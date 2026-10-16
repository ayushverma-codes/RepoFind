"""Run orchestration.

Run-level workflow (LangGraph):   plan -> (run already exists? END) -> execute -> check -> finalize -> END

* plan      deterministic planner builds bounded tasks; the store records them (idempotent per run id)
* execute   asyncio workers, one per task, bounded by a semaphore. Each worker claims its task in
            SQLite (so a task runs once), runs the blocking specialist in a thread, and records the
            result. A timeout, error or cancellation affects only that task.
* check     validates and independently checks candidate findings (see reviewharness.checking)
* finalize  sweeps leftovers and derives the run status from task outcomes.

Retries: transient LLM errors are retried inside the specialist graph (settings.max_retries). The
orchestrator does not retry again, so total attempts per task stay bounded by max_retries + 1.

Known limit: Python cannot kill a thread. A task that times out is recorded as TIMED_OUT and its
result is discarded, but the provider call keeps running until the provider's own timeout.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TypedDict

from reviewharness.checking.adjudicate import adjudicate
from reviewharness.checking.pipeline import MAX_CHECKED_FINDINGS, check_run
from reviewharness.config import ConfigError, Settings
from reviewharness.ingest.models import IngestResult
from reviewharness.llm.base import LLMProvider
from reviewharness.logging_setup import redact
from reviewharness.models import RunRecord, RunStatus, TaskStatus
from reviewharness.orchestration.models import PlannedTask
from reviewharness.orchestration.planner import SPECIALISTS, make_run_id, plan_tasks
from reviewharness.orchestration.states import TERMINAL_RUN_STATES, derive_run_status
from reviewharness.orchestration.store import RunStore
from reviewharness.specialists import SpecialistResult, review as run_specialist

log = logging.getLogger("reviewharness.orchestration")


class OrchestrationError(Exception):
    """The run cannot be started (for example a previous identical run is still marked running)."""


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    status: RunStatus
    reused: bool  # True: an identical finished run already existed and nothing was re-executed


@dataclass(frozen=True)
class RunContext:
    run_id: str
    ingest: IngestResult
    deadline: float  # time.monotonic() value
    sem: asyncio.Semaphore

    @property
    def root(self) -> Path:
        return Path(self.ingest.snapshot.tree_path)


class _State(TypedDict, total=False):
    run_id: str
    plans: list[PlannedTask]
    reused: bool
    status: str
    deadline: float


ReviewFn = Callable[..., SpecialistResult]


class Orchestrator:
    def __init__(
        self,
        store: RunStore,
        settings: Settings,
        provider: LLMProvider,
        review_fn: ReviewFn = run_specialist,
        adjudicate_fn: Callable = adjudicate,
    ) -> None:
        self.store = store
        self.settings = settings
        self.provider = provider
        self.review_fn = review_fn
        self.adjudicate_fn = adjudicate_fn

    # ------------------------------------------------------------------ API

    async def run(self, ingest: IngestResult, *, new_run: bool = False) -> RunOutcome:
        """Execute a review run. Re-running an identical finished request returns the stored run."""
        nonce = f"{time.time_ns()}" if new_run else ""
        run_id = make_run_id(ingest.snapshot.snapshot_id, list(SPECIALISTS), self.provider.name, self.provider.model, nonce)
        app = self._build_graph(ingest)
        try:
            final = await app.ainvoke({"run_id": run_id}, config={"recursion_limit": 10})
        except asyncio.CancelledError:
            self._close_out(run_id, "run cancelled", cancelled=True)
            raise
        record = self.store.get_run(run_id)
        assert record is not None
        return RunOutcome(run_id=run_id, status=record.status, reused=bool(final.get("reused")))

    # ---------------------------------------------------------------- graph

    def _build_graph(self, ingest: IngestResult):
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:
            raise ConfigError('the "langgraph" package is not installed; run: pip install -e ".[agents]"') from None

        async def plan(state: _State) -> _State:
            run_id = state["run_id"]
            plans = plan_tasks(run_id, ingest.inventory, self.settings)
            existing = self.store.get_run(run_id)
            if existing is not None and existing.status not in TERMINAL_RUN_STATES:
                raise OrchestrationError(
                    f"run {run_id} exists but is still '{existing.status.value}' (another process, or it crashed). "
                    "Start a fresh run with --new-run."
                )
            created = self.store.create_run(run_id, ingest.snapshot.snapshot_id, ingest.snapshot.source, plans)
            if created:
                for p in plans:
                    if p.skip_reason:
                        self.store.skip_task(p.task.task_id, p.skip_reason)
            return {"plans": plans, "reused": not created}

        async def execute(state: _State) -> _State:
            run_id = state["run_id"]
            self.store.start_run(run_id)
            self.store.add_event(run_id, "run_budgets", self._budgets())
            ctx = RunContext(
                run_id=run_id,
                ingest=ingest,
                deadline=time.monotonic() + self.settings.run_timeout_s,
                sem=asyncio.Semaphore(self.settings.max_concurrency),
            )
            record = self.store.get_run(run_id)
            assert record is not None
            todo = [t for t in record.tasks if t.status is TaskStatus.PENDING]
            outcomes = await asyncio.gather(*(self._work(ctx, t) for t in todo), return_exceptions=True)
            for t, out in zip(todo, outcomes):
                if isinstance(out, BaseException):
                    log.error("worker for %s crashed: %s", t.task_id, redact(f"{type(out).__name__}: {out}"))
            return {"deadline": ctx.deadline}

        async def check(state: _State) -> _State:
            await check_run(
                self.store, self.settings, self.provider, Path(ingest.snapshot.tree_path), ingest.inventory,
                state["run_id"], state["deadline"],
                adjudicate_fn=self.adjudicate_fn,
            )
            return {}

        async def finalize(state: _State) -> _State:
            run_id = state["run_id"]
            self._close_out(run_id, "worker ended without recording a result", cancelled=False)
            return {}

        graph = StateGraph(_State)
        graph.add_node("plan", plan)
        graph.add_node("execute", execute)
        graph.add_node("check", check)
        graph.add_node("finalize", finalize)
        graph.add_edge(START, "plan")
        graph.add_conditional_edges("plan", lambda s: "end" if s.get("reused") else "execute", {"end": END, "execute": "execute"})
        graph.add_edge("execute", "check")
        graph.add_edge("check", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    # --------------------------------------------------------------- workers

    async def _work(self, ctx: RunContext, task) -> None:
        async with ctx.sem:
            remaining = ctx.deadline - time.monotonic()
            if remaining <= 0:
                self.store.skip_task(task.task_id, "run time budget exhausted before the task started")
                return
            if sum(self.store.token_usage(ctx.run_id)) >= self.settings.max_total_tokens:
                self.store.skip_task(
                    task.task_id,
                    f"model-usage budget of {self.settings.max_total_tokens} tokens exhausted before the task started",
                )
                return
            if not self.store.claim_task(task.task_id):
                return  # someone else owns it; never execute twice
            limit = min(self.settings.task_timeout_s, remaining)
            reason = "task timeout" if self.settings.task_timeout_s <= remaining else "run time budget"
            call = functools.partial(
                self.review_fn,
                task.specialist,
                ctx.root,
                ctx.ingest.inventory,
                ctx.ingest.context,
                self.provider,
                self.settings,
                scope_paths=task.scope_paths,
            )
            try:
                result = await asyncio.wait_for(asyncio.to_thread(call), timeout=limit)
            except asyncio.TimeoutError:
                self.store.finish_task(task.task_id, TaskStatus.TIMED_OUT, error=f"exceeded {reason} of {limit:.1f}s")
            except asyncio.CancelledError:
                self.store.finish_task(task.task_id, TaskStatus.CANCELLED, error="cancelled")
                raise
            except Exception as exc:  # a crashing worker must not take the others down
                self.store.finish_task(task.task_id, TaskStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
            else:
                if result.status is TaskStatus.SUCCEEDED:
                    self.store.finish_task(task.task_id, TaskStatus.SUCCEEDED, result=result)
                else:
                    self.store.finish_task(task.task_id, result.status, error=result.error or "unknown failure", result=result)

    def _budgets(self) -> dict:
        """The limits this run was started with, recorded so a replayed report states them accurately."""
        s = self.settings
        return {
            "provider": self.provider.name,
            "model": self.provider.model,
            "max_concurrency": s.max_concurrency,
            "max_retries": s.max_retries,
            "max_revisions": s.max_revisions,
            "max_total_tokens": s.max_total_tokens,
            "max_checked_findings": MAX_CHECKED_FINDINGS,
            "llm_timeout_s": s.llm_timeout_s,
            "task_timeout_s": s.task_timeout_s,
            "run_timeout_s": s.run_timeout_s,
        }

    # -------------------------------------------------------------- wrap-up

    def _close_out(self, run_id: str, reason: str, *, cancelled: bool) -> None:
        """Sweep unfinished tasks, then set the run status from what actually happened."""
        run = self.store.get_run(run_id)
        if run is None or run.status in TERMINAL_RUN_STATES:
            return
        self.store.sweep_unfinished(run_id, reason, cancelled=cancelled)
        run = self.store.get_run(run_id)
        assert run is not None
        status = derive_run_status(
            [t.status for t in run.tasks], cancelled=cancelled, check_incomplete=self.store.check_incomplete(run_id)
        )
        counts: dict[str, int] = {}
        for t in run.tasks:
            counts[t.status.value] = counts.get(t.status.value, 0) + 1
        if run.status is RunStatus.CREATED:
            self.store.start_run(run_id)
        self.store.finish_run(run_id, status, {"tasks": counts})
