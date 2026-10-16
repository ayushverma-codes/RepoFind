"""Checking stage driver: validate -> check concurrently under budgets -> record every outcome.

Nothing reaches the accepted report except findings with an ACCEPTED outcome. Findings that
could not be checked (checker down, time or count budget) are recorded UNCERTAIN, never accepted,
and make the run PARTIAL.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from reviewharness.checking.adjudicate import adjudicate
from reviewharness.checking.models import CheckSummary, FindingOutcome, INCOMPLETE_CODES, OutcomeKind
from reviewharness.checking.validation import validate_candidates
from reviewharness.config import Settings
from reviewharness.ingest.models import Inventory
from reviewharness.llm.base import LLMProvider
from reviewharness.logging_setup import redact
from reviewharness.models import CheckerStatus, Finding, Report, TaskStatus, priority_key

if TYPE_CHECKING:
    from reviewharness.orchestration.store import RunStore

log = logging.getLogger("reviewharness.checking")

MAX_CHECKED_FINDINGS = 40  # model-usage bound for one run


def _uncertain(f: Finding, code: str, reason: str) -> FindingOutcome:
    f = f.model_copy(update={"checker_status": CheckerStatus.UNCERTAIN, "confidence": round(min(f.confidence, 0.5), 2)})
    return FindingOutcome(finding=f, kind=OutcomeKind.UNCERTAIN, reason_code=code, reason=reason)


async def check_run(
    store: "RunStore",
    settings: Settings,
    provider: LLMProvider,
    root: Path,
    inventory: Inventory,
    run_id: str,
    deadline: float,
    *,
    adjudicate_fn: Callable[..., FindingOutcome] = adjudicate,
    sleep: Callable[[float], None] = time.sleep,
) -> CheckSummary:
    decided = {o.finding.finding_id for o in store.finding_outcomes(run_id)}
    candidates = [f for f in store.candidate_findings(run_id) if f.finding_id not in decided]
    if candidates:
        store.add_event(run_id, "check_started", {"candidates": len(candidates)})
        result = validate_candidates(root, inventory, candidates)
        for bad in result.invalid:
            f = bad.finding.model_copy(update={"checker_status": CheckerStatus.REJECTED})
            store.record_outcome(run_id, FindingOutcome(finding=f, kind=OutcomeKind.INVALID, reason_code=bad.code, reason=bad.reason))
        for dup in result.duplicates:
            store.record_outcome(
                run_id,
                FindingOutcome(finding=dup.finding, kind=OutcomeKind.DUPLICATE, reason_code="duplicate", reason=f"overlaps {dup.kept_id}"),
            )
        to_check, over_budget = result.valid[:MAX_CHECKED_FINDINGS], result.valid[MAX_CHECKED_FINDINGS:]
        for f in over_budget:
            store.record_outcome(run_id, _uncertain(f, "check_budget", f"more than {MAX_CHECKED_FINDINGS} findings; this one was not checked"))

        sem = asyncio.Semaphore(settings.max_concurrency)

        async def one(f: Finding) -> None:
            async with sem:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    outcome = _uncertain(f, "run_deadline", "run time budget exhausted before this finding was checked")
                elif sum(store.token_usage(run_id)) >= settings.max_total_tokens:
                    outcome = _uncertain(
                        f, "check_budget",
                        f"model-usage budget of {settings.max_total_tokens} tokens exhausted before this finding was checked",
                    )
                else:
                    limit = min(settings.task_timeout_s, remaining)
                    call = functools.partial(adjudicate_fn, f, root, provider, settings, sleep=sleep)
                    try:
                        outcome = await asyncio.wait_for(asyncio.to_thread(call), timeout=limit)
                    except asyncio.TimeoutError:
                        outcome = _uncertain(f, "check_timeout", f"check exceeded {limit:.1f}s")
                    except Exception as exc:  # one finding's crash must not stop the others
                        outcome = _uncertain(f, "checker_error", redact(f"{type(exc).__name__}: {exc}")[:300])
                store.record_outcome(run_id, outcome)

        results = await asyncio.gather(*(one(f) for f in to_check), return_exceptions=True)
        for r in results:
            if isinstance(r, BaseException):
                log.error("check worker crashed: %s", redact(f"{type(r).__name__}: {r}"))

    outcomes = store.finding_outcomes(run_id)
    counts: dict[str, int] = {}
    for o in outcomes:
        counts[o.kind.value] = counts.get(o.kind.value, 0) + 1
    incomplete = store.check_incomplete(run_id)
    if candidates:
        store.add_event(run_id, "check_finished", {"counts": counts, "incomplete": incomplete})
    return CheckSummary(counts=counts, incomplete=incomplete)


def assemble_report(store: "RunStore", run_id: str) -> Report:
    """Build the Report from recorded outcomes only. Report's own validators reject anything unverified."""
    run = store.get_run(run_id)
    if run is None:
        raise KeyError(run_id)
    outcomes = store.finding_outcomes(run_id)

    accepted = sorted((o.finding for o in outcomes if o.kind is OutcomeKind.ACCEPTED), key=priority_key)
    unresolved = sorted((o.finding for o in outcomes if o.kind is OutcomeKind.UNCERTAIN), key=priority_key)
    bad = (TaskStatus.FAILED, TaskStatus.TIMED_OUT, TaskStatus.CANCELLED)
    return Report(
        run_id=run_id,
        run_status=run.status,
        findings=accepted,
        unresolved=unresolved,
        failed_task_ids=[t.task_id for t in run.tasks if t.status in bad],
        skipped_task_ids=[t.task_id for t in run.tasks if t.status is TaskStatus.SKIPPED],
    )
