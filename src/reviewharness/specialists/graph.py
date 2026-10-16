"""One specialist review, expressed as a small LangGraph state machine.

    START -> call_llm -> parse -> END
                ^          |
                +--retry---+      (also call_llm -> call_llm on transient errors)

Attempts are bounded (settings.max_retries + 1). Transient provider errors get a
capped exponential backoff (or a valid provider delay); malformed replies are retried once per attempt with a
corrective note. A permanent error stops immediately. review() never raises for
provider problems: it returns a FAILED/TIMED_OUT result so a caller can tell
"the review broke" apart from "the review found nothing".
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Callable, TypedDict

from reviewharness.config import ConfigError, Settings
from reviewharness.ingest.models import Inventory, RepositoryContext
from reviewharness.llm.base import (
    LLMError,
    LLMProvider,
    LLMResponseError,
    LLMTimeoutError,
    LLMTransientError,
)
from reviewharness.logging_setup import redact
from reviewharness.models import SpecialistSource, TaskStatus
from reviewharness.specialists.excerpts import build_excerpts, select_scope
from reviewharness.specialists.models import Excerpt, SpecialistResult
from reviewharness.specialists.parsing import ResponseFormatError, extract_findings_list, validate_findings
from reviewharness.specialists.prompts import repair_note, system_prompt, user_prompt

log = logging.getLogger("reviewharness.specialists")

BACKOFF_BASE_S = 0.5
BACKOFF_CAP_S = 30.0


class ReviewState(TypedDict, total=False):
    attempt: int
    outcome: str  # parse | retry | failed | done
    error: str
    error_kind: str  # timeout | transient | response | permanent
    feedback: str
    raw: str
    findings: list
    rejected: list
    prompt_tokens: int
    completion_tokens: int
    retry_after_s: float | None


def _build_graph(call_node: Callable, parse_node: Callable):
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        raise ConfigError('the "langgraph" package is not installed; run: pip install -e ".[agents]"') from None

    def after_call(state: ReviewState) -> str:
        return {"parse": "parse", "retry": "retry"}.get(state["outcome"], "end")

    def after_parse(state: ReviewState) -> str:
        return "retry" if state["outcome"] == "retry" else "end"

    graph = StateGraph(ReviewState)
    graph.add_node("call_llm", call_node)
    graph.add_node("parse", parse_node)
    graph.add_edge(START, "call_llm")
    graph.add_conditional_edges("call_llm", after_call, {"parse": "parse", "retry": "call_llm", "end": END})
    graph.add_conditional_edges("parse", after_parse, {"retry": "call_llm", "end": END})
    return graph.compile()


def review(
    specialist: SpecialistSource,
    root: Path,
    inventory: Inventory,
    context: RepositoryContext,
    provider: LLMProvider,
    settings: Settings,
    *,
    sleep: Callable[[float], None] = time.sleep,
    max_total_chars: int | None = None,
    scope_paths: list[str] | None = None,
) -> SpecialistResult:
    started = time.monotonic()
    max_attempts = settings.max_retries + 1

    kwargs = {} if max_total_chars is None else {"max_total_chars": max_total_chars}
    excerpts, coverage = build_excerpts(
        root, inventory, select_scope(inventory, specialist) if scope_paths is None else scope_paths, **kwargs
    )
    system = system_prompt(specialist)
    base_user = user_prompt(specialist, context, excerpts)

    def retry_or_fail(
        state: ReviewState, attempt: int, kind: str, message: str,
        feedback: str = "", retry_after_s: float | None = None,
    ) -> ReviewState:
        outcome = "retry" if attempt < max_attempts else "failed"
        log.warning("%s attempt %d/%d failed (%s): %s", specialist.value, attempt, max_attempts, kind, redact(message))
        return {"attempt": attempt, "outcome": outcome, "error": message, "error_kind": kind, "feedback": feedback, "retry_after_s": retry_after_s}

    def call_llm(state: ReviewState) -> ReviewState:
        attempt = state.get("attempt", 0) + 1
        if attempt > 1 and state.get("error_kind") in ("timeout", "transient"):
            backoff = min(BACKOFF_BASE_S * 2 ** (attempt - 2), BACKOFF_CAP_S)
            requested = state.get("retry_after_s")
            # Provider delay is authoritative, not capped to a smaller delay.
            # If it is too long, fail without violating the server's request.
            if requested is not None and isinstance(requested, (int, float)) and not isinstance(requested, bool):
                if math.isfinite(requested) and requested >= 0:
                    backoff = requested
            remaining = settings.task_timeout_s - (time.monotonic() - started)
            if backoff > BACKOFF_CAP_S or backoff >= remaining:
                reason = (f"retry delay {backoff:g}s exceeds safe wait bound {BACKOFF_CAP_S:g}s"
                          if backoff > BACKOFF_CAP_S else
                          f"retry delay {backoff:g}s exceeds remaining task budget")
                log.warning("%s: %s", specialist.value, reason)
                return {"attempt": attempt - 1, "outcome": "failed", "error": reason,
                        "error_kind": "transient", "retry_after_s": None}
            sleep(backoff)
        feedback = state.get("feedback", "")
        user = base_user + (repair_note(feedback) if feedback else "")
        try:
            result = provider.complete(system, user)
        except LLMTimeoutError as exc:
            return retry_or_fail(state, attempt, "timeout", str(exc), retry_after_s=exc.retry_after_s)
        except LLMTransientError as exc:
            return retry_or_fail(state, attempt, "transient", str(exc), retry_after_s=exc.retry_after_s)
        except LLMResponseError as exc:
            return retry_or_fail(state, attempt, "response", str(exc), feedback=str(exc))
        except LLMError as exc:
            log.error("%s permanent provider error: %s", specialist.value, redact(str(exc)))
            return {"attempt": attempt, "outcome": "failed", "error": str(exc), "error_kind": "permanent"}
        return {
            "attempt": attempt,
            "outcome": "parse",
            "raw": result.text,
            "error": "",
            "error_kind": "",
            "feedback": "",
            "retry_after_s": None,
            "prompt_tokens": state.get("prompt_tokens", 0) + result.prompt_tokens,
            "completion_tokens": state.get("completion_tokens", 0) + result.completion_tokens,
        }

    def parse(state: ReviewState) -> ReviewState:
        attempt = state["attempt"]
        try:
            items = extract_findings_list(state["raw"])
        except ResponseFormatError as exc:
            return retry_or_fail(state, attempt, "response", str(exc), feedback=str(exc))
        findings, rejected = validate_findings(items, specialist, excerpts)
        return {"outcome": "done", "findings": findings, "rejected": rejected, "error": "", "error_kind": ""}

    app = _build_graph(call_llm, parse)
    final = app.invoke({"attempt": 0, "prompt_tokens": 0, "completion_tokens": 0}, config={"recursion_limit": 25})

    done = final.get("outcome") == "done"
    if done:
        status, error = TaskStatus.SUCCEEDED, None
    else:
        status = TaskStatus.TIMED_OUT if final.get("error_kind") == "timeout" else TaskStatus.FAILED
        error = redact(final.get("error") or "unknown failure")[:300]

    return SpecialistResult(
        specialist=specialist,
        status=status,
        provider=provider.name,
        model=provider.model,
        findings=final.get("findings", []) if done else [],
        rejected=final.get("rejected", []) if done else [],
        coverage=coverage,
        attempts=final["attempt"],
        prompt_tokens=final.get("prompt_tokens", 0),
        completion_tokens=final.get("completion_tokens", 0),
        duration_s=round(time.monotonic() - started, 3),
        error=error,
    )
