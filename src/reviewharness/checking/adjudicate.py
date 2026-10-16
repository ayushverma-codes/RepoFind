"""Check one finding: checker -> (revise -> checker)* with a hard revision budget, as a LangGraph.

    START -> check --accept/reject/uncertain--> END
               |  ^
        revise |  | revised finding (re-validated deterministically first)
               v  |
              revise --withdrawn / invalid--> END

Policy (deterministic, applied after the model speaks):
  * severity can only be lowered by the checker, never raised;
  * the specialist's confidence is a hint, not proof: accepted findings keep at most 0.95, reduced
    10% per revision; uncertain findings are capped at 0.5;
  * ACCEPTED requires a recorded checker "accept" decision (enforced by FindingOutcome).
Every provider call is retried at most settings.max_retries times; revisions at most settings.max_revisions.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Callable, TypedDict

from pydantic import ValidationError

from reviewharness.checking.models import CheckerReply, FindingOutcome, OutcomeKind
from reviewharness.checking.prompts import CHECKER_SYSTEM, REVISER_SYSTEM, checker_user, reviser_user
from reviewharness.checking.source import read_lines
from reviewharness.checking.validation import location_problem
from reviewharness.config import ConfigError, Settings
from reviewharness.llm.base import LLMError, LLMProvider, LLMResponseError, LLMTimeoutError, LLMTransientError
from reviewharness.logging_setup import redact
from reviewharness.models import CheckerDecision, CheckerStatus, CheckerVerdict, Finding, Severity
from reviewharness.specialists.parsing import (
    ALLOWED_CATEGORIES, FINDING_FIELDS, ResponseFormatError, extract_json_object, mask_secrets,
)
from reviewharness.specialists.prompts import repair_note

log = logging.getLogger("reviewharness.checking")

BACKOFF_BASE_S = 0.5
BACKOFF_CAP_S = 30.0
MAX_ACCEPTED_CONFIDENCE = 0.95
MAX_UNCERTAIN_CONFIDENCE = 0.5
REVISION_CONFIDENCE_FACTOR = 0.9


class CallFailed(Exception):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class AdjState(TypedDict, total=False):
    finding: Finding
    revisions: int
    decisions: list
    feedback: str
    route: str  # check | revise | end
    kind: str
    code: str
    reason: str
    ptok: int
    ctok: int


def _short(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        err = exc.errors()[0]  # location + message only; never echo values
        return f"schema: {'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
    return str(exc)


def call_model(provider: LLMProvider, system: str, user: str, parse: Callable[[str], Any], settings: Settings, sleep: Callable[[float], None]):
    """One bounded model call. Returns (parsed, prompt_tokens, completion_tokens) or raises CallFailed."""
    max_attempts = settings.max_retries + 1
    feedback, kind, message = "", "response", ""
    ptok = ctok = 0
    started = time.monotonic()
    retry_after_s = None
    for attempt in range(1, max_attempts + 1):
        if attempt > 1 and kind in ("timeout", "transient"):
            delay = min(BACKOFF_BASE_S * 2 ** (attempt - 2), BACKOFF_CAP_S)
            if retry_after_s is not None and math.isfinite(retry_after_s) and retry_after_s >= 0:
                delay = retry_after_s
            remaining = settings.task_timeout_s - (time.monotonic() - started)
            if delay > BACKOFF_CAP_S or delay >= remaining:
                raise CallFailed("transient", "provider retry delay exceeds safe wait or task budget")
            sleep(delay)
        try:
            result = provider.complete(system, user + (repair_note(feedback) if feedback else ""))
        except LLMTimeoutError as exc:
            kind, message, feedback = "timeout", str(exc), ""
            retry_after_s = exc.retry_after_s
            continue
        except LLMTransientError as exc:
            kind, message, feedback = "transient", str(exc), ""
            retry_after_s = exc.retry_after_s
            continue
        except LLMResponseError as exc:
            kind, message, feedback = "response", str(exc), str(exc)
            continue
        except LLMError as exc:
            raise CallFailed("permanent", str(exc)) from None
        ptok += result.prompt_tokens
        ctok += result.completion_tokens
        try:
            return parse(result.text), ptok, ctok
        except (ResponseFormatError, ValueError) as exc:  # pydantic ValidationError is a ValueError
            kind, message = "response", _short(exc)
            feedback = message
    raise CallFailed(kind, message or "no usable reply")


def parse_checker_reply(text: str) -> CheckerReply:
    return CheckerReply.model_validate(extract_json_object(text))


def _lower_only(current: Severity, suggested: Severity | None) -> Severity:
    return suggested if suggested is not None and suggested.rank > current.rank else current


def accepted_confidence(confidence: float, revisions: int) -> float:
    return round(min(confidence, MAX_ACCEPTED_CONFIDENCE) * REVISION_CONFIDENCE_FACTOR**revisions, 2)


def _build_revision(original: Finding, item: dict, lines: list[str]) -> tuple[Finding | None, str]:
    """Validate a revised finding exactly like a new one. Returns (finding, "") or (None, reason)."""
    data = {k: item[k] for k in FINDING_FIELDS if k in item}
    data["file_path"] = original.file_path  # a revision may not move to another file
    try:
        revised = Finding(**data, source=original.source, finding_id=original.finding_id)
    except ValidationError as exc:
        return None, _short(exc)
    if revised.category not in ALLOWED_CATEGORIES[original.source]:
        return None, "revised category is outside the specialist's role"
    revised = revised.model_copy(
        update={k: mask_secrets(getattr(revised, k)) for k in ("evidence", "rationale", "impact")}
    )
    problem = location_problem(revised, lines)
    if problem:
        return None, problem[1]
    return revised, ""


def _build_graph(check_node: Callable, revise_node: Callable):
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        raise ConfigError('the "langgraph" package is not installed; run: pip install -e ".[agents]"') from None
    graph = StateGraph(AdjState)
    graph.add_node("check", check_node)
    graph.add_node("revise", revise_node)
    graph.add_edge(START, "check")
    graph.add_conditional_edges("check", lambda s: s["route"], {"revise": "revise", "end": END})
    graph.add_conditional_edges("revise", lambda s: s["route"], {"check": "check", "end": END})
    return graph.compile()


def adjudicate(
    finding: Finding,
    root: Path,
    provider: LLMProvider,
    settings: Settings,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> FindingOutcome:
    lines = read_lines(root, finding.file_path)
    if lines is None:
        return _outcome(finding, finding, OutcomeKind.UNCERTAIN, "checker_error", "source could not be read for checking", 0, [], 0, 0)

    def check_node(state: AdjState) -> AdjState:
        f, revisions = state["finding"], state["revisions"]
        try:
            reply, p, c = call_model(provider, CHECKER_SYSTEM, checker_user(f, lines), parse_checker_reply, settings, sleep)
        except CallFailed as exc:
            log.warning("checker failed for %s (%s): %s", f.finding_id, exc.kind, redact(str(exc)))
            return {"route": "end", "kind": "uncertain", "code": "checker_error", "reason": f"checker unavailable ({exc.kind}): {redact(str(exc))}"[:500]}
        decision = CheckerDecision(
            finding_id=f.finding_id, verdict=reply.verdict, reasoning=redact(reply.reasoning),
            counterexample=reply.counterexample, attempt=revisions,
        )
        update: AdjState = {"decisions": state["decisions"] + [decision], "ptok": state["ptok"] + p, "ctok": state["ctok"] + c}
        if reply.verdict is CheckerVerdict.ACCEPT:
            lowered = _lower_only(f.severity, reply.suggested_severity)
            return {**update, "finding": f.model_copy(update={"severity": lowered}), "route": "end", "kind": "accepted", "code": "checker_accepted", "reason": reply.reasoning[:500]}
        if reply.verdict is CheckerVerdict.REJECT:
            why = reply.counterexample or reply.reasoning
            return {**update, "route": "end", "kind": "rejected", "code": "checker_rejected", "reason": why[:500]}
        if revisions >= settings.max_revisions:
            return {**update, "route": "end", "kind": "uncertain", "code": "revision_budget_exhausted",
                    "reason": f"still disputed after {revisions} revision(s): {reply.reasoning}"[:500]}
        return {**update, "route": "revise", "feedback": reply.reasoning + (f" Counterexample: {reply.counterexample}" if reply.counterexample else "")}

    def revise_node(state: AdjState) -> AdjState:
        f = state["finding"]
        try:
            item, p, c = call_model(provider, REVISER_SYSTEM, reviser_user(f, lines, state["feedback"]), _parse_revision, settings, sleep)
        except CallFailed as exc:
            return {"route": "end", "kind": "uncertain", "code": "checker_error", "reason": f"reviser unavailable ({exc.kind}): {redact(str(exc))}"[:500]}
        base: AdjState = {"ptok": state["ptok"] + p, "ctok": state["ctok"] + c}
        if item is None:
            return {**base, "route": "end", "kind": "rejected", "code": "withdrawn_by_specialist", "reason": "the specialist withdrew the finding after the checker's challenge"}
        revised, problem = _build_revision(f, item, lines)
        if revised is None:
            return {**base, "route": "end", "kind": "uncertain", "code": "revision_invalid", "reason": f"the revised finding failed validation: {problem}"[:500]}
        return {**base, "finding": revised, "revisions": state["revisions"] + 1, "route": "check"}

    app = _build_graph(check_node, revise_node)
    limit = 2 * (settings.max_revisions + 1) + 4
    final = app.invoke(
        {"finding": finding, "revisions": 0, "decisions": [], "ptok": 0, "ctok": 0},
        config={"recursion_limit": limit},
    )
    return _outcome(
        finding, final["finding"], OutcomeKind(final["kind"]), final["code"], final.get("reason", ""),
        final["revisions"], final["decisions"], final["ptok"], final["ctok"],
    )


def _parse_revision(text: str) -> dict | None:
    data = extract_json_object(text)
    if "finding" not in data:
        raise ResponseFormatError('the JSON must contain a "finding" key (an object, or null to withdraw)')
    item = data["finding"]
    if item is not None and not isinstance(item, dict):
        raise ResponseFormatError('"finding" must be an object or null')
    return item


def _outcome(original: Finding, final: Finding, kind: OutcomeKind, code: str, reason: str, revisions: int, decisions: list, ptok: int, ctok: int) -> FindingOutcome:
    if kind is OutcomeKind.ACCEPTED:
        final = final.model_copy(update={"checker_status": CheckerStatus.ACCEPTED, "confidence": accepted_confidence(final.confidence, revisions)})
    elif kind is OutcomeKind.UNCERTAIN:
        final = final.model_copy(update={"checker_status": CheckerStatus.UNCERTAIN, "confidence": round(min(final.confidence, MAX_UNCERTAIN_CONFIDENCE), 2)})
    else:
        final = final.model_copy(update={"checker_status": CheckerStatus.REJECTED})
    return FindingOutcome(
        finding=final, original=original if revisions > 0 else None, kind=kind, reason_code=code,
        reason=reason, revisions=revisions, decisions=decisions, prompt_tokens=ptok, completion_tokens=ctok,
    )
