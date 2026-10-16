"""Build a RunReport from the SQLite run store.

Everything here is read-only and deterministic: the same stored run always yields the same report.
All repository-derived or error text is masked for secrets before it can reach a report.
"""
from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from typing import Any, Optional

from reviewharness.checking.models import INCOMPLETE_CODES, OutcomeKind
from reviewharness.checking.pipeline import assemble_report
from reviewharness.logging_setup import redact
from reviewharness.models import Finding, RunStatus, TaskStatus, priority_key
from reviewharness.orchestration.inspect import peak_concurrency
from reviewharness.orchestration.store import RunStore
from reviewharness.reporting.models import (
    Budgets,
    CoverageGap,
    CoverageSummary,
    RunReport,
    TaskSummary,
    UnresolvedFinding,
    VerificationSummary,
)
from reviewharness.specialists.parsing import mask_secrets

log = logging.getLogger("reviewharness.reporting")

MAX_GAPS_LISTED = 100
_GAP_STATES = frozenset({"truncated", "omitted", "not_reviewed", "over_scope_cap", "pending"})
_LIMIT_BY_CODE = {
    "check_budget": "checked-finding or model-usage budget",
    "run_deadline": "run time budget",
    "check_timeout": "per-check time limit",
}


def clean(text: str) -> str:
    """Mask secret-looking literals and API keys. Applied to every free-text field in a report."""
    return redact(mask_secrets(text))


def _clean_finding(f: Finding) -> Finding:
    return f.model_copy(update={k: clean(getattr(f, k)) for k in ("title", "evidence", "rationale", "impact")})


def _duration(created: Optional[str], finished: Optional[str]) -> Optional[float]:
    if not created or not finished:
        return None
    return round((datetime.fromisoformat(finished) - datetime.fromisoformat(created)).total_seconds(), 3)


def _skip_limit(error: str) -> Optional[str]:
    if "model-usage budget" in error:
        return "model-usage budget"
    if "run time budget" in error:
        return "run time budget"
    return None


def build_run_report(store: RunStore, run_id: str) -> RunReport:
    """Raises KeyError for an unknown run."""
    run = store.get_run(run_id)
    if run is None:
        raise KeyError(run_id)
    meta = store.run_meta(run_id) or {}
    base = assemble_report(store, run_id)  # its validators refuse anything not checker-accepted
    outcomes = store.finding_outcomes(run_id)
    candidates = store.candidate_findings(run_id)
    events = store.events(run_id)
    results = store.task_results(run_id)
    prompt_tokens, completion_tokens = store.token_usage(run_id)

    findings = [_clean_finding(f) for f in base.findings]
    unresolved = sorted(
        (
            UnresolvedFinding(finding=_clean_finding(o.finding), reason_code=o.reason_code, reason=clean(o.reason), revisions=o.revisions)
            for o in outcomes
            if o.kind is OutcomeKind.UNCERTAIN
        ),
        key=lambda u: priority_key(u.finding),
    )

    kinds = Counter(o.kind for o in outcomes)
    decided = {o.finding.finding_id for o in outcomes}
    verification = VerificationSummary(
        candidates=len(candidates),
        accepted=kinds[OutcomeKind.ACCEPTED],
        unresolved=kinds[OutcomeKind.UNCERTAIN],
        rejected=kinds[OutcomeKind.REJECTED],
        invalid=kinds[OutcomeKind.INVALID],
        duplicate=kinds[OutcomeKind.DUPLICATE],
        unchecked=sum(1 for c in candidates if c.finding_id not in decided),
        checks_incomplete=sum(1 for o in outcomes if o.reason_code in INCOMPLETE_CODES),
    )

    tasks = []
    for t in sorted(run.tasks, key=lambda t: t.specialist.value):
        r = results.get(t.task_id)
        tasks.append(
            TaskSummary(
                specialist=t.specialist.value, status=t.status.value, attempts=t.attempts, max_attempts=t.max_attempts,
                findings=len(r.findings) if r else 0,
                prompt_tokens=r.prompt_tokens if r else 0, completion_tokens=r.completion_tokens if r else 0,
                duration_s=r.duration_s if r else 0.0,
                error=clean(t.error) if t.error else None,
            )
        )

    rows = store.coverage(run_id)
    by_spec: dict[str, Counter] = {}
    for row in rows:
        by_spec.setdefault(row.specialist, Counter())[row.state] += 1
    gap_rows = sorted((r for r in rows if r.state in _GAP_STATES), key=lambda r: (r.specialist, r.path))
    coverage = CoverageSummary(
        by_specialist={s: dict(sorted(c.items())) for s, c in sorted(by_spec.items())},
        gaps=[CoverageGap(specialist=r.specialist, path=r.path, state=r.state) for r in gap_rows[:MAX_GAPS_LISTED]],
        gaps_total=len(gap_rows),
    )

    budget_events = [e for e in events if e.type == "run_budgets"]
    configured: Optional[dict[str, Any]] = budget_events[-1].payload if budget_events else None
    limits_hit = {_LIMIT_BY_CODE[o.reason_code] for o in outcomes if o.reason_code in _LIMIT_BY_CODE}
    limits_hit |= {"task time limit" for t in run.tasks if t.status is TaskStatus.TIMED_OUT}
    limits_hit |= {lim for t in run.tasks if t.status is TaskStatus.SKIPPED and t.error and (lim := _skip_limit(t.error))}
    if configured and prompt_tokens + completion_tokens > configured.get("max_total_tokens", float("inf")):
        limits_hit.add("model-usage budget (calls already in flight cannot be interrupted)")
    budgets = Budgets(
        configured=configured,
        observed={
            "duration_s": _duration(meta.get("created_at"), meta.get("finished_at")),
            "peak_concurrency": peak_concurrency(events),
            "total_tokens": prompt_tokens + completion_tokens,
            "findings_checked": sum(1 for o in outcomes if o.decisions),
            "max_revisions_used": max((o.revisions for o in outcomes), default=0),
            "max_task_attempts": max((t.attempts for t in run.tasks), default=0),
        },
        limits_hit=sorted(limits_hit),
    )

    reasons: list[str] = []
    if run.status is not RunStatus.COMPLETED:
        reasons.append(f"run status is '{run.status.value}', not 'completed'")
    not_ok = [t for t in run.tasks if t.status is not TaskStatus.SUCCEEDED]
    if not_ok:
        reasons.append("review task(s) did not succeed: " + ", ".join(f"{t.specialist.value} ({t.status.value})" for t in not_ok))
    if verification.unchecked:
        reasons.append(f"{verification.unchecked} candidate finding(s) were never checked and are excluded from this report")
    if verification.checks_incomplete:
        reasons.append(f"verification could not finish for {verification.checks_incomplete} finding(s); they are listed as unresolved")
    if coverage.gaps_total:
        reasons.append(f"{coverage.gaps_total} file/task pair(s) were not fully reviewed (truncated, omitted, over scope cap or not reviewed)")

    report = RunReport(
        run_id=run_id,
        run_status=run.status.value,
        complete=not reasons,
        incomplete_reasons=reasons,
        repository={"input_type": run.repository.input_type.value, "location": clean(run.repository.location)},
        snapshot_id=str(meta.get("snapshot_id", "")),
        created_at=str(meta.get("created_at", "")),
        finished_at=meta.get("finished_at"),
        counts_by_severity=base.counts_by_severity,
        counts_by_category=dict(sorted(Counter(f.category.value for f in findings).items())),
        verification=verification,
        findings=findings,
        unresolved=unresolved,
        tasks=tasks,
        coverage=coverage,
        budgets=budgets,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    log.info("report built run=%s complete=%s accepted=%d unresolved=%d reasons=%d",
             run_id, report.complete, len(findings), len(unresolved), len(reasons))
    return report
