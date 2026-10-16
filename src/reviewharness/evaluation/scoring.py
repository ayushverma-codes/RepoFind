"""Deterministic scoring of reported findings against ground truth.

A finding matches an expected issue when it is in the same file, in the same category group
and its line range overlaps the expected range (widened by `tolerance` lines). Matching is
one-to-one: a second finding for an already-matched issue is a duplicate, and a finding that
matches nothing is a false positive. Any real problem that is not in the ground truth also
counts as a false positive, so precision is a lower bound.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from reviewharness.evaluation.models import ExpectedIssue, SystemScore
from reviewharness.models import Category, Finding, priority_key

_GROUP = {
    Category.BUG: "bug",
    Category.SECURITY: "security",
    Category.TESTING: "quality",
    Category.MAINTAINABILITY: "quality",
}


@dataclass(frozen=True)
class MatchResult:
    matched: dict[str, str]  # expected id -> finding id
    true_positives: list[Finding]
    duplicates: list[Finding]
    false_positives: list[Finding]
    missed: list[ExpectedIssue]


def _hits(f: Finding, e: ExpectedIssue, tolerance: int) -> bool:
    return (
        f.file_path == e.file
        and _GROUP[f.category] == _GROUP[e.category]
        and f.start_line <= e.end_line + tolerance
        and e.start_line <= f.end_line + tolerance
    )


def match_findings(findings: list[Finding], expected: list[ExpectedIssue], tolerance: int = 1) -> MatchResult:
    matched: dict[str, str] = {}
    true_positives: list[Finding] = []
    duplicates: list[Finding] = []
    false_positives: list[Finding] = []
    for f in sorted(findings, key=priority_key):
        candidates = [e for e in expected if _hits(f, e, tolerance)]
        if not candidates:
            false_positives.append(f)
            continue
        free = [e for e in candidates if e.id not in matched]
        if not free:
            duplicates.append(f)
            continue
        best = min(free, key=lambda e: (abs(e.start_line - f.start_line), e.id))
        matched[best.id] = f.finding_id
        true_positives.append(f)
    missed = [e for e in expected if e.id not in matched]
    return MatchResult(matched, true_positives, duplicates, false_positives, missed)


def _ratio(num: int, den: int) -> Optional[float]:
    return round(num / den, 3) if den else None


def build_score(
    system: str,
    *,
    ok: bool,
    status: str,
    findings: list[Finding],
    expected: list[ExpectedIssue],
    tolerance: int,
    latency_s: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    unresolved: int = 0,
    complete: Optional[bool] = None,
    error: Optional[str] = None,
) -> SystemScore:
    common = dict(
        system=system, status=status, latency_s=round(latency_s, 3), prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens, unresolved=unresolved, complete=complete, error=error,
    )
    if not ok:
        # A failed system is "not measured", never "found nothing".
        return SystemScore(ok=False, missed=[e.id for e in expected], **common)
    m = match_findings(findings, expected, tolerance)
    return SystemScore(
        ok=True,
        reported=len(findings),
        true_positives=len(m.true_positives),
        duplicates=len(m.duplicates),
        false_positives=len(m.false_positives),
        matched_expected=len(m.matched),
        precision=_ratio(len(m.true_positives), len(findings)),
        recall=_ratio(len(m.matched), len(expected)),
        missed=[e.id for e in m.missed],
        false_positive_findings=[f"{f.file_path}:{f.start_line} {f.title}" for f in m.false_positives],
        **common,
    )
