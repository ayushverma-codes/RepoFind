"""Report contract. `complete` is derived by the builder from recorded facts, never set by callers."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from reviewharness.models import Finding

SCHEMA_VERSION = 1


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskSummary(_M):
    specialist: str
    status: str
    attempts: int
    max_attempts: int
    findings: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_s: float = 0.0
    error: Optional[str] = None


class CoverageGap(_M):
    specialist: str
    path: str
    state: str


class CoverageSummary(_M):
    by_specialist: dict[str, dict[str, int]] = Field(default_factory=dict)
    gaps: list[CoverageGap] = Field(default_factory=list)  # capped list; gaps_total is exact
    gaps_total: int = 0


class UnresolvedFinding(_M):
    finding: Finding
    reason_code: str
    reason: str
    revisions: int = 0


class VerificationSummary(_M):
    candidates: int = 0
    accepted: int = 0
    unresolved: int = 0
    rejected: int = 0
    invalid: int = 0
    duplicate: int = 0
    unchecked: int = 0  # candidates with no recorded outcome
    checks_incomplete: int = 0  # unresolved only because checking could not finish


class Budgets(_M):
    configured: Optional[dict[str, Any]] = None  # None: the run predates budget recording
    observed: dict[str, Any] = Field(default_factory=dict)
    limits_hit: list[str] = Field(default_factory=list)


class RunReport(_M):
    schema_version: int = SCHEMA_VERSION
    run_id: str
    run_status: str
    complete: bool
    incomplete_reasons: list[str] = Field(default_factory=list)
    repository: dict[str, str]
    snapshot_id: str
    created_at: str
    finished_at: Optional[str] = None
    counts_by_severity: dict[str, int] = Field(default_factory=dict)  # accepted findings only
    counts_by_category: dict[str, int] = Field(default_factory=dict)
    verification: VerificationSummary
    findings: list[Finding] = Field(default_factory=list)  # accepted, priority order
    unresolved: list[UnresolvedFinding] = Field(default_factory=list)
    tasks: list[TaskSummary] = Field(default_factory=list)
    coverage: CoverageSummary
    budgets: Budgets
    prompt_tokens: int = 0
    completion_tokens: int = 0
