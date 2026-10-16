"""Canonical data contracts shared by every component.

Everything an LLM or an input source produces must pass through these models
before downstream use.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class InputType(str, Enum):
    GITHUB_URL = "github_url"
    ZIP_FILE = "zip_file"
    LOCAL_FOLDER = "local_folder"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def rank(self) -> int:
        """Lower rank = more severe. Used for deterministic prioritization."""
        return _SEVERITY_RANK[self]


_SEVERITY_RANK = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
    Severity.INFO: 4,
}


class Category(str, Enum):
    BUG = "bug"
    SECURITY = "security"
    TESTING = "testing"
    MAINTAINABILITY = "maintainability"


class SpecialistSource(str, Enum):
    BUG = "bug"
    SECURITY = "security"
    TESTING_MAINTAINABILITY = "testing_maintainability"
    BASELINE = "baseline"  # single-agent comparison baseline


class CheckerStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    REVISION_REQUESTED = "revision_requested"
    UNCERTAIN = "uncertain"  # unresolved after the revision budget


class CheckerVerdict(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    REQUEST_REVISION = "request_revision"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"  # finished; may legitimately have found nothing
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"  # every task succeeded
    PARTIAL = "partial"  # finished, but some tasks failed/skipped/cancelled
    FAILED = "failed"
    CANCELLED = "cancelled"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

_WIN_DRIVE = re.compile(r"^[A-Za-z]:")
_GITHUB_URL = re.compile(r"^https://github\.com/[\w.-]+/[\w.-]+?(\.git)?/?$")


def normalize_repo_path(value: str) -> str:
    """Return a clean repo-relative POSIX path or raise ValueError.

    Rejects empty, absolute, drive-letter, null-byte and '..' paths so that
    nothing downstream can be pointed outside the repository snapshot.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("path must not be empty")
    if "\x00" in value:
        raise ValueError("path contains a null byte")
    path = value.strip().replace("\\", "/")
    if path.startswith("/") or _WIN_DRIVE.match(path):
        raise ValueError("path must be relative to the repository root")
    parts = [p for p in path.split("/") if p not in ("", ".")]
    if not parts:
        raise ValueError("path must not be empty")
    if ".." in parts:
        raise ValueError("path must not contain '..'")
    return "/".join(parts)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _plain(value: Any) -> str:
    return value.value if isinstance(value, Enum) else str(value)


def make_finding_id(category: Any, file_path: Any, start_line: Any, end_line: Any, title: Any) -> str:
    """Deterministic ID so identical findings collapse instead of duplicating."""
    raw = "|".join(
        [
            _plain(category).strip().lower(),
            str(file_path).strip().replace("\\", "/"),
            str(start_line),
            str(end_line),
            str(title).strip().lower(),
        ]
    )
    return "F-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


class _Strict(BaseModel):
    """Reject unknown fields so malformed LLM output fails loudly."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


class RepositoryInput(_Strict):
    input_type: InputType
    location: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_location(self) -> "RepositoryInput":
        loc = self.location
        if "\x00" in loc:
            raise ValueError("location contains a null byte")
        if self.input_type is InputType.GITHUB_URL and not _GITHUB_URL.match(loc):
            raise ValueError("github_url must look like https://github.com/<owner>/<repo>")
        if self.input_type is InputType.ZIP_FILE and not loc.lower().endswith(".zip"):
            raise ValueError("zip_file location must end with .zip")
        return self


class Finding(_Strict):
    finding_id: str = Field(default="", pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    category: Category
    title: str = Field(min_length=1, max_length=200)
    severity: Severity
    file_path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    evidence: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    impact: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    source: SpecialistSource
    checker_status: CheckerStatus = CheckerStatus.PENDING

    @model_validator(mode="before")
    @classmethod
    def _fill_finding_id(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("finding_id"):
            keys = ("category", "file_path", "start_line", "end_line", "title")
            if all(k in data for k in keys):
                data = {**data, "finding_id": make_finding_id(*(data[k] for k in keys))}
            else:
                data = {k: v for k, v in data.items() if k != "finding_id"}
        return data

    @field_validator("file_path")
    @classmethod
    def _safe_path(cls, v: str) -> str:
        return normalize_repo_path(v)

    @model_validator(mode="after")
    def _check_lines(self) -> "Finding":
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        return self


class CheckerDecision(_Strict):
    finding_id: str = Field(min_length=1)
    verdict: CheckerVerdict
    reasoning: str = Field(min_length=1)
    counterexample: Optional[str] = None
    attempt: int = Field(default=0, ge=0)
    decided_at: datetime = Field(default_factory=utcnow)


class ReviewTask(_Strict):
    task_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    specialist: SpecialistSource
    scope_paths: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=3, ge=1, le=10)
    error: Optional[str] = None

    @field_validator("scope_paths")
    @classmethod
    def _safe_scope(cls, paths: list[str]) -> list[str]:
        return [normalize_repo_path(p) for p in paths]

    @model_validator(mode="after")
    def _check_state(self) -> "ReviewTask":
        if self.attempts > self.max_attempts:
            raise ValueError("attempts must not exceed max_attempts")
        if self.status in (TaskStatus.FAILED, TaskStatus.TIMED_OUT) and not self.error:
            raise ValueError("a failed or timed-out task must record an error")
        return self


class RunRecord(_Strict):
    run_id: str = Field(min_length=1)
    repository: RepositoryInput
    status: RunStatus = RunStatus.CREATED
    created_at: datetime = Field(default_factory=utcnow)
    tasks: list[ReviewTask] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_tasks(self) -> "RunRecord":
        ids = [t.task_id for t in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate task_id in run")
        if any(t.run_id != self.run_id for t in self.tasks):
            raise ValueError("task belongs to a different run")
        return self


class Report(_Strict):
    run_id: str = Field(min_length=1)
    run_status: RunStatus
    findings: list[Finding] = Field(default_factory=list)  # checker-accepted only
    unresolved: list[Finding] = Field(default_factory=list)  # uncertain after revisions
    failed_task_ids: list[str] = Field(default_factory=list)
    skipped_task_ids: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="before")
    @classmethod
    def _drop_derived_fields(cls, data: Any) -> Any:
        # is_complete / counts_by_severity are derived on output; a saved report
        # contains them, but they must never be trusted or set from input.
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if k not in ("is_complete", "counts_by_severity")}
        return data

    @model_validator(mode="after")
    def _check_integrity(self) -> "Report":
        if any(f.checker_status is not CheckerStatus.ACCEPTED for f in self.findings):
            raise ValueError("report findings must all be checker-accepted")
        if any(f.checker_status is not CheckerStatus.UNCERTAIN for f in self.unresolved):
            raise ValueError("unresolved findings must have checker_status 'uncertain'")
        if self.run_status is RunStatus.COMPLETED and (self.failed_task_ids or self.skipped_task_ids):
            raise ValueError("a run with failed or skipped tasks cannot be labelled completed")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_complete(self) -> bool:
        return (
            self.run_status is RunStatus.COMPLETED
            and not self.failed_task_ids
            and not self.skipped_task_ids
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def counts_by_severity(self) -> dict[str, int]:
        counts = {s.value: 0 for s in Severity}
        for f in self.findings:
            counts[f.severity.value] += 1
        return counts
