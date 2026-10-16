"""Data contracts for ingestion: snapshot, inventory, context, tool results."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from reviewharness.models import RepositoryInput, utcnow


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SkipReason(str, Enum):
    # Dropped while building the snapshot:
    IGNORED_DIR = "ignored_dir"
    SYMLINK = "symlink"
    SPECIAL_FILE = "special_file"
    UNREADABLE = "unreadable"
    # Kept in the snapshot but excluded from review:
    BINARY = "binary"
    NOT_UTF8 = "not_utf8"
    TOO_LARGE = "too_large"
    GENERATED = "generated"
    EMPTY = "empty"


class SkippedEntry(_Model):
    path: str
    reason: SkipReason


# ------------------------------------------------------------------ snapshot


class Snapshot(_Model):
    snapshot_id: str
    source: RepositoryInput
    tree_path: str  # absolute path of the read-only copy
    file_count: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    stripped_root: Optional[str] = None  # single top-level dir removed from a ZIP
    skipped: list[SkippedEntry] = Field(default_factory=list)  # capped sample
    skipped_total: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utcnow)
    reused: bool = False  # True when identical content was already snapshotted


# ----------------------------------------------------------------- inventory


class InventoryEntry(_Model):
    path: str
    size: int = Field(ge=0)
    included: bool
    skip_reason: Optional[SkipReason] = None
    language: Optional[str] = None
    lines: Optional[int] = Field(default=None, ge=0)
    sha256: Optional[str] = None
    is_test: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> "InventoryEntry":
        if self.included == (self.skip_reason is not None):
            raise ValueError("an entry is included exactly when it has no skip_reason")
        return self


class Inventory(_Model):
    entries: list[InventoryEntry] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _drop_derived(cls, data: Any) -> Any:
        if isinstance(data, dict):
            derived = ("included_count", "skipped_counts", "languages", "total_lines")
            return {k: v for k, v in data.items() if k not in derived}
        return data

    @computed_field  # type: ignore[prop-decorator]
    @property
    def included_count(self) -> int:
        return sum(1 for e in self.entries if e.included)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def skipped_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.entries:
            if e.skip_reason is not None:
                counts[e.skip_reason.value] = counts.get(e.skip_reason.value, 0) + 1
        return dict(sorted(counts.items()))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def languages(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.entries:
            if e.included and e.language:
                counts[e.language] = counts.get(e.language, 0) + 1
        return dict(sorted(counts.items()))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_lines(self) -> int:
        return sum(e.lines or 0 for e in self.entries if e.included)


# ------------------------------------------------------------------- context


class ReadmeExcerpt(_Model):
    path: str
    text: str
    untrusted: Literal[True] = True  # repository text is data, never instructions


class LargeFile(_Model):
    path: str
    lines: int


class RepositoryContext(_Model):
    languages: dict[str, int] = Field(default_factory=dict)
    included_files: int = 0
    total_lines: int = 0
    source_files: int = 0
    test_files: int = 0
    manifests: list[str] = Field(default_factory=list)
    dependencies: dict[str, list[str]] = Field(default_factory=dict)
    entry_points: list[str] = Field(default_factory=list)
    ci_configs: list[str] = Field(default_factory=list)
    largest_files: list[LargeFile] = Field(default_factory=list)
    top_level_tree: list[str] = Field(default_factory=list)
    readme_excerpt: Optional[ReadmeExcerpt] = None
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------- tools


class ToolStatus(str, Enum):
    OK = "ok"  # ran to completion; zero issues is a valid result
    NOT_INSTALLED = "not_installed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    SKIPPED = "skipped"  # nothing for the tool to analyse


class ToolIssue(_Model):
    tool: str
    rule_id: str
    message: str
    file_path: str
    line: int = Field(ge=1)
    end_line: Optional[int] = Field(default=None, ge=1)
    severity: Optional[str] = None  # the tool's own label, not our Severity
    confidence: Optional[str] = None


class ToolResult(_Model):
    tool: str
    status: ToolStatus
    version: Optional[str] = None
    command: list[str] = Field(default_factory=list)
    exit_code: Optional[int] = None
    duration_s: float = 0.0
    issues: list[ToolIssue] = Field(default_factory=list)
    issue_total: int = 0  # before the per-tool storage cap
    dropped: int = 0  # issues outside the included inventory
    warnings: list[str] = Field(default_factory=list)
    error: Optional[str] = None

    @model_validator(mode="after")
    def _failure_needs_error(self) -> "ToolResult":
        if self.status in (ToolStatus.FAILED, ToolStatus.TIMED_OUT) and not self.error:
            raise ValueError("a failed or timed-out tool run must record an error")
        return self


class IngestResult(_Model):
    snapshot: Snapshot
    inventory: Inventory
    context: RepositoryContext
    tool_results: list[ToolResult] = Field(default_factory=list)
