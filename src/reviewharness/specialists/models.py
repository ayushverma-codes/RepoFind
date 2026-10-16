"""Contracts for a specialist run. A failed run and an empty successful run differ by `status`."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reviewharness.models import Finding, SpecialistSource, TaskStatus


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Excerpt(_M):
    """Source text a specialist was actually shown. Always starts at line 1."""

    path: str
    total_lines: int = Field(ge=1)
    shown_lines: int = Field(ge=1)
    text: str  # the shown lines, un-numbered, joined with \n

    @property
    def truncated(self) -> bool:
        return self.shown_lines < self.total_lines


class Coverage(_M):
    reviewed: list[str] = Field(default_factory=list)  # shown in full
    truncated: list[str] = Field(default_factory=list)  # shown partially
    omitted: list[str] = Field(default_factory=list)  # in scope but not shown (budget/unreadable)


class RejectedFinding(_M):
    """A model-proposed finding dropped by deterministic validation, with the reason."""

    title: str
    reason: str


class SpecialistResult(_M):
    specialist: SpecialistSource
    status: TaskStatus
    provider: str
    model: str
    findings: list[Finding] = Field(default_factory=list)
    rejected: list[RejectedFinding] = Field(default_factory=list)
    coverage: Coverage = Field(default_factory=Coverage)
    attempts: int = Field(ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    duration_s: float = Field(default=0.0, ge=0)
    error: Optional[str] = None

    @model_validator(mode="after")
    def _check(self) -> "SpecialistResult":
        if self.status not in (TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.TIMED_OUT):
            raise ValueError("a specialist run ends as succeeded, failed or timed_out")
        if self.status is TaskStatus.SUCCEEDED and self.error:
            raise ValueError("a succeeded run must not carry an error")
        if self.status is not TaskStatus.SUCCEEDED and not self.error:
            raise ValueError("a failed or timed-out run must record an error")
        if self.status is not TaskStatus.SUCCEEDED and self.findings:
            raise ValueError("a failed run cannot report findings")
        return self
