"""Contracts for benchmark ground truth and results."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reviewharness.models import Category

SCHEMA_VERSION = 1


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExpectedIssue(_M):
    id: str = Field(min_length=1)
    file: str = Field(min_length=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    category: Category
    note: str = ""

    @model_validator(mode="after")
    def _lines(self) -> "ExpectedIssue":
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        return self


class BenchmarkSpec(_M):
    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    description: str
    expected: list[ExpectedIssue] = Field(min_length=1)
    traps: list[str] = Field(default_factory=list)  # documented false-positive traps and blind spots

    @model_validator(mode="after")
    def _unique_ids(self) -> "BenchmarkSpec":
        ids = [e.id for e in self.expected]
        if len(ids) != len(set(ids)):
            raise ValueError("expected issue ids must be unique")
        return self


class SystemScore(_M):
    system: str  # "baseline" | "multi_agent"
    ok: bool  # False: the system produced no usable result (metrics are then not measured, not zero)
    status: str
    reported: int = 0
    true_positives: int = 0
    duplicates: int = 0
    false_positives: int = 0
    matched_expected: int = 0
    precision: Optional[float] = None
    recall: Optional[float] = None
    missed: list[str] = Field(default_factory=list)
    false_positive_findings: list[str] = Field(default_factory=list)
    unresolved: int = 0  # multi-agent only: findings the checker could not settle
    complete: Optional[bool] = None  # multi-agent only: the run report's own completeness verdict
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None


class BenchmarkResult(_M):
    name: str
    expected_total: int
    baseline: SystemScore
    multi_agent: SystemScore


class AggregateScore(_M):
    system: str
    benchmarks_ok: int
    benchmarks_total: int
    incomplete_runs: int = 0
    reported: int = 0
    true_positives: int = 0
    false_positives: int = 0
    duplicates: int = 0
    expected_total: int = 0
    matched_expected: int = 0
    precision: Optional[float] = None  # over benchmarks where the system produced a result
    recall: Optional[float] = None  # over ALL expected issues; a failed benchmark counts as missed
    total_latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class BenchmarkReport(_M):
    schema_version: int = SCHEMA_VERSION
    mode: str  # "test" | "live"
    provider: str
    model: str
    line_tolerance: int
    results: list[BenchmarkResult]
    aggregate: list[AggregateScore]
    limitations: list[str]
