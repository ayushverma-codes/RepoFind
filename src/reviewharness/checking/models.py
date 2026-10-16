"""Contracts for the checking stage."""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from reviewharness.models import CheckerDecision, CheckerStatus, CheckerVerdict, Finding, Severity


class OutcomeKind(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"  # checker (or the specialist) refuted it
    UNCERTAIN = "uncertain"  # unresolved: revision budget spent, or the check could not be completed
    INVALID = "invalid"  # failed deterministic validation; never reached the checker
    DUPLICATE = "duplicate"  # overlaps a better finding; never reached the checker


# Outcomes with these codes mean "we could not finish checking", so the run cannot be called complete.
INCOMPLETE_CODES = frozenset({"checker_error", "checker_inconsistent", "check_timeout", "check_budget", "run_deadline"})


class CheckerReply(BaseModel):
    """What the checker model must return. Anything else is rejected and retried."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    verdict: CheckerVerdict
    reasoning: str = Field(min_length=1, max_length=1500)
    counterexample: Optional[str] = Field(default=None, max_length=1500)
    suggested_severity: Optional[Severity] = None  # may only lower severity; enforced by policy

    @field_validator("suggested_severity", "counterexample", mode="before")
    @classmethod
    def _blank_is_none(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip().lower() in ("", "none", "null", "n/a"):
            return None
        return v


class FindingOutcome(BaseModel):
    """The recorded verification result for one candidate finding."""

    model_config = ConfigDict(extra="forbid")

    finding: Finding  # final state (checker_status set)
    original: Optional[Finding] = None  # as first proposed, when it was revised
    kind: OutcomeKind
    reason_code: str = Field(min_length=1)
    reason: str = Field(default="", max_length=600)
    revisions: int = Field(default=0, ge=0)
    decisions: list[CheckerDecision] = Field(default_factory=list)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _invariants(self) -> "FindingOutcome":
        if self.kind is OutcomeKind.ACCEPTED:
            if self.finding.checker_status is not CheckerStatus.ACCEPTED:
                raise ValueError("accepted outcome needs checker_status accepted")
            if not self.decisions or self.decisions[-1].verdict is not CheckerVerdict.ACCEPT:
                raise ValueError("an accepted finding needs a recorded checker accept decision")
        if self.kind is OutcomeKind.UNCERTAIN and self.finding.checker_status is not CheckerStatus.UNCERTAIN:
            raise ValueError("uncertain outcome needs checker_status uncertain")
        return self


class CheckSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    counts: dict[str, int] = Field(default_factory=dict)
    incomplete: bool = False
