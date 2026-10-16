"""Small contracts used only by orchestration."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from reviewharness.models import ReviewTask


class PlannedTask(BaseModel):
    """A validated task plus planning facts that the store records in the coverage ledger."""

    model_config = ConfigDict(extra="forbid")

    task: ReviewTask
    omitted: list[str] = Field(default_factory=list)  # in scope but cut by the scope cap
    skip_reason: Optional[str] = None  # set when there is nothing to review


class RunEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seq: int
    run_id: str
    task_id: Optional[str] = None
    ts: datetime
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class CoverageRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    specialist: str
    path: str
    state: str  # pending | reviewed | truncated | omitted | not_reviewed | over_scope_cap
