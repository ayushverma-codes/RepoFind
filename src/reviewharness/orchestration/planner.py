"""Lead/planner. Deterministic on purpose: scope selection needs no LLM and must be reproducible."""
from __future__ import annotations

import hashlib

from reviewharness.config import Settings
from reviewharness.ingest.models import Inventory
from reviewharness.models import ReviewTask, SpecialistSource
from reviewharness.orchestration.models import PlannedTask
from reviewharness.specialists.excerpts import select_scope

MAX_SCOPE_FILES = 100  # hard bound on files assigned to one task

SPECIALISTS = (
    SpecialistSource.BUG,
    SpecialistSource.SECURITY,
    SpecialistSource.TESTING_MAINTAINABILITY,
)


def make_run_id(snapshot_id: str, specialists: list[SpecialistSource], provider_name: str, model: str, nonce: str = "") -> str:
    """Same snapshot + plan + provider -> same run id, so a repeated request is detected, not duplicated."""
    raw = "|".join([snapshot_id, ",".join(sorted(s.value for s in specialists)), provider_name, model, nonce])
    return "R-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def plan_tasks(run_id: str, inventory: Inventory, settings: Settings, specialists=SPECIALISTS) -> list[PlannedTask]:
    plans: list[PlannedTask] = []
    for specialist in specialists:
        scope = select_scope(inventory, specialist)
        kept, omitted = scope[:MAX_SCOPE_FILES], scope[MAX_SCOPE_FILES:]
        task = ReviewTask(
            task_id=f"{run_id}:{specialist.value}",
            run_id=run_id,
            specialist=specialist,
            scope_paths=kept,
            max_attempts=settings.max_retries + 1,
        )
        reason = None if kept else "no reviewable source files in scope"
        plans.append(PlannedTask(task=task, omitted=omitted, skip_reason=reason))
    return plans
