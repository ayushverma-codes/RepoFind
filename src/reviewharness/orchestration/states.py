"""Explicit state machines for tasks and runs. The store enforces these in SQL."""
from __future__ import annotations

from reviewharness.models import RunStatus, TaskStatus

T = TaskStatus
R = RunStatus

TASK_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    T.PENDING: frozenset({T.RUNNING, T.SKIPPED, T.CANCELLED}),
    T.RUNNING: frozenset({T.SUCCEEDED, T.FAILED, T.TIMED_OUT, T.CANCELLED}),
    T.SUCCEEDED: frozenset(),
    T.FAILED: frozenset(),
    T.TIMED_OUT: frozenset(),
    T.CANCELLED: frozenset(),
    T.SKIPPED: frozenset(),
}

RUN_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    R.CREATED: frozenset({R.RUNNING, R.FAILED, R.CANCELLED}),
    R.RUNNING: frozenset({R.COMPLETED, R.PARTIAL, R.FAILED, R.CANCELLED}),
    R.COMPLETED: frozenset(),
    R.PARTIAL: frozenset(),
    R.FAILED: frozenset(),
    R.CANCELLED: frozenset(),
}

TERMINAL_RUN_STATES = frozenset(s for s, nxt in RUN_TRANSITIONS.items() if not nxt)


def task_sources(new: TaskStatus) -> frozenset[TaskStatus]:
    """States a task may move to `new` from."""
    return frozenset(s for s, nxt in TASK_TRANSITIONS.items() if new in nxt)


def run_sources(new: RunStatus) -> frozenset[RunStatus]:
    return frozenset(s for s, nxt in RUN_TRANSITIONS.items() if new in nxt)


def derive_run_status(task_statuses: list[TaskStatus], *, cancelled: bool = False) -> RunStatus:
    """Accurate run status from task outcomes. Never COMPLETED unless every task succeeded."""
    if cancelled:
        return R.CANCELLED
    if task_statuses and all(s is T.SUCCEEDED for s in task_statuses):
        return R.COMPLETED
    if any(s is T.SUCCEEDED for s in task_statuses):
        return R.PARTIAL
    return R.FAILED
