"""SQLite run store: durable run/task state, coverage ledger, candidate findings, event history.

Rules enforced here (not left to callers):
  * every state change is one conditional UPDATE inside a transaction together with its event,
    so state and history can never disagree;
  * a task can be claimed once: the second claim updates zero rows and is refused;
  * `events` is append-only (triggers abort UPDATE and DELETE);
  * candidate findings are keyed (run_id, finding_id), so a repeated finding cannot duplicate;
  * one failed task never touches another task's rows.

All calls are made from the event-loop thread; worker threads never touch the connection.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from reviewharness.logging_setup import redact
from reviewharness.models import (
    Finding,
    RepositoryInput,
    ReviewTask,
    RunRecord,
    RunStatus,
    TaskStatus,
    utcnow,
)
from reviewharness.orchestration.models import CoverageRow, PlannedTask, RunEvent
from reviewharness.orchestration.states import run_sources, task_sources
from reviewharness.specialists.models import SpecialistResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    snapshot_id TEXT NOT NULL,
    repository_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    specialist TEXT NOT NULL,
    scope_json TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL,
    error TEXT,
    started_at TEXT,
    finished_at TEXT,
    result_json TEXT,
    UNIQUE (run_id, specialist)
);
CREATE TABLE IF NOT EXISTS coverage (
    task_id TEXT NOT NULL REFERENCES tasks(task_id),
    path TEXT NOT NULL,
    state TEXT NOT NULL,
    PRIMARY KEY (task_id, path)
);
CREATE TABLE IF NOT EXISTS candidate_findings (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    finding_id TEXT NOT NULL,
    task_id TEXT NOT NULL REFERENCES tasks(task_id),
    finding_json TEXT NOT NULL,
    PRIMARY KEY (run_id, finding_id)
);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    task_id TEXT,
    ts TEXT NOT NULL,
    type TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, seq);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
"""

_TASK_EVENT = {
    TaskStatus.SUCCEEDED: "task_succeeded",
    TaskStatus.FAILED: "task_failed",
    TaskStatus.TIMED_OUT: "task_timed_out",
    TaskStatus.CANCELLED: "task_cancelled",
    TaskStatus.SKIPPED: "task_skipped",
}
_ERROR_MAX = 300


class InvalidTransition(Exception):
    """A state change that the state machine forbids (or that lost a race)."""


def _now() -> str:
    return utcnow().isoformat()


def _in(states) -> tuple[str, list[str]]:
    values = sorted(s.value for s in states)
    return ",".join("?" for _ in values), values


class RunStore:
    def __init__(self, db_path: Path) -> None:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), timeout=5.0, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    def _emit(self, run_id: str, task_id: Optional[str], type_: str, payload: Optional[dict[str, Any]] = None) -> None:
        text = redact(json.dumps(payload or {}, sort_keys=True, default=str))
        self._conn.execute(
            "INSERT INTO events (run_id, task_id, ts, type, payload_json) VALUES (?,?,?,?,?)",
            (run_id, task_id, _now(), type_, text),
        )

    # ----------------------------------------------------------------- runs

    def create_run(self, run_id: str, snapshot_id: str, repository: RepositoryInput, plans: list[PlannedTask]) -> bool:
        """Create the run, its tasks and its coverage ledger. False if the run already exists."""
        with self._tx() as c:
            if c.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone():
                self._emit(run_id, None, "run_reuse_requested", {"snapshot_id": snapshot_id})
                return False
            c.execute(
                "INSERT INTO runs (run_id, snapshot_id, repository_json, status, created_at) VALUES (?,?,?,?,?)",
                (run_id, snapshot_id, repository.model_dump_json(), RunStatus.CREATED.value, _now()),
            )
            self._emit(run_id, None, "run_created", {"snapshot_id": snapshot_id, "tasks": len(plans)})
            for plan in plans:
                t = plan.task
                c.execute(
                    "INSERT INTO tasks (task_id, run_id, specialist, scope_json, status, max_attempts) VALUES (?,?,?,?,?,?)",
                    (t.task_id, run_id, t.specialist.value, json.dumps(t.scope_paths), TaskStatus.PENDING.value, t.max_attempts),
                )
                c.executemany(
                    "INSERT INTO coverage (task_id, path, state) VALUES (?,?,?)",
                    [(t.task_id, p, "pending") for p in t.scope_paths] + [(t.task_id, p, "over_scope_cap") for p in plan.omitted],
                )
                self._emit(
                    run_id, t.task_id, "task_planned",
                    {"specialist": t.specialist.value, "files": len(t.scope_paths), "over_scope_cap": len(plan.omitted)},
                )
        return True

    def _set_run_status(self, run_id: str, new: RunStatus, type_: str, payload: Optional[dict] = None) -> None:
        marks, values = _in(run_sources(new))
        finished = _now() if new not in (RunStatus.RUNNING,) else None
        with self._tx() as c:
            cur = c.execute(
                f"UPDATE runs SET status=?, finished_at=COALESCE(?, finished_at) WHERE run_id=? AND status IN ({marks})",
                (new.value, finished, run_id, *values),
            )
            if cur.rowcount != 1:
                raise InvalidTransition(f"run {run_id} cannot move to {new.value}")
            self._emit(run_id, None, type_, payload)

    def start_run(self, run_id: str) -> None:
        self._set_run_status(run_id, RunStatus.RUNNING, "run_started")

    def finish_run(self, run_id: str, status: RunStatus, payload: Optional[dict] = None) -> None:
        self._set_run_status(run_id, status, "run_finished", {"status": status.value, **(payload or {})})

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        row = self._conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            return None
        tasks = [
            ReviewTask(
                task_id=r["task_id"], run_id=run_id, specialist=r["specialist"],
                scope_paths=json.loads(r["scope_json"]), status=r["status"], attempts=r["attempts"],
                max_attempts=r["max_attempts"], error=r["error"],
            )
            for r in self._conn.execute("SELECT * FROM tasks WHERE run_id=? ORDER BY task_id", (run_id,))
        ]
        return RunRecord(
            run_id=run_id, repository=RepositoryInput.model_validate_json(row["repository_json"]),
            status=row["status"], created_at=row["created_at"], tasks=tasks,
        )

    def run_meta(self, run_id: str) -> Optional[dict[str, Any]]:
        row = self._conn.execute("SELECT snapshot_id, created_at, finished_at FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def list_runs(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT run_id, snapshot_id, status, created_at FROM runs ORDER BY created_at DESC, run_id")
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------- tasks

    def claim_task(self, task_id: str) -> bool:
        """PENDING -> RUNNING exactly once. False means another worker already owns the task."""
        marks, values = _in(task_sources(TaskStatus.RUNNING))
        with self._tx() as c:
            row = c.execute("SELECT run_id FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is None:
                raise InvalidTransition(f"unknown task {task_id}")
            cur = c.execute(
                f"UPDATE tasks SET status=?, started_at=? WHERE task_id=? AND status IN ({marks})",
                (TaskStatus.RUNNING.value, _now(), task_id, *values),
            )
            if cur.rowcount != 1:
                self._emit(row["run_id"], task_id, "task_claim_rejected", {"reason": "already claimed or finished"})
                return False
            self._emit(row["run_id"], task_id, "task_started", {})
            return True

    def skip_task(self, task_id: str, reason: str) -> None:
        self._finish(task_id, TaskStatus.SKIPPED, error=reason)

    def finish_task(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        error: Optional[str] = None,
        result: Optional[SpecialistResult] = None,
    ) -> None:
        """Record the end of a RUNNING task. Findings and coverage are written in the same transaction."""
        if status is TaskStatus.SUCCEEDED and result is None:
            raise ValueError("a succeeded task needs its result")
        if status in (TaskStatus.FAILED, TaskStatus.TIMED_OUT) and not error:
            raise ValueError("a failed or timed-out task needs an error")
        self._finish(task_id, status, error=error, result=result)

    def _finish(self, task_id: str, status: TaskStatus, *, error: Optional[str], result: Optional[SpecialistResult] = None) -> None:
        marks, values = _in(task_sources(status))
        error = redact(error)[:_ERROR_MAX] if error else None
        attempts = result.attempts if result else 0
        with self._tx() as c:
            row = c.execute("SELECT run_id FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is None:
                raise InvalidTransition(f"unknown task {task_id}")
            run_id = row["run_id"]
            cur = c.execute(
                f"UPDATE tasks SET status=?, error=?, attempts=?, finished_at=?, result_json=? "
                f"WHERE task_id=? AND status IN ({marks})",
                (status.value, error, attempts, _now(), result.model_dump_json() if result else None, task_id, *values),
            )
            if cur.rowcount != 1:
                raise InvalidTransition(f"task {task_id} cannot move to {status.value}")
            payload: dict[str, Any] = {"status": status.value}
            if error:
                payload["error"] = error
            if result is not None:
                payload.update(attempts=result.attempts, prompt_tokens=result.prompt_tokens,
                               completion_tokens=result.completion_tokens, duration_s=result.duration_s)
            if status is TaskStatus.SUCCEEDED and result is not None:
                cov = result.coverage
                for state, paths in (("reviewed", cov.reviewed), ("truncated", cov.truncated), ("omitted", cov.omitted)):
                    c.executemany(
                        "UPDATE coverage SET state=? WHERE task_id=? AND path=? AND state='pending'",
                        [(state, task_id, p) for p in paths],
                    )
                payload["coverage"] = {"reviewed": len(cov.reviewed), "truncated": len(cov.truncated), "omitted": len(cov.omitted)}
                new_ids, dup_ids = [], []
                for f in result.findings:
                    cur = c.execute(
                        "INSERT OR IGNORE INTO candidate_findings (run_id, finding_id, task_id, finding_json) VALUES (?,?,?,?)",
                        (run_id, f.finding_id, task_id, f.model_dump_json()),
                    )
                    (new_ids if cur.rowcount == 1 else dup_ids).append(f.finding_id)
                payload.update(findings=len(new_ids), duplicate_findings=dup_ids, rejected=len(result.rejected))
            # Whatever the task did not actually review stays visible as not_reviewed.
            c.execute("UPDATE coverage SET state='not_reviewed' WHERE task_id=? AND state='pending'", (task_id,))
            self._emit(run_id, task_id, _TASK_EVENT[status], payload)

    def sweep_unfinished(self, run_id: str, reason: str, *, cancelled: bool) -> list[str]:
        """Close tasks left PENDING/RUNNING (cancel, crash). Finished tasks are never touched."""
        swept: list[str] = []
        rows = self._conn.execute(
            "SELECT task_id, status FROM tasks WHERE run_id=? AND status IN ('pending','running') ORDER BY task_id", (run_id,)
        ).fetchall()
        for r in rows:
            if r["status"] == TaskStatus.PENDING.value:
                new = TaskStatus.CANCELLED if cancelled else TaskStatus.SKIPPED
            else:
                new = TaskStatus.CANCELLED if cancelled else TaskStatus.FAILED
            try:
                self._finish(r["task_id"], new, error=reason)
                swept.append(r["task_id"])
            except InvalidTransition:
                continue  # finished concurrently; leave it alone
        return swept

    # ------------------------------------------------------------- read side

    def events(self, run_id: str) -> list[RunEvent]:
        rows = self._conn.execute("SELECT * FROM events WHERE run_id=? ORDER BY seq", (run_id,))
        return [
            RunEvent(seq=r["seq"], run_id=r["run_id"], task_id=r["task_id"], ts=r["ts"], type=r["type"],
                     payload=json.loads(r["payload_json"]))
            for r in rows
        ]

    def coverage(self, run_id: str) -> list[CoverageRow]:
        rows = self._conn.execute(
            "SELECT c.task_id, t.specialist, c.path, c.state FROM coverage c JOIN tasks t ON t.task_id=c.task_id "
            "WHERE t.run_id=? ORDER BY c.task_id, c.path", (run_id,),
        )
        return [CoverageRow(task_id=r["task_id"], specialist=r["specialist"], path=r["path"], state=r["state"]) for r in rows]

    def candidate_findings(self, run_id: str) -> list[Finding]:
        rows = self._conn.execute("SELECT finding_json FROM candidate_findings WHERE run_id=? ORDER BY finding_id", (run_id,))
        return [Finding.model_validate_json(r["finding_json"]) for r in rows]

    def task_results(self, run_id: str) -> dict[str, SpecialistResult]:
        rows = self._conn.execute("SELECT task_id, result_json FROM tasks WHERE run_id=? AND result_json IS NOT NULL", (run_id,))
        return {r["task_id"]: SpecialistResult.model_validate_json(r["result_json"]) for r in rows}
