import sqlite3

import pytest

from reviewharness.models import (
    Finding, InputType, RepositoryInput, ReviewTask, RunStatus, SpecialistSource, TaskStatus,
)
from reviewharness.orchestration import InvalidTransition, PlannedTask, RunStore
from reviewharness.orchestration.states import TASK_TRANSITIONS, derive_run_status
from reviewharness.specialists.models import Coverage, SpecialistResult

REPO = RepositoryInput(input_type=InputType.LOCAL_FOLDER, location="./x")


def _plan(run_id="R-1", specialist=SpecialistSource.BUG, files=("a.py", "b.py"), omitted=()):
    task = ReviewTask(task_id=f"{run_id}:{specialist.value}", run_id=run_id, specialist=specialist, scope_paths=list(files))
    return PlannedTask(task=task, omitted=list(omitted))


def _finding(title="Bare except", line=3):
    return Finding(category="bug", title=title, severity="low", file_path="a.py", start_line=line, end_line=line,
                   evidence="except:", rationale="r", impact="i", confidence=0.5, source="bug")


def _result(findings=(), reviewed=("a.py",), omitted=("b.py",)):
    return SpecialistResult(specialist=SpecialistSource.BUG, status=TaskStatus.SUCCEEDED, provider="fake", model="m",
                            findings=list(findings), coverage=Coverage(reviewed=list(reviewed), omitted=list(omitted)), attempts=1)


@pytest.fixture
def store(tmp_path):
    s = RunStore(tmp_path / "runs.db")
    yield s
    s.close()


def test_create_run_is_idempotent(store):
    assert store.create_run("R-1", "S-1", REPO, [_plan()]) is True
    assert store.create_run("R-1", "S-1", REPO, [_plan()]) is False
    assert len(store.list_runs()) == 1
    assert [e.type for e in store.events("R-1")] == ["run_created", "task_planned", "run_reuse_requested"]


def test_task_can_be_claimed_only_once(store):
    store.create_run("R-1", "S-1", REPO, [_plan()])
    assert store.claim_task("R-1:bug") is True
    assert store.claim_task("R-1:bug") is False
    types = [e.type for e in store.events("R-1")]
    assert types.count("task_started") == 1 and "task_claim_rejected" in types


def test_illegal_transitions_are_refused(store):
    store.create_run("R-1", "S-1", REPO, [_plan()])
    with pytest.raises(InvalidTransition):  # PENDING -> SUCCEEDED skips RUNNING
        store.finish_task("R-1:bug", TaskStatus.SUCCEEDED, result=_result())
    store.claim_task("R-1:bug")
    store.finish_task("R-1:bug", TaskStatus.FAILED, error="boom")
    with pytest.raises(InvalidTransition):  # terminal states never change
        store.finish_task("R-1:bug", TaskStatus.SUCCEEDED, result=_result())
    assert store.get_run("R-1").tasks[0].status is TaskStatus.FAILED


def test_terminal_states_have_no_exits():
    for status in (TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.TIMED_OUT, TaskStatus.CANCELLED, TaskStatus.SKIPPED):
        assert TASK_TRANSITIONS[status] == frozenset()


def test_failed_task_requires_error(store):
    store.create_run("R-1", "S-1", REPO, [_plan()])
    store.claim_task("R-1:bug")
    with pytest.raises(ValueError):
        store.finish_task("R-1:bug", TaskStatus.FAILED)


def test_events_are_append_only(store):
    store.create_run("R-1", "S-1", REPO, [_plan()])
    with pytest.raises(sqlite3.DatabaseError):
        store._conn.execute("UPDATE events SET type='x'")
    with pytest.raises(sqlite3.DatabaseError):
        store._conn.execute("DELETE FROM events")
    assert len(store.events("R-1")) == 2


def test_success_records_coverage_and_dedupes_findings(store):
    store.create_run("R-1", "S-1", REPO, [_plan(omitted=["z.py"])])
    store.claim_task("R-1:bug")
    f = _finding()
    store.finish_task("R-1:bug", TaskStatus.SUCCEEDED, result=_result([f, f]))
    states = {r.path: r.state for r in store.coverage("R-1")}
    assert states == {"a.py": "reviewed", "b.py": "omitted", "z.py": "over_scope_cap"}
    assert len(store.candidate_findings("R-1")) == 1
    done = [e for e in store.events("R-1") if e.type == "task_succeeded"][0]
    assert done.payload["findings"] == 1 and done.payload["duplicate_findings"] == [f.finding_id]


def test_failed_task_coverage_is_not_reviewed_even_if_result_claims_otherwise(store):
    store.create_run("R-1", "S-1", REPO, [_plan()])
    store.claim_task("R-1:bug")
    failed = SpecialistResult(specialist=SpecialistSource.BUG, status=TaskStatus.FAILED, provider="fake", model="m",
                              coverage=Coverage(reviewed=["a.py"]), attempts=1, error="x")
    store.finish_task("R-1:bug", TaskStatus.FAILED, error="x", result=failed)
    assert {r.state for r in store.coverage("R-1")} == {"not_reviewed"}


def test_events_redact_secrets(store):
    store.create_run("R-1", "S-1", REPO, [_plan()])
    store.claim_task("R-1:bug")
    store.finish_task("R-1:bug", TaskStatus.FAILED, error="auth failed api_key=gsk_abcdefghijklmnop1234")
    text = " ".join(str(e.payload) for e in store.events("R-1")) + (store.get_run("R-1").tasks[0].error or "")
    assert "gsk_abcdefghijklmnop1234" not in text


def test_run_status_transitions(store):
    store.create_run("R-1", "S-1", REPO, [_plan()])
    with pytest.raises(InvalidTransition):
        store.finish_run("R-1", RunStatus.COMPLETED)  # not started yet
    store.start_run("R-1")
    store.finish_run("R-1", RunStatus.PARTIAL)
    with pytest.raises(InvalidTransition):
        store.finish_run("R-1", RunStatus.COMPLETED)


@pytest.mark.parametrize("statuses, cancelled, expected", [
    ([TaskStatus.SUCCEEDED] * 3, False, RunStatus.COMPLETED),
    ([TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.SUCCEEDED], False, RunStatus.PARTIAL),
    ([TaskStatus.SUCCEEDED, TaskStatus.SKIPPED, TaskStatus.TIMED_OUT], False, RunStatus.PARTIAL),
    ([TaskStatus.FAILED] * 3, False, RunStatus.FAILED),
    ([TaskStatus.SKIPPED] * 3, False, RunStatus.FAILED),
    ([TaskStatus.SUCCEEDED, TaskStatus.CANCELLED], True, RunStatus.CANCELLED),
])
def test_derive_run_status(statuses, cancelled, expected):
    assert derive_run_status(statuses, cancelled=cancelled) is expected
