import asyncio
import threading
import time
from functools import partial

import pytest

from reviewharness.ingest import ingest_repository
from reviewharness.llm import LLMTransientError
from reviewharness.llm.fake import FaultInjectingProvider, RuleBasedProvider
from reviewharness.models import RunStatus, SpecialistSource, TaskStatus
from reviewharness.orchestration import OrchestrationError, Orchestrator, RunStore, make_run_id, plan_tasks
from reviewharness.orchestration.inspect import format_run_report, peak_concurrency
from reviewharness.orchestration.planner import MAX_SCOPE_FILES, SPECIALISTS
from reviewharness.orchestration.runner import RunContext
from reviewharness.specialists import review as real_review


@pytest.fixture
def settings(make_settings):
    return make_settings(test_mode=True)


@pytest.fixture
def ingest(sample_repo, settings):
    return ingest_repository(str(sample_repo), None, settings, run_tools=False)


@pytest.fixture
def store(tmp_path):
    s = RunStore(tmp_path / "runs.db")
    yield s
    s.close()


def go(store, settings, ingest, provider=None, review_fn=real_review, **kw):
    orch = Orchestrator(store, settings, provider or RuleBasedProvider(), review_fn)
    return asyncio.run(orch.run(ingest, **kw))


def statuses(store, run_id):
    return {t.specialist.value: t.status for t in store.get_run(run_id).tasks}


def test_happy_path_completes_and_records_everything(store, settings, ingest):
    out = go(store, settings, ingest)
    assert out.status is RunStatus.COMPLETED and not out.reused
    assert set(statuses(store, out.run_id).values()) == {TaskStatus.SUCCEEDED}
    types = [e.type for e in store.events(out.run_id)]
    assert types[0] == "run_created" and types[-1] == "run_finished" and types.count("task_succeeded") == 3
    assert store.candidate_findings(out.run_id)
    assert "pending" not in {r.state for r in store.coverage(out.run_id)}


def test_specialists_really_run_concurrently(store, settings, ingest):
    lock, live, peak = threading.Lock(), [0], [0]

    def slow(*a, **k):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.3)
        with lock:
            live[0] -= 1
        return real_review(*a, **k)

    t0 = time.monotonic()
    out = go(store, settings, ingest, review_fn=slow)
    assert out.status is RunStatus.COMPLETED
    assert peak[0] == 3 and time.monotonic() - t0 < 0.8  # sequential would take >= 0.9s
    assert peak_concurrency(store.events(out.run_id)) == 3


def test_concurrency_limit_is_enforced(store, make_settings, sample_repo):
    s = make_settings(test_mode=True, max_concurrency=1)
    ing = ingest_repository(str(sample_repo), None, s, run_tools=False)
    lock, live, peak = threading.Lock(), [0], [0]

    def slow(*a, **k):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.05)
        with lock:
            live[0] -= 1
        return real_review(*a, **k)

    out = go(store, s, ing, review_fn=slow)
    assert out.status is RunStatus.COMPLETED and peak[0] == 1
    assert peak_concurrency(store.events(out.run_id)) == 1


def test_one_failed_worker_does_not_erase_the_others(store, settings, ingest):
    provider = FaultInjectingProvider(RuleBasedProvider(), fail_specialists=["security"])
    out = go(store, settings, ingest, provider)
    st = statuses(store, out.run_id)
    assert out.status is RunStatus.PARTIAL
    assert st["security"] is TaskStatus.FAILED
    assert st["bug"] is st["testing_maintainability"] is TaskStatus.SUCCEEDED
    assert store.candidate_findings(out.run_id)  # the good results survived
    sec = [t for t in store.get_run(out.run_id).tasks if t.specialist is SpecialistSource.SECURITY][0]
    assert "simulated provider failure" in sec.error
    states = {(r.specialist, r.state) for r in store.coverage(out.run_id)}
    assert ("security", "not_reviewed") in states and ("security", "reviewed") not in states
    assert "INCOMPLETE" in format_run_report(store, out.run_id)


def test_all_workers_failing_gives_failed_not_completed(store, settings, ingest):
    provider = FaultInjectingProvider(RuleBasedProvider(), fail_specialists=["bug", "security", "testing_maintainability"])
    out = go(store, settings, ingest, provider)
    assert out.status is RunStatus.FAILED and not store.candidate_findings(out.run_id)


def test_crashing_worker_is_recorded_as_failed(store, settings, ingest):
    def crashy(specialist, *a, **k):
        if specialist is SpecialistSource.BUG:
            raise RuntimeError("worker exploded")
        return real_review(specialist, *a, **k)

    out = go(store, settings, ingest, review_fn=crashy)
    assert out.status is RunStatus.PARTIAL and statuses(store, out.run_id)["bug"] is TaskStatus.FAILED
    assert "worker exploded" in [t for t in store.get_run(out.run_id).tasks if t.specialist is SpecialistSource.BUG][0].error


def test_task_timeout_marks_only_that_task(store, make_settings, sample_repo):
    s = make_settings(test_mode=True, task_timeout_s=0.1)
    ing = ingest_repository(str(sample_repo), None, s, run_tools=False)

    def slow_security(specialist, *a, **k):
        if specialist is SpecialistSource.SECURITY:
            time.sleep(0.5)
        return real_review(specialist, *a, **k)

    out = go(store, s, ing, review_fn=slow_security)
    st = statuses(store, out.run_id)
    assert out.status is RunStatus.PARTIAL and st["security"] is TaskStatus.TIMED_OUT
    assert st["bug"] is TaskStatus.SUCCEEDED
    assert not [f for f in store.candidate_findings(out.run_id) if f.source is SpecialistSource.SECURITY]


def test_run_time_budget_skips_tasks_that_never_started(store, settings, ingest):
    s = settings.model_copy(update={"max_concurrency": 1, "run_timeout_s": 0.05})

    def slow(*a, **k):
        time.sleep(0.3)
        return real_review(*a, **k)

    out = go(store, s, ingest, review_fn=slow)
    st = set(statuses(store, out.run_id).values())
    assert out.status is RunStatus.FAILED
    assert st == {TaskStatus.TIMED_OUT, TaskStatus.SKIPPED}
    assert "run time budget" in format_run_report(store, out.run_id)


def test_transient_errors_are_retried_a_bounded_number_of_times(store, make_settings, sample_repo):
    s = make_settings(test_mode=True, max_retries=1)
    ing = ingest_repository(str(sample_repo), None, s, run_tools=False)
    calls = []

    class Flaky:
        name, model = "flaky", "m"

        def complete(self, system, user):
            calls.append(1)
            raise LLMTransientError("429 rate limited")

    out = go(store, s, ing, Flaky(), review_fn=partial(real_review, sleep=lambda _: None))
    assert out.status is RunStatus.FAILED
    assert all(t.attempts == 2 for t in store.get_run(out.run_id).tasks)  # max_retries + 1, never more
    assert len(calls) == 6


def test_cancellation_keeps_finished_work_and_closes_the_rest(store, make_settings, sample_repo):
    s = make_settings(test_mode=True, max_concurrency=2)
    ing = ingest_repository(str(sample_repo), None, s, run_tools=False)

    def mixed(specialist, *a, **k):
        if specialist is not SpecialistSource.BUG:
            time.sleep(0.6)
        return real_review(specialist, *a, **k)

    async def main():
        orch = Orchestrator(store, s, RuleBasedProvider(), mixed)
        task = asyncio.create_task(orch.run(ing))
        await asyncio.sleep(0.25)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    run = store.list_runs()[0]
    assert run["status"] == "cancelled"
    st = statuses(store, run["run_id"])
    assert st["bug"] is TaskStatus.SUCCEEDED
    assert st["security"] is TaskStatus.CANCELLED and st["testing_maintainability"] is TaskStatus.CANCELLED
    assert store.candidate_findings(run["run_id"])  # partial results are kept
    assert not any(t.status in (TaskStatus.RUNNING, TaskStatus.PENDING) for t in store.get_run(run["run_id"]).tasks)


def test_duplicate_execution_of_one_task_is_prevented(store, settings, ingest):
    calls = []

    def counting(*a, **k):
        calls.append(1)
        time.sleep(0.05)
        return real_review(*a, **k)

    orch = Orchestrator(store, settings, RuleBasedProvider(), counting)
    plans = plan_tasks("R-dup", ingest.inventory, settings)
    store.create_run("R-dup", ingest.snapshot.snapshot_id, ingest.snapshot.source, plans)
    store.start_run("R-dup")
    task = plans[0].task

    async def main():
        ctx = RunContext("R-dup", ingest, time.monotonic() + 30, asyncio.Semaphore(3))
        await asyncio.gather(orch._work(ctx, task), orch._work(ctx, task))

    asyncio.run(main())
    assert len(calls) == 1
    assert "task_claim_rejected" in [e.type for e in store.events("R-dup")]


def test_identical_request_reuses_the_finished_run(store, settings, ingest):
    calls = []

    def counting(*a, **k):
        calls.append(1)
        return real_review(*a, **k)

    first = go(store, settings, ingest, review_fn=counting)
    n = len(calls)
    second = go(store, settings, ingest, review_fn=counting)
    assert second.reused and second.run_id == first.run_id and len(calls) == n
    assert len(store.list_runs()) == 1
    third = go(store, settings, ingest, review_fn=counting, new_run=True)
    assert third.run_id != first.run_id and len(store.list_runs()) == 2


def test_unfinished_run_with_same_id_is_refused(store, settings, ingest):
    run_id = make_run_id(ingest.snapshot.snapshot_id, list(SPECIALISTS), "fake", "rule-based-v1")
    store.create_run(run_id, ingest.snapshot.snapshot_id, ingest.snapshot.source, plan_tasks(run_id, ingest.inventory, settings))
    store.start_run(run_id)
    with pytest.raises(OrchestrationError, match="--new-run"):
        go(store, settings, ingest)


def test_repository_without_code_is_skipped_not_reported_complete(store, settings, tmp_path):
    repo = tmp_path / "docs_only"
    repo.mkdir()
    (repo / "README.md").write_text("# nothing to review\n")
    ing = ingest_repository(str(repo), None, settings, run_tools=False)
    out = go(store, settings, ing)
    assert set(statuses(store, out.run_id).values()) == {TaskStatus.SKIPPED}
    assert out.status is RunStatus.FAILED  # nothing was reviewed, so it must not look like success


def test_planner_bounds_scope_and_validates_tasks(ingest, settings):
    plans = plan_tasks("R-x", ingest.inventory, settings)
    assert [p.task.specialist for p in plans] == list(SPECIALISTS)
    assert all(len(p.task.scope_paths) <= MAX_SCOPE_FILES for p in plans)
    assert all(p.task.max_attempts == settings.max_retries + 1 for p in plans)
