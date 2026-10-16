import pytest
from pydantic import ValidationError

from reviewharness.models import (
    CheckerDecision,
    CheckerStatus,
    Finding,
    InputType,
    Report,
    RepositoryInput,
    ReviewTask,
    RunRecord,
    RunStatus,
    Severity,
    SpecialistSource,
    TaskStatus,
    normalize_repo_path,
)


def finding_data(**overrides):
    data = dict(
        category="security",
        title="SQL injection",
        severity="critical",
        file_path="src/db.py",
        start_line=5,
        end_line=7,
        evidence='cur.execute("SELECT * FROM t WHERE id=" + uid)',
        rationale="User input concatenated into SQL.",
        impact="Arbitrary data access.",
        confidence=0.9,
        source="security",
    )
    data.update(overrides)
    return data


def test_valid_finding_defaults():
    f = Finding(**finding_data())
    assert f.checker_status is CheckerStatus.PENDING
    assert f.finding_id.startswith("F-")


def test_finding_id_is_deterministic_and_preserved():
    assert Finding(**finding_data()).finding_id == Finding(**finding_data()).finding_id
    assert Finding(**finding_data(finding_id="custom-1")).finding_id == "custom-1"


@pytest.mark.parametrize(
    "overrides",
    [
        {"file_path": "../etc/passwd"},
        {"file_path": "/etc/passwd"},
        {"file_path": "C:\\Windows\\x.py"},
        {"file_path": ""},
        {"start_line": 0},
        {"start_line": 9, "end_line": 3},
        {"evidence": "   "},
        {"confidence": 1.5},
        {"confidence": -0.1},
        {"severity": "catastrophic"},
        {"category": "style"},
        {"source": "nobody"},
        {"unexpected": "field"},
    ],
)
def test_invalid_finding_rejected(overrides):
    with pytest.raises(ValidationError):
        Finding(**finding_data(**overrides))


def test_missing_required_field_rejected():
    data = finding_data()
    del data["title"]
    with pytest.raises(ValidationError):
        Finding(**data)


def test_path_normalization():
    assert normalize_repo_path("./src\\a//b.py") == "src/a/b.py"
    with pytest.raises(ValueError):
        normalize_repo_path("a/../../b")


def test_severity_rank_orders_critical_first():
    assert Severity.CRITICAL.rank < Severity.HIGH.rank < Severity.INFO.rank


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(input_type="github_url", location="https://github.com/owner/repo"),
        dict(input_type="github_url", location="https://github.com/owner/repo.git"),
        dict(input_type="zip_file", location="project.ZIP"),
        dict(input_type="local_folder", location="./some/dir"),
    ],
)
def test_valid_repository_inputs(kwargs):
    assert RepositoryInput(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(input_type="github_url", location="http://github.com/owner/repo"),
        dict(input_type="github_url", location="https://evil.com/owner/repo"),
        dict(input_type="github_url", location="https://github.com/owner"),
        dict(input_type="zip_file", location="project.tar"),
        dict(input_type="local_folder", location=""),
        dict(input_type="ftp", location="x"),
    ],
)
def test_invalid_repository_inputs(kwargs):
    with pytest.raises(ValidationError):
        RepositoryInput(**kwargs)


def test_failed_task_requires_error():
    with pytest.raises(ValidationError):
        ReviewTask(task_id="t", run_id="r", specialist=SpecialistSource.BUG, status=TaskStatus.FAILED)
    task = ReviewTask(
        task_id="t", run_id="r", specialist=SpecialistSource.BUG, status=TaskStatus.FAILED, error="boom"
    )
    assert task.error == "boom"


def test_task_attempts_bounded_and_scope_validated():
    with pytest.raises(ValidationError):
        ReviewTask(task_id="t", run_id="r", specialist=SpecialistSource.BUG, attempts=4, max_attempts=3)
    with pytest.raises(ValidationError):
        ReviewTask(task_id="t", run_id="r", specialist=SpecialistSource.BUG, scope_paths=["../x"])


def test_run_rejects_duplicate_or_foreign_tasks():
    repo = RepositoryInput(input_type=InputType.LOCAL_FOLDER, location=".")
    t = ReviewTask(task_id="t", run_id="r", specialist=SpecialistSource.BUG)
    with pytest.raises(ValidationError):
        RunRecord(run_id="r", repository=repo, tasks=[t, t])
    with pytest.raises(ValidationError):
        RunRecord(run_id="other", repository=repo, tasks=[t])


def test_checker_decision_validation():
    d = CheckerDecision(finding_id="F-1", verdict="accept", reasoning="Confirmed in source.")
    assert d.attempt == 0
    with pytest.raises(ValidationError):
        CheckerDecision(finding_id="F-1", verdict="maybe", reasoning="x")
    with pytest.raises(ValidationError):
        CheckerDecision(finding_id="F-1", verdict="accept", reasoning="")


def test_report_only_accepts_accepted_findings():
    pending = Finding(**finding_data())
    with pytest.raises(ValidationError):
        Report(run_id="r", run_status=RunStatus.COMPLETED, findings=[pending])
    accepted = pending.model_copy(update={"checker_status": CheckerStatus.ACCEPTED})
    report = Report(run_id="r", run_status=RunStatus.COMPLETED, findings=[accepted])
    assert report.is_complete
    assert report.counts_by_severity["critical"] == 1


def test_report_never_labels_incomplete_run_complete():
    with pytest.raises(ValidationError):
        Report(run_id="r", run_status=RunStatus.COMPLETED, failed_task_ids=["t1"])
    partial = Report(run_id="r", run_status=RunStatus.PARTIAL, failed_task_ids=["t1"])
    assert not partial.is_complete
