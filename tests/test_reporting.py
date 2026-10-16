import json

import pytest

from reviewharness.cli import main
from reviewharness.models import Category, Finding, Severity, priority_key
from reviewharness.orchestration import RunStore
from reviewharness.reporting import build_run_report, clean, render_json, render_markdown
from reviewharness.reporting.render import code, fence, inline


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setattr("reviewharness.cli.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("REVIEW_TEST_MODE", raising=False)
    monkeypatch.delenv("REVIEW_MAX_TOKENS", raising=False)
    monkeypatch.delenv("REVIEW_MAX_CONCURRENCY", raising=False)
    monkeypatch.setenv("REVIEW_DATA_DIR", str(tmp_path / "data"))


def run_id_of(capsys, repo, *extra):
    main(["run", str(repo), "--test-mode", "--json", *extra])
    return json.loads(capsys.readouterr().out)["run_id"]


def report_json(capsys, run_id):
    assert main(["report", run_id, "--format", "json"]) == 0
    return json.loads(capsys.readouterr().out)


def make(sev, conf, cat=Category.BUG, path="a.py", line=1, title="t"):
    return Finding(category=cat, title=title, severity=sev, file_path=path, start_line=line, end_line=line,
                   evidence="x", rationale="r", impact="i", confidence=conf, source="bug")


# ---------------------------------------------------------------- prioritization


def test_priority_order_is_severity_then_confidence_then_category_then_location():
    items = [
        make(Severity.LOW, 0.9, title="low"),
        make(Severity.HIGH, 0.5, title="high-lowconf"),
        make(Severity.HIGH, 0.9, Category.TESTING, title="high-testing"),
        make(Severity.HIGH, 0.9, Category.SECURITY, title="high-security"),
        make(Severity.CRITICAL, 0.1, title="critical"),
        make(Severity.HIGH, 0.9, Category.BUG, "a.py", 7, title="high-bug-later"),
        make(Severity.HIGH, 0.9, Category.BUG, "a.py", 2, title="high-bug-early"),
    ]
    got = [f.title for f in sorted(items, key=priority_key)]
    assert got == ["critical", "high-security", "high-bug-early", "high-bug-later", "high-testing", "high-lowconf", "low"]
    assert [f.title for f in sorted(reversed(items), key=priority_key)] == got  # input order does not matter


# ---------------------------------------------------------------- rendering safety


def test_clean_masks_secrets():
    text = clean('API_KEY = "gsk_abcdefghijklmnopqrstuvwx" and password = "hunter2hunter2"')
    assert "gsk_abcdefghijklmnop" not in text and "hunter2hunter2" not in text


def test_markdown_helpers_neutralise_repository_text():
    assert "<script>" not in inline("<script>alert(1)</script>")
    assert "\n" not in inline("line1\nline2")
    assert "`" not in code("a`b")[1:-1]
    block = fence("x = 1\n```\n# pretend the report ended\n")
    assert block.startswith("````") and block.endswith("````")  # fence is longer than the inner backtick run


# ---------------------------------------------------------------- complete run


def test_complete_run_writes_reports_and_labels_them_complete(capsys, sample_repo, tmp_path):
    assert main(["run", str(sample_repo), "--test-mode"]) == 0
    out = capsys.readouterr().out
    assert "Reports:" in out
    reports = list((tmp_path / "data" / "reports").glob("R-*"))
    assert len(reports) == 1
    md = (reports[0] / "report.md").read_text(encoding="utf-8")
    data = json.loads((reports[0] / "report.json").read_text(encoding="utf-8"))
    assert "## Review status: COMPLETE" in md and "INCOMPLETE" not in md
    assert data["complete"] is True and data["incomplete_reasons"] == [] and data["run_status"] == "completed"
    assert len(data["findings"]) == data["verification"]["accepted"] == 7
    assert sum(data["counts_by_severity"].values()) == 7
    assert all(f["checker_status"] == "accepted" for f in data["findings"])
    assert data["verification"]["unchecked"] == 0 and data["coverage"]["gaps_total"] == 0


def test_findings_are_in_priority_order(capsys, sample_repo):
    data = report_json(capsys, run_id_of(capsys, sample_repo))
    findings = [Finding.model_validate(f) for f in data["findings"]]
    assert [f.finding_id for f in findings] == [f.finding_id for f in sorted(findings, key=priority_key)]


def test_report_is_replayable_and_identical(capsys, sample_repo, tmp_path):
    rid = run_id_of(capsys, sample_repo)
    assert main(["report", rid]) == 0
    first = capsys.readouterr().out
    assert main(["report", rid]) == 0
    assert capsys.readouterr().out == first
    assert main(["report", rid, "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "report.md").read_text(encoding="utf-8") == first
    db = RunStore(tmp_path / "data" / "runs.db")
    try:
        assert render_markdown(build_run_report(db, rid)) == first
        assert json.loads(render_json(build_run_report(db, rid)))["run_id"] == rid
    finally:
        db.close()


def test_budgets_are_recorded_and_disclosed(capsys, sample_repo):
    rid = run_id_of(capsys, sample_repo)
    data = report_json(capsys, rid)
    cfg = data["budgets"]["configured"]
    assert cfg["max_total_tokens"] == 500000 and cfg["max_concurrency"] == 3 and cfg["max_revisions"] == 2
    assert data["budgets"]["observed"]["peak_concurrency"] == 3 and data["budgets"]["limits_hit"] == []
    assert data["prompt_tokens"] > 0


def test_unknown_run_and_missing_database(capsys, sample_repo):
    assert main(["report", "R-nope"]) == 1
    assert "no run database" in capsys.readouterr().err
    run_id_of(capsys, sample_repo)
    assert main(["report", "R-nope"]) == 1
    assert "unknown run" in capsys.readouterr().err


# ---------------------------------------------------------------- incomplete runs are never called complete


def test_failed_specialist_makes_the_report_incomplete(capsys, sample_repo):
    rid = run_id_of(capsys, sample_repo, "--simulate-fail", "security")
    data = report_json(capsys, rid)
    assert data["complete"] is False and data["run_status"] == "partial"
    assert any("security" in r and "failed" in r for r in data["incomplete_reasons"])
    statuses = {t["specialist"]: t["status"] for t in data["tasks"]}
    assert statuses["security"] == "failed" and statuses["bug"] == "succeeded"  # a failure is not "no issues"
    assert data["coverage"]["gaps_total"] > 0
    assert main(["report", rid]) == 0
    md = capsys.readouterr().out
    assert "## Review status: INCOMPLETE" in md and "## Review status: COMPLETE" not in md


def test_checker_outage_leaves_nothing_accepted(capsys, sample_repo):
    data = report_json(capsys, run_id_of(capsys, sample_repo, "--simulate-fail", "checker"))
    assert data["complete"] is False and data["findings"] == []
    assert data["verification"]["accepted"] == 0 and data["verification"]["unresolved"] == 7
    assert data["verification"]["checks_incomplete"] == 7
    assert {u["reason_code"] for u in data["unresolved"]} == {"checker_error"}


def test_unchecked_candidates_are_disclosed(capsys, sample_repo, tmp_path):
    rid = run_id_of(capsys, sample_repo)
    db = RunStore(tmp_path / "data" / "runs.db")
    try:
        db._conn.execute("DELETE FROM finding_outcomes WHERE run_id=? AND finding_id=(SELECT MIN(finding_id) FROM finding_outcomes WHERE run_id=?)", (rid, rid))
        report = build_run_report(db, rid)
    finally:
        db.close()
    assert report.verification.unchecked == 1 and report.complete is False
    assert any("never checked" in r for r in report.incomplete_reasons)


# ---------------------------------------------------------------- budgets enforced


def test_token_budget_stops_new_work_and_is_disclosed(capsys, sample_repo, monkeypatch):
    monkeypatch.setenv("REVIEW_MAX_TOKENS", "1")
    monkeypatch.setenv("REVIEW_MAX_CONCURRENCY", "1")
    main(["run", str(sample_repo), "--test-mode", "--json"])
    rid = json.loads(capsys.readouterr().out)["run_id"]
    data = report_json(capsys, rid)
    statuses = [t["status"] for t in data["tasks"]]
    assert statuses.count("succeeded") == 1 and statuses.count("skipped") == 2  # only the first task fit the budget
    assert data["complete"] is False and data["run_status"] == "partial"
    assert "model-usage budget" in data["budgets"]["limits_hit"]
    assert data["findings"] == []  # nothing could be checked, so nothing is accepted
    assert {u["reason_code"] for u in data["unresolved"]} == {"check_budget"}
    assert data["budgets"]["configured"]["max_total_tokens"] == 1
