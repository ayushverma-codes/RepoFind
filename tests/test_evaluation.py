import json
import shutil
from pathlib import Path

import pytest

from reviewharness.cli import main
from reviewharness.config import Settings
from reviewharness.evaluation import (
    DEFAULT_BENCHMARKS_DIR,
    EvaluationError,
    ExpectedIssue,
    load_benchmarks,
    match_findings,
    run_benchmarks,
    run_failure_demo,
)
from reviewharness.evaluation.scoring import build_score
from reviewharness.llm import make_provider
from reviewharness.models import Finding


def finding(file, line, category="security", title="t", end=None):
    return Finding(
        category=category, title=title, severity="high", file_path=file, start_line=line, end_line=end or line,
        evidence="x", rationale="r", impact="i", confidence=0.8, source="baseline",
    )


def issue(id_, file, line, category="security"):
    return ExpectedIssue(id=id_, file=file, start_line=line, end_line=line, category=category)


# ------------------------------------------------------------------ scoring

def test_match_is_one_to_one_and_classifies_everything():
    expected = [issue("A", "a.py", 10), issue("B", "a.py", 50), issue("C", "b.py", 1, "bug")]
    findings = [
        finding("a.py", 10, title="exact"),
        finding("a.py", 11, title="duplicate of A (within tolerance)"),
        finding("a.py", 51, title="near B"),
        finding("zzz.py", 3, title="wrong file"),
        finding("b.py", 1, category="security", title="wrong category group"),
    ]
    m = match_findings(findings, expected, tolerance=1)
    assert set(m.matched) == {"A", "B"}
    assert len(m.true_positives) == 2 and len(m.duplicates) == 1 and len(m.false_positives) == 2
    assert [e.id for e in m.missed] == ["C"]


def test_tolerance_zero_requires_overlap_and_testing_matches_maintainability():
    e = [issue("A", "a.py", 10, "testing")]
    assert match_findings([finding("a.py", 11, "testing")], e, tolerance=0).missed
    assert not match_findings([finding("a.py", 10, "maintainability")], e, tolerance=0).missed


def test_failed_system_is_not_measured_rather_than_zero():
    s = build_score("baseline", ok=False, status="failed", findings=[], expected=[issue("A", "a.py", 1)],
                    tolerance=1, latency_s=0.5, error="boom")
    assert s.ok is False and s.precision is None and s.recall is None and s.missed == ["A"] and s.error == "boom"


def test_precision_is_none_when_nothing_reported():
    s = build_score("baseline", ok=True, status="succeeded", findings=[], expected=[issue("A", "a.py", 1)],
                    tolerance=1, latency_s=0.1)
    assert s.precision is None and s.recall == 0.0


# --------------------------------------------------------- benchmark loading

def test_shipped_benchmarks_load_and_reference_real_files():
    specs = load_benchmarks(DEFAULT_BENCHMARKS_DIR)
    assert [s.name for s, _ in specs] == ["data_pipeline", "test_suite", "web_service"]
    assert sum(len(s.expected) for s, _ in specs) == 14


def test_loader_rejects_bad_definitions(tmp_path):
    with pytest.raises(EvaluationError):
        load_benchmarks(tmp_path / "missing")
    with pytest.raises(EvaluationError):
        load_benchmarks(tmp_path)  # empty: no benchmarks
    with pytest.raises(EvaluationError):
        load_benchmarks(DEFAULT_BENCHMARKS_DIR, ["nope"])

    bad = tmp_path / "x"
    (bad / "repo").mkdir(parents=True)
    (bad / "expected.json").write_text("{not json")
    with pytest.raises(EvaluationError):
        load_benchmarks(tmp_path)

    (bad / "expected.json").write_text(json.dumps({"name": "other", "description": "d", "expected": [
        {"id": "I", "file": "f.py", "start_line": 1, "end_line": 1, "category": "bug"}]}))
    with pytest.raises(EvaluationError, match="does not match"):
        load_benchmarks(tmp_path)

    (bad / "expected.json").write_text(json.dumps({"name": "x", "description": "d", "expected": [
        {"id": "I", "file": "missing.py", "start_line": 1, "end_line": 1, "category": "bug"}]}))
    with pytest.raises(EvaluationError, match="missing file"):
        load_benchmarks(tmp_path)


# ------------------------------------------------ baseline vs multi-agent run

@pytest.fixture
def report(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", test_mode=True)
    return run_benchmarks(DEFAULT_BENCHMARKS_DIR, tmp_path / "out", settings, make_provider(settings))


def _by_name(report):
    return {r.name: r for r in report.results}


def test_checker_removes_the_comment_false_positive(report):
    ws = _by_name(report)["web_service"]
    assert ws.baseline.false_positives == 1 and ws.baseline.false_positive_findings[0].startswith("app.py:12")
    assert ws.multi_agent.false_positives == 0
    assert ws.multi_agent.precision == 1.0 and ws.multi_agent.recall == 1.0
    assert ws.baseline.precision == pytest.approx(5 / 6, abs=0.001) and ws.baseline.recall == 1.0


def test_known_miss_and_known_false_positive_are_reported_honestly(report):
    dp = _by_name(report)["data_pipeline"]
    for s in (dp.baseline, dp.multi_agent):
        assert s.missed == ["DP-OFFBYONE"] and s.recall == 0.75
        assert s.false_positives == 1 and s.false_positive_findings[0].startswith("pipeline.py:25")


def test_testing_benchmark_is_fully_found_by_both(report):
    ts = _by_name(report)["test_suite"]
    for s in (ts.baseline, ts.multi_agent):
        assert s.precision == 1.0 and s.recall == 1.0 and s.missed == []


def test_aggregate_numbers_and_completeness(report):
    base, multi = {a.system: a for a in report.aggregate}["baseline"], {a.system: a for a in report.aggregate}["multi_agent"]
    assert (base.reported, base.true_positives, base.false_positives) == (15, 13, 2)
    assert (multi.reported, multi.true_positives, multi.false_positives) == (14, 13, 1)
    assert base.recall == multi.recall == pytest.approx(13 / 14, abs=0.001)
    assert multi.precision > base.precision
    assert multi.incomplete_runs == 0 and all(r.multi_agent.complete for r in report.results)
    assert multi.prompt_tokens > 0 and base.prompt_tokens > 0
    assert report.mode == "test" and report.limitations


def _scores(report):
    return [
        {k: v for k, v in s.model_dump().items() if k not in ("latency_s",)}
        for r in report.results for s in (r.baseline, r.multi_agent)
    ]


def test_benchmark_scores_are_reproducible(report, tmp_path):
    settings = Settings(data_dir=tmp_path / "data2", test_mode=True)
    again = run_benchmarks(DEFAULT_BENCHMARKS_DIR, tmp_path / "out2", settings, make_provider(settings))
    assert _scores(again) == _scores(report)


def test_reports_are_written_and_do_not_leak_the_planted_secret(report, tmp_path):
    text = "".join(p.read_text(encoding="utf-8") for p in (tmp_path / "out" / "web_service" / "reports").rglob("report.*"))
    assert "demo-token-value-1234" not in text and "Hardcoded credential" in text


# ----------------------------------------------------------- failure demo

def test_failure_demo_passes(tmp_path):
    settings = Settings(data_dir=tmp_path / "data")
    results = run_failure_demo(DEFAULT_BENCHMARKS_DIR, tmp_path / "demo", settings)
    assert [r.scenario for r in results] == ["worker_failure", "checker_outage"]
    for r in results:
        assert r.passed, [c for c in r.checks if not c.passed]


# --------------------------------------------------------------------- CLI

@pytest.fixture
def cli_env(monkeypatch, tmp_path):
    monkeypatch.setattr("reviewharness.cli.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("REVIEW_TEST_MODE", raising=False)
    monkeypatch.setenv("REVIEW_DATA_DIR", str(tmp_path / "data"))
    return tmp_path


def test_cli_benchmark_prints_table_and_writes_files(capsys, cli_env):
    assert main(["benchmark"]) == 0
    out = capsys.readouterr().out
    assert "Aggregate" in out and "baseline" in out and "multi_agent" in out and "Limitations" in out
    d = cli_env / "data" / "benchmarks"
    data = json.loads((d / "benchmark_results.json").read_text())
    assert data["mode"] == "test" and len(data["results"]) == 3
    assert "| web_service | baseline |" in (d / "benchmark_results.md").read_text()


def test_cli_benchmark_only_and_json(capsys, cli_env):
    assert main(["benchmark", "--only", "web_service", "--json"]) == 0
    out = capsys.readouterr().out
    assert json.loads(out[: out.rindex("}") + 1])["results"][0]["name"] == "web_service"


def test_cli_benchmark_errors(capsys, cli_env):
    assert main(["benchmark", "--only", "nope"]) == 2
    assert "unknown benchmark" in capsys.readouterr().err
    assert main(["benchmark", "--live"]) == 2
    assert "GROQ_API_KEY" in capsys.readouterr().err
    assert main(["benchmark", "--benchmarks-dir", str(cli_env / "none")]) == 2


def test_cli_failure_demo(capsys, cli_env):
    assert main(["failure-demo"]) == 0
    out = capsys.readouterr().out
    assert "FAILURE DEMO: PASSED" in out and "[FAIL]" not in out and "checker_outage" in out
