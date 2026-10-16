"""Static-analysis wrappers: parsers, failure recording, sandboxing, real tools."""
import json
import subprocess
from pathlib import Path

import pytest

from reviewharness.ingest import tools
from reviewharness.ingest.inventory import build_inventory
from reviewharness.ingest.models import ToolResult, ToolStatus
from reviewharness.ingest.snapshot import create_snapshot
from reviewharness.ingest.tools import find_executable, parse_bandit, parse_ruff, run_analysis_tools, run_python_syntax
from reviewharness.models import InputType, RepositoryInput


@pytest.fixture
def sample(make_settings, sample_repo):
    settings = make_settings()
    snap = create_snapshot(RepositoryInput(input_type=InputType.LOCAL_FOLDER, location=str(sample_repo)), settings)
    root = Path(snap.tree_path)
    return root, build_inventory(root, 512 * 1024), settings


def completed(argv, rc=0, out=b"[]", err=b""):
    return subprocess.CompletedProcess(argv, rc, out, err)


class Recorder:
    """Fake subprocess runner that records every call."""

    def __init__(self, rc=0, out=b"[]", err=b"", raises=None):
        self.rc, self.out, self.err, self.raises, self.calls = rc, out, err, raises, []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if self.raises and "--version" not in argv:
            raise self.raises
        return completed(argv, self.rc, self.out, self.err)


def by_tool(results):
    return {r.tool: r for r in results}


# ------------------------------------------------------------- built-in check


def test_python_syntax_flags_only_the_broken_file(sample):
    root, inv, _ = sample
    result = run_python_syntax(root, inv)
    assert result.status is ToolStatus.OK
    assert [(i.file_path, i.line, i.rule_id) for i in result.issues] == [("broken.py", 1, "SYNTAX")]


def test_python_syntax_clean_repo_is_ok_with_zero_issues(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    result = run_python_syntax(tmp_path, build_inventory(tmp_path, 1024))
    assert result.status is ToolStatus.OK and result.issue_total == 0


def test_no_python_files_marks_tools_skipped(tmp_path, make_settings):
    (tmp_path / "a.js").write_text("let x = 1;\n")
    results = by_tool(run_analysis_tools(tmp_path, build_inventory(tmp_path, 1024), make_settings()))
    assert {r.status for r in results.values()} == {ToolStatus.SKIPPED}


# -------------------------------------------------------------------- parsers


def test_parse_ruff_maps_paths_and_drops_outside_root(tmp_path):
    payload = json.dumps(
        [
            {"code": "F401", "message": "unused", "filename": str(tmp_path / "a.py"),
             "location": {"row": 3, "column": 1}, "end_location": {"row": 3, "column": 9}},
            {"code": None, "message": "bad syntax", "filename": str(tmp_path / "b.py"),
             "location": {"row": 1, "column": 1}},
            {"code": "F401", "message": "elsewhere", "filename": "/etc/passwd",
             "location": {"row": 1, "column": 1}},
        ]
    )
    parsed = parse_ruff(payload, tmp_path)
    assert [(i.rule_id, i.file_path, i.line) for i in parsed.issues] == [("F401", "a.py", 3), ("syntax-error", "b.py", 1)]
    assert parsed.dropped == 1


def test_parse_bandit_maps_results_and_errors(tmp_path):
    payload = json.dumps(
        {
            "results": [{"filename": "./src/x.py", "line_number": 7, "line_range": [7, 8], "test_id": "B602",
                         "issue_text": "shell=True", "issue_severity": "HIGH", "issue_confidence": "HIGH"}],
            "errors": [{"filename": "./src/y.py", "reason": "syntax error"}],
        }
    )
    parsed = parse_bandit(payload, tmp_path)
    issue = parsed.issues[0]
    assert (issue.rule_id, issue.file_path, issue.line, issue.end_line, issue.severity) == ("B602", "src/x.py", 7, 8, "high")
    assert parsed.warnings == ["src/y.py: syntax error"]


@pytest.mark.parametrize("parser", [parse_ruff, parse_bandit])
def test_parsers_reject_malformed_output(parser, tmp_path):
    for bad in ("not json", '"string"', "42"):
        with pytest.raises(ValueError):
            parser(bad, tmp_path)


def test_messages_are_redacted(tmp_path):
    payload = json.dumps([{"code": "X", "message": "key gsk_abcdefghijklmnopqrstuv leaked",
                           "filename": str(tmp_path / "a.py"), "location": {"row": 1}}])
    assert "gsk_abcdefghijklmnopqrstuv" not in parse_ruff(payload, tmp_path).issues[0].message


# ------------------------------------------------------ failures are recorded


def run_with(sample, runner, monkeypatch):
    root, inv, settings = sample
    monkeypatch.setattr(tools, "find_executable", lambda name: f"/fake/{name}")
    return by_tool(run_analysis_tools(root, inv, settings, runner=runner))


def test_missing_tools_are_recorded_not_fatal(sample, monkeypatch):
    root, inv, settings = sample
    monkeypatch.setattr(tools, "find_executable", lambda name: None)
    results = by_tool(run_analysis_tools(root, inv, settings))
    assert results["ruff"].status is ToolStatus.NOT_INSTALLED and "not installed" in results["ruff"].error
    assert results["bandit"].status is ToolStatus.NOT_INSTALLED
    assert results["python_syntax"].status is ToolStatus.OK  # built-in still ran


def test_timeout_is_recorded(sample, monkeypatch):
    results = run_with(sample, Recorder(raises=subprocess.TimeoutExpired("ruff", 1)), monkeypatch)
    assert results["ruff"].status is ToolStatus.TIMED_OUT and "timed out" in results["ruff"].error


def test_nonzero_exit_is_failed_with_redacted_stderr(sample, monkeypatch):
    results = run_with(sample, Recorder(rc=2, err=b"boom api_key=hunter2"), monkeypatch)
    assert results["ruff"].status is ToolStatus.FAILED and results["ruff"].exit_code == 2
    assert "hunter2" not in results["ruff"].error


def test_garbage_output_is_failed(sample, monkeypatch):
    results = run_with(sample, Recorder(out=b"<<<garbage>>>"), monkeypatch)
    assert results["ruff"].status is ToolStatus.FAILED and "could not parse" in results["ruff"].error


def test_failed_differs_from_clean_run(sample, monkeypatch):
    clean = run_with(sample, Recorder(out=b"[]"), monkeypatch)["ruff"]
    failed = run_with(sample, Recorder(rc=2), monkeypatch)["ruff"]
    assert clean.status is ToolStatus.OK and clean.issue_total == 0
    assert failed.status is ToolStatus.FAILED and failed.error


def test_tool_invocation_is_sandboxed(sample, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_abcdefghijklmnopqrstuv")
    rec = Recorder()
    run_with(sample, rec, monkeypatch)
    root = sample[0]
    ruff_argv, ruff_kw = next(c for c in rec.calls if c[0][0] == "/fake/ruff" and "check" in c[0])
    assert "--isolated" in ruff_argv  # ignores the reviewed repo's own ruff config
    for argv, kw in rec.calls:
        cwd = Path(kw["cwd"]).resolve()
        assert cwd != root.resolve() and root.resolve() not in cwd.parents  # never inside the snapshot
        assert kw["shell"] is False and kw["timeout"] > 0
        assert "GROQ_API_KEY" not in kw["env"]
        assert kw["stdin"] == subprocess.DEVNULL


def test_issues_outside_inventory_are_counted_as_dropped(sample, monkeypatch):
    root, _, _ = sample
    out = json.dumps([{"code": "F401", "message": "m", "filename": str(root / "assets" / "logo.png"),
                       "location": {"row": 1}}]).encode()
    result = run_with(sample, Recorder(out=out), monkeypatch)["ruff"]
    assert result.issue_total == 0 and result.dropped == 1


# ----------------------------------------------------- real tools (if present)

HAVE_RUFF, HAVE_BANDIT = bool(find_executable("ruff")), bool(find_executable("bandit"))


def pairs(result: ToolResult):
    return {(i.rule_id, i.file_path, i.line) for i in result.issues}


@pytest.mark.skipif(not HAVE_RUFF, reason="ruff not installed (pip install -e '.[analysis]')")
def test_real_ruff_on_sample(sample):
    root, inv, settings = sample
    result = by_tool(run_analysis_tools(root, inv, settings))["ruff"]
    assert result.status is ToolStatus.OK and result.version
    assert {("F401", "app.py", 1), ("B006", "app.py", 8), ("F841", "app.py", 9)} <= pairs(result)
    assert any(i.file_path == "broken.py" for i in result.issues)  # syntax error surfaced
    assert not any(i.file_path == "utils.py" for i in result.issues)  # clean file stays clean


@pytest.mark.skipif(not HAVE_BANDIT, reason="bandit not installed (pip install -e '.[analysis]')")
def test_real_bandit_on_sample(sample):
    root, inv, settings = sample
    result = by_tool(run_analysis_tools(root, inv, settings))["bandit"]
    assert result.status is ToolStatus.OK and result.version
    assert {("B105", "app.py", 5), ("B602", "app.py", 10), ("B307", "app.py", 11)} <= pairs(result)
    assert any("broken.py" in w for w in result.warnings)  # unparseable file disclosed


@pytest.mark.skipif(not (HAVE_RUFF and HAVE_BANDIT), reason="needs ruff and bandit")
def test_real_tool_results_are_deterministic(sample):
    root, inv, settings = sample
    strip = lambda rs: [(r.tool, r.status, r.issues) for r in rs]  # noqa: E731 - ignore timings
    assert strip(run_analysis_tools(root, inv, settings)) == strip(run_analysis_tools(root, inv, settings))


@pytest.mark.skipif(not HAVE_RUFF, reason="ruff not installed")
def test_repo_config_cannot_disable_ruff(make_settings, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("import os\n")
    (repo / "ruff.toml").write_text('[lint]\nignore = ["F401"]\nexclude = ["a.py"]\n')
    settings = make_settings()
    snap = create_snapshot(RepositoryInput(input_type=InputType.LOCAL_FOLDER, location=str(repo)), settings)
    root = Path(snap.tree_path)
    result = by_tool(run_analysis_tools(root, build_inventory(root, 1024), settings))["ruff"]
    assert ("F401", "a.py", 1) in pairs(result)


# --------------------------------------------- regression: Windows launcher junk


class JunkWritingRunner(Recorder):
    """Simulates the Windows launcher that writes '~/AppData/...' relative to its cwd."""

    def __call__(self, argv, **kwargs):
        junk = Path(kwargs["cwd"]) / "~" / "AppData" / "Local" / "Python Entry Points" / "abc"
        junk.parent.mkdir(parents=True, exist_ok=True)
        junk.write_text("junk")
        return super().__call__(argv, **kwargs)


def test_tool_side_effects_cannot_reach_the_snapshot(sample, monkeypatch):
    from reviewharness.ingest.snapshot import verify_snapshot  # noqa: F401 - documents the invariant

    root, inv, settings = sample
    before = sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))
    run_with(sample, JunkWritingRunner(), monkeypatch)
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*")) == before


def test_pipeline_fails_loudly_if_a_tool_modifies_the_snapshot(make_settings, sample_repo, monkeypatch):
    from reviewharness.ingest import pipeline
    from reviewharness.ingest.errors import IngestError

    def evil_tools(root, inventory, settings, *, runner=None):
        (root / "injected.py").write_text("x = 1\n")
        return []

    monkeypatch.setattr(pipeline, "run_analysis_tools", evil_tools)
    with pytest.raises(IngestError, match="changed during analysis"):
        pipeline.ingest_repository(str(sample_repo), InputType.LOCAL_FOLDER, make_settings())


def test_version_is_normalised(sample, monkeypatch):
    out = b"python.exe C:\\proj\\.venv\\Scripts\\bandit 1.9.4\n  python version = 3.14.7\n"
    assert tools._version("bandit", "/fake/bandit", Path("."), Recorder(out=out)) == "bandit 1.9.4"
    assert tools._version("ruff", "/fake/ruff", Path("."), Recorder(out=b"ruff 0.17.1\n")) == "ruff 0.17.1"


def test_windows_profile_vars_are_passed_but_secrets_are_not(monkeypatch):
    from reviewharness.ingest.subproc import safe_env

    monkeypatch.setenv("LOCALAPPDATA", "C:\\Users\\x\\AppData\\Local")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_abcdefghijklmnopqrstuv")
    env = safe_env()
    assert env["LOCALAPPDATA"] == "C:\\Users\\x\\AppData\\Local" and "GROQ_API_KEY" not in env
