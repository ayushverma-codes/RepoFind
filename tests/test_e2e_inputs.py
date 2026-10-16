"""End-to-end CLI workflow for every supported input type, in test mode (no API key, no network)."""
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from reviewharness.cli import main

BENCH_REPO = Path(__file__).parent.parent / "benchmarks" / "web_service" / "repo"


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setattr("reviewharness.cli.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("REVIEW_TEST_MODE", raising=False)
    monkeypatch.setenv("REVIEW_DATA_DIR", str(tmp_path / "data"))


@pytest.fixture
def zip_source(tmp_path):
    path = tmp_path / "repo.zip"
    with zipfile.ZipFile(path, "w") as z:
        for f in sorted(BENCH_REPO.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(BENCH_REPO).as_posix())
    return path


@pytest.fixture
def github_source(monkeypatch):
    def fake_clone(url, dest, timeout_s, runner=None):
        shutil.copytree(BENCH_REPO, dest, dirs_exist_ok=True)

    monkeypatch.setattr("reviewharness.ingest.snapshot.clone_github", fake_clone)
    return "https://github.com/example/demo"


def _run(capsys, source):
    assert main(["run", str(source), "--test-mode", "--json", "--new-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    report_dir = Path(payload["report_dir"])
    md, js = report_dir / "report.md", report_dir / "report.json"
    assert md.is_file() and js.is_file()
    return payload["run_id"], json.loads(js.read_text(encoding="utf-8")), md.read_text(encoding="utf-8")


def _signature(report):
    return sorted((f["file_path"], f["start_line"], f["title"]) for f in report["findings"])


def test_all_three_input_types_give_the_same_complete_report(capsys, zip_source, github_source):
    folder = _run(capsys, BENCH_REPO)
    archive = _run(capsys, zip_source)
    github = _run(capsys, github_source)
    for _, report, md in (folder, archive, github):
        assert report["complete"] is True and report["incomplete_reasons"] == []
        assert report["verification"]["accepted"] == 5 and report["verification"]["unchecked"] == 0
        assert "demo-token-value-1234" not in md
        assert "Review status: COMPLETE" in md
    assert _signature(folder[1]) == _signature(archive[1]) == _signature(github[1])
    assert {r[1]["repository"]["input_type"] for r in (folder, archive, github)} == {"local_folder", "zip_file", "github_url"}


def test_report_command_replays_the_stored_run(capsys):
    run_id, stored, _ = _run(capsys, BENCH_REPO)
    assert main(["report", run_id, "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out) == stored
    assert main(["inspect", run_id, "--events"]) == 0
    assert "run_finished" in capsys.readouterr().out


def test_unsupported_source_is_rejected_cleanly(capsys, tmp_path):
    assert main(["run", str(tmp_path / "missing"), "--test-mode"]) == 1
    assert "INGEST FAILED" in capsys.readouterr().err
