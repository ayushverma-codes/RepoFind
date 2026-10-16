"""End-to-end ingestion through the CLI."""
import json
import zipfile
from pathlib import Path

import pytest

from reviewharness.cli import main


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("REVIEW_DATA_DIR", str(tmp_path / "data"))


def test_cli_ingest_folder_no_tools(capsys, sample_repo):
    assert main(["ingest", str(sample_repo), "--no-tools"]) == 0
    out = capsys.readouterr().out
    assert "(new)" in out and "6 reviewable of 11 files" in out and "Tools: not run" in out


def test_cli_ingest_json_and_artifacts(capsys, sample_repo, tmp_path):
    assert main(["ingest", str(sample_repo), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["inventory"]["included_count"] == 6
    assert {t["tool"] for t in data["tool_results"]} == {"python_syntax", "ruff", "bandit"}
    out_dir = Path(data["snapshot"]["tree_path"]).parent
    for name in ("snapshot.json", "manifest.json", "inventory.json", "context.json", "tools.json"):
        assert (out_dir / name).is_file()
    assert not list(out_dir.glob("*.tmp"))


def test_cli_ingest_twice_reuses_snapshot(capsys, sample_repo):
    main(["ingest", str(sample_repo), "--no-tools"])
    capsys.readouterr()
    main(["ingest", str(sample_repo), "--no-tools"])
    assert "(reused)" in capsys.readouterr().out


def test_cli_ingest_zip(capsys, sample_repo, tmp_path):
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for p in sorted(sample_repo.rglob("*")):
            if p.is_file():
                zf.write(p, "sample/" + p.relative_to(sample_repo).as_posix())
    assert main(["ingest", str(archive), "--no-tools"]) == 0
    assert "zip_file" in capsys.readouterr().out


def test_cli_hostile_zip_exits_1(capsys, tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../evil.py", "boom")
    assert main(["ingest", str(archive)]) == 1
    assert "INGEST FAILED" in capsys.readouterr().err


def test_cli_bad_sources(capsys, tmp_path):
    assert main(["ingest", str(tmp_path / "nothing-here")]) == 1
    assert "cannot tell" in capsys.readouterr().err
    assert main(["ingest", "https://evil.com/o/r", "--type", "github"]) == 2
    assert "Invalid input" in capsys.readouterr().err


def test_secrets_in_repo_text_do_not_reach_artifacts(tmp_path, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("key: gsk_abcdefghijklmnopqrstuv\n")
    (repo / "a.py").write_text("x = 1\n")
    assert main(["ingest", str(repo), "--json"]) == 0
    out = capsys.readouterr().out
    assert "gsk_abcdefghijklmnopqrstuv" not in out
    for f in (tmp_path / "data" / "snapshots").glob("*/context.json"):
        assert "gsk_abcdefghijklmnopqrstuv" not in f.read_text()
