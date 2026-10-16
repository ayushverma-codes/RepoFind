import json
import zipfile

import pytest

from reviewharness.cli import main
from reviewharness.llm import LLMPermanentError
from reviewharness.llm.fake import ScriptedProvider
from reviewharness.specialists import SpecialistResult


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setattr("reviewharness.cli.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("REVIEW_TEST_MODE", raising=False)
    monkeypatch.setenv("REVIEW_DATA_DIR", str(tmp_path / "data"))


def test_review_all_in_test_mode(capsys, sample_repo, tmp_path):
    assert main(["review", str(sample_repo), "--test-mode"]) == 0
    out = capsys.readouterr().out
    for name in ("bug", "security", "testing_maintainability"):
        assert f"{name}: succeeded" in out
    files = sorted((tmp_path / "data" / "snapshots").glob("*/reviews/*.json"))
    assert [f.name for f in files] == ["bug.json", "security.json", "testing_maintainability.json"]
    for f in files:
        SpecialistResult.model_validate_json(f.read_text(encoding="utf-8"))


def test_review_json_output(capsys, sample_repo):
    assert main(["review", str(sample_repo), "--test-mode", "--specialist", "baseline", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data) == 1 and data[0]["status"] == "succeeded" and data[0]["specialist"] == "baseline"


def test_review_test_mode_from_environment(capsys, sample_repo, monkeypatch):
    monkeypatch.setenv("REVIEW_TEST_MODE", "1")
    assert main(["review", str(sample_repo), "--specialist", "bug"]) == 0


def test_review_zip_input(capsys, sample_repo, tmp_path):
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(sample_repo.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(sample_repo).as_posix())
    assert main(["review", str(archive), "--test-mode", "--specialist", "security"]) == 0
    assert "security: succeeded" in capsys.readouterr().out


def test_review_without_key_fails_clearly_before_ingesting(capsys, sample_repo, tmp_path):
    assert main(["review", str(sample_repo)]) == 2
    assert "GROQ_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "data" / "snapshots").exists()


def test_review_exit_code_and_output_when_provider_fails(capsys, sample_repo, monkeypatch):
    provider = ScriptedProvider([LLMPermanentError("HTTP 401")])
    monkeypatch.setattr("reviewharness.cli.make_provider", lambda settings: provider)
    assert main(["review", str(sample_repo), "--specialist", "bug"]) == 1
    out = capsys.readouterr().out
    assert "bug: failed" in out and "error:" in out
