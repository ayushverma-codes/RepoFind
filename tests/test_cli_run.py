import json

import pytest

from reviewharness.cli import main


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setattr("reviewharness.cli.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("REVIEW_TEST_MODE", raising=False)
    monkeypatch.setenv("REVIEW_DATA_DIR", str(tmp_path / "data"))


def test_run_completes_in_test_mode(capsys, sample_repo, tmp_path):
    assert main(["run", str(sample_repo), "--test-mode"]) == 0
    out = capsys.readouterr().out
    assert ": COMPLETED" in out and "peak concurrency: 3" in out
    assert (tmp_path / "data" / "runs.db").exists()


def test_run_reports_partial_when_a_specialist_fails(capsys, sample_repo):
    assert main(["run", str(sample_repo), "--test-mode", "--simulate-fail", "security"]) == 1
    out = capsys.readouterr().out
    assert ": PARTIAL" in out and "INCOMPLETE" in out
    assert "security" in out and "failed" in out and "succeeded" in out


def test_simulation_flags_require_test_mode(capsys, sample_repo):
    assert main(["run", str(sample_repo), "--simulate-delay", "1"]) == 2
    assert "--test-mode" in capsys.readouterr().err


def test_run_without_key_fails_clearly(capsys, sample_repo):
    assert main(["run", str(sample_repo)]) == 2
    assert "GROQ_API_KEY" in capsys.readouterr().err


def test_second_identical_run_is_reused_and_new_run_forces_another(capsys, sample_repo):
    assert main(["run", str(sample_repo), "--test-mode", "--json"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert main(["run", str(sample_repo), "--test-mode", "--json"]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["run_id"] == first["run_id"] and second["reused"] is True
    assert main(["run", str(sample_repo), "--test-mode", "--json", "--new-run"]) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] != first["run_id"]


def test_inspect_lists_and_replays_events(capsys, sample_repo):
    main(["run", str(sample_repo), "--test-mode", "--json"])
    run_id = json.loads(capsys.readouterr().out)["run_id"]
    assert main(["inspect"]) == 0
    assert run_id in capsys.readouterr().out
    assert main(["inspect", run_id, "--events"]) == 0
    out = capsys.readouterr().out
    assert "run_created" in out and "task_started" in out and "run_finished" in out
    assert main(["inspect", "R-nope"]) == 1


def test_inspect_without_database(capsys):
    assert main(["inspect"]) == 1
    assert "no run database" in capsys.readouterr().err


def test_run_prints_verification_summary(capsys, sample_repo):
    assert main(["run", str(sample_repo), "--test-mode"]) == 0
    out = capsys.readouterr().out
    assert "Verification: accepted=7" in out and "unchecked=0" in out and "Accepted findings:" in out


def test_checker_outage_is_disclosed_and_nothing_is_accepted(capsys, sample_repo):
    assert main(["run", str(sample_repo), "--test-mode", "--simulate-fail", "checker"]) == 1
    out = capsys.readouterr().out
    assert ": PARTIAL" in out and "accepted=0" in out and "uncertain=7" in out
    assert "Accepted findings:" not in out and "checker_error" in out
