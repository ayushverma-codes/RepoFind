import json

import pytest

from reviewharness.cli import main

VALID = {
    "category": "bug",
    "title": "Off-by-one",
    "severity": "high",
    "file_path": "src/app.py",
    "start_line": 3,
    "end_line": 4,
    "evidence": "range(len(x)+1)",
    "rationale": "Overruns list.",
    "impact": "IndexError.",
    "confidence": 0.7,
    "source": "bug",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("GROQ_API_KEY", "REVIEW_MAX_CONCURRENCY"):
        monkeypatch.delenv(var, raising=False)


def test_version(capsys):
    assert main(["version"]) == 0
    assert "reviewharness" in capsys.readouterr().out


def test_smoke(capsys):
    assert main(["smoke"]) == 0
    assert "SMOKE OK" in capsys.readouterr().out


def test_config_redacts_key(capsys, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_abcdefghijklmnopqrstuv")
    assert main(["config"]) == 0
    out = capsys.readouterr().out
    assert "gsk_abcdefghijklmnopqrstuv" not in out
    assert '"groq_api_key": "set"' in out


def test_invalid_config_exit_code(capsys, monkeypatch):
    monkeypatch.setenv("REVIEW_MAX_CONCURRENCY", "0")
    assert main(["version"]) == 2
    assert "Invalid configuration" in capsys.readouterr().err


def test_validate_finding_ok_and_bad(tmp_path, capsys):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(VALID))
    assert main(["validate-finding", str(good)]) == 0
    assert "VALID" in capsys.readouterr().out

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**VALID, "file_path": "../secret", "confidence": 3}))
    assert main(["validate-finding", str(bad)]) == 1
    assert "INVALID" in capsys.readouterr().err

    assert main(["validate-finding", str(tmp_path / "missing.json")]) == 2


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        main(["nope"])
