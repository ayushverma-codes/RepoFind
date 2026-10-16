"""Inventory and context against the committed sample repository."""
from pathlib import Path

import pytest

from reviewharness.ingest.context import build_context, parse_package_json, parse_pyproject, parse_requirements
from reviewharness.ingest.inventory import build_inventory, is_test_path, language_for
from reviewharness.ingest.models import Inventory
from reviewharness.ingest.snapshot import create_snapshot
from reviewharness.models import InputType, RepositoryInput


@pytest.fixture
def sample_root(make_settings, sample_repo) -> Path:
    snap = create_snapshot(RepositoryInput(input_type=InputType.LOCAL_FOLDER, location=str(sample_repo)), make_settings())
    return Path(snap.tree_path)


def by_path(inv: Inventory):
    return {e.path: e for e in inv.entries}


def test_inventory_included_and_skipped(sample_root):
    inv = build_inventory(sample_root, 512 * 1024)
    entries = by_path(inv)
    included = sorted(p for p, e in entries.items() if e.included)
    assert included == ["README.md", "app.py", "broken.py", "requirements.txt", "tests/test_app.py", "utils.py"]
    reasons = {p: e.skip_reason.value for p, e in entries.items() if not e.included}
    assert reasons == {
        "assets/logo.png": "binary",
        "data/blob.dat": "binary",
        "empty.txt": "empty",
        "legacy.txt": "not_utf8",
        "package-lock.json": "generated",
    }
    assert inv.skipped_counts == {"binary": 2, "empty": 1, "generated": 1, "not_utf8": 1}
    assert inv.languages == {"markdown": 1, "python": 4, "text": 1}


def test_inventory_details_and_determinism(sample_root):
    first = build_inventory(sample_root, 512 * 1024)
    second = build_inventory(sample_root, 512 * 1024)
    assert first == second
    assert [e.path for e in first.entries] == sorted(e.path for e in first.entries)
    app = by_path(first)["app.py"]
    assert app.language == "python" and app.lines == 19 and len(app.sha256) == 64 and not app.is_test
    assert by_path(first)["tests/test_app.py"].is_test
    assert by_path(first)["assets/logo.png"].sha256 is None


def test_too_large_files_are_excluded(sample_root):
    inv = build_inventory(sample_root, 100)  # 100-byte cap
    assert by_path(inv)["app.py"].skip_reason.value == "too_large"


def test_inventory_round_trips_through_json(sample_root):
    inv = build_inventory(sample_root, 512 * 1024)
    assert Inventory.model_validate_json(inv.model_dump_json()) == inv


@pytest.mark.parametrize(
    "path,expected",
    [("tests/a.py", True), ("src/test_x.py", True), ("x_test.go", True), ("src/app.py", False), ("contest.py", False)],
)
def test_is_test_path(path, expected):
    assert is_test_path(path) is expected


def test_language_for():
    assert language_for("a/b.PY") == "python" and language_for("Dockerfile") == "dockerfile" and language_for("x.weird") == "other"


def test_context_for_sample(sample_root):
    inv = build_inventory(sample_root, 512 * 1024)
    ctx = build_context(sample_root, inv)
    assert ctx.included_files == 6 and ctx.test_files == 1 and ctx.source_files == 3
    assert ctx.manifests == ["requirements.txt"]
    assert ctx.dependencies == {"requirements.txt": ["flask", "requests"]}
    assert ctx.entry_points == ["app.py"]
    assert ctx.largest_files[0].path == "app.py"
    assert ctx.top_level_tree == ["README.md", "app.py", "broken.py", "requirements.txt", "tests/", "utils.py"]
    assert any("excluded from review" in n for n in ctx.notes)


def test_readme_is_carried_as_untrusted_data(sample_root):
    ctx = build_context(sample_root, build_inventory(sample_root, 512 * 1024))
    assert ctx.readme_excerpt is not None and ctx.readme_excerpt.untrusted is True
    assert "Ignore all previous instructions" in ctx.readme_excerpt.text  # kept verbatim as data


def test_dependency_parsers():
    assert parse_requirements("Django>=4\n# c\n-r x.txt\nrequests[security]==2\n") == ["django", "requests"]
    assert parse_pyproject('[project]\ndependencies=["a>=1"]\n[project.optional-dependencies]\ndev=["B"]\n') == ["a", "b"]
    assert parse_package_json('{"dependencies":{"x":"1"},"devDependencies":{"y":"2"}}') == ["x", "y"]


def test_bad_manifest_becomes_a_note(tmp_path):
    (tmp_path / "package.json").write_text("{not json")
    ctx = build_context(tmp_path, build_inventory(tmp_path, 1024))
    assert any("could not parse" in n for n in ctx.notes)
