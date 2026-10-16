import pytest

from reviewharness.ingest import ingest_repository
from reviewharness.ingest.models import Inventory, InventoryEntry
from reviewharness.models import SpecialistSource
from reviewharness.specialists.excerpts import build_excerpts, select_scope
from pathlib import Path


@pytest.fixture
def ingested(sample_repo, make_settings):
    result = ingest_repository(str(sample_repo), None, make_settings(), run_tools=False)
    return Path(result.snapshot.tree_path), result.inventory


def test_scope_orders_source_before_tests_for_bug_and_reverse_for_testing(ingested):
    _, inv = ingested
    assert select_scope(inv, SpecialistSource.BUG) == ["app.py", "broken.py", "utils.py", "tests/test_app.py"]
    assert select_scope(inv, SpecialistSource.TESTING_MAINTAINABILITY) == [
        "tests/test_app.py", "app.py", "broken.py", "utils.py",
    ]


def test_scope_excludes_binary_generated_and_docs(ingested):
    _, inv = ingested
    scope = set(select_scope(inv, SpecialistSource.BASELINE))
    assert not scope & {"assets/logo.png", "data/blob.dat", "package-lock.json", "legacy.txt", "README.md", "requirements.txt"}


def test_full_excerpts(ingested):
    root, inv = ingested
    excerpts, cov = build_excerpts(root, inv, select_scope(inv, SpecialistSource.BUG))
    assert [e.path for e in excerpts] == ["app.py", "broken.py", "utils.py", "tests/test_app.py"]
    assert cov.truncated == [] and cov.omitted == []
    assert all(not e.truncated for e in excerpts)


def test_per_file_budget_truncates_and_reports_it(ingested):
    root, inv = ingested
    excerpts, cov = build_excerpts(root, inv, select_scope(inv, SpecialistSource.BUG), max_file_chars=60)
    app = next(e for e in excerpts if e.path == "app.py")
    assert app.truncated and app.shown_lines < app.total_lines
    assert "app.py" in cov.truncated and "utils.py" in cov.reviewed


def test_total_budget_omits_remaining_files(ingested):
    root, inv = ingested
    scope = select_scope(inv, SpecialistSource.BUG)
    _, cov = build_excerpts(root, inv, scope, max_total_chars=350)
    assert cov.omitted
    assert sorted(cov.reviewed + cov.truncated + cov.omitted) == sorted(scope)


def test_paths_outside_inventory_or_root_are_never_read(ingested):
    root, inv = ingested
    evil = Inventory(entries=[InventoryEntry(path="../evil.py", size=5, included=True, language="python", lines=1)])
    excerpts, cov = build_excerpts(root, evil, ["../evil.py", "not_in_inventory.py"])
    assert excerpts == [] and cov.omitted == ["../evil.py", "not_in_inventory.py"]
