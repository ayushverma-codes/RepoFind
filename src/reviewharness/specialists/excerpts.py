"""Scoped, bounded source excerpts. The only code that feeds repository text to a specialist.

Scope rules (all deterministic):
  * only files the inventory marked `included` can be shown;
  * every read goes through resolve_inside() (no traversal, no symlinks);
  * total and per-file character budgets bound prompt size; whatever does not
    fit is recorded as truncated/omitted so coverage is never overstated.
"""
from __future__ import annotations

from pathlib import Path

from reviewharness.ingest.errors import IngestError
from reviewharness.ingest.models import Inventory
from reviewharness.ingest.safepath import resolve_inside
from reviewharness.models import SpecialistSource
from reviewharness.specialists.models import Coverage, Excerpt

MAX_TOTAL_CHARS = 12000  # keeps one request well inside small free-tier token limits
MAX_FILE_CHARS = 6000
LINE_OVERHEAD = 8  # "  123| " prefix added when rendering
MIN_USEFUL_CHARS = 300

_NON_CODE = frozenset({"markdown", "rst", "text", "json", "yaml", "toml", "ini", "other", None})


def select_scope(inventory: Inventory, specialist: SpecialistSource) -> list[str]:
    """Ordered list of reviewable code files for this specialist (sorted by path)."""
    code = [e for e in inventory.entries if e.included and e.language not in _NON_CODE]
    source = sorted(e.path for e in code if not e.is_test)
    tests = sorted(e.path for e in code if e.is_test)
    if specialist is SpecialistSource.TESTING_MAINTAINABILITY:
        return tests + source
    return source + tests


def build_excerpts(
    root: Path,
    inventory: Inventory,
    paths: list[str],
    *,
    max_total_chars: int = MAX_TOTAL_CHARS,
    max_file_chars: int = MAX_FILE_CHARS,
) -> tuple[list[Excerpt], Coverage]:
    included = {e.path for e in inventory.entries if e.included}
    excerpts: list[Excerpt] = []
    coverage = Coverage()
    remaining = max_total_chars

    for rel in paths:
        if rel not in included or remaining < MIN_USEFUL_CHARS:
            coverage.omitted.append(rel)
            continue
        try:
            text = resolve_inside(root, rel).read_text(encoding="utf-8")
        except (IngestError, OSError, UnicodeDecodeError):
            coverage.omitted.append(rel)
            continue

        lines = text.splitlines()
        limit = min(max_file_chars, remaining)
        shown: list[str] = []
        used = 0
        for line in lines:
            cost = len(line) + LINE_OVERHEAD
            if used + cost > limit:
                break
            shown.append(line)
            used += cost
        if not shown:
            coverage.omitted.append(rel)
            continue

        remaining -= used
        excerpts.append(Excerpt(path=rel, total_lines=len(lines), shown_lines=len(shown), text="\n".join(shown)))
        (coverage.truncated if len(shown) < len(lines) else coverage.reviewed).append(rel)
    return excerpts, coverage
