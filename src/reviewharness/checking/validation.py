"""Deterministic validation and de-duplication. Runs before any checker model is called."""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Optional

from reviewharness.checking.source import read_lines
from reviewharness.ingest.models import Inventory
from reviewharness.models import Finding
from reviewharness.specialists.parsing import MAX_SPAN_LINES, mask_secrets, normalize_text, strip_numbering


class Invalid(NamedTuple):
    finding: Finding
    code: str
    reason: str


class Duplicate(NamedTuple):
    finding: Finding
    kept_id: str


class ValidationResult(NamedTuple):
    valid: list[Finding]
    invalid: list[Invalid]
    duplicates: list[Duplicate]


def location_problem(finding: Finding, lines: Optional[list[str]]) -> Optional[tuple[str, str]]:
    """(code, reason) if the finding's location or evidence is not backed by the file, else None."""
    if lines is None:
        return "invalid_path", "file could not be read from the snapshot"
    if finding.end_line > len(lines):
        return "invalid_location", f"line {finding.end_line} is beyond the end of the file ({len(lines)} lines)"
    if finding.end_line - finding.start_line + 1 > MAX_SPAN_LINES:
        return "invalid_location", f"line range spans more than {MAX_SPAN_LINES} lines"
    cited = normalize_text(mask_secrets("\n".join(lines[finding.start_line - 1 : finding.end_line])))
    needle = normalize_text(strip_numbering(finding.evidence))
    if not needle or needle not in cited:
        return "evidence_mismatch", "evidence text does not appear in the cited lines"
    return None


def _overlaps(a: Finding, b: Finding) -> bool:
    return a.file_path == b.file_path and a.category is b.category and a.start_line <= b.end_line and b.start_line <= a.end_line


def validate_candidates(root: Path, inventory: Inventory, findings: list[Finding]) -> ValidationResult:
    included = {e.path for e in inventory.entries if e.included}
    cache: dict[str, Optional[list[str]]] = {}
    valid: list[Finding] = []
    invalid: list[Invalid] = []

    for f in sorted(findings, key=lambda x: x.finding_id):
        if f.file_path not in included:
            invalid.append(Invalid(f, "invalid_path", "file is not part of the reviewed snapshot inventory"))
            continue
        if f.file_path not in cache:
            cache[f.file_path] = read_lines(root, f.file_path)
        problem = location_problem(f, cache[f.file_path])
        if problem:
            invalid.append(Invalid(f, *problem))
        else:
            valid.append(f)

    # Best finding first, so the survivor of an overlap is the most severe, then most confident.
    valid.sort(key=lambda x: (x.severity.rank, -x.confidence, x.finding_id))
    kept: list[Finding] = []
    duplicates: list[Duplicate] = []
    for f in valid:
        twin = next((k for k in kept if _overlaps(k, f)), None)
        if twin is None:
            kept.append(f)
        else:
            duplicates.append(Duplicate(f, twin.finding_id))
    return ValidationResult(kept, invalid, duplicates)
