"""Independent source access for the checker. Reads the snapshot itself, never the specialist's excerpt."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from reviewharness.ingest.errors import IngestError
from reviewharness.ingest.safepath import resolve_inside

CONTEXT_LINES = 12
MAX_WINDOW_LINES = 120


def read_lines(root: Path, rel: str) -> Optional[list[str]]:
    """All lines of a snapshot file, or None if it cannot be read safely."""
    try:
        return resolve_inside(root, rel).read_text(encoding="utf-8").splitlines()
    except (IngestError, OSError, UnicodeDecodeError):
        return None


def render_window(path: str, lines: list[str], start: int, end: int) -> str:
    """Numbered source around [start, end] with real line numbers, as an untrusted-data block."""
    span = end - start + 1
    ctx = max(0, min(CONTEXT_LINES, (MAX_WINDOW_LINES - span) // 2))
    first, last = max(1, start - ctx), min(len(lines), end + ctx)
    body = "\n".join(f"{n:>5}| {lines[n - 1]}" for n in range(first, last + 1))
    return f"##### FILE {path} (lines {first}-{last} of {len(lines)}) #####\n{body}\n##### END FILE #####"
