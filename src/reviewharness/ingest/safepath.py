"""The one place that turns an untrusted relative path into a real path.

Every file access inside a snapshot (extraction, inventory, tools, later the
specialists' source excerpts) should go through resolve_inside().
"""
from __future__ import annotations

from pathlib import Path

from reviewharness.ingest.errors import PathEscapeError
from reviewharness.models import normalize_repo_path


def resolve_inside(root: Path, rel: str) -> Path:
    """Return the absolute path of `rel` under `root`, or raise PathEscapeError.

    Rejects '..', absolute and drive-letter paths, null bytes, any symlink
    component, and anything whose resolved location is outside `root`.
    """
    try:
        clean = normalize_repo_path(rel)
    except ValueError as exc:
        raise PathEscapeError(f"unsafe path {str(rel)[:100]!r}: {exc}") from None

    root_resolved = root.resolve()
    current = root_resolved
    for part in clean.split("/"):
        current = current / part
        if current.is_symlink():
            raise PathEscapeError(f"symlink in path {clean[:100]!r}")

    resolved = current.resolve()
    if resolved != root_resolved and not resolved.is_relative_to(root_resolved):
        raise PathEscapeError(f"path escapes the repository root: {clean[:100]!r}")
    return resolved
