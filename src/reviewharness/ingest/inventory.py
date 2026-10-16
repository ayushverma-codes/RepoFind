"""Deterministic file inventory: what is in the snapshot and what is reviewable."""
from __future__ import annotations

import codecs
import hashlib
import os
import stat
from pathlib import Path, PurePosixPath

from reviewharness.ingest.models import Inventory, InventoryEntry, SkipReason

SNIFF_BYTES = 8192

LANGUAGES = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".java": "java", ".kt": "kotlin",
    ".go": "go", ".rs": "rust", ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp",
    ".hpp": "cpp", ".cs": "csharp", ".rb": "ruby", ".php": "php", ".swift": "swift",
    ".sh": "shell", ".bash": "shell", ".sql": "sql", ".html": "html", ".css": "css",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".ini": "ini",
    ".cfg": "ini", ".md": "markdown", ".rst": "rst", ".txt": "text",
}
NAME_LANGUAGES = {"Dockerfile": "dockerfile", "Makefile": "makefile"}

BINARY_EXTS = frozenset(
    ".png .jpg .jpeg .gif .bmp .ico .webp .tiff .svgz .pdf .zip .gz .tgz .bz2 .xz .7z .rar "
    ".tar .jar .war .class .pyc .pyo .so .dll .dylib .exe .o .a .bin .dat .woff .woff2 .ttf "
    ".otf .eot .mp3 .mp4 .mov .avi .wav .ogg .webm .sqlite .db .pkl".split()
)
GENERATED_NAMES = frozenset(
    {
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Pipfile.lock",
        "Cargo.lock", "composer.lock", "Gemfile.lock", "go.sum",
    }
)
GENERATED_SUFFIXES = (".min.js", ".min.css", ".map")
TEST_DIRS = frozenset({"tests", "test", "__tests__", "spec"})


def language_for(path: str) -> str:
    name = PurePosixPath(path).name
    if name in NAME_LANGUAGES:
        return NAME_LANGUAGES[name]
    return LANGUAGES.get(PurePosixPath(path).suffix.lower(), "other")


def is_test_path(path: str) -> bool:
    p = PurePosixPath(path)
    name = p.name.lower()
    if any(part.lower() in TEST_DIRS for part in p.parts[:-1]):
        return True
    return (
        name.startswith("test_")
        or name.endswith(("_test.py", "_test.go", ".test.js", ".test.ts", ".spec.js", ".spec.ts"))
    )


def _looks_binary(sample: bytes) -> bool:
    return b"\x00" in sample


def _classify(abs_path: Path, rel: str, max_file_bytes: int) -> InventoryEntry:
    def skipped(size: int, reason: SkipReason) -> InventoryEntry:
        return InventoryEntry(path=rel, size=size, included=False, skip_reason=reason)

    st = os.lstat(abs_path)
    if stat.S_ISLNK(st.st_mode):
        return skipped(0, SkipReason.SYMLINK)
    if not stat.S_ISREG(st.st_mode):
        return skipped(0, SkipReason.SPECIAL_FILE)
    size = st.st_size
    name = PurePosixPath(rel).name
    if size == 0:
        return skipped(size, SkipReason.EMPTY)
    if PurePosixPath(rel).suffix.lower() in BINARY_EXTS:
        return skipped(size, SkipReason.BINARY)
    if name in GENERATED_NAMES or name.lower().endswith(GENERATED_SUFFIXES):
        return skipped(size, SkipReason.GENERATED)
    if size > max_file_bytes:
        return skipped(size, SkipReason.TOO_LARGE)

    data = abs_path.read_bytes()
    if _looks_binary(data[:SNIFF_BYTES]):
        return skipped(size, SkipReason.BINARY)
    try:
        text = codecs.decode(data, "utf-8")
    except UnicodeDecodeError:
        return skipped(size, SkipReason.NOT_UTF8)

    lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    return InventoryEntry(
        path=rel,
        size=size,
        included=True,
        language=language_for(rel),
        lines=lines,
        sha256=hashlib.sha256(data).hexdigest(),
        is_test=is_test_path(rel),
    )


def build_inventory(root: Path, max_file_bytes: int) -> Inventory:
    """Classify every file under `root`. Output is sorted, so it is reproducible."""
    entries: list[InventoryEntry] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for name in sorted(filenames):
            abs_path = Path(dirpath) / name
            entries.append(_classify(abs_path, abs_path.relative_to(root).as_posix(), max_file_bytes))
    entries.sort(key=lambda e: e.path)
    return Inventory(entries=entries)
