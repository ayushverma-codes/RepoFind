"""Safe acquisition: copy untrusted repository content into a staging tree.

Three sources (local folder, ZIP, GitHub clone) all funnel through _Sink, which
enforces the file-count and byte limits and writes only inside the staging tree.
Nothing from the repository is ever executed.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable

from reviewharness.ingest.errors import (
    AcquisitionError,
    RepositoryTooLargeError,
    UnsafeArchiveError,
)
from reviewharness.ingest.models import SkippedEntry, SkipReason
from reviewharness.ingest.safepath import resolve_inside
from reviewharness.ingest.subproc import safe_env
from reviewharness.logging_setup import redact
from reviewharness.models import normalize_repo_path

MAX_RECORDED_SKIPS = 1000
_CHUNK = 64 * 1024

IGNORED_DIR_NAMES = frozenset(
    {
        ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
        ".tox", ".nox", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".eggs",
        ".idea", ".vscode", ".reviewharness", "__MACOSX",
    }
)


def is_ignored_dir(name: str) -> bool:
    return name in IGNORED_DIR_NAMES or name.endswith(".egg-info")


@dataclass(frozen=True)
class Limits:
    max_total_bytes: int
    max_files: int


class Sink:
    """Writes files into `dest` while enforcing limits and recording skips."""

    def __init__(self, dest: Path, limits: Limits) -> None:
        self.dest = dest
        self.limits = limits
        self.files = 0
        self.bytes = 0
        self.skipped: list[SkippedEntry] = []
        self.skipped_total = 0
        self.stripped_root: str | None = None

    def skip(self, path: str, reason: SkipReason) -> None:
        self.skipped_total += 1
        if len(self.skipped) < MAX_RECORDED_SKIPS:
            self.skipped.append(SkippedEntry(path=path, reason=reason))

    def write(self, rel: str, reader: BinaryIO) -> None:
        if self.files >= self.limits.max_files:
            raise RepositoryTooLargeError(f"more than {self.limits.max_files} files (REVIEW_MAX_FILES)")
        target = resolve_inside(self.dest, rel)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            # "x" refuses to overwrite: duplicate archive members cannot clobber files.
            with open(target, "xb") as out:
                while chunk := reader.read(_CHUNK):
                    self.bytes += len(chunk)
                    if self.bytes > self.limits.max_total_bytes:
                        raise RepositoryTooLargeError("repository exceeds the size limit (REVIEW_MAX_REPO_MB)")
                    out.write(chunk)
        except OSError as exc:
            raise AcquisitionError(f"cannot write {rel[:100]!r}: {exc.strerror}") from None
        self.files += 1


# --------------------------------------------------------------- local folder


def copy_local(src: Path, sink: Sink, data_dir: Path) -> None:
    """Copy a folder without following symlinks and without touching `data_dir`."""
    try:
        src = src.expanduser().resolve(strict=True)
    except OSError:
        raise AcquisitionError("local folder does not exist") from None
    if not src.is_dir():
        raise AcquisitionError("local path is not a directory")
    data_dir_resolved = data_dir.resolve()

    for dirpath, dirnames, filenames in os.walk(src, followlinks=False):
        here = Path(dirpath)
        keep = []
        for name in sorted(dirnames):
            child = here / name
            rel = child.relative_to(src).as_posix()
            if child.is_symlink():
                sink.skip(rel, SkipReason.SYMLINK)
            elif is_ignored_dir(name) or child.resolve() == data_dir_resolved:
                sink.skip(rel, SkipReason.IGNORED_DIR)
            else:
                keep.append(name)
        dirnames[:] = keep

        for name in sorted(filenames):
            path = here / name
            rel = path.relative_to(src).as_posix()
            try:
                mode = os.lstat(path).st_mode
                if stat.S_ISLNK(mode):
                    sink.skip(rel, SkipReason.SYMLINK)
                elif not stat.S_ISREG(mode):
                    sink.skip(rel, SkipReason.SPECIAL_FILE)
                else:
                    with open(path, "rb") as fh:
                        sink.write(rel, fh)
            except PermissionError:
                sink.skip(rel, SkipReason.UNREADABLE)


# ------------------------------------------------------------------------ ZIP


def extract_zip(zip_path: Path, sink: Sink) -> None:
    """Extract a ZIP defensively. Hostile archives are rejected as a whole."""
    zip_path = zip_path.expanduser()
    if not zip_path.is_file():
        raise AcquisitionError("zip file does not exist")
    try:
        archive = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError):
        raise AcquisitionError("not a valid zip archive") from None

    with archive:
        infos = archive.infolist()
        if len(infos) > sink.limits.max_files * 2:
            raise RepositoryTooLargeError("archive has too many entries (REVIEW_MAX_FILES)")

        files: list[tuple[zipfile.ZipInfo, str]] = []
        declared = 0
        for info in infos:
            try:
                rel = normalize_repo_path(info.filename)
            except ValueError:
                if info.is_dir() and info.filename.strip("./\\") == "":
                    continue
                raise UnsafeArchiveError(f"unsafe path in archive: {info.filename[:100]!r}") from None
            if info.is_dir():
                continue
            if info.flag_bits & 0x1:
                raise UnsafeArchiveError("encrypted archive entries are not supported")
            declared += info.file_size
            files.append((info, rel))
        if declared > sink.limits.max_total_bytes:
            raise RepositoryTooLargeError("archive declares more data than the size limit (REVIEW_MAX_REPO_MB)")

        usable = []
        for info, rel in files:
            if rel.split("/")[0] == "__MACOSX":
                sink.skip(rel, SkipReason.IGNORED_DIR)
            else:
                usable.append((info, rel))

        # GitHub-style archives wrap everything in one folder: strip it.
        strip = 0
        parts = [rel.split("/") for _, rel in usable]
        if parts and all(len(p) >= 2 for p in parts) and len({p[0] for p in parts}) == 1:
            sink.stripped_root = parts[0][0]
            strip = 1

        recorded_dirs: set[str] = set()
        for (info, _), segs in zip(usable, parts):
            segs = segs[strip:]
            rel = "/".join(segs)
            ignored_at = next((i for i, s in enumerate(segs[:-1]) if is_ignored_dir(s)), None)
            if ignored_at is not None:
                prefix = "/".join(segs[: ignored_at + 1])
                if prefix not in recorded_dirs:
                    recorded_dirs.add(prefix)
                    sink.skip(prefix, SkipReason.IGNORED_DIR)
                continue
            unix_mode = info.external_attr >> 16 if info.create_system == 3 else 0
            file_type = stat.S_IFMT(unix_mode)  # 0 when the archiver stored no type bits
            if file_type == stat.S_IFLNK:
                sink.skip(rel, SkipReason.SYMLINK)
                continue
            if file_type not in (0, stat.S_IFREG):
                sink.skip(rel, SkipReason.SPECIAL_FILE)
                continue
            try:
                with archive.open(info) as fh:
                    sink.write(rel, fh)
            except (zipfile.BadZipFile, NotImplementedError, RuntimeError, zlib.error, EOFError):
                raise AcquisitionError(f"corrupt or unsupported archive entry: {rel[:100]!r}") from None


# --------------------------------------------------------------------- GitHub

Runner = Callable[..., "subprocess.CompletedProcess[bytes]"]


def clone_github(url: str, dest: Path, timeout_s: float, runner: Runner = subprocess.run) -> None:
    """Shallow-clone a GitHub repo with hooks, filters and symlinks neutralised.

    Limitation: the clone itself is bounded by the timeout, not by bytes; the
    size limit is enforced when the clone is copied into the snapshot.
    """
    git = shutil.which("git")
    if git is None:
        raise AcquisitionError("git is not installed; use a ZIP or local folder instead")
    argv = [
        git,
        "-c", f"core.hooksPath={os.devnull}",
        "-c", "core.fsmonitor=false",
        "-c", "core.symlinks=false",
        "-c", "init.templateDir=",
        "-c", "protocol.allow=never",
        "-c", "protocol.https.allow=always",
        "-c", "transfer.fsckObjects=true",
        "clone", "--depth", "1", "--single-branch", "--no-tags", "--quiet",
        "--", url, str(dest),
    ]
    env = safe_env(network=True)
    env.update(
        GIT_TERMINAL_PROMPT="0",
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_LFS_SKIP_SMUDGE="1",
    )
    try:
        proc = runner(argv, env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout_s, shell=False, check=False)
    except subprocess.TimeoutExpired:
        raise AcquisitionError(f"git clone timed out after {timeout_s:g}s (REVIEW_GIT_TIMEOUT_S)") from None
    except OSError as exc:
        raise AcquisitionError(f"could not run git: {exc.strerror}") from None
    if proc.returncode != 0:
        detail = redact(proc.stderr.decode("utf-8", "replace").strip())[-300:]
        raise AcquisitionError(f"git clone failed (exit {proc.returncode}): {detail}")
