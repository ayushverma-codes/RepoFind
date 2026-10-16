"""Build an immutable, content-addressed snapshot of the repository.

Layout under <data_dir>/snapshots/<snapshot_id>/:
    tree/           read-only copy of the repository
    manifest.json   {relative path: sha256} used to detect later tampering
    snapshot.json   Snapshot metadata
The snapshot id is derived from content, so ingesting the same repository twice
(from any source type) reuses one snapshot instead of creating a duplicate.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

from reviewharness.config import Settings
from reviewharness.ingest.acquire import Limits, Runner, Sink, clone_github, copy_local, extract_zip
from reviewharness.ingest.errors import AcquisitionError, IngestError
from reviewharness.ingest.models import Snapshot
from reviewharness.models import InputType, RepositoryInput


def snapshot_dir(snapshot: Snapshot) -> Path:
    return Path(snapshot.tree_path).parent


def hash_tree(tree: Path) -> dict[str, str]:
    """sha256 of every regular file, keyed by POSIX relative path."""
    manifest: dict[str, str] = {}
    for dirpath, _dirs, files in os.walk(tree, followlinks=False):
        for name in files:
            path = Path(dirpath) / name
            digest = hashlib.sha256()
            with open(path, "rb") as fh:
                while chunk := fh.read(1 << 20):
                    digest.update(chunk)
            manifest[path.relative_to(tree).as_posix()] = digest.hexdigest()
    return dict(sorted(manifest.items()))


def _snapshot_id(manifest: dict[str, str]) -> str:
    raw = "\n".join(f"{p}\0{h}" for p, h in manifest.items())
    return "S-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def remove_tree(path: Path) -> None:
    """Delete a tree even if its files were made read-only."""
    if not path.exists():
        return
    for dirpath, dirs, files in os.walk(path):
        for name in dirs + files:
            try:
                os.chmod(Path(dirpath) / name, stat.S_IRWXU)
            except OSError:
                pass
    shutil.rmtree(path, ignore_errors=True)


def _make_read_only(tree: Path) -> None:
    for dirpath, _dirs, files in os.walk(tree):
        for name in files:
            os.chmod(Path(dirpath) / name, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def verify_snapshot(snapshot: Snapshot) -> list[str]:
    """Return a list of problems (empty = snapshot matches its recorded manifest)."""
    manifest_path = snapshot_dir(snapshot) / "manifest.json"
    try:
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ["manifest.json is missing or unreadable"]
    actual = hash_tree(Path(snapshot.tree_path))
    problems = [f"modified: {p}" for p in recorded if p in actual and actual[p] != recorded[p]]
    problems += [f"missing: {p}" for p in recorded if p not in actual]
    problems += [f"unexpected: {p}" for p in actual if p not in recorded]
    return problems


def _acquire(repo: RepositoryInput, sink: Sink, settings: Settings, staging: Path, runner: Runner) -> None:
    if repo.input_type is InputType.LOCAL_FOLDER:
        copy_local(Path(repo.location), sink, settings.data_dir)
    elif repo.input_type is InputType.ZIP_FILE:
        extract_zip(Path(repo.location), sink)
    else:
        clone_dir = staging / "clone"
        clone_github(repo.location, clone_dir, settings.git_timeout_s, runner)
        copy_local(clone_dir, sink, settings.data_dir)


def create_snapshot(repo: RepositoryInput, settings: Settings, *, runner: Runner = subprocess.run) -> Snapshot:
    data_dir = settings.data_dir
    tmp_root, snaps_root = data_dir / "tmp", data_dir / "snapshots"
    tmp_root.mkdir(parents=True, exist_ok=True)
    snaps_root.mkdir(parents=True, exist_ok=True)

    staging = Path(tempfile.mkdtemp(prefix="stage-", dir=tmp_root))
    try:
        tree = staging / "tree"
        tree.mkdir()
        sink = Sink(tree, Limits(settings.max_repo_mb * 1024 * 1024, settings.max_files))
        _acquire(repo, sink, settings, staging, runner)
        if sink.files == 0:
            raise AcquisitionError("repository contains no reviewable files")

        manifest = hash_tree(tree)
        snapshot_id = _snapshot_id(manifest)
        final = snaps_root / snapshot_id

        if (final / "tree").is_dir():
            return _reuse(final, repo)

        snapshot = Snapshot(
            snapshot_id=snapshot_id,
            source=repo,
            tree_path=str((final / "tree").resolve()),
            file_count=sink.files,
            total_bytes=sink.bytes,
            stripped_root=sink.stripped_root,
            skipped=sink.skipped,
            skipped_total=sink.skipped_total,
        )
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        (staging / "snapshot.json").write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
        _make_read_only(tree)
        try:
            os.replace(staging, final)
        except OSError:
            if (final / "tree").is_dir():  # another process created it first
                return _reuse(final, repo)
            raise
        return snapshot
    finally:
        remove_tree(staging)


def _reuse(final: Path, repo: RepositoryInput) -> Snapshot:
    try:
        existing = Snapshot.model_validate_json((final / "snapshot.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise IngestError(f"existing snapshot {final.name} is unreadable; delete it and retry") from None
    existing = existing.model_copy(update={"source": repo, "reused": True, "tree_path": str((final / "tree").resolve())})
    problems = verify_snapshot(existing)
    if problems:
        raise IngestError(f"existing snapshot {final.name} was modified: {problems[0]}")
    return existing
