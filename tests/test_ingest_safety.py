"""Path safety, hostile archives, limits, snapshot immutability and git handling."""
import os
import stat
import subprocess
import warnings
import zipfile
from pathlib import Path

import pytest

from reviewharness.ingest.acquire import clone_github
from reviewharness.ingest.errors import (
    AcquisitionError,
    IngestError,
    PathEscapeError,
    RepositoryTooLargeError,
    UnsafeArchiveError,
)
from reviewharness.ingest.safepath import resolve_inside
from reviewharness.ingest.snapshot import create_snapshot, snapshot_dir, verify_snapshot
from reviewharness.models import InputType, RepositoryInput


def local(path) -> RepositoryInput:
    return RepositoryInput(input_type=InputType.LOCAL_FOLDER, location=str(path))


def zipped(path) -> RepositoryInput:
    return RepositoryInput(input_type=InputType.ZIP_FILE, location=str(path))


def make_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def symlink_or_skip(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported here")


def no_staging_left(settings) -> bool:
    tmp = settings.data_dir / "tmp"
    return not tmp.exists() or not any(tmp.iterdir())


# ------------------------------------------------------------- resolve_inside


def test_resolve_inside_accepts_normal_paths(tmp_path):
    (tmp_path / "a").mkdir()
    assert resolve_inside(tmp_path, "a/b.py") == tmp_path.resolve() / "a" / "b.py"


@pytest.mark.parametrize("bad", ["../x", "a/../../x", "/etc/passwd", "C:\\Windows\\x", "a\x00b", "", ".."])
def test_resolve_inside_rejects_unsafe(tmp_path, bad):
    with pytest.raises(PathEscapeError):
        resolve_inside(tmp_path, bad)


def test_resolve_inside_rejects_symlink_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("s")
    root = tmp_path / "root"
    root.mkdir()
    symlink_or_skip(root / "link", outside)
    with pytest.raises(PathEscapeError):
        resolve_inside(root, "link/secret.txt")


# ----------------------------------------------------------------------- ZIP


@pytest.mark.parametrize("name", ["../evil.py", "a/../../evil.py", "/abs/evil.py", "..\\evil.py", "C:\\evil.py"])
def test_zip_path_traversal_rejects_whole_archive(tmp_path, make_settings, name):
    settings = make_settings()
    archive = make_zip(tmp_path / "bad.zip", {"ok.py": b"x = 1\n", name: b"boom"})
    with pytest.raises(UnsafeArchiveError):
        create_snapshot(zipped(archive), settings)
    assert not (tmp_path / "evil.py").exists()
    assert no_staging_left(settings)


def test_zip_symlink_member_is_skipped_not_extracted(tmp_path, make_settings):
    archive = tmp_path / "link.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("ok.py", "x = 1\n")
        info = zipfile.ZipInfo("link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "/etc/passwd")
    snap = create_snapshot(zipped(archive), make_settings())
    assert not (Path(snap.tree_path) / "link").exists()
    assert [(s.path, s.reason.value) for s in snap.skipped] == [("link", "symlink")]


def test_zip_duplicate_members_rejected(tmp_path, make_settings):
    archive = tmp_path / "dup.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("a.py", "one")
            zf.writestr("a.py", "two")
    with pytest.raises(IngestError):
        create_snapshot(zipped(archive), make_settings())


def test_zip_bomb_rejected_by_declared_size(tmp_path, make_settings):
    archive = make_zip(tmp_path / "bomb.zip", {"big.txt": b"\0" * (3 * 1024 * 1024)})
    assert archive.stat().st_size < 50_000  # tiny on disk, large when extracted
    settings = make_settings(max_repo_mb=1)
    with pytest.raises(RepositoryTooLargeError):
        create_snapshot(zipped(archive), settings)
    assert no_staging_left(settings)


def test_zip_file_count_limit(tmp_path, make_settings):
    archive = make_zip(tmp_path / "many.zip", {f"f{i}.py": b"x=1\n" for i in range(5)})
    with pytest.raises(RepositoryTooLargeError):
        create_snapshot(zipped(archive), make_settings(max_files=3))


def test_zip_single_top_level_folder_is_stripped(tmp_path, make_settings):
    archive = make_zip(tmp_path / "gh.zip", {"proj-main/a.py": b"x=1\n", "proj-main/src/b.py": b"y=2\n"})
    snap = create_snapshot(zipped(archive), make_settings())
    assert snap.stripped_root == "proj-main"
    assert (Path(snap.tree_path) / "src" / "b.py").is_file()


def test_zip_ignored_dirs_and_macosx_skipped(tmp_path, make_settings):
    archive = make_zip(
        tmp_path / "x.zip",
        {"a.py": b"x=1\n", "node_modules/pkg/i.js": b"1", ".git/config": b"[core]", "__MACOSX/._a.py": b"junk"},
    )
    snap = create_snapshot(zipped(archive), make_settings())
    assert not (Path(snap.tree_path) / "node_modules").exists()
    assert not (Path(snap.tree_path) / ".git").exists()
    assert snap.file_count == 1 and snap.skipped_total == 3


def test_zip_invalid_or_missing(tmp_path, make_settings):
    fake = tmp_path / "fake.zip"
    fake.write_text("not a zip")
    with pytest.raises(AcquisitionError):
        create_snapshot(zipped(fake), make_settings())
    with pytest.raises(AcquisitionError):
        create_snapshot(zipped(tmp_path / "missing.zip"), make_settings())


# --------------------------------------------------------------- local folder


def test_local_symlinks_are_never_followed(tmp_path, make_settings):
    secret_dir = tmp_path / "outside"
    secret_dir.mkdir()
    (secret_dir / "secret.txt").write_text("TOP-SECRET")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("x = 1\n")
    symlink_or_skip(repo / "file_link", secret_dir / "secret.txt")
    symlink_or_skip(repo / "dir_link", secret_dir)

    snap = create_snapshot(local(repo), make_settings())
    tree = Path(snap.tree_path)
    assert sorted(p.name for p in tree.rglob("*")) == ["a.py"]
    assert {s.reason.value for s in snap.skipped} == {"symlink"}
    assert "TOP-SECRET" not in "".join(p.read_text() for p in tree.rglob("*") if p.is_file())


def test_local_ignored_dirs_and_data_dir_excluded(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "config").write_text("[core]")
    (repo / "node_modules" / "p").mkdir(parents=True)
    (repo / "node_modules" / "p" / "i.js").write_text("1")
    (repo / "keep.py").write_text("x = 1\n")
    from reviewharness.config import Settings

    settings = Settings(data_dir=repo / ".reviewharness")  # data dir lives inside the repo
    first = create_snapshot(local(repo), settings)
    second = create_snapshot(local(repo), settings)  # must not recurse into its own snapshots
    assert first.file_count == 1 and first.snapshot_id == second.snapshot_id


def test_local_missing_or_not_a_directory(tmp_path, make_settings):
    with pytest.raises(AcquisitionError):
        create_snapshot(local(tmp_path / "nope"), make_settings())
    f = tmp_path / "file.txt"
    f.write_text("x")
    with pytest.raises(AcquisitionError):
        create_snapshot(local(f), make_settings())


def test_local_size_limit_and_cleanup(tmp_path, make_settings):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "big.txt").write_bytes(b"a" * (2 * 1024 * 1024))
    settings = make_settings(max_repo_mb=1)
    with pytest.raises(RepositoryTooLargeError):
        create_snapshot(local(repo), settings)
    assert no_staging_left(settings)
    assert not (settings.data_dir / "snapshots").exists() or not any((settings.data_dir / "snapshots").iterdir())


def test_empty_repo_rejected(tmp_path, make_settings):
    (tmp_path / "empty").mkdir()
    with pytest.raises(AcquisitionError):
        create_snapshot(local(tmp_path / "empty"), make_settings())


# ------------------------------------------------------------------ snapshot


def test_same_content_reuses_snapshot_across_source_types(tmp_path, make_settings, sample_repo):
    settings = make_settings()
    first = create_snapshot(local(sample_repo), settings)
    again = create_snapshot(local(sample_repo), settings)
    assert not first.reused and again.reused and first.snapshot_id == again.snapshot_id

    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for p in sorted(sample_repo.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(sample_repo).as_posix())
    from_zip = create_snapshot(zipped(archive), settings)
    assert from_zip.reused and from_zip.snapshot_id == first.snapshot_id
    assert len(list((settings.data_dir / "snapshots").iterdir())) == 1


def test_snapshot_files_are_read_only_and_tampering_detected(make_settings, sample_repo):
    snap = create_snapshot(local(sample_repo), make_settings())
    target = Path(snap.tree_path) / "utils.py"
    assert not os.access(target, os.W_OK) or os.name == "nt" or os.geteuid() == 0
    assert verify_snapshot(snap) == []

    target.chmod(0o644)
    target.write_text("def add(a, b):\n    return 0\n")
    assert verify_snapshot(snap) == ["modified: utils.py"]
    with pytest.raises(IngestError, match="modified"):
        create_snapshot(local(sample_repo), make_settings())


def test_snapshot_dir_layout(make_settings, sample_repo):
    snap = create_snapshot(local(sample_repo), make_settings())
    d = snapshot_dir(snap)
    assert (d / "tree").is_dir() and (d / "manifest.json").is_file() and (d / "snapshot.json").is_file()


# -------------------------------------------------------------------- GitHub


class FakeGit:
    def __init__(self, returncode=0, stderr=b"", files=None, raises=None):
        self.returncode, self.stderr, self.files, self.raises = returncode, stderr, files or {}, raises
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if self.raises:
            raise self.raises
        dest = Path(argv[-1])
        for name, data in self.files.items():
            (dest / name).parent.mkdir(parents=True, exist_ok=True)
            (dest / name).write_bytes(data)
        return subprocess.CompletedProcess(argv, self.returncode, b"", self.stderr)


@pytest.fixture
def need_git(monkeypatch):
    monkeypatch.setattr("reviewharness.ingest.acquire.shutil.which", lambda name: "/usr/bin/git")


GH = RepositoryInput(input_type=InputType.GITHUB_URL, location="https://github.com/octocat/Hello-World")


def test_github_clone_is_hardened_and_snapshotted(monkeypatch, make_settings, need_git):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_abcdefghijklmnopqrstuv")
    fake = FakeGit(files={"a.py": b"x = 1\n", ".git/config": b"[core]", "README.md": b"hi\n"})
    snap = create_snapshot(GH, make_settings(), runner=fake)
    argv, kwargs = fake.calls[0]
    assert argv[argv.index("--") + 1] == GH.location
    assert any(a.startswith("core.hooksPath=") for a in argv)
    assert "protocol.allow=never" in argv and kwargs["shell"] is False
    assert "GROQ_API_KEY" not in kwargs["env"]
    assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert sorted(p.name for p in Path(snap.tree_path).iterdir()) == ["README.md", "a.py"]  # .git dropped


def test_github_failure_is_redacted(tmp_path, need_git):
    fake = FakeGit(returncode=128, stderr=b"fatal: auth failed token=gsk_abcdefghijklmnopqrstuv")
    with pytest.raises(AcquisitionError) as exc:
        clone_github(GH.location, tmp_path / "c", 5, fake)
    assert "gsk_abcdefghijklmnopqrstuv" not in str(exc.value) and "128" in str(exc.value)


def test_github_timeout_and_missing_git(tmp_path, monkeypatch, need_git):
    with pytest.raises(AcquisitionError, match="timed out"):
        clone_github(GH.location, tmp_path / "c", 5, FakeGit(raises=subprocess.TimeoutExpired("git", 5)))
    monkeypatch.setattr("reviewharness.ingest.acquire.shutil.which", lambda name: None)
    with pytest.raises(AcquisitionError, match="git is not installed"):
        clone_github(GH.location, tmp_path / "c", 5, FakeGit())
