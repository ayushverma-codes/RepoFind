"""Deterministic repository context: facts a planner/specialist needs up front.

Everything here is extracted by plain code. README text is carried as an
explicitly untrusted excerpt; it is data, never instructions.
"""
from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

from reviewharness.ingest.models import InventoryEntry, Inventory, LargeFile, ReadmeExcerpt, RepositoryContext
from reviewharness.ingest.safepath import resolve_inside
from reviewharness.logging_setup import redact

README_CHARS = 1500
MAX_LIST = 200
MAX_DEPS = 100

MANIFEST_NAMES = frozenset(
    {
        "pyproject.toml", "setup.py", "setup.cfg", "Pipfile", "package.json", "go.mod",
        "Cargo.toml", "pom.xml", "build.gradle", "Gemfile", "composer.json", "Dockerfile", "Makefile",
    }
)
ENTRY_NAMES = frozenset(
    {"main.py", "__main__.py", "app.py", "cli.py", "manage.py", "wsgi.py", "asgi.py",
     "index.js", "server.js", "main.go", "main.rs", "Main.java"}
)
_MAIN_GUARD = re.compile(r"""if\s+__name__\s*==\s*["']__main__["']""")
_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _read(root: Path, entry: InventoryEntry) -> str:
    return resolve_inside(root, entry.path).read_text(encoding="utf-8", errors="replace")


def parse_requirements(text: str) -> list[str]:
    names = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-", "git+", "http")):
            continue
        match = _REQ_NAME.match(line)
        if match:
            names.append(match.group(1).lower())
    return sorted(set(names))[:MAX_DEPS]


def parse_pyproject(text: str) -> list[str]:
    data = tomllib.loads(text)
    project = data.get("project", {})
    specs = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        specs.extend(group)
    names = []
    for spec in specs:
        match = _REQ_NAME.match(str(spec))
        if match:
            names.append(match.group(1).lower())
    return sorted(set(names))[:MAX_DEPS]


def parse_package_json(text: str) -> list[str]:
    data = json.loads(text)
    names = set()
    for key in ("dependencies", "devDependencies"):
        section = data.get(key, {})
        if isinstance(section, dict):
            names.update(section)
    return sorted(names)[:MAX_DEPS]


def _dependencies(root: Path, manifests: list[InventoryEntry], notes: list[str]) -> dict[str, list[str]]:
    deps: dict[str, list[str]] = {}
    for entry in manifests:
        name = PurePosixPath(entry.path).name
        parser = None
        if name.startswith("requirements") and name.endswith(".txt"):
            parser = parse_requirements
        elif name == "pyproject.toml":
            parser = parse_pyproject
        elif name == "package.json":
            parser = parse_package_json
        if parser is None:
            continue
        try:
            deps[entry.path] = parser(_read(root, entry))
        except (ValueError, AttributeError, TypeError):  # TOMLDecodeError/JSONDecodeError subclass ValueError
            notes.append(f"could not parse dependencies in {entry.path}")
    return deps


def build_context(root: Path, inventory: Inventory) -> RepositoryContext:
    included = [e for e in inventory.entries if e.included]
    notes: list[str] = []

    manifests = [
        e for e in included
        if PurePosixPath(e.path).name in MANIFEST_NAMES
        or (PurePosixPath(e.path).name.startswith("requirements") and e.path.endswith(".txt"))
    ]

    entry_points = {e.path for e in included if PurePosixPath(e.path).name in ENTRY_NAMES}
    for e in included:
        if e.language == "python" and not e.is_test and e.path not in entry_points:
            if _MAIN_GUARD.search(_read(root, e)):
                entry_points.add(e.path)

    ci = [
        e.path for e in included
        if e.path.startswith((".github/workflows/", ".circleci/")) or PurePosixPath(e.path).name in (".gitlab-ci.yml", "Jenkinsfile")
    ]

    readme = None
    for e in included:
        if "/" not in e.path and PurePosixPath(e.path).name.lower().startswith("readme"):
            readme = ReadmeExcerpt(path=e.path, text=redact(_read(root, e)[:README_CHARS]))
            break

    test_files = sum(1 for e in included if e.is_test)
    code_files = [e for e in included if e.language not in ("markdown", "rst", "text", "json", "yaml", "toml", "ini", "other", None)]
    source_files = sum(1 for e in code_files if not e.is_test)

    if not included:
        notes.append("no reviewable files found")
    if test_files == 0:
        notes.append("no test files detected")
    if not manifests:
        notes.append("no dependency manifest found")
    skipped = len(inventory.entries) - len(included)
    if skipped:
        notes.append(f"{skipped} file(s) excluded from review; see inventory skip reasons")

    largest = sorted(included, key=lambda e: (-(e.lines or 0), e.path))[:10]
    tree = sorted(
        {(p[0] + "/" if len(p) > 1 else p[0]) for p in (PurePosixPath(e.path).parts for e in included)}
    )[:MAX_LIST]  # top-level entries; directories end with "/"

    return RepositoryContext(
        languages=inventory.languages,
        included_files=len(included),
        total_lines=inventory.total_lines,
        source_files=source_files,
        test_files=test_files,
        manifests=sorted(e.path for e in manifests)[:MAX_LIST],
        dependencies=_dependencies(root, manifests, notes),
        entry_points=sorted(entry_points)[:MAX_LIST],
        ci_configs=sorted(ci)[:MAX_LIST],
        largest_files=[LargeFile(path=e.path, lines=e.lines or 0) for e in largest],
        top_level_tree=tree,
        readme_excerpt=readme,
        notes=notes,
    )
