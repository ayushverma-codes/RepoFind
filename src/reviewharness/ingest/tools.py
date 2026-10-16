"""Approved static-analysis tools, run safely and recorded as structured data.

Approved set: a built-in Python syntax check (ast.parse, which never executes
code), plus ruff and bandit when installed. Tools run with shell=False, a
scrubbed environment, the snapshot as cwd, a timeout, and ignore the reviewed
repository's own configuration. A missing tool or a failure is recorded in the
ToolResult; it never aborts ingestion and never looks like "no issues found".
"""
from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from reviewharness.config import Settings
from reviewharness.ingest.models import Inventory, ToolIssue, ToolResult, ToolStatus
from reviewharness.ingest.safepath import resolve_inside
from reviewharness.ingest.subproc import safe_env
from reviewharness.logging_setup import redact

MAX_ISSUES_PER_TOOL = 500
MAX_OUTPUT_BYTES = 10 * 1024 * 1024
MAX_WARNINGS = 20
RUFF_RULES = "E9,F,B"  # syntax errors, pyflakes, bugbear


def find_executable(name: str) -> Optional[str]:
    """Prefer the tool installed next to this interpreter (works without venv activation)."""
    suffix = ".exe" if sys.platform == "win32" else ""
    sibling = Path(sys.executable).parent / f"{name}{suffix}"
    if sibling.is_file():
        return str(sibling)
    return shutil.which(name)


@dataclass
class _Parsed:
    issues: list[ToolIssue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dropped: int = 0  # reported against paths outside the snapshot


def _to_rel(raw: str, root: Path) -> Optional[str]:
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return None


def _positive(value: object, default: int = 1) -> int:
    return value if isinstance(value, int) and value >= 1 else default


# ------------------------------------------------------------------- parsers


def parse_ruff(stdout: str, root: Path) -> _Parsed:
    data = json.loads(stdout or "[]")
    if not isinstance(data, list):
        raise ValueError("unexpected ruff output shape")
    out = _Parsed()
    for item in data:
        rel = _to_rel(str(item.get("filename", "")), root)
        if rel is None:
            out.dropped += 1
            continue
        loc, end = item.get("location") or {}, item.get("end_location") or {}
        out.issues.append(
            ToolIssue(
                tool="ruff",
                rule_id=str(item.get("code") or "syntax-error"),
                message=redact(str(item.get("message", ""))),
                file_path=rel,
                line=_positive(loc.get("row")),
                end_line=_positive(end.get("row"), 0) or None,
            )
        )
    return out


def parse_bandit(stdout: str, root: Path) -> _Parsed:
    data = json.loads(stdout)
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError("unexpected bandit output shape")
    out = _Parsed()
    for item in data["results"]:
        rel = _to_rel(str(item.get("filename", "")), root)
        if rel is None:
            out.dropped += 1
            continue
        lines = item.get("line_range") or []
        out.issues.append(
            ToolIssue(
                tool="bandit",
                rule_id=str(item.get("test_id", "unknown")),
                message=redact(str(item.get("issue_text", ""))),
                file_path=rel,
                line=_positive(item.get("line_number")),
                end_line=_positive(max(lines), 0) or None if lines and all(isinstance(n, int) for n in lines) else None,
                severity=str(item.get("issue_severity", "")).lower() or None,
                confidence=str(item.get("issue_confidence", "")).lower() or None,
            )
        )
    for err in data.get("errors") or []:
        rel = _to_rel(str(err.get("filename", "")), root) or "<outside snapshot>"
        out.warnings.append(redact(f"{rel}: {err.get('reason', 'error')}"))
    return out


# ------------------------------------------------------------------- runners

Runner = Callable[..., "subprocess.CompletedProcess[bytes]"]


def _finish(tool: str, parsed: _Parsed, included: set[str], base: dict) -> ToolResult:
    kept = [i for i in parsed.issues if i.file_path in included]
    dropped = parsed.dropped + len(parsed.issues) - len(kept)
    kept.sort(key=lambda i: (i.file_path, i.line, i.rule_id, i.message))
    return ToolResult(
        tool=tool,
        status=ToolStatus.OK,
        issues=kept[:MAX_ISSUES_PER_TOOL],
        issue_total=len(kept),
        dropped=dropped,
        warnings=parsed.warnings[:MAX_WARNINGS],
        **base,
    )


_VERSION = re.compile(r"\d+\.\d+(?:\.\d+)?\S*")


def _version(tool: str, exe: str, cwd: Path, runner: Runner) -> Optional[str]:
    try:
        proc = runner([exe, "--version"], cwd=str(cwd), env=safe_env(), stdin=subprocess.DEVNULL,
                      capture_output=True, timeout=10, shell=False, check=False)
        match = _VERSION.search(proc.stdout.decode("utf-8", "replace"))
        return f"{tool} {match.group(0)}" if proc.returncode == 0 and match else None
    except (OSError, subprocess.SubprocessError):
        return None


def _run_external(
    tool: str,
    args: list[str],
    parser: Callable[[str, Path], _Parsed],
    root: Path,
    cwd: Path,
    included: set[str],
    settings: Settings,
    runner: Runner,
) -> ToolResult:
    """Run `args` (which contain the snapshot path) with `cwd` outside the snapshot."""
    exe = find_executable(tool)
    if exe is None:
        return ToolResult(
            tool=tool,
            status=ToolStatus.NOT_INSTALLED,
            error=f"{tool} is not installed (pip install 'reviewharness[analysis]')",
        )
    argv = [exe, *args]
    shown = ["<snapshot>" if a == str(root) else a for a in args]
    base: dict = {"command": [tool, *shown], "version": _version(tool, exe, cwd, runner)}
    start = time.monotonic()
    try:
        proc = runner(argv, cwd=str(cwd), env=safe_env(), stdin=subprocess.DEVNULL,
                      capture_output=True, timeout=settings.tool_timeout_s, shell=False, check=False)
    except subprocess.TimeoutExpired:
        return ToolResult(tool=tool, status=ToolStatus.TIMED_OUT, duration_s=round(time.monotonic() - start, 3),
                          error=f"{tool} timed out after {settings.tool_timeout_s:g}s (REVIEW_TOOL_TIMEOUT_S)", **base)
    except OSError as exc:
        return ToolResult(tool=tool, status=ToolStatus.FAILED, error=f"could not start {tool}: {exc.strerror}", **base)

    base["duration_s"] = round(time.monotonic() - start, 3)
    base["exit_code"] = proc.returncode
    stdout = proc.stdout[:MAX_OUTPUT_BYTES].decode("utf-8", "replace")
    if proc.returncode != 0:
        detail = redact(proc.stderr.decode("utf-8", "replace").strip())[-300:]
        return ToolResult(tool=tool, status=ToolStatus.FAILED, error=f"{tool} exited with {proc.returncode}: {detail}", **base)
    try:
        parsed = parser(stdout, root)
    except (ValueError, AttributeError, TypeError) as exc:
        return ToolResult(tool=tool, status=ToolStatus.FAILED, error=f"could not parse {tool} output: {exc}"[:300], **base)
    return _finish(tool, parsed, included, base)


def run_python_syntax(root: Path, inventory: Inventory) -> ToolResult:
    """Parse every included .py file with ast.parse. Parsing never executes code."""
    files = [e for e in inventory.entries if e.included and e.language == "python"]
    if not files:
        return ToolResult(tool="python_syntax", status=ToolStatus.SKIPPED, error="no Python files")
    start = time.monotonic()
    issues: list[ToolIssue] = []
    for entry in files:
        source = resolve_inside(root, entry.path).read_bytes()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ast.parse(source, filename=entry.path)
        except SyntaxError as exc:
            line = _positive(exc.lineno)
            issues.append(ToolIssue(tool="python_syntax", rule_id="SYNTAX", message=redact(str(exc.msg))[:200],
                                    file_path=entry.path, line=line, severity="high"))
        except (ValueError, RecursionError, MemoryError) as exc:
            issues.append(ToolIssue(tool="python_syntax", rule_id="UNPARSEABLE", message=type(exc).__name__,
                                    file_path=entry.path, line=1, severity="high"))
    issues.sort(key=lambda i: (i.file_path, i.line, i.rule_id))
    return ToolResult(
        tool="python_syntax",
        status=ToolStatus.OK,
        version=f"python {sys.version_info.major}.{sys.version_info.minor}",
        duration_s=round(time.monotonic() - start, 3),
        issues=issues[:MAX_ISSUES_PER_TOOL],
        issue_total=len(issues),
    )


def run_analysis_tools(root: Path, inventory: Inventory, settings: Settings, *, runner: Runner = subprocess.run) -> list[ToolResult]:
    """Run the approved tools sequentially (deterministic order) and return every result."""
    results = [run_python_syntax(root, inventory)]
    included = {e.path for e in inventory.entries if e.included}
    if not any(e.language == "python" for e in inventory.entries if e.included):
        results += [ToolResult(tool=t, status=ToolStatus.SKIPPED, error="no Python files") for t in ("ruff", "bandit")]
        return results
    # Tools run from a throwaway scratch directory, never from inside the snapshot,
    # so nothing a tool (or its launcher) writes can end up in the immutable tree.
    scratch_root = settings.data_dir / "tmp"
    scratch_root.mkdir(parents=True, exist_ok=True)
    target = str(root)
    with tempfile.TemporaryDirectory(prefix="tool-", dir=scratch_root, ignore_cleanup_errors=True) as scratch:
        cwd = Path(scratch)
        results.append(_run_external(
            "ruff",
            ["check", "--isolated", "--no-cache", "--no-respect-gitignore", "--output-format", "json",
             "--select", RUFF_RULES, "--exit-zero", target],
            parse_ruff, root, cwd, included, settings, runner,
        ))
        results.append(_run_external(
            "bandit", ["-r", target, "-f", "json", "-q", "--exit-zero"],
            parse_bandit, root, cwd, included, settings, runner,
        ))
    return results
