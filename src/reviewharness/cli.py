"""Command-line entry point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from dotenv import load_dotenv
from pydantic import ValidationError

from reviewharness import __version__
from reviewharness.config import Settings
from reviewharness.ingest import IngestError, IngestResult, ingest_repository
from reviewharness.ingest.snapshot import snapshot_dir
from reviewharness.logging_setup import setup_logging
from reviewharness.models import (
    CheckerStatus,
    Finding,
    InputType,
    Report,
    RepositoryInput,
    ReviewTask,
    RunRecord,
    RunStatus,
    SpecialistSource,
)


def _format_errors(exc: ValidationError) -> str:
    # Only location and message: never echo input values (they may be secrets).
    return "\n".join(
        f"  - {'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()
    )


def run_smoke() -> None:
    """Build one of every contract, round-trip through JSON, and check invariants."""
    repo = RepositoryInput(input_type=InputType.LOCAL_FOLDER, location="./sample")
    finding = Finding(
        category="bug",
        title="Off-by-one in loop bound",
        severity="high",
        file_path="src/app.py",
        start_line=10,
        end_line=12,
        evidence="for i in range(len(items) + 1):",
        rationale="Index goes one past the end of the list.",
        impact="IndexError at runtime.",
        confidence=0.8,
        source="bug",
    )
    task = ReviewTask(task_id="t1", run_id="r1", specialist=SpecialistSource.BUG)
    run = RunRecord(run_id="r1", repository=repo, tasks=[task])
    accepted = finding.model_copy(update={"checker_status": CheckerStatus.ACCEPTED})
    report = Report(run_id=run.run_id, run_status=RunStatus.COMPLETED, findings=[accepted])

    restored = Report.model_validate_json(report.model_dump_json())
    assert restored.is_complete and restored.counts_by_severity["high"] == 1
    assert Finding.model_validate_json(finding.model_dump_json()) == finding


def _cmd_version(_: argparse.Namespace) -> int:
    print(f"reviewharness {__version__}")
    return 0


def _cmd_config(_: argparse.Namespace) -> int:
    settings = Settings.from_env()
    print(json.dumps(settings.public_dict(), indent=2))
    return 0


def _cmd_validate_finding(args: argparse.Namespace) -> int:
    path = Path(args.path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"cannot read {path}: {exc.strerror}", file=sys.stderr)
        return 2
    try:
        finding = Finding.model_validate_json(raw)
    except ValidationError as exc:
        print("INVALID finding:\n" + _format_errors(exc), file=sys.stderr)
        return 1
    print(f"VALID finding {finding.finding_id}")
    return 0


def _cmd_smoke(_: argparse.Namespace) -> int:
    try:
        run_smoke()
    except (ValidationError, AssertionError) as exc:
        print(f"SMOKE FAILED: {type(exc).__name__}", file=sys.stderr)
        return 1
    print("SMOKE OK")
    return 0


_INGEST_TYPES = {
    "auto": None,
    "github": InputType.GITHUB_URL,
    "zip": InputType.ZIP_FILE,
    "folder": InputType.LOCAL_FOLDER,
}


def format_ingest_summary(result: IngestResult) -> str:
    snap, inv, ctx = result.snapshot, result.inventory, result.context
    skipped = ", ".join(f"{k}={v}" for k, v in inv.skipped_counts.items()) or "none"
    langs = ", ".join(f"{k}={v}" for k, v in inv.languages.items()) or "none"
    lines = [
        f"Snapshot {snap.snapshot_id} ({'reused' if snap.reused else 'new'})",
        f"  source:   {snap.source.input_type.value} {snap.source.location}",
        f"  copied:   {snap.file_count} files, {snap.total_bytes} bytes; dropped at copy: {snap.skipped_total}",
        f"Inventory: {inv.included_count} reviewable of {len(inv.entries)} files, {inv.total_lines} lines",
        f"  excluded: {skipped}",
        f"  languages: {langs}",
        f"Context: source_files={ctx.source_files} test_files={ctx.test_files} "
        f"manifests={len(ctx.manifests)} entry_points={len(ctx.entry_points)}",
    ]
    lines += [f"  note: {n}" for n in ctx.notes]
    if result.tool_results:
        lines.append("Tools:")
        for t in result.tool_results:
            version = f" ({t.version})" if t.version else ""
            detail = f"{t.issue_total} issue(s)" if t.status.value == "ok" else (t.error or "")
            lines.append(f"  {t.tool}{version}: {t.status.value} - {detail}")
            lines += [f"    warning: {w}" for w in t.warnings[:3]]
    else:
        lines.append("Tools: not run (--no-tools)")
    lines.append(f"Artifacts: {snapshot_dir(snap)}")
    return "\n".join(lines)


def _cmd_ingest(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    try:
        result = ingest_repository(
            args.source, _INGEST_TYPES[args.type], settings, run_tools=not args.no_tools
        )
    except ValidationError as exc:
        print("Invalid input:\n" + _format_errors(exc), file=sys.stderr)
        return 2
    except IngestError as exc:
        print(f"INGEST FAILED: {exc}", file=sys.stderr)
        return 1
    print(result.model_dump_json(indent=2) if args.json else format_ingest_summary(result))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reviewharness", description="Multi-agent repository review harness")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="print version").set_defaults(func=_cmd_version)
    sub.add_parser("config", help="show effective configuration (secrets redacted)").set_defaults(func=_cmd_config)
    sub.add_parser("smoke", help="run the built-in schema smoke test").set_defaults(func=_cmd_smoke)

    vf = sub.add_parser("validate-finding", help="validate a finding JSON file against the schema")
    vf.add_argument("path")
    vf.set_defaults(func=_cmd_validate_finding)

    ing = sub.add_parser("ingest", help="snapshot a repository and run deterministic analysis")
    ing.add_argument("source", help="GitHub URL, .zip file or local folder")
    ing.add_argument("--type", choices=sorted(_INGEST_TYPES), default="auto", help="input type (default: auto-detect)")
    ing.add_argument("--no-tools", action="store_true", help="skip static-analysis tools")
    ing.add_argument("--json", action="store_true", help="print the full result as JSON")
    ing.set_defaults(func=_cmd_ingest)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_dotenv(override=False)  # real environment variables win over .env
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env()
    except ValidationError as exc:
        print("Invalid configuration:\n" + _format_errors(exc), file=sys.stderr)
        return 2
    setup_logging(settings.log_level)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
