"""Phase 2 entry point: source -> snapshot -> inventory -> context -> tool results."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from reviewharness.config import Settings
from reviewharness.ingest.acquire import Runner
from reviewharness.ingest.context import build_context
from reviewharness.ingest.errors import IngestError
from reviewharness.ingest.inventory import build_inventory
from reviewharness.ingest.models import IngestResult
from reviewharness.ingest.snapshot import create_snapshot, snapshot_dir, verify_snapshot
from reviewharness.ingest.tools import run_analysis_tools
from reviewharness.models import InputType, RepositoryInput


def detect_input_type(source: str) -> InputType:
    s = source.strip()
    if s.lower().startswith("https://github.com/"):
        return InputType.GITHUB_URL
    path = Path(s).expanduser()
    if path.is_dir():
        return InputType.LOCAL_FOLDER
    if path.is_file() and s.lower().endswith(".zip"):
        return InputType.ZIP_FILE
    raise IngestError("cannot tell what the source is: expected a GitHub URL, a .zip file or a folder")


def write_json_atomic(path: Path, model: BaseModel | list[BaseModel]) -> None:
    if isinstance(model, list):
        text = "[" + ",".join(m.model_dump_json() for m in model) + "]"
    else:
        text = model.model_dump_json(indent=2)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def ingest_repository(
    source: str,
    input_type: Optional[InputType],
    settings: Settings,
    *,
    run_tools: bool = True,
    runner: Runner = subprocess.run,
) -> IngestResult:
    repo = RepositoryInput(input_type=input_type or detect_input_type(source), location=source)
    snapshot = create_snapshot(repo, settings, runner=runner)
    root = Path(snapshot.tree_path)

    inventory = build_inventory(root, settings.max_file_kb * 1024)
    context = build_context(root, inventory)
    tool_results = run_analysis_tools(root, inventory, settings, runner=runner) if run_tools else []

    # Invariant: nothing we ran may have modified the immutable snapshot.
    problems = verify_snapshot(snapshot)
    if problems:
        raise IngestError(f"snapshot {snapshot.snapshot_id} changed during analysis: {problems[0]}")

    out = snapshot_dir(snapshot)
    write_json_atomic(out / "inventory.json", inventory)
    write_json_atomic(out / "context.json", context)
    if run_tools:
        write_json_atomic(out / "tools.json", tool_results)
    return IngestResult(snapshot=snapshot, inventory=inventory, context=context, tool_results=tool_results)
