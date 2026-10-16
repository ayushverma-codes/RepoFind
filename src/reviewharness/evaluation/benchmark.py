"""Run the baseline (one general reviewer, unchecked) and the multi-agent system on the same
snapshot of each benchmark repository, then score both against the ground truth.

Test mode uses the deterministic rule-based provider: it verifies the pipeline and the scoring,
and says nothing about model quality. Live mode uses the configured Groq model.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from pydantic import ValidationError

from reviewharness.config import Settings
from reviewharness.evaluation.models import (
    AggregateScore,
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkSpec,
    SystemScore,
)
from reviewharness.evaluation.scoring import build_score
from reviewharness.ingest import ingest_repository
from reviewharness.ingest.models import IngestResult
from reviewharness.llm.base import LLMProvider
from reviewharness.models import InputType, RunStatus, SpecialistSource, TaskStatus
from reviewharness.orchestration import Orchestrator, RunStore
from reviewharness.reporting import build_run_report, write_reports
from reviewharness.specialists import review as run_specialist

log = logging.getLogger("reviewharness.evaluation")

DEFAULT_BENCHMARKS_DIR = Path(__file__).resolve().parents[3] / "benchmarks"
DEFAULT_TOLERANCE = 1

LIMITATIONS = [
    "Three tiny benchmark repositories (see expected.json files); results do not show general production readiness.",
    "Ground truth was written by the project authors. A real problem that is not listed counts as a false positive, so precision is a lower bound.",
    "A finding matches an issue by file, category group and line range (+/- the stated tolerance); wrong explanations at the right line still match.",
    "The baseline is one general reviewer over the same files with no checker. It is limited to 10 findings per call, like each specialist.",
    "Test mode uses a rule-based provider and estimates tokens as characters/4: it measures the harness, not model quality or cost.",
    "Latency is wall-clock time on this machine and varies between runs; scores in test mode are deterministic.",
]


class EvaluationError(Exception):
    """The benchmark definitions are missing or invalid."""


def load_benchmarks(root: Path, only: Optional[Sequence[str]] = None) -> list[tuple[BenchmarkSpec, Path]]:
    root = Path(root)
    if not root.is_dir():
        raise EvaluationError(f"benchmarks directory not found: {root}")
    found: list[tuple[BenchmarkSpec, Path]] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        expected, repo = d / "expected.json", d / "repo"
        if not expected.is_file() or not repo.is_dir():
            continue
        try:
            spec = BenchmarkSpec.model_validate_json(expected.read_text(encoding="utf-8"))
        except (OSError, ValidationError):
            raise EvaluationError(f"invalid benchmark definition: {expected}") from None
        if spec.name != d.name:
            raise EvaluationError(f"{expected}: name '{spec.name}' does not match directory '{d.name}'")
        for issue in spec.expected:
            if not (repo / issue.file).is_file():
                raise EvaluationError(f"{expected}: {issue.id} refers to missing file {issue.file}")
        found.append((spec, repo))
    if only:
        names = {s.name for s, _ in found}
        unknown = sorted(set(only) - names)
        if unknown:
            raise EvaluationError(f"unknown benchmark(s): {', '.join(unknown)}; available: {', '.join(sorted(names))}")
        found = [(s, r) for s, r in found if s.name in set(only)]
    if not found:
        raise EvaluationError(f"no benchmarks found under {root}")
    return found


def _baseline(spec: BenchmarkSpec, ingest: IngestResult, provider: LLMProvider, settings: Settings, tolerance: int) -> SystemScore:
    started = time.perf_counter()
    result = run_specialist(
        SpecialistSource.BASELINE, Path(ingest.snapshot.tree_path), ingest.inventory, ingest.context, provider, settings
    )
    return build_score(
        "baseline",
        ok=result.status is TaskStatus.SUCCEEDED,
        status=result.status.value,
        findings=result.findings,
        expected=spec.expected,
        tolerance=tolerance,
        latency_s=time.perf_counter() - started,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        error=result.error,
    )


def _multi_agent(
    spec: BenchmarkSpec, ingest: IngestResult, provider: LLMProvider, settings: Settings, tolerance: int
) -> SystemScore:
    store = RunStore(settings.data_dir / "runs.db")
    try:
        started = time.perf_counter()
        outcome = asyncio.run(Orchestrator(store, settings, provider).run(ingest, new_run=True))
        latency = time.perf_counter() - started
        report = build_run_report(store, outcome.run_id)
        write_reports(report, settings.data_dir / "reports" / outcome.run_id)
    finally:
        store.close()
    return build_score(
        "multi_agent",
        ok=outcome.status not in (RunStatus.FAILED, RunStatus.CANCELLED),
        status=outcome.status.value,
        findings=report.findings,
        expected=spec.expected,
        tolerance=tolerance,
        latency_s=latency,
        prompt_tokens=report.prompt_tokens,
        completion_tokens=report.completion_tokens,
        unresolved=len(report.unresolved),
        complete=report.complete,
    )


def _aggregate(system: str, scores: list[tuple[SystemScore, int]]) -> AggregateScore:
    ok = [s for s, _ in scores if s.ok]
    reported = sum(s.reported for s in ok)
    tp = sum(s.true_positives for s in ok)
    expected_total = sum(n for _, n in scores)
    matched = sum(s.matched_expected for s in ok)
    return AggregateScore(
        system=system,
        benchmarks_ok=len(ok),
        benchmarks_total=len(scores),
        incomplete_runs=sum(1 for s in ok if s.complete is False),
        reported=reported,
        true_positives=tp,
        false_positives=sum(s.false_positives for s in ok),
        duplicates=sum(s.duplicates for s in ok),
        expected_total=expected_total,
        matched_expected=matched,
        precision=round(tp / reported, 3) if reported else None,
        recall=round(matched / expected_total, 3) if expected_total else None,
        total_latency_s=round(sum(s.latency_s for s, _ in scores), 3),
        prompt_tokens=sum(s.prompt_tokens for s, _ in scores),
        completion_tokens=sum(s.completion_tokens for s, _ in scores),
    )


def run_benchmarks(
    root: Path,
    out_dir: Path,
    settings: Settings,
    provider: LLMProvider,
    *,
    only: Optional[Sequence[str]] = None,
    tolerance: int = DEFAULT_TOLERANCE,
) -> BenchmarkReport:
    results: list[BenchmarkResult] = []
    for spec, repo in load_benchmarks(root, only):
        work = Path(out_dir) / spec.name
        bench_settings = settings.model_copy(update={"data_dir": work})
        log.info("benchmark %s: ingesting %s", spec.name, repo)
        ingest = ingest_repository(str(repo), InputType.LOCAL_FOLDER, bench_settings, run_tools=False)
        baseline = _baseline(spec, ingest, provider, bench_settings, tolerance)
        multi = _multi_agent(spec, ingest, provider, bench_settings, tolerance)
        results.append(BenchmarkResult(name=spec.name, expected_total=len(spec.expected), baseline=baseline, multi_agent=multi))
    aggregate = [
        _aggregate("baseline", [(r.baseline, r.expected_total) for r in results]),
        _aggregate("multi_agent", [(r.multi_agent, r.expected_total) for r in results]),
    ]
    return BenchmarkReport(
        mode="test" if settings.test_mode else "live",
        provider=provider.name,
        model=provider.model,
        line_tolerance=tolerance,
        results=results,
        aggregate=aggregate,
        limitations=LIMITATIONS,
    )


# ------------------------------------------------------------------ output

def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_benchmark_text(report: BenchmarkReport) -> str:
    lines = [
        f"Benchmark ({report.mode} mode) provider={report.provider} model={report.model} line_tolerance={report.line_tolerance}",
        "",
        f"{'benchmark':<16}{'system':<13}{'status':<11}{'rep':>4}{'TP':>4}{'FP':>4}{'miss':>5}{'prec':>7}{'recall':>8}{'sec':>8}{'tokens':>9}",
    ]
    for r in report.results:
        for s in (r.baseline, r.multi_agent):
            tokens = s.prompt_tokens + s.completion_tokens
            lines.append(
                f"{r.name:<16}{s.system:<13}{s.status:<11}{s.reported:>4}{s.true_positives:>4}{s.false_positives:>4}"
                f"{len(s.missed) if s.ok else '-':>5}{_fmt(s.precision):>7}{_fmt(s.recall):>8}{s.latency_s:>8.2f}{tokens:>9}"
            )
            if s.error:
                lines.append(f"    error: {s.error}")
            lines += [f"    false positive: {fp}" for fp in s.false_positive_findings]
            lines += [f"    missed: {', '.join(s.missed)}"] if s.ok and s.missed else []
            if s.system == "multi_agent" and (s.complete is False or s.unresolved):
                lines.append(f"    NOT COMPLETE: complete={s.complete} unresolved={s.unresolved}")
    lines += ["", "Aggregate (micro-average; recall counts failed benchmarks as missed):"]
    for a in report.aggregate:
        lines.append(
            f"  {a.system:<12} ok={a.benchmarks_ok}/{a.benchmarks_total} reported={a.reported} TP={a.true_positives} "
            f"FP={a.false_positives} precision={_fmt(a.precision)} recall={_fmt(a.recall)} "
            f"latency={a.total_latency_s:.2f}s tokens={a.prompt_tokens + a.completion_tokens}"
            + (f" incomplete_runs={a.incomplete_runs}" if a.incomplete_runs else "")
        )
    lines += ["", "Limitations:"] + [f"  - {t}" for t in report.limitations]
    return "\n".join(lines)


def render_benchmark_markdown(report: BenchmarkReport) -> str:
    out = [
        "# Benchmark results",
        "",
        f"Mode: **{report.mode}** | provider: `{report.provider}` | model: `{report.model}` | line tolerance: {report.line_tolerance}",
        "",
        "| Benchmark | System | Status | Reported | TP | FP | Missed | Precision | Recall | Latency (s) | Tokens |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in report.results:
        for s in (r.baseline, r.multi_agent):
            out.append(
                f"| {r.name} | {s.system} | {s.status} | {s.reported} | {s.true_positives} | {s.false_positives} | "
                f"{len(s.missed) if s.ok else '-'} | {_fmt(s.precision)} | {_fmt(s.recall)} | {s.latency_s:.2f} | "
                f"{s.prompt_tokens + s.completion_tokens} |"
            )
    out += ["", "## Aggregate", "", "| System | OK | Reported | TP | FP | Precision | Recall | Latency (s) | Tokens |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a in report.aggregate:
        out.append(
            f"| {a.system} | {a.benchmarks_ok}/{a.benchmarks_total} | {a.reported} | {a.true_positives} | {a.false_positives} | "
            f"{_fmt(a.precision)} | {_fmt(a.recall)} | {a.total_latency_s:.2f} | {a.prompt_tokens + a.completion_tokens} |"
        )
    detail = [
        f"- **{r.name} / {s.system}**: " + "; ".join(
            ([f"false positives: {', '.join(s.false_positive_findings)}"] if s.false_positive_findings else [])
            + ([f"missed: {', '.join(s.missed)}"] if s.ok and s.missed else [])
            + ([f"error: {s.error}"] if s.error else [])
        )
        for r in report.results for s in (r.baseline, r.multi_agent)
        if s.false_positive_findings or (s.ok and s.missed) or s.error
    ]
    if detail:
        out += ["", "## False positives, misses and errors", ""] + detail
    out += ["", "## Limitations", ""] + [f"- {t}" for t in report.limitations]
    return "\n".join(out) + "\n"


def write_benchmark_report(report: BenchmarkReport, out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    js, md = out_dir / "benchmark_results.json", out_dir / "benchmark_results.md"
    for path, text in ((js, json.dumps(report.model_dump(mode="json"), indent=2) + "\n"), (md, render_benchmark_markdown(report))):
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    return js, md
