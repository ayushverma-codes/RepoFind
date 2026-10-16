"""Controlled failure demonstration (test mode only).

Scenario 1: the bug specialist's provider calls fail. Expected: the other specialists' findings
survive, the run is PARTIAL, and the report says it is incomplete and names the failed task.
Scenario 2: the checker is down. Expected: nothing is accepted, every candidate is listed as
unresolved, and the run is PARTIAL (unverified findings never reach the accepted list).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from reviewharness.config import Settings
from reviewharness.evaluation.benchmark import load_benchmarks
from reviewharness.ingest import ingest_repository
from reviewharness.llm.fake import FaultInjectingProvider, RuleBasedProvider
from reviewharness.models import InputType
from reviewharness.orchestration import Orchestrator, RunStore
from reviewharness.reporting import RunReport, build_run_report, write_reports

DEMO_BENCHMARK = "web_service"


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class DemoResult:
    scenario: str
    run_id: str
    run_status: str
    checks: list[Check]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def _run(root: Path, out_dir: Path, settings: Settings, fail: list[str], scenario: str) -> tuple[str, RunReport]:
    s = settings.model_copy(update={"test_mode": True, "data_dir": out_dir / scenario})
    ingest = ingest_repository(str(root), InputType.LOCAL_FOLDER, s, run_tools=False)
    provider = FaultInjectingProvider(RuleBasedProvider(), fail_specialists=fail)
    store = RunStore(s.data_dir / "runs.db")
    try:
        outcome = asyncio.run(Orchestrator(store, s, provider).run(ingest, new_run=True))
        report = build_run_report(store, outcome.run_id)
        write_reports(report, s.data_dir / "reports" / outcome.run_id)
    finally:
        store.close()
    return outcome.run_id, report


def run_failure_demo(benchmarks_dir: Path, out_dir: Path, settings: Settings) -> list[DemoResult]:
    (spec, repo), = load_benchmarks(benchmarks_dir, [DEMO_BENCHMARK])
    results: list[DemoResult] = []

    run_id, rep = _run(repo, out_dir, settings, ["bug"], "worker_failure")
    by_task = {t.specialist: t.status for t in rep.tasks}
    results.append(DemoResult("worker_failure", run_id, rep.run_status, [
        Check("run status is partial", rep.run_status == "partial", rep.run_status),
        Check("failed worker is reported as failed", by_task.get("bug") == "failed", str(by_task)),
        Check("other workers still succeeded", by_task.get("security") == "succeeded" and by_task.get("testing_maintainability") == "succeeded", str(by_task)),
        Check("their findings were kept", len(rep.findings) > 0, f"{len(rep.findings)} accepted finding(s)"),
        Check("report does not claim completeness", rep.complete is False and any("bug" in r for r in rep.incomplete_reasons), "; ".join(rep.incomplete_reasons)),
    ]))

    run_id, rep = _run(repo, out_dir, settings, ["checker"], "checker_outage")
    results.append(DemoResult("checker_outage", run_id, rep.run_status, [
        Check("run status is partial", rep.run_status == "partial", rep.run_status),
        Check("nothing unverified was accepted", len(rep.findings) == 0, f"{len(rep.findings)} accepted"),
        Check("candidates are listed as unresolved", len(rep.unresolved) > 0, f"{len(rep.unresolved)} unresolved"),
        Check("report does not claim completeness", rep.complete is False, "; ".join(rep.incomplete_reasons)),
    ]))
    return results
