"""Benchmark evaluation: ground truth, scoring, baseline-vs-multi-agent comparison, failure demo."""
from reviewharness.evaluation.benchmark import (
    DEFAULT_BENCHMARKS_DIR,
    EvaluationError,
    load_benchmarks,
    run_benchmarks,
    write_benchmark_report,
)
from reviewharness.evaluation.failure_demo import run_failure_demo
from reviewharness.evaluation.models import BenchmarkReport, BenchmarkSpec, ExpectedIssue
from reviewharness.evaluation.scoring import match_findings

__all__ = [
    "BenchmarkReport",
    "BenchmarkSpec",
    "DEFAULT_BENCHMARKS_DIR",
    "EvaluationError",
    "ExpectedIssue",
    "load_benchmarks",
    "match_findings",
    "run_benchmarks",
    "run_failure_demo",
    "write_benchmark_report",
]
