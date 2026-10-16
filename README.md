# reviewharness

Deterministic-first, multi-agent repository review harness. Input: a GitHub URL, a ZIP file or a local
folder. Output: a prioritized Markdown and JSON report of bugs, security issues and testing or
maintainability problems. Every accepted finding has source evidence, a location, severity, confidence,
rationale and an independent checker decision.

Pipeline: safe snapshot and inventory, context builder, deterministic planner, three parallel specialists
(LangGraph + asyncio), deterministic validation and de-duplication, independent checker with a bounded
revision loop, deterministic prioritization, reports. State and an append-only event log live in SQLite.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,analysis,agents]"
cp .env.example .env                                   # then set GROQ_API_KEY for live runs
pytest
```

Python 3.10+. `analysis` (ruff, bandit) is optional: missing tools are recorded, not fatal.

## Configuration

Environment variables (or `.env`; real environment variables win). `reviewharness config` prints the
effective values with the key redacted. See `.env.example` for all of them.

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | none | needed for live runs only |
| `REVIEW_LLM_MODEL` | `openai/gpt-oss-20b` | Groq model |
| `REVIEW_TEST_MODE` | `0` | `1`: deterministic rule-based provider, no key, no network |
| `REVIEW_MAX_CONCURRENCY` / `_RETRIES` / `_REVISIONS` | 3 / 2 / 2 | bounds |
| `REVIEW_MAX_TOKENS` | 500000 | model-usage budget per run |
| `REVIEW_TASK_TIMEOUT_S` / `REVIEW_RUN_TIMEOUT_S` | 300 / 900 | time budgets |
| `REVIEW_DATA_DIR` | `.reviewharness` | snapshots, `runs.db`, reports |

## Usage

```bash
# Any of the three input types (auto-detected; --type github|zip|folder to force)
reviewharness run ./path/to/folder
reviewharness run ./repo.zip
reviewharness run https://github.com/owner/repo

reviewharness run ./path --test-mode        # no API key; deterministic provider
reviewharness inspect                        # list runs
reviewharness inspect <run_id> --events      # state, coverage, event history
reviewharness report <run_id> --out ./out    # rebuild report.md/report.json from stored state (no model calls)
```

`run` writes `report.md` and `report.json` to `<data dir>/reports/<run_id>/`. Exit code 0 means the
run was COMPLETED; 1 means PARTIAL, FAILED or refused. An incomplete review is always labelled
INCOMPLETE in the report, with the reasons. "No findings" in an incomplete report does not mean clean.

An identical request (same snapshot, plan, provider and model) reuses the stored run; use `--new-run` to force another.

## Benchmarks and failure demonstration

```bash
reviewharness benchmark                 # test mode: deterministic, no key
reviewharness benchmark --live          # configured Groq model (needs GROQ_API_KEY)
reviewharness benchmark --only web_service
reviewharness failure-demo              # worker failure + checker outage; exit 0 only if behaviour is correct
```

`benchmarks/<name>/repo` is a small safe repository and `benchmarks/<name>/expected.json` its ground
truth (14 planted issues in three repositories, plus documented false-positive traps and one blind spot).
The benchmark runs the **baseline** (one general reviewer, no checker) and the **multi-agent** system
(three specialists plus checker) on the same snapshot, matches findings to ground truth by file, category
group and line (+/- 1), and reports precision, recall, false positives, misses, latency and tokens to
`<data dir>/benchmarks/benchmark_results.{json,md}`.

### Reading the results honestly

* Test-mode scores are deterministic and show that the harness and the checker work as designed. The
  rule-based provider is a test double: **they say nothing about LLM quality**. Use `--live` for that, and
  expect run-to-run variation.
* Three tiny repositories are not evidence of general production readiness.
* Ground truth is authored by the project. A real issue that is not listed counts as a false positive,
  so precision is a lower bound.
* Token counts in test mode are estimates (characters / 4). Cost is not computed.

## Limitations

* A task that times out is recorded as TIMED_OUT, but Python cannot kill its provider call, which runs on until the provider's own timeout.
* GitHub clones are bounded by time, not bytes; the size limit applies when the clone is copied into the snapshot.
* Repository code is never executed. Only the approved static-analysis tools run, with a timeout and a scrubbed environment.
* Checker confidence is a model opinion, not proof; findings that cannot be settled within the revision budget are listed as unresolved.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `CONFIG ERROR: GROQ_API_KEY is not set` | set it in `.env`, or use `--test-mode` |
| `the "langgraph" package is not installed` | `pip install -e ".[agents]"` |
| `RUN REFUSED: ... still 'running'` | a previous identical run crashed; use `--new-run` |
| `git is not installed` / clone fails | use a ZIP or local folder instead |
| run ends PARTIAL | `reviewharness inspect <run_id>`: failed/timed-out tasks, unchecked findings and coverage gaps are listed |
| rate-limit errors in live mode | lower `REVIEW_MAX_CONCURRENCY`; transient errors are retried up to `REVIEW_MAX_RETRIES` |
