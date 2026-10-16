"""Plain-text views over a recorded run. Reads the store only; never executes anything."""
from __future__ import annotations

from collections import Counter

from reviewharness.models import TaskStatus
from reviewharness.orchestration.models import RunEvent
from reviewharness.orchestration.store import RunStore

_ENDED = {"task_succeeded", "task_failed", "task_timed_out", "task_cancelled"}


def peak_concurrency(events: list[RunEvent]) -> int:
    """Most tasks in flight at once, computed from the recorded start/end events."""
    live = peak = 0
    for e in events:
        if e.type == "task_started":
            live += 1
            peak = max(peak, live)
        elif e.type in _ENDED:
            live = max(0, live - 1)
    return peak


def format_run_list(store: RunStore) -> str:
    runs = store.list_runs()
    if not runs:
        return "no runs recorded"
    return "\n".join(f"{r['run_id']}  {r['status']:<9} snapshot={r['snapshot_id']}  {r['created_at']}" for r in runs)


def format_run_report(store: RunStore, run_id: str, *, reused: bool = False, events: bool = False) -> str:
    run = store.get_run(run_id)
    meta = store.run_meta(run_id) or {}
    evs = store.events(run_id)
    results = store.task_results(run_id)
    head = f"Run {run_id}: {run.status.value.upper()}"
    if reused:
        head += " (identical run already existed; nothing was re-executed)"
    lines = [head, f"  snapshot: {meta.get('snapshot_id')}  source: {run.repository.location}"]
    lines.append("Tasks (Phase 4: findings below are unchecked candidates):")
    for t in run.tasks:
        r = results.get(t.task_id)
        extra = ""
        if r is not None:
            extra = f" findings={len(r.findings)} tokens={r.prompt_tokens}/{r.completion_tokens} {r.duration_s}s"
        lines.append(f"  {t.specialist.value:<24} {t.status.value:<9} attempts={t.attempts}/{t.max_attempts}{extra}")
        if t.error:
            lines.append(f"    reason: {t.error}")
    cov = Counter(row.state for row in store.coverage(run_id))
    lines.append("Coverage (file/task pairs): " + (", ".join(f"{k}={v}" for k, v in sorted(cov.items())) or "none"))
    cand = store.candidate_findings(run_id)
    lines.append(f"Candidate findings (deduplicated by id): {len(cand)}")
    not_ok = [t for t in run.tasks if t.status is not TaskStatus.SUCCEEDED]
    if not_ok:
        lines.append(f"INCOMPLETE: {len(not_ok)} task(s) did not succeed; their files are NOT covered.")
    lines.append(f"Events: {len(evs)}  peak concurrency: {peak_concurrency(evs)}")
    if events:
        for e in evs:
            who = e.task_id.split(":", 1)[-1] if e.task_id else "-"
            lines.append(f"  #{e.seq:<3} {e.ts.strftime('%H:%M:%S.%f')[:-3]} {e.type:<20} {who:<24} {e.payload}")
    return "\n".join(lines)
