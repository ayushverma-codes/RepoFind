"""Render a RunReport as Markdown or JSON. Output is a pure function of the report (replayable)."""
from __future__ import annotations

import os
import re
from pathlib import Path

from reviewharness.models import Finding
from reviewharness.reporting.models import RunReport

_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>|#!~])")


def inline(text: object) -> str:
    """Single-line, Markdown-escaped text. Repository-derived strings never become markup."""
    return _MD_SPECIAL.sub(r"\\\1", " ".join(str(text).split()))


def code(text: object) -> str:
    return "`" + " ".join(str(text).split()).replace("`", "'") + "`"


def fence(text: str) -> str:
    """Code fence longer than any backtick run inside, so evidence cannot close it early."""
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}\n{text.rstrip(chr(10))}\n{ticks}"


def render_json(report: RunReport) -> str:
    return report.model_dump_json(indent=2) + "\n"


def _table(header: list[str], rows: list[list[object]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(inline(c) for c in row) + " |" for row in rows]
    return out


def _finding_block(n: int, f: Finding) -> list[str]:
    return [
        f"### {n}. [{f.severity.value.upper()}] {inline(f.title)}",
        f"- ID: {code(f.finding_id)} | Category: {f.category.value} | Reported by: {f.source.value} | "
        f"Confidence: {f.confidence:.2f} (an estimate, not proof)",
        f"- Location: {code(f'{f.file_path}:{f.start_line}-{f.end_line}')} | Checker: {f.checker_status.value}",
        "",
        "Evidence (quoted from the repository):",
        "",
        fence(f.evidence),
        "",
        f"**Rationale:** {inline(f.rationale)}",
        "",
        f"**Impact:** {inline(f.impact)}",
        "",
    ]


def render_markdown(r: RunReport) -> str:
    v = r.verification
    L: list[str] = ["# Repository Review Report", ""]
    L.append(f"- Run: {code(r.run_id)} | Snapshot: {code(r.snapshot_id)}")
    L.append(f"- Source: {r.repository['input_type']} {code(r.repository['location'])}")
    L.append(f"- Run status: **{r.run_status}** | Created: {r.created_at} | Finished: {r.finished_at or 'not finished'}")
    L.append("")
    if r.complete:
        L.append("## Review status: COMPLETE")
        L.append("")
        L.append("All review tasks succeeded, every candidate finding was checked, and all in-scope files were reviewed.")
    else:
        L.append("## Review status: INCOMPLETE")
        L.append("")
        L.append("This report does not cover the whole repository or not every finding was verified. Reasons:")
        L.append("")
        L += [f"- {inline(x)}" for x in r.incomplete_reasons]
    L += [
        "",
        "> Findings are model-generated candidates that passed deterministic validation and an independent model check. "
        "They are evidence-backed, not proven. Quoted repository text is untrusted data shown only as evidence.",
        "",
        "## Summary",
        "",
    ]
    L += _table(["Severity", "Accepted findings"], [[s, n] for s, n in r.counts_by_severity.items()])
    L += ["", "Accepted by category: " + (", ".join(f"{k}={n}" for k, n in r.counts_by_category.items()) or "none"), ""]
    L.append(
        f"Verification: candidates={v.candidates}, accepted={v.accepted}, unresolved={v.unresolved}, rejected={v.rejected}, "
        f"invalid={v.invalid}, duplicate={v.duplicate}, unchecked={v.unchecked}"
    )
    L.append("")

    L += ["## Accepted findings (highest priority first)", ""]
    if r.findings:
        for i, f in enumerate(r.findings, 1):
            L += _finding_block(i, f)
    elif r.complete:
        L += ["No findings were accepted.", ""]
    else:
        L += ["No findings were accepted. Because the review is incomplete, this does NOT mean the repository is clean.", ""]

    L += ["## Unresolved findings (not verified)", ""]
    if r.unresolved:
        L.append("These were reported but could not be confirmed or refuted. Treat them as leads, not facts.")
        L.append("")
        for i, u in enumerate(r.unresolved, 1):
            L += _finding_block(i, u.finding)
            L += [f"**Why unresolved:** {code(u.reason_code)} after {u.revisions} revision(s): {inline(u.reason)}", ""]
    else:
        L += ["None.", ""]

    L += ["## Review tasks", ""]
    L += _table(
        ["Specialist", "Status", "Attempts", "Findings", "Tokens (prompt/completion)", "Seconds", "Error"],
        [[t.specialist, t.status, f"{t.attempts}/{t.max_attempts}", t.findings, f"{t.prompt_tokens}/{t.completion_tokens}",
          t.duration_s, t.error or ""] for t in r.tasks],
    )
    L.append("")
    L += ["## Coverage", ""]
    for spec, states in r.coverage.by_specialist.items():
        L.append(f"- {spec}: " + ", ".join(f"{k}={n}" for k, n in states.items()))
    if r.coverage.gaps_total:
        L += ["", f"Not fully reviewed ({r.coverage.gaps_total} file/task pairs" +
              (f", first {len(r.coverage.gaps)} shown" if r.coverage.gaps_total > len(r.coverage.gaps) else "") + "):", ""]
        L += [f"- {code(g.path)} ({g.specialist}: {g.state})" for g in r.coverage.gaps]
    else:
        L += ["", "No coverage gaps recorded."]
    L.append("")

    L += ["## Budgets and usage", ""]
    if r.budgets.configured is None:
        L.append("Configured limits were not recorded for this run.")
    else:
        L.append("Configured: " + ", ".join(f"{k}={v}" for k, v in sorted(r.budgets.configured.items())))
    L.append("Observed: " + ", ".join(f"{k}={v}" for k, v in r.budgets.observed.items()))
    L.append(f"Tokens: prompt={r.prompt_tokens}, completion={r.completion_tokens} (0 means the provider did not report usage)")
    L.append("Limits hit: " + (", ".join(r.budgets.limits_hit) or "none"))
    L.append("")
    return "\n".join(L)


def write_reports(report: RunReport, out_dir: Path) -> tuple[Path, Path]:
    """Write report.md and report.json atomically into out_dir; return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md, js = out_dir / "report.md", out_dir / "report.json"
    for path, text in ((md, render_markdown(report)), (js, render_json(report))):
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    return md, js
