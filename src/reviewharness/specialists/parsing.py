"""Turn raw model text into validated findings. Nothing here trusts the model.

Pipeline: extract JSON -> per-finding pydantic validation -> role check ->
grounding against the excerpt the model was shown -> de-duplication -> secret masking.
One bad finding is rejected individually (with a reason); it never discards the rest.
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from reviewharness.models import Category, Finding, SpecialistSource
from reviewharness.specialists.models import Excerpt, RejectedFinding

MAX_FINDINGS = 10
MAX_SPAN_LINES = 80
LINE_SLACK = 2  # tolerance for models that are off by a line or two

FINDING_FIELDS = (
    "category", "title", "severity", "file_path", "start_line", "end_line",
    "evidence", "rationale", "impact", "confidence",
)

ALLOWED_CATEGORIES = {
    SpecialistSource.BUG: {Category.BUG},
    SpecialistSource.SECURITY: {Category.SECURITY},
    SpecialistSource.TESTING_MAINTAINABILITY: {Category.TESTING, Category.MAINTAINABILITY},
    SpecialistSource.BASELINE: set(Category),
}

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_NUMBER_PREFIX = re.compile(r"^\s*\d+\s*\|\s?")
_SECRET_LITERAL = re.compile(
    r"""(?i)((?:pass(?:word|wd)?|pwd|secret|token|api[_-]?key|private[_-]?key)\w*\s*[:=]\s*)(["'])[^"']+\2"""
)


class ResponseFormatError(ValueError):
    """The reply is not the JSON object we asked for."""


def extract_json_object(text: str) -> dict:
    """Parse the JSON object in a model reply (plain, fenced, or surrounded by prose)."""
    candidate = text.strip()
    fenced = _FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    data = None
    try:
        data = json.loads(candidate)
    except ValueError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(candidate[start : end + 1])
            except ValueError:
                data = None
    if data is None:
        raise ResponseFormatError("the reply was not valid JSON")
    if not isinstance(data, dict):
        raise ResponseFormatError("the reply must be a JSON object")
    return data


def extract_findings_list(text: str) -> list[Any]:
    """Return the raw `findings` list from a model reply, or raise ResponseFormatError."""
    data = extract_json_object(text)
    if not isinstance(data.get("findings"), list):
        raise ResponseFormatError('the JSON must be an object with a "findings" array')
    return data["findings"]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_numbering(evidence: str) -> str:
    """Models sometimes copy the '<n>| ' prefix; drop it only if every line has it."""
    lines = [ln for ln in evidence.splitlines() if ln.strip()]
    if lines and all(_NUMBER_PREFIX.match(ln) for ln in lines):
        return "\n".join(_NUMBER_PREFIX.sub("", ln) for ln in lines)
    return evidence


def mask_secrets(text: str) -> str:
    """Hide quoted values assigned to secret-looking names so reports never repeat them."""
    return _SECRET_LITERAL.sub(r"\1\2[REDACTED]\2", text)


def check_grounding(finding: Finding, excerpts: dict[str, Excerpt]) -> str | None:
    """Return a rejection reason, or None if the finding is tied to code the model saw."""
    ex = excerpts.get(finding.file_path)
    if ex is None:
        return "file was not part of the reviewed excerpts"
    if finding.end_line > ex.shown_lines:
        return "line range lies outside the reviewed excerpt"
    if finding.end_line - finding.start_line + 1 > MAX_SPAN_LINES:
        return f"line range spans more than {MAX_SPAN_LINES} lines"
    lines = ex.text.split("\n")
    window = lines[max(0, finding.start_line - 1 - LINE_SLACK) : finding.end_line + LINE_SLACK]
    needle = _normalize(_strip_numbering(finding.evidence))
    if not needle:
        return "evidence is empty"
    if needle not in _normalize(" ".join(window)):
        return "evidence not found in the cited lines"
    return None


normalize_text = _normalize
strip_numbering = _strip_numbering


def _first_error(exc: ValidationError) -> str:
    err = exc.errors()[0]  # location + message only; never echo values
    loc = ".".join(str(p) for p in err["loc"]) or "<root>"
    return f"schema: {loc}: {err['msg']}"


def validate_findings(
    items: list[Any], specialist: SpecialistSource, excerpts: list[Excerpt]
) -> tuple[list[Finding], list[RejectedFinding]]:
    by_path = {e.path: e for e in excerpts}
    findings: list[Finding] = []
    rejected: list[RejectedFinding] = []
    seen: set[str] = set()

    if len(items) > MAX_FINDINGS:
        rejected.append(
            RejectedFinding(title="(excess findings)", reason=f"reply had {len(items)} findings; only the first {MAX_FINDINGS} were considered")
        )

    for item in items[:MAX_FINDINGS]:
        if not isinstance(item, dict):
            rejected.append(RejectedFinding(title="(not an object)", reason="finding is not a JSON object"))
            continue
        title = str(item.get("title", ""))[:80] or "(untitled)"
        data = {k: item[k] for k in FINDING_FIELDS if k in item}  # model may not set id/source/status
        try:
            finding = Finding(**data, source=specialist)
        except ValidationError as exc:
            rejected.append(RejectedFinding(title=title, reason=_first_error(exc)))
            continue
        if finding.category not in ALLOWED_CATEGORIES[specialist]:
            rejected.append(RejectedFinding(title=title, reason=f"category '{finding.category.value}' is outside this specialist's role"))
            continue
        reason = check_grounding(finding, by_path)
        if reason:
            rejected.append(RejectedFinding(title=title, reason=reason))
            continue
        if finding.finding_id in seen:
            rejected.append(RejectedFinding(title=title, reason="duplicate of an earlier finding in the same reply"))
            continue
        seen.add(finding.finding_id)
        findings.append(
            finding.model_copy(
                update={
                    "evidence": mask_secrets(finding.evidence),
                    "rationale": mask_secrets(finding.rationale),
                    "impact": mask_secrets(finding.impact),
                }
            )
        )
    return findings, rejected
