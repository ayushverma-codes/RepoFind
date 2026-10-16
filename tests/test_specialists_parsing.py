import pytest

from reviewharness.models import CheckerStatus, SpecialistSource
from reviewharness.specialists.models import Excerpt
from reviewharness.specialists.parsing import (
    MAX_FINDINGS,
    ResponseFormatError,
    extract_findings_list,
    validate_findings,
)

SRC = [
    "import subprocess",
    "",
    "def run(cmd, items=[]):",
    "    subprocess.call(cmd, shell=True)",
    "    return eval(cmd)",
    'PASSWORD = "hunter2"',
]
EX = Excerpt(path="app.py", total_lines=6, shown_lines=6, text="\n".join(SRC))
SEC = SpecialistSource.SECURITY


def item(**over):
    base = dict(
        category="security", title="shell=True", severity="high", file_path="app.py",
        start_line=4, end_line=4, evidence="subprocess.call(cmd, shell=True)",
        rationale="r", impact="i", confidence=0.7,
    )
    base.update(over)
    return base


def run(items, specialist=SEC):
    return validate_findings(items, specialist, [EX])


# ---- JSON extraction
def test_extract_plain_fenced_and_wrapped():
    assert extract_findings_list('{"findings": []}') == []
    assert extract_findings_list('```json\n{"findings": [{"a": 1}]}\n```') == [{"a": 1}]
    assert extract_findings_list('Sure! Here you go: {"findings": []} Hope that helps.') == []


@pytest.mark.parametrize("text", ["not json", "[]", '{"other": 1}', '{"findings": "x"}', ""])
def test_extract_rejects_bad_replies(text):
    with pytest.raises(ResponseFormatError):
        extract_findings_list(text)


# ---- validation
def test_valid_finding_is_filled_in_by_us_not_the_model():
    findings, rejected = run([item()])
    assert rejected == [] and len(findings) == 1
    f = findings[0]
    assert f.source is SEC and f.checker_status is CheckerStatus.PENDING and f.finding_id.startswith("F-")


def test_model_cannot_set_id_source_or_status():
    findings, _ = run([item(finding_id="X", source="bug", checker_status="accepted")])
    f = findings[0]
    assert f.finding_id != "X" and f.source is SEC and f.checker_status is CheckerStatus.PENDING


def test_numbered_evidence_and_small_line_offset_are_accepted():
    findings, rejected = run([item(evidence="    4| subprocess.call(cmd, shell=True)", start_line=6, end_line=6)])
    assert len(findings) == 1 and rejected == []


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"evidence": "os.system(cmd)"}, "evidence not found"),
        ({"start_line": 1, "end_line": 1}, "evidence not found"),  # real line is 4: too far away
        ({"file_path": "other.py"}, "not part of the reviewed"),
        ({"start_line": 4, "end_line": 40}, "outside the reviewed excerpt"),
        ({"file_path": "../x.py"}, "schema: file_path"),
        ({"severity": "catastrophic"}, "schema: severity"),
        ({"confidence": 1.5}, "schema: confidence"),
        ({"evidence": "   "}, "schema: evidence"),
    ],
)
def test_bad_findings_are_rejected_with_reason(overrides, reason):
    findings, rejected = run([item(**overrides)])
    assert findings == []
    assert len(rejected) == 1 and reason in rejected[0].reason


def test_category_outside_role_is_rejected():
    findings, rejected = run([item(category="security")], specialist=SpecialistSource.BUG)
    assert findings == [] and "outside this specialist's role" in rejected[0].reason


def test_baseline_accepts_any_category():
    findings, _ = run([item(category="security")], specialist=SpecialistSource.BASELINE)
    assert len(findings) == 1


def test_non_object_item_rejected():
    findings, rejected = run(["just text"])
    assert findings == [] and rejected[0].reason == "finding is not a JSON object"


def test_one_bad_finding_does_not_discard_the_good_one():
    findings, rejected = run([item(evidence="nope"), item(title="second", start_line=5, end_line=5, evidence="return eval(cmd)")])
    assert [f.title for f in findings] == ["second"] and len(rejected) == 1


def test_duplicate_in_reply_is_dropped():
    findings, rejected = run([item(), item()])
    assert len(findings) == 1 and "duplicate" in rejected[0].reason


def test_secret_values_are_masked_after_grounding():
    findings, _ = run([item(title="Hardcoded credential", start_line=6, end_line=6, evidence='PASSWORD = "hunter2"')])
    assert len(findings) == 1
    assert "hunter2" not in findings[0].evidence and "[REDACTED]" in findings[0].evidence


def test_findings_per_reply_are_capped():
    items = [item(title=f"t{i}") for i in range(MAX_FINDINGS + 2)]
    findings, rejected = run(items)
    assert len(findings) == MAX_FINDINGS
    assert any(r.title == "(excess findings)" for r in rejected)
