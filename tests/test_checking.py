import asyncio
import json
import re
import threading
import time

import pytest
from pydantic import ValidationError

from reviewharness.checking import FindingOutcome, OutcomeKind, adjudicate, assemble_report, check_run, validate_candidates
from reviewharness.checking import pipeline as check_pipeline
from reviewharness.checking.prompts import checker_user
from reviewharness.checking.source import read_lines
from reviewharness.ingest import ingest_repository
from reviewharness.llm import LLMPermanentError, LLMResult, LLMTransientError
from reviewharness.llm.fake import FaultInjectingProvider, RuleBasedProvider
from reviewharness.models import CheckerStatus, CheckerVerdict, Finding, Report, RunStatus, SpecialistSource, TaskStatus
from reviewharness.orchestration import Orchestrator, RunStore
from reviewharness.specialists.models import Coverage, SpecialistResult
from reviewharness.specialists.parsing import mask_secrets

APP = """import hashlib
PASSWORD = "hunter2hunter2"
def digest(data):
    return hashlib.md5(data).hexdigest()
# eval(user_input) is dangerous
def f(x=[]):
    return x
def g(y=[]):
    return y
"""


@pytest.fixture
def settings(make_settings):
    return make_settings(test_mode=True, max_revisions=2, max_retries=1)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "app.py").write_text(APP)
    return root


@pytest.fixture
def ingest(repo, settings):
    return ingest_repository(str(repo), None, settings, run_tools=False)


@pytest.fixture
def root(ingest):
    from pathlib import Path
    return Path(ingest.snapshot.tree_path)


@pytest.fixture
def store(tmp_path):
    s = RunStore(tmp_path / "runs.db")
    yield s
    s.close()


def mk(title="Mutable default argument", start=6, end=6, evidence="def f(x=[]):", severity="medium",
       confidence=0.8, category="bug", source="bug", path="app.py") -> Finding:
    return Finding(category=category, title=title, severity=severity, file_path=path, start_line=start, end_line=end,
                   evidence=evidence, rationale="shared state", impact="surprising behaviour", confidence=confidence, source=source)


def revision_item(**over):
    base = {"category": "bug", "title": "Mutable default argument", "severity": "medium", "file_path": "app.py",
            "start_line": 6, "end_line": 6, "evidence": "def f(x=[]):", "rationale": "shared", "impact": "bugs", "confidence": 0.8}
    return {"finding": {**base, **over}}


def acc(sev=None):
    return {"verdict": "accept", "reasoning": "Line 6 has a mutable default.", "counterexample": None, "suggested_severity": sev}


def rej():
    return {"verdict": "reject", "reasoning": "Not real.", "counterexample": "It is a comment.", "suggested_severity": None}


def rev():
    return {"verdict": "request_revision", "reasoning": "Line range is imprecise.", "counterexample": None, "suggested_severity": None}


class Routed:
    """Replies by (role, claim title). The last scripted reply repeats."""

    name, model = "routed", "m"

    def __init__(self, checker=None, reviser=None):
        self.scripts = {"checker": checker or {}, "reviser": reviser or {}}
        self.calls, self.prompts, self._lock = [], [], threading.Lock()

    def count(self, role, title=None):
        return sum(1 for r, t in self.calls if r == role and (title is None or t == title))

    def complete(self, system, user):
        role = "checker" if user.startswith("ROLE: checker") else "reviser"
        title = re.search(r"^- title: (.*)$", user, re.MULTILINE).group(1)
        with self._lock:
            n = self.count(role, title) + 1
            self.calls.append((role, title))
            self.prompts.append((role, system, user))
        script = self.scripts[role][title]
        item = script[min(n - 1, len(script) - 1)]
        if isinstance(item, Exception):
            raise item
        return LLMResult(text=item if isinstance(item, str) else json.dumps(item), prompt_tokens=1, completion_tokens=1)


# ------------------------------------------------------------ deterministic validation

def test_valid_finding_passes_validation(root, ingest):
    res = validate_candidates(root, ingest.inventory, [mk()])
    assert [f.finding_id for f in res.valid] == [mk().finding_id] and not res.invalid and not res.duplicates


@pytest.mark.parametrize("finding, code", [
    (mk(start=90, end=90, evidence="x"), "invalid_location"),
    (mk(start=6, end=7, evidence="def totally_different():"), "evidence_mismatch"),
    (mk(path="nope.py"), "invalid_path"),
])
def test_invalid_findings_are_caught_before_the_checker(root, ingest, finding, code):
    res = validate_candidates(root, ingest.inventory, [finding])
    assert not res.valid and res.invalid[0].code == code


def test_masked_secret_evidence_still_validates(root, ingest):
    evidence = mask_secrets('PASSWORD = "hunter2hunter2"')
    assert "hunter2" not in evidence
    f = mk(title="Hardcoded credential", start=2, end=2, evidence=evidence, category="security", source="security")
    assert validate_candidates(root, ingest.inventory, [f]).valid


def test_overlapping_findings_are_deduplicated_keeping_the_most_severe(root, ingest):
    low = mk(title="Same problem, other words", start=6, end=7, evidence="def f(x=[]):", severity="low")
    high = mk(severity="high")
    res = validate_candidates(root, ingest.inventory, [low, high])
    assert [f.severity.value for f in res.valid] == ["high"]
    assert res.duplicates[0].kept_id == high.finding_id and res.duplicates[0].finding.finding_id == low.finding_id


def test_same_lines_different_category_are_not_duplicates(root, ingest):
    a = mk(category="bug")
    b = mk(title="Security view", category="security", source="security")
    assert len(validate_candidates(root, ingest.inventory, [a, b]).valid) == 2


# ------------------------------------------------------------ adjudication

def test_accept_records_decision_and_lowers_never_raises_severity(root, settings):
    p = Routed({"Mutable default argument": [acc("low")]})
    out = adjudicate(mk(severity="medium"), root, p, settings)
    assert out.kind is OutcomeKind.ACCEPTED and out.finding.severity.value == "low"
    assert out.finding.checker_status is CheckerStatus.ACCEPTED and out.decisions[0].verdict is CheckerVerdict.ACCEPT
    p2 = Routed({"Mutable default argument": [acc("critical")]})
    assert adjudicate(mk(severity="medium"), root, p2, settings).finding.severity.value == "medium"


def test_accepted_confidence_is_capped_not_trusted(root, settings):
    out = adjudicate(mk(confidence=1.0), root, Routed({"Mutable default argument": [acc()]}), settings)
    assert out.finding.confidence == 0.95


def test_false_positive_is_rejected_with_counterexample(root, settings):
    out = adjudicate(mk(), root, Routed({"Mutable default argument": [rej()]}), settings)
    assert out.kind is OutcomeKind.REJECTED and out.finding.checker_status is CheckerStatus.REJECTED
    assert "comment" in out.reason and out.decisions[0].counterexample


def test_revision_then_accept_keeps_id_and_records_the_original(root, settings):
    orig = mk(start=6, end=7, evidence="def f(x=[]):\n    return x", severity="high")
    p = Routed({"Mutable default argument": [rev(), acc()]}, {"Mutable default argument": [revision_item(severity="medium")]})
    out = adjudicate(orig, root, p, settings)
    assert out.kind is OutcomeKind.ACCEPTED and out.revisions == 1
    assert out.finding.finding_id == orig.finding_id and out.original.severity.value == "high"
    assert out.finding.start_line == 6 and out.finding.end_line == 6 and out.finding.severity.value == "medium"
    assert out.finding.confidence == 0.72  # 0.8 * 0.9 for one revision
    assert [d.verdict.value for d in out.decisions] == ["request_revision", "accept"]
    assert [d.attempt for d in out.decisions] == [0, 1]


def test_revision_budget_exhausted_leaves_finding_uncertain(root, settings):
    p = Routed({"Mutable default argument": [rev()]}, {"Mutable default argument": [revision_item()]})
    out = adjudicate(mk(confidence=0.9), root, p, settings)
    assert out.kind is OutcomeKind.UNCERTAIN and out.reason_code == "revision_budget_exhausted"
    assert p.count("checker") == 3 and p.count("reviser") == 2  # max_revisions=2, never more
    assert out.finding.checker_status is CheckerStatus.UNCERTAIN and out.finding.confidence == 0.5


def test_zero_revision_budget_means_no_reviser_call(root, make_settings):
    s = make_settings(test_mode=True, max_revisions=0)
    p = Routed({"Mutable default argument": [rev()]})
    out = adjudicate(mk(), root, p, s)
    assert out.kind is OutcomeKind.UNCERTAIN and p.count("reviser") == 0


def test_revision_with_invalid_location_is_not_trusted(root, settings):
    p = Routed({"Mutable default argument": [rev()]}, {"Mutable default argument": [revision_item(start_line=99, end_line=99)]})
    out = adjudicate(mk(), root, p, settings)
    assert out.kind is OutcomeKind.UNCERTAIN and out.reason_code == "revision_invalid"


def test_revision_cannot_move_to_another_file(root, settings):
    p = Routed({"Mutable default argument": [rev(), acc()]},
               {"Mutable default argument": [revision_item(file_path="other.py")]})
    out = adjudicate(mk(), root, p, settings)
    assert out.kind is OutcomeKind.ACCEPTED and out.finding.file_path == "app.py"


def test_withdrawn_finding_is_rejected(root, settings):
    p = Routed({"Mutable default argument": [rev()]}, {"Mutable default argument": [{"finding": None}]})
    out = adjudicate(mk(), root, p, settings)
    assert out.kind is OutcomeKind.REJECTED and out.reason_code == "withdrawn_by_specialist"


def test_malformed_checker_reply_is_retried_then_accepted(root, settings):
    p = Routed({"Mutable default argument": ["not json at all", acc()]})
    out = adjudicate(mk(), root, p, settings)
    assert out.kind is OutcomeKind.ACCEPTED and p.count("checker") == 2


def test_always_malformed_checker_reply_is_uncertain_not_accepted(root, settings):
    out = adjudicate(mk(), root, Routed({"Mutable default argument": ['{"verdict": "maybe"}']}), settings)
    assert out.kind is OutcomeKind.UNCERTAIN and out.reason_code == "checker_error"


def test_transient_errors_are_bounded_and_never_accept(root, settings):
    p = Routed({"Mutable default argument": [LLMTransientError("429")]})
    out = adjudicate(mk(), root, p, settings, sleep=lambda s: None)
    assert out.kind is OutcomeKind.UNCERTAIN and out.reason_code == "checker_error"
    assert p.count("checker") == settings.max_retries + 1


def test_permanent_error_is_not_retried(root, settings):
    p = Routed({"Mutable default argument": [LLMPermanentError("401")]})
    out = adjudicate(mk(), root, p, settings)
    assert out.reason_code == "checker_error" and p.count("checker") == 1


def test_accepted_outcome_requires_a_recorded_accept_decision():
    f = mk().model_copy(update={"checker_status": CheckerStatus.ACCEPTED})
    with pytest.raises(ValidationError):
        FindingOutcome(finding=f, kind=OutcomeKind.ACCEPTED, reason_code="x")


def test_checker_prompt_hides_confidence_and_treats_source_as_data(root, ingest):
    f = mk(confidence=0.77)
    prompt = checker_user(f, read_lines(root, "app.py"))
    assert "0.77" not in prompt and "confidence" not in prompt.lower() and "evidence" not in prompt.lower()
    assert prompt.index("##### END FILE #####") < prompt.index("TASK:")
    assert "    6| def f(x=[]):" in prompt  # real line numbers from the checker's own read


# ------------------------------------------------------------ end to end through the orchestrator

def canned(findings):
    def fn(specialist, root, inventory, context, provider, settings, scope_paths=None, **_):
        mine = findings if specialist is SpecialistSource.BUG else []
        return SpecialistResult(specialist=specialist, status=TaskStatus.SUCCEEDED, provider="t", model="m", findings=list(mine),
                                coverage=Coverage(reviewed=list(scope_paths or [])), attempts=1)
    return fn


def run(store, settings, ingest, provider, findings):
    return asyncio.run(Orchestrator(store, settings, provider, canned(findings)).run(ingest))


def test_gate_valid_false_positive_invalid_duplicate_and_uncertain(store, settings, ingest):
    valid = mk(severity="high", confidence=0.9)
    false_pos = mk(title="Dynamic eval", start=5, end=5, evidence="# eval(user_input) is dangerous", severity="high")
    bad_loc = mk(title="Ghost line", start=90, end=90, evidence="nothing here")
    dup = mk(title="Same bug again", start=6, end=7, evidence="def f(x=[]):", severity="low")
    disputed = mk(title="Weak hash", start=4, end=4, evidence="return hashlib.md5(data).hexdigest()", severity="medium")
    provider = Routed(
        {"Mutable default argument": [acc()], "Dynamic eval": [rej()], "Weak hash": [rev()]},
        {"Weak hash": [revision_item(title="Weak hash", start_line=4, end_line=4, evidence="return hashlib.md5(data).hexdigest()")]},
    )
    out = run(store, settings, ingest, provider, [valid, false_pos, bad_loc, dup, disputed])

    kinds = {o.finding.title: o.kind for o in store.finding_outcomes(out.run_id)}
    assert kinds == {
        "Mutable default argument": OutcomeKind.ACCEPTED, "Dynamic eval": OutcomeKind.REJECTED, "Ghost line": OutcomeKind.INVALID,
        "Same bug again": OutcomeKind.DUPLICATE, "Weak hash": OutcomeKind.UNCERTAIN,
    }
    assert out.status is RunStatus.COMPLETED  # uncertainty is disclosed, not a failed check
    assert provider.count("checker", "Ghost line") == 0 and provider.count("checker", "Same bug again") == 0  # never reached the model
    report = assemble_report(store, out.run_id)
    assert [f.title for f in report.findings] == ["Mutable default argument"]
    assert [f.title for f in report.unresolved] == ["Weak hash"]
    assert report.findings[0].checker_status is CheckerStatus.ACCEPTED
    decisions = store.checker_decisions(out.run_id, disputed.finding_id)
    assert [d.verdict.value for _, d in decisions] == ["request_revision"] * 3
    types = [e.type for e in store.events(out.run_id)]
    for t in ("check_started", "finding_accepted", "finding_rejected", "finding_invalid", "finding_duplicate", "finding_uncertain", "check_finished"):
        assert t in types
    assert types.index("task_succeeded") < types.index("check_started") < types.index("run_finished")


def test_report_refuses_unverified_findings():
    with pytest.raises(ValidationError):
        Report(run_id="r", run_status=RunStatus.COMPLETED, findings=[mk()])
    with pytest.raises(ValidationError):
        Report(run_id="r", run_status=RunStatus.COMPLETED, unresolved=[mk()])


def test_checker_outage_marks_run_partial_and_accepts_nothing(store, settings, ingest):
    provider = FaultInjectingProvider(RuleBasedProvider(), fail_specialists=["checker"])
    out = asyncio.run(Orchestrator(store, settings, provider).run(ingest))
    assert out.status is RunStatus.PARTIAL
    outcomes = store.finding_outcomes(out.run_id)
    assert outcomes and {o.reason_code for o in outcomes} == {"checker_error"}
    report = assemble_report(store, out.run_id)
    assert not report.findings and len(report.unresolved) == len(outcomes)


def test_default_fake_checker_rejects_a_match_inside_a_comment(store, settings, tmp_path):
    repo = tmp_path / "r2"
    repo.mkdir()
    (repo / "app.py").write_text("# never call eval(user_input)\nx = 1\n")
    ing = ingest_repository(str(repo), None, settings, run_tools=False)
    out = asyncio.run(Orchestrator(store, settings, RuleBasedProvider()).run(ing))
    by_title = {o.finding.title: o for o in store.finding_outcomes(out.run_id)}
    assert by_title["Dynamic code execution with eval/exec"].kind is OutcomeKind.REJECTED
    assert "Dynamic code execution with eval/exec" not in [f.title for f in assemble_report(store, out.run_id).findings]


def test_check_budget_leaves_extra_findings_uncertain_and_run_partial(store, settings, ingest, monkeypatch):
    monkeypatch.setattr(check_pipeline, "MAX_CHECKED_FINDINGS", 1)
    a = mk(severity="high")
    b = mk(title="Second", start=8, end=8, evidence="def g(y=[]):", severity="low")
    provider = Routed({"Mutable default argument": [acc()], "Second": [acc()]})
    out = run(store, settings, ingest, provider, [a, b])
    kinds = {o.finding.title: (o.kind, o.reason_code) for o in store.finding_outcomes(out.run_id)}
    assert kinds["Mutable default argument"][0] is OutcomeKind.ACCEPTED  # most severe is checked first
    assert kinds["Second"] == (OutcomeKind.UNCERTAIN, "check_budget") and provider.count("checker", "Second") == 0
    assert out.status is RunStatus.PARTIAL


def test_slow_check_times_out_without_blocking_others(store, make_settings, repo):
    s = make_settings(test_mode=True, task_timeout_s=0.2)
    ing = ingest_repository(str(repo), None, s, run_tools=False)
    fast, slow = mk(severity="high"), mk(title="Slow", start=8, end=8, evidence="def g(y=[]):", severity="low")

    def adj(f, root, provider, settings, *, sleep):
        if f.title == "Slow":
            time.sleep(0.6)
        return adjudicate(f, root, provider, settings, sleep=sleep)

    provider = Routed({"Mutable default argument": [acc()], "Slow": [acc()]})
    out = asyncio.run(Orchestrator(store, s, provider, canned([fast, slow]), adj).run(ing))
    kinds = {o.finding.title: (o.kind, o.reason_code) for o in store.finding_outcomes(out.run_id)}
    assert kinds["Mutable default argument"][0] is OutcomeKind.ACCEPTED
    assert kinds["Slow"] == (OutcomeKind.UNCERTAIN, "check_timeout")
    assert out.status is RunStatus.PARTIAL


def test_rechecking_a_run_does_not_duplicate_outcomes_or_calls(store, settings, ingest, root):
    provider = Routed({"Mutable default argument": [acc()]})
    out = run(store, settings, ingest, provider, [mk()])
    n = provider.count("checker")
    asyncio.run(check_run(store, settings, provider, root, ingest.inventory, out.run_id, time.monotonic() + 30))
    assert provider.count("checker") == n and len(store.finding_outcomes(out.run_id)) == 1
    assert store.record_outcome(out.run_id, store.finding_outcomes(out.run_id)[0]) is False


def test_checker_decisions_are_append_only(store, settings, ingest):
    import sqlite3
    out = run(store, settings, ingest, Routed({"Mutable default argument": [acc()]}), [mk()])
    with pytest.raises(sqlite3.DatabaseError):
        store._conn.execute("UPDATE checker_decisions SET verdict='reject'")
    with pytest.raises(sqlite3.DatabaseError):
        store._conn.execute("DELETE FROM checker_decisions")
    assert len(store.checker_decisions(out.run_id)) == 1
