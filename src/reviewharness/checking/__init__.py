"""Verification of candidate findings: deterministic validation, independent checker, bounded revision."""
from reviewharness.checking.adjudicate import adjudicate
from reviewharness.checking.models import CheckSummary, CheckerReply, FindingOutcome, OutcomeKind
from reviewharness.checking.pipeline import assemble_report, check_run
from reviewharness.checking.validation import validate_candidates

__all__ = [
    "CheckSummary", "CheckerReply", "FindingOutcome", "OutcomeKind",
    "adjudicate", "assemble_report", "check_run", "validate_candidates",
]
