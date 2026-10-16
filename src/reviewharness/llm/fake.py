"""Deterministic providers that need no API key and no network.

RuleBasedProvider is the REVIEW_TEST_MODE provider: it reads the numbered source
blocks in the prompt and applies fixed regex rules, so its findings are
source-grounded and reproducible. It is a test double, not a quality claim.

ScriptedProvider replays a fixed list of replies/exceptions; tests use it to
simulate malformed output, timeouts and provider outages.

Prompt format parsed here (produced by specialists/prompts.py):
    SPECIALIST: <name>
    ##### FILE <path> (lines 1-<shown> of <total>) #####
    <n>| <source line>
    ##### END FILE #####
"""
from __future__ import annotations

import json
import re
from pathlib import PurePosixPath
from typing import Sequence, Union

from reviewharness.llm.base import LLMResult

_SPECIALIST = re.compile(r"^SPECIALIST: (\w+)$", re.MULTILINE)
_BLOCK = re.compile(r"^##### FILE (?P<path>.+?) \(lines \d+-\d+ of \d+\) #####\n(?P<body>.*?)\n##### END FILE #####$", re.MULTILINE | re.DOTALL)
_NUMBERED = re.compile(r"^\s*(\d+)\| ?(.*)$")

# (specialist, category, severity, title, pattern, rationale, impact, confidence)
_RULES = [
    ("bug", "bug", "medium", "Mutable default argument",
     re.compile(r"def\s+\w+\(.*=\s*(\[\]|\{\}|dict\(\)|list\(\))"),
     "The default object is created once and shared across calls.",
     "State leaks between calls and causes surprising behaviour.", 0.8),
    ("bug", "bug", "low", "Bare except clause",
     re.compile(r"^\s*except\s*:"),
     "A bare except also catches KeyboardInterrupt and SystemExit and hides real errors.",
     "Failures are silently swallowed.", 0.6),
    ("bug", "bug", "low", "Comparison to None with ==",
     re.compile(r"[=!]=\s*None\b"),
     "Identity checks against None should use 'is' / 'is not'.",
     "Objects that override __eq__ can give wrong results.", 0.5),
    ("security", "security", "high", "Shell command execution with shell=True",
     re.compile(r"shell\s*=\s*True"),
     "Passing a string to the shell allows command injection if any part is attacker-controlled.",
     "Arbitrary command execution.", 0.8),
    ("security", "security", "high", "Dynamic code execution with eval/exec",
     re.compile(r"(?<![\w.])(eval|exec)\("),
     "eval/exec run arbitrary Python from a string.",
     "Arbitrary code execution if the input is attacker-influenced.", 0.8),
    ("security", "security", "high", "Hardcoded credential",
     re.compile(r"(?i)^\s*\w*(password|passwd|secret|api_key|token)\w*\s*=\s*[\"'][^\"']+[\"']"),
     "A credential is committed in source code.",
     "Anyone with repository access can use the credential.", 0.7),
    ("security", "security", "medium", "Weak hash algorithm",
     re.compile(r"hashlib\.(md5|sha1)\("),
     "MD5 and SHA-1 are broken for security purposes.",
     "Collisions allow forged integrity checks.", 0.6),
    ("security", "security", "high", "TLS certificate verification disabled",
     re.compile(r"verify\s*=\s*False"),
     "Disabling verification removes protection against man-in-the-middle attacks.",
     "Traffic can be intercepted or altered.", 0.8),
    ("testing_maintainability", "maintainability", "low", "TODO/FIXME marker left in code",
     re.compile(r"#\s*(TODO|FIXME)"),
     "An unresolved marker indicates unfinished work.",
     "Known gaps are easy to forget.", 0.5),
    ("testing_maintainability", "testing", "medium", "Trivial assertion",
     re.compile(r"assert\s+True\b"),
     "The assertion can never fail, so the test proves nothing.",
     "False confidence in test coverage.", 0.8),
]


def _parse_prompt(user: str) -> tuple[str, dict[str, dict[int, str]]]:
    match = _SPECIALIST.search(user)
    specialist = match.group(1) if match else "baseline"
    files: dict[str, dict[int, str]] = {}
    for block in _BLOCK.finditer(user):
        lines: dict[int, str] = {}
        for raw in block.group("body").split("\n"):
            m = _NUMBERED.match(raw)
            if m:
                lines[int(m.group(1))] = m.group(2)
        files[block.group("path")] = lines
    return specialist, files


def _finding(path: str, line: int, text: str, category: str, severity: str, title: str,
             rationale: str, impact: str, confidence: float) -> dict:
    return {
        "category": category, "title": title, "severity": severity,
        "file_path": path, "start_line": line, "end_line": line,
        "evidence": text.strip(), "rationale": rationale, "impact": impact,
        "confidence": confidence,
    }


def _is_test_path(path: str) -> bool:
    p = PurePosixPath(path)
    return p.name.startswith("test_") or any(part in ("tests", "test") for part in p.parts[:-1])


def _structural_findings(specialist: str, files: dict[str, dict[int, str]]) -> list[dict]:
    if specialist not in ("testing_maintainability", "baseline"):
        return []
    out: list[dict] = []
    test_names = {PurePosixPath(p).name for p in files if _is_test_path(p)}
    for path, lines in sorted(files.items()):
        numbers = sorted(lines)
        if _is_test_path(path):
            for i, n in enumerate(numbers):
                text = lines[n]
                m = re.match(r"^(\s*)def\s+test_\w+\(", text)
                if not m:
                    continue
                indent = len(m.group(1))
                body = []
                for n2 in numbers[i + 1:]:
                    t2 = lines[n2]
                    if t2.strip() and len(t2) - len(t2.lstrip()) <= indent:
                        break
                    body.append(t2)
                if not any("assert" in b or "raises" in b for b in body):
                    out.append(_finding(path, n, text, "testing", "medium", "Test function contains no assertion",
                                        "The test runs code but never checks a result.",
                                        "Regressions pass unnoticed.", 0.6))
        elif path.endswith(".py") and any(re.match(r"^\s*def\s", t) for t in lines.values()):
            stem = PurePosixPath(path).stem
            if f"test_{stem}.py" not in test_names and numbers:
                first = next((n for n in numbers if lines[n].strip()), numbers[0])
                out.append(_finding(path, first, lines[first], "testing", "low",
                                    f"No test module named test_{stem}.py",
                                    "No test file follows the naming convention for this module.",
                                    "Changes to this module may be untested.", 0.4))
    return out


class RuleBasedProvider:
    name = "fake"
    model = "rule-based-v1"

    def complete(self, system: str, user: str) -> LLMResult:
        if user.startswith("ROLE: checker"):
            return self._check(user)
        if user.startswith("ROLE: reviser"):
            text = json.dumps({"finding": None})
            return LLMResult(text=text, prompt_tokens=len(user) // 4, completion_tokens=len(text) // 4)
        specialist, files = _parse_prompt(user)
        findings: list[dict] = []
        for path, lines in sorted(files.items()):
            if _is_test_path(path) and specialist != "testing_maintainability" and specialist != "baseline":
                continue
            for number in sorted(lines):
                for spec, cat, sev, title, pattern, why, impact, conf in _RULES:
                    if specialist in (spec, "baseline") and pattern.search(lines[number]):
                        findings.append(_finding(path, number, lines[number], cat, sev, title, why, impact, conf))
        findings += _structural_findings(specialist, files)
        text = json.dumps({"findings": findings})
        return LLMResult(text=text, prompt_tokens=(len(system) + len(user)) // 4, completion_tokens=len(text) // 4)


    def _check(self, user: str) -> LLMResult:
        """Re-apply the matching rule to the cited lines of an independently rendered window."""
        title = re.search(r"^- title: (.*)$", user, re.MULTILINE)
        cited = re.search(r"^- lines: (\d+)-(\d+)$", user, re.MULTILINE)
        path = re.search(r"^- file: (.*)$", user, re.MULTILINE)
        _, files = _parse_prompt(user)
        lines = files.get(path.group(1), {}) if path else {}
        start, end = (int(cited.group(1)), int(cited.group(2))) if cited else (0, 0)
        text_lines = [lines[n] for n in range(start, end + 1) if n in lines]
        rule = next((r for r in _RULES if title and r[3] == title.group(1)), None)
        if rule is None:  # structural findings (missing tests etc.) cannot be re-derived from one window
            reply = {"verdict": "accept", "reasoning": "The cited lines exist and are consistent with the claim.", "counterexample": None, "suggested_severity": None}
        else:
            hits = [t for t in text_lines if rule[4].search(t)]
            if not hits:
                reply = {"verdict": "reject", "reasoning": "No cited line matches the described pattern.",
                         "counterexample": f"None of lines {start}-{end} contains the pattern for '{rule[3]}'.", "suggested_severity": None}
            elif all(t.lstrip().startswith("#") for t in hits) and rule[3] != "TODO/FIXME marker left in code":
                reply = {"verdict": "reject", "reasoning": "The match is inside a comment, not executable code.",
                         "counterexample": "The only matching text is a comment.", "suggested_severity": None}
            else:
                reply = {"verdict": "accept", "reasoning": "The cited lines contain the described pattern.", "counterexample": None, "suggested_severity": None}
        out = json.dumps(reply)
        return LLMResult(text=out, prompt_tokens=len(user) // 4, completion_tokens=len(out) // 4)


ScriptItem = Union[str, LLMResult, Exception]


class ScriptedProvider:
    """Replays `script` in order. Exceptions are raised, strings/results are returned."""

    name = "scripted"
    model = "scripted"

    def __init__(self, script: Sequence[ScriptItem]) -> None:
        self._script = list(script)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> LLMResult:
        self.calls.append((system, user))
        if len(self.calls) > len(self._script):
            raise RuntimeError("ScriptedProvider script exhausted: more calls than scripted replies")
        item = self._script[len(self.calls) - 1]
        if isinstance(item, Exception):
            raise item
        if isinstance(item, LLMResult):
            return item
        return LLMResult(text=item, prompt_tokens=10, completion_tokens=5)


class FaultInjectingProvider:
    """Wraps another provider for demos/tests: adds latency and fails chosen specialists.

    Used only with test mode (CLI --simulate-delay / --simulate-fail) to show concurrency
    and partial-failure handling without a real outage.
    """

    def __init__(self, inner, *, delay_s: float = 0.0, fail_specialists: Sequence[str] = ()) -> None:
        self._inner = inner
        self._delay_s = delay_s
        self._fail = frozenset(fail_specialists)
        self.name = f"{inner.name}+faults"
        self.model = inner.model

    def complete(self, system: str, user: str) -> LLMResult:
        import time

        from reviewharness.llm.base import LLMPermanentError

        match = _SPECIALIST.search(user)
        specialist = match.group(1) if match else ""
        if user.startswith("ROLE: checker"):
            specialist = "checker"
        if self._delay_s:
            time.sleep(self._delay_s)
        if specialist in self._fail:
            raise LLMPermanentError(f"simulated provider failure for {specialist}")
        return self._inner.complete(system, user)
