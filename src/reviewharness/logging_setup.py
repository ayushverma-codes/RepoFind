"""Logging with secret redaction."""
from __future__ import annotations

import logging
import re
import sys

LOGGER_NAME = "reviewharness"

_PATTERNS = [
    re.compile(r"gsk_[A-Za-z0-9]{10,}"),  # Groq API keys
    re.compile(r"org_[A-Za-z0-9]{10,}"),  # Provider organization identifiers
    re.compile(r"(?i)(api[_-]?key|authorization|bearer|token|secret)(\"?\s*[:=]\s*\"?)[^\s\"',}]+"),
]


def redact(text: str) -> str:
    text = _PATTERNS[0].sub("[REDACTED]", text)
    text = _PATTERNS[1].sub("[REDACTED_ORG]", text)
    return _PATTERNS[2].sub(r"\1\2[REDACTED]", text)


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = ()
        return True


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure the package logger on stderr. Safe to call repeatedly."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.addFilter(RedactingFilter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
