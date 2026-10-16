"""Minimal environment for child processes.

Children (git, ruff, bandit) never inherit the parent's environment, so
GROQ_API_KEY and any other secret cannot leak into them.
"""
from __future__ import annotations

import os

# Windows launchers (e.g. the Python install manager) misbehave without the profile
# variables and fall back to literal "~" paths relative to the cwd.
_BASE = (
    "PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "TMPDIR",
    "USERPROFILE", "HOME", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "APPDATA", "PROGRAMDATA",
    "LANG", "LC_ALL",
)
_NETWORK = (
    "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY",
    "https_proxy", "http_proxy", "all_proxy", "no_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "GIT_SSL_CAINFO",
)


def safe_env(network: bool = False) -> dict[str, str]:
    names = _BASE + (_NETWORK if network else ())
    env = {k: os.environ[k] for k in names if k in os.environ}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env
