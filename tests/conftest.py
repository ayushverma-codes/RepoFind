"""Shared fixtures. The sample repo under fixtures/ is data, not tests to collect."""
from pathlib import Path

import pytest

from reviewharness.config import Settings

collect_ignore = ["fixtures"]
SAMPLE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


@pytest.fixture
def sample_repo() -> Path:
    return SAMPLE_REPO


@pytest.fixture
def make_settings(tmp_path):
    def _make(**overrides) -> Settings:
        return Settings(data_dir=tmp_path / "data", **overrides)

    return _make
