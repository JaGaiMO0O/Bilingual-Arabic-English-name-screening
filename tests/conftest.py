"""Shared fixtures.

The default test run must pass with no network, no model download and no dataset.
Anything that needs the embedding model carries ``@pytest.mark.requires_model``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def seed_csv(repo_root: Path) -> Path:
    return repo_root / "data" / "watchlist_seed.csv"


@pytest.fixture
def eval_csv(repo_root: Path) -> Path:
    return repo_root / "data" / "eval_cases.csv"


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Stop configuration leaking between tests that patch the environment."""
    from name_screening.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
