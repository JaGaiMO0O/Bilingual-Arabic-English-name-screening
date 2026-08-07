"""Configuration tests.

Deliberately real tests rather than placeholders: config is the one module that is
fully implemented in the skeleton, and the prefix defaults it carries are load-bearing.
"""

from __future__ import annotations

import pytest

from name_screening.config import Settings, get_settings


def test_defaults_are_valid():
    Settings().validate()


def test_e5_prefixes_are_present_by_default():
    """e5 is asymmetric. Losing these costs accuracy and raises nothing."""
    settings = Settings()
    assert settings.query_prefix.startswith("query")
    assert settings.passage_prefix.startswith("passage")


def test_threshold_is_a_cosine_similarity():
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        Settings(match_threshold=42.0).validate()


def test_top_k_must_be_positive():
    with pytest.raises(ValueError, match="top_k"):
        Settings(top_k=0).validate()


def test_env_overrides_threshold(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NAME_SCREENING_MATCH_THRESHOLD", "0.91")
    assert Settings().match_threshold == pytest.approx(0.91)


def test_non_numeric_threshold_fails_loudly(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NAME_SCREENING_MATCH_THRESHOLD", "high")
    with pytest.raises(ValueError, match="must be a number"):
        Settings()


def test_paths_resolve_under_the_repo_root():
    settings = Settings()
    assert settings.seed_watchlist_path.name == "watchlist_seed.csv"
    assert settings.index_path.parent == settings.artifacts_dir
    assert settings.metadata_path.parent == settings.artifacts_dir


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


def test_committed_data_files_exist(seed_csv, eval_csv):
    """A clean clone must be runnable without a download."""
    assert seed_csv.exists()
    assert eval_csv.exists()
