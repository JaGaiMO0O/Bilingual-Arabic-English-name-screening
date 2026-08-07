"""Central configuration.

Every tunable in this project lives here. No magic numbers anywhere else — if a
threshold, a batch size or a path appears in another module, it is a bug.

Values are read from the environment (see ``.env.example``) with working defaults,
so a clean checkout runs without any setup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# src/name_screening/config.py -> src/name_screening -> src -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]

# --- Defaults ---------------------------------------------------------------
# Named rather than inlined so they are greppable and so the dataclass below
# stays readable.
DEFAULT_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_QUERY_PREFIX = "query:"
DEFAULT_PASSAGE_PREFIX = "passage:"
DEFAULT_EMBED_BATCH_SIZE = 64
DEFAULT_MATCH_THRESHOLD = 0.85
DEFAULT_TOP_K = 10
DEFAULT_DATA_DIR = "data"
DEFAULT_ARTIFACTS_DIR = "artifacts"


def _env_str(key: str, default: str) -> str:
    value = os.environ.get(key)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number, got {raw!r}") from exc


def _env_path(key: str, default: str) -> Path:
    raw = _env_str(key, default)
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass(frozen=True)
class Settings:
    """Resolved configuration for one process."""

    # --- Model ---------------------------------------------------------------
    # e5 models are asymmetric: they are trained with instruction prefixes and
    # silently lose accuracy without them. Watchlist entries are encoded as
    # "passage: <name>", incoming queries as "query: <name>". Nothing errors if
    # you omit them, which is exactly why they are configuration and not a
    # literal buried in index.py.
    model_name: str = field(default_factory=lambda: _env_str("NAME_SCREENING_MODEL", DEFAULT_MODEL))
    query_prefix: str = field(
        default_factory=lambda: _env_str("NAME_SCREENING_QUERY_PREFIX", DEFAULT_QUERY_PREFIX)
    )
    passage_prefix: str = field(
        default_factory=lambda: _env_str("NAME_SCREENING_PASSAGE_PREFIX", DEFAULT_PASSAGE_PREFIX)
    )
    embed_batch_size: int = field(
        default_factory=lambda: _env_int(
            "NAME_SCREENING_EMBED_BATCH_SIZE", DEFAULT_EMBED_BATCH_SIZE
        )
    )

    # --- Retrieval -----------------------------------------------------------
    # Vectors are L2-normalised and the index is inner-product, so scores are
    # cosine similarities in [-1, 1] and this threshold is directly comparable
    # across runs.
    #
    # PLACEHOLDER. The committed value is replaced in build step 7 with the
    # operating point chosen from the precision/recall curve, and the README
    # must explain why that point and not another. Screening is asymmetric:
    # a missed sanctioned party and a false hit on an innocent customer do not
    # cost the same, and the threshold is where that judgement is expressed.
    match_threshold: float = field(
        default_factory=lambda: _env_float(
            "NAME_SCREENING_MATCH_THRESHOLD", DEFAULT_MATCH_THRESHOLD
        )
    )
    top_k: int = field(default_factory=lambda: _env_int("NAME_SCREENING_TOP_K", DEFAULT_TOP_K))

    # --- Paths ---------------------------------------------------------------
    data_dir: Path = field(
        default_factory=lambda: _env_path("NAME_SCREENING_DATA_DIR", DEFAULT_DATA_DIR)
    )
    artifacts_dir: Path = field(
        default_factory=lambda: _env_path("NAME_SCREENING_ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR)
    )

    # --- Derived paths -------------------------------------------------------
    @property
    def seed_watchlist_path(self) -> Path:
        """Small bilingual seed list, committed, so the repo runs immediately."""
        return self.data_dir / "watchlist_seed.csv"

    @property
    def eval_cases_path(self) -> Path:
        """Labelled evaluation set, committed."""
        return self.data_dir / "eval_cases.csv"

    @property
    def index_path(self) -> Path:
        """Serialised FAISS index."""
        return self.artifacts_dir / "watchlist.faiss"

    @property
    def metadata_path(self) -> Path:
        """Sidecar mapping FAISS row ordinals back to watchlist records.

        The index stores vectors and nothing else, so this file is what turns a
        hit into a person. It must be written and loaded in lockstep with the
        index or every result is silently mislabelled.
        """
        return self.artifacts_dir / "watchlist_meta.jsonl"

    @property
    def eval_report_path(self) -> Path:
        return self.artifacts_dir / "eval_report.md"

    def validate(self) -> None:
        """Fail loudly on nonsense configuration rather than at query time."""
        if not -1.0 <= self.match_threshold <= 1.0:
            raise ValueError(
                "match_threshold must be a cosine similarity in [-1, 1], "
                f"got {self.match_threshold}"
            )
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.embed_batch_size < 1:
            raise ValueError(f"embed_batch_size must be >= 1, got {self.embed_batch_size}")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, resolved once.

    Cached so that CLI and API share one instance. Tests that manipulate the
    environment should call ``get_settings.cache_clear()`` or construct
    ``Settings()`` directly.
    """
    settings = Settings()
    settings.validate()
    return settings
