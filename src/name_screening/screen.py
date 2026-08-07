"""The screening path: query -> normalise -> embed -> search -> threshold -> decision.

One function, ``screen``, is the only way a name gets checked. The CLI, the API and
the evaluation harness all go through it, so the numbers in the README describe the
same code path a user exercises.
"""

from __future__ import annotations

from dataclasses import dataclass

from .index import WatchlistIndex
from .normalize import Script


@dataclass(frozen=True)
class Candidate:
    """One watchlist person retrieved for a query, with the evidence for the score."""

    record_id: str
    name: str
    score: float
    matched_form: str
    matched_via_alias: bool
    is_match: bool


@dataclass(frozen=True)
class ScreenResult:
    """The full outcome of screening one name."""

    query: str
    normalized_query: str
    detected_script: Script
    candidates: tuple[Candidate, ...]
    threshold: float

    @property
    def has_match(self) -> bool:
        """True if any candidate cleared the threshold."""
        raise NotImplementedError

    @property
    def top_score(self) -> float:
        """Highest similarity seen, match or not. ``0.0`` when nothing was retrieved."""
        raise NotImplementedError


def screen(
    name: str,
    index: WatchlistIndex,
    *,
    top_k: int | None = None,
    threshold: float | None = None,
) -> ScreenResult:
    """Screen one name against the watchlist.

    ``top_k`` and ``threshold`` default to the configured values; the overrides
    exist so ``evaluate.py`` can sweep the threshold without rebuilding anything.

    Candidates are deduped to one row per ``record_id``, keeping that record's
    best-scoring name variant. Without this, a person carrying five aliases
    occupies the whole result list and pushes real alternatives out.
    """
    raise NotImplementedError


def screen_batch(
    names: list[str],
    index: WatchlistIndex,
    *,
    top_k: int | None = None,
    threshold: float | None = None,
) -> list[ScreenResult]:
    """Screen many names, embedding them in one batch.

    Same results as calling ``screen`` in a loop — that equivalence is worth a test,
    because batching is where subtle prefix and ordering bugs hide.
    """
    raise NotImplementedError
