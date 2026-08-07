"""Evaluation harness — the reason this project exists.

Runs the labelled cases in ``data/eval_cases.csv`` through the same ``screen()``
path a user hits, and emits precision, recall, F1 and the confusion counts behind
them. The README quotes these numbers, so nothing here may special-case the
evaluation set or bypass normalisation.

**Hard negatives are mandatory.** Names that are genuinely similar but are not the
same person: two people sharing a common surname, a first/last transposition, a
one-letter difference. Without them precision is meaningless — any multilingual
embedding model scores near-perfectly on a set of easy positives and unrelated
negatives, and reporting that number is the most common way a portfolio ML repo
misleads its reader. The count of hard negatives goes in the README.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .index import WatchlistIndex


@dataclass(frozen=True)
class EvalCase:
    """One labelled screening case.

    ``target_id`` is the watchlist record the case concerns — for a positive, the
    person the query should find; for a hard negative, the confusable record it
    must *not* match. ``should_match`` is the label.
    """

    case_id: str
    query: str
    target_id: str
    should_match: bool
    category: str
    notes: str = ""


@dataclass(frozen=True)
class Metrics:
    """Confusion counts and the rates derived from them, at one threshold."""

    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        """TP / (TP + FP). Defined as 0.0 when nothing was flagged."""
        raise NotImplementedError

    @property
    def recall(self) -> float:
        """TP / (TP + FN). Defined as 0.0 when there are no positives."""
        raise NotImplementedError

    @property
    def f1(self) -> float:
        raise NotImplementedError


def load_eval_cases(path: Path) -> list[EvalCase]:
    """Read the labelled set.

    Expected columns: ``case_id,query,target_id,should_match,category,notes``.
    Lines beginning ``#`` are comments and must be skipped.

    Raises if any ``target_id`` is absent from the watchlist — a typo there
    quietly turns a positive case into an unwinnable one and depresses recall
    for reasons no one will find by reading the metrics.
    """
    raise NotImplementedError


def evaluate_at_threshold(
    cases: list[EvalCase], index: WatchlistIndex, threshold: float
) -> Metrics:
    """Score every case at one threshold."""
    raise NotImplementedError


def sweep_thresholds(
    cases: list[EvalCase], index: WatchlistIndex, thresholds: list[float]
) -> list[Metrics]:
    """Evaluate across a range of thresholds to produce the precision/recall curve.

    Embeds each query once and re-thresholds the cached scores rather than
    re-running retrieval per threshold.
    """
    raise NotImplementedError


def metrics_by_category(
    cases: list[EvalCase], index: WatchlistIndex, threshold: float
) -> dict[str, Metrics]:
    """Break the same run down per category.

    The aggregate number hides the interesting part. Cross-script recall and
    hard-negative precision are the two figures that say whether this works, and
    an aggregate F1 can look healthy while cross-script matching is failing
    outright.
    """
    raise NotImplementedError


def format_report(overall: Metrics, per_category: dict[str, Metrics], curve: list[Metrics]) -> str:
    """Render the markdown report that the README quotes.

    Includes the size of the evaluation set and the hard-negative count, so the
    numbers cannot be read without their denominator.
    """
    raise NotImplementedError


def write_report(report: str, path: Path) -> None:
    """Write the report to ``artifacts/`` (gitignored; the README carries the numbers)."""
    raise NotImplementedError
