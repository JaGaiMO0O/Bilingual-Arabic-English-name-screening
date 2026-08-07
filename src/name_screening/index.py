"""Embedding and FAISS index construction, persistence and loading.

Heavy imports (``sentence_transformers``, ``faiss``) are deliberately deferred into
function bodies. Importing this module must stay cheap so that ``cli.py`` can offer
``--help`` instantly and so the offline test suite can import the package without
pulling in torch.

Two things here are easy to get wrong and produce no error when you do:

1. **Prefixes.** e5 is asymmetric. Watchlist entries are encoded as
   ``"passage: <name>"``, queries as ``"query: <name>"``. Both come from
   ``config.Settings`` so the two paths cannot drift apart.
2. **Normalise, then inner product.** Call ``faiss.normalize_L2()`` on the matrix
   and use ``IndexFlatIP``. Inner product over unit vectors is cosine similarity,
   which is what the threshold in config assumes. ``IndexFlatL2`` without
   normalisation returns squared distances — smaller is better, the range is
   unbounded, and every threshold comparison in this codebase silently inverts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .ingest import PersonRecord

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import numpy as np


@dataclass(frozen=True)
class IndexEntry:
    """One indexed vector's provenance.

    A record contributes several rows to the index — one per name variant — so
    the mapping from FAISS ordinal to person is many-to-one. ``record_id`` is
    what dedupes hits back down to people.
    """

    ordinal: int
    record_id: str
    surface_form: str
    normalized_form: str
    is_alias: bool


@dataclass
class WatchlistIndex:
    """A loaded FAISS index plus the metadata that gives its rows meaning."""

    faiss_index: Any
    entries: list[IndexEntry]
    model_name: str

    def __len__(self) -> int:
        raise NotImplementedError

    def record_ids(self) -> set[str]:
        raise NotImplementedError


def load_model(model_name: str) -> Any:
    """Load the sentence-transformers model. Imported lazily; cached by the library."""
    raise NotImplementedError


def embed_passages(texts: list[str], *, model: Any = None) -> np.ndarray:
    """Encode watchlist entries with the passage prefix.

    Returns a float32 array of shape ``(len(texts), dim)``, L2-normalised and
    ready for an inner-product index.
    """
    raise NotImplementedError


def embed_queries(texts: list[str], *, model: Any = None) -> np.ndarray:
    """Encode incoming queries with the query prefix. Same shape and normalisation."""
    raise NotImplementedError


def build_index(records: list[PersonRecord]) -> WatchlistIndex:
    """Normalise every name variant of every record, embed, and build an ``IndexFlatIP``.

    Must be deterministic: the same input produces the same ordinals, so a rebuilt
    index is comparable with the evaluation numbers quoted in the README.
    """
    raise NotImplementedError


def save_index(index: WatchlistIndex, index_path: Path, metadata_path: Path) -> None:
    """Write the FAISS index and its JSONL metadata sidecar.

    The sidecar records ``model_name``. Loading an index built by a different
    model gives plausible-looking nonsense, so ``load_index`` checks it.
    """
    raise NotImplementedError


def load_index(index_path: Path, metadata_path: Path) -> WatchlistIndex:
    """Load index and sidecar, verifying they agree with each other and with config.

    Raises if the row count and metadata length disagree, or if the sidecar was
    written by a different embedding model.
    """
    raise NotImplementedError
