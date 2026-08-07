"""Watchlist ingestion — seed CSV and streaming OpenSanctions FollowTheMoney reader.

Both sources produce ``PersonRecord`` objects, so everything downstream is unaware
of which one it is fed. That is what lets the repo be demo-able from the committed
seed file while still handling the real dataset.

The FtM reader must be **streaming**: the PEP dataset does not belong in memory, and
a reviewer running this on a laptop will notice if it tries.

Data source: OpenSanctions <https://www.opensanctions.org/datasets/peps/>, licensed
CC-BY-NC 4.0. Attribution is required and lives in the README.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PersonRecord:
    """One person on the watchlist.

    ``name`` is the primary display name as published. ``aliases`` holds every
    other spelling the source gives, which is where most of the cross-script
    value lives — OpenSanctions frequently carries both the Arabic and the
    romanised form of the same person, and both should be indexed.
    """

    record_id: str
    name: str
    aliases: tuple[str, ...] = ()
    birth_date: str | None = None
    countries: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    source: str = "seed"

    def all_names(self) -> list[str]:
        """Primary name followed by aliases, deduped, order-stable."""
        raise NotImplementedError


def load_seed_watchlist(path: Path) -> list[PersonRecord]:
    """Read the committed bilingual seed CSV.

    Expected columns: ``record_id,name,aliases,birth_date,countries,topics``.
    Multi-valued fields are ``;``-separated. Lines beginning ``#`` are comments
    and must be skipped — the committed seed documents its own schema inline, and
    ``csv.DictReader`` will happily parse those comments into records otherwise.

    Raises on a missing or malformed header rather than silently producing empty
    records.
    """
    raise NotImplementedError


def stream_ftm_persons(path: Path) -> Iterator[PersonRecord]:
    """Yield ``PersonRecord`` objects from a FollowTheMoney JSON-lines export.

    Reads line by line and never materialises the file. Filters to
    ``schema == "Person"`` and skips entries with no usable name.

    FtM puts values in ``properties`` as lists, so ``properties["name"]`` is a
    list even when there is one name — treating it as a string is the standard
    way to get a one-character record id.
    """
    raise NotImplementedError


def load_watchlist(path: Path) -> list[PersonRecord]:
    """Dispatch on file extension: ``.csv`` to the seed reader, ``.json``/``.jsonl`` to FtM."""
    raise NotImplementedError
