"""Watchlist ingestion — seed CSV and streaming OpenSanctions FollowTheMoney reader.

Both sources produce :class:`PersonRecord` objects, so everything downstream is
unaware of which one it is fed. That is what lets the repo be demo-able from the
committed seed file while still handling the real dataset.

The FtM reader is a generator and stays one: the PEP export does not belong in
memory, and a reviewer running this on a laptop will notice if it tries.
:func:`load_watchlist` materialises for convenience, which is fine for the seed and
wrong for the bulk file — call :func:`stream_ftm_persons` directly for that.

Data source: OpenSanctions <https://www.opensanctions.org/datasets/peps/>, licensed
CC-BY-NC 4.0. Attribution is required and lives in the README.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Columns the seed CSV cannot do without. Everything else is optional.
SEED_REQUIRED_COLUMNS = frozenset({"record_id", "name"})

#: Separator for multi-valued CSV fields (aliases, countries, topics).
MULTI_VALUE_SEPARATOR = ";"

#: Lines beginning with this are documentation, not data. The committed CSVs carry
#: their schema and their category budget inline, and csv.DictReader would
#: otherwise parse those comments into records.
COMMENT_PREFIX = "#"

#: FtM schema name for a natural person. Companies, vessels and addresses share the
#: same file and are not screened here.
PERSON_SCHEMA = "Person"

#: FtM property holding the published primary name.
FTM_PRIMARY_NAME_PROPERTY = "name"

#: FtM properties holding every other spelling, in the order they are preferred.
#: ``weakAlias`` is included deliberately: OpenSanctions uses it for low-confidence
#: spellings, which is exactly where cross-script variants tend to sit.
FTM_ALIAS_PROPERTIES = ("alias", "weakAlias", "previousName")

#: FtM properties that carry a country. Merged: the distinction does not matter for
#: screening, and records populate one or the other inconsistently.
FTM_COUNTRY_PROPERTIES = ("country", "nationality")


class IngestError(ValueError):
    """A watchlist source is malformed or unusable.

    Deliberately loud. A silently skipped record is a person who is not screened,
    which is the one failure mode this project exists to avoid.
    """


def _dedupe(values: Iterable[str]) -> list[str]:
    """Strip, drop blanks, remove duplicates, preserve first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _split_multi(value: str | None) -> tuple[str, ...]:
    """Split a ``;``-separated CSV field."""
    if not value:
        return ()
    return tuple(_dedupe(value.split(MULTI_VALUE_SEPARATOR)))


def _strip_comments(lines: Iterable[str]) -> Iterator[str]:
    """Drop full-line comments before the CSV reader sees them."""
    for line in lines:
        if line.lstrip().startswith(COMMENT_PREFIX):
            continue
        yield line


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
        return _dedupe([self.name, *self.aliases])


def load_seed_watchlist(path: Path) -> list[PersonRecord]:
    """Read the committed bilingual seed CSV.

    Expected columns: ``record_id,name,aliases,birth_date,countries,topics``.
    Multi-valued fields are ``;``-separated. Lines beginning ``#`` are comments
    and are skipped.

    Raises :class:`IngestError` on a missing header, an absent required column, a
    record with no name, or a duplicate ``record_id``. That last one matters more
    than it looks: the index metadata is keyed by ``record_id``, so a duplicate
    silently merges two people into one search result.
    """
    if not path.exists():
        raise IngestError(f"seed watchlist not found: {path}")

    # utf-8-sig so a BOM from a spreadsheet export does not end up inside the
    # first column name, which produces a baffling "missing record_id" error.
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(_strip_comments(handle))

        if reader.fieldnames is None:
            raise IngestError(f"{path} is empty — expected a CSV header row")

        columns = {(field or "").strip() for field in reader.fieldnames}
        missing = SEED_REQUIRED_COLUMNS - columns
        if missing:
            raise IngestError(
                f"{path} is missing required column(s): {', '.join(sorted(missing))}. "
                f"Found: {', '.join(sorted(columns))}"
            )

        records: list[PersonRecord] = []
        seen_ids: set[str] = set()

        for row_number, row in enumerate(reader, start=1):
            record_id = (row.get("record_id") or "").strip()
            if not record_id:
                raise IngestError(f"{path}: data row {row_number} has no record_id")

            name = (row.get("name") or "").strip()
            if not name:
                raise IngestError(f"{path}: record {record_id!r} has no name")

            if record_id in seen_ids:
                raise IngestError(f"{path}: duplicate record_id {record_id!r}")
            seen_ids.add(record_id)

            birth_date = (row.get("birth_date") or "").strip() or None

            records.append(
                PersonRecord(
                    record_id=record_id,
                    name=name,
                    aliases=_split_multi(row.get("aliases")),
                    birth_date=birth_date,
                    countries=tuple(c.upper() for c in _split_multi(row.get("countries"))),
                    topics=_split_multi(row.get("topics")),
                    source="seed",
                )
            )

    return records


def _ftm_values(properties: dict[str, Any], key: str) -> list[str]:
    """Read one FtM property as a list of strings.

    FtM puts every value in a list, even when there is exactly one — reading
    ``properties["name"]`` as a string is the standard way to end up with a
    one-character record. A bare string is tolerated anyway, because hand-made
    fixtures and some downstream exports do not honour the convention.
    """
    raw = properties.get(key)
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(value) for value in raw if isinstance(value, str | int | float)]
    return []


def _person_from_ftm(entity: dict[str, Any]) -> PersonRecord | None:
    """Convert one FtM entity to a record, or ``None`` if it is not a usable person.

    Returns ``None`` — rather than raising — for entities that are simply not our
    concern: a non-Person schema, or a person with no name at all. Those are
    expected in a mixed export, unlike malformed JSON.
    """
    if entity.get("schema") != PERSON_SCHEMA:
        return None

    record_id = str(entity.get("id") or "").strip()
    if not record_id:
        return None

    properties = entity.get("properties")
    if not isinstance(properties, dict):
        properties = {}

    caption = entity.get("caption")
    names = _dedupe(
        [
            *_ftm_values(properties, FTM_PRIMARY_NAME_PROPERTY),
            # caption sits between the primary name and the aliases so that it is
            # only promoted to primary when the name property is absent entirely.
            *([caption] if isinstance(caption, str) else []),
            *(value for key in FTM_ALIAS_PROPERTIES for value in _ftm_values(properties, key)),
        ]
    )
    if not names:
        return None

    countries = _dedupe(
        value.upper() for key in FTM_COUNTRY_PROPERTIES for value in _ftm_values(properties, key)
    )
    birth_dates = _ftm_values(properties, "birthDate")

    return PersonRecord(
        record_id=record_id,
        name=names[0],
        aliases=tuple(names[1:]),
        birth_date=birth_dates[0] if birth_dates else None,
        countries=tuple(countries),
        topics=tuple(_dedupe(_ftm_values(properties, "topics"))),
        source="opensanctions",
    )


def stream_ftm_persons(path: Path) -> Iterator[PersonRecord]:
    """Yield :class:`PersonRecord` objects from a FollowTheMoney JSON-lines export.

    Reads line by line and never materialises the file. Filters to
    ``schema == "Person"`` and skips entities with no usable name.

    Malformed JSON raises :class:`IngestError` naming the line number, rather than
    being skipped. A truncated download is the likely cause, and silently
    screening against a partial watchlist is worse than failing the build.
    """
    if not path.exists():
        raise IngestError(f"watchlist not found: {path}")

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                entity = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise IngestError(f"{path}:{line_number}: malformed JSON: {exc}") from exc

            if not isinstance(entity, dict):
                raise IngestError(
                    f"{path}:{line_number}: expected a JSON object, got {type(entity).__name__}"
                )

            record = _person_from_ftm(entity)
            if record is not None:
                yield record


def load_watchlist(path: Path) -> list[PersonRecord]:
    """Dispatch on file extension and return every record.

    ``.csv`` goes to the seed reader; ``.json``, ``.jsonl`` and ``.ndjson`` to the
    FtM reader. OpenSanctions names its export ``entities.ftm.json`` even though
    the contents are JSON lines, which is why ``.json`` maps to the streaming
    reader and not to :func:`json.load`.

    This materialises the whole list. Correct for the seed file; for the bulk
    export, iterate :func:`stream_ftm_persons` instead.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_seed_watchlist(path)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return list(stream_ftm_persons(path))
    raise IngestError(
        f"unsupported watchlist format {path.suffix!r} for {path}. "
        "Expected .csv (seed) or .json/.jsonl/.ndjson (FollowTheMoney)."
    )
