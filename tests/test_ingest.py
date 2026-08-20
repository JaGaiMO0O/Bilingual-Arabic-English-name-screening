"""Tests for watchlist ingestion — the committed seed and the FtM streaming reader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from name_screening.ingest import (
    IngestError,
    PersonRecord,
    load_seed_watchlist,
    load_watchlist,
    stream_ftm_persons,
)


def write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def write_jsonl(path: Path, entities: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(entity, ensure_ascii=False) for entity in entities),
        encoding="utf-8",
    )
    return path


def ftm_person(entity_id: str, **properties) -> dict:
    return {"id": entity_id, "schema": "Person", "properties": properties}


# --- The committed seed ------------------------------------------------------


def test_loads_the_committed_seed(seed_csv: Path):
    """The repo must be runnable from a clean clone with no download."""
    records = load_seed_watchlist(seed_csv)
    assert len(records) == 6
    assert all(isinstance(record, PersonRecord) for record in records)
    assert {record.record_id for record in records} == {f"seed-00{n}" for n in range(1, 7)}


def test_seed_comments_are_not_parsed_as_records(seed_csv: Path):
    """The seed documents its schema inline; DictReader would ingest those lines."""
    records = load_seed_watchlist(seed_csv)
    assert not any(record.record_id.startswith("#") for record in records)
    assert not any(record.name.startswith("#") for record in records)


def test_seed_splits_aliases_and_preserves_both_scripts(seed_csv: Path):
    records = {record.record_id: record for record in load_seed_watchlist(seed_csv)}
    hashimi = records["seed-001"]
    assert hashimi.name == "طارق الهاشمي"
    assert "Tariq Al-Hashimi" in hashimi.aliases
    assert "طارق الهاشمى" in hashimi.aliases
    assert hashimi.countries == ("IQ",)


def test_seed_records_carry_several_names(seed_csv: Path):
    """Most cross-script value lives in the aliases, not the primary name."""
    records = load_seed_watchlist(seed_csv)
    assert all(len(record.all_names()) >= 2 for record in records)


# --- PersonRecord ------------------------------------------------------------


def test_all_names_dedupes_and_keeps_order():
    record = PersonRecord(
        record_id="x",
        name="Tariq",
        aliases=("Tarek", "Tariq", "  Tarek  ", "طارق"),
    )
    assert record.all_names() == ["Tariq", "Tarek", "طارق"]


def test_all_names_on_a_record_with_no_aliases():
    assert PersonRecord(record_id="x", name="Tariq").all_names() == ["Tariq"]


# --- Seed validation ---------------------------------------------------------


def test_missing_required_column_raises(tmp_path: Path):
    path = write_csv(tmp_path / "bad.csv", "record_id,alias\nseed-1,foo\n")
    with pytest.raises(IngestError, match="missing required column"):
        load_seed_watchlist(path)


def test_duplicate_record_id_raises(tmp_path: Path):
    """A duplicate silently merges two people into one search result."""
    path = write_csv(tmp_path / "dupe.csv", "record_id,name\nseed-1,Tariq\nseed-1,Khalid\n")
    with pytest.raises(IngestError, match="duplicate record_id"):
        load_seed_watchlist(path)


def test_record_without_a_name_raises(tmp_path: Path):
    path = write_csv(tmp_path / "noname.csv", "record_id,name\nseed-1,\n")
    with pytest.raises(IngestError, match="has no name"):
        load_seed_watchlist(path)


def test_record_without_an_id_raises(tmp_path: Path):
    path = write_csv(tmp_path / "noid.csv", "record_id,name\n,Tariq\n")
    with pytest.raises(IngestError, match="no record_id"):
        load_seed_watchlist(path)


def test_empty_file_raises(tmp_path: Path):
    path = write_csv(tmp_path / "empty.csv", "")
    with pytest.raises(IngestError, match="empty"):
        load_seed_watchlist(path)


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(IngestError, match="not found"):
        load_seed_watchlist(tmp_path / "nope.csv")


def test_byte_order_mark_does_not_break_the_header(tmp_path: Path):
    """A spreadsheet export prepends a BOM, which lands inside the first column name."""
    path = tmp_path / "bom.csv"
    path.write_text("record_id,name\nseed-1,Tariq\n", encoding="utf-8-sig")
    assert load_seed_watchlist(path)[0].record_id == "seed-1"


# --- FollowTheMoney streaming ------------------------------------------------


def test_ftm_reads_single_element_lists_as_whole_strings(tmp_path: Path):
    """FtM wraps every value in a list. Reading it as a string yields one character."""
    path = write_jsonl(tmp_path / "e.json", [ftm_person("q1", name=["Tariq Al-Hashimi"])])
    (record,) = stream_ftm_persons(path)
    assert record.name == "Tariq Al-Hashimi"


def test_ftm_tolerates_a_bare_string_value(tmp_path: Path):
    path = write_jsonl(tmp_path / "e.json", [ftm_person("q1", name="Tariq")])
    (record,) = stream_ftm_persons(path)
    assert record.name == "Tariq"


def test_ftm_filters_to_persons(tmp_path: Path):
    """Companies, vessels and addresses share the export and are not screened here."""
    path = write_jsonl(
        tmp_path / "e.json",
        [
            ftm_person("q1", name=["Tariq"]),
            {"id": "q2", "schema": "Company", "properties": {"name": ["Acme LLC"]}},
            {"id": "q3", "schema": "Vessel", "properties": {"name": ["MV Example"]}},
        ],
    )
    assert [record.record_id for record in stream_ftm_persons(path)] == ["q1"]


def test_ftm_collects_aliases_from_every_name_property(tmp_path: Path):
    path = write_jsonl(
        tmp_path / "e.json",
        [
            ftm_person(
                "q1",
                name=["طارق الهاشمي"],
                alias=["Tariq Al-Hashimi"],
                weakAlias=["T. Hashimi"],
                previousName=["Tarek Alhashimi"],
            )
        ],
    )
    (record,) = stream_ftm_persons(path)
    assert record.name == "طارق الهاشمي"
    assert record.aliases == ("Tariq Al-Hashimi", "T. Hashimi", "Tarek Alhashimi")


def test_ftm_falls_back_to_caption_when_there_is_no_name_property(tmp_path: Path):
    path = write_jsonl(
        tmp_path / "e.json",
        [{"id": "q1", "schema": "Person", "caption": "Tariq", "properties": {}}],
    )
    (record,) = stream_ftm_persons(path)
    assert record.name == "Tariq"


def test_ftm_skips_persons_with_no_usable_name(tmp_path: Path):
    path = write_jsonl(
        tmp_path / "e.json",
        [{"id": "q1", "schema": "Person", "properties": {"birthDate": ["1965"]}}],
    )
    assert list(stream_ftm_persons(path)) == []


def test_ftm_skips_entities_with_no_id(tmp_path: Path):
    path = write_jsonl(tmp_path / "e.json", [{"schema": "Person", "properties": {"name": ["X"]}}])
    assert list(stream_ftm_persons(path)) == []


def test_ftm_merges_country_and_nationality_uppercased(tmp_path: Path):
    path = write_jsonl(
        tmp_path / "e.json",
        [ftm_person("q1", name=["Tariq"], country=["iq"], nationality=["iq", "jo"])],
    )
    (record,) = stream_ftm_persons(path)
    assert record.countries == ("IQ", "JO")


def test_ftm_takes_the_first_birth_date(tmp_path: Path):
    path = write_jsonl(
        tmp_path / "e.json",
        [ftm_person("q1", name=["Tariq"], birthDate=["1965-03-12", "1965"])],
    )
    (record,) = stream_ftm_persons(path)
    assert record.birth_date == "1965-03-12"


def test_ftm_records_are_marked_with_their_source(tmp_path: Path):
    path = write_jsonl(tmp_path / "e.json", [ftm_person("q1", name=["Tariq"])])
    (record,) = stream_ftm_persons(path)
    assert record.source == "opensanctions"


def test_ftm_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "e.json"
    path.write_text(
        json.dumps(ftm_person("q1", name=["Tariq"])) + "\n\n\n",
        encoding="utf-8",
    )
    assert len(list(stream_ftm_persons(path))) == 1


def test_ftm_malformed_json_names_the_line(tmp_path: Path):
    """A truncated download must fail the build, not screen against a partial list."""
    path = tmp_path / "e.json"
    path.write_text(
        json.dumps(ftm_person("q1", name=["Tariq"])) + "\n{ this is not json\n",
        encoding="utf-8",
    )
    with pytest.raises(IngestError, match=r":2: malformed JSON"):
        list(stream_ftm_persons(path))


def test_ftm_rejects_a_json_array_line(tmp_path: Path):
    path = tmp_path / "e.json"
    path.write_text('["not", "an", "entity"]\n', encoding="utf-8")
    with pytest.raises(IngestError, match="expected a JSON object"):
        list(stream_ftm_persons(path))


def test_ftm_missing_file_raises(tmp_path: Path):
    with pytest.raises(IngestError, match="not found"):
        list(stream_ftm_persons(tmp_path / "nope.json"))


def test_ftm_reader_is_lazy(tmp_path: Path):
    """The bulk export does not belong in memory.

    Proven by putting malformed JSON on line 2: a reader that materialised the
    file would raise on the first next(), and this asserts it does not.
    """
    path = tmp_path / "e.json"
    path.write_text(
        json.dumps(ftm_person("q1", name=["Tariq"])) + "\n{ broken\n",
        encoding="utf-8",
    )
    stream = stream_ftm_persons(path)
    assert next(stream).record_id == "q1"
    with pytest.raises(IngestError):
        next(stream)


# --- Dispatch ----------------------------------------------------------------


def test_load_watchlist_dispatches_to_the_seed_reader(seed_csv: Path):
    assert len(load_watchlist(seed_csv)) == 6


@pytest.mark.parametrize("filename", ["entities.ftm.json", "entities.jsonl", "entities.ndjson"])
def test_load_watchlist_dispatches_to_the_ftm_reader(tmp_path: Path, filename: str):
    """OpenSanctions names its JSON-lines export .json, so .json must not mean json.load."""
    path = write_jsonl(tmp_path / filename, [ftm_person("q1", name=["Tariq"])])
    assert [record.record_id for record in load_watchlist(path)] == ["q1"]


def test_load_watchlist_rejects_an_unknown_format(tmp_path: Path):
    path = tmp_path / "watchlist.xlsx"
    path.write_text("", encoding="utf-8")
    with pytest.raises(IngestError, match="unsupported watchlist format"):
        load_watchlist(path)
