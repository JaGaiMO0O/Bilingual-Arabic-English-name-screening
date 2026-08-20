"""Tests for script-aware name normalisation — one per rule, with a real example.

Combining marks and invisible characters are written as ``\\u`` escapes rather than
literals. A fatha or a zero-width joiner is invisible in source, so a literal would
leave the reader unable to tell what the test actually asserts.
"""

from __future__ import annotations

import pytest

from name_screening.normalize import (
    Script,
    detect_script,
    fold_alef,
    fold_alef_maqsura,
    fold_hamza,
    fold_persian_forms,
    fold_ta_marbuta,
    normalization_variants,
    normalize_arabic,
    normalize_latin,
    normalize_name,
    strip_definite_article,
    strip_diacritics,
    strip_invisible,
    strip_tatweel,
)

# محمد — the bare, unvocalised form every rule below should converge on.
MUHAMMAD = "محمد"


# --- Script detection --------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("طارق الهاشمي", Script.ARABIC),
        ("Tariq Al-Hashimi", Script.LATIN),
        ("طارق (Tariq)", Script.MIXED),
        ("", Script.UNKNOWN),
        ("1965-03-12", Script.UNKNOWN),
        ("   ", Script.UNKNOWN),
        ("محمد 1965", Script.ARABIC),
    ],
)
def test_detect_script(text: str, expected: Script):
    assert detect_script(text) is expected


def test_diacritics_do_not_vote_on_script():
    """Harakat are combining marks, not letters — a vocalised name is still Arabic."""
    vocalised = "مُحَمَّد"  # مُحَمَّد
    assert detect_script(vocalised) is Script.ARABIC


# --- Individual rules --------------------------------------------------------


def test_strip_diacritics_removes_harakat():
    """مُحَمَّد -> محمد. Damma, fatha and shadda carry pronunciation, not identity."""
    vocalised = "مُحَمَّد"
    assert strip_diacritics(vocalised) == MUHAMMAD


def test_strip_diacritics_removes_superscript_alef():
    """U+0670 is a mark, not a letter, and is invisible in most fonts."""
    assert strip_diacritics("رحمٰن") == "رحمن"


def test_strip_tatweel_removes_kashida():
    """مـــحمد -> محمد. Tatweel is justification, pure typography."""
    stretched = "م" + "ـ" * 3 + "حمد"
    assert strip_tatweel(stretched) == MUHAMMAD


@pytest.mark.parametrize("variant", ["أحمد", "إحمد", "آحمد", "ٱحمد"])
def test_fold_alef_collapses_every_carrier(variant: str):
    """أ إ آ ٱ -> ا"""
    assert fold_alef(variant) == "احمد"


def test_fold_ta_marbuta():
    """فاطمة -> فاطمه. Final ta marbuta is written both ways in the same document."""
    assert fold_ta_marbuta("فاطمة") == "فاطمه"


def test_fold_alef_maqsura():
    """مصطفى -> مصطفي. Dotless final yeh is a regional habit, not a different name."""
    assert fold_alef_maqsura("مصطفى") == "مصطفي"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("مؤمن", "مومن"),  # ؤ -> و
        ("رئيس", "رييس"),  # ئ -> ي
        ("ماء", "ما"),  # bare ء dropped entirely
    ],
)
def test_fold_hamza(text: str, expected: str):
    assert fold_hamza(text) == expected


def test_fold_persian_forms():
    """Farsi yeh and keheh are visually near-identical to their Arabic counterparts."""
    assert fold_persian_forms("علی") == "علي"  # علی -> علي
    assert fold_persian_forms("کمال") == "كمال"  # کمال


def test_strip_invisible_removes_zero_width_joiner():
    """A ZWJ survives copy-paste, renders as nothing, and defeats exact comparison."""
    contaminated = "مح‍مد"
    assert contaminated != MUHAMMAD
    assert strip_invisible(contaminated) == MUHAMMAD


def test_strip_invisible_removes_bidi_and_bom():
    assert strip_invisible("﻿‫Tariq‬") == "Tariq"


# --- Definite article --------------------------------------------------------


def test_strip_definite_article_arabic():
    """الهاشمي -> هاشمي"""
    assert strip_definite_article("الهاشمي") == "هاشمي"


def test_strip_definite_article_guards_short_tokens():
    """الف must survive: those letters are the word, not an article.

    This is the guard that stops article stripping from being destructive.
    """
    assert strip_definite_article("الف") == "الف"


def test_strip_definite_article_latin_whole_token_only():
    assert strip_definite_article("al hashimi") == "hashimi"
    assert strip_definite_article("el sayed") == "sayed"


def test_strip_definite_article_does_not_eat_ali():
    """The reason the Latin rule matches whole tokens: a prefix rule turns Ali into i."""
    assert strip_definite_article("ali hassan") == "ali hassan"


def test_strip_definite_article_keeps_a_lone_article():
    assert strip_definite_article("al") == "al"


def test_strip_definite_article_on_empty_input():
    assert strip_definite_article("") == ""


# --- Chains ------------------------------------------------------------------


def test_normalize_arabic_full_chain():
    """Vocalised, stretched and hamza-carrying input reduces to the bare form."""
    messy = "مُحـَمَّد"  # مُحـَمَّد with tatweel
    assert normalize_arabic(messy) == MUHAMMAD


def test_normalize_arabic_collapses_a_real_pair():
    """The seed file carries both of these spellings of the same person."""
    assert normalize_arabic("طارق الهاشمى") == normalize_arabic("طارق الهاشمي")


def test_normalize_arabic_handles_presentation_forms():
    """PDF extraction yields U+FB50-U+FEFF shapes; NFKC must run before folding."""
    assert normalize_arabic("ﻟﺍ") == normalize_arabic("لا")


@pytest.mark.parametrize("spelling", ["Al-Hashimi", "AL-HASHIMI", "al  hashimi", "Al Hashimi"])
def test_normalize_latin_converges_on_spacing_and_case(spelling: str):
    assert normalize_latin(spelling) == "al hashimi"


def test_normalize_latin_folds_accents():
    """Béchir -> bechir. Accents are lost in transliteration constantly."""
    assert normalize_latin("Béchir") == "bechir"


def test_normalize_latin_folds_letters_nfkd_cannot_reach():
    """ø and ł have no canonical decomposition, so they need an explicit map."""
    assert normalize_latin("Jørgen") == "jorgen"
    assert normalize_latin("Wałęsa") == "walesa"


def test_normalize_latin_strips_punctuation():
    assert normalize_latin("O'Brien, Sean.") == "o brien sean"


# --- Entry point -------------------------------------------------------------


def test_normalize_name_dispatches_by_script():
    assert normalize_name("Al-Hashimi") == "al hashimi"
    assert normalize_name("الهاشمي") == "الهاشمي"


def test_normalize_name_handles_mixed_script():
    """Both chains run; each only touches characters of its own script."""
    assert normalize_name("طارق (Tariq)") == "طارق tariq"


@pytest.mark.parametrize("blank", ["", "   ", "​", "\t\n"])
def test_normalize_name_on_blank_input(blank: str):
    assert normalize_name(blank) == ""


def test_normalize_name_does_not_strip_the_article():
    """Stripping here would be destructive; the stripped form is a variant instead."""
    assert normalize_name("الهاشمي") == "الهاشمي"


@pytest.mark.parametrize(
    "text",
    ["طارق الهاشمي", "Al-Hashimi", "مُحَمَّد", "طارق (Tariq)", "O'Brien", ""],
)
def test_normalize_name_is_idempotent(text: str):
    """Normalising twice must equal normalising once.

    Worth asserting because ingest and query paths both call this, and a
    non-idempotent step would make an indexed name unreachable by its own text.
    """
    once = normalize_name(text)
    assert normalize_name(once) == once


# --- Variants ----------------------------------------------------------------


def test_variants_include_article_stripped_form():
    variants = normalization_variants("الهاشمي")
    assert "الهاشمي" in variants
    assert "هاشمي" in variants


def test_variants_are_deduped_and_order_stable():
    variants = normalization_variants("Tariq")
    assert variants == ["tariq"]


def test_variants_on_blank_input():
    assert normalization_variants("  ") == []


def test_the_three_latin_spellings_meet_on_a_common_variant():
    """The reason the compact variant exists.

    Al-Hashimi / al hashimi normalise to "al hashimi"; AlHashimi normalises to
    "alhashimi". Neither depunctuation strategy alone unifies all three, so the
    whitespace-free form is emitted as a variant and they meet there.
    """
    spellings = ["Al-Hashimi", "AlHashimi", "al hashimi"]
    shared = set.intersection(*(set(normalization_variants(s)) for s in spellings))
    assert "alhashimi" in shared


def test_variants_stay_small():
    """Each extra form is another chance at a false positive."""
    assert len(normalization_variants("طارق الهاشمي")) <= 3
    assert len(normalization_variants("Tariq Al-Hashimi")) <= 3


def test_variants_are_themselves_normalised():
    """A variant must be reachable by a query that normalises to it."""
    for variant in normalization_variants("Tariq Al-Hashimi"):
        assert normalize_name(variant) == variant
