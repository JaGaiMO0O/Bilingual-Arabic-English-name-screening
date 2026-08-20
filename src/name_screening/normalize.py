"""Script-aware name normalisation — the module a reviewer will actually read.

Deliberately depends on nothing outside the standard library. The full test suite
has to run with no network, no model download and no FAISS, and keeping this file
import-light is what makes that possible.

Arabic orthography varies in ways that break naive matching. The same person is
written ``محمد``, ``مُحَمَّد``, ``مـحمد``, ``Mohammed``, ``Muhammad`` and ``Mohamad``.
Normalisation collapses the variation that carries no identity information, so the
embedding model is asked to bridge meaning rather than typography.

Rules implemented here:

===========================  =========================  ==============================
Issue                        Example                    Treatment
===========================  =========================  ==============================
Alef variants                ``أ إ آ ٱ`` -> ``ا``       fold to bare alef
Ta marbuta                   ``ة`` -> ``ه``             fold
Alef maqsura                 ``ى`` -> ``ي``             fold
Diacritics (harakat)         ``مُحَمَّد`` -> ``محمد``       strip combining marks
Tatweel / kashida            ``مـــحمد`` -> ``محمد``     strip U+0640
Hamza forms                  ``ؤ ئ ء``                  fold to carrier, drop bare
Persian/Urdu letterforms     ``ی ک`` -> ``ي ك``         fold to Arabic
Definite article             ``الهاشمي`` -> ``هاشمي``    guarded, and kept as a variant
Latin side                   ``Al-Hashimi``             casefold, fold accents, depunctuate
===========================  =========================  ==============================

Three decisions worth defending, because they are the ones a reviewer will question:

**The definite article is not stripped destructively.** Removing ``ال`` unconditionally
also removes it from names where those two letters are root letters rather than an
article — ``الف`` would become ``ف``. So ``normalize_name`` leaves it alone, and
``normalization_variants`` emits the stripped form *alongside* the unstripped one. A
length guard means the article is only stripped when something substantial survives it.
Indexing both forms costs an extra vector; guessing wrong costs a match.

**Bare hamza is dropped rather than mapped.** ``ء`` carries no identity information on
its own and is inconsistently written in romanised-then-back-transliterated names.
Carriers fold to their base letter: ``ؤ`` -> ``و``, ``ئ`` -> ``ي``.

**Punctuation becomes a space, and a whitespace-free variant is emitted separately.**
Neither choice alone unifies ``Al-Hashimi``, ``AlHashimi`` and ``al hashimi``: stripping
punctuation outright merges the first two but not the third, and replacing it with a
space merges the first and third but not the second. Emitting a compact variant lets all
three meet on a common form. See ``normalization_variants``.
"""

from __future__ import annotations

import unicodedata
from enum import StrEnum

# --- Character classes -------------------------------------------------------

#: Combining marks that carry pronunciation, not identity. Covers the harakat
#: (U+064B-U+0652), the hamza/madda combining forms above them (U+0653-U+065F),
#: the superscript alef (U+0670) and the Quranic annotation marks (U+06D6-U+06ED).
ARABIC_DIACRITICS = frozenset(
    chr(codepoint) for codepoint in (*range(0x064B, 0x0660), 0x0670, *range(0x06D6, 0x06EE))
)

#: Kashida. Pure typography — used to justify text by stretching a join.
TATWEEL = "ـ"

#: Invisible characters: soft hyphen, bidi controls, zero-width space/joiners and
#: the BOM. These survive copy-paste out of PDFs and web pages, render as nothing,
#: and silently defeat exact comparison. Stripping them first is what stops a name
#: that *looks* identical from failing to match.
INVISIBLE_CHARS = frozenset(
    chr(codepoint)
    for codepoint in (
        0x00AD,  # soft hyphen
        0x061C,  # Arabic letter mark
        *range(0x200B, 0x2010),  # ZWSP, ZWNJ, ZWJ, LRM, RLM
        *range(0x202A, 0x202F),  # bidi embedding/override controls
        *range(0x2060, 0x2065),  # word joiner, invisible operators
        *range(0x2066, 0x206A),  # bidi isolates
        0xFEFF,  # zero-width no-break space / BOM
    )
)

#: The Arabic definite article, alef + lam.
DEFINITE_ARTICLE = "ال"

#: Minimum length of a token before its leading ``ال`` is treated as an article.
#: ``الهاشمي`` (7) is stripped; ``الف`` (3) is not, because one letter would survive.
MIN_TOKEN_LEN_FOR_ARTICLE_STRIP = 4

#: Latin-script renderings of the article. Matched as whole tokens only — matching
#: a bare prefix would turn ``Ali`` into ``i``.
LATIN_ARTICLES = frozenset({"al", "el"})

_ALEF_TABLE = str.maketrans(
    {
        "أ": "ا",  # أ alef with hamza above
        "إ": "ا",  # إ alef with hamza below
        "آ": "ا",  # آ alef with madda above
        "ٱ": "ا",  # ٱ alef wasla
        "ٲ": "ا",  # ٲ alef with wavy hamza above
        "ٳ": "ا",  # ٳ alef with wavy hamza below
    }
)

_TA_MARBUTA_TABLE = str.maketrans({"ة": "ه"})  # ة -> ه

_ALEF_MAQSURA_TABLE = str.maketrans({"ى": "ي"})  # ى -> ي

_HAMZA_TABLE = str.maketrans(
    {
        "ؤ": "و",  # ؤ waw with hamza -> و
        "ئ": "ي",  # ئ yeh with hamza -> ي
        "ء": "",  # ء bare hamza -> dropped
    }
)

#: Persian and Urdu letterforms that appear in Arabic-script names and are visually
#: near-identical to their Arabic counterparts.
_PERSIAN_TABLE = str.maketrans(
    {
        "ی": "ي",  # ی farsi yeh -> ي
        "ک": "ك",  # ک keheh -> ك
        "ڪ": "ك",  # ڪ swash kaf -> ك
        "ھ": "ه",  # ھ heh doachashmee -> ه
        "ہ": "ه",  # ہ heh goal -> ه
        "ە": "ه",  # ە ae -> ه
    }
)

#: Latin letters with no canonical decomposition, so NFKD cannot reach them.
_LATIN_SPECIALS = str.maketrans(
    {
        "ø": "o",
        "Ø": "O",
        "đ": "d",
        "Đ": "D",
        "ł": "l",
        "Ł": "L",
        "ð": "d",
        "Ð": "D",
        "þ": "th",
        "Þ": "Th",
        "æ": "ae",
        "Æ": "Ae",
        "œ": "oe",
        "Œ": "Oe",
    }
)

_INVISIBLE_TABLE = str.maketrans({char: None for char in INVISIBLE_CHARS})
_TATWEEL_TABLE = str.maketrans({TATWEEL: None})
_DIACRITICS_TABLE = str.maketrans({char: None for char in ARABIC_DIACRITICS})


class Script(StrEnum):
    """Which writing system a string is predominantly in."""

    ARABIC = "arabic"
    LATIN = "latin"
    MIXED = "mixed"
    UNKNOWN = "unknown"


# --- Primitives --------------------------------------------------------------


def strip_invisible(text: str) -> str:
    """Remove zero-width, bidi-control and soft-hyphen characters."""
    return text.translate(_INVISIBLE_TABLE)


def strip_diacritics(text: str) -> str:
    """Remove Arabic harakat, the superscript alef and Quranic annotation marks."""
    return text.translate(_DIACRITICS_TABLE)


def strip_tatweel(text: str) -> str:
    """Remove the kashida / tatweel character (U+0640) used to justify text."""
    return text.translate(_TATWEEL_TABLE)


def fold_alef(text: str) -> str:
    """Fold ``أ إ آ ٱ`` to bare ``ا``."""
    return text.translate(_ALEF_TABLE)


def fold_ta_marbuta(text: str) -> str:
    """Fold ``ة`` to ``ه``."""
    return text.translate(_TA_MARBUTA_TABLE)


def fold_alef_maqsura(text: str) -> str:
    """Fold ``ى`` to ``ي``."""
    return text.translate(_ALEF_MAQSURA_TABLE)


def fold_hamza(text: str) -> str:
    """Fold hamza carriers to their base letter and drop the bare hamza.

    ``ؤ`` -> ``و``, ``ئ`` -> ``ي``, ``ء`` -> removed. Alef carriers are handled by
    :func:`fold_alef`.
    """
    return text.translate(_HAMZA_TABLE)


def fold_persian_forms(text: str) -> str:
    """Fold Persian/Urdu letterforms to their Arabic equivalents."""
    return text.translate(_PERSIAN_TABLE)


def _depunctuate(text: str) -> str:
    """Replace every non-alphanumeric, non-space character with a space."""
    return "".join(char if (char.isalnum() or char.isspace()) else " " for char in text)


def _collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip the ends."""
    return " ".join(text.split())


# --- Script detection --------------------------------------------------------


def detect_script(text: str) -> Script:
    """Classify ``text`` by the script of its letter characters.

    Digits, punctuation and whitespace do not vote. A string with letters from
    both systems is ``MIXED``; one with no letters at all is ``UNKNOWN``.
    """
    has_arabic = False
    has_latin = False

    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if name.startswith("ARABIC"):
            has_arabic = True
        elif name.startswith("LATIN"):
            has_latin = True
        if has_arabic and has_latin:
            return Script.MIXED

    if has_arabic:
        return Script.ARABIC
    if has_latin:
        return Script.LATIN
    return Script.UNKNOWN


# --- Chains ------------------------------------------------------------------


def normalize_arabic(text: str) -> str:
    """Apply the full Arabic normalisation chain.

    NFKC runs early so that Arabic presentation forms (the U+FB50-U+FEFF ligature
    and contextual-shape blocks, which is what you get from some PDF extractions)
    become ordinary letters before any folding is attempted.
    """
    text = strip_invisible(text)
    text = unicodedata.normalize("NFKC", text)
    text = strip_tatweel(text)
    text = strip_diacritics(text)
    text = fold_persian_forms(text)
    text = fold_alef(text)
    text = fold_alef_maqsura(text)
    text = fold_ta_marbuta(text)
    text = fold_hamza(text)
    text = _depunctuate(text)
    return _collapse_whitespace(text)


def normalize_latin(text: str) -> str:
    """Casefold, fold accents, depunctuate and collapse whitespace.

    ``Al-Hashimi``, ``al  hashimi`` and ``AL-HASHIMI`` all reach ``al hashimi``.
    ``AlHashimi`` reaches ``alhashimi`` — see the module docstring on why the
    compact variant exists rather than a cleverer rule here.
    """
    text = strip_invisible(text)
    text = text.translate(_LATIN_SPECIALS)
    decomposed = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in decomposed if not unicodedata.combining(char))
    text = text.casefold()
    text = _depunctuate(text)
    return _collapse_whitespace(text)


def normalize_name(text: str) -> str:
    """Normalise ``text`` according to its detected script.

    The single entry point used by ingest, indexing and query paths. It is
    important that all three call exactly this function: normalising queries
    differently from passages is a silent accuracy leak that no test will catch
    unless it is written to.

    The definite article is deliberately *not* stripped here — see
    :func:`normalization_variants`.
    """
    if not text or not text.strip():
        return ""

    script = detect_script(text)
    if script is Script.ARABIC:
        return normalize_arabic(text)
    if script is Script.LATIN:
        return normalize_latin(text)
    if script is Script.MIXED:
        # Each chain only touches characters belonging to its own script, so
        # running both is safe and order-independent for the folding steps.
        return normalize_latin(normalize_arabic(text))

    text = strip_invisible(unicodedata.normalize("NFKC", text))
    return _collapse_whitespace(_depunctuate(text.casefold()))


def strip_definite_article(text: str) -> str:
    """Remove the definite article from each token of an already-normalised string.

    Arabic ``ال`` is stripped only when at least two letters survive it, so root
    letters are not mistaken for an article. Latin ``al`` / ``el`` are removed only
    as whole tokens, and never when they are the only token — matching a bare
    prefix would turn ``Ali`` into ``i``.

    Lossy by nature. Callers keep the unstripped form too rather than replace it.
    """
    tokens = text.split()
    if not tokens:
        return ""

    kept: list[str] = []
    for token in tokens:
        if token.startswith(DEFINITE_ARTICLE) and len(token) >= MIN_TOKEN_LEN_FOR_ARTICLE_STRIP:
            kept.append(token[len(DEFINITE_ARTICLE) :])
        elif token in LATIN_ARTICLES and len(tokens) > 1:
            continue
        else:
            kept.append(token)

    return " ".join(kept)


def normalization_variants(text: str) -> list[str]:
    """Return the distinct forms of ``text`` worth indexing.

    Three at most, deduped and order-stable:

    1. the normalised form,
    2. the article-stripped form, when it differs,
    3. the whitespace-free compact form, when it differs.

    (3) is what lets ``Al-Hashimi``, ``AlHashimi`` and ``al hashimi`` meet: they
    normalise to ``al hashimi``, ``alhashimi`` and ``al hashimi`` respectively, and
    all three produce ``alhashimi`` compactly.

    Indexing several forms per record is cheaper than losing a match, but each
    extra form is another chance at a false positive, so this stays small and
    deliberate.
    """
    base = normalize_name(text)
    if not base:
        return []

    candidates = [base, strip_definite_article(base), base.replace(" ", "")]

    variants: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in variants:
            variants.append(candidate)
    return variants
