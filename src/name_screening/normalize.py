"""Script-aware name normalisation — the module a reviewer will actually read.

Deliberately depends on nothing outside the standard library. The full test suite
has to run with no network, no model download and no FAISS, and keeping this file
import-light is what makes that possible.

Arabic orthography varies in ways that break naive matching. The same person is
written ``محمد``, ``مُحَمَّد``, ``مـحمد``, ``Mohammed``, ``Muhammad`` and ``Mohamad``.
Normalisation collapses the variation that carries no identity information, so the
embedding model is asked to bridge meaning rather than typography.

Rules to implement (build step 2), each with a unit test carrying a real example:

===========================  =========================  ==============================
Issue                        Example                    Treatment
===========================  =========================  ==============================
Alef variants                ``أ إ آ ٱ`` -> ``ا``       fold to bare alef
Ta marbuta                   ``ة`` -> ``ه``             fold, or strip when final
Alef maqsura                 ``ى`` -> ``ي``             fold
Diacritics (harakat)         ``مُحَمَّد`` -> ``محمد``       strip U+064B-U+0652
Tatweel / kashida            ``مـــحمد`` -> ``محمد``     strip U+0640
Hamza forms                  ``ؤ ئ ء``                  normalise consistently
Definite article             ``الهاشمي`` / ``هاشمي``     handle the ``ال`` prefix
Latin side                   ``Al-Hashimi``, ``al hashimi``  casefold, strip punctuation
===========================  =========================  ==============================

The definite article is the one judgement call in the table. Stripping ``ال``
unconditionally also strips it from names where those two letters are root
letters rather than an article, so the intended approach is to index both the
stripped and unstripped forms rather than to destroy information. Whichever way
it goes, the README should say which and why.
"""

from __future__ import annotations

from enum import StrEnum


class Script(StrEnum):
    """Which writing system a string is predominantly in."""

    ARABIC = "arabic"
    LATIN = "latin"
    MIXED = "mixed"
    UNKNOWN = "unknown"


def detect_script(text: str) -> Script:
    """Classify ``text`` by the script of its letter characters.

    Digits, punctuation and whitespace do not vote. A string with letters from
    both systems is ``MIXED``; one with no letters at all is ``UNKNOWN``.
    """
    raise NotImplementedError


def strip_diacritics(text: str) -> str:
    """Remove Arabic harakat (U+064B-U+0652) and the superscript alef (U+0670)."""
    raise NotImplementedError


def strip_tatweel(text: str) -> str:
    """Remove the kashida / tatweel character (U+0640) used to justify text."""
    raise NotImplementedError


def fold_alef(text: str) -> str:
    """Fold ``أ إ آ ٱ`` to bare ``ا``."""
    raise NotImplementedError


def fold_ta_marbuta(text: str) -> str:
    """Fold ``ة`` to ``ه``."""
    raise NotImplementedError


def fold_alef_maqsura(text: str) -> str:
    """Fold ``ى`` to ``ي``."""
    raise NotImplementedError


def fold_hamza(text: str) -> str:
    """Normalise the hamza carriers ``ؤ ئ ء`` to a single consistent form."""
    raise NotImplementedError


def strip_definite_article(text: str) -> str:
    """Remove a leading ``ال`` from each token, and ``al-`` / ``el-`` on the Latin side.

    Lossy by nature — see the module docstring. Callers are expected to keep the
    unstripped form too rather than replace it.
    """
    raise NotImplementedError


def normalize_arabic(text: str) -> str:
    """Apply the full Arabic normalisation chain."""
    raise NotImplementedError


def normalize_latin(text: str) -> str:
    """Casefold, strip punctuation and diacritics, collapse whitespace.

    Handles ``Al-Hashimi``, ``AlHashimi`` and ``al hashimi`` landing on a common
    form, and folds Latin-1 accents (``Béchir`` -> ``bechir``).
    """
    raise NotImplementedError


def normalize_name(text: str) -> str:
    """Normalise ``text`` according to its detected script.

    The single entry point used by ingest, indexing and query paths. It is
    important that all three call exactly this function: normalising queries
    differently from passages is a silent accuracy leak that no test will catch
    unless it is written to.
    """
    raise NotImplementedError


def normalization_variants(text: str) -> list[str]:
    """Return the distinct forms of ``text`` worth indexing.

    At minimum the fully-normalised form and the article-stripped form, deduped
    and order-stable. Indexing several forms per record is cheaper than losing a
    match, but each extra form is another chance at a false positive, so this
    stays small and deliberate.
    """
    raise NotImplementedError
