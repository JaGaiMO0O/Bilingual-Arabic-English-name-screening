# Name screening across Arabic and English script

Screens a person's name against a sanctions and politically-exposed-persons watchlist,
matching **across writing systems**: query `طارق الهاشمي` and it finds `Tariq Al-Hashimi`,
and the reverse.

Exact matching fails immediately on this problem — the same person is written `Mohammed`,
`Muhammad`, `Mohamad`, `محمد` and `مُحَمَّد` — and transliteration tables are brittle. This
uses script-aware normalisation followed by multilingual embeddings and vector retrieval,
and measures the result against a labelled evaluation set rather than asserting it works.

> **Status: skeleton.** Module interfaces and configuration are in place; the
> implementations are not. Nothing below marked TODO is true yet. This banner comes out
> when the numbers go in.

---

## Results

**TODO (build step 9).** This section is the deliverable. It must carry:

- Precision, recall, F1 and the confusion counts, at the chosen threshold.
- The size of the evaluation set and **how many of those cases are hard negatives**.
- The same breakdown per category — cross-script recall and hard-negative precision are
  the two figures that actually say whether this works.

An aggregate F1 with no hard-negative count underneath it is not a result, and a reviewer
who has shipped something will read it as one.

## Choice of threshold

**TODO (build step 7).** Which operating point was taken off the precision/recall curve,
and why that one. A false negative (a sanctioned party passes screening) and a false
positive (an innocent customer is frozen) do not cost the same, and the threshold is where
that asymmetry gets decided. Showing the reasoning is worth more than the score.

## Known limitations

**TODO (build step 9).** Concrete, not hedged. At minimum: which name types fail, what
happens to very short names and single-token names, and what a purely semantic approach
cannot do that a deterministic rules layer would catch.

---

## How it works

```
query ──► normalise (script-aware) ──► embed ("query: ") ──► FAISS inner product
                                                                    │
watchlist ──► normalise ──► variants ──► embed ("passage: ") ──► index
                                                                    │
                                                          threshold ──► decision
```

Two details that are easy to get wrong and that fail silently:

- **e5 models require prefixes.** Watchlist entries are encoded `passage: <name>`, queries
  `query: <name>`. Omitting them costs real accuracy and raises nothing. Both live in
  `config.py` so the two paths cannot drift.
- **Normalise the vectors, then use inner product.** `faiss.normalize_L2` plus
  `IndexFlatIP` gives cosine similarity, which is what the configured threshold assumes.
  `IndexFlatL2` without normalising returns squared distances — smaller is better and the
  range is unbounded, so every threshold comparison silently inverts.

`normalize.py` is the substantive module: alef and hamza folding, ta marbuta, alef maqsura,
harakat and tatweel stripping, the `ال` definite article, and Latin-side casefolding and
punctuation handling — each rule with a unit test carrying a real example pair.

## Quickstart

**TODO (build step 9)** — verify every line of this from a clean clone before shipping it.

```bash
pip install -e ".[dev]"
name-screening build                          # embed the seed watchlist, write the index
name-screening screen --name "طارق الهاشمي"
name-screening evaluate                       # P/R/F1 against data/eval_cases.csv
```

Tests run with no network, no model download and no dataset:

```bash
pytest
```

Requires Python 3.11+.

## Data and attribution

Watchlist data comes from [OpenSanctions](https://www.opensanctions.org/datasets/peps/),
in FollowTheMoney format, licensed **CC-BY-NC 4.0** — free for non-commercial use with
attribution.

The bulk dataset is not committed: it is large, it goes stale, and redistributing it is a
licensing question this repo does not need to have. `scripts/download_data.py` fetches it.

`data/watchlist_seed.csv` is committed so a clean clone runs immediately. **Its entries are
invented.** They are not real people and the file asserts nothing about anyone; they exist
to give normalisation bilingual pairs to exercise.

## Licence

MIT — see [LICENSE](LICENSE).
