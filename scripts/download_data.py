"""Fetch the OpenSanctions PEP dataset into ``data/``.

Not committed to git: the bulk export is large, it goes stale, and redistributing
it is a licensing question this repo does not need to have. The committed seed
file is what makes a clean clone runnable; this script is what makes it real.

Source:  https://www.opensanctions.org/datasets/peps/
Format:  FollowTheMoney, JSON lines
Licence: CC-BY-NC 4.0 — free for non-commercial use, attribution required.
         A portfolio repository is non-commercial. The attribution belongs in
         the README, not only here.

Streams to disk in chunks; the file is not held in memory at any point.
"""

from __future__ import annotations

FTM_PEPS_URL = "https://data.opensanctions.org/datasets/latest/peps/entities.ftm.json"


def download(url: str = FTM_PEPS_URL, *, force: bool = False) -> None:
    """Stream the dataset to ``data/``, skipping the fetch if it is already present."""
    raise NotImplementedError


if __name__ == "__main__":
    download()
