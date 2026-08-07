"""Import smoke tests.

These exist to protect a property that is easy to lose: importing the package must
not pull in torch, sentence-transformers or faiss. If it does, the offline test run
gets slow and eventually needs a model download to collect at all.
"""

from __future__ import annotations

import subprocess
import sys


def test_package_imports():
    import name_screening

    assert name_screening.__version__


def test_normalize_has_no_heavy_dependencies():
    """normalize.py is stdlib-only by design — it is the module tested offline."""
    code = (
        "import sys, name_screening.normalize;"
        "heavy = {'torch', 'sentence_transformers', 'faiss'} & set(sys.modules);"
        "assert not heavy, heavy"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
