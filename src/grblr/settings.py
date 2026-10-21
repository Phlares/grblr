"""Where the engine finds its corpus and keeps its files. Read from the
environment each call, so nothing machine-specific is baked into the code.

GRBLR_DB           corpus database (VC_DB is the older name, still honoured)
GRBLR_CORPUS_ROOT  root the corpus audio paths resolve against
GRBLR_DATA         working files: renders, presets, filter_models (default ~/.grblr)
"""

from __future__ import annotations

import os
from pathlib import Path


def db_path() -> Path | None:
    v = os.environ.get("GRBLR_DB") or os.environ.get("VC_DB")
    return Path(v) if v else None


def corpus_root() -> Path | None:
    v = os.environ.get("GRBLR_CORPUS_ROOT")
    return Path(v) if v else None


def data_dir(name: str) -> Path:
    return Path(os.environ.get("GRBLR_DATA") or Path.home() / ".grblr") / name
