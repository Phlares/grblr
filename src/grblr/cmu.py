"""The CMU Pronouncing Dictionary, read from the copy shipped in data/.

The dictionary is BSD-licensed (data/LICENSE.cmudict); reading it directly
keeps the GPL `cmudict` wrapper package out of the dependency tree."""

from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources

_VARIANT = re.compile(r"\(\d+\)$")


@lru_cache(maxsize=1)
def load() -> dict[str, list[list[str]]]:
    """word (lowercase) -> its pronunciations, each a list of ARPAbet phones."""
    out: dict[str, list[list[str]]] = {}
    text = (
        resources.files("grblr")
        .joinpath("data", "cmudict.dict")
        .read_text(encoding="utf-8")
    )
    for line in text.splitlines():
        parts = line.split("#", 1)[0].split()
        if parts:
            out.setdefault(_VARIANT.sub("", parts[0]), []).append(parts[1:])
    return out
