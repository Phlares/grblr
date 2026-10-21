"""Where a clip comes from: its source id, content category and language hint.

The engine stores only neutral ids (`src_` + 8 hex digits). Real names, and the
folder layouts that reveal them, live in an optional private JSON file named by
GRBLR_SOURCES:

    {"source":   [[regex, template], ...],   # template may use \\1 groups
     "category": [[regex, value], ...],
     "language": [[regex, value], ...],
     "names":    {"src_1234abcd": "display name"}}

Each list is ordered; the first matching regex wins. Without the file every
clip resolves to (no source, "voice", no language hint).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CATEGORY = "voice"


def source_id(name: str | None) -> str | None:
    """Neutral id for a source name. The v23 backfill uses the same function."""
    if name is None:
        return None
    return "src_" + hashlib.sha1(name.encode()).hexdigest()[:8]


@dataclass(frozen=True)
class ManifestRow:
    """One manifest v0 row: an audio file and what the engine stores about it."""

    audio_key: str
    source_id: str | None = None
    category: str = DEFAULT_CATEGORY
    lang_hint: str | None = None


def _first(rules: tuple[tuple[re.Pattern, str], ...], path: str) -> str | None:
    for pattern, template in rules:
        m = pattern.search(path)
        if m:
            return m.expand(template)
    return None


def _compile(pairs) -> tuple[tuple[re.Pattern, str], ...]:
    return tuple((re.compile(p), t) for p, t in pairs)


@dataclass(frozen=True)
class PathRules:
    source: tuple[tuple[re.Pattern, str], ...] = ()
    category: tuple[tuple[re.Pattern, str], ...] = ()
    language: tuple[tuple[re.Pattern, str], ...] = ()
    names: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> PathRules:
        return cls(
            source=_compile(d.get("source", ())),
            category=_compile(d.get("category", ())),
            language=_compile(d.get("language", ())),
            names=dict(d.get("names", {})),
        )

    def resolve(self, path: str) -> ManifestRow:
        return ManifestRow(
            path,
            source_id(_first(self.source, path)),
            _first(self.category, path) or DEFAULT_CATEGORY,
            _first(self.language, path),
        )


NO_RULES = PathRules()


def load() -> PathRules:
    f = os.environ.get("GRBLR_SOURCES")
    if not f:
        return NO_RULES
    return PathRules.from_dict(json.loads(Path(f).read_text(encoding="utf-8")))


def read_manifest(path: Path | str) -> list[ManifestRow]:
    """Manifest v0: JSON Lines, one ManifestRow per line. Blank lines skip."""
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        d = json.loads(line)
        try:
            rows.append(ManifestRow(**{k: v for k, v in d.items() if v is not None}))
        except TypeError as e:
            raise ValueError(f"{path}:{n}: {e}") from None
    return rows
