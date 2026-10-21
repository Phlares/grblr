from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Literal

from grblr import phonetics
from grblr.normalize import normalize

Strategy = Literal["literal", "syllable_split", "miss"]


@dataclass
class SearchUnit:
    target: str
    strategy: Strategy
    parts: list[str]


@dataclass
class SearchPlan:
    sentence: str
    units: list[SearchUnit] = field(default_factory=list)
    log: list[str] = field(default_factory=list)


@dataclass
class DecomposerConfig:
    enable_syllable_split: bool = False  # v1.1
    weirdness: float = 0.0  # v1.2
    rng_seed: int | None = None


def _word_exists(db: sqlite3.Connection, text_lower: str) -> int:
    return db.execute(
        "SELECT COUNT(*) FROM words WHERE text_lower = ?", (text_lower,)
    ).fetchone()[0]


def _strip_stress_list(phs: list[str]) -> list[str]:
    return [p.rstrip("0123456789") for p in phs]


def _resolve_syllable(
    syllable: list[str],
    corpus: list[tuple[str, list[str]]],
) -> str | None:
    """Find the best corpus text_lower whose phonemes correspond to this syllable.

    Match order: exact stress-stripped equality → corpus phonemes start with
    syllable → syllable starts with corpus phonemes → containment → nearest by
    phoneme_distance (bounded to len(syllable)).
    """
    syl = _strip_stress_list(syllable)
    if not syl:
        return None

    best_exact: str | None = None
    best_prefix: tuple[int, str] | None = None
    best_contains: tuple[int, str] | None = None
    best_dist: tuple[int, int, str] | None = None  # (distance, len_diff, text)

    for text, raw_phs in corpus:
        cph = _strip_stress_list(raw_phs)
        if not cph:
            continue
        if cph == syl:
            return text
        if cph[: len(syl)] == syl and len(cph) > len(syl):
            cand = (len(cph) - len(syl), text)
            if best_prefix is None or cand < best_prefix:
                best_prefix = cand
            continue
        if syl[: len(cph)] == cph and len(syl) > len(cph):
            cand = (len(syl) - len(cph), text)
            if best_prefix is None or cand < best_prefix:
                best_prefix = cand
            continue
        # substring containment in either direction
        for i in range(len(cph) - len(syl) + 1):
            if cph[i : i + len(syl)] == syl:
                cand = (len(cph) - len(syl), text)
                if best_contains is None or cand < best_contains:
                    best_contains = cand
                break
        else:
            for i in range(len(syl) - len(cph) + 1):
                if syl[i : i + len(cph)] == cph:
                    cand = (len(syl) - len(cph), text)
                    if best_contains is None or cand < best_contains:
                        best_contains = cand
                    break
        d = phonetics.phoneme_distance(syl, cph)
        if d <= max(1, len(syl)):
            cand_d = (d, abs(len(cph) - len(syl)), text)
            if best_dist is None or cand_d < best_dist:
                best_dist = cand_d

    if best_exact is not None:
        return best_exact
    if best_prefix is not None:
        return best_prefix[1]
    if best_contains is not None:
        return best_contains[1]
    if best_dist is not None:
        return best_dist[2]
    return None


def _load_corpus_phonemes(db: sqlite3.Connection) -> list[tuple[str, list[str]]]:
    rows = db.execute(
        "SELECT DISTINCT text_lower, phonemes FROM words WHERE phonemes IS NOT NULL"
    ).fetchall()
    return [(r[0], r[1].split()) for r in rows if r[1]]


def _try_syllable_split(
    token: str,
    normalized: str,
    db: sqlite3.Connection,
    plan: SearchPlan,
) -> SearchUnit | None:
    phs = phonetics.word_to_phonemes(normalized)
    if not phs:
        plan.log.append(f"[{token}] syllable split: no G2P entry — MISS")
        return None

    syllables = phonetics.split_into_syllables(phs)
    if not syllables:
        plan.log.append(f"[{token}] syllable split: no syllables — MISS")
        return None

    corpus = _load_corpus_phonemes(db)
    if not corpus:
        plan.log.append(f"[{token}] syllable split: empty phoneme corpus — MISS")
        return None

    parts: list[str] = []
    for syl in syllables:
        match = _resolve_syllable(syl, corpus)
        if match is None:
            plan.log.append(
                f"[{token}] syllable split: syllable {syl} unresolved — MISS"
            )
            return None
        parts.append(match)

    plan.log.append(f"[{token}] syllable split → {parts}")
    return SearchUnit(token, "syllable_split", parts)


def decompose(
    sentence: str, db: sqlite3.Connection, config: DecomposerConfig
) -> SearchPlan:
    """Turn a sentence into a SearchPlan against the given corpus DB."""
    plan = SearchPlan(sentence=sentence, units=[])
    for token in sentence.split():
        normalized = normalize(token)
        if not normalized:
            continue
        n_variants = _word_exists(db, normalized)
        if n_variants > 0:
            plan.units.append(SearchUnit(token, "literal", [normalized]))
            plan.log.append(f"[{token}] literal match found ({n_variants} variants)")
            continue
        if config.enable_syllable_split:
            unit = _try_syllable_split(token, normalized, db, plan)
            if unit is not None:
                plan.units.append(unit)
                continue
            plan.units.append(SearchUnit(token, "miss", []))
            continue
        plan.units.append(SearchUnit(token, "miss", []))
        plan.log.append(
            f"[{token}] no literal match — MISS (syllable split disabled in v1)"
        )
    return plan
