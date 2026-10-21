"""Contiguous-run unit selection (greedy v1). Builds an in-memory positional
index over the already-filtered English corpus and greedily tiles a target word
sequence with the longest available contiguous runs. The renderer reuses these
as wider `Pick` spans (one clip, [first.start, last.end]) — no renderer change.
Lattice/Viterbi is the roadmapped graduation; this is its candidate generator.

Spec: docs/superpowers/specs/2026-05-31-contiguous-run-unit-selection-design.md
"""

from __future__ import annotations

import dataclasses
import random as _random
import sqlite3
from dataclasses import dataclass

from grblr.selector import SelectionBias, _bucket_clauses


@dataclass
class InstanceFeatures:
    word_id: int
    duration_ms: int
    rms_db: float | None
    f0: float | None
    centroid: float | None
    voiced: float | None
    quality: str | None
    keep_prob: float | None


@dataclass
class PositionIndex:
    by_word: dict[str, list[tuple[int, int]]]
    by_pos: dict[tuple[int, int], tuple[str, int, int]]
    feat: dict[tuple[int, int], InstanceFeatures] | None = None


def build_position_index(
    db: sqlite3.Connection, bias: SelectionBias | None, with_features: bool = False
) -> PositionIndex:
    """Index the filtered corpus by (clip, position). Honors lang_bucket /
    content_class / dupe filters but NOT min_word_duration_ms. When with_features
    is True, also attaches per-position InstanceFeatures (word_id + acoustic +
    quality + latest keep_prob) so the lattice cost models need no per-render
    queries. Default False keeps the greedy path unchanged."""
    index_bias = (
        dataclasses.replace(bias, min_word_duration_ms=None)
        if bias is not None
        else None
    )
    filt, fp = _bucket_clauses(index_bias)
    cols = "w.clip_id, w.text_lower, w.start_ms, w.end_ms"
    if with_features:
        cols += (
            ", w.id, w.duration_ms, w.rms_db, w.median_f0_hz, "
            "w.spectral_centroid_hz, w.voiced_fraction"
        )
    rows = db.execute(
        f"SELECT {cols} FROM words w JOIN clips c ON c.id = w.clip_id "
        f"WHERE 1=1{filt} ORDER BY w.clip_id, w.start_ms",
        fp,
    ).fetchall()

    quality_map: dict[int, str] = {}
    keep_map: dict[int, float] = {}
    if with_features:
        quality_map = {
            wid: q for wid, q in db.execute("SELECT word_id, quality FROM word_quality")
        }
        # Latest score per word wins. Order by rowid (insertion order = chronological:
        # backfills/rescores append newer rows) rather than the model_version STRING,
        # which would mis-sort a future numeric/semver scheme (e.g. 'v10' < 'v2').
        for wid, kp in db.execute(
            "SELECT word_id, keep_prob FROM clip_filter ORDER BY rowid"
        ):
            keep_map[wid] = kp

    by_word: dict[str, list[tuple[int, int]]] = {}
    by_pos: dict[tuple[int, int], tuple[str, int, int]] = {}
    feat: dict[tuple[int, int], InstanceFeatures] | None = {} if with_features else None
    cur_clip = None
    pos = 0
    for row in rows:
        clip_id, word, start, end = row[0], row[1], row[2], row[3]
        if clip_id != cur_clip:
            cur_clip, pos = clip_id, 0
        by_word.setdefault(word, []).append((clip_id, pos))
        by_pos[(clip_id, pos)] = (word, start, end)
        if feat is not None:
            wid, dur, rms, f0, cen, vf = row[4], row[5], row[6], row[7], row[8], row[9]
            feat[(clip_id, pos)] = InstanceFeatures(
                word_id=wid,
                duration_ms=dur,
                rms_db=rms,
                f0=f0,
                centroid=cen,
                voiced=vf,
                quality=quality_map.get(wid),
                keep_prob=keep_map.get(wid),
            )
        pos += 1
    return PositionIndex(by_word=by_word, by_pos=by_pos, feat=feat)


@dataclass
class Tile:
    words: list[str]
    length: int
    candidates: list[tuple[int, int, int]]  # (clip_id, start_ms, end_ms)


def _extend_run(
    idx: PositionIndex, target: list[str], i: int, clip_id: int, pos: int
) -> int:
    """Max contiguous match length L>=1 of target[i:] against the clip starting at
    (clip_id, pos): extend while the successor word matches the next target word."""
    L = 1
    while True:
        nxt = idx.by_pos.get((clip_id, pos + L))
        if (i + L) >= len(target) or nxt is None or nxt[0] != target[i + L]:
            return L
        L += 1


def _run_candidates(
    idx: PositionIndex, target: list[str], i: int
) -> tuple[int, list[tuple[int, int, int]]]:
    """The LONGEST L>=1 with >=1 contiguous match of target[i:i+L], and the clip
    spans for it. Returns (L, candidates); (1, []) when target[i] is absent."""
    best_len = 0
    best: list[tuple[int, int, int]] = []
    for clip_id, pos in idx.by_word.get(target[i], []):
        L = _extend_run(idx, target, i, clip_id, pos)
        if L < best_len:
            continue
        span = (
            clip_id,
            idx.by_pos[(clip_id, pos)][1],
            idx.by_pos[(clip_id, pos + L - 1)][2],
        )
        if L > best_len:
            best_len, best = L, [span]
        else:
            best.append(span)
    if best_len == 0:
        return (1, [])
    best.sort()
    return (best_len, best)


def cover_target(idx: PositionIndex, target: list[str]) -> list[Tile]:
    """Greedy leftmost-longest tiling. length-1 tiles carry empty candidates so
    the caller routes them to the existing per-word selector (the fallback)."""
    tiles: list[Tile] = []
    i = 0
    while i < len(target):
        L, cands = _run_candidates(idx, target, i)
        if L >= 2:
            tiles.append(Tile(words=target[i : i + L], length=L, candidates=cands))
        else:
            tiles.append(Tile(words=[target[i]], length=1, candidates=[]))
        i += L
    return tiles


def pick_run_ranked(candidates, pos_of, feats, target_cost, rng, run_len=1):
    """Choose one (clip_id, start_ms, end_ms) by lowest target cost over the WHOLE
    run (mean of its run_len words' costs — consistent with the lattice), not just
    the anchor word. pos_of maps a candidate's clip_id -> its run's first
    (clip_id, pos). Ties broken by seeded rng over a stable sort (reproducible)."""
    if not candidates:
        raise ValueError("pick_run_ranked called with no candidates")
    scored = []
    for c in candidates:
        clip_id, pos = pos_of[c[0]]
        positions = [(clip_id, pos + k) for k in range(run_len)]
        scored.append((target_cost.run_cost(positions, feats), c[0], c))
    best = min(s[0] for s in scored)
    tied = sorted(s[2] for s in scored if abs(s[0] - best) < 1e-12)
    return rng.choice(tied)


def all_run_candidates(
    idx: PositionIndex, target: list[str], i: int
) -> list[tuple[int, int, int]]:
    """Every contiguous run starting at target[i]: a list of (length, clip_id,
    first_pos). For a clip that matches target[i:i+M] contiguously, emits one
    tuple per length 1..M (prefix-closed). Empty when target[i] is absent."""
    out: list[tuple[int, int, int]] = []
    for clip_id, pos in idx.by_word.get(target[i], []):
        L = _extend_run(idx, target, i, clip_id, pos)
        for length in range(1, L + 1):
            out.append((length, clip_id, pos))
    return out


def _run_matches(idx, words, clip_id, pos):
    for k, w in enumerate(words):
        cell = idx.by_pos.get((clip_id, pos + k))
        if cell is None or cell[0] != w:
            return False
    return True


def pick_run(
    db: sqlite3.Connection,
    words: list[str],
    candidates: list[tuple[int, int, int]],
    rng: _random.Random,
) -> tuple[int, int, int]:
    """Choose one (clip_id, start_ms, end_ms) for a run. Drops candidate clips
    whose first run-word is rated 'ignore', then seeded-uniform choice. Falls
    back to the full list if the ignore filter empties it."""
    if not candidates:
        raise ValueError("pick_run called with no candidates")
    ignored = {
        r[0]
        for r in db.execute(
            "SELECT DISTINCT w.clip_id FROM words w "
            "JOIN word_quality q ON q.word_id = w.id "
            "WHERE q.quality = 'ignore' AND w.text_lower = ? "
            f"AND w.clip_id IN ({','.join('?' * len(candidates))})",
            [words[0], *[c[0] for c in candidates]],
        ).fetchall()
    }
    pool = [c for c in candidates if c[0] not in ignored] or candidates
    return rng.choice(pool)
