from __future__ import annotations

import math
import random
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from grblr import phonetics
from grblr.decomposer import SearchPlan

WeightFn = Callable[[sqlite3.Row], float]
MissStrategy = Literal["silence", "random", "skip", "sound-alike"]
_MISS_DURATION_FALLBACK_MS = 400


@dataclass
class SelectionBias:
    prefer_longer: bool = False
    prefer_deep: bool = False
    prefer_high: bool = False
    prefer_bright: bool = False
    prefer_dark: bool = False
    pitch_min_hz: float | None = None
    pitch_max_hz: float | None = None
    min_voiced_fraction: float | None = None
    quality_bias: bool = True
    exclude_ignored: bool = True
    source_ids: frozenset[str] | None = None
    """If set, only variants whose source_id is in this set are eligible.
    None means no source filtering."""
    exclude_source_ids: frozenset[str] | None = None
    """If set, variants from these source_ids are EXCLUDED (the inverse of
    source_ids — an allowlist removal for known-bad batches, e.g. a game whose
    extraction is corrupted). None means no exclusion."""
    lang_buckets: frozenset[str] | None = None
    """If set, only variants from clips whose v2.2 lang_bucket is in this set
    are eligible. CLI/UI default to frozenset({'en'}) for English TTS;
    None (the dataclass default) means no language filter — keeps legacy
    tests and other callers unaffected. Buckets: en/ja/neither/unknown."""
    exclude_dupes: bool = True
    """Skip clips with dupe_of NOT NULL (sequestered duplicates moved out of
    the active corpus path). Default True because dupes literally aren't at
    the location stored in clips.path anymore — loading them would fail."""
    min_word_duration_ms: int | None = None
    """Reject word variants shorter than this. wav2vec2-CTC's ~20ms stride
    produces boundary errors where the aligner lands inside a single phoneme
    instead of on a word edge — those slices are sub-perceptual garbage that
    sound like static. CLI/UI default 80ms; pass 0/None to disable. The full
    fix is the aligner upgrade + re-index; this is the cheap render-time gate."""
    max_word_duration_ms: int | None = None
    """Reject word variants LONGER than this — the ceiling partner of the floor.
    A single word spanning multiple seconds is almost always a too-long boundary
    error (the aligner ran the span into trailing silence/cutscene); such spans
    drag and read as 'slowed'. Only ~0.01% of corpus words exceed 4000ms, so a
    high ceiling sheds garbage at negligible cost. None disables."""
    content_classes: frozenset[str] | None = None
    """If set, only variants from clips whose segment-0 content_class is in
    this set are eligible. CLI/UI default frozenset({'speech-en'}) for English
    TTS, which excludes clips that lang_bucket flagged as English by path/dir
    prior but whose audio PANNs content-classified as music/sung/speech-other/
    speech-ja — the JP-in-EN contamination vector when whisperX hallucinates
    English tokens on Japanese audio. None means no content filter (preserves
    backward compat for tests and legacy callers)."""
    prefer_runs: bool = False
    """When True, select() covers the target with the longest available
    contiguous corpus runs (multi-word spans from one clip) before falling back
    to single-word selection. Reduced function words are kept inside runs (never
    cut), so min_word_duration_ms applies only to length-1 fallback units.
    Default False preserves the legacy per-word path for existing callers."""
    prefer_lattice: bool = False
    """When True, cover the target with the cost-weighted instance-level Viterbi
    lattice (spec 2026-06-01) instead of the greedy cover. Implies the run path and
    takes precedence over prefer_runs. Default False = greedy/per-word unchanged."""


@dataclass
class Pick:
    part: str
    clip_path: str | None
    start_ms: int | None
    end_ms: int | None
    duration_ms: int
    variants_available: int
    rms_db: float | None
    source_id: str | None
    reason: str


@dataclass
class RenderPlan:
    sentence: str
    seed: int
    picks: list[Pick]
    log: list[str] = field(default_factory=list)


def _avg_corpus_duration_ms(db: sqlite3.Connection) -> int:
    row = db.execute("SELECT AVG(duration_ms) FROM words").fetchone()
    return int(row[0]) if row and row[0] is not None else _MISS_DURATION_FALLBACK_MS


def _bucket_clauses(bias: SelectionBias | None) -> tuple[str, list]:
    """SQL `AND`-clauses + bind params for v2.2 corpus filters (lang_bucket,
    dupe_of, min_word_duration_ms, content_classes, source_ids). Returns
    ('', []) when no filter applies — caller can splice directly into a WHERE
    clause that has `clips c` aliased and `words w` available.

    source_ids is filtered here (not only in the per-word Python path) so the
    contiguous-run index (build_position_index) honors it too — otherwise
    prefer_runs would silently draw runs from deselected games."""
    if bias is None:
        return "", []
    clauses = []
    params: list = []
    if bias.exclude_dupes:
        clauses.append("AND c.dupe_of IS NULL")
    if bias.lang_buckets:  # truthy: None or empty set → skip (empty would build IN ())
        placeholders = ",".join("?" * len(bias.lang_buckets))
        clauses.append(f"AND c.lang_bucket IN ({placeholders})")
        params.extend(sorted(bias.lang_buckets))
    if bias.min_word_duration_ms is not None and bias.min_word_duration_ms > 0:
        clauses.append("AND w.duration_ms >= ?")
        params.append(bias.min_word_duration_ms)
    if (
        bias.content_classes
    ):  # truthy: None or empty set → skip (empty would build IN ())
        placeholders = ",".join("?" * len(bias.content_classes))
        clauses.append(
            "AND (SELECT content_class FROM clip_content "
            f"WHERE clip_id=c.id AND segment_idx=0) IN ({placeholders})"
        )
        params.extend(sorted(bias.content_classes))
    if bias.source_ids:  # truthy: None or empty set → skip (empty would build IN ())
        placeholders = ",".join("?" * len(bias.source_ids))
        clauses.append(f"AND c.source_id IN ({placeholders})")
        params.extend(sorted(bias.source_ids))
    if bias.max_word_duration_ms is not None and bias.max_word_duration_ms > 0:
        clauses.append("AND w.duration_ms <= ?")
        params.append(bias.max_word_duration_ms)
    if bias.exclude_source_ids:
        placeholders = ",".join("?" * len(bias.exclude_source_ids))
        clauses.append(f"AND c.source_id NOT IN ({placeholders})")
        params.extend(sorted(bias.exclude_source_ids))
    return (" " + " ".join(clauses) if clauses else ""), params


def _query_variants(
    db: sqlite3.Connection, text_lower: str, bias: SelectionBias | None = None
) -> list[sqlite3.Row]:
    db.row_factory = sqlite3.Row
    filt, fp = _bucket_clauses(bias)
    return db.execute(
        "SELECT w.id, w.text_lower, w.start_ms, w.end_ms, w.duration_ms, "
        "       w.rms_db, w.peak_db, w.median_f0_hz, w.spectral_centroid_hz, "
        "       w.voiced_fraction, w.phonemes, c.path, c.source_id, "
        "       q.quality "
        "FROM words w JOIN clips c ON c.id = w.clip_id "
        "LEFT JOIN word_quality q ON q.word_id = w.id "
        f"WHERE w.text_lower = ?{filt} "
        "ORDER BY w.id",
        [text_lower, *fp],
    ).fetchall()


def _random_corpus_word(
    db: sqlite3.Connection, rng: random.Random, bias: SelectionBias | None = None
) -> sqlite3.Row | None:
    db.row_factory = sqlite3.Row
    filt, fp = _bucket_clauses(bias)
    total = db.execute(
        f"SELECT COUNT(*) FROM words w JOIN clips c ON c.id = w.clip_id "
        f"WHERE 1=1{filt}",
        fp,
    ).fetchone()[0]
    if total == 0:
        return None
    offset = rng.randrange(total)
    return db.execute(
        "SELECT w.id, w.text_lower, w.start_ms, w.end_ms, w.duration_ms, "
        "       w.rms_db, w.peak_db, c.path, c.source_id "
        "FROM words w JOIN clips c ON c.id = w.clip_id "
        f"WHERE 1=1{filt} "
        "ORDER BY w.id LIMIT 1 OFFSET ?",
        [*fp, offset],
    ).fetchone()


def _sound_alike_neighbor(
    db: sqlite3.Connection,
    target_phonemes: list[str],
    rng: random.Random,
    bias: SelectionBias | None = None,
) -> tuple[sqlite3.Row, int] | None:
    db.row_factory = sqlite3.Row
    filt, fp = _bucket_clauses(bias)
    rows = db.execute(
        "SELECT w.id, w.text_lower, w.start_ms, w.end_ms, w.duration_ms, "
        "       w.rms_db, w.peak_db, w.phonemes, c.path, c.source_id "
        "FROM words w JOIN clips c ON c.id = w.clip_id "
        f"WHERE w.phonemes IS NOT NULL{filt} "
        "ORDER BY w.id",
        fp,
    ).fetchall()
    if not rows:
        return None
    best_d = math.inf
    best: list[sqlite3.Row] = []
    for r in rows:
        ph = str(r["phonemes"] or "").split()
        if not ph:
            continue
        d = phonetics.phoneme_distance(target_phonemes, ph)
        if d < best_d:
            best_d = d
            best = [r]
        elif d == best_d:
            best.append(r)
    if not best:
        return None
    return rng.choice(best), int(best_d)


def _weighted_choice(rng: random.Random, rows: list, weights: list[float]):
    total = sum(weights)
    if total <= 0:
        return rng.choice(rows)
    r = rng.uniform(0, total)
    upto = 0.0
    for row, w in zip(rows, weights, strict=False):
        upto += w
        if upto >= r:
            return row
    return rows[-1]


def _bias_weight(row: sqlite3.Row, bias: SelectionBias) -> float:
    log_w = 0.0
    f0 = row["median_f0_hz"] if "median_f0_hz" in row.keys() else None
    centroid = (
        row["spectral_centroid_hz"] if "spectral_centroid_hz" in row.keys() else None
    )
    dur = row["duration_ms"] if "duration_ms" in row.keys() else None
    quality = row["quality"] if "quality" in row.keys() else None

    if bias.prefer_deep and f0 is not None and f0 > 0:
        # lower f0 → larger weight; log domain so it composes additively.
        log_w += -math.log(f0 / 100.0) * 6.0
    if bias.prefer_high and f0 is not None and f0 > 0:
        log_w += math.log(f0 / 100.0) * 6.0
    if bias.prefer_bright and centroid is not None and centroid > 0:
        log_w += math.log(centroid / 1000.0) * 3.0
    if bias.prefer_dark and centroid is not None and centroid > 0:
        log_w += -math.log(centroid / 1000.0) * 3.0
    if bias.prefer_longer and dur is not None and dur > 0:
        log_w += math.log(dur / 100.0) * 1.0
    if bias.quality_bias and quality == "good":
        log_w += math.log(3.0)
    return math.exp(log_w)


def _hard_filter(rows: list[sqlite3.Row], bias: SelectionBias) -> list[sqlite3.Row]:
    out: list[sqlite3.Row] = []
    for r in rows:
        f0 = r["median_f0_hz"] if "median_f0_hz" in r.keys() else None
        vf = r["voiced_fraction"] if "voiced_fraction" in r.keys() else None
        sg = r["source_id"] if "source_id" in r.keys() else None
        if bias.pitch_min_hz is not None and (f0 is None or f0 < bias.pitch_min_hz):
            continue
        if bias.pitch_max_hz is not None and (f0 is None or f0 > bias.pitch_max_hz):
            continue
        if bias.min_voiced_fraction is not None and (
            vf is None or vf < bias.min_voiced_fraction
        ):
            continue
        if (
            bias.source_ids and sg not in bias.source_ids
        ):  # truthy: matches _bucket_clauses (empty → no filter)
            continue
        out.append(r)
    return out


def _pick_word_pick(db, part, rng, eff_bias, weight_fn, bias, miss_duration):
    rows = _query_variants(db, part, eff_bias)
    if not rows:
        return (
            Pick(
                part=part,
                clip_path=None,
                start_ms=None,
                end_ms=None,
                duration_ms=miss_duration,
                variants_available=0,
                rms_db=None,
                source_id=None,
                reason="MISS (no variants at select-time)",
            ),
            0,
            [],
        )
    pool = rows
    extra_log: list[str] = []
    if eff_bias.exclude_ignored:
        kept = [
            r
            for r in pool
            if (r["quality"] if "quality" in r.keys() else None) != "ignore"
        ]
        if not kept:
            extra_log.append(f"[{part}] all variants ignored — fallback to unfiltered")
        else:
            pool = kept
    if (
        eff_bias.pitch_min_hz is not None
        or eff_bias.pitch_max_hz is not None
        or eff_bias.min_voiced_fraction is not None
        or eff_bias.source_ids
    ):
        filtered = _hard_filter(pool, eff_bias)
        if not filtered:
            extra_log.append(
                f"[{part}] hard filter emptied pool — fallback to unfiltered"
            )
        else:
            pool = filtered
    has_quality_signal = eff_bias.quality_bias and any(
        (r["quality"] if "quality" in r.keys() else None) == "good" for r in pool
    )
    if weight_fn is None and bias is None and not has_quality_signal:
        chosen = rng.choice(pool)
        reason = f"uniform random among {len(pool)} variants"
    else:
        weights = []
        for r in pool:
            w = max(0.0, weight_fn(r)) if weight_fn is not None else 1.0
            w *= _bias_weight(r, eff_bias)
            weights.append(w)
        chosen = _weighted_choice(rng, pool, weights)
        reason = f"weighted random among {len(pool)} variants"
    return (
        Pick(
            part=part,
            clip_path=chosen["path"],
            start_ms=chosen["start_ms"],
            end_ms=chosen["end_ms"],
            duration_ms=chosen["duration_ms"],
            variants_available=len(pool),
            rms_db=chosen["rms_db"],
            source_id=chosen["source_id"],
            reason=reason,
        ),
        len(pool),
        extra_log,
    )


def refine_end(
    end_ms: int, start_ms: int, n_words: int, *, max_word_ms: int = 1200
) -> int:
    """Cap a pick span at `start + n_words*max_word_ms` (pure). A few words can't
    legitimately run for many seconds, so this kills VAD/boundary blowouts — e.g.
    audit #15's 17 s 'go' (gunfire tail) and #26's 5.7 s.

    NOTE: an earlier version also o2o-EXTENDED a clipped word-tail toward the next
    word's onset, but a blind old-vs-new A/B rejected it 17-3 (p=0.003): extending
    toward whisperX onsets (~60 ms off on defect edges) dragged in audible BLEED,
    worse than the clean-but-short cut. The real cut-off fix needs the boundary
    re-align; here we keep only the unambiguous blowout cap."""
    return min(end_ms, start_ms + n_words * max_word_ms)


def _refine_spans(picks: list[Pick]) -> None:
    """In-place: cap every audio pick's span via refine_end. One post-pass so run
    AND single-word picks get the same absurd-duration guard. No DB lookups — the
    cap uses only the pick's own fields."""
    for p in picks:
        if p.clip_path is None or p.start_ms is None or p.end_ms is None:
            continue
        new_end = refine_end(p.end_ms, p.start_ms, len(p.part.split()))
        if new_end != p.end_ms:
            p.end_ms = new_end
            p.duration_ms = new_end - p.start_ms


def _secs(ms: int | None) -> str:
    return "?" if ms is None else f"{ms / 1000:.3f}s"


def _run_pick(db, words, clip_id, start_ms, end_ms, length, variants_available):
    """Realize a chosen run instance (one clip span) into a Pick. Shared by the
    greedy and lattice run paths — both look up the clip path/game and build the
    same `run:N` Pick; only their log wording differs (kept at the call sites).
    Returns (pick, clip_path)."""
    row = db.execute(
        "SELECT path, source_id FROM clips WHERE id=?", (clip_id,)
    ).fetchone()
    return Pick(
        part=" ".join(words),
        clip_path=row[0],
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=end_ms - start_ms,
        variants_available=variants_available,
        rms_db=None,
        source_id=row[1],
        reason=f"run:{length}",
    ), row[0]


def _select_with_runs(
    plan,
    db,
    seed,
    rng,
    eff_bias,
    weight_fn,
    bias,
    miss_duration,
    miss_strategy,
    position_index=None,
):
    """v1 greedy contiguous-run cover (spec 2026-05-31). A run pick carries its
    word count in `reason` as 'run:N'. Runs span consecutive single-word literal
    units; miss/syllable-split units route through _emit_unit_legacy. The lattice
    graduation (spec §8) will unify this with the legacy per-word loop."""
    from grblr import lattice as _lat
    from grblr import run_select
    from grblr.cost import AverageProxyJoinCost, TargetCost, compute_norm_stats

    rp = RenderPlan(sentence=plan.sentence, seed=seed, picks=[], log=list(plan.log))
    idx = position_index or run_select.build_position_index(
        db, eff_bias, with_features=True
    )
    if idx.feat is None:
        raise ValueError(
            "run/lattice selection needs a feature-bearing index; a supplied "
            "position_index must be built with build_position_index(..., with_features=True)"
        )
    ns = compute_norm_stats(idx.feat)
    target_cost = TargetCost(ns)
    join_cost = AverageProxyJoinCost(ns)
    use_lattice = eff_bias.prefer_lattice
    units = plan.units
    i = 0
    while i < len(units):
        u = units[i]
        if u.strategy == "miss" or len(u.parts) != 1:
            _emit_unit_legacy(rp, u, db, weight_fn, bias, eff_bias, miss_strategy)
            i += 1
            continue
        j = i
        words = []
        while (
            j < len(units) and units[j].strategy != "miss" and len(units[j].parts) == 1
        ):
            words.append(units[j].parts[0])
            j += 1
        if use_lattice:
            lat = _lat.build_lattice(idx, words, target_cost, top_k=16)
            for t in _lat.viterbi(lat, join_cost, idx.feat, rng):
                if t.clip_id is not None:
                    pick, path = _run_pick(
                        db,
                        t.words,
                        t.clip_id,
                        t.start_ms,
                        t.end_ms,
                        t.length,
                        t.variants,
                    )
                    rp.picks.append(pick)
                    rp.log.append(
                        f"[{' '.join(t.words)}] lattice run:{t.length} -> {path} "
                        f"@ {_secs(t.start_ms)}-{_secs(t.end_ms)}"
                    )
                else:  # fallback node => per-word path
                    pick, n, extra_log = _pick_word_pick(
                        db, t.words[0], rng, eff_bias, weight_fn, bias, miss_duration
                    )
                    rp.picks.append(pick)
                    rp.log.extend(extra_log)
                    rp.log.append(
                        f"[{t.words[0]}] (lattice fallback) {n} variants -> {pick.clip_path}"
                    )
        else:
            for t in run_select.cover_target(idx, words):
                if t.length >= 2:
                    pos_of = {}
                    for clip_id, pos in idx.by_word.get(t.words[0], []):
                        if run_select._run_matches(idx, t.words, clip_id, pos):
                            pos_of.setdefault(clip_id, (clip_id, pos))
                    clip_id, start, end = run_select.pick_run_ranked(
                        t.candidates, pos_of, idx.feat, target_cost, rng, len(t.words)
                    )
                    pick, path = _run_pick(
                        db, t.words, clip_id, start, end, t.length, len(t.candidates)
                    )
                    rp.picks.append(pick)
                    rp.log.append(
                        f"[{' '.join(t.words)}] run:{t.length} → {path} "
                        f"@ {start / 1000:.3f}s–{end / 1000:.3f}s"
                    )
                else:
                    pick, n, extra_log = _pick_word_pick(
                        db, t.words[0], rng, eff_bias, weight_fn, bias, miss_duration
                    )
                    rp.picks.append(pick)
                    rp.log.extend(extra_log)
                    rp.log.append(
                        f"[{t.words[0]}] (fallback) {n} variants → {pick.clip_path}"
                    )
        i = j
    return rp


def _emit_unit_legacy(rp, unit, db, weight_fn, bias, eff_bias, miss_strategy):
    """Handle one miss/syllable-split unit via the existing per-unit path by
    re-entering select() with prefer_runs disabled (no recursion). Reuses the
    full miss-handling logic rather than duplicating it. NOTE: this re-seeds a
    fresh rng from rp.seed per call, so non-literal units in one sentence draw
    from correlated streams — deterministic and fine for v1; the lattice
    graduation will thread a single shared rng."""
    import dataclasses

    from grblr.decomposer import SearchPlan

    legacy_bias = dataclasses.replace(eff_bias, prefer_runs=False, prefer_lattice=False)
    out = select(
        SearchPlan(sentence=rp.sentence, units=[unit]),
        db,
        seed=rp.seed,
        weight_fn=weight_fn,
        bias=legacy_bias,
        miss_strategy=miss_strategy,
    )
    rp.picks.extend(out.picks)
    rp.log.extend(out.log)


def _emit_silence_miss(
    rp: RenderPlan, target: str, miss_duration: int, note: str
) -> None:
    rp.picks.append(
        Pick(
            part=target,
            clip_path=None,
            start_ms=None,
            end_ms=None,
            duration_ms=miss_duration,
            variants_available=0,
            rms_db=None,
            source_id=None,
            reason="MISS",
        )
    )
    rp.log.append(f"[{target}] 0 variants — MISS ({note})")


def select(
    plan: SearchPlan,
    db: sqlite3.Connection,
    *,
    seed: int,
    weight_fn: WeightFn | None = None,
    bias: SelectionBias | None = None,
    miss_strategy: MissStrategy = "silence",
    position_index=None,
) -> RenderPlan:
    """Choose one clip per part of each SearchUnit. Seeded for determinism.

    weight_fn and bias compose multiplicatively (weight_fn first, then bias).
    Hard filters from bias (pitch range, voiced fraction) apply before weighting;
    if filtering empties the candidate pool, the selector logs a note and falls
    back to the unfiltered set.

    miss_strategy:
      - "silence":     emit a Pick with clip_path=None
      - "skip":        don't emit any Pick
      - "random":      substitute a uniformly random corpus word
      - "sound-alike": substitute the phonetically nearest corpus word

    position_index: an optional prebuilt run_select.PositionIndex to reuse across
    calls (the eval harness builds it once per config). When prefer_runs/prefer_lattice
    is set it MUST have been built with_features=True (the cost models need idx.feat).
    None (default) builds the index per call, unchanged.
    """
    rng = random.Random(seed)
    rp = RenderPlan(sentence=plan.sentence, seed=seed, picks=[], log=list(plan.log))
    miss_duration = _avg_corpus_duration_ms(db)
    eff_bias = bias if bias is not None else SelectionBias()

    if eff_bias.prefer_lattice or eff_bias.prefer_runs:
        rp = _select_with_runs(
            plan,
            db,
            seed,
            rng,
            eff_bias,
            weight_fn,
            bias,
            miss_duration,
            miss_strategy,
            position_index,
        )
        _refine_spans(rp.picks)
        return rp

    for unit in plan.units:
        if unit.strategy == "miss":
            if miss_strategy == "skip":
                rp.log.append(f"[{unit.target}] 0 variants — MISS (skipped)")
                continue
            if miss_strategy == "random":
                sub = _random_corpus_word(db, rng, eff_bias)
                if sub is not None:
                    rp.picks.append(
                        Pick(
                            part=unit.target,
                            clip_path=sub["path"],
                            start_ms=sub["start_ms"],
                            end_ms=sub["end_ms"],
                            duration_ms=sub["duration_ms"],
                            variants_available=0,
                            rms_db=sub["rms_db"],
                            source_id=sub["source_id"],
                            reason=f"MISS substituted with random corpus word {sub['text_lower']!r}",
                        )
                    )
                    rp.log.append(
                        f"[{unit.target}] 0 variants — MISS, substituted with "
                        f"{sub['text_lower']!r} from {sub['source_id'] or '?'}"
                    )
                    continue
            if miss_strategy == "sound-alike":
                tph = phonetics.word_to_phonemes(unit.target)
                if not tph:
                    _emit_silence_miss(
                        rp,
                        unit.target,
                        miss_duration,
                        "sound-alike: no G2P entry, fell back to silence",
                    )
                    continue
                hit = _sound_alike_neighbor(db, tph, rng, eff_bias)
                if hit is None:
                    _emit_silence_miss(
                        rp,
                        unit.target,
                        miss_duration,
                        "sound-alike: no phoneme corpus, fell back to silence",
                    )
                    continue
                sub, dist = hit
                rp.picks.append(
                    Pick(
                        part=unit.target,
                        clip_path=sub["path"],
                        start_ms=sub["start_ms"],
                        end_ms=sub["end_ms"],
                        duration_ms=sub["duration_ms"],
                        variants_available=0,
                        rms_db=sub["rms_db"],
                        source_id=sub["source_id"],
                        reason=f"MISS sound-alike substitute {sub['text_lower']!r} (phoneme dist {dist})",
                    )
                )
                rp.log.append(
                    f"[{unit.target}] no literal — sound-alike: phoneme distance "
                    f"{dist} → substituted {sub['text_lower']!r} from "
                    f"{sub['source_id'] or '?'}"
                )
                continue
            _emit_silence_miss(rp, unit.target, miss_duration, "silence placeholder")
            continue

        for part in unit.parts:
            pick, n, extra_log = _pick_word_pick(
                db, part, rng, eff_bias, weight_fn, bias, miss_duration
            )
            rp.picks.append(pick)
            rp.log.extend(extra_log)
            if pick.clip_path is None:
                rp.log.append(f"[{part}] 0 variants — MISS")
            else:
                rp.log.append(
                    f"[{part}] {n} variants → picked {pick.clip_path} "
                    f"@ {_secs(pick.start_ms)}–{_secs(pick.end_ms)} ({pick.reason})"
                )
    _refine_spans(rp.picks)
    return rp
