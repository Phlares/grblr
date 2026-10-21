import random
import sqlite3

from grblr.cost import AverageProxyJoinCost, TargetCost, compute_norm_stats
from grblr.lattice import FALLBACK, build_lattice, viterbi
from grblr.run_select import build_position_index
from grblr.selector import SelectionBias


def _db(rows_words):
    db = sqlite3.connect(":memory:")
    db.executescript("""
        CREATE TABLE clips(id INTEGER PRIMARY KEY, dupe_of INTEGER, lang_bucket TEXT);
        CREATE TABLE words(id INTEGER PRIMARY KEY, clip_id INTEGER, text_lower TEXT,
            start_ms INTEGER, end_ms INTEGER, duration_ms INTEGER, rms_db REAL,
            median_f0_hz REAL, spectral_centroid_hz REAL, voiced_fraction REAL);
        CREATE TABLE word_quality(word_id INTEGER, quality TEXT);
        CREATE TABLE clip_filter(word_id INTEGER, model_version TEXT, keep_prob REAL, excluded INTEGER);
    """)
    for cid in sorted({r[1] for r in rows_words}):  # derive clips from word rows
        db.execute("INSERT INTO clips VALUES (?,NULL,'en')", (cid,))
    db.executemany("INSERT INTO words VALUES (?,?,?,?,?,?,?,?,?,?)", rows_words)
    return db


def test_lattice_nodes_are_instances_and_cover_all_positions():
    # clip1: 'a b', clip2: 'b c'  => target 'a b c'
    db = _db(
        [
            (1, 1, "a", 0, 100, 100, -20.0, 120.0, 1500.0, 0.8),
            (2, 1, "b", 100, 200, 100, -20.0, 120.0, 1500.0, 0.8),
            (3, 2, "b", 0, 100, 100, -20.0, 120.0, 1500.0, 0.8),
            (4, 2, "c", 100, 200, 100, -20.0, 120.0, 1500.0, 0.8),
        ]
    )
    idx = build_position_index(
        db, SelectionBias(lang_buckets=None, content_classes=None), with_features=True
    )
    assert idx.feat is not None
    tc = TargetCost(compute_norm_stats(idx.feat))
    lat = build_lattice(idx, ["a", "b", "c"], tc, top_k=8)
    starts = {n.i for n in lat.nodes}
    assert {0, 1, 2}.issubset(starts)
    ab = [n for n in lat.nodes if n.i == 0 and n.j == 2]
    assert ab and ab[0].clip_id == 1 and ab[0].instance is not None


def test_lattice_inserts_fallback_node_for_zero_instance_word():
    # 'a' present, 'zzz' absent from corpus  => target 'a zzz'
    db = _db([(1, 1, "a", 0, 100, 100, -20.0, 120.0, 1500.0, 0.8)])
    idx = build_position_index(
        db, SelectionBias(lang_buckets=None, content_classes=None), with_features=True
    )
    assert idx.feat is not None
    tc = TargetCost(compute_norm_stats(idx.feat))
    lat = build_lattice(idx, ["a", "zzz"], tc, top_k=8)
    fb = [n for n in lat.nodes if n.i == 1 and n.j == 2]
    assert fb and fb[0].instance is FALLBACK


def test_lattice_topk_prunes_lowest_cost_instances():
    # three single-word 'x' instances; quality spreads target cost; top_k=2 keeps
    # the two lowest-cost and drops the 'ignore' one.
    db = _db(
        [
            (1, 1, "x", 0, 100, 100, -20.0, 120.0, 1500.0, 0.8),
            (2, 2, "x", 0, 100, 100, -20.0, 120.0, 1500.0, 0.8),
            (3, 3, "x", 0, 100, 100, -20.0, 120.0, 1500.0, 0.8),
        ]
    )
    db.execute("INSERT INTO word_quality VALUES (1,'good')")  # lowest cost
    db.execute("INSERT INTO word_quality VALUES (3,'ignore')")  # highest cost
    idx = build_position_index(
        db, SelectionBias(lang_buckets=None, content_classes=None), with_features=True
    )
    assert idx.feat is not None
    tc = TargetCost(compute_norm_stats(idx.feat))
    lat = build_lattice(idx, ["x"], tc, top_k=2)
    x_nodes = [n for n in lat.nodes if n.i == 0 and n.j == 1]
    assert len(x_nodes) == 2  # pruned 3 -> top_k
    assert 3 not in {n.clip_id for n in x_nodes}  # 'ignore' (highest cost) dropped


def test_lattice_distinct_instances_and_span_ms():
    # two clips both say 'a b' => span (0,2) yields TWO distinct nodes with correct ms
    db = _db(
        [
            (1, 1, "a", 0, 100, 100, -20.0, 120.0, 1500.0, 0.8),
            (2, 1, "b", 100, 200, 100, -20.0, 120.0, 1500.0, 0.8),
            (3, 2, "a", 0, 100, 100, -20.0, 120.0, 1500.0, 0.8),
            (4, 2, "b", 100, 200, 100, -20.0, 120.0, 1500.0, 0.8),
        ]
    )
    idx = build_position_index(
        db, SelectionBias(lang_buckets=None, content_classes=None), with_features=True
    )
    assert idx.feat is not None
    tc = TargetCost(compute_norm_stats(idx.feat))
    lat = build_lattice(idx, ["a", "b"], tc, top_k=8)
    ab = [n for n in lat.nodes if n.i == 0 and n.j == 2]
    assert {n.clip_id for n in ab} == {1, 2}  # both kept, distinct
    assert all(
        n.start_ms == 0 and n.end_ms == 200 for n in ab
    )  # [first.start, last.end]


def test_viterbi_picks_instance_for_smoother_seam():
    # target 'a b': clip1 has 'a'(f0=120); two 'b' instances — clip2 f0=122 (smooth),
    # clip3 f0=400 (rough). Single-word 'a' forces the seam a->b; Viterbi must pick
    # the f0=122 'b' instance, proving instance-level (not span-level) optimization.
    db = _db(
        [
            (1, 1, "a", 0, 100, 100, -20.0, 120.0, 1500.0, 0.8),
            (2, 2, "b", 0, 100, 100, -20.0, 122.0, 1500.0, 0.8),
            (3, 3, "b", 0, 100, 100, -20.0, 400.0, 1500.0, 0.8),
        ]
    )
    idx = build_position_index(
        db, SelectionBias(lang_buckets=None, content_classes=None), with_features=True
    )
    assert idx.feat is not None
    ns = compute_norm_stats(idx.feat)
    tc = TargetCost(ns)
    jc = AverageProxyJoinCost(ns)
    lat = build_lattice(idx, ["a", "b"], tc, top_k=8)
    tiles = viterbi(lat, jc, idx.feat, random.Random(0))
    b_tile = [t for t in tiles if t.words == ["b"]][0]
    assert b_tile.clip_id == 2


def test_viterbi_deterministic_under_seed():
    db = _db(
        [
            (1, 1, "a", 0, 100, 100, -20.0, 120.0, 1500.0, 0.8),
            (2, 1, "b", 100, 200, 100, -20.0, 120.0, 1500.0, 0.8),
        ]
    )
    idx = build_position_index(
        db, SelectionBias(lang_buckets=None, content_classes=None), with_features=True
    )
    assert idx.feat is not None
    ns = compute_norm_stats(idx.feat)
    lat = build_lattice(idx, ["a", "b"], TargetCost(ns), top_k=8)
    jc = AverageProxyJoinCost(ns)
    a = viterbi(lat, jc, idx.feat, random.Random(7))
    b = viterbi(lat, jc, idx.feat, random.Random(7))
    assert [(t.words, t.clip_id) for t in a] == [(t.words, t.clip_id) for t in b]


def test_viterbi_tie_break_is_seeded_and_reproducible():
    # two IDENTICAL 'b' instances => genuine seam tie; rng.choice must be exercised
    # over both, deterministic per seed, and the pick is one of the tied clips.
    db = _db(
        [
            (1, 1, "a", 0, 100, 100, -20.0, 120.0, 1500.0, 0.8),
            (2, 2, "b", 0, 100, 100, -20.0, 120.0, 1500.0, 0.8),
            (3, 3, "b", 0, 100, 100, -20.0, 120.0, 1500.0, 0.8),
        ]
    )
    idx = build_position_index(
        db, SelectionBias(lang_buckets=None, content_classes=None), with_features=True
    )
    assert idx.feat is not None
    ns = compute_norm_stats(idx.feat)
    tc = TargetCost(ns)
    jc = AverageProxyJoinCost(ns)
    lat = build_lattice(idx, ["a", "b"], tc, top_k=8)
    picks = {viterbi(lat, jc, idx.feat, random.Random(s))[1].clip_id for s in range(8)}
    # tie is broken by the seed: across seeds BOTH tied instances get chosen
    assert picks == {2, 3}
    # and a fixed seed is reproducible
    one = viterbi(lat, jc, idx.feat, random.Random(3))[1].clip_id
    two = viterbi(lat, jc, idx.feat, random.Random(3))[1].clip_id
    assert one == two


def test_viterbi_prefers_contiguous_run_on_cost_tie():
    # one clip 'go to': the run [go to] and the split [go]+[to] tie on cost (crude
    # features identical, join ~0). The lexicographic (cost, joins) min must ALWAYS
    # pick the single contiguous run, never the split — no join regression vs greedy.
    db = _db(
        [
            (1, 1, "go", 0, 100, 100, -20.0, 120.0, 1500.0, 0.8),
            (2, 1, "to", 100, 200, 100, -20.0, 120.0, 1500.0, 0.8),
        ]
    )
    idx = build_position_index(
        db, SelectionBias(lang_buckets=None, content_classes=None), with_features=True
    )
    assert idx.feat is not None
    ns = compute_norm_stats(idx.feat)
    tc = TargetCost(ns)
    jc = AverageProxyJoinCost(ns)
    lat = build_lattice(idx, ["go", "to"], tc, top_k=8)
    for seed in range(8):
        tiles = viterbi(lat, jc, idx.feat, random.Random(seed))
        assert len(tiles) == 1 and tiles[0].length == 2  # the run, every seed


def test_viterbi_completes_path_through_fallback():
    # target 'a zzz': 'a' present, 'zzz' absent => fallback node at position 1; the
    # path must complete through it and the gap tile carries clip_id=None (the
    # selector routes those to the per-word fallback).
    db = _db([(1, 1, "a", 0, 100, 100, -20.0, 120.0, 1500.0, 0.8)])
    idx = build_position_index(
        db, SelectionBias(lang_buckets=None, content_classes=None), with_features=True
    )
    assert idx.feat is not None
    ns = compute_norm_stats(idx.feat)
    tc = TargetCost(ns)
    jc = AverageProxyJoinCost(ns)
    lat = build_lattice(idx, ["a", "zzz"], tc, top_k=8)
    tiles = viterbi(lat, jc, idx.feat, random.Random(0))
    assert [t.words for t in tiles] == [["a"], ["zzz"]]
    zzz = [t for t in tiles if t.words == ["zzz"]][0]
    assert zzz.clip_id is None


def test_lattice_tile_reports_candidate_count_not_run_length():
    # 3 single-word 'x' instances => the chosen tile.variants == 3 (pool size), and
    # length == 1 — proving variants is the candidate count, not the run word-count.
    db = _db(
        [
            (1, 1, "x", 0, 100, 100, -20.0, 120.0, 1500.0, 0.8),
            (2, 2, "x", 0, 100, 100, -20.0, 120.0, 1500.0, 0.8),
            (3, 3, "x", 0, 100, 100, -20.0, 120.0, 1500.0, 0.8),
        ]
    )
    idx = build_position_index(
        db, SelectionBias(lang_buckets=None, content_classes=None), with_features=True
    )
    assert idx.feat is not None
    ns = compute_norm_stats(idx.feat)
    tiles = viterbi(
        build_lattice(idx, ["x"], TargetCost(ns), top_k=8),
        AverageProxyJoinCost(ns),
        idx.feat,
        random.Random(0),
    )
    assert tiles[0].length == 1 and tiles[0].variants == 3
