import random

from grblr import run_select
from grblr.db import open_db
from grblr.selector import SelectionBias


def _clip(conn, cid, game="g", lang="en", content="speech-en", dupe=None):
    conn.execute(
        "INSERT INTO clips (id, path, duration_ms, source_id, sample_rate, "
        "channels, whisper_model, indexed_at, lang_bucket, dupe_of) "
        "VALUES (?, ?, 5000, ?, 16000, 1, 'm', 0, ?, ?)",
        (cid, f"c{cid}.wav", game, lang, dupe),
    )
    if content is not None:
        conn.execute(
            "INSERT INTO clip_content (clip_id, segment_idx, content_class) "
            "VALUES (?, 0, ?)",
            (cid, content),
        )


def _word(conn, wid, cid, text, start, end):
    conn.execute(
        "INSERT INTO words (id, clip_id, text_lower, text_original, start_ms, "
        "end_ms, duration_ms, rms_db, peak_db) VALUES (?,?,?,?,?,?,?,-20,-3)",
        (wid, cid, text, text, start, end, end - start),
    )


def test_position_index_orders_words_and_maps_successors():
    conn = open_db(":memory:")
    _clip(conn, 1)
    _word(conn, 11, 1, "to", 300, 600)
    _word(conn, 10, 1, "go", 0, 300)
    _word(conn, 12, 1, "home", 600, 1000)
    conn.commit()
    idx = run_select.build_position_index(conn, SelectionBias())
    assert idx.by_word["go"] == [(1, 0)]
    assert idx.by_pos[(1, 0)] == ("go", 0, 300)
    assert idx.by_pos[(1, 1)] == ("to", 300, 600)
    assert idx.by_pos[(1, 2)] == ("home", 600, 1000)


def test_position_index_respects_lang_content_dupe_filters():
    conn = open_db(":memory:")
    _clip(conn, 1, lang="en", content="speech-en")
    _clip(conn, 2, lang="ja", content="speech-ja")
    _clip(conn, 3, lang="en", content="music")
    _clip(conn, 4, lang="en", content="speech-en", dupe=1)
    _word(conn, 10, 1, "keep", 0, 300)
    _word(conn, 20, 2, "drop", 0, 300)
    _word(conn, 30, 3, "drop", 0, 300)
    _word(conn, 40, 4, "drop", 0, 300)
    conn.commit()
    bias = SelectionBias(
        lang_buckets=frozenset({"en"}), content_classes=frozenset({"speech-en"})
    )
    idx = run_select.build_position_index(conn, bias)
    assert "keep" in idx.by_word
    assert "drop" not in idx.by_word


def test_position_index_ignores_min_word_duration():
    conn = open_db(":memory:")
    _clip(conn, 1)
    _word(conn, 10, 1, "go", 0, 300)
    _word(conn, 11, 1, "to", 300, 340)
    conn.commit()
    bias = SelectionBias(min_word_duration_ms=80)
    idx = run_select.build_position_index(conn, bias)
    assert idx.by_pos[(1, 1)] == ("to", 300, 340)


def test_cover_picks_longest_run_then_falls_back():
    conn = open_db(":memory:")
    _clip(conn, 1)
    _word(conn, 10, 1, "go", 0, 300)
    _word(conn, 11, 1, "to", 300, 600)
    _word(conn, 12, 1, "home", 600, 1000)
    conn.commit()
    idx = run_select.build_position_index(conn, SelectionBias())
    tiles = run_select.cover_target(idx, ["go", "to", "mars"])
    assert [(t.length, t.words) for t in tiles] == [(2, ["go", "to"]), (1, ["mars"])]
    assert tiles[0].candidates == [(1, 0, 600)]
    assert tiles[1].candidates == []


def test_cover_all_singles_when_no_runs():
    conn = open_db(":memory:")
    _clip(conn, 1)
    _word(conn, 10, 1, "go", 0, 300)
    conn.commit()
    idx = run_select.build_position_index(conn, SelectionBias())
    tiles = run_select.cover_target(idx, ["go", "go"])
    assert [t.length for t in tiles] == [1, 1]


def test_cover_is_deterministic_candidate_order():
    conn = open_db(":memory:")
    _clip(conn, 1)
    _clip(conn, 2)
    for cid in (1, 2):
        _word(conn, cid * 10, cid, "go", 0, 300)
        _word(conn, cid * 10 + 1, cid, "to", 300, 600)
    conn.commit()
    idx = run_select.build_position_index(conn, SelectionBias())
    tiles = run_select.cover_target(idx, ["go", "to"])
    assert tiles[0].length == 2
    assert tiles[0].candidates == [(1, 0, 600), (2, 0, 600)]


def test_pick_run_is_seeded_and_skips_ignored():
    conn = open_db(":memory:")
    _clip(conn, 1)
    _clip(conn, 2)
    for cid in (1, 2):
        _word(conn, cid * 10, cid, "go", 0, 300)
        _word(conn, cid * 10 + 1, cid, "to", 300, 600)
    conn.execute(
        "INSERT INTO word_quality (word_id, quality, rated_at) VALUES (10, 'ignore', 0)"
    )
    conn.commit()
    cands = [(1, 0, 600), (2, 0, 600)]
    chosen = run_select.pick_run(conn, ["go", "to"], cands, random.Random(42))
    assert chosen == (2, 0, 600)


def test_pick_run_deterministic_for_seed():
    conn = open_db(":memory:")
    for cid in (1, 2, 3):
        _clip(conn, cid)
        _word(conn, cid * 10, cid, "go", 0, 300)
        _word(conn, cid * 10 + 1, cid, "to", 300, 600)
    conn.commit()
    cands = [(1, 0, 600), (2, 0, 600), (3, 0, 600)]
    a = run_select.pick_run(conn, ["go", "to"], cands, random.Random(7))
    b = run_select.pick_run(conn, ["go", "to"], cands, random.Random(7))
    assert a == b


def test_position_index_with_features(tmp_path):
    import sqlite3

    from grblr.run_select import build_position_index
    from grblr.selector import SelectionBias

    db = sqlite3.connect(":memory:")
    db.executescript("""
        CREATE TABLE clips(id INTEGER PRIMARY KEY, dupe_of INTEGER, lang_bucket TEXT);
        CREATE TABLE words(id INTEGER PRIMARY KEY, clip_id INTEGER, text_lower TEXT,
            start_ms INTEGER, end_ms INTEGER, duration_ms INTEGER, rms_db REAL,
            median_f0_hz REAL, spectral_centroid_hz REAL, voiced_fraction REAL);
        CREATE TABLE word_quality(word_id INTEGER, quality TEXT);
        CREATE TABLE clip_filter(word_id INTEGER, model_version TEXT, keep_prob REAL, excluded INTEGER);
        INSERT INTO clips VALUES (1, NULL, 'en');
        INSERT INTO words VALUES (10,1,'go',0,100,100,-20.0,120.0,1500.0,0.9);
        INSERT INTO words VALUES (11,1,'home',100,300,200,-18.0,NULL,1800.0,0.8);
        INSERT INTO word_quality VALUES (10,'good');
        INSERT INTO clip_filter VALUES (10,'v1',0.5,0);
        INSERT INTO clip_filter VALUES (10,'v2',0.91,0);  -- latest wins
    """)
    bias = SelectionBias(lang_buckets=None, content_classes=None, exclude_dupes=True)
    idx = build_position_index(db, bias, with_features=True)
    assert idx.feat is not None
    f = idx.feat[(1, 0)]  # 'go'
    assert f.word_id == 10
    assert f.quality == "good"
    assert f.keep_prob is not None and abs(f.keep_prob - 0.91) < 1e-9  # latest
    assert f.rms_db is not None and abs(f.rms_db + 20.0) < 1e-9
    assert idx.feat[(1, 1)].f0 is None  # 'home' has NULL f0
    assert idx.feat[(1, 1)].keep_prob is None  # no clip_filter row


def test_pick_run_prefers_higher_target_quality():
    import random

    from grblr.cost import TargetCost, compute_norm_stats
    from grblr.run_select import InstanceFeatures, pick_run_ranked

    feats = {
        (1, 0): InstanceFeatures(1, 200, -20.0, 120.0, 1500.0, 0.8, "good", 0.95),
        (2, 0): InstanceFeatures(2, 200, -20.0, 120.0, 1500.0, 0.8, "ignore", 0.10),
    }
    ns = compute_norm_stats(feats)
    tc = TargetCost(ns)
    cands = [(1, 0, 200), (2, 0, 200)]  # (clip_id, start_ms, end_ms)
    pos_of = {1: (1, 0), 2: (2, 0)}  # clip_id -> run's first position
    chosen = pick_run_ranked(cands, pos_of, feats, tc, random.Random(0))
    assert chosen[0] == 1  # higher quality clip wins deterministically


def test_all_run_candidates_enumerates_every_length():
    import sqlite3

    from grblr.run_select import all_run_candidates, build_position_index
    from grblr.selector import SelectionBias

    db = sqlite3.connect(":memory:")
    db.executescript("""
        CREATE TABLE clips(id INTEGER PRIMARY KEY, dupe_of INTEGER, lang_bucket TEXT);
        CREATE TABLE words(id INTEGER PRIMARY KEY, clip_id INTEGER, text_lower TEXT,
            start_ms INTEGER, end_ms INTEGER);
        INSERT INTO clips VALUES (1, NULL, 'en');
        INSERT INTO words VALUES (1,1,'a',0,100),(2,1,'b',100,200),(3,1,'c',200,300);
    """)
    idx = build_position_index(
        db, SelectionBias(lang_buckets=None, content_classes=None)
    )
    target = ["a", "b", "c"]
    runs = all_run_candidates(idx, target, 0)
    lengths = sorted(L for (L, clip_id, pos) in runs)
    assert lengths == [1, 2, 3]
    assert all(clip_id == 1 for (_, clip_id, _) in runs)


def test_all_run_candidates_partial_prefix_and_absent():
    import sqlite3

    from grblr.run_select import all_run_candidates, build_position_index
    from grblr.selector import SelectionBias

    db = sqlite3.connect(":memory:")
    db.executescript("""
        CREATE TABLE clips(id INTEGER PRIMARY KEY, dupe_of INTEGER, lang_bucket TEXT);
        CREATE TABLE words(id INTEGER PRIMARY KEY, clip_id INTEGER, text_lower TEXT,
            start_ms INTEGER, end_ms INTEGER);
        INSERT INTO clips VALUES (1, NULL, 'en');
        INSERT INTO words VALUES (1,1,'a',0,100),(2,1,'b',100,200);  -- clip has only 'a b'
    """)
    idx = build_position_index(
        db, SelectionBias(lang_buckets=None, content_classes=None)
    )
    # target 'a b c' but clip only matches the 'a b' prefix => lengths [1, 2], never 3
    runs = all_run_candidates(idx, ["a", "b", "c"], 0)
    assert sorted(L for (L, _, _) in runs) == [1, 2]
    # absent word at position i => empty
    assert all_run_candidates(idx, ["zzz", "b"], 0) == []


def test_pick_run_ranked_scores_whole_run_not_just_first_word():
    import random

    from grblr.cost import TargetCost, compute_norm_stats
    from grblr.run_select import InstanceFeatures, pick_run_ranked

    # clip1 'go'(good)+'to'(good); clip2 'go'(good)+'to'(ignore). First words tie;
    # whole-run scoring (run_len=2) must prefer clip1 (better 2nd word).
    feats = {
        (1, 0): InstanceFeatures(10, 200, -20.0, 120.0, 1500.0, 0.8, "good", None),
        (1, 1): InstanceFeatures(11, 200, -20.0, 120.0, 1500.0, 0.8, "good", None),
        (2, 0): InstanceFeatures(20, 200, -20.0, 120.0, 1500.0, 0.8, "good", None),
        (2, 1): InstanceFeatures(21, 200, -20.0, 120.0, 1500.0, 0.8, "ignore", None),
    }
    ns = compute_norm_stats(feats)
    tc = TargetCost(ns)
    cands = [(1, 0, 400), (2, 0, 400)]
    pos_of = {1: (1, 0), 2: (2, 0)}
    chosen = pick_run_ranked(cands, pos_of, feats, tc, random.Random(0), run_len=2)
    assert (
        chosen[0] == 1
    )  # clip1 wins on its 2nd word, ignored by first-word-only scoring
