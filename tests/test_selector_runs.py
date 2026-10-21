from grblr.db import open_db
from grblr.decomposer import DecomposerConfig, decompose
from grblr.selector import SelectionBias, select


def _clip(conn, cid):
    conn.execute(
        "INSERT INTO clips (id, path, duration_ms, source_id, sample_rate, "
        "channels, whisper_model, indexed_at, lang_bucket) "
        "VALUES (?, ?, 5000, 'g', 16000, 1, 'm', 0, 'en')",
        (cid, f"c{cid}.wav"),
    )
    conn.execute(
        "INSERT INTO clip_content (clip_id, segment_idx, content_class) "
        "VALUES (?, 0, 'speech-en')",
        (cid,),
    )


def _word(conn, wid, cid, text, start, end):
    conn.execute(
        "INSERT INTO words (id, clip_id, text_lower, text_original, start_ms, "
        "end_ms, duration_ms, rms_db, peak_db) VALUES (?,?,?,?,?,?,?,-20,-3)",
        (wid, cid, text, text, start, end, end - start),
    )


def test_select_uses_run_when_prefer_runs():
    conn = open_db(":memory:")
    _clip(conn, 1)
    _word(conn, 10, 1, "go", 0, 300)
    _word(conn, 11, 1, "to", 300, 600)
    conn.commit()
    plan = decompose("go to", conn, DecomposerConfig())
    bias = SelectionBias(
        lang_buckets=frozenset({"en"}),
        content_classes=frozenset({"speech-en"}),
        prefer_runs=True,
    )
    rp = select(plan, conn, seed=1, bias=bias)
    assert len(rp.picks) == 1
    p = rp.picks[0]
    assert p.part == "go to" and p.start_ms == 0 and p.end_ms == 600
    assert p.reason.startswith("run:2")


def test_select_falls_back_to_singles_without_run():
    # "go" and "fast" are non-adjacent in the clip: a third word "now" sits
    # between them at the corpus position level (pos 0=go, pos 1=now, pos 2=fast),
    # so no contiguous "go fast" run exists — both must fall back to singles.
    conn = open_db(":memory:")
    _clip(conn, 1)
    _word(conn, 10, 1, "go", 0, 300)
    _word(conn, 15, 1, "now", 300, 600)  # gap word — breaks the run
    _word(conn, 20, 1, "fast", 600, 900)
    conn.commit()
    plan = decompose("go fast", conn, DecomposerConfig())
    bias = SelectionBias(
        lang_buckets=frozenset({"en"}),
        content_classes=frozenset({"speech-en"}),
        prefer_runs=True,
    )
    rp = select(plan, conn, seed=1, bias=bias)
    assert [p.part for p in rp.picks] == ["go", "fast"]


def _clip_game(conn, cid, game):
    conn.execute(
        "INSERT INTO clips (id, path, duration_ms, source_id, sample_rate, "
        "channels, whisper_model, indexed_at, lang_bucket) "
        "VALUES (?, ?, 5000, ?, 16000, 1, 'm', 0, 'en')",
        (cid, f"c{cid}.wav", game),
    )
    conn.execute(
        "INSERT INTO clip_content (clip_id, segment_idx, content_class) "
        "VALUES (?, 0, 'speech-en')",
        (cid,),
    )


def test_runs_respect_source_ids():
    # Regression: prefer_runs must honor source_ids. A 'go to' run exists ONLY
    # in the EXCLUDED game 'A'. The INCLUDED game 'B' has both words but with a
    # gap word between them, so no contiguous run exists there. With
    # source_ids={'B'} the run path must NOT reach into 'A' — it should fall
    # back to single words, all sourced from 'B'.
    conn = open_db(":memory:")
    _clip_game(conn, 1, "A")
    _word(conn, 10, 1, "go", 0, 300)
    _word(conn, 11, 1, "to", 300, 600)  # contiguous 'go to' run in A (excluded)
    _clip_game(conn, 2, "B")
    _word(conn, 20, 2, "go", 0, 300)
    _word(conn, 21, 2, "x", 300, 600)  # gap word breaks the run in B
    _word(conn, 22, 2, "to", 600, 900)
    conn.commit()
    plan = decompose("go to", conn, DecomposerConfig())
    bias = SelectionBias(
        source_ids=frozenset({"B"}),
        lang_buckets=frozenset({"en"}),
        content_classes=frozenset({"speech-en"}),
        prefer_runs=True,
    )
    rp = select(plan, conn, seed=1, bias=bias)
    used = [p.source_id for p in rp.picks if p.clip_path]
    assert used, "expected at least one real pick"
    assert all(g == "B" for g in used), f"source filter bypassed by run path: {used}"
    assert not any(p.reason.startswith("run") for p in rp.picks), (
        "the only contiguous run lives in excluded game 'A'; it must not be used"
    )


def test_bucket_clauses_max_duration_and_exclude_games():
    from grblr.selector import _bucket_clauses

    filt, params = _bucket_clauses(
        SelectionBias(
            max_word_duration_ms=4000, exclude_source_ids=frozenset({"BadGame"})
        )
    )
    assert "w.duration_ms <= ?" in filt and 4000 in params
    assert "c.source_id NOT IN" in filt and "BadGame" in params
    f2, _ = _bucket_clauses(SelectionBias())  # disabled by default
    assert "duration_ms <= " not in f2 and "NOT IN" not in f2


def test_max_word_duration_excludes_long_word_from_index():
    from grblr import run_select

    conn = open_db(":memory:")
    _clip(conn, 1)
    _word(conn, 10, 1, "ok", 0, 300)  # 300ms kept
    _word(conn, 11, 1, "drag", 300, 8300)  # 8000ms garbage, excluded by the ceiling
    conn.commit()
    idx = run_select.build_position_index(
        conn,
        SelectionBias(
            lang_buckets=frozenset({"en"}),
            content_classes=frozenset({"speech-en"}),
            max_word_duration_ms=4000,
        ),
    )
    assert "ok" in idx.by_word and "drag" not in idx.by_word


def test_exclude_source_ids_removes_game_from_index():
    from grblr import run_select

    conn = open_db(":memory:")
    _clip(conn, 1)
    _word(conn, 10, 1, "keep", 0, 300)
    conn.execute(
        "INSERT INTO clips (id, path, duration_ms, source_id, sample_rate, "
        "channels, whisper_model, indexed_at, lang_bucket) "
        "VALUES (2, 'c2.wav', 5000, 'BadGame', 16000, 1, 'm', 0, 'en')"
    )
    conn.execute(
        "INSERT INTO clip_content (clip_id, segment_idx, content_class) "
        "VALUES (2, 0, 'speech-en')"
    )
    _word(conn, 20, 2, "drop", 0, 300)
    conn.commit()
    idx = run_select.build_position_index(
        conn,
        SelectionBias(
            lang_buckets=frozenset({"en"}),
            content_classes=frozenset({"speech-en"}),
            exclude_source_ids=frozenset({"BadGame"}),
        ),
    )
    assert "keep" in idx.by_word and "drop" not in idx.by_word


def test_bucket_clauses_empty_frozensets_dont_build_invalid_in():
    # An empty (non-None) IN-set must NOT emit `IN ()` (a SQLite syntax error).
    from grblr.selector import _bucket_clauses

    filt, params = _bucket_clauses(
        SelectionBias(
            source_ids=frozenset(),
            lang_buckets=frozenset(),
            content_classes=frozenset(),
        )
    )
    assert "IN ()" not in filt
    assert "lang_bucket" not in filt and "source_id IN" not in filt  # empty → skipped
    # and an actual query built with such a bias must execute, not raise.
    conn = open_db(":memory:")
    _clip(conn, 1)
    _word(conn, 10, 1, "go", 0, 300)
    conn.commit()
    plan = decompose("go", conn, DecomposerConfig())
    rp = select(plan, conn, seed=1, bias=SelectionBias(source_ids=frozenset()))
    # empty set means "no filter" consistently: no sqlite3.OperationalError, the
    # clip is found, and we DON'T go through the drop-everything → fallback path
    # (which would mean the SQL layer and the Python _hard_filter disagreed).
    assert rp.picks and rp.picks[0].clip_path is not None
    assert not any("fallback" in line for line in rp.log)
