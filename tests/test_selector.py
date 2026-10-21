from grblr.db import open_db
from grblr.decomposer import SearchPlan, SearchUnit
from grblr.phonetics import word_to_phonemes
from grblr.selector import SelectionBias, select


def _seed(db, rows: list[tuple[str, str, int, int, float]]):
    """rows: (text_lower, path, start_ms, end_ms, rms_db)"""
    paths_seen = {}
    for text, path, start, end, rms in rows:
        if path not in paths_seen:
            db.execute(
                "INSERT INTO clips (path, duration_ms, source_id, sample_rate, channels, "
                "whisper_model, indexed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (path, 5000, "TestGame", 16000, 1, "medium", 0),
            )
            paths_seen[path] = db.execute(
                "SELECT id FROM clips WHERE path = ?", (path,)
            ).fetchone()[0]
        clip_id = paths_seen[path]
        db.execute(
            "INSERT INTO words (clip_id, text_lower, text_original, start_ms, end_ms, "
            "duration_ms, rms_db, peak_db) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (clip_id, text, text, start, end, end - start, rms, -3.0),
        )
    db.commit()


def _phonemes(word):
    ph = word_to_phonemes(word)
    assert ph is not None
    return " ".join(ph)


def _seed_with_features(
    db,
    rows: list[tuple[str, str, float | None, float | None, float | None, str | None]],
):
    """rows: (text_lower, path, median_f0_hz, spectral_centroid_hz, voiced_fraction, phonemes)"""
    paths_seen = {}
    for text, path, f0, centroid, vf, ph in rows:
        if path not in paths_seen:
            db.execute(
                "INSERT INTO clips (path, duration_ms, source_id, sample_rate, channels, "
                "whisper_model, indexed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (path, 5000, "TestGame", 16000, 1, "medium", 0),
            )
            paths_seen[path] = db.execute(
                "SELECT id FROM clips WHERE path = ?", (path,)
            ).fetchone()[0]
        clip_id = paths_seen[path]
        db.execute(
            "INSERT INTO words (clip_id, text_lower, text_original, start_ms, end_ms, "
            "duration_ms, rms_db, peak_db, median_f0_hz, spectral_centroid_hz, "
            "voiced_fraction, phonemes) "
            "VALUES (?, ?, ?, 0, 500, 500, -20.0, -3.0, ?, ?, ?, ?)",
            (clip_id, text, text, f0, centroid, vf, ph),
        )
    db.commit()


def test_select_literal_picks_one_variant(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    _seed(db, [("hello", "/a.wav", 0, 500, -20.0)])
    plan = SearchPlan("hello", [SearchUnit("hello", "literal", ["hello"])])

    rp = select(plan, db, seed=42)

    assert rp.seed == 42
    assert len(rp.picks) == 1
    assert rp.picks[0].clip_path == "/a.wav"
    assert rp.picks[0].start_ms == 0
    assert rp.picks[0].end_ms == 500
    assert rp.picks[0].variants_available == 1
    assert rp.picks[0].source_id == "TestGame"


def test_select_is_deterministic_for_seed(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    _seed(
        db,
        [
            ("hello", "/a.wav", 0, 500, -20.0),
            ("hello", "/b.wav", 0, 500, -15.0),
            ("hello", "/c.wav", 0, 500, -25.0),
        ],
    )
    plan = SearchPlan("hello", [SearchUnit("hello", "literal", ["hello"])])

    rp1 = select(plan, db, seed=7)
    rp2 = select(plan, db, seed=7)

    assert rp1.picks[0].clip_path == rp2.picks[0].clip_path


def test_select_different_seeds_can_differ(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    _seed(
        db,
        [
            ("hello", "/a.wav", 0, 500, -20.0),
            ("hello", "/b.wav", 0, 500, -15.0),
        ],
    )
    plan = SearchPlan("hello", [SearchUnit("hello", "literal", ["hello"])])

    # Try a handful of seeds and confirm we see both variants somewhere.
    seen = set()
    for s in range(20):
        rp = select(plan, db, seed=s)
        seen.add(rp.picks[0].clip_path)
    assert len(seen) == 2


def test_select_miss_emits_pick_with_none(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    _seed(db, [("hello", "/a.wav", 0, 500, -20.0)])  # provides corpus for AVG duration
    plan = SearchPlan(
        "hello universe",
        [
            SearchUnit("hello", "literal", ["hello"]),
            SearchUnit("universe", "miss", []),
        ],
    )

    rp = select(plan, db, seed=0)

    assert rp.picks[0].clip_path == "/a.wav"
    assert rp.picks[1].clip_path is None
    assert rp.picks[1].variants_available == 0
    assert rp.picks[1].duration_ms > 0
    assert "MISS" in rp.picks[1].reason


def test_select_miss_uses_400ms_fallback_when_corpus_empty(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    plan = SearchPlan("hi", [SearchUnit("hi", "miss", [])])

    rp = select(plan, db, seed=0)

    assert rp.picks[0].duration_ms == 400


def test_select_weight_fn_biases_pick(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    _seed(
        db,
        [
            ("hello", "/quiet.wav", 0, 500, -40.0),
            ("hello", "/loud.wav", 0, 500, -5.0),
        ],
    )
    plan = SearchPlan("hello", [SearchUnit("hello", "literal", ["hello"])])

    # Weight strongly toward higher rms_db → "loud.wav" should win every time.
    loud_count = 0
    for s in range(10):
        rp = select(
            plan,
            db,
            seed=s,
            weight_fn=lambda row: 1000.0 if row["rms_db"] > -10 else 0.001,
        )
        if rp.picks[0].clip_path == "/loud.wav":
            loud_count += 1
    assert loud_count >= 9


def test_select_prefer_deep_picks_lower_f0(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    _seed_with_features(
        db,
        [
            ("hello", "/deep.wav", 120.0, 1500.0, 0.9, None),
            ("hello", "/high.wav", 250.0, 1500.0, 0.9, None),
        ],
    )
    plan = SearchPlan("hello", [SearchUnit("hello", "literal", ["hello"])])

    for s in range(10):
        rp = select(plan, db, seed=s, bias=SelectionBias(prefer_deep=True))
        assert rp.picks[0].clip_path == "/deep.wav"


def test_select_prefer_high_picks_higher_f0(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    _seed_with_features(
        db,
        [
            ("hello", "/deep.wav", 120.0, 1500.0, 0.9, None),
            ("hello", "/high.wav", 250.0, 1500.0, 0.9, None),
        ],
    )
    plan = SearchPlan("hello", [SearchUnit("hello", "literal", ["hello"])])

    for s in range(10):
        rp = select(plan, db, seed=s, bias=SelectionBias(prefer_high=True))
        assert rp.picks[0].clip_path == "/high.wav"


def test_select_pitch_max_filters_out_high(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    _seed_with_features(
        db,
        [
            ("hello", "/deep.wav", 120.0, 1500.0, 0.9, None),
            ("hello", "/high.wav", 250.0, 1500.0, 0.9, None),
        ],
    )
    plan = SearchPlan("hello", [SearchUnit("hello", "literal", ["hello"])])

    for s in range(10):
        rp = select(plan, db, seed=s, bias=SelectionBias(pitch_max_hz=180.0))
        assert rp.picks[0].clip_path == "/deep.wav"


def test_select_pitch_range_falls_back_if_empty(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    _seed_with_features(
        db,
        [
            ("hello", "/deep.wav", 120.0, 1500.0, 0.9, None),
            ("hello", "/high.wav", 250.0, 1500.0, 0.9, None),
        ],
    )
    plan = SearchPlan("hello", [SearchUnit("hello", "literal", ["hello"])])

    rp = select(plan, db, seed=0, bias=SelectionBias(pitch_min_hz=400.0))
    assert rp.picks[0].clip_path in ("/deep.wav", "/high.wav")
    assert any(
        "fallback" in line.lower() or "fell back" in line.lower() for line in rp.log
    )


def test_select_sound_alike_miss_picks_phonetic_neighbor(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    _seed_with_features(
        db,
        [
            ("hat", "/hat.wav", None, None, None, _phonemes("hat")),
            ("dog", "/dog.wav", None, None, None, _phonemes("dog")),
        ],
    )
    plan = SearchPlan("cat", [SearchUnit("cat", "miss", [])])

    rp = select(plan, db, seed=0, miss_strategy="sound-alike")
    assert rp.picks[0].clip_path == "/hat.wav"


def _rate(db, word_id: int, quality: str) -> None:
    db.execute(
        "INSERT OR REPLACE INTO word_quality (word_id, quality, rated_at, notes) "
        "VALUES (?, ?, ?, NULL)",
        (word_id, quality, 0),
    )
    db.commit()


def test_select_excludes_ignored_by_default(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    _seed(
        db,
        [
            ("hello", "/keep.wav", 0, 500, -20.0),
            ("hello", "/skip.wav", 0, 500, -20.0),
        ],
    )
    skip_id = db.execute(
        "SELECT id FROM words WHERE text_lower='hello' "
        "AND clip_id=(SELECT id FROM clips WHERE path='/skip.wav')"
    ).fetchone()[0]
    _rate(db, skip_id, "ignore")
    plan = SearchPlan("hello", [SearchUnit("hello", "literal", ["hello"])])

    for s in range(20):
        rp = select(plan, db, seed=s)
        assert rp.picks[0].clip_path == "/keep.wav"


def test_select_excludes_ignored_can_be_disabled(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    _seed(
        db,
        [
            ("hello", "/a.wav", 0, 500, -20.0),
            ("hello", "/b.wav", 0, 500, -20.0),
        ],
    )
    b_id = db.execute(
        "SELECT id FROM words WHERE text_lower='hello' "
        "AND clip_id=(SELECT id FROM clips WHERE path='/b.wav')"
    ).fetchone()[0]
    _rate(db, b_id, "ignore")
    plan = SearchPlan("hello", [SearchUnit("hello", "literal", ["hello"])])

    seen = set()
    for s in range(20):
        rp = select(plan, db, seed=s, bias=SelectionBias(exclude_ignored=False))
        seen.add(rp.picks[0].clip_path)
    assert seen == {"/a.wav", "/b.wav"}


def test_select_excludes_ignored_falls_back_if_all_ignored(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    _seed(db, [("hello", "/only.wav", 0, 500, -20.0)])
    only_id = db.execute("SELECT id FROM words WHERE text_lower='hello'").fetchone()[0]
    _rate(db, only_id, "ignore")
    plan = SearchPlan("hello", [SearchUnit("hello", "literal", ["hello"])])

    rp = select(plan, db, seed=0)
    assert rp.picks[0].clip_path == "/only.wav"
    assert any("ignored" in line for line in rp.log)


def test_select_prefers_good_variants(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    _seed(
        db,
        [
            ("hello", "/good.wav", 0, 500, -20.0),
            ("hello", "/unrated.wav", 0, 500, -20.0),
        ],
    )
    good_id = db.execute(
        "SELECT id FROM words WHERE text_lower='hello' "
        "AND clip_id=(SELECT id FROM clips WHERE path='/good.wav')"
    ).fetchone()[0]
    _rate(db, good_id, "good")
    plan = SearchPlan("hello", [SearchUnit("hello", "literal", ["hello"])])

    good_wins = 0
    n = 200
    for s in range(n):
        rp = select(plan, db, seed=s)
        if rp.picks[0].clip_path == "/good.wav":
            good_wins += 1
    # With +log(3) bias, good should win ~75% of the time.
    assert good_wins > n * 0.6


def test_select_quality_bias_can_be_disabled(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    _seed(
        db,
        [
            ("hello", "/good.wav", 0, 500, -20.0),
            ("hello", "/unrated.wav", 0, 500, -20.0),
        ],
    )
    good_id = db.execute(
        "SELECT id FROM words WHERE text_lower='hello' "
        "AND clip_id=(SELECT id FROM clips WHERE path='/good.wav')"
    ).fetchone()[0]
    _rate(db, good_id, "good")
    plan = SearchPlan("hello", [SearchUnit("hello", "literal", ["hello"])])

    counts = {"/good.wav": 0, "/unrated.wav": 0}
    for s in range(60):
        rp = select(plan, db, seed=s, bias=SelectionBias(quality_bias=False))
        path = rp.picks[0].clip_path
        assert path is not None
        counts[path] += 1
    # Without bias both should appear a fair amount.
    assert counts["/good.wav"] > 10
    assert counts["/unrated.wav"] > 10


def test_select_sound_alike_falls_back_to_silence_when_no_phonemes(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    _seed_with_features(
        db,
        [
            ("hat", "/hat.wav", None, None, None, _phonemes("hat")),
        ],
    )
    plan = SearchPlan("qwxyzzzz", [SearchUnit("qwxyzzzz", "miss", [])])

    rp = select(plan, db, seed=0, miss_strategy="sound-alike")
    assert rp.picks[0].clip_path is None
    assert rp.picks[0].variants_available == 0


# v2.2 corpus filters: lang_buckets and exclude_dupes


def _seed_with_buckets(db, rows):
    """rows: (text, path, lang_bucket, dupe_of_id_or_None)"""
    paths = {}
    for text, path, bucket, dupe_of in rows:
        if path not in paths:
            db.execute(
                "INSERT INTO clips (path, duration_ms, source_id, sample_rate, channels, "
                "whisper_model, indexed_at, lang_bucket, dupe_of) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (path, 5000, "TestGame", 16000, 1, "medium", 0, bucket, dupe_of),
            )
            paths[path] = db.execute(
                "SELECT id FROM clips WHERE path=?", (path,)
            ).fetchone()[0]
        db.execute(
            "INSERT INTO words (clip_id, text_lower, text_original, start_ms, end_ms, "
            "duration_ms, rms_db, peak_db) VALUES (?, ?, ?, 0, 500, 500, -20.0, -3.0)",
            (paths[path], text, text),
        )
    db.commit()
    return paths


def test_select_lang_buckets_restricts_to_en(tmp_path):
    """English render must NOT pick a word from a Japanese-labeled clip even if
    whisper transcribed an English token. This is the whole point of v2.2 for
    the renderer — language bucket gates corpus selection."""
    db = open_db(str(tmp_path / "t.db"))
    _seed_with_buckets(
        db,
        [
            ("hello", "/en.wav", "en", None),
            ("hello", "/ja.wav", "ja", None),  # decoy: same text, wrong language
            ("hello", "/unk.wav", "unknown", None),
        ],
    )
    plan = SearchPlan("hello", [SearchUnit("hello", "literal", ["hello"])])
    rp = select(plan, db, seed=0, bias=SelectionBias(lang_buckets=frozenset({"en"})))
    assert rp.picks[0].clip_path == "/en.wav"


def test_select_lang_buckets_none_means_no_filter(tmp_path):
    """Backward compat: omitting lang_buckets keeps the legacy "any clip" behavior."""
    db = open_db(str(tmp_path / "t.db"))
    _seed_with_buckets(
        db,
        [
            ("hello", "/ja.wav", "ja", None),
        ],
    )
    plan = SearchPlan("hello", [SearchUnit("hello", "literal", ["hello"])])
    rp = select(plan, db, seed=0, bias=SelectionBias(lang_buckets=None))
    assert rp.picks[0].clip_path == "/ja.wav"


def test_select_lang_buckets_union(tmp_path):
    """frozenset({'en','unknown'}) is the "include recoverable" mode — for when
    the user wants to mine the unknown bucket later (WC3 horse-PERFECT case)."""
    db = open_db(str(tmp_path / "t.db"))
    _seed_with_buckets(
        db,
        [
            ("hello", "/en.wav", "en", None),
            ("hello", "/unk.wav", "unknown", None),
            ("hello", "/ja.wav", "ja", None),
        ],
    )
    plan = SearchPlan("hello", [SearchUnit("hello", "literal", ["hello"])])
    seen = set()
    for s in range(20):
        rp = select(
            plan,
            db,
            seed=s,
            bias=SelectionBias(lang_buckets=frozenset({"en", "unknown"})),
        )
        seen.add(rp.picks[0].clip_path)
    assert seen == {"/en.wav", "/unk.wav"}  # ja never picked


def test_select_exclude_dupes_skips_sequestered(tmp_path):
    """Sequestered dupes have dupe_of NOT NULL — selector must skip
    them because the audio file is no longer at clips.path."""
    db = open_db(str(tmp_path / "t.db"))
    paths = _seed_with_buckets(
        db,
        [
            ("hello", "/canon.wav", "en", None),
        ],
    )
    canon_id = paths["/canon.wav"]
    _seed_with_buckets(
        db,
        [
            ("hello", "/dupe.wav", "en", canon_id),  # sequestered duplicate
        ],
    )
    plan = SearchPlan("hello", [SearchUnit("hello", "literal", ["hello"])])
    # default exclude_dupes=True: only the canonical wins, every seed
    for s in range(10):
        rp = select(plan, db, seed=s, bias=SelectionBias())
        assert rp.picks[0].clip_path == "/canon.wav"


def test_select_exclude_dupes_false_includes_them(tmp_path):
    """Escape hatch: callers who explicitly want dupes (debug / archaeology)
    can set exclude_dupes=False."""
    db = open_db(str(tmp_path / "t.db"))
    paths = _seed_with_buckets(db, [("hello", "/canon.wav", "en", None)])
    _seed_with_buckets(db, [("hello", "/dupe.wav", "en", paths["/canon.wav"])])
    plan = SearchPlan("hello", [SearchUnit("hello", "literal", ["hello"])])
    seen = set()
    for s in range(20):
        rp = select(plan, db, seed=s, bias=SelectionBias(exclude_dupes=False))
        seen.add(rp.picks[0].clip_path)
    assert "/dupe.wav" in seen


def _seed_with_word_durations(db, rows):
    """rows: (text, path, word_dur_ms)"""
    paths = {}
    for text, path, dur in rows:
        if path not in paths:
            db.execute(
                "INSERT INTO clips (path, duration_ms, source_id, sample_rate, channels, "
                "whisper_model, indexed_at, lang_bucket) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'en')",
                (path, 5000, "TestGame", 16000, 1, "medium", 0),
            )
            paths[path] = db.execute(
                "SELECT id FROM clips WHERE path=?", (path,)
            ).fetchone()[0]
        db.execute(
            "INSERT INTO words (clip_id, text_lower, text_original, start_ms, end_ms, "
            "duration_ms, rms_db, peak_db) VALUES (?, ?, ?, 0, ?, ?, -20.0, -3.0)",
            (paths[path], text, text, dur, dur),
        )
    db.commit()


def test_select_min_word_duration_drops_boundary_errors(tmp_path):
    """The 'eat 22ms' case: whisperX alignment lands mid-phoneme, leaving a
    sub-perceptual slice tagged as a real word. min_word_duration_ms gates
    these out at render time."""
    db = open_db(str(tmp_path / "t.db"))
    _seed_with_word_durations(
        db,
        [
            ("eat", "/broken.wav", 22),  # the boundary-error case
            ("eat", "/good.wav", 350),  # a real take
        ],
    )
    plan = SearchPlan("eat", [SearchUnit("eat", "literal", ["eat"])])
    for s in range(10):
        rp = select(
            plan,
            db,
            seed=s,
            bias=SelectionBias(lang_buckets=frozenset({"en"}), min_word_duration_ms=80),
        )
        assert rp.picks[0].clip_path == "/good.wav"


def test_select_min_word_duration_zero_disables(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    _seed_with_word_durations(db, [("eat", "/broken.wav", 22)])
    plan = SearchPlan("eat", [SearchUnit("eat", "literal", ["eat"])])
    rp = select(
        plan,
        db,
        seed=0,
        bias=SelectionBias(lang_buckets=frozenset({"en"}), min_word_duration_ms=0),
    )
    assert rp.picks[0].clip_path == "/broken.wav"  # not filtered


def test_select_content_classes_filter(tmp_path):
    """A clip can be lang_bucket='en' via path/dir prior while its audio was
    actually content-classified as speech-ja/music/etc. The
    JP-in-EN contamination vector. content_classes filter removes them."""
    db = open_db(str(tmp_path / "t.db"))
    # two clips, both lang_bucket='en' but different content_class on segment 0
    for path, klass in [
        ("/clean.wav", "speech-en"),
        ("/contaminated.wav", "speech-ja"),
    ]:
        db.execute(
            "INSERT INTO clips (path, duration_ms, source_id, sample_rate, channels, "
            "whisper_model, indexed_at, lang_bucket) VALUES (?, 5000, 'G', 16000, 1, 'm', 0, 'en')",
            (path,),
        )
        cid = db.execute("SELECT id FROM clips WHERE path=?", (path,)).fetchone()[0]
        db.execute(
            "INSERT INTO clip_content (clip_id, segment_idx, content_class, speech_prom) "
            "VALUES (?, 0, ?, 0.8)",
            (cid, klass),
        )
        db.execute(
            "INSERT INTO words (clip_id, text_lower, text_original, start_ms, end_ms, "
            "duration_ms, rms_db, peak_db) VALUES (?, 'hello', 'hello', 0, 500, 500, -20.0, -3.0)",
            (cid,),
        )
    db.commit()
    plan = SearchPlan("hello", [SearchUnit("hello", "literal", ["hello"])])
    rp = select(
        plan,
        db,
        seed=0,
        bias=SelectionBias(
            lang_buckets=frozenset({"en"}), content_classes=frozenset({"speech-en"})
        ),
    )
    assert rp.picks[0].clip_path == "/clean.wav"


def test_select_content_classes_excluded_clip_with_no_content_row_drops(tmp_path):
    """A clip with no clip_content row at segment 0 produces NULL from the
    subquery, which is NOT IN {'speech-en'} → excluded. This is correct: an
    un-content-scored clip is unsafe to include under a strict filter."""
    db = open_db(str(tmp_path / "t.db"))
    db.execute(
        "INSERT INTO clips (path, duration_ms, source_id, sample_rate, channels, "
        "whisper_model, indexed_at, lang_bucket) VALUES ('/a.wav', 5000, 'G', 16000, 1, 'm', 0, 'en')"
    )
    cid = db.execute("SELECT id FROM clips WHERE path='/a.wav'").fetchone()[0]
    db.execute(
        "INSERT INTO words (clip_id, text_lower, text_original, start_ms, end_ms, "
        "duration_ms, rms_db, peak_db) VALUES (?, 'hello', 'hello', 0, 500, 500, -20.0, -3.0)",
        (cid,),
    )
    db.commit()
    plan = SearchPlan("hello", [SearchUnit("hello", "literal", ["hello"])])
    rp = select(
        plan,
        db,
        seed=0,
        bias=SelectionBias(
            lang_buckets=frozenset({"en"}), content_classes=frozenset({"speech-en"})
        ),
    )
    assert rp.picks[0].clip_path is None  # filtered to empty → MISS
    assert rp.picks[0].reason.startswith("MISS")


def test_select_lang_buckets_emptied_pool_falls_to_miss(tmp_path):
    """If lang_buckets filters out every variant for a word, the selector emits
    a MISS rather than silently breaking the seam."""
    db = open_db(str(tmp_path / "t.db"))
    _seed_with_buckets(
        db,
        [
            ("hello", "/ja.wav", "ja", None),  # only ja exists for "hello"
        ],
    )
    plan = SearchPlan("hello", [SearchUnit("hello", "literal", ["hello"])])
    rp = select(plan, db, seed=0, bias=SelectionBias(lang_buckets=frozenset({"en"})))
    assert rp.picks[0].clip_path is None
    assert rp.picks[0].reason.startswith("MISS")


def test_selection_bias_prefer_runs_defaults_false():
    from grblr.selector import SelectionBias

    assert SelectionBias().prefer_runs is False
    assert SelectionBias(prefer_runs=True).prefer_runs is True


def test_select_without_pick_span(tmp_path, monkeypatch):
    """A Pick that names a clip but carries no span must not crash the log."""
    from grblr import selector
    from grblr.selector import Pick

    db = open_db(str(tmp_path / "t.db"))
    _seed(db, [("hello", "/a.wav", 0, 300, -20.0)])
    stub = Pick(
        part="hello",
        clip_path="/a.wav",
        start_ms=None,
        end_ms=None,
        duration_ms=300,
        variants_available=1,
        rms_db=None,
        source_id=None,
        reason="stub",
    )
    monkeypatch.setattr(selector, "_pick_word_pick", lambda *a, **k: (stub, 1, []))
    plan = SearchPlan("hello", [SearchUnit("hello", "literal", ["hello"])])
    rp = select(plan, db, seed=0)
    assert rp.picks == [stub]
    assert rp.log[-1] == "[hello] 1 variants → picked /a.wav @ ?–? (stub)"
