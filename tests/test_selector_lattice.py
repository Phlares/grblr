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


def test_prefer_lattice_yields_instance_chosen_run():
    conn = open_db(":memory:")
    _clip(conn, 1)
    _word(conn, 10, 1, "go", 0, 300)
    _word(conn, 11, 1, "to", 300, 600)
    conn.commit()
    plan = decompose("go to", conn, DecomposerConfig())
    bias = SelectionBias(
        lang_buckets=frozenset({"en"}),
        content_classes=frozenset({"speech-en"}),
        prefer_lattice=True,
    )
    rp = select(plan, conn, seed=1, bias=bias)
    assert len(rp.picks) == 1
    p = rp.picks[0]
    assert p.reason.startswith("run:2") and p.start_ms == 0 and p.end_ms == 600


def test_prefer_lattice_precedence_routes_like_lattice():
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
        prefer_lattice=True,
    )
    rp = select(plan, conn, seed=1, bias=bias)
    assert len(rp.picks) == 1 and rp.picks[0].reason.startswith("run:2")


def test_prefer_lattice_with_miss_unit_does_not_recurse():
    # A miss/unknown word under prefer_lattice routes through _emit_unit_legacy,
    # which re-enters select(). BOTH run flags must be cleared there, else select()
    # recurses into _select_with_runs forever. A low recursion limit would expose it.
    import sys

    conn = open_db(":memory:")
    _clip(conn, 1)
    _word(conn, 10, 1, "go", 0, 300)
    _word(conn, 11, 1, "to", 300, 600)
    conn.commit()
    plan = decompose("go to zzzzqx", conn, DecomposerConfig())
    bias = SelectionBias(
        lang_buckets=frozenset({"en"}),
        content_classes=frozenset({"speech-en"}),
        prefer_lattice=True,
    )
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(200)  # infinite recursion would RecursionError here
    try:
        rp = select(plan, conn, seed=1, bias=bias)
    finally:
        sys.setrecursionlimit(old)
    assert "go to" in [p.part for p in rp.picks]  # run realized, call returned


def test_select_rejects_featureless_prebuilt_index():
    import pytest

    from grblr import run_select

    conn = open_db(":memory:")
    _clip(conn, 1)
    _word(conn, 10, 1, "go", 0, 300)
    _word(conn, 11, 1, "to", 300, 600)
    conn.commit()
    idx = run_select.build_position_index(
        conn, SelectionBias(lang_buckets=frozenset({"en"})), with_features=False
    )
    plan = decompose("go to", conn, DecomposerConfig())
    bias = SelectionBias(
        lang_buckets=frozenset({"en"}),
        content_classes=frozenset({"speech-en"}),
        prefer_lattice=True,
    )
    with pytest.raises(ValueError):
        select(plan, conn, seed=1, bias=bias, position_index=idx)
