# tests/test_render_service.py
import os
import time

import numpy as np
import pytest
import soundfile as sf

import grblr.render_service as rs
from grblr.render_service import word_timeline
from grblr.selector import Pick, RenderPlan, SelectionBias


def _patch_pipeline(monkeypatch, tmp_path):
    """Stub decompose/select/render so no corpus is needed."""
    rp = RenderPlan(
        sentence="get to mining",
        seed=7,
        picks=[
            Pick("get", "a.wav", 0, 300, 300, 0, None, None, "x"),
            Pick("to", "b.wav", 0, 150, 150, 0, None, None, "x"),
            Pick("mining", "c.wav", 0, 500, 500, 0, None, None, "x"),
        ],
        log=[],
    )
    monkeypatch.setattr(rs, "decompose", lambda *a, **k: object())
    monkeypatch.setattr(rs, "select", lambda *a, **k: rp)

    def fake_render(plan, config, out_wav, out_log):
        sf.write(str(out_wav), np.zeros(int(0.5 * 48000), dtype="float32"), 48000)
        return out_wav, out_log

    monkeypatch.setattr(rs, "render", fake_render)
    return rp


def test_render_phrase_produces_ogg_words_and_meta(monkeypatch, tmp_path):
    _patch_pipeline(monkeypatch, tmp_path)
    r = rs.render_phrase(
        conn=None, phrase="get to mining", seed=7, version="best", render_dir=tmp_path
    )
    assert r.version == "best-effort-v0"
    assert r.seed == 7
    assert [w["text"] for w in r.words] == ["get", "to", "mining"]
    assert r.ogg_path.endswith(".ogg")
    assert os.path.basename(r.ogg_path).startswith("get-to-mining-best-effort-v0-7-")
    assert os.path.exists(r.ogg_path)


def test_render_phrase_caches_by_phrase_seed_version(monkeypatch, tmp_path):
    _patch_pipeline(monkeypatch, tmp_path)
    calls = {"n": 0}
    orig = rs.select
    monkeypatch.setattr(
        rs,
        "select",
        lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), orig(*a, **k))[1],
    )
    a = rs.render_phrase(
        conn=None, phrase="x", seed=1, version="best", render_dir=tmp_path
    )
    b = rs.render_phrase(
        conn=None, phrase="x", seed=1, version="best", render_dir=tmp_path
    )
    assert a is b
    assert calls["n"] == 1  # second call served from cache


def test_word_timeline_hit_and_miss_with_head_and_gap():
    picks = [
        Pick(
            part="get",
            clip_path="a.wav",
            start_ms=100,
            end_ms=400,
            duration_ms=300,
            variants_available=1,
            rms_db=None,
            source_id=None,
            reason="x",
        ),
        Pick(
            part="to",
            clip_path=None,
            start_ms=None,
            end_ms=None,
            duration_ms=150,  # miss
            variants_available=0,
            rms_db=None,
            source_id=None,
            reason="MISS",
        ),
        Pick(
            part="mining",
            clip_path="b.wav",
            start_ms=0,
            end_ms=520,
            duration_ms=520,
            variants_available=1,
            rms_db=None,
            source_id=None,
            reason="x",
        ),
    ]
    words = word_timeline(picks, head_silence_ms=50, pad_ms=5, gap_ms=120)

    # hit dur = (end-start) + 2*pad ; miss dur = duration_ms
    assert words == [
        {"text": "get", "onset_ms": 50, "dur_ms": 310},
        {"text": "to", "onset_ms": 480, "dur_ms": 150},  # 50+310+120
        {"text": "mining", "onset_ms": 750, "dur_ms": 530},  # 480+150+120
    ]


def test_word_timeline_empty():
    assert word_timeline([], head_silence_ms=0, pad_ms=5, gap_ms=120) == []


def test_cache_lru_eviction(monkeypatch, tmp_path):
    """Inserting beyond _CACHE_MAX evicts the oldest entry (LRU semantics)."""
    from collections import OrderedDict

    import grblr.render_service as rs

    # Clear and shrink the cache for this test so we don't need 128+ renders.
    monkeypatch.setattr(rs, "_CACHE_MAX", 2)
    monkeypatch.setattr(rs, "_CACHE", OrderedDict())
    _patch_pipeline(monkeypatch, tmp_path)

    # Render three distinct (phrase, seed) pairs — only the last two should remain.
    r1 = rs.render_phrase(
        conn=None, phrase="alpha", seed=1, version="best", render_dir=tmp_path
    )
    r2 = rs.render_phrase(
        conn=None, phrase="beta", seed=2, version="best", render_dir=tmp_path
    )
    r3 = rs.render_phrase(
        conn=None, phrase="gamma", seed=3, version="best", render_dir=tmp_path
    )

    assert len(rs._CACHE) == 2, "cache should be capped at _CACHE_MAX=2"

    # The oldest entry ("alpha", 1, ...) must have been evicted.
    assert ("alpha", 1, rs.resolve_version("best")) not in rs._CACHE
    # The two most-recent entries are still present.
    assert ("beta", 2, rs.resolve_version("best")) in rs._CACHE
    assert ("gamma", 3, rs.resolve_version("best")) in rs._CACHE

    # Cache-hit identity still holds for surviving entries.
    r2b = rs.render_phrase(
        conn=None, phrase="beta", seed=2, version="best", render_dir=tmp_path
    )
    assert r2b is r2
    r3b = rs.render_phrase(
        conn=None, phrase="gamma", seed=3, version="best", render_dir=tmp_path
    )
    assert r3b is r3

    # r1 (evicted) triggers a re-render (a new object, not is r1).
    r1b = rs.render_phrase(
        conn=None, phrase="alpha", seed=1, version="best", render_dir=tmp_path
    )
    assert r1b is not r1


def test_render_phrase_rejects_unknown_version(tmp_path):
    # A version outside the allowlist is rejected before any rendering happens.
    with pytest.raises(ValueError):
        rs.render_phrase(
            conn=None, phrase="x", seed=1, version="rogue-v9", render_dir=tmp_path
        )


def test_render_phrase_rejects_path_traversal_version(tmp_path):
    # A ?version= path-traversal attempt must raise and write nothing anywhere —
    # resolve_version rejects it before render_dir is even created.
    render_dir = tmp_path / "renders"
    before = set(tmp_path.rglob("*"))
    with pytest.raises(ValueError):
        rs.render_phrase(
            conn=None,
            phrase="x",
            seed=1,
            version="../../../etc/evil",
            render_dir=render_dir,
        )
    assert set(tmp_path.rglob("*")) == before  # nothing created in or above render_dir


def test_render_bias_matches_cli_quality_gates():
    """_RENDER_BIAS must mirror the CLI render defaults (cli.py `render` command):
    English lang bucket, speech-en content class, no sub-80ms slivers, no dupes,
    and prefer_runs OFF (word_timeline() assumes single-word picks)."""
    assert rs._RENDER_BIAS == SelectionBias(
        lang_buckets=frozenset({"en"}),
        content_classes=frozenset({"speech-en"}),
        min_word_duration_ms=80,
        exclude_dupes=True,
        prefer_runs=False,
    )


def test_render_phrase_passes_gated_bias_into_select(monkeypatch, tmp_path):
    """render_phrase must call select(..., bias=_RENDER_BIAS), not a bare
    SelectionBias() — the quality gates only take effect if the real bias
    object reaches select()."""
    _patch_pipeline(monkeypatch, tmp_path)
    captured = {}
    orig = rs.select

    def spy(*a, **k):
        captured.update(k)
        return orig(*a, **k)

    monkeypatch.setattr(rs, "select", spy)

    rs.render_phrase(
        conn=None, phrase="bias check", seed=1, version="best", render_dir=tmp_path
    )
    assert captured["bias"] is rs._RENDER_BIAS


def test_prune_render_dir_deletes_oldest_beyond_keep(tmp_path):
    paths = []
    now = time.time()
    for i in range(3):
        ogg = tmp_path / f"clip{i}.ogg"
        js = tmp_path / f"clip{i}.json"
        ogg.write_bytes(b"x")
        js.write_text("{}")
        # Stagger mtimes so clip0 is oldest, clip2 is newest.
        mtime = now - (3 - i) * 100
        os.utime(ogg, (mtime, mtime))
        os.utime(js, (mtime, mtime))
        paths.append((ogg, js))

    rs._prune_render_dir(tmp_path, keep=2)

    assert not paths[0][0].exists()  # oldest ogg pruned
    assert not paths[0][1].exists()  # its matching json pruned too
    assert paths[1][0].exists() and paths[2][0].exists()
    assert paths[1][1].exists() and paths[2][1].exists()


def test_render_phrase_prunes_render_dir_after_write(monkeypatch, tmp_path):
    """The GC call site: render_phrase must prune render_dir right after writing
    the ogg, bounding on-disk artifacts (not just the in-memory LRU)."""
    _patch_pipeline(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(rs, "_prune_render_dir", lambda d, **k: calls.append(d))
    rs.render_phrase(
        conn=None, phrase="prune wiring", seed=1, version="best", render_dir=tmp_path
    )
    assert calls == [tmp_path]


def test_render_phrase_falls_through_on_stale_cache(monkeypatch, tmp_path):
    """A cache hit whose OGG file was pruned off disk must NOT be returned as-is
    (the caller would 500 on a FileResponse to a missing path) — it must be
    evicted and re-rendered."""
    _patch_pipeline(monkeypatch, tmp_path)
    a = rs.render_phrase(
        conn=None, phrase="stale cache", seed=1, version="best", render_dir=tmp_path
    )
    assert os.path.exists(a.ogg_path)

    # Evict the file out from under the cache entry, simulating a GC prune.
    os.remove(a.ogg_path)

    b = rs.render_phrase(
        conn=None, phrase="stale cache", seed=1, version="best", render_dir=tmp_path
    )
    assert b is not a  # re-rendered, not the stale cached object
    assert os.path.exists(b.ogg_path)  # the re-render actually produced a file


def test_phrases_with_the_same_slug_keep_their_own_files(monkeypatch, tmp_path):
    _patch_pipeline(monkeypatch, tmp_path)
    a = rs.render_phrase(
        conn=None, phrase="it's here", seed=1, version="best", render_dir=tmp_path
    )
    b = rs.render_phrase(
        conn=None, phrase="it s here", seed=1, version="best", render_dir=tmp_path
    )
    assert a.ogg_path != b.ogg_path


def test_a_long_phrase_gets_a_short_file_name(monkeypatch, tmp_path):
    _patch_pipeline(monkeypatch, tmp_path)
    r = rs.render_phrase(
        conn=None, phrase="word " * 200, seed=1, version="best", render_dir=tmp_path
    )
    assert len(os.path.basename(r.ogg_path)) <= 80
