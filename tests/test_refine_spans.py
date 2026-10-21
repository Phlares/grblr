"""_refine_spans caps every audio pick's span in place (run + single), leaving
normal picks untouched and skipping misses. No DB lookups."""

from grblr.selector import Pick, _refine_spans


def _pick(part, start, end):
    return Pick(
        part=part,
        clip_path="c.wav",
        start_ms=start,
        end_ms=end,
        duration_ms=end - start,
        variants_available=1,
        rms_db=None,
        source_id=None,
        reason="x",
    )


def test_caps_absurd_run_15_bug():
    p = _pick("let me go", 873, 17959)
    _refine_spans([p])
    assert p.end_ms == 873 + 3 * 1200 and p.duration_ms == 3 * 1200


def test_normal_pick_unchanged():
    p = _pick("hello", 0, 500)
    _refine_spans([p])
    assert p.end_ms == 500 and p.duration_ms == 500


def test_miss_pick_untouched():
    p = Pick(
        part="x",
        clip_path=None,
        start_ms=None,
        end_ms=None,
        duration_ms=120,
        variants_available=0,
        rms_db=None,
        source_id=None,
        reason="MISS",
    )
    _refine_spans([p])
    assert p.end_ms is None and p.duration_ms == 120
