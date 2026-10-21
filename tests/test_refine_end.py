"""refine_end: cap a pick span at start + n_words*MAX_WORD_MS to kill VAD/boundary
blowouts (audit #15's 17 s 'go'). The o2o trailing-extension was reverted after a
blind A/B (bleed from sloppy whisperX onsets), so this is now a pure cap."""

from grblr.selector import refine_end


def test_caps_absurd_duration_15_bug():
    # "let me go" (3 words) span 873..17959 → cap to start + 3*1200.
    assert refine_end(17959, 873, 3, max_word_ms=1200) == 873 + 3600


def test_single_word_blowout_capped():
    assert refine_end(5000, 0, 1, max_word_ms=1200) == 1200


def test_normal_span_left_unchanged():
    # a reasonable span is returned exactly — no extension, no trailing pad.
    assert refine_end(1000, 0, 2) == 1000


def test_exactly_at_cap_unchanged():
    assert refine_end(2400, 0, 2, max_word_ms=1200) == 2400
