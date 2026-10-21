import json

import numpy as np
import soundfile as sf

from grblr.renderer import RendererConfig, render, silence_handler
from grblr.selector import Pick, RenderPlan


def test_silence_handler_returns_zeros():
    out = silence_handler("missing", 250, sample_rate=16000)
    assert out.shape == (4000,)  # 250ms @ 16kHz
    assert np.all(out == 0)


def test_render_concatenates_two_clips(tmp_path, synth_wav):
    a = synth_wav(name="a.wav", duration_ms=500, sample_rate=16000, freq_hz=440)
    b = synth_wav(name="b.wav", duration_ms=300, sample_rate=16000, freq_hz=880)
    plan = RenderPlan(
        sentence="a b",
        seed=1,
        picks=[
            Pick("a", str(a), 0, 500, 500, 1, -10.0, "Test", "uniform"),
            Pick("b", str(b), 0, 300, 300, 1, -10.0, "Test", "uniform"),
        ],
    )
    out_wav = tmp_path / "out.wav"
    out_log = tmp_path / "out.json"

    wav_path, log_path = render(
        plan, RendererConfig(tail_silence_ms=0), out_wav=out_wav, out_log=out_log
    )

    samples, sr = sf.read(str(wav_path))
    assert sr == 16000
    # 800ms at 16kHz = 12800 samples (allow small tolerance for slice rounding)
    assert abs(len(samples) - 12800) < 32


def test_render_handles_miss_with_silence(tmp_path, synth_wav):
    a = synth_wav(name="a.wav", duration_ms=500)
    plan = RenderPlan(
        sentence="a x",
        seed=1,
        picks=[
            Pick("a", str(a), 0, 500, 500, 1, -10.0, "Test", "uniform"),
            Pick("x", None, None, None, 400, 0, None, None, "MISS"),
        ],
    )
    out_wav = tmp_path / "out.wav"
    out_log = tmp_path / "out.json"

    render(plan, RendererConfig(tail_silence_ms=0), out_wav=out_wav, out_log=out_log)

    samples, sr = sf.read(str(out_wav))
    # 500 + 400 = 900ms at 16kHz = 14400 samples
    assert abs(len(samples) - 14400) < 32


def test_render_writes_decision_log(tmp_path, synth_wav):
    a = synth_wav(name="a.wav", duration_ms=500)
    plan = RenderPlan(
        sentence="a",
        seed=42,
        picks=[
            Pick("a", str(a), 0, 500, 500, 7, -10.0, "src_a", "uniform random"),
        ],
    )
    out_wav = tmp_path / "out.wav"
    out_log = tmp_path / "out.json"

    render(plan, RendererConfig(), out_wav=out_wav, out_log=out_log)

    log = json.loads(out_log.read_text(encoding="utf-8"))
    assert log["sentence"] == "a"
    assert log["seed"] == 42
    assert log["total_duration_ms"] > 0
    assert log["picks"][0]["part"] == "a"
    assert log["picks"][0]["source_id"] == "src_a"
    assert log["picks"][0]["variants_available"] == 7


def test_render_resamples_mixed_rate_clips(tmp_path, synth_wav):
    a = synth_wav(name="a.wav", duration_ms=500, sample_rate=22050)
    b = synth_wav(name="b.wav", duration_ms=500, sample_rate=44100)
    plan = RenderPlan(
        sentence="a b",
        seed=1,
        picks=[
            Pick("a", str(a), 0, 500, 500, 1, -10.0, "Test", "uniform"),
            Pick("b", str(b), 0, 500, 500, 1, -10.0, "Test", "uniform"),
        ],
    )
    out_wav = tmp_path / "out.wav"
    out_log = tmp_path / "out.json"

    render(plan, RendererConfig(), out_wav=out_wav, out_log=out_log)

    _, sr = sf.read(str(out_wav))
    assert sr == 44100  # max of inputs, capped at 48k


def test_render_caps_sample_rate_at_48k(tmp_path, synth_wav):
    a = synth_wav(name="a.wav", duration_ms=500, sample_rate=48000)
    plan = RenderPlan(
        sentence="a",
        seed=1,
        picks=[
            Pick("a", str(a), 0, 500, 500, 1, -10.0, "Test", "uniform"),
        ],
    )
    out_wav = tmp_path / "out.wav"
    out_log = tmp_path / "out.json"

    render(
        plan, RendererConfig(target_sample_rate=48000), out_wav=out_wav, out_log=out_log
    )

    _, sr = sf.read(str(out_wav))
    assert sr == 48000


def test_render_treats_missing_clip_as_miss(tmp_path, synth_wav):
    """A pick whose clip_path doesn't exist should fall back to silence,
    not crash. Spec requirement: "treat it as a miss and log; do not crash"."""
    a = synth_wav(name="a.wav", duration_ms=500, sample_rate=16000)
    plan = RenderPlan(
        sentence="a missing",
        seed=1,
        picks=[
            Pick("a", str(a), 0, 500, 500, 1, -10.0, "Test", "uniform"),
            Pick(
                "missing",
                "/nonexistent/clip.wav",
                0,
                400,
                400,
                1,
                -10.0,
                "Test",
                "uniform",
            ),
        ],
    )
    out_wav = tmp_path / "out.wav"
    out_log = tmp_path / "out.json"

    # Should not raise
    render(plan, RendererConfig(tail_silence_ms=0), out_wav=out_wav, out_log=out_log)

    samples, sr = sf.read(str(out_wav))
    # 500 + 400 = 900ms at 16kHz = 14400 samples
    assert abs(len(samples) - 14400) < 64

    log = json.loads(out_log.read_text(encoding="utf-8"))
    assert any("clip load failed" in line for line in log["log"])
