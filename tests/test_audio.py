import numpy as np

from grblr.audio import (
    load_wav,
    peak_db,
    resample,
    rms_db,
    slice_ms,
    to_mono,
)


def test_load_wav_returns_samples_and_metadata(synth_wav):
    p = synth_wav(duration_ms=500, sample_rate=16000)
    samples, sr, channels = load_wav(str(p))
    assert sr == 16000
    assert channels == 1
    assert samples.shape[0] == 8000  # 500ms * 16kHz


def test_load_wav_stereo(synth_wav):
    p = synth_wav(duration_ms=500, sample_rate=16000, channels=2)
    samples, sr, channels = load_wav(str(p))
    assert channels == 2
    assert samples.shape == (8000, 2)


def test_slice_ms_returns_correct_range():
    samples = np.arange(16000, dtype=np.float32)  # 1s at 16kHz
    sliced = slice_ms(samples, 16000, start_ms=250, end_ms=750)
    assert len(sliced) == 8000  # 500ms
    assert sliced[0] == 4000  # 250ms offset = sample 4000


def test_slice_ms_clamps_to_buffer():
    samples = np.arange(1000, dtype=np.float32)
    sliced = slice_ms(samples, 16000, start_ms=0, end_ms=999999)
    assert len(sliced) == 1000


def test_to_mono_already_mono():
    samples = np.zeros(1000, dtype=np.float32)
    assert to_mono(samples) is samples or np.array_equal(to_mono(samples), samples)


def test_to_mono_stereo_averages_channels():
    stereo = np.column_stack(
        [
            np.ones(100, dtype=np.float32),
            np.full(100, 3.0, dtype=np.float32),
        ]
    )
    mono = to_mono(stereo)
    assert mono.shape == (100,)
    assert np.allclose(mono, 2.0)


def test_resample_no_op_when_rates_match():
    samples = np.arange(1000, dtype=np.float32)
    out = resample(samples, 16000, 16000)
    assert np.array_equal(out, samples)


def test_resample_changes_length_proportionally():
    samples = np.zeros(16000, dtype=np.float32)  # 1s at 16kHz
    out = resample(samples, 16000, 48000)
    # Allow 1-sample tolerance for resampler rounding
    assert abs(len(out) - 48000) <= 1


def test_rms_db_of_silence_is_very_negative():
    samples = np.zeros(1000, dtype=np.float32)
    # Silence should clamp to a very low dB floor, not -inf
    assert rms_db(samples) <= -90.0


def test_rms_db_of_full_scale_is_zero():
    samples = np.ones(1000, dtype=np.float32)
    assert abs(rms_db(samples) - 0.0) < 0.01


def test_peak_db_of_half_scale():
    samples = np.full(1000, 0.5, dtype=np.float32)
    # 0.5 amplitude = -6.02 dBFS
    assert abs(peak_db(samples) - (-6.02)) < 0.05
