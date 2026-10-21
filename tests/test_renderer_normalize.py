"""Per-pick loudness normalization from the buffer's MEASURED RMS, so multi-word
runs (which carry no stored rms_db) get leveled like single words do."""

import numpy as np

from grblr.renderer import normalize_buffer


def _rms_db(buf):
    rms = float(np.sqrt(np.mean(buf**2)))
    return 20.0 * np.log10(rms)


def test_quiet_buffer_amplified_toward_target_within_cap():
    # measured RMS ≈ -40 dB; target -20 dB wants +20 dB; the cap is 12 dB → ≈ -28 dB.
    buf = np.full(2000, 0.01, dtype=np.float32)
    out = normalize_buffer(buf, target_db=-20.0, max_gain_db=12.0)
    assert abs(_rms_db(out) - (-28.0)) < 0.5


def test_at_target_is_unchanged():
    buf = np.full(2000, 0.1, dtype=np.float32)  # ≈ -20 dB already
    out = normalize_buffer(buf, target_db=-20.0, max_gain_db=12.0)
    assert np.allclose(out, buf, atol=1e-4)


def test_silent_buffer_not_amplified():
    buf = np.zeros(2000, dtype=np.float32)
    out = normalize_buffer(buf, target_db=-20.0, max_gain_db=12.0)
    assert np.array_equal(out, buf)


def test_empty_buffer_safe():
    buf = np.zeros(0, dtype=np.float32)
    out = normalize_buffer(buf, target_db=-20.0, max_gain_db=12.0)
    assert out.size == 0


def test_uses_measured_rms_not_a_stored_value():
    # A loud buffer (≈ -6 dB) is attenuated toward -20 dB regardless of any
    # external rms_db — proves it measures the buffer itself (runs have rms_db=None).
    buf = np.full(2000, 0.5, dtype=np.float32)
    out = normalize_buffer(buf, target_db=-20.0, max_gain_db=12.0)
    assert _rms_db(out) < _rms_db(buf)
