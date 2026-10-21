from __future__ import annotations

import numpy as np
import soundfile as sf

_DB_FLOOR = -120.0


def load_wav(path: str) -> tuple[np.ndarray, int, int]:
    """Load a WAV file. Returns (samples, sample_rate, channels).

    Samples are float32 in [-1, 1]. Mono shape: (N,). Stereo shape: (N, 2).
    """
    samples, sr = sf.read(path, dtype="float32", always_2d=False)
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    return samples, sr, channels


def slice_ms(
    samples: np.ndarray, sample_rate: int, start_ms: int, end_ms: int
) -> np.ndarray:
    """Slice samples by millisecond range. Clamps to buffer length."""
    start = max(0, int(sample_rate * start_ms / 1000))
    end = min(len(samples), int(sample_rate * end_ms / 1000))
    if start >= end:
        return samples[start:start]
    return samples[start:end]


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Downmix to mono by averaging channels. Already-mono passes through."""
    if samples.ndim == 1:
        return samples
    return samples.mean(axis=1).astype(samples.dtype)


def resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Linear resample. Sufficient for v1 concatenation; no anti-aliasing."""
    if from_rate == to_rate:
        return samples
    n_out = int(round(len(samples) * to_rate / from_rate))
    if n_out <= 0:
        return np.zeros(0, dtype=samples.dtype)
    src_idx = np.linspace(0, len(samples) - 1, n_out)
    return np.interp(src_idx, np.arange(len(samples)), samples).astype(samples.dtype)


def rms_db(samples: np.ndarray) -> float:
    """RMS amplitude in dBFS. Returns _DB_FLOOR for silence."""
    if len(samples) == 0:
        return _DB_FLOOR
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    if rms <= 0:
        return _DB_FLOOR
    return max(_DB_FLOOR, 20.0 * np.log10(rms))


def peak_db(samples: np.ndarray) -> float:
    """Peak amplitude in dBFS. Returns _DB_FLOOR for silence."""
    if len(samples) == 0:
        return _DB_FLOOR
    peak = float(np.max(np.abs(samples)))
    if peak <= 0:
        return _DB_FLOOR
    return max(_DB_FLOOR, 20.0 * np.log10(peak))
