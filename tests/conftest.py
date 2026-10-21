import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def synth_wav(tmp_path):
    """Factory: create a synthetic WAV file with given duration_ms and sample_rate."""

    def _make(
        name: str = "clip.wav",
        duration_ms: int = 1000,
        sample_rate: int = 16000,
        freq_hz: float = 440.0,
        amplitude: float = 0.5,
        channels: int = 1,
    ):
        n_samples = int(sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, n_samples, endpoint=False)
        wave = (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
        if channels == 2:
            wave = np.column_stack([wave, wave])
        path = tmp_path / name
        sf.write(str(path), wave, sample_rate, subtype="PCM_16")
        return path

    return _make


@pytest.fixture
def fake_transcribe():
    """Factory: a callable that returns canned Whisper-style output."""

    def _make(segments: list[dict]):
        def _transcribe(path: str):
            return segments

        return _transcribe

    return _make
