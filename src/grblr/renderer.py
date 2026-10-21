from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from grblr.audio import load_wav, resample, slice_ms, to_mono
from grblr.selector import RenderPlan

log = logging.getLogger(__name__)

MissHandler = Callable[[str, int, int], np.ndarray]


def silence_handler(part: str, duration_ms: int, sample_rate: int) -> np.ndarray:
    """Default miss handler: zero-filled buffer at the target sample rate."""
    n = int(sample_rate * duration_ms / 1000)
    return np.zeros(n, dtype=np.float32)


@dataclass
class RendererConfig:
    target_sample_rate: int | None = None
    bit_depth: int = 16
    miss_handler: MissHandler = silence_handler
    pad_ms: int = 0
    """Extend each pick's slice by this many ms on both sides into the source
    clip's surrounding audio. Restores chopped consonants and brings in more
    of the original media's texture across the seams."""
    gap_ms: int = 0
    """Insert this much silence between picks for natural pacing."""
    ramp_ms: int = 0
    """Fade each pick in and out over this many ms at the edges. Kills the
    click/pop artifacts that hard cuts produce when the waveform doesn't
    land at a zero crossing."""
    normalize_rms_db: float | None = None
    """If set, scale each pick toward this target RMS loudness in dBFS
    using the stored rms_db. Levels out volume jumps between clips
    coming from different games / speakers. None disables."""
    normalize_max_gain_db: float = 12.0
    """Cap on per-pick gain when normalizing, to prevent silent clips
    from being amplified into hiss."""
    head_silence_ms: int = 0
    """Silence prepended before the first pick. Gives the sentence a
    natural lead-in instead of starting on the consonant."""
    tail_silence_ms: int = 250
    """Silence appended after the last pick. Without this, wav files
    end abruptly during the last pick's fade-out and the final word
    sounds clipped."""
    # Reserved for later, ignored in v1:
    crossfade_ms: int = 0
    gain_per_pick_db: float = 0.0
    normalize_to_lufs: float | None = None


def _apply_edge_ramps(buf: np.ndarray, sample_rate: int, ramp_ms: int) -> np.ndarray:
    """Fade in/out the first and last `ramp_ms` to suppress click artifacts
    at hard-cut boundaries. No-op if the buffer is shorter than 2x ramp."""
    n = int(sample_rate * ramp_ms / 1000)
    if n < 1 or len(buf) < 2 * n:
        return buf
    ramp_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
    ramp_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
    out = buf.copy()
    out[:n] *= ramp_in
    out[-n:] *= ramp_out
    return out


def _normalize_rms(
    buf: np.ndarray, current_rms_db: float, target_db: float, max_gain_db: float
) -> np.ndarray:
    """Scale buf toward target RMS dBFS, clamping the gain to ±max_gain_db
    so silent/SFX clips aren't blown up into hiss."""
    gain_db = target_db - current_rms_db
    gain_db = max(-max_gain_db, min(max_gain_db, gain_db))
    gain = 10.0 ** (gain_db / 20.0)
    return (buf * gain).astype(np.float32)


def normalize_buffer(
    buf: np.ndarray, target_db: float, max_gain_db: float
) -> np.ndarray:
    """Level a pick toward target RMS dBFS using the buffer's OWN measured RMS.
    Unlike the stored-`rms_db` path, this works for multi-word runs (whose
    `rms_db` is None) — so a quiet run no longer plays back softer than the words
    around it. Silent/empty buffers are returned unchanged (no hiss amplification)."""
    if buf.size == 0:
        return buf
    rms = float(np.sqrt(np.mean(buf**2)))
    if rms <= 0.0:
        return buf
    return _normalize_rms(buf, 20.0 * float(np.log10(rms)), target_db, max_gain_db)


def _decide_sample_rate(plan: RenderPlan, config: RendererConfig) -> int:
    if config.target_sample_rate is not None:
        return min(config.target_sample_rate, 48000)
    rates: list[int] = []
    for p in plan.picks:
        if p.clip_path is None:
            continue
        try:
            info = sf.info(p.clip_path)
            rates.append(info.samplerate)
        except Exception as e:
            log.warning("Cannot read sample rate from %s: %s", p.clip_path, e)
            continue
    if not rates:
        return 44100
    return min(max(rates), 48000)


def render(
    plan: RenderPlan,
    config: RendererConfig,
    out_wav: Path | str,
    out_log: Path | str,
) -> tuple[Path, Path]:
    """Concatenate picks into a WAV at out_wav and write the decision log to out_log."""
    out_wav = Path(out_wav)
    out_log = Path(out_log)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    out_log.parent.mkdir(parents=True, exist_ok=True)

    target_sr = _decide_sample_rate(plan, config)
    gap_samples = (
        np.zeros(int(target_sr * config.gap_ms / 1000), dtype=np.float32)
        if config.gap_ms > 0
        else None
    )
    buffers: list[np.ndarray] = []
    if config.head_silence_ms > 0:
        buffers.append(
            np.zeros(int(target_sr * config.head_silence_ms / 1000), dtype=np.float32)
        )

    for i, pick in enumerate(plan.picks):
        if pick.clip_path is None:
            buf = config.miss_handler(pick.part, pick.duration_ms, target_sr)
        else:
            try:
                samples, sr, _ = load_wav(pick.clip_path)
                samples = to_mono(samples)
                start_ms = max(0, (pick.start_ms or 0) - config.pad_ms)
                end_ms = (pick.end_ms or 0) + config.pad_ms
                slice_ = slice_ms(samples, sr, start_ms, end_ms)
                if sr != target_sr:
                    slice_ = resample(slice_, sr, target_sr)
                buf = slice_.astype(np.float32)
                if config.normalize_rms_db is not None:
                    # measure the buffer's own RMS so runs (rms_db=None) level too
                    buf = normalize_buffer(
                        buf, config.normalize_rms_db, config.normalize_max_gain_db
                    )
            except Exception as e:
                log.warning(
                    "Failed to load %s: %s — substituting miss", pick.clip_path, e
                )
                plan.log.append(
                    f"[{pick.part}] clip load failed ({type(e).__name__}) — substituting miss"
                )
                buf = config.miss_handler(pick.part, pick.duration_ms, target_sr)
        if config.ramp_ms > 0:
            buf = _apply_edge_ramps(buf, target_sr, config.ramp_ms)
        buffers.append(buf)
        if gap_samples is not None and i < len(plan.picks) - 1:
            buffers.append(gap_samples)
    if config.tail_silence_ms > 0:
        buffers.append(
            np.zeros(int(target_sr * config.tail_silence_ms / 1000), dtype=np.float32)
        )

    final = np.concatenate(buffers) if buffers else np.zeros(0, dtype=np.float32)
    final = np.clip(final, -1.0, 1.0)

    subtype = "PCM_16" if config.bit_depth == 16 else "PCM_24"
    sf.write(str(out_wav), final, target_sr, subtype=subtype)

    log_obj = {
        "sentence": plan.sentence,
        "seed": plan.seed,
        "rendered_at": datetime.now(UTC).isoformat(),
        "output_wav": str(out_wav.resolve()),
        "total_duration_ms": int(1000 * len(final) / target_sr),
        "target_sample_rate": target_sr,
        "picks": [asdict(p) for p in plan.picks],
        "log": list(plan.log),
    }
    out_log.write_text(json.dumps(log_obj, indent=2), encoding="utf-8")
    return out_wav, out_log
