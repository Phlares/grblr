"""Synthetic corpus and render snapshot shared by both golden layers.

The public layer renders from this generated corpus. A second layer can run
the same snapshot on a real corpus. Both pin render_service.render_phrase output."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf

import grblr.render_service as rs
from grblr.db import open_db

VOCAB = (
    "the a to and of in it is you that he was for on are with as i his they "
    "be at one have this from by hot word but what some we can out other were "
    "all there when up use your how said"
).split()
SR = 16000


def build_corpus(root: Path) -> Path:
    """3 sources x 6 clips of seeded tone+noise, 3-4 words each, all en/voice/speech-en."""
    root.mkdir(parents=True, exist_ok=True)
    db = root / "golden.db"
    conn = open_db(str(db))
    rng = np.random.default_rng(1234)
    cid = 0
    for src in ("src_a", "src_b", "src_c"):
        for _ in range(6):
            cid += 1
            n = int(rng.integers(3, 5))
            words = [VOCAB[int(i)] for i in rng.integers(0, len(VOCAB), n)]
            t = np.arange(int(1.5 * SR)) / SR
            f0 = float(rng.uniform(150, 400))
            audio = 0.2 * np.sin(2 * np.pi * f0 * t) * (0.5 + 0.5 * rng.random(t.size))
            wav = root / f"{cid:03d}.wav"
            sf.write(str(wav), audio.astype("float32"), SR, subtype="PCM_16")
            conn.execute(
                "INSERT INTO clips (id, path, duration_ms, source_id, category, sample_rate, "
                "channels, whisper_model, indexed_at, lang_bucket, needs_review, bucket_conf) "
                "VALUES (?,?,1500,?,'voice',?,1,'m',0,'en',0,0.9)",
                (cid, str(wav), src, SR),
            )
            span = 1400 // n
            for i, w in enumerate(words):
                s, e = 50 + i * span, 50 + (i + 1) * span - 20
                conn.execute(
                    "INSERT INTO words (clip_id, text_lower, text_original, start_ms, end_ms, "
                    "duration_ms, rms_db, peak_db) VALUES (?,?,?,?,?,?,-20,-6)",
                    (cid, w, w, s, e, e - s),
                )
            conn.execute(
                "INSERT INTO clip_content (clip_id, segment_idx, start_ms, end_ms, content_class) "
                "VALUES (?,0,0,1500,'speech-en')",
                (cid,),
            )
    conn.commit()
    conn.close()
    return db


@contextmanager
def _capture_wav_hash(sink: list):
    orig = rs.render

    def wrapped(plan, config, out_wav, out_log):
        out = orig(plan, config, out_wav=out_wav, out_log=out_log)
        data, _sr = sf.read(str(out_wav), dtype="int16")
        sink.append(hashlib.sha256(data.tobytes()).hexdigest())
        return out

    rs.render = wrapped
    try:
        yield
    finally:
        rs.render = orig


def golden_snapshot(
    db: Path, render_dir: Path, phrases: list[str], seed: int = 1, connect=open_db
) -> dict:
    """Per phrase: PCM hash of the rendered WAV (pre-OGG), karaoke words, total_ms.

    Hashes int16 PCM, not the OGG: Vorbis bytes vary by encoder build."""
    rs._CACHE.clear()
    conn = connect(str(db))
    out = {}
    try:
        for phrase in phrases:
            hashes: list[str] = []
            with _capture_wav_hash(hashes):
                r = rs.render_phrase(
                    conn, phrase, seed=seed, version=None, render_dir=render_dir
                )
            out[phrase] = {
                "pcm_sha256": hashes[0],
                "words": r.words,
                "total_ms": r.total_ms,
            }
    finally:
        conn.close()
        rs._CACHE.clear()
    return out
