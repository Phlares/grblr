# src/grblr/render_service.py
"""Live phrase rendering for the game's karaoke reward (Part A).

Orchestrates the CLI render pipeline (decompose -> select -> render) over the
shared corpus connection, encodes OGG, and derives a per-word karaoke timeline.
Deterministic given (phrase, seed, version)."""

from __future__ import annotations

import hashlib
import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from grblr.decomposer import DecomposerConfig, decompose
from grblr.renderer import RendererConfig, render
from grblr.selector import Pick, SelectionBias, select

MODEL_VERSION = "best-effort-v0"

# Render shape: short, gapped, lightly padded — matches the CLI defaults that
# produce the cleanest short phrases. Tunable here, single source of truth.
_PAD_MS = 5
_GAP_MS = 120
_HEAD_MS = 0
_TAIL_MS = 250
_RAMP_MS = 5
_NORMALIZE_DB = -20.0


# `version` is interpolated into on-disk render filenames, so it must never carry
# path separators or arbitrary values. Constrain it to a small allowlist of model
# versions we control (extend as fine-tuned versions ship) — this closes the
# ?version= path-traversal / arbitrary-file-write vector at the single resolution
# point. Callers translate the ValueError to a 400.
_ALLOWED_VERSIONS = {MODEL_VERSION}

# Quality gates matching the CLI render defaults (cli.py `render` command):
# English lang bucket, speech-en content class, no sub-80ms slivers, no dupes.
# prefer_runs stays False — word_timeline() depends on per-word picks.
_RENDER_BIAS = SelectionBias(
    lang_buckets=frozenset({"en"}),
    content_classes=frozenset({"speech-en"}),
    min_word_duration_ms=80,
    exclude_dupes=True,
    prefer_runs=False,
)

# Bound on-disk render artifacts: the in-memory LRU (_CACHE_MAX) only bounds RAM,
# not disk — every render still writes an .ogg + .json to render_dir. Keep the
# newest _DISK_KEEP of each; older ones are pruned after every render.
_DISK_KEEP = 256


def _prune_render_dir(render_dir: Path, keep: int = _DISK_KEEP) -> None:
    """Keep the newest `keep` OGGs (and their JSON logs) in render_dir; delete
    the rest. The in-memory LRU only bounds RAM, not disk."""

    def mtime(p: Path) -> float:
        # A concurrent request's prune can delete a file between glob and stat.
        try:
            return p.stat().st_mtime
        except FileNotFoundError:
            return 0.0

    oggs = sorted(render_dir.glob("*.ogg"), key=mtime)
    for old in oggs[: max(0, len(oggs) - keep)]:
        old.unlink(missing_ok=True)
        old.with_suffix(".json").unlink(missing_ok=True)


def resolve_version(v: str | None) -> str:
    resolved = MODEL_VERSION if v in (None, "best") else v
    if resolved not in _ALLOWED_VERSIONS:
        raise ValueError(f"unknown version: {resolved!r}")
    return resolved


def versions() -> dict:
    """The model-version registry (GDD §10.6 evolution-view seed): the active
    serving version + everything renderable via ?version=. Grows per fine-tune."""
    return {"active": MODEL_VERSION, "available": sorted(_ALLOWED_VERSIONS)}


@dataclass(frozen=True)
class RenderResult:
    ogg_path: str
    words: list
    seed: int
    version: str
    total_ms: int


_CACHE_MAX = 128
_CACHE: OrderedDict[tuple[str, int, str], RenderResult] = OrderedDict()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "phrase"


def render_phrase(
    conn, phrase: str, *, seed: int | None, version: str | None, render_dir
) -> RenderResult:
    version = resolve_version(version)
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    key = (phrase, seed, version)
    if key in _CACHE:
        # A cache hit whose OGG was pruned off disk (see _prune_render_dir) must
        # NOT be returned — /api/render/audio would 500 on a FileResponse to a
        # missing path. Evict and fall through to a re-render instead.
        if not Path(_CACHE[key].ogg_path).exists():
            del _CACHE[key]
        else:
            _CACHE.move_to_end(key)
            return _CACHE[key]

    render_dir = Path(render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)
    rp = select(
        decompose(phrase, conn, DecomposerConfig(rng_seed=seed)),
        conn,
        seed=seed,
        bias=_RENDER_BIAS,
        miss_strategy="silence",
    )

    # The slug is lossy ("it's" and "it s" match) and unbounded, so a phrase hash
    # keeps each cache entry on its own file and the name short.
    digest = hashlib.sha256(phrase.encode()).hexdigest()[:8]
    stem = f"{_slug(phrase)[:40]}-{version}-{seed}-{digest}"
    wav = render_dir / f"{stem}.wav"
    log = render_dir / f"{stem}.json"
    ogg = render_dir / f"{stem}.ogg"
    # Defense in depth: `version` is allowlisted and `phrase` is slugged, so the
    # stem already cannot contain separators — but assert every output path stays
    # inside render_dir before any write/unlink, so a future loosening can't
    # silently reintroduce traversal.
    root = render_dir.resolve()
    for p in (wav, log, ogg):
        if not p.resolve().is_relative_to(root):
            raise ValueError("render path escaped render_dir")
    cfg = RendererConfig(
        pad_ms=_PAD_MS,
        gap_ms=_GAP_MS,
        ramp_ms=_RAMP_MS,
        normalize_rms_db=_NORMALIZE_DB,
        head_silence_ms=_HEAD_MS,
        tail_silence_ms=_TAIL_MS,
    )
    render(rp, cfg, out_wav=wav, out_log=log)

    data, sr = sf.read(str(wav))
    wav.unlink(missing_ok=True)
    sf.write(str(ogg), data, sr, format="OGG", subtype="VORBIS")
    _prune_render_dir(render_dir)

    words = word_timeline(
        rp.picks, head_silence_ms=_HEAD_MS, pad_ms=_PAD_MS, gap_ms=_GAP_MS
    )
    total_ms = int(1000 * len(data) / sr) if sr else 0
    result = RenderResult(
        ogg_path=str(ogg), words=words, seed=seed, version=version, total_ms=total_ms
    )
    _CACHE[key] = result
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return result


def _pick_dur_ms(p: Pick, pad_ms: int) -> int:
    if p.clip_path is None or p.start_ms is None or p.end_ms is None:
        return p.duration_ms
    return (p.end_ms - p.start_ms) + 2 * pad_ms


def word_timeline(
    picks, *, head_silence_ms: int, pad_ms: int, gap_ms: int
) -> list[dict]:
    """Per-word [{text, onset_ms, dur_ms}] mirroring renderer buffer assembly.

    Emits one karaoke entry per pick and therefore assumes single-word picks
    (depends on the render bias keeping runs/lattice OFF).  A future bias change
    that enables multi-word runs would need this function to split run picks into
    per-word entries.
    """
    out: list[dict] = []
    cursor = head_silence_ms
    for i, p in enumerate(picks):
        dur = _pick_dur_ms(p, pad_ms)
        out.append({"text": p.part, "onset_ms": cursor, "dur_ms": dur})
        cursor += dur
        if i < len(picks) - 1:
            cursor += gap_ms
    return out
