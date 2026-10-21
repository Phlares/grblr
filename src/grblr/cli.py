from __future__ import annotations

import argparse
import logging
import random
import re
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

from grblr import settings
from grblr.db import open_db
from grblr.decomposer import DecomposerConfig, decompose
from grblr.normalize import normalize
from grblr.renderer import RendererConfig, render
from grblr.selector import MissStrategy, SelectionBias, select

AddCommands = Callable[[Any], None]


def make_parser(
    prog: str = "grblr", add_commands: AddCommands | None = None
) -> argparse.ArgumentParser:
    """The render commands, plus any `add_commands(subparsers)` registers.
    Each command stores its handler as `run` (set_defaults)."""
    p = argparse.ArgumentParser(
        prog=prog, description="Concatenative speech synthesis."
    )
    p.add_argument(
        "--db",
        default=settings.db_path(),
        help="path to corpus SQLite database (default: $GRBLR_DB)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_lookup = sub.add_parser("lookup", help="List variants of a word in the corpus.")
    p_lookup.add_argument("word")
    p_lookup.set_defaults(run=lambda a: cmd_lookup(a.word, a.db))

    p_render = sub.add_parser("render", help="Render a sentence to WAV.")
    p_render.add_argument("text")
    p_render.add_argument("--seed", type=int, default=None)
    p_render.add_argument("--out", default=None)
    p_render.add_argument("--log", default=None)
    p_render.add_argument(
        "--pad-ms",
        type=int,
        default=5,
        help="extend each word slice by N ms on both sides "
        "(default 5; aligned boundaries are tight enough "
        "that bigger pads drag in adjacent-word artifacts)",
    )
    p_render.add_argument(
        "--gap-ms",
        type=int,
        default=120,
        help="insert N ms of silence between words for pacing "
        "(default 120; set to 0 for constant-bark style)",
    )
    p_render.add_argument(
        "--ramp-ms",
        type=int,
        default=5,
        help="fade each pick in/out over N ms to suppress "
        "click artifacts at hard-cut boundaries (default 5)",
    )
    p_render.add_argument(
        "--normalize-db",
        type=float,
        default=-20.0,
        help="RMS-normalize each pick toward this dBFS level "
        "to even out volume between games/speakers "
        "(default -20; pass a large negative number "
        "like -100 to effectively disable)",
    )
    p_render.add_argument(
        "--no-normalize",
        action="store_true",
        help="disable RMS normalization entirely (preserves "
        "original loudness contrast between picks)",
    )
    p_render.add_argument(
        "--prefer-longer",
        action="store_true",
        help="bias variant selection toward longer (more "
        "enunciated) takes of each word — useful when "
        "short syllables yanked from larger barks are "
        "perceptually too brief to register",
    )
    p_render.add_argument(
        "--prefer-deep",
        action="store_true",
        help="bias toward lower-pitched (deeper) takes",
    )
    p_render.add_argument(
        "--prefer-high", action="store_true", help="bias toward higher-pitched takes"
    )
    p_render.add_argument(
        "--prefer-bright",
        action="store_true",
        help="bias toward brighter (higher spectral centroid) takes",
    )
    p_render.add_argument(
        "--prefer-dark",
        action="store_true",
        help="bias toward darker (lower spectral centroid) takes",
    )
    p_render.add_argument(
        "--pitch-min",
        type=float,
        default=None,
        help="reject takes with median F0 below this Hz "
        "(hard filter — useful to exclude unusually deep barks)",
    )
    p_render.add_argument(
        "--pitch-max",
        type=float,
        default=None,
        help="reject takes with median F0 above this Hz "
        "(hard filter — useful to exclude screechy takes)",
    )
    p_render.add_argument(
        "--min-voiced",
        type=float,
        default=None,
        help="reject takes whose voiced fraction is below this "
        "(0.0-1.0; filters out grunts/screams/breath noise)",
    )
    p_render.add_argument(
        "--syllable-split",
        action="store_true",
        help="when a word isn't in the corpus, attempt to "
        "assemble it from corpus words whose phonemes "
        "match each of its syllables (CAT-ASS-TROPHY mode)",
    )
    p_render.add_argument(
        "--miss-mode",
        choices=["silence", "random", "skip", "sound-alike"],
        default="silence",
        help="what to do when a word isn't in the corpus and "
        "syllable-split also failed: 'silence' (default), "
        "'random' substitutes a random corpus word, "
        "'skip' drops the word entirely, 'sound-alike' "
        "substitutes the phonetically nearest corpus word",
    )
    p_render.add_argument(
        "--head-ms",
        type=int,
        default=0,
        help="leading silence before the first pick (default 0)",
    )
    p_render.add_argument(
        "--tail-ms",
        type=int,
        default=250,
        help="trailing silence after the last pick "
        "(default 250; prevents the last word "
        "from sounding clipped at the wav file end)",
    )
    p_render.add_argument(
        "--lang",
        default="en",
        help="comma-separated v2.2 lang_buckets to draw from "
        "(default 'en' — English-only corpus selection). "
        "Use 'en,unknown' to include the recoverable bucket, "
        "'all' to disable language filtering entirely.",
    )
    p_render.add_argument(
        "--include-dupes",
        action="store_true",
        help="include sequestered dupe_of clips. Default OFF "
        "because their audio file was moved to <game>/_dupe/ "
        "and loading them would fail. Debug-only.",
    )
    p_render.add_argument(
        "--min-word-ms",
        type=int,
        default=80,
        help="reject word variants shorter than N ms. wav2vec2-CTC "
        "boundary errors can land mid-phoneme producing 20-60ms "
        "slices that sound like static. Default 80 (drops ~8%% "
        "of en-bucket variants); pass 0 to disable.",
    )
    p_render.add_argument(
        "--content-classes",
        default="speech-en",
        help="comma-separated PANNs content_class allowlist. Default "
        "'speech-en' (drops ~3.6%% of en-bucket variants) — "
        "removes clips whose audio was classified as music / "
        "sung / speech-other / speech-ja but whose lang_bucket "
        "ended up 'en' via path or dir prior (Japanese audio in "
        "an English folder). Pass 'all' to disable.",
    )
    p_render.add_argument(
        "--runs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="contiguous-run unit selection: stitch the longest multi-word "
        "runs available from a single clip, keeping reduced function "
        "words in their natural coarticulation. Default ON "
        "(intelligible); --no-runs reverts to single-word selection "
        "(more chaotic, more variety).",
    )

    p_render.set_defaults(run=_run_render)

    sub.add_parser("stats", help="Show corpus statistics.").set_defaults(
        run=lambda a: cmd_stats(a.db)
    )
    if add_commands is not None:
        add_commands(sub)
    return p


def _slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return s[:max_len] or "render"


def cmd_lookup(word: str, db: str) -> int:
    conn = open_db(db)
    rows = conn.execute(
        "SELECT c.path, w.start_ms, w.end_ms, w.duration_ms, w.rms_db, c.source_id "
        "FROM words w JOIN clips c ON c.id = w.clip_id "
        "WHERE w.text_lower = ? ORDER BY c.path, w.start_ms",
        (normalize(word),),
    ).fetchall()
    conn.close()
    if not rows:
        print(f"No variants found for '{word}'.")
        return 1
    print(f"{len(rows)} variant(s) for '{word}':")
    for path, start_ms, end_ms, dur, rms, game in rows:
        print(
            f"  [{game or '?'}] {path}  {start_ms / 1000:.2f}s-{end_ms / 1000:.2f}s  "
            f"({dur}ms, {rms:.1f}dBFS)"
        )
    return 0


def cmd_stats(db: str) -> int:
    conn = open_db(db)
    n_clips = conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0]
    n_words = conn.execute("SELECT COUNT(*) FROM words").fetchone()[0]
    n_unique = conn.execute("SELECT COUNT(DISTINCT text_lower) FROM words").fetchone()[
        0
    ]
    top = conn.execute(
        "SELECT text_lower, COUNT(*) c FROM words GROUP BY text_lower ORDER BY c DESC LIMIT 10"
    ).fetchall()
    conn.close()
    print(f"Corpus: {n_clips} clip(s), {n_words} word(s), {n_unique} unique word(s).")
    if top:
        print("Top 10 most-common words:")
        for text, count in top:
            print(f"  {text}: {count}")
    return 0


def cmd_render(
    text: str,
    db: str,
    out: str | None,
    log: str | None,
    seed: int | None,
    pad_ms: int = 5,
    gap_ms: int = 120,
    ramp_ms: int = 5,
    normalize_db: float | None = -20.0,
    prefer_longer: bool = False,
    prefer_deep: bool = False,
    prefer_high: bool = False,
    prefer_bright: bool = False,
    prefer_dark: bool = False,
    pitch_min: float | None = None,
    pitch_max: float | None = None,
    min_voiced: float | None = None,
    syllable_split: bool = False,
    head_ms: int = 0,
    tail_ms: int = 250,
    miss_mode: MissStrategy = "silence",
    lang: str = "en",
    include_dupes: bool = False,
    min_word_ms: int = 80,
    content_classes: str = "speech-en",
    prefer_runs: bool = True,
) -> int:
    conn = open_db(db)
    seed = seed if seed is not None else random.randint(0, 2**31 - 1)

    plan = decompose(
        text,
        conn,
        DecomposerConfig(
            rng_seed=seed,
            enable_syllable_split=syllable_split,
        ),
    )
    lang_buckets: frozenset[str] | None
    if lang == "all":
        lang_buckets = None
    else:
        lang_buckets = frozenset(b.strip() for b in lang.split(",") if b.strip())
    content_cls: frozenset[str] | None
    if content_classes == "all":
        content_cls = None
    else:
        content_cls = frozenset(
            c.strip() for c in content_classes.split(",") if c.strip()
        )
    bias = SelectionBias(
        prefer_longer=prefer_longer,
        prefer_deep=prefer_deep,
        prefer_high=prefer_high,
        prefer_bright=prefer_bright,
        prefer_dark=prefer_dark,
        pitch_min_hz=pitch_min,
        pitch_max_hz=pitch_max,
        min_voiced_fraction=min_voiced,
        lang_buckets=lang_buckets,
        exclude_dupes=not include_dupes,
        min_word_duration_ms=(min_word_ms if min_word_ms > 0 else None),
        content_classes=content_cls,
        prefer_runs=prefer_runs,
    )
    rp = select(plan, conn, seed=seed, bias=bias, miss_strategy=miss_mode)

    today = date.today().isoformat()
    slug = _slug(text)
    default_out = str(settings.data_dir("renders") / f"{today}_{slug}_{seed}.wav")
    default_log = str(settings.data_dir("renders") / f"{today}_{slug}_{seed}.json")

    out_wav = Path(out) if out else Path(default_out)
    out_log = Path(log) if log else Path(default_log)

    render(
        rp,
        RendererConfig(
            pad_ms=pad_ms,
            gap_ms=gap_ms,
            ramp_ms=ramp_ms,
            normalize_rms_db=normalize_db,
            head_silence_ms=head_ms,
            tail_silence_ms=tail_ms,
        ),
        out_wav=out_wav,
        out_log=out_log,
    )
    print(f"Wrote {out_wav}")
    print(f"Log  {out_log}")
    conn.close()
    return 0


def _run_render(a: argparse.Namespace) -> int:
    normalize_db = None if a.no_normalize else a.normalize_db
    return cmd_render(
        a.text,
        a.db,
        a.out,
        a.log,
        a.seed,
        pad_ms=a.pad_ms,
        gap_ms=a.gap_ms,
        ramp_ms=a.ramp_ms,
        normalize_db=normalize_db,
        prefer_longer=a.prefer_longer,
        prefer_deep=a.prefer_deep,
        prefer_high=a.prefer_high,
        prefer_bright=a.prefer_bright,
        prefer_dark=a.prefer_dark,
        pitch_min=a.pitch_min,
        pitch_max=a.pitch_max,
        min_voiced=a.min_voiced,
        syllable_split=a.syllable_split,
        head_ms=a.head_ms,
        tail_ms=a.tail_ms,
        miss_mode=a.miss_mode,
        lang=a.lang,
        include_dupes=a.include_dupes,
        min_word_ms=a.min_word_ms,
        content_classes=a.content_classes,
        prefer_runs=a.runs,
    )


def main(
    argv: list[str] | None = None,
    prog: str = "grblr",
    add_commands: AddCommands | None = None,
) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = make_parser(prog, add_commands)
    args = parser.parse_args(argv)
    if args.db is None:
        parser.error("set --db or GRBLR_DB to the corpus database")
    args.db = str(args.db)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
