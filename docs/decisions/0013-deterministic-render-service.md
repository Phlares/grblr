# 0013. Make a live render a pure function of phrase, seed, version and corpus

Date: 2026-06-29. Status: accepted.

## Context

The live game renders phrases on request and plays them with a word-by-word karaoke timeline. Players replay and share renders. Refactors and the engine split must not change output. Both needs require that one input gives one output.

## Decision

- `render_service.render_phrase(conn, phrase, seed, version, render_dir)` runs decompose, select and render, then encodes OGG Vorbis.
- All randomness comes from `random.Random(seed)`. A request without a seed draws one and returns it, so the render can be replayed.
- `version` must be in an allowlist (today only `best-effort-v0`). `best` and no value map to the active version. Any other value raises `ValueError`.
- The service uses one fixed selection bias:
  - English bucket and `speech-en` content (the optional `clips.lang_bucket` and `clip_content` inputs);
  - 80 ms word floor and duplicate exclusion (0002; `clips.dupe_of`);
  - per-word picks, runs off (0004).
- The service uses the render defaults of 0012.
- `word_timeline` returns `{text, onset_ms, dur_ms}` per pick. It mirrors the renderer's buffer layout: head silence, pad, gap.
- An in-memory LRU of 128 entries caches results by `(phrase, seed, version)`. A cache hit whose OGG file is gone re-renders.
- The render folder keeps the newest 256 OGG and JSON pairs.

## Consequences

- Output depends on the corpus too. A changed database or re-scored words can change a render with the same phrase and seed.
- The JSON decision log carries a wall-clock `rendered_at`. The audio does not.
- A new selection model needs a new version string, so old seeds keep their old sound.
- Two golden tests pin output. A public one renders from a generated corpus. A private one renders from the real corpus.

## Evidence

- `src/grblr/render_service.py`: `MODEL_VERSION`, `_ALLOWED_VERSIONS`, `_RENDER_BIAS`, `_CACHE_MAX = 128`, `_DISK_KEEP = 256`, `word_timeline`.
- `tests/test_render_service.py`: cache keyed by phrase, seed and version; version allowlist and path-traversal rejection; bias parity with the CLI gates.
- `tests/golden/test_render_golden.py` and `tests/golden/render_v0.json`: 8 phrases over a 3-source, 18-clip generated corpus, plus a repeat-run determinism test (added 2026-10-20).
- A real-corpus golden check that the maintainer runs outside this repository.
- Commit history: `word_timeline` on 2026-06-29; version allowlist on 2026-06-29; CLI-parity quality gates and the disk bound on 2026-07-01.
- The engine-split design: render output must not change when the engine moves to its own repository.
