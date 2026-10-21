# 0012. Render with short pads, fixed gaps, edge ramps and per-pick loudness levelling

Date: 2026-05-16. Status: accepted.

## Context

The renderer concatenates picks from many sources. Early test renders had four audible problems:
- clicks at hard cuts;
- loudness jumps of more than 15 dB between speakers;
- rushed pacing, because aligned words are tight;
- a clipped final word at the end of the file.

The v1 design kept seams raw on purpose, so each fix had to stay small.

## Decision

The renderer defaults, shared by the CLI and the live render service:

| Setting | Default | Effect |
|---|---|---|
| `pad_ms` | 5 | Widen each slice by 5 ms on both sides. Larger pads pull in neighbouring words. |
| `gap_ms` | 120 | Insert 120 ms of silence between picks. |
| `ramp_ms` | 5 | Fade each pick in and out over 5 ms. |
| `normalize_rms_db` | -20.0 | Scale each pick toward -20 dBFS RMS, measured on the pick's own buffer. |
| `normalize_max_gain_db` | 12.0 | Clamp the levelling gain to +/-12 dB, so near-silent picks do not become hiss. |
| `head_silence_ms` | 0 | No lead-in. |
| `tail_silence_ms` | 250 | Add 250 ms after the last pick. |
| sample rate | highest input rate, capped at 48 kHz | Resample every pick to it. |
| format | mono, 16-bit PCM | Downmix stereo by channel mean. |

- No crossfade. Picks abut after their ramps and gaps.
- A miss renders as silence of the corpus mean word length (0003).
- A clip that fails to load renders as a miss and is logged. The render does not fail.
- Every render writes a JSON decision log next to the audio.

## Consequences

- Seams stay audible by design. The output keeps a collage sound.
- Levelling hides the loudness contrast between sources. `--no-normalize` restores it.
- `crossfade_ms`, `gain_per_pick_db` and `normalize_to_lufs` exist on `RendererConfig` but do nothing.

## Evidence

- Hard cuts and the decision log, sample-rate rule.
- `src/grblr/renderer.py`: `RendererConfig` defaults, `_decide_sample_rate`, `normalize_buffer`.
- `src/grblr/cli.py`: `--pad-ms 5`, `--gap-ms 120`, `--ramp-ms 5`, `--normalize-db -20`, `--tail-ms 250`.
- `src/grblr/render_service.py`: `_PAD_MS = 5`, `_GAP_MS = 120`, `_RAMP_MS = 5`, `_NORMALIZE_DB = -20.0`, `_TAIL_MS = 250`.
- Commit history: pad 5 ms set on 2026-05-15; ramps, levelling and the 80 to 120 ms gap on 2026-05-16; tail silence on 2026-05-16.
- 2026-06-05: levelling switched from the stored `rms_db` to the measured buffer RMS. Multi-word runs have no stored `rms_db`, so quiet runs played softer than single words.
