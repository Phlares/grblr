# 0004. Prefer the longest contiguous corpus run over per-word picks

Date: 2026-05-31. Status: accepted.

## Context

Short function words ("to", "of", "the", "and") are reduced in connected speech. Sliced out alone, they are blips, not words. No aligner can fix this: the full word was never in the audio. Each per-word pick also adds a seam, and every seam exposes boundary error (0002).

## Decision

- With `SelectionBias.prefer_runs = True`, the selector covers each stretch of literal words with the longest run that occurs contiguously in one clip.
  - The cover is greedy, leftmost-longest.
  - A run is one slice `[first_word.start, last_word.end]` from one clip. It has no internal cuts.
  - `run_select.build_position_index` builds an in-memory `(clip, position)` index over the filtered corpus.
  - Where no run of 2 or more words exists, the word falls back to the per-word path.
- Runs obey the language, content and duplicate filters. The 80 ms floor applies only to single-word fallbacks, so a reduced word inside a run stays.
- Among clips that hold the same run, the selector picks the lowest mean target cost. The cost reads the optional `word_quality` and `clip_filter` inputs. A seeded RNG breaks exact ties.
- The CLI (`--runs`) and the local UI default to runs on.
- The live render service keeps runs off (`prefer_runs=False`). Its karaoke timeline needs one pick per word.

## Consequences

- Fewer seams, and reduced words keep their real coarticulation.
- Output has less voice variety per sentence, because neighbouring words often come from one speaker. `--no-runs` restores per-word variety.
- Two render paths exist. CLI renders and live renders of the same phrase and seed differ.
- Turning runs on in the live service first needs `word_timeline` to split run picks into words.

## Evidence

- The design and the reduced-word argument.
- On "i need to get out of the city before night", joins fall from 9 to 2 with one 5-word run. A blind listen judged greedy runs a clear intelligibility win.
- Over the 13 eval sentences, greedy leaves zero isolated function-word singles.
- `src/grblr/run_select.py`: `cover_target`, `pick_run_ranked` (quality-ranked instance choice, added 2026-06-01 in place of a seeded-uniform pick).
- `src/grblr/cli.py`: `--runs` uses `BooleanOptionalAction` with `default=True` (2026-06-01).
- `src/grblr/render_service.py`: `_RENDER_BIAS` sets `prefer_runs=False`; `word_timeline` assumes single-word picks.
