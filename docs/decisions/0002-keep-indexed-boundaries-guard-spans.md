# 0002. Keep the indexed word boundaries and guard spans at render time

Date: 2026-06-02. Status: accepted.

## Context

wav2vec2 alignment has a ~20 ms frame stride. Most word edges are close to right. A minority are far off and clip an onset or run into silence. We ran an aligner bake-off against hand-labeled edges to decide whether to re-align the corpus.

The hand labels start from the whisperX span and the annotator nudges it. So the truth is anchored to the baseline. A flat mean error favours the baseline. We judge only the edges the annotator moved 50 ms or more ("big" and "huge"), where anchoring cannot help the baseline.

## Decision

- The `words` table keeps the whisperX + wav2vec2 boundaries. No aligner re-indexes the corpus.
- The selector guards spans at render time instead:
  - `min_word_duration_ms = 80` drops single-word variants shorter than 80 ms. These are mostly mid-phoneme slivers.
  - `refine_end` caps every pick at `start + n_words * 1200 ms`. This removes multi-second blowouts.
  - `max_word_duration_ms` exists as an opt-in ceiling. It is off by default.
- The selector does not extend a word end toward the next word's onset. A blind A/B rejected that.
- Contiguous runs (0004) avoid internal cuts, so internal boundary error does not reach the output.

## Consequences

- The ~15% of bad edges stay in the corpus. Renders still contain some cut-off and too-long words.
- No cheap signal finds the bad edges, so a targeted re-align has no trigger.
- The best boundary source we measured is the median of a five-aligner ensemble. It is not built into the engine. It is the next step if boundaries become the limit again.
- Scores and filters computed on current spans stay valid, because spans do not move.

## Evidence

- 150 hand-labeled clips, 598 words, 1,196 edges. 1,017 edges (85%) were accepted or barely moved.
- Word-edge MAE in ms on moved edges.
  - Big bucket (50-100 ms, n = 151): baseline 64.3, MFA 52.4, ctc-forced-aligner 74.9, MMS_FA 36.0, Charsiu 37.9.
  - Huge bucket (> 100 ms, n = 28): baseline 128.2, MFA 85.4, ctc-forced-aligner 152.9, MMS_FA 86.2, Charsiu 78.2.
  - No aligner reaches the 10-15 ms bar. All share a ~25-30 ms floor.
- Same report, auto-targeting test (2026-06-03): no cheap signal separates bad edges.
  - Confidence: 0.876 good, 0.859 big, 0.830 huge.
  - A 20 ms disagreement gate catches 75% of bad edges and false-flags 45% of good ones.
  - A 75 ms gate false-flags ~6% but catches only ~12%.
- Same report, addendum (2026-06-03 to 2026-06-04): label-prior CTC wins the huge bucket (median 30 ms closer than MMS_FA, 95% CI [-61, -12], n = 22). It loses a 45-trial blind check on good edges: listeners preferred MMS_FA 25 to 18 overall and 18 to 6 in round 2 (p = 0.023). It runs long and bleeds.
- Five-aligner ensemble. The median is a tie with MMS_FA on big and huge edges (median delta -5.5 ms, CI [-10.5, 0.0]). Ensemble spread predicts error (Spearman 0.438, p = 1.2e-50; top spread quintile has 3.2x the error of the bottom).
- On a cut-off and bleed metric, MMS_FA clips 35.8% of words short. CrisperWhisper onset-to-onset beats MMS_FA in 22 of 22 decidable blind context votes (p ~ 5e-7) and ties the ensemble. CrisperWhisper is CC-BY-NC. Granite has ~26% bad edges.
- A cloud speech model ends all 22 matched words late (median +52 ms) on a coarse grid. It is not a boundary source.
- `src/grblr/selector.py` (`refine_end`): the onset-to-onset end extension lost a blind A/B 17 to 3 (p = 0.003) on 2026-06-05, because whisperX onsets are ~60 ms off on defect edges.
- One real cut-off. The aligned start is 220 ms after the word's energy onset.
