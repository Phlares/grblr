# 0003. Make syllable assembly an opt-in fallback; render misses as silence

Date: 2026-05-16. Status: accepted.

## Context

A target word that is absent from the corpus is a miss. v1 rendered every miss as silence. The v1.1 design added a syllable fallback: split the missing word into syllables and fill each syllable with a whole corpus word whose phonemes match. It also added a sound-alike substitute. The v1.1 design planned the syllable fallback as the default.

## Decision

- The decomposer marks a word `literal` when any `words` row has its normalized text. Otherwise it is a miss.
- Syllable assembly is off by default. `DecomposerConfig.enable_syllable_split` defaults to `False`. The CLI turns it on with `--syllable-split`.
- Syllable assembly works like this:
  - Look up the word in the bundled CMU Pronouncing Dictionary. Use the first pronunciation.
  - Split the phonemes at vowel nuclei. Give consonant clusters to the next syllable when they form a legal English onset.
  - Match each syllable to a whole corpus word, in this order: exact match, prefix match, containment, then nearest by stress-free phoneme edit distance.
  - If any syllable has no match, the word stays a miss.
- Assembly never slices inside a word. Each part is a whole indexed word, so it passes the same filters and quality ranking as a literal word.
- Miss handling is a separate choice (`--miss-mode`): `silence` (default), `skip`, `random` corpus word, or `sound-alike` (nearest word by ARPAbet edit distance).
- The live render service uses no syllable assembly and renders misses as silence.

## Consequences

- Default renders never invent a word from parts. A missing word is an audible gap of the corpus mean word length (400 ms when the corpus is empty).
- A word outside the CMU dictionary cannot be assembled or sound-alike matched. It stays a miss.
- The decomposer counts unfiltered rows. A word that exists only in filtered-out clips is `literal` at decompose time and becomes a miss at select time.
- Sub-word slicing is still future work. The scoring code is span-agnostic, so it can score sub-word spans later.

## Evidence

- The syllable and sound-alike design. It planned the fallback as default on.
- Assembled parts are whole words and go through the same quality gating.
- `src/grblr/decomposer.py`: `enable_syllable_split: bool = False`, `_resolve_syllable` match order.
- `src/grblr/cli.py`: `--syllable-split` is `store_true`; `--miss-mode` defaults to `silence`.
- `src/grblr/render_service.py`: `DecomposerConfig(rng_seed=seed)` and `miss_strategy="silence"`.
- `tests/golden/test_render_golden.py`: the phrase "we were here" pins the miss path.
