# Decision records

One decision per record, with the date it was made and the numbers behind it. A record describes what the code does today.

Each record summarizes its evidence and quotes the numbers it relies on. Evidence
with no path comes from research notes kept outside this repository.

- [0002](0002-keep-indexed-boundaries-guard-spans.md) Keep the indexed word boundaries and guard spans at render time. Accepted.
- [0003](0003-syllable-fallback-opt-in.md) Make syllable assembly an opt-in fallback; render misses as silence. Accepted.
- [0004](0004-contiguous-runs-over-per-word-picks.md) Prefer the longest contiguous corpus run over per-word picks. Accepted.
- [0005](0005-viterbi-lattice-stays-experimental.md) Keep the Viterbi lattice behind a flag; greedy runs stay the default. Accepted.
- [0006](0006-cut-at-word-edges.md) Cut at word edges, not at phone midpoints. Accepted.
- [0012](0012-render-defaults.md) Render with short pads, fixed gaps, edge ramps and per-pick loudness levelling. Accepted.
- [0013](0013-deterministic-render-service.md) Make a live render a pure function of phrase, seed, version and corpus. Accepted.
