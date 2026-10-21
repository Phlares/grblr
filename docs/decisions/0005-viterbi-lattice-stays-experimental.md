# 0005. Keep the Viterbi lattice behind a flag; greedy runs stay the default

Date: 2026-06-03. Status: accepted.

## Context

Greedy leftmost-longest cover (0004) can strand a word. For target "A B C D" with runs "A B C" and "B C D", greedy takes "A B C" and leaves "D". The standard fix is unit selection with a target cost plus a join cost, solved as a min-cost path (the 1996 target-plus-join-cost formulation).

We built that lattice:
- Nodes are `(span, specific clip)`.
- Each span keeps its top 16 instances by target cost.
- Every position gets a length-1 fallback node, so the graph stays connected.
- The search minimizes total cost, then unit count, so an exact cost tie keeps the longer run.
- Join cost compares per-word average F0, RMS and spectral centroid (`AverageProxyJoinCost`).

## Decision

- The lattice ships in `lattice.py` and `cost.py`, behind `SelectionBias.prefer_lattice` (default `False`).
- No CLI flag or UI control turns it on. The eval script and tests use it.
- Greedy contiguous runs stay the default selection path.
- The target cost model and its ranking stay in use for greedy run instances (0004).

## Consequences

- The search code is ready for a better join cost. An edge-frame join cost can replace `AverageProxyJoinCost` behind the same protocol.
- Until the join cost tracks perception, the lattice adds run time and no audible gain.
- Join weights stay equal across normalized sub-costs. No perceptual regression trains them yet.

## Evidence

- A probe over the 13 eval sentences finds zero joins a min-join search can remove. Greedy is already join-optimal there. In 5 of 13 sentences an equal-join alternative exists, so only cost can choose.
- A single-blind greedy-vs-lattice render comparison showed no perceptual win for the lattice.
- The join-cost proxy agrees with listener preference 0.60 of the time against 0.50 chance (10 decisive trials). A weight grid search did not improve it.
- Target cost had `keep_prob` for ~1,887 of 796,881 units, and edge-frame features were unpopulated.
- `src/grblr/selector.py`: `prefer_lattice: bool = False`; `top_k=16`.
