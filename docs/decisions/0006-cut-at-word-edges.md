# 0006. Cut at word edges, not at phone midpoints

Date: 2026-06-02. Status: accepted.

## Context

Diphone synthesis cuts in the steady middle of a phone. A mid-phone cut can hide boundary error and give smoother joins than a cut at a word edge. Adopting it would move the engine from a word inventory to a sub-word inventory. We tested whether it helps on this corpus first.

## Decision

- The engine cuts at word edges only. It keeps no phone tier and no phone-midpoint cut points.
- Boundary quality work targets the word edge itself (0002), not the cut location.

## Consequences

- Units stay whole words or whole runs. Syllable assembly (0003) also uses whole words.
- A sub-word inventory stays future work, together with phoneme-level matching.

## Evidence

- The controlled, phone-matched test design.
- 400 English clips aligned with Charsiu, 7,534 phone instances, 4,240 phone-matched pairs.
  - Mean join cost: 218.8 at the word edge, 224.8 at the phone midpoint. The midpoint is ~2.8% worse. Paired Wilcoxon p = 0.014, with a tiny effect size.
  - Likely confound: word edges sit in low-energy regions, so cross-clip distances there are small.
- Same sources, blind listen on 24 trial pairs: 14 too close to tell. Of 10 decisive calls, 6 preferred the word edge and 4 the midpoint.
- Diphone coverage is not the blocker: 751 diphone types, 76.7% with 2 or more instances in 400 clips.
- Caveat: short single-seam trials have low power. A large effect would still likely have shown.
