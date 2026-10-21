## Change

<!-- One or two sentences: what changes and why. Link the issue: Closes #N. -->

## Checks

- [ ] A test fails without this change and passes with it.
- [ ] `uv run pytest`, `uv run pyright` and `uv run prek run --all-files --hook-stage manual` pass.
- [ ] Render output is unchanged, or the change is intended and the goldens are updated in this pull request.
- [ ] A new model, threshold or algorithm has a decision record in `docs/decisions/`.
- [ ] New dependencies and models are MIT-compatible and listed in `MODELS.md`. No GPL code is imported in-process.

## AI use

<!-- Tool and extent, or "none". See AI_POLICY.md. -->

## Content rule

- [ ] This pull request adds no audio, transcripts or source names from a voice without consent. Test fixtures are synthetic, consented or openly licensed (name the dataset and license).
