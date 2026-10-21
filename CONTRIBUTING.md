# Contributing to grblr

> [!NOTE]
> grblr is a personal project. I maintain it in my free time and I read every
> issue myself. Please read this page before you open one. Thank you.

## The one rule

**Understand what you submit.** If you can't explain what your change does and
how it fits the rest of grblr without help from an AI tool, don't send it. AI
tools are fine to use; see the [AI policy](AI_POLICY.md).

## This is a tool to build your own library, we don't share or publish private, proprietary, or illegally obtained materials.

grblr publishes tools, the format and docs. It never publishes, hosts or
lists audio, transcripts or voice libraries made from someone else's voice!

In practice:

- Do not attach, link or quote audio, transcripts or a corpus built from a
  voice you do not have rights to. This applies to issues, pull requests,
  test fixtures and docs.
- Do not name the source of a private corpus (a game, a show, a person) in
  an issue or a pull request. Write "my corpus" or "a 40-hour corpus of
  one speaker".
- Test fixtures and examples use synthetic audio, your own consented voice,
  or an openly licensed dataset such as LJ Speech or LibriTTS-R. Name the
  dataset and its license.
- Bug reports can include numbers, logs and stack traces. Remove file paths
  that reveal a private source.

I remove content that breaks this rule, and may close the issue. Repeat breaks
lead to a block.

## How to contribute right now

Open an issue. That is the way in for now.

- **Bug:** use the bug form. It needs the steps, the expected result, the actual
  result and the grblr version (`uv pip show grblr`).
- **Idea:** use the idea form. Say what you can't do today, not only the fix you
  have in mind.
- **Security problem:** don't open an issue. Use "Report a vulnerability" on the
  Security tab. See [SECURITY.md](SECURITY.md).

Pull requests are invite-only for now. When I accept an issue (label
`accepted`), I either fix it or invite you to. An accepted issue is ready to
work on: the problem is understood and the approach is agreed.

If an open issue already matches yours, add a 👍 reaction instead of a "+1"
comment. Every comment emails everyone on the issue.

## If I invite you to send a pull request

1. Work against the accepted issue. One issue per pull request; keep it small.
2. Install the dev tools: `uv sync --all-groups`, then `uv run prek install`
   for the commit hooks.
3. Write a failing test first, then the code that makes it pass.
4. Run the checks before you push:
   ```
   uv run pytest
   uv run pyright
   uv run prek run --all-files --hook-stage manual
   ```
5. Fill in the pull request template. Pull requests are not the place to
   design a feature; that happens on the issue.

## Rules for code

- Render output must not change by accident. The golden render tests pin it.
  If a change is meant to alter output, say so in the pull request and update
  the goldens in the same commit.
- A new model, threshold or algorithm needs a decision record in
  `docs/decisions/`, with the numbers that justify it.
- New dependencies must have a license compatible with MIT. GPL code may not be
  imported in-process. List each new model and its license in `MODELS.md`.
- The engine depends on numpy and soundfile only. Open an issue before you
  add a dependency.

## License of contributions

By contributing, you agree that your contribution is licensed under the MIT
license of this project.

## Conduct

Everyone in this project follows the [code of conduct](CODE_OF_CONDUCT.md).
