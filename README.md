# grblr

grblr turns a corpus of voice recordings into a voice you can play.

It stitches new phrases from real recorded words: word-level concatenative
speech, not a synthetic voice. The same phrase, seed and corpus always give
the same audio.

```
grblr render "the quick brown fox" --seed 7 --out fox.wav
```

Status: v0. This repository holds the play half: word lookup, selection and
rendering. The build tools that make a corpus (transcribe, align, score,
curate) are not public yet. They open as a build recipe once it works end to
end. Until then, a corpus comes from your own tools in the [format v0](docs/format-v0.md)
layout.

## This is a tool to build your own library, we don't share or publish private, proprietary, or illegally obtained materials.

grblr publishes tools, the format and docs. It never publishes, hosts or
lists audio, transcripts or voice libraries made from someone else's voice.
Examples and demos use only openly licensed or consented voices. A library
you build from recordings you own stays on your machine.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before you open an issue. Pull
requests are invite-only for now.

If you find content in this project that uses your voice or your work without
permission, open a private report (Security tab, "Report a vulnerability")
with "Removal" in the title. We remove it.

## Install

grblr needs Python 3.11 or later. It depends on numpy and soundfile only.

```
git clone https://github.com/Phlares/grblr
cd grblr
uv sync
```

With pip: `pip install git+https://github.com/Phlares/grblr@v0.1.0`.

## Use

1. **Point grblr at a corpus.** Set `GRBLR_DB` to the SQLite file, or pass
   `--db`. [Format v0](docs/format-v0.md) describes the tables.
2. **Look up a word.** `grblr lookup hello` lists its recorded variants.
3. **Render.** `grblr render "<text>" --seed N --out out.wav` writes a WAV
   and a JSON log of every pick.
4. **Check the corpus.** `grblr stats` counts clips and words.

`grblr --help` and `grblr render --help` list every option. By default
`render` draws English speech only (`--lang en --content-classes speech-en`);
pass `--lang all --content-classes all` for a corpus that does not fill those
columns.

## How it works

| Step | What it does | Decision |
|---|---|---|
| Decompose | Split the text into words; optional syllable fallback | [0003](docs/decisions/0003-syllable-fallback-opt-in.md) |
| Select | Prefer the longest contiguous run of words from one clip | [0004](docs/decisions/0004-contiguous-runs-over-per-word-picks.md) |
| Cut | Cut at the stored word edges, with guard spans | [0002](docs/decisions/0002-keep-indexed-boundaries-guard-spans.md), [0006](docs/decisions/0006-cut-at-word-edges.md) |
| Render | Short pads, fixed gaps, edge ramps, per-pick loudness levelling | [0012](docs/decisions/0012-render-defaults.md) |

Each choice is a numbered decision record with the evidence behind it, in
[`docs/decisions/`](docs/decisions/). The long-term direction is in
[`docs/vision.md`](docs/vision.md).

## License

MIT, see [LICENSE](LICENSE). The bundled CMU Pronouncing Dictionary keeps its
own BSD license: see [NOTICE](NOTICE) and [MODELS.md](MODELS.md).
