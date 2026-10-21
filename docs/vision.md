# GRBLR vision

GRBLR turns a pile of voice recordings into a voice you can play. You bring the
recordings. GRBLR slices them into words, syllables and phonemes, and stitches
new phrases from the slices.

## What a person does with it

1. **Collect.** Gather recordings you own or have consent to use: your own
   voice, or a friend reading a script.
2. **Build a grblr.** The build tools transcribe the lines, find word and phoneme
   boundaries, score each slice for quality and keep the good ones. The result is
   *a grblr*: one personal voice library.
3. **Check coverage.** Each grblr has a completeness rating. It measures how much
   of a language's sound inventory the slices cover, against a phonetically
   balanced reference set such as the Harvard sentences. It also lists the sounds
   that are missing and lines to record that would fill them.
4. **Play it.** Every consumer reads the same grblr format:
   - a DAW instrument, for a chaotic vocoder voice;
   - a text-to-speech engine for your own apps and sites;
   - a browser extension or phone app that reads text aloud.

## Two halves

| | Build | Play |
|---|---|---|
| Job | Turn recordings into a grblr | Turn text into audio from a grblr |
| Weight | Heavy: ASR, forced alignment, ML scoring | Light: lookup, selection, concatenation |
| Runs on | Desktop or cloud | Anything, down to a phone or a browser |
| Speed target | Hours for a large corpus is fine | Near real-time, snappy audio feedback |

**Play is the main optimization target.** It is likely to move from Python to a
compiled core (Rust or Zig) that can also build to WebAssembly. Build stays in
Python, where the ASR and alignment ecosystem lives.

## Principles

- **A grblr holds slices, not full lines.** It keeps what playback needs: the
  slices, their timings, phonemes and quality scores. It can record a source file
  by its hash instead of keeping a copy.
- **Aligners plug in.** whisperX today. Montreal Forced Aligner, our own
  fine-tuned boundary model, or anything else can slot into the build step. The
  decision records say which one is the default and why.
- **One format, many players.** The format is the standard. The API, the
  instrument and the extension are all readers of it.
- **Measured, not guessed.** Each choice of model, threshold or algorithm is a
  numbered decision record with the evidence behind it (`docs/decisions/`).

## This is a tool to build your own library, we don't share or publish private, proprietary, or illegally obtained materials.

GRBLR publishes tools, the format and docs. It never publishes, hosts or lists
audio, transcripts or grblrs made from someone else's voice. Examples and demos
use only openly licensed or consented voices. A grblr you build from recordings
you own stays on your machine.

## Where it stands (v0)

- Storage is one SQLite database plus a folder of audio. This is "format v0".
- This repository holds the play half: selection and rendering in Python.
- The build tools (index, score, curate) are not public yet. They open as a
  build recipe once it works end to end.
- One game uses it today: a browser game that collects boundary labels from
  players.

## What comes next

The roadmap lives in the issue tracker. In rough order:

1. **grblr format v1**: slice-only storage and the choice of container, migrated from v0.
2. **Coverage rating** against a balanced reference set.
3. **Phoneme-level matching** in selection.
4. **A compiled play core** (Rust or Zig, plus WebAssembly).
5. **An aligner plug-in interface** and better boundary detection.
6. **An open demo corpus** (LJ Speech or LibriTTS-R) so the engine works on install.
7. **More languages.**
8. **Readers of the format:** an HTTP API, the DAW instrument, the browser extension and mobile.
