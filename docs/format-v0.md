# Format v0

A v0 corpus is one SQLite database plus the audio files it points to. The
engine reads it; it never changes the audio. Format v1 will replace v0, with a
migration (see [vision](vision.md)).

`grblr.db.open_db` creates every table below on an empty file, so a build tool
can start from it. The renderer opens the database read-only.

## Versions

`schema_meta` holds one row per component that owns tables in the file.

| Column | Type | Meaning |
|---|---|---|
| `component` | TEXT, key | `engine` for the tables on this page |
| `version` | INTEGER | Schema version of that component. The engine is at 23. |

Other tools may keep their own tables in the same file under their own
`component` row. The engine ignores them.

## Required tables

### clips

One row per audio file.

| Column | Type | Meaning |
|---|---|---|
| `id` | INTEGER, key | |
| `path` | TEXT, unique | Path the renderer opens. Absolute, or relative to the working directory. |
| `duration_ms` | INTEGER | Length of the file |
| `sample_rate`, `channels` | INTEGER | Audio format. The renderer resamples and downmixes. |
| `whisper_model` | TEXT | Name of the transcriber that made the word rows |
| `indexed_at` | INTEGER | Unix time |
| `source_id` | TEXT | Neutral source id, `src_` + 8 hex digits (see Sources) |
| `category` | TEXT | Content type; `voice` by default |
| `lang_bucket` | TEXT | Language used for selection, for example `en` |
| `dupe_of` | INTEGER | When set, this clip duplicates that clip and is skipped |
| `lang_hint` | TEXT | Language hint from the manifest |

Other `clips` columns (`source_game`, `language`, `lid_prob`, `bucket_conf`,
`bucket_source`, `needs_review`, `review_reason`) belong to build tools. The
renderer does not read them. `source_game` is the old name of
`source_id` and goes away in a later version.

### words

One row per transcribed word.

| Column | Type | Meaning |
|---|---|---|
| `id` | INTEGER, key | |
| `clip_id` | INTEGER | `clips.id`. Runs of words follow `start_ms` inside a clip. |
| `text_lower`, `text_original` | TEXT | The word, lowercase and as transcribed |
| `start_ms`, `end_ms`, `duration_ms` | INTEGER | Word edges in the clip |
| `rms_db`, `peak_db` | REAL | Loudness of the word span |
| `confidence` | REAL | Transcriber confidence, 0 to 1 |
| `median_f0_hz`, `spectral_centroid_hz`, `voiced_fraction` | REAL | Pitch, brightness and voicing, for selection bias |
| `phonemes` | TEXT | ARPAbet, space separated (CMUdict style) |

## Optional tables

Each one refines selection. An empty table changes nothing.

| Table | What the renderer does with it |
|---|---|
| `word_quality` (`word_id`, `quality`) | `ignore` drops a word. `good` words are preferred. |
| `clip_content` (`clip_id`, `segment_idx`, `content_class`) | With `--content-classes`, only clips whose segment 0 class is listed are used. |
| `clip_filter` (`word_id`, `model_version`, `keep_prob`, `excluded`) | `keep_prob` feeds the experimental lattice cost. |
| `filter_models` | Describes each `clip_filter` model version. |

## Selection defaults

`grblr render` draws from clips with `lang_bucket = 'en'` and a
`clip_content` class of `speech-en`. A corpus that fills neither column
renders silence for every word. Fill them, or pass
`--lang all --content-classes all`.

## Sources

The database stores neutral source ids only. An optional JSON file named by
`GRBLR_SOURCES` maps paths to sources and holds display names:

```json
{"source":   [["^speaker_(\\w+)/", "\\1"]],
 "category": [["/sfx/", "sfx"]],
 "language": [["/fr/", "fr"]],
 "names":    {"src_1234abcd": "Speaker A"}}
```

Each list is ordered; the first matching regex wins. `grblr.sources.source_id`
turns a source name into its id.

## Manifest v0

A build tool can take a manifest instead of path rules: JSON Lines, one object
per audio file. Fields: `audio_key` (the path), `source_id`, `category`
(default `voice`) and `lang_hint`. Blank lines are skipped.
`grblr.sources.read_manifest` reads it.
