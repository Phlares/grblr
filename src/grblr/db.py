import sqlite3
from pathlib import Path

from grblr.sources import source_id

# The tables the renderer reads. Other components may keep their own tables
# in the same file; each records its own version in schema_meta.
SCHEMA_VERSION = 23  # v9 was clips.language (ALTER-only, no _V9_ADDITIONS block); v10 adds clips.lid_prob + clip_content; v12 adds clips triage columns; v14 adds clips.dupe_of (ALTER-only); v23 adds clips.source_id + clips.lang_hint (source_game stays one release)

_V1_1_COLUMNS: list[tuple[str, str]] = [
    ("median_f0_hz", "REAL"),
    ("spectral_centroid_hz", "REAL"),
    ("voiced_fraction", "REAL"),
    ("phonemes", "TEXT"),
]

_V3_ADDITIONS = """
CREATE TABLE IF NOT EXISTS word_quality (
    word_id  INTEGER PRIMARY KEY REFERENCES words(id),
    quality  TEXT NOT NULL,
    rated_at INTEGER NOT NULL,
    notes    TEXT
);
CREATE INDEX IF NOT EXISTS idx_word_quality_quality ON word_quality(quality);
"""

_V8_ADDITIONS = """
CREATE TABLE IF NOT EXISTS filter_models (
    version       TEXT PRIMARY KEY,
    created_at    INTEGER NOT NULL,
    threshold     REAL NOT NULL,
    n_labels      INTEGER NOT NULL,
    metrics       TEXT,
    feature_names TEXT,
    model_path    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS clip_filter (
    word_id       INTEGER NOT NULL REFERENCES words(id),
    model_version TEXT NOT NULL REFERENCES filter_models(version),
    keep_prob     REAL NOT NULL,
    excluded      INTEGER NOT NULL,
    PRIMARY KEY (word_id, model_version)
);
"""

_V10_ADDITIONS = """
CREATE TABLE IF NOT EXISTS clip_content (
    clip_id        INTEGER NOT NULL REFERENCES clips(id),
    segment_idx    INTEGER NOT NULL DEFAULT 0,
    start_ms       INTEGER,
    end_ms         INTEGER,
    speech_prom    REAL,
    music_prob     REAL,
    sfx_prob       REAL,
    content_class  TEXT,
    lid_lang       TEXT,
    lid_prob       REAL,
    scored_at      INTEGER,
    score_version  TEXT,
    PRIMARY KEY (clip_id, segment_idx)
);
CREATE INDEX IF NOT EXISTS idx_clip_content_class ON clip_content(content_class);
"""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    component TEXT PRIMARY KEY,
    version   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS clips (
    id            INTEGER PRIMARY KEY,
    path          TEXT UNIQUE NOT NULL,
    duration_ms   INTEGER NOT NULL,
    source_game   TEXT,
    category      TEXT,
    sample_rate   INTEGER NOT NULL,
    channels      INTEGER NOT NULL,
    whisper_model TEXT NOT NULL,
    indexed_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS words (
    id                   INTEGER PRIMARY KEY,
    clip_id              INTEGER NOT NULL REFERENCES clips(id),
    text_lower           TEXT NOT NULL,
    text_original        TEXT NOT NULL,
    start_ms             INTEGER NOT NULL,
    end_ms               INTEGER NOT NULL,
    duration_ms          INTEGER NOT NULL,
    rms_db               REAL NOT NULL,
    peak_db              REAL NOT NULL,
    confidence           REAL,
    median_f0_hz         REAL,
    spectral_centroid_hz REAL,
    voiced_fraction      REAL,
    phonemes             TEXT
);

CREATE INDEX IF NOT EXISTS idx_words_text ON words(text_lower);
CREATE INDEX IF NOT EXISTS idx_words_clip ON words(clip_id);

CREATE TABLE IF NOT EXISTS word_quality (
    word_id  INTEGER PRIMARY KEY REFERENCES words(id),
    quality  TEXT NOT NULL,
    rated_at INTEGER NOT NULL,
    notes    TEXT
);
CREATE INDEX IF NOT EXISTS idx_word_quality_quality ON word_quality(quality);

CREATE TABLE IF NOT EXISTS filter_models (
    version       TEXT PRIMARY KEY,
    created_at    INTEGER NOT NULL,
    threshold     REAL NOT NULL,
    n_labels      INTEGER NOT NULL,
    metrics       TEXT,
    feature_names TEXT,
    model_path    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS clip_filter (
    word_id       INTEGER NOT NULL REFERENCES words(id),
    model_version TEXT NOT NULL REFERENCES filter_models(version),
    keep_prob     REAL NOT NULL,
    excluded      INTEGER NOT NULL,
    PRIMARY KEY (word_id, model_version)
);

CREATE TABLE IF NOT EXISTS clip_content (
    clip_id        INTEGER NOT NULL REFERENCES clips(id),
    segment_idx    INTEGER NOT NULL DEFAULT 0,
    start_ms       INTEGER,
    end_ms         INTEGER,
    speech_prom    REAL,
    music_prob     REAL,
    sfx_prob       REAL,
    content_class  TEXT,
    lid_lang       TEXT,
    lid_prob       REAL,
    scored_at      INTEGER,
    score_version  TEXT,
    PRIMARY KEY (clip_id, segment_idx)
);
CREATE INDEX IF NOT EXISTS idx_clip_content_class ON clip_content(content_class);

"""


def _migrate(conn: sqlite3.Connection) -> None:
    existing = {r[1] for r in conn.execute("PRAGMA table_info(words)").fetchall()}
    for name, sql_type in _V1_1_COLUMNS:
        if name not in existing:
            conn.execute(f"ALTER TABLE words ADD COLUMN {name} {sql_type}")
    conn.executescript(_V3_ADDITIONS)
    # v4: clips.category — content-type label (voice/cutscene/sfx/bgm) for
    # corpus inclusion/exclusion. Set at ingest (sources.PathRules / manifest).
    clip_cols = {r[1] for r in conn.execute("PRAGMA table_info(clips)").fetchall()}
    if "category" not in clip_cols:
        conn.execute("ALTER TABLE clips ADD COLUMN category TEXT")
    clip_cols = {r[1] for r in conn.execute("PRAGMA table_info(clips)").fetchall()}
    # v9: clips.language — spoken language ('en'/'ja'/...); NULL when uncertain
    # or non-speech (BGM/SFX). Stamped per index run / backfilled per game.
    if "language" not in clip_cols:
        conn.execute("ALTER TABLE clips ADD COLUMN language TEXT")
    # v10: clips.lid_prob — LID confidence score for the language column value.
    if "lid_prob" not in clip_cols:
        conn.execute("ALTER TABLE clips ADD COLUMN lid_prob REAL")
    triage_cols = {r[1] for r in conn.execute("PRAGMA table_info(clips)").fetchall()}
    for col, typ in (
        ("lang_bucket", "TEXT"),
        ("bucket_conf", "REAL"),
        ("bucket_source", "TEXT"),
        ("needs_review", "INTEGER"),
        ("review_reason", "TEXT"),
    ):
        if col not in triage_cols:
            conn.execute(f"ALTER TABLE clips ADD COLUMN {col} {typ}")
    # v14: clips.dupe_of — when set, this clip is a byte-identical duplicate of
    # the referenced canonical clip and should be excluded from corpus selection
    # / triage sampling. Source files are sequestered to a `_dupe/` subtree at
    # the game root (path is rewritten to the new location).
    dedup_cols = {r[1] for r in conn.execute("PRAGMA table_info(clips)").fetchall()}
    if "dupe_of" not in dedup_cols:
        conn.execute(
            "ALTER TABLE clips ADD COLUMN dupe_of INTEGER REFERENCES clips(id)"
        )
    conn.executescript(_V8_ADDITIONS)
    conn.executescript(_V10_ADDITIONS)
    _migrate_v23(conn)


def add_column(conn: sqlite3.Connection, table: str, name: str, decl: str) -> bool:
    """ALTER TABLE ADD COLUMN unless present. True when it added the column."""
    if name in {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}:
        return False
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
    return True


def _migrate_v23(conn: sqlite3.Connection) -> None:
    # Expand: source_id beside source_game. The previous release still reads
    # and writes source_game, so it stays until the next release (contract).
    # ALTER-only so the column order matches on every DB (apply_delta._guard).
    conn.create_function("source_id", 1, source_id, deterministic=True)
    # The backfills run on every open: ALTER commits on its own, so a crash
    # before the UPDATE commits must not leave the column empty for good.
    add_column(conn, "clips", "source_id", "TEXT")
    conn.execute(
        "UPDATE clips SET source_id = source_id(source_game)"
        " WHERE source_id IS NULL AND source_game IS NOT NULL"
    )
    # The manifest's language hint (sources.ManifestRow); triage's path vote.
    add_column(conn, "clips", "lang_hint", "TEXT")


def set_version(conn: sqlite3.Connection, component: str, version: int) -> None:
    """Record a component's schema version. The engine, the lab and the game
    share one DB file and version their own tables here."""
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta (component, version) VALUES (?, ?)",
        (component, version),
    )


def open_db(path: str) -> sqlite3.Connection:
    """Open (and lazily initialize) a grblr corpus database.

    Schema creation is idempotent — safe to call against an existing DB.
    Foreign key enforcement is enabled per-connection.
    Parent directories are created if they don't exist.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=60.0)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 60000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(_SCHEMA)
    _migrate(conn)
    set_version(conn, "engine", SCHEMA_VERSION)
    conn.commit()
    return conn


def open_db_readonly(path: str) -> sqlite3.Connection:
    """Open the corpus DB read-only — no migrations, no writes possible.

    Safe to use alongside a live indexer holding write transactions: SQLite's
    read-only mode never acquires the reserved/exclusive lock, so SELECTs go
    through even while the indexer is committing.
    """
    return sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
