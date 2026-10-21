import sqlite3

from grblr.db import open_db

_V1_COLUMNS = {
    "id",
    "clip_id",
    "text_lower",
    "text_original",
    "start_ms",
    "end_ms",
    "duration_ms",
    "rms_db",
    "peak_db",
    "confidence",
}
_V1_1_NEW_COLUMNS = {
    "median_f0_hz",
    "spectral_centroid_hz",
    "voiced_fraction",
    "phonemes",
}

_V1_SCHEMA = """
CREATE TABLE clips (
    id            INTEGER PRIMARY KEY,
    path          TEXT UNIQUE NOT NULL,
    duration_ms   INTEGER NOT NULL,
    source_game   TEXT,
    sample_rate   INTEGER NOT NULL,
    channels      INTEGER NOT NULL,
    whisper_model TEXT NOT NULL,
    indexed_at    INTEGER NOT NULL
);

CREATE TABLE words (
    id            INTEGER PRIMARY KEY,
    clip_id       INTEGER NOT NULL REFERENCES clips(id),
    text_lower    TEXT NOT NULL,
    text_original TEXT NOT NULL,
    start_ms      INTEGER NOT NULL,
    end_ms        INTEGER NOT NULL,
    duration_ms   INTEGER NOT NULL,
    rms_db        REAL NOT NULL,
    peak_db       REAL NOT NULL,
    confidence    REAL
);

CREATE INDEX idx_words_text ON words(text_lower);
CREATE INDEX idx_words_clip ON words(clip_id);
"""


def test_open_db_creates_v1_1_schema_from_scratch(tmp_path):
    conn = open_db(str(tmp_path / "fresh.db"))
    cols = {r[1] for r in conn.execute("PRAGMA table_info(words)").fetchall()}
    assert cols >= _V1_COLUMNS
    assert cols >= _V1_1_NEW_COLUMNS
    conn.close()


def test_open_db_migrates_v1_schema_in_place(tmp_path):
    db_path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(str(db_path))
    legacy.executescript(_V1_SCHEMA)
    legacy.execute(
        "INSERT INTO clips (path, duration_ms, sample_rate, channels, whisper_model, indexed_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("/legacy.wav", 1500, 22050, 1, "medium", 42),
    )
    legacy.execute(
        "INSERT INTO words (clip_id, text_lower, text_original, start_ms, end_ms, "
        "duration_ms, rms_db, peak_db, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (1, "hello", "Hello", 100, 500, 400, -18.5, -6.2, 0.93),
    )
    legacy.commit()
    legacy.close()

    conn = open_db(str(db_path))
    cols = {r[1] for r in conn.execute("PRAGMA table_info(words)").fetchall()}
    assert cols >= _V1_COLUMNS | _V1_1_NEW_COLUMNS

    row = conn.execute(
        "SELECT text_lower, text_original, start_ms, end_ms, duration_ms, "
        "rms_db, peak_db, confidence, median_f0_hz, spectral_centroid_hz, "
        "voiced_fraction, phonemes FROM words"
    ).fetchone()
    assert row[0] == "hello"
    assert row[1] == "Hello"
    assert row[2] == 100
    assert row[3] == 500
    assert row[4] == 400
    assert row[5] == -18.5
    assert row[6] == -6.2
    assert row[7] == 0.93
    assert row[8] is None
    assert row[9] is None
    assert row[10] is None
    assert row[11] is None
    conn.close()


def test_open_db_is_idempotent(tmp_path):
    db_path = tmp_path / "twice.db"
    conn1 = open_db(str(db_path))
    cols1 = [r[1] for r in conn1.execute("PRAGMA table_info(words)").fetchall()]
    conn1.close()

    conn2 = open_db(str(db_path))
    cols2 = [r[1] for r in conn2.execute("PRAGMA table_info(words)").fetchall()]
    conn2.close()

    assert cols1 == cols2
    assert len(cols2) == len(set(cols2))


def test_open_db_v1_1_shape_gains_word_quality_table(tmp_path):
    """A v1.1-shaped DB (words has the v1.1 cols but no word_quality table)
    must gain word_quality on open."""
    db_path = tmp_path / "v1_1.db"
    legacy = sqlite3.connect(str(db_path))
    legacy.executescript(_V1_SCHEMA)
    for name, sql_type in (
        ("median_f0_hz", "REAL"),
        ("spectral_centroid_hz", "REAL"),
        ("voiced_fraction", "REAL"),
        ("phonemes", "TEXT"),
    ):
        legacy.execute(f"ALTER TABLE words ADD COLUMN {name} {sql_type}")
    legacy.commit()
    legacy.close()

    conn = open_db(str(db_path))
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "word_quality" in tables
    cols = {r[1] for r in conn.execute("PRAGMA table_info(word_quality)").fetchall()}
    assert cols == {"word_id", "quality", "rated_at", "notes"}
    conn.close()


def test_word_quality_table_idempotent_across_opens(tmp_path):
    db_path = tmp_path / "idem.db"
    conn1 = open_db(str(db_path))
    conn1.execute(
        "INSERT INTO clips (path, duration_ms, sample_rate, channels, whisper_model, indexed_at) "
        "VALUES ('/a.wav', 500, 16000, 1, 'medium', 0)"
    )
    conn1.execute(
        "INSERT INTO words (clip_id, text_lower, text_original, start_ms, end_ms, "
        "duration_ms, rms_db, peak_db) VALUES (1, 'hi', 'Hi', 0, 500, 500, -20, -10)"
    )
    conn1.execute(
        "INSERT INTO word_quality (word_id, quality, rated_at) VALUES (1, 'good', 100)"
    )
    conn1.commit()
    conn1.close()

    conn2 = open_db(str(db_path))
    row = conn2.execute(
        "SELECT quality, rated_at FROM word_quality WHERE word_id=1"
    ).fetchone()
    assert row == ("good", 100)
    conn2.close()


def test_foreign_keys_still_enforced(tmp_path):
    conn = open_db(str(tmp_path / "fk.db"))
    try:
        conn.execute(
            "INSERT INTO words (clip_id, text_lower, text_original, start_ms, end_ms, "
            "duration_ms, rms_db, peak_db) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (9999, "a", "a", 0, 1, 1, -20.0, -10.0),
        )
        conn.commit()
        raise AssertionError("expected IntegrityError on orphan clip_id")
    except sqlite3.IntegrityError:
        pass
    conn.close()


def test_open_db_creates_filter_tables(tmp_path):
    conn = open_db(str(tmp_path / "f.db"))
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {"filter_models", "clip_filter"} <= tables
    conn.close()
