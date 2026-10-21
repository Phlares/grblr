import sqlite3

from grblr.db import SCHEMA_VERSION, open_db


def test_open_db_creates_schema(tmp_path):
    db_path = tmp_path / "test.db"
    conn = open_db(str(db_path))
    cur = conn.cursor()
    tables = {
        r[0]
        for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "clips" in tables
    assert "words" in tables
    conn.close()


def test_open_db_idempotent(tmp_path):
    db_path = tmp_path / "test.db"
    conn1 = open_db(str(db_path))
    conn1.close()
    conn2 = open_db(str(db_path))
    cur = conn2.cursor()
    tables = {
        r[0]
        for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "clips" in tables
    assert "words" in tables
    conn2.close()


def test_clips_columns(tmp_path):
    conn = open_db(str(tmp_path / "t.db"))
    cols = {r[1] for r in conn.execute("PRAGMA table_info(clips)").fetchall()}
    assert cols >= {
        "id",
        "path",
        "duration_ms",
        "source_game",
        "sample_rate",
        "channels",
        "whisper_model",
        "indexed_at",
    }
    conn.close()


def test_words_columns(tmp_path):
    conn = open_db(str(tmp_path / "t.db"))
    cols = {r[1] for r in conn.execute("PRAGMA table_info(words)").fetchall()}
    assert cols >= {
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
    conn.close()


def test_text_lower_index_exists(tmp_path):
    conn = open_db(str(tmp_path / "t.db"))
    indexes = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    }
    assert "idx_words_text" in indexes
    conn.close()


def test_path_uniqueness(tmp_path):
    conn = open_db(str(tmp_path / "t.db"))
    conn.execute(
        "INSERT INTO clips (path, duration_ms, sample_rate, channels, whisper_model, indexed_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("/x.wav", 1000, 16000, 1, "medium", 0),
    )
    conn.commit()
    try:
        conn.execute(
            "INSERT INTO clips (path, duration_ms, sample_rate, channels, whisper_model, indexed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("/x.wav", 2000, 22050, 2, "medium", 1),
        )
        conn.commit()
        raise AssertionError("expected IntegrityError on duplicate path")
    except sqlite3.IntegrityError:
        pass
    conn.close()


def test_foreign_keys_enforced(tmp_path):
    conn = open_db(str(tmp_path / "t.db"))
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


def test_schema_version_is_int():
    assert isinstance(SCHEMA_VERSION, int)
    assert SCHEMA_VERSION >= 19


def test_open_db_creates_parent_dirs(tmp_path):
    """open_db() should create missing parent directories so first-run
    against a fresh corpus path doesn't fail."""
    nested = tmp_path / "deep" / "nested" / "dir" / "corpus.db"
    assert not nested.parent.exists()
    conn = open_db(str(nested))
    assert nested.parent.exists()
    assert nested.exists()
    conn.close()
