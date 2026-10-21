from grblr.db import open_db


def _cols(conn, table):
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def test_clips_has_lid_prob(tmp_path):
    conn = open_db(str(tmp_path / "c.db"))
    assert "lid_prob" in _cols(conn, "clips")
    assert "language" in _cols(conn, "clips")  # v9, still present
    conn.close()


def test_clip_content_table_exists(tmp_path):
    conn = open_db(str(tmp_path / "c.db"))
    cols = _cols(conn, "clip_content")
    assert {
        "clip_id",
        "segment_idx",
        "speech_prom",
        "music_prob",
        "content_class",
        "lid_lang",
        "score_version",
    }.issubset(cols)
    conn.close()


def test_migration_is_idempotent(tmp_path):
    p = str(tmp_path / "c.db")
    open_db(p).close()
    conn = open_db(p)  # second open must not raise
    assert "lid_prob" in _cols(conn, "clips")
