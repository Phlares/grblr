import json

from grblr.cli import cmd_lookup, cmd_render, cmd_stats, make_parser
from grblr.db import open_db


def test_make_parser_has_all_subcommands():
    p = make_parser()
    args = p.parse_args(["lookup", "hello"])
    assert args.command == "lookup"
    args = p.parse_args(["render", "hello world", "--seed", "7"])
    assert args.command == "render"
    assert args.seed == 7
    args = p.parse_args(["stats"])
    assert args.command == "stats"


def test_render_runs_flag_defaults_on():
    p = make_parser()
    assert p.parse_args(["render", "hello"]).runs is True  # default: intelligible
    assert p.parse_args(["render", "hello", "--no-runs"]).runs is False  # chaos lever
    assert p.parse_args(["render", "hello", "--runs"]).runs is True


def test_cmd_lookup_prints_variants(tmp_path, capsys):
    db_path = tmp_path / "t.db"
    db = open_db(str(db_path))
    db.execute(
        "INSERT INTO clips (path, duration_ms, sample_rate, channels, whisper_model, indexed_at) "
        "VALUES ('/a.wav', 1000, 16000, 1, 'medium', 0)"
    )
    db.execute(
        "INSERT INTO words (clip_id, text_lower, text_original, start_ms, end_ms, "
        "duration_ms, rms_db, peak_db) VALUES (1, 'hello', 'Hello', 0, 500, 500, -20, -10)"
    )
    db.commit()
    db.close()

    rc = cmd_lookup(word="hello", db=str(db_path))
    assert rc == 0
    out = capsys.readouterr().out
    assert "hello" in out
    assert "/a.wav" in out


def test_cmd_lookup_unknown_word_returns_nonzero(tmp_path, capsys):
    db_path = tmp_path / "t.db"
    open_db(str(db_path)).close()
    rc = cmd_lookup(word="nope", db=str(db_path))
    assert rc == 1


def test_cmd_stats(tmp_path, capsys):
    db_path = tmp_path / "t.db"
    db = open_db(str(db_path))
    db.execute(
        "INSERT INTO clips (path, duration_ms, sample_rate, channels, whisper_model, indexed_at) "
        "VALUES ('/a.wav', 1000, 16000, 1, 'medium', 0)"
    )
    db.execute(
        "INSERT INTO words (clip_id, text_lower, text_original, start_ms, end_ms, "
        "duration_ms, rms_db, peak_db) VALUES (1, 'hello', 'Hello', 0, 500, 500, -20, -10)"
    )
    db.commit()
    db.close()

    rc = cmd_stats(db=str(db_path))
    assert rc == 0
    out = capsys.readouterr().out
    assert "1 clip" in out
    assert "1 word" in out
    assert "1 unique" in out


def test_cmd_render_end_to_end(tmp_path, synth_wav, capsys):
    # Seed a corpus by inserting directly (no Whisper required)
    a = synth_wav(name="hello.wav", duration_ms=500)
    db_path = tmp_path / "t.db"
    db = open_db(str(db_path))
    db.execute(
        "INSERT INTO clips (path, duration_ms, sample_rate, channels, whisper_model, indexed_at) "
        "VALUES (?, 500, 16000, 1, 'medium', 0)",
        (str(a),),
    )
    db.execute(
        "INSERT INTO words (clip_id, text_lower, text_original, start_ms, end_ms, "
        "duration_ms, rms_db, peak_db) VALUES (1, 'hello', 'Hello', 0, 500, 500, -20, -10)"
    )
    db.commit()
    db.close()

    out_wav = tmp_path / "out.wav"
    out_log = tmp_path / "out.json"
    rc = cmd_render(
        text="hello",
        db=str(db_path),
        out=str(out_wav),
        log=str(out_log),
        seed=42,
    )
    assert rc == 0
    assert out_wav.exists()
    log = json.loads(out_log.read_text(encoding="utf-8"))
    assert log["sentence"] == "hello"
    assert log["seed"] == 42


def test_cmd_lookup_normalizes_input(tmp_path, capsys):
    """Lookup must use normalize() so curly apostrophes and punctuation
    in user input match indexer-normalized rows."""
    db_path = tmp_path / "t.db"
    db = open_db(str(db_path))
    db.execute(
        "INSERT INTO clips (path, duration_ms, sample_rate, channels, whisper_model, indexed_at) "
        "VALUES ('/a.wav', 1000, 16000, 1, 'medium', 0)"
    )
    db.execute(
        "INSERT INTO words (clip_id, text_lower, text_original, start_ms, end_ms, "
        'duration_ms, rms_db, peak_db) VALUES (1, "don\'t", "Don\'t", 0, 500, 500, -20, -10)'
    )
    db.commit()
    db.close()

    # Input with curly apostrophe should still match
    rc = cmd_lookup(word="Don’t", db=str(db_path))
    assert rc == 0

    # Input with trailing punctuation should still match
    rc = cmd_lookup(word="don't,", db=str(db_path))
    assert rc == 0
