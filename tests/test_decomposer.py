from grblr.db import open_db
from grblr.decomposer import (
    DecomposerConfig,
    SearchUnit,
    decompose,
)
from grblr.phonetics import word_to_phonemes


def _seed_words(db, words: list[tuple[str, int]], with_phonemes: bool = False):
    db.execute(
        "INSERT INTO clips (path, duration_ms, sample_rate, channels, whisper_model, indexed_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("/c.wav", 1000, 16000, 1, "medium", 0),
    )
    clip_id = db.execute("SELECT id FROM clips").fetchone()[0]
    for text, n_copies in words:
        ph = " ".join(word_to_phonemes(text) or []) if with_phonemes else None
        ph = ph if ph else None
        for _ in range(n_copies):
            db.execute(
                "INSERT INTO words (clip_id, text_lower, text_original, start_ms, end_ms, "
                "duration_ms, rms_db, peak_db, phonemes) "
                "VALUES (?, ?, ?, 0, 100, 100, -20.0, -10.0, ?)",
                (clip_id, text, text, ph),
            )
    db.commit()


def test_decompose_all_words_present(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    _seed_words(db, [("hello", 1), ("world", 1)])

    plan = decompose("hello world", db, DecomposerConfig())

    assert plan.sentence == "hello world"
    assert len(plan.units) == 2
    assert plan.units[0] == SearchUnit("hello", "literal", ["hello"])
    assert plan.units[1] == SearchUnit("world", "literal", ["world"])


def test_decompose_missing_word_emits_miss(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    _seed_words(db, [("hello", 1)])

    plan = decompose("hello universe", db, DecomposerConfig())

    assert plan.units[0].strategy == "literal"
    assert plan.units[1].strategy == "miss"
    assert plan.units[1].parts == []


def test_decompose_normalizes_punctuation(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    _seed_words(db, [("hello", 1)])

    plan = decompose("Hello!", db, DecomposerConfig())
    assert plan.units[0].strategy == "literal"
    assert plan.units[0].parts == ["hello"]


def test_decompose_empty_string(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    plan = decompose("", db, DecomposerConfig())
    assert plan.units == []


def test_decompose_log_records_each_decision(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    _seed_words(db, [("hello", 3)])
    plan = decompose("hello universe", db, DecomposerConfig())
    assert any("hello" in line and "literal match" in line for line in plan.log)
    assert any("universe" in line and "MISS" in line for line in plan.log)


def test_decompose_curly_apostrophe_matches_ascii(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    _seed_words(db, [("don't", 1)])  # corpus stored with ASCII apostrophe
    plan = decompose("Don’t", db, DecomposerConfig())  # input with curly apostrophe
    assert plan.units[0].strategy == "literal"
    assert plan.units[0].parts == ["don't"]


def test_decompose_syllable_split_disabled_by_default(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    _seed_words(db, [("cat", 1), ("ass", 1), ("trophy", 1)])
    plan = decompose("catastrophe", db, DecomposerConfig())
    # With syllable_split disabled, "catastrophe" is a miss even though cat/ass/trophy exist.
    assert plan.units[0].strategy == "miss"


def test_decompose_syllable_split_finds_parts(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    _seed_words(db, [("cat", 1), ("ass", 1), ("trophy", 1)], with_phonemes=True)
    plan = decompose("catastrophe", db, DecomposerConfig(enable_syllable_split=True))
    assert plan.units[0].strategy == "syllable_split"
    assert len(plan.units[0].parts) >= 2
    corpus_texts = {
        row[0] for row in db.execute("SELECT DISTINCT text_lower FROM words")
    }
    for part in plan.units[0].parts:
        assert part in corpus_texts


def test_decompose_syllable_split_falls_back_to_miss(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    _seed_words(db, [("cat", 1)], with_phonemes=True)
    plan = decompose("catastrophe", db, DecomposerConfig(enable_syllable_split=True))
    assert plan.units[0].strategy == "miss"


def test_decompose_existing_disabled_default_still_misses(tmp_path):
    db = open_db(str(tmp_path / "t.db"))
    _seed_words(db, [("cat", 1), ("ass", 1), ("trophy", 1)], with_phonemes=True)
    plan = decompose("catastrophe", db, DecomposerConfig())
    assert plan.units[0].strategy == "miss"
