import json

import pytest

from grblr import sources
from grblr.sources import ManifestRow, PathRules, read_manifest, source_id

RULES = {
    "source": [[r"[/\\]+library[/\\]+([^/\\]+)", r"\1"]],
    "category": [["(?i)cutscene", "cutscene"], ["(?i)sfx|se_cm", "sfx"]],
    "language": [[r"(?i)[\\/]sfx[\\/]", "neither"], ["(?i)english|_uk", "en"]],
    "names": {"src_aaaaaaaa": "Source A"},
}


def test_source_id_is_src_plus_first_8_hex_of_sha1():
    import hashlib

    want = "src_" + hashlib.sha1(b"Source A").hexdigest()[:8]
    assert source_id("Source A") == want
    assert len(want) == 12


def test_source_id_none_passes_through():
    assert source_id(None) is None


def test_no_rules_resolves_to_neutral_defaults():
    row = PathRules().resolve("/corpus/library/Source A/sfx/a.wav")
    assert row == ManifestRow("/corpus/library/Source A/sfx/a.wav", None, "voice", None)


def test_rules_resolve_source_category_and_language():
    rules = PathRules.from_dict(RULES)
    row = rules.resolve(r"corpus\library\Source B\English_UK\cutscene_01.wav")
    assert row.source_id == source_id("Source B")
    assert row.category == "cutscene"
    assert row.lang_hint == "en"


def test_first_matching_rule_wins():
    rules = PathRules.from_dict(RULES)
    row = rules.resolve("/x/library/Source C/English/sfx/b.wav")
    assert row.category == "sfx"
    assert row.lang_hint == "neither"


def test_unmatched_path_gets_defaults():
    row = PathRules.from_dict(RULES).resolve("/random/clip.wav")
    assert (row.source_id, row.category, row.lang_hint) == (None, "voice", None)


def test_load_reads_the_private_file_named_by_env(tmp_path, monkeypatch):
    f = tmp_path / "sources.json"
    f.write_text(json.dumps(RULES), encoding="utf-8")
    monkeypatch.setenv("GRBLR_SOURCES", str(f))
    assert sources.load().resolve("/a/library/Source A/x.wav").source_id == (
        source_id("Source A")
    )


def test_load_without_env_is_the_empty_rule_set(monkeypatch):
    monkeypatch.delenv("GRBLR_SOURCES", raising=False)
    assert sources.load() == PathRules()


def test_read_manifest_jsonl(tmp_path):
    f = tmp_path / "m.jsonl"
    f.write_text(
        '{"audio_key": "a/1.wav", "source_id": "src_1", "category": "voice",'
        ' "lang_hint": "en"}\n'
        "\n"
        '{"audio_key": "a/2.wav"}\n',
        encoding="utf-8",
    )
    assert read_manifest(f) == [
        ManifestRow("a/1.wav", "src_1", "voice", "en"),
        ManifestRow("a/2.wav", None, "voice", None),
    ]


def test_read_manifest_rejects_unknown_fields(tmp_path):
    f = tmp_path / "m.jsonl"
    f.write_text('{"audio_key": "a.wav", "source_game": "X"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="source_game"):
        read_manifest(f)


def test_read_manifest_names_the_line_of_a_row_without_audio_key(tmp_path):
    f = tmp_path / "m.jsonl"
    f.write_text('{"audio_key": "a.wav"}\n{"source_id": "src_1"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"m.jsonl:2:.*audio_key"):
        read_manifest(f)
