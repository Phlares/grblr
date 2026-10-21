import json
import sys
from pathlib import Path

from tests.golden.corpus import build_corpus, golden_snapshot

GOLDEN = Path(__file__).with_name("render_v0.json")
PHRASES = [
    "the word is out",
    "you can use it",
    "what was said",
    "they are all up there",
    "how hot is your word",
    "we were here",  # "here" is not in the corpus: exercises the miss path
    "i have this one",
    "the other one said what",
]


def test_render_matches_golden(tmp_path):
    db = build_corpus(tmp_path / "corpus")
    assert golden_snapshot(db, tmp_path / "renders", PHRASES) == json.loads(
        GOLDEN.read_text()
    )


def test_snapshot_is_deterministic(tmp_path):
    db = build_corpus(tmp_path / "corpus")
    a = golden_snapshot(db, tmp_path / "r1", PHRASES)
    b = golden_snapshot(db, tmp_path / "r2", PHRASES)
    assert a == b


if __name__ == "__main__" and "--regen" in sys.argv:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        snap = golden_snapshot(
            build_corpus(Path(d) / "corpus"), Path(d) / "renders", PHRASES
        )
    GOLDEN.write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN}")
