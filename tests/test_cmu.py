import hashlib
import json
from importlib import resources

from grblr import cmu


def test_known_word():
    assert cmu.load()["hello"] == [["HH", "AH0", "L", "OW1"], ["HH", "EH0", "L", "OW1"]]


def test_inline_comments_are_not_phones():
    assert all(
        "#" not in ph for prons in cmu.load().values() for pron in prons for ph in pron
    )


# sha256 of the GPL `cmudict` wrapper's dict(), minus the 21 entries carrying an
# inline "#" comment (the wrapper keeps the comment words as phones). Recorded
# once from cmudict 1.1.3 when the wrapper was dropped.
WRAPPER_SHA256 = "6158447448b9d30265c5e1995867ae91eff90c31a14001e8239124db5d8e385b"


def test_load_matches_the_wrapper_it_replaced():
    data = (
        resources.files("grblr")
        .joinpath("data", "cmudict.dict")
        .read_text(encoding="utf-8")
    )
    commented = {
        line.split()[0].split("(")[0] for line in data.splitlines() if "#" in line
    }
    kept = {k: v for k, v in sorted(cmu.load().items()) if k not in commented}
    assert len(commented) == 21
    assert hashlib.sha256(json.dumps(kept).encode()).hexdigest() == WRAPPER_SHA256


def test_license_ships():
    text = resources.files("grblr").joinpath("data", "LICENSE.cmudict").read_text()
    assert "Carnegie Mellon University" in text
