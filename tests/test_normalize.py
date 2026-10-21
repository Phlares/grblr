from grblr.normalize import normalize


def test_basic_lowercase():
    assert normalize("Hello") == "hello"


def test_strip_trailing_punctuation():
    assert normalize("Christmas!") == "christmas"


def test_strip_leading_punctuation():
    assert normalize('"hello') == "hello"


def test_strip_surrounding_punctuation():
    assert normalize("(hello).") == "hello"


def test_curly_apostrophe_normalized():
    assert normalize("Don’t") == "don't"


def test_left_curly_apostrophe_normalized():
    assert normalize("‘tis") == "'tis"


def test_internal_apostrophe_preserved():
    assert normalize("don't") == "don't"


def test_hyphenated_kept_as_one():
    assert normalize("fire-breathing") == "fire-breathing"


def test_empty_string():
    assert normalize("") == ""


def test_only_punctuation():
    assert normalize("...") == ""
