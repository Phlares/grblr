from grblr.phonetics import (
    phoneme_distance,
    split_into_syllables,
    word_to_phonemes,
)


def test_word_to_phonemes_cat():
    assert word_to_phonemes("cat") == ["K", "AE1", "T"]


def test_word_to_phonemes_mixed_case():
    result = word_to_phonemes("Catastrophe")
    assert result is not None
    assert len(result) == 10
    assert result[0] == "K"
    assert "AE1" in result


def test_word_to_phonemes_empty():
    assert word_to_phonemes("") is None


def test_word_to_phonemes_oov():
    assert word_to_phonemes("qwxyzzzz") is None


def test_word_to_phonemes_apostrophe():
    result = word_to_phonemes("don't")
    assert result is not None
    assert result[0] == "D"


def test_word_to_phonemes_curly_apostrophe():
    result = word_to_phonemes("don’t")
    assert result is not None
    assert result == word_to_phonemes("don't")


def test_word_to_phonemes_returns_first_pronunciation():
    result = word_to_phonemes("don't")
    assert result == ["D", "OW1", "N", "T"]


def test_split_syllables_cat():
    assert split_into_syllables(["K", "AE1", "T"]) == [["K", "AE1", "T"]]


def test_split_syllables_catastrophe():
    phonemes = ["K", "AH0", "T", "AE1", "S", "T", "R", "AH0", "F", "IY0"]
    syllables = split_into_syllables(phonemes)
    assert len(syllables) == 4
    for syl in syllables:
        vowels = [p for p in syl if any(c.isdigit() for c in p)]
        assert len(vowels) == 1


def test_split_syllables_empty():
    assert split_into_syllables([]) == []


def test_split_syllables_no_vowel():
    assert split_into_syllables(["K"]) == [["K"]]


def test_split_syllables_happy():
    assert split_into_syllables(["HH", "AE1", "P", "IY0"]) == [
        ["HH", "AE1"],
        ["P", "IY0"],
    ]


def test_phoneme_distance_cat_hat():
    assert phoneme_distance(["K", "AE1", "T"], ["HH", "AE1", "T"]) == 1


def test_phoneme_distance_identity():
    assert phoneme_distance(["K", "AE1", "T"], ["K", "AE1", "T"]) == 0


def test_phoneme_distance_cat_cats():
    assert phoneme_distance(["K", "AE1", "T"], ["K", "AE1", "T", "S"]) == 1


def test_phoneme_distance_stress_stripped():
    assert phoneme_distance(["AE1"], ["AE0"]) == 0


def test_phoneme_distance_empty_to_one():
    assert phoneme_distance([], ["K"]) == 1
