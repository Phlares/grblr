from functools import lru_cache

from grblr import cmu

# Common English 2-consonant onset clusters (legal syllable-initial pairs).
_VALID_2_ONSETS = frozenset(
    {
        ("B", "L"),
        ("B", "R"),
        ("CH", "R"),
        ("D", "R"),
        ("D", "W"),
        ("F", "L"),
        ("F", "R"),
        ("G", "L"),
        ("G", "R"),
        ("G", "W"),
        ("HH", "Y"),
        ("K", "L"),
        ("K", "R"),
        ("K", "W"),
        ("K", "Y"),
        ("P", "L"),
        ("P", "R"),
        ("P", "Y"),
        ("S", "K"),
        ("S", "L"),
        ("S", "M"),
        ("S", "N"),
        ("S", "P"),
        ("S", "T"),
        ("S", "W"),
        ("S", "F"),
        ("SH", "R"),
        ("T", "R"),
        ("T", "W"),
        ("T", "Y"),
        ("TH", "R"),
        ("TH", "W"),
        ("V", "Y"),
    }
)

# Common English 3-consonant onset clusters (all start with S).
_VALID_3_ONSETS = frozenset(
    {
        ("S", "K", "R"),
        ("S", "K", "W"),
        ("S", "K", "Y"),
        ("S", "P", "L"),
        ("S", "P", "R"),
        ("S", "P", "Y"),
        ("S", "T", "R"),
        ("S", "T", "Y"),
    }
)


@lru_cache(maxsize=1)
def _cmu() -> dict:
    return cmu.load()


def _is_vowel(p: str) -> bool:
    return any(c.isdigit() for c in p)


def _strip_stress(p: str) -> str:
    return p.rstrip("0123456789")


def word_to_phonemes(word: str) -> list[str] | None:
    if not word:
        return None
    key = word.replace("’", "'").replace("‘", "'").lower()
    entries = _cmu().get(key)
    if not entries:
        return None
    return list(entries[0])


def split_into_syllables(phonemes: list[str]) -> list[list[str]]:
    if not phonemes:
        return []

    vowel_indices = [i for i, p in enumerate(phonemes) if _is_vowel(p)]
    if not vowel_indices:
        return [list(phonemes)]

    boundaries = [0]
    for i in range(len(vowel_indices) - 1):
        v1 = vowel_indices[i]
        v2 = vowel_indices[i + 1]
        consonants = phonemes[v1 + 1 : v2]
        n = len(consonants)

        if n == 0:
            onset = 0
        elif n == 1:
            onset = 1
        else:
            onset = 1
            if (
                n >= 3
                and tuple(_strip_stress(c) for c in consonants[-3:]) in _VALID_3_ONSETS
            ):
                onset = 3
            elif tuple(_strip_stress(c) for c in consonants[-2:]) in _VALID_2_ONSETS:
                onset = 2

        boundaries.append(v2 - onset)

    boundaries.append(len(phonemes))
    return [
        phonemes[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)
    ]


def phoneme_distance(a: list[str], b: list[str]) -> int:
    aa = [_strip_stress(p) for p in a]
    bb = [_strip_stress(p) for p in b]
    m, n = len(aa), len(bb)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if aa[i - 1] == bb[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]
