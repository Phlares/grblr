_PUNCT = '.,!?;:"()[]'


def normalize(token: str) -> str:
    """Normalize a token for corpus lookup.

    Strips surrounding ASCII punctuation, normalizes curly apostrophes (U+2018
    and U+2019) to ASCII, and lowercases. Internal apostrophes (e.g. in "don't")
    are preserved. Hyphenated tokens stay as a single token.
    """
    return token.strip(_PUNCT).replace("’", "'").replace("‘", "'").lower()
