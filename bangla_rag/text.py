"""Bangla-aware text normalisation and tokenisation.

Python's ``\\w`` splits Bangla words at vowel signs (matras) because those are
combining marks, so we never use it. Instead we replace punctuation and symbols
with spaces and split on whitespace, keeping combining marks attached.
"""
from __future__ import annotations

import re
import unicodedata

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_WS = re.compile(r"\s+")


def normalize(text: str | None) -> str:
    """NFC-normalise, map Bangla digits to ASCII, drop punctuation, lowercase."""
    s = unicodedata.normalize("NFC", text or "")
    s = s.translate(_BN_DIGITS)
    s = s.replace("​", "").replace("﻿", "").replace(" ", " ")
    s = "".join(" " if unicodedata.category(c)[0] in ("P", "S") else c for c in s)
    return _WS.sub(" ", s.lower()).strip()


def tokenize(text: str | None) -> list[str]:
    n = normalize(text)
    return n.split() if n else []
