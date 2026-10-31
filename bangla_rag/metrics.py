"""Answer-quality and retrieval metrics (no heavy dependencies)."""
from __future__ import annotations

from collections import Counter
from typing import Iterable

from .text import normalize, tokenize

CORRECT_F1_THRESHOLD = 0.5  # an answer counts as "correct" when token-F1 >= this


def exact_match(pred: str, refs: Iterable[str]) -> float:
    p = normalize(pred)
    return float(bool(p) and any(p == normalize(r) for r in refs))


def f1_score(pred: str, refs: Iterable[str]) -> float:
    """Best token-level F1 against any reference answer."""
    pt = tokenize(pred)
    best = 0.0
    for r in refs:
        rt = tokenize(r)
        if not pt or not rt:
            continue
        overlap = sum((Counter(pt) & Counter(rt)).values())
        if overlap == 0:
            continue
        prec, rec = overlap / len(pt), overlap / len(rt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def answer_in_text(refs: Iterable[str], text: str) -> bool:
    """True if any reference answer appears (after normalisation) in ``text``."""
    t = normalize(text)
    return any(n and n in t for n in (normalize(r) for r in refs))


def is_correct(f1: float) -> bool:
    return f1 >= CORRECT_F1_THRESHOLD
