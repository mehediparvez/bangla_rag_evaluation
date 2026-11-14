"""BanglaRQA scoring, so our numbers are comparable to the dataset authors' published ones.

Reimplemented from the evaluation logic in the authors' public notebook
(github.com/sartajekram419/BanglaRQA, BanglaT5_Testing.ipynb; CC BY-NC-SA 4.0).
Behaviours kept on purpose, even where a different choice would be arguable:

* only the FIRST reference answer (``answer_text[0]``) is scored;
* text normalisation is lower-case, ASCII punctuation removal, English article
  removal and whitespace collapse (the Bangla danda is NOT removed);
* an unanswerable question (empty gold answer) is correct only if the
  prediction is also empty;
* multi-span answers are ';'-separated and scored with a Hungarian-assignment F1
  averaged over max(#gold spans, #predicted spans), so extra or missing spans
  both lower the score.

Inputs to ``score_item`` are expected to be already passed through
csebuetnlp's ``normalize`` (see ``t5_data.py``), as the authors do.
"""
from __future__ import annotations

import re
import string
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

_PUNCT = set(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def normalize_text(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def compute_exact_match(prediction: str, truth: str) -> int:
    return int(normalize_text(prediction) == normalize_text(truth))


def compute_f1(prediction: str, truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    # no-answer on either side: 1 if both are empty, otherwise 0
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    common = sum((Counter(pred_tokens) & Counter(truth_tokens)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(pred_tokens), common / len(truth_tokens)
    return 2 * prec * rec / (prec + rec)


def multi_span_f1(prediction: str, truth: str) -> float:
    pred_items, truth_items = prediction.split(";"), truth.split(";")
    scores = np.zeros((len(truth_items), len(pred_items)))
    for gi, g in enumerate(truth_items):
        for pi, p in enumerate(pred_items):
            scores[gi, pi] = compute_f1(p, g)
    rows, cols = linear_sum_assignment(-scores)
    best = np.zeros(max(len(truth_items), len(pred_items)))
    for r, c in zip(rows, cols):
        best[r] = max(best[r], scores[r, c])
    return float(best.mean())


def score_item(prediction: str, gold: str, answer_type: str) -> tuple[int, float]:
    """Return (exact match, F1) for one question."""
    f1 = multi_span_f1(prediction, gold) if answer_type == "multiple spans" else compute_f1(prediction, gold)
    return compute_exact_match(prediction, gold), f1


def contains_match(prediction: str, truth: str) -> int:
    """Relaxed match: every ';'-separated reference span appears inside the prediction.

    Both sides are normalised and the Bangla danda is ignored. An unanswerable question (empty reference)
    is still correct only if the prediction is empty. This favours long answers, so it is a secondary measure.
    """
    squash = lambda s: normalize_text(s.replace("।", " "))
    pred = squash(prediction)
    spans = [s for s in (squash(x) for x in truth.split(";")) if s]
    if not spans:
        return int(pred == "")
    return int(pred != "" and all(s in pred for s in spans))


def _mean(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def summarise(items: list[dict], preds: list[str]) -> dict:
    """Aggregate scores in the authors' breakdowns plus HasAns / NoAns and the empty-prediction rate.

    ``items`` come from ``t5_data.load_items``; ``preds`` is aligned with it.
    Returns the summary and per-item scores (for paired tests).
    """
    per_item = []
    for it, pred in zip(items, preds):
        em, f1 = score_item(pred, it["answer"], it["answer_type"])
        per_item.append({"question_id": it["question_id"], "em": em, "f1": f1,
                         "answerable": it["answer"] != "", "pred_empty": normalize_text(pred) == ""})

    def agg(mask):
        rows = [p for p, m in zip(per_item, mask) if m]
        return {"n": len(rows), "em": _mean([r["em"] for r in rows]), "f1": _mean([r["f1"] for r in rows])}

    ans = [p["answerable"] for p in per_item]
    summary = {
        "overall": agg([True] * len(per_item)),
        "has_answer": agg(ans),
        "no_answer": agg([not a for a in ans]),
        "empty_pred_rate_on_answerable": _mean([float(p["pred_empty"]) for p in per_item if p["answerable"]]),
        "by_question_type": {t: agg([it["question_type"] == t for it in items])
                             for t in ("factoid", "causal", "confirmation", "list")},
        "by_answer_type": {t: agg([it["answer_type"] == t and a for it, a in zip(items, ans)])
                           for t in ("single span", "multiple spans", "yes/no")},
    }
    return {"summary": summary, "per_item": per_item}
