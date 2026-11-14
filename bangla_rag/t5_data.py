"""Examples for fine-tuning and scoring BanglaT5 on BanglaRQA.

Format follows the dataset authors' notebooks: question and context are
normalised with csebuetnlp's ``normalize``, the target is the normalised
``answer_text[0]`` (empty string for an unanswerable question), and every
question is kept, answerable or not. The tokenizer encodes (question, context)
as a pair with the context truncated.
"""
from __future__ import annotations

import warnings
from pathlib import Path

from .data import load_split


def get_normalizer():
    """csebuetnlp normalizer (required for real runs); identity fallback only for local tests."""
    try:
        from normalizer import normalize

        return normalize
    except ImportError:
        warnings.warn("csebuetnlp normalizer not installed; using identity (fine for tests, not for training)")
        return lambda s: s


def load_items(data_dir: str | Path, split: str, normalize=None) -> list[dict]:
    """One dict per question (answerable or not), with the gold passage as ``context``."""
    normalize = normalize or get_normalizer()
    items = []
    for p in load_split(data_dir, split):
        gold_context = normalize(p["context"])
        for qa in p["qas"]:
            ans = qa["answers"]
            items.append(
                {
                    "question_id": qa["question_id"],
                    "passage_id": p["passage_id"],
                    "question": normalize(qa["question_text"]),
                    "question_raw": qa["question_text"].strip(),
                    "context": gold_context,
                    "answer": normalize(ans["answer_text"][0]),
                    "answer_type": normalize(ans["answer_type"][0]),
                    "question_type": normalize(qa["question_type"]),
                    "is_answerable": str(qa.get("is_answerable")) == "1",
                }
            )
    return items


def question_key(question_id: str, question_text: str) -> str:
    """Unique key for a question. BanglaRQA reuses a few question ids (five in the Test split)."""
    return f"{question_id}|{question_text.strip()}"


def length_grouped_batches(lengths: list[int], batch_size: int, rng, mega: int = 50) -> list[list[int]]:
    """Batches of example indices with similar lengths, so padding wastes little compute.

    Shuffle, cut into chunks of ``batch_size * mega``, sort each chunk by length,
    split into batches, then shuffle the batch order. Every index appears once.
    """
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    chunk = batch_size * mega
    batches = []
    for s in range(0, len(idx), chunk):
        part = sorted(idx[s : s + chunk], key=lambda i: lengths[i], reverse=True)
        batches += [part[b : b + batch_size] for b in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    return batches


def load_passages(data_dir: str | Path, normalize=None, splits=("Train", "Validation", "Test")) -> dict[str, str]:
    """passage_id -> normalised passage text, over the given splits (all 3,000 by default)."""
    normalize = normalize or get_normalizer()
    return {p["passage_id"]: normalize(p["context"]) for s in splits for p in load_split(data_dir, s)}
