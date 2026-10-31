"""BanglaRQA loading. Run ``python -m bangla_rag.data`` to download the dataset.

BanglaRQA (Ekram et al., EMNLP Findings 2022) is licensed CC BY-NC-SA 4.0:
non-commercial research use with attribution. We download it at run time and
never redistribute it from this repository.
"""
from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path

HF_URL = "https://huggingface.co/datasets/sartajekram/BanglaRQA/resolve/main/{}.json"
SPLITS = ("Train", "Validation", "Test")


def download(data_dir: str | Path = "data/banglarqa") -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        path = data_dir / f"{split}.json"
        if path.exists() and path.stat().st_size > 0:
            continue
        print(f"downloading {split} ...")
        urllib.request.urlretrieve(HF_URL.format(split), path)
    return data_dir


def load_split(data_dir: str | Path, split: str) -> list[dict]:
    with open(Path(data_dir) / f"{split}.json", encoding="utf-8") as f:
        d = json.load(f)
    return d["data"] if isinstance(d, dict) and "data" in d else d


def build_corpus(data_dir: str | Path = "data/banglarqa", splits=SPLITS) -> list[dict]:
    """All passages from every split form one retrieval pool."""
    corpus = []
    for split in splits:
        for p in load_split(data_dir, split):
            corpus.append(
                {"passage_id": p["passage_id"], "title": p.get("title", ""), "text": p["context"]}
            )
    return corpus


def _clean_answers(answer_text) -> list[str]:
    if isinstance(answer_text, str):
        answer_text = [answer_text]
    out = []
    for a in answer_text or []:
        a = (a or "").strip()
        if a and a not in out:
            out.append(a)
    return out


def sample_questions(
    data_dir: str | Path = "data/banglarqa",
    n: int = 200,
    seed: int = 13,
    split: str = "Test",
    answerable_only: bool = True,
) -> list[dict]:
    """Deterministic sample of questions; ``n <= 0`` returns all of them."""
    questions = []
    for p in load_split(data_dir, split):
        for qa in p["qas"]:
            answerable = str(qa.get("is_answerable")) == "1"
            if answerable_only and not answerable:
                continue
            refs = _clean_answers(qa["answers"].get("answer_text"))
            if answerable_only and not refs:
                continue
            questions.append(
                {
                    "question_id": qa["question_id"],
                    "passage_id": p["passage_id"],
                    "question": qa["question_text"].strip(),
                    "answers": refs,
                    "question_type": qa.get("question_type", ""),
                }
            )
    questions.sort(key=lambda q: q["question_id"])
    if n and 0 < n < len(questions):
        questions = random.Random(seed).sample(questions, n)
        questions.sort(key=lambda q: q["question_id"])
    return questions


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download BanglaRQA")
    ap.add_argument("--data-dir", default="data/banglarqa")
    args = ap.parse_args()
    download(args.data_dir)
    corpus = build_corpus(args.data_dir)
    qs = sample_questions(args.data_dir, n=0)
    print(f"passages: {len(corpus)}; answerable test questions: {len(qs)}")
