"""Step 0: how many BanglaRQA passages are too long for BanglaT5's 512-token input?

Run this first (Colab or local). It decides whether whole passages can be used
as the retrieval unit, or whether long passages get truncated and lose answers.

Setup:
    pip install transformers sentencepiece
    pip install git+https://github.com/csebuetnlp/normalizer
    python -m bangla_rag.data            # download BanglaRQA
    python scripts/check_token_lengths.py
"""
from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from normalizer import normalize  # csebuetnlp/normalizer, required by BanglaT5
from transformers import AutoTokenizer

from bangla_rag.data import SPLITS, load_split

MAX_LEN = 512
tok = AutoTokenizer.from_pretrained("csebuetnlp/banglat5", use_fast=False)


def n_tokens(text: str) -> int:
    return len(tok(normalize(text), add_special_tokens=True)["input_ids"])


for split in SPLITS:
    passages = load_split("data/banglarqa", split)
    lens = sorted(n_tokens(p["context"]) for p in passages)
    over = sum(l > MAX_LEN for l in lens)
    print(
        f"{split:10s} passages={len(lens)}  median={st.median(lens):.0f}  "
        f"p90={lens[int(0.9 * len(lens))]}  max={lens[-1]}  "
        f">{MAX_LEN} tokens: {over} ({100 * over / len(lens):.1f}%)"
    )
