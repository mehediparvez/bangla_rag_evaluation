"""Greedy answer generation with a fine-tuned BanglaT5, using the dataset authors' decoding settings."""
from __future__ import annotations

import torch


def pad(seqs: list[list[int]], value: int) -> torch.Tensor:
    n = max(len(s) for s in seqs)
    return torch.tensor([s + [value] * (n - len(s)) for s in seqs])


@torch.no_grad()
def predict(model, tok, questions: list[str], contexts: list[str], device, batch_size: int = 32,
            max_source: int = 1024, max_target: int = 256, bf16: bool = False) -> list[str]:
    """Answer each (question, context) pair; results are returned in the input order.

    An empty context is allowed (no-passage condition). The context is truncated, never the question.
    """
    model.eval()
    enc = tok(questions, contexts, max_length=max_source, truncation="only_second")["input_ids"]
    order = sorted(range(len(questions)), key=lambda i: len(enc[i]))  # similar lengths together
    preds = [""] * len(questions)
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        ids = pad([enc[i] for i in idx], 0).to(device)  # T5 pad id is 0
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
            out = model.generate(input_ids=ids, attention_mask=(ids != 0).long(), num_beams=1,
                                 max_length=max_target, repetition_penalty=2.5, length_penalty=1.0,
                                 early_stopping=True, use_cache=True)
        for i, text in zip(idx, tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)):
            preds[i] = text
    return preds
