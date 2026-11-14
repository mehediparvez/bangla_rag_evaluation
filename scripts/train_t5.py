"""Fine-tune csebuetnlp/banglat5 on BanglaRQA question answering.

Setup follows the dataset authors' notebooks (github.com/sartajekram419/BanglaRQA):
input = (question, gold passage) as a tokenizer pair, max 1024 tokens, context truncated;
target = first reference answer, empty string for unanswerable questions;
all train questions used; validation EM/F1 after every epoch with their greedy decoding.

Two recipes, both overridable from the command line:
  authors  15 epochs, batch 2, AdamW 5e-5, constant lr, no clipping (their published recipe)
  fast     10 epochs, effective batch 16, AdamW 1e-4, 5% warmup + linear decay, clip 1.0

Outputs in --out-dir:  config.json, log.jsonl, best/ (model with best validation F1),
last.pt (full state, overwritten each epoch, used by --resume), training_curve.png.

Speed / precision check:  --max-steps 150 --precision bf16   (then fp32) and compare loss and examples/s.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from normalizer import normalize  # csebuetnlp/normalizer: required by BanglaT5, no fallback here
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, get_constant_schedule,
                          get_linear_schedule_with_warmup)

from bangla_rag.qa_eval import summarise
from bangla_rag.t5_data import length_grouped_batches, load_items

PRESETS = {
    "authors": dict(epochs=15, batch_size=2, grad_accum=1, lr=5e-5, scheduler="constant", warmup=0.0, clip=0.0),
    "fast": dict(epochs=10, batch_size=8, grad_accum=2, lr=1e-4, scheduler="linear", warmup=0.05, clip=1.0),
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", choices=list(PRESETS), default="fast")
    ap.add_argument("--data-dir", default="data/banglarqa")
    ap.add_argument("--out-dir", required=True, help="put this on Google Drive on Colab")
    for name in ("epochs", "batch_size", "grad_accum"):
        ap.add_argument(f"--{name.replace('_', '-')}", type=int)
    for name in ("lr", "warmup", "clip"):
        ap.add_argument(f"--{name}", type=float)
    ap.add_argument("--scheduler", choices=["constant", "linear"])
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16", help="never fp16: T5 gives NaN")
    ap.add_argument("--max-source", type=int, default=1024)
    ap.add_argument("--max-target", type=int, default=256)
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--val-n", type=int, default=0, help="validate on the first N questions only; 0 = all")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (speed test, no save)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--resume", action="store_true", help="continue from out-dir/last.pt if it exists")
    args = ap.parse_args()
    for k, v in PRESETS[args.recipe].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


def encode(items, tok, args):
    src = tok([i["question"] for i in items], [i["context"] for i in items],
              max_length=args.max_source, truncation="only_second")["input_ids"]
    tgt = tok([i["answer"] for i in items], max_length=args.max_target, truncation=True)["input_ids"]
    return [{"input_ids": s, "labels": t} for s, t in zip(src, tgt)]


def pad(seqs, value):
    n = max(len(s) for s in seqs)
    return torch.tensor([s + [value] * (n - len(s)) for s in seqs])


def collate(examples, device):
    ids = pad([e["input_ids"] for e in examples], 0)  # T5 pad id is 0
    return {"input_ids": ids.to(device), "attention_mask": (ids != 0).long().to(device),
            "labels": pad([e["labels"] for e in examples], -100).to(device)}


@torch.no_grad()
def predict(model, tok, items, args, device, autocast):
    """Greedy decoding with the authors' settings; returns predictions in the order of ``items``."""
    model.eval()
    enc = tok([i["question"] for i in items], [i["context"] for i in items],
              max_length=args.max_source, truncation="only_second")["input_ids"]
    order = sorted(range(len(items)), key=lambda i: len(enc[i]))
    preds = [""] * len(items)
    for s in range(0, len(order), args.eval_batch):
        idx = order[s : s + args.eval_batch]
        ids = pad([enc[i] for i in idx], 0).to(device)
        with autocast():
            out = model.generate(input_ids=ids, attention_mask=(ids != 0).long(), num_beams=1,
                                 max_length=args.max_target, repetition_penalty=2.5,
                                 length_penalty=1.0, early_stopping=True, use_cache=True)
        for i, text in zip(idx, tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)):
            preds[i] = text
    model.train()
    return preds


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)  # a disconnect mid-save cannot corrupt the previous file


def plot_curve(log, path: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ep = [r["epoch"] + 1 for r in log]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(ep, [r["train_loss"] for r in log], marker="o", label="train loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("train loss")
    ax2 = ax.twinx()
    ax2.plot(ep, [r["val_f1"] for r in log], marker="s", color="tab:orange", label="validation F1")
    ax2.set_ylabel("validation F1")
    fig.legend(loc="center right"); fig.tight_layout(); fig.savefig(path, dpi=150)


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16" and device.type == "cuda")

    import transformers
    (out / "config.json").write_text(json.dumps(
        {**vars(args), "torch": torch.__version__, "transformers": transformers.__version__}, indent=2), encoding="utf-8")

    tok = AutoTokenizer.from_pretrained("csebuetnlp/banglat5", use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/banglat5").to(device)

    train_items = load_items(args.data_dir, "Train", normalize)
    val_items = load_items(args.data_dir, "Validation", normalize)
    if args.val_n:
        val_items = val_items[: args.val_n]
    train = encode(train_items, tok, args)
    lengths = [len(e["input_ids"]) for e in train]
    print(f"train {len(train)} | validation {len(val_items)} | source tokens median {int(np.median(lengths))} "
          f"max {max(lengths)} | truncated {sum(l >= args.max_source for l in lengths)}")
    print(f"recipe={args.recipe} epochs={args.epochs} batch={args.batch_size}x{args.grad_accum} lr={args.lr} "
          f"sched={args.scheduler} precision={args.precision}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    batches_per_epoch = math.ceil(len(train) / args.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    sched = (get_constant_schedule(opt) if args.scheduler == "constant"
             else get_linear_schedule_with_warmup(opt, int(args.warmup * total_steps), total_steps))

    start_epoch, best_f1, log = 0, -1.0, []
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_epoch, best_f1, log = ck["epoch"], ck["best_f1"], ck["log"]
        print(f"resumed: starting at epoch {start_epoch + 1}, best validation F1 so far {best_f1:.4f}")

    model.train()
    step = 0
    for epoch in range(start_epoch, args.epochs):
        rng = random.Random(args.seed + epoch)
        batches = length_grouped_batches(lengths, args.batch_size, rng)
        t0, loss_sum, n_seen = time.time(), 0.0, 0
        opt.zero_grad()
        for b, idx in enumerate(batches):
            batch = collate([train[i] for i in idx], device)
            with autocast():
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                sys.exit(f"non-finite loss at epoch {epoch + 1}, batch {b}: stop and check precision")
            (loss / args.grad_accum).backward()
            loss_sum += loss.item() * len(idx); n_seen += len(idx)
            if (b + 1) % args.grad_accum == 0 or b + 1 == len(batches):
                if args.clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 100 == 0:
                    print(f"epoch {epoch + 1} step {step} loss {loss_sum / n_seen:.4f} "
                          f"{n_seen / (time.time() - t0):.1f} ex/s", flush=True)
                if args.max_steps and step >= args.max_steps:
                    print(f"SPEED TEST: {step} steps, mean loss {loss_sum / n_seen:.4f}, "
                          f"{n_seen / (time.time() - t0):.1f} examples/s, peak GPU memory "
                          f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
                    return
        train_secs = time.time() - t0

        preds = predict(model, tok, val_items, args, device, autocast)
        s = summarise(val_items, preds)["summary"]
        rec = {"epoch": epoch, "train_loss": loss_sum / n_seen, "val_em": s["overall"]["em"], "val_f1": s["overall"]["f1"],
               "val_has_answer_em": s["has_answer"]["em"], "val_no_answer_em": s["no_answer"]["em"],
               "train_secs": round(train_secs), "total_secs": round(time.time() - t0)}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        with open(out / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if rec["val_f1"] > best_f1:
            best_f1 = rec["val_f1"]
            model.save_pretrained(out / "best"); tok.save_pretrained(out / "best")
            print(f"new best validation F1 {best_f1:.4f}: saved to {out / 'best'}")
        atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "epoch": epoch + 1, "best_f1": best_f1, "log": log}, out / "last.pt")
        plot_curve(log, out / "training_curve.png")
    print(f"done. best validation F1 {best_f1:.4f}")


if __name__ == "__main__":
    main()
