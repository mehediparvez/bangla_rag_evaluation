"""Evaluate a fine-tuned BanglaT5 on the BanglaRQA test set under one passage condition.

  --condition gold       the question's own passage (upper bound; compare with the published 62.42 EM / 78.11 F1)
  --condition none       no passage at all (extra, the model was not trained for this)
  --condition retrieved  the top-ranked passage from a retrieval file (--retrieved results.json),
                         a JSON object {question_id: [passage_id, ...]} in ranked order

Writes <out-dir>/<name>_preds.jsonl (one line per question) and <name>_summary.json, and prints a summary.
Scoring is the dataset authors' (see bangla_rag/qa_eval.py). Default precision is fp32, as in their notebook.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from normalizer import normalize  # csebuetnlp/normalizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from bangla_rag.qa_eval import summarise
from bangla_rag.t5_data import load_items, load_passages, question_key
from bangla_rag.t5_infer import predict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="e.g. <Drive>/bangla_t5_rag/run_fast/best")
    ap.add_argument("--condition", choices=["gold", "none", "retrieved"], required=True)
    ap.add_argument("--retrieved", help="retrieval results JSON (required for --condition retrieved)")
    ap.add_argument("--name", help="output file prefix; defaults to the condition (or the retrieval file name)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-dir", default="data/banglarqa")
    ap.add_argument("--split", default="Test")
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke test)")
    args = ap.parse_args()
    if args.condition == "retrieved" and not args.retrieved:
        sys.exit("--condition retrieved needs --retrieved <file.json>")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    name = args.name or (Path(args.retrieved).stem if args.condition == "retrieved" else args.condition)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)
    assert model.lm_head.weight.data_ptr() != model.shared.weight.data_ptr(), "embeddings are tied: wrong checkpoint?"

    items = load_items(args.data_dir, args.split, normalize)
    if args.limit:
        items = items[: args.limit]
    passages = load_passages(args.data_dir, normalize)

    used_pid = [None] * len(items)
    if args.condition == "gold":
        contexts = [it["context"] for it in items]
        used_pid = [it["passage_id"] for it in items]
    elif args.condition == "none":
        contexts = [""] * len(items)
    else:
        ranked = json.loads(Path(args.retrieved).read_text(encoding="utf-8"))
        used_pid = [ranked[question_key(it["question_id"], it["question_raw"])][0] for it in items]
        contexts = [passages[p] for p in used_pid]

    t0 = time.time()
    preds = predict(model, tok, [it["question"] for it in items], contexts, device,
                    batch_size=args.eval_batch, bf16=args.precision == "bf16")
    result = summarise(items, preds)
    summary = result["summary"]
    if args.condition != "none":
        hits = [p == it["passage_id"] for p, it in zip(used_pid, items)]
        summary["passage_hit_at_1"] = sum(hits) / len(hits)

    with open(out / f"{name}_preds.jsonl", "w", encoding="utf-8") as f:
        for it, pred, sc, pid in zip(items, preds, result["per_item"], used_pid):
            f.write(json.dumps({"question_id": it["question_id"], "question": it["question_raw"],
                                "question_type": it["question_type"],
                                "answer_type": it["answer_type"], "gold_passage_id": it["passage_id"],
                                "used_passage_id": pid, "reference": it["answer"], "prediction": pred,
                                "em": sc["em"], "f1": sc["f1"], "answerable": sc["answerable"]},
                               ensure_ascii=False) + "\n")
    (out / f"{name}_summary.json").write_text(json.dumps({"args": vars(args), "summary": summary}, indent=2), encoding="utf-8")

    pct = lambda x: "-" if x is None else f"{100 * x:5.2f}"
    print(f"\n{name}: {len(items)} {args.split} questions in {time.time() - t0:.0f}s ({args.precision})")
    print(f"  overall      EM {pct(summary['overall']['em'])}  F1 {pct(summary['overall']['f1'])}   (published gold: 62.42 / 78.11)")
    print(f"  answerable   EM {pct(summary['has_answer']['em'])}  F1 {pct(summary['has_answer']['f1'])}   n={summary['has_answer']['n']}   (published: 54.89 / 75.73)")
    print(f"  unanswerable EM {pct(summary['no_answer']['em'])}                     n={summary['no_answer']['n']}   (published: 85.37)")
    print(f"  empty answers on answerable questions: {pct(summary['empty_pred_rate_on_answerable'])}%")
    if "passage_hit_at_1" in summary:
        print(f"  passage hit@1: {pct(summary['passage_hit_at_1'])}%")
    for group in ("by_question_type", "by_answer_type"):
        for k, v in summary[group].items():
            print(f"  {k:15s} EM {pct(v['em'])}  F1 {pct(v['f1'])}  n={v['n']}")


if __name__ == "__main__":
    main()
