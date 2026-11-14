"""Retrieve passages for the BanglaRQA questions with BM25, dense (BGE-M3) and hybrid (RRF) retrieval.

The library is the 3,000 BanglaRQA passages, each kept whole. For every question the script writes the
ranked passage ids to <out-dir>/retrieval_<method>.json ({question_id: [passage_id, ...]}), which
scripts/eval_t5.py reads with --condition retrieved, and prints hit@1 / hit@3 / hit@depth and MRR.
A hit means the question's own passage is among the top results.

BM25 needs only rank_bm25 and runs anywhere. Dense and hybrid need sentence-transformers and a GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bangla_rag.data import build_corpus, sample_questions
from bangla_rag.retrieval import BM25Retriever, DenseRetriever, HybridRetriever


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/banglarqa")
    ap.add_argument("--split", default="Test")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--methods", default="bm25,dense,hybrid", help="comma-separated subset of bm25,dense,hybrid")
    ap.add_argument("--depth", type=int, default=10, help="how many ranked passages to save per question")
    ap.add_argument("--cache", default="cache", help="folder for cached passage embeddings")
    ap.add_argument("--embed-max-len", type=int, default=1024, help="max tokens BGE-M3 reads per passage")
    ap.add_argument("--rrf-k", type=int, default=60)
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    corpus = build_corpus(args.data_dir)  # all splits, one record per passage; index i <-> corpus[i]
    pids = [c["passage_id"] for c in corpus]
    questions = sample_questions(args.data_dir, n=0, split=args.split, answerable_only=False)
    print(f"{len(corpus)} passages | {len(questions)} {args.split} questions")

    retrievers = {}
    t0 = time.time()
    if "bm25" in methods or "hybrid" in methods:
        retrievers["bm25"] = BM25Retriever(corpus)
    if "dense" in methods or "hybrid" in methods:
        retrievers["dense"] = DenseRetriever(corpus, "bge-m3", cache_dir=args.cache, max_seq_length=args.embed_max_len)
    if "hybrid" in methods:
        retrievers["hybrid"] = HybridRetriever(retrievers["bm25"], retrievers["dense"], depth=50, k_rrf=args.rrf_k)
    print(f"indexes ready in {time.time() - t0:.0f}s")

    summary = {}
    for name in methods:
        r = retrievers[name]
        ranked, ranks = {}, []
        t0 = time.time()
        for q in questions:
            hits = r.search(q["question"], args.depth)
            ids = [pids[i] for i, _ in hits]
            ranked[q["question_id"]] = ids
            ranks.append(ids.index(q["passage_id"]) + 1 if q["passage_id"] in ids else None)
        n = len(questions)
        at = lambda k: sum(1 for x in ranks if x is not None and x <= k) / n
        summary[name] = {"hit@1": at(1), "hit@3": at(3), f"hit@{args.depth}": at(args.depth),
                         "mrr": sum(1 / x for x in ranks if x) / n, "n": n}
        (out / f"retrieval_{name}.json").write_text(json.dumps(ranked), encoding="utf-8")
        s = summary[name]
        print(f"{name:7s} hit@1 {100 * s['hit@1']:5.1f}  hit@3 {100 * s['hit@3']:5.1f}  "
              f"hit@{args.depth} {100 * s[f'hit@{args.depth}']:5.1f}  MRR {s['mrr']:.3f}   ({time.time() - t0:.0f}s)")
    (out / "retrieval_summary.json").write_text(json.dumps({"args": vars(args), "summary": summary}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
