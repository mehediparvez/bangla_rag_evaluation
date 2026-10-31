"""Run the RQ1 retrieval ablation.

Examples
--------
Retrieval metrics only (no API key, no cost)::

    python -m bangla_rag.run --preset core --retrieval-only --n 0

Full run with Gemini (resumable; Ctrl-C and re-run safely)::

    python -m bangla_rag.run --preset core --n 200 --rpm 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

from .chunking import build_chunks
from .data import build_corpus, download, sample_questions
from .metrics import answer_in_text, exact_match, f1_score, is_correct
from .prompts import build_prompt, parse_response

DEFAULT_CHUNK, DEFAULT_K, DEFAULT_EMBEDDER = 512, 3, "bge-m3"


def make_spec(retrieval: str, embedder: str = DEFAULT_EMBEDDER, chunk: int = DEFAULT_CHUNK, k: int = DEFAULT_K):
    if retrieval == "none":
        return {"run_id": "none", "retrieval": "none"}
    kind = {"bm25": "bm25", "dense": f"dense-{embedder}", "hybrid": f"hybrid-{embedder}"}[retrieval]
    return {
        "run_id": f"{kind}_c{chunk}_k{k}",
        "retrieval": retrieval,
        "embedder": embedder,
        "chunk": chunk,
        "k": k,
    }


# One factor changes at a time from the default (dense bge-m3, 512 chars, top-3).
ALL_SPECS = [
    make_spec("none"),
    make_spec("bm25"),
    make_spec("dense"),
    make_spec("hybrid"),
    make_spec("dense", embedder="qwen3-emb"),
    make_spec("dense", embedder="multilingual-e5"),
    make_spec("dense", chunk=256),
    make_spec("dense", chunk=1000),
    make_spec("dense", k=5),
]
RUNS = {s["run_id"]: s for s in ALL_SPECS}
DEFAULT_RUN = make_spec("dense")["run_id"]
PRESETS = {
    "core": ["none", "bm25_c512_k3", DEFAULT_RUN, make_spec("hybrid")["run_id"]],
    "full": [s["run_id"] for s in ALL_SPECS],
}


class RetrieverFactory:
    """Builds chunks and retrievers lazily and reuses them across runs."""

    def __init__(self, corpus, cache_dir, overlap_ratio):
        self.corpus, self.cache_dir, self.overlap_ratio = corpus, cache_dir, overlap_ratio
        self._chunks, self._bm25, self._dense = {}, {}, {}

    def chunks(self, size):
        if size not in self._chunks:
            self._chunks[size] = build_chunks(self.corpus, size, self.overlap_ratio)
            print(f"[chunks] size={size}: {len(self._chunks[size])} chunks")
        return self._chunks[size]

    def get(self, spec):
        from .retrieval import BM25Retriever, DenseRetriever, HybridRetriever

        size = spec["chunk"]
        chunks = self.chunks(size)
        if spec["retrieval"] in ("bm25", "hybrid") and size not in self._bm25:
            self._bm25[size] = BM25Retriever(chunks)
        if spec["retrieval"] == "bm25":
            return self._bm25[size]
        dkey = (spec["embedder"], size)
        if dkey not in self._dense:
            self._dense[dkey] = DenseRetriever(chunks, spec["embedder"], self.cache_dir)
        if spec["retrieval"] == "dense":
            return self._dense[dkey]
        return HybridRetriever(self._bm25[size], self._dense[dkey])


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["question_id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def process(q, spec, retriever, chunks, gen):
    rec = {"run_id": spec["run_id"], "question_id": q["question_id"], "question": q["question"],
           "references": q["answers"], "question_type": q["question_type"], "gold_passage_id": q["passage_id"]}
    contexts = []
    if retriever is not None:
        hits = retriever.search(q["question"], spec["k"])
        retrieved = [chunks[i] for i, _ in hits]
        contexts = [c["text"] for c in retrieved]
        pids = [c["passage_id"] for c in retrieved]
        rec["retrieved_chunk_ids"] = [c["chunk_id"] for c in retrieved]
        rec["contexts"] = contexts
        rec["passage_hit"] = q["passage_id"] in pids
        rec["gold_rank"] = pids.index(q["passage_id"]) + 1 if q["passage_id"] in pids else None
        rec["answer_in_context"] = answer_in_text(q["answers"], " ".join(contexts))
    if gen is not None:
        resp = gen.generate(build_prompt(q["question"], contexts))
        answer, explanation = parse_response(resp["text"])
        rec.update(answer=answer, explanation=explanation, raw_response=resp["text"])
        rec["em"] = exact_match(answer, q["answers"])
        rec["f1"] = f1_score(answer, q["answers"])
        rec["correct"] = is_correct(rec["f1"])
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS), default="core")
    ap.add_argument("--runs", help="comma-separated run ids (overrides --preset)")
    ap.add_argument("--n", type=int, default=200, help="number of questions; 0 = all answerable test questions")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--split", default="Test")
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument("--rpm", type=float, default=10, help="max Gemini requests per minute (free tier is low)")
    ap.add_argument("--overlap", type=float, default=0.1, help="chunk overlap as a fraction of chunk size")
    ap.add_argument("--data-dir", default="data/banglarqa")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--retrieval-only", action="store_true", help="skip generation; compute retrieval metrics only")
    args = ap.parse_args(argv)

    run_ids = args.runs.split(",") if args.runs else PRESETS[args.preset]
    unknown = [r for r in run_ids if r not in RUNS]
    if unknown:
        sys.exit(f"unknown run ids {unknown}; choose from {list(RUNS)}")

    download(args.data_dir)
    corpus = build_corpus(args.data_dir)
    questions = sample_questions(args.data_dir, n=args.n, seed=args.seed, split=args.split)
    print(f"corpus: {len(corpus)} passages | questions: {len(questions)} | runs: {run_ids}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    factory = RetrieverFactory(corpus, args.cache, args.overlap)
    gen = None
    if not args.retrieval_only:
        from .generation import GeminiGenerator

        gen = GeminiGenerator(args.model, rpm=args.rpm, cache_path=out / "gemini_cache.jsonl")

    from .generation import DailyQuotaExceeded

    for run_id in run_ids:
        spec = RUNS[run_id]
        if args.retrieval_only and spec["retrieval"] == "none":
            continue
        path = out / (f"{run_id}.retrieval.jsonl" if args.retrieval_only else f"{run_id}.jsonl")
        seen = done_ids(path)
        todo = [q for q in questions if q["question_id"] not in seen]
        print(f"\n== {run_id}: {len(todo)} to do ({len(seen)} already done)")
        if not todo:
            continue
        retriever = factory.get(spec) if spec["retrieval"] != "none" else None
        chunks = factory.chunks(spec["chunk"]) if retriever is not None else None
        try:
            with open(path, "a", encoding="utf-8") as f:
                for q in tqdm(todo):
                    rec = process(q, spec, retriever, chunks, gen)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
        except DailyQuotaExceeded as e:
            print(f"\nDaily Gemini quota reached. Progress is saved; re-run the same command tomorrow.\n{e}")
            sys.exit(2)
    print("\ndone. Next: python -m bangla_rag.analyze")


if __name__ == "__main__":
    main()
