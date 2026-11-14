"""The question-answering system as one object: retrieve a passage, then let the fine-tuned BanglaT5 read it.

``BanglaRAGSystem`` takes its parts as arguments so it can be tested with fakes; ``BanglaRAGSystem.load``
builds the real thing (BM25, BGE-M3 dense, hybrid, and the fine-tuned model from a checkpoint folder).
"""
from __future__ import annotations

import time

from .qa_eval import normalize_text

NO_ANSWER_TEXT = "উত্তর পাওয়া যায়নি (no answer found in the retrieved passage)"
METHOD_LABELS = {"bm25": "BM25 (keywords)", "dense": "Dense (BGE-M3, meaning)", "hybrid": "Hybrid (BM25 + dense, RRF)"}


class BanglaRAGSystem:
    def __init__(self, corpus: list[dict], retrievers: dict, reader, normalize):
        """corpus: [{'passage_id','title','text'}]; retrievers: {name: object with .search(query, k)};
        reader: callable(question, passage) -> answer string (inputs already normalised)."""
        self.corpus, self.retrievers, self.reader, self.normalize = corpus, retrievers, reader, normalize

    def retrieve(self, question: str, method: str, k: int = 3) -> list[dict]:
        hits = self.retrievers[method].search(question, k)
        return [{"rank": r + 1, "passage_id": self.corpus[i]["passage_id"], "title": self.corpus[i].get("title", ""),
                 "text": self.corpus[i]["text"], "score": float(s)} for r, (i, s) in enumerate(hits)]

    def answer(self, question: str, method: str = "bm25", show: int = 3) -> dict:
        """Search, give the top passage to the reader, and return the answer with the evidence shown."""
        t0 = time.time()
        passages = self.retrieve(question, method, k=max(show, 1))
        raw = self.reader(self.normalize(question), self.normalize(passages[0]["text"]))
        abstained = normalize_text(raw) == ""
        return {"answer": NO_ANSWER_TEXT if abstained else raw.strip(), "abstained": abstained, "method": method,
                "passages": passages, "seconds": time.time() - t0}

    @classmethod
    def load(cls, model_dir: str, data_dir: str = "data/banglarqa", cache_dir: str = "cache", bf16: bool = False):
        import torch
        from normalizer import normalize
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        from .data import build_corpus
        from .retrieval import BM25Retriever, DenseRetriever, HybridRetriever
        from .t5_infer import predict

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device).eval()
        corpus = build_corpus(data_dir)
        bm25 = BM25Retriever(corpus)
        dense = DenseRetriever(corpus, "bge-m3", cache_dir=cache_dir, max_seq_length=1024)
        retrievers = {"bm25": bm25, "dense": dense, "hybrid": HybridRetriever(bm25, dense, depth=50, k_rrf=60)}

        def reader(question: str, passage: str) -> str:
            return predict(model, tok, [question], [passage], device, batch_size=1, bf16=bf16)[0]

        return cls(corpus, retrievers, reader, normalize)
