"""Retrievers: BM25, dense (sentence-transformers + FAISS) and hybrid (RRF)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .text import tokenize

# Embedders compared in RQ1. Prefixes follow each model's documented usage.
EMBEDDERS = {
    "bge-m3": {
        "name": "BAAI/bge-m3",
        "query_prefix": "",
        "passage_prefix": "",
    },
    "multilingual-e5": {
        "name": "intfloat/multilingual-e5-base",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
    "qwen3-emb": {
        "name": "Qwen/Qwen3-Embedding-0.6B",
        "query_prefix": "Instruct: Given a Bangla question, retrieve the passage that answers it\nQuery:",
        "passage_prefix": "",
    },
    "bengali-sbert": {
        "name": "l3cube-pune/bengali-sentence-similarity-sbert",
        "query_prefix": "",
        "passage_prefix": "",
    },
}


def top_k(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    order = np.argsort(-scores, kind="stable")[:k]
    return [(int(i), float(scores[i])) for i in order]


def rrf(rankings: list[list[tuple[int, float]]], top: int, k_rrf: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion of several ranked lists of (index, score)."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _) in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k_rrf + rank + 1)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:top]


class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        return top_k(np.asarray(self.bm25.get_scores(tokenize(query))), k)


class DenseRetriever:
    def __init__(
        self,
        chunks: list[dict],
        embedder: str = "bge-m3",
        cache_dir: str | Path = "cache",
        batch_size: int = 32,
        device: str | None = None,
        max_seq_length: int = 512,
    ):
        self.spec = EMBEDDERS[embedder]
        self.device = device
        self.max_seq_length = max_seq_length
        self._model = None
        texts = [c["text"] for c in chunks]
        digest = hashlib.md5(
            ("\n".join(texts) + self.spec["name"] + self.spec["passage_prefix"] + str(max_seq_length)).encode("utf-8")
        ).hexdigest()[:12]
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"emb_{embedder}_{digest}.npy"
        if path.exists():
            emb = np.load(path)
        else:
            emb = self._encode([self.spec["passage_prefix"] + t for t in texts], batch_size)
            np.save(path, emb)
        self.emb = np.ascontiguousarray(emb, dtype="float32")
        try:
            import faiss

            self.index = faiss.IndexFlatIP(self.emb.shape[1])
            self.index.add(self.emb)
        except ImportError:  # numpy fallback; exact search either way
            self.index = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.spec["name"], device=self.device)
            self._model.max_seq_length = min(self._model.max_seq_length or self.max_seq_length, self.max_seq_length)
        return self._model

    def _encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self._get_model().encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self._encode([self.spec["query_prefix"] + query], batch_size=1).astype("float32")
        if self.index is not None:
            scores, ids = self.index.search(q, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        return top_k(self.emb @ q[0], k)


class HybridRetriever:
    """Reciprocal-rank fusion of BM25 and a dense retriever."""

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, depth: int = 50, k_rrf: int = 60):
        self.bm25, self.dense, self.depth, self.k_rrf = bm25, dense, depth, k_rrf

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        return rrf(
            [self.bm25.search(query, self.depth), self.dense.search(query, self.depth)],
            top=k,
            k_rrf=self.k_rrf,
        )
