"""Character-based chunking (the unit the proposal varies: 256 / 512 / 1000)."""
from __future__ import annotations

_BREAKS = (" ", "\n", "।", "?", "!")


def chunk_text(text: str, size: int, overlap: int = 0) -> list[str]:
    """Split ``text`` into windows of about ``size`` characters.

    Windows end at a whitespace or sentence mark when one is found in the last
    30% of the window, so words are not cut in half.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    overlap = max(0, min(overlap, size - 1))
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window_floor = start + int(size * 0.7)
            cut = max(text.rfind(b, window_floor, end) for b in _BREAKS)
            if cut > start:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(corpus: list[dict], size: int, overlap_ratio: float = 0.1) -> list[dict]:
    overlap = int(size * overlap_ratio)
    chunks = []
    for p in corpus:
        for i, piece in enumerate(chunk_text(p["text"], size, overlap)):
            chunks.append(
                {"chunk_id": f"{p['passage_id']}#{i}", "passage_id": p["passage_id"], "text": piece}
            )
    return chunks
