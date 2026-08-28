"""Chunking Strategy 3: Semantic (embedding-based) chunking"""

import re
import numpy as np
from numpy.linalg import norm


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on sentence-ending punctuation."""
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw if s.strip()]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    dot = float(np.dot(a_arr, b_arr))
    norm_a = float(norm(a_arr))
    norm_b = float(norm(b_arr))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _find_boundaries(
    vecs: list[list[float]],
    threshold: float,
    window_size: int,
) -> list[int]:
    """Find sentence indices where similarity drops below threshold."""
    boundaries = []
    step = max(1, window_size // 2)
    for i in range(1, len(vecs)):
        s = max(0, i - step)
        e = min(len(vecs), i + step)
        left_mean = np.mean(vecs[s:i], axis=0) if i > s else np.array(vecs[i])
        right_mean = np.mean(vecs[i:e], axis=0) if e > i else np.array(vecs[i])
        sim = _cosine_sim(left_mean.tolist(), right_mean.tolist())
        if sim < threshold:
            boundaries.append(i)
    return boundaries


def semantic_chunk(
    text: str,
    embed_fn: callable,
    threshold: float = 0.5,
    window_size: int = 2,
) -> list[str]:
    """Split text at points where embedding similarity drops below threshold.

    Args:
        text: Input text to chunk.
        embed_fn: Function that takes list[str] and returns list[list[float]].
        threshold: Cosine similarity threshold for cutting.
        window_size: Number of sentences in sliding window.

    Returns:
        List of chunk text strings.
    """
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text]

    vecs = embed_fn(sentences)
    boundaries = _find_boundaries(vecs, threshold, window_size)

    if not boundaries:
        return [text]

    boundaries = sorted(set(boundaries))
    chunks = []
    start = 0
    for b in boundaries:
        chunks.append(" ".join(sentences[start:b]))
        start = b
    chunks.append(" ".join(sentences[start:]))
    return [c for c in chunks if c.strip()]
