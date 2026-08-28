"""Context assembly: pack retrieved chunks into one string within token budget."""

import tiktoken
from config import TOKENIZER_MODEL


def assemble_context(chunks: list[dict], max_tokens: int = 4000) -> tuple[str, int]:
    """Greedy-fill chunks by score until token budget is reached.

    Args:
        chunks: List of dicts with "text" and "score" keys, sorted by score desc.
        max_tokens: Maximum token budget for the LLM context.

    Returns:
        Tuple of (assembled_context_string, actual_token_count).
        Chunks are joined with "\n\n" separator.
        If a single chunk exceeds the budget, it's included anyway (no partial chunks).
    """
    enc = tiktoken.get_encoding(TOKENIZER_MODEL)

    # Sort by score descending (highest relevance first)
    sorted_chunks = sorted(chunks, key=lambda c: c.get("score", 0), reverse=True)

    selected = []
    total_tokens = 0

    for chunk in sorted_chunks:
        text = chunk.get("text", "")
        if not text.strip():
            continue

        chunk_tokens = len(enc.encode(text, disallowed_special=()))

        # If adding this chunk would exceed budget and we already have some, stop
        if total_tokens + chunk_tokens > max_tokens and selected:
            break

        selected.append(text)
        total_tokens += chunk_tokens

    # Join with double newline (same delimiter compressor uses in AutoRAG)
    context = "\n\n".join(selected)
    return context, total_tokens
