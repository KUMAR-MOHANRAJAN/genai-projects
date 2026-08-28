"""Chunking Strategy 1: Fixed-size token chunking"""

import tiktoken
from config import TOKENIZER_MODEL


def fixed_size_chunk(text: str, chunk_size: int = 256, overlap: int = 0) -> list[str]:
    """Split text into fixed-size token windows with optional overlap."""
    enc = tiktoken.get_encoding(TOKENIZER_MODEL)
    tokens = enc.encode(text, disallowed_special=())
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += chunk_size - overlap
    return chunks
