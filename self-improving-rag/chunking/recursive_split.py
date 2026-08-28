"""Chunking Strategy 2: Recursive character text splitting"""


def recursive_split_chunk(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Split text recursively by separators: \\n\\n -> \\n -> . -> space."""
    separators = ["\n\n", "\n", ". ", " "]
    texts = [text]

    for sep in separators:
        next_texts = []
        for t in texts:
            if len(t) > chunk_size:
                next_texts.extend(t.split(sep))
            else:
                next_texts.append(t)
        texts = next_texts

    # Merge small pieces up to chunk_size
    chunks = []
    current = ""
    for t in texts:
        t = t.strip()
        if not t:
            continue
        if len(current) + len(t) <= chunk_size:
            current += t
        else:
            if current:
                chunks.append(current)
            if overlap > 0:
                current = current[-overlap:] + t
            else:
                current = t

    if current:
        chunks.append(current)
    return chunks
