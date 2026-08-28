#!/usr/bin/env python3
"""Ingest documents into Chroma for a given config version.

Usage:
  python ingest.py                                    # default config, v1
  python ingest.py --strategy recursive --size 512     # custom config
  python ingest.py --pages 100                         # ingest more pages
"""

import argparse
import hashlib
import re
from embeddings import EmbeddingClient
from vector_store import ChromaStore
from chunking import fixed_size_chunk, recursive_split_chunk, semantic_chunk
from config import BOOK_PATHS, DEFAULT_CONFIG


def load_book(path: str, max_pages: int = 50, start_page: int = 1) -> str:
    """Load book text, from start_page to start_page + max_pages.

    Page markers look like: '-- 1 of 598 --'
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Split on page markers: '-- N of M --'
    pages = re.split(r"-- \d+ of \d+ --", text)
    # First element is usually empty (before first marker)
    pages = [p.strip() for p in pages if p.strip()]
    # start_page is 1-indexed
    start_idx = max(0, start_page - 1)
    selected = pages[start_idx:start_idx + max_pages]
    return "\n\n".join(selected)


def load_pdf(path: str, max_pages: int = 50) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        text = page.get_text().strip()
        if text:
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def load_document(path: str, max_pages: int = 50, start_page: int = 1) -> str:
    """Load a document — dispatches on file extension."""
    if path.lower().endswith(".pdf"):
        return load_pdf(path, max_pages)
    return load_book(path, max_pages, start_page=start_page)


def get_chunker(strategy: str, chunk_size: int, chunk_overlap: int):
    """Return a chunking function based on strategy name."""
    if strategy == "fixed_size":
        return lambda text: fixed_size_chunk(text, chunk_size, chunk_overlap)
    elif strategy == "recursive_split":
        return lambda text: recursive_split_chunk(text, chunk_size, chunk_overlap)
    elif strategy == "semantic":
        client = EmbeddingClient()
        return lambda text: semantic_chunk(text, client.embed, threshold=0.5)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def make_id(strategy: str, source: str, index: int, text: str) -> str:
    """Deterministic chunk ID for idempotent re-ingestion."""
    md5 = hashlib.md5(text[:100].encode()).hexdigest()[:8]
    return f"{strategy}_{source}_{index:04d}_{md5}"


def ingest(
    strategy: str = "fixed_size",
    chunk_size: int = 256,
    chunk_overlap: int = 0,
    version: str = "v1",
    pages: int = 50,
    start_page: int = 1,
    book_path: str | None = None,
) -> str:
    """Ingest book into Chroma collection. Skip if collection exists.

    Returns the collection name.
    """
    book_path = book_path or BOOK_PATHS[0]
    collection_name = f"rag_{version}_{strategy}_{chunk_size}"

    # Check if collection already exists
    store = ChromaStore(collection_name)
    if store.count() > 0:
        print(f"  Collection '{collection_name}' already exists with {store.count()} chunks. Skipping.")
        return collection_name

    # Load
    print(f"  Loading book: {book_path}")
    text = load_document(book_path, max_pages=pages, start_page=start_page)
    print(f"  Loaded {len(text):,} characters ({pages} pages)")

    # Chunk
    print(f"  Chunking: strategy='{strategy}', size={chunk_size}, overlap={chunk_overlap}")
    chunker = get_chunker(strategy, chunk_size, chunk_overlap)
    chunks = chunker(text)
    print(f"  Created {len(chunks)} chunks")

    # Embed (batch — one API call for all chunks)
    print("  Embedding chunks (batch)...")
    client = EmbeddingClient()
    vecs = client.embed(chunks)

    # Upsert
    print(f"  Upserting to '{collection_name}'...")
    ids = [make_id(strategy, "book", i, c) for i, c in enumerate(chunks)]
    metadatas = [{"source": book_path, "chunk_index": i} for i in range(len(chunks))]
    store.upsert(ids=ids, embeddings=vecs, metadatas=metadatas, documents=chunks)

    print(f"  Done: {len(chunks)} chunks in '{collection_name}'")
    return collection_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Chroma")
    parser.add_argument("--strategy", default=DEFAULT_CONFIG["chunk_strategy"])
    parser.add_argument("--size", type=int, default=DEFAULT_CONFIG["chunk_size"])
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--pages", type=int, default=50)
    parser.add_argument("--book", default=None)
    args = parser.parse_args()

    print("=" * 50)
    print("  Ingestion")
    print("=" * 50)
    ingest(
        strategy=args.strategy,
        chunk_size=args.size,
        chunk_overlap=args.overlap,
        version=args.version,
        pages=args.pages,
        book_path=args.book,
    )
    print("=" * 50)
