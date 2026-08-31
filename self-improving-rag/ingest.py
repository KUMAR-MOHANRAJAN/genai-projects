#!/usr/bin/env python3
"""Ingest documents into Chroma for a given config version.

Usage:
  python ingest.py                                    # default config, v1
  python ingest.py --strategy recursive --size 512     # custom config
  python ingest.py --pages 100                         # ingest more pages
"""

import argparse
import hashlib
import os
import re
from embeddings import EmbeddingClient
from vector_store import ChromaStore
from chunking import fixed_size_chunk, recursive_split_chunk, semantic_chunk
from config import CORPUS_FILES, DEFAULT_CONFIG
from utils import build_collection_name


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


def load_whole_document(path: str) -> str:
    """Load a short document's full text, no page windowing.

    Used for the multi-file business-docs corpus (CORPUS_FILES) — these
    files have no '-- N of M --' page markers, so the page-window logic in
    load_document()/load_book() doesn't apply (and would silently return an
    empty slice given the default INGEST_START_PAGE).
    """
    if path.lower().endswith(".pdf"):
        return load_pdf(path, max_pages=100000)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


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
    book_paths: list[str] | None = None,
) -> str:
    """Ingest one or more documents into a Chroma collection. Skip if exists.

    Two modes:
      - Multi-file (default when book_path is not given): ingests every file
        in `book_paths`, or CORPUS_FILES if book_paths is also None. Each
        file is loaded and chunked independently (chunks never blend across
        document boundaries) and tagged with its real source filename.
      - Legacy single-book: pass book_path for the old page-windowed flow
        (used by the original two large textbook corpus files).

    Returns the collection name.
    """
    collection_name = build_collection_name(
        {"chunk_strategy": strategy, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        version,
    )

    # Check if collection already exists
    store = ChromaStore(collection_name)
    if store.count() > 0:
        print(f"  Collection '{collection_name}' already exists with {store.count()} chunks. Skipping.")
        return collection_name

    chunker = get_chunker(strategy, chunk_size, chunk_overlap)
    print(f"  Chunking: strategy='{strategy}', size={chunk_size}, overlap={chunk_overlap}")

    chunks: list[str] = []
    sources: list[str] = []

    if book_path:
        # Legacy single-book, page-windowed flow.
        print(f"  Loading book: {book_path}")
        text = load_document(book_path, max_pages=pages, start_page=start_page)
        print(f"  Loaded {len(text):,} characters ({pages} pages)")
        doc_chunks = chunker(text)
        chunks.extend(doc_chunks)
        sources.extend([book_path] * len(doc_chunks))
    else:
        paths = book_paths or CORPUS_FILES
        for path in paths:
            print(f"  Loading document: {path}")
            text = load_whole_document(path)
            doc_chunks = chunker(text)
            chunks.extend(doc_chunks)
            sources.extend([path] * len(doc_chunks))
        print(f"  Loaded {len(paths)} documents")

    print(f"  Created {len(chunks)} chunks")

    # Embed (batch — one API call for all chunks)
    print("  Embedding chunks (batch)...")
    client = EmbeddingClient()
    vecs = client.embed(chunks)

    # Upsert
    print(f"  Upserting to '{collection_name}'...")
    ids = [make_id(strategy, os.path.basename(sources[i]), i, c) for i, c in enumerate(chunks)]
    metadatas = [{"source": sources[i], "chunk_index": i} for i in range(len(chunks))]
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
