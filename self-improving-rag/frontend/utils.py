"""Bridge between Streamlit frontend and the RAG pipeline backend.

Adds the project root to sys.path so we can import pipeline modules directly.
No API server needed — Streamlit runs in the same Python process.
"""

import os
import sys

# Add project root to path so we can import pipeline modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline import run_pipeline  # noqa: E402
from ingest import ingest  # noqa: E402
from config import DEFAULT_CONFIG, INGEST_PAGES, INGEST_START_PAGE  # noqa: E402
from ground_truth import TEST_QUERIES  # noqa: E402
from agents.optimizer import run_optimization  # noqa: E402
from run_history import save_query_run, save_optimization_run, load_history, clear_history  # noqa: E402

CORPUS_DIR = os.path.join(PROJECT_ROOT, "corpus")


def list_corpus_files() -> list[str]:
    """List all .txt and .pdf files in the corpus directory."""
    if not os.path.isdir(CORPUS_DIR):
        return []
    return sorted(
        f for f in os.listdir(CORPUS_DIR)
        if f.lower().endswith((".txt", ".pdf"))
    )


def save_uploaded_file(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to the corpus directory.

    Returns the full path to the saved file.
    """
    os.makedirs(CORPUS_DIR, exist_ok=True)
    dest = os.path.join(CORPUS_DIR, uploaded_file.name)
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def run_query(query: str, config: dict, version: str = "v1",
              book_path: str | None = None, pages: int = 50) -> dict:
    """Run the full pipeline: auto-ingest if needed, then query.

    Returns the RunState dict with answer, scores, cost, latency, etc.
    """
    strategy = config.get("chunk_strategy", "fixed_size")
    chunk_size = config.get("chunk_size", 256)
    chunk_overlap = config.get("chunk_overlap", 0)

    # Auto-ingest if collection is empty
    from vector_store import ChromaStore
    collection_name = f"rag_{version}_{strategy}_{chunk_size}"
    store = ChromaStore(collection_name)
    if store.count() == 0:
        ingest(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            version=version,
            pages=pages,
            book_path=book_path,
        )

    return run_pipeline(query, config, version=version)


def get_ground_truth_queries() -> list[str]:
    """Return just the question strings from the test queries."""
    return [q[0] for q in TEST_QUERIES]


# ── Bad config presets for demo ───────────────────────────────────────────────

BAD_CONFIG = {
    "chunk_strategy": "fixed_size",
    "chunk_size": 64,
    "chunk_overlap": 0,
    "retrieval_k": 1,
    "max_context_tokens": 4000,
    "prompt_template": "v1",
}


def run_optimization_ui(
    query: str,
    config: dict,
    version: str = "g1",
    target_score: float = 0.85,
    max_iterations: int = 5,
) -> dict:
    """Wrapper for the optimizer — called from Streamlit UI.

    Returns the full optimization report dict with iteration history.
    """
    return run_optimization(
        query=query,
        config=config,
        version=version,
        target_score=target_score,
        max_iterations=max_iterations,
    )
