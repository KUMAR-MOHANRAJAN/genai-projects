"""Bridge between Streamlit frontend and the RAG pipeline backend.

Adds the project root to sys.path so we can import pipeline modules directly.
No API server needed — Streamlit runs in the same Python process.
"""

import os
import sys

# Add project root to path so we can import pipeline modules.
# IMPORTANT: insert at index 0 so project-root modules are found before
# this directory's own modules (avoids circular import between
# frontend/utils.py and project-root utils.py).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import project-root utils FIRST (before pipeline, which also imports it)
# to avoid the partially-initialized module error.
import importlib
_root_utils = importlib.import_module("utils")
build_collection_name = _root_utils.build_collection_name

from pipeline import run_pipeline  # noqa: E402
from ingest import ingest  # noqa: E402
from config import DEFAULT_CONFIG, INGEST_PAGES, INGEST_START_PAGE  # noqa: E402
from ground_truth import TEST_QUERIES  # noqa: E402
from agents.optimizer import run_optimization  # noqa: E402
from agents.mlflow_logger import start_query_context  # noqa: E402
from run_history import save_query_run, save_optimization_run, load_history, clear_history  # noqa: E402
from vector_store import ChromaStore  # noqa: E402

CORPUS_DIR = os.path.join(PROJECT_ROOT, "corpus")


def list_corpus_files() -> list[str]:
    """List all .txt and .pdf files directly in the corpus directory (flat,
    legacy single-book layout)."""
    if not os.path.isdir(CORPUS_DIR):
        return []
    return sorted(
        f for f in os.listdir(CORPUS_DIR)
        if f.lower().endswith((".txt", ".pdf"))
    )


def list_corpus_files_recursive() -> list[str]:
    """Discover every .txt/.pdf file under corpus/, including domain
    subfolders (corpus/hr/, corpus/technical/, etc.). Returns paths relative
    to CORPUS_DIR (e.g. "hr/leave_policy.txt") for display and lookup.
    """
    if not os.path.isdir(CORPUS_DIR):
        return []
    found = []
    for root, _dirs, files in os.walk(CORPUS_DIR):
        for f in files:
            if f.lower().endswith((".txt", ".pdf")):
                rel = os.path.relpath(os.path.join(root, f), CORPUS_DIR)
                found.append(rel.replace(os.sep, "/"))
    return sorted(found)


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

    Uses build_collection_name() from agents/builder.py — single source
    of truth for collection naming.

    Wraps graph.invoke() in an MLflow run context so @mlflow.trace()
    spans attach correctly.
    """
    strategy = config.get("chunk_strategy", "fixed_size")
    chunk_size = config.get("chunk_size", 256)
    chunk_overlap = config.get("chunk_overlap", 0)

    # Use build_collection_name for consistency (single source of truth)
    collection_name = config.get("collection_name") or build_collection_name(config, version)
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

    # Wrap pipeline in MLflow run context so @mlflow.trace() spans attach
    with start_query_context(query) as mlflow_run_id:
        result = run_pipeline(query, config, version=version)
        # Store run_id in result so save_query_run() can log summary to it
        if mlflow_run_id:
            result["_mlflow_run_id"] = mlflow_run_id
        return result


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
    baseline_result: dict | None = None,
    force_continue: bool = False,
) -> dict:
    """Wrapper for the optimizer — called from Streamlit UI.

    Returns the full optimization report dict with iteration history.
    If baseline_result is provided, the optimizer skips its first graph
    invocation and uses the provided result as iteration 0.
    If force_continue is True, faithfulness veto and HITL stops are skipped.
    """
    return run_optimization(
        query=query,
        config=config,
        version=version,
        target_score=target_score,
        max_iterations=max_iterations,
        baseline_result=baseline_result,
        force_continue=force_continue,
    )


def get_collections() -> list[dict]:
    """List all ChromaDB collections with chunk counts.

    Returns list of {"name": str, "count": int} sorted by name.
    Only returns non-empty collections.
    """
    return [c for c in ChromaStore.list_collections() if c["count"] > 0]


def parse_collection_name(name: str) -> dict:
    """Parse collection name into config components.

    'rag_g1_fixed_size_256_o0' → {version: 'g1', strategy: 'fixed_size', chunk_size: 256, chunk_overlap: 0}
    """
    parts = name.split("_")
    # Format: rag_{version}_{strategy_word1..N}_{chunk_size}_o{overlap}. Older
    # collections (pre-overlap-in-name) won't have the trailing "oN" part —
    # default overlap to 0 for those. Falls back to defaults on any mismatch
    # (e.g. single-word "semantic" strategy has fewer underscore-separated parts).
    try:
        if parts[0] != "rag":
            raise ValueError("not a rag collection name")
        version = parts[1]
        overlap = 0
        if parts[-1][:1] == "o" and parts[-1][1:].isdigit():
            overlap = int(parts[-1][1:])
            parts = parts[:-1]
        chunk_size = int(parts[-1])
        strategy = "_".join(parts[2:-1])
        return {
            "version": version,
            "chunk_strategy": strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": overlap,
        }
    except (IndexError, ValueError):
        return {"version": "g1", "chunk_strategy": "fixed_size", "chunk_size": 256, "chunk_overlap": 0}


def collection_label(col: dict) -> str:
    """Build a display label for a collection dropdown.

    'rag_g1_fixed_size_256' (19 chunks) → 'rag_g1_fixed_size_256 (19 chunks)'
    """
    return f"{col['name']} ({col['count']} chunks)"
