"""Agent: Builder node — ingest-if-needed + retrieval.

Combines ingestion and retrieval into a single node since this project
does not require PII masking, a separate storage layer, or a
multi-document pipeline.

Responsibilities:
  1. Read variant_config from state (chunk strategy, size, overlap, version)
  2. Build collection name (SINGLE source of truth — build_collection_name())
  3. Check if collection exists; if empty, ingest the corpus
  4. Run retrieval: embed query → Chroma search → ranked chunks
  5. Write retrieved_chunks, collection_name, chunk_count to state

This node OWNS ingestion + retrieval. No other node or utility should
duplicate the "does this collection exist?" check or build collection
names independently.
"""

import copy
import logging
import time

from state import RunState
from config import DEFAULT_CONFIG, INGEST_PAGES, INGEST_START_PAGE, validate_config
from ingest import ingest
from vector_store import ChromaStore
from retrieval.search import search
from utils import build_collection_name
from agents.trace import traced_node

# MLflow tracing — best-effort, graceful fallback
try:
    import mlflow
    _mlflow_trace = mlflow.trace
except ImportError:
    _mlflow_trace = lambda **kwargs: lambda fn: fn  # no-op decorator

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Builder Node — LangGraph node function
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@_mlflow_trace(name="builder", span_type="retrieval")
@traced_node("builder")
def builder_node(state: RunState) -> dict:
    """LangGraph node: ingest if needed, then retrieve.

    Reads from state:
      - query, config, version

    Writes to state:
      - collection_name, chunk_count
      - retrieved_chunks (list of {chunk_id, text, score, metadata})

    Pipeline flow:
      - Reads variant_config from state
      - Builds collection name (version isolation)
      - Checks if collection exists, ingests if empty
      - Runs retrieval and writes results to state
      - Logs structured events for observability

    Uses single-stage dense retrieval (ChromaDB cosine), which is
    sufficient for relative comparison (v1 vs v2 vs v3) in the optimizer.
    """
    query = state["query"]
    raw_cfg = state.get("config", copy.deepcopy(DEFAULT_CONFIG))
    cfg = validate_config(raw_cfg)
    version = state.get("version", "v1")

    # Build collection name (single source of truth)
    collection_name = build_collection_name(cfg, version)

    strategy = cfg.get("chunk_strategy", "fixed_size")
    chunk_size = cfg.get("chunk_size", 256)
    chunk_overlap = cfg.get("chunk_overlap", 0)
    k = cfg.get("retrieval_k", 5)

    logger.info(
        "builder_node_started",
        extra={
            "collection_name": collection_name,
            "version": version,
            "strategy": strategy,
            "chunk_size": chunk_size,
            "retrieval_k": k,
        },
    )

    # ── Step 1: Auto-ingest if collection doesn't exist ───────────────────
    # When the optimizer changes chunk_size or strategy, the collection name
    # changes (e.g., rag_v2_fixed_size_320). If that collection doesn't
    # exist yet, we re-ingest the corpus with the new chunking params.
    # This matches the pattern where each optimizer iteration
    # re-runs ingestion (full pipeline from scratch with new config).
    node_start = time.perf_counter()

    store = ChromaStore(collection_name)
    if store.count() == 0:
        logger.info(
            "builder_ingesting",
            extra={"collection_name": collection_name, "reason": "empty_collection"},
        )
        ingest(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            version=version,
            pages=INGEST_PAGES,
            start_page=INGEST_START_PAGE,
        )
        logger.info(
            "builder_ingestion_complete",
            extra={
                "collection_name": collection_name,
                "chunk_count": store.count(),
            },
        )
    else:
        logger.info(
            "builder_collection_exists",
            extra={
                "collection_name": collection_name,
                "chunk_count": store.count(),
            },
        )

    # ── Step 2: Retrieval ─────────────────────────────────────────────────
    # Single-stage dense retrieval via ChromaDB cosine similarity.
    # Sufficient for the optimization loop mechanics and for relative
    # comparison (v1 vs v2 vs v3) in the optimizer.
    chunks = search(collection_name, query, k=k)

    retrieval_latency_ms = int((time.perf_counter() - node_start) * 1000)

    logger.info(
        "builder_node_complete",
        extra={
            "collection_name": collection_name,
            "retrieved_count": len(chunks),
            "retrieval_latency_ms": retrieval_latency_ms,
        },
    )

    return {
        "collection_name": collection_name,
        "retrieved_chunks": chunks,
        "chunk_count": store.count(),
    }
