#!/usr/bin/env python3
"""Self-Improving RAG — Query the pipeline and see scored results.

Usage:
  python main.py                                    # default query
  python main.py "What is tokenization?"            # custom query
"""

import sys
from config import DEFAULT_CONFIG
from pipeline import run_pipeline
from utils import build_collection_name

DEFAULT_QUERY = "How do embeddings represent meaning?"


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    config = DEFAULT_CONFIG

    strategy = config.get("chunk_strategy", "fixed_size")
    chunk_size = config.get("chunk_size", 256)
    k = config.get("retrieval_k", 5)
    version = "v1"
    collection_name = build_collection_name(config, version)

    print("=" * 50)
    print("  Self-Improving RAG")
    print("=" * 50)
    print(f"  Query:   {query}")
    print(f"  Config:  strategy={strategy}, chunk_size={chunk_size}, k={k}")
    print(f"  Version: {version}")
    print()

    # Auto-ingest if collection doesn't exist
    try:
        from vector_store import ChromaStore
        store = ChromaStore(collection_name)
        if store.count() == 0:
            print(f"  Collection '{collection_name}' not found — ingesting first...")
            print()
            from ingest import ingest
            ingest(strategy=strategy, chunk_size=chunk_size, version=version)
            print()
    except Exception:
        print(f"  Collection '{collection_name}' not found — ingesting first...")
        print()
        from ingest import ingest
        ingest(strategy=strategy, chunk_size=chunk_size, version=version)
        print()

    # Run pipeline
    print("  Running pipeline...")
    print()
    state = run_pipeline(query, config, version=version)

    # Helper for None-safe formatting
    def fmt(val, suffix=""):
        return f"{val:.2f}{suffix}" if val is not None else "N/A (judge failed)"

    # Print results
    print("-" * 50)
    print("  RESULTS")
    print("-" * 50)
    print(f"  Answer: {state['answer'][:300]}...")
    print()
    print(f"  Faithfulness:    {fmt(state['faithfulness'])}")
    print(f"  Relevance:       {fmt(state['relevance'])}")
    if state.get("correctness") is not None:
        print(f"  Correctness:     {fmt(state['correctness'])}")
    else:
        print(f"  Correctness:     N/A (no ground truth)")
    print(f"  Retrieval:       {fmt(state['retrieval_score'])}")
    print(f"  Unified Score:   {fmt(state['unified_score'])}")
    print()
    print(f"  Cost:            ${state['cost_usd']:.4f}")
    print(f"  Latency:         {state['latency_ms']}ms")
    print(f"  Chunks:          {state['chunk_count']}")
    print(f"  Context tokens:  {state['context_tokens']}")
    print()
    print(f"  Gate Decision:   {state.get('gate_decision', 'N/A')}")
    print(f"  Gate Reason:     {state.get('gate_reason', 'N/A')}")
    print("-" * 50)


if __name__ == "__main__":
    main()
