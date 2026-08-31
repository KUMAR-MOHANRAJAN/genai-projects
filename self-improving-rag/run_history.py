"""Run History — append-only JSONL store for pipeline and optimization runs.

Saves each run as a single JSON line in data/runs_history.jsonl.
No database needed — just a flat file. Each record has:
  - timestamp, run_type ("query" or "optimization"), query, config,
    scores, gate_decision, stop_reason (if optimization), etc.

Also logs to MLflow (if available) for experiment tracking UI.

Used by:
  - Frontend Tab 3 (History) to display past runs in a table
  - MLflow UI for side-by-side run comparison and metric charts
  - Future: drift detection (compare recent scores to historical average)
"""

import json
import os
from datetime import datetime, timezone

from agents.mlflow_logger import (
    log_query_run as _mlflow_log_query,
    log_summary_to_run,
    get_active_run_id,
)

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
HISTORY_FILE = os.path.join(_DATA_DIR, "runs_history.jsonl")


def _ensure_dir():
    os.makedirs(_DATA_DIR, exist_ok=True)


def _slim_iterations(iterations: list[dict]) -> list[dict]:
    """Create a compact version of iteration records for JSONL storage.

    Keeps scores, diagnosis, config, and applied variant.
    Drops large fields (retrieved_chunks full text, full answer) to save space.
    """
    slim = []
    for rec in iterations:
        entry = {
            "iteration": rec.get("iteration"),
            "unified_score": rec.get("unified_score"),
            "faithfulness": rec.get("faithfulness"),
            "relevance": rec.get("relevance"),
            "correctness": rec.get("correctness"),
            "retrieval_score": rec.get("retrieval_score"),
            "gate_decision": rec.get("gate_decision"),
            "failure_type": rec.get("failure_type"),
            "remediation_hint": rec.get("remediation_hint"),
            "config": rec.get("config"),
            "cost_usd": rec.get("cost_usd"),
            "latency_ms": rec.get("latency_ms"),
            "chunk_count": rec.get("chunk_count", 0),
            "answer_preview": rec.get("answer", "") or "",
        }
        variant = rec.get("applied_variant")
        if variant:
            entry["applied_variant"] = {
                "variant_id": variant.get("variant_id"),
                "failure_type": variant.get("failure_type"),
                "delta": variant.get("delta"),
                "rationale": variant.get("rationale"),
            }
        entry["execution_trace"] = rec.get("execution_trace", [])
        entry["judge_details"] = rec.get("judge_details", {})
        slim.append(entry)
    return slim


def save_query_run(query: str, config: dict, result: dict, version: str = "v1") -> dict:
    """Save a single pipeline query run to history.

    Args:
        query: The question asked.
        config: Pipeline config used.
        result: The pipeline result dict (from run_pipeline).
        version: Collection version string.

    Returns:
        The saved record dict.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_type": "query",
        "query": query,
        "version": version,
        "config": {
            "chunk_strategy": config.get("chunk_strategy"),
            "chunk_size": config.get("chunk_size"),
            "chunk_overlap": config.get("chunk_overlap"),
            "retrieval_k": config.get("retrieval_k"),
            "prompt_template": config.get("prompt_template"),
        },
        "unified_score": result.get("unified_score"),
        "faithfulness": result.get("faithfulness"),
        "relevance": result.get("relevance"),
        "correctness": result.get("correctness"),
        "retrieval_score": result.get("retrieval_score"),
        "gate_decision": result.get("gate_decision"),
        "cost_usd": result.get("cost_usd", result.get("generation_cost_usd", 0)),
        "latency_ms": result.get("latency_ms", result.get("generation_latency_ms", 0)),
        "chunk_count": result.get("chunk_count", 0),
        "answer_preview": result.get("answer", "") or "",
        "execution_trace": result.get("execution_trace", []),
        "judge_details": result.get("judge_details", {}),
    }
    _append(record)

    # Log summary params/metrics to the existing MLflow run (traces already attached)
    mlflow_run_id = result.get("_mlflow_run_id")
    if mlflow_run_id:
        log_summary_to_run(
            run_id=mlflow_run_id,
            params={
                "chunk_strategy": config.get("chunk_strategy"),
                "chunk_size": config.get("chunk_size"),
                "chunk_overlap": config.get("chunk_overlap"),
                "retrieval_k": config.get("retrieval_k"),
                "max_context_tokens": config.get("max_context_tokens"),
                "prompt_template": config.get("prompt_template"),
                "version": version,
            },
            metrics={
                "unified_score": result.get("unified_score") or 0.0,
                "faithfulness": result.get("faithfulness") or 0.0,
                "relevance": result.get("relevance") or 0.0,
                "correctness": result.get("correctness") or 0.0,
                "retrieval_score": result.get("retrieval_score") or 0.0,
                "latency_ms": result.get("latency_ms") or result.get("generation_latency_ms") or 0,
                "cost_usd": result.get("cost_usd") or result.get("generation_cost_usd") or 0.0,
                "chunk_count": result.get("chunk_count") or 0,
                "context_tokens": result.get("context_tokens") or 0,
            },
            tags={
                "gate_decision": result.get("gate_decision", "unknown"),
            },
            artifact_name="query_result.json",
            artifact_data=result,
        )
    else:
        # Fallback: no active run context (e.g. CLI usage), create standalone run
        _mlflow_log_query(query, result, config, version)

    return record


def save_optimization_run(query: str, report: dict) -> dict:
    """Save an optimization run to history.

    Args:
        query: The question optimized for.
        report: The optimization report dict (from run_optimization).

    Returns:
        The saved record dict.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_type": "optimization",
        "query": query,
        "initial_config": report.get("initial_config"),
        "final_config": report.get("final_config"),
        "initial_score": report.get("initial_score"),
        "final_score": report.get("final_score"),
        "improvement": report.get("improvement"),
        "stop_reason": report.get("stop_reason"),
        "total_iterations": report.get("total_iterations"),
        "iterations": _slim_iterations(report.get("iterations", [])),
    }
    _append(record)
    # MLflow logging handled by optimizer.py (wraps graph.invoke() in active run)
    return record


def load_history(limit: int = 100) -> list[dict]:
    """Load recent run history records.

    Args:
        limit: Maximum number of records to return (most recent first).

    Returns:
        List of record dicts, newest first.
    """
    if not os.path.exists(HISTORY_FILE):
        return []

    records = []
    with open(HISTORY_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # skip corrupt lines

    # Return most recent first, limited
    return list(reversed(records[-limit:]))


def clear_history() -> int:
    """Delete all history records. Returns count of deleted records."""
    count = len(load_history(limit=999999))
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return count


def _append(record: dict):
    """Append a single record as a JSON line."""
    _ensure_dir()
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
