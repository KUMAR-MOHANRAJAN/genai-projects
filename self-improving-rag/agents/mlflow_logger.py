"""MLflow Logger — thin wrapper for experiment tracking.

Integrates MLflow Tracking into the Self-Improving RAG pipeline.
Logs parameters, metrics, tags, and artifacts for every run.

Design:
  - Best-effort: if MLflow server isn't running, all calls are no-ops
  - Never replaces JSONL or execution_trace — adds a parallel logging path
  - Two entry points: log_query_run() for single queries, log_optimization_run()
    for the full optimization loop with nested child runs per iteration

Usage:
    from agents.mlflow_logger import log_query_run, log_optimization_run

    # Single query
    log_query_run(query="What is tokenization?", result=pipeline_result, config=config)

    # Optimization loop
    log_optimization_run(query="What is tokenization?", report=optimizer_report)
"""

import logging
import os
import json
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# MLflow is optional — graceful degradation if not installed or server is down.
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

_EXPERIMENT_NAME = "self-improving-rag"
_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def _safe_log(fn, *args, **kwargs):
    """Execute an MLflow logging call with graceful fallback."""
    if not _MLFLOW_AVAILABLE:
        return
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        logger.debug(f"MLflow log skipped: {exc}")


def _set_experiment():
    """Ensure the experiment exists and is active."""
    if not _MLFLOW_AVAILABLE:
        return
    try:
        mlflow.set_tracking_uri(_TRACKING_URI)
        mlflow.set_experiment(_EXPERIMENT_NAME)
    except Exception as exc:
        logger.debug(f"MLflow experiment setup skipped: {exc}")


def log_query_run(query: str, result: dict, config: dict, version: str = "v1") -> str | None:
    """Log a single pipeline query run to MLflow.

    Args:
        query: The question asked.
        result: Pipeline result dict (from run_pipeline or graph invoke).
        config: Pipeline config used.
        version: Collection version string.

    Returns:
        MLflow run_id if successful, None otherwise.
    """
    _set_experiment()

    try:
        with mlflow.start_run(run_name=f"query: {query[:60]}") as run:
            # ── Parameters (things you configure) ──────────────────────
            _safe_log(mlflow.log_param, "chunk_strategy", config.get("chunk_strategy"))
            _safe_log(mlflow.log_param, "chunk_size", config.get("chunk_size"))
            _safe_log(mlflow.log_param, "chunk_overlap", config.get("chunk_overlap"))
            _safe_log(mlflow.log_param, "retrieval_k", config.get("retrieval_k"))
            _safe_log(mlflow.log_param, "max_context_tokens", config.get("max_context_tokens"))
            _safe_log(mlflow.log_param, "prompt_template", config.get("prompt_template"))
            _safe_log(mlflow.log_param, "version", version)

            # ── Metrics (things you measure) ───────────────────────────
            _safe_log(mlflow.log_metric, "unified_score", result.get("unified_score") or 0.0)
            _safe_log(mlflow.log_metric, "faithfulness", result.get("faithfulness") or 0.0)
            _safe_log(mlflow.log_metric, "relevance", result.get("relevance") or 0.0)
            _safe_log(mlflow.log_metric, "correctness", result.get("correctness") or 0.0)
            _safe_log(mlflow.log_metric, "retrieval_score", result.get("retrieval_score") or 0.0)
            _safe_log(mlflow.log_metric, "latency_ms", result.get("latency_ms") or result.get("generation_latency_ms") or 0)
            _safe_log(mlflow.log_metric, "cost_usd", result.get("cost_usd") or result.get("generation_cost_usd") or 0.0)
            _safe_log(mlflow.log_metric, "chunk_count", result.get("chunk_count") or 0)
            _safe_log(mlflow.log_metric, "context_tokens", result.get("context_tokens") or 0)

            # ── Tags (metadata for filtering) ──────────────────────────
            _safe_log(mlflow.set_tag, "run_type", "query")
            _safe_log(mlflow.set_tag, "gate_decision", result.get("gate_decision", "unknown"))
            _safe_log(mlflow.set_tag, "query", query[:200])

            # ── Artifact (full report as JSON) ─────────────────────────
            _log_artifact_json("query_result.json", result)

            logger.info(f"MLflow query run logged: {run.info.run_id}")
            return run.info.run_id

    except Exception as exc:
        logger.debug(f"MLflow query run logging failed: {exc}")
        return None


def log_optimization_run(query: str, report: dict) -> str | None:
    """Log a full optimization loop to MLflow with nested child runs.

    Creates a parent run for the overall optimization, then a child run
    for each iteration. This mirrors MLflow's nested run pattern and
    makes the UI show iterations grouped under their parent.

    Args:
        query: The question optimized for.
        report: The optimization report dict (from run_optimization).

    Returns:
        MLflow parent run_id if successful, None otherwise.
    """
    _set_experiment()

    try:
        # ── Parent run: the overall optimization ───────────────────────
        parent_name = f"optimize: {query[:60]}"
        with mlflow.start_run(run_name=parent_name) as parent_run:
            # ── Parent parameters ──────────────────────────────────────
            init_cfg = report.get("initial_config", {})
            final_cfg = report.get("final_config", {})

            _safe_log(mlflow.log_param, "initial_chunk_size", init_cfg.get("chunk_size"))
            _safe_log(mlflow.log_param, "initial_retrieval_k", init_cfg.get("retrieval_k"))
            _safe_log(mlflow.log_param, "final_chunk_size", final_cfg.get("chunk_size"))
            _safe_log(mlflow.log_param, "final_retrieval_k", final_cfg.get("retrieval_k"))
            _safe_log(mlflow.log_param, "stop_reason", report.get("stop_reason"))

            # ── Parent metrics ─────────────────────────────────────────
            _safe_log(mlflow.log_metric, "initial_score", report.get("initial_score") or 0.0)
            _safe_log(mlflow.log_metric, "final_score", report.get("final_score") or 0.0)
            _safe_log(mlflow.log_metric, "improvement", report.get("improvement") or 0.0)
            _safe_log(mlflow.log_metric, "total_iterations", report.get("total_iterations") or 0)

            # ── Parent tags ────────────────────────────────────────────
            _safe_log(mlflow.set_tag, "run_type", "optimization")
            _safe_log(mlflow.set_tag, "stop_reason", report.get("stop_reason", "unknown"))
            _safe_log(mlflow.set_tag, "query", query[:200])

            # ── Child runs: one per iteration ──────────────────────────
            iterations = report.get("iterations", [])
            for i, rec in enumerate(iterations):
                iter_name = f"iter-{i}: {rec.get('gate_decision', '?')}"
                with mlflow.start_run(run_name=iter_name, nested=True):
                    # Iteration params
                    iter_cfg = rec.get("config", {})
                    _safe_log(mlflow.log_param, "chunk_size", iter_cfg.get("chunk_size"))
                    _safe_log(mlflow.log_param, "retrieval_k", iter_cfg.get("retrieval_k"))
                    _safe_log(mlflow.log_param, "chunk_overlap", iter_cfg.get("chunk_overlap"))
                    _safe_log(mlflow.log_param, "prompt_template", iter_cfg.get("prompt_template"))

                    # Iteration metrics
                    _safe_log(mlflow.log_metric, "unified_score", rec.get("unified_score") or 0.0)
                    _safe_log(mlflow.log_metric, "faithfulness", rec.get("faithfulness") or 0.0)
                    _safe_log(mlflow.log_metric, "relevance", rec.get("relevance") or 0.0)
                    _safe_log(mlflow.log_metric, "retrieval_score", rec.get("retrieval_score") or 0.0)
                    _safe_log(mlflow.log_metric, "latency_ms", rec.get("latency_ms") or 0)
                    _safe_log(mlflow.log_metric, "cost_usd", rec.get("cost_usd") or 0.0)

                    # Iteration tags
                    _safe_log(mlflow.set_tag, "gate_decision", rec.get("gate_decision", "unknown"))
                    _safe_log(mlflow.set_tag, "failure_type", rec.get("failure_type", "none"))
                    _safe_log(mlflow.set_tag, "iteration", i)

                    # Applied variant info
                    variant = rec.get("applied_variant")
                    if variant:
                        _safe_log(mlflow.set_tag, "variant_id", variant.get("variant_id", ""))
                        _safe_log(mlflow.set_tag, "delta", str(variant.get("delta", {})))
                        _safe_log(mlflow.set_tag, "rationale", variant.get("rationale", "")[:200])

                    # Iteration artifact
                    _log_artifact_json(f"iteration_{i}_result.json", rec)

            # ── Parent artifact (full report) ──────────────────────────
            _log_artifact_json("optimization_report.json", report)

            logger.info(f"MLflow optimization run logged: {parent_run.info.run_id}")
            return parent_run.info.run_id

    except Exception as exc:
        logger.debug(f"MLflow optimization run logging failed: {exc}")
        return None


def _log_artifact_json(filename: str, data: dict):
    """Write a dict to a temp JSON file and log it as an MLflow artifact."""
    if not _MLFLOW_AVAILABLE:
        return
    try:
        import tempfile
        import os as _os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir="/tmp") as f:
            json.dump(data, f, indent=2, default=str)
            tmp_path = f.name
        mlflow.log_artifact(tmp_path, artifact_path="reports")
        _os.unlink(tmp_path)
    except Exception as exc:
        logger.debug(f"MLflow artifact logging skipped: {exc}")


def is_available() -> bool:
    """Check if MLflow is available and the server is reachable."""
    if not _MLFLOW_AVAILABLE:
        return False
    try:
        mlflow.set_tracking_uri(_TRACKING_URI)
        mlflow.set_experiment(_EXPERIMENT_NAME)
        return True
    except Exception:
        return False
