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
import urllib.request
from contextlib import contextmanager
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

# Cache the server availability check so we don't probe on every call.
_server_available: bool | None = None

# Track the current active run_id so summary logging can attach to it.
_active_run_id: str | None = None


def _check_server() -> bool:
    """Quick HTTP probe to see if the MLflow tracking server is reachable."""
    global _server_available
    if _server_available is not None:
        return _server_available
    try:
        req = urllib.request.Request(_TRACKING_URI, method="HEAD")
        urllib.request.urlopen(req, timeout=2)
        _server_available = True
        return True
    except Exception:
        _server_available = False
        return False


def _safe_log(fn, *args, **kwargs):
    """Execute an MLflow logging call with graceful fallback."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        logger.debug(f"MLflow log skipped: {exc}")


def _set_experiment():
    """Ensure the experiment exists and is active."""
    try:
        mlflow.set_tracking_uri(_TRACKING_URI)
        mlflow.set_experiment(_EXPERIMENT_NAME)
    except Exception as exc:
        logger.debug(f"MLflow experiment setup skipped: {exc}")


@contextmanager
def start_optimization_context(query: str):
    """Context manager that wraps an optimization loop in a single MLflow run.

    The @mlflow.trace() spans created during graph.invoke() attach to this run.
    Yields the run_id so summary metrics can be logged to the same run later.

    Usage:
        with start_optimization_context(query) as run_id:
            # ... graph invocations happen here, traces attach automatically
            pass
        # After context exits, log summary metrics using run_id
    """
    global _active_run_id
    if not _MLFLOW_AVAILABLE or not _check_server():
        yield None
        return

    _set_experiment()

    run_id = None
    try:
        with mlflow.start_run(run_name=f"optimize: {query[:60]}") as run:
            run_id = run.info.run_id
            _active_run_id = run_id
            _safe_log(mlflow.set_tag, "run_type", "optimization")
            _safe_log(mlflow.set_tag, "query", query[:200])
            yield run_id
    except Exception as exc:
        logger.debug(f"MLflow optimization context failed: {exc}")
        yield None
    finally:
        _active_run_id = None


@contextmanager
def start_query_context(query: str):
    """Context manager that wraps a single query pipeline in an MLflow run.

    The @mlflow.trace() spans created during graph.invoke() attach to this run.
    Stores the run_id in _active_run_id so log_summary_to_run() can find it.
    """
    global _active_run_id
    if not _MLFLOW_AVAILABLE or not _check_server():
        yield None
        return

    _set_experiment()

    run_id = None
    try:
        with mlflow.start_run(run_name=f"query: {query[:60]}") as run:
            run_id = run.info.run_id
            _active_run_id = run_id
            _safe_log(mlflow.set_tag, "run_type", "query")
            _safe_log(mlflow.set_tag, "query", query[:200])
            yield run_id
    except Exception as exc:
        logger.debug(f"MLflow query context failed: {exc}")
        yield None
    finally:
        _active_run_id = None


def get_active_run_id() -> str | None:
    """Get the run_id of the current active MLflow run context."""
    return _active_run_id


def log_summary_to_run(run_id: str | None = None, params: dict | None = None,
                       metrics: dict | None = None, tags: dict | None = None,
                       artifact_name: str | None = None, artifact_data: dict | None = None):
    """Log params, metrics, tags, and an artifact to an existing MLflow run.

    Used to log summary data after the graph invocations complete.
    The run must have been started by start_optimization_context() or
    start_query_context() in the same process.

    If run_id is None, tries to use the active run_id from the context.
    """
    if run_id is None:
        run_id = _active_run_id
    if not _MLFLOW_AVAILABLE or not _check_server() or run_id is None:
        return

    try:
        # Re-attach to the existing run to log additional data
        with mlflow.start_run(run_id=run_id, nested=True):
            if params:
                for k, v in params.items():
                    _safe_log(mlflow.log_param, k, v)
            if metrics:
                for k, v in metrics.items():
                    _safe_log(mlflow.log_metric, k, v or 0.0)
            if tags:
                for k, v in tags.items():
                    _safe_log(mlflow.set_tag, k, v)
            if artifact_name and artifact_data:
                _log_artifact_json(artifact_name, artifact_data)
    except Exception as exc:
        logger.debug(f"MLflow summary logging skipped: {exc}")


def log_query_run(query: str, result: dict, config: dict, version: str = "v1") -> str | None:
    """Log a single pipeline query run to MLflow."""
    if not _MLFLOW_AVAILABLE or not _check_server():
        return None

    _set_experiment()

    try:
        with mlflow.start_run(run_name=f"query: {query[:60]}") as run:
            # ── Parameters ─────────────────────────────────────────────
            _safe_log(mlflow.log_param, "chunk_strategy", config.get("chunk_strategy"))
            _safe_log(mlflow.log_param, "chunk_size", config.get("chunk_size"))
            _safe_log(mlflow.log_param, "chunk_overlap", config.get("chunk_overlap"))
            _safe_log(mlflow.log_param, "retrieval_k", config.get("retrieval_k"))
            _safe_log(mlflow.log_param, "max_context_tokens", config.get("max_context_tokens"))
            _safe_log(mlflow.log_param, "prompt_template", config.get("prompt_template"))
            _safe_log(mlflow.log_param, "version", version)

            # ── Metrics ────────────────────────────────────────────────
            _safe_log(mlflow.log_metric, "unified_score", result.get("unified_score") or 0.0)
            _safe_log(mlflow.log_metric, "faithfulness", result.get("faithfulness") or 0.0)
            _safe_log(mlflow.log_metric, "relevance", result.get("relevance") or 0.0)
            _safe_log(mlflow.log_metric, "correctness", result.get("correctness") or 0.0)
            _safe_log(mlflow.log_metric, "retrieval_score", result.get("retrieval_score") or 0.0)
            _safe_log(mlflow.log_metric, "latency_ms", result.get("latency_ms") or result.get("generation_latency_ms") or 0)
            _safe_log(mlflow.log_metric, "cost_usd", result.get("cost_usd") or result.get("generation_cost_usd") or 0.0)
            _safe_log(mlflow.log_metric, "chunk_count", result.get("chunk_count") or 0)
            _safe_log(mlflow.log_metric, "context_tokens", result.get("context_tokens") or 0)

            # ── Tags ───────────────────────────────────────────────────
            _safe_log(mlflow.set_tag, "run_type", "query")
            _safe_log(mlflow.set_tag, "gate_decision", result.get("gate_decision", "unknown"))
            _safe_log(mlflow.set_tag, "query", query[:200])

            # ── Artifact ───────────────────────────────────────────────
            _log_artifact_json("query_result.json", result)

            logger.info(f"MLflow query run logged: {run.info.run_id}")
            return run.info.run_id

    except Exception as exc:
        logger.debug(f"MLflow query run logging failed: {exc}")
        return None


def log_optimization_run(query: str, report: dict, run_id: str | None = None) -> str | None:
    """Log a full optimization loop to MLflow with nested child runs.

    If run_id is provided, logs to the existing run (started by
    start_optimization_context()) instead of creating a new one.
    This ensures @mlflow.trace() spans attach correctly.
    """
    if not _MLFLOW_AVAILABLE or not _check_server():
        return None

    _set_experiment()

    try:
        init_cfg = report.get("initial_config", {})
        final_cfg = report.get("final_config", {})

        params = {
            "initial_chunk_size": init_cfg.get("chunk_size"),
            "initial_retrieval_k": init_cfg.get("retrieval_k"),
            "final_chunk_size": final_cfg.get("chunk_size"),
            "final_retrieval_k": final_cfg.get("retrieval_k"),
            "stop_reason": report.get("stop_reason"),
        }
        metrics = {
            "initial_score": report.get("initial_score") or 0.0,
            "final_score": report.get("final_score") or 0.0,
            "improvement": report.get("improvement") or 0.0,
            "total_iterations": report.get("total_iterations") or 0,
        }
        tags = {
            "stop_reason": report.get("stop_reason", "unknown"),
        }

        if run_id:
            # Log to the existing run (traces already attached)
            log_summary_to_run(
                run_id,
                params=params,
                metrics=metrics,
                tags=tags,
                artifact_name="optimization_report.json",
                artifact_data=report,
            )
            return run_id

        # Fallback: create a new run (no traces — standalone logging)
        parent_name = f"optimize: {query[:60]}"
        with mlflow.start_run(run_name=parent_name) as parent_run:
            for k, v in params.items():
                _safe_log(mlflow.log_param, k, v)
            for k, v in metrics.items():
                _safe_log(mlflow.log_metric, k, v)
            _safe_log(mlflow.set_tag, "run_type", "optimization")
            for k, v in tags.items():
                _safe_log(mlflow.set_tag, k, v)
            _safe_log(mlflow.set_tag, "query", query[:200])

            iterations = report.get("iterations", [])
            for i, rec in enumerate(iterations):
                iter_name = f"iter-{i}: {rec.get('gate_decision', '?')}"
                with mlflow.start_run(run_name=iter_name, nested=True):
                    iter_cfg = rec.get("config", {})
                    _safe_log(mlflow.log_param, "chunk_size", iter_cfg.get("chunk_size"))
                    _safe_log(mlflow.log_param, "retrieval_k", iter_cfg.get("retrieval_k"))
                    _safe_log(mlflow.log_param, "chunk_overlap", iter_cfg.get("chunk_overlap"))
                    _safe_log(mlflow.log_param, "prompt_template", iter_cfg.get("prompt_template"))

                    _safe_log(mlflow.log_metric, "unified_score", rec.get("unified_score") or 0.0)
                    _safe_log(mlflow.log_metric, "faithfulness", rec.get("faithfulness") or 0.0)
                    _safe_log(mlflow.log_metric, "relevance", rec.get("relevance") or 0.0)
                    _safe_log(mlflow.log_metric, "retrieval_score", rec.get("retrieval_score") or 0.0)
                    _safe_log(mlflow.log_metric, "latency_ms", rec.get("latency_ms") or 0)
                    _safe_log(mlflow.log_metric, "cost_usd", rec.get("cost_usd") or 0.0)

                    _safe_log(mlflow.set_tag, "gate_decision", rec.get("gate_decision", "unknown"))
                    _safe_log(mlflow.set_tag, "failure_type", rec.get("failure_type", "none"))
                    _safe_log(mlflow.set_tag, "iteration", i)

                    variant = rec.get("applied_variant")
                    if variant:
                        _safe_log(mlflow.set_tag, "variant_id", variant.get("variant_id", ""))
                        _safe_log(mlflow.set_tag, "delta", str(variant.get("delta", {})))
                        _safe_log(mlflow.set_tag, "rationale", variant.get("rationale", "")[:200])

                    _log_artifact_json(f"iteration_{i}_result.json", rec)

            _log_artifact_json("optimization_report.json", report)

            logger.info(f"MLflow optimization run logged: {parent_run.info.run_id}")
            return parent_run.info.run_id

    except Exception as exc:
        logger.debug(f"MLflow optimization run logging failed: {exc}")
        return None


def _log_artifact_json(filename: str, data: dict):
    """Write a dict to a temp JSON file and log it as an MLflow artifact."""
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
    return _check_server()
