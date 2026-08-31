"""Execution trace — append-only observability events for each graph node.

Lightweight in-state tracing pattern (no MLflow, no external tracker).
Every wrapped node's output is packaged as a `TraceEvent` and appended to
state["execution_trace"], which uses the `operator.add` reducer (see
state.py) so events accumulate across the linear graph run instead of being
overwritten by the next node's state update.

The graph owns run tracing: nodes are wrapped at registration time in
agents/graph.py via traced_node(), not inside each agent module.
"""

import time
import logging
import functools
from datetime import datetime, timezone
from typing import Any, Callable, TypedDict

logger = logging.getLogger(__name__)

# Known state keys that carry token/cost/latency numbers, scanned in order.
_TOKEN_KEYS = ("context_tokens",)
_COST_KEYS = ("generation_cost_usd", "cost_usd")

# Fields excluded from summaries to avoid unbounded self-reference growth.
# Everything else is kept but bounded in size via _truncate() below, so
# e.g. "answer" and "context" still show up (truncated), just not raw chunk lists.
_LARGE_FIELDS = ("execution_trace",)
_MAX_STR_LEN = 200


class TraceEvent(TypedDict, total=False):
    node: str
    timestamp: str
    status: str                 # "success" | "error"
    latency_ms: int
    tokens: int | None
    cost_usd: float | None
    input_summary: dict
    output_summary: dict
    error: str | None
    retry_count: int


def _truncate(value: Any) -> Any:
    """Bound a value's size so trace events stay cheap to store/render."""
    if isinstance(value, str) and len(value) > _MAX_STR_LEN:
        return value[:_MAX_STR_LEN] + "…"
    if isinstance(value, list):
        return f"<list len={len(value)}>"
    if isinstance(value, dict):
        return {k: _truncate(v) for k, v in value.items() if k not in _LARGE_FIELDS}
    return value


def _summarize(data: dict) -> dict:
    """Shallow, size-bounded summary of a state/result dict for tracing."""
    return {k: _truncate(v) for k, v in data.items() if k not in _LARGE_FIELDS}


def _first_present(data: dict, keys: tuple) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def traced_node(node_name: str) -> Callable:
    """Decorator: wrap a LangGraph node fn to emit an append-only trace event.

    On success, appends one TraceEvent to the node's returned state update
    (merged via the execution_trace reducer). On error, the trace event is
    logged — LangGraph discards a node's state update when it raises, so
    there is nothing to append to — and the original exception is re-raised
    so upstream retry/error handling still fires.
    """

    def decorator(node_fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
        @functools.wraps(node_fn)
        def wrapper(state: dict) -> dict:
            started = time.perf_counter()
            retry_count = state.get("improvement_attempt", 0)
            try:
                result = node_fn(state)
            except Exception as exc:
                event: TraceEvent = {
                    "node": node_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "error",
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "tokens": None,
                    "cost_usd": None,
                    "input_summary": _summarize(state),
                    "output_summary": {},
                    "error": str(exc),
                    "retry_count": retry_count,
                }
                logger.error("node_trace_error", extra={"trace_event": event})
                raise

            event: TraceEvent = {
                "node": node_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "success",
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "tokens": _first_present(result, _TOKEN_KEYS),
                "cost_usd": _first_present(result, _COST_KEYS),
                "input_summary": _summarize(state),
                "output_summary": _summarize(result),
                "error": None,
                "retry_count": retry_count,
            }
            return {**result, "execution_trace": [event]}

        return wrapper

    return decorator
