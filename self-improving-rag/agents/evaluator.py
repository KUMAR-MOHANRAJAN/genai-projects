"""Evaluator Agent — LangGraph node for pipeline evaluation.

Wraps the existing LLM judge functions from pipeline.py as a LangGraph node.
Does NOT duplicate judge code — imports and calls the existing functions.

Responsibilities:
  1. Run 3 LLM judges (faithfulness, relevance, correctness)
  2. Compute retrieval metrics (keyword-based, no LLM)
  3. Compute quality sub-score and unified score (ADR-004 formula v1.2)
  4. Make the gate decision (deploy / HITL / block)
  5. Package everything as a state update

Latency accounting:
  Uses generation_latency_ms (the LLM API call time) — NOT full wall-clock.
  In production, the user gets the answer after generation; judge calls run
  async in the background. The latency penalty should reflect what the user
  experiences, not internal grading overhead.

Cost accounting:
  Uses generation_cost_usd only — judge calls are evaluation overhead,
  not pipeline cost the user pays for.

Architecture: LangGraph evaluation node with LLM judges and unified scoring.
"""

import sys
import os

# Ensure project root is on sys.path so we can import pipeline, config, etc.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from state import RunState
from config import UNIFIED_TARGET, HITL_LOW, FAITHFULNESS_FLOOR
from ground_truth import TEST_QUERIES
from utils import compute_gate_decision
from agents.trace import traced_node
from pipeline import (
    _judge_faithfulness,
    _judge_relevance,
    _judge_correctness,
    _compute_quality_score,
    _compute_unified_score,
    _compute_retrieval_score,
    _find_keywords,
    precision_at_k,
    recall_at_k,
)

# MLflow tracing — best-effort, graceful fallback
try:
    import mlflow
    _mlflow_trace = mlflow.trace
except ImportError:
    _mlflow_trace = lambda **kwargs: lambda fn: fn  # no-op decorator


@_mlflow_trace(name="evaluator", span_type="evaluation")
@traced_node("evaluator")
def evaluator_node(state: RunState) -> dict:
    """LangGraph node: evaluate a completed pipeline run.

    Reads from state:
      - answer, context, query          (for LLM judges)
      - retrieved_chunks, config        (for retrieval metrics)
      - generation_latency_ms           (for latency penalty)
      - generation_cost_usd             (for cost penalty)

    Writes to state:
      - unified_score, faithfulness, relevance, correctness
      - retrieval_score, retrieval_precision, retrieval_recall
      - gate_decision, gate_reason
      - judge_reasoning
      - cost_usd, latency_ms
    """
    query = state["query"]
    answer = state["answer"]
    context = state["context"]
    chunks = state.get("retrieved_chunks", [])
    cfg = state.get("config", {})
    k = cfg.get("retrieval_k", 5)

    # ── 1. Retrieval Metrics (keyword-based, NO LLM call) ────────────────
    keywords = _find_keywords(query)
    precision = precision_at_k(chunks, keywords, k=k) if keywords else 0.0
    recall_kw = recall_at_k(chunks, keywords, k=k) if keywords else 0.0
    retrieval_score = _compute_retrieval_score(chunks, keywords, k=k)

    # ── 2. LLM Judges (3 separate calls) ─────────────────────────────────
    judge_reasoning = {}
    judge_details = {}

    faithfulness, faith_reasoning, faith_detail = _judge_faithfulness(answer, context)
    if faith_reasoning:
        judge_reasoning["faithfulness"] = faith_reasoning
    judge_details["faithfulness"] = faith_detail

    relevance, rel_reasoning, rel_detail = _judge_relevance(answer, query)
    if rel_reasoning:
        judge_reasoning["relevance"] = rel_reasoning
    judge_details["relevance"] = rel_detail

    # Correctness requires ground truth — only available for test queries
    expected_answer = None
    for q, ea, _ in TEST_QUERIES:
        if q.lower() == query.lower():
            expected_answer = ea
            break

    correctness = None
    if expected_answer:
        correctness, corr_reasoning, corr_detail = _judge_correctness(answer, query, expected_answer)
        if corr_reasoning:
            judge_reasoning["correctness"] = corr_reasoning
        judge_details["correctness"] = corr_detail

    # ── 3. Sub-scores and Unified Score ──────────────────────────────────
    quality = _compute_quality_score(relevance, correctness)

    # Latency = generation LLM call only (not judge overhead)
    # Cost = generation LLM call only (judges are eval overhead)
    latency_ms = state.get("generation_latency_ms", 0)
    cost_usd = state.get("generation_cost_usd", 0.0)

    unified_score = _compute_unified_score(
        recall=retrieval_score,
        quality=quality,
        faithfulness=faithfulness,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )

    # ── 4. Gate Decision (single source of truth) ─────────────────────────
    gate_decision, gate_reason = compute_gate_decision(unified_score, faithfulness)

    # ── 5. Return state updates ──────────────────────────────────────────
    return {
        "unified_score": unified_score,
        "faithfulness": faithfulness,
        "relevance": relevance,
        "correctness": correctness,
        "retrieval_score": retrieval_score,
        "retrieval_precision": precision,
        "retrieval_recall": recall_kw,
        "gate_decision": gate_decision,
        "gate_reason": gate_reason,
        "judge_reasoning": judge_reasoning,
        "judge_details": judge_details,
        "cost_usd": cost_usd,
        "latency_ms": latency_ms,
    }
