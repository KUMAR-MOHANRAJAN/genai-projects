"""RunState — typed dict that carries data through the pipeline.

Mirrors AutoRAG's GraphState (app/core/state.py) but simplified for learning.
Every node reads and writes fields from this shared state.
"""

import operator
from typing import TypedDict, Any, Annotated


class RunState(TypedDict, total=False):
    """Fields that flow through the pipeline.

    total=False means all fields are optional — nodes only write what they own.
    """
    # Input
    query: str
    config: dict[str, Any]          # {chunk_strategy, chunk_size, k, ...}
    version: str                    # "v1", "v2", ...

    # Ingestion
    collection_name: str
    chunk_count: int

    # Retrieval
    retrieved_chunks: list[dict]    # [{chunk_id, text, score, metadata}, ...]
    retrieval_score: float          # 0.5 × precision + 0.5 × recall
    retrieval_precision: float      # precision@k (keyword-based)
    retrieval_recall: float         # recall@k (keyword-based)

    # Context assembly
    context: str
    context_tokens: int

    # Generation
    answer: str
    generation_cost_usd: float
    generation_latency_ms: int

    # Evaluation
    unified_score: float
    gate_decision: str              # "deploy_eligible", "hitl_required", "hard_block"
    gate_reason: str                # human-readable explanation of gate decision
    faithfulness: float
    relevance: float
    correctness: float | None          # None if no ground truth
    cost_usd: float
    latency_ms: int
    judge_reasoning: dict[str, str]    # {metric_name: reasoning} — feeds diagnoser
    judge_details: dict[str, dict]     # {metric_name: {claims, supported, reasoning, ...}}

    # Diagnosis (filled by diagnoser node)
    failure_type: str               # "F-01", "F-02", "F-03", "F-04", "F-05"
    confidence: float
    remediation_hint: str
    root_cause_analysis: str        # explains WHY with actual metric values

    # Improvement (filled by improver node)
    improver_candidates: list[dict] # [{variant_id, config, estimated_score, is_winner}, ...]

    # Loop control (optimizer tracks retries externally; improvement_attempt
    # is passed in by the optimizer so the improver knows which playbook
    # entry to use — attempt 0 = conservative, 1 = moderate, 2 = aggressive)
    improvement_attempt: int

    # Execution trace (filled by agents/trace.py's traced_node wrapper).
    # Annotated with operator.add so each node's single-event list is
    # concatenated onto the running trace instead of overwriting it.
    execution_trace: Annotated[list[dict], operator.add]


def initial_state(
    query: str,
    config: dict[str, Any],
    version: str = "v1",
) -> RunState:
    """Create a fresh RunState with defaults."""
    return RunState(
        query=query,
        config=config,
        version=version,
        improvement_attempt=0,
        execution_trace=[],
    )
