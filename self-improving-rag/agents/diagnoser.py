"""Diagnoser Agent — LangGraph node for failure classification.

Pure rule cascade — no LLM call. Deterministic, free, transparent, testable.

Classifies the root cause of a pipeline failure into exactly one of 5 types.
Rules are checked in fixed priority order (first match wins). Priority is
based on severity: safety-critical failures (hallucination) before retrieval
issues before quality issues.

The 5 Failure Types:
  F-01  Retrieval Miss       — didn't find relevant chunks
  F-02  Context Overflow     — found good chunks but couldn't fit them
  F-03  Hallucination        — model made up facts not in context
  F-04  Answer Incomplete    — answer correct but missing key details
  F-05  Latency Spike        — answer fine but too slow

Origin: AutoRAG's app/graph/nodes/diagnoser.py:166 _rule_based_classify()
        Failure types from .claude/specs/agents/07-diagnoser.md
"""

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from state import RunState
from config import FAITHFULNESS_FLOOR, RETRIEVAL_SIM_FLOOR, LATENCY_CAP_MS


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Remediation hints — pre-written constants, deterministic, no LLM needed.
# Each maps a failure code to a human-readable explanation + suggested fix.
# These also feed the improver's playbook lookup.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_FAILURE_CATALOG = {
    "F-01": {
        "name": "Retrieval Miss",
        "remediation_hint": (
            "Retrieved chunks lack relevant content. "
            "Try: increase retrieval_k, reduce chunk_size for finer granularity, "
            "switch chunking strategy (e.g., recursive_split), or increase chunk_overlap."
        ),
    },
    "F-02": {
        "name": "Context Overflow",
        "remediation_hint": (
            "Relevant chunks found but context window is saturated. "
            "Try: reduce retrieval_k to fetch fewer chunks, reduce chunk_size, "
            "or increase max_context_tokens budget."
        ),
    },
    "F-03": {
        "name": "Hallucination",
        "remediation_hint": (
            "Answer contains claims not supported by the retrieved context. "
            "Try: switch to stricter prompt template (v2), increase chunk_overlap "
            "for better context continuity, or increase retrieval_k for more grounding."
        ),
    },
    "F-04": {
        "name": "Answer Incomplete",
        "remediation_hint": (
            "Answer is on-topic but missing key details. "
            "Try: increase chunk_size for more context per chunk, increase retrieval_k "
            "to retrieve more content, or increase chunk_overlap."
        ),
    },
    "F-05": {
        "name": "Latency Spike",
        "remediation_hint": (
            "Pipeline response time exceeds threshold. "
            "Try: reduce retrieval_k (fewer chunks = less context = faster generation), "
            "reduce chunk_size, or simplify prompt template."
        ),
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule Cascade — checked in fixed priority order (first match wins)
#
# Priority order rationale:
#   1. F-03 Hallucination   — most dangerous, safety-critical
#   2. F-01 Retrieval Miss  — if retrieval failed, everything downstream is compromised
#   3. F-02 Context Overflow — found good content but couldn't use it all
#   4. F-05 Latency Spike   — answer was fine but too slow
#   5. F-04 Answer Incomplete — catch-all for low scores with no specific failure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _classify_failure(state: RunState) -> tuple[str, float, str]:
    """Apply rule cascade to classify failure type.

    Returns:
        (failure_type, confidence, root_cause_analysis)
    """
    faithfulness = state.get("faithfulness")
    retrieval_score = state.get("retrieval_score", 0.0)
    context_tokens = state.get("context_tokens", 0)
    max_context = state.get("config", {}).get("max_context_tokens", 4000)
    latency_ms = state.get("latency_ms", 0)
    unified_score = state.get("unified_score", 0.0)

    # ── Priority 1: Hallucination (F-03) ─────────────────────────────────
    # Most dangerous — answer contains unsupported claims.
    # Checked first because a hallucinated answer is actively harmful
    # regardless of other metrics.
    if faithfulness is not None and faithfulness < FAITHFULNESS_FLOOR:
        return (
            "F-03",
            0.97,
            f"Faithfulness {faithfulness:.2f} < {FAITHFULNESS_FLOOR} floor. "
            f"Answer contains claims not grounded in retrieved context.",
        )

    # ── Priority 2: Retrieval Miss (F-01) ────────────────────────────────
    # Didn't find relevant content. If retrieval fails, everything downstream
    # is compromised — no point analyzing answer quality.
    chunks = state.get("retrieved_chunks", [])
    if not chunks or retrieval_score < RETRIEVAL_SIM_FLOOR:
        return (
            "F-01",
            0.90,
            f"Retrieval score {retrieval_score:.2f} < {RETRIEVAL_SIM_FLOOR} floor. "
            f"Retrieved {len(chunks)} chunks but content lacks relevant information.",
        )

    # ── Priority 3: Context Overflow (F-02) ──────────────────────────────
    # Found good content but couldn't fit it all into the context window.
    # Token budget is saturated.
    if context_tokens >= max_context * 0.95:  # 95% = effectively full
        return (
            "F-02",
            0.88,
            f"Context tokens {context_tokens} approaching budget {max_context}. "
            f"Relevant content may have been truncated.",
        )

    # ── Priority 4: Latency Spike (F-05) ─────────────────────────────────
    # Answer was fine but pipeline is too slow. Lower priority because a
    # slow correct answer is better than a fast wrong one.
    if latency_ms > LATENCY_CAP_MS:
        return (
            "F-05",
            0.80,
            f"Latency {latency_ms}ms > {LATENCY_CAP_MS}ms cap. "
            f"Pipeline response time exceeds acceptable threshold.",
        )

    # ── Priority 5: Answer Incomplete (F-04) — catch-all ─────────────────
    # Nothing critical went wrong, but the score is still low. Usually
    # means the answer needs more context or a better prompt.
    return (
        "F-04",
        0.70,
        f"Unified score {unified_score:.4f} below target with no specific failure pattern. "
        f"Answer may be on-topic but missing key details.",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LangGraph Node
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def diagnoser_node(state: RunState) -> dict:
    """LangGraph node: classify pipeline failure into one of 5 types.

    Reads from state:
      - faithfulness, retrieval_score, context_tokens, latency_ms, unified_score
      - retrieved_chunks, config (for context budget)

    Writes to state:
      - failure_type      (F-01 through F-05)
      - confidence        (0.0-1.0, higher for more certain classifications)
      - remediation_hint  (human-readable suggestion for the improver)
      - root_cause_analysis (explains WHY with actual metric values)
    """
    failure_type, confidence, root_cause = _classify_failure(state)

    catalog_entry = _FAILURE_CATALOG[failure_type]

    return {
        "failure_type": failure_type,
        "confidence": confidence,
        "remediation_hint": catalog_entry["remediation_hint"],
        "root_cause_analysis": root_cause,
    }
