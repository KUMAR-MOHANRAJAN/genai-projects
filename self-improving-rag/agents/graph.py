"""Self-Improving RAG Graph — LangGraph StateGraph wiring.

LINEAR graph — one pass per invocation. The optimizer service handles
all retry/improvement logic externally by calling this graph repeatedly
with different configs.

Nodes:
  builder_node    — ingest-if-needed + retrieval (from agents/builder.py)
  pipeline_node   — assemble context + generate answer (NO retrieval)
  evaluator_node  — 3 LLM judges + unified score + gate decision
  hitl_node       — interrupt() for human approval in gray band
  diagnoser_node  — rule-cascade failure classification (F-01..F-05)
  improver_node   — playbook lookup + config delta → candidate config

Graph flow:
  START → builder → pipeline → evaluator → [route]
    deploy_eligible → END
    hitl_required   → hitl → END (approve) or diagnoser (reject)
    hard_block      → diagnoser → improver → END (candidates in state)

The graph NEVER loops. After one pass, the final state contains:
  - The answer + scores (always)
  - failure_type + improver_candidates (only if hard_block or HITL reject)

The optimizer reads improver_candidates from the result and decides
whether to apply the winner and dispatch another graph run.

Architecture: Linear StateGraph with conditional routing and external optimizer.
"""

import sys
import os
import copy

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from state import RunState
from config import DEFAULT_CONFIG
from generation.context_assembly import assemble_context
from generation.generator import generate
from agents.builder import builder_node
from agents.evaluator import evaluator_node
from agents.diagnoser import diagnoser_node
from agents.improver import improver_node


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pipeline Node — assemble context + generate (NO ingestion or retrieval)
#
# Builder handles ingestion + retrieval. This node receives pre-retrieved
# chunks from state and focuses on context assembly + LLM generation.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def pipeline_node(state: RunState) -> dict:
    """LangGraph node: assemble context + generate answer.

    Builder has already populated state with retrieved_chunks.

    Reads from state:
      - query, config
      - retrieved_chunks (from builder_node)

    Writes to state:
      - context, context_tokens
      - answer, generation_cost_usd, generation_latency_ms
    """
    query = state["query"]
    cfg = state.get("config", copy.deepcopy(DEFAULT_CONFIG))
    chunks = state.get("retrieved_chunks", [])

    # ── Step 1: Context Assembly ──────────────────────────────────────────
    max_tokens = cfg.get("max_context_tokens", 4000)
    context, context_tokens = assemble_context(chunks, max_tokens=max_tokens)

    # ── Step 2: Generation ────────────────────────────────────────────────
    prompt_version = cfg.get("prompt_template", "v1")
    gen_result = generate(context, query, prompt_version=prompt_version)

    return {
        "context": context,
        "context_tokens": context_tokens,
        "answer": gen_result["answer"],
        "generation_cost_usd": gen_result["cost_usd"],
        "generation_latency_ms": gen_result["latency_ms"],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HITL Node — interrupt() for human review
#
# When the evaluator puts the run in the gray band (0.70-0.85), this node
# pauses the graph and asks the human to approve or reject the answer.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def hitl_node(state: RunState) -> dict:
    """LangGraph node: pause for human approval.

    Calls interrupt() with the answer + score for the human to review.
    Returns the human's decision ("approve" or "reject").
    """
    human_decision = interrupt({
        "message": "Answer is in the gray band — human review required.",
        "answer": state.get("answer", ""),
        "unified_score": state.get("unified_score", 0.0),
        "gate_reason": state.get("gate_reason", ""),
        "options": ["approve", "reject"],
    })

    return {"gate_decision": human_decision}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Conditional Routing Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def route_after_eval(state: RunState) -> str:
    """Route based on evaluator's gate decision.

    deploy_eligible → END (answer passes quality gate)
    hitl_required   → hitl (pause for human review)
    hard_block      → diagnoser (diagnose + generate improvement candidates)
    """
    gate = state.get("gate_decision", "hard_block")

    if gate == "deploy_eligible":
        return END
    elif gate == "hitl_required":
        return "hitl"
    else:  # hard_block
        return "diagnoser"


def route_after_hitl(state: RunState) -> str:
    """Route based on human's HITL decision.

    approve → END (human accepted the answer)
    reject  → diagnoser (diagnose + generate improvement candidates)
    """
    decision = state.get("gate_decision", "reject")

    if decision == "approve":
        return END
    else:
        return "diagnoser"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Build the Graph
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_graph(checkpointer=None):
    """Build and compile the linear RAG + evaluation graph.

    The graph runs ONE pass: pipeline → evaluator → (optional: diagnoser →
    improver) → END. No internal loop. The optimizer service calls this
    graph repeatedly with different configs to implement the retry loop.

    Args:
        checkpointer: LangGraph checkpointer for state persistence.
                      Required for interrupt() (HITL). Pass MemorySaver()
                      for dev, or a database-backed one for production.
                      If None, MemorySaver() is used by default.

    Returns:
        Compiled LangGraph application.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = StateGraph(RunState)

    # ── Register Nodes ────────────────────────────────────────────────────
    graph.add_node("builder", builder_node)
    graph.add_node("pipeline", pipeline_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("hitl", hitl_node)
    graph.add_node("diagnoser", diagnoser_node)
    graph.add_node("improver", improver_node)

    # ── Edges ─────────────────────────────────────────────────────────────

    # START → builder (ingest if needed, then retrieve)
    graph.add_edge(START, "builder")

    # builder → pipeline (context assembly + generation)
    graph.add_edge("builder", "pipeline")

    # pipeline → evaluator (always evaluate after generating)
    graph.add_edge("pipeline", "evaluator")

    # evaluator → conditional routing (deploy / hitl / diagnose)
    graph.add_conditional_edges("evaluator", route_after_eval, {
        END: END,
        "hitl": "hitl",
        "diagnoser": "diagnoser",
    })

    # hitl → conditional routing (approve / reject)
    graph.add_conditional_edges("hitl", route_after_hitl, {
        END: END,
        "diagnoser": "diagnoser",
    })

    # diagnoser → improver (always)
    graph.add_edge("diagnoser", "improver")

    # improver → END (candidates are in state; optimizer reads them)
    graph.add_edge("improver", END)

    return graph.compile(checkpointer=checkpointer)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Standalone test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 70)
    print("GRAPH — Structure Verification (Linear, with Builder)")
    print("=" * 70)

    app = build_graph()

    graph_obj = app.get_graph()
    print(f"\nNodes: {list(graph_obj.nodes.keys())}")
    print(f"\nEdges:")
    for edge in graph_obj.edges:
        cond = " (conditional)" if edge.conditional else ""
        print(f"  {edge.source} → {edge.target}{cond}")

    print(f"\nGraph compiled successfully with {len(graph_obj.nodes)} nodes.")
    print("\nExpected flow (linear — no loop):")
    print("  START → builder → pipeline → evaluator → [route]")
    print("    deploy_eligible → END")
    print("    hitl_required   → hitl → [approve→END | reject→diagnoser]")
    print("    hard_block      → diagnoser → improver → END")
    print()
    print("  Builder handles: ingest-if-needed + retrieval")
    print("  Pipeline handles: context assembly + generation")
    print("  Optimizer reads improver_candidates from result and decides")
    print("  whether to dispatch another graph run with the new config.")
    print()
    print("=" * 70)
    print("STRUCTURE TEST PASSED")
    print("=" * 70)
