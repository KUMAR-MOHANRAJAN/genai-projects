"""Optimizer Service — external orchestrator for the self-improving loop.

NOT a LangGraph node. Sits outside the graph and dispatches full graph runs
repeatedly with different configs until a stop condition fires.

Architecture:
  - The optimizer is a service layer, not a graph node
  - It dispatches whole-graph runs via the existing entry point
  - The graph is linear (one pass) — the optimizer handles all retry logic
  - Every score is REAL (never fabricated or estimated)

Flow:
  1. Dispatch graph with current config → get result
  2. Read score + improver_candidates from result
  3. Check stop conditions
  4. If continuing: apply winner's config, loop back to step 1

Stop conditions (5 total):
  - target_reached: score >= target (success)
  - blocked_faithfulness: faithfulness < floor (safety)
  - hitl_required: score in gray band (need human)
  - no_candidates: graph produced no improvement candidates
  - no_improvement: 3 consecutive iterations with delta < 0.01 (plateau)
  - max_iterations_reached: exhausted the budget
"""

import sys
import os
import copy
import uuid

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import (
    DEFAULT_CONFIG,
    UNIFIED_TARGET,
    HITL_LOW,
    FAITHFULNESS_FLOOR,
    MAX_ITERATIONS,
    NO_IMPROVEMENT_DELTA,
    validate_config,
)
from agents.graph import build_graph
from agents.mlflow_logger import (
    log_optimization_run as _mlflow_log_opt,
    start_optimization_context,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Result container for each iteration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _iteration_record(
    iteration: int,
    config: dict,
    result: dict,
    applied_variant: dict | None = None,
) -> dict:
    """Build a summary record for one iteration."""
    return {
        "iteration": iteration,
        "config": copy.deepcopy(config),
        "unified_score": result.get("unified_score"),
        "faithfulness": result.get("faithfulness"),
        "relevance": result.get("relevance"),
        "correctness": result.get("correctness"),
        "retrieval_score": result.get("retrieval_score"),
        "gate_decision": result.get("gate_decision"),
        "gate_reason": result.get("gate_reason"),
        "failure_type": result.get("failure_type"),
        "remediation_hint": result.get("remediation_hint"),
        "latency_ms": result.get("generation_latency_ms"),
        "cost_usd": result.get("generation_cost_usd"),
        "answer": result.get("answer", ""),
        "retrieved_chunks": result.get("retrieved_chunks", []),
        "chunk_count": result.get("chunk_count", 0),
        "context_tokens": result.get("context_tokens", 0),
        "applied_variant": applied_variant,
        "execution_trace": result.get("execution_trace", []),
        "judge_details": result.get("judge_details", {}),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Core optimization loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_optimization(
    query: str,
    config: dict | None = None,
    version: str = "v1",
    target_score: float = UNIFIED_TARGET,
    max_iterations: int = MAX_ITERATIONS,
    baseline_result: dict | None = None,
    force_continue: bool = False,
) -> dict:
    """Run the bounded autonomous optimization loop.

    Dispatches the graph repeatedly, applying the improver's winning
    candidate after each iteration, until a stop condition fires.

    Args:
        query: The question to optimize for.
        config: Starting pipeline config. Defaults to DEFAULT_CONFIG.
        version: Collection version string.
        target_score: Stop when unified_score >= this.
        max_iterations: Maximum number of graph dispatches.
        baseline_result: Optional pre-computed baseline (from Playground).
            If provided, skips the first graph invocation and uses this
            as iteration 0. The optimizer starts improvement at iteration 1.

    Returns:
        Report dict with:
          - iterations: list of per-iteration records
          - stop_reason: why the loop stopped
          - final_config: the config at the end (may differ from input)
          - final_score: last unified_score achieved
          - improvement: final_score - initial_score
          - total_iterations: how many iterations ran
    """
    current_config = validate_config(config or DEFAULT_CONFIG)
    iterations = []
    last_score = None
    consecutive_no_improvement = 0
    stop_reason = "max_iterations_reached"

    print(f"\n{'=' * 70}")
    print(f"OPTIMIZER — Starting optimization loop")
    print(f"  Query: {query[:80]}...")
    print(f"  Target: {target_score}")
    print(f"  Max iterations: {max_iterations}")
    if baseline_result:
        print(f"  Baseline: pre-supplied (score={baseline_result.get('unified_score')})")
    print(f"{'=' * 70}")

    # Wrap the entire optimization loop in an MLflow run so that
    # @mlflow.trace() spans from graph.invoke() attach to this run.
    mlflow_run_id = None
    with start_optimization_context(query) as mlflow_run_id:

        # ── Handle pre-supplied baseline ──────────────────────────────────────
        if baseline_result:
            score = baseline_result.get("unified_score")
            faithfulness = baseline_result.get("faithfulness")
            gate = baseline_result.get("gate_decision", "hard_block")

            print(f"\n── Baseline (from Playground) ──")
            print(f"  Score: {score:.4f}" if score else "  Score: None")
            print(f"  Gate:  {gate}")
            print(f"  Faithfulness: {faithfulness}")

            # Check stop conditions on baseline
            if faithfulness is not None and faithfulness < FAITHFULNESS_FLOOR:
                if force_continue:
                    print(f"  WARNING: faithfulness {faithfulness:.2f} < {FAITHFULNESS_FLOOR} (force_continue=True, continuing)")
                else:
                    record = _iteration_record(0, current_config, baseline_result)
                    iterations.append(record)
                    stop_reason = "blocked_faithfulness"
                    print(f"  STOP: {stop_reason} (faithfulness {faithfulness:.2f} < {FAITHFULNESS_FLOOR})")
            
            if stop_reason == "max_iterations_reached" and score is not None and score >= target_score:
                record = _iteration_record(0, current_config, baseline_result)
                iterations.append(record)
                stop_reason = "target_reached"
                print(f"  STOP: {stop_reason} (score {score:.4f} >= {target_score})")

            if stop_reason == "max_iterations_reached":
                # A Playground baseline may already have been explicitly rejected
                # by the user, so a HITL-band baseline must still enter the loop.
                record = _iteration_record(0, current_config, baseline_result)
                iterations.append(record)
                last_score = score

            # If we stopped on baseline, skip to report
            if stop_reason != "max_iterations_reached":
                # Build report and return early
                initial_score = iterations[0]["unified_score"] if iterations else None
                final_score = iterations[-1]["unified_score"] if iterations else None
                improvement = (
                    round(final_score - initial_score, 4)
                    if initial_score is not None and final_score is not None
                    else None
                )
                report = {
                    "stop_reason": stop_reason,
                    "total_iterations": len(iterations),
                    "initial_score": initial_score,
                    "final_score": final_score,
                    "improvement": improvement,
                    "initial_config": config or DEFAULT_CONFIG,
                    "final_config": current_config,
                    "iterations": iterations,
                }
                print(f"\n{'=' * 70}")
                print(f"OPTIMIZER — Loop finished (stopped on baseline)")
                print(f"  Stop reason: {stop_reason}")
                print(f"{'=' * 70}\n")
                _mlflow_log_opt(query, report, run_id=mlflow_run_id)
                return report

        for iteration in range(1, max_iterations + 1):
            print(f"\n── Iteration {iteration}/{max_iterations} ──")
            print(f"  Config: k={current_config.get('retrieval_k')}, "
                  f"chunk={current_config.get('chunk_size')}, "
                  f"overlap={current_config.get('chunk_overlap')}, "
                  f"prompt={current_config.get('prompt_template')}")

            # ── 1. Dispatch graph ─────────────────────────────────────────────
            # Each iteration gets a fresh graph + thread to avoid state leakage.
            app = build_graph()
            thread_id = f"opt-{uuid.uuid4().hex[:8]}"

            initial_state = {
                "query": query,
                "config": copy.deepcopy(current_config),
                "version": version,
                "improvement_attempt": iteration - 1,
            }

            result = app.invoke(
                initial_state,
                config={"configurable": {"thread_id": thread_id}},
            )

            score = result.get("unified_score")
            faithfulness = result.get("faithfulness")
            gate = result.get("gate_decision", "hard_block")

            print(f"  Score: {score:.4f}" if score else "  Score: None")
            print(f"  Gate:  {gate}")
            print(f"  Faithfulness: {faithfulness}")

            # ── 2. Check stop conditions ──────────────────────────────────────

            # Safety first: faithfulness hard block
            if faithfulness is not None and faithfulness < FAITHFULNESS_FLOOR:
                if force_continue:
                    print(f"  WARNING: faithfulness {faithfulness:.2f} < {FAITHFULNESS_FLOOR} (force_continue=True, continuing)")
                else:
                    record = _iteration_record(iteration, current_config, result)
                    iterations.append(record)
                    stop_reason = "blocked_faithfulness"
                    print(f"  STOP: {stop_reason} (faithfulness {faithfulness:.2f} < {FAITHFULNESS_FLOOR})")
                    break

            # Target reached
            if score is not None and score >= target_score:
                record = _iteration_record(iteration, current_config, result)
                iterations.append(record)
                stop_reason = "target_reached"
                print(f"  STOP: {stop_reason} (score {score:.4f} >= {target_score})")
                break

            # HITL required — optimizer cannot bypass human approval
            if gate == "hitl_required":
                record = _iteration_record(iteration, current_config, result)
                iterations.append(record)
                stop_reason = "hitl_required"
                print(f"  STOP: {stop_reason} (score in gray band, needs human)")
                break

            # No improvement plateau
            if last_score is not None and score is not None:
                delta = score - last_score
                if delta < NO_IMPROVEMENT_DELTA:
                    consecutive_no_improvement += 1
                else:
                    consecutive_no_improvement = 0

                if consecutive_no_improvement >= 3:
                    record = _iteration_record(iteration, current_config, result)
                    iterations.append(record)
                    stop_reason = "no_improvement"
                    print(f"  STOP: {stop_reason} (3 consecutive iterations with delta < {NO_IMPROVEMENT_DELTA})")
                    break
            last_score = score

            # ── 3. Extract winner candidate ───────────────────────────────────
            candidates = result.get("improver_candidates", [])
            if not candidates:
                record = _iteration_record(iteration, current_config, result)
                iterations.append(record)
                stop_reason = "no_candidates"
                print(f"  STOP: {stop_reason} (graph produced no improvement candidates)")
                break

            # The last candidate in the list is from this iteration
            winner = candidates[-1]

            print(f"  Failure: {winner.get('failure_type', '?')}")
            print(f"  Fix:     {winner.get('rationale', '?')}")
            print(f"  Delta:   {winner.get('delta', {})}")

            # Record this iteration (with the variant that will be applied)
            record = _iteration_record(iteration, current_config, result, applied_variant=winner)
            iterations.append(record)

            # ── 4. Apply winner's config for next iteration ───────────────────
            new_config = winner.get("config_after")
            if new_config is None:
                stop_reason = "no_candidates"
                print(f"  STOP: {stop_reason} (winner has no config_after)")
                break

            current_config = validate_config(new_config)
            print(f"  → Applied. New config for next iteration: "
                  f"k={current_config.get('retrieval_k')}, "
                  f"chunk={current_config.get('chunk_size')}")

        # ── Build report ──────────────────────────────────────────────────────
        initial_score = iterations[0]["unified_score"] if iterations else None
        final_score = iterations[-1]["unified_score"] if iterations else None
        improvement = (
            round(final_score - initial_score, 4)
            if initial_score is not None and final_score is not None
            else None
        )

        report = {
            "stop_reason": stop_reason,
            "total_iterations": len(iterations),
            "initial_score": initial_score,
            "final_score": final_score,
            "improvement": improvement,
            "initial_config": config or DEFAULT_CONFIG,
            "final_config": current_config,
            "iterations": iterations,
        }

        print(f"\n{'=' * 70}")
        print(f"OPTIMIZER — Loop finished")
        print(f"  Stop reason:     {stop_reason}")
        print(f"  Iterations:      {len(iterations)}")
        print(f"  Initial score:   {initial_score}")
        print(f"  Final score:     {final_score}")
        print(f"  Improvement:     {improvement}")
        print(f"{'=' * 70}\n")

        # Log to MLflow — pass run_id so summary attaches to the same run as traces
        _mlflow_log_opt(query, report, run_id=mlflow_run_id)

    return report


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Print report helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_report(report: dict) -> None:
    """Pretty-print an optimization report."""
    print(f"\n{'=' * 70}")
    print(f"OPTIMIZATION REPORT")
    print(f"{'=' * 70}")
    print(f"  Stop reason:   {report['stop_reason']}")
    print(f"  Iterations:    {report['total_iterations']}")
    print(f"  Initial score: {report['initial_score']}")
    print(f"  Final score:   {report['final_score']}")
    print(f"  Improvement:   {report['improvement']}")
    print()

    print("  Config changes:")
    init = report["initial_config"]
    final = report["final_config"]
    for key in sorted(set(init) | set(final)):
        old = init.get(key)
        new = final.get(key)
        if old != new:
            print(f"    {key}: {old} → {new}")

    print()
    print("  Per-iteration scores:")
    for rec in report["iterations"]:
        variant_info = ""
        if rec.get("applied_variant"):
            v = rec["applied_variant"]
            variant_info = f" → applied {v.get('variant_id', '?')} ({v.get('rationale', '')[:50]})"
        print(f"    [{rec['iteration']}] score={rec['unified_score']:.4f}  "
              f"gate={rec['gate_decision']}"
              f"  failure={rec.get('failure_type', '-')}"
              f"{variant_info}")

    print(f"\n{'=' * 70}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Standalone test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    from ground_truth import TEST_QUERIES

    # Use the first ground-truth query for testing
    query, expected_answer, expected_keywords = TEST_QUERIES[0]

    print(f"Test query: {query}")
    print(f"Expected answer: {expected_answer[:100]}...")
    print()

    report = run_optimization(
        query=query,
        config=copy.deepcopy(DEFAULT_CONFIG),
        target_score=UNIFIED_TARGET,
        max_iterations=MAX_ITERATIONS,
    )

    print_report(report)
