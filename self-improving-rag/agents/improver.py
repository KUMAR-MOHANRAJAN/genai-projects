"""Improver Agent — LangGraph node that proposes a config fix.

Given a failure type from the diagnoser, looks up a static playbook of
config deltas and applies the best-guess fix to the current config.

Design decisions:
  - ONE candidate per invocation (sequential trial, not batch).
    The graph loop re-runs the pipeline with this config. If it still fails,
    the diagnoser re-classifies (failure type may shift!) and the improver
    proposes a different fix. Max 3 retries.
  - No LLM call — deterministic lookup + arithmetic.
  - Each invocation APPENDS to state["improver_candidates"] so the
    optimizer can see the full history of what was tried.
  - _apply_delta() clamps values to sane bounds.

Smarter alternatives (not implemented — documented for reference):
  1. LLM-assisted variant suggestion:
     Ask the LLM "given these metrics and failure type, propose a config change."
     Pros: can discover novel fixes beyond the playbook, adapts to edge cases.
     Cons: adds latency + cost per retry, non-deterministic (same failure may
     get different fixes on different runs), harder to test/debug.

  2. Historical learning:
     Track which deltas worked for which failure types across past runs.
     Prioritize deltas with higher historical success rates.
     Requires a run history database (e.g., MLflow or PostgreSQL).
     Pros: improves over time, data-driven.
     Cons: cold-start problem (no history = no signal), requires infra.

  This project uses the deterministic playbook because it's transparent,
  testable, free, and sufficient for demonstrating the self-improvement loop.
  In production, you'd layer LLM suggestions on top with the playbook as
  fallback.

Architecture: Deterministic playbook-based config improvement.
"""

import sys
import os
import copy

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from state import RunState
from config import DEFAULT_CONFIG

# MLflow tracing — best-effort, graceful fallback
try:
    import mlflow
    _mlflow_trace = mlflow.trace
except ImportError:
    _mlflow_trace = lambda **kwargs: lambda fn: fn  # no-op decorator


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Bounds — clamp values to prevent nonsensical configs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_BOUNDS = {
    "chunk_size":          (64, 1024),    # tokens per chunk
    "chunk_overlap":       (0, 128),      # overlap between chunks
    "retrieval_k":         (1, 20),       # how many chunks to retrieve
    "max_context_tokens":  (1000, 8000),  # context window budget
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Playbook — failure type → config delta
#
# Each entry is a list of deltas tried in order across retry iterations.
# On attempt 0 we use index 0, attempt 1 → index 1, etc.
# If attempt >= len(deltas), we wrap around (shouldn't happen with max 3 retries).
#
# Numeric values are ADDITIVE (retrieval_k: +3 means add 3 to current).
# String values are REPLACEMENTS (prompt_template: "v2" means switch to v2).
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_PLAYBOOK: dict[str, list[dict]] = {
    # F-01 Retrieval Miss — didn't find relevant chunks
    # Strategy: cast a wider net, use finer granularity
    "F-01": [
        {   # Attempt 0: conservative — retrieve more chunks
            "delta": {"retrieval_k": +3},
            "rationale": "Increase retrieval_k to cast a wider net for relevant content.",
        },
        {   # Attempt 1: moderate — smaller chunks + more overlap
            "delta": {"chunk_size": -64, "chunk_overlap": +32},
            "rationale": "Reduce chunk_size for finer granularity and add overlap to avoid losing content at boundaries.",
        },
        {   # Attempt 2: aggressive — all three knobs
            "delta": {"retrieval_k": +5, "chunk_size": -128, "chunk_overlap": +64},
            "rationale": "Aggressive retrieval fix: more chunks, smaller and overlapping for maximum recall.",
        },
    ],

    # F-02 Context Overflow — found good content but context window is full
    # Strategy: reduce volume flowing into context
    "F-02": [
        {   # Attempt 0: retrieve fewer chunks
            "delta": {"retrieval_k": -2},
            "rationale": "Reduce retrieval_k to fit fewer chunks into context window.",
        },
        {   # Attempt 1: smaller chunks
            "delta": {"chunk_size": -64, "retrieval_k": -1},
            "rationale": "Smaller chunks + fewer of them to reduce total context tokens.",
        },
        {   # Attempt 2: aggressive reduction
            "delta": {"retrieval_k": -3, "chunk_size": -128},
            "rationale": "Aggressive context reduction: significantly fewer and smaller chunks.",
        },
    ],

    # F-03 Hallucination — model made up facts not in context
    # Strategy: stricter grounding instructions + better retrieval for grounding
    "F-03": [
        {   # Attempt 0: switch to stricter prompt + more retrieval for grounding
            "delta": {"prompt_template": "v2", "retrieval_k": +2},
            "rationale": "Switch to prompt v2 with stricter grounding + retrieve more chunks for better coverage.",
        },
        {   # Attempt 1: stricter prompt + bigger chunks for more context per chunk
            "delta": {"prompt_template": "v2", "chunk_size": +64, "chunk_overlap": +32},
            "rationale": "Stricter prompt + larger overlapping chunks to provide more coherent context for grounding.",
        },
        {   # Attempt 2: aggressive — more retrieval + bigger chunks + overlap
            "delta": {"prompt_template": "v2", "retrieval_k": +3, "chunk_size": +128, "chunk_overlap": +32},
            "rationale": "Aggressive grounding fix: stricter prompt + much more context for the model to ground claims.",
        },
    ],

    # F-04 Answer Incomplete — response is partial or abstained despite context
    # Strategy: first give the model more content to work with
    "F-04": [
        {   # Attempt 0: retrieve more chunks
            "delta": {"retrieval_k": +2},
            "rationale": "Retrieve more chunks to provide additional information for a complete answer.",
        },
        {   # Attempt 1: bigger chunks + overlap
            "delta": {"chunk_size": +64, "chunk_overlap": +32},
            "rationale": "Larger chunks with overlap to capture more context per chunk.",
        },
        {   # Attempt 2: more of everything
            "delta": {"retrieval_k": +3, "chunk_size": +128, "chunk_overlap": +64},
            "rationale": "Aggressive completeness fix: more chunks, larger and overlapping.",
        },
    ],

    # F-05 Latency Spike — answer is fine but too slow
    # Strategy: reduce work for the LLM (less context = faster)
    "F-05": [
        {   # Attempt 0: fewer chunks
            "delta": {"retrieval_k": -2},
            "rationale": "Retrieve fewer chunks to reduce context size and speed up generation.",
        },
        {   # Attempt 1: fewer + smaller chunks
            "delta": {"retrieval_k": -2, "chunk_size": -64},
            "rationale": "Fewer and smaller chunks for faster generation with less context.",
        },
        {   # Attempt 2: minimal context
            "delta": {"retrieval_k": -3, "chunk_size": -128},
            "rationale": "Aggressive latency reduction: minimal context for fastest generation.",
        },
    ],
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _apply_delta — merge a delta dict into a config, clamping to bounds
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _apply_delta(base_config: dict, delta: dict) -> dict:
    """Produce a new config by applying delta to base_config.

    Numeric deltas are additive:  base + delta_value
    String deltas are replacements: overwrite with delta_value

    All numeric values are clamped to _BOUNDS.
    Keys not in the delta are copied unchanged.

    Args:
        base_config: Current pipeline config dict.
        delta: Dict of changes, e.g. {"retrieval_k": +3, "prompt_template": "v2"}.

    Returns:
        New config dict (does not mutate base_config).
    """
    new_config = copy.deepcopy(base_config)

    for key, delta_value in delta.items():
        if key not in new_config:
            # New key — just set it (e.g., prompt_template if missing)
            new_config[key] = delta_value
            continue

        current = new_config[key]

        if isinstance(delta_value, str):
            # String replacement (e.g., prompt_template: "v2")
            new_config[key] = delta_value
        elif isinstance(current, (int, float)):
            # Numeric addition + clamping
            raw = current + delta_value
            if key in _BOUNDS:
                lo, hi = _BOUNDS[key]
                raw = max(lo, min(hi, raw))
            # Preserve int type if original was int
            new_config[key] = int(raw) if isinstance(current, int) else raw
        else:
            # Fallback: just overwrite
            new_config[key] = delta_value

    return new_config


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LangGraph Node
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@_mlflow_trace(name="improver", span_type="improvement")
def improver_node(state: RunState) -> dict:
    """LangGraph node: propose a config fix based on the diagnosed failure type.

    Reads from state:
      - failure_type          (F-01 through F-05, set by diagnoser)
      - config                (current pipeline config)
      - improvement_attempt   (which retry iteration we're on, 0-indexed)
      - improver_candidates   (history of previous attempts, may be empty)

    Writes to state:
      - config                (UPDATED config for next pipeline run)
      - improver_candidates   (APPENDED with this attempt's candidate)
      - improvement_attempt   (incremented by 1)
    """
    failure_type = state.get("failure_type", "F-04")  # default to catch-all
    current_config = state.get("config", copy.deepcopy(DEFAULT_CONFIG))
    attempt = state.get("improvement_attempt", 0)
    history = list(state.get("improver_candidates", []))  # copy to avoid mutation

    # Look up playbook for this failure type
    plays = _PLAYBOOK.get(failure_type, _PLAYBOOK["F-04"])  # fallback to F-04

    # Pick the play for this attempt (wrap around if somehow > len)
    play = plays[attempt % len(plays)]

    # Apply the delta to produce a new config
    new_config = _apply_delta(current_config, play["delta"])

    # Build candidate record for history
    candidate = {
        "variant_id": f"{failure_type}-v{attempt}",
        "failure_type": failure_type,
        "attempt": attempt,
        "delta": play["delta"],
        "rationale": play["rationale"],
        "config_before": current_config,
        "config_after": new_config,
    }

    history.append(candidate)

    return {
        "config": new_config,
        "improver_candidates": history,
        "improvement_attempt": attempt + 1,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Standalone test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import json

    print("=" * 70)
    print("IMPROVER — Standalone Tests")
    print("=" * 70)

    base = copy.deepcopy(DEFAULT_CONFIG)

    # ── Test 1: F-01 across 3 attempts ───────────────────────────────────
    print("\n── Test 1: F-01 (Retrieval Miss) × 3 attempts ──")
    state = {"failure_type": "F-01", "config": copy.deepcopy(base),
             "improvement_attempt": 0, "improver_candidates": []}

    for i in range(3):
        result = improver_node(state)
        cand = result["improver_candidates"][-1]
        print(f"  Attempt {i}: {cand['variant_id']}")
        print(f"    Delta:     {cand['delta']}")
        print(f"    Rationale: {cand['rationale']}")
        print(f"    k: {cand['config_before']['retrieval_k']} → {cand['config_after']['retrieval_k']}")
        # Feed result back into state for next iteration
        state["config"] = result["config"]
        state["improvement_attempt"] = result["improvement_attempt"]
        state["improver_candidates"] = result["improver_candidates"]

    # ── Test 2: F-03 (Hallucination) — prompt switch ────────────────────
    print("\n── Test 2: F-03 (Hallucination) — prompt switch ──")
    state = {"failure_type": "F-03", "config": copy.deepcopy(base),
             "improvement_attempt": 0, "improver_candidates": []}
    result = improver_node(state)
    cand = result["improver_candidates"][-1]
    print(f"  Prompt: {cand['config_before']['prompt_template']} → {cand['config_after']['prompt_template']}")
    print(f"  Rationale: {cand['rationale']}")

    # ── Test 3: Clamping — chunk_size can't go below 64 ─────────────────
    print("\n── Test 3: Clamping bounds ──")
    tiny_config = copy.deepcopy(base)
    tiny_config["chunk_size"] = 80   # close to lower bound
    tiny_config["retrieval_k"] = 2   # close to lower bound
    new = _apply_delta(tiny_config, {"chunk_size": -128, "retrieval_k": -5})
    print(f"  chunk_size: 80 + (-128) → {new['chunk_size']} (clamped to 64)")
    print(f"  retrieval_k: 2 + (-5) → {new['retrieval_k']} (clamped to 1)")
    assert new["chunk_size"] == 64, f"Expected 64, got {new['chunk_size']}"
    assert new["retrieval_k"] == 1, f"Expected 1, got {new['retrieval_k']}"

    # ── Test 4: Failure type shift across iterations ─────────────────────
    print("\n── Test 4: Failure type shift (F-01 → F-03) ──")
    state = {"failure_type": "F-01", "config": copy.deepcopy(base),
             "improvement_attempt": 0, "improver_candidates": []}
    # First iteration: diagnoser says F-01
    result = improver_node(state)
    print(f"  Iter 0: {result['improver_candidates'][-1]['variant_id']} (F-01 fix)")
    # Second iteration: diagnoser now says F-03 (failure type shifted!)
    state["config"] = result["config"]
    state["improvement_attempt"] = result["improvement_attempt"]
    state["improver_candidates"] = result["improver_candidates"]
    state["failure_type"] = "F-03"  # shifted!
    result = improver_node(state)
    cand = result["improver_candidates"][-1]
    print(f"  Iter 1: {cand['variant_id']} (F-03 fix — adapted to new failure!)")
    print(f"    Prompt: {cand['config_after'].get('prompt_template', '?')}")

    # ── Test 5: History accumulates ──────────────────────────────────────
    print(f"\n── Test 5: History length = {len(result['improver_candidates'])} (expected 2) ──")
    assert len(result["improver_candidates"]) == 2

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
