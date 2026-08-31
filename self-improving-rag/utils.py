"""Shared utility functions — single sources of truth.

Contains functions that multiple modules need but that would create
circular imports if placed in any single module:
  - build_collection_name(): deterministic Chroma collection naming
  - compute_gate_decision(): deployment gate routing logic

These are pure functions with no heavy dependencies (only config constants).
"""

from config import UNIFIED_TARGET, HITL_LOW, FAITHFULNESS_FLOOR


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Collection Naming — single source of truth
#
# Every place that needs a collection name MUST call this function.
# Format: rag_{version}_{strategy}_{chunk_size}
# This ensures version isolation (immutable vN collections).
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_collection_name(config: dict, version: str = "v1") -> str:
    """Build a deterministic Chroma collection name from config + version.

    This is the ONE place collection names are constructed. Every consumer
    (builder, frontend, optimizer, ingest) calls this instead of building
    the string ad-hoc.

    Each trial/version gets its own collection, never mutating a
    previously-measured version — this is how honest before/after
    comparisons work in the optimizer loop.

    chunk_overlap is included so a config change that only touches overlap
    still produces a new collection name — otherwise ingestion would be
    silently skipped (collection already has chunks) and the new overlap
    would never actually take effect.

    Args:
        config: Pipeline config dict with chunk_strategy, chunk_size, chunk_overlap.
        version: Version string (e.g. "v1", "v2").

    Returns:
        Collection name like "rag_v1_fixed_size_256_o0".
    """
    strategy = config.get("chunk_strategy", "fixed_size")
    chunk_size = config.get("chunk_size", 256)
    chunk_overlap = config.get("chunk_overlap", 0)
    return f"rag_{version}_{strategy}_{chunk_size}_o{chunk_overlap}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Gate Decision — single source of truth
#
# The deployment gate logic that routes after evaluation:
#   deploy_eligible, hitl_required, or hard_block.
# Both pipeline.py and agents/evaluator.py call this instead of
# duplicating the threshold comparisons.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_gate_decision(
    unified_score: float,
    faithfulness: float | None,
) -> tuple[str, str]:
    """Compute the deployment gate decision from scores.

    This is the ONE place gate logic lives. Every consumer (pipeline.py,
    evaluator_node) calls this instead of duplicating threshold comparisons.

    Order matters — safety checks fire FIRST:
      1. Faithfulness veto (< FAITHFULNESS_FLOOR) → hard_block
      2. Score >= UNIFIED_TARGET → deploy_eligible
      3. Score in [HITL_LOW, UNIFIED_TARGET) → hitl_required
      4. Score < HITL_LOW → hard_block

    Args:
        unified_score: The computed unified score.
        faithfulness: Faithfulness score (None if judge failed).

    Returns:
        Tuple of (gate_decision, gate_reason).
    """
    if faithfulness is not None and faithfulness < FAITHFULNESS_FLOOR:
        return (
            "hard_block",
            f"Faithfulness veto: {faithfulness:.2f} < {FAITHFULNESS_FLOOR}",
        )
    if unified_score >= UNIFIED_TARGET:
        return (
            "deploy_eligible",
            f"Score {unified_score:.4f} >= {UNIFIED_TARGET} target",
        )
    if unified_score >= HITL_LOW:
        return (
            "hitl_required",
            f"Score {unified_score:.4f} in gray band [{HITL_LOW}, {UNIFIED_TARGET})",
        )
    return (
        "hard_block",
        f"Score {unified_score:.4f} < {HITL_LOW} minimum",
    )
