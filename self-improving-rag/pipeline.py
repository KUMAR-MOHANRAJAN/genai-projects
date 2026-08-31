"""RAG pipeline — single-query execution with inline evaluation.

Runs the full chain: retrieval → context assembly → generation → evaluation → scoring.

This file serves two roles:
  1. TODAY: the only way to run the pipeline (no agents/ layer yet).
  2. FUTURE: the "plain Python" pipeline for quick single-query runs, CLI usage,
     and Streamlit UI. The LangGraph graph (agents/graph.py) will be the primary
     pipeline for the optimizer loop, with proper node separation and conditional
     routing.

Architecture note — what lives where:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ THIS FILE (pipeline.py)                                                │
  │   run_pipeline()        — sequential orchestrator (no graph)           │
  │   _judge_*()            — LLM-judge functions (→ agents/evaluator.py)  │
  │   precision_at_k() etc  — retrieval metrics  (→ agents/evaluator.py)   │
  │   _compute_*_score()    — score formulas     (→ agents/evaluator.py)   │
  │                                                                        │
  │ When agents/ is built, the evaluation logic (~lines 31-241) migrates   │
  │ to agents/evaluator.py. This file keeps run_pipeline() as the simple   │
  │ non-graph entry point.                                                 │
  └─────────────────────────────────────────────────────────────────────────┘

Evaluation design:
  - We NEVER skip evaluation based on retrieval quality. The diagnoser needs
    the full picture to classify failures (F-01 vs F-03).
  - Unified score formula v1.2:
    0.25×Recall + 0.35×Quality + 0.25×Faithfulness − latency_pen − cost_pen
  - Cost = generation LLM call only (judge calls are eval overhead, not pipeline cost).
  - Latency = full pipeline wall-clock (retrieval + generation + judge calls).
"""

import json
from openai import OpenAI
from config import (
    DEFAULT_CONFIG, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, JUDGE_MODEL,
    PRICE_INPUT_PER_M, PRICE_OUTPUT_PER_M,
    UNIFIED_TARGET, HITL_LOW, FAITHFULNESS_FLOOR,
    validate_config,
)
from state import RunState, initial_state
from retrieval import search
from generation import assemble_context, generate
from ground_truth import TEST_QUERIES
from agents.llm_utils import judge_call as _judge_call_via_utils
from utils import build_collection_name, compute_gate_decision

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM-Judge Evaluation (→ migrates to agents/evaluator.py)
#
# Three separate LLM calls scoring different aspects of the answer.
# Hand-rolled mini-RAGAS: same recipes as the RAGAS library but explicit
# (~40 lines each) so the mechanics are transparent for learning.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_FAITHFULNESS_PROMPT = """You are an evaluation judge. Analyze whether the ANSWER is faithful to the CONTEXT.

Step 1: Break the ANSWER into individual atomic claims (one fact per claim).
Step 2: For EACH claim, determine if it is verifiable from the CONTEXT.
Step 3: Return faithfulness = supported_claims / total_claims.

Respond ONLY with valid JSON:
{{"claims": ["claim 1", "claim 2", ...], "supported": [true, false, ...], "faithfulness": 0.85}}

CONTEXT:
{context}

ANSWER:
{answer}
"""

_RELEVANCE_PROMPT = """You are an evaluation judge. Does the ANSWER address the QUESTION?

Score 1.0 if the answer directly addresses the question.
Score 0.0 if the answer is about a different topic.
Score in between for partial relevance.

Respond ONLY with valid JSON:
{{"score": 0.85, "reasoning": "one brief sentence"}}

QUESTION:
{question}

ANSWER:
{answer}
"""

_CORRECTNESS_PROMPT = """You are an evaluation judge. Compare the ANSWER to the EXPECTED_ANSWER.

Score 1.0 if the answer conveys the same information as the expected answer.
Score 0.0 if the answer contradicts or misses the key facts.
Score in between for partial correctness.

Respond ONLY with valid JSON:
{{"score": 0.85, "reasoning": "one brief sentence"}}

QUESTION:
{question}

EXPECTED_ANSWER:
{expected_answer}

ANSWER:
{answer}
"""


def _llm_judge(prompt: str) -> dict:
    """Call the judge LLM and parse JSON response. Returns empty dict on failure.

    Uses agents/llm_utils.judge_call() for provider failover and proper
    error handling. This replaces the bare OpenAI call that had no retry
    or error classification.
    """
    return _judge_call_via_utils(prompt, agent_name="judge")


def _judge_faithfulness(answer: str, context: str) -> tuple[float | None, str | None, dict]:
    """LLM-judge: break answer into claims, verify each against context.

    Returns (score, reasoning, detail) tuple. Score is None if judge fails
    (never fabricates). detail carries the raw claims/supported breakdown.
    """
    prompt = _FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    result = _llm_judge(prompt)
    score = result.get("faithfulness")
    # Build reasoning from claims analysis
    claims = result.get("claims", [])
    supported = result.get("supported", [])
    reasoning = None
    if claims and supported:
        reasoning = f"{sum(supported)}/{len(claims)} claims supported"
    detail = {"claims": claims, "supported": supported}
    if score is not None:
        return float(score), reasoning, detail
    return None, None, detail  # judge failure — caller must handle missing metric


def _judge_relevance(answer: str, query: str) -> tuple[float | None, str | None, dict]:
    """LLM-judge: does the answer address the question?

    Returns (score, reasoning, detail) tuple. Score is None if judge fails (never fabricates).
    """
    prompt = _RELEVANCE_PROMPT.format(question=query, answer=answer)
    result = _llm_judge(prompt)
    score = result.get("score")
    reasoning = result.get("reasoning")
    detail = {"reasoning": reasoning}
    if score is not None:
        return float(score), reasoning, detail
    return None, None, detail


def _judge_correctness(answer: str, query: str, expected_answer: str) -> tuple[float | None, str | None, dict]:
    """LLM-judge: compare answer to ground truth. Returns (score, reasoning, detail) tuple."""
    prompt = _CORRECTNESS_PROMPT.format(
        question=query, expected_answer=expected_answer, answer=answer
    )
    result = _llm_judge(prompt)
    score = result.get("score")
    reasoning = result.get("reasoning")
    detail = {"reasoning": reasoning, "expected_answer": expected_answer}
    if score is not None:
        return float(score), reasoning, detail
    return None, None, detail


_RETRIEVAL_RELEVANCE_PROMPT = """You are an evaluation judge. Determine whether ANY of the RETRIEVED CHUNKS below contain information relevant to answering the QUESTION.

Respond ONLY with valid JSON:
{{"relevant": true, "relevance_score": 0.8, "reasoning": "one brief sentence"}}

QUESTION:
{question}

RETRIEVED CHUNKS:
{chunks_text}
"""


def _judge_retrieval_relevance(query: str, chunks: list[dict]) -> tuple[bool | None, float | None, str | None]:
    """LLM-judge: is any retrieved chunk actually relevant to the query?

    NOT part of the standard evaluation — this is a targeted fallback used
    only by the diagnoser when keyword-based retrieval_score is unavailable
    (ad-hoc query with no golden-set ground truth) and the answer is an
    abstention. Scoped narrowly so it never adds cost to the golden-set/
    optimizer path, which already has free keyword-based retrieval scoring.

    Returns (relevant, relevance_score, reasoning). All None if the judge
    fails (never fabricates a verdict).
    """
    chunks_text = "\n\n".join(
        f"[Chunk {i + 1}] {c.get('text', '')[:500]}" for i, c in enumerate(chunks)
    )
    prompt = _RETRIEVAL_RELEVANCE_PROMPT.format(question=query, chunks_text=chunks_text)
    result = _llm_judge(prompt)
    relevant = result.get("relevant")
    score = result.get("relevance_score")
    reasoning = result.get("reasoning")
    if relevant is not None:
        return bool(relevant), (float(score) if score is not None else None), reasoning
    return None, None, None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Retrieval Metrics (→ migrates to agents/evaluator.py)
#
# Keyword-based, NO LLM call. Cheap and deterministic.
# Uses ground truth keywords rather than chunk IDs because chunk IDs change
# with every config version (different chunking = different chunks).
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _find_keywords(query: str) -> list[str] | None:
    """Find ground truth keywords for a query. Returns None if not found."""
    for q, _, kws in TEST_QUERIES:
        if q.lower() == query.lower():
            return kws
    return None


def precision_at_k(chunks: list[dict], keywords: list[str], k: int = 5) -> float:
    """Fraction of top-k retrieved chunks that contain at least one keyword.

    Precision@k = relevant chunks in top-k / k
    A chunk is "relevant" if its text contains at least one ground truth keyword.
    """
    if not chunks or not keywords:
        return 0.0
    top_k = chunks[:k]
    relevant = sum(
        1 for c in top_k
        if any(kw.lower() in c.get("text", "").lower() for kw in keywords)
    )
    return relevant / len(top_k)


def recall_at_k(chunks: list[dict], keywords: list[str], k: int = 5) -> float:
    """Fraction of keywords found in the top-k retrieved chunks.

    Recall@k = unique keywords found in top-k / total keywords
    Approximates "did we find all the relevant stuff?" without ground truth IDs.
    """
    if not chunks or not keywords:
        return 0.0
    top_k = chunks[:k]
    found_keywords = set()
    for c in top_k:
        text = c.get("text", "").lower()
        for kw in keywords:
            if kw.lower() in text:
                found_keywords.add(kw.lower())
    return len(found_keywords) / len(keywords)


def _compute_retrieval_score(
    chunks: list[dict],
    keywords: list[str] | None,
    k: int = 5,
) -> float:
    """Retrieval sub-score: 0.5 × precision@k + 0.5 × recall@k.

    Feeds the Unified Score "R" term.
    NOTE: These are classical IR metrics (keyword-based, no LLM), NOT the RAGAS
    context_precision / context_recall metrics (which are LLM-judged, Layer 2 only).

    When keywords exist (ground truth query): uses precision@k + recall@k.
    When no keywords (ad-hoc query): falls back to avg chunk similarity.
    """
    if not chunks:
        return 0.0

    if keywords:
        precision = precision_at_k(chunks, keywords, k=k)
        recall = recall_at_k(chunks, keywords, k=k)
        return 0.5 * precision + 0.5 * recall

    # No ground truth keywords — cannot compute meaningful retrieval score.
    # Production systems use LLM-judged RAGAS context_precision/recall here.
    # We skip that to save LLM cost and return None instead of a misleading
    # embedding-similarity average.
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Score Computation (→ migrates to agents/evaluator.py)
#
# Sub-scores (quality, retrieval) and the unified score formula.
# The unified score is the single number that drives all routing decisions:
#   >= 0.85 → accept,  0.70-0.84 → HITL review,  < 0.70 → diagnose/improve
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _compute_quality_score(
    answer_relevancy: float | None,
    answer_correctness: float | None,
) -> float | None:
    """Quality sub-score.

    Quality = 0.6 × AnswerCorrectness + 0.4 × AnswerRelevancy
    Falls back to AnswerRelevancy alone if no ground truth.
    Returns None if relevancy is missing (cannot compute quality without it).
    """
    if answer_relevancy is None:
        return None
    if answer_correctness is not None:
        return 0.6 * answer_correctness + 0.4 * answer_relevancy
    return answer_relevancy


def _compute_unified_score(
    recall: float | None,
    quality: float | None,
    faithfulness: float | None,
    latency_ms: int,
    cost_usd: float,
) -> float:
    """AutoRAG's unified score formula v1.2, with graceful None handling.

    Full formula (all metrics available):
      Score = 0.25*Recall + 0.35*Quality + 0.25*Faithfulness
              - 0.10*min(latency/3000, 1)
              - 0.05*min(cost/MAX_QUERY_COST, 1)

    When a metric is None (no ground truth or judge failure), that term is
    dropped and remaining positive weights are re-normalized to sum to 0.85.
    This avoids fabricating scores -- the unified score simply becomes
    less confident (based on fewer signals) rather than wrong.

    MAX_QUERY_COST = $0.01 (generous cap for this project)
    """
    MAX_QUERY_COST = 0.01
    latency_penalty = 0.10 * min(latency_ms / 3000.0, 1.0)
    cost_penalty = 0.05 * min(cost_usd / MAX_QUERY_COST, 1.0)

    # Build weighted terms from available metrics
    terms = []
    if recall is not None:
        terms.append((0.25, recall))
    if quality is not None:
        terms.append((0.35, quality))
    if faithfulness is not None:
        terms.append((0.25, faithfulness))

    # Re-normalize weights so positive terms sum to 0.85 (preserving penalty budget)
    raw_weight_sum = sum(w for w, _ in terms)
    target_weight_sum = 0.85  # total positive budget (1.0 - 0.10 - 0.05)
    scale = target_weight_sum / raw_weight_sum if raw_weight_sum > 0 else 1.0

    score = sum(w * scale * v for w, v in terms) - latency_penalty - cost_penalty
    return round(max(0.0, min(1.0, score)), 4)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pipeline Orchestrator (stays in this file)
#
# Sequential execution: retrieve → metrics → assemble → generate → evaluate → score.
# This is the "plain Python" pipeline — no LangGraph, no conditional routing.
# When agents/graph.py is built, the LangGraph version adds:
#   - Conditional edges (score-based routing to diagnoser/improver)
#   - HITL interrupt() for the gray band
#   - Retry loop (max 3)
# But this function remains useful for CLI, Streamlit UI, and testing.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_pipeline(
    query: str,
    config: dict | None = None,
    version: str = "v1",
) -> RunState:
    """Run the full RAG pipeline with LLM-judge evaluation.

    Flow: retrieval → retrieval metrics → context assembly → generation → LLM-judge scoring

    Args:
        query: The user's question.
        config: Pipeline config dict. Defaults to DEFAULT_CONFIG.
        version: Version string for collection naming ("v1", "v2", ...).

    Returns:
        RunState with answer, LLM-judge scores, retrieval metrics, costs, and metadata.
    """
    cfg = validate_config(config or DEFAULT_CONFIG)
    state = initial_state(query=query, config=cfg, version=version)

    # Use explicit collection_name from config if provided, else build from params
    collection_name = cfg.get("collection_name") or build_collection_name(cfg, version)
    state["collection_name"] = collection_name

    print(f"\n{'=' * 70}")
    print(f"PIPELINE — run_pipeline()")
    print(f"  Query:      {query[:80]}")
    print(f"  Collection: {collection_name}")
    print(f"  Config:     k={cfg.get('retrieval_k')}, chunk={cfg.get('chunk_size')}, "
          f"overlap={cfg.get('chunk_overlap', 0)}, "
          f"max_ctx={cfg.get('max_context_tokens', 4000)}")
    print(f"{'=' * 70}")

    # ── Step 1: Retrieval ─────────────────────────────────────────────────
    k = cfg.get("retrieval_k", 5)
    print(f"\n  [1/8] Retrieval: searching '{collection_name}' for top-{k} chunks...")
    chunks = search(collection_name, query, k=k)
    state["retrieved_chunks"] = chunks
    state["chunk_count"] = len(chunks)
    print(f"         Retrieved {len(chunks)} chunks")

    # ── Step 2: Retrieval Metrics (keyword-based, NO LLM call) ────────────
    # Computed FIRST — cheap, no LLM. But we NEVER skip evaluation based on
    # these results. AutoRAG computes all metrics regardless because the
    # diagnoser needs the full picture to classify failures.
    print(f"  [2/8] Retrieval metrics (keyword, no LLM)...")
    keywords = _find_keywords(query)
    precision = precision_at_k(chunks, keywords, k=k) if keywords else 0.0
    recall_kw = recall_at_k(chunks, keywords, k=k) if keywords else 0.0
    retrieval_score = _compute_retrieval_score(chunks, keywords, k=k)
    if retrieval_score is not None:
        print(f"         precision={precision:.2f}, recall={recall_kw:.2f}, "
              f"retrieval_score={retrieval_score:.2f}")
    else:
        print(f"         No ground truth keywords — retrieval score: N/A")

    state["retrieval_score"] = retrieval_score
    state["retrieval_precision"] = precision
    state["retrieval_recall"] = recall_kw

    # ── Step 3: Context Assembly ──────────────────────────────────────────
    max_tokens = cfg.get("max_context_tokens", 4000)
    print(f"  [3/8] Context assembly (budget: {max_tokens} tokens)...")
    context, context_tokens = assemble_context(chunks, max_tokens=max_tokens)
    state["context"] = context
    state["context_tokens"] = context_tokens
    print(f"         Assembled {context_tokens} tokens")

    # ── Step 4: Generation ────────────────────────────────────────────────
    prompt_version = cfg.get("prompt_template", "v1")
    print(f"  [4/8] Generation (prompt={prompt_version})...")
    gen_result = generate(context, query, prompt_version=prompt_version)
    state["answer"] = gen_result["answer"]
    state["generation_cost_usd"] = gen_result["cost_usd"]
    state["generation_latency_ms"] = gen_result["latency_ms"]
    print(f"         Done. latency={gen_result['latency_ms']}ms, "
          f"cost=${gen_result['cost_usd']:.4f}")
    print(f"         Answer: {gen_result['answer'][:100]}...")

    # ── Step 5: LLM-Judge Evaluation (mirrors AutoRAG's evaluator_node) ───
    # Always computed, even when retrieval is garbage. The diagnoser needs
    # all metrics to distinguish F-01 (bad retrieval) from F-03 (hallucination).
    answer = gen_result["answer"]
    judge_reasoning = {}
    judge_details = {}

    print(f"  [5/8] Judge: faithfulness...")
    faithfulness, faith_reasoning, faith_detail = _judge_faithfulness(answer, context)
    if faith_reasoning:
        judge_reasoning["faithfulness"] = faith_reasoning
    judge_details["faithfulness"] = faith_detail
    print(f"         faithfulness={faithfulness}")

    print(f"  [6/8] Judge: relevance...")
    relevance, rel_reasoning, rel_detail = _judge_relevance(answer, query)
    if rel_reasoning:
        judge_reasoning["relevance"] = rel_reasoning
    judge_details["relevance"] = rel_detail
    print(f"         relevance={relevance}")

    expected_answer = None
    for q, ea, _ in TEST_QUERIES:
        if q.lower() == query.lower():
            expected_answer = ea
            break
    correctness = None
    if expected_answer:
        print(f"  [7/8] Judge: correctness (ground truth found)...")
        correctness, corr_reasoning, corr_detail = _judge_correctness(answer, query, expected_answer)
        if corr_reasoning:
            judge_reasoning["correctness"] = corr_reasoning
        judge_details["correctness"] = corr_detail
        print(f"         correctness={correctness}")
    else:
        print(f"  [7/8] Judge: correctness — skipped (no ground truth)")

    # ── Step 6: Compute sub-scores (AutoRAG's exact recipe) ───────────────
    quality = _compute_quality_score(relevance, correctness)

    # ── Step 7: Unified Score (AutoRAG formula v1.2) ──────────────────────
    # Latency = generation LLM call only (not judge overhead).
    # In production, user gets the answer after generation; judges run async.
    # The latency penalty reflects what the user experiences.
    latency_ms = gen_result["latency_ms"]
    cost_usd = gen_result["cost_usd"]

    unified_score = _compute_unified_score(
        recall=retrieval_score,
        quality=quality,
        faithfulness=faithfulness,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )

    state["unified_score"] = unified_score
    state["faithfulness"] = faithfulness
    state["relevance"] = relevance
    state["correctness"] = correctness
    state["cost_usd"] = cost_usd
    state["latency_ms"] = latency_ms

    # ── Step 8: Gate Decision (single source of truth) ─────────────────────
    # Uses compute_gate_decision() — one place for all threshold logic.
    gate_decision, gate_reason = compute_gate_decision(unified_score, faithfulness)

    state["gate_decision"] = gate_decision
    state["gate_reason"] = gate_reason
    state["judge_reasoning"] = judge_reasoning
    state["judge_details"] = judge_details

    print(f"\n  [8/8] Gate decision: {gate_decision}")
    print(f"         {gate_reason}")
    print(f"         unified={unified_score:.4f}, faith={faithfulness}, "
          f"relev={relevance}, correct={correctness}")
    print(f"{'=' * 70}\n")

    return state
