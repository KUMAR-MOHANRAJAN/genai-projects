# Comparison Analysis: Self-Improving RAG vs AutoRAG Architect

> This document traces every component of the learning project (`self-improving-rag`)
> back to its origin in the parent platform (`PROJECT-6-AutoRAG-Architect`).

---

## 1. Architecture Overview

| Dimension | AutoRAG (Parent) | Self-Improving RAG (Child) |
|---|---|---|
| **Graph nodes** | 9 (ingestion, query_rewriter, builder, compressor, evaluator, observer, diagnoser, improver, deployer) | 6 (builder, pipeline, evaluator, hitl, diagnoser, improver) |
| **Graph topology** | Linear with conditional routing; no in-graph loop (Phase E1 fix) | Linear with conditional routing; no in-graph loop |
| **Optimizer** | Service layer (`OptimizerService`), async, DB-backed, experiment_runs tracking | Service function (`run_optimization()`), sync, in-memory |
| **Persistence** | PostgreSQL (evaluation_runs, experiment_runs, pipeline_versions) | JSONL append-only (`data/runs_history.jsonl`) |
| **Deployment** | ECS Fargate, blue/green canary, ALB, rollback automation | Not in scope — CLI + Streamlit UI only |
| **Multi-tenancy** | organization_id, user_id, session_id, pipeline_id | Single-user learning environment |

---

## 2. Component-by-Component Mapping

### 2.1 Graph Architecture

| Aspect | Parent | Child | Match |
|---|---|---|---|
| **Framework** | LangGraph StateGraph | LangGraph StateGraph | Exact |
| **Linear design** | Yes — Phase E1 removed in-graph loop-back | Yes — improver always ends at END | Exact |
| **Conditional routing** | `_route_after_evaluator`, `_route_after_observer`, `_route_after_improver` | `route_after_eval`, `route_after_hitl` | Strong |
| **HITL** | `interrupt()` in `_hitl_approval_node` | `interrupt()` in `hitl_node` | Exact |
| **Checkpointer** | Database-backed | `MemorySaver()` | Simplified |

**Parent source:** `autorag-server/app/core/workflow.py:242` (`build_autorag_graph`)
**Child source:** `agents/graph.py:159` (`build_graph`)

### 2.2 State Management

| Aspect | Parent | Child | Match |
|---|---|---|---|
| **Type** | `GraphState` TypedDict (~403 lines) | `RunState` TypedDict (~84 lines) | Strong — same pattern |
| **Version field** | `graph_state_version` ("v1.0") | `version` ("v1", "v2") | Simplified |
| **Execution trace** | `execution_trace: list[dict]` | `execution_trace: Annotated[list[dict], operator.add]` | Strong |
| **Multi-tenancy fields** | `organization_id`, `user_id`, `session_id`, `run_id`, `x_trace_id` | None | Excluded (scope) |
| **Deployer fields** | `ecs_task_arn`, `blue_task_arn`, `green_task_arn`, `rollback_reason` | None | Excluded (scope) |

**Parent source:** `autorag-server/app/core/state.py`
**Child source:** `state.py`

### 2.3 Evaluator Agent

| Aspect | Parent | Child | Match |
|---|---|---|---|
| **LLM Judges** | Faithfulness, Relevance, Correctness (via RAGAS service) | Faithfulness, Relevance, Correctness (hand-rolled) | Strong |
| **Unified Score** | Formula v1.2: `0.25·R + 0.35·Q + 0.25·F - 0.10·L - 0.05·C` | Same formula with graceful None handling | Strong |
| **Faithfulness Veto** | `< 0.50 → guardrail_blocked=True` | `< 0.50 → hard_block` | Strong |
| **Gate thresholds** | From `slo.yaml` deployment_gate | Hardcoded constants (mirrors slo.yaml) | Strong |
| **Score provenance** | `build_score_provenance()` | Not yet implemented | Gap |
| **Adversarial gate** | 10-question hallucination benchmark | Not in scope | Excluded |
| **Context truncation fix** | `build_ragas_context_texts()` — splits compressed_context | Uses exact generation context directly | Strong (avoids the bug) |

**Parent source:** `autorag-server/app/graph/nodes/evaluator.py` (761 lines), `autorag-server/app/services/evaluation_service.py` (491 lines), `autorag-server/app/services/ragas_service.py` (872 lines)
**Child source:** `agents/evaluator.py` (144 lines), `pipeline.py` lines 119-351

### 2.4 Diagnoser Agent

| Aspect | Parent | Child | Match |
|---|---|---|---|
| **Failure types** | F-01 through F-06 | F-01 through F-05 | Gap (F-06 missing) |
| **Thresholds** | `_FAITHFULNESS_THRESHOLD=0.5`, `_LATENCY_THRESHOLD_MS=3000`, `_RETRIEVAL_SCORE_FLOOR=0.3`, `_TOKEN_BUDGET=3800` | `FAITHFULNESS_FLOOR=0.50`, `LATENCY_CAP_MS=3000`, `RETRIEVAL_SIM_FLOOR=0.30` | Strong |
| **Priority order** | F-03 → F-05 → F-01 → F-02 → F-06 → LLM fallback | Abstention → F-03 → F-01 → F-02 → F-05 → F-04 | Partial — child adds abstention handling |
| **LLM fallback** | `_CLASSIFICATION_PROMPT` when rules don't match | Not yet (rule cascade only) | Gap |
| **F-06 Quality Drift** | Fires on `observation_flags.quality_drift=True` | Scaffolding exists (`DRIFT_WINDOW`, `DRIFT_MIN_DROP`) but no implementation | Gap |

**Parent source:** `autorag-server/app/graph/nodes/diagnoser.py` (303 lines), `_rule_based_classify()` at line 166
**Child source:** `agents/diagnoser.py` (280 lines), `_classify_failure()` at line 129

### 2.5 Improver Agent

| Aspect | Parent | Child | Match |
|---|---|---|---|
| **Playbook** | `_VARIANT_PLAYBOOK` — 3 variants per failure type | `_PLAYBOOK` — 1 candidate per iteration | Partial |
| **Score estimates** | `_SCORE_DELTA_ESTIMATES` per failure type | Not implemented | Gap |
| **Trade-offs** | `_KNOWN_TRADEOFFS` — real operational trade-off text | Not implemented | Gap |
| **Rationale** | `_VARIANT_RATIONALE` — deterministic explanation per variant | `rationale` per play entry | Strong |
| **Candidate generation** | Generates all 3 at once, flags `is_winner` | Generates 1 per graph invocation, accumulates across retries | Partial |
| **Guardrails** | TokenGuardrail + CostGuardrail enforcement | Bounds clamping (`_BOUNDS`) | Simplified |
| **MLflow logging** | `_log_to_mlflow()` per variant | Not implemented | Excluded (scope) |
| **Delta application** | `_apply_delta()` — numeric sum (min 1), string replace | `_apply_delta()` — numeric sum + clamp, string replace | Strong |

**Parent source:** `autorag-server/app/graph/nodes/improver.py` (450 lines), `_VARIANT_PLAYBOOK` at line 58, `_SCORE_DELTA_ESTIMATES` at line 88
**Child source:** `agents/improver.py` (334 lines), `_PLAYBOOK` at line 73

### 2.6 Optimizer Loop

| Aspect | Parent | Child | Match |
|---|---|---|---|
| **Stop conditions** | blocked_faithfulness → target_reached → hitl_required → no_candidates → no_improvement (×3) → max_iterations | Same 6 conditions, same priority order | Strong |
| **Baseline tracking** | `baseline_run_id`, `baseline_version`, `build_improvement_result()` | `baseline_result` parameter, manual report | Strong |
| **Version isolation** | `format_pipeline_version()`, new collection per version | `version` string, new ChromaDB collection per config | Strong |
| **Plateau detection** | `consecutive_no_improvement >= 3`, delta < 0.01 | Same logic, same threshold | Exact |
| **DB persistence** | `experiment_runs`, `evaluation_runs` rows | JSONL append-only | Simplified |
| **Config diff** | `_diff_config()` — field-by-field before/after | Not implemented | Gap |

**Parent source:** `autorag-server/app/services/optimizer_service.py` (635 lines), `run_optimization_loop()` at line 302, stop conditions at lines 473-499
**Child source:** `agents/optimizer.py` (402 lines), `run_optimization()` at line 87

### 2.7 LLM Infrastructure

| Aspect | Parent | Child | Match |
|---|---|---|---|
| **Provider failover** | `LLMService.call()` with chain resolution | `llm_call()` with EURI → Google chain | Strong |
| **Judge decoupling** | ADR-012: separate judge provider/model | `JUDGE_MODEL` constant, same model as generation in v1 | Partial |
| **Cost accounting** | Centralized in `LLMService` | Centralized in `llm_utils.py` | Strong |
| **Retry/backoff** | Bounded retry (max_attempts=2) on unparseable JSON | 2 attempts, exponential backoff on rate limit | Strong |
| **Judge failure** | Returns None — caller treats as missing metric | Returns None — never fabricates 0.0 | Exact |

**Parent source:** `autorag-server/app/services/llm_service.py`, ADR-012 (`ai-component-configuration.md`)
**Child source:** `agents/llm_utils.py` (393 lines)

### 2.8 Ingestion & Retrieval

| Aspect | Parent | Child | Match |
|---|---|---|---|
| **Document source** | Supabase Storage | Local text files + PDF | Simplified |
| **Chunking** | Semantic, hierarchical, late-interaction | Fixed-size, recursive_split, semantic | Strong |
| **Vector store** | pgvector (PostgreSQL) | ChromaDB | Simplified |
| **Version isolation** | `pipeline_chunks` partitioned by version string (migration 0016) | `rag_{version}_{strategy}_{chunk_size}` collection naming | Strong — same concept |
| **Auto-ingest** | Ingestion node runs on every graph dispatch | Builder node auto-ingests if collection empty | Strong |
| **PII masking** | Every page masked before chunking | Not in scope | Excluded |

**Parent source:** `autorag-server/app/graph/nodes/ingestion.py` (509 lines), migration 0016
**Child source:** `agents/builder.py` (142 lines), `ingest.py` (191 lines), `vector_store.py` (50 lines)

### 2.9 Observability

| Aspect | Parent | Child | Match |
|---|---|---|---|
| **Execution trace** | Append-only `execution_trace` in GraphState | Append-only `execution_trace` via `@traced_node` decorator | Strong |
| **Trace fields** | node, timestamp, x_trace_id, latency_ms, status, cost, tokens | node, timestamp, latency_ms, status, tokens, cost, input/output summary | Strong |
| **MLflow** | Full experiment tracking, artifact storage, web UI | Not implemented | Excluded (scope) |
| **Prometheus/Grafana** | Metrics dashboard, alert rules, runbooks | Not in scope | Excluded |
| **Run history** | `evaluation_runs` + `experiment_runs` tables | `data/runs_history.jsonl` | Simplified |

**Parent source:** `observability/` directory, `autorag-server/app/services/mlflow_service.py`
**Child source:** `agents/trace.py` (117 lines), `run_history.py` (169 lines)

---

## 3. Thresholds Comparison

| Constant | Parent (slo.yaml / code) | Child (config.py) | Match |
|---|---|---|---|
| `UNIFIED_TARGET` | 0.85 | 0.85 | Exact |
| `HITL_LOW` | 0.70 | 0.70 | Exact |
| `FAITHFULNESS_FLOOR` | 0.50 | 0.50 | Exact |
| `RETRIEVAL_SIM_FLOOR` | 0.30 | 0.30 | Exact |
| `LATENCY_CAP_MS` | 3000 | 3000 | Exact |
| `MAX_RETRIES` | 3 | 3 | Exact |
| `MAX_ITERATIONS` | 3 | 3 | Exact |
| `NO_IMPROVEMENT_DELTA` | 0.01 | 0.01 | Exact |
| `DRIFT_WINDOW` | 5 | 5 | Exact |
| `DRIFT_MIN_DROP` | 0.05 | 0.05 | Exact |

---

## 4. Design Principles Inherited

| Principle | Parent Evidence | Child Implementation |
|---|---|---|
| **Propose vs verify split** | Phase E1: removed in-graph improver→ingestion edge | Improver always ends at END; optimizer re-invokes graph |
| **Immutable versions** | `pipeline_chunks` partitioned by version; `create_version()` per trial | `rag_v{N}_{strategy}_{size}` — new collection per config |
| **Honest accounting** | Negative score_delta persisted, never clamped (optimizer_service.py:562-565) | Score deltas recorded as-is in optimizer report |
| **Veto above score** | `guardrail_blocked=True` regardless of unified_score (evaluator.py:569) | `hard_block` when faithfulness < 0.50, regardless of score |
| **Bounded loops** | max_retries=3, max_iterations=3, plateau detection | Same bounds, same plateau detection |
| **Judge context = generator context** | ADR-012 P0 fix: `build_ragas_context_texts()` splits compressed_context | Child uses exact context that generation saw (no compressor layer) |

---

## 5. Summary

The child project successfully captures the **core self-improvement loop mechanics** from AutoRAG:

- Linear graph with conditional routing (same Phase E1 fix)
- Evaluator with 3 LLM judges + unified score v1.2 + faithfulness veto
- Diagnoser with deterministic rule cascade (5 of 6 failure types)
- Improver with playbook-based config deltas
- Optimizer with 6 stop conditions in safety-first priority order
- HITL via LangGraph `interrupt()`
- Append-only execution trace
- Versioned immutable collections
- Provider failover for LLM calls
- Pydantic config validation

Key simplifications are intentional and align with the project's learning scope:

- No multi-tenancy, deployment, or observability infrastructure
- JSONL persistence instead of PostgreSQL
- Single candidate per iteration instead of 3-variant batch
- ChromaDB instead of pgvector
- Local files instead of Supabase Storage
