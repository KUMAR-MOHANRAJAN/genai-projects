# Top 8 Recommendations for Demo Readiness

> Prioritized by ROI for a half-day internal interview/demo.
> Goal: demonstrate production-grade understanding of the AutoRAG self-improvement loop.

---

## 1. Learning Map Dashboard Tab (Highest ROI — 30 min)

**What:** Add a 4th Streamlit tab showing the component-to-AutoRAG origin mapping table.

**Why:** This is the interview killer. It proves you understand the parent architecture and can trace every design decision back to its source. Reviewers immediately see the lineage.

**How:** Create a new tab in `frontend/app.py` with a table mirroring §8 of the project plan. Each row links the child file to the parent file and explains what was inherited, simplified, or excluded.

**Effort:** ~30 min

---

## 2. Execution Trace Visualizer (High ROI — 30 min)

**What:** Render `execution_trace` as a timeline/table in the Streamlit UI showing each node's latency, status, tokens, and cost.

**Why:** The trace infrastructure already exists (`agents/trace.py`) but isn't surfaced. Visualizing it screams "production-grade observability" without needing MLflow. Shows you built real monitoring, not just a pipeline.

**How:** Add a collapsible section in Tab 1 (Pipeline) that displays trace events as a dataframe or timeline chart. Each row shows: node name, timestamp, status, latency_ms, tokens, cost_usd, input_summary, output_summary.

**Effort:** ~30 min

---

## 3. Before vs After Comparison View (High ROI — 30 min)

**What:** Side-by-side comparison of baseline vs final config with config diff, score deltas, and chunk comparison.

**Why:** Parent has this pattern (`_diff_config()` + `build_improvement_result()`). It's the most demo-friendly way to show the loop actually improved something.

**How:** Add to Tab 2 (Optimizer) — after optimization completes, render a two-column layout: left=baseline, right=final, with a diff table in between showing which knobs changed and by how much.

**Effort:** ~30 min

---

## 4. Improver 3-Candidate Generation (Medium ROI — 45 min)

**What:** Generate all 3 variants at once (like parent), flag `is_winner`, add `_SCORE_DELTA_ESTIMATES` and `_KNOWN_TRADEOFFS`.

**Why:** Matches the parent's "propose vs verify" split exactly. Shows you understand why the parent generates 3 candidates (auditability, trade-off transparency) rather than 1 sequential attempt.

**How:** Refactor `agents/improver.py` to generate all 3 candidates in one node invocation, select winner by estimated score, persist all 3 to `state["improver_candidates"]` with `is_winner` flag. Add `_SCORE_DELTA_ESTIMATES` dict and `_KNOWN_TRADEOFFS` dict from parent's pattern.

**Effort:** ~45 min

---

## 5. F-06 Quality Drift Detection (Medium ROI — 30 min)

**What:** Implement the Observer-style drift rule: check last 5 runs from `runs_history.jsonl`, flag if ≥2 declining pairs AND cumulative drop ≥ 0.05.

**Why:** You already have `DRIFT_WINDOW=5` and `DRIFT_MIN_DROP=0.05` in config. Completes the 6-failure-type taxonomy. Shows trend-vs-acute detection pattern (different from single-run failures).

**How:** Add a `check_quality_drift()` function in `agents/diagnoser.py` that reads recent history and returns F-06 if conditions met. Insert into the rule cascade after F-02 (matching parent's position).

**Effort:** ~30 min

---

## 6. Score Provenance Dict (Medium ROI — 20 min)

**What:** Store `{formula_version, weights, raw_inputs}` alongside every unified score.

**Why:** Parent's `build_score_provenance()` pattern. Shows auditability — any score can be re-derived later. Important for compliance/debugging in production.

**How:** Add a `_build_score_provenance()` function in `pipeline.py` that returns a dict with formula version, weight breakdown, and raw metric values. Include its output in the evaluator's state update as `score_provenance`.

**Effort:** ~20 min

---

## 7. Score Trend Chart in History (Low ROI — 20 min)

**What:** Line chart of unified scores over time in Tab 3 (History).

**Why:** Visual polish that makes the history tab feel like a real monitoring dashboard. Uses existing JSONL data — no new infrastructure needed.

**How:** Use `st.line_chart()` or matplotlib to plot `unified_score` vs `timestamp` from `load_history()`. Separate lines for query runs vs optimization runs.

**Effort:** ~20 min

---

## 8. Diagnoser Priority Order Alignment (Low ROI — 15 min)

**What:** Align child's priority order with parent: F-03 → F-05 → F-01 → F-02 → F-04.

**Why:** Child currently checks abstention first, then F-03, F-01, F-02, F-05, F-04. Parent checks F-03, F-05, F-01, F-02, F-06. The child's abstention handling is thoughtful but diverges from the parent's spec. For the demo, matching shows direct lineage.

**How:** Reorder the `_classify_failure()` cascade in `agents/diagnoser.py`. Keep abstention as a sub-check within F-01/F-04 rather than a top-level priority. Move F-05 (latency) before F-01 (retrieval miss) to match parent.

**Effort:** ~15 min

---

## What NOT to Add (and Why)

| Item | Why Excluded |
|---|---|
| **MLflow** | Overkill for a half-day demo. Requires server setup, environment config. Your JSONL + trace already covers the learning goal. Parent's MLflow is Phase 3 material. |
| **Adversarial Testing** | Parent's 10-question hallucination benchmark requires a separate adversarial service, curated question sets, and pass-rate thresholds. Out of scope per the project plan (§1 "Explicitly EXCLUDED"). |
| **PII Masking / Guardrails** | Real production concerns but orthogonal to understanding evaluate→diagnose→improve→optimize. Parent's guardrails are in `app/guardrails/` — a separate subsystem. |
| **Reranking** | Parent uses cross-encoder reranking (`rerank_service.py`). Child's scope is retrieval + generation + evaluation loop. Adding reranking would require a new model, new service, and changes to every agent's input/output. |
| **Database Persistence** | PostgreSQL + SQLAlchemy adds schema management, migrations, connection pooling. JSONL is append-only, human-readable, and sufficient for demo. Parent uses Postgres because it's multi-tenant — child is single-user. |
| **Deployment (ECS/Fargate)** | Parent's deployer agent handles blue/green canary rollbacks via ECS. Child has no deployment target. Adding this would require AWS credentials, Docker images, Terraform — a full infra project. |
| **Prometheus/Grafana** | Parent's observability stack (`observability/` directory) requires running Prometheus + Grafana servers, writing PromQL queries, building dashboards. The `@traced_node` decorator + JSONL history already provides the same visibility at learning-project scale. |
| **Multi-tenancy** | Parent's `organization_id`, `user_id`, `session_id` fields exist for a SaaS platform. Child is a single-user learning project. Adding multi-tenancy would touch every file and add no demo value. |
| **Query Rewriter / Compressor** | Parent has 9 agents including query_rewriter (HyDE) and compressor (context compression). Child consolidates these into builder + pipeline nodes. Adding them would increase complexity without demonstrating the core loop better. |

---

## Recommended Execution Order (Total: ~3.5 hours)

| Order | Item | Effort | Cumulative |
|---|---|---|---|
| 1 | Learning Map Dashboard Tab | 30 min | 30 min |
| 2 | Execution Trace Visualizer | 30 min | 1 hr |
| 3 | Before vs After Comparison | 30 min | 1.5 hr |
| 4 | Improver 3-Candidate Generation | 45 min | 2.25 hr |
| 5 | F-06 Quality Drift Detection | 30 min | 2.75 hr |
| 6 | Score Provenance Dict | 20 min | 3 hr |
| 7 | Score Trend Chart | 20 min | 3.3 hr |
| 8 | Diagnoser Priority Alignment | 15 min | 3.5 hr |

If time is tighter than half a day, execute items 1-3 only (1.5 hr) — they deliver 80% of the demo impact.
