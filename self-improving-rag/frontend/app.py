"""Self-Improving RAG — Streamlit UI

Tabs:
  1. Test Playground — select collection, run queries, see results, trigger optimization
  2. Optimizer      — run improvement loop, view iterations, HITL decisions
  3. History        — view past runs and score trends
"""

import streamlit as st
import os
import sys
import time
import copy

# Ensure frontend/ is on sys.path so "frontend_utils" can be found,
# and project root is on sys.path for pipeline modules.
_frontend_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_frontend_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from agents.diagnoser import diagnoser_node, _is_abstention
from pipeline import _compute_quality_score

# Import frontend utilities — use importlib to avoid collision with
# project-root utils.py (both are named "utils").
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "frontend_utils", os.path.join(_frontend_dir, "utils.py")
)
_fe_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fe_utils)

list_corpus_files_recursive = _fe_utils.list_corpus_files_recursive
save_uploaded_file = _fe_utils.save_uploaded_file
run_query = _fe_utils.run_query
get_ground_truth_queries = _fe_utils.get_ground_truth_queries
run_optimization_ui = _fe_utils.run_optimization_ui
save_query_run = _fe_utils.save_query_run
save_optimization_run = _fe_utils.save_optimization_run
load_history = _fe_utils.load_history
clear_history = _fe_utils.clear_history
get_collections = _fe_utils.get_collections
parse_collection_name = _fe_utils.parse_collection_name
build_collection_name = _fe_utils.build_collection_name
collection_label = _fe_utils.collection_label
CORPUS_DIR = _fe_utils.CORPUS_DIR
DEFAULT_CONFIG = _fe_utils.DEFAULT_CONFIG
BAD_CONFIG = _fe_utils.BAD_CONFIG
INGEST_PAGES = _fe_utils.INGEST_PAGES


def _render_judge_details(judge_details: dict, key_prefix: str):
    """Render per-judge breakdown: claims for faithfulness, reasoning for the rest.

    Collapsible via a selectbox (same interaction as the execution trace
    inspector) rather than always showing every judge at once.
    """
    if not judge_details:
        return

    available = [name for name in ("faithfulness", "relevance", "correctness") if judge_details.get(name)]
    if not available:
        return

    st.markdown("**Judge Breakdown**")
    picked = st.selectbox("Inspect judge", available, key=f"judge_pick_{key_prefix}")
    detail = judge_details.get(picked, {})

    if picked == "faithfulness" and detail.get("claims"):
        claims = detail["claims"]
        supported = detail.get("supported", [])
        rows = [
            {"claim": c, "supported": (supported[i] if i < len(supported) else None)}
            for i, c in enumerate(claims)
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True, key=f"judge_faith_{key_prefix}")
    elif detail.get("reasoning"):
        st.caption(detail["reasoning"])
        if picked == "correctness" and detail.get("expected_answer"):
            st.caption(f"Expected: \"{detail['expected_answer'][:150]}\"")


def _format_config(cfg: dict) -> str:
    return (
        f"k={cfg.get('retrieval_k')}, "
        f"chunk={cfg.get('chunk_size')}, "
        f"overlap={cfg.get('chunk_overlap')}, "
        f"prompt={cfg.get('prompt_template')}"
    )


def _resolve_next_config(rec: dict, iterations_list: list, it_num, final_config: dict | None = None):
    """Find the config the improver's fix actually produced.

    Prefers applied_variant['config_after'] (present on live/in-memory
    reports); falls back to the next iteration's recorded config (still
    correct after history's slimming strips config_after); falls back to
    the run's final_config if this was the last iteration.
    """
    variant = rec.get("applied_variant") or {}
    if variant.get("config_after"):
        return variant["config_after"]
    for other in iterations_list:
        if other.get("iteration") == it_num + 1:
            return other.get("config")
    return final_config


def _render_diagnosis_block(failure_type: str, remediation_hint: str):
    """Highlight the diagnoser's finding so it doesn't blend into plain text."""
    if remediation_hint:
        st.warning(f"**Diagnosis {failure_type}:** {remediation_hint}")


def _render_applied_variant(variant: dict, next_config: dict | None = None):
    """Highlight the fix the improver proposed — this changes the config for
    the NEXT iteration, not the one that produced the answer above."""
    if not variant:
        return
    lines = [
        f"**Fix Applied → Next Iteration's Config ({variant.get('variant_id', '?')}):** "
        f"{variant.get('rationale', 'N/A')}",
        f"**Delta:** `{variant.get('delta', {})}`",
    ]
    if next_config:
        lines.append(f"**Resulting Config:** {_format_config(next_config)}")
    st.info("  \n".join(lines))


def _render_answer_block(answer: str, max_len: int = 800):
    """Show the generated answer, highlighting abstentions distinctly."""
    if not answer:
        return
    text = answer[:max_len]
    if _is_abstention(answer):
        st.error(f"**Answer (abstained):** {text}")
    else:
        st.success(f"**Answer:** {text}")


def _render_execution_trace(trace: list, key_prefix: str):
    """Render one iteration's append-only execution_trace as a compact table
    plus per-event input/output summaries.

    Note: no st.expander here — Streamlit forbids nesting expanders, and this
    is always called from within an already-open iteration expander.
    """
    if not trace:
        return
    rows = [
        {
            "node": ev.get("node"),
            "status": ev.get("status"),
            "latency_ms": ev.get("latency_ms"),
            "tokens": ev.get("tokens"),
            "cost_usd": ev.get("cost_usd"),
            "error": ev.get("error"),
        }
        for ev in trace
    ]
    st.markdown("**Execution Trace**")
    st.dataframe(rows, use_container_width=True, hide_index=True, key=f"trace_{key_prefix}")

    labels = [
        f"{i}: {ev.get('node')} ({ev.get('status')})"
        for i, ev in enumerate(trace)
    ]
    picked = st.selectbox(
        "Inspect node input/output", labels, key=f"trace_pick_{key_prefix}"
    )
    ev = trace[labels.index(picked)]
    io_cols = st.columns(2)
    with io_cols[0]:
        st.markdown("**Input**")
        st.json(ev.get("input_summary", {}))
    with io_cols[1]:
        st.markdown("**Output**")
        st.json(ev.get("output_summary", {}))


def _fmt(val, fmt=".2f", prefix="", suffix=""):
    if val is None:
        return "-"
    if isinstance(val, (int, float)) and val == 0:
        return "-"
    return f"{prefix}{val:{fmt}}{suffix}"


def _render_unified_score_breakdown(rec: dict, key_prefix: str):
    """Show the unified score arithmetic (mirrors _compute_unified_score in pipeline.py).

    Reuses the real _compute_quality_score() for the quality sub-score so
    this display can't drift from the actual scoring logic; the weight
    re-normalization and penalties are re-derived here purely for display.
    """
    recall = rec.get("retrieval_score")
    faithfulness = rec.get("faithfulness")
    quality = _compute_quality_score(rec.get("relevance"), rec.get("correctness"))

    MAX_QUERY_COST = 0.01
    latency_penalty = 0.10 * min((rec.get("latency_ms") or 0) / 3000.0, 1.0)
    cost_penalty = 0.05 * min((rec.get("cost_usd") or 0.0) / MAX_QUERY_COST, 1.0)

    terms = []
    if recall is not None:
        terms.append(("Retrieval (Recall)", 0.25, recall))
    if quality is not None:
        terms.append(("Quality (0.6×Correctness + 0.4×Relevance)", 0.35, quality))
    if faithfulness is not None:
        terms.append(("Faithfulness", 0.25, faithfulness))

    raw_weight_sum = sum(w for _, w, _ in terms)
    scale = 0.85 / raw_weight_sum if raw_weight_sum > 0 else 1.0

    st.markdown("**Unified Score Calculation**")
    st.caption(
        "Score = Σ(effective weight × value) − latency penalty − cost penalty. "
        "Weights re-normalize to sum to 0.85 when a metric is missing (no fabricated scores)."
    )
    if terms:
        rows = [
            {
                "term": name,
                "base weight": w,
                "value": round(v, 4),
                "effective weight": round(w * scale, 4),
                "contribution": round(w * scale * v, 4),
            }
            for name, w, v in terms
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True, key=f"score_calc_{key_prefix}")

    pen_cols = st.columns(2)
    with pen_cols[0]:
        st.metric("Latency Penalty", f"-{latency_penalty:.4f}")
    with pen_cols[1]:
        st.metric("Cost Penalty", f"-{cost_penalty:.4f}")

    computed = round(max(0.0, min(1.0, sum(w * scale * v for _, w, v in terms) - latency_penalty - cost_penalty)), 4)
    st.caption(f"= **{computed}** (recorded unified score: {rec.get('unified_score')})")


def _render_iteration_workflow(
    rec: dict, iterations_list: list, it_num: int, key_prefix: str,
    final_config: dict | None, query: str,
):
    """Render one iteration in the order the graph actually executes it:
    input (query/config) -> answer (builder+pipeline) -> evaluation
    (evaluator) -> diagnosis (diagnoser) -> fix (improver) -> cost.
    Shared by the Optimizer tab and the History tab so both stay in sync.
    """
    st.markdown("**Input**")
    st.caption(f"Query: {query}")
    st.caption(f"Config used this iteration: {_format_config(rec.get('config', {}))}")

    st.divider()
    st.markdown("**Answer**")
    _render_answer_block(rec.get("answer") or rec.get("answer_preview", ""))

    st.divider()
    st.markdown("**Evaluation**")
    score_cols = st.columns(5)
    with score_cols[0]:
        st.metric("Faithfulness", _fmt(rec.get("faithfulness")))
    with score_cols[1]:
        st.metric("Relevance", _fmt(rec.get("relevance")))
    with score_cols[2]:
        st.metric("Correctness", _fmt(rec.get("correctness")))
    with score_cols[3]:
        st.metric("Retrieval", _fmt(rec.get("retrieval_score")))
    with score_cols[4]:
        st.metric("Unified", _fmt(rec.get("unified_score"), ".4f"))
    _render_judge_details(rec.get("judge_details", {}), key_prefix=key_prefix)
    _render_unified_score_breakdown(rec, key_prefix=key_prefix)

    if rec.get("failure_type"):
        st.divider()
        st.markdown("**Diagnosis**")
        _render_diagnosis_block(rec["failure_type"], rec.get("remediation_hint", ""))

    if rec.get("applied_variant"):
        st.divider()
        st.markdown("**Fix → Next Iteration's Config**")
        next_cfg = _resolve_next_config(rec, iterations_list, it_num, final_config)
        _render_applied_variant(rec.get("applied_variant"), next_cfg)

    st.divider()
    st.markdown("**Cost for This Iteration**")
    meta_cols = st.columns(3)
    with meta_cols[0]:
        st.metric("Cost", f"${rec.get('cost_usd') or 0:.4f}")
    with meta_cols[1]:
        st.metric("Latency", f"{rec.get('latency_ms', 0)}ms")
    with meta_cols[2]:
        st.metric("Chunks", rec.get("chunk_count", 0))

    _render_execution_trace(rec.get("execution_trace", []), key_prefix=key_prefix)


# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Self-Improving RAG",
    page_icon="🔍",
    layout="wide",
)

# ─── Tab Navigation (session-state driven, survives reruns) ───────────────────
TAB_OPTIONS = ["Test Playground", "Optimizer", "History"]

# nav_radio is the widget's own key — it's the single source of truth.
# Once a keyed widget has a value in session_state, Streamlit ignores the
# index= argument on every later rerun, so programmatic tab switches must
# write to nav_radio directly. But Streamlit also forbids setting a keyed
# widget's state AFTER it's instantiated in the same run — buttons further
# down the script can't touch nav_radio directly (it's already rendered by
# the time they run). So they stash the target tab in "_pending_tab"
# instead, and we apply it here, before the widget is created.
if "_pending_tab" in st.session_state:
    st.session_state["nav_radio"] = st.session_state.pop("_pending_tab")

if "nav_radio" not in st.session_state:
    st.session_state["nav_radio"] = TAB_OPTIONS[0]

active_tab = st.radio(
    "Navigation",
    TAB_OPTIONS,
    horizontal=True,
    key="nav_radio",
    label_visibility="collapsed",
)
st.session_state["active_tab"] = active_tab

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Test Playground
# ═══════════════════════════════════════════════════════════════════════════════
if active_tab == "Test Playground":
    st.title("Test Playground")
    st.caption(
        "Run queries against an ingested collection. "
        "If scores are low, send to the Optimizer for automatic improvement."
    )

    # ── Setup Section ─────────────────────────────────────────────────────
    st.subheader("1. Corpus / Ingestion")
    st.caption("Selects the underlying corpus and chunking scheme (the fixed, ingested part of the config).")

    # Two paths: select existing collection OR ingest new
    collections = get_collections()

    # Check if optimizer sent us back with a pre-selected config
    if "opt_result_config" in st.session_state:
        opt_cfg_back = st.session_state.pop("opt_result_config")
        opt_coll_back = st.session_state.pop("opt_result_collection", None)
        # Pre-fill query-time params from optimizer result
        st.session_state["pg_retrieval_k"] = opt_cfg_back.get("retrieval_k", 5)
        st.session_state["pg_max_context_tokens"] = opt_cfg_back.get("max_context_tokens", 4000)
        if opt_coll_back:
            st.session_state["pg_selected_collection"] = opt_coll_back
            existing_names = [c["name"] for c in collections]
            if opt_coll_back in existing_names:
                # pg_collection_select is the selectbox's own key and stores
                # the display *label*, not the raw name — index= alone won't
                # move it once the key already has a value, so set it directly.
                st.session_state["pg_collection_select"] = collection_label(
                    collections[existing_names.index(opt_coll_back)]
                )
        st.info(
            f"Config loaded from Optimizer: k={opt_cfg_back.get('retrieval_k')}, "
            f"collection={opt_coll_back or 'auto'}"
        )

    if collections:
        # Build collection options
        col_options = [collection_label(c) for c in collections]
        col_names = [c["name"] for c in collections]

        # Find default index
        default_idx = 0
        if "pg_selected_collection" in st.session_state:
            try:
                default_idx = col_names.index(st.session_state["pg_selected_collection"])
            except ValueError:
                default_idx = 0

        selected_label = st.selectbox(
            "Select collection",
            col_options,
            index=default_idx,
            key="pg_collection_select",
        )
        selected_collection = col_names[col_options.index(selected_label)]
        st.session_state["pg_selected_collection"] = selected_collection

        # Parse config from collection name
        parsed = parse_collection_name(selected_collection)
        st.caption(
            f"Version: `{parsed['version']}` | "
            f"Strategy: `{parsed['chunk_strategy']}` | "
            f"Chunk size: `{parsed['chunk_size']}`"
        )
    else:
        st.warning("No collections found. Ingest a document first.")
        selected_collection = None

    # Ingest documents — pick any subset of the corpus and the chunking
    # config to ingest them with. Defaults to selecting everything (the
    # golden query set in ground_truth.py assumes the full 9-doc corpus).
    all_corpus_files = list_corpus_files_recursive()
    if all_corpus_files:
        ing_selected = st.multiselect(
            "Documents to ingest", all_corpus_files, default=all_corpus_files,
            key="ing_selected_files",
        )
        ing_cols = st.columns(3)
        with ing_cols[0]:
            ing_strategy = st.selectbox(
                "Strategy", ["fixed_size", "recursive_split", "semantic"],
                key="ing_strategy",
            )
        with ing_cols[1]:
            ing_chunk_size = st.number_input(
                "Chunk size", min_value=32, max_value=512, value=256, step=32,
                key="ing_chunk_size",
            )
        with ing_cols[2]:
            ing_overlap = st.number_input(
                "Overlap", min_value=0, max_value=128, value=0, step=16,
                key="ing_overlap",
            )

        if st.button("Ingest Selected Documents", type="primary", key="ing_button", disabled=not ing_selected):
            book_paths = [os.path.join(CORPUS_DIR, f) for f in ing_selected]
            with st.status(f"Ingesting {len(ing_selected)} document(s)...", expanded=True) as status:
                for f in ing_selected:
                    st.write(f"- {f}")
                st.write(f"Strategy: {ing_strategy}, Size: {ing_chunk_size}, Overlap: {ing_overlap}")
                try:
                    from ingest import ingest
                    collection_name = ingest(
                        strategy=ing_strategy,
                        chunk_size=ing_chunk_size,
                        chunk_overlap=ing_overlap,
                        version="g1",
                        book_paths=book_paths,
                    )
                    st.session_state["pg_selected_collection"] = collection_name
                    status.update(label=f"Done: {collection_name}", state="complete")
                    st.rerun()
                except Exception as e:
                    status.update(label="Ingestion failed", state="error")
                    st.error(str(e))

        if len(ing_selected) < len(all_corpus_files):
            st.caption(
                "Note: a partial selection with the same strategy/size/overlap as a "
                "previous ingestion reuses that collection's chunks (collection "
                "identity doesn't track which specific files were ingested)."
            )
    else:
        st.info("No files in corpus/. Add .txt or .pdf files to the corpus/ directory (subfolders are discovered too).")

    st.divider()


    # ── Baseline Config Section ────────────────────────────────────────────
    st.subheader("2. Baseline Config")
    st.caption(
        "This is the exact baseline configuration used for this run — "
        "the reference point the Optimizer will try to beat."
    )

    if selected_collection:
        parsed_cfg = parse_collection_name(selected_collection)
        baseline_overlap = parsed_cfg["chunk_overlap"]

        base_cols = st.columns(2)
        with base_cols[0]:
            with st.container(border=True):
                st.markdown("**Ingestion Config**")
                st.caption("Baked into the collection — change these means re-ingesting.")
                ic_cols = st.columns(3)
                with ic_cols[0]:
                    st.metric("Chunk Strategy", parsed_cfg["chunk_strategy"])
                with ic_cols[1]:
                    st.metric("Chunk Size", parsed_cfg["chunk_size"])
                with ic_cols[2]:
                    st.metric("Chunk Overlap", baseline_overlap)
        with base_cols[1]:
            with st.container(border=True):
                st.markdown("**Retrieval Config**")
                st.caption("Query-time knobs — free to tweak per query, no re-ingestion needed.")
                rc_cols = st.columns(2)
                with rc_cols[0]:
                    pg_k = st.number_input(
                        "retrieval_k", min_value=1, max_value=20,
                        value=st.session_state.get("pg_retrieval_k", 5),
                        key="pg_k_input",
                    )
                with rc_cols[1]:
                    pg_max_ctx = st.number_input(
                        "max_context_tokens", min_value=1000, max_value=8000, step=500,
                        value=st.session_state.get("pg_max_context_tokens", 4000),
                        key="pg_ctx_input",
                    )

        # Store for cross-tab use
        st.session_state["pg_retrieval_k"] = pg_k
        st.session_state["pg_max_context_tokens"] = pg_max_ctx
    else:
        st.info("Select or ingest a collection above to see its baseline config.")

    st.divider()

    # ── Query Section ─────────────────────────────────────────────────────
    st.subheader("3. Query")

    if selected_collection:
        # Query input
        gt_queries = get_ground_truth_queries()
        query_options = ["(type custom query below)"] + gt_queries
        selected_query_option = st.selectbox(
            "Select a test query (ground truth enables correctness scoring)",
            query_options,
            key="pg_query_select",
        )

        if selected_query_option == "(type custom query below)":
            pg_query = st.text_input(
                "Custom query:",
                placeholder="e.g. How do embeddings represent meaning?",
                key="pg_custom_query",
            )
        else:
            pg_query = selected_query_option

        # Build full config from baseline (corpus) + query-time params
        full_config = {
            "collection_name": selected_collection,
            "chunk_strategy": parsed_cfg["chunk_strategy"],
            "chunk_size": parsed_cfg["chunk_size"],
            "chunk_overlap": baseline_overlap,
            "retrieval_k": pg_k,
            "max_context_tokens": pg_max_ctx,
            "prompt_template": "v1",
        }


        if st.button("Run Query", type="primary", key="pg_run", disabled=not pg_query):
            with st.spinner("Running pipeline..."):
                try:
                    t0 = time.time()
                    result = run_query(
                        query=pg_query,
                        config=copy.deepcopy(full_config),
                        version=parsed_cfg["version"],
                        pages=INGEST_PAGES,
                    )
                    elapsed = time.time() - t0
                    result["_elapsed"] = round(elapsed, 2)
                    result["_collection"] = selected_collection
                    if result.get("gate_decision") == "hard_block":
                        result.update(diagnoser_node(result))
                    st.session_state["pg_last_result"] = result
                    st.session_state["pg_last_query"] = pg_query
                    st.session_state["pg_last_config"] = full_config

                    # Save to history
                    save_query_run(pg_query, full_config, result, version=parsed_cfg["version"])

                    # Add to session history
                    if "pg_session_history" not in st.session_state:
                        st.session_state["pg_session_history"] = []
                    st.session_state["pg_session_history"].append({
                        "query": pg_query[:60],
                        "score": result.get("unified_score"),
                        "gate": result.get("gate_decision"),
                        "collection": selected_collection,
                        "timestamp": time.strftime("%H:%M:%S"),
                    })
                except Exception as e:
                    st.error(f"Pipeline error: {e}")

        # ── Results Display ───────────────────────────────────────────────
        if "pg_last_result" in st.session_state:
            state = st.session_state["pg_last_result"]
            handoff_query = st.session_state.get("pg_last_query", "")
            handoff_config = copy.deepcopy(st.session_state.get("pg_last_config", full_config))
            handoff_collection = state.get("_collection", handoff_config.get("collection_name", ""))
            handoff_version = parse_collection_name(handoff_collection).get("version", "g1")

            st.divider()
            st.subheader("4. Results")

            # Answer
            st.markdown("**Answer:**")
            st.info(state.get("answer", "No answer"))

            # Score metrics row
            score_cols = st.columns(5)
            with score_cols[0]:
                faith = state.get("faithfulness")
                st.metric("Faithfulness", f"{faith:.2f}" if faith is not None else "N/A")
            with score_cols[1]:
                rel = state.get("relevance")
                st.metric("Relevance", f"{rel:.2f}" if rel is not None else "N/A")
            with score_cols[2]:
                corr = state.get("correctness")
                st.metric("Correctness", f"{corr:.2f}" if corr is not None else "N/A")
            with score_cols[3]:
                ret = state.get("retrieval_score")
                st.metric("Retrieval", f"{ret:.2f}" if ret is not None else "N/A")
            with score_cols[4]:
                unified = state.get("unified_score")
                st.metric("Unified Score", f"{unified:.2f}" if unified is not None else "N/A")

            # Gate decision banner
            gate = state.get("gate_decision")
            if unified is not None:
                if unified >= 0.85:
                    st.success(f"Score {unified:.2f} >= 0.85 — deploy_eligible")
                elif unified >= 0.70:
                    st.warning(f"Score {unified:.2f} in [0.70, 0.85) — hitl_required")
                else:
                    st.error(f"Score {unified:.2f} < 0.70 — hard_block")

            if faith is not None and faith < 0.50:
                st.error("VETO: Faithfulness < 0.50 — blocked regardless of unified score")

            if state.get("failure_type"):
                st.subheader("Diagnosis")
                st.warning(
                    f"{state['failure_type']}: "
                    f"{state.get('root_cause_analysis', 'No root cause analysis available.')}"
                )
                st.caption(state.get("remediation_hint", ""))

            # Metadata
            meta_cols = st.columns(5)
            with meta_cols[0]:
                st.metric("Cost", f"${state.get('cost_usd') or 0:.4f}")
            with meta_cols[1]:
                st.metric("Latency", f"{state.get('latency_ms', 0)}ms")
            with meta_cols[2]:
                st.metric("Chunks", state.get("chunk_count", 0))
            with meta_cols[3]:
                st.metric("Context Tokens", state.get("context_tokens", 0))
            with meta_cols[4]:
                # Source file name (basename only)
                coll_name = state.get("_collection", state.get("collection_name", ""))
                st.metric("Collection", coll_name.split("_")[-2] + "_" + coll_name.split("_")[-1] if "_" in coll_name else coll_name)

            # Collection clearly shown
            st.caption(f"Executed against: `{state.get('_collection', state.get('collection_name', 'N/A'))}`")

            # Retrieved chunks — collapsible per chunk so each can be
            # inspected individually without one giant wall of text.
            chunks = state.get("retrieved_chunks", [])
            st.subheader(f"Retrieved Chunks ({len(chunks)})")
            if chunks:
                for i, ch in enumerate(chunks):
                    if isinstance(ch, dict):
                        meta = ch.get("metadata", {}) if isinstance(ch.get("metadata"), dict) else {}
                        source = meta.get("source", "")
                        source_display = os.path.basename(source) if source else "unknown"
                        chunk_idx = meta.get("chunk_index", "?")
                        sim_score = ch.get("score")
                        sim_str = f"{sim_score:.4f}" if sim_score is not None else "N/A"

                        label = f"Chunk {i + 1} | Similarity: {sim_str} | Source: {source_display}"
                        with st.expander(label, expanded=(i == 0)):
                            meta_cols = st.columns(4)
                            with meta_cols[0]:
                                st.caption("Source")
                                st.markdown(f"`{source_display}`")
                            with meta_cols[1]:
                                st.caption("Chunk Index")
                                st.markdown(f"`{chunk_idx}`")
                            with meta_cols[2]:
                                st.caption("Chunk ID")
                                st.markdown(f"`{ch.get('chunk_id', '?')}`")
                            with meta_cols[3]:
                                st.caption("Collection")
                                st.markdown(f"`{state.get('_collection', state.get('collection_name', ''))}`")
                            st.code(ch.get("text", str(ch)), language=None)
            else:
                st.caption("No chunks retrieved.")


            # ── Score Nudge — Suggest Optimization ────────────────────────
            if unified is not None and unified < 0.70:
                st.divider()
                st.markdown(
                    "**This collection has a low evaluation score.** "
                    "The optimizer can automatically improve retrieval configuration."
                )
                if st.button("Optimize this collection", type="primary", key="pg_to_opt"):
                    # Preserve the query and config that produced this result.
                    st.session_state["opt_prefill_query"] = handoff_query
                    st.session_state["opt_prefill_config"] = handoff_config
                    st.session_state["opt_prefill_baseline"] = state
                    st.session_state["opt_prefill_version"] = handoff_version
                    st.session_state["opt_config"] = copy.deepcopy(handoff_config)
                    st.session_state["opt_k"] = handoff_config["retrieval_k"]
                    st.session_state["opt_chunk"] = handoff_config["chunk_size"]
                    st.session_state["opt_overlap"] = handoff_config["chunk_overlap"]
                    st.session_state["opt_ctx"] = handoff_config["max_context_tokens"]
                    st.session_state["opt_query_select"] = (
                        handoff_query if handoff_query in get_ground_truth_queries() else "Custom query"
                    )
                    st.session_state["opt_custom_query"] = handoff_query
                    st.session_state.pop("opt_report", None)
                    st.session_state["_pending_tab"] = "Optimizer"
                    st.rerun()

            # ── HITL Decision (gray band) ─────────────────────────────────
            elif unified is not None and 0.70 <= unified < 0.85:
                st.divider()
                st.subheader("Human-in-the-Loop Decision")
                st.markdown(
                    f"Score **{unified:.4f}** is in the HITL band [0.70, 0.85). "
                    "This result needs human review before deployment."
                )

                hitl_cols = st.columns(3)
                with hitl_cols[0]:
                    if st.button("Approve (deploy as-is)", type="primary", key="pg_hitl_approve"):
                        st.session_state["pg_hitl_decision"] = "approved"
                        st.rerun()
                with hitl_cols[1]:
                    if st.button("Reject & Optimize", key="pg_hitl_reject"):
                        st.session_state["opt_prefill_query"] = handoff_query
                        st.session_state["opt_prefill_config"] = handoff_config
                        st.session_state["opt_prefill_baseline"] = state
                        st.session_state["opt_prefill_version"] = handoff_version
                        st.session_state["opt_config"] = copy.deepcopy(handoff_config)
                        st.session_state["opt_k"] = handoff_config["retrieval_k"]
                        st.session_state["opt_chunk"] = handoff_config["chunk_size"]
                        st.session_state["opt_overlap"] = handoff_config["chunk_overlap"]
                        st.session_state["opt_ctx"] = handoff_config["max_context_tokens"]
                        st.session_state["opt_query_select"] = (
                            handoff_query if handoff_query in get_ground_truth_queries() else "Custom query"
                        )
                        st.session_state["opt_custom_query"] = handoff_query
                        st.session_state.pop("opt_report", None)
                        st.session_state["_pending_tab"] = "Optimizer"
                        st.rerun()
                with hitl_cols[2]:
                    if st.button("Reject (discard)", key="pg_hitl_discard"):
                        st.session_state["pg_hitl_decision"] = "discarded"
                        st.rerun()

                decision = st.session_state.get("pg_hitl_decision")
                if decision == "approved":
                    st.success("Approved! This result is accepted for deployment.")
                elif decision == "discarded":
                    st.info("Discarded. Try a different query or config.")

            # ── How metrics are measured (collapsible) ────────────────────
            with st.expander("How are these metrics measured?"):
                st.markdown("""
**Faithfulness** (LLM-judge, 0-1): Breaks answer into claims, checks each against context.
Score = supported claims / total claims. If < 0.50, triggers safety veto.

**Relevance** (LLM-judge, 0-1): Does the answer address the question?

**Correctness** (LLM-judge, 0-1): Compares to expected answer. Only for test queries.

**Retrieval** (keyword-based, no LLM, 0-1): 0.5 x Precision@k + 0.5 x Recall@k.

**Unified Score** (formula, 0-1):
`0.25 x Retrieval + 0.35 x Quality + 0.25 x Faithfulness - 0.10 x Latency - 0.05 x Cost`

Where Quality = 0.6 x Correctness + 0.4 x Relevance.
Gate bands: >= 0.85 deploy, 0.70-0.84 HITL, < 0.70 blocked.
""")

    else:
        st.info("Select or ingest a collection above to start querying.")

    # ── Session History ───────────────────────────────────────────────────
    session_hist = st.session_state.get("pg_session_history", [])
    if session_hist:
        st.divider()
        st.subheader("Session History")
        import pandas as pd
        df = pd.DataFrame(session_hist)
        st.dataframe(
            df[["timestamp", "query", "score", "gate", "collection"]],
            use_container_width=True,
            hide_index=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Optimizer — self-improving loop
# ═══════════════════════════════════════════════════════════════════════════════
if active_tab == "Optimizer":
    st.title("Optimizer")
    st.caption(
        "Run the self-improving loop: evaluate → diagnose → improve → repeat. "
        "Can be triggered from the Test Playground or configured manually."
    )

    # ── Setup Section ─────────────────────────────────────────────────────
    st.subheader("1. Setup")

    # Check for pre-fill from Playground
    prefilled = False
    if "opt_prefill_query" in st.session_state:
        prefilled = True
        opt_query_default = st.session_state.get("opt_prefill_query", "")
        opt_config_default = st.session_state.get("opt_prefill_config", copy.deepcopy(BAD_CONFIG))
        opt_baseline_result = st.session_state.get("opt_prefill_baseline")
        opt_version = st.session_state.get("opt_prefill_version", "g1")
        st.success(
            f"Pre-filled from Playground: \"{opt_query_default[:60]}\" | "
            f"Baseline score: {opt_baseline_result.get('unified_score', 'N/A') if opt_baseline_result else 'N/A'}"
        )
    else:
        opt_query_default = ""
        opt_config_default = copy.deepcopy(BAD_CONFIG)
        opt_baseline_result = None
        opt_version = "g1"

    # Query selection
    gt_queries_opt = get_ground_truth_queries()
    custom_query_option = "Custom query"
    query_options = [custom_query_option, *gt_queries_opt]
    if prefilled and opt_query_default in gt_queries_opt:
        default_q_idx = query_options.index(opt_query_default)
    else:
        default_q_idx = 0
    selected_query = st.selectbox(
        "Query", query_options, index=default_q_idx, key="opt_query_select",
    )
    if selected_query == custom_query_option:
        opt_query = st.text_area(
            "Custom query",
            value=opt_query_default if prefilled else "",
            key="opt_custom_query",
        )
    else:
        opt_query = selected_query

    # If user changes query from pre-filled, invalidate baseline
    if prefilled and opt_query != opt_query_default:
        opt_baseline_result = None

    # Config
    st.markdown("**Starting Config**")

    if not prefilled:
        col_bad, col_default = st.columns(2)
        with col_bad:
            if st.button("Use Bad Config (k=1, chunk=64)", key="opt_bad"):
                st.session_state["opt_config"] = copy.deepcopy(BAD_CONFIG)
                st.rerun()
        with col_default:
            if st.button("Use Default Config (k=5, chunk=256)", key="opt_default"):
                st.session_state["opt_config"] = copy.deepcopy(DEFAULT_CONFIG)
                st.rerun()

    if "opt_config" not in st.session_state:
        st.session_state["opt_config"] = copy.deepcopy(opt_config_default)

    opt_cfg = st.session_state["opt_config"]

    # Editable config
    cfg_cols = st.columns(4)
    with cfg_cols[0]:
        opt_cfg["retrieval_k"] = st.number_input(
            "retrieval_k", min_value=1, max_value=20,
            value=opt_cfg.get("retrieval_k", 1), key="opt_k",
        )
    with cfg_cols[1]:
        opt_cfg["chunk_size"] = st.number_input(
            "chunk_size", min_value=32, max_value=512, step=32,
            value=opt_cfg.get("chunk_size", 64), key="opt_chunk",
        )
    with cfg_cols[2]:
        opt_cfg["chunk_overlap"] = st.number_input(
            "chunk_overlap", min_value=0, max_value=128, step=16,
            value=opt_cfg.get("chunk_overlap", 0), key="opt_overlap",
        )
    with cfg_cols[3]:
        opt_cfg["max_context_tokens"] = st.number_input(
            "max_context_tokens", min_value=1000, max_value=8000, step=500,
            value=opt_cfg.get("max_context_tokens", 4000), key="opt_ctx",
        )

    # If user modified config from pre-filled baseline, invalidate baseline
    if prefilled and opt_baseline_result:
        baseline_cfg = st.session_state.get("opt_prefill_config", {})
        if (opt_cfg.get("retrieval_k") != baseline_cfg.get("retrieval_k") or
                opt_cfg.get("chunk_size") != baseline_cfg.get("chunk_size")):
            opt_baseline_result = None
            st.caption("Config changed from Playground baseline — will run fresh baseline.")

    # Target and iterations
    t_cols = st.columns(3)
    with t_cols[0]:
        opt_target = st.number_input(
            "Target Score", min_value=0.50, max_value=1.00, step=0.05,
            value=0.85, key="opt_target",
        )
    with t_cols[1]:
        opt_max_iter = st.number_input(
            "Max Iterations", min_value=1, max_value=10,
            value=5, key="opt_max_iter",
        )
    with t_cols[2]:
        opt_force = st.checkbox(
            "Force (skip faithfulness veto)",
            value=True, key="opt_force",
            help="When checked, the optimizer will not stop on faithfulness veto (< 0.50). "
                 "Useful for demos where a bad config produces low faithfulness initially. "
                 "HITL stops are always respected.",
        )

    # Show baseline info if available
    if opt_baseline_result:
        st.divider()
        st.markdown(
            f"**Baseline (from Playground):** "
            f"score={opt_baseline_result.get('unified_score', 'N/A')}, "
            f"gate={opt_baseline_result.get('gate_decision', 'N/A')}"
        )

    st.divider()

    # ── Run Optimizer ─────────────────────────────────────────────────────
    st.subheader("2. Run Optimizer")

    if st.button("Run Optimizer", type="primary", key="run_optimizer"):
        if not opt_query.strip():
            st.error("Enter a custom query or select a golden-set query.")
        else:
            with st.spinner("Running optimization loop... (multiple LLM calls per iteration)"):
                try:
                    report = run_optimization_ui(
                        query=opt_query,
                        config=copy.deepcopy(opt_cfg),
                        version=opt_version,
                        target_score=opt_target,
                        max_iterations=opt_max_iter,
                        baseline_result=opt_baseline_result,
                        force_continue=opt_force,
                    )
                    st.session_state["opt_report"] = report
                    save_optimization_run(opt_query, report)
                    # Clear pre-fill state
                    st.session_state.pop("opt_prefill_query", None)
                    st.session_state.pop("opt_prefill_config", None)
                    st.session_state.pop("opt_prefill_baseline", None)
                    st.session_state.pop("opt_prefill_version", None)
                except Exception as e:
                    st.error(f"Optimizer error: {e}")

    # ── Display Results ───────────────────────────────────────────────────
    if "opt_report" in st.session_state:
        report = st.session_state["opt_report"]
        iterations = report.get("iterations", [])
        stop = report.get("stop_reason", "unknown")
        improvement = report.get("improvement")
        final_score = report.get("final_score")
        initial_score = report.get("initial_score")

        # Summary banner
        if stop == "target_reached":
            st.success(
                f"Target reached! Score {final_score:.4f} >= {opt_target:.2f} "
                f"in {report['total_iterations']} iteration(s). "
                f"Improvement: +{improvement:.4f}"
            )
        elif stop == "hitl_required":
            st.warning(
                f"HITL Required — Score {final_score:.4f} is in the gray band "
                f"[0.70, 0.85). Human review needed."
                + (f" Improvement: +{improvement:.4f}" if improvement else "")
            )
        elif stop == "blocked_faithfulness":
            st.error(
                f"Blocked — Faithfulness below safety floor (0.50). "
                f"Score: {final_score:.4f}. Cannot proceed."
            )
        elif stop == "no_improvement":
            st.warning(f"Plateau — 3 consecutive iterations with delta < 0.01. Final: {final_score:.4f}.")
        elif stop == "no_candidates":
            st.warning(f"No candidates — improver could not generate fixes. Final: {final_score:.4f}.")
        else:
            st.info(f"Max iterations reached ({report['total_iterations']}). Final: {final_score:.4f}.")

        # Per-iteration details
        st.markdown("**Iteration Details**")
        for rec in iterations:
            it_num = rec["iteration"]
            if it_num == 0:
                # Baseline duplicates what was already shown in the Playground run.
                continue
            it_score = rec.get("unified_score", 0)
            it_gate = rec.get("gate_decision", "?")
            it_fail = rec.get("failure_type", "-")

            prev_score = iterations[it_num - 1]["unified_score"] if it_num > 0 and it_num < len(iterations) else None
            delta_str = ""
            if it_num > 0 and prev_score is not None and it_score is not None:
                prev = iterations[it_num - 1]["unified_score"] if it_num - 1 >= 0 else None
                if prev is not None:
                    delta = it_score - prev
                    delta_str = f" (delta: {'+' if delta >= 0 else ''}{delta:.4f})"

            label = (
                f"{'Baseline' if it_num == 0 else f'Iteration {it_num}'}: "
                f"score={it_score:.4f}  |  gate={it_gate}  |  failure={it_fail}{delta_str}"
            )

            with st.expander(label, expanded=(it_num == len(iterations) - 1)):
                _render_iteration_workflow(
                    rec, iterations, it_num, key_prefix=f"opt_{it_num}",
                    final_config=report.get("final_config"), query=opt_query,
                )

        st.divider()

        # ── HITL Decision ─────────────────────────────────────────────────
        if stop == "hitl_required":
            st.subheader("3. Human-in-the-Loop Decision")
            st.markdown(
                f"Score **{final_score:.4f}** is in the HITL band [0.70, 0.85)."
            )

            col_approve, col_reject = st.columns(2)
            with col_approve:
                if st.button("Approve & Deploy / Test in Playground", type="primary", key="hitl_approve"):
                    # Push optimized config to Playground
                    final_cfg = report.get("final_config", {})
                    opt_collection = build_collection_name(final_cfg, "g1")
                    st.session_state["opt_result_config"] = final_cfg
                    st.session_state["opt_result_collection"] = opt_collection
                    st.session_state["hitl_decision"] = "approved"
                    st.session_state.pop("opt_report", None)
                    st.session_state["_pending_tab"] = "Test Playground"
                    st.rerun()
            with col_reject:
                if st.button("Reject — Continue improving", key="hitl_reject"):
                    st.session_state["hitl_decision"] = "rejected"
                    # Re-run optimizer from final config
                    st.session_state["opt_config"] = copy.deepcopy(report["final_config"])
                    st.session_state.pop("opt_report", None)
                    st.rerun()

            if st.session_state.get("hitl_decision") == "approved":
                st.success("Approved! Config accepted.")

        # ── Before vs After ───────────────────────────────────────────────
        if len(iterations) >= 2:
            st.subheader("4. Before vs After")

            col_before, col_after = st.columns(2)
            init_cfg = report.get("initial_config", {})
            final_cfg = report.get("final_config", {})

            with col_before:
                st.markdown("**Before**")
                st.markdown(
                    f"- k = {init_cfg.get('retrieval_k')}\n"
                    f"- chunk_size = {init_cfg.get('chunk_size')}\n"
                    f"- overlap = {init_cfg.get('chunk_overlap')}\n"
                    f"- prompt = {init_cfg.get('prompt_template')}"
                )
                if initial_score is not None:
                    st.metric("Score", f"{initial_score:.4f}")

            with col_after:
                st.markdown("**After**")
                st.markdown(
                    f"- k = {final_cfg.get('retrieval_k')}\n"
                    f"- chunk_size = {final_cfg.get('chunk_size')}\n"
                    f"- overlap = {final_cfg.get('chunk_overlap')}\n"
                    f"- prompt = {final_cfg.get('prompt_template')}"
                )
                if final_score is not None:
                    st.metric("Score", f"{final_score:.4f}")

            # Config diff
            config_changes = {}
            for key in sorted(set(init_cfg) | set(final_cfg)):
                old = init_cfg.get(key)
                new = final_cfg.get(key)
                if old != new:
                    config_changes[key] = (old, new)
            if config_changes:
                st.markdown("**Config Changes:**")
                for key, (old, new) in config_changes.items():
                    st.markdown(f"- `{key}`: {old} → {new}")

            if improvement is not None:
                if improvement > 0:
                    st.success(f"Total improvement: +{improvement:.4f}")
                elif improvement == 0:
                    st.info("No score change.")
                else:
                    st.error(f"Score decreased: {improvement:.4f}")

        st.markdown(f"**Stop Reason:** `{stop}`")
        st.markdown(f"**Total Iterations:** {report['total_iterations']}")

        # ── "Test in Playground" button ───────────────────────────────────
        st.divider()
        if st.button("Test optimized config in Playground", key="opt_to_playground"):
            final_cfg = report.get("final_config", {})
            # Determine the optimized collection name
            opt_collection = build_collection_name(final_cfg, "g1")
            st.session_state["opt_result_config"] = final_cfg
            st.session_state["opt_result_collection"] = opt_collection
            # Clear optimizer state for clean transition
            st.session_state.pop("opt_report", None)
            st.session_state.pop("hitl_decision", None)
            st.session_state["_pending_tab"] = "Test Playground"
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: History — view past runs and score trends
# ═══════════════════════════════════════════════════════════════════════════════
if active_tab == "History":
    st.title("Run History")
    st.caption(
        "All runs are logged automatically. "
        "**Pipeline Query** = single question from the Playground. "
        "**Optimization Loop** = multi-iteration improvement from the Optimizer."
    )

    # Controls
    col_refresh, col_clear = st.columns([3, 1])
    with col_refresh:
        if st.button("Refresh", key="hist_refresh"):
            st.rerun()
    with col_clear:
        if st.button("Clear History", type="secondary", key="hist_clear"):
            count = clear_history()
            st.success(f"Cleared {count} records.")
            st.rerun()

    history = load_history(limit=100)

    if not history:
        st.info("No runs recorded yet. Use the Playground or Optimizer to generate data.")
    else:
        st.markdown(f"**{len(history)} runs** (newest first)")

        # Summary stats
        query_runs = [r for r in history if r.get("run_type") == "query"]
        opt_runs = [r for r in history if r.get("run_type") == "optimization"]

        stat_cols = st.columns(4)
        with stat_cols[0]:
            st.metric("Total Runs", len(history))
        with stat_cols[1]:
            st.metric("Pipeline Queries", len(query_runs))
        with stat_cols[2]:
            st.metric("Optimization Loops", len(opt_runs))
        with stat_cols[3]:
            scores = [r["unified_score"] for r in query_runs if r.get("unified_score") is not None]
            avg_score = sum(scores) / len(scores) if scores else None
            st.metric("Avg Score", f"{avg_score:.3f}" if avg_score else "-")

        st.divider()

        # Run table
        st.subheader("All Runs")
        filter_type = st.selectbox("Filter", ["All", "Queries", "Optimizations"], key="hist_filter")
        filtered = history
        if filter_type == "Queries":
            filtered = query_runs
        elif filter_type == "Optimizations":
            filtered = opt_runs

        for record in filtered:
            run_type = record.get("run_type", "unknown")
            ts = record.get("timestamp", "")[:19].replace("T", " ")

            if run_type == "query":
                score = record.get("unified_score")
                gate = record.get("gate_decision", "?")
                q = record.get("query", "")[:60]
                score_str = f"{score:.3f}" if score is not None else "-"
                cfg = record.get("config", {})

                with st.expander(f"[{ts}] Query | score={score_str} | gate={gate} | \"{q}\""):
                    st.markdown(f"**Query:** {record.get('query', '')}")
                    q_cols = st.columns(5)
                    with q_cols[0]:
                        st.metric("Unified", score_str)
                    with q_cols[1]:
                        st.metric("Faithfulness", _fmt(record.get("faithfulness")))
                    with q_cols[2]:
                        st.metric("Relevance", _fmt(record.get("relevance")))
                    with q_cols[3]:
                        st.metric("Correctness", _fmt(record.get("correctness")))
                    with q_cols[4]:
                        st.metric("Gate", gate)

                    st.markdown(
                        f"**Config:** k={cfg.get('retrieval_k')}, "
                        f"chunk={cfg.get('chunk_size')}, "
                        f"overlap={cfg.get('chunk_overlap')}"
                    )

                    preview = record.get("answer_preview", "")
                    _render_answer_block(preview)

                    _render_judge_details(record.get("judge_details", {}), key_prefix=f"hist_q_{ts}")
                    _render_execution_trace(record.get("execution_trace", []), key_prefix=f"hist_q_{ts}")

            elif run_type == "optimization":
                f_score = record.get("final_score")
                imp = record.get("improvement")
                stop_r = record.get("stop_reason", "?")
                q = record.get("query", "")[:60]
                score_str = f"{f_score:.3f}" if f_score is not None else "-"
                imp_str = f"+{imp:.4f}" if imp is not None and imp >= 0 else (
                    f"{imp:.4f}" if imp is not None else "-"
                )

                # Not an expander — real per-iteration expanders below can't
                # nest inside one, and this mirrors the Optimizer tab's layout.
                with st.container(border=True):
                    st.markdown(f"**[{ts}] Optimization | final={score_str} | {imp_str} | stop={stop_r} | \"{q}\"**")
                    st.markdown(f"**Query:** {record.get('query', '')}")
                    o_cols = st.columns(4)
                    with o_cols[0]:
                        st.metric("Initial", _fmt(record.get("initial_score"), ".3f"))
                    with o_cols[1]:
                        st.metric("Final", score_str)
                    with o_cols[2]:
                        st.metric("Improvement", imp_str)
                    with o_cols[3]:
                        st.metric("Iterations", record.get("total_iterations", 0))

                    st.markdown(f"**Stop:** `{stop_r}`")

                    init_cfg = record.get("initial_config", {})
                    final_cfg = record.get("final_config", {})
                    st.markdown(
                        f"**Config:** k={init_cfg.get('retrieval_k')} -> {final_cfg.get('retrieval_k')}, "
                        f"chunk={init_cfg.get('chunk_size')} -> {final_cfg.get('chunk_size')}"
                    )

                    # Each iteration gets its own real expander — same UX as
                    # the Optimizer tab's "Iteration Details" section.
                    all_iters = record.get("iterations", [])
                    st.markdown("**Iteration Details**")
                    for it in all_iters:
                        it_num = it.get("iteration", 0)
                        if it_num == 0:
                            # Baseline duplicates the Playground run that seeded this optimization.
                            continue
                        it_score = it.get("unified_score", 0)
                        it_gate = it.get("gate_decision", "?")
                        it_fail = it.get("failure_type", "-")

                        prev_score = (
                            all_iters[it_num - 1]["unified_score"]
                            if 0 < it_num < len(all_iters) else None
                        )
                        delta_str = ""
                        if prev_score is not None and it_score is not None:
                            delta = it_score - prev_score
                            delta_str = f" (delta: {'+' if delta >= 0 else ''}{delta:.4f})"

                        label = f"Iteration {it_num}: score={it_score:.4f} | gate={it_gate} | failure={it_fail}{delta_str}"

                        with st.expander(label, expanded=(it_num == len(all_iters) - 1)):
                            _render_iteration_workflow(
                                it, all_iters, it_num,
                                key_prefix=f"hist_{record.get('timestamp', '')}_{it_num}",
                                final_config=record.get("final_config"),
                                query=record.get("query", ""),
                            )

