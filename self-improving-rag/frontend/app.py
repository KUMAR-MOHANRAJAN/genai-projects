"""Self-Improving RAG — Streamlit UI

Tabs:
  1. Pipeline  — upload docs, configure, ask questions, see scored results
  2. Optimize  — (placeholder) run optimization loop, view iterations
  3. History   — (placeholder) view past runs and score trends
"""

import streamlit as st
import os
import time
import copy

from utils import (
    list_corpus_files,
    save_uploaded_file,
    run_query,
    get_ground_truth_queries,
    run_optimization_ui,
    save_query_run,
    save_optimization_run,
    load_history,
    clear_history,
    CORPUS_DIR,
    DEFAULT_CONFIG,
    BAD_CONFIG,
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Self-Improving RAG",
    page_icon="🔍",
    layout="wide",
)

# ─── Sidebar: pipeline configuration ─────────────────────────────────────────
with st.sidebar:
    st.header("Pipeline Config")

    chunk_strategy = st.selectbox(
        "Chunking Strategy",
        ["fixed_size", "recursive_split", "semantic"],
        index=0,
        help="fixed_size: token windows. recursive_split: split on boundaries. semantic: embedding-based.",
    )
    chunk_size = st.slider(
        "Chunk Size (tokens)", 64, 512, DEFAULT_CONFIG["chunk_size"], step=32
    )
    chunk_overlap = st.slider(
        "Chunk Overlap (tokens)", 0, 128, DEFAULT_CONFIG["chunk_overlap"], step=16
    )
    retrieval_k = st.slider(
        "Retrieval k (top chunks)", 3, 15, DEFAULT_CONFIG["retrieval_k"]
    )
    max_context_tokens = st.slider(
        "Max Context Tokens", 1000, 8000, DEFAULT_CONFIG["max_context_tokens"], step=500
    )
    version = st.text_input("Version", value="v1")
    max_pages = st.slider("Max Pages to Ingest", 10, 200, 50, step=10)

    st.divider()
    st.caption("Config changes require re-ingestion (new version recommended).")


def build_config() -> dict:
    """Build a config dict from the current sidebar values."""
    return {
        "chunk_strategy": chunk_strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "retrieval_k": retrieval_k,
        "max_context_tokens": max_context_tokens,
        "prompt_template": "v1",
    }


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_pipeline, tab_optimize, tab_history = st.tabs(
    ["Pipeline", "Optimization", "History"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Pipeline — upload, ingest, query, see results
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pipeline:
    st.title("Self-Improving RAG")

    # ── Document Upload / Selection ───────────────────────────────────────
    col_upload, col_corpus = st.columns(2)

    with col_upload:
        st.subheader("Upload Document")
        uploaded = st.file_uploader(
            "Upload a .txt or .pdf file",
            type=["txt", "pdf"],
            help="File will be saved to the corpus/ directory.",
        )
        if uploaded:
            path = save_uploaded_file(uploaded)
            st.success(f"Saved: {uploaded.name}")
            # Store as selected file
            st.session_state["selected_file"] = uploaded.name

    with col_corpus:
        st.subheader("Select from Corpus")
        corpus_files = list_corpus_files()
        if corpus_files:
            default_idx = 0
            if "selected_file" in st.session_state:
                try:
                    default_idx = corpus_files.index(st.session_state["selected_file"])
                except ValueError:
                    default_idx = 0
            selected = st.selectbox("Available files", corpus_files, index=default_idx)
            st.session_state["selected_file"] = selected
        else:
            st.info("No files in corpus/. Upload a document first.")

    # ── Initialize Pipeline ───────────────────────────────────────────────
    st.divider()

    if st.button("Initialize Pipeline", type="primary",
                 disabled=not corpus_files and uploaded is None):
        config = build_config()
        book_path = os.path.join(CORPUS_DIR, st.session_state.get("selected_file", ""))

        if not os.path.exists(book_path):
            st.error(f"File not found: {book_path}")
        else:
            with st.status("Ingesting...", expanded=True) as status:
                st.write(f"File: {os.path.basename(book_path)}")
                st.write(f"Strategy: {config['chunk_strategy']}, "
                         f"Size: {config['chunk_size']}, "
                         f"Overlap: {config['chunk_overlap']}")

                try:
                    from ingest import ingest
                    collection = ingest(
                        strategy=config["chunk_strategy"],
                        chunk_size=config["chunk_size"],
                        chunk_overlap=config["chunk_overlap"],
                        version=version,
                        pages=max_pages,
                        book_path=book_path,
                    )
                    st.session_state["collection"] = collection
                    st.session_state["book_path"] = book_path
                    status.update(label=f"Done: {collection}", state="complete")
                except Exception as e:
                    status.update(label="Ingestion failed", state="error")
                    st.error(str(e))

    # Show current collection if set
    if "collection" in st.session_state:
        st.caption(f"Active collection: `{st.session_state['collection']}`")

    # ── Query Section ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Ask a Question")

    # Two input methods: free-text always visible, plus optional test query picker
    query = st.text_input(
        "Type your question:",
        placeholder="e.g. How do embeddings represent meaning?",
    )

    gt_queries = get_ground_truth_queries()
    with st.expander("Or pick a test query (with ground truth for correctness scoring)"):
        selected_gt = st.selectbox(
            "Test queries",
            ["(none)"] + gt_queries,
            label_visibility="collapsed",
        )
        if selected_gt != "(none)":
            query = selected_gt

    if st.button("Ask", type="primary", disabled=not query):
        config = build_config()
        book_path = st.session_state.get("book_path",
                                          os.path.join(CORPUS_DIR, corpus_files[0]) if corpus_files else "")

        with st.spinner("Running pipeline..."):
            try:
                t0 = time.time()
                state = run_query(
                    query=query,
                    config=config,
                    version=version,
                    book_path=book_path,
                    pages=max_pages,
                )
                elapsed = time.time() - t0
                st.session_state["last_result"] = state
                save_query_run(query, config, state, version=version)
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

    # ── Results Display ───────────────────────────────────────────────────
    if "last_result" in st.session_state:
        state = st.session_state["last_result"]

        st.divider()
        st.subheader("Results")

        # Answer
        st.markdown("**Answer:**")
        st.info(state.get("answer", "No answer"))

        # Score metrics row
        col1, col2, col3, col4, col5 = st.columns(5)

        def score_color(val, threshold_green=0.85, threshold_amber=0.70):
            if val is None:
                return "off"
            if val >= threshold_green:
                return "normal"
            elif val >= threshold_amber:
                return "off"      # amber — neutral
            return "inverse"      # red

        with col1:
            faith = state.get("faithfulness")
            st.metric("Faithfulness", f"{faith:.2f}" if faith is not None else "N/A")
        with col2:
            rel = state.get("relevance")
            st.metric("Relevance", f"{rel:.2f}" if rel is not None else "N/A")
        with col3:
            corr = state.get("correctness")
            st.metric("Correctness", f"{corr:.2f}" if corr is not None else "N/A")
        with col4:
            ret = state.get("retrieval_score")
            st.metric("Retrieval", f"{ret:.2f}" if ret is not None else "N/A")
        with col5:
            unified = state.get("unified_score")
            st.metric("Unified Score", f"{unified:.2f}" if unified is not None else "N/A")

        # Score band indicator
        if unified is not None:
            if unified >= 0.85:
                st.success(f"Score {unified:.2f} >= 0.85 — Autonomous accept band")
            elif unified >= 0.70:
                st.warning(f"Score {unified:.2f} in [0.70, 0.85) — HITL review band")
            else:
                st.error(f"Score {unified:.2f} < 0.70 — Needs improvement")

        # Faithfulness veto check
        if faith is not None and faith < 0.50:
            st.error("VETO: Faithfulness < 0.50 — deployment blocked regardless of unified score")

        # ── Metric Explanations ───────────────────────────────────────────
        with st.expander("How are these metrics measured?"):
            st.markdown("""
**Faithfulness** (LLM-judge, 0-1)
The LLM breaks the answer into atomic claims and checks each one against the
retrieved context. Score = supported claims / total claims.
If < 0.50 the answer is considered a hallucination and triggers a hard veto.

**Relevance** (LLM-judge, 0-1)
The LLM judges whether the answer actually addresses the question asked.
A high-faithfulness but low-relevance answer means the model quoted the
context accurately but answered the wrong question.

**Correctness** (LLM-judge, 0-1, only for test queries)
The LLM compares the answer to a known expected answer from the ground truth
set. Only available for the 6 pre-defined test queries — shows N/A for
custom questions.

**Retrieval** (keyword-based, no LLM, 0-1)
For test queries: `0.5 x Precision@k + 0.5 x Recall@k` where a chunk is
"relevant" if it contains at least one ground truth keyword.
For custom queries: falls back to average chunk similarity score from the
vector search.

**Unified Score** (computed, 0-1) — AutoRAG formula v1.2:
```
Score = 0.25 x Retrieval + 0.35 x Quality + 0.25 x Faithfulness
        - 0.10 x min(latency / 3000ms, 1)
        - 0.05 x min(cost / $0.01, 1)
```
Where Quality = `0.6 x Correctness + 0.4 x Relevance` (or just Relevance
if no ground truth). The score bands are:
- **>= 0.85**: autonomous accept (deploy)
- **0.70 - 0.84**: HITL review required
- **< 0.70**: needs improvement (triggers diagnose/improve loop)

---

**Cost** = generation LLM call only (tiktoken token count x gpt-4o-mini
price). Does NOT include the 3 LLM-judge calls — those are evaluation
overhead, not pipeline cost.

**Latency** = full pipeline wall-clock time: retrieval + context assembly +
generation + all 3 LLM-judge evaluations. This is why it's high (~10-15s) —
the judge calls dominate. The latency penalty in the unified score saturates
at 3000ms (any latency above 3s gets the same -0.10 penalty).
""")

        # Cost / latency / metadata
        col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
        with col_meta1:
            st.metric("Cost", f"${state.get('cost_usd') or 0:.4f}")
        with col_meta2:
            st.metric("Latency", f"{state.get('latency_ms', 0)}ms")
        with col_meta3:
            st.metric("Chunks", state.get("chunk_count", 0))
        with col_meta4:
            st.metric("Context Tokens", state.get("context_tokens", 0))

        # Expandable raw state
        with st.expander("Raw pipeline state"):
            # Filter out long fields for readability
            display_state = {
                k: v for k, v in state.items()
                if k not in ("chunks", "context")
            }
            st.json(display_state)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Optimization — self-improving loop UI
# ═══════════════════════════════════════════════════════════════════════════════
with tab_optimize:
    st.title("Optimization Loop")
    st.caption(
        "Run the self-improving loop: evaluate → diagnose → improve → repeat. "
        "Start with a bad config, watch the optimizer fix it automatically."
    )

    # ── Setup Section ─────────────────────────────────────────────────────
    st.subheader("1. Setup")

    # Query selection
    gt_queries_opt = get_ground_truth_queries()
    opt_query = st.selectbox(
        "Select a test query (ground truth required for correctness scoring)",
        gt_queries_opt,
        key="opt_query",
    )

    # Config presets
    st.markdown("**Starting Config**")
    col_bad, col_default = st.columns(2)
    with col_bad:
        if st.button("Use Bad Config (k=1, chunk=64)", type="secondary"):
            st.session_state["opt_config"] = copy.deepcopy(BAD_CONFIG)
    with col_default:
        if st.button("Use Default Config (k=5, chunk=256)", type="secondary"):
            st.session_state["opt_config"] = copy.deepcopy(DEFAULT_CONFIG)

    # Initialize with bad config if not set
    if "opt_config" not in st.session_state:
        st.session_state["opt_config"] = copy.deepcopy(BAD_CONFIG)

    opt_cfg = st.session_state["opt_config"]

    # Editable config display
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        opt_cfg["retrieval_k"] = st.number_input(
            "retrieval_k", min_value=1, max_value=15,
            value=opt_cfg.get("retrieval_k", 1), key="opt_k",
        )
    with col_c2:
        opt_cfg["chunk_size"] = st.number_input(
            "chunk_size", min_value=32, max_value=512, step=32,
            value=opt_cfg.get("chunk_size", 64), key="opt_chunk",
        )
    with col_c3:
        opt_cfg["chunk_overlap"] = st.number_input(
            "chunk_overlap", min_value=0, max_value=128, step=16,
            value=opt_cfg.get("chunk_overlap", 0), key="opt_overlap",
        )
    with col_c4:
        opt_cfg["max_context_tokens"] = st.number_input(
            "max_context_tokens", min_value=1000, max_value=8000, step=500,
            value=opt_cfg.get("max_context_tokens", 4000), key="opt_ctx",
        )

    # Target score and max iterations
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        opt_target = st.number_input(
            "Target Score", min_value=0.50, max_value=1.00, step=0.05,
            value=0.85, key="opt_target",
        )
    with col_t2:
        opt_max_iter = st.number_input(
            "Max Iterations", min_value=1, max_value=10,
            value=3, key="opt_max_iter",
        )

    st.divider()

    # ── Baseline Run Section ──────────────────────────────────────────────
    st.subheader("2. Baseline Run")
    st.caption("Run the pipeline once with the starting config to see the initial score.")

    if st.button("Run Baseline", type="primary", key="run_baseline"):
        with st.spinner("Running baseline pipeline..."):
            try:
                from utils import run_query as rq
                t0 = time.time()
                baseline_state = rq(
                    query=opt_query,
                    config=copy.deepcopy(opt_cfg),
                    version="g1",
                    pages=10,
                )
                elapsed = time.time() - t0
                baseline_state["_elapsed"] = round(elapsed, 2)
                st.session_state["opt_baseline"] = baseline_state
            except Exception as e:
                st.error(f"Baseline error: {e}")

    # Display baseline results
    if "opt_baseline" in st.session_state:
        bl = st.session_state["opt_baseline"]

        st.markdown("**Answer:**")
        st.info(bl.get("answer", "No answer"))

        bl_cols = st.columns(6)
        metrics = [
            ("Faithfulness", "faithfulness"),
            ("Relevance", "relevance"),
            ("Correctness", "correctness"),
            ("Retrieval", "retrieval_score"),
            ("Unified", "unified_score"),
            ("Gate", "gate_decision"),
        ]
        for col, (label, key) in zip(bl_cols, metrics):
            with col:
                val = bl.get(key)
                if key == "gate_decision":
                    st.metric(label, val or "N/A")
                elif val is not None:
                    st.metric(label, f"{val:.2f}")
                else:
                    st.metric(label, "N/A")

        bl_meta = st.columns(4)
        with bl_meta[0]:
            st.metric("Cost", f"${bl.get('cost_usd') or 0:.4f}")
        with bl_meta[1]:
            st.metric("Latency", f"{bl.get('latency_ms', 0)}ms")
        with bl_meta[2]:
            st.metric("Chunks", bl.get("chunk_count", 0))
        with bl_meta[3]:
            st.metric("Wall Clock", f"{bl.get('_elapsed') or 0:.1f}s")

        with st.expander("Retrieved Chunks"):
            chunks = bl.get("retrieved_chunks") or bl.get("chunks", [])
            if chunks:
                for i, ch in enumerate(chunks):
                    if isinstance(ch, dict):
                        st.markdown(f"**Chunk {i+1}** (score: {ch.get('score', 'N/A')})")
                        st.text(ch.get("text", str(ch))[:500])
                    else:
                        st.text(str(ch)[:500])
            else:
                st.caption("No chunks available.")

    st.divider()

    # ── Optimizer Run Section ─────────────────────────────────────────────
    st.subheader("3. Run Optimizer")
    st.caption(
        "The optimizer dispatches the full graph (pipeline → evaluator → diagnoser → "
        "improver) repeatedly, applying config fixes until a stop condition fires."
    )

    if st.button("Run Optimizer", type="primary", key="run_optimizer"):
        with st.spinner("Running optimization loop... (this makes multiple LLM calls)"):
            try:
                report = run_optimization_ui(
                    query=opt_query,
                    config=copy.deepcopy(opt_cfg),
                    version="g1",
                    target_score=opt_target,
                    max_iterations=opt_max_iter,
                )
                st.session_state["opt_report"] = report
                save_optimization_run(opt_query, report)
            except Exception as e:
                st.error(f"Optimizer error: {e}")

    # Display optimizer results
    if "opt_report" in st.session_state:
        report = st.session_state["opt_report"]
        iterations = report.get("iterations", [])

        # Summary banner
        stop = report.get("stop_reason", "unknown")
        improvement = report.get("improvement")
        final_score = report.get("final_score")
        initial_score = report.get("initial_score")

        if stop == "target_reached":
            st.success(
                f"Target reached! Score {final_score:.4f} >= {opt_target:.2f} "
                f"in {report['total_iterations']} iteration(s). "
                f"Improvement: +{improvement:.4f}"
            )
        elif stop == "hitl_required":
            st.warning(
                f"HITL Required — Score {final_score:.4f} is in the gray band "
                f"[0.70, 0.85). Human review needed. "
                f"Improvement so far: +{improvement:.4f}" if improvement else
                f"HITL Required — Score {final_score:.4f} is in the gray band. "
                f"Human review needed."
            )
        elif stop == "blocked_faithfulness":
            st.error(
                f"Blocked — Faithfulness below safety floor (0.50). "
                f"Score: {final_score:.4f}. Cannot proceed."
            )
        elif stop == "no_improvement":
            st.warning(
                f"Plateau — 3 consecutive iterations with delta < 0.01. "
                f"Final score: {final_score:.4f}."
            )
        elif stop == "no_candidates":
            st.warning(
                f"No candidates — the improver could not generate config changes. "
                f"Final score: {final_score:.4f}."
            )
        else:
            st.info(
                f"Max iterations reached ({report['total_iterations']}). "
                f"Final score: {final_score:.4f}."
            )

        # Per-iteration details
        st.markdown("**Iteration Details**")
        for rec in iterations:
            it_num = rec["iteration"]
            it_score = rec.get("unified_score", 0)
            it_gate = rec.get("gate_decision", "?")
            it_fail = rec.get("failure_type", "-")

            # Score delta from previous
            prev_score = iterations[it_num - 2]["unified_score"] if it_num > 1 else None
            delta_str = ""
            if prev_score is not None and it_score is not None:
                delta = it_score - prev_score
                delta_str = f" (delta: {'+' if delta >= 0 else ''}{delta:.4f})"

            with st.expander(
                f"Iteration {it_num}: score={it_score:.4f}  |  "
                f"gate={it_gate}  |  failure={it_fail}{delta_str}",
                expanded=(it_num == len(iterations)),  # expand last
            ):
                # Metrics row
                it_cols = st.columns(5)
                with it_cols[0]:
                    st.metric("Faithfulness", f"{rec.get('faithfulness') or 0:.2f}")
                with it_cols[1]:
                    st.metric("Relevance", f"{rec.get('relevance') or 0:.2f}")
                with it_cols[2]:
                    corr = rec.get("correctness")
                    st.metric("Correctness", f"{corr:.2f}" if corr is not None else "N/A")
                with it_cols[3]:
                    st.metric("Retrieval", f"{rec.get('retrieval_score') or 0:.2f}")
                with it_cols[4]:
                    st.metric("Unified", f"{it_score:.4f}")

                # Config used
                cfg = rec.get("config", {})
                st.markdown(
                    f"**Config:** k={cfg.get('retrieval_k')}, "
                    f"chunk={cfg.get('chunk_size')}, "
                    f"overlap={cfg.get('chunk_overlap')}, "
                    f"prompt={cfg.get('prompt_template')}"
                )

                # Diagnosis + fix applied
                if rec.get("remediation_hint"):
                    st.markdown(f"**Diagnosis:** {it_fail} — {rec['remediation_hint']}")

                variant = rec.get("applied_variant")
                if variant:
                    st.markdown(
                        f"**Fix Applied:** {variant.get('rationale', 'N/A')}"
                    )
                    st.markdown(f"**Delta:** `{variant.get('delta', {})}`")

                # Answer
                ans = rec.get("answer", "")
                if ans:
                    st.markdown("**Answer:**")
                    st.text(ans[:1000])

                # Metadata
                meta_cols = st.columns(3)
                with meta_cols[0]:
                    st.metric("Cost", f"${rec.get('cost_usd') or 0:.4f}")
                with meta_cols[1]:
                    st.metric("Latency", f"{rec.get('latency_ms', 0)}ms")
                with meta_cols[2]:
                    st.metric("Chunks", rec.get("chunk_count", 0))

                # Retrieved chunks
                r_chunks = rec.get("retrieved_chunks", [])
                if r_chunks:
                    with st.expander(f"Retrieved Chunks ({len(r_chunks)})"):
                        for ci, ch in enumerate(r_chunks):
                            if isinstance(ch, dict):
                                st.markdown(f"**Chunk {ci+1}** (score: {ch.get('score', 'N/A')})")
                                st.text(ch.get("text", str(ch))[:500])
                            else:
                                st.text(str(ch)[:500])

        st.divider()

        # ── HITL Decision Section ─────────────────────────────────────────
        if stop == "hitl_required":
            st.subheader("4. Human-in-the-Loop Decision")
            st.markdown(
                f"The optimizer stopped because the score **{final_score:.4f}** "
                f"is in the HITL band [0.70, 0.85). You can:"
            )

            col_approve, col_reject = st.columns(2)
            with col_approve:
                if st.button("Approve — Accept this result", type="primary",
                             key="hitl_approve"):
                    st.session_state["hitl_decision"] = "approved"
            with col_reject:
                if st.button("Reject — Continue improving", type="secondary",
                             key="hitl_reject"):
                    st.session_state["hitl_decision"] = "pending_confirm"

            if st.session_state.get("hitl_decision") == "approved":
                st.success(
                    "Approved! The current config and result are accepted. "
                    "See the comparison below."
                )
            elif st.session_state.get("hitl_decision") == "pending_confirm":
                st.warning(
                    "Are you sure? This will run the optimizer again from the "
                    f"current best config (k={report['final_config'].get('retrieval_k')}, "
                    f"chunk={report['final_config'].get('chunk_size')}) "
                    f"for up to {opt_max_iter} more iterations."
                )
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("Yes, continue optimizing", type="primary",
                                 key="hitl_confirm_yes"):
                        st.session_state["hitl_decision"] = "running"
                        st.rerun()
                with col_no:
                    if st.button("No, cancel", type="secondary",
                                 key="hitl_confirm_no"):
                        st.session_state["hitl_decision"] = None
                        st.rerun()
            elif st.session_state.get("hitl_decision") == "running":
                # Auto-run the optimizer from the current best config
                new_start_config = copy.deepcopy(report["final_config"])
                st.session_state["opt_config"] = new_start_config
                with st.spinner("Running optimizer from current best config..."):
                    try:
                        new_report = run_optimization_ui(
                            query=opt_query,
                            config=new_start_config,
                            version="g1",
                            target_score=opt_target,
                            max_iterations=opt_max_iter,
                        )
                        st.session_state["opt_report"] = new_report
                        save_optimization_run(opt_query, new_report)
                        st.session_state["hitl_decision"] = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Optimizer error: {e}")
                        st.session_state["hitl_decision"] = None

        # ── Before vs After Comparison ────────────────────────────────────
        st.subheader("5. Before vs After")

        col_before, col_after = st.columns(2)

        init_cfg = report.get("initial_config", {})
        final_cfg = report.get("final_config", {})

        with col_before:
            st.markdown("**Before (Initial Config)**")
            st.markdown(
                f"- k = {init_cfg.get('retrieval_k')}\n"
                f"- chunk_size = {init_cfg.get('chunk_size')}\n"
                f"- chunk_overlap = {init_cfg.get('chunk_overlap')}\n"
                f"- prompt = {init_cfg.get('prompt_template')}"
            )
            if initial_score is not None:
                st.metric("Score", f"{initial_score:.4f}")
            if iterations:
                first_ans = iterations[0].get("answer", "")
                if first_ans:
                    st.markdown("**Answer:**")
                    st.text(first_ans[:500])
                first_chunks = iterations[0].get("retrieved_chunks", [])
                if first_chunks:
                    with st.expander(f"Retrieved Chunks ({len(first_chunks)})"):
                        for ci, ch in enumerate(first_chunks):
                            if isinstance(ch, dict):
                                score_val = ch.get("score", "N/A")
                                st.markdown(f"**Chunk {ci+1}** (similarity: {score_val})")
                                meta = {k: v for k, v in ch.items() if k not in ("text", "score")}
                                if meta:
                                    st.caption(f"Metadata: {meta}")
                                st.text(ch.get("text", str(ch))[:500])
                            else:
                                st.text(str(ch)[:500])

        with col_after:
            st.markdown("**After (Final Config)**")
            st.markdown(
                f"- k = {final_cfg.get('retrieval_k')}\n"
                f"- chunk_size = {final_cfg.get('chunk_size')}\n"
                f"- chunk_overlap = {final_cfg.get('chunk_overlap')}\n"
                f"- prompt = {final_cfg.get('prompt_template')}"
            )
            if final_score is not None:
                st.metric("Score", f"{final_score:.4f}")
            if iterations:
                last_ans = iterations[-1].get("answer", "")
                if last_ans:
                    st.markdown("**Answer:**")
                    st.text(last_ans[:500])
                last_chunks = iterations[-1].get("retrieved_chunks", [])
                if last_chunks:
                    with st.expander(f"Retrieved Chunks ({len(last_chunks)})"):
                        for ci, ch in enumerate(last_chunks):
                            if isinstance(ch, dict):
                                score_val = ch.get("score", "N/A")
                                st.markdown(f"**Chunk {ci+1}** (similarity: {score_val})")
                                meta = {k: v for k, v in ch.items() if k not in ("text", "score")}
                                if meta:
                                    st.caption(f"Metadata: {meta}")
                                st.text(ch.get("text", str(ch))[:500])
                            else:
                                st.text(str(ch)[:500])

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

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: History — view past runs and score trends
# ═══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.title("Run History")
    st.caption(
        "All runs are logged automatically. "
        "**Pipeline Query** = single question from the Pipeline tab. "
        "**Optimization Loop** = multi-iteration improvement from the Optimization tab."
    )

    # Controls row
    col_refresh, col_clear = st.columns([3, 1])
    with col_refresh:
        if st.button("Refresh", key="hist_refresh"):
            st.rerun()
    with col_clear:
        if st.button("Clear History", type="secondary", key="hist_clear"):
            count = clear_history()
            st.success(f"Cleared {count} records.")
            st.rerun()

    # Helper for displaying values — show dash for missing/None/zero
    def _fmt(val, fmt=".2f", prefix="", suffix=""):
        if val is None:
            return "—"
        if isinstance(val, (int, float)) and val == 0:
            return "—"
        return f"{prefix}{val:{fmt}}{suffix}"

    # Load history
    history = load_history(limit=100)

    if not history:
        st.info("No runs recorded yet. Use the Pipeline or Optimization tab to generate data.")
    else:
        st.markdown(f"**{len(history)} runs** (newest first)")

        # ── Summary stats ─────────────────────────────────────────────
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
            st.metric("Avg Query Score", f"{avg_score:.3f}" if avg_score else "—")

        # ── Score trend chart ─────────────────────────────────────────
        if query_runs:
            st.subheader("Score Trend")
            import pandas as pd

            chart_data = []
            for r in reversed(query_runs):  # oldest first for chart
                if r.get("unified_score") is not None:
                    chart_data.append({
                        "timestamp": r.get("timestamp", "")[:19].replace("T", " "),
                        "unified_score": r["unified_score"],
                        "faithfulness": r.get("faithfulness") or 0,
                        "retrieval_score": r.get("retrieval_score") or 0,
                    })

            if chart_data:
                df = pd.DataFrame(chart_data)
                st.line_chart(df, x="timestamp", y=["unified_score", "faithfulness", "retrieval_score"])

        st.divider()

        # ── Run table ─────────────────────────────────────────────────
        st.subheader("All Runs")

        # Filter
        filter_type = st.selectbox(
            "Filter by type",
            ["All", "Pipeline Queries", "Optimization Loops"],
            key="hist_filter",
        )

        filtered = history
        if filter_type == "Pipeline Queries":
            filtered = query_runs
        elif filter_type == "Optimization Loops":
            filtered = opt_runs

        for i, record in enumerate(filtered):
            run_type = record.get("run_type", "unknown")
            ts = record.get("timestamp", "")[:19].replace("T", " ")

            if run_type == "query":
                score = record.get("unified_score")
                gate = record.get("gate_decision", "?")
                q = record.get("query", "")[:60]
                score_str = f"{score:.3f}" if score is not None else "—"
                cfg = record.get("config", {})

                with st.expander(
                    f"[{ts}]  Pipeline Query  |  score={score_str}  |  "
                    f"gate={gate}  |  \"{q}\"",
                    expanded=False,
                ):
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
                        f"overlap={cfg.get('chunk_overlap')}, "
                        f"prompt={cfg.get('prompt_template')}"
                    )

                    meta_cols = st.columns(3)
                    with meta_cols[0]:
                        st.metric("Cost", _fmt(record.get("cost_usd"), ".4f", prefix="$"))
                    with meta_cols[1]:
                        st.metric("Latency", _fmt(record.get("latency_ms"), "d", suffix="ms"))
                    with meta_cols[2]:
                        st.metric("Chunks", _fmt(record.get("chunk_count"), "d"))

                    preview = record.get("answer_preview", "")
                    if preview:
                        st.markdown("**Answer preview:**")
                        st.text(preview)

            elif run_type == "optimization":
                f_score = record.get("final_score")
                imp = record.get("improvement")
                stop_r = record.get("stop_reason", "?")
                q = record.get("query", "")[:60]
                score_str = f"{f_score:.3f}" if f_score is not None else "—"
                imp_str = f"+{imp:.4f}" if imp is not None and imp >= 0 else (
                    f"{imp:.4f}" if imp is not None else "—"
                )

                with st.expander(
                    f"[{ts}]  Optimization Loop  |  final={score_str}  |  "
                    f"{imp_str}  |  stop={stop_r}  |  \"{q}\"",
                    expanded=False,
                ):
                    st.markdown(f"**Query:** {record.get('query', '')}")

                    o_cols = st.columns(4)
                    with o_cols[0]:
                        init_s = record.get("initial_score")
                        st.metric("Initial Score", f"{init_s:.3f}" if init_s is not None else "—")
                    with o_cols[1]:
                        st.metric("Final Score", score_str)
                    with o_cols[2]:
                        st.metric("Improvement", imp_str)
                    with o_cols[3]:
                        st.metric("Iterations", record.get("total_iterations", 0))

                    st.markdown(f"**Stop Reason:** `{stop_r}`")

                    # Config comparison
                    init_cfg = record.get("initial_config", {})
                    final_cfg = record.get("final_config", {})
                    cfg_cols = st.columns(2)
                    with cfg_cols[0]:
                        st.markdown("**Initial Config:**")
                        st.markdown(
                            f"- k={init_cfg.get('retrieval_k')}, "
                            f"chunk={init_cfg.get('chunk_size')}, "
                            f"overlap={init_cfg.get('chunk_overlap')}"
                        )
                    with cfg_cols[1]:
                        st.markdown("**Final Config:**")
                        st.markdown(
                            f"- k={final_cfg.get('retrieval_k')}, "
                            f"chunk={final_cfg.get('chunk_size')}, "
                            f"overlap={final_cfg.get('chunk_overlap')}"
                        )

                    # Per-iteration breakdown
                    saved_iters = record.get("iterations", [])
                    if saved_iters:
                        st.markdown("---")
                        st.markdown("**Per-Iteration Decision Chain:**")
                        for it_rec in saved_iters:
                            it_n = it_rec.get("iteration", "?")
                            it_s = it_rec.get("unified_score")
                            it_gate = it_rec.get("gate_decision", "?")
                            it_fail = it_rec.get("failure_type", "—")
                            it_s_str = f"{it_s:.4f}" if it_s is not None else "—"
                            it_cfg = it_rec.get("config", {})

                            st.markdown(
                                f"**Iteration {it_n}:** score={it_s_str} | "
                                f"gate={it_gate} | failure={it_fail}"
                            )

                            # Scores detail
                            it_detail_cols = st.columns(5)
                            with it_detail_cols[0]:
                                st.metric("Faith", _fmt(it_rec.get("faithfulness")),
                                          label_visibility="visible")
                            with it_detail_cols[1]:
                                st.metric("Relev", _fmt(it_rec.get("relevance")),
                                          label_visibility="visible")
                            with it_detail_cols[2]:
                                st.metric("Corr", _fmt(it_rec.get("correctness")),
                                          label_visibility="visible")
                            with it_detail_cols[3]:
                                st.metric("Retrieval", _fmt(it_rec.get("retrieval_score")),
                                          label_visibility="visible")
                            with it_detail_cols[4]:
                                st.metric("Config k", it_cfg.get("retrieval_k", "—"),
                                          label_visibility="visible")

                            # Diagnosis and fix
                            if it_rec.get("remediation_hint"):
                                st.caption(
                                    f"Diagnosis: {it_fail} — {it_rec['remediation_hint']}"
                                )
                            variant = it_rec.get("applied_variant")
                            if variant:
                                st.caption(
                                    f"Fix: {variant.get('rationale', '—')} | "
                                    f"Delta: {variant.get('delta', {})}"
                                )

                            # Answer preview
                            it_preview = it_rec.get("answer_preview", "")
                            if it_preview:
                                with st.expander(f"Answer (iteration {it_n})"):
                                    st.text(it_preview)
                    else:
                        st.caption("No per-iteration details saved for this run.")
