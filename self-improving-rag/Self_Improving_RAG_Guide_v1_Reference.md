# Self-Improving RAG: Technical Guide

**Version**: v1.0 | **Audience**: ML/AI Engineers, Data Engineers, Technical Reviewers

---

## Table of Contents

- [Chapter 1: Introduction](#chapter-1-introduction)
- [Chapter 2: Architecture at a Glance](#chapter-2-architecture-at-a-glance)
- [Chapter 3: Core Concepts](#chapter-3-core-concepts)
- [Chapter 4: The Agent Flow](#chapter-4-the-agent-flow)
  - [Agent 1: Pipeline Agent](#agent-1-pipeline-agent)
  - [Agent 2: Evaluator (How the Pipeline Is Scored)](#agent-2-evaluator-how-the-pipeline-is-scored)
  - [Agent 3: Diagnoser (Why a Pipeline Underperforms)](#agent-3-diagnoser-why-a-pipeline-underperforms)
  - [Agent 4: Improver (Applying a Targeted Fix)](#agent-4-improver-applying-a-targeted-fix)
- [Chapter 5: The Optimizer Loop](#chapter-5-the-optimizer-loop)
- [Chapter 6: Using the Streamlit UI](#chapter-6-using-the-streamlit-ui)
- [Chapter 7: Configuration and Ingestion](#chapter-7-configuration-and-ingestion)
- [Chapter 8: Troubleshooting and FAQ](#chapter-8-troubleshooting-and-faq)

---

# Chapter 1: Introduction

### What Problem It Solves

A RAG pipeline answers questions by retrieving relevant text from documents and passing it to a
language model. Getting one working is straightforward. Getting one that retrieves the *right*
passages, doesn't hallucinate, and improves itself when it fails. That requires a structured
evaluation and improvement loop.

This system implements that loop: **evaluate** the pipeline's output, **diagnose** why it
underperforms, **improve** the configuration, and **re-evaluate**, in a controlled manner,
bounded by configurable iteration limits, before escalating to a human reviewer.

```mermaid
flowchart LR
    A[Query + Config] --> B[Pipeline]
    B --> C[Evaluator]
    C -->|Score too low| D[Diagnoser]
    D --> E[Improver]
    E -->|New config| B
    C -->|Score acceptable| F[Done]
```

### System Overview

Self-Improving RAG is an end-to-end RAG system with autonomous quality control. It implements
five core agents, each with a single responsibility, orchestrated as a LangGraph state machine:

| Agent | Responsibility |
|-------|---------------|
| **Pipeline** | Retrieves chunks, assembles context, generates an answer |
| **Evaluator** | Grades the answer (faithfulness, relevance, correctness) and determines the gate decision |
| **Diagnoser** | Classifies the root cause of underperformance (one of 5 failure types) |
| **Improver** | Proposes a specific configuration change based on the diagnosis |
| **Optimizer** | Orchestrates the improvement loop: baseline → trial iterations → convergence |

### Roadmap

The current release covers the core evaluation and self-improvement loop. The following
capabilities are planned for subsequent phases:

| Capability | Status |
|-----------|--------|
| Evaluate → Diagnose → Improve loop | Implemented |
| HITL (Human-in-the-Loop) gate | Implemented |
| Automated ingestion and re-ingestion | Implemented |
| Production deployment (canary/blue-green rollout) | Planned for Phase 2 |
| Live monitoring and alerting (Observer agent) | Planned for Phase 2 |
| F-06 Quality Drift detection | Planned for Phase 2 |
| MLflow experiment tracking | Planned for Phase 3 |
| Prometheus/Grafana metrics dashboard | Planned for Phase 3 |
| Audit logging and compliance trail | Planned for Phase 3 |

[Back to Top](#table-of-contents)

---

# Chapter 2: Architecture at a Glance

### Two-Layer Design

The system uses a two-layer architecture that separates a single evaluation attempt (the graph)
from retry orchestration (the optimizer):

### Layer 1: The Graph (single pass)

Each graph invocation runs the full pipeline once: retrieve → generate → evaluate → route.
The graph never loops internally. It produces a result and exits.

```mermaid
flowchart TD
    START([START]) --> pipeline[Pipeline Agent]
    pipeline --> evaluator[Evaluator Agent]
    evaluator --> route{Gate Decision}

    route -->|"score >= 0.85\nAND faith >= 0.50"| deploy_end([END: deploy_eligible])

    route -->|"0.70 <= score < 0.85\nAND faith >= 0.50"| hitl[HITL Gate\ninterrupt]
    hitl -->|approve| hitl_end([END: approved])
    hitl -->|reject| diag_hitl[Diagnoser Agent]
    diag_hitl --> imp_hitl[Improver Agent]
    imp_hitl --> hitl_done([END: new config proposed])

    route -->|"score < 0.70\nOR faith < 0.50"| diag[Diagnoser Agent]
    diag --> imp[Improver Agent]
    imp --> block_end([END: new config proposed])
```

### Layer 2: The Optimizer (external loop)

The Optimizer sits outside the graph and controls retries. It invokes the graph repeatedly,
extracting the Improver's suggested configuration after each iteration and feeding it as the
next iteration's input.

```mermaid
flowchart TD
    A[Initial Config] --> B[Invoke Graph\nbaseline]
    B --> C{Stop Condition Met?}
    C -->|target_reached| D([Done: target met])
    C -->|hitl_required| E([Done: human decides])
    C -->|blocked_faithfulness| F([Done: safety block])
    C -->|no| G[Extract new config\nfrom Improver]
    G --> H[Invoke Graph\niteration N]
    H --> I{Stop Condition Met?}
    I -->|yes| J([Done: return report])
    I -->|no, iterations left| G
    I -->|max_iterations| K([Done: budget exhausted])
```

### Why Linear, Not Circular

A circular graph (where the Improver loops back to the Pipeline Agent) risks infinite loops
and makes debugging harder. The linear design means each graph invocation is a pure function:
same input → same output. The Optimizer owns the retry logic and applies its own stop
conditions between iterations.

[Back to Top](#table-of-contents)

---

# Chapter 3: Core Concepts

### Unified Score

A single number (0 to 1) that answers: *"Overall, how good is this pipeline?"*

It blends five components using a weighted formula:

```mermaid
flowchart LR
    R[Retrieval Recall\nweight: 0.30] --> U((Unified\nScore))
    Q[Quality Score\nweight: 0.25] --> U
    F[Faithfulness\nweight: 0.25] --> U
    L[Latency Penalty\nweight: 0.10] --> U
    C[Cost Penalty\nweight: 0.10] --> U
```

```
unified = 0.30 * recall_score
        + 0.25 * quality_score
        + 0.25 * faithfulness
        + 0.10 * latency_penalty
        + 0.10 * cost_penalty
```

Where:
- **recall_score**: keyword-based retrieval precision/recall (no LLM call)
- **quality_score**: average of relevance and correctness (LLM judge)
- **faithfulness**: are claims grounded in the retrieved context? (LLM judge)
- **latency_penalty**: 1.0 if fast, decreasing if slow (saturates at 3000ms)
- **cost_penalty**: 1.0 if cheap, decreasing with cost

### Gate Decision

The Evaluator uses the unified score and faithfulness to produce a gate decision:

```mermaid
flowchart TD
    E[Pipeline Evaluation] --> F{"Faithfulness\n< 0.50?"}
    F -->|Yes| VETO[hard_block\nSafety Veto]
    F -->|No| U{"Unified Score\n>= 0.85?"}
    U -->|Yes| DEPLOY[deploy_eligible\nAuto-accept]
    U -->|No| H{"Unified Score\n>= 0.70?"}
    H -->|Yes| HITL[hitl_required\nHuman review]
    H -->|No| BLOCK[hard_block\nAuto-reject]
```

| Unified Score | Faithfulness | Decision |
|---------------|-------------|----------|
| >= 0.85 | >= 0.50 | `deploy_eligible` (autonomous acceptance) |
| 0.70 - 0.84 | >= 0.50 | `hitl_required` (requires human approval) |
| < 0.70 | (any) | `hard_block` (rejected, triggers improvement loop) |
| (any) | < 0.50 | `hard_block` (safety veto, non-negotiable) |

**The faithfulness veto is non-negotiable.** A pipeline that scores 0.90 overall but
hallucinates (faithfulness < 0.50) is blocked. High quality scores never excuse invented facts.

### Failure Types (F-01 to F-05)

When the gate blocks a pipeline, the Diagnoser classifies the root cause:

| Code | Name | Trigger | Meaning |
|------|------|---------|---------|
| F-01 | Retrieval Miss | No chunks retrieved or retrieval_score < 0.30 | Failed to find relevant passages |
| F-02 | Context Overflow | Context fills 95%+ of token budget | Retrieved too much, relevant material was truncated |
| F-03 | Hallucination | faithfulness < 0.50 | Answer contains claims not supported by the context |
| F-04 | Answer Incomplete | (catch-all) | Retrieved relevant material but the answer didn't use it |
| F-05 | Latency Spike | latency > 3000ms | Response time exceeds the acceptable threshold |

**Priority order matters.** The Diagnoser checks F-03 first (safety), then F-01, F-02, F-05,
and F-04 last (catch-all). First match wins. This prevents a hallucinating pipeline from being
misdiagnosed as merely "slow."

> **F-06 (Quality Drift)**, detecting score degradation over time across production traffic,
> is planned for Phase 2, alongside the Observer agent and live monitoring capabilities.
> The configuration scaffolding (`DRIFT_WINDOW`, `DRIFT_MIN_DROP`) is already defined.

### LLM Judges

Three separate LLM calls evaluate the pipeline's output. The judge model is architecturally
independent from the generation model, allowing them to be swapped or scaled separately:

| Judge | What it scores | Scale |
|-------|---------------|-------|
| Faithfulness | Are all claims in the answer supported by the retrieved context? | 0.0 - 1.0 |
| Relevance | Does the answer address the question asked? | 0.0 - 1.0 |
| Correctness | Does the answer match the expected ground truth answer? | 0.0 - 1.0 |

Correctness runs only when ground truth exists for the query (from the golden dataset). For
ad-hoc questions, it is skipped.

[Back to Top](#table-of-contents)

---

# Chapter 4: The Agent Flow

This chapter traces data through each agent: what it receives, what it does, and what it
returns. Each agent is implemented as a LangGraph node with a defined state contract.

### Workflow at a Glance

```mermaid
sequenceDiagram
    participant Q as Query
    participant P as Pipeline Agent
    participant E as Evaluator Agent
    participant D as Diagnoser Agent
    participant I as Improver Agent

    Q->>P: query + config
    P->>P: Ingest (if collection empty)
    P->>P: Retrieve top-k chunks
    P->>P: Assemble context
    P->>P: Generate answer (LLM)
    P->>E: answer, chunks, context, cost, latency

    E->>E: Retrieval metrics (keyword, no LLM)
    E->>E: LLM Judge: faithfulness
    E->>E: LLM Judge: relevance
    E->>E: LLM Judge: correctness
    E->>E: Compute unified score
    E->>E: Gate decision

    alt deploy_eligible (score >= 0.85)
        E-->>Q: Done (auto-accept)
    else hitl_required (0.70 <= score < 0.85)
        E-->>Q: Paused (awaiting human review)
    else hard_block (score < 0.70 or faith < 0.50)
        E->>D: scores, metrics, config
        D->>D: Rule cascade (F-03 then F-01 then F-02 then F-05 then F-04)
        D->>I: failure_type, confidence, remediation hint
        I->>I: Lookup playbook for failure_type
        I->>I: Apply delta, clamp to bounds
        I-->>Q: New config proposed
    end
```

---

### Agent 1: Pipeline Agent

**File:** `agents/graph.py`, `pipeline_node()`

**Input:** query, config (chunk_size, retrieval_k, etc.), version

```mermaid
flowchart LR
    A[Query] --> B[Embed Query]
    B --> C[Search ChromaDB\ntop-k chunks]
    C --> D[Assemble Context\nrespect token budget]
    D --> E[Generate Answer\nLLM call]
    E --> F[answer + chunks\n+ cost + latency]
```

**Process:**
1. Build collection name: `rag_{version}_{strategy}_{chunk_size}`
2. Auto-ingest if collection is empty (chunk the corpus, embed, store in ChromaDB)
3. Embed the query, search ChromaDB for top-k chunks
4. Assemble context from retrieved chunks (respecting `max_context_tokens`)
5. Generate answer using the LLM with the assembled context + query

**Output:** answer, retrieved_chunks, context, context_tokens, latency_ms, cost_usd

---

### Agent 2: Evaluator (How the Pipeline Is Scored)

**File:** `agents/evaluator.py`, `evaluator_node()`

The Evaluator executes **5 jobs** in sequence. Two are free (no LLM), three require LLM judge calls:

```mermaid
flowchart TD
    A[Pipeline Output] --> B[Job 1: Retrieval Metrics\nkeyword precision/recall\nNO LLM call]
    A --> C[Job 2: Faithfulness Judge\nLLM call]
    A --> D[Job 3: Relevance Judge\nLLM call]
    A --> E["Job 4: Correctness Judge\nLLM call (only with ground truth)"]

    B --> F[Job 5: Unified Score\nWeighted formula]
    C --> F
    D --> F
    E --> F

    F --> G{Gate Decision}
    G -->|">= 0.85 AND faith >= 0.50"| H[deploy_eligible]
    G -->|"0.70-0.84 AND faith >= 0.50"| I[hitl_required]
    G -->|"< 0.70 OR faith < 0.50"| J[hard_block]
```

**Input:** query, answer, context, retrieved_chunks, config, latency_ms, cost_usd

**The 5 Jobs:**

| # | Job | LLM? | What it computes |
|---|-----|------|-----------------|
| 1 | Retrieval Metrics | No | Keyword precision@k and recall@k against ground truth keywords |
| 2 | Faithfulness Judge | Yes | "How many claims in the answer are supported by the context?" |
| 3 | Relevance Judge | Yes | "Does the answer address the question asked?" |
| 4 | Correctness Judge | Yes* | "Does the answer match the expected answer?" |
| 5 | Unified Score | No | Weighted formula combining all metrics → gate decision |

*Correctness only runs when ground truth exists for the query.

**Why this order matters:** Retrieval metrics are computed first because they are free and
instant. The three LLM judges run next (each is a separate API call). The unified score
is computed last because it requires all preceding inputs.

**Output:** unified_score, faithfulness, relevance, correctness, gate_decision, gate_reason,
judge_reasoning

---

### Agent 3: Diagnoser (Why a Pipeline Underperforms)

**File:** `agents/diagnoser.py`, `diagnoser_node()`

The Diagnoser activates only when the gate blocks a pipeline. It applies a **deterministic
rule cascade** with no LLM call. First matching rule wins.

```mermaid
flowchart TD
    A[Blocked Pipeline] --> B{"F-03: Hallucination?\nfaithfulness < 0.50"}
    B -->|Yes| F03[F-03 Hallucination\nconfidence: 0.97]
    B -->|No| C{"F-01: Retrieval Miss?\nno chunks OR\nretrieval_score < 0.30"}
    C -->|Yes| F01[F-01 Retrieval Miss\nconfidence: 0.90]
    C -->|No| D{"F-02: Context Overflow?\ncontext_tokens >= 95%\nof max_context_tokens"}
    D -->|Yes| F02[F-02 Context Overflow\nconfidence: 0.88]
    D -->|No| E{"F-05: Latency Spike?\nlatency > 3000ms"}
    E -->|Yes| F05[F-05 Latency Spike\nconfidence: 0.80]
    E -->|No| F04[F-04 Answer Incomplete\ncatch-all\nconfidence: 0.70]
```

**Input:** faithfulness, retrieval_score, context_tokens, latency_ms, config

**Why priority order matters:**
- **F-03 (Hallucination)** is checked first because safety trumps all other concerns. A
  hallucinating pipeline must not be misdiagnosed as a retrieval or latency issue.
- **F-01 (Retrieval Miss)** is checked next because if retrieval fails, everything downstream
  (context, answer) is unreliable regardless of other symptoms.
- **F-04 (Answer Incomplete)** is the catch-all. If no specific pattern matched, the
  answer simply didn't use the retrieved material effectively.

**Output:** failure_type (F-01..F-05), confidence, remediation_hint, root_cause_analysis

```
Example:
  Input:  faithfulness=0.80, retrieval_score=0.15, latency=1200ms
  Output: F-01 (Retrieval Miss), confidence=0.90
          hint: "Increase retrieval_k or reduce chunk_size"
          analysis: "retrieval_score 0.15 < floor 0.30"
```

---

### Agent 4: Improver (Applying a Targeted Fix)

**File:** `agents/improver.py`, `improver_node()`

The Improver maps each failure type to a **static playbook** of 3 ordered remediation
strategies. No LLM call, just deterministic lookup and arithmetic.

```mermaid
flowchart TD
    A[Failure Type\nfrom Diagnoser] --> B{Which failure?}

    B -->|F-01\nRetrieval Miss| C["Attempt 0: k += 3\nAttempt 1: chunk_size -= 64\nAttempt 2: overlap += 32"]
    B -->|F-02\nContext Overflow| D["Attempt 0: k -= 2\nAttempt 1: chunk_size -= 64\nAttempt 2: (both)"]
    B -->|F-03\nHallucination| E["Attempt 0: prompt = v2\nAttempt 1: k -= 2\nAttempt 2: chunk_size -= 64"]
    B -->|F-04\nIncomplete| F["Attempt 0: k += 2\nAttempt 1: chunk_size += 64\nAttempt 2: overlap += 32"]
    B -->|F-05\nLatency Spike| G["Attempt 0: k -= 2\nAttempt 1: chunk_size -= 64\nAttempt 2: (both)"]

    C --> H[Apply Delta\nClamp to safe bounds]
    D --> H
    E --> H
    F --> H
    G --> H

    H --> I[New Config +\nCandidate Record]
```

**Input:** failure_type, current config, improvement_attempt (0/1/2)

**How deltas are applied:**
- **Numeric deltas** are additive: `retrieval_k: +3` means add 3 to current value
- **String deltas** are replacements: `prompt_template: "v2"` replaces the current value
- All numeric values are **clamped** to safe bounds after applying the delta:

| Knob | Min | Max |
|------|-----|-----|
| `chunk_size` | 64 | 1024 |
| `chunk_overlap` | 0 | 128 |
| `retrieval_k` | 1 | 20 |
| `max_context_tokens` | 1000 | 8000 |

**Output:** new config, candidate record (variant_id, delta, rationale, config before/after)

```
Example: F-01 (Retrieval Miss), attempt 0:
  Delta:     { retrieval_k: +3 }
  Before:    { retrieval_k: 1, chunk_size: 64 }
  After:     { retrieval_k: 4, chunk_size: 64 }
  Rationale: "Increase retrieval breadth to capture more relevant passages"
```

**Why a static playbook?** Determinism, safety, and cost. The playbook guarantees: (a) the
same diagnosis always produces the same fix, (b) all fixes are bounded within safe ranges,
and (c) zero additional LLM calls per iteration.

---

### Complete Data Flow

```mermaid
flowchart TD
    subgraph "Pipeline Agent"
        P1[Embed Query] --> P2[Search ChromaDB]
        P2 --> P3[Assemble Context]
        P3 --> P4[Generate Answer]
    end

    subgraph "Evaluator Agent"
        E1[Retrieval Metrics\nno LLM] --> E5[Unified Score]
        E2[Faithfulness Judge\nLLM] --> E5
        E3[Relevance Judge\nLLM] --> E5
        E4[Correctness Judge\nLLM] --> E5
        E5 --> E6{Gate}
    end

    subgraph "Diagnoser Agent"
        D1{"F-03?"} -->|no| D2{"F-01?"}
        D2 -->|no| D3{"F-02?"}
        D3 -->|no| D4{"F-05?"}
        D4 -->|no| D5[F-04]
    end

    subgraph "Improver Agent"
        I1[Lookup Playbook] --> I2[Apply Delta]
        I2 --> I3[Clamp Bounds]
        I3 --> I4[New Config]
    end

    P4 -->|"answer, chunks,\ncontext, cost"| E1
    P4 --> E2
    P4 --> E3
    P4 --> E4

    E6 -->|deploy_eligible| DONE([Done])
    E6 -->|hitl_required| HITL([Human Review])
    E6 -->|hard_block| D1

    D1 -->|matched| I1
    D2 -->|matched| I1
    D3 -->|matched| I1
    D4 -->|matched| I1
    D5 --> I1

    I4 -->|"Config returned\nto Optimizer"| OPT([Optimizer picks up\nnew config for next iteration])
```

[Back to Top](#table-of-contents)

---

# Chapter 5: The Optimizer Loop

The Optimizer sits outside the graph. It invokes `graph.invoke()` repeatedly, extracting the
Improver's suggested configuration after each iteration and feeding it as the next iteration's
starting config.

### Loop Structure

```mermaid
flowchart TD
    A[Start: query + initial_config] --> B[Invoke Graph → baseline result]
    B --> C{Check Stop Conditions}

    C -->|"1. faith < 0.50"| S1([STOP: blocked_faithfulness])
    C -->|"2. score >= 0.85"| S2([STOP: target_reached])
    C -->|"3. gate = hitl"| S3([STOP: hitl_required])
    C -->|"4. delta < 0.01 x3"| S4([STOP: no_improvement])
    C -->|"5. no candidates"| S5([STOP: no_candidates])
    C -->|"6. budget exhausted"| S6([STOP: max_iterations])

    C -->|"None triggered"| D[Extract new config\nfrom Improver candidates]
    D --> E[Invoke Graph → iteration N result]
    E --> F[Compare score to previous]
    F --> C
```

### Stop Condition Priority

Stop conditions are checked in a **safety-first** order:

1. **Faithfulness block**: a hallucinating pipeline must never be retried with a different
   retrieval config. The problem is the model, not the settings.
2. **Target reached**: success. The pipeline meets the quality bar.
3. **HITL required**: the score is in the gray band (0.70-0.85). The Optimizer cannot
   make this decision. A human reviewer must.
4. **No improvement**: three consecutive iterations with delta < 0.01 indicates the
   playbook is exhausted for this failure type.
5. **No candidates**: the Improver produced no fix. This is a defensive check and should
   not occur with the current playbook.
6. **Max iterations**: budget cap. Default is 3.

### Illustrative Example

**Starting point:** query = "How do embeddings represent meaning?"

```mermaid
flowchart LR
    B["Baseline\nk=1, chunk=64\nscore: 0.43\nF-01"] -->|"fix: k +3"| I1["Iteration 1\nk=4, chunk=64\nscore: 0.58\nF-04"]
    I1 -->|"fix: k +3"| I2["Iteration 2\nk=7, chunk=64\nscore: 0.72\nhitl_required"]
    I2 --> STOP([STOP\nhitl_required\nScore in gray band])
```

| Iteration | Config | Score | Failure | Fix Applied |
|-----------|--------|-------|---------|-------------|
| Baseline | k=1, chunk=64 | 0.43 | F-01 (Retrieval Miss) | (initial) |
| 1 | k=4, chunk=64 | 0.58 | F-04 (Answer Incomplete) | k +3 |
| 2 | k=7, chunk=64 | 0.72 | (HITL band) | k +3 |
| Stop | | | hitl_required | Score in gray band |

The Optimizer improved the score from 0.43 to 0.72 in two iterations by increasing retrieval
breadth, then stopped because the score entered the HITL band where a human reviewer must
decide.

### Before vs After

| Metric | Before | After |
|--------|--------|-------|
| Unified Score | 0.43 | 0.72 |
| Faithfulness | 1.00 | 1.00 |
| Relevance | 0.60 | 1.00 |
| Retrieval k | 1 | 7 |
| Gate Decision | hard_block | hitl_required |

### Optimizer Scope

The Optimizer adjusts retrieval and chunking configuration knobs within bounded ranges. It does
**not**:

- Modify the underlying model or fine-tune weights
- Re-train or swap embedding models
- Add new documents to the corpus
- Perform prompt engineering beyond switching between predefined prompt template versions

[Back to Top](#table-of-contents)

---

# Chapter 6: Using the Streamlit UI

The interface is organized into three tabs, each serving a distinct purpose in the workflow.

```mermaid
flowchart LR
    subgraph "Tab 1: Pipeline"
        T1[Run single query\nInspect all steps]
    end
    subgraph "Tab 2: Optimization"
        T2[Run improvement loop\nView iterations]
    end
    subgraph "Tab 3: History"
        T3[Review past runs\nScore trends]
    end
    T1 --> T2 --> T3
```

### Tab 1: Pipeline

**Purpose:** Execute a single RAG query and inspect every step of the pipeline output.

| Section | What it shows |
|---------|--------------|
| Document Selection | Select a document from the corpus or upload a new one |
| Initialize Pipeline | Ingest the document with the configured chunking strategy |
| Ask a Question | Free-text input or select from the golden dataset |
| Results | Answer, all scores (faithfulness, relevance, correctness, unified), gate decision |
| Metadata | Cost, latency, chunk count, collection name |
| Retrieved Chunks | The text chunks retrieved, with similarity scores |

**Sidebar controls** affect Tab 1: chunk strategy, chunk size, overlap, retrieval k, max
context tokens, version string, max pages to ingest.

### Tab 2: Optimization

**Purpose:** Run the self-improvement loop and observe each iteration's diagnosis and fix.

| Section | What it shows |
|---------|--------------|
| Setup | Query selection, config presets (Degraded / Default), editable config fields |
| Baseline Run | Full pipeline result with all metrics |
| Optimizer Run | Per-iteration expandable details: scores, diagnosis, fix applied, delta |
| HITL Decision | Accept or reject. Rejection triggers re-optimization |
| Before vs After | Side-by-side comparison of initial and final configs, scores, and chunks |

**Config presets:**
- **Degraded Config**: k=1, chunk_size=64, intentionally suboptimal for demonstrating improvement
- **Default Config**: k=5, chunk_size=256, baseline for the current corpus

### Tab 3: History

**Purpose:** Review historical runs and track quality trends over time.

| Section | What it shows |
|---------|--------------|
| Pipeline Queries | Past single-query runs with scores and configurations |
| Optimization Loops | Past optimizer runs with per-iteration decision chains |
| Summary Stats | Aggregate score metrics across all runs |
| Score Trend | Line chart of unified scores over time |

History is persisted in `data/runs_history.jsonl` as an append-only store. MLflow integration
for full experiment tracking is planned for Phase 3.

[Back to Top](#table-of-contents)

---

# Chapter 7: Configuration and Ingestion

### Configuration Knobs

All configuration is centralized in `config.py`. The Optimizer sweeps these knobs during the
improvement loop:

| Knob | Default | Range | What it controls |
|------|---------|-------|-----------------|
| `chunk_size` | 256 | 64 - 1024 | Token count per chunk |
| `chunk_overlap` | 0 | 0 - 128 | Overlap between consecutive chunks |
| `retrieval_k` | 5 | 1 - 20 | Number of chunks retrieved per query |
| `max_context_tokens` | 4000 | 1000 - 8000 | Token budget for assembled context |
| `prompt_template` | "v1" | v1, v2 | Generation prompt version |
| `chunk_strategy` | "fixed_size" | fixed_size, recursive_split, semantic | Chunking algorithm |

### Collection Naming

Each unique combination of version + strategy + chunk_size maps to its own ChromaDB collection:

```
rag_{version}_{strategy}_{chunk_size}

Examples:
  rag_g1_fixed_size_256    ← default config, Google embeddings
  rag_g1_fixed_size_64     ← degraded config preset
  rag_g1_fixed_size_128    ← created by Optimizer when Improver adjusts chunk_size
```

Collections are **immutable**. Once ingested, they are never modified. A new chunk_size
produces a new collection with a new name, not an update to the existing one.

### Auto-Ingest

```mermaid
flowchart TD
    A[Graph receives config\nchunk_size changed] --> B[Build collection name\nrag_g1_fixed_size_128]
    B --> C{Collection\nexists?}
    C -->|Yes, has chunks| D[Skip ingest\nProceed to retrieval]
    C -->|Empty or missing| E[Auto-ingest]
    E --> F[Load corpus pages]
    F --> G[Chunk with new size]
    G --> H[Embed chunks]
    H --> I[Store in ChromaDB]
    I --> D
```

The graph pipeline auto-ingests when it encounters an empty collection. This occurs
transparently during optimizer runs when the Improver changes `chunk_size`:

1. Improver suggests `chunk_size: 128` (was 256)
2. Next graph invocation builds collection name: `rag_g1_fixed_size_128`
3. Collection does not exist → auto-ingest: load corpus → chunk at 128 → embed → store
4. Retrieval proceeds against the new collection

### Corpus

The default corpus is *Hands-On Large Language Models* by Jay Alammar. Current ingest range:

- **Start page:** 25 (skips overview/preamble content)
- **Page count:** 25 (pages 25-49)
- **Content covered:** tokenization, bag-of-words, word2vec, embeddings, RNNs, attention
  mechanisms, transformers, BERT, GPT, context windows, fine-tuning

### Golden Dataset

Seven test queries with expected answers and retrieval keywords, defined in `ground_truth.py`:

| # | Query | Key Topic |
|---|-------|-----------|
| 0 | How do embeddings represent meaning? | Embeddings, vectors, semantic similarity |
| 1 | What is the attention mechanism? | Attention, token weighting |
| 2 | What is a transformer? | Transformer architecture |
| 3 | What is tokenization? | Text splitting into tokens |
| 4 | What is a context window? | Maximum token limit |
| 5 | What is fine-tuning? | Adapting pre-trained models |
| 6 | How does word2vec generate word embeddings? | Word2vec training process |

The golden dataset serves two purposes:
1. **Correctness scoring**: the Evaluator compares the pipeline's answer to the expected answer
2. **Retrieval metrics**: keyword presence in retrieved chunks provides precision and recall

[Back to Top](#table-of-contents)

---

# Chapter 8: Troubleshooting and FAQ

### Common Issues

| Symptom | Cause | Resolution |
|---------|-------|------------|
| `429 RESOURCE_EXHAUSTED` | API rate limit exceeded (15 req/min on free tier) | Wait 60s and retry. Optimizer runs are rate-limit-intensive (4+ LLM calls per iteration). |
| Score = 0.0 on all metrics, but answer is reasonable | Judge calls failed silently due to rate limiting; scores defaulted to None | Wait for rate limit to reset and re-run the query. |
| Optimizer shows score regression | Larger chunks on a small corpus produce fewer total chunks; high k retrieves too large a fraction | Expected with small corpora. The bounds (k max=20) and max_context_tokens (4000) are the guardrails. |
| Collection has 0 chunks | Collection was deleted but not re-ingested | Tab 2 auto-ingests. For Tab 1, use "Initialize Pipeline" to trigger ingestion. |
| "I don't have enough information" answer | Retrieved chunks do not contain relevant content for the query | Verify the corpus page range covers the topic. Check against the golden dataset queries. |

### FAQ

**Q: Why is the graph linear instead of having an internal retry loop?**

The Optimizer owns retry logic externally. A linear graph is easier to debug (each invocation
is a pure function), eliminates infinite loop risks, and provides clean separation between
"evaluate once" and "decide whether to retry."

**Q: Why are there 5 failure types instead of 6?**

F-06 (Quality Drift) detects score degradation over time across production traffic. It
requires the Observer agent and live monitoring, which are planned for Phase 2. The
configuration scaffolding (`DRIFT_WINDOW`, `DRIFT_MIN_DROP`) is already in place.

**Q: Why use keyword-based retrieval metrics instead of LLM-based retrieval evaluation?**

Cost and speed. Keyword precision/recall is free, instant, and sufficient for the retrieval
component of the unified score. The LLM judges handle the more subjective assessments
(faithfulness, relevance, correctness).

**Q: What would MLflow replace in this system?**

The current JSONL-based history store (`run_history.py`). MLflow would add: experiment
grouping, SQL-queryable run history, artifact storage, a web UI for comparing runs, and a
model registry for promoting validated configurations. The core self-improving logic
(evaluate → diagnose → improve) is independent of the logging backend. MLflow integration
is planned for Phase 3.

**Q: Why does the Improver use a static playbook instead of LLM-generated fixes?**

Determinism, safety, and cost. The playbook maps each failure type to exactly 3 ordered
remediation strategies, guaranteeing: (a) identical diagnoses always produce identical fixes,
(b) all configuration changes are bounded within safe ranges, and (c) zero additional LLM
calls per improvement iteration.

**Q: What happens when the Optimizer exhausts all 3 iterations without reaching the target?**

It returns `stop_reason: "max_iterations_reached"` with the best score achieved. The full
iteration history (configs tried, scores achieved, diagnoses, fixes applied) is available
for manual review and further tuning decisions.

[Back to Top](#table-of-contents)
