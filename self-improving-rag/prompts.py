"""Prompt templates for RAG generation.

Mirrors AutoRAG's evaluator.py _ANSWER_GENERATION_PROMPT_V1 — the instruction
that makes the LLM answer from context only, making hallucination measurable.
"""

SYSTEM_PROMPT = """You are a precise document Q&A assistant.

Answer based ONLY on the Context below.
If the Context does not contain enough information, respond:
"I don't have enough information in the provided context to answer this question."
Be concise, accurate, and grounded.
Do not invent facts if the context is empty or insufficient."""

QA_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""

# ─── Template versions (optimizer can swap these) ────────────────────────────
# v1: standard (above)
# v2: stricter grounding — used by improver's F-03 playbook
SYSTEM_PROMPT_V2 = """You are a precise document Q&A assistant.

Answer based ONLY on the Context below. Every claim in your answer MUST be
directly supported by a sentence in the Context.
If the Context does not contain enough information, respond:
"I don't have enough information in the provided context to answer this question."
Be concise. Do not add information not present in the Context."""

PROMPT_TEMPLATES = {
    "v1": {"system": SYSTEM_PROMPT, "qa": QA_TEMPLATE},
    "v2": {"system": SYSTEM_PROMPT_V2, "qa": QA_TEMPLATE},
}
