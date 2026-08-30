"""LLM generation via llm_call() with provider failover.

Uses agents/llm_utils.py as the single entry point for all LLM calls.
"""

import logging
from agents.llm_utils import llm_call
from prompts import PROMPT_TEMPLATES

logger = logging.getLogger(__name__)


def generate(
    context: str,
    question: str,
    prompt_version: str = "v1",
) -> dict:
    """Call LLM with assembled context + question, return answer + metadata.

    Uses llm_call() for provider failover (EURI → Google).
    Cost is computed inside llm_call() — cost calculation is centralized
    there, never in agent nodes.

    Args:
        context: Assembled context string from assemble_context().
        question: The user's question.
        prompt_version: Which prompt template to use ("v1" or "v2").

    Returns:
        Dict with: answer, cost_usd, input_tokens, output_tokens, latency_ms.
    """
    templates = PROMPT_TEMPLATES.get(prompt_version, PROMPT_TEMPLATES["v1"])
    system_prompt = templates["system"]
    qa_template = templates["qa"]

    user_prompt = qa_template.format(context=context, question=question)

    response = llm_call(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model_key="chat_model",
        temperature=0.0,
        agent_name="generator",
    )

    return {
        "answer": response.content.strip(),
        "cost_usd": response.cost_usd,
        "input_tokens": response.prompt_tokens,
        "output_tokens": response.completion_tokens,
        "latency_ms": response.latency_ms,
        "model": response.model,
        "provider": response.provider,
        "prompt_version": prompt_version,
    }
