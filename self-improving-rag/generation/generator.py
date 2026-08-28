"""LLM generation via active provider (EURI or Google Gemini)."""

import time
from openai import OpenAI
from config import (
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
    PRICE_INPUT_PER_M, PRICE_OUTPUT_PER_M,
)
from prompts import SYSTEM_PROMPT, QA_TEMPLATE, PROMPT_TEMPLATES


def generate(
    context: str,
    question: str,
    prompt_version: str = "v1",
) -> dict:
    """Call EURI LLM with assembled context + question, return answer + metadata.

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

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,  # deterministic for evaluation consistency
    )
    latency_ms = int((time.perf_counter() - start) * 1000)

    answer = resp.choices[0].message.content or ""
    input_tokens = resp.usage.prompt_tokens
    output_tokens = resp.usage.completion_tokens

    # Cost accounting
    cost_usd = (
        (input_tokens / 1_000_000) * PRICE_INPUT_PER_M
        + (output_tokens / 1_000_000) * PRICE_OUTPUT_PER_M
    )

    return {
        "answer": answer.strip(),
        "cost_usd": round(cost_usd, 6),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "model": LLM_MODEL,
        "prompt_version": prompt_version,
    }
