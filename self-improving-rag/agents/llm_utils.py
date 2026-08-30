"""LLM call wrapper — EURI → Google Gemini failover chain.

Single entry point for ALL LLM calls with provider failover, structured
responses, and proper error classification.

Design:
  - llm_call() tries providers in order; on failure, falls back to the next
  - On all-fail: raises LLMProviderError
  - LLMResponse dataclass carries content + cost + latency + tokens
  - Judge failures → None (never fabricate a score)

Cost accounting:
  Cost is computed HERE, not in agent nodes. This centralizes cost
  calculation in a single place for consistency and auditability.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI, APIError, APITimeoutError, RateLimitError, AuthenticationError

from config import (
    EURI_API_KEY, EURI_BASE_URL,
    GOOGLE_API_KEY, GOOGLE_BASE_URL,
    LLM_PROVIDER,
    PRICE_INPUT_PER_M, PRICE_OUTPUT_PER_M,
)

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public Types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class RAGError(Exception):
    """Base exception for all Self-Improving RAG errors."""


class LLMProviderError(RAGError):
    """Raised when all LLM providers in the failover chain have failed.

    Raised only when ALL providers are exhausted, never on a single
    provider failure (that's a fallback trigger, not an error).
    """

    def __init__(self, message: str, last_provider: str = "", last_error: str = ""):
        self.last_provider = last_provider
        self.last_error = last_error
        super().__init__(message)


class IngestionError(RAGError):
    """Raised when document ingestion fails (chunking, embedding, or upsert)."""


class ConfigValidationError(RAGError):
    """Raised when pipeline config validation fails."""


@dataclass
class LLMResponse:
    """Structured result of a single LLM call.

      - content: the LLM's text response
      - provider: which provider actually served this ("euri" or "google")
      - prompt_tokens/completion_tokens: from the API usage response
      - cost_usd: computed here, never in agent nodes
      - latency_ms: wall-clock time for the API call
      - model: which model was used
    """

    content: str
    provider: str           # "euri" or "google"
    model: str              # actual model name used
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: int


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Provider Chain
#
# Each provider is a (label, api_key, base_url, models) tuple.
# The chain is tried in order; first success wins.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_provider_chain() -> list[dict[str, Any]]:
    """Build the ordered provider failover chain.

    Each slot resolves to a (label, api_key, base_url, chat_model,
    judge_model) tuple.

    Order:
      1. Active provider (from LLM_PROVIDER env var — "euri" or "google")
      2. Fallback provider (the other one)

    If a provider has no API key configured, it's skipped (can't call it).
    """
    providers = []

    euri = {
        "label": "euri",
        "api_key": EURI_API_KEY,
        "base_url": EURI_BASE_URL,
        "chat_model": "gpt-4o-mini",
        "judge_model": "llama-3.3-70b-versatile",
    }

    google = {
        "label": "google",
        "api_key": GOOGLE_API_KEY,
        "base_url": GOOGLE_BASE_URL,
        "chat_model": "gemini-3.5-flash-lite",
        "judge_model": "gemini-3.5-flash-lite",
    }

    # Active provider first, fallback second
    if LLM_PROVIDER == "google":
        if GOOGLE_API_KEY:
            providers.append(google)
        if EURI_API_KEY:
            providers.append(euri)
    else:
        if EURI_API_KEY:
            providers.append(euri)
        if GOOGLE_API_KEY:
            providers.append(google)

    return providers


def _compute_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost from token counts.

    Single place for cost calculation. Uses the price constants from config.py.
    """
    cost = (
        (prompt_tokens / 1_000_000) * PRICE_INPUT_PER_M
        + (completion_tokens / 1_000_000) * PRICE_OUTPUT_PER_M
    )
    return round(cost, 8)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Core LLM Call — with provider failover
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _call_provider(
    *,
    provider: dict[str, Any],
    messages: list[dict[str, str]],
    model_key: str = "chat_model",
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> LLMResponse:
    """Make a single LLM call to one provider.

    Provider-agnostic — returns structured LLMResponse on success, raises
    on failure.

    Error classification:
      - RateLimitError: retryable → failover to next provider
      - APITimeoutError: retryable → failover to next provider
      - AuthenticationError: non-retryable → failover (wrong key)
      - APIError: catch-all for provider errors → failover
      - Exception: unexpected → failover
    """
    label = provider["label"]
    model = provider[model_key]
    client = OpenAI(api_key=provider["api_key"], base_url=provider["base_url"])

    call_start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency_ms = int((time.perf_counter() - call_start) * 1000)

    content = resp.choices[0].message.content or ""
    prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
    completion_tokens = resp.usage.completion_tokens if resp.usage else 0

    return LLMResponse(
        content=content,
        provider=label,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=_compute_cost(prompt_tokens, completion_tokens),
        latency_ms=latency_ms,
    )


def llm_call(
    messages: list[dict[str, str]],
    *,
    model_key: str = "chat_model",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    agent_name: str = "unknown",
) -> LLMResponse:
    """Call LLM with automatic failover. Raises LLMProviderError if all fail.

    This is the SINGLE entry point for all LLM calls in the project.
    Tries each provider in chain order, logs failures, and fails over
    to the next.

    Args:
        messages: Chat messages (system + user).
        model_key: Which model to use from provider config ("chat_model"
                   or "judge_model").
        temperature: LLM temperature (0.0 for deterministic).
        max_tokens: Maximum response tokens.
        agent_name: For logging — which agent is making this call.

    Returns:
        LLMResponse with content, cost, latency, tokens.

    Raises:
        LLMProviderError: When all providers in the chain have failed.
    """
    providers = _build_provider_chain()
    if not providers:
        raise LLMProviderError(
            "No LLM providers configured — set EURI_API_KEY or GOOGLE_API_KEY",
            last_provider="none",
            last_error="no_api_keys",
        )

    last_provider = ""
    last_error = ""

    for provider in providers:
        label = provider["label"]
        try:
            response = _call_provider(
                provider=provider,
                messages=messages,
                model_key=model_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            logger.info(
                "llm_call_success",
                extra={
                    "provider": label,
                    "model": response.model,
                    "agent_name": agent_name,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "cost_usd": response.cost_usd,
                    "latency_ms": response.latency_ms,
                },
            )
            return response

        except RateLimitError as e:
            last_provider = label
            last_error = f"rate_limit: {e}"
            logger.warning(
                "llm_provider_rate_limited",
                extra={"provider": label, "agent_name": agent_name, "error": str(e)},
            )

        except APITimeoutError as e:
            last_provider = label
            last_error = f"timeout: {e}"
            logger.warning(
                "llm_provider_timeout",
                extra={"provider": label, "agent_name": agent_name, "error": str(e)},
            )

        except AuthenticationError as e:
            last_provider = label
            last_error = f"auth_error: {e}"
            logger.warning(
                "llm_provider_auth_failed",
                extra={"provider": label, "agent_name": agent_name, "error": str(e)},
            )

        except APIError as e:
            last_provider = label
            last_error = f"api_error ({e.status_code}): {e.message}"
            logger.warning(
                "llm_provider_api_error",
                extra={
                    "provider": label,
                    "agent_name": agent_name,
                    "status_code": e.status_code,
                    "error": e.message,
                },
            )

        except Exception as e:
            last_provider = label
            last_error = f"unexpected: {e}"
            logger.warning(
                "llm_provider_unexpected_error",
                extra={"provider": label, "agent_name": agent_name, "error": str(e)},
            )

    # All providers exhausted
    raise LLMProviderError(
        f"All LLM providers exhausted. Last failure ({last_provider}): {last_error}",
        last_provider=last_provider,
        last_error=last_error,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Judge Call — LLM call + JSON parsing, returns None on failure
#
# RAGAS judge calls:
#   - Use the judge_model (may differ from chat_model)
#   - Return None on failure (never fabricate a score)
#   - Parse JSON from LLM response (strip code fences, etc.)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _parse_json_response(raw: str) -> dict:
    """Parse JSON from an LLM response, handling markdown code fences.

    LLMs sometimes wrap JSON in ```json ... ``` — strip that before parsing.
    Returns empty dict on parse failure (caller decides what None means).
    """
    text = raw.strip()

    # Strip markdown code fences
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(
            "json_parse_failed",
            extra={"error": str(e), "raw_length": len(raw)},
        )
        return {}


def judge_call(
    prompt: str,
    *,
    agent_name: str = "judge",
) -> dict:
    """Call the judge LLM and parse JSON response.

    Returns parsed JSON dict on success, empty dict on failure.
    Judge failures produce None metrics (never fabricated scores),
    handled by the evaluator's weight re-normalization in the unified
    score formula.

    Uses judge_model (may differ from chat_model — judge calls are
    explicitly decoupled from generation calls so each can use a
    different provider/model).
    """
    try:
        response = llm_call(
            messages=[{"role": "user", "content": prompt}],
            model_key="judge_model",
            temperature=0.0,
            max_tokens=1024,
            agent_name=agent_name,
        )
        return _parse_json_response(response.content)

    except LLMProviderError:
        logger.error(
            "judge_call_all_providers_failed",
            extra={"agent_name": agent_name},
        )
        return {}

    except Exception as e:
        logger.error(
            "judge_call_unexpected_error",
            extra={"agent_name": agent_name, "error": str(e)},
        )
        return {}
