"""Configuration — single source of truth for all knobs and thresholds.

Pipeline config validation uses Pydantic models to catch bad values
at startup rather than letting them silently break the pipeline.
"""
import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

load_dotenv()
os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt")

# ─── LLM / Embedding ─────────────────────────────────────────────────────────
EURI_API_KEY = os.getenv("EURI_API_KEY", "")
EURI_BASE_URL = os.getenv("EURI_BASE_URL", "https://api.euron.one/api/v1/euri")

# Google Gemini fallback (used when EURI daily limit is hit)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Active provider: "euri" or "google"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google")

# Model names per provider — hardcoded per provider, not from .env
# (.env has EURI model names which would break Google if os.getenv picked them up)
if LLM_PROVIDER == "google":
    LLM_MODEL = "gemini-3.5-flash-lite"
    JUDGE_MODEL = "gemini-3.5-flash-lite"
    EMBEDDING_MODEL = "gemini-embedding-001"
    LLM_API_KEY = GOOGLE_API_KEY
    LLM_BASE_URL = GOOGLE_BASE_URL
else:
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_API_KEY = EURI_API_KEY
    LLM_BASE_URL = EURI_BASE_URL

EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "256"))
TOKENIZER_MODEL = os.getenv("TOKENIZER_MODEL", "o200k_base")

# ─── Ingest defaults ─────────────────────────────────────────────────────────
INGEST_PAGES = int(os.getenv("INGEST_PAGES", "25"))
INGEST_START_PAGE = int(os.getenv("INGEST_START_PAGE", "25"))  # skip overview (21-24), start at tokenization

# ─── Book / Data ─────────────────────────────────────────────────────────────
BOOK_PATHS = [
    os.path.join(os.path.dirname(__file__), "corpus", "Hands-On_Large_Language_Models_-_Jay_Alammar.txt"),
    os.path.join(os.path.dirname(__file__), "corpus", "AI Engineering_ Building Applications With Foundation Models by Chip Huyen (1).txt"),
]

# ─── Multi-file demo corpus (whole-document ingestion, no page windowing) ────
# 9 short business documents across 4 domains (HR, Technical, Legal, Finance),
# curated for genuine cross-domain golden-set queries. See ground_truth.py.
CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")
CORPUS_FILES = [
    os.path.join(CORPUS_DIR, "hr", "leave_policy.txt"),
    os.path.join(CORPUS_DIR, "hr", "benefits_guide.txt"),
    os.path.join(CORPUS_DIR, "hr", "code_of_conduct.txt"),
    os.path.join(CORPUS_DIR, "hr", "statutory_holidays_policy.txt"),
    os.path.join(CORPUS_DIR, "technical", "production_runbook.txt"),
    os.path.join(CORPUS_DIR, "technical", "developer_setup_guide.txt"),
    os.path.join(CORPUS_DIR, "legal", "vendor_contract_template.txt"),
    os.path.join(CORPUS_DIR, "legal", "nda_template.txt"),
    os.path.join(CORPUS_DIR, "finance", "expense_policy.txt"),
    os.path.join(CORPUS_DIR, "finance", "budget_guidelines_2026.txt"),
]
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "data", "chroma")

# ─── Pipeline knobs (optimizer sweeps these) ─────────────────────────────────
DEFAULT_CONFIG = {
    "chunk_strategy": "fixed_size",
    "chunk_size": 256,
    "chunk_overlap": 0,
    "retrieval_k": 5,
    "max_context_tokens": 4000,
    "prompt_template": "v1",
}

# ─── Gate thresholds (mirrors AutoRAG slo.yaml) ─────────────────────────────
UNIFIED_TARGET = 0.85       # autonomous accept
HITL_LOW = 0.70             # gray band lower bound (interrupt)
FAITHFULNESS_FLOOR = 0.50   # hard veto — non-negotiable
RETRIEVAL_SIM_FLOOR = 0.30  # F-01 trigger
LATENCY_CAP_MS = 3000       # latency penalty saturation
MAX_RETRIES = 3             # in-graph diagnose/improve cap
MAX_ITERATIONS = 3          # optimizer trial cap
NO_IMPROVEMENT_DELTA = 0.01 # plateau threshold
DRIFT_WINDOW = 5            # runs to check for F-06
DRIFT_MIN_DROP = 0.05       # cumulative drop to flag drift

# ─── Cost accounting (gpt-4o-mini approx, per 1M tokens) ─────────────────────
PRICE_INPUT_PER_M = 0.15
PRICE_OUTPUT_PER_M = 0.60

# ─── Known chunking strategies ───────────────────────────────────────────────
KNOWN_STRATEGIES = {"fixed_size", "recursive_split", "semantic"}
KNOWN_PROMPT_TEMPLATES = {"v1", "v2"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pipeline Config Validation — Pydantic model
#
# Validates pipeline config at startup and before each pipeline run.
# Bad values (negative k, chunk_size=0) raise ConfigValidationError with
# clear messages instead of silently breaking deep in the pipeline.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PipelineConfig(BaseModel):
    """Validated pipeline configuration.

    Every field has bounds that prevent silent failures:
      - chunk_size >= 32 (smaller chunks lose semantic coherence)
      - chunk_overlap >= 0 and < chunk_size (overlap can't exceed chunk)
      - retrieval_k >= 1 (must retrieve at least one chunk)
      - max_context_tokens >= 500 (too small = unusable context)
      - chunk_strategy must be a known strategy
      - prompt_template must be a known template version

    Usage:
        validated = PipelineConfig(**raw_config_dict)
        config_dict = validated.to_dict()
    """

    chunk_strategy: str = Field(
        default="fixed_size",
        description="Chunking strategy: fixed_size, recursive_split, or semantic",
    )
    chunk_size: int = Field(
        default=256,
        ge=32,
        description="Chunk size in characters/tokens. Must be >= 32.",
    )
    chunk_overlap: int = Field(
        default=0,
        ge=0,
        description="Overlap between consecutive chunks. Must be >= 0 and < chunk_size.",
    )
    retrieval_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of chunks to retrieve. Must be 1-50.",
    )
    max_context_tokens: int = Field(
        default=4000,
        ge=500,
        description="Maximum tokens in assembled context. Must be >= 500.",
    )
    prompt_template: str = Field(
        default="v1",
        description="Prompt template version: v1 or v2.",
    )

    @model_validator(mode="after")
    def _validate_bounds(self) -> "PipelineConfig":
        """Cross-field validation: overlap must be less than chunk_size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        if self.chunk_strategy not in KNOWN_STRATEGIES:
            raise ValueError(
                f"Unknown chunk_strategy '{self.chunk_strategy}'. "
                f"Must be one of: {', '.join(sorted(KNOWN_STRATEGIES))}"
            )
        if self.prompt_template not in KNOWN_PROMPT_TEMPLATES:
            raise ValueError(
                f"Unknown prompt_template '{self.prompt_template}'. "
                f"Must be one of: {', '.join(sorted(KNOWN_PROMPT_TEMPLATES))}"
            )
        return self

    def to_dict(self) -> dict:
        """Convert to plain dict for passing to pipeline functions."""
        return self.model_dump()


def validate_config(config: dict) -> dict:
    """Validate a pipeline config dict and return the validated version.

    This is the convenience function that wraps PipelineConfig.
    Call this before any pipeline run to catch bad values early.

    Args:
        config: Raw config dict (may have bad values).

    Returns:
        Validated config dict with all fields present.

    Raises:
        ValueError: If any field fails validation (with a clear message).
    """
    validated = PipelineConfig(**config)
    return validated.to_dict()
