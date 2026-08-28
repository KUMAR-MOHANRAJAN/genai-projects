"""Configuration — single source of truth for all knobs and thresholds.

Mirrors AutoRAG's slo.yaml + config.py combined into one file for simplicity.
"""
import os
from dotenv import load_dotenv

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
