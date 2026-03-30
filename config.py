"""AlignEval configuration."""
import os

# LLM configuration
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_MOCK_MODE = os.getenv("LLM_MOCK_MODE", "false").lower() == "true"

# Knowledge graph settings
KG_SIMILARITY_THRESHOLD = float(os.getenv("KG_SIMILARITY_THRESHOLD", "0.85"))
MAX_TRIPLES_PER_ENTITY = int(os.getenv("MAX_TRIPLES_PER_ENTITY", "20"))

# Probing settings
PROBE_LEVELS = ["factual", "relational", "reverse"]
MAX_PROBE_ENTITIES = int(os.getenv("MAX_PROBE_ENTITIES", "50"))

# App settings
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
DATA_DIR = os.getenv("DATA_DIR", "data")
