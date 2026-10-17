# ── Model configuration ────────────────────────────────────────────────────────
# Current Gemini Flash model ID
GEMINI_MODEL = "gemini-3.8-flash"

# Default generation settings for grading calls
DEFAULT_TEMPERATURE = 0.0   # deterministic for reproducibility

# Thinking configuration:
# Gemini supports tunable thinking budget. Setting thinking_budget to 0 turns off
# thinking/reasoning token generation completely, eliminating reasoning token usage.
THINKING_BUDGET = 0
THINKING_LEVEL = None  # Disabled to enforce lowest thinking budget (off)

# Maximum output tokens by method (prevents runaway long outputs)
MAX_OUTPUT_TOKENS = {
    "direct": 300,
    "direct_v1": 300,
    "direct_v2": 300,
    "rubric": 500,
    "rubric_v1": 500,
    "rubric_v2": 500,
    "rag_rubric": 500,
    "rag_rubric_v2": 500,
}
DEFAULT_MAX_OUTPUT_TOKENS = 500

# Rate limiting: free tier for gemini-3.8-flash is 5 requests per minute
RPM_LIMIT = 5
REQUEST_DELAY_SECONDS = 12.5  # Safe spacing (60s / 5 RPM = 12.0s + 0.5s buffer)

# ── Retrieval & RAG configuration ─────────────────────────────────────────────
RETRIEVAL_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_QUERY_MODE = "question"  # "question" (default) or "question+reference"
RAG_TOP_K = 3                      # Top-k retrieved chunks for RAG grading
RAG_MAX_CHUNK_WORDS = 150          # Maximum words per retrieved chunk in RAG context
