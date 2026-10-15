# ── Model configuration ────────────────────────────────────────────────────────
# Current Gemini Flash model ID
GEMINI_MODEL = "gemini-3.8-flash"

# Default generation settings for grading calls
DEFAULT_TEMPERATURE = 0.0   # deterministic for reproducibility

# Thinking configuration:
# Gemini 3.8 Flash supports tunable thinking levels ("low", "medium", "high").
# By default, Gemini 3.8 Flash has thinking enabled (defaulting to "medium").
# We set THINKING_LEVEL explicitly to "low" for grading evaluation to minimize latency
# and token overhead while providing structured reasoning, and apply it consistently.
THINKING_LEVEL = "low"

# Rate limiting: free tier for gemini-3.8-flash is 5 requests per minute
RPM_LIMIT = 5
REQUEST_DELAY_SECONDS = 12.5  # Safe spacing (60s / 5 RPM = 12.0s + 0.5s buffer)

# ── Retrieval configuration ────────────────────────────────────────────────────
RETRIEVAL_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_QUERY_MODE = "question"  # "question" (default) or "question+reference"
