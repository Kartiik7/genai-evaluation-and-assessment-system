"""
src/llm_client.py

generate_json(prompt, model, temperature, cache_key_extra="")
  - Loads GEMINI_API_KEY from .env
  - Calls Gemini via google-genai (models.generate_content) with JSON response mode
  - Disk-caches every response in results/llm_cache/ keyed by SHA-256 hash of
    (model, temperature, prompt, cache_key_extra)
  - Exponential back-off on rate-limit / transient errors (max 5 attempts)
  - One JSON-parse retry; on second failure returns {"error": "invalid_json", "raw": ...}
  - Sleeps CALL_DELAY_S between calls to stay under free-tier RPM limit
  - Logs every call to results/llm_log.csv
"""

import os
import json
import time
import hashlib
import logging
import datetime
import csv

from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

try:
    from config import THINKING_LEVEL, REQUEST_DELAY_SECONDS
except ImportError:
    THINKING_LEVEL = "low"
    REQUEST_DELAY_SECONDS = 4.0

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

CACHE_DIR   = Path("results/llm_cache")
LOG_PATH    = Path("results/llm_log.csv")
CALL_DELAY_S = REQUEST_DELAY_SECONDS          # seconds between API calls (respecting RPM limit)
MAX_ATTEMPTS = 5
BACKOFF_BASE = 2.0          # seconds; doubles each retry


# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Lazy singleton client ─────────────────────────────────────────────────────
_client: genai.Client | None = None

def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment / .env")
        _client = genai.Client(api_key=api_key)
    return _client


# ── Helpers ───────────────────────────────────────────────────────────────────
def _cache_key(model: str, temperature: float,
               prompt: str, cache_key_extra: str) -> str:
    raw = f"{model}|{temperature}|{prompt}|{cache_key_extra}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.json"


def _log_call(model: str, temperature: float, cached: bool) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["timestamp", "model", "temperature", "cached"])
        writer.writerow([
            datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            model,
            temperature,
            cached,
        ])


def _parse_json_from_text(text: str) -> dict:
    """Strip markdown code fences if present, then parse JSON."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # drop first and last fence lines
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


# ── Main public function ──────────────────────────────────────────────────────
def generate_json(prompt: str,
                  model: str,
                  temperature: float,
                  cache_key_extra: str = "") -> dict:
    """
    Call Gemini and return a parsed JSON dict.

    Parameters
    ----------
    prompt          : The full prompt string to send.
    model           : Gemini model ID, e.g. "gemini-2.0-flash".
    temperature     : Sampling temperature (0.0 = deterministic).
    cache_key_extra : Optional string mixed into the cache hash (e.g. row index).

    Returns
    -------
    dict – parsed JSON response, or {"error": "invalid_json", "raw": <text>}
           if the model returns non-parseable output after 2 attempts.
    """
    key   = _cache_key(model, temperature, prompt, cache_key_extra)
    cpath = _cache_path(key)

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if cpath.exists():
        logger.info("CACHE HIT  model=%s temp=%s key=%s", model, temperature, key[:12])
        _log_call(model, temperature, cached=True)
        with cpath.open(encoding="utf-8") as fh:
            return json.load(fh)

    # ── API call with retry ───────────────────────────────────────────────────
    client = _get_client()
    
    # Configure thinking level if specified
    thinking_cfg = None
    if THINKING_LEVEL:
        try:
            thinking_cfg = types.ThinkingConfig(thinking_level=THINKING_LEVEL)
        except Exception:
            pass

    config_kwargs = {
        "temperature": temperature,
        "response_mime_type": "application/json",
    }
    if thinking_cfg is not None:
        config_kwargs["thinking_config"] = thinking_cfg

    config = types.GenerateContentConfig(**config_kwargs)


    last_exc = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            logger.info("API CALL   model=%s temp=%s attempt=%d/%d",
                        model, temperature, attempt, MAX_ATTEMPTS)
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            raw_text = response.text
            break                          # success
        except Exception as exc:
            last_exc = exc
            exc_str  = str(exc).lower()
            # Retryable: rate limit (429) or transient 5xx
            if any(tok in exc_str for tok in ("429", "rate", "quota",
                                               "500", "503", "unavailable")):
                wait = BACKOFF_BASE ** attempt
                # If error contains recommended retry delay, honor it
                import re
                retry_match = re.search(r"retry in (\d+(?:\.\d+)?)s", exc_str)
                if not retry_match:
                    retry_match = re.search(r"retrydelay': '(\d+)s", exc_str)
                if retry_match:
                    wait = max(wait, float(retry_match.group(1)) + 1.0)
                else:
                    wait = max(wait, 12.0)  # at least 12s on rate limit
                logger.warning("Retryable error (attempt %d): %s — sleeping %.1fs",
                               attempt, exc, wait)
                time.sleep(wait)
            else:
                raise   # non-retryable: bubble up immediately

    else:
        raise RuntimeError(
            f"All {MAX_ATTEMPTS} attempts failed. Last: {last_exc}") from last_exc

    # ── JSON parse (with one retry on bad output) ─────────────────────────────
    for parse_attempt in range(2):
        try:
            result = _parse_json_from_text(raw_text)
            break
        except json.JSONDecodeError:
            if parse_attempt == 0:
                logger.warning("JSON parse failed — retrying API call once")
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config,
                    )
                    raw_text = response.text
                except Exception as exc:
                    logger.error("Retry API call failed: %s", exc)
                    result = {"error": "invalid_json", "raw": raw_text}
                    break
            else:
                logger.error("JSON parse failed after retry. raw=%r", raw_text[:200])
                result = {"error": "invalid_json", "raw": raw_text}
                break

    # ── Cache write & log ─────────────────────────────────────────────────────
    with cpath.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    _log_call(model, temperature, cached=False)
    time.sleep(CALL_DELAY_S)   # rate-limit courtesy pause

    return result
