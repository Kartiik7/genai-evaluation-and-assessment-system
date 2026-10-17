"""
src/llm_client.py

generate_json(prompt, model, temperature, cache_key_extra="", max_output_tokens=None)
  - Loads GEMINI_API_KEY from .env
  - Calls Gemini via google-genai (models.generate_content) with JSON response mode
  - Configures lowest thinking budget (THINKING_BUDGET = 0, off) across all methods
  - Caps max_output_tokens to prevent runaway token usage
  - Disk-caches every response in results/llm_cache/ keyed by SHA-256 hash of
    (model, temperature, prompt, cache_key_extra)
  - Stops immediately on quota error (429 resource_exhausted) without burning retries
  - Exponential back-off on transient 5xx server errors (max 3 attempts)
  - One JSON-parse retry; on second failure returns {"error": "invalid_json", "raw": ...}
  - Sleeps REQUEST_DELAY_SECONDS between calls to strictly respect RPM limits
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
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

try:
    from config import (
        THINKING_BUDGET,
        REQUEST_DELAY_SECONDS,
        DEFAULT_MAX_OUTPUT_TOKENS,
    )
except ImportError:
    THINKING_BUDGET = 0
    REQUEST_DELAY_SECONDS = 12.5
    DEFAULT_MAX_OUTPUT_TOKENS = 500

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

CACHE_DIR = Path("results/llm_cache")
LOG_PATH = Path("results/llm_log.csv")
CALL_DELAY_S = REQUEST_DELAY_SECONDS
MAX_ATTEMPTS = 3
BACKOFF_BASE = 2.0

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Lazy singleton client ─────────────────────────────────────────────────────
_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment / .env")
        _client = genai.Client(api_key=api_key)
    return _client


# ── Helpers ───────────────────────────────────────────────────────────────────
def _cache_key(model: str, temperature: float, prompt: str, cache_key_extra: str) -> str:
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
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


# ── Main public function ──────────────────────────────────────────────────────
def generate_json(
    prompt: str,
    model: str,
    temperature: float,
    cache_key_extra: str = "",
    max_output_tokens: Optional[int] = None,
) -> dict:
    """
    Call Gemini and return a parsed JSON dict.
    
    Parameters
    ----------
    prompt            : The full prompt string to send.
    model             : Gemini model ID.
    temperature       : Sampling temperature (0.0 = deterministic).
    cache_key_extra   : Optional string mixed into cache hash (e.g. row index).
    max_output_tokens : Optional ceiling on generated output tokens.
    """
    key = _cache_key(model, temperature, prompt, cache_key_extra)
    cpath = _cache_path(key)

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if cpath.exists():
        logger.info("CACHE HIT  model=%s temp=%s key=%s", model, temperature, key[:12])
        _log_call(model, temperature, cached=True)
        with cpath.open(encoding="utf-8") as fh:
            return json.load(fh)

    # ── API call with retry ───────────────────────────────────────────────────
    client = _get_client()

    # Lowest thinking budget (0 = off) across all methods
    thinking_cfg = types.ThinkingConfig(thinking_budget=THINKING_BUDGET)

    out_tokens = max_output_tokens if max_output_tokens is not None else DEFAULT_MAX_OUTPUT_TOKENS
    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        thinking_config=thinking_cfg,
        max_output_tokens=out_tokens,
    )

    last_exc = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            logger.info("API CALL   model=%s temp=%s attempt=%d/%d max_tokens=%d",
                        model, temperature, attempt, MAX_ATTEMPTS, out_tokens)
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            raw_text = response.text
            break
        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()

            # Quota exhausted: stop immediately as per integrity rules
            if any(tok in exc_str for tok in ("resource_exhausted", "quota", "daily limit", "free_tier_requests")):
                logger.error("FATAL: Gemini API quota exceeded (%s). Stopping immediately.", exc)
                raise RuntimeError(f"Gemini API quota exceeded. Stopping immediately: {exc}") from exc

            # Transient 5xx server errors
            if any(tok in exc_str for tok in ("500", "503", "unavailable", "server error")):
                wait = BACKOFF_BASE ** attempt
                logger.warning("Transient error (attempt %d/%d): %s — sleeping %.1fs",
                               attempt, MAX_ATTEMPTS, exc, wait)
                time.sleep(wait)
            else:
                raise

    else:
        raise RuntimeError(f"All {MAX_ATTEMPTS} attempts failed. Last: {last_exc}") from last_exc

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
    time.sleep(CALL_DELAY_S)

    return result
