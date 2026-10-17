"""
scripts/estimate_cost.py

Pre-run estimator for LLM grading calls.
Prints for a given method and split:
  - total rows
  - cached calls count
  - uncached calls count (requires API calls)
  - approximate input tokens
  - approximate max output tokens
  - estimated runtime under RPM limits
MAKES ZERO LLM API CALLS.
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    GEMINI_MODEL,
    DEFAULT_TEMPERATURE,
    MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    REQUEST_DELAY_SECONDS,
    RAG_TOP_K,
    RAG_MAX_CHUNK_WORDS,
)
from src.prompts import render
from src.rag import retrieve_rag_context


def _cache_key(model: str, temperature: float, prompt: str, cache_key_extra: str) -> str:
    raw = f"{model}|{temperature}|{prompt}|{cache_key_extra}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def approx_tokens(text: str) -> int:
    """Standard approximation for Gemini models: ~4 characters per token."""
    return (len(text) + 3) // 4


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate token usage and uncached calls for a grading run.")
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["direct", "rubric", "rag", "rag_rubric", "all", "direct_v2", "rubric_v2", "rag_v2", "rag_rubric_v2"],
        help="Grading method / prompt version (default: all).",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default="v2",
        help="Prompt version family (default: v2).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["dev", "test"],
        help="Data split to evaluate (default: dev).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit (e.g. 30).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to CSV data file (default: data/processed/{split}.csv).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="results/llm_cache",
        help="Path to LLM disk cache directory (default: results/llm_cache).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=GEMINI_MODEL,
        help="Model ID (default from config.GEMINI_MODEL).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperature (default from config.DEFAULT_TEMPERATURE).",
    )
    return parser.parse_args()


def estimate_single_method(method: str, version: str, df: pd.DataFrame, args, cache_dir: Path):
    clean_method = method.replace(f"_{version}", "")
    template_name = f"{clean_method}_{version}" if not method.endswith(f"_{version}") else method

    max_out_tok = MAX_OUTPUT_TOKENS.get(clean_method, DEFAULT_MAX_OUTPUT_TOKENS)

    cached_count = 0
    uncached_count = 0
    total_input_tokens = 0
    uncached_input_tokens = 0

    for idx, row in df.iterrows():
        q = str(row.get("question", ""))
        ref = str(row.get("reference_answer", ""))
        ans = str(row.get("student_answer", ""))
        row_id = str(row.get("row_id", f"{args.split}_{idx:03d}"))

        fields = {
            "question": q,
            "reference_answer": ref,
            "student_answer": ans,
        }
        if clean_method in ("rag", "rag_rubric"):
            _, ctx = retrieve_rag_context(q)
            fields["context"] = ctx
            fields["retrieved_context"] = ctx

        prompt, _ = render(template_name, **fields)
        in_tok = approx_tokens(prompt)
        total_input_tokens += in_tok

        cache_extra = f"{clean_method}_{row_id}"
        key = _cache_key(args.model, args.temperature, prompt, cache_key_extra=cache_extra)
        cpath = cache_dir / f"{key}.json"

        if cpath.exists():
            cached_count += 1
        else:
            uncached_count += 1
            uncached_input_tokens += in_tok

    total_rows = len(df)
    uncached_max_out = uncached_count * max_out_tok
    est_runtime_s = uncached_count * REQUEST_DELAY_SECONDS

    return {
        "method": clean_method,
        "template": template_name,
        "total_rows": total_rows,
        "cached": cached_count,
        "uncached": uncached_count,
        "uncached_in_tok": uncached_input_tokens,
        "uncached_max_out_tok": uncached_max_out,
        "total_uncached_tok": uncached_input_tokens + uncached_max_out,
        "est_runtime_min": est_runtime_s / 60.0,
    }


def main():
    args = parse_args()
    data_path = Path(args.data) if args.data else Path(f"data/processed/{args.split}.csv")
    cache_dir = Path(args.cache_dir)

    if not data_path.exists():
        print(f"Error: dataset file not found at {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    if "row_id" in df.columns:
        df = df.sort_values("row_id").reset_index(drop=True)

    if args.limit is not None and args.limit > 0:
        df = df.head(args.limit)

    methods = ["direct", "rubric", "rag", "rag_rubric"] if args.method == "all" else [args.method]

    print("=" * 68)
    print("PRE-RUN COST & TOKEN ESTIMATOR (ZERO API CALLS MADE)")
    print(f"Split: {args.split} (first {len(df)} rows sorted by row_id), Temp: {args.temperature}, Model: {args.model}")
    print("=" * 68)

    total_uncached_all = 0
    total_tokens_all = 0
    total_runtime_all = 0.0

    print(f"{'Method':<12} {'Template':<15} {'Rows':<6} {'Cached':<8} {'Uncached':<10} {'Est. InTok':<12} {'Est. Time'}")
    print("-" * 68)

    for m in methods:
        res = estimate_single_method(m, args.prompt_version, df, args, cache_dir)
        total_uncached_all += res["uncached"]
        total_tokens_all += res["total_uncached_tok"]
        total_runtime_all += res["est_runtime_min"]
        print(
            f"{res['method']:<12} {res['template']:<15} {res['total_rows']:<6} "
            f"{res['cached']:<8} {res['uncached']:<10} ~{res['uncached_in_tok']:<11,} "
            f"~{res['est_runtime_min']:.1f}m"
        )

    print("-" * 68)
    print(f"TOTAL UNCACHED CALLS ACROSS ALL METHODS: {total_uncached_all}")
    print(f"TOTAL UNCACHED TOKENS (IN + MAX OUT)  : ~{total_tokens_all:,}")
    print(f"TOTAL ESTIMATED RUNTIME (SPACED)      : ~{total_runtime_all:.1f} minutes")
    print("=" * 68)


if __name__ == "__main__":
    main()
