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
from src.rag import build_rag_context


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
        default="rubric_v2",
        choices=["direct_v1", "direct_v2", "rubric_v1", "rubric_v2", "rag_rubric_v2"],
        help="Grading method / prompt version (default: rubric_v2).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["dev", "test"],
        help="Data split to evaluate (default: dev).",
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


def main():
    args = parse_args()
    data_path = Path(args.data) if args.data else Path(f"data/processed/{args.split}.csv")
    cache_dir = Path(args.cache_dir)

    if not data_path.exists():
        print(f"Error: dataset file not found at {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    total_rows = len(df)

    # If method is RAG, pre-load retrieval top-3 and chunk texts
    rag_context_map = {}
    if "rag" in args.method.lower():
        top3_path = Path("results/retrieval_top3.csv")
        chunks_path = Path("data/kb/chunks.csv")
        if top3_path.exists() and chunks_path.exists():
            top3_df = pd.read_csv(top3_path)
            chunks_df = pd.read_csv(chunks_path).set_index("chunk_id")["text"].to_dict()
            for qid, group in top3_df.groupby("question_id"):
                top_chunks = group.sort_values("rank").head(RAG_TOP_K)["chunk_id"].tolist()
                texts = [chunks_df.get(cid, "") for cid in top_chunks]
                rag_context_map[str(qid)] = build_rag_context(texts, max_chunks=RAG_TOP_K, max_words_per_chunk=RAG_MAX_CHUNK_WORDS)
                rag_context_map[float(qid) if "." in str(qid) else qid] = rag_context_map[str(qid)]

    # Determine max output tokens
    method_key = args.method.lower()
    max_out_tok = MAX_OUTPUT_TOKENS.get(method_key, DEFAULT_MAX_OUTPUT_TOKENS)

    cached_count = 0
    uncached_count = 0
    total_input_tokens = 0
    uncached_input_tokens = 0

    for idx, row in df.iterrows():
        q = str(row.get("question", ""))
        ref = str(row.get("reference_answer", ""))
        ans = str(row.get("student_answer", ""))
        qid = row.get("question_id", "")
        row_id = str(row.get("row_id", idx))

        fields = {
            "question": q,
            "reference_answer": ref,
            "student_answer": ans,
        }
        if "rag" in args.method.lower():
            fields["context"] = rag_context_map.get(qid, rag_context_map.get(str(qid), ""))

        prompt, _ = render(args.method, **fields)
        in_tok = approx_tokens(prompt)
        total_input_tokens += in_tok

        key = _cache_key(args.model, args.temperature, prompt, cache_key_extra=row_id)
        cpath = cache_dir / f"{key}.json"

        if cpath.exists():
            cached_count += 1
        else:
            uncached_count += 1
            uncached_input_tokens += in_tok

    uncached_max_out = uncached_count * max_out_tok
    est_runtime_s = uncached_count * REQUEST_DELAY_SECONDS

    print("=" * 60)
    print("PRE-RUN COST & TOKEN ESTIMATOR (NO API CALLS MADE)")
    print("=" * 60)
    print(f"Split                : {args.split} ({data_path})")
    print(f"Method / Prompt      : {args.method}")
    print(f"Model ID             : {args.model}")
    print(f"Total Rows           : {total_rows}")
    print(f"Cached Calls         : {cached_count} ({cached_count/total_rows*100:.1f}%)")
    print(f"Uncached Calls       : {uncached_count} ({uncached_count/total_rows*100:.1f}%)")
    print("-" * 60)
    print(f"Uncached Input Toks  : ~{uncached_input_tokens:,} tokens (avg ~{uncached_input_tokens/max(1, uncached_count):.0f}/call)")
    print(f"Uncached Max Out Toks: ~{uncached_max_out:,} tokens (cap {max_out_tok}/call)")
    print(f"Total Uncached Toks  : ~{uncached_input_tokens + uncached_max_out:,} tokens")
    print(f"Est. Runtime Spacing : ~{est_runtime_s/60:.1f} minutes ({REQUEST_DELAY_SECONDS}s delay/call)")
    print("=" * 60)


if __name__ == "__main__":
    main()
