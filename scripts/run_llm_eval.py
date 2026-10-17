"""
scripts/run_llm_eval.py

Evaluates LLM grading on a dataset split using direct, rubric, rag, or rag_rubric prompting.

Usage:
    python scripts/run_llm_eval.py --method direct --split dev
    python scripts/run_llm_eval.py --method rubric --split dev
    python scripts/run_llm_eval.py --method rag --split dev
    python scripts/run_llm_eval.py --method rag_rubric --split dev

Behavior:
- Loads the chosen split (data/processed/{split}.csv).
- For RAG methods, retrieves top-3 chunks using ONLY the question text, truncates each
  to ~150 words, and populates {retrieved_context}.
- Renders the prompt ({method}_v1.txt).
- Calls generate_json from src/llm_client.py (disk-cached, quota-guarded, rate-limited).
- Validates output via src/llm_eval.py.
- Saves one row per response to results/llm_{method}_{split}.csv with retrieved_chunk_ids.
- Prints evaluation summary at the end.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    GEMINI_MODEL,
    DEFAULT_TEMPERATURE,
    MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS,
)
from src.prompts import render
from src.llm_client import generate_json, _cache_key, _cache_path
from src.llm_eval import validate_response, RUBRIC_CRITERIA
from src.rag import retrieve_rag_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM evaluation using Gemini Flash.")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["direct", "rubric", "rag", "rag_rubric"],
        help="Prompting method: 'direct', 'rubric', 'rag', or 'rag_rubric'.",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["dev", "test"],
        help="Dataset split: 'dev' or 'test'.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=GEMINI_MODEL,
        help=f"Gemini model ID (default: {GEMINI_MODEL}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit (e.g. 3 for smoke testing).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    split_file = Path(f"data/processed/{args.split}.csv")
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    df = pd.read_csv(split_file)
    if args.limit is not None and args.limit > 0:
        df = df.head(args.limit)
        logger.info("Limiting evaluation to first %d rows for smoke test.", len(df))
    else:
        logger.info("Loaded split '%s' with %d rows from %s", args.split, len(df), split_file)

    template_name = f"{args.method}_v1"
    output_path = Path(f"results/llm_{args.method}_{args.split}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine max output tokens cap
    max_out_tokens = MAX_OUTPUT_TOKENS.get(args.method, DEFAULT_MAX_OUTPUT_TOKENS)

    cache_hits = 0
    parse_failures = 0
    records = []

    logger.info(
        "Starting LLM evaluation: method=%s, split=%s, model=%s, temp=%.2f, max_out_tokens=%d",
        args.method,
        args.split,
        args.model,
        args.temperature,
        max_out_tokens,
    )

    for idx, row in df.iterrows():
        question = str(row["question"])
        reference_answer = str(row["reference_answer"])
        student_answer = str(row["student_answer"])
        human_score = float(row["human_score"])
        question_id = row.get("question_id", idx)
        row_id = str(row.get("row_id", f"{args.split}_{idx:03d}"))

        retrieved_chunk_ids = []
        render_kwargs = {
            "question": question,
            "reference_answer": reference_answer,
            "student_answer": student_answer,
        }

        # For RAG methods, retrieve context using ONLY question text
        if args.method in ("rag", "rag_rubric"):
            chunk_ids, context_text = retrieve_rag_context(question=question)
            retrieved_chunk_ids = chunk_ids
            render_kwargs["retrieved_context"] = context_text

        # Render prompt
        rendered_prompt, prompt_version = render(template_name, **render_kwargs)

        cache_extra = f"{args.method}_{row_id}"
        key = _cache_key(args.model, args.temperature, rendered_prompt, cache_key_extra=cache_extra)
        cpath = _cache_path(key)
        if cpath.exists():
            cache_hits += 1

        # Call LLM client
        raw_result = generate_json(
            prompt=rendered_prompt,
            model=args.model,
            temperature=args.temperature,
            cache_key_extra=cache_extra,
            max_output_tokens=max_out_tokens,
        )

        # Validate structured output
        val_res = validate_response(raw_result, prompt_type=args.method)

        parse_ok = bool(val_res["is_valid"])
        if not parse_ok:
            parse_failures += 1
            logger.warning("Row %s: parse failure. Error: %s", row_id, val_res.get("error"))

        pred_score = val_res["score"]
        reasoning = val_res.get("reasoning", "")

        record = {
            "row_id": row_id,
            "question_id": question_id,
            "human_score": human_score,
            "pred_score": pred_score,
            "parse_ok": parse_ok,
            "reasoning": reasoning,
            "retrieved_chunk_ids": ",".join(retrieved_chunk_ids),
            "prompt_version": prompt_version,
            "model": args.model,
            "temperature": args.temperature,
        }

        # Add rubric-specific fields
        if args.method in ("rubric", "rag_rubric"):
            crit_scores = val_res.get("criterion_scores", {})
            for crit in RUBRIC_CRITERIA:
                record[f"score_{crit}"] = crit_scores.get(crit, np.nan)

            record["strengths"] = json.dumps(val_res.get("strengths", []), ensure_ascii=False)
            record["missing_concepts"] = json.dumps(val_res.get("missing_concepts", []), ensure_ascii=False)
            record["errors"] = json.dumps(val_res.get("errors", []), ensure_ascii=False)
            record["feedback"] = val_res.get("feedback", "")

        records.append(record)

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_path, index=False)
    logger.info("Saved %d evaluation rows to %s", len(out_df), output_path)

    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY: {args.method} ({args.split} split)")
    print("=" * 60)
    print(f"Total items evaluated : {len(df)}")
    print(f"Cache hits            : {cache_hits} / {len(df)}")
    print(f"Parse failures        : {parse_failures} / {len(df)}")
    print(f"Valid responses       : {len(df) - parse_failures} / {len(df)}")
    print(f"Output saved to       : {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
