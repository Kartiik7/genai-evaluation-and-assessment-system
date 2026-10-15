"""
scripts/run_llm_eval.py

Evaluates LLM grading on a dataset split using either direct or rubric prompting.

Usage:
    python scripts/run_llm_eval.py --method direct --split dev
    python scripts/run_llm_eval.py --method rubric --split test --temperature 0.0

Behavior:
- Loads the chosen split (data/processed/{split}.csv).
- Renders the prompt ({method}_v1.txt) with question, reference_answer, student_answer.
- Calls generate_json from src/llm_client.py (cache-enabled).
- Validates the output via src/llm_eval.py.
- Saves one row per response to results/llm_{method}_{split}.csv.
- Prints the number of parse failures and cache hits at the end.
- Safe to rerun if interrupted (the cache makes it resume instantly).
- Does NOT append anything to main_results.csv for the dev split.
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

from config import GEMINI_MODEL, DEFAULT_TEMPERATURE
from src.prompts import render
from src.llm_client import generate_json, _cache_key, _cache_path
from src.llm_eval import validate_response, RUBRIC_CRITERIA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM evaluation using Gemini Flash.")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["direct", "rubric"],
        help="Prompting method: 'direct' or 'rubric'.",
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
    return parser.parse_args()


def main():
    args = parse_args()

    split_file = Path(f"data/processed/{args.split}.csv")
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    df = pd.read_csv(split_file)
    logger.info("Loaded split '%s' with %d rows from %s", args.split, len(df), split_file)

    template_name = f"{args.method}_v1"
    output_path = Path(f"results/llm_{args.method}_{args.split}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cache_hits = 0
    parse_failures = 0
    records = []

    logger.info(
        "Starting LLM evaluation: method=%s, split=%s, model=%s, temp=%.2f",
        args.method,
        args.split,
        args.model,
        args.temperature,
    )

    for idx, row in df.iterrows():
        question = str(row["question"])
        reference_answer = str(row["reference_answer"])
        student_answer = str(row["student_answer"])
        human_score = float(row["human_score"])
        question_id = row.get("question_id", idx)

        # Render prompt
        rendered_prompt, prompt_version = render(
            template_name,
            question=question,
            reference_answer=reference_answer,
            student_answer=student_answer,
        )

        # Cache key check beforehand to accurately count cache hits
        key = _cache_key(args.model, args.temperature, rendered_prompt, cache_key_extra=f"{args.method}_{idx}")
        cpath = _cache_path(key)
        if cpath.exists():
            cache_hits += 1

        # Call LLM client
        raw_result = generate_json(
            prompt=rendered_prompt,
            model=args.model,
            temperature=args.temperature,
            cache_key_extra=f"{args.method}_{idx}",
        )

        # Validate structured output
        val_res = validate_response(raw_result, prompt_type=args.method)

        parse_ok = bool(val_res["is_valid"])
        if not parse_ok:
            parse_failures += 1
            logger.warning(
                "Row %d: parse failure. Error: %s", idx, val_res.get("error")
            )

        pred_score = val_res["score"]
        reasoning = val_res.get("reasoning", "")

        record = {
            "row_id": row.get("row_id", f"{args.split}_{idx:03d}"),
            "question_id": question_id,
            "human_score": human_score,
            "pred_score": pred_score,
            "parse_ok": parse_ok,
            "reasoning": reasoning,
            "prompt_version": prompt_version,
            "model": args.model,
            "temperature": args.temperature,
        }

        # Add rubric-specific fields
        if args.method == "rubric":
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
