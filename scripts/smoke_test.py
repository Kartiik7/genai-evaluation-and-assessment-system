"""
scripts/smoke_test.py

Sends 3 short grading prompts drawn from dev.csv through generate_json()
and prints the parsed JSON responses.

Usage (from project root):
    python scripts/smoke_test.py
"""

import sys
import json
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from config import GEMINI_MODEL, DEFAULT_TEMPERATURE
from src.llm_client import generate_json

PROMPT_TEMPLATE = """\
You are an expert short-answer grader.

Question:
{question}

Reference answer:
{reference_answer}

Student answer:
{student_answer}

Grade the student answer on a scale from 0 to 5, where:
  0 = completely wrong / irrelevant
  5 = correct and complete

Respond ONLY with a JSON object in this exact format:
{{
  "score": <number from 0 to 5, decimals allowed>,
  "reasoning": "<one-sentence justification>"
}}
"""

def main():
    df_dev = pd.read_csv("data/processed/dev.csv")
    # Pick 3 rows spanning low, mid, high human scores
    samples = pd.concat([
        df_dev.nsmallest(1, "human_score"),
        df_dev.iloc[(len(df_dev) // 2) : (len(df_dev) // 2) + 1],
        df_dev.nlargest(1, "human_score"),
    ]).reset_index(drop=True)

    print(f"Model  : {GEMINI_MODEL}")
    print(f"Temp   : {DEFAULT_TEMPERATURE}")
    print("=" * 70)

    for i, row in samples.iterrows():
        prompt = PROMPT_TEMPLATE.format(
            question=row["question"],
            reference_answer=row["reference_answer"],
            student_answer=row["student_answer"],
        )

        print(f"\n[Sample {i+1}]")
        print(f"  Question : {row['question'][:80]}...")
        print(f"  Student  : {row['student_answer'][:80]}")
        print(f"  Human    : {row['human_score']}")

        result = generate_json(
            prompt=prompt,
            model=GEMINI_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            cache_key_extra=f"smoke_{i}",
        )

        print(f"  Gemini   : {json.dumps(result, indent=None)}")

    print("\n" + "=" * 70)
    print("Smoke test complete. Check results/llm_log.csv for call log.")

if __name__ == "__main__":
    main()
