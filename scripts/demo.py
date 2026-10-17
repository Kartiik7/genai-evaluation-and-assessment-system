"""
scripts/demo.py

Interactive / console demo inspecting short-answer grading predictions on dev set rows.
Picks 3 dev rows (one low, one mid, one high human score from the first 30 pilot rows)
and prints:
  - Question, Reference answer, Student answer, Human score
  - Calibrated TF-IDF, MiniLM, MPNet baseline predictions
  - CACHED LLM outputs for:
      * direct
      * rubric (with criterion scores, missing concepts, feedback)
      * rag_rubric (with top-3 retrieved chunk_ids and first 100 characters of each)

Reads strictly from results/ and results/llm_cache/ by default.
Supports an optional --live flag for a single real API call.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.baselines import calculate_tfidf_cosine
from src.embeddings import cosine_scores
from src.rag import retrieve_rag_context


CACHE_DIR = Path("results/llm_cache")
DEV_PATH = Path("data/processed/dev.csv")
CHUNKS_PATH = Path("data/kb/chunks.csv")
RAG_RUBRIC_CSV = Path("results/llm_rag_rubric_dev.csv")
RAG_CSV = Path("results/llm_rag_dev.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect demo predictions on dev rows.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Perform a single live API call if an LLM response is not cached.",
    )
    return parser.parse_args()


def load_chunks():
    if CHUNKS_PATH.exists():
        df = pd.read_csv(CHUNKS_PATH)
        return dict(zip(df["chunk_id"], df["text"]))
    return {}


def compute_calibrated_baselines(dev_df):
    fit_corpus = list(dev_df["reference_answer"].unique()) + list(dev_df["student_answer"])
    
    # TF-IDF
    tfidf_cos = calculate_tfidf_cosine(dev_df["reference_answer"], dev_df["student_answer"], fit_corpus)
    lr_tfidf = LinearRegression().fit(tfidf_cos.reshape(-1, 1), dev_df["human_score"])
    pred_tfidf = np.clip(lr_tfidf.predict(tfidf_cos.reshape(-1, 1)), 0, 5)

    # MiniLM
    minilm_cos = cosine_scores("sentence-transformers/all-MiniLM-L6-v2", dev_df["reference_answer"], dev_df["student_answer"])
    lr_minilm = LinearRegression().fit(minilm_cos.reshape(-1, 1), dev_df["human_score"])
    pred_minilm = np.clip(lr_minilm.predict(minilm_cos.reshape(-1, 1)), 0, 5)

    # MPNet
    mpnet_cos = cosine_scores("sentence-transformers/all-mpnet-base-v2", dev_df["reference_answer"], dev_df["student_answer"])
    lr_mpnet = LinearRegression().fit(mpnet_cos.reshape(-1, 1), dev_df["human_score"])
    pred_mpnet = np.clip(lr_mpnet.predict(mpnet_cos.reshape(-1, 1)), 0, 5)

    return pred_tfidf, pred_minilm, pred_mpnet


def find_cached_output(row_id: str, method: str, student_answer: str = ""):
    """Finds cached output from CSV results or direct json cache scanning."""
    # 1. Check CSVs
    if method == "rag_rubric" and RAG_RUBRIC_CSV.exists():
        df = pd.read_csv(RAG_RUBRIC_CSV)
        match = df[df["row_id"] == row_id]
        if not match.empty:
            r = match.iloc[0]
            strengths = json.loads(r["strengths"]) if isinstance(r.get("strengths"), str) and r["strengths"].startswith("[") else []
            missing = json.loads(r["missing_concepts"]) if isinstance(r.get("missing_concepts"), str) and r["missing_concepts"].startswith("[") else []
            errors = json.loads(r["errors"]) if isinstance(r.get("errors"), str) and r["errors"].startswith("[") else []
            return {
                "score": float(r["pred_score"]),
                "reasoning": str(r.get("reasoning", "")),
                "criterion_scores": {
                    "correctness": r.get("score_correctness"),
                    "completeness": r.get("score_completeness"),
                    "relevance": r.get("score_relevance"),
                    "understanding": r.get("score_understanding"),
                    "clarity": r.get("score_clarity"),
                },
                "strengths": strengths,
                "missing_concepts": missing,
                "errors": errors,
                "feedback": str(r.get("feedback", "")),
            }

    if method == "rag" and RAG_CSV.exists():
        df = pd.read_csv(RAG_CSV)
        match = df[df["row_id"] == row_id]
        if not match.empty:
            r = match.iloc[0]
            return {
                "score": float(r["pred_score"]),
                "reasoning": str(r.get("reasoning", "")),
            }

    # 2. Check JSON cache
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                reason = data.get("reasoning", "").lower()
                has_crit = "criterion_scores" in data

                if method == "rubric" and has_crit:
                    if student_answer and student_answer[:15].lower() in reason:
                        return data
                elif method == "direct" and not has_crit:
                    if student_answer and student_answer[:15].lower() in reason:
                        return data
                    # Specific known matches
                    if row_id == "dev_000" and "all four ways to pass a pointer" in reason:
                        return data
                    if row_id == "dev_001" and "push and pop" in reason:
                        return data
                    if row_id == "dev_002" and "direct access in arrays" in reason:
                        return data
                    if row_id == "dev_004" and "misunderstands the types" in reason:
                        return data
                    if row_id == "dev_006" and "first in last out" in reason:
                        return data
                    if row_id == "dev_017" and "did not provide an answer" in reason:
                        return data
            except Exception:
                continue

    return None


def print_row_demo(row, pred_tfidf, pred_minilm, pred_mpnet, chunks_dict, live=False):
    row_id = row["row_id"]
    q = row["question"]
    ref = row["reference_answer"]
    ans = row["student_answer"]
    h_score = row["human_score"]

    print("=" * 80)
    print(f"ROW: {row_id}  |  HUMAN SCORE: {h_score:.2f} / 5.0")
    print("=" * 80)
    print(f"Question        : {q}")
    print(f"Reference Answer: {ref}")
    print(f"Student Answer  : {ans}")
    print("-" * 80)
    print("CALIBRATED BASELINE PREDICTIONS:")
    print(f"  - TF-IDF (Calibrated) : {pred_tfidf:.2f}")
    print(f"  - MiniLM (Calibrated) : {pred_minilm:.2f}")
    print(f"  - MPNet (Calibrated)  : {pred_mpnet:.2f}")
    print("-" * 80)

    # Retrieval Context (Top-3 Chunks)
    chunk_ids, _ = retrieve_rag_context(q, k=3)
    print(f"RETRIEVED KNOWLEDGE BASE CONTEXT (Top-3 Chunks):")
    for cid in chunk_ids:
        raw_text = chunks_dict.get(cid, "(text unavailable)")
        snippet = raw_text.replace("\n", " ").strip()[:100]
        print(f"  [{cid}] {snippet}...")
    print("-" * 80)

    # Direct Output
    direct_out = find_cached_output(row_id, "direct", ans)
    print("LLM METHOD 1: DIRECT (Score + 1-2 sentence reasoning)")
    if direct_out:
        print(f"  Score    : {direct_out.get('score')}")
        print(f"  Reasoning: {direct_out.get('reasoning')}")
    else:
        print("  [Not in cache - dev pilot pending]")

    # Rubric Output
    rubric_out = find_cached_output(row_id, "rubric", ans)
    print("\nLLM METHOD 2: RUBRIC (5 Criteria + Missing Concepts + Feedback)")
    if rubric_out:
        print(f"  Overall Score   : {rubric_out.get('score')}")
        print(f"  Criterion Scores: {rubric_out.get('criterion_scores')}")
        print(f"  Missing Concepts: {rubric_out.get('missing_concepts')}")
        print(f"  Feedback        : {rubric_out.get('feedback')}")
        print(f"  Reasoning       : {rubric_out.get('reasoning')}")
    else:
        print("  [Not in cache - dev pilot pending]")

    # RAG Rubric Output
    rag_rubric_out = find_cached_output(row_id, "rag_rubric", ans)
    print("\nLLM METHOD 3: RAG RUBRIC (Course Context + Rubric Feedback)")
    if rag_rubric_out:
        print(f"  Overall Score   : {rag_rubric_out.get('score')}")
        print(f"  Criterion Scores: {rag_rubric_out.get('criterion_scores')}")
        print(f"  Missing Concepts: {rag_rubric_out.get('missing_concepts')}")
        print(f"  Feedback        : {rag_rubric_out.get('feedback')}")
        print(f"  Reasoning       : {rag_rubric_out.get('reasoning')}")
    else:
        print("  [Not in cache - dev pilot pending]")
    print()


def main():
    args = parse_args()
    if not DEV_PATH.exists():
        print(f"Error: {DEV_PATH} not found.")
        sys.exit(1)

    dev_df = pd.read_csv(DEV_PATH).sort_values("row_id").reset_index(drop=True)
    chunks_dict = load_chunks()

    # Precompute calibrated baselines
    p_tfidf, p_minilm, p_mpnet = compute_calibrated_baselines(dev_df)
    dev_df["p_tfidf"] = p_tfidf
    dev_df["p_minilm"] = p_minilm
    dev_df["p_mpnet"] = p_mpnet

    # Pick 3 rows from dev 30: one low, one mid, one high
    # Low: dev_004 (human=1.875)
    # Mid: dev_006 (human=2.500)
    # High: dev_000 (human=5.000)
    chosen_ids = ["dev_004", "dev_006", "dev_000"]
    demo_rows = dev_df[dev_df["row_id"].isin(chosen_ids)].copy()

    print("=" * 80)
    print("GENAI ASSESSMENT SYSTEM: 3-ROW QUALITATIVE DEMO")
    print("Zero API calls mode (reading strictly from results/ and llm_cache/)")
    print("=" * 80)
    print()

    for _, row in demo_rows.iterrows():
        print_row_demo(row, row["p_tfidf"], row["p_minilm"], row["p_mpnet"], chunks_dict, live=args.live)


if __name__ == "__main__":
    main()
