"""
scripts/retrieve_test_questions.py

Retrieves the top-3 knowledge base chunks for each unique question in the dev
and test splits using the configured retrieval mode (question text only).
Outputs results to results/retrieval_top3.csv with columns:
  - question_id
  - rank
  - chunk_id
  - score
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import RETRIEVAL_QUERY_MODE, RETRIEVAL_MODEL_NAME
from src.retrieval import build_index, retrieve, format_query


def parse_args():
    parser = argparse.ArgumentParser(description="Retrieve top-3 KB chunks for all unique dev/test questions.")
    parser.add_argument(
        "--dev",
        type=str,
        default="data/processed/dev.csv",
        help="Path to dev.csv (default: data/processed/dev.csv).",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="data/processed/test.csv",
        help="Path to test.csv (default: data/processed/test.csv).",
    )
    parser.add_argument(
        "--chunks",
        type=str,
        default="data/kb/chunks.csv",
        help="Path to chunks.csv (default: data/kb/chunks.csv).",
    )
    parser.add_argument(
        "--index",
        type=str,
        default="data/kb/index.faiss",
        help="Path to index.faiss (default: data/kb/index.faiss).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/retrieval_top3.csv",
        help="Path for output retrieval CSV (default: results/retrieval_top3.csv).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of top chunks to retrieve per question (default: 3).",
    )
    parser.add_argument(
        "--query-mode",
        type=str,
        default=RETRIEVAL_QUERY_MODE,
        choices=["question", "question+reference"],
        help="Query formulation mode (default: question).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dev_path = Path(args.dev)
    test_path = Path(args.test)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure index exists
    index_path = Path(args.index)
    chunks_path = Path(args.chunks)
    if not index_path.exists():
        print(f"Index not found at {index_path}. Building index...")
        build_index(chunks_path=chunks_path, index_path=index_path, model_name=RETRIEVAL_MODEL_NAME)

    # Load dev and test datasets to find all unique questions
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)
    combined = pd.concat([dev_df, test_df], ignore_index=True)

    # Drop duplicate questions based on question_id
    unique_questions = (
        combined[["question_id", "question", "reference_answer"]]
        .drop_duplicates(subset=["question_id"])
        .sort_values("question_id")
    )
    print(f"Loaded {len(unique_questions)} unique questions across dev ({len(dev_df)} rows) and test ({len(test_df)} rows).")
    print(f"Retrieval query mode: '{args.query_mode}' (model: {RETRIEVAL_MODEL_NAME})")

    results = []
    for _, row in unique_questions.iterrows():
        qid = row["question_id"]
        q_text = row["question"]
        ref_text = row.get("reference_answer", "")

        query = format_query(question=q_text, reference=ref_text, mode=args.query_mode)
        chunk_ids, texts, scores = retrieve(
            query=query,
            k=args.k,
            index_path=index_path,
            chunks_path=chunks_path,
            model_name=RETRIEVAL_MODEL_NAME,
        )

        for rank, (cid, score) in enumerate(zip(chunk_ids, scores), start=1):
            results.append({
                "question_id": qid,
                "rank": rank,
                "chunk_id": cid,
                "score": round(score, 6),
            })

    out_df = pd.DataFrame(results, columns=["question_id", "rank", "chunk_id", "score"])
    out_df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("RETRIEVAL SUMMARY")
    print("=" * 60)
    print(f"Total unique questions queried : {len(unique_questions)}")
    print(f"Total retrieval rows generated : {len(out_df)}")
    print(f"Output saved to                : {output_path}")
    print("Sample retrieved rows:")
    print(out_df.head(9).to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
