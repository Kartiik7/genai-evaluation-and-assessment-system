"""
scripts/make_retrieval_labels.py

Creates a labeling dataset for retrieval evaluation:
1. Selects 30 unique questions from the TEST set (data/processed/test.csv) with random_state=42.
2. For each question, retrieves the top-10 knowledge base chunks using FAISS cosine similarity.
3. Shuffles the 10 retrieved chunks for each question to avoid rank bias and hides ranks and scores.
4. Outputs results to results/retrieval_labeling.csv with columns:
     - question_id
     - question_text
     - chunk_id
     - chunk_text
     - relevant (blank, to be manually filled with 1 or 0)
"""

import os
import sys
import random
import argparse
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import RETRIEVAL_MODEL_NAME, RETRIEVAL_QUERY_MODE
from src.retrieval import build_index, retrieve, format_query


def parse_args():
    parser = argparse.ArgumentParser(description="Generate retrieval labeling CSV for manual annotation.")
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
        default="results/retrieval_labeling.csv",
        help="Path for labeling output CSV (default: results/retrieval_labeling.csv).",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=30,
        help="Number of unique questions to sample (default: 30).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of retrieved chunks per question to pool (default: 10).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling questions and shuffling chunks (default: 42).",
    )
    parser.add_argument(
        "--query-mode",
        type=str,
        default=RETRIEVAL_QUERY_MODE,
        choices=["question", "question+reference"],
        help="Retrieval query mode (default: question).",
    )
    parser.add_argument(
        "--hide-question-text",
        action="store_true",
        help="If set, omit question_text column from the CSV.",
    )
    return parser.parse_args()


def generate_labeling_dataset(
    test_path: str = "data/processed/test.csv",
    chunks_path: str = "data/kb/chunks.csv",
    index_path: str = "data/kb/index.faiss",
    output_path: str = "results/retrieval_labeling.csv",
    n_questions: int = 30,
    k: int = 10,
    random_state: int = 42,
    query_mode: str = RETRIEVAL_QUERY_MODE,
    include_question_text: bool = True,
) -> pd.DataFrame:
    """
    Selects n_questions unique questions from test_path with random_state,
    retrieves top-k chunks, shuffles them per question, and saves to output_path.
    """
    test_path = Path(test_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure index exists
    index_path = Path(index_path)
    chunks_path = Path(chunks_path)
    if not index_path.exists():
        print(f"Index not found at {index_path}. Building index...")
        build_index(chunks_path=chunks_path, index_path=index_path, model_name=RETRIEVAL_MODEL_NAME)

    # Load test questions
    test_df = pd.read_csv(test_path)
    unique_questions = (
        test_df[["question_id", "question"]]
        .drop_duplicates(subset=["question_id"])
        .sort_values("question_id")
        .reset_index(drop=True)
    )

    if len(unique_questions) < n_questions:
        raise ValueError(
            f"Requested {n_questions} questions, but test set only has {len(unique_questions)} unique questions."
        )

    # Sample n_questions with random_state
    sampled_questions = (
        unique_questions.sample(n=n_questions, random_state=random_state)
        .sort_values("question_id")
        .reset_index(drop=True)
    )

    print(f"Sampled {len(sampled_questions)} unique questions from {len(unique_questions)} total test questions.")
    print(f"Query mode: '{query_mode}' | Retrieving top-{k} chunks per question...")

    rows = []
    rng = random.Random(random_state)

    for i, (_, q_row) in enumerate(sampled_questions.iterrows()):
        qid = q_row["question_id"]
        q_text = q_row["question"]

        query = format_query(question=q_text, mode=query_mode)
        top_chunk_ids, top_texts, _ = retrieve(
            query=query,
            k=k,
            index_path=index_path,
            chunks_path=chunks_path,
            model_name=RETRIEVAL_MODEL_NAME,
        )

        # Prepare chunk entries
        q_items = []
        for cid, text in zip(top_chunk_ids, top_texts):
            item = {
                "question_id": qid,
                "chunk_id": cid,
                "chunk_text": text,
                "relevant": "",  # Blank, to be filled by human labeler (1 or 0)
            }
            if include_question_text:
                item["question_text"] = q_text
            q_items.append(item)

        # Shuffle the retrieved chunks for this question to avoid rank bias
        rng.shuffle(q_items)
        rows.extend(q_items)

    # Assemble dataframe with exact column order
    if include_question_text:
        columns = ["question_id", "question_text", "chunk_id", "chunk_text", "relevant"]
    else:
        columns = ["question_id", "chunk_id", "chunk_text", "relevant"]

    out_df = pd.DataFrame(rows)[columns]
    out_df.to_csv(output_path, index=False)

    print(f"Saved labeling dataset to {output_path} with {len(out_df)} rows ({len(sampled_questions)} questions x {k} chunks).")
    print("Columns:", list(out_df.columns))
    print("Relevant column left blank for manual labeling.")
    return out_df


def main():
    args = parse_args()
    generate_labeling_dataset(
        test_path=args.test,
        chunks_path=args.chunks,
        index_path=args.index,
        output_path=args.output,
        n_questions=args.n_questions,
        k=args.k,
        random_state=args.random_state,
        query_mode=args.query_mode,
        include_question_text=not args.hide_question_text,
    )


if __name__ == "__main__":
    main()
