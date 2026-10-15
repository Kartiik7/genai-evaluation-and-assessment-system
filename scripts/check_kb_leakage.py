"""
scripts/check_kb_leakage.py

Audits knowledge base chunks for word-level n-gram leakage (n >= 8) against
reference answers and student answers in data/processed/mohler_clean.csv.

Computes the longest consecutive matching word sequence between each chunk
and any target answer. Lists any chunk with an 8-gram or longer match for review.
Does not delete or alter any chunks.
"""

import os
import sys
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Set
import pandas as pd


def tokenize(text: str) -> List[str]:
    """Tokenize string into lowercase alphanumeric words."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    return re.findall(r"\b[a-zA-Z0-9']+\b", text.lower())


def build_target_index(mohler_df: pd.DataFrame, n: int = 8) -> Tuple[Dict[Tuple[str, ...], List[Dict]], int]:
    """
    Build an index mapping each n-gram to occurrences in reference and student answers.
    """
    ngram_index = defaultdict(list)
    total_targets = 0

    # Index unique reference answers
    refs = mohler_df[["question_id", "reference_answer"]].drop_duplicates()
    for _, row in refs.iterrows():
        tokens = tokenize(row["reference_answer"])
        total_targets += 1
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i : i + n])
            ngram_index[gram].append({
                "type": "reference_answer",
                "question_id": row["question_id"],
                "target_text": row["reference_answer"],
                "tokens": tokens,
                "start_idx": i,
            })

    # Index unique student answers
    students = mohler_df[["question_id", "student_answer"]].drop_duplicates()
    for _, row in students.iterrows():
        tokens = tokenize(row["student_answer"])
        total_targets += 1
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i : i + n])
            ngram_index[gram].append({
                "type": "student_answer",
                "question_id": row["question_id"],
                "target_text": row["student_answer"],
                "tokens": tokens,
                "start_idx": i,
            })

    return ngram_index, total_targets


def check_chunk_leakage(
    chunk_text: str,
    ngram_index: Dict[Tuple[str, ...], List[Dict]],
    n: int = 8,
) -> List[Dict]:
    """
    Check a single chunk text against the n-gram index.
    Returns all distinct matches with their longest extended overlap length.
    """
    tokens = tokenize(chunk_text)
    if len(tokens) < n:
        return []

    matches = []
    seen_matches: Set[Tuple[str, float, str]] = set()

    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        if gram in ngram_index:
            for hit in ngram_index[gram]:
                t_tokens = hit["tokens"]
                j = hit["start_idx"]

                # Extend match as far as possible
                k = n
                while (i + k < len(tokens)) and (j + k < len(t_tokens)) and (tokens[i + k] == t_tokens[j + k]):
                    k += 1

                overlap_phrase = " ".join(tokens[i : i + k])
                match_key = (hit["type"], hit["question_id"], overlap_phrase)
                if match_key not in seen_matches:
                    seen_matches.add(match_key)
                    matches.append({
                        "overlap_length": k,
                        "overlap_phrase": overlap_phrase,
                        "match_type": hit["type"],
                        "question_id": hit["question_id"],
                        "target_text": hit["target_text"],
                    })

    return matches


def parse_args():
    parser = argparse.ArgumentParser(description="Check KB chunks for n-gram leakage with Mohler dataset.")
    parser.add_argument(
        "--chunks",
        type=str,
        default="data/kb/chunks.csv",
        help="Path to chunks CSV (default: data/kb/chunks.csv).",
    )
    parser.add_argument(
        "--mohler",
        type=str,
        default="data/processed/mohler_clean.csv",
        help="Path to mohler_clean.csv (default: data/processed/mohler_clean.csv).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=8,
        help="Minimum n-gram length to check (default: 8).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    chunks_path = Path(args.chunks)
    mohler_path = Path(args.mohler)

    if not chunks_path.exists():
        print(f"Error: Chunks file '{chunks_path}' does not exist. Run scripts/build_kb.py first.")
        sys.exit(1)

    if not mohler_path.exists():
        print(f"Error: Mohler file '{mohler_path}' does not exist.")
        sys.exit(1)

    print("=" * 70)
    print(f"KB LEAKAGE AUDIT (n-gram >= {args.n})")
    print("=" * 70)

    chunks_df = pd.read_csv(chunks_path)
    mohler_df = pd.read_csv(mohler_path)
    print(f"Loaded {len(chunks_df)} chunks from {chunks_path}")
    print(f"Loaded {len(mohler_df)} items from {mohler_path}")

    print(f"Indexing {args.n}-grams from reference and student answers...")
    ngram_index, total_targets = build_target_index(mohler_df, n=args.n)
    print(f"Indexed {len(ngram_index):,} unique {args.n}-grams across {total_targets:,} target texts.\n")

    leaked_chunks = []
    for _, row in chunks_df.iterrows():
        chunk_id = row["chunk_id"]
        topic = row.get("topic", "")
        source_file = row.get("source_file", "")
        text = str(row.get("text", ""))

        matches = check_chunk_leakage(text, ngram_index, n=args.n)
        if matches:
            # Sort by longest overlap length descending
            matches.sort(key=lambda x: x["overlap_length"], reverse=True)
            longest_match = matches[0]
            leaked_chunks.append({
                "chunk_id": chunk_id,
                "topic": topic,
                "source_file": source_file,
                "longest_overlap": longest_match["overlap_length"],
                "match_count": len(matches),
                "matches": matches,
            })

    print("-" * 70)
    print(f"RESULTS: {len(leaked_chunks)} of {len(chunks_df)} chunks contain an {args.n}+ gram match.")
    print("-" * 70)

    if leaked_chunks:
        print("\nList of matching chunks for human review (NO chunks deleted):")
        for item in leaked_chunks:
            top_match = item["matches"][0]
            print(f"\n[Chunk ID]: {item['chunk_id']} (File: {item['source_file']}, Topic: {item['topic']})")
            print(f"  Longest overlap length : {top_match['overlap_length']} words (Total {item['match_count']} matching phrases)")
            print(f"  Matched target         : {top_match['match_type']} (question_id: {top_match['question_id']})")
            print(f"  Matching phrase        : \"{top_match['overlap_phrase']}\"")
            print(f"  Full target text       : \"{top_match['target_text']}\"")
    else:
        print(f"\nNo {args.n}-gram word-level overlaps detected between KB chunks and Mohler dataset.")
        print("Knowledge base chunks are clean of verbatim 8+ gram leakage.")

    print("\n" + "=" * 70)
    print("Audit complete. Note: As per policy, no files or chunks were modified or deleted.")
    print("=" * 70)


if __name__ == "__main__":
    main()
