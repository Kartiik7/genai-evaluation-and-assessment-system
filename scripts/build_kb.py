"""
scripts/build_kb.py

Builds knowledge base chunks from markdown files in data/kb/raw/.
Outputs chunks to data/kb/chunks.csv and reports chunk statistics:
  - total chunk count
  - mean, min, and max words per chunk
  - chunk count per file
"""

import os
import sys
import argparse
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.kb import build_kb


def parse_args():
    parser = argparse.ArgumentParser(description="Build Knowledge Base chunks from markdown files.")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/kb/raw",
        help="Directory containing raw markdown files (default: data/kb/raw).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/kb/chunks.csv",
        help="Path for exported chunks CSV (default: data/kb/chunks.csv).",
    )
    parser.add_argument(
        "--target-min-words",
        type=int,
        default=120,
        help="Target minimum words per chunk (default: 120).",
    )
    parser.add_argument(
        "--target-max-words",
        type=int,
        default=250,
        help="Target maximum words per chunk (default: 250).",
    )
    parser.add_argument(
        "--overlap-words",
        type=int,
        default=20,
        help="Approximate word overlap between consecutive chunks (default: 20).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_path = Path(args.output)

    if not raw_dir.exists():
        print(f"Warning: Directory '{raw_dir}' does not exist. Creating it.")
        raw_dir.mkdir(parents=True, exist_ok=True)

    df_chunks = build_kb(
        raw_dir=raw_dir,
        output_path=output_path,
        target_min_words=args.target_min_words,
        target_max_words=args.target_max_words,
        overlap_words=args.overlap_words,
    )

    num_chunks = len(df_chunks)
    print("\n" + "=" * 60)
    print("KNOWLEDGE BASE CHUNKING REPORT")
    print("=" * 60)
    print(f"Total number of chunks : {num_chunks}")

    if num_chunks > 0:
        mean_words = df_chunks["word_count"].mean()
        min_words = df_chunks["word_count"].min()
        max_words = df_chunks["word_count"].max()
        print(f"Mean words per chunk   : {mean_words:.1f}")
        print(f"Min words per chunk    : {min_words}")
        print(f"Max words per chunk    : {max_words}")
        print("\nChunks per file:")
        per_file = df_chunks.groupby("source_file").size()
        for filename, count in per_file.items():
            print(f"  - {filename:<35} : {count:>3} chunks")
    else:
        print("Mean words per chunk   : N/A")
        print("Min words per chunk    : N/A")
        print("Max words per chunk    : N/A")
        print("\nChunks per file        : 0 files processed")

    print("=" * 60)
    print(f"Saved chunks to: {output_path}\n")


if __name__ == "__main__":
    main()
