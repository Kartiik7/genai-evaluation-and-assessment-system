"""
src/kb.py

Core module for knowledge base ingestion and chunking.
Reads raw markdown files with source URLs and licenses,
splits documents into cohesive semantic chunks bounded
between ~120 and 250 words with ~20 words overlap,
and exports structured chunk metadata.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd


def parse_header(first_line: str) -> Tuple[str, str]:
    """
    Extract source URL and license from the first line of a markdown document.

    Supports formats like:
      Source: https://... | License: CC BY-SA 4.0
      URL: https://... | License: MIT
      <!-- Source: https://... | License: CC BY-SA 4.0 -->
      https://... | CC BY-SA 4.0
    """
    clean_line = first_line.strip().strip("<>!-#* ")
    
    # Extract URL
    url_match = re.search(r"https?://[^\s|),>]+", clean_line)
    source_url = url_match.group(0) if url_match else ""

    # Extract license
    license_val = ""
    lic_match = re.search(r"(?:license|licence)\s*[:=]\s*([^|\n\r>]+)", clean_line, re.IGNORECASE)
    if lic_match:
        license_val = lic_match.group(1).strip()
    else:
        # Check pipe-separated fallback
        parts = [p.strip() for p in clean_line.split("|") if p.strip()]
        for part in parts:
            if not part.startswith("http") and any(k in part.lower() for k in ["license", "cc", "mit", "apache", "public"]):
                license_val = part
                break

    return source_url, license_val


def chunk_markdown(
    text: str,
    target_min_words: int = 120,
    target_max_words: int = 250,
    overlap_words: int = 20
) -> List[str]:
    """
    Chunks markdown text by headings and paragraphs into segments of
    approximately 120-250 words with ~20 words overlap.
    """
    # Split text by double newlines into structural blocks (paragraphs / headings)
    raw_blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if not raw_blocks:
        return []

    # Break down blocks so that no individual block exceeds target_max_words
    blocks: List[str] = []
    for block in raw_blocks:
        b_words = block.split()
        if len(b_words) <= target_max_words:
            blocks.append(block)
        else:
            # Split block into smaller sub-blocks by sentences
            sentences = re.split(r"(?<=[.!?])\s+", block)
            sub_words: List[str] = []
            for s in sentences:
                s_words = s.split()
                if len(sub_words) + len(s_words) > target_max_words and len(sub_words) >= target_min_words:
                    blocks.append(" ".join(sub_words))
                    sub_words = s_words
                else:
                    sub_words.extend(s_words)
            if sub_words:
                blocks.append(" ".join(sub_words))

    chunks: List[str] = []
    current_tokens: List[str] = []

    for block in blocks:
        b_tokens = block.split()
        if not b_tokens:
            continue

        # If adding the block exceeds target_max_words and current chunk is near/above target_min_words
        if len(current_tokens) + len(b_tokens) > target_max_words and len(current_tokens) >= (target_min_words - 20):
            chunks.append(" ".join(current_tokens))
            # Start next chunk with trailing overlap words
            overlap = current_tokens[-overlap_words:] if len(current_tokens) >= overlap_words else current_tokens
            current_tokens = list(overlap) + b_tokens
        else:
            current_tokens.extend(b_tokens)
            # If current buffer became larger than target_max_words, emit window
            while len(current_tokens) >= target_max_words:
                chunks.append(" ".join(current_tokens[:target_max_words]))
                overlap = current_tokens[target_max_words - overlap_words:target_max_words]
                current_tokens = list(overlap) + current_tokens[target_max_words:]

    if current_tokens:
        # Merge very small trailing fragment into the previous chunk if reasonable
        if chunks and len(current_tokens) < (target_min_words // 2) and (len(chunks[-1].split()) + len(current_tokens) <= target_max_words + 35):
            chunks[-1] = chunks[-1] + " " + " ".join(current_tokens)
        else:
            chunks.append(" ".join(current_tokens))

    return chunks


def process_markdown_file(
    file_path: Path,
    target_min_words: int = 120,
    target_max_words: int = 250,
    overlap_words: int = 20
) -> List[Dict]:
    """
    Process a single markdown file: parse header, extract metadata, chunk body.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if not lines:
        return []

    # First line contains source URL and license
    first_line = lines[0]
    source_url, license_val = parse_header(first_line)

    # Remainder is the document body
    body_text = "".join(lines[1:]).strip()
    topic = file_path.stem
    source_file = file_path.name

    raw_chunks = chunk_markdown(
        body_text,
        target_min_words=target_min_words,
        target_max_words=target_max_words,
        overlap_words=overlap_words
    )

    records: List[Dict] = []
    for idx, chunk_text in enumerate(raw_chunks):
        word_count = len(chunk_text.split())
        chunk_id = f"{topic}_{idx:03d}"
        records.append({
            "chunk_id": chunk_id,
            "source_file": source_file,
            "source_url": source_url,
            "license": license_val,
            "topic": topic,
            "text": chunk_text,
            "word_count": word_count,
        })

    return records


def build_kb(
    raw_dir: Path,
    output_path: Path,
    target_min_words: int = 120,
    target_max_words: int = 250,
    overlap_words: int = 20
) -> pd.DataFrame:
    """
    Build the knowledge base chunks dataframe from all markdown files in raw_dir
    and write to output_path (CSV).
    """
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    md_files = sorted(list(raw_dir.glob("*.md")))
    all_chunks: List[Dict] = []

    for md_file in md_files:
        chunks = process_markdown_file(
            md_file,
            target_min_words=target_min_words,
            target_max_words=target_max_words,
            overlap_words=overlap_words
        )
        all_chunks.extend(chunks)

    df_chunks = pd.DataFrame(all_chunks, columns=[
        "chunk_id", "source_file", "source_url", "license", "topic", "text", "word_count"
    ])
    df_chunks.to_csv(output_path, index=False)
    return df_chunks
