"""
src/rag.py

Context formatting and retrieval utilities for Retrieval-Augmented Generation (RAG) grading.
Enforces limits:
  - top-k = 3 chunks maximum
  - max words per chunk = ~150 words
  - retrieval uses query = question text only
"""

from typing import List, Tuple
try:
    from config import RAG_TOP_K, RAG_MAX_CHUNK_WORDS
except ImportError:
    RAG_TOP_K = 3
    RAG_MAX_CHUNK_WORDS = 150

from src.retrieval import retrieve


def truncate_chunk(text: str, max_words: int = RAG_MAX_CHUNK_WORDS) -> str:
    """Truncates chunk text to max_words."""
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]) + "..."


def build_rag_context(
    chunk_texts: List[str],
    max_chunks: int = RAG_TOP_K,
    max_words_per_chunk: int = RAG_MAX_CHUNK_WORDS,
) -> str:
    """
    Builds a concatenated context string from the top-k chunks,
    with each chunk limited to max_words_per_chunk.
    """
    selected = chunk_texts[:max_chunks]
    truncated_chunks = [truncate_chunk(t, max_words=max_words_per_chunk) for t in selected]
    return "\n\n---\n\n".join(truncated_chunks)


def retrieve_rag_context(
    question: str,
    k: int = RAG_TOP_K,
    max_words_per_chunk: int = RAG_MAX_CHUNK_WORDS,
) -> Tuple[List[str], str]:
    """
    Retrieves top-k chunks for question using ONLY the question text.
    Returns (chunk_ids, formatted_context_string).
    """
    chunk_ids, texts, _ = retrieve(query=question, k=k)
    context = build_rag_context(texts, max_chunks=k, max_words_per_chunk=max_words_per_chunk)
    return list(chunk_ids), context
