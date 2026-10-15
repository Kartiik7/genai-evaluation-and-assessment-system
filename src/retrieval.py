"""
src/retrieval.py

Knowledge Base Retrieval Module using Sentence Transformers and FAISS.
Builds an inner-product (cosine similarity) index over knowledge base chunks
and retrieves relevant context for grading queries.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import faiss

# Ensure project root is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import RETRIEVAL_MODEL_NAME, RETRIEVAL_QUERY_MODE
from src.embeddings import embed_texts

_index_cache = {}
_chunks_cache = {}


def build_index(
    chunks_path: Union[str, Path] = "data/kb/chunks.csv",
    index_path: Union[str, Path] = "data/kb/index.faiss",
    model_name: str = RETRIEVAL_MODEL_NAME,
) -> faiss.IndexFlatIP:
    """
    Embeds all chunk texts from chunks_path (normalized),
    builds a FAISS inner-product index, and saves it to index_path.
    """
    chunks_path = Path(chunks_path)
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found at: {chunks_path}")

    df_chunks = pd.read_csv(chunks_path)
    texts = df_chunks["text"].fillna("").tolist()

    if not texts:
        raise ValueError(f"No chunks found in {chunks_path}")

    # Generate normalized embeddings
    embeddings = embed_texts(model_name, texts)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    # Ensure L2 normalization for inner-product index (IP on normalized = cosine similarity)
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(index_path))

    # Update cache
    _index_cache[str(index_path)] = index
    _chunks_cache[str(chunks_path)] = df_chunks

    return index


def format_query(
    question: str,
    reference: Optional[str] = None,
    mode: str = RETRIEVAL_QUERY_MODE,
) -> str:
    """
    Formats the retrieval query according to mode.
    Modes:
      - 'question': QUESTION TEXT ONLY (default)
      - 'question+reference': Question text concatenated with reference answer
    """
    q_str = str(question).strip() if question is not None else ""
    if mode == "question":
        return q_str
    elif mode == "question+reference":
        ref_str = str(reference).strip() if reference is not None else ""
        if ref_str:
            return f"{q_str}\nReference Answer: {ref_str}"
        return q_str
    else:
        raise ValueError(f"Unsupported RETRIEVAL_QUERY_MODE: '{mode}'")


def retrieve(
    query: str,
    k: int = 3,
    index_path: Union[str, Path] = "data/kb/index.faiss",
    chunks_path: Union[str, Path] = "data/kb/chunks.csv",
    model_name: str = RETRIEVAL_MODEL_NAME,
) -> Tuple[List[str], List[str], List[float]]:
    """
    Retrieves the top-k chunk_ids, texts, and scores for a given query string.
    Returns:
        (chunk_ids, texts, scores)
    """
    index_path = Path(index_path)
    chunks_path = Path(chunks_path)

    # Load chunks dataframe (cached)
    chunks_key = str(chunks_path)
    if chunks_key not in _chunks_cache:
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found at: {chunks_path}")
        _chunks_cache[chunks_key] = pd.read_csv(chunks_path)
    df_chunks = _chunks_cache[chunks_key]

    # Load FAISS index (cached)
    index_key = str(index_path)
    if index_key not in _index_cache:
        if not index_path.exists():
            build_index(chunks_path=chunks_path, index_path=index_path, model_name=model_name)
        else:
            _index_cache[index_key] = faiss.read_index(str(index_path))
    index = _index_cache[index_key]

    # Embed query
    query_str = str(query) if query is not None else ""
    q_emb = embed_texts(model_name, [query_str])
    q_emb = np.ascontiguousarray(q_emb, dtype=np.float32)
    faiss.normalize_L2(q_emb)

    actual_k = min(k, index.ntotal)
    if actual_k <= 0:
        return [], [], []

    scores_matrix, indices_matrix = index.search(q_emb, actual_k)
    top_indices = indices_matrix[0]
    top_scores = [float(s) for s in scores_matrix[0]]

    top_chunk_ids = [str(df_chunks.iloc[idx]["chunk_id"]) for idx in top_indices]
    top_texts = [str(df_chunks.iloc[idx]["text"]) for idx in top_indices]

    return top_chunk_ids, top_texts, top_scores
