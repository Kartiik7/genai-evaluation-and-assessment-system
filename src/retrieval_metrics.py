"""
src/retrieval_metrics.py

Retrieval evaluation metrics module for the GenAI Evaluation & Assessment System.

Computes standard Information Retrieval (IR) metrics from human relevance annotations:
  - Precision@1, Precision@3, Precision@5
  - Recall@3, Recall@5
  - Mean Reciprocal Rank (MRR)

IMPORTANT NOTE ON RECALL:
In this pooled evaluation workflow, recall is measured strictly against the labeled pool
(the top-10 candidate chunks retrieved for each question), not against the entire
knowledge base. Because human annotators evaluate only the pooled candidates rather than
exhaustively labeling all documents in the corpus, the ground-truth relevant set for each
question is defined as all chunks within the top-10 pool judged relevant (relevant == 1).

RANDOM BASELINE:
Computes expected performance when retrieving k chunks uniformly at random without
replacement from the full knowledge base corpus across 1000 repetitions (random_state=42).
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import RETRIEVAL_MODEL_NAME, RETRIEVAL_QUERY_MODE
from src.retrieval import retrieve, format_query


def precision_at_k(ranked_labels: List[int], k: int) -> float:
    """
    Computes Precision@k: fraction of top-k retrieved items that are relevant.
    """
    if k <= 0:
        return 0.0
    cutoff_labels = ranked_labels[:k]
    if not cutoff_labels:
        return 0.0
    return float(sum(cutoff_labels) / k)


def recall_at_k(ranked_labels: List[int], total_relevant_in_pool: int, k: int) -> float:
    """
    Computes Recall@k: fraction of relevant items in the labeled pool that appear in top-k.
    If total_relevant_in_pool is 0, returns 0.0.
    """
    if total_relevant_in_pool <= 0 or k <= 0:
        return 0.0
    cutoff_labels = ranked_labels[:k]
    return float(sum(cutoff_labels) / total_relevant_in_pool)


def reciprocal_rank(ranked_labels: List[int]) -> float:
    """
    Computes Reciprocal Rank (RR): 1 / rank of the first relevant item (1-indexed).
    Returns 0.0 if no relevant items are found.
    """
    for rank_idx, label in enumerate(ranked_labels, start=1):
        if label == 1:
            return 1.0 / rank_idx
    return 0.0


def get_real_retrieval_ranking(
    labeled_df: pd.DataFrame,
    k: int = 10,
    index_path: Union[str, Path] = "data/kb/index.faiss",
    chunks_path: Union[str, Path] = "data/kb/chunks.csv",
    model_name: str = RETRIEVAL_MODEL_NAME,
    query_mode: str = RETRIEVAL_QUERY_MODE,
) -> Dict[Any, List[str]]:
    """
    Computes the real retrieval ranking (in descending order of cosine similarity score)
    for each unique question in labeled_df using FAISS index.
    Returns:
        dict mapping question_id -> list of top-k chunk_ids in rank order (rank 1 to k).
    """
    q_col = "question_text" if "question_text" in labeled_df.columns else "question"
    if q_col not in labeled_df.columns:
        # Fallback to test.csv if question text is not present in labeled_df
        test_path = Path("data/processed/test.csv")
        if test_path.exists():
            test_df = pd.read_csv(test_path)
            q_map = dict(zip(test_df["question_id"], test_df["question"]))
        else:
            raise ValueError("Question text column missing from labeled_df and test.csv not found.")
    else:
        q_map = dict(zip(labeled_df["question_id"], labeled_df[q_col]))

    rankings = {}
    for qid, q_text in q_map.items():
        query = format_query(question=str(q_text), mode=query_mode)
        top_chunk_ids, _, _ = retrieve(
            query=query,
            k=k,
            index_path=index_path,
            chunks_path=chunks_path,
            model_name=model_name,
        )
        rankings[qid] = top_chunk_ids

    return rankings


def compute_retrieval_metrics(
    labeled_df: pd.DataFrame,
    real_ranking: Optional[Dict[Any, List[str]]] = None,
    index_path: Union[str, Path] = "data/kb/index.faiss",
    chunks_path: Union[str, Path] = "data/kb/chunks.csv",
    model_name: str = RETRIEVAL_MODEL_NAME,
    query_mode: str = RETRIEVAL_QUERY_MODE,
) -> Dict[str, float]:
    """
    Computes Precision@1, @3, @5, Recall@3, @5, and MRR, averaged over labeled questions.
    Recall is measured against the labeled pool (top-10).
    """
    if "relevant" not in labeled_df.columns:
        raise ValueError("Column 'relevant' missing from labeled_df.")

    # Check that labels are not completely empty
    non_empty = labeled_df["relevant"].dropna()
    valid_labels = non_empty[non_empty.astype(str).str.strip() != ""]
    if len(valid_labels) == 0:
        raise ValueError(
            "Column 'relevant' contains no annotations yet. "
            "Please label the chunks (1 or 0) before computing metrics."
        )

    # Clean labeled_df
    df = labeled_df.copy()
    df["relevant"] = pd.to_numeric(df["relevant"], errors="coerce").fillna(0).astype(int)

    # Obtain real retrieval ranking if not supplied
    if real_ranking is None:
        real_ranking = get_real_retrieval_ranking(
            labeled_df=df,
            k=10,
            index_path=index_path,
            chunks_path=chunks_path,
            model_name=model_name,
            query_mode=query_mode,
        )

    p1_list, p3_list, p5_list = [], [], []
    r3_list, r5_list = [], []
    mrr_list = []

    unique_qids = df["question_id"].unique()

    for qid in unique_qids:
        q_subset = df[df["question_id"] == qid]
        label_dict = dict(zip(q_subset["chunk_id"], q_subset["relevant"]))
        total_relevant = sum(label_dict.values())

        ranked_cids = real_ranking.get(qid, [])
        ranked_labels = [label_dict.get(cid, 0) for cid in ranked_cids]

        p1_list.append(precision_at_k(ranked_labels, 1))
        p3_list.append(precision_at_k(ranked_labels, 3))
        p5_list.append(precision_at_k(ranked_labels, 5))
        r3_list.append(recall_at_k(ranked_labels, total_relevant, 3))
        r5_list.append(recall_at_k(ranked_labels, total_relevant, 5))
        mrr_list.append(reciprocal_rank(ranked_labels))

    return {
        "Precision@1": float(np.mean(p1_list)),
        "Precision@3": float(np.mean(p3_list)),
        "Precision@5": float(np.mean(p5_list)),
        "Recall@3": float(np.mean(r3_list)),
        "Recall@5": float(np.mean(r5_list)),
        "MRR": float(np.mean(mrr_list)),
    }


def compute_random_baseline(
    labeled_df: pd.DataFrame,
    n_repetitions: int = 1000,
    seed: int = 42,
    chunks_path: Union[str, Path] = "data/kb/chunks.csv",
    kb_chunk_ids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Computes random baseline retrieval performance over n_repetitions (seed 42).
    For each question, k random chunks are retrieved without replacement from the knowledge base.
    """
    if "relevant" not in labeled_df.columns:
        raise ValueError("Column 'relevant' missing from labeled_df.")

    df = labeled_df.copy()
    df["relevant"] = pd.to_numeric(df["relevant"], errors="coerce").fillna(0).astype(int)

    # Determine corpus universe
    if kb_chunk_ids is None:
        chunks_path = Path(chunks_path)
        if chunks_path.exists():
            kb_df = pd.read_csv(chunks_path)
            kb_chunk_ids = kb_df["chunk_id"].dropna().unique().tolist()
        else:
            # Fallback to chunks present in labeled_df
            kb_chunk_ids = df["chunk_id"].unique().tolist()

    all_chunks = np.array(kb_chunk_ids)
    num_kb_chunks = len(all_chunks)

    rng = np.random.RandomState(seed)
    unique_qids = df["question_id"].unique()

    # Pre-parse question relevant sets
    q_data = {}
    for qid in unique_qids:
        q_subset = df[df["question_id"] == qid]
        rel_set = set(q_subset[q_subset["relevant"] == 1]["chunk_id"])
        q_data[qid] = {
            "relevant_set": rel_set,
            "total_relevant": len(rel_set),
        }

    rep_p1, rep_p3, rep_p5 = [], [], []
    rep_r3, rep_r5 = [], []
    rep_mrr = []

    k_retrieve = min(10, num_kb_chunks)

    for _ in range(n_repetitions):
        q_p1, q_p3, q_p5 = [], [], []
        q_r3, q_r5 = [], []
        q_rr = []

        for qid in unique_qids:
            rel_set = q_data[qid]["relevant_set"]
            total_rel = q_data[qid]["total_relevant"]

            # Sample k distinct chunks randomly from the KB
            sampled_indices = rng.choice(num_kb_chunks, size=k_retrieve, replace=False)
            sampled_chunks = all_chunks[sampled_indices]
            random_labels = [1 if cid in rel_set else 0 for cid in sampled_chunks]

            q_p1.append(precision_at_k(random_labels, 1))
            q_p3.append(precision_at_k(random_labels, 3))
            q_p5.append(precision_at_k(random_labels, 5))
            q_r3.append(recall_at_k(random_labels, total_rel, 3))
            q_r5.append(recall_at_k(random_labels, total_rel, 5))
            q_rr.append(reciprocal_rank(random_labels))

        rep_p1.append(np.mean(q_p1))
        rep_p3.append(np.mean(q_r3))
        rep_p5.append(np.mean(q_p5))
        rep_r3.append(np.mean(q_r3))
        rep_r5.append(np.mean(q_r5))
        rep_mrr.append(np.mean(q_rr))

    return {
        "Precision@1": float(np.mean(rep_p1)),
        "Precision@3": float(np.mean(rep_p3)),
        "Precision@5": float(np.mean(rep_p5)),
        "Recall@3": float(np.mean(rep_r3)),
        "Recall@5": float(np.mean(rep_r5)),
        "MRR": float(np.mean(rep_mrr)),
    }


def compare_retrieval_to_baseline(
    labeled_df: pd.DataFrame,
    real_ranking: Optional[Dict[Any, List[str]]] = None,
    n_repetitions: int = 1000,
    seed: int = 42,
    index_path: Union[str, Path] = "data/kb/index.faiss",
    chunks_path: Union[str, Path] = "data/kb/chunks.csv",
    model_name: str = RETRIEVAL_MODEL_NAME,
) -> pd.DataFrame:
    """
    Computes and compares metrics for both the actual retriever and random baseline.
    Returns a formatted pandas DataFrame.
    """
    model_metrics = compute_retrieval_metrics(
        labeled_df=labeled_df,
        real_ranking=real_ranking,
        index_path=index_path,
        chunks_path=chunks_path,
        model_name=model_name,
    )
    random_metrics = compute_random_baseline(
        labeled_df=labeled_df,
        n_repetitions=n_repetitions,
        seed=seed,
        chunks_path=chunks_path,
    )

    rows = [
        {"Method": f"Retriever ({model_name})", **model_metrics},
        {"Method": "Random Baseline (1000 runs)", **random_metrics},
    ]
    comparison_df = pd.DataFrame(rows)
    return comparison_df


def find_retrieval_failures(
    labeled_df: pd.DataFrame,
    real_ranking: Optional[Dict[Any, List[str]]] = None,
    k: int = 3,
    index_path: Union[str, Path] = "data/kb/index.faiss",
    chunks_path: Union[str, Path] = "data/kb/chunks.csv",
    model_name: str = RETRIEVAL_MODEL_NAME,
) -> pd.DataFrame:
    """
    Identifies questions where no relevant chunk was retrieved in the top k (default: 3).
    Returns a DataFrame of failure cases with question details and retrieved candidates.
    """
    df = labeled_df.copy()
    df["relevant"] = pd.to_numeric(df["relevant"], errors="coerce").fillna(0).astype(int)

    if real_ranking is None:
        real_ranking = get_real_retrieval_ranking(
            labeled_df=df,
            k=10,
            index_path=index_path,
            chunks_path=chunks_path,
            model_name=model_name,
        )

    q_col = "question_text" if "question_text" in df.columns else "question"
    q_map = dict(zip(df["question_id"], df[q_col])) if q_col in df.columns else {}

    failures = []
    for qid in df["question_id"].unique():
        q_subset = df[df["question_id"] == qid]
        label_dict = dict(zip(q_subset["chunk_id"], q_subset["relevant"]))
        total_relevant = sum(label_dict.values())

        ranked_cids = real_ranking.get(qid, [])
        top_k_cids = ranked_cids[:k]
        top_k_labels = [label_dict.get(cid, 0) for cid in top_k_cids]

        # A retrieval failure occurs if zero relevant chunks were retrieved in top-k
        if sum(top_k_labels) == 0:
            failures.append({
                "question_id": qid,
                "question_text": q_map.get(qid, ""),
                "total_relevant_in_pool": total_relevant,
                "top_k_retrieved": top_k_cids,
                "top_k_labels": top_k_labels,
            })

    return pd.DataFrame(failures)
