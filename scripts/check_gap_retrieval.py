import sys
sys.path.insert(0, ".")
import pandas as pd
from src.retrieval import retrieve

gap_qids = ['1.6', '2.5', '3.4', '3.7', '4.1', '7.4', '8.7']
labels_df = pd.read_csv('results/retrieval_labeling.csv')
qid_to_text = labels_df.groupby('question_id')['question_text'].first().to_dict()

for qid in gap_qids:
    q_text = qid_to_text[float(qid) if '.' in qid else qid]
    chunk_ids, texts, scores = retrieve(q_text, k=10)
    print(f"\n=== QID {qid}: {q_text} ===")
    for rank, (cid, score) in enumerate(zip(chunk_ids, scores), 1):
        print(f"  Rank {rank}: {cid} (score: {score:.4f})")
