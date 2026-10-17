import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from src.retrieval import retrieve


# Load current labeling CSV
curr_df = pd.read_csv('results/retrieval_labeling.csv')
unique_qids = curr_df['question_id'].unique()

# Map each question to its question_text
qid_to_text = curr_df.groupby('question_id')['question_text'].first().to_dict()

# Retrieve new top 10 for each question
new_top10 = {}
for qid in unique_qids:
    q_text = qid_to_text[qid]
    retrieved = retrieve(q_text, k=10)
    new_top10[qid] = retrieved

# Compare pools
changed_qids = []
for qid in unique_qids:
    old_chunks = set(curr_df[curr_df['question_id'] == qid]['chunk_id'])
    new_chunks = set(new_top10[qid][0])
    if old_chunks != new_chunks:
        changed_qids.append(qid)

print(f"Total questions: {len(unique_qids)}")
print(f"Changed questions: {len(changed_qids)} -> {changed_qids}")
for qid in changed_qids:
    old_c = set(curr_df[curr_df['question_id'] == qid]['chunk_id'])
    new_c = set(new_top10[qid][0])
    print(f"\nQID {qid}: {qid_to_text[qid]}")
    print(f"  Added to top10: {new_c - old_c}")
    print(f"  Dropped from top10: {old_c - new_c}")

