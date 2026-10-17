import pandas as pd

df = pd.read_csv('results/retrieval_labeling.csv')

# Let's inspect each row: (question_id, chunk_id)
# We will evaluate each chunk for relevance.
# A chunk is 1 if it contains information needed to answer the question correctly.

for qid in df['question_id'].unique():
    q_df = df[df['question_id'] == qid]
    q_text = q_df.iloc[0]['question_text']
    print(f"=== QID {qid}: {q_text} ===")
    for _, row in q_df.iterrows():
        cid = row['chunk_id']
        txt = row['chunk_text']
        print(f"  [{cid}]")
