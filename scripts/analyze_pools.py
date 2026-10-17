import pandas as pd

df = pd.read_csv('results/retrieval_labeling.csv')
chunks = pd.read_csv('data/kb/chunks.csv').set_index('chunk_id')['text'].to_dict()

# Print each question and let's inspect the pool of retrieved chunks
for i, qid in enumerate(df['question_id'].unique(), 1):
    q_df = df[df['question_id'] == qid]
    q_text = q_df.iloc[0]['question_text']
    pool_chunks = q_df['chunk_id'].tolist()
    print(f"\nQ{i} (ID: {qid}): {q_text}")
    print(f"Pool: {pool_chunks}")
