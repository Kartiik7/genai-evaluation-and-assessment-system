import pandas as pd

df = pd.read_csv('results/retrieval_labeling.csv')
print(f"Total rows: {len(df)}")
unique_qs = df['question_id'].unique()
print(f"Unique questions count: {len(unique_qs)}")

for i, qid in enumerate(unique_qs[:6], 1):
    g = df[df['question_id'] == qid]
    q_text = g.iloc[0]['question_text']
    print(f"\n[{i}/6] QID: {qid} -> {q_text}")
    for _, row in g.iterrows():
        print(f"   - {row['chunk_id']}")

