import pandas as pd

df = pd.read_csv('results/retrieval_labeling.csv')
for qid in df['question_id'].unique()[:10]:
    sub = df[df['question_id'] == qid]
    print("="*70)
    print(f"QUESTION ID: {qid}")
    print(f"QUESTION: {sub.iloc[0]['question_text']}")
    print("="*70)
    for idx, row in sub.iterrows():
        print(f"[{row['chunk_id']}]")
        print(row['chunk_text'][:250] + "...")
        print()
