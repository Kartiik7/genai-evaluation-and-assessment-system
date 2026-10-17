import pandas as pd

chunks = pd.read_csv('data/kb/chunks.csv')
for idx in range(2):
    row = chunks.iloc[idx]
    print(f"=== {row['chunk_id']} ({row['source_file']}) ===")
    print(row['text'])
    print("\n" + "="*50 + "\n")



