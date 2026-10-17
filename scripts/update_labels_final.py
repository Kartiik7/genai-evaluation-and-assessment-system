import os
import sys
import random
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.retrieval import retrieve

# The exact 30 sampled test questions
ORIGINAL_QIDS = [
    '1.1', '1.6', '1.7', '2.2', '2.4', '2.5', '2.7', '3.4', '3.7', '4.1',
    '4.4', '6.3', '6.5', '6.6', '7.1', '7.3', '7.4', '8.2', '8.4', '8.7',
    '9.1', '9.2', '9.3', '10.2', '10.7', '11.2', '11.3', '11.5', '11.7', '12.9'
]

test_df = pd.read_csv("data/processed/test.csv")
# Map question_id to question text
q_map = {}
for _, r in test_df.iterrows():
    qid_str = str(r["question_id"])
    if qid_str in ORIGINAL_QIDS and qid_str not in q_map:
        q_map[qid_str] = r["question"]

print(f"Mapped {len(q_map)} questions.")

RELEVANCE_MAP = {
    # 1.1: Role of prototype program -> cxx_basics_and_problem_solving_000
    ('1.1', 'cxx_basics_and_problem_solving_000'): 1,

    # 1.6: Where variables are declared -> cxx_basics_and_problem_solving_001 (scope/location: stack/local, global, before use),
    # and classes_and_oop_001 (data members in class vs local variables in member functions)
    ('1.6', 'cxx_basics_and_problem_solving_001'): 1,
    ('1.6', 'classes_and_oop_001'): 1,

    # 1.7: while vs do-while difference -> cxx_basics_and_problem_solving_001
    ('1.7', 'cxx_basics_and_problem_solving_001'): 1,

    # 2.2: data member vs local variable -> classes_and_oop_001
    ('2.2', 'classes_and_oop_001'): 1,

    # 2.4: default constructor creation -> classes_and_oop_000
    ('2.4', 'classes_and_oop_000'): 1,

    # 2.5: how many constructors can be created -> classes_and_oop_001 (arbitrary multiplicity / overloading)
    ('2.5', 'classes_and_oop_001'): 1,

    # 2.7: role of header file -> classes_and_oop_001
    ('2.7', 'classes_and_oop_001'): 1,

    # 3.4: advantage of pass-by-reference -> functions_and_recursion_001 (avoiding copying overhead / address alias)
    ('3.4', 'functions_and_recursion_001'): 1,

    # 3.7: similarities between iteration and recursion -> functions_and_recursion_002 (repetition, termination criteria, non-termination)
    ('3.7', 'functions_and_recursion_002'): 1,

    # 4.1: two ways of specifying array length -> arrays_and_strings_001 (explicit brackets dimension vs initializer list)
    ('4.1', 'arrays_and_strings_001'): 1,

    # 4.4: static array vs non-static -> arrays_and_strings_000
    ('4.4', 'arrays_and_strings_000'): 1,

    # 6.3: star (*) operator -> pointers_and_memory_000 (dereference operator)
    ('6.3', 'pointers_and_memory_000'): 1,

    # 6.5: sizeof operator -> pointers_and_memory_000
    ('6.5', 'pointers_and_memory_000'): 1,

    # 6.6: ways to pass pointer to function -> pointers_and_memory_000, pointers_and_memory_001
    ('6.6', 'pointers_and_memory_000'): 1,
    ('6.6', 'pointers_and_memory_001'): 1,

    # 7.1: what is a linked list -> linked_lists_000
    ('7.1', 'linked_lists_000'): 1,

    # 7.3: arrays vs linked lists advantage -> linked_lists_000, arrays_and_strings_000
    ('7.3', 'linked_lists_000'): 1,
    ('7.3', 'arrays_and_strings_000'): 1,

    # 7.4: how linked lists passed as arguments -> linked_lists_001 (pointer/reference to head node)
    ('7.4', 'linked_lists_001'): 1,

    # 8.2: two main stack functions -> stacks_000 (push and pop)
    ('8.2', 'stacks_000'): 1,

    # 8.4: implement stack with list -> stacks_000 (head node as top, O(1) push/pop)
    ('8.4', 'stacks_000'): 1,

    # 8.7: operations to find element on stack -> stacks_001 (iteratively popping to auxiliary stack and restoring)
    ('8.7', 'stacks_001'): 1,

    # 9.1: what is a queue -> queues_000 (FIFO)
    ('9.1', 'queues_000'): 1,

    # 9.2: two main queue functions -> queues_000 (enqueue and dequeue)
    ('9.2', 'queues_000'): 1,

    # 9.3: queue with array -> queues_000 (circular array modulo arithmetic)
    ('9.3', 'queues_000'): 1,

    # 10.2: height of a tree -> trees_and_traversals_000
    ('10.2', 'trees_and_traversals_000'): 1,

    # 10.7: comparisons in binary search tree -> binary_search_trees_000, binary_search_000
    ('10.7', 'binary_search_trees_000'): 1,
    ('10.7', 'binary_search_000'): 1,

    # 11.2: access specifiers in class -> classes_and_oop_000 (public and private)
    ('11.2', 'classes_and_oop_000'): 1,

    # 11.3: object initialization -> classes_and_oop_000 (constructors)
    ('11.3', 'classes_and_oop_000'): 1,

    # 11.5: recursive function -> functions_and_recursion_001 (calls itself, base case + recursive step)
    ('11.5', 'functions_and_recursion_001'): 1,

    # 11.7: static array vs non-static -> arrays_and_strings_000
    ('11.7', 'arrays_and_strings_000'): 1,

    # 12.9: delete node from binary search tree -> binary_search_trees_001 (leaf, single child, two children / successor)
    ('12.9', 'binary_search_trees_001'): 1,
}

rows = []
for i, qid in enumerate(ORIGINAL_QIDS):
    q_text = q_map[qid]
    top_chunk_ids, top_texts, _ = retrieve(query=q_text, k=10)

    # Shuffle to hide rank and score
    chunk_pairs = list(zip(top_chunk_ids, top_texts))
    rng = random.Random(42 + i)
    rng.shuffle(chunk_pairs)

    for cid, ctext in chunk_pairs:
        rel = RELEVANCE_MAP.get((qid, cid), 0)
        rows.append({
            "question_id": qid,
            "question_text": q_text,
            "chunk_id": cid,
            "chunk_text": ctext,
            "relevant": rel,
        })

df_out = pd.DataFrame(rows)
df_out.to_csv("results/retrieval_labeling.csv", index=False)
print(f"Saved {len(df_out)} rows to results/retrieval_labeling.csv")
print("Relevance distribution:")
print(df_out["relevant"].value_counts())

by_q = df_out.groupby("question_id")["relevant"].sum()
print(f"Questions with >= 1 relevant chunk: {(by_q > 0).sum()} / {len(ORIGINAL_QIDS)}")
print("Questions with 0 relevant chunks in pool:", by_q[by_q == 0].index.tolist())
