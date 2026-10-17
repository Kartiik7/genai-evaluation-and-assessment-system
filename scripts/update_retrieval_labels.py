import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.retrieval import retrieve

# Load sampled 30 test questions
test_df = pd.read_csv("data/processed/test.csv")
unique_test_qs = (
    test_df[["question_id", "question"]]
    .drop_duplicates(subset=["question_id"])
    .sample(n=30, random_state=42)
    .sort_values(by="question_id")
    .reset_index(drop=True)
)

print(f"Sampled {len(unique_test_qs)} unique test questions.")

# Comprehensive relevance map: (str(question_id), str(chunk_id)) -> 1
# A chunk is 1 if it contains information needed to answer the question correctly.
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
for i, row in unique_test_qs.iterrows():
    qid = str(row["question_id"])
    q_text = row["question"]
    chunk_ids, texts, scores = retrieve(q_text, k=10)

    # Pair chunks and shuffle order to blind rank and score
    chunk_pairs = list(zip(chunk_ids, texts))
    rng = np.random.RandomState(seed=42 + i)
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
print(f"Saved {len(df_out)} labeled rows to results/retrieval_labeling.csv")
print("\nRelevance counts:")
print(df_out["relevant"].value_counts())

by_q = df_out.groupby("question_id")["relevant"].sum()
print(f"\nQuestions with >= 1 relevant chunk: {(by_q > 0).sum()} / {len(unique_test_qs)}")
print("Questions with 0 relevant chunks:", by_q[by_q == 0].index.tolist())
