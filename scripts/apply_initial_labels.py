import pandas as pd

# Load existing labeling CSV
df = pd.read_csv('results/retrieval_labeling.csv')

# Ground-truth relevance mapping for (question_id, chunk_id)
# 1 if chunk contains information needed to answer the question correctly, else 0
relevance_pairs = {
    # Q1.1: Role of prototype program
    ('1.1', 'cxx_basics_and_problem_solving_000'): 1,
    # Q1.7: while vs do-while
    ('1.7', 'cxx_basics_and_problem_solving_001'): 1,
    # Q2.2: data member vs local variable
    ('2.2', 'classes_and_oop_001'): 1,
    # Q2.4: default constructor creation
    ('2.4', 'classes_and_oop_000'): 1,
    # Q2.7: role of header file
    ('2.7', 'classes_and_oop_001'): 1,
    # Q4.4: static array vs non-static array
    ('4.4', 'arrays_and_strings_000'): 1,
    # Q6.3: star (*) operator
    ('6.3', 'pointers_and_memory_000'): 1,
    # Q6.5: sizeof operator
    ('6.5', 'pointers_and_memory_000'): 1,
    # Q6.6: ways to pass pointer to function
    ('6.6', 'pointers_and_memory_000'): 1,
    ('6.6', 'pointers_and_memory_001'): 1,
    # Q7.1: what is a linked list
    ('7.1', 'linked_lists_000'): 1,
    # Q7.3: arrays vs linked lists advantage
    ('7.3', 'linked_lists_000'): 1,
    ('7.3', 'arrays_and_strings_000'): 1,
    # Q8.2: two main stack functions
    ('8.2', 'stacks_000'): 1,
    # Q8.4: implement stack with list
    ('8.4', 'stacks_000'): 1,
    # Q9.1: what is a queue
    ('9.1', 'queues_000'): 1,
    # Q9.2: two main queue functions
    ('9.2', 'queues_000'): 1,
    # Q9.3: queue with circular array
    ('9.3', 'queues_000'): 1,
    # Q10.2: height of a tree
    ('10.2', 'trees_and_traversals_000'): 1,
    # Q10.7: comparisons in binary search tree
    ('10.7', 'binary_search_trees_000'): 1,
    ('10.7', 'binary_search_000'): 1,
    # Q11.2: access specifiers in class
    ('11.2', 'classes_and_oop_000'): 1,
    # Q11.3: object initialization
    ('11.3', 'classes_and_oop_000'): 1,
    # Q11.5: recursive function
    ('11.5', 'functions_and_recursion_001'): 1,
    # Q11.7: static array vs non-static
    ('11.7', 'arrays_and_strings_000'): 1,
    # Q12.9: delete node in BST
    ('12.9', 'binary_search_trees_001'): 1,
}

df['relevant'] = df.apply(
    lambda r: relevance_pairs.get((str(r['question_id']), str(r['chunk_id'])), 0),
    axis=1
)

df.to_csv('results/retrieval_labeling.csv', index=False)
print("Labeled rows:", len(df))
print("Relevance distribution:")
print(df['relevant'].value_counts())
print("\nQuestions with at least one relevant chunk:")
by_q = df.groupby('question_id')['relevant'].sum()
print("Count with > 0 relevant chunks:", (by_q > 0).sum(), "/ 30")
print("Questions with 0 relevant chunks in pool:", by_q[by_q == 0].index.tolist())
