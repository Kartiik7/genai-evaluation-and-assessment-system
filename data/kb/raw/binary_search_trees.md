Source: https://en.wikipedia.org/wiki/Binary_search_tree | License: CC BY-SA 4.0

# Binary Search Trees: Invariants, Lookup, and Node Deletion

## The Binary Search Tree Invariant
A binary search tree (BST) is an ordered node-based binary tree data structure where each node stores a distinct key value. The fundamental invariant governing a binary search tree mandates that for any given node, all keys located in its left subtree must be strictly less than the node's key, while all keys stored in its right subtree must be strictly greater than (or equal to) the node's key. This structural ordering allows binary search trees to perform lookup, insertion, and deletion operations in logarithmic O(log N) average time when balanced, though degenerate unbalanced trees degrade to linear O(N) chains.

## Deletion Scenarios in Binary Search Trees
Deleting a designated target node from a binary search tree requires handling three structural topological cases:
1. The target node is a leaf possessing no children: deletion simply detaches the node from its parent pointer.
2. The target node has exactly one child: the node is spliced out by repointing the parent directly to the single surviving child.
3. The target node possesses two children: the node cannot be abruptly detached without breaking the tree structure. Instead, the node's key is replaced with the key of either its in-order successor (the leftmost node in its right subtree, possessing the smallest key greater than the target) or its in-order predecessor (the rightmost node in its left subtree). Subsequently, that successor or predecessor node—which by definition has at most one child—is deleted using the simpler case 1 or case 2 logic.
