Source: https://en.wikipedia.org/wiki/Binary_tree | License: CC BY-SA 4.0

# Trees, Hierarchical Data Structures, and Traversals

## Tree Terminology and Binary Tree Characteristics
A tree is a non-linear, hierarchical data structure composed of nodes interconnected by directed edges, originating at a unique root node with no parent. A binary tree restricts each constituent node to having at most two child nodes, conventionally distinguished as the left child and the right child. Key geometric metrics of trees include the depth of an individual node (the exact count of edges from the root to that node) and the height of a tree (the length of the longest path from the root down to any terminal leaf node). A leaf node is defined as any node devoid of descendant children.

## Systematic Tree Traversals: Inorder, Preorder, and Postorder
Traversing a binary tree involves systematically visiting every node in the hierarchy exactly once according to well-defined recursive protocols. In preorder traversal, the algorithm processes the current node first, followed recursively by the left subtree and then the right subtree. In inorder traversal, the left subtree is processed first, followed by the current node, and finally the right subtree (a sequence that yields monotonically sorted keys in binary search trees). In postorder traversal, both the left and right subtrees are visited prior to evaluating the current node, making postorder ideal for bottom-up cleanup operations such as tree destruction and memory deallocation.
