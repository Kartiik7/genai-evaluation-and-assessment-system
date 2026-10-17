Source: https://en.wikipedia.org/wiki/Linked_list | License: CC BY-SA 4.0

# Linked Lists and Dynamic Node Chains

## Node Structures and Dynamic Memory Allocation
A linked list is a linear collection of discrete data elements termed nodes, where linear ordering is maintained not by physical contiguity in memory, but by explicit pointer references stored inside each node. In a singly linked list, every node encapsulates a payload data field alongside a pointer referencing the succeeding node in the chain, culminating in a null pointer at the list tail. Unlike static arrays whose dimensions are fixed upon initialization, linked lists are dynamically allocated node by node at runtime, allowing seamless expansion and contraction without preallocating contiguous memory pools.

## Trade-offs: Linked Lists versus Contiguous Arrays
The architectural differences between contiguous arrays and pointer-linked nodes introduce distinct computational trade-offs. The primary advantage of arrays lies in direct constant-time O(1) indexed access to any arbitrary element, whereas linked lists require sequential traversal from the head node, incurring linear O(N) access time to reach arbitrary indices. Conversely, linked lists excel at insertion and deletion operations: once an insertion position is identified, a node can be spliced in or removed in constant O(1) time simply by updating adjacent pointer references, completely bypassing the expensive element shifts necessitated by array buffers.

## Doubly Linked Lists and Bidirectional Traversal
A doubly linked list extends the singly linked list topology by providing two distinct pointers within every node structure: a next pointer referencing the succeeding node and a previous pointer referencing the preceding node. While storing two pointer variables per node increases overall memory overhead, bidirectional linking substantially simplifies operations such as backward traversals and node deletions. In particular, deleting an arbitrary target node from a doubly linked list does not require searching from the list head to establish a trailing pointer, because the preceding node is immediately accessible via the node's previous link.

## Passing Linked Lists as Function Arguments
In C++, dynamic linked lists are typically passed into functions by supplying a pointer referencing the head node of the chain. Passing the head pointer by value allows the invoked function to traverse and read node values or mutate payload fields throughout the list without affecting the caller's head variable. Alternatively, passing the head pointer by reference (or utilizing a double pointer) enables the routine to modify the original head reference itself, which is required when executing prepending operations, deleting the initial node, or modifying the root topology of the list structure.
