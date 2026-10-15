Source: https://en.wikipedia.org/wiki/Queue_(abstract_data_type) | License: CC BY-SA 4.0

# Queues: First-In First-Out Principles and Circular Buffers

## The FIFO Principle and Core Operations
A queue is an abstract data structure governed by the First-In, First-Out (FIFO) ordering policy, where the earliest entity inserted into the structure is the earliest to be extracted. In contrast to stacks, queue operations occur at opposite physical terminals: insertion occurs at the rear (tail) of the queue via the enqueue operation, while removal occurs from the front (head) via the dequeue operation. This queuing model accurately mirrors real-world waiting lines and forms the foundational basis for asynchronous task scheduling, message buffering, and breadth-first search traversals in graph algorithms.

## Circular Array Implementation of Queues
Implementing a queue within a linear array introduces an apparent capacity exhaustion problem: as items are repeatedly enqueued at the rear and dequeued from the front, both boundary pointers drift monotonically toward the end of the array, leaving unreachable vacant slots at the beginning. To circumvent this, queues are implemented using circular arrays, where the index arithmetic utilizes modulo operations (e.g., index = (index + 1) % capacity). When the rear or front pointer reaches the physical boundary of the underlying buffer, it seamlessly wraps around to index zero, ensuring full buffer utilization in continuous constant-time O(1) operations.
