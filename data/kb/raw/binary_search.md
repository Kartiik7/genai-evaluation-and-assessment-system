Source: https://en.wikipedia.org/wiki/Binary_search_algorithm | License: CC BY-SA 4.0

# Binary Search, Divide-and-Conquer, and Logarithmic Efficiency

## The Divide-and-Conquer Paradigm in Searching
Searching is a fundamental operation across computer algorithms. While linear search sequentially scans items with worst-case linear time O(N), sorted collections permit the application of binary search via the divide-and-conquer strategy. At each step, binary search compares the target value against the median element of the active search range. If the target matches the median, the index is immediately returned. If the target is strictly smaller, the algorithm discards the entire upper half of the range and recursively repeats the search on the lower half; if larger, it discards the lower half and searches the upper segment.

## Complexity Analysis and Preconditions
The prerequisite for binary search is that elements must be sorted according to a total ordering relation and stored within a random-access data structure like an array. Because each comparison halves the remaining search space, the maximum number of comparison steps required to locate an element or determine its absence is bounded by log2(N). Consequently, binary search exhibits logarithmic O(log N) worst-case time complexity, offering exponential computational acceleration over linear scans when querying large indexed databases.
