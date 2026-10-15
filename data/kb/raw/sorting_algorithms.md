Source: https://en.wikipedia.org/wiki/Sorting_algorithm | License: CC BY-SA 4.0

# Sorting Algorithms: Insertion Sort and Selection Sort

## Mechanics and Invariants of Insertion Sort
Insertion sort is an intuitive, comparison-based sorting algorithm that operates iteratively by partitioning an array into a sorted subarray on the left and an unsorted subarray on the right. At each iteration step, the algorithm extracts the first element from the unsorted segment and compares it backward against elements in the sorted portion, shifting larger items rightward to create space until the correct insertion position is reached. Insertion sort is adaptive and particularly efficient for small datasets or lists that are already nearly sorted, attaining linear O(N) best-case performance while exhibiting quadratic O(N^2) average and worst-case complexity.

## Mechanics and Behavior of Selection Sort
Selection sort is another elementary in-place comparison sort that divides the input array into sorted and unsorted segments. During each pass over the remaining unsorted subarray, the algorithm scans every unsorted item to locate the minimum element, and then swaps that minimum with the leftmost unsorted element, permanently extending the boundary of the sorted region by one. Unlike insertion sort, selection sort always performs a fixed number of comparisons regardless of the initial order of the input, resulting in an invariant quadratic O(N^2) time complexity across best, average, and worst-case scenarios, though it minimizes total data write operations to at most N swaps.
