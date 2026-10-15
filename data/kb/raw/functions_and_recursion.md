Source: https://en.wikipedia.org/wiki/Subroutine | License: CC BY-SA 4.0

# Functions, Signatures, Overloading, and Recursion

## Function Signatures and Overloading Resolution
In compiled languages like C++, a function signature consists of the function identifier and the ordered sequence of parameter types (including qualifiers such as const or reference). The return type of a function does not constitute part of the signature used for overload resolution. Function overloading permits multiple functions in the same lexical scope to share identical names provided their parameter lists diverge in arity, argument types, or order. At compile time, the compiler performs overload resolution to uniquely bind each call site to the most suitable candidate function based on exact or promoting type conversions.

## Inline Functions and Performance Considerations
Function call overhead includes passing parameters into registers or onto the call stack, preserving return addresses, branching execution, and restoring stack frames upon return. By prefixing a function declaration with the inline specifier, a programmer instructs the compiler that the function body should ideally be expanded directly at the call site. Inline expansion eliminates function call branching and enables downstream optimization passes like constant folding, though indiscriminate inlining of large functions can trigger instruction cache misses through binary code bloat.

## Recursive Functions and Infinite Recursion Pitfalls
Recursion occurs when a function solves a problem by repeatedly invoking itself on smaller sub-problems. Every valid recursive routine requires two structural components: one or more base cases that compute results directly without further recursion, and a recursive step that steadily reduces the input domain toward those base cases. Infinite recursion, which leads directly to call stack exhaustion and stack overflow segmentation faults, typically arises from omitting the terminating base case entirely, or crafting recursive transitions that fail to progress toward the base condition under edge cases. Establishing formal inductive proofs and monitoring recursion depth safeguard against infinite recursion errors.
