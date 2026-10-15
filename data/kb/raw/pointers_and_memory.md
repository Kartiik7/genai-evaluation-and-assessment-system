Source: https://en.wikipedia.org/wiki/Pointer_(computer_programming) | License: CC BY-SA 4.0

# Pointers, Memory Addresses, and Function Pointers

## Pointer Fundamentals and Address Operators
A pointer in C and C++ is a specialized variable that stores the physical memory address of another variable or object rather than storing a direct data value. Pointers are closely coupled with two fundamental operators: the address-of operator (&), which yields the hexadecimal memory location occupied by its operand, and the dereference operator (*), which accesses or modifies the value stored at the referenced memory address. Additionally, the compile-time sizeof operator evaluates the memory footprint (in bytes) required by an object, variable, or data type within the host platform architecture.

## Pointer Qualification and Passing Conventions
When passing pointers into functions as arguments, C++ supports fine-grained const qualifiers to establish immutable contracts over both the pointer object and the target data. There are four primary parameter configurations: non-constant pointer to non-constant data, non-constant pointer to constant data (which permits altering which variable is pointed to while disallowing modifications to the underlying value), constant pointer to non-constant data (which freezes pointer address reassignment while permitting in-place mutation of the referenced content), and constant pointer to constant data (which strictly forbids mutating either the pointer address or the referenced data).

## Function Pointers and Dynamic Call Dispatch
In addition to referencing scalar variables and composite objects, pointers can also store the starting execution address of executable code within memory. A function pointer holds the entry address of a compiled function possessing a specific signature and return type. Function pointers enable dynamic dispatch, event-driven callbacks, and flexible routing table implementations in low-level systems programming, allowing client code to pass callable routines as first-class function arguments across decoupled modular layers. Uninitialized or dangling pointers represent severe security risks, making explicit initialization to nullptr a critical defensive programming practice.
