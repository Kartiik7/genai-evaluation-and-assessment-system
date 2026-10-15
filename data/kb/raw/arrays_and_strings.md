Source: https://en.wikipedia.org/wiki/Array_(data_structure) | License: CC BY-SA 4.0

# Arrays, Memory Layout, and String Representations

## Contiguous Storage and Memory Semantics of Arrays
An array is a linear data structure storing fixed-capacity collections of elements of homogeneous data types within contiguous memory locations. Because memory addresses are contiguous, arrays provide instantaneous constant-time random access, where the address of the i-th element is calculated via simple base address offset arithmetic. However, primitive arrays suffer from fixed sizing defined at compile time (or dynamic allocation overhead), making arbitrary insertions and deletions expensive because existing elements must be shifted in memory to preserve order.

## Static Arrays vs Dynamic and Automatic Storage
In C and C++, the static storage duration specifier alters the lifecycle and initialization behavior of variables and arrays. While standard automatic local arrays allocated within a function scope are constructed upon function entry and discarded upon return, static arrays persist in the program's static data segment across all function invocations throughout program lifetime. Static arrays are initialized exactly once upon initial program startup, maintaining their preserved values between subsequent function calls, whereas non-static local arrays are reallocated on the stack frame during every call.

## C-Style Character Arrays vs Standard Library Strings
Textual data in systems programming is traditionally manipulated either as primitive null-terminated character arrays or through high-level string objects such as std::string. C-style strings are raw character buffers whose boundaries are demarcated solely by an explicit null terminator byte ('\0'). Failing to reserve space for or accidentally overwriting this trailing null character leads to out-of-bounds reads and undefined memory behavior. In contrast, std::string wraps internal character buffers with dynamic heap reallocation, automated capacity tracking, bounds-checked indexing, and comprehensive utility methods that fundamentally prevent common buffer overflow vulnerabilities.
