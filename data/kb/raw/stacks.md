Source: https://en.wikipedia.org/wiki/Stack_(abstract_data_type) | License: CC BY-SA 4.0

# Stacks: Abstract Data Type and Practical Applications

## The LIFO Principle and Fundamental Operations
A stack is an abstract data structure that adheres strictly to the Last-In, First-Out (LIFO) access discipline. All insertions and removals occur exclusively at a single designated boundary known as the top of the stack. The two primary primitive operations defined by a stack are push, which adds a new element onto the top of the collection, and pop, which removes the most recently inserted item. Auxiliary inspection routines typically include peek (or top), which observes the current top element without removal, and isEmpty, which evaluates whether any elements remain stored on the stack.

## Array and Linked Implementations
Stacks can be implemented efficiently using either contiguous arrays or singly linked lists. In an array-based stack implementation, an integer index tracks the current top of the stack; pushing increments this index and assigns the new payload, while popping decrements the index. If fixed-capacity arrays are utilized, pushing to a full stack triggers stack overflow exceptions. Alternatively, a linked stack maintains a pointer to the head node representing the top element; pushing allocates a new node whose next pointer references the prior top, achieving seamless dynamic scaling with O(1) time complexity for all fundamental operations.

## Infix to Postfix Conversion and Expression Evaluation
In computing, arithmetic expressions are commonly authored by humans in infix notation, where operators reside between operands (e.g., A + B). Compilers and evaluation engines systematically translate infix expressions into postfix notation (Reverse Polish Notation) to eliminate parenthetical ambiguity and enforce operator precedence. This conversion process relies heavily on an operator stack to buffer operators until higher-precedence operations are resolved. Once in postfix form, evaluation proceeds sequentially using an operand stack: numbers are pushed immediately, and encountering an operator triggers popping the required operands, performing the arithmetic operation, and pushing the resulting value back onto the stack.
