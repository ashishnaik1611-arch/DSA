# Stack - Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Stack Fundamentals](#stack-fundamentals)
3. [Implementation](#implementation)
4. [Basic Operations](#basic-operations)
5. [Expression Problems](#expression-problems)
6. [Important Stack Patterns](#important-stack-patterns)
7. [Common Problems](#common-problems)
8. [Practice Problems](#practice-problems)

---

## Introduction

**What is a Stack?**
- A linear data structure following **LIFO** (Last In First Out) principle
- Elements are added and removed from the same end (top)
- Like a stack of plates - you add/remove from top only
- Two main operations: Push (add) and Pop (remove)

**Why Learn Stacks?**
- Essential for recursion, backtracking, undo operations
- Used in expression evaluation, parsing
- Browser history, function call stack
- Very common in interviews (15-20% of problems)

**Real-World Applications:**
- Function call stack in programming
- Undo/Redo in editors
- Browser back button
- Expression evaluation
- Syntax checking (balanced parentheses)
- Backtracking algorithms

---

## Stack Fundamentals

### LIFO Principle

```
Push operations:      Pop operations:

Step 1: Push(5)       Step 1: Pop() → 30
[5]                   [5, 10, 20]

Step 2: Push(10)      Step 2: Pop() → 20
[5, 10]               [5, 10]

Step 3: Push(20)      Step 3: Pop() → 10
[5, 10, 20]           [5]

Step 4: Push(30)      Step 4: Pop() → 5
[5, 10, 20, 30]       []
     ↑ TOP                ↑ TOP
```

### Stack Operations

| Operation | Description | Time Complexity |
|-----------|-------------|-----------------|
| push(x) | Add element to top | O(1) |
| pop() | Remove and return top element | O(1) |
| peek() / top() | Return top element without removing | O(1) |
| isEmpty() | Check if stack is empty | O(1) |
| size() | Return number of elements | O(1) |

---

## Implementation

### 1. Using Python List

```python
class Stack:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        return self.items.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        return self.items[-1]

    def size(self):
        return len(self.items)

    def display(self):
        print("Stack:", self.items, "← TOP")

# Usage
stack = Stack()
stack.push(1)
stack.push(2)
stack.push(3)
stack.display()  # Stack: [1, 2, 3] ← TOP
print(stack.pop())  # 3
print(stack.peek())  # 2
```

### 2. Using collections.deque (More Efficient)

```python
from collections import deque

class Stack:
    def __init__(self):
        self.items = deque()

    def is_empty(self):
        return len(self.items) == 0

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        return self.items.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        return self.items[-1]

    def size(self):
        return len(self.items)
```

**Why deque over list?**
- More efficient for stack operations
- Thread-safe append and pop operations
- Better performance for large stacks

### 3. Using Linked List

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class Stack:
    def __init__(self):
        self.top = None
        self._size = 0

    def is_empty(self):
        return self.top is None

    def push(self, item):
        new_node = Node(item)
        new_node.next = self.top
        self.top = new_node
        self._size += 1

    def pop(self):
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        data = self.top.data
        self.top = self.top.next
        self._size -= 1
        return data

    def peek(self):
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        return self.top.data

    def size(self):
        return self._size
```

### 4. Fixed Size Stack (Using Array)

```python
class FixedStack:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = [None] * capacity
        self.top_index = -1

    def is_empty(self):
        return self.top_index == -1

    def is_full(self):
        return self.top_index == self.capacity - 1

    def push(self, item):
        if self.is_full():
            raise OverflowError("Stack is full")
        self.top_index += 1
        self.items[self.top_index] = item

    def pop(self):
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        item = self.items[self.top_index]
        self.top_index -= 1
        return item

    def peek(self):
        if self.is_empty():
            raise IndexError("Peek from empty stack")
        return self.items[self.top_index]

    def size(self):
        return self.top_index + 1
```

---

## Basic Operations

### Time & Space Complexity

**Time Complexity:**
- Push: O(1)
- Pop: O(1)
- Peek: O(1)
- isEmpty: O(1)
- Size: O(1)

**Space Complexity:**
- O(n) where n is number of elements

---

## Expression Problems

### 1. Balanced Parentheses

```python
def is_valid_parentheses(s):
    """
    Check if parentheses are balanced
    Example: "()[]{}" → True
             "(]" → False
    """
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}

    for char in s:
        if char in mapping:
            # Closing bracket
            top = stack.pop() if stack else '#'
            if mapping[char] != top:
                return False
        else:
            # Opening bracket
            stack.append(char)

    return not stack

# Examples
print(is_valid_parentheses("()"))      # True
print(is_valid_parentheses("()[]{}"))  # True
print(is_valid_parentheses("(]"))      # False
print(is_valid_parentheses("([)]"))    # False
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

### 2. Infix to Postfix Conversion

```python
def infix_to_postfix(expression):
    """
    Convert infix to postfix notation
    Example: "A+B*C" → "ABC*+"
    """
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    stack = []
    output = []

    for char in expression:
        # Operand
        if char.isalnum():
            output.append(char)

        # Left parenthesis
        elif char == '(':
            stack.append(char)

        # Right parenthesis
        elif char == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()  # Remove '('

        # Operator
        else:
            while (stack and stack[-1] != '(' and
                   precedence.get(stack[-1], 0) >= precedence.get(char, 0)):
                output.append(stack.pop())
            stack.append(char)

    # Pop remaining operators
    while stack:
        output.append(stack.pop())

    return ''.join(output)

# Example
print(infix_to_postfix("A+B*C"))        # ABC*+
print(infix_to_postfix("(A+B)*C"))      # AB+C*
print(infix_to_postfix("A+B*C-D/E"))    # ABC*+DE/-
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

### 3. Postfix Evaluation

```python
def evaluate_postfix(expression):
    """
    Evaluate postfix expression
    Example: "23*5+" → 11
    """
    stack = []

    for char in expression:
        # Operand
        if char.isdigit():
            stack.append(int(char))

        # Operator
        else:
            operand2 = stack.pop()
            operand1 = stack.pop()

            if char == '+':
                stack.append(operand1 + operand2)
            elif char == '-':
                stack.append(operand1 - operand2)
            elif char == '*':
                stack.append(operand1 * operand2)
            elif char == '/':
                stack.append(operand1 // operand2)

    return stack.pop()

# Examples
print(evaluate_postfix("23*5+"))   # 11 (2*3 + 5)
print(evaluate_postfix("53+2*"))   # 16 ((5+3)*2)
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

### 4. Infix Evaluation

```python
def evaluate_infix(expression):
    """
    Evaluate infix expression directly
    """
    def apply_operation(operators, values):
        operator = operators.pop()
        right = values.pop()
        left = values.pop()

        if operator == '+':
            values.append(left + right)
        elif operator == '-':
            values.append(left - right)
        elif operator == '*':
            values.append(left * right)
        elif operator == '/':
            values.append(left // right)

    def precedence(op):
        if op in ['+', '-']:
            return 1
        if op in ['*', '/']:
            return 2
        return 0

    values = []
    operators = []
    i = 0

    while i < len(expression):
        if expression[i] == ' ':
            i += 1
            continue

        # Opening bracket
        if expression[i] == '(':
            operators.append(expression[i])

        # Number
        elif expression[i].isdigit():
            num = 0
            while i < len(expression) and expression[i].isdigit():
                num = num * 10 + int(expression[i])
                i += 1
            values.append(num)
            i -= 1

        # Closing bracket
        elif expression[i] == ')':
            while operators and operators[-1] != '(':
                apply_operation(operators, values)
            operators.pop()

        # Operator
        else:
            while (operators and operators[-1] != '(' and
                   precedence(operators[-1]) >= precedence(expression[i])):
                apply_operation(operators, values)
            operators.append(expression[i])

        i += 1

    while operators:
        apply_operation(operators, values)

    return values[-1]

# Example
print(evaluate_infix("10 + 2 * 6"))  # 22
```

---

## Important Stack Patterns

### 1. Next Greater Element

**Problem:** For each element, find the next greater element to its right.

```python
def next_greater_element(arr):
    """
    Find next greater element for each element
    Example: [4, 5, 2, 25] → [5, 25, 25, -1]
    """
    n = len(arr)
    result = [-1] * n
    stack = []

    # Traverse from right to left
    for i in range(n - 1, -1, -1):
        # Pop elements smaller than current
        while stack and stack[-1] <= arr[i]:
            stack.pop()

        # If stack not empty, top is next greater
        if stack:
            result[i] = stack[-1]

        # Push current element
        stack.append(arr[i])

    return result

# Example
print(next_greater_element([4, 5, 2, 25]))  # [5, 25, 25, -1]
print(next_greater_element([13, 7, 6, 12]))  # [-1, 12, 12, -1]
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

### 2. Next Greater Element (Circular Array)

```python
def next_greater_circular(arr):
    """
    Find next greater element in circular array
    """
    n = len(arr)
    result = [-1] * n
    stack = []

    # Traverse twice (circular)
    for i in range(2 * n - 1, -1, -1):
        while stack and stack[-1] <= arr[i % n]:
            stack.pop()

        if i < n and stack:
            result[i] = stack[-1]

        stack.append(arr[i % n])

    return result

# Example
print(next_greater_circular([1, 2, 1]))  # [2, -1, 2]
```

### 3. Previous Greater Element

```python
def previous_greater_element(arr):
    """
    Find previous greater element for each element
    """
    n = len(arr)
    result = [-1] * n
    stack = []

    # Traverse from left to right
    for i in range(n):
        while stack and stack[-1] <= arr[i]:
            stack.pop()

        if stack:
            result[i] = stack[-1]

        stack.append(arr[i])

    return result
```

### 4. Stock Span Problem

```python
def calculate_span(prices):
    """
    Calculate span of stock prices
    Span = number of consecutive days before current day
           where price was less than or equal to current price
    """
    n = len(prices)
    span = [1] * n
    stack = []  # Stack of indices

    for i in range(n):
        # Pop elements with smaller or equal price
        while stack and prices[stack[-1]] <= prices[i]:
            stack.pop()

        # Calculate span
        span[i] = i + 1 if not stack else i - stack[-1]

        stack.append(i)

    return span

# Example
prices = [100, 80, 60, 70, 60, 75, 85]
print(calculate_span(prices))  # [1, 1, 1, 2, 1, 4, 6]
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

### 5. Largest Rectangle in Histogram

```python
def largest_rectangle_area(heights):
    """
    Find largest rectangle area in histogram
    """
    stack = []
    max_area = 0
    index = 0

    while index < len(heights):
        # If current bar is higher, push it
        if not stack or heights[index] >= heights[stack[-1]]:
            stack.append(index)
            index += 1
        else:
            # Calculate area with top as smallest bar
            top = stack.pop()
            width = index if not stack else index - stack[-1] - 1
            area = heights[top] * width
            max_area = max(max_area, area)

    # Process remaining bars
    while stack:
        top = stack.pop()
        width = index if not stack else index - stack[-1] - 1
        area = heights[top] * width
        max_area = max(max_area, area)

    return max_area

# Example
print(largest_rectangle_area([2, 1, 5, 6, 2, 3]))  # 10
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

### 6. Min Stack (Get Minimum in O(1))

```python
class MinStack:
    """
    Stack that supports getMin() in O(1)
    """
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self):
        if self.stack:
            val = self.stack.pop()
            if val == self.min_stack[-1]:
                self.min_stack.pop()
            return val

    def top(self):
        return self.stack[-1] if self.stack else None

    def get_min(self):
        return self.min_stack[-1] if self.min_stack else None

# Usage
min_stack = MinStack()
min_stack.push(3)
min_stack.push(5)
print(min_stack.get_min())  # 3
min_stack.push(2)
print(min_stack.get_min())  # 2
min_stack.pop()
print(min_stack.get_min())  # 3
```

### 7. Implement Queue Using Stacks

```python
class QueueUsingStacks:
    def __init__(self):
        self.stack1 = []  # For enqueue
        self.stack2 = []  # For dequeue

    def enqueue(self, x):
        self.stack1.append(x)

    def dequeue(self):
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())

        if not self.stack2:
            raise IndexError("Dequeue from empty queue")

        return self.stack2.pop()

    def peek(self):
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())

        if not self.stack2:
            raise IndexError("Peek from empty queue")

        return self.stack2[-1]

    def is_empty(self):
        return not self.stack1 and not self.stack2
```

---

## Common Problems

### Easy Level

**1. Valid Parentheses (Already shown above)**

**2. Remove All Adjacent Duplicates**
```python
def remove_duplicates(s):
    """
    Remove all adjacent duplicate characters
    Example: "abbaca" → "ca"
    """
    stack = []

    for char in s:
        if stack and stack[-1] == char:
            stack.pop()
        else:
            stack.append(char)

    return ''.join(stack)
```

**3. Baseball Game**
```python
def cal_points(ops):
    """
    Calculate points based on operations
    "5" - Add 5 to record
    "+" - Add sum of previous 2
    "D" - Add double of previous
    "C" - Remove previous
    """
    stack = []

    for op in ops:
        if op == '+':
            stack.append(stack[-1] + stack[-2])
        elif op == 'D':
            stack.append(2 * stack[-1])
        elif op == 'C':
            stack.pop()
        else:
            stack.append(int(op))

    return sum(stack)
```

**4. Make The String Great**
```python
def make_good(s):
    """
    Remove adjacent characters with different cases
    Example: "leEeetcode" → "leetcode"
    """
    stack = []

    for char in s:
        if stack and stack[-1].swapcase() == char:
            stack.pop()
        else:
            stack.append(char)

    return ''.join(stack)
```

### Medium Level

**1. Daily Temperatures**
```python
def daily_temperatures(temperatures):
    """
    Find how many days until warmer temperature
    Example: [73,74,75,71,69,72,76,73]
    Output: [1,1,4,2,1,1,0,0]
    """
    n = len(temperatures)
    answer = [0] * n
    stack = []  # Stack of indices

    for i in range(n):
        while stack and temperatures[i] > temperatures[stack[-1]]:
            prev_index = stack.pop()
            answer[prev_index] = i - prev_index

        stack.append(i)

    return answer
```

**2. Decode String**
```python
def decode_string(s):
    """
    Decode encoded string
    Example: "3[a2[c]]" → "accaccacc"
    """
    stack = []
    current_num = 0
    current_str = ""

    for char in s:
        if char.isdigit():
            current_num = current_num * 10 + int(char)
        elif char == '[':
            stack.append(current_str)
            stack.append(current_num)
            current_str = ""
            current_num = 0
        elif char == ']':
            num = stack.pop()
            prev_str = stack.pop()
            current_str = prev_str + num * current_str
        else:
            current_str += char

    return current_str
```

**3. Asteroid Collision**
```python
def asteroid_collision(asteroids):
    """
    Simulate asteroid collisions
    Positive = moving right, Negative = moving left
    Example: [5, 10, -5] → [5, 10]
    """
    stack = []

    for asteroid in asteroids:
        while stack and asteroid < 0 < stack[-1]:
            if stack[-1] < -asteroid:
                stack.pop()
                continue
            elif stack[-1] == -asteroid:
                stack.pop()
            break
        else:
            stack.append(asteroid)

    return stack
```

**4. Remove K Digits**
```python
def remove_k_digits(num, k):
    """
    Remove k digits to get smallest possible number
    Example: "1432219", k=3 → "1219"
    """
    stack = []

    for digit in num:
        while k > 0 and stack and stack[-1] > digit:
            stack.pop()
            k -= 1
        stack.append(digit)

    # Remove remaining k digits from end
    stack = stack[:-k] if k else stack

    # Remove leading zeros
    result = ''.join(stack).lstrip('0')

    return result if result else '0'
```

**5. Simplify Path**
```python
def simplify_path(path):
    """
    Simplify Unix-style file path
    Example: "/a/./b/../../c/" → "/c"
    """
    stack = []
    parts = path.split('/')

    for part in parts:
        if part == '..' and stack:
            stack.pop()
        elif part and part != '.' and part != '..':
            stack.append(part)

    return '/' + '/'.join(stack)
```

### Hard Level

**1. Longest Valid Parentheses**
```python
def longest_valid_parentheses(s):
    """
    Find length of longest valid parentheses substring
    Example: "(()" → 2, ")()())" → 4
    """
    stack = [-1]
    max_length = 0

    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)
            else:
                max_length = max(max_length, i - stack[-1])

    return max_length
```

**2. Basic Calculator**
```python
def calculate(s):
    """
    Evaluate expression with +, -, (, )
    Example: "(1+(4+5+2)-3)+(6+8)" → 23
    """
    stack = []
    operand = 0
    result = 0
    sign = 1

    for char in s:
        if char.isdigit():
            operand = operand * 10 + int(char)
        elif char == '+':
            result += sign * operand
            sign = 1
            operand = 0
        elif char == '-':
            result += sign * operand
            sign = -1
            operand = 0
        elif char == '(':
            stack.append(result)
            stack.append(sign)
            result = 0
            sign = 1
        elif char == ')':
            result += sign * operand
            result *= stack.pop()  # sign
            result += stack.pop()  # previous result
            operand = 0

    return result + sign * operand
```

**3. Maximal Rectangle**
```python
def maximal_rectangle(matrix):
    """
    Find largest rectangle containing only 1s
    """
    if not matrix:
        return 0

    max_area = 0
    heights = [0] * len(matrix[0])

    for row in matrix:
        for i in range(len(row)):
            heights[i] = heights[i] + 1 if row[i] == '1' else 0

        max_area = max(max_area, largest_rectangle_area(heights))

    return max_area
```

---

## Practice Problems

### Beginner (Easy)

1. Valid parentheses
2. Implement stack using array
3. Implement stack using queue
4. Remove all adjacent duplicates
5. Baseball game
6. Make the string great
7. Backspace string compare
8. Next greater element I
9. Build an array with stack operations
10. Final prices with special discount

### Intermediate (Medium)

11. Min stack
12. Implement queue using stacks
13. Daily temperatures
14. Next greater element II
15. Online stock span
16. Decode string
17. Asteroid collision
18. Remove K digits
19. Simplify path
20. Validate stack sequences
21. Score of parentheses
22. Remove duplicate letters
23. Sum of subarray minimums
24. 132 pattern
25. Evaluate reverse polish notation
26. Design browser history
27. Car fleet
28. Largest rectangle in histogram
29. Exclusive time of functions
30. Tag validator

### Advanced (Hard)

31. Longest valid parentheses
32. Basic calculator
33. Basic calculator II
34. Basic calculator III
35. Maximal rectangle
36. Trapping rain water (using stack)
37. Number of atoms
38. Parsing a boolean expression
39. Reverse substrings between parentheses
40. Minimum number of swaps to balance

---

## Tips for Stack Problems

### When to Use Stack

✅ **Use stack when you see:**
- Balanced parentheses/brackets
- Expression evaluation
- Next greater/smaller element
- Undo/redo operations
- Backtracking
- Function call simulation
- Reverse order needed
- Nested structures

### Pattern Recognition

- **"Balanced/Valid"** → Stack with matching
- **"Next greater"** → Monotonic decreasing stack
- **"Next smaller"** → Monotonic increasing stack
- **"Expression evaluation"** → Two stacks (operators & operands)
- **"Decode/Parse"** → Stack for nesting
- **"Undo operation"** → Stack naturally

### Common Techniques

**1. Monotonic Stack:**
```python
# For next greater element (decreasing stack)
stack = []
for i in range(len(arr)):
    while stack and arr[i] > arr[stack[-1]]:
        # Process
        stack.pop()
    stack.append(i)
```

**2. Stack with Index:**
```python
# Store indices instead of values
stack = []  # stores indices
for i in range(len(arr)):
    # Use arr[stack[-1]] to access value
    pass
```

**3. Two Stacks:**
```python
# For expression evaluation
operators = []
operands = []
```

### Common Mistakes

❌ Not checking if stack is empty before pop/peek
❌ Forgetting to handle remaining elements in stack
❌ Using wrong stack for the problem
❌ Not considering edge cases (empty string, single element)

✅ Always check `if stack` before popping
✅ Process remaining elements after loop
✅ Use appropriate data structure
✅ Handle edge cases first

---

## Python Stack Tips

### Built-in Stack Operations

```python
# Using list as stack
stack = []
stack.append(1)      # Push
stack.pop()          # Pop
stack[-1]            # Peek (top)
len(stack)           # Size
not stack            # isEmpty
```

### Using deque (Better Performance)

```python
from collections import deque

stack = deque()
stack.append(1)      # Push - O(1)
stack.pop()          # Pop - O(1)
stack[-1]            # Peek - O(1)
```

### Stack Tricks

```python
# Reverse using stack
def reverse(s):
    stack = list(s)
    return ''.join(reversed(stack))

# Check balanced in one pass
def is_balanced(s):
    count = 0
    for char in s:
        count += 1 if char == '(' else -1
        if count < 0:
            return False
    return count == 0
```

---

## Summary

### Must Know Concepts
✅ LIFO principle
✅ Push, Pop, Peek operations (all O(1))
✅ Implementation using array/list
✅ Balanced parentheses
✅ Expression conversion and evaluation
✅ Next greater/smaller element pattern
✅ Monotonic stack concept
✅ Min/Max stack

### Key Takeaways
- Stack follows LIFO (Last In First Out)
- All operations are O(1)
- Perfect for reversing, backtracking, nesting
- Use monotonic stack for next greater/smaller
- Always check empty before pop/peek
- Common in 15-20% of interview problems

### Time Complexity Cheat Sheet
- Push: O(1)
- Pop: O(1)
- Peek: O(1)
- Search: O(n)
- Space: O(n)

---

**Next Steps:**
1. Implement stack using array and linked list
2. Solve all expression problems
3. Master next greater element pattern
4. Practice 15-20 medium problems
5. Learn monotonic stack applications

Good luck with your stack learning journey! 🚀

---

*Last Updated: November 2024*
