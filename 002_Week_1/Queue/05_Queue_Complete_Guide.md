# Queue - Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Queue Fundamentals](#queue-fundamentals)
3. [Types of Queues](#types-of-queues)
4. [Implementation](#implementation)
5. [Important Queue Patterns](#important-queue-patterns)
6. [Common Problems](#common-problems)
7. [Practice Problems](#practice-problems)

---

## Introduction

**What is a Queue?**
- A linear data structure following **FIFO** (First In First Out) principle
- Elements are added at rear/back and removed from front
- Like a queue of people - first person in line is served first
- Two main operations: Enqueue (add) and Dequeue (remove)

**Why Learn Queues?**
- Essential for BFS (Breadth First Search)
- Used in scheduling, buffering, resource sharing
- Message queues, task scheduling
- Common in system design and interviews

**Real-World Applications:**
- Print queue (printer spooler)
- CPU task scheduling
- BFS in graphs and trees
- Handling requests in web servers
- Message queues in distributed systems
- Call center phone systems

---

## Queue Fundamentals

### FIFO Principle

```
Enqueue operations:       Dequeue operations:

Step 1: Enqueue(5)        Step 1: Dequeue() → 5
Front [5] Rear            Front [10, 20, 30] Rear

Step 2: Enqueue(10)       Step 2: Dequeue() → 10
Front [5, 10] Rear        Front [20, 30] Rear

Step 3: Enqueue(20)       Step 3: Dequeue() → 20
Front [5, 10, 20] Rear    Front [30] Rear

Step 4: Enqueue(30)       Step 4: Dequeue() → 30
Front [5, 10, 20, 30] Rear    Front [] Rear (Empty)
```

### Queue Operations

| Operation | Description | Time Complexity |
|-----------|-------------|-----------------|
| enqueue(x) | Add element to rear | O(1) |
| dequeue() | Remove and return front element | O(1) |
| front() / peek() | Return front element without removing | O(1) |
| isEmpty() | Check if queue is empty | O(1) |
| size() | Return number of elements | O(1) |

---

## Types of Queues

### 1. Simple Queue (Linear Queue)
- Standard FIFO queue
- Elements added at rear, removed from front

### 2. Circular Queue
- Last position is connected to first position
- Efficient space utilization
- Overcomes limitation of simple queue

### 3. Deque (Double-Ended Queue)
- Elements can be added/removed from both ends
- More flexible than simple queue

### 4. Priority Queue
- Elements have priority
- Higher priority elements dequeued first
- Implemented using heap

---

## Implementation

### 1. Simple Queue Using List

```python
class Queue:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        return self.items.pop(0)  # O(n) - inefficient!

    def front(self):
        if self.is_empty():
            raise IndexError("Front from empty queue")
        return self.items[0]

    def size(self):
        return len(self.items)

    def display(self):
        print("Front", self.items, "Rear")
```

**Problem:** `pop(0)` is O(n) because it shifts all elements.

### 2. Queue Using collections.deque (Recommended)

```python
from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()

    def is_empty(self):
        return len(self.items) == 0

    def enqueue(self, item):
        self.items.append(item)  # O(1)

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        return self.items.popleft()  # O(1)

    def front(self):
        if self.is_empty():
            raise IndexError("Front from empty queue")
        return self.items[0]

    def size(self):
        return len(self.items)

    def display(self):
        print("Front", list(self.items), "Rear")

# Usage
queue = Queue()
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)
queue.display()  # Front [1, 2, 3] Rear
print(queue.dequeue())  # 1
print(queue.front())    # 2
```

### 3. Circular Queue Using Array

```python
class CircularQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = [None] * capacity
        self.front = -1
        self.rear = -1
        self.count = 0

    def is_empty(self):
        return self.count == 0

    def is_full(self):
        return self.count == self.capacity

    def enqueue(self, item):
        if self.is_full():
            raise OverflowError("Queue is full")

        if self.front == -1:
            self.front = 0

        self.rear = (self.rear + 1) % self.capacity
        self.queue[self.rear] = item
        self.count += 1

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")

        item = self.queue[self.front]
        self.front = (self.front + 1) % self.capacity
        self.count -= 1

        if self.count == 0:
            self.front = -1
            self.rear = -1

        return item

    def front_element(self):
        if self.is_empty():
            raise IndexError("Front from empty queue")
        return self.queue[self.front]

    def size(self):
        return self.count

    def display(self):
        if self.is_empty():
            print("Queue is empty")
            return

        i = self.front
        elements = []
        for _ in range(self.count):
            elements.append(self.queue[i])
            i = (i + 1) % self.capacity

        print("Front", elements, "Rear")
```

### 4. Queue Using Linked List

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class Queue:
    def __init__(self):
        self.front = None
        self.rear = None
        self._size = 0

    def is_empty(self):
        return self.front is None

    def enqueue(self, item):
        new_node = Node(item)

        if self.rear is None:
            self.front = self.rear = new_node
        else:
            self.rear.next = new_node
            self.rear = new_node

        self._size += 1

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")

        item = self.front.data
        self.front = self.front.next

        if self.front is None:
            self.rear = None

        self._size -= 1
        return item

    def front_element(self):
        if self.is_empty():
            raise IndexError("Front from empty queue")
        return self.front.data

    def size(self):
        return self._size
```

### 5. Deque (Double-Ended Queue)

```python
from collections import deque

class Deque:
    def __init__(self):
        self.items = deque()

    def is_empty(self):
        return len(self.items) == 0

    # Add operations
    def add_front(self, item):
        self.items.appendleft(item)

    def add_rear(self, item):
        self.items.append(item)

    # Remove operations
    def remove_front(self):
        if self.is_empty():
            raise IndexError("Remove from empty deque")
        return self.items.popleft()

    def remove_rear(self):
        if self.is_empty():
            raise IndexError("Remove from empty deque")
        return self.items.pop()

    # Peek operations
    def front(self):
        if self.is_empty():
            raise IndexError("Front from empty deque")
        return self.items[0]

    def rear(self):
        if self.is_empty():
            raise IndexError("Rear from empty deque")
        return self.items[-1]

    def size(self):
        return len(self.items)

# Usage
dq = Deque()
dq.add_rear(1)
dq.add_rear(2)
dq.add_front(0)
print(dq.items)  # deque([0, 1, 2])
```

### 6. Priority Queue Using heapq

```python
import heapq

class PriorityQueue:
    def __init__(self):
        self.heap = []

    def is_empty(self):
        return len(self.heap) == 0

    def enqueue(self, item, priority):
        # Python heapq is min-heap
        # Use negative priority for max-heap behavior
        heapq.heappush(self.heap, (priority, item))

    def dequeue(self):
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        return heapq.heappop(self.heap)[1]

    def front(self):
        if self.is_empty():
            raise IndexError("Front from empty queue")
        return self.heap[0][1]

    def size(self):
        return len(self.heap)

# Usage
pq = PriorityQueue()
pq.enqueue("task1", 3)
pq.enqueue("task2", 1)  # Higher priority (lower number)
pq.enqueue("task3", 2)
print(pq.dequeue())  # task2 (priority 1)
```

---

## Important Queue Patterns

### 1. Implement Stack Using Queues

```python
from collections import deque

class StackUsingQueues:
    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()

    def push(self, x):
        # Add to q2
        self.q2.append(x)

        # Move all elements from q1 to q2
        while self.q1:
            self.q2.append(self.q1.popleft())

        # Swap q1 and q2
        self.q1, self.q2 = self.q2, self.q1

    def pop(self):
        if not self.q1:
            raise IndexError("Pop from empty stack")
        return self.q1.popleft()

    def top(self):
        if not self.q1:
            raise IndexError("Top from empty stack")
        return self.q1[0]

    def is_empty(self):
        return not self.q1
```

### 2. Reverse First K Elements of Queue

```python
def reverse_first_k(queue, k):
    """
    Reverse first k elements of queue
    """
    if k <= 0 or k > queue.size():
        return

    stack = []

    # Dequeue first k elements and push to stack
    for _ in range(k):
        stack.append(queue.dequeue())

    # Enqueue back from stack (reversed)
    while stack:
        queue.enqueue(stack.pop())

    # Move remaining elements to back
    for _ in range(queue.size() - k):
        queue.enqueue(queue.dequeue())

    return queue
```

### 3. Generate Binary Numbers 1 to N

```python
def generate_binary_numbers(n):
    """
    Generate binary numbers from 1 to n using queue
    Example: n=5 → ["1", "10", "11", "100", "101"]
    """
    result = []
    queue = deque(['1'])

    for _ in range(n):
        current = queue.popleft()
        result.append(current)

        # Generate next numbers
        queue.append(current + '0')
        queue.append(current + '1')

    return result

# Example
print(generate_binary_numbers(5))
# ['1', '10', '11', '100', '101']
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

### 4. First Non-Repeating Character in Stream

```python
from collections import deque

class FirstNonRepeating:
    def __init__(self):
        self.queue = deque()
        self.freq = {}

    def add_char(self, char):
        """Add character and return first non-repeating"""
        # Update frequency
        self.freq[char] = self.freq.get(char, 0) + 1

        # Add to queue
        self.queue.append(char)

        # Remove repeating characters from front
        while self.queue and self.freq[self.queue[0]] > 1:
            self.queue.popleft()

        # Return first non-repeating or -1
        return self.queue[0] if self.queue else -1

# Usage
stream = FirstNonRepeating()
for char in "aabcc":
    print(stream.add_char(char))
# Output: a, -1, b, b, -1
```

### 5. Sliding Window Maximum Using Deque

```python
from collections import deque

def max_sliding_window(nums, k):
    """
    Find maximum in each sliding window of size k
    Example: nums = [1,3,-1,-3,5,3,6,7], k = 3
    Output: [3,3,5,5,6,7]
    """
    if not nums:
        return []

    result = []
    dq = deque()  # Store indices

    for i in range(len(nums)):
        # Remove elements outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove smaller elements from rear
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        # Add to result if window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result

# Example
print(max_sliding_window([1,3,-1,-3,5,3,6,7], 3))
# [3, 3, 5, 5, 6, 7]
```

**Time Complexity:** O(n)
**Space Complexity:** O(k)

### 6. Interleave First Half with Second Half

```python
from collections import deque

def interleave_queue(queue):
    """
    Interleave first half with second half
    Example: [1,2,3,4,5,6] → [1,4,2,5,3,6]
    """
    if queue.size() % 2 != 0:
        raise ValueError("Queue size must be even")

    n = queue.size() // 2
    first_half = deque()

    # Move first half to temporary queue
    for _ in range(n):
        first_half.append(queue.dequeue())

    # Interleave
    while first_half:
        queue.enqueue(first_half.popleft())
        queue.enqueue(queue.dequeue())

    return queue
```

### 7. Circular Tour (Gas Station Problem)

```python
def can_complete_circuit(gas, cost):
    """
    Find starting gas station for circular tour
    Return -1 if no solution exists
    """
    if sum(gas) < sum(cost):
        return -1

    start = 0
    tank = 0

    for i in range(len(gas)):
        tank += gas[i] - cost[i]

        if tank < 0:
            start = i + 1
            tank = 0

    return start

# Example
gas = [1, 2, 3, 4, 5]
cost = [3, 4, 5, 1, 2]
print(can_complete_circuit(gas, cost))  # 3
```

---

## Common Problems

### Easy Level

**1. Implement Queue Using Stacks**
```python
class QueueUsingStacks:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def enqueue(self, x):
        self.stack1.append(x)

    def dequeue(self):
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2.pop() if self.stack2 else None

    def front(self):
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2[-1] if self.stack2 else None

    def is_empty(self):
        return not self.stack1 and not self.stack2
```

**2. Number of Recent Calls**
```python
from collections import deque

class RecentCounter:
    def __init__(self):
        self.queue = deque()

    def ping(self, t):
        """Count requests in last 3000ms"""
        self.queue.append(t)

        # Remove requests older than 3000ms
        while self.queue[0] < t - 3000:
            self.queue.popleft()

        return len(self.queue)
```

**3. Time Needed to Buy Tickets**
```python
def time_required_to_buy(tickets, k):
    """
    Calculate time for person at position k to buy tickets
    """
    time = 0

    for i in range(len(tickets)):
        if i <= k:
            time += min(tickets[i], tickets[k])
        else:
            time += min(tickets[i], tickets[k] - 1)

    return time
```

### Medium Level

**1. Design Circular Queue**
```python
class CircularQueue:
    def __init__(self, k):
        self.size = k
        self.queue = [None] * k
        self.front = -1
        self.rear = -1

    def enqueue(self, value):
        if self.is_full():
            return False

        if self.is_empty():
            self.front = 0

        self.rear = (self.rear + 1) % self.size
        self.queue[self.rear] = value
        return True

    def dequeue(self):
        if self.is_empty():
            return False

        if self.front == self.rear:
            self.front = -1
            self.rear = -1
        else:
            self.front = (self.front + 1) % self.size

        return True

    def front_element(self):
        return -1 if self.is_empty() else self.queue[self.front]

    def rear_element(self):
        return -1 if self.is_empty() else self.queue[self.rear]

    def is_empty(self):
        return self.front == -1

    def is_full(self):
        return (self.rear + 1) % self.size == self.front
```

**2. Dota2 Senate**
```python
from collections import deque

def predict_party_victory(senate):
    """
    Simulate voting rounds
    Example: "RDD" → "Dire" wins
    """
    radiant = deque()
    dire = deque()

    # Store indices
    for i, s in enumerate(senate):
        if s == 'R':
            radiant.append(i)
        else:
            dire.append(i)

    n = len(senate)

    while radiant and dire:
        r_index = radiant.popleft()
        d_index = dire.popleft()

        # Earlier senator bans the other
        if r_index < d_index:
            radiant.append(r_index + n)
        else:
            dire.append(d_index + n)

    return "Radiant" if radiant else "Dire"
```

**3. Jump Game VI**
```python
from collections import deque

def max_result(nums, k):
    """
    Find maximum score jumping at most k steps
    """
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    dq = deque([0])

    for i in range(1, n):
        # Remove indices outside window
        while dq and dq[0] < i - k:
            dq.popleft()

        dp[i] = nums[i] + dp[dq[0]]

        # Maintain decreasing deque
        while dq and dp[dq[-1]] <= dp[i]:
            dq.pop()

        dq.append(i)

    return dp[-1]
```

**4. Reveal Cards in Increasing Order**
```python
from collections import deque

def deck_revealed_increasing(deck):
    """
    Arrange deck to reveal cards in increasing order
    """
    deck.sort()
    result = deque()

    for card in reversed(deck):
        if result:
            result.appendleft(result.pop())
        result.appendleft(card)

    return list(result)
```

### Hard Level

**1. Shortest Subarray with Sum at Least K**
```python
from collections import deque

def shortest_subarray(nums, k):
    """
    Find shortest subarray with sum >= k
    """
    n = len(nums)
    prefix = [0] * (n + 1)

    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]

    result = float('inf')
    dq = deque()

    for i in range(n + 1):
        # Check if we can form subarray
        while dq and prefix[i] - prefix[dq[0]] >= k:
            result = min(result, i - dq.popleft())

        # Maintain increasing deque
        while dq and prefix[i] <= prefix[dq[-1]]:
            dq.pop()

        dq.append(i)

    return result if result != float('inf') else -1
```

**2. Constrained Subsequence Sum**
```python
from collections import deque

def constrained_subset_sum(nums, k):
    """
    Maximum sum of non-empty subsequence with at most k distance
    """
    n = len(nums)
    dp = nums[:]
    dq = deque()

    for i in range(n):
        # Remove elements outside window
        while dq and dq[0] < i - k:
            dq.popleft()

        # Calculate dp[i]
        if dq:
            dp[i] = max(dp[i], nums[i] + dp[dq[0]])

        # Maintain decreasing deque
        while dq and dp[dq[-1]] <= dp[i]:
            dq.pop()

        if dp[i] > 0:
            dq.append(i)

    return max(dp)
```

**3. Shortest Path in Binary Matrix with Obstacles Elimination**
```python
from collections import deque

def shortest_path(grid, k):
    """
    Find shortest path with ability to eliminate k obstacles
    """
    n = len(grid)
    if grid[0][0] == 1 or grid[n-1][n-1] == 1:
        if k == 0:
            return -1

    queue = deque([(0, 0, k, 0)])  # row, col, remaining_k, steps
    visited = {(0, 0, k)}

    directions = [(0,1), (1,0), (0,-1), (-1,0)]

    while queue:
        row, col, remaining_k, steps = queue.popleft()

        if row == n-1 and col == n-1:
            return steps

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            if 0 <= new_row < n and 0 <= new_col < n:
                new_k = remaining_k - grid[new_row][new_col]

                if new_k >= 0 and (new_row, new_col, new_k) not in visited:
                    visited.add((new_row, new_col, new_k))
                    queue.append((new_row, new_col, new_k, steps + 1))

    return -1
```

---

## Practice Problems

### Beginner (Easy)

1. Implement queue using array
2. Implement queue using stacks
3. Implement stack using queues
4. Number of recent calls
5. Time needed to buy tickets
6. Number of students unable to eat lunch
7. Design circular queue
8. First unique character in string
9. Moving average from data stream
10. Generate binary numbers

### Intermediate (Medium)

11. Reverse first K elements of queue
12. First non-repeating character in stream
13. Interleave first half with second half
14. Design hit counter
15. Dota2 senate
16. Reveal cards in increasing order
17. Jump game VI
18. Task scheduler
19. K closest points to origin
20. Design snake game
21. Shortest subarray with sum at least K
22. Constrained subsequence sum
23. Sliding window maximum
24. Maximum of minimums of every window size
25. Distance of nearest cell having 1
26. Rotting oranges
27. Open the lock
28. Perfect squares (BFS)
29. Walls and gates
30. 01 Matrix

### Advanced (Hard)

31. Shortest path in binary matrix with obstacles
32. Sliding window median
33. Max value of equation
34. Count unique characters in substrings
35. Minimum cost to hire K workers
36. IPO
37. Trapping rain water II
38. Swim in rising water
39. Cut off trees for golf event
40. Bus routes

---

## Tips for Queue Problems

### When to Use Queue

✅ **Use queue when you see:**
- BFS (Breadth First Search)
- Level order traversal
- First come first served
- Process in order of arrival
- Sliding window (with deque)
- Shortest path (unweighted)
- "First" or "Earliest" in problem

### Pattern Recognition

- **"Level by level"** → BFS with queue
- **"Shortest path"** → BFS with queue
- **"First non-repeating"** → Queue + frequency
- **"Sliding window maximum"** → Monotonic deque
- **"Process in order"** → Simple queue
- **"Add/remove from both ends"** → Deque
- **"Priority-based"** → Priority queue

### Common Techniques

**1. BFS Template:**
```python
from collections import deque

def bfs(start):
    queue = deque([start])
    visited = {start}

    while queue:
        node = queue.popleft()

        # Process node

        for neighbor in get_neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

**2. Level Order Template:**
```python
def level_order(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result
```

**3. Monotonic Deque Template:**
```python
from collections import deque

def sliding_window_pattern(arr, k):
    dq = deque()
    result = []

    for i in range(len(arr)):
        # Remove elements outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Maintain monotonic property
        while dq and arr[dq[-1]] < arr[i]:
            dq.pop()

        dq.append(i)

        if i >= k - 1:
            result.append(arr[dq[0]])

    return result
```

### Common Mistakes

❌ Using list.pop(0) - O(n) operation
❌ Not checking empty before dequeue
❌ Forgetting to process level by level in BFS
❌ Not using deque for sliding window

✅ Use collections.deque for O(1) operations
✅ Always check if queue is empty
✅ Track level size in BFS
✅ Use monotonic deque for window problems

---

## Queue vs Other Structures

### Queue vs Stack
- Queue: FIFO (First In First Out)
- Stack: LIFO (Last In First Out)
- Use Queue for: BFS, ordering, scheduling
- Use Stack for: DFS, recursion, backtracking

### Simple Queue vs Circular Queue
- Simple: Wasted space when elements dequeued
- Circular: Efficient space utilization
- Use Circular when: Fixed size, continuous operations

### Queue vs Deque
- Queue: Add rear, remove front
- Deque: Add/remove both ends
- Use Deque for: Sliding window, palindrome check

---

## Python Queue Tips

### Using collections.deque

```python
from collections import deque

# Create deque
dq = deque()

# Queue operations
dq.append(1)        # Enqueue (rear)
dq.popleft()        # Dequeue (front)
dq[0]               # Front/peek
len(dq)             # Size
not dq              # isEmpty

# Deque operations
dq.appendleft(0)    # Add to front
dq.pop()            # Remove from rear

# Initialize with values
dq = deque([1, 2, 3])

# Max length (circular behavior)
dq = deque(maxlen=5)
```

### Using queue module

```python
from queue import Queue

# Thread-safe queue
q = Queue()
q.put(1)            # Enqueue
q.get()             # Dequeue (blocks if empty)
q.qsize()           # Size
q.empty()           # Check empty
```

---

## Summary

### Must Know Concepts
✅ FIFO principle
✅ Enqueue, Dequeue operations (both O(1))
✅ Implementation using deque
✅ Circular queue concept
✅ Deque (double-ended queue)
✅ BFS pattern with queue
✅ Sliding window with monotonic deque
✅ Priority queue basics

### Key Takeaways
- Queue follows FIFO (First In First Out)
- Use collections.deque for efficient operations
- Perfect for BFS and level-order traversal
- Deque for sliding window problems
- Always check empty before dequeue
- Common in tree and graph problems

### Time Complexity Cheat Sheet
- Enqueue: O(1)
- Dequeue: O(1)
- Front/Peek: O(1)
- Search: O(n)
- Space: O(n)

---

**Next Steps:**
1. Implement all queue types
2. Master BFS pattern
3. Practice sliding window with deque
4. Solve 15-20 medium problems
5. Learn priority queue applications

Good luck with your queue learning journey! 🚀

---

*Last Updated: November 2024*
