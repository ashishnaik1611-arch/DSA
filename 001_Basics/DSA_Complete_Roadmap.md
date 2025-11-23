# Complete DSA Roadmap - Python Edition

A comprehensive guide to Data Structures and Algorithms for mastering technical interviews and problem-solving.

---

## Table of Contents
1. [Arrays](#1-arrays)
2. [Strings](#2-strings)
3. [Linked Lists](#3-linked-lists)
4. [Stacks](#4-stacks)
5. [Queues](#5-queues)
6. [Hashing](#6-hashing)
7. [Recursion](#7-recursion)
8. [Backtracking](#8-backtracking)
9. [Trees](#9-trees)
10. [Heaps / Priority Queue](#10-heaps--priority-queue)
11. [Graphs](#11-graphs)
12. [Dynamic Programming](#12-dynamic-programming)
13. [Greedy Algorithms](#13-greedy-algorithms)
14. [Divide and Conquer](#14-divide-and-conquer)
15. [Bit Manipulation](#15-bit-manipulation)
16. [Mathematical Algorithms](#16-mathematical-algorithms)
17. [Searching & Sorting](#17-searching--sorting)
18. [Advanced Data Structures](#18-advanced-data-structures)
19. [Advanced Topics](#19-advanced-topics)
20. [Problem-Solving Patterns](#20-problem-solving-patterns)

---

## 1. ARRAYS

### Topics to Learn
- **Basic Operations**
  - Insertion (beginning, middle, end) - O(n)
  - Deletion (beginning, middle, end) - O(n)
  - Access by index - O(1)
  - Searching - O(n) linear, O(log n) binary
  - Traversal techniques

- **Two Pointer Technique**
  - Two pointers from start and end
  - Fast and slow pointers
  - Meeting point problems
  - Sorting-based two pointer

- **Sliding Window**
  - Fixed-size window
  - Variable-size window
  - Maximum/minimum in window
  - Substring problems

- **Prefix Sum / Cumulative Sum**
  - Building prefix sum array
  - Range sum queries
  - Subarray sum problems

- **Subarray Problems**
  - Maximum subarray sum (Kadane's Algorithm)
  - Minimum subarray sum
  - Subarray with given sum
  - Longest subarray with conditions
  - Count subarrays with conditions

- **Sorting Algorithms**
  - Bubble Sort - O(n²)
  - Selection Sort - O(n²)
  - Insertion Sort - O(n²)
  - Merge Sort - O(n log n)
  - Quick Sort - O(n log n) average
  - Counting Sort - O(n+k)

- **Searching Algorithms**
  - Linear Search - O(n)
  - Binary Search - O(log n)
  - Search in rotated array
  - Find first/last occurrence
  - Search in 2D matrix

- **Array Manipulation**
  - Merging arrays
  - Finding duplicates
  - Removing duplicates
  - Array rotation (left/right)
  - Rearrangement problems
  - Array partitioning

- **Advanced Patterns**
  - Dutch National Flag (3-way partitioning)
  - Moore's Voting Algorithm (majority element)
  - Next Greater/Smaller Element
  - Stock buy-sell problems
  - Trapping rainwater
  - Container with most water

- **Matrix (2D Arrays)**
  - Matrix traversal (row-wise, column-wise, spiral)
  - Matrix rotation (90°, 180°)
  - Matrix transpose
  - Searching in sorted matrix
  - Diagonal traversal

### Key Concepts
- Time Complexity: Access O(1), Search O(n), Insert/Delete O(n)
- Space Complexity: O(n)
- Dynamic vs Static arrays
- In-place algorithms

---

## 2. STRINGS

### Topics to Learn
- **String Basics**
  - String creation and initialization
  - String immutability in Python
  - String methods (split, join, strip, replace, find)
  - String slicing and indexing
  - String concatenation

- **Pattern Matching**
  - Naive pattern matching - O(n*m)
  - KMP Algorithm - O(n+m)
  - Rabin-Karp Algorithm
  - Boyer-Moore Algorithm
  - Z Algorithm

- **String Manipulation**
  - Anagram problems
  - Palindrome problems
  - Subsequence & substring problems
  - String reversal
  - Longest common prefix/suffix
  - String compression
  - Remove duplicates

- **Advanced String Problems**
  - Longest palindromic substring
  - Longest repeating subsequence
  - Edit distance
  - Wildcard pattern matching
  - Regular expression matching
  - Word break problem
  - Interleaving strings

### Key Concepts
- Strings are immutable in Python
- Use list for mutable operations
- Character frequency using Counter
- ASCII values: ord(), chr()

---

## 3. LINKED LISTS

### Topics to Learn
- **Types of Linked Lists**
  - Singly Linked List
  - Doubly Linked List
  - Circular Linked List
  - Skip List

- **Basic Operations**
  - Insert (beginning, middle, end) - O(1) at head, O(n) otherwise
  - Delete (beginning, middle, end) - O(1) at head, O(n) otherwise
  - Search - O(n)
  - Traverse - O(n)

- **Common Problems**
  - Reverse a linked list (iterative & recursive)
  - Detect cycle (Floyd's Cycle Detection)
  - Find middle element (slow-fast pointer)
  - Merge two sorted lists
  - Remove duplicates
  - Intersection point of two lists
  - Rotate a linked list
  - Flatten a linked list
  - Clone a linked list with random pointer
  - Palindrome linked list

- **Advanced Problems**
  - LRU Cache implementation
  - Sort linked list (merge sort)
  - Reverse in groups of k
  - Add two numbers represented as lists
  - Segregate even-odd nodes

### Key Concepts
- No random access (no indexing)
- Dynamic size
- Efficient insertion/deletion at beginning
- Extra space for pointers

---

## 4. STACKS

### Topics to Learn
- **Stack Basics**
  - LIFO (Last In First Out) principle
  - Push, Pop, Peek operations - O(1)
  - Implementation using array
  - Implementation using linked list
  - Stack using Python list

- **Expression Problems**
  - Infix to Postfix conversion
  - Infix to Prefix conversion
  - Postfix evaluation
  - Prefix evaluation
  - Balanced parentheses

- **Stack Applications**
  - Function call stack
  - Undo/Redo operations
  - Browser back button
  - Expression evaluation

- **Advanced Stack Problems**
  - Next Greater Element
  - Next Smaller Element
  - Previous Greater Element
  - Stock span problem
  - Largest rectangle in histogram
  - Min/Max stack (get min in O(1))
  - Stack using queues
  - Celebrity problem
  - Implement k stacks in array

- **Monotonic Stack**
  - Increasing monotonic stack
  - Decreasing monotonic stack
  - Applications in array problems

### Key Concepts
- Time Complexity: Push O(1), Pop O(1), Peek O(1)
- Space Complexity: O(n)
- Use collections.deque for efficient stack

---

## 5. QUEUES

### Topics to Learn
- **Queue Basics**
  - FIFO (First In First Out) principle
  - Enqueue, Dequeue operations - O(1)
  - Implementation using array
  - Implementation using linked list
  - Queue using Python collections.deque

- **Types of Queues**
  - Simple Queue
  - Circular Queue
  - Deque (Double-ended queue)
  - Priority Queue

- **Queue Applications**
  - BFS (Breadth First Search)
  - Level order traversal
  - Job scheduling
  - Buffer management

- **Common Problems**
  - Queue using stacks
  - Stack using queues
  - First non-repeating character in stream
  - Sliding window maximum (using deque)
  - Generate binary numbers
  - Implement k queues in array
  - Reverse first k elements
  - Interleave first half with second half

### Key Concepts
- Time Complexity: Enqueue O(1), Dequeue O(1)
- Space Complexity: O(n)
- Use collections.deque for efficient queue

---

## 6. HASHING

### Topics to Learn
- **Hash Table Basics**
  - Hash function concept
  - Collision handling
    - Chaining (separate chaining)
    - Open addressing (linear probing, quadratic probing, double hashing)
  - Load factor
  - Dictionary in Python (dict)
  - Set in Python (set)

- **Common Applications**
  - Frequency counting
  - Fast lookups
  - Caching
  - Removing duplicates
  - Checking existence

- **Hash-based Problems**
  - Two Sum problem
  - Three Sum problem
  - Four Sum problem
  - Subarray sum equals k
  - Longest consecutive sequence
  - Group anagrams
  - Count distinct elements
  - First repeating element
  - Pairs with given sum
  - Check if arrays are equal
  - Find symmetric pairs

- **Advanced Hashing**
  - Rolling hash (Rabin-Karp)
  - Count distinct elements in window
  - Longest subarray with sum k
  - Count subarrays with given XOR

### Key Concepts
- Average Time Complexity: Insert O(1), Search O(1), Delete O(1)
- Worst Case: O(n) with poor hash function
- Space Complexity: O(n)
- Use Counter from collections for frequency

---

## 7. RECURSION

### Topics to Learn
- **Recursion Basics**
  - What is recursion?
  - Base case and recursive case
  - Call stack understanding
  - Stack overflow
  - Recursion vs Iteration

- **Types of Recursion**
  - Direct recursion
  - Indirect recursion
  - Tail recursion
  - Head recursion
  - Tree recursion

- **Basic Recursive Problems**
  - Factorial
  - Fibonacci sequence
  - Sum of N numbers
  - Power calculation
  - GCD (Euclidean algorithm)
  - Print N to 1 and 1 to N
  - Sum of digits

- **Intermediate Problems**
  - Tower of Hanoi
  - Generate all subsequences
  - Generate all subsets (power set)
  - Permutations
  - Combinations
  - Print all paths in matrix
  - Flood fill algorithm

- **Advanced Recursive Problems**
  - N-Queens problem
  - Sudoku solver
  - Maze problems
  - Word search in matrix
  - Generate valid parentheses
  - Letter combinations of phone number

### Key Concepts
- Time Complexity: varies by problem
- Space Complexity: O(n) for call stack
- Memoization to optimize recursion
- Recursion depth limit in Python (default 1000)

---

## 8. BACKTRACKING

### Topics to Learn
- **Backtracking Concept**
  - What is backtracking?
  - State space tree
  - Pruning
  - When to use backtracking

- **Classic Backtracking Problems**
  - N-Queens problem
  - Sudoku solver
  - Rat in a maze
  - Knight's tour problem
  - Hamiltonian path
  - Graph coloring

- **Combination & Permutation**
  - Generate all permutations
  - Generate all subsets
  - Combination sum
  - Combination sum II (with duplicates)
  - Subsets with duplicates
  - Permutations with duplicates

- **String Backtracking**
  - Word break problem
  - Palindrome partitioning
  - Letter case permutation
  - Generate valid parentheses
  - Restore IP addresses
  - Word search in grid

- **Advanced Backtracking**
  - M-coloring problem
  - Subset sum problem
  - Partition equal subset sum
  - Expression add operators
  - Remove invalid parentheses

### Key Concepts
- Time Complexity: Often exponential
- Space Complexity: O(n) for recursion stack
- Pruning reduces search space
- Choose-Explore-Unchoose pattern

---

## 9. TREES

### Binary Trees

#### Topics to Learn
- **Tree Basics**
  - Tree terminology (root, leaf, parent, child, sibling)
  - Types of binary trees (full, complete, perfect, balanced)
  - Tree representation

- **Tree Traversals**
  - Inorder (Left-Root-Right)
  - Preorder (Root-Left-Right)
  - Postorder (Left-Right-Root)
  - Level order (BFS)
  - Iterative traversals using stack
  - Morris traversal (no extra space)

- **Basic Tree Problems**
  - Height/depth of tree
  - Count nodes
  - Count leaf nodes
  - Count non-leaf nodes
  - Sum of all nodes
  - Check if tree is balanced
  - Check if two trees are identical

- **Intermediate Problems**
  - Diameter of tree
  - Mirror/symmetric tree
  - Lowest common ancestor (LCA)
  - Construct tree from traversals
  - Path sum problems
  - Maximum path sum
  - Print all root-to-leaf paths
  - Check if sum path exists

- **Advanced Tree Problems**
  - Vertical order traversal
  - Top view / Bottom view
  - Left view / Right view
  - Boundary traversal
  - Zigzag level order traversal
  - Diagonal traversal
  - Connect nodes at same level
  - Serialize and deserialize tree
  - Flatten tree to linked list

### Binary Search Trees (BST)

#### Topics to Learn
- **BST Basics**
  - BST property (left < root < right)
  - Search in BST - O(h)
  - Insert in BST - O(h)
  - Delete in BST - O(h)
  - Find min/max element

- **BST Problems**
  - Check if tree is BST
  - Inorder successor/predecessor
  - Kth smallest/largest element
  - Convert sorted array to BST
  - Two sum in BST
  - Merge two BSTs
  - BST to sorted linked list
  - Fix BST with two nodes swapped
  - Range sum query
  - Count BST nodes in range

### Advanced Trees

#### Topics to Learn
- **Self-Balancing Trees**
  - AVL Trees (height-balanced)
  - Red-Black Trees
  - Rotations (left, right, left-right, right-left)

- **Special Trees**
  - B-Trees & B+ Trees
  - Segment Trees (range queries)
  - Fenwick Tree / Binary Indexed Tree (BIT)
  - Trie (Prefix Tree)
  - Suffix Trees

### Key Concepts
- Height of balanced tree: O(log n)
- Height of skewed tree: O(n)
- Binary tree: max 2 children per node
- BST: inorder gives sorted order

---

## 10. HEAPS / PRIORITY QUEUE

### Topics to Learn
- **Heap Basics**
  - Min heap (parent < children)
  - Max heap (parent > children)
  - Complete binary tree property
  - Heap representation using array
  - Parent-child index relations

- **Heap Operations**
  - Insert (heappush) - O(log n)
  - Extract min/max (heappop) - O(log n)
  - Heapify - O(n)
  - Peek - O(1)
  - Decrease/Increase key - O(log n)

- **Python heapq Module**
  - heapq.heappush()
  - heapq.heappop()
  - heapq.heapify()
  - heapq.nlargest()
  - heapq.nsmallest()

- **Heap Problems**
  - Heap sort - O(n log n)
  - K largest/smallest elements
  - Kth largest/smallest element
  - Sort k-sorted array
  - Merge K sorted arrays
  - Merge K sorted lists
  - Top K frequent elements
  - Frequency sort
  - Find median in stream
  - Connect n ropes with minimum cost
  - Sliding window median
  - Task scheduler

- **Advanced Heap Problems**
  - Reorganize string
  - Minimum cost to hire K workers
  - Maximum CPU load
  - IPO problem
  - Find K pairs with smallest sums

### Key Concepts
- Time Complexity: Insert O(log n), Delete O(log n), Peek O(1)
- Space Complexity: O(n)
- Python's heapq is min heap by default
- For max heap, negate values

---

## 11. GRAPHS

### Graph Basics

#### Topics to Learn
- **Graph Representation**
  - Adjacency Matrix - O(V²) space
  - Adjacency List - O(V+E) space
  - Edge List
  - Weighted vs Unweighted
  - Directed vs Undirected

- **Graph Types**
  - Directed Acyclic Graph (DAG)
  - Cyclic graph
  - Weighted graph
  - Unweighted graph
  - Connected vs Disconnected
  - Complete graph
  - Bipartite graph

### Graph Traversal

#### Topics to Learn
- **Depth First Search (DFS)**
  - Recursive DFS
  - Iterative DFS (using stack)
  - Time: O(V+E), Space: O(V)
  - Applications: path finding, connectivity

- **Breadth First Search (BFS)**
  - Using queue
  - Time: O(V+E), Space: O(V)
  - Applications: shortest path in unweighted graph

- **Graph Traversal Problems**
  - Count connected components
  - Check if graph is connected
  - Find all paths between two nodes
  - Clone a graph
  - Course schedule (detect cycle in DAG)
  - Number of islands
  - Flood fill
  - Surrounded regions

### Cycle Detection

#### Topics to Learn
- Cycle detection in undirected graph (DFS/BFS)
- Cycle detection in directed graph (DFS with colors)
- Detect cycle using Union-Find
- Find the redundant connection

### Shortest Path Algorithms

#### Topics to Learn
- **Single Source Shortest Path**
  - Dijkstra's algorithm - O((V+E) log V)
  - Bellman-Ford algorithm - O(VE)
  - 0-1 BFS - O(V+E)
  - Shortest path in unweighted graph (BFS)

- **All Pairs Shortest Path**
  - Floyd-Warshall algorithm - O(V³)

- **Problems**
  - Network delay time
  - Cheapest flights within K stops
  - Path with minimum effort
  - Swim in rising water

### Minimum Spanning Tree

#### Topics to Learn
- **MST Algorithms**
  - Kruskal's algorithm - O(E log E)
  - Prim's algorithm - O(E log V)
  - Boruvka's algorithm

- **Union-Find / Disjoint Set**
  - Union by rank
  - Path compression
  - Applications in Kruskal's algorithm

- **Problems**
  - Connecting cities with minimum cost
  - Min cost to connect all points
  - Redundant connection

### Topological Sorting

#### Topics to Learn
- Topological sort using DFS
- Topological sort using BFS (Kahn's algorithm)
- Course schedule problems
- Alien dictionary
- Sequence reconstruction

### Advanced Graph Algorithms

#### Topics to Learn
- **Strongly Connected Components**
  - Kosaraju's algorithm
  - Tarjan's algorithm

- **Bridges and Articulation Points**
  - Critical connections in network
  - Bridges in graph

- **Special Graph Problems**
  - Eulerian path and circuit
  - Hamiltonian path and circuit
  - Bipartite graph check
  - Graph coloring
  - Traveling Salesman Problem (TSP)
  - Word ladder
  - Snakes and ladders

### Key Concepts
- V = vertices, E = edges
- DFS uses stack (or recursion)
- BFS uses queue
- Dijkstra doesn't work with negative weights
- Bellman-Ford works with negative weights

---

## 12. DYNAMIC PROGRAMMING

### DP Basics

#### Topics to Learn
- **Fundamentals**
  - What is Dynamic Programming?
  - Overlapping subproblems
  - Optimal substructure
  - Memoization (Top-down)
  - Tabulation (Bottom-up)
  - When to use DP

- **DP vs Recursion vs Greedy**
  - Differences and use cases
  - Time-space tradeoffs

### Classic DP Patterns

#### 1D DP
- Fibonacci sequence
- Climbing stairs
- House robber
- Decode ways
- Jump game
- Minimum cost climbing stairs
- Counting bits
- Perfect squares

#### 2D DP
- Unique paths in grid
- Minimum path sum
- Longest common subsequence (LCS)
- Edit distance
- Distinct subsequences
- Regular expression matching
- Wildcard matching
- Interleaving strings

#### Knapsack Problems
- 0/1 Knapsack
- Unbounded knapsack
- Subset sum problem
- Partition equal subset sum
- Partition to K equal sum subsets
- Target sum
- Count of subset sum
- Minimum subset sum difference

#### String DP
- Longest palindromic substring
- Longest palindromic subsequence
- Palindrome partitioning
- Word break problem
- Scrambled string
- Count palindromic substrings

#### Subsequence DP
- Longest Increasing Subsequence (LIS) - O(n²) and O(n log n)
- Longest Decreasing Subsequence
- Number of LIS
- Russian Doll Envelopes
- Maximum length of pair chain

#### Interval DP
- Matrix chain multiplication
- Burst balloons
- Minimum cost to merge stones
- Minimum score triangulation of polygon

#### Stock Problems
- Best time to buy and sell stock
- Best time to buy and sell stock II (multiple transactions)
- Best time to buy and sell stock III (2 transactions)
- Best time to buy and sell stock IV (k transactions)
- Best time to buy and sell stock with cooldown
- Best time to buy and sell stock with fee

#### Game Theory DP
- Stone game
- Predict the winner
- Can I win

### Advanced DP

#### Topics to Learn
- **DP on Trees**
  - Maximum path sum
  - Tree diameter
  - House robber III

- **DP on Graphs**
  - Longest path in DAG
  - All paths with target sum

- **Digit DP**
  - Count numbers with unique digits
  - Numbers at most N given digit set

- **Bitmask DP**
  - Traveling Salesman Problem
  - Partition to K equal sum subsets
  - Shortest path visiting all nodes

- **State Machine DP**
  - Stock problems with states
  - Maximum alternating subsequence sum

### Key Concepts
- Identify overlapping subproblems
- Define state and transitions
- Start with recursive solution
- Add memoization (top-down)
- Convert to iterative (bottom-up)
- Optimize space if possible

---

## 13. GREEDY ALGORITHMS

### Topics to Learn
- **Greedy Method Basics**
  - What is greedy approach?
  - Greedy choice property
  - Optimal substructure
  - When greedy works vs when it doesn't
  - Greedy vs DP

- **Classic Greedy Problems**
  - Activity selection problem
  - Fractional knapsack
  - Job sequencing with deadlines
  - Huffman coding
  - Minimum platforms required
  - Minimum number of coins
  - Egyptian fraction

- **Array-based Greedy**
  - Meeting rooms / intervals
  - Non-overlapping intervals
  - Minimum arrows to burst balloons
  - Remove K digits
  - Partition labels
  - Queue reconstruction by height
  - Gas station problem

- **Jump Problems**
  - Jump game I
  - Jump game II (minimum jumps)
  - Reach a number

- **Greedy on Sequences**
  - Candy distribution
  - Assign cookies
  - Boats to save people
  - Two city scheduling

- **Advanced Greedy**
  - Minimum deletions to make character frequencies unique
  - Maximum ice cream bars
  - Reorganize string
  - Task scheduler

### Key Concepts
- Makes locally optimal choice
- Doesn't always give global optimum
- Often involves sorting first
- Usually O(n log n) due to sorting

---

## 14. DIVIDE AND CONQUER

### Topics to Learn
- **Concept**
  - Divide problem into subproblems
  - Conquer subproblems recursively
  - Combine solutions

- **Classic Algorithms**
  - Binary search - O(log n)
  - Merge sort - O(n log n)
  - Quick sort - O(n log n) average
  - Closest pair of points - O(n log n)
  - Strassen's matrix multiplication
  - Karatsuba multiplication
  - Count inversions - O(n log n)

- **Problems**
  - Maximum subarray sum (divide & conquer approach)
  - Median of two sorted arrays
  - Kth largest element
  - Search in rotated sorted array
  - Find peak element
  - Power function (x^n) - O(log n)

### Key Concepts
- Recurrence relations
- Master theorem for time complexity
- Often results in O(n log n) or O(log n)

---

## 15. BIT MANIPULATION

### Topics to Learn
- **Bitwise Operators**
  - AND (&)
  - OR (|)
  - XOR (^)
  - NOT (~)
  - Left shift (<<)
  - Right shift (>>)

- **Basic Bit Operations**
  - Check if ith bit is set
  - Set ith bit
  - Clear ith bit
  - Toggle ith bit
  - Remove last set bit
  - Check if power of 2
  - Count set bits (Brian Kernighan's algorithm)

- **XOR Properties**
  - a ^ a = 0
  - a ^ 0 = a
  - XOR is commutative and associative
  - Find unique element (XOR all elements)

- **Common Problems**
  - Single number (find unique in array)
  - Single number II (one appears once, others thrice)
  - Single number III (two unique numbers)
  - Missing number
  - Power of two
  - Power of four
  - Number of 1 bits
  - Reverse bits
  - Hamming distance
  - Total Hamming distance

- **Advanced Bit Manipulation**
  - Subsets using bit masking
  - Power set generation
  - Counting bits (0 to n)
  - Bitwise AND of numbers range
  - Maximum XOR of two numbers
  - Divide two integers (using bits)

- **Bit Masking**
  - Represent subsets as integers
  - DP with bitmask
  - Traveling Salesman using bitmask

### Key Concepts
- Very fast operations
- Useful for optimization
- Often O(1) or O(log n) operations
- Common in interviews

---

## 16. MATHEMATICAL ALGORITHMS

### Topics to Learn
- **Prime Numbers**
  - Check if prime - O(√n)
  - Sieve of Eratosthenes - O(n log log n)
  - Segmented sieve
  - Count primes up to n
  - Prime factorization

- **GCD and LCM**
  - Euclidean algorithm - O(log min(a,b))
  - Extended Euclidean algorithm
  - LCM calculation
  - GCD of array

- **Modular Arithmetic**
  - (a + b) % m
  - (a * b) % m
  - Modular exponentiation - O(log n)
  - Modular inverse
  - Fermat's little theorem

- **Combinatorics**
  - Factorial
  - nCr (combinations)
  - nPr (permutations)
  - Pascal's triangle
  - Catalan numbers

- **Number Theory**
  - Fibonacci numbers
  - Fibonacci using matrix exponentiation - O(log n)
  - Euler's totient function
  - Chinese remainder theorem
  - Trailing zeros in factorial
  - Power of 2 in factorial

- **Mathematical Problems**
  - Happy number
  - Ugly number
  - Perfect squares
  - Count primes
  - Sqrt(x) - binary search approach
  - Pow(x, n)
  - Multiply strings
  - Add binary
  - Add strings

### Key Concepts
- Optimize using math properties
- Avoid overflow using modulo
- Fast exponentiation is key
- Prime numbers are fundamental

---

## 17. SEARCHING & SORTING

### Searching Algorithms

#### Topics to Learn
- **Linear Search** - O(n)
  - Simple iteration
  - Sentinel linear search

- **Binary Search** - O(log n)
  - Iterative binary search
  - Recursive binary search
  - Lower bound / Upper bound
  - Search in rotated sorted array
  - Find first/last occurrence
  - Find peak element
  - Search in 2D matrix
  - Find minimum in rotated sorted array
  - Median of two sorted arrays

- **Ternary Search** - O(log₃ n)
  - Used for unimodal functions

- **Exponential Search** - O(log n)
  - Unbounded binary search

- **Interpolation Search** - O(log log n) average
  - Better for uniformly distributed data

- **Binary Search on Answer**
  - Minimize maximum (or maximize minimum)
  - Allocate minimum pages
  - Aggressive cows
  - Split array largest sum
  - Koko eating bananas

### Sorting Algorithms

#### Topics to Learn
- **Comparison-based Sorts**
  - Bubble Sort - O(n²)
  - Selection Sort - O(n²)
  - Insertion Sort - O(n²)
  - Merge Sort - O(n log n)
  - Quick Sort - O(n log n) average, O(n²) worst
  - Heap Sort - O(n log n)

- **Non-comparison Sorts**
  - Counting Sort - O(n+k)
  - Radix Sort - O(d*(n+k))
  - Bucket Sort - O(n+k)

- **Hybrid Sorts**
  - Tim Sort (Python's default)
  - Intro Sort

- **Sorting Problems**
  - Sort colors (Dutch National Flag)
  - Sort array by parity
  - Sort array by increasing frequency
  - Custom sort string
  - Largest number
  - Merge intervals
  - Meeting rooms

### Key Concepts
- Binary search requires sorted array
- In-place vs out-of-place sorting
- Stable vs unstable sorting
- Comparison sorts can't beat O(n log n)

---

## 18. ADVANCED DATA STRUCTURES

### Topics to Learn
- **Trie (Prefix Tree)**
  - Insert, Search, Delete - O(L) where L is length
  - Prefix search
  - Word search problems
  - Autocomplete
  - Longest common prefix
  - Replace words
  - Word search II

- **Segment Tree**
  - Range query - O(log n)
  - Point update - O(log n)
  - Range update with lazy propagation
  - Sum, min, max, GCD queries

- **Fenwick Tree (Binary Indexed Tree)**
  - Range sum query - O(log n)
  - Point update - O(log n)
  - More space efficient than segment tree
  - Count inversions

- **Disjoint Set Union (Union-Find)**
  - Union by rank
  - Path compression
  - Find - O(α(n)) amortized (nearly O(1))
  - Applications: Kruskal's MST, connected components
  - Number of provinces
  - Redundant connection
  - Accounts merge

- **Suffix Array & Suffix Tree**
  - Pattern matching
  - Longest repeated substring
  - Longest common substring

- **Sparse Table**
  - Range min/max query - O(1)
  - Build - O(n log n)
  - Immutable arrays

- **Skip List**
  - Probabilistic data structure
  - Search, insert, delete - O(log n) expected

- **Bloom Filter**
  - Probabilistic membership testing
  - Space efficient
  - False positives possible

- **Cache Implementations**
  - LRU Cache (using OrderedDict or HashMap + DLL)
  - LFU Cache
  - Time-based cache

### Key Concepts
- Specialized structures for specific problems
- Often trade space for time
- Learn when to use each structure

---

## 19. ADVANCED TOPICS

### Topics to Learn
- **Sliding Window Maximum**
  - Using deque - O(n)
  - Monotonic deque

- **Two Pointers Advanced**
  - Multiple pointers
  - Partition variations

- **Meet in the Middle**
  - Reduce exponential to √exponential
  - Subset sum problems

- **Square Root Decomposition**
  - Divide array into √n blocks
  - Query and update in O(√n)

- **Mo's Algorithm**
  - Query ordering for range queries
  - O((n+q)√n)

- **Heavy-Light Decomposition**
  - Tree path queries

- **Centroid Decomposition**
  - Divide tree into smaller trees

- **Persistent Data Structures**
  - Maintain multiple versions
  - Persistent segment tree

- **Randomized Algorithms**
  - Reservoir sampling
  - Fisher-Yates shuffle
  - Monte Carlo methods

### Key Concepts
- These are competition-level topics
- Not always needed for interviews
- Learn after mastering fundamentals

---

## 20. PROBLEM-SOLVING PATTERNS

### Common Patterns to Recognize

#### Pattern 1: Frequency Counter
- Use hash map to count frequencies
- Avoid nested loops
- Example: Anagram problems, frequency-based problems

#### Pattern 2: Multiple Pointers
- Two pointers from different positions
- Reduces time complexity from O(n²) to O(n)
- Example: Two sum in sorted array, container with most water

#### Pattern 3: Sliding Window
- Create a window that slides through array
- Fixed or variable size
- Example: Maximum sum subarray, longest substring

#### Pattern 4: Divide and Conquer
- Break into smaller subproblems
- Solve recursively and combine
- Example: Merge sort, quick sort, binary search

#### Pattern 5: Dynamic Programming
- Break into overlapping subproblems
- Store results to avoid recomputation
- Example: Fibonacci, knapsack, LIS

#### Pattern 6: Greedy
- Make locally optimal choice
- Hope for global optimum
- Example: Activity selection, Huffman coding

#### Pattern 7: Backtracking
- Try all possibilities
- Undo if doesn't work
- Example: N-Queens, Sudoku, permutations

#### Pattern 8: Two Heaps
- One max heap, one min heap
- Example: Find median in stream

#### Pattern 9: Top K Elements
- Use heap to maintain K elements
- Example: K largest elements, K closest points

#### Pattern 10: K-way Merge
- Merge K sorted arrays/lists
- Use heap
- Example: Merge K sorted lists

#### Pattern 11: Modified Binary Search
- Binary search variations
- Example: Search in rotated array, find peak

#### Pattern 12: Cyclic Sort
- Sort array with numbers in range [1, n]
- Example: Find missing number, find duplicate

#### Pattern 13: Fast and Slow Pointers
- Two pointers at different speeds
- Example: Linked list cycle, find middle

#### Pattern 14: BFS/DFS
- Tree/graph traversal
- Level-wise vs depth-wise
- Example: Level order traversal, connected components

#### Pattern 15: Monotonic Stack/Queue
- Maintain increasing/decreasing order
- Example: Next greater element, sliding window maximum

---

## RECOMMENDED LEARNING PATH

### Phase 1: Fundamentals (4-6 weeks)
**Priority: MUST LEARN**
1. **Arrays** (1 week)
   - Basic operations, two pointer, sliding window
2. **Strings** (1 week)
   - String manipulation, pattern matching basics
3. **Hashing** (3-4 days)
   - Hash maps, sets, frequency counting
4. **Recursion** (1 week)
   - Basic recursion, tree recursion
5. **Sorting & Searching** (1 week)
   - All sorting algorithms, binary search

**Practice:** 50-70 easy problems

---

### Phase 2: Linear Data Structures (3-4 weeks)
**Priority: MUST LEARN**
6. **Linked Lists** (1 week)
   - All types, reversal, cycle detection
7. **Stacks** (1 week)
   - Implementation, expression problems
8. **Queues** (1 week)
   - All types, deque, priority queue basics

**Practice:** 40-50 easy-medium problems

---

### Phase 3: Non-Linear Data Structures (4-6 weeks)
**Priority: MUST LEARN**
9. **Trees** (2 weeks)
   - Binary trees, BST, all traversals
10. **Heaps** (1 week)
    - Min/max heap, heap problems
11. **Graphs** (2-3 weeks)
    - BFS, DFS, shortest path, MST

**Practice:** 50-70 medium problems

---

### Phase 4: Advanced Algorithms (6-8 weeks)
**Priority: HIGH**
12. **Dynamic Programming** (3-4 weeks)
    - 1D, 2D DP, classic patterns
13. **Greedy Algorithms** (1 week)
    - Activity selection, interval problems
14. **Backtracking** (1-2 weeks)
    - Permutations, combinations, N-Queens
15. **Advanced Graph Algorithms** (1-2 weeks)
    - Dijkstra, Bellman-Ford, topological sort

**Practice:** 70-100 medium-hard problems

---

### Phase 5: Specialized Topics (4-6 weeks)
**Priority: MEDIUM (Learn based on goals)**
16. **Bit Manipulation** (3-4 days)
17. **Mathematical Algorithms** (1 week)
18. **Advanced Data Structures** (2-3 weeks)
    - Trie, Segment Tree, Union-Find
19. **Problem-solving Patterns** (Ongoing)

**Practice:** 50-70 hard problems

---

### Phase 6: Competition Topics (Optional)
**Priority: LOW (Only for competitive programming)**
20. **Advanced Topics**
    - Mo's algorithm, Heavy-light decomposition

---

## TIME ESTIMATES

### Based on Study Hours per Day

**3-4 hours/day (Recommended for working professionals)**
- Phase 1: 4-6 weeks
- Phase 2: 3-4 weeks
- Phase 3: 4-6 weeks
- Phase 4: 6-8 weeks
- Phase 5: 4-6 weeks
- **Total: 5-7 months for interview readiness**

**6-8 hours/day (Full-time students / intensive prep)**
- Phase 1: 2-3 weeks
- Phase 2: 2 weeks
- Phase 3: 3-4 weeks
- Phase 4: 4-5 weeks
- Phase 5: 3-4 weeks
- **Total: 3-4 months for interview readiness**

**1-2 hours/day (Casual learning)**
- Phase 1: 8-10 weeks
- Phase 2: 6-8 weeks
- Phase 3: 8-10 weeks
- Phase 4: 10-12 weeks
- Phase 5: 8-10 weeks
- **Total: 10-12 months**

---

## PROBLEM PRACTICE GUIDELINES

### Difficulty Distribution
- **Easy:** 30-40% (build confidence, learn patterns)
- **Medium:** 50-60% (most interview questions)
- **Hard:** 10-20% (competitive companies, advanced rounds)

### Topic-wise Problem Count (Minimum)
- Arrays: 50 problems
- Strings: 30 problems
- Linked Lists: 25 problems
- Stacks & Queues: 30 problems
- Trees: 50 problems
- Graphs: 40 problems
- Dynamic Programming: 60 problems
- Others: 65 problems
- **Total: 350+ problems for strong preparation**

### Practice Platforms
1. **LeetCode** (most popular for interviews)
2. **GeeksforGeeks** (good explanations)
3. **HackerRank** (structured learning)
4. **Codeforces** (competitive programming)
5. **CodeChef** (contests)
6. **InterviewBit** (interview focused)

---

## RESOURCES

### Books
- "Cracking the Coding Interview" by Gayle Laakmann McDowell
- "Introduction to Algorithms" by CLRS
- "Algorithm Design Manual" by Steven Skiena
- "Elements of Programming Interviews in Python"

### Online Courses
- Coursera: Algorithms Specialization (Stanford)
- MIT OpenCourseWare: Introduction to Algorithms
- Abdul Bari (YouTube): Algorithms
- freeCodeCamp: Data Structures and Algorithms

### Python-Specific
- Python collections module documentation
- Python heapq module documentation
- Python bisect module documentation

---

## INTERVIEW PREPARATION CHECKLIST

### Technical Skills
- [ ] Solve 350+ problems across all topics
- [ ] Master time & space complexity analysis
- [ ] Practice on whiteboard / paper
- [ ] Learn to explain your approach clearly
- [ ] Practice with timer (45 min per problem)

### Problem-Solving Approach
1. **Clarify requirements** (5 min)
   - Ask questions
   - Understand input/output
   - Edge cases

2. **Think out loud** (10 min)
   - Discuss approach
   - Mention tradeoffs
   - Start with brute force

3. **Code** (20 min)
   - Write clean code
   - Use meaningful variable names
   - Add comments if helpful

4. **Test** (10 min)
   - Walk through example
   - Consider edge cases
   - Fix bugs

### Behavioral Preparation
- [ ] Prepare STAR stories
- [ ] Know your projects deeply
- [ ] Practice explaining technical concepts simply
- [ ] Research company culture

---

## TIPS FOR SUCCESS

### Study Tips
1. **Consistency over intensity** - 2 hours daily > 14 hours on Sunday
2. **Active learning** - Code everything yourself
3. **Understand don't memorize** - Know why solution works
4. **Review regularly** - Revisit problems after 1 week, 1 month
5. **Track progress** - Maintain spreadsheet of solved problems

### During Problem Solving
1. Start with brute force approach
2. Optimize step by step
3. Consider multiple approaches
4. Think about edge cases
5. Analyze time and space complexity
6. Test with examples

### Common Mistakes to Avoid
1. Jumping to code without thinking
2. Not asking clarifying questions
3. Ignoring edge cases
4. Poor variable naming
5. Not testing the code
6. Giving up too quickly
7. Not learning from mistakes

---

## FINAL WORDS

DSA is a marathon, not a sprint. Don't get discouraged by hard problems. Everyone struggles initially. The key is consistent practice and learning from mistakes.

**Remember:**
- Master fundamentals first (Arrays, Strings, Hashing, Recursion)
- Learn patterns, not individual problems
- Practice explaining your approach
- Time yourself during practice
- Review and revise regularly

**Good luck with your DSA journey! 🚀**

---

*Last Updated: November 2024*
