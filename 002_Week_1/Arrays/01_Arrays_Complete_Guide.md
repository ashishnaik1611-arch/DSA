# Arrays - Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Array Fundamentals](#array-fundamentals)
3. [Basic Operations](#basic-operations)
4. [Time & Space Complexity](#time--space-complexity)
5. [Array Traversal Techniques](#array-traversal-techniques)
6. [Two Pointer Technique](#two-pointer-technique)
7. [Sliding Window](#sliding-window)
8. [Prefix Sum](#prefix-sum)
9. [Sorting Algorithms](#sorting-algorithms)
10. [Searching Algorithms](#searching-algorithms)
11. [Important Patterns](#important-patterns)
12. [Common Problems](#common-problems)
13. [Practice Problems](#practice-problems)

---

## Introduction

**What is an Array?**
- An array is a collection of elements stored at contiguous memory locations
- In Python, we use **lists** which are dynamic arrays
- Arrays store elements of the same type (though Python lists can store mixed types)
- Elements can be accessed using an index (0-based indexing)

**Why Learn Arrays?**
- Most fundamental data structure
- Building block for other data structures
- Very common in interviews
- Many real-world applications

---

## Array Fundamentals

### Array in Memory
```
Array: [10, 20, 30, 40, 50]
Index:   0   1   2   3   4

Memory Addresses (example):
1000 -> 10
1004 -> 20
1008 -> 30
1012 -> 40
1016 -> 50
```

### Static vs Dynamic Arrays

**Static Arrays:**
- Fixed size at creation
- Cannot grow or shrink
- Languages: C, C++, Java

**Dynamic Arrays (Python Lists):**
- Can grow or shrink
- Automatically resize when needed
- Python lists are dynamic arrays

### Creating Arrays in Python

```python
# Method 1: Direct initialization
arr1 = [1, 2, 3, 4, 5]

# Method 2: Empty array
arr2 = []

# Method 3: Array with default values
arr3 = [0] * 5  # [0, 0, 0, 0, 0]

# Method 4: Using range
arr4 = list(range(1, 6))  # [1, 2, 3, 4, 5]

# Method 5: List comprehension
arr5 = [i**2 for i in range(5)]  # [0, 1, 4, 9, 16]

# 2D Array
matrix = [[0] * 3 for _ in range(3)]  # 3x3 matrix
```

---

## Basic Operations

### 1. Access
```python
arr = [10, 20, 30, 40, 50]

# Access by index
print(arr[0])    # 10
print(arr[2])    # 30
print(arr[-1])   # 50 (last element)
print(arr[-2])   # 40 (second last)
```
**Time Complexity:** O(1)

### 2. Update
```python
arr[2] = 100
print(arr)  # [10, 20, 100, 40, 50]
```
**Time Complexity:** O(1)

### 3. Insert

```python
arr = [10, 20, 30, 40, 50]

# Insert at end
arr.append(60)  # [10, 20, 30, 40, 50, 60]

# Insert at beginning
arr.insert(0, 5)  # [5, 10, 20, 30, 40, 50, 60]

# Insert at specific position
arr.insert(3, 25)  # [5, 10, 20, 25, 30, 40, 50, 60]

# Extend array (add multiple elements)
arr.extend([70, 80])  # [5, 10, 20, 25, 30, 40, 50, 60, 70, 80]
```

**Time Complexity:**
- `append()` at end: O(1) amortized
- `insert()` at beginning: O(n)
- `insert()` at position i: O(n)

### 4. Delete

```python
arr = [10, 20, 30, 40, 50]

# Remove from end
arr.pop()  # Returns 50, arr = [10, 20, 30, 40]

# Remove from beginning
arr.pop(0)  # Returns 10, arr = [20, 30, 40]

# Remove by value
arr.remove(30)  # arr = [20, 40]

# Remove by index
del arr[1]  # arr = [20]

# Clear entire array
arr.clear()  # arr = []
```

**Time Complexity:**
- `pop()` from end: O(1)
- `pop(0)` from beginning: O(n)
- `remove()`: O(n)
- `del arr[i]`: O(n)

### 5. Search

```python
arr = [10, 20, 30, 40, 50]

# Check if element exists
if 30 in arr:
    print("Found")

# Get index of element
index = arr.index(30)  # Returns 2

# Count occurrences
count = arr.count(30)  # Returns 1
```

**Time Complexity:** O(n)

---

## Time & Space Complexity

### Time Complexity Summary

| Operation | Time Complexity |
|-----------|----------------|
| Access by index | O(1) |
| Update by index | O(1) |
| Append at end | O(1) amortized |
| Insert at beginning | O(n) |
| Insert at position | O(n) |
| Delete from end | O(1) |
| Delete from beginning | O(n) |
| Delete from position | O(n) |
| Search | O(n) |
| Binary Search (sorted) | O(log n) |

### Space Complexity
- **Space Complexity:** O(n) where n is number of elements

---

## Array Traversal Techniques

### 1. Using Index
```python
arr = [10, 20, 30, 40, 50]

for i in range(len(arr)):
    print(f"Index {i}: {arr[i]}")
```

### 2. Direct Iteration
```python
for element in arr:
    print(element)
```

### 3. Using Enumerate
```python
for index, value in enumerate(arr):
    print(f"Index {index}: {value}")
```

### 4. Reverse Traversal
```python
# Method 1
for i in range(len(arr) - 1, -1, -1):
    print(arr[i])

# Method 2
for element in reversed(arr):
    print(element)
```

### 5. Using While Loop
```python
i = 0
while i < len(arr):
    print(arr[i])
    i += 1
```

---

## Two Pointer Technique

**When to use:**
- Array is sorted or can be sorted
- Need to find pairs with certain conditions
- Reverse, partition problems

### Pattern 1: Start and End Pointers

**Example: Reverse Array**
```python
def reverse_array(arr):
    left = 0
    right = len(arr) - 1

    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1

    return arr

# Example
arr = [1, 2, 3, 4, 5]
print(reverse_array(arr))  # [5, 4, 3, 2, 1]
```

**Example: Two Sum in Sorted Array**
```python
def two_sum_sorted(arr, target):
    left = 0
    right = len(arr) - 1

    while left < right:
        current_sum = arr[left] + arr[right]

        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1

    return []

# Example
arr = [1, 2, 3, 4, 6]
print(two_sum_sorted(arr, 7))  # [1, 4] (2 + 6 = 7)
```

### Pattern 2: Fast and Slow Pointers

**Example: Remove Duplicates from Sorted Array**
```python
def remove_duplicates(arr):
    if not arr:
        return 0

    slow = 0

    for fast in range(1, len(arr)):
        if arr[fast] != arr[slow]:
            slow += 1
            arr[slow] = arr[fast]

    return slow + 1

# Example
arr = [1, 1, 2, 2, 3, 4, 4]
length = remove_duplicates(arr)
print(arr[:length])  # [1, 2, 3, 4]
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## Sliding Window

**When to use:**
- Find subarrays with certain conditions
- Maximum/minimum in contiguous subarrays
- Problems involving "consecutive elements"

### Pattern 1: Fixed Size Window

**Example: Maximum Sum of Subarray of Size K**
```python
def max_sum_subarray(arr, k):
    if len(arr) < k:
        return None

    # Calculate sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum

    # Slide the window
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)

    return max_sum

# Example
arr = [1, 4, 2, 10, 23, 3, 1, 0, 20]
print(max_sum_subarray(arr, 4))  # 39 (10+23+3+1)
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

### Pattern 2: Variable Size Window

**Example: Smallest Subarray with Sum >= Target**
```python
def smallest_subarray_with_sum(arr, target):
    min_length = float('inf')
    window_sum = 0
    window_start = 0

    for window_end in range(len(arr)):
        window_sum += arr[window_end]

        while window_sum >= target:
            min_length = min(min_length, window_end - window_start + 1)
            window_sum -= arr[window_start]
            window_start += 1

    return 0 if min_length == float('inf') else min_length

# Example
arr = [2, 1, 5, 2, 3, 2]
print(smallest_subarray_with_sum(arr, 7))  # 2 (5+2)
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

---

## Prefix Sum

**When to use:**
- Range sum queries
- Subarray sum problems
- Multiple queries on same array

### Basic Prefix Sum

**Concept:**
```
Original:   [1, 2, 3, 4, 5]
Prefix Sum: [1, 3, 6, 10, 15]

Each element = sum of all elements from 0 to i
```

**Implementation:**
```python
def build_prefix_sum(arr):
    prefix = [0] * len(arr)
    prefix[0] = arr[0]

    for i in range(1, len(arr)):
        prefix[i] = prefix[i-1] + arr[i]

    return prefix

# Range sum query
def range_sum(prefix, left, right):
    if left == 0:
        return prefix[right]
    return prefix[right] - prefix[left - 1]

# Example
arr = [1, 2, 3, 4, 5]
prefix = build_prefix_sum(arr)
print(range_sum(prefix, 1, 3))  # 9 (2+3+4)
```

**Time Complexity:**
- Build: O(n)
- Query: O(1)

### Subarray Sum Equals K

```python
def subarray_sum_k(arr, k):
    count = 0
    prefix_sum = 0
    sum_freq = {0: 1}  # Important: initialize with 0:1

    for num in arr:
        prefix_sum += num

        # Check if (prefix_sum - k) exists
        if prefix_sum - k in sum_freq:
            count += sum_freq[prefix_sum - k]

        # Add current prefix_sum to map
        sum_freq[prefix_sum] = sum_freq.get(prefix_sum, 0) + 1

    return count

# Example
arr = [1, 1, 1]
print(subarray_sum_k(arr, 2))  # 2 ([1,1] at positions (0,1) and (1,2))
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

---

## Sorting Algorithms

### 1. Bubble Sort
**Idea:** Repeatedly swap adjacent elements if they're in wrong order

```python
def bubble_sort(arr):
    n = len(arr)

    for i in range(n):
        swapped = False

        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True

        if not swapped:
            break

    return arr
```

**Time Complexity:** O(n²)
**Space Complexity:** O(1)
**Stable:** Yes

### 2. Selection Sort
**Idea:** Find minimum element and place it at beginning

```python
def selection_sort(arr):
    n = len(arr)

    for i in range(n):
        min_idx = i

        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j

        arr[i], arr[min_idx] = arr[min_idx], arr[i]

    return arr
```

**Time Complexity:** O(n²)
**Space Complexity:** O(1)
**Stable:** No

### 3. Insertion Sort
**Idea:** Build sorted array one element at a time

```python
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1

        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1

        arr[j + 1] = key

    return arr
```

**Time Complexity:** O(n²)
**Space Complexity:** O(1)
**Stable:** Yes

### 4. Merge Sort
**Idea:** Divide array, sort halves, merge them

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])

    return result
```

**Time Complexity:** O(n log n)
**Space Complexity:** O(n)
**Stable:** Yes

### 5. Quick Sort
**Idea:** Pick pivot, partition array, recursively sort

```python
def quick_sort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1

    if low < high:
        pi = partition(arr, low, high)
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)

    return arr

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
```

**Time Complexity:** O(n log n) average, O(n²) worst
**Space Complexity:** O(log n)
**Stable:** No

### Python's Built-in Sort

```python
arr = [3, 1, 4, 1, 5, 9, 2, 6]

# Sort in place
arr.sort()
print(arr)  # [1, 1, 2, 3, 4, 5, 6, 9]

# Sort and return new array
arr2 = sorted(arr, reverse=True)
print(arr2)  # [9, 6, 5, 4, 3, 2, 1, 1]

# Custom sort key
arr3 = [(1, 2), (3, 1), (2, 4)]
arr3.sort(key=lambda x: x[1])
print(arr3)  # [(3, 1), (1, 2), (2, 4)]
```

---

## Searching Algorithms

### 1. Linear Search

```python
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

### 2. Binary Search

**Requirement:** Array must be sorted

**Iterative:**
```python
def binary_search(arr, target):
    left = 0
    right = len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```

**Recursive:**
```python
def binary_search_recursive(arr, target, left=0, right=None):
    if right is None:
        right = len(arr) - 1

    if left > right:
        return -1

    mid = left + (right - left) // 2

    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
```

**Time Complexity:** O(log n)
**Space Complexity:** O(1) iterative, O(log n) recursive

### Binary Search Variations

**Find First Occurrence:**
```python
def find_first(arr, target):
    left, right = 0, len(arr) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result
```

**Find Last Occurrence:**
```python
def find_last(arr, target):
    left, right = 0, len(arr) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            result = mid
            left = mid + 1  # Continue searching right
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result
```

---

## Important Patterns

### 1. Kadane's Algorithm (Maximum Subarray Sum)

```python
def max_subarray_sum(arr):
    max_sum = float('-inf')
    current_sum = 0

    for num in arr:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)

    return max_sum

# Example
arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
print(max_subarray_sum(arr))  # 6 ([4,-1,2,1])
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

### 2. Dutch National Flag (3-Way Partitioning)

**Problem:** Sort array of 0s, 1s, and 2s

```python
def sort_colors(arr):
    low = 0
    mid = 0
    high = len(arr) - 1

    while mid <= high:
        if arr[mid] == 0:
            arr[low], arr[mid] = arr[mid], arr[low]
            low += 1
            mid += 1
        elif arr[mid] == 1:
            mid += 1
        else:  # arr[mid] == 2
            arr[mid], arr[high] = arr[high], arr[mid]
            high -= 1

    return arr

# Example
arr = [2, 0, 2, 1, 1, 0]
print(sort_colors(arr))  # [0, 0, 1, 1, 2, 2]
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

### 3. Moore's Voting Algorithm (Majority Element)

**Problem:** Find element that appears more than n/2 times

```python
def majority_element(arr):
    candidate = None
    count = 0

    # Find candidate
    for num in arr:
        if count == 0:
            candidate = num
        count += 1 if num == candidate else -1

    # Verify candidate (optional, if majority is guaranteed)
    count = sum(1 for num in arr if num == candidate)
    return candidate if count > len(arr) // 2 else None

# Example
arr = [2, 2, 1, 1, 1, 2, 2]
print(majority_element(arr))  # 2
```

**Time Complexity:** O(n)
**Space Complexity:** O(1)

### 4. Next Greater Element

```python
def next_greater_element(arr):
    result = [-1] * len(arr)
    stack = []

    for i in range(len(arr) - 1, -1, -1):
        while stack and stack[-1] <= arr[i]:
            stack.pop()

        if stack:
            result[i] = stack[-1]

        stack.append(arr[i])

    return result

# Example
arr = [4, 5, 2, 25]
print(next_greater_element(arr))  # [5, 25, 25, -1]
```

**Time Complexity:** O(n)
**Space Complexity:** O(n)

---

## Common Problems

### Easy Level

1. **Find Maximum Element**
```python
def find_max(arr):
    return max(arr)  # O(n)
```

2. **Find Second Largest**
```python
def second_largest(arr):
    if len(arr) < 2:
        return None

    first = second = float('-inf')

    for num in arr:
        if num > first:
            second = first
            first = num
        elif num > second and num != first:
            second = num

    return second if second != float('-inf') else None
```

3. **Check if Sorted**
```python
def is_sorted(arr):
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            return False
    return True
```

4. **Rotate Array by K**
```python
def rotate(arr, k):
    k = k % len(arr)
    arr[:] = arr[-k:] + arr[:-k]
    return arr
```

### Medium Level

1. **Two Sum**
```python
def two_sum(arr, target):
    seen = {}

    for i, num in enumerate(arr):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i

    return []
```

2. **3Sum (Find triplets that sum to 0)**
```python
def three_sum(arr):
    arr.sort()
    result = []

    for i in range(len(arr) - 2):
        if i > 0 and arr[i] == arr[i - 1]:
            continue

        left = i + 1
        right = len(arr) - 1

        while left < right:
            total = arr[i] + arr[left] + arr[right]

            if total == 0:
                result.append([arr[i], arr[left], arr[right]])

                while left < right and arr[left] == arr[left + 1]:
                    left += 1
                while left < right and arr[right] == arr[right - 1]:
                    right -= 1

                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1

    return result
```

3. **Product of Array Except Self**
```python
def product_except_self(arr):
    n = len(arr)
    result = [1] * n

    # Left products
    left = 1
    for i in range(n):
        result[i] = left
        left *= arr[i]

    # Right products
    right = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right
        right *= arr[i]

    return result
```

### Hard Level

1. **Trapping Rain Water**
```python
def trap_rain_water(height):
    if not height:
        return 0

    left = 0
    right = len(height) - 1
    left_max = right_max = 0
    water = 0

    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1

    return water
```

2. **Median of Two Sorted Arrays**
```python
def find_median_sorted_arrays(arr1, arr2):
    if len(arr1) > len(arr2):
        arr1, arr2 = arr2, arr1

    m, n = len(arr1), len(arr2)
    left, right = 0, m

    while left <= right:
        partition1 = (left + right) // 2
        partition2 = (m + n + 1) // 2 - partition1

        maxLeft1 = float('-inf') if partition1 == 0 else arr1[partition1 - 1]
        minRight1 = float('inf') if partition1 == m else arr1[partition1]

        maxLeft2 = float('-inf') if partition2 == 0 else arr2[partition2 - 1]
        minRight2 = float('inf') if partition2 == n else arr2[partition2]

        if maxLeft1 <= minRight2 and maxLeft2 <= minRight1:
            if (m + n) % 2 == 0:
                return (max(maxLeft1, maxLeft2) + min(minRight1, minRight2)) / 2
            else:
                return max(maxLeft1, maxLeft2)
        elif maxLeft1 > minRight2:
            right = partition1 - 1
        else:
            left = partition1 + 1
```

---

## Practice Problems

### Beginner (Easy)

1. Find the largest element in an array
2. Find the second largest element
3. Check if array is sorted
4. Remove duplicates from sorted array
5. Move zeros to end
6. Find missing number (1 to n)
7. Maximum consecutive ones
8. Single number (find unique element)
9. Intersection of two arrays
10. Plus one (add 1 to number represented as array)

### Intermediate (Medium)

11. Two Sum
12. 3Sum
13. 4Sum
14. Container with most water
15. Product of array except self
16. Maximum subarray sum (Kadane's)
17. Best time to buy and sell stock
18. Rotate array
19. Find all duplicates in array
20. Subarray sum equals K
21. Longest consecutive sequence
22. Spiral matrix
23. Rotate image (matrix)
24. Set matrix zeroes
25. Search in rotated sorted array
26. Find peak element
27. Kth largest element
28. Top K frequent elements
29. Sort colors (Dutch flag)
30. Next permutation

### Advanced (Hard)

31. Trapping rain water
32. First missing positive
33. Longest increasing subsequence
34. Maximum product subarray
35. Median of two sorted arrays
36. Merge intervals
37. Insert interval
38. Jump game II
39. Sliding window maximum
40. Minimum window substring

---

## Tips for Array Problems

### General Approach
1. **Understand the problem**
   - Read carefully
   - Identify input/output
   - Ask clarifying questions

2. **Think of approaches**
   - Brute force first
   - Can we sort?
   - Can we use two pointers?
   - Is sliding window applicable?
   - Do we need extra space?

3. **Analyze complexity**
   - Time complexity
   - Space complexity
   - Can we optimize?

4. **Code carefully**
   - Handle edge cases
   - Use meaningful variable names
   - Test with examples

### Common Edge Cases
- Empty array: `[]`
- Single element: `[1]`
- Two elements: `[1, 2]`
- All same elements: `[1, 1, 1]`
- Negative numbers: `[-1, -2, -3]`
- Mixed positive/negative: `[-1, 0, 1]`
- Sorted array
- Reverse sorted array

### Pattern Recognition
- **See "consecutive"** → Think sliding window
- **See "sorted"** → Think binary search or two pointers
- **See "subarray sum"** → Think prefix sum
- **See "maximum/minimum"** → Think greedy or DP
- **See "pairs with sum"** → Think two pointers or hashing
- **See "frequency"** → Think hashing

---

## Python-Specific Tips

### List Methods
```python
arr = [1, 2, 3, 4, 5]

# Useful methods
arr.append(6)        # Add to end
arr.extend([7, 8])   # Add multiple
arr.insert(0, 0)     # Insert at position
arr.pop()            # Remove from end
arr.pop(0)           # Remove from start
arr.remove(3)        # Remove by value
arr.clear()          # Clear all
arr.reverse()        # Reverse in place
arr.sort()           # Sort in place
len(arr)             # Length
sum(arr)             # Sum of elements
min(arr)             # Minimum
max(arr)             # Maximum
arr.count(1)         # Count occurrences
arr.index(2)         # Find index
```

### List Slicing
```python
arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

arr[2:5]      # [2, 3, 4]
arr[:4]       # [0, 1, 2, 3]
arr[5:]       # [5, 6, 7, 8, 9]
arr[::2]      # [0, 2, 4, 6, 8]
arr[::-1]     # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
arr[-3:]      # [7, 8, 9]
arr[1:8:2]    # [1, 3, 5, 7]
```

### List Comprehensions
```python
# Create list
squares = [x**2 for x in range(10)]

# Filter
evens = [x for x in range(10) if x % 2 == 0]

# Transform
doubled = [x * 2 for x in [1, 2, 3]]

# Nested
matrix = [[i+j for j in range(3)] for i in range(3)]
```

### Important: Copying Arrays
```python
# WRONG - Reference copy
arr1 = [1, 2, 3]
arr2 = arr1
arr2.append(4)
print(arr1)  # [1, 2, 3, 4] - arr1 also changed!

# CORRECT - Shallow copy
arr3 = arr1.copy()  # or arr1[:]
arr3.append(5)
print(arr1)  # [1, 2, 3, 4] - arr1 unchanged

# Deep copy (for 2D arrays)
import copy
matrix1 = [[1, 2], [3, 4]]
matrix2 = copy.deepcopy(matrix1)
```

---

## Resources for Practice

### Online Platforms
1. **LeetCode** - Best for interview prep
2. **GeeksforGeeks** - Good explanations
3. **HackerRank** - Structured learning
4. **Codeforces** - Competitive programming

### Recommended Problem Sets
- LeetCode: Arrays Easy (50 problems)
- LeetCode: Arrays Medium (100 problems)
- GeeksforGeeks: Array Practice

### Study Plan
- **Week 1:** Basics, traversal, basic operations (10-15 problems)
- **Week 2:** Two pointers, sliding window (10-15 problems)
- **Week 3:** Sorting, searching (10-15 problems)
- **Week 4:** Advanced patterns (10-15 problems)

---

## Summary

### Must Know Concepts
✅ Array indexing and slicing
✅ Time complexity of operations
✅ Two pointer technique
✅ Sliding window
✅ Prefix sum
✅ Binary search
✅ Common sorting algorithms
✅ Important patterns (Kadane's, Dutch flag, etc.)

### Key Takeaways
- Arrays are the foundation of DSA
- Master the basics before moving to complex problems
- Practice pattern recognition
- Optimize space when possible
- Always analyze time and space complexity

---

**Next Steps:**
1. Complete beginner problems (1-10)
2. Learn and practice two pointers
3. Master sliding window
4. Move to medium problems
5. Study advanced patterns

Good luck with your array learning journey! 🚀

---

*Last Updated: November 2024*
