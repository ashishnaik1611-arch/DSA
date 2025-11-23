# Hashing - Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Hashing Fundamentals](#hashing-fundamentals)
3. [Hash Functions](#hash-functions)
4. [Collision Handling](#collision-handling)
5. [Python Hash Structures](#python-hash-structures)
6. [Important Hashing Patterns](#important-hashing-patterns)
7. [Common Problems](#common-problems)
8. [Practice Problems](#practice-problems)

---

## Introduction

**What is Hashing?**
- A technique to map data to a fixed-size array (hash table)
- Uses a hash function to compute an index (hash code)
- Enables fast data retrieval in O(1) average time
- Key-value pair storage

**Why Learn Hashing?**
- Fast lookups, insertions, and deletions
- Essential for frequency counting, caching, indexing
- Used in databases, compilers, cryptography
- Very common in interviews (25-30% of problems)

**Real-World Applications:**
- Databases indexing
- Caching (LRU Cache, Redis)
- Password storage
- Compiler symbol tables
- Spell checkers
- Network routers
- Blockchain

---

## Hashing Fundamentals

### How Hashing Works

```
Key → Hash Function → Hash Code → Index in Array → Value

Example:
"apple" → hash("apple") → 12345 → 12345 % 10 = 5 → array[5]

Hash Table:
Index  Value
  0    None
  1    None
  2    None
  3    ("banana", 2)
  4    None
  5    ("apple", 5)    ← Stored here
  6    None
  7    ("orange", 3)
  8    None
  9    None
```

### Basic Operations

| Operation | Average Case | Worst Case |
|-----------|--------------|------------|
| Insert | O(1) | O(n) |
| Search | O(1) | O(n) |
| Delete | O(1) | O(n) |
| Space | O(n) | O(n) |

**Average case:** Good hash function with few collisions
**Worst case:** All keys hash to same index (poor hash function)

---

## Hash Functions

### Properties of Good Hash Function

1. **Deterministic:** Same input always gives same output
2. **Uniform Distribution:** Keys distributed evenly across table
3. **Efficient:** Quick to compute
4. **Minimize Collisions:** Different keys should hash to different values

### Common Hash Functions

**1. Division Method**
```python
def hash_division(key, table_size):
    """
    h(key) = key % table_size
    Simple and fast
    """
    return key % table_size

# Example
print(hash_division(123, 10))  # 3
```

**2. Multiplication Method**
```python
def hash_multiplication(key, table_size):
    """
    h(key) = floor(table_size * (key * A % 1))
    A = constant (0 < A < 1), often 0.6180339887 (golden ratio)
    """
    A = 0.6180339887
    return int(table_size * ((key * A) % 1))
```

**3. String Hashing (Polynomial Rolling Hash)**
```python
def hash_string(s, table_size):
    """
    Hash string using polynomial method
    """
    hash_value = 0
    prime = 31  # or 37, 41, etc.

    for char in s:
        hash_value = (hash_value * prime + ord(char)) % table_size

    return hash_value

# Example
print(hash_string("hello", 100))
```

**4. Python's Built-in Hash**
```python
# Python has built-in hash function
print(hash("hello"))      # Integer hash
print(hash(123))          # Integer hash
print(hash((1, 2, 3)))    # Tuple hash

# Note: Lists and dicts are not hashable (mutable)
# print(hash([1, 2, 3]))  # TypeError
```

### Load Factor

```
Load Factor (α) = Number of elements / Table size

α = 0.5  → Table is 50% full
α = 0.75 → Table is 75% full (good balance)
α = 1.0  → Table is 100% full
α > 1.0  → More elements than slots (only with chaining)

When α exceeds threshold (usually 0.75), resize table
```

---

## Collision Handling

### What is a Collision?

When two different keys hash to the same index.

```
hash("apple") = 5
hash("banana") = 5   ← Collision!
```

### 1. Chaining (Separate Chaining)

Store multiple elements at same index using linked list or list.

```python
class HashTableChaining:
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(size)]

    def _hash(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        index = self._hash(key)

        # Update if key exists
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index][i] = (key, value)
                return

        # Add new key-value pair
        self.table[index].append((key, value))

    def search(self, key):
        index = self._hash(key)

        for k, v in self.table[index]:
            if k == key:
                return v

        return None

    def delete(self, key):
        index = self._hash(key)

        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                del self.table[index][i]
                return True

        return False

    def display(self):
        for i, bucket in enumerate(self.table):
            if bucket:
                print(f"{i}: {bucket}")

# Usage
ht = HashTableChaining()
ht.insert("apple", 5)
ht.insert("banana", 7)
ht.insert("orange", 3)
print(ht.search("apple"))  # 5
ht.display()
```

**Pros:**
- Simple to implement
- Table never fills up
- Less sensitive to hash function

**Cons:**
- Extra memory for links
- Cache performance may be poor

### 2. Open Addressing

All elements stored in table itself. When collision occurs, probe for next empty slot.

#### a. Linear Probing

```python
class HashTableLinearProbing:
    def __init__(self, size=10):
        self.size = size
        self.table = [None] * size
        self.count = 0

    def _hash(self, key, i=0):
        """Linear probing: (h(key) + i) % size"""
        return (hash(key) + i) % self.size

    def insert(self, key, value):
        if self.count >= self.size:
            raise Exception("Hash table is full")

        i = 0
        while i < self.size:
            index = self._hash(key, i)

            if self.table[index] is None or self.table[index] == "DELETED":
                self.table[index] = (key, value)
                self.count += 1
                return
            elif self.table[index][0] == key:
                self.table[index] = (key, value)  # Update
                return

            i += 1

        raise Exception("Hash table is full")

    def search(self, key):
        i = 0
        while i < self.size:
            index = self._hash(key, i)

            if self.table[index] is None:
                return None
            elif self.table[index] != "DELETED" and self.table[index][0] == key:
                return self.table[index][1]

            i += 1

        return None

    def delete(self, key):
        i = 0
        while i < self.size:
            index = self._hash(key, i)

            if self.table[index] is None:
                return False
            elif self.table[index] != "DELETED" and self.table[index][0] == key:
                self.table[index] = "DELETED"
                self.count -= 1
                return True

            i += 1

        return False
```

**Pros:**
- No extra memory for links
- Better cache performance

**Cons:**
- Table can fill up
- Primary clustering (consecutive occupied slots)

#### b. Quadratic Probing

```python
def _hash_quadratic(self, key, i):
    """Quadratic probing: (h(key) + i²) % size"""
    return (hash(key) + i * i) % self.size
```

**Pros:**
- Reduces primary clustering

**Cons:**
- Secondary clustering
- May not probe all slots

#### c. Double Hashing

```python
def _hash_double(self, key, i):
    """Double hashing: (h1(key) + i * h2(key)) % size"""
    h1 = hash(key) % self.size
    h2 = 1 + (hash(key) % (self.size - 1))
    return (h1 + i * h2) % self.size
```

**Pros:**
- Uniform probing
- No clustering

**Cons:**
- More complex
- Requires good second hash function

---

## Python Hash Structures

### 1. Dictionary (dict)

```python
# Create dictionary
d = {}
d = dict()
d = {"apple": 5, "banana": 7}

# Insert/Update
d["orange"] = 3
d["apple"] = 10  # Update

# Access
print(d["apple"])  # 5
print(d.get("grape"))  # None (no error)
print(d.get("grape", 0))  # 0 (default value)

# Delete
del d["apple"]
d.pop("banana")  # Returns value
d.pop("grape", None)  # No error if key doesn't exist

# Check existence
if "apple" in d:
    print("Found")

# Iterate
for key in d:
    print(key, d[key])

for key, value in d.items():
    print(key, value)

for key in d.keys():
    print(key)

for value in d.values():
    print(value)

# Other methods
d.clear()  # Remove all
len(d)     # Number of items
d.update({"x": 1, "y": 2})  # Add multiple
```

### 2. Set

```python
# Create set
s = set()
s = {1, 2, 3}
s = set([1, 2, 3, 2])  # {1, 2, 3} - duplicates removed

# Add
s.add(4)

# Remove
s.remove(2)  # Raises KeyError if not found
s.discard(2)  # No error if not found
s.pop()  # Remove and return arbitrary element

# Check existence
if 3 in s:
    print("Found")

# Set operations
a = {1, 2, 3}
b = {3, 4, 5}

print(a | b)  # Union: {1, 2, 3, 4, 5}
print(a & b)  # Intersection: {3}
print(a - b)  # Difference: {1, 2}
print(a ^ b)  # Symmetric difference: {1, 2, 4, 5}

# Methods
a.union(b)
a.intersection(b)
a.difference(b)
a.symmetric_difference(b)
a.issubset(b)
a.issuperset(b)
```

### 3. Counter (from collections)

```python
from collections import Counter

# Create counter
c = Counter()
c = Counter([1, 2, 2, 3, 3, 3])
c = Counter("hello")  # {'h': 1, 'e': 1, 'l': 2, 'o': 1}

# Count elements
text = "hello world"
c = Counter(text)
print(c['l'])  # 3
print(c['x'])  # 0 (no KeyError)

# Most common
print(c.most_common(2))  # [('l', 3), ('o', 2)]

# Operations
c1 = Counter("hello")
c2 = Counter("world")
print(c1 + c2)  # Combine
print(c1 - c2)  # Subtract (keep positive)
print(c1 & c2)  # Intersection (min)
print(c1 | c2)  # Union (max)

# Update
c.update("more text")
c.update({'a': 2, 'b': 3})

# Methods
c.elements()  # Iterator over elements
c.most_common(n)  # n most common elements
```

### 4. defaultdict (from collections)

```python
from collections import defaultdict

# Create with default factory
d = defaultdict(int)  # Default value 0
d = defaultdict(list)  # Default value []
d = defaultdict(set)  # Default value set()

# No KeyError
d = defaultdict(int)
d['apple'] += 1  # No need to initialize
print(d['banana'])  # 0 (default)

# Group by
from collections import defaultdict

def group_anagrams(words):
    groups = defaultdict(list)
    for word in words:
        key = ''.join(sorted(word))
        groups[key].append(word)
    return list(groups.values())

words = ["eat", "tea", "tan", "ate", "nat", "bat"]
print(group_anagrams(words))
# [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
```

### 5. OrderedDict (from collections)

```python
from collections import OrderedDict

# Maintains insertion order (dict in Python 3.7+ also maintains order)
od = OrderedDict()
od['a'] = 1
od['b'] = 2
od['c'] = 3

# Move to end
od.move_to_end('a')  # {'b': 2, 'c': 3, 'a': 1}

# Pop items
od.popitem(last=True)   # Remove last
od.popitem(last=False)  # Remove first
```

---

## Important Hashing Patterns

### 1. Frequency Counting

```python
def count_frequencies(arr):
    """Count frequency of each element"""
    freq = {}
    for num in arr:
        freq[num] = freq.get(num, 0) + 1
    return freq

# Or using Counter
from collections import Counter
freq = Counter(arr)
```

### 2. Two Sum Problem

```python
def two_sum(nums, target):
    """
    Find two numbers that add up to target
    Example: nums = [2, 7, 11, 15], target = 9
    Output: [0, 1] (2 + 7 = 9)
    """
    seen = {}

    for i, num in enumerate(nums):
        complement = target - num

        if complement in seen:
            return [seen[complement], i]

        seen[num] = i

    return []

# Time: O(n), Space: O(n)
```

### 3. Check if Array Contains Duplicates

```python
def contains_duplicate(nums):
    """Check if any value appears at least twice"""
    return len(nums) != len(set(nums))

# Or
def contains_duplicate(nums):
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False
```

### 4. First Unique Character

```python
def first_unique_char(s):
    """
    Find first non-repeating character
    Example: "leetcode" → 0 (index of 'l')
    """
    from collections import Counter
    count = Counter(s)

    for i, char in enumerate(s):
        if count[char] == 1:
            return i

    return -1
```

### 5. Anagram Check

```python
def is_anagram(s, t):
    """Check if two strings are anagrams"""
    # Method 1: Sort
    return sorted(s) == sorted(t)

def is_anagram(s, t):
    """Method 2: Count characters"""
    if len(s) != len(t):
        return False

    from collections import Counter
    return Counter(s) == Counter(t)
```

### 6. Group Anagrams

```python
def group_anagrams(strs):
    """
    Group strings that are anagrams
    Example: ["eat","tea","tan","ate","nat","bat"]
    Output: [["eat","tea","ate"],["tan","nat"],["bat"]]
    """
    from collections import defaultdict

    groups = defaultdict(list)

    for s in strs:
        # Use sorted string as key
        key = ''.join(sorted(s))
        groups[key].append(s)

    return list(groups.values())

# Time: O(n * k log k) where n = number of strings, k = max length
```

### 7. Longest Consecutive Sequence

```python
def longest_consecutive(nums):
    """
    Find length of longest consecutive sequence
    Example: [100, 4, 200, 1, 3, 2] → 4 (1,2,3,4)
    """
    if not nums:
        return 0

    num_set = set(nums)
    max_length = 0

    for num in num_set:
        # Only start sequence from smallest number
        if num - 1 not in num_set:
            current = num
            length = 1

            while current + 1 in num_set:
                current += 1
                length += 1

            max_length = max(max_length, length)

    return max_length

# Time: O(n), Space: O(n)
```

### 8. Subarray Sum Equals K

```python
def subarray_sum(nums, k):
    """
    Count subarrays with sum equal to k
    Example: nums = [1,1,1], k = 2 → 2
    """
    count = 0
    prefix_sum = 0
    sum_freq = {0: 1}  # Important: prefix sum 0 occurs once

    for num in nums:
        prefix_sum += num

        # Check if (prefix_sum - k) exists
        if prefix_sum - k in sum_freq:
            count += sum_freq[prefix_sum - k]

        # Add current prefix_sum to map
        sum_freq[prefix_sum] = sum_freq.get(prefix_sum, 0) + 1

    return count

# Time: O(n), Space: O(n)
```

### 9. LRU Cache

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return -1

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)

        self.cache[key] = value

        # Remove least recently used if over capacity
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

# Usage
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))  # 1
cache.put(3, 3)      # Evicts key 2
print(cache.get(2))  # -1 (not found)
```

### 10. Top K Frequent Elements

```python
def top_k_frequent(nums, k):
    """
    Find k most frequent elements
    Example: nums = [1,1,1,2,2,3], k = 2 → [1,2]
    """
    from collections import Counter
    import heapq

    # Count frequencies
    count = Counter(nums)

    # Use heap to find top k
    return heapq.nlargest(k, count.keys(), key=count.get)

# Or using bucket sort
def top_k_frequent(nums, k):
    from collections import Counter

    count = Counter(nums)
    buckets = [[] for _ in range(len(nums) + 1)]

    # Put numbers in buckets by frequency
    for num, freq in count.items():
        buckets[freq].append(num)

    # Collect top k
    result = []
    for i in range(len(buckets) - 1, 0, -1):
        result.extend(buckets[i])
        if len(result) >= k:
            return result[:k]

    return result

# Time: O(n), Space: O(n)
```

---

## Common Problems

### Easy Level

**1. Valid Anagram**
```python
def is_anagram(s, t):
    from collections import Counter
    return Counter(s) == Counter(t)
```

**2. Contains Duplicate**
```python
def contains_duplicate(nums):
    return len(nums) != len(set(nums))
```

**3. Single Number**
```python
def single_number(nums):
    """Every element appears twice except one"""
    # XOR method (without hashing)
    result = 0
    for num in nums:
        result ^= num
    return result

# Or using Counter
def single_number(nums):
    from collections import Counter
    count = Counter(nums)
    for num, freq in count.items():
        if freq == 1:
            return num
```

**4. Intersection of Two Arrays**
```python
def intersection(nums1, nums2):
    """Find unique common elements"""
    return list(set(nums1) & set(nums2))
```

**5. Happy Number**
```python
def is_happy(n):
    """
    Happy number: sum of squares of digits eventually equals 1
    """
    seen = set()

    while n != 1 and n not in seen:
        seen.add(n)
        n = sum(int(digit) ** 2 for digit in str(n))

    return n == 1
```

### Medium Level

**1. Longest Substring Without Repeating Characters**
```python
def length_of_longest_substring(s):
    """
    Find length of longest substring without repeating chars
    Example: "abcabcbb" → 3 ("abc")
    """
    char_index = {}
    max_length = 0
    start = 0

    for i, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1

        char_index[char] = i
        max_length = max(max_length, i - start + 1)

    return max_length
```

**2. 4Sum II**
```python
def four_sum_count(nums1, nums2, nums3, nums4):
    """
    Count tuples (i,j,k,l) where
    nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0
    """
    from collections import defaultdict

    sum_count = defaultdict(int)

    # Store all sums of first two arrays
    for a in nums1:
        for b in nums2:
            sum_count[a + b] += 1

    count = 0

    # Check if complement exists
    for c in nums3:
        for d in nums4:
            count += sum_count[-(c + d)]

    return count
```

**3. Contiguous Array**
```python
def find_max_length(nums):
    """
    Find maximum length of contiguous subarray
    with equal number of 0s and 1s
    """
    # Convert 0s to -1s, problem becomes subarray sum = 0
    sum_index = {0: -1}
    max_length = 0
    cumsum = 0

    for i, num in enumerate(nums):
        cumsum += 1 if num == 1 else -1

        if cumsum in sum_index:
            max_length = max(max_length, i - sum_index[cumsum])
        else:
            sum_index[cumsum] = i

    return max_length
```

**4. Brick Wall**
```python
def least_bricks(wall):
    """
    Find vertical line that crosses least bricks
    """
    from collections import defaultdict

    edge_count = defaultdict(int)

    for row in wall:
        edge_position = 0

        # Don't count last edge (end of wall)
        for i in range(len(row) - 1):
            edge_position += row[i]
            edge_count[edge_position] += 1

    # Maximum edges = minimum bricks crossed
    max_edges = max(edge_count.values()) if edge_count else 0
    return len(wall) - max_edges
```

### Hard Level

**1. Longest Consecutive Sequence (Already shown)**

**2. Substring with Concatenation of All Words**
```python
def find_substring(s, words):
    """
    Find starting indices of substrings that are
    concatenation of all words
    """
    if not s or not words:
        return []

    from collections import Counter

    word_len = len(words[0])
    word_count = len(words)
    total_len = word_len * word_count
    word_freq = Counter(words)
    result = []

    for i in range(len(s) - total_len + 1):
        seen = {}
        j = 0

        while j < word_count:
            word_start = i + j * word_len
            word = s[word_start:word_start + word_len]

            if word not in word_freq:
                break

            seen[word] = seen.get(word, 0) + 1

            if seen[word] > word_freq[word]:
                break

            j += 1

        if j == word_count:
            result.append(i)

    return result
```

**3. Longest Duplicate Substring**
```python
def longest_dup_substring(s):
    """
    Find longest duplicate substring using rolling hash
    """
    def search(L):
        """Check if there's duplicate substring of length L"""
        seen = set()
        base = 26
        mod = 2**63 - 1

        h = 0
        for i in range(L):
            h = (h * base + ord(s[i])) % mod

        seen.add(h)
        aL = pow(base, L, mod)

        for i in range(1, len(s) - L + 1):
            h = (h * base - ord(s[i-1]) * aL + ord(s[i+L-1])) % mod

            if h in seen:
                return i

            seen.add(h)

        return -1

    # Binary search on length
    left, right = 0, len(s)
    result_start = 0

    while left <= right:
        mid = (left + right) // 2
        start = search(mid)

        if start != -1:
            result_start = start
            left = mid + 1
        else:
            right = mid - 1

    return s[result_start:result_start + right]
```

---

## Practice Problems

### Beginner (Easy)

1. Two sum
2. Valid anagram
3. Contains duplicate
4. Contains duplicate II
5. Single number
6. Intersection of two arrays
7. Intersection of two arrays II
8. Happy number
9. Isomorphic strings
10. Word pattern
11. Ransom note
12. First unique character in string
13. Find the difference
14. Jewels and stones
15. Unique number of occurrences

### Intermediate (Medium)

16. Group anagrams
17. Top K frequent elements
18. Longest substring without repeating characters
19. Longest consecutive sequence
20. Subarray sum equals K
21. 4Sum II
22. Find duplicate subtrees
23. Contiguous array
24. Longest palindrome by concatenating two words
25. Maximum number of pairs in array
26. Sort characters by frequency
27. Design hashset
28. Design hashmap
29. Design Twitter
30. Insert delete getrandom O(1)
31. LRU cache
32. Encode and decode TinyURL
33. All O one data structure
34. Time based key-value store
35. Brick wall

### Advanced (Hard)

36. Substring with concatenation of all words
37. Longest duplicate substring
38. LFU cache
39. Contains duplicate III
40. Count of range sum

---

## Tips for Hashing Problems

### When to Use Hashing

✅ **Use hashing when you see:**
- "Find duplicate/unique"
- "Count frequency"
- "First/last occurrence"
- "Two sum / K sum"
- "Anagram"
- "Substring/subarray with condition"
- "Check existence in O(1)"
- "Group by some property"

### Pattern Recognition

- **"Two sum"** → Hash map to store complements
- **"Anagram"** → Count characters or sort
- **"Frequency"** → Counter or defaultdict
- **"Subarray sum"** → Prefix sum + hash map
- **"Longest substring"** → Sliding window + hash set
- **"Group by"** → defaultdict(list)
- **"LRU/LFU"** → OrderedDict or custom

### Common Techniques

**1. Hash Map for Complement:**
```python
# Two sum pattern
seen = {}
for i, num in enumerate(arr):
    complement = target - num
    if complement in seen:
        return [seen[complement], i]
    seen[num] = i
```

**2. Frequency Counter:**
```python
from collections import Counter
freq = Counter(arr)
```

**3. Prefix Sum with Hash:**
```python
prefix_sum = 0
sum_count = {0: 1}
for num in arr:
    prefix_sum += num
    # Check or update sum_count
```

**4. Sliding Window with Set:**
```python
window = set()
left = 0
for right in range(len(arr)):
    while arr[right] in window:
        window.remove(arr[left])
        left += 1
    window.add(arr[right])
```

### Common Mistakes

❌ Forgetting to handle duplicate keys
❌ Not initializing default values properly
❌ Using list as dictionary key (not hashable)
❌ Modifying dict while iterating

✅ Use dict.get(key, default)
✅ Use defaultdict for automatic defaults
✅ Convert list to tuple for hashing
✅ Create copy if modifying during iteration

---

## Performance Tips

### Do's
✅ Use built-in dict and set (highly optimized)
✅ Use Counter for frequency counting
✅ Use defaultdict to avoid KeyError
✅ Use set for membership testing
✅ Use in operator (O(1) for dict/set)

### Don'ts
❌ Don't use list for searching (O(n))
❌ Don't create unnecessary copies
❌ Don't use dict.keys() for iteration (just iterate dict)
❌ Don't forget dict is unordered in Python < 3.7

---

## Summary

### Must Know Concepts
✅ Hash function basics
✅ Collision handling (chaining, open addressing)
✅ Load factor and rehashing
✅ dict, set, Counter, defaultdict
✅ Two sum pattern
✅ Frequency counting
✅ Anagram problems
✅ Subarray sum with prefix sum + hash

### Key Takeaways
- Hashing provides O(1) average time operations
- Python's dict and set are hash-based
- Perfect for lookups, counting, grouping
- Use Counter for frequency problems
- Use defaultdict to avoid KeyError
- Very common in interviews (25-30% of problems)

### Time Complexity Cheat Sheet
- Insert: O(1) average, O(n) worst
- Search: O(1) average, O(n) worst
- Delete: O(1) average, O(n) worst
- Space: O(n)

---

**Next Steps:**
1. Practice two sum variations
2. Master frequency counting problems
3. Learn anagram patterns
4. Solve subarray sum problems
5. Implement LRU cache
6. Practice 20-30 medium problems

Good luck with your hashing learning journey! 🚀

---

*Last Updated: November 2024*
