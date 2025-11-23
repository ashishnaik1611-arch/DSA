# Strings - Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [String Fundamentals](#string-fundamentals)
3. [String Operations](#string-operations)
4. [String Methods](#string-methods)
5. [String Manipulation Techniques](#string-manipulation-techniques)
6. [Pattern Matching Algorithms](#pattern-matching-algorithms)
7. [Important String Patterns](#important-string-patterns)
8. [Common String Problems](#common-string-problems)
9. [Practice Problems](#practice-problems)

---

## Introduction

**What is a String?**
- A string is a sequence of characters
- In Python, strings are immutable (cannot be changed after creation)
- Stored as array of characters internally
- Can contain letters, digits, symbols, spaces

**Why Learn Strings?**
- Very common in interviews (20-30% of problems)
- Used in text processing, parsing, validation
- Foundation for pattern matching and algorithms
- Real-world applications: search engines, DNA sequencing, text editors

---

## String Fundamentals

### String Creation

```python
# Method 1: Single quotes
str1 = 'Hello'

# Method 2: Double quotes
str2 = "World"

# Method 3: Triple quotes (multiline)
str3 = """This is a
multiline string"""

# Method 4: From list of characters
str4 = ''.join(['H', 'e', 'l', 'l', 'o'])

# Method 5: String repetition
str5 = 'Ha' * 3  # 'HaHaHa'

# Method 6: Empty string
str6 = ''
```

### String Immutability

```python
# Strings are IMMUTABLE in Python
s = "Hello"
# s[0] = 'h'  # This will cause ERROR!

# To modify, create new string
s = 'h' + s[1:]  # 'hello'

# Or convert to list
chars = list(s)
chars[0] = 'H'
s = ''.join(chars)
```

### String Indexing

```python
s = "Python"
#     012345  (positive indexing)
#    -654321  (negative indexing)

print(s[0])    # 'P'
print(s[5])    # 'n'
print(s[-1])   # 'n' (last character)
print(s[-2])   # 'o' (second last)
```

### String Slicing

```python
s = "Programming"

print(s[0:4])     # 'Prog'
print(s[:4])      # 'Prog' (start from beginning)
print(s[4:])      # 'ramming' (till end)
print(s[::2])     # 'Poamn' (every 2nd character)
print(s[::-1])    # 'gnimmargorP' (reverse)
print(s[2:8:2])   # 'orm' (from 2 to 8, step 2)
```

---

## String Operations

### Time Complexity of Operations

| Operation | Time Complexity |
|-----------|----------------|
| Access by index | O(1) |
| Concatenation (+) | O(n+m) |
| Length | O(1) |
| Slicing | O(k) where k is slice length |
| Search (in) | O(n) |
| Replace | O(n) |
| Split | O(n) |
| Join | O(n) |
| Upper/Lower | O(n) |

### Basic Operations

```python
# Concatenation
s1 = "Hello"
s2 = "World"
s3 = s1 + " " + s2  # "Hello World"

# Repetition
s = "Ha" * 3  # "HaHaHa"

# Length
length = len(s)  # 6

# Membership
print('H' in s)      # True
print('X' in s)      # False
print('Ha' in s)     # True

# Comparison
print("apple" < "banana")   # True (lexicographic)
print("abc" == "abc")       # True
```

---

## String Methods

### Case Conversion

```python
s = "Hello World"

s.upper()       # "HELLO WORLD"
s.lower()       # "hello world"
s.capitalize()  # "Hello world"
s.title()       # "Hello World"
s.swapcase()    # "hELLO wORLD"
```

### Checking Methods

```python
s = "Hello123"

s.isalpha()      # False (has digits)
s.isdigit()      # False (has letters)
s.isalnum()      # True (alphanumeric)
s.isspace()      # False
s.isupper()      # False
s.islower()      # False
s.startswith('H') # True
s.endswith('3')   # True
```

### Search Methods

```python
s = "Hello World Hello"

# Find (returns -1 if not found)
s.find('World')      # 6
s.find('Python')     # -1

# Index (raises error if not found)
s.index('World')     # 6

# Count occurrences
s.count('Hello')     # 2
s.count('l')         # 5
```

### Modification Methods

```python
s = "  Hello World  "

# Strip whitespace
s.strip()       # "Hello World"
s.lstrip()      # "Hello World  "
s.rstrip()      # "  Hello World"

# Replace
s.replace('World', 'Python')  # "  Hello Python  "

# Split
s = "apple,banana,orange"
fruits = s.split(',')  # ['apple', 'banana', 'orange']

s = "Hello World"
words = s.split()      # ['Hello', 'World'] (splits on whitespace)

# Join
words = ['Hello', 'World']
s = ' '.join(words)    # "Hello World"
s = '-'.join(words)    # "Hello-World"
```

### Other Useful Methods

```python
# Center, Left, Right justify
s = "Hello"
s.center(10)      # "  Hello   "
s.ljust(10)       # "Hello     "
s.rjust(10)       # "     Hello"

# Zero fill
s = "42"
s.zfill(5)        # "00042"

# Partition
s = "Hello-World"
s.partition('-')  # ('Hello', '-', 'World')
```

---

## String Manipulation Techniques

### 1. Character Frequency Count

```python
# Method 1: Using dictionary
def count_freq(s):
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    return freq

# Method 2: Using Counter
from collections import Counter
s = "hello"
freq = Counter(s)  # Counter({'l': 2, 'h': 1, 'e': 1, 'o': 1})
```

### 2. String Reversal

```python
# Method 1: Slicing (fastest)
s = "Hello"
reversed_s = s[::-1]  # "olleH"

# Method 2: reversed() function
reversed_s = ''.join(reversed(s))

# Method 3: Loop
def reverse_string(s):
    chars = list(s)
    left, right = 0, len(chars) - 1
    while left < right:
        chars[left], chars[right] = chars[right], chars[left]
        left += 1
        right -= 1
    return ''.join(chars)
```

### 3. Palindrome Check

```python
def is_palindrome(s):
    # Method 1: Simple
    return s == s[::-1]

def is_palindrome_alphanumeric(s):
    # Method 2: Ignore non-alphanumeric and case
    s = ''.join(c.lower() for c in s if c.isalnum())
    return s == s[::-1]

# Two pointer approach
def is_palindrome_two_pointer(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
```

### 4. Anagram Check

```python
def are_anagrams(s1, s2):
    # Method 1: Sort
    return sorted(s1) == sorted(s2)

def are_anagrams_freq(s1, s2):
    # Method 2: Frequency count
    if len(s1) != len(s2):
        return False

    from collections import Counter
    return Counter(s1) == Counter(s2)
```

### 5. Remove Duplicates

```python
def remove_duplicates(s):
    # Preserving order
    seen = set()
    result = []
    for char in s:
        if char not in seen:
            result.append(char)
            seen.add(char)
    return ''.join(result)

# Example
s = "programming"
print(remove_duplicates(s))  # "progamin"
```

### 6. String Compression

```python
def compress_string(s):
    """
    Compress string using count of repeated characters.
    Example: "aaabbc" -> "a3b2c1"
    """
    if not s:
        return ""

    result = []
    count = 1

    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            count += 1
        else:
            result.append(s[i-1] + str(count))
            count = 1

    result.append(s[-1] + str(count))

    compressed = ''.join(result)
    return compressed if len(compressed) < len(s) else s
```

---

## Pattern Matching Algorithms

### 1. Naive Pattern Matching

```python
def naive_search(text, pattern):
    """
    Simple pattern matching algorithm.
    Time: O(n*m) where n=len(text), m=len(pattern)
    """
    n = len(text)
    m = len(pattern)
    indices = []

    for i in range(n - m + 1):
        j = 0
        while j < m and text[i + j] == pattern[j]:
            j += 1

        if j == m:
            indices.append(i)

    return indices

# Example
text = "AABAACAADAABAABA"
pattern = "AABA"
print(naive_search(text, pattern))  # [0, 9, 12]
```

**Time Complexity:** O(n*m)
**Space Complexity:** O(1)

### 2. KMP Algorithm (Knuth-Morris-Pratt)

```python
def build_lps(pattern):
    """
    Build Longest Prefix Suffix array.
    LPS[i] = length of longest proper prefix which is also suffix
    """
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1

    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1

    return lps

def kmp_search(text, pattern):
    """
    KMP pattern matching algorithm.
    Time: O(n+m) where n=len(text), m=len(pattern)
    """
    n = len(text)
    m = len(pattern)

    lps = build_lps(pattern)
    indices = []

    i = 0  # index for text
    j = 0  # index for pattern

    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1

        if j == m:
            indices.append(i - j)
            j = lps[j - 1]
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return indices

# Example
text = "AABAACAADAABAABA"
pattern = "AABA"
print(kmp_search(text, pattern))  # [0, 9, 12]
```

**Time Complexity:** O(n + m)
**Space Complexity:** O(m)

### 3. Rabin-Karp Algorithm (Rolling Hash)

```python
def rabin_karp(text, pattern):
    """
    Rolling hash based pattern matching.
    Time: O(n+m) average, O(n*m) worst
    """
    d = 256  # number of characters in alphabet
    q = 101  # prime number

    n = len(text)
    m = len(pattern)

    h = pow(d, m - 1) % q
    p = 0  # hash value for pattern
    t = 0  # hash value for text
    indices = []

    # Calculate initial hash values
    for i in range(m):
        p = (d * p + ord(pattern[i])) % q
        t = (d * t + ord(text[i])) % q

    # Slide pattern over text
    for i in range(n - m + 1):
        if p == t:
            # Check characters one by one
            if text[i:i+m] == pattern:
                indices.append(i)

        # Calculate hash for next window
        if i < n - m:
            t = (d * (t - ord(text[i]) * h) + ord(text[i + m])) % q
            if t < 0:
                t += q

    return indices
```

**Time Complexity:** O(n + m) average
**Space Complexity:** O(1)

---

## Important String Patterns

### 1. Two Pointer Technique

**Valid Palindrome:**
```python
def is_palindrome(s):
    left, right = 0, len(s) - 1

    while left < right:
        # Skip non-alphanumeric characters
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1

        if s[left].lower() != s[right].lower():
            return False

        left += 1
        right -= 1

    return True

# Example
print(is_palindrome("A man, a plan, a canal: Panama"))  # True
```

### 2. Sliding Window

**Longest Substring Without Repeating Characters:**
```python
def length_of_longest_substring(s):
    char_set = set()
    left = 0
    max_length = 0

    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1

        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)

    return max_length

# Example
print(length_of_longest_substring("abcabcbb"))  # 3 (abc)
```

**Longest Substring with K Distinct Characters:**
```python
def longest_substring_k_distinct(s, k):
    if k == 0:
        return 0

    char_count = {}
    left = 0
    max_length = 0

    for right in range(len(s)):
        char_count[s[right]] = char_count.get(s[right], 0) + 1

        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1

        max_length = max(max_length, right - left + 1)

    return max_length
```

### 3. String Building (Use List)

```python
# INEFFICIENT - O(n²) due to string immutability
def build_string_bad(n):
    result = ""
    for i in range(n):
        result += str(i)  # Creates new string each time!
    return result

# EFFICIENT - O(n)
def build_string_good(n):
    result = []
    for i in range(n):
        result.append(str(i))
    return ''.join(result)
```

### 4. Character Mapping

**Isomorphic Strings:**
```python
def is_isomorphic(s, t):
    if len(s) != len(t):
        return False

    map_s_to_t = {}
    map_t_to_s = {}

    for c1, c2 in zip(s, t):
        if c1 in map_s_to_t:
            if map_s_to_t[c1] != c2:
                return False
        else:
            map_s_to_t[c1] = c2

        if c2 in map_t_to_s:
            if map_t_to_s[c2] != c1:
                return False
        else:
            map_t_to_s[c2] = c1

    return True

# Example
print(is_isomorphic("egg", "add"))  # True
print(is_isomorphic("foo", "bar"))  # False
```

### 5. Subsequence vs Substring

```python
# Check if s is subsequence of t
def is_subsequence(s, t):
    i = 0
    for char in t:
        if i < len(s) and char == s[i]:
            i += 1
    return i == len(s)

# Check if s is substring of t
def is_substring(s, t):
    return s in t  # O(n*m)
```

---

## Common String Problems

### Easy Level

**1. Reverse String**
```python
def reverse_string(s):
    return s[::-1]
```

**2. First Unique Character**
```python
def first_unique_char(s):
    from collections import Counter
    count = Counter(s)

    for i, char in enumerate(s):
        if count[char] == 1:
            return i

    return -1
```

**3. Valid Anagram**
```python
def is_anagram(s, t):
    return sorted(s) == sorted(t)
```

**4. Longest Common Prefix**
```python
def longest_common_prefix(strs):
    if not strs:
        return ""

    prefix = strs[0]

    for string in strs[1:]:
        while not string.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""

    return prefix

# Example
strs = ["flower", "flow", "flight"]
print(longest_common_prefix(strs))  # "fl"
```

### Medium Level

**1. Longest Palindromic Substring**
```python
def longest_palindrome(s):
    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1

    if not s:
        return ""

    start = 0
    max_len = 0

    for i in range(len(s)):
        # Odd length palindrome
        len1 = expand_around_center(i, i)
        # Even length palindrome
        len2 = expand_around_center(i, i + 1)

        length = max(len1, len2)

        if length > max_len:
            max_len = length
            start = i - (length - 1) // 2

    return s[start:start + max_len]
```

**2. Group Anagrams**
```python
def group_anagrams(strs):
    from collections import defaultdict
    anagrams = defaultdict(list)

    for s in strs:
        key = ''.join(sorted(s))
        anagrams[key].append(s)

    return list(anagrams.values())

# Example
strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
print(group_anagrams(strs))
# [["eat","tea","ate"], ["tan","nat"], ["bat"]]
```

**3. String to Integer (atoi)**
```python
def my_atoi(s):
    s = s.lstrip()

    if not s:
        return 0

    sign = 1
    i = 0

    if s[0] in ['+', '-']:
        sign = -1 if s[0] == '-' else 1
        i = 1

    result = 0

    while i < len(s) and s[i].isdigit():
        result = result * 10 + int(s[i])
        i += 1

    result *= sign

    # Clamp to 32-bit integer range
    INT_MAX = 2**31 - 1
    INT_MIN = -2**31

    if result > INT_MAX:
        return INT_MAX
    if result < INT_MIN:
        return INT_MIN

    return result
```

**4. Zigzag Conversion**
```python
def convert(s, num_rows):
    if num_rows == 1 or num_rows >= len(s):
        return s

    rows = [''] * num_rows
    current_row = 0
    going_down = False

    for char in s:
        rows[current_row] += char

        if current_row == 0 or current_row == num_rows - 1:
            going_down = not going_down

        current_row += 1 if going_down else -1

    return ''.join(rows)
```

### Hard Level

**1. Minimum Window Substring**
```python
def min_window(s, t):
    from collections import Counter

    if not s or not t:
        return ""

    dict_t = Counter(t)
    required = len(dict_t)

    left = 0
    formed = 0
    window_counts = {}

    ans = float("inf"), None, None

    for right in range(len(s)):
        char = s[right]
        window_counts[char] = window_counts.get(char, 0) + 1

        if char in dict_t and window_counts[char] == dict_t[char]:
            formed += 1

        while left <= right and formed == required:
            char = s[left]

            if right - left + 1 < ans[0]:
                ans = (right - left + 1, left, right)

            window_counts[char] -= 1
            if char in dict_t and window_counts[char] < dict_t[char]:
                formed -= 1

            left += 1

    return "" if ans[0] == float("inf") else s[ans[1]:ans[2] + 1]
```

**2. Regular Expression Matching**
```python
def is_match(s, p):
    """
    '.' Matches any single character
    '*' Matches zero or more of the preceding element
    """
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]

    dp[0][0] = True

    # Handle patterns like a*, a*b*, a*b*c*
    for j in range(2, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 2]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == s[i - 1] or p[j - 1] == '.':
                dp[i][j] = dp[i - 1][j - 1]
            elif p[j - 1] == '*':
                dp[i][j] = dp[i][j - 2]  # Zero occurrence
                if p[j - 2] == s[i - 1] or p[j - 2] == '.':
                    dp[i][j] = dp[i][j] or dp[i - 1][j]  # One or more

    return dp[m][n]
```

---

## Practice Problems

### Beginner (Easy)

1. Reverse a string
2. Check if palindrome
3. Valid anagram
4. First unique character
5. Implement strStr() (find substring)
6. Longest common prefix
7. Valid parentheses
8. Count and say
9. Add binary
10. Length of last word

### Intermediate (Medium)

11. Longest substring without repeating characters
12. Longest palindromic substring
13. Group anagrams
14. Generate parentheses
15. String to integer (atoi)
16. Zigzag conversion
17. Letter combinations of phone number
18. Reverse words in a string
19. Simplify path
20. Multiply strings
21. Compare version numbers
22. Decode string
23. Encode and decode strings
24. Palindromic substrings
25. Longest repeating character replacement
26. Permutation in string
27. Find all anagrams in string
28. Word pattern
29. Isomorphic strings
30. Valid palindrome II

### Advanced (Hard)

31. Minimum window substring
32. Regular expression matching
33. Wildcard matching
34. Edit distance
35. Distinct subsequences
36. Scramble string
37. Interleaving string
38. Text justification
39. Word break II
40. Palindrome partitioning II

---

## Tips for String Problems

### General Approach

1. **Understand requirements**
   - Case sensitive or not?
   - Special characters handling?
   - Empty string behavior?
   - Unicode or ASCII only?

2. **Choose right technique**
   - Two pointers for palindrome/reversal
   - Sliding window for substrings
   - HashMap for frequency/anagram
   - Stack for parentheses/nesting
   - DP for edit distance/matching

3. **Optimize space**
   - Strings are immutable in Python
   - Use list for multiple modifications
   - Join at the end

4. **Handle edge cases**
   - Empty string
   - Single character
   - All same characters
   - Special characters
   - Very long strings

### Common Patterns

- **"Palindrome"** → Two pointers
- **"Anagram"** → Frequency count or sort
- **"Substring"** → Sliding window
- **"Subsequence"** → Two pointers or DP
- **"Pattern matching"** → KMP or Rabin-Karp
- **"Parentheses"** → Stack
- **"Edit distance"** → Dynamic Programming

### Python String Tricks

```python
# Check if all characters are unique
def is_unique(s):
    return len(s) == len(set(s))

# Remove all whitespace
s = "Hello World"
no_space = s.replace(" ", "")
no_space = ''.join(s.split())

# Check if string contains only letters
s.isalpha()

# Convert to list for modifications
chars = list(s)
chars[0] = 'X'
s = ''.join(chars)

# ASCII value
ord('A')  # 65
chr(65)   # 'A'

# String formatting
name = "Alice"
age = 25
f"Name: {name}, Age: {age}"
"Name: {}, Age: {}".format(name, age)
```

---

## Important String Concepts

### Character Sets
- **ASCII:** 128 characters (0-127)
- **Extended ASCII:** 256 characters (0-255)
- **Unicode:** Millions of characters

### String Encoding
```python
# Encode string to bytes
s = "Hello"
encoded = s.encode('utf-8')  # b'Hello'

# Decode bytes to string
decoded = encoded.decode('utf-8')  # "Hello"
```

### Regular Expressions (Basics)
```python
import re

# Match pattern
pattern = r'\d+'  # one or more digits
text = "I have 2 apples and 3 oranges"
matches = re.findall(pattern, text)  # ['2', '3']

# Replace pattern
result = re.sub(r'\d+', 'X', text)
# "I have X apples and X oranges"

# Split by pattern
parts = re.split(r'\d+', text)
# ['I have ', ' apples and ', ' oranges']
```

---

## Performance Tips

### Do's
✅ Use `''.join(list)` for building strings
✅ Use string methods (built-in C implementation)
✅ Use `in` operator for substring search
✅ Use `Counter` for frequency counting
✅ Use list comprehension when applicable

### Don'ts
❌ Don't use `+=` in loops for string building
❌ Don't convert to list unnecessarily
❌ Don't use nested loops if avoidable
❌ Don't forget string is immutable

---

## Resources for Practice

### Online Platforms
1. **LeetCode** - String tag (150+ problems)
2. **GeeksforGeeks** - String practice
3. **HackerRank** - String challenges
4. **Codeforces** - String problems

### Study Plan
- **Week 1:** Basics, reversal, palindrome (10 problems)
- **Week 2:** Anagrams, frequency, substrings (10 problems)
- **Week 3:** Two pointers, sliding window (10 problems)
- **Week 4:** Pattern matching, advanced (10 problems)

---

## Summary

### Must Know Concepts
✅ String immutability in Python
✅ String methods (split, join, strip, replace, etc.)
✅ Two pointer technique
✅ Sliding window for substrings
✅ Character frequency using Counter
✅ Pattern matching basics (naive, KMP)
✅ Common patterns (palindrome, anagram, subsequence)

### Key Takeaways
- Strings are immutable - use lists for modifications
- Learn to recognize patterns (two pointer, sliding window, etc.)
- Master string methods - they're optimized in C
- Practice both easy and hard problems
- Always consider edge cases

---

**Next Steps:**
1. Practice 10-15 easy string problems
2. Master two pointer and sliding window
3. Learn pattern matching algorithms
4. Solve medium and hard problems
5. Time yourself during practice

Good luck with your string learning journey! 🚀

---

*Last Updated: November 2024*
