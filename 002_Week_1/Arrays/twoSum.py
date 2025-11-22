from typing import List

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i, num in enumerate(nums):
            a = target - num
            if a in nums:
                j = nums.index(a)
                if i != j: 
                    return [i, j]

if __name__ == "__main__":
    nums = [2, 7, 11, 15]
    target = 9
    solution = Solution()
    result = solution.twoSum(nums, target)
    print(result)



"""
Explanation of the Two Sum solution:

1. We loop through the list using enumerate(), which gives both
   the index (i) and the value (num) of each element.

2. For each number, we compute the complement 'a' such that:
       a = target - num
   This is the value we need to find in the list to make the sum equal to 'target'.

3. We check if this complement exists in the list:
       if a in nums

4. If it exists, we get its index using:
       j = nums.index(a)

5. We must ensure that we do not use the same index twice.
   Example: target = 6, nums = [3, 3]
   If i == j, it means we found the same element, so we skip it.

6. When we find two valid indices, we return them as a list:
       return [i, j]

Note:
This approach works but is not optimal.
- Time complexity: O(n^2)
A better solution uses a hash map (dictionary) in O(n) time.
"""
