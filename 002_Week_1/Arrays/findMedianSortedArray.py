from typing import List

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        left = 0
        right = 1
        nums = nums1+nums2
        print(nums)
        total = 0
        n = len(nums)
        
        while left < (len(nums)-1):
            while right <(len(nums)):
                if nums[left]>nums[right]:
                    temp = 0
                    temp = nums[left]
                    nums[left] = nums[right]
                    nums[right] = temp
                    right += 1
                else:
                    right += 1
                
            left += 1
            right = left + 1
                
        print("Sorted",nums)
            
        if len(nums)%2 != 0:
            mid = (n//2) 
            nmb = float(nums[mid])
            return nmb
            
        else:
             mid = n//2
             nmb = float((nums[mid-1]+nums[mid])/2)
             return nmb