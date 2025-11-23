class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        max_count = 0
        count = 0
        list1 = []
        
        for i, l in enumerate(s):
            count = 0
            list1.append(l)
            count +=1
            max_count = max(max_count, count)
            for j, m in enumerate(s[i+1:]):
                if m in list1:
                    max_count = max(max_count, count)
                    count = 0
                    list1 = []
                    break
                else:
                    count +=1
                    list1.append(m)
        return max_count