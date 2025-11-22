from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        carry = 0
        dummy = ListNode()
        tail = dummy
        
        while l1 or l2 or carry:
            x = l1.val if l1 else 0
            y = l2.val if l2 else 0
            
            number = x + y + carry
            
            if number > 9:
                carry = 1
                number -= 10
            else:
                carry = 0
            
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
            
            tail.next = ListNode(number)
            tail = tail.next  
        
        return dummy.next


# Helper functions to build and print linked lists
def build_list(values):
    head = ListNode(values[0])
    current = head
    for v in values[1:]:
        current.next = ListNode(v)
        current = current.next
    return head

def print_list(node):
    result = []
    while node:
        result.append(node.val)
        node = node.next
    print(result)


if __name__ == "__main__":
    # Example input
    l1 = build_list([9,9,9,9,9,9,9])
    l2 = build_list([9,9,9,9])
    
    solution = Solution()
    result = solution.addTwoNumbers(l1, l2)

    # Print result linked list
    print_list(result)




"""
carry holds the overflow when digits add to more than 9.

dummy is a placeholder node so we don’t need special handling for the
first node of the result linked list. It simplifies list construction.

tail always points to the last node in the result list, allowing us to
append new nodes easily.

The loop runs as long as:
- l1 has remaining nodes, or
- l2 has remaining nodes, or
- there is still a carry from the previous digit addition.

x and y represent the current digit values from l1 and l2.
If a list has already ended, its value is treated as 0.

number = x + y + carry calculates the sum of the digits and previous carry.

If number > 9, we take only the last digit (number - 10) and set carry = 1.
Otherwise, carry becomes 0.

We move l1 and l2 to their next nodes if they exist.

We then create a new ListNode holding the resulting digit.
This new node is attached to tail.next, and tail moves forward.

Finally, we return dummy.next because dummy is only a placeholder and
the actual result list starts from dummy.next.
"""

