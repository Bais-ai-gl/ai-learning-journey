# LeetCode 206：反转链表（Reverse Linked List）
# 求职冲刺 · 简单题 · 链表 · 双指针
# 用法：py 07_reverse_linked_list.py

from typing import Optional


class ListNode:
    """单链表节点"""
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """原地反转单链表，返回新头节点"""
    prev: Optional[ListNode] = None
    cur = head
    while cur:
        nxt = cur.next      # 先记住下一个，防止断链
        cur.next = prev     # 反转指针方向
        prev = cur          # 两个指针同步前移
        cur = nxt
    return prev             # 循环结束时 prev 就是新的头


def build_list(vals: list[int]) -> Optional[ListNode]:
    """辅助：数组转链表"""
    dummy = ListNode()
    cur = dummy
    for v in vals:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next


def to_list(head: Optional[ListNode]) -> list[int]:
    """辅助：链表转数组，便于打印"""
    res = []
    while head:
        res.append(head.val)
        head = head.next
    return res


# ---- 自测 ----
if __name__ == "__main__":
    head = build_list([1, 2, 3, 4, 5])
    ans = to_list(reverse_list(head))
    print("输入：[1,2,3,4,5]")
    print("反转后：", ans)          # 期望 [5, 4, 3, 2, 1]
    assert ans == [5, 4, 3, 2, 1], "测试失败"
    print("测试通过")

# 复杂度：时间 O(n)，空间 O(1)（原地反转，只用常数个指针）
# 思考：递归写法空间 O(n)（递归栈），迭代写法空间 O(1) 更优；面试先把迭代写熟。
#      关键点：先存 nxt 再改 cur.next，否则会「断链」找不到下一个节点。
