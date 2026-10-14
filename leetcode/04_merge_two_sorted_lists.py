# LeetCode 21：合并两个有序链表（Merge Two Sorted Lists）
# 求职冲刺 · 简单题 · 链表 · 哨兵节点
# 用法：py 04_merge_two_sorted_lists.py

from typing import Optional


class ListNode:
    """单链表节点"""
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next


def merge_two_lists(
    list1: Optional[ListNode], list2: Optional[ListNode]
) -> Optional[ListNode]:
    """合并两个升序链表，返回新链表的头节点"""
    dummy = ListNode()               # 哨兵节点：省去「头节点是谁」的分支判断
    cur = dummy
    while list1 and list2:           # 两条都有时，谁小接谁
        if list1.val <= list2.val:
            cur.next = list1
            list1 = list1.next
        else:
            cur.next = list2
            list2 = list2.next
        cur = cur.next
    cur.next = list1 or list2        # 剩一条没走完，直接接上
    return dummy.next


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
    l1 = build_list([1, 2, 4])
    l2 = build_list([1, 3, 4])
    ans = to_list(merge_two_lists(l1, l2))
    print("输入：[1,2,4] 和 [1,3,4]")
    print("输出：", ans)             # 期望 [1, 1, 2, 3, 4, 4]
    assert ans == [1, 1, 2, 3, 4, 4], "测试失败"
    print("测试通过")

# 复杂度：时间 O(m+n)（两链表各遍历一次），空间 O(1)（只移动指针，无额外分配）
# 思考：递归写法更简洁但空间 O(m+n)（递归栈）；迭代 + 哨兵节点空间 O(1)，是更稳的面试答案。
#      哨兵节点 dummy 的作用：避免单独判断「第一个节点从哪条链表来」。
