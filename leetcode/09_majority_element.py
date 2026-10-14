# LeetCode 169：多数元素（Majority Element）
# 求职冲刺 · 简单题 · 摩尔投票法
# 用法：py 09_majority_element.py

from typing import List


def majority_element(nums: List[int]) -> int:
    """返回出现次数超过 n/2 的多数元素（题目保证存在）"""
    candidate = nums[0]
    count = 0
    for x in nums:
        if count == 0:                 # 票数归零，换一个候选
            candidate = x
        count += 1 if x == candidate else -1
    return candidate


# ---- 自测 ----
if __name__ == "__main__":
    nums = [2, 2, 1, 1, 1, 2, 2]
    ans = majority_element(nums)
    print("输入：", nums)
    print("多数元素：", ans)          # 期望 2
    assert ans == 2, "测试失败"
    print("测试通过")

# 复杂度：时间 O(n)，空间 O(1)
# 思考：摩尔投票的直觉——多数元素比其他所有元素加起来还多，正负抵消后剩下的候选人就是它。
#      排序取中位数 O(n log n)、哈希计数 O(n)/O(n)；摩尔投票 O(n)/O(1) 是最优解。
