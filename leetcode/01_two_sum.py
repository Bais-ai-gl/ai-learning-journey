# LeetCode 01：两数之和（Two Sum）
# 求职冲刺 · 简单题 · 哈希表
# 用法：py 01_two_sum.py

from typing import List


def two_sum(nums: List[int], target: int) -> List[int]:
    """返回和为 target 的两个数的下标（题目保证有且仅有一组解）"""
    seen: dict[int, int] = {}          # 值 -> 下标
    for i, num in enumerate(nums):
        need = target - num            # 我要找的另一个数
        if need in seen:               # 之前见过 -> 直接返回
            return [seen[need], i]
        seen[num] = i                  # 没找到 -> 记下当前值
    return []                          # 题目保证有解，这里仅兜底


# ---- 自测 ----
if __name__ == "__main__":
    nums, target = [2, 7, 11, 15], 9
    ans = two_sum(nums, target)
    print("输入：", nums, "target =", target)
    print("输出：", ans)               # 期望 [0, 1]，因为 2 + 7 = 9
    assert ans == [0, 1], "测试失败"
    print("测试通过")

# 复杂度：时间 O(n)（一遍遍历），空间 O(n)（哈希表）
# 思考：为什么不用双重 for 循环？那是 O(n²)，数据一大就超时。
#      哈希表是「用空间换时间」：遍历时顺手存下每个值，之后 O(1) 就能查到补数。
