# LeetCode 53：最大子数组和（Maximum Subarray）
# 求职冲刺 · 简单题 · 动态规划 / Kadane 算法
# 用法：py 06_max_subarray.py

from typing import List


def max_sub_array(nums: List[int]) -> int:
    """返回连续子数组的最大和（Kadane 算法）"""
    cur = nums[0]      # 以当前元素结尾的最大子数组和
    best = nums[0]     # 全局最大
    for x in nums[1:]:
        cur = max(x, cur + x)   # 要么接着前面，要么另起炉灶
        best = max(best, cur)
    return best


# ---- 自测 ----
if __name__ == "__main__":
    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    ans = max_sub_array(nums)
    print("输入：", nums)
    print("最大子数组和：", ans)   # 期望 6，对应子数组 [4, -1, 2, 1]
    assert ans == 6, "测试失败"
    print("测试通过")

# 复杂度：时间 O(n)（一遍遍历），空间 O(1)
# 思考：cur = max(x, cur + x) 的含义——以当前元素结尾的最优解：
#      若「前面接上来的和」是负数，宁可只取 x 自己，重新开始。
#      分治法也能做（O(n log n)），但 Kadane 是这道题的最优解。
