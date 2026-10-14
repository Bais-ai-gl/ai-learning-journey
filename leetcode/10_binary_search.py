# LeetCode 704：二分查找（Binary Search）
# 求职冲刺 · 简单题 · 二分查找
# 用法：py 10_binary_search.py

from typing import List


def binary_search(nums: List[int], target: int) -> int:
    """在升序数组 nums 中查找 target，返回下标或 -1（闭区间写法）"""
    lo, hi = 0, len(nums) - 1
    while lo <= hi:                  # 闭区间 [lo, hi]，lo == hi 时还要检查一次
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            lo = mid + 1             # target 在右半
        else:
            hi = mid - 1             # target 在左半
    return -1


# ---- 自测 ----
if __name__ == "__main__":
    nums = [-1, 0, 3, 5, 9, 12]
    for t in (9, 2):
        ans = binary_search(nums, t)
        print(f"nums={nums}, target={t} -> {ans}")
    assert binary_search(nums, 9) == 4    # 9 在下标 4
    assert binary_search(nums, 2) == -1   # 2 不存在
    print("测试通过")

# 复杂度：时间 O(log n)（每次砍半），空间 O(1)
# 思考：边界写法要背熟——闭区间用 lo <= hi；mid 用 (lo + hi) // 2。
#      进阶变体：找「第一个 >= target」的下界、找「最后一个 <= target」的上界，是二分高频题。
