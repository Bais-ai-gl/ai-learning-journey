# 脚本 10：综合小算法
# 练习：排序、二分查找、字符频率统计（为 LeetCode 预热）

from collections import Counter

def binary_search(arr, target):
    """在有序数组中二分查找，返回下标或 -1"""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

nums = sorted([5, 2, 9, 1, 7])          # 排序
print("排序后：", nums)
print("查找 7 的下标：", binary_search(nums, 7))
print("查找 100 的下标：", binary_search(nums, 100))

# 字符频率统计
freq = Counter("aabbcccdddd")
print("字符频率：", dict(freq))
print("出现最多：", freq.most_common(1))
