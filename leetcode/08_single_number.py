# LeetCode 136：只出现一次的数字（Single Number）
# 求职冲刺 · 简单题 · 位运算（异或）
# 用法：py 08_single_number.py

from typing import List


def single_number(nums: List[int]) -> int:
    """只有一个数出现一次，其余都出现两次，找出那个数"""
    res = 0
    for x in nums:
        res ^= x   # 相同的数异或抵消为 0，最后剩下的就是只出现一次的数
    return res


# ---- 自测 ----
if __name__ == "__main__":
    nums = [4, 1, 2, 1, 2]
    ans = single_number(nums)
    print("输入：", nums)
    print("只出现一次的数：", ans)   # 期望 4
    assert ans == 4, "测试失败"
    print("测试通过")

# 复杂度：时间 O(n)，空间 O(1)
# 思考：为什么异或可行？a ^ a = 0、a ^ 0 = a，且异或满足交换律、结合律，
#      所以成对出现的数两两抵消，只剩那个单身的数。
#      通用解法（哈希表计数）空间 O(n)；异或把空间压到 O(1)，是这道题的加分点。
