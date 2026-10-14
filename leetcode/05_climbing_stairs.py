# LeetCode 70：爬楼梯（Climbing Stairs）
# 求职冲刺 · 简单题 · 动态规划
# 用法：py 05_climbing_stairs.py


def climb_stairs(n: int) -> int:
    """n 阶楼梯，每次爬 1 或 2 阶，返回不同方法数"""
    # dp[i] = dp[i-1] + dp[i-2]，即斐波那契数列
    if n <= 2:
        return n
    prev, cur = 1, 2                 # prev = dp[1]，cur = dp[2]
    for _ in range(3, n + 1):
        prev, cur = cur, prev + cur  # 滚动数组，只保留两个数，省空间
    return cur


# ---- 自测 ----
if __name__ == "__main__":
    for n, expected in [(2, 2), (3, 3), (4, 5)]:
        got = climb_stairs(n)
        print(f"n={n} -> {got} 种方法")
        assert got == expected, f"n={n} 测试失败"
    print("测试通过")

# 复杂度：时间 O(n)，空间 O(1)（滚动数组只存两个变量）
# 思考：状态转移为什么是 dp[i] = dp[i-1] + dp[i-2]？
#      最后一步要么跨 1 阶（来自 i-1），要么跨 2 阶（来自 i-2），两种情况方法数相加。
#      递归 + 记忆化也能做，但迭代写法空间最优、最好写。
