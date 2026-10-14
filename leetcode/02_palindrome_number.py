# LeetCode 09：回文数（Palindrome Number）
# 求职冲刺 · 简单题 · 数学
# 用法：py 02_palindrome_number.py


def is_palindrome(x: int) -> bool:
    """判断整数 x 是否回文（不转字符串：反转一半数字）"""
    # 负数、或以 0 结尾但本身不是 0 的数，都不可能是回文
    if x < 0 or (x % 10 == 0 and x != 0):
        return False

    reversed_half = 0
    while x > reversed_half:            # 反转后半段，直到 x 不再大于反转数
        reversed_half = reversed_half * 10 + x % 10
        x //= 10

    # 偶数位：两半相等；奇数位：去掉中间一位后相等
    return x == reversed_half or x == reversed_half // 10


# ---- 自测 ----
if __name__ == "__main__":
    for num, expected in [(121, True), (-121, False), (10, False)]:
        got = is_palindrome(num)
        print(f"{num} -> {got}")
        assert got == expected, f"{num} 测试失败"
    print("测试通过")

# 复杂度：时间 O(log10(n))（每次除以 10），空间 O(1)
# 思考：最省事的写法是 str(x) == str(x)[::-1]，但面试可能追问「不转字符串」。
#      反转一半的好处：既不用字符串，又不会像反转整个数那样整数溢出（Python 虽无此问题）。
