# LeetCode 20：有效的括号（Valid Parentheses）
# 求职冲刺 · 简单题 · 栈
# 用法：py 03_valid_parentheses.py


def is_valid(s: str) -> bool:
    """判断括号是否有效：左括号入栈，右括号与栈顶配对"""
    pairs = {")": "(", "]": "[", "}": "{"}   # 右括号 -> 对应左括号
    stack: list[str] = []

    for ch in s:
        if ch in "([{":                 # 左括号：入栈
            stack.append(ch)
        else:                            # 右括号：必须匹配栈顶
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack                     # 栈空才说明全部配对成功


# ---- 自测 ----
if __name__ == "__main__":
    cases = [("()[]{}", True), ("(]", False), ("([)]", False), ("{[]}", True)]
    for s, expected in cases:
        got = is_valid(s)
        print(f"{s} -> {got}")
        assert got == expected, f"{s} 测试失败"
    print("测试通过")

# 复杂度：时间 O(n)（遍历一次），空间 O(n)（栈最坏存下全部左括号）
# 思考：为什么用栈？括号是「后进先出」的嵌套结构——最近的左括号要和当前右括号配对。
#      "([)]" 这种交叉配对，栈能立刻识破：遇到 ) 时栈顶是 [，不匹配 -> False。
