# Day 1 · 脚本 3：函数与参数
# 练习：默认参数、*args、**kwargs、lambda

def greet(name, greeting="你好"):        # 默认参数
    return f"{greeting}，{name}"

def total(*args):                        # 任意个位置参数
    return sum(args)

def describe(**kwargs):                  # 任意个关键字参数
    return ", ".join(f"{k}={v}" for k, v in kwargs.items())

# lambda：一行匿名函数
double = lambda x: x * 2

print(greet("小王"))
print(greet("小李", greeting="早上好"))
print("总和：", total(1, 2, 3, 4, 5))
print("描述：", describe(name="模型", params=7, type="LLM"))
print("翻倍：", double(21))
