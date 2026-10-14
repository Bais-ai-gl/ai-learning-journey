# Day 1 · 脚本 6：装饰器
# 练习：@ 语法、带参数的装饰器（计时器）

import time

# 计时器装饰器：包装任意函数，打印耗时
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        cost = (time.perf_counter() - start) * 1000
        print(f"[{func.__name__}] 耗时 {cost:.2f} ms")
        return result
    return wrapper

@timer
def slow_sum(n):
    return sum(range(n))

print("结果：", slow_sum(1000000))
