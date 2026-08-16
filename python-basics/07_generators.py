# 脚本 7：生成器
# 练习：yield、生成器表达式、内存优势

import sys

def squares(n):                          # 生成器函数
    for i in range(n):
        yield i * i                      # 每次产出一个值

# 生成器表达式（惰性求值）
gen = (x * x for x in range(100000))
lst = [x * x for x in range(100000)]     # 一次性全部算好

print("生成器前5个：", [next(squares(5)) for _ in range(5)])

# 内存对比：生成器几乎不占内存，列表占几 MB
print(f"生成器占用：{sys.getsizeof(gen)} 字节")
print(f"列表占用：{sys.getsizeof(lst)} 字节")
