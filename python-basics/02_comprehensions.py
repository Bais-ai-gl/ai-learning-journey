# Day 1 · 脚本 2：列表与字典推导式
# 练习：推导式（过滤、映射、嵌套）

nums = list(range(1, 21))          # 1~20

# 列表推导式：偶数的平方
even_squares = [n * n for n in nums if n % 2 == 0]

# 带条件的映射：能被3整除则翻倍，否则原值
mapped = [n * 2 if n % 3 == 0 else n for n in nums]

# 字典推导式：数字 -> 它的平方（只取奇数）
square_map = {n: n ** 2 for n in nums if n % 2 == 1}

# 嵌套推导式：乘法表（展平成列表）
table = [i * j for i in range(1, 10) for j in range(1, 10)]

print("偶数的平方：", even_squares)
print("3的倍数翻倍：", mapped)
print("奇数平方字典：", square_map)
print("乘法表前10项：", table[:10])
print(nums)
