# 脚本 1：类型与输入输出
# 练习：基础类型、类型转换、f-string 格式化

name = "AI"                     # str
age = 3                         # int
score = 98.5                    # float
is_learning = True              # bool
skills = ["python", "pandas"]   # list
profile = {"city": "北京", "level": 1}  # dict
tags = {"ml", "dl"}             # set（无序、去重）

# 类型转换
age_str = str(age)              # int -> str
score_int = int(score)          # float -> int（截断）

# f-string 格式化
print(f"我叫{name}，学AI已经{age}年了，分数{score}")
print(f"技能数量：{len(skills)}，城市：{profile['city']}")
print(f"年龄字符串类型：{type(age_str).__name__}")
print(f"tags 集合：{tags}")
