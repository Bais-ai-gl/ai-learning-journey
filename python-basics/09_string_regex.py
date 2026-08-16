# 脚本 9：字符串与正则
# 练习：切片、split/join、re 模块

import re

text = "  我的邮箱是 abc@example.com，另一个是 xyz123@test.cn  "

# 切片
print("前5个字符：", text.strip()[:5])

# split / join
words = text.strip().split("，")
print("按逗号切分：", words)
print("重新拼接：", " | ".join(words))

# 正则：提取所有邮箱
emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
print("提取到的邮箱：", emails)

# 替换
masked = re.sub(r"\d+", "#", "订单号 12345，金额 678")
print("数字脱敏：", masked)
