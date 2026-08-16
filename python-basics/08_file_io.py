# 脚本 8：文件读写
# 练习：open/with、读写文本、处理路径

from pathlib import Path

data = "第一行\n第二行\nAI 学习\n"

# 用 with 自动关闭文件
with open("test.txt", "w", encoding="utf-8") as f:
    f.write(data)

with open("test.txt", "r", encoding="utf-8") as f:
    content = f.read()
    print("全部内容：\n" + content)

with open("test.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    print("按行读：", lines)

# 路径处理
p = Path("test.txt")
print("文件名：", p.name, "| 存在：", p.exists())
