# 脚本 4：类与继承
# 练习：__init__、方法、继承、super()

class Model:
    def __init__(self, name, params):    # 构造函数
        self.name = name
        self.params = params

    def describe(self):
        return f"{self.name}（{self.params} 参数）"

    def run(self):
        return f"{self.name} 正在推理…"

# 继承：ChatModel 复用 Model 的属性和方法
class ChatModel(Model):
    def __init__(self, name, params, temperature):
        super().__init__(name, params)   # 调用父类构造
        self.temperature = temperature

    def run(self):                        # 重写父类方法
        return f"{self.name} 以温度 {self.temperature} 生成回答…"

base = Model("gpt", 175)
chat = ChatModel("qwen", 7, 0.7)

print(base.describe())
print(chat.describe())
print(chat.run())                        # 调用子类重写后的方法
