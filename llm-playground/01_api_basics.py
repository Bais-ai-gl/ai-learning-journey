# Day 3：调用大模型 API（OpenAI 兼容接口）
# 用法：
#   1. 注册智谱(bigmodel.cn) 或 通义百炼(bailian.aliyun.com)，拿到 API Key
#   2. 在 PowerShell 里设置环境变量（不要写死在代码里）：
#        $env:OPENAI_API_KEY="你的key"
#   3. 运行：  py 01_api_basics.py
# 注意：装依赖  py -m pip install openai

import os
from openai import OpenAI

# ---- 配置（按你注册的平台改这两个值）----
API_KEY = os.environ.get("OPENAI_API_KEY")          # 从环境变量读取
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"   # 智谱；通义填 https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL = "glm-4-flash"                               # 智谱模型名；通义可填 qwen-plus

if not API_KEY:
    print('请先设置 API Key，例如 PowerShell 里执行：$env:OPENAI_API_KEY="你的key"')
    raise SystemExit(1)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ---- 1. 最简调用 ----
resp = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "你是一个乐于助人的 AI 助手。"},
        {"role": "user", "content": "用一句话介绍你自己"},
    ],
    temperature=0.7,
)
print("最简回复：", resp.choices[0].message.content)

# ---- 2. 参数实验：同一问题，不同 temperature ----
# 注意：智谱 GLM 的 temperature 取值范围是 [0, 1]（OpenAI 才是 [0, 2]），超过 1 会报错
question = "写一句表白的话"
for temp in [0, 0.7, 1.0]:
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
        temperature=temp,
    )
    print(f"\n[temperature={temp}] {r.choices[0].message.content}")

# 观察：temperature 越低越确定/稳定，越高越发散/有创意。
