# Day 4：Prompt 工程 · 6 个技巧实测
# 用法：先设 API Key（$env:OPENAI_API_KEY="你的key"），再  py 01_six_techniques.py

import os
import json
from openai import OpenAI

API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"   # 智谱；通义换 base_url 和模型名
MODEL = "glm-4-flash"

if not API_KEY:
    print('请先设置 API Key：$env:OPENAI_API_KEY="你的key"')
    raise SystemExit(1)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def ask(messages, temperature=0.7):
    r = client.chat.completions.create(model=MODEL, messages=messages, temperature=temperature)
    return r.choices[0].message.content

# ---- 技巧1：角色设定 ----
print("== 技巧1 角色设定 ==")
print(ask([
    {"role": "system", "content": "你是资深劳动法律师，回答严谨、通俗、给出依据。"},
    {"role": "user", "content": "试用期被辞退有赔偿吗？"},
]))

# ---- 技巧2：few-shot（给示例再问）----
print("\n== 技巧2 few-shot ==")
print(ask([
    {"role": "user", "content": (
        "把句子情绪分类为【正面/中性/负面】。\n"
        "示例：太棒了！ → 正面\n"
        "示例：还行吧。 → 中性\n"
        "示例：真倒霉。 → 负面\n"
        "现在分类：\"简直糟透了\""
    )},
]))

# ---- 技巧3：思维链（CoT）----
print("\n== 技巧3 思维链 ==")
print(ask([
    {"role": "user", "content": "小明有3个苹果，买了5个，吃掉2个，又送朋友1个。他还剩几个？请一步一步思考。"},
]))

# ---- 技巧4：结构化 JSON 输出 ----
print("\n== 技巧4 结构化 JSON 输出 ==")
raw = ask([
    {"role": "user", "content": '分析"这个产品很好用但太贵了"，只返回 JSON，格式 {"优点":str, "缺点":str}，不要输出多余文字。'},
])
try:
    parsed = json.loads(raw)
    print("JSON 解析成功：", parsed)
except json.JSONDecodeError:
    print("解析失败，原样输出：", raw)

# ---- 技巧5：输出约束 ----
print("\n== 技巧5 输出约束（限 20 字）==")
print(ask([{"role": "user", "content": "用不超过 20 个字介绍 Python。"}]))

# ---- 技巧6：自一致性（问 3 次取多数）----
print("\n== 技巧6 自一致性 ==")
answers = [ask([{"role": "user", "content": "1+1=？只回答一个数字"}], temperature=0.8) for _ in range(3)]
print("三次回答：", answers)
