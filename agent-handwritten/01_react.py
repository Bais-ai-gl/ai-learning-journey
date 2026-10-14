# Day 11：手写 ReAct Agent（Thought → Action → Observation → Answer）
# 用法：py 01_react.py
# 依赖：py -m pip install openai   （ast / json / sqlite3 / operator 都是内置库）

import os
# 关键修复：本机若开了系统代理(127.0.0.1:7897)，openai 库底层的 httpx 默认会走代理，
# 导致本地 Ollama 请求(127.0.0.1:11434)被代理拦截、返回 502。
# 设置 NO_PROXY 让 httpx 直连本地地址（必须在创建任何客户端之前设置）。
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import ast
import json
import operator
import sqlite3

import httpx  # openai 库自带，无需额外安装
from openai import OpenAI

# ===========================================================================
# 工具 1：calculator —— 用 ast 白名单方式安全计算表达式
# ===========================================================================
# 为什么不用 eval()？eval 会执行任意代码（如 __import__('os').system('rm -rf')）。
# ast.parse 只把字符串解析成语法树，我们再手动递归求值，只放行我们允许的节点类型。
_ALLOWED_OPS = {
    ast.Add: operator.add,      # +
    ast.Sub: operator.sub,      # -
    ast.Mult: operator.mul,     # *
    ast.Div: operator.truediv,  # /
    ast.Mod: operator.mod,      # %
    ast.Pow: operator.pow,      # **
    ast.USub: operator.neg,     # 一元负号 -3
    ast.UAdd: operator.pos,     # 一元正号 +3
}


def _eval_ast(node):
    """递归求值 AST 节点，只允许数字、加减乘除/取余/乘方和正负号。"""
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_ast(node.operand))
    raise ValueError("表达式里含有不允许的运算")


def calculator(expr):
    """安全计算数学表达式，返回数值。"""
    return _eval_ast(ast.parse(expr.strip(), mode="eval"))  # mode="eval" 只允许单个表达式


# ===========================================================================
# 工具 2：get_weather —— 模拟天气（学习用，非真实数据）
# ===========================================================================
_WEATHER = {
    "北京": "晴，25℃，西北风 3 级，空气质量优",
    "上海": "多云，28℃，东南风 2 级",
    "深圳": "雷阵雨，30℃，湿度 85%",
    "广州": "阴，27℃，湿度 70%",
}


def get_weather(city):
    """返回模拟天气。"""
    return _WEATHER.get(city, f"{city}：晴，22℃，空气质量优（模拟数据）")


# ===========================================================================
# 工具 3：search_db —— 在内存 SQLite 里查商品库存
# ===========================================================================
def _build_db():
    """建一个内存库并灌入几条商品数据。连接对象在进程内一直有效。"""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE products (name TEXT, category TEXT, price REAL, stock INTEGER)")
    rows = [
        ("苹果", "水果", 8.5, 120),
        ("香蕉", "水果", 3.2, 80),
        ("西瓜", "水果", 15.0, 0),
        ("牛奶", "乳制品", 6.5, 200),
        ("面包", "烘焙", 12.0, 45),
        ("咖啡豆", "饮品", 68.0, 30),
    ]
    conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", rows)
    return conn


_DB = _build_db()


def search_db(keyword):
    """按名称/分类模糊查询商品库存。"""
    rows = _DB.execute(
        "SELECT name, category, price, stock FROM products WHERE name LIKE ? OR category LIKE ?",
        (f"%{keyword}%", f"%{keyword}%"),
    ).fetchall()
    if not rows:
        return f"没有找到与「{keyword}」相关的商品"
    return "\n".join(
        f"{name}({category})：单价 {price} 元，库存 {stock} 件" for name, category, price, stock in rows
    )


# ===========================================================================
# 把三个工具登记成 OpenAI Function Calling 的 Schema（模型据此决定调哪个）
# ===========================================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "安全计算一个数学表达式并返回数值，例如 '3.5*7+2'",
            "parameters": {
                "type": "object",
                "properties": {"expr": {"type": "string", "description": "要计算的数学表达式"}},
                "required": ["expr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某个城市的天气（模拟数据）",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名，如 北京"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_db",
            "description": "在本地商品库存库中按名称或分类模糊查询库存",
            "parameters": {
                "type": "object",
                "properties": {"keyword": {"type": "string", "description": "查询关键词，如 牛奶"}},
                "required": ["keyword"],
            },
        },
    },
]

# 工具名 → 实际函数（执行工具时按名分发）
EXECUTORS = {"calculator": calculator, "get_weather": get_weather, "search_db": search_db}


# ===========================================================================
# 手写 ReAct 循环（核心！这就是 Agent 的“思考-行动-观察”引擎）
# ===========================================================================
def run_react(question, max_rounds=10):
    client = OpenAI(api_key="ollama", base_url="http://127.0.0.1:11434/v1")
    model = "qwen2.5:7b"
    messages = [{"role": "user", "content": question}]  # 整个对话历史都在这里累积

    for round_no in range(1, max_rounds + 1):
        # 1) Thought：模型看到历史 + 工具列表，决定「思考→要不要调工具→调哪个」
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, temperature=0.0,
        )
        msg = resp.choices[0].message

        # 把 assistant 这条消息写回历史（带 tool_calls，下一轮模型要据此对齐）
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        # 2) 没有 tool_calls → 模型觉得信息够了，直接给出最终 Answer，结束循环
        if not msg.tool_calls:
            print(f"[轮次 {round_no}] 模型直接给出最终答案，循环结束\n")
            return msg.content

        # 3) Action + Observation：逐个执行模型要求的工具，把结果回填给模型
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            print(f"[轮次 {round_no}] Action : 调用 {name}({args})")
            try:
                result = EXECUTORS[name](**args)
            except Exception as e:
                result = f"工具执行出错：{type(e).__name__}: {e}"
            print(f"[轮次 {round_no}] Observation : {result}\n")
            # role="tool" 的结果必须用 tool_call_id 和上面的调用一一对应
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

    return "达到最大轮数仍未得到最终答案"


def warm_up(model):
    """预热模型：/v1 接口在模型未加载时会直接 502，原生接口会耐心等加载完。"""
    print("正在预热大模型（冷加载约几秒~几十秒，请耐心等待）...")
    try:
        r = httpx.post(
            "http://127.0.0.1:11434/api/generate",
            json={"model": model, "prompt": "你好", "stream": False},
            timeout=180,
        )
        print("模型预热完成" if r.status_code == 200 else f"[预热] 返回 {r.status_code}：{r.text[:150]}")
    except Exception as e:
        print(f"[预热] 失败：{type(e).__name__} {e}")


def main():
    warm_up("qwen2.5:7b")
    # 测试场景：一句话里藏着两个工具调用（天气 + 计算），考验模型的 ReAct 规划
    question = "北京天气怎么样？顺便算一下 3.5*7+2"
    print("问题：", question)
    print("=" * 70)
    answer = run_react(question)
    print("=" * 70)
    print("最终答案：\n", answer)


if __name__ == "__main__":
    main()


# ===========================================================================
# 自测 / 思考（面试题）
# ===========================================================================
# 1. 为什么调工具的结果要包成 role="tool" 且带 tool_call_id 回填？漏掉会怎样？
# 2. calculator 为什么用 ast.parse + 递归求值，而不是 eval()？ast 白名单漏了哪类节点？
# 3. 如果模型在一次回复里同时发起 3 个 tool_calls（并行），上面代码怎么处理？结果回填顺序重要吗？
# 4. ReAct 和「把工具结果一次性塞进 prompt」相比，优点是什么？代价（成本/轮次）是什么？
# 5. 怎么加一个「最多轮数 + 提前终止」之外的护栏，防止模型无限调用工具？
