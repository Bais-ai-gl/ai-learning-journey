# Day 12：用 LangGraph 重写 ReAct Agent（状态图 + 条件分支）
# 用法：py 01_agent_langgraph.py
# 依赖：py -m pip install langgraph langchain-openai openai
#   （langchain-openai 提供 ChatOpenAI；langgraph 会自动带上 langchain-core）

import os
# 关键修复：本机若开了系统代理，httpx 默认走代理会拦截本地 Ollama 请求(127.0.0.1:11434)。
# 设置 NO_PROXY 让 httpx 直连本地（必须在创建任何客户端之前设置）。
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import ast
import operator
import sqlite3
from typing import Annotated

import httpx
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, create_react_agent

# ===========================================================================
# 用 @tool 装饰器定义三个工具（docstring 当描述、类型注解当参数 Schema）
# ===========================================================================
_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval_ast(node):
    """安全求值 AST 节点（白名单方式，杜绝 eval 注入）。"""
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_ast(node.operand))
    raise ValueError("表达式里含有不允许的运算")


@tool
def calculator(expr: str) -> str:
    """安全计算一个数学表达式并返回数值，例如输入 '3.5*7+2'。"""
    return str(_eval_ast(ast.parse(expr.strip(), mode="eval")))


_WEATHER = {"北京": "晴，25℃，西北风 3 级", "上海": "多云，28℃", "深圳": "雷阵雨，30℃"}


@tool
def get_weather(city: str) -> str:
    """查询某个城市的天气（模拟数据，学习用）。"""
    return _WEATHER.get(city, f"{city}：晴，22℃，空气质量优（模拟数据）")


def _build_db():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE products (name TEXT, category TEXT, price REAL, stock INTEGER)")
    conn.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?)",
        [("苹果", "水果", 8.5, 120), ("香蕉", "水果", 3.2, 80), ("牛奶", "乳制品", 6.5, 200),
         ("面包", "烘焙", 12.0, 45), ("咖啡豆", "饮品", 68.0, 30)],
    )
    return conn


_DB = _build_db()


@tool
def search_db(keyword: str) -> str:
    """在本地商品库存库中按名称或分类模糊查询，返回库存信息。"""
    rows = _DB.execute(
        "SELECT name, category, price, stock FROM products WHERE name LIKE ? OR category LIKE ?",
        (f"%{keyword}%", f"%{keyword}%"),
    ).fetchall()
    if not rows:
        return f"没有找到与「{keyword}」相关的商品"
    return "\n".join(f"{n}({c})：单价 {p} 元，库存 {s} 件" for n, c, p, s in rows)


TOOLS = [calculator, get_weather, search_db]

# ===========================================================================
# 手写底层状态图（等价于 Day 11 的 while 循环，但用“图 + 条件分支”表达）
# ===========================================================================
class AgentState(TypedDict):
    """Agent 的共享状态：本质上就是一段不断累加的消息列表。
    add_messages 表示每次节点返回的消息是“追加”而不是“覆盖”。"""
    messages: Annotated[list[BaseMessage], add_messages]


def build_manual_graph(model):
    # bind_tools：把工具 Schema 交给模型，让模型输出 tool_calls（Day 11 里的 tools 参数）
    model_with_tools = model.bind_tools(TOOLS)

    def call_model(state: AgentState):
        """agent 节点：调模型，返回的消息由 add_messages 追加进状态。"""
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def should_continue(state: AgentState):
        """条件分支（图的“if”）：
        最后一条消息带 tool_calls → 走 "tools"；否则 → END 结束。"""
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)          # 思考节点
    graph.add_node("tools", ToolNode(TOOLS))     # 工具节点：批量执行 tool_calls 并回填结果
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")             # 工具执行完 → 回到 agent 继续推理
    return graph.compile()


def warm_up(model_name):
    """预热模型：/v1 接口冷加载会直接 502，原生接口会耐心等加载完。"""
    print("正在预热大模型（冷加载约几秒~几十秒，请耐心等待）...")
    try:
        r = httpx.post(
            "http://127.0.0.1:11434/api/generate",
            json={"model": model_name, "prompt": "你好", "stream": False}, timeout=180,
        )
        print("模型预热完成" if r.status_code == 200 else f"[预热] 返回 {r.status_code}：{r.text[:150]}")
    except Exception as e:
        print(f"[预热] 失败：{type(e).__name__} {e}")


def main():
    warm_up("qwen2.5:7b")

    # ChatOpenAI 直连本地 Ollama 的 OpenAI 兼容接口
    model = ChatOpenAI(
        base_url="http://127.0.0.1:11434/v1",
        api_key="ollama",           # Ollama 不校验 key，随便填
        model="qwen2.5:7b",
        temperature=0,
    )
    question = "北京天气怎么样？顺便算一下 3.5*7+2"

    # 方式 A：手写底层状态图（能看清 agent/tools 两个节点 + 条件分支，教学用）
    print("\n===== 方式 A：手写 StateGraph + ToolNode =====")
    app = build_manual_graph(model)
    result = app.invoke({"messages": [("user", question)]})
    print("最终答案：\n", result["messages"][-1].content)

    # 方式 B：一行预置函数（最简，LangGraph 0.2+ 推荐日常直接用它）
    #   agent = create_react_agent(model, TOOLS)
    #   result = agent.invoke({"messages": [("user", question)]})
    #   print(result["messages"][-1].content)


if __name__ == "__main__":
    main()


# ===========================================================================
# 自测 / 思考（面试题）
# ===========================================================================
# 1. LangGraph 的 StateGraph 和 Day 11 的 while 循环，本质区别是什么？
#    （提示：循环是“跑一段代码”，状态图是“在节点间按条件跳转”，天然可持久化/可视化/加子图）
# 2. add_messages 这个 reducer 有什么用？如果去掉它，节点返回 {"messages": [...]} 会发生什么？
# 3. add_conditional_edges 里的 should_continue 相当于循环里的哪一步？END 又对应什么？
# 4. ToolNode 帮你省掉了哪些样板代码（对比 Day 11 手动回填 role="tool" 的部分）？
# 5. create_react_agent 底层就是上面这张图，去读它的源码，能找到 agent/tools 节点和条件边吗？
