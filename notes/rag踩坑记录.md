# RAG 实战踩坑记录（Day 8 / Day 10）

> 日期：2026-09-06　｜　对应文件：`rag-handwritten/01_rag_solution.py`、`01_rag_skeleton.py`
> 背景：手写最小 RAG（文档加载 → 分块 → embedding → FAISS 检索 → 大模型生成），本地 Ollama 跑 qwen2.5:7b。

---

## 一句话总结

跑本地大模型应用，最常见的三个坑都撞上了：

1. **调本地 Ollama 报 502** —— 系统代理拦截了 httpx 的本地请求。
2. **下载模型报 10060 超时** —— huggingface.co 被墙，要用国内镜像。
3. **RAGAS 导入报 ModuleNotFoundError** —— ragas 与 langchain-community 版本错配。

前两个是「环境问题」，第三个是「依赖版本错配」，都和代码逻辑无关。但**排查过程本身就是一次很好的工程调试练习**。

---

## 坑一：调用本地 Ollama 返回 502（空响应体）

### 现象

```python
from openai import OpenAI
client = OpenAI(api_key="ollama", base_url="http://127.0.0.1:11434/v1")
r = client.chat.completions.create(...)
# 报错：openai.InternalServerError: Error code: 502
```

诡异的是：**时好时坏**。`ollama run qwen2.5:7b "你好"`（命令行）能正常对话，但脚本里调 API 就 502。

### 排查过程（关键证据链）

| 检查项 | 结果 | 结论 |
|---|---|---|
| `ollama list` | ✅ 模型在册 | 模型没丢 |
| `ollama run`（命令行） | ✅ 正常回复 | 模型能加载 |
| PowerShell `Invoke-RestMethod` 调 `/v1` | ✅ 200 | 接口本身能通 |
| **Python `openai`/`httpx` 调 `/v1`** | ❌ **502** | 只有 Python 走不通 |
| Ollama 日志搜 5xx | **完全没有 502** | 502 不是 Ollama 返回的 |

**最关键的发现**：Ollama 服务端日志里**从未出现过 502**，说明 502 是「中间某个东西」返回的，请求根本没到 Ollama。

### 根因

查系统代理发现：

```
ProxyEnable : 1
ProxyServer : 127.0.0.1:7897     ← Clash/樱花猫 类代理
ProxyOverride: localhost;127.*;...  ← Windows 有绕过名单
```

而 Python 检测到的代理是 `http://127.0.0.1:7897`。

**核心矛盾**：

- **PowerShell（.NET）和 `ollama run`（Go）** → 遵守 Windows 的「绕过代理名单」，本地地址直连 → 成功 ✅
- **Python 的 `httpx`（openai 库底层）** → **只认 `NO_PROXY` 环境变量**，不读 Windows 注册表的绕过名单，而 `NO_PROXY` 是空的 → 把 `127.0.0.1:11434` 也送进代理 → 代理返回 **502** ❌

### 修复

在脚本顶部、**创建任何客户端之前**设置：

```python
import os
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
```

> 只绕开本地地址，以后调智谱/通义等**远程 API 仍会正常走代理**，互不影响。

---

## 坑二：Hugging Face 下载模型报 10060 超时

### 现象

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
# 报错：[WinError 10060] ... thrown while requesting
#       HEAD https://huggingface.co/BAAI/bge-small-zh-v1.5/resolve/main/adapter_config.json
```

### 根因

huggingface.co 在国内不稳定/被墙。sentence-transformers 加载模型时会先向 huggingface.co 发请求检查文件（如 `adapter_config.json`），连接超时。

### 修复

用国内镜像 `hf-mirror.com`（官方推荐、速度快）：

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

> ⚠️ 必须在 **import sentence_transformers 之前**设置。因为 huggingface_hub 在导入时会读一次 `HF_ENDPOINT` 并缓存，之后改环境变量无效。

### 顺带记录：离线加载

如果模型已完整缓存，想完全离线加载（不联网）：

```python
os.environ["HF_HUB_OFFLINE"] = "1"   # 只用缓存，不联网
```

---

## 坑三：RAGAS 导入报 ModuleNotFoundError（Day 10）

### 现象

```python
# py 03_ragas_eval.py
from ragas import evaluate, EvaluationDataset, SingleTurnSample
# ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
```

### 根因

`ragas 0.4.3` 内部仍在 import 旧路径 `langchain_community.chat_models.vertexai`，但新版 `langchain-community 0.4.2` 已把该模块移除（迁到独立的 `langchain-google-vertexai` 包）。而 ragas 的依赖声明里只写 `langchain-community`（不锁版本），pip 装了最新版 → 版本错配。（官方 issue：#2741 / #2753）

### 修复

```powershell
py -m pip install "langchain-community<0.4.2"   # 降到 0.4.1
```

### 附：ragas 0.4.x 的 API 变化（和 0.2.x 完全不同）

降级解决 import 后，脚本还可能有 API 报错（`TypeError: All metrics must be initialised metric objects`）。0.4.x 的正确写法：

```python
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall  # 大写类
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from openai import OpenAI

llm = llm_factory("qwen2.5:7b", client=OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama"))
embeddings = HuggingFaceEmbeddings(model="BAAI/bge-small-zh-v1.5")

# 指标要实例化，并把 llm/embeddings 传进构造器
metrics = [
    Faithfulness(llm=llm),
    AnswerRelevancy(llm=llm, embeddings=embeddings),
    ContextPrecision(llm=llm),
    ContextRecall(llm=llm),
]
result = evaluate(dataset, metrics=metrics)
```

关键变化：① 指标从小写模块 → 大写类并实例化 ② `LangchainLLMWrapper` → `llm_factory` ③ `LangchainEmbeddingsWrapper` → `HuggingFaceEmbeddings`。

### 通用排查思路

遇到 `ModuleNotFoundError: langchain_community.xxx` 这类错，先查版本再动手：

```powershell
py -m pip show ragas langchain-community   # 先看版本
```

ragas 的 API 迭代极快、langchain-community 也频繁移除旧模块，两者版本错配是 RAG 评测最常见的坑。**先查版本，再决定升 ragas 还是降 langchain-community，不要盲目改代码。**

---

## 排查方法论复盘（比结果更值钱）

这次从「报 502」到「找到根因」走的路径，可以抽象成一个通用流程：

1. **先分清「代码问题」还是「环境问题」**：代码能跑通别的东西（`ollama run` 能通），说明代码逻辑大概率没错。
2. **做对照实验**：同一个接口，换调用方式（命令行 / PowerShell / Python），看谁通谁不通。差异点往往就是线索。
3. **查「中间层」**：502 且服务端日志没有记录 → 请求没到服务端 → 怀疑代理/防火墙/VPN。
4. **查环境差异**：不同语言对「系统代理」的处理方式不同（.NET 读注册表绕过名单，Python 只读 `NO_PROXY` 环境变量）。
5. **验证修复**：改动后用最小可复现脚本验证，再端到端跑一遍。

**教训**：排查时容易被「冷启动」「IPv6」「重复进程」等表象带偏，最后发现是**环境变量的锅**。以后遇到「同一请求不同客户端结果不同」，优先怀疑代理。

---

## 附：最终修复代码（两个文件顶部都加了）

```python
import os
# 修复1：系统代理会拦截 httpx 的本地请求导致 502，让本地直连
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
# 修复2：huggingface.co 被墙，改用国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 注意：以上必须在 import sentence_transformers / 创建 OpenAI 客户端之前设置
```

---

## 面试要点（这些能聊）

1. **「为什么调本地服务要设 NO_PROXY」** → Python 的 httpx/requests 不读 Windows 系统代理的绕过名单，只认 `NO_PROXY` 环境变量；VPN 开着就会把 localhost 请求吞掉。
2. **「国内下载 Hugging Face 模型怎么办」** → `HF_ENDPOINT=https://hf-mirror.com`，国内标配。
3. **「RAG 生产化要考虑什么」** → 除了检索效果，还有：模型冷启动（预热）、请求重试、超时、代理/网络环境。这次的 `warm_up` 预热和 `generate` 重试就是这个思路的落地。
4. **「base_url 用 localhost 还是 127.0.0.1」** → 用 `127.0.0.1` 更稳，避免 Windows 上 localhost 解析成 IPv6(::1) 而服务只绑 IPv4 导致的间歇性连不上。
5. **「RAGAS 报错怎么定位」** → ragas 依赖 langchain-community 但不锁版本，遇到 `ModuleNotFoundError` 先 `pip show` 对版本，通常降 `langchain-community<0.4.2` 或升 ragas 即可。
