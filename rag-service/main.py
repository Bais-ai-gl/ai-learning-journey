# -*- coding: utf-8 -*-
"""
Day 13（项目整合）：把 Day 8 手写 RAG 包装成 FastAPI 服务
Day 14（Docker 部署）：配合 Dockerfile / requirements.txt / README.md 打包运行

功能：POST /ask 接收 {"question": "..."}，返回 {"answer": "...", "sources": ["...", "..."]}

用法（本地）：
    cd rag-service
    py -m pip install -r requirements.txt
    py -m uvicorn main:app --host 127.0.0.1 --port 8000
    # 或直接运行：py main.py

测试：
    curl -X POST http://127.0.0.1:8000/ask ^
         -H "Content-Type: application/json" ^
         -d "{\"question\":\"星火科技的主打产品是什么？\"}"
"""

import os
# 关键修复1：本机若开了系统代理(如 127.0.0.1:7897)，openai 库底层 httpx 会把
# 本地 Ollama(127.0.0.1:11434) 也走代理导致 502。设置 NO_PROXY 让 httpx 直连本地。
# 必须在创建任何客户端之前设置。
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
# 关键修复2：huggingface.co 在国内不稳定/被墙，下载 embedding 模型会超时。
# 改用镜像 hf-mirror.com（必须在加载模型之前设置）。
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import faiss
import numpy as np
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ============================ 配置区（可用环境变量覆盖） ============================
# Ollama 地址：本地用 127.0.0.1；Docker 容器内要用宿主机地址（见 README 踩坑记录）。
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}/v1"
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
TOP_K = int(os.environ.get("TOP_K", "3"))
# 检索相似度下限：低于该值的块视为“不相关”，用于“无检索结果”兜底。
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.25"))


def _resolve_docs_dir() -> str:
    """确定文档目录：环境变量 DOCS_DIR > 本目录 ./docs > 兄弟目录 rag-handwritten/docs"""
    if os.environ.get("DOCS_DIR"):
        return os.environ["DOCS_DIR"]
    local = Path(__file__).parent / "docs"
    if local.is_dir():
        return str(local)
    sibling = Path(__file__).parent.parent / "rag-handwritten" / "docs"
    if sibling.is_dir():
        return str(sibling)
    return str(local)


DOCS_DIR = _resolve_docs_dir()

# ============================ 日志 ============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag-service")


# ======================= 手写 RAG 环节（复用 Day 8 的 5 个环节）=======================
def load_documents(directory):
    """环节1：读取目录下所有 .txt / .md 文件，返回 list[str]"""
    docs = []
    for p in sorted(Path(directory).glob("*.txt")) + sorted(Path(directory).glob("*.md")):
        docs.append(p.read_text(encoding="utf-8"))
    return docs


def chunk_text(text, chunk_size=500, overlap=50):
    """环节2：把长文本切成若干块，相邻块重叠 overlap 个字符"""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def embed_chunks(chunks, model):
    """环节3：把文本块转成向量，归一化后内积 = 余弦相似度"""
    return model.encode(chunks, normalize_embeddings=True)


def build_index(vectors):
    """环节4：用 faiss 内积索引（配合归一化向量 = 余弦相似度）"""
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors.astype("float32"))
    return index


def retrieve(index, query_vec, chunks, top_k=TOP_K):
    """环节5：检索 top_k 个最相似的块，返回 [(score, chunk), ...]。
    相比 Day 8 增加 idx<0 保护：当 top_k 超过总块数时 faiss 会返回 -1，需跳过。"""
    query_vec = query_vec.astype("float32").reshape(1, -1)
    scores, idxs = index.search(query_vec, top_k)
    hits = []
    for i in range(len(idxs[0])):
        idx = idxs[0][i]
        if idx < 0:  # faiss 用 -1 表示“没有更多结果”
            continue
        hits.append((float(scores[0][i]), chunks[idx]))
    return hits


def warm_up(model_name):
    """预热模型：用 Ollama 原生 /api/generate 加载。
    /v1 兼容接口在模型未加载时会直接返回 502，而原生接口会耐心等模型加载完。"""
    logger.info("正在预热大模型（冷加载约几秒~几十秒，请耐心等待）...")
    try:
        r = httpx.post(
            f"http://{OLLAMA_HOST}/api/generate",
            json={"model": model_name, "prompt": "你好", "stream": False},
            timeout=180,
        )
        if r.status_code == 200:
            logger.info("模型预热完成")
            return True
        logger.warning(f"[预热] 返回状态 {r.status_code}：{r.text[:200]}")
        return False
    except Exception as e:
        logger.warning(f"[预热] 失败：{type(e).__name__} {e}")
        return False


def generate(question, context_chunks, client, model_name):
    """环节6：把检索结果拼进 prompt，要求模型引用来源、不知道就说不知道。
    模型冷启动可能瞬时报 502，自动重试（最多 5 次），全部失败则抛异常由上层兜底。"""
    context = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context_chunks))
    prompt = (
        "根据下面的资料回答问题。如果资料中没有答案，请回答“资料中没有相关信息”，不要编造。\n\n"
        f"资料：\n{context}\n\n"
        f"问题：{question}\n\n"
        "请回答，并在末尾注明引用了哪几条资料（如 [1][2]）。"
    )
    for attempt in range(1, 6):
        try:
            r = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                timeout=120,  # 单次调用超时上限，避免请求无限挂起
            )
            return r.choices[0].message.content
        except Exception as e:
            logger.warning(f"[生成] 第 {attempt} 次调用失败：{type(e).__name__} {e}")
            if attempt == 5:
                raise
            time.sleep(15)


# ============================ 全局状态（启动时填充） ============================
class RAGState:
    """保存启动时加载好的资源，避免每个请求都重新加载模型（很慢）。"""

    def __init__(self):
        self.embedding_model = None   # embedding 模型（bge-small-zh）
        self.chunks = []              # 所有文本块
        self.index = None             # FAISS 索引
        self.client = None            # OpenAI 兼容客户端（指向 Ollama）
        self.llm_ready = False        # 大模型是否预热成功（供 /health 查看）
        self.error = None             # 初始化失败原因


state = RAGState()

# 兜底文案：模型未加载 / 超时 / 服务异常时返回
FALLBACK_ANSWER = "抱歉，RAG 服务当前不可用（模型未加载或调用超时），请稍后重试或检查 Ollama 服务是否已启动。"


def init_rag():
    """启动时初始化：加载文档 -> 分块 -> 向量化 -> 建索引 -> 建客户端 -> 预热"""
    # 1) 加载 embedding 模型（首次会从 hf-mirror 下载约 100MB）
    logger.info(f"加载 embedding 模型：{EMBED_MODEL}（首次会下载约 100MB）...")
    state.embedding_model = SentenceTransformer(EMBED_MODEL)

    # 2) 加载文档并分块
    logger.info(f"读取文档目录：{DOCS_DIR}")
    docs = load_documents(DOCS_DIR)
    all_chunks = []
    for d in docs:
        all_chunks += chunk_text(d)
    logger.info(f"加载 {len(docs)} 篇文档，切出 {len(all_chunks)} 块")
    state.chunks = all_chunks

    # 3) 向量化 + 4) 建索引
    vectors = embed_chunks(all_chunks, state.embedding_model)
    logger.info(f"向量形状：{vectors.shape}")
    state.index = build_index(vectors)

    # 5) 建大模型客户端并预热
    state.client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)
    state.llm_ready = warm_up(LLM_MODEL)
    if state.llm_ready:
        logger.info("RAG 初始化完成，服务就绪 ✅")
    else:
        logger.warning("大模型预热失败，生成环节将走兜底，请检查 Ollama 是否启动/模型是否已 pull")


# ============================ FastAPI 应用 ============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期：启动时初始化 RAG，退出时打印日志。
    用 try/except 包住初始化：即使模型下载失败，进程也不崩溃，/ask 会返回兜底文案。"""
    try:
        init_rag()
    except Exception as e:
        state.error = f"{type(e).__name__}: {e}"
        logger.exception(f"RAG 初始化失败：{state.error}")
    yield
    logger.info("服务关闭")


app = FastAPI(title="手写 RAG 服务（Day 13/14）", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    """健康检查：查看模型是否就绪、块数、错误信息（可用于 Docker HEALTHCHECK）"""
    return {
        "status": "ok",
        "llm_ready": state.llm_ready,
        "chunks": len(state.chunks),
        "docs_dir": DOCS_DIR,
        "error": state.error,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    logger.info(f"收到问题：{req.question[:80]}")

    # 0) 参数校验：空问题直接兜底
    question = (req.question or "").strip()
    if not question:
        logger.warning("问题为空，返回兜底")
        return AskResponse(answer="问题不能为空，请输入有效的问题。", sources=[])

    # 1) 服务未就绪（embedding 模型或索引加载失败）-> 兜底
    if state.index is None or state.embedding_model is None:
        logger.warning(f"服务未就绪，返回兜底。原因：{state.error}")
        return AskResponse(answer=FALLBACK_ANSWER, sources=[])

    # 2) 检索（环节5）
    try:
        q_vec = state.embedding_model.encode([question], normalize_embeddings=True)
        hits = retrieve(state.index, q_vec, state.chunks, top_k=TOP_K)
    except Exception as e:
        logger.exception(f"检索失败：{type(e).__name__} {e}")
        return AskResponse(answer=FALLBACK_ANSWER, sources=[])

    # 3) 无检索结果：所有命中块都低于相似度阈值 -> 兜底
    valid = [(s, c) for s, c in hits if s >= MIN_SCORE]
    if not valid:
        top_score = hits[0][0] if hits else 0.0
        logger.info(f"无有效检索结果（最高分 {top_score:.3f} < {MIN_SCORE}），返回兜底")
        return AskResponse(answer="资料中没有相关信息，无法回答该问题。", sources=[])

    for s, c in valid:
        logger.info(f"命中块（相似度 {s:.3f}）：{c[:50]}...")

    # 4) 生成（环节6）：大模型客户端不存在 -> 兜底；存在则调用，失败兜底
    if state.client is None:
        logger.warning("大模型客户端不可用，返回兜底")
        return AskResponse(answer=FALLBACK_ANSWER, sources=[c for _, c in valid])

    try:
        answer = generate(question, [c for _, c in valid], state.client, LLM_MODEL)
    except Exception as e:
        logger.exception(f"生成失败：{type(e).__name__} {e}")
        # 生成失败也把检索到的资料附上，方便排查“检索没问题、生成超时”的情况
        return AskResponse(answer=FALLBACK_ANSWER, sources=[c for _, c in valid])

    logger.info(f"返回回答（{len(answer)} 字）")
    return AskResponse(answer=answer, sources=[c for _, c in valid])


if __name__ == "__main__":
    # 直接运行：py main.py 也能启动，等价于 uvicorn main:app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
