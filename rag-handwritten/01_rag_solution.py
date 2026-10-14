# Day 8：手写 RAG（参考答案版）
# 说明：先自己完成 01_rag_skeleton.py，卡住再看本文件对照。
# 依赖：py -m pip install sentence-transformers faiss-cpu numpy openai
# 运行：py 01_rag_solution.py

import os
# 关键修复：本机开了系统代理(127.0.0.1:7897)，openai 库底层的 httpx 默认会走代理，
# 导致本地 Ollama 请求(127.0.0.1:11434)被代理拦截、返回 502。
# 设置 NO_PROXY 让 httpx 直连本地地址（必须在创建任何客户端之前设置）。
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"

# 关键修复2：huggingface.co 在国内不稳定/被墙，下载模型会超时(10060)。
# 改用国内镜像 hf-mirror.com（必须在加载模型之前设置）。
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path
import time
import numpy as np
import faiss
import httpx   # openai 库自带，无需额外安装
from sentence_transformers import SentenceTransformer
from openai import OpenAI


def load_documents(directory):
    """读取目录下所有 .txt / .md 文件，返回 list[str]"""
    docs = []
    for p in sorted(Path(directory).glob("*.txt")) + sorted(Path(directory).glob("*.md")):
        docs.append(p.read_text(encoding="utf-8"))
    return docs


def chunk_text(text, chunk_size=500, overlap=50):
    """把长文本切成若干块，相邻块重叠 overlap 个字符"""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def embed_chunks(chunks, model):
    """把文本块转成向量，归一化后内积 = 余弦相似度"""
    return model.encode(chunks, normalize_embeddings=True)


def build_index(vectors):
    """用 faiss 内积索引（配合归一化向量 = 余弦相似度）"""
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors.astype("float32"))
    return index


def retrieve(index, query_vec, chunks, top_k=3):
    """检索 top_k 个最相似的块"""
    query_vec = query_vec.astype("float32").reshape(1, -1)
    scores, idxs = index.search(query_vec, top_k)
    return [(float(scores[0][i]), chunks[idxs[0][i]]) for i in range(len(idxs[0]))]


def generate(question, context_chunks, client, model_name):
    """把检索结果拼进 prompt，要求模型引用来源、不知道就说不知道"""
    context = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context_chunks))
    prompt = (
        "根据下面的资料回答问题。如果资料中没有答案，请回答“资料中没有相关信息”，不要编造。\n\n"
        f"资料：\n{context}\n\n"
        f"问题：{question}\n\n"
        "请回答，并在末尾注明引用了哪几条资料（如 [1][2]）。"
    )
    # 模型冷启动/加载中可能瞬时报 502，自动重试（RTX 5070 冷加载约 30~60 秒，间隔拉长）
    for attempt in range(1, 6):
        try:
            r = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return r.choices[0].message.content
        except Exception as e:
            print(f"[警告] 第 {attempt} 次调用失败：{type(e).__name__} {e}")
            if attempt == 5:
                raise
            time.sleep(15)


def warm_up(model_name):
    """预热模型：用 Ollama 原生接口加载。
    RTX 5070 上 /v1 兼容接口在模型未加载时会直接返回 502，
    而原生 /api/generate 会耐心等模型加载完，所以用它来预热。"""
    print("正在预热大模型（冷加载约几秒~几十秒，请耐心等待）...")
    try:
        r = httpx.post(
            "http://127.0.0.1:11434/api/generate",
            json={"model": model_name, "prompt": "你好", "stream": False},
            timeout=180,
        )
        if r.status_code == 200:
            print("模型预热完成")
        else:
            print(f"  [预热] 返回状态 {r.status_code}：{r.text[:200]}")
    except Exception as e:
        print(f"  [预热] 失败：{type(e).__name__} {e}")


def main():
    print("正在加载 embedding 模型（首次会下载约 100MB）...")
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    # 本地 Ollama；换 API 就改这里
    # 用 127.0.0.1 而非 localhost：Windows 上 localhost 可能解析成 IPv6(::1)，而 Ollama 有时只绑 IPv4，会连不上报 502
    client = OpenAI(api_key="ollama", base_url="http://127.0.0.1:11434/v1")
    model_name = "qwen2.5:7b"

    docs = load_documents("./docs")
    all_chunks = []
    for d in docs:
        all_chunks += chunk_text(d)
    print(f"加载 {len(docs)} 篇文档，切出 {len(all_chunks)} 块")

    vectors = embed_chunks(all_chunks, model)
    print("向量形状：", vectors.shape)

    index = build_index(vectors)

    question = input("\n请输入你的问题：")
    q_vec = model.encode([question], normalize_embeddings=True)
    hits = retrieve(index, q_vec, all_chunks, top_k=3)

    print("\n命中的 3 块（相似度）：")
    for score, chunk in hits:
        print(f"  [{score:.3f}] {chunk[:60]}...")

    warm_up(model_name)   # 生成前预热：确保模型已加载（/v1 冷启动会 502）
    answer = generate(question, [c for _, c in hits], client, model_name)
    print("\n回答：\n", answer)


if __name__ == "__main__":
    main()
