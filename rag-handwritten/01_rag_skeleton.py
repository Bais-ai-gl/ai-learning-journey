# Day 8：手写 RAG（骨架版 · 留白练习）
# 目标：不借助 LangChain 等框架，手写 RAG 的 5 个环节
# 依赖：py -m pip install sentence-transformers faiss-cpu numpy openai
# 用法：先在 ./docs 目录放几个 .txt 文档，再运行  py 01_rag_skeleton.py
#
# 5 个环节（对应下方 5 个函数）：
#   ① 加载文档 load_documents
#   ② 分块 chunk_text
#   ③ 向量化 embed_chunks
#   ④ 建索引 + 检索 build_index / retrieve
#   ⑤ 生成 generate
#
# 用法说明：每个函数里都有一处 `raise NotImplementedError`，请你把那一行换成真正实现。
# 每个 TODO 下面都给了"提示"，照着提示写 1~3 行即可。卡住再看 01_rag_solution.py。

import os
# 关键修复：本机开了系统代理(127.0.0.1:7897)，openai 库底层的 httpx 默认会走代理，
# 导致本地 Ollama 请求(127.0.0.1:11434)被代理拦截、返回 502。
# 设置 NO_PROXY 让 httpx 直连本地地址（必须在创建任何客户端之前设置）。
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"

# 关键修复2：huggingface.co 在国内不稳定/被墙，下载模型会超时(10060)。
# 改用国内镜像 hf-mirror.com（必须在加载模型之前设置）。
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ---------- ① 加载文档 ----------
def load_documents(directory):
    """读取目录下所有 .txt / .md 文件，返回 list[str]（每个文件一个字符串）"""
    docs = []
    # TODO(1): 遍历 directory 下的 *.txt 和 *.md 文件，读出文本内容加入 docs
    # 提示：
    #   for p in Path(directory).glob("*.txt"):
    #       docs.append(p.read_text(encoding="utf-8"))
    # 记得 .md 文件也照做一遍。
    raise NotImplementedError("① 加载文档：请实现")
    return docs


# ---------- ② 分块 ----------
def chunk_text(text, chunk_size=500, overlap=50):
    """把长文本切成若干块，返回 list[str]。
    每块 chunk_size 个字符，相邻块重叠 overlap 个字符。"""
    chunks = []
    # TODO(2): 用 while 循环切块
    # 提示：
    #   start = 0
    #   while start < len(text):
    #       chunks.append(text[start:start+chunk_size])
    #       start += chunk_size - overlap   # 步长 = 块大小 - 重叠
    raise NotImplementedError("② 分块：请实现")
    return chunks


# ---------- ③ 向量化 ----------
def embed_chunks(chunks, model):
    """把每个文本块转成向量，返回 numpy 数组，形状 (n, dim)"""
    # TODO(3): 调用 model.encode，注意要 normalize_embeddings=True（归一化后内积=余弦相似度）
    # 提示：return model.encode(chunks, normalize_embeddings=True)
    raise NotImplementedError("③ 向量化：请实现")


# ---------- ④ 建索引 + 检索 ----------
def build_index(vectors):
    """用 faiss 建索引，返回 index"""
    # TODO(4): 用 faiss.IndexFlatIP（内积），然后 index.add(vectors)
    # 提示：
    #   dim = vectors.shape[1]
    #   index = faiss.IndexFlatIP(dim)      # IP=Inner Product 内积
    #   index.add(vectors.astype("float32"))
    #   return index
    raise NotImplementedError("④ 建索引：请实现")


def retrieve(index, query_vec, chunks, top_k=3):
    """检索与 query_vec 最相似的 top_k 个块，返回 [(相似度, 文本块), ...]"""
    # TODO(5): 调 index.search 拿到分数和下标，按下标取回对应 chunks
    # 提示：
    #   query_vec = query_vec.astype("float32").reshape(1, -1)
    #   scores, idxs = index.search(query_vec, top_k)
    #   return [(float(scores[0][i]), chunks[idxs[0][i]]) for i in range(len(idxs[0]))]
    raise NotImplementedError("④ 检索：请实现")


# ---------- ⑤ 生成 ----------
def generate(question, context_chunks, client, model_name):
    """把检索到的块拼进 prompt，调大模型生成带引用的回答，返回 str"""
    # TODO(6): 拼 prompt（把 context_chunks 编号，要求模型引用来源、不知道就说不）再调模型
    # 提示：
    #   context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))
    #   prompt = f"根据以下资料回答问题，资料没有就答'不知道'，不要编造：\n\n资料：\n{context}\n\n问题：{question}\n\n请回答并注明引用编号如 [1][2]。"
    #   r = client.chat.completions.create(model=model_name, messages=[{"role":"user","content":prompt}], temperature=0.3)
    #   return r.choices[0].message.content
    raise NotImplementedError("⑤ 生成：请实现")


# ---------- 主流程（已写好，不用改） ----------
def main():
    # 0. 准备模型和客户端
    print("正在加载 embedding 模型（首次会下载约 100MB）...")
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    # 生成用本地 Ollama（无需 API Key）；要换智谱/通义就改这两行
    # 用 127.0.0.1 而非 localhost：Windows 上 localhost 可能解析成 IPv6(::1)，而 Ollama 有时只绑 IPv4，会连不上报 502
    client = OpenAI(api_key="ollama", base_url="http://127.0.0.1:11434/v1")
    model_name = "qwen2.5:7b"

    # 1. 加载 + 分块
    docs = load_documents("./docs")
    all_chunks = []
    for d in docs:
        all_chunks += chunk_text(d)
    print(f"加载 {len(docs)} 篇文档，切出 {len(all_chunks)} 块")

    # 2. 向量化
    vectors = embed_chunks(all_chunks, model)
    print("向量形状：", vectors.shape)

    # 3. 建索引
    index = build_index(vectors)

    # 4. 提问 → 检索 → 生成
    question = input("\n请输入你的问题：")
    q_vec = model.encode([question], normalize_embeddings=True)
    hits = retrieve(index, q_vec, all_chunks, top_k=3)

    print("\n命中的 3 块（相似度）：")
    for score, chunk in hits:
        print(f"  [{score:.3f}] {chunk[:60]}...")

    answer = generate(question, [c for _, c in hits], client, model_name)
    print("\n回答：\n", answer)


if __name__ == "__main__":
    main()
