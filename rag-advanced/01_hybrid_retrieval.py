# Day 10：混合检索 · BM25 + 向量检索 7:3 融合
# 用法：py 01_hybrid_retrieval.py
# 依赖：py -m pip install rank_bm25 sentence-transformers faiss-cpu numpy
# 目的：单靠向量检索会漏掉「关键词精确命中」，单靠 BM25 会漏掉「语义相近但用词不同」，
#       两者融合（7:3）能互补，是 RAG 从 60 分到 80 分的关键一步。

import os
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"      # 系统代理会拦截 httpx 本地请求
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"     # huggingface 国内镜像

from pathlib import Path
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ---------- 复用 Day 8 的文档加载 / 分块 ----------
def load_documents(directory):
    docs = []
    for p in sorted(Path(directory).glob("*.txt")):
        docs.append(p.read_text(encoding="utf-8"))
    return docs


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


# ---------- ① 关键词检索：BM25 ----------
def build_bm25(chunks):
    """BM25 需要先分词；中文这里用「按字符切」做最简分词（真项目用 jieba）"""
    tokenized = [list(c) for c in chunks]   # 把每块切成字符序列
    return BM25Okapi(tokenized), tokenized


def bm25_search(bm25, tokenized_chunks, question, top_k=20):
    scores = bm25.get_scores(list(question))
    # 取分数最高的 top_k 个下标（注意排除得分为 0 的）
    idxs = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in idxs if scores[i] > 0]


# ---------- ② 语义检索：向量 + FAISS ----------
def build_vector_index(vectors):
    index = faiss.IndexFlatIP(vectors.shape[1])          # IP 内积（配合归一化向量 = 余弦）
    index.add(vectors.astype("float32"))
    return index


def vector_search(index, query_vec, top_k=20):
    query_vec = query_vec.astype("float32").reshape(1, -1)
    scores, idxs = index.search(query_vec, top_k)
    return [(int(idxs[0][i]), float(scores[0][i])) for i in range(len(idxs[0])) if idxs[0][i] >= 0]


# ---------- ③ 融合：归一化后 7:3 加权 ----------
def fuse(vec_hits, bm25_hits, w_vec=0.7, w_bm25=0.3, top_k=5):
    """两种检索的分数量纲不同，先各自 min-max 归一化到 [0,1]，再按 7:3 加权"""
    def normalize(hits):
        if not hits:
            return {}
        vals = [s for _, s in hits]
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return {i: 1.0 for i, _ in hits}
        return {i: (s - lo) / (hi - lo) for i, s in hits}

    vec_n = normalize(vec_hits)
    bm25_n = normalize(bm25_hits)

    fused = {}
    for idx in set(vec_n) | set(bm25_n):
        fused[idx] = w_vec * vec_n.get(idx, 0.0) + w_bm25 * bm25_n.get(idx, 0.0)

    # 按融合分降序取 top_k
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]


def main():
    print("加载文档 + 建索引...")
    docs = load_documents("../rag-handwritten/docs")
    all_chunks = []
    for d in docs:
        all_chunks += chunk_text(d)
    print(f"共 {len(all_chunks)} 块")

    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    vectors = model.encode(all_chunks, normalize_embeddings=True)
    vec_index = build_vector_index(vectors)
    bm25, tokenized = build_bm25(all_chunks)

    question = input("\n请输入问题：")
    q_vec = model.encode([question], normalize_embeddings=True)

    vec_hits = vector_search(vec_index, q_vec, top_k=20)
    bm25_hits = bm25_search(bm25, tokenized, question, top_k=20)
    fused = fuse(vec_hits, bm25_hits)

    print("\n=== 纯向量 top3 ===")
    for idx, score in vec_hits[:3]:
        print(f"  [{score:.3f}] {all_chunks[idx][:50]}...")
    print("=== 纯 BM25 top3 ===")
    for idx, score in bm25_hits[:3]:
        print(f"  [{score:.3f}] {all_chunks[idx][:50]}...")
    print("=== 混合(7:3) top5 ===")
    for idx, score in fused:
        print(f"  [{score:.3f}] {all_chunks[idx][:50]}...")


if __name__ == "__main__":
    main()

# 思考：
# 1. 为什么「向量检索」和「BM25」各自会漏掉什么？举个具体例子。
# 2. 7:3 的权重怎么定？什么时候应该让 BM25 占更大比例？
# 3. 融合前为什么必须归一化？（提示：BM25 分数和余弦相似度的量纲完全不同）
