# Day 10：重排（Rerank）· bge-reranker-base 把 top-20 精排成 top-5
# 用法：py 02_rerank.py
# 依赖：py -m pip install sentence-transformers faiss-cpu numpy
# 目的：向量检索是「粗排」（快但只看整体语义），重排是「精排」（慢但逐字比对 query 和文档）。
#       先粗排 top-20，再用交叉编码器精排取 top-5，召回质量明显提升。

import os
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder


def load_documents(directory):
    return [p.read_text(encoding="utf-8") for p in sorted(Path(directory).glob("*.txt"))]


def chunk_text(text, chunk_size=500, overlap=50):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def main():
    docs = load_documents("../rag-handwritten/docs")
    all_chunks = []
    for d in docs:
        all_chunks += chunk_text(d)
    print(f"共 {len(all_chunks)} 块")

    # ① 粗排：bi-encoder（bge-small）向量检索，快，取 top-20
    bi_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    vectors = bi_model.encode(all_chunks, normalize_embeddings=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype("float32"))

    # ② 精排：cross-encoder（bge-reranker）逐对打分，慢但准
    #   CrossEncoder 会同时接收 (query, doc) 并输出相关性分数，比向量内积更精细
    print("加载 reranker（首次会下载约 1GB）...")
    reranker = CrossEncoder("BAAI/bge-reranker-base")

    question = input("\n请输入问题：")
    q_vec = bi_model.encode([question], normalize_embeddings=True)
    _, idxs = index.search(q_vec.astype("float32").reshape(1, -1), 20)

    candidates = [all_chunks[i] for i in idxs[0] if i >= 0]

    # 交叉编码器打分：[(query, doc)] 批量输入，返回每个 pair 的相关性分数
    pairs = [[question, doc] for doc in candidates]
    scores = reranker.predict(pairs)

    # 按 rerank 分数降序取 top-5
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

    print("\n=== 重排后 top5 ===")
    for score, doc in ranked[:5]:
        print(f"  [{score:.3f}] {doc[:50]}...")

    print("\n=== 对比：纯向量检索第 1 名（重排前）===")
    print(f"  {candidates[0][:50]}...")


if __name__ == "__main__":
    main()

# 思考：
# 1. bi-encoder 和 cross-encoder 的本质区别？（一个把 query/doc 各自编码再点积，一个把两者拼一起过网络）
# 2. 为什么不直接对所有文档用 reranker？（性能：reranker 太慢，只能精排少量候选）
# 3. 重排能解决「检索到的 top-20 里没有正确答案」的问题吗？（不能，重排只排序不召回）
