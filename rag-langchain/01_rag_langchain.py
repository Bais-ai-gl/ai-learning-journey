# Day 9：用 LangChain 重写手写 RAG
# 用法：py 01_rag_langchain.py
# 依赖：py -m pip install langchain langchain-community langchain-text-splitters langchain-huggingface faiss-cpu sentence-transformers openai

import os
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"   # 系统代理会拦截 httpx 本地请求导致502
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # huggingface 国内镜像

import time
from pathlib import Path
import httpx                       # openai 库自带，无需额外安装
from openai import OpenAI
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 内置兜底文档（内容同 rag-handwritten/docs），./docs 缺失或为空时使用
MOCK_DOCS = {
    "公司介绍.txt": "星火科技成立于 2018 年，是一家专注于企业级人工智能解决方案的公司。公司总部位于北京，在上海、深圳设有分公司。核心产品包括智能客服系统、知识库问答平台和数据分析助手。公司拥有超过 200 名员工，其中研发人员占比超过 60%。2024 年公司完成了 B 轮融资，估值超过 10 亿元人民币。公司的使命是用人工智能技术帮助企业降本增效。",
    "产品说明.txt": "智能客服系统是星火科技的主打产品，主要功能包括：自动回答客户常见问题、7x24 小时在线服务、支持多轮对话、支持接入企业微信和网页。该系统基于大语言模型和检索增强生成技术，能够根据企业知识库自动生成回答，回答准确率超过 92%。系统支持私有化部署，也提供云端 SaaS 版本。收费模式按照坐席数量和调用量计费，基础版每月每坐席 99 元。",
    "常见问题.txt": "常见问题：如何申请试用？请访问官网填写试用申请表单，销售人员会在 1 个工作日内联系您。是否支持私有化部署？支持，企业版可部署在客户自己的服务器上。数据是否安全？所有数据均加密存储，公司通过 ISO27001 信息安全认证。如何收费？基础版每月每坐席 99 元，专业版每月每坐席 199 元，企业版按需定制。售后支持时间是每周一至周五 9 点到 18 点，紧急问题可拨打 400 热线。",
}
class HashEmbeddings:
    """离线兜底向量：bigram 哈希成 256 维，仅演示流程（同一进程内 hash 稳定可对齐）。"""
    def _embed(self, text):
        v = [0.0] * 256
        for i in range(len(text) - 1):
            v[hash(text[i:i + 2]) % 256] += 1.0
        n = (sum(x * x for x in v) ** 0.5) or 1.0
        return [x / n for x in v]
    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]
    def embed_query(self, text):
        return self._embed(text)

def build_embeddings():
    """③ 向量化：优先 bge-small-zh-v1.5，下载失败退回哈希向量兜底。"""
    try:
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # 归一化：L2 距离与余弦相似度等价
        )
    except Exception as e:
        print(f"[兜底] 加载真实 embedding 失败（{type(e).__name__}），改用哈希向量（仅演示流程）")
        return HashEmbeddings()

def load_documents(directory="./docs"):
    """① DocumentLoader：TextLoader 加载目录下所有 .txt。"""
    paths = sorted(Path(directory).glob("*.txt"))
    if paths:
        docs = []
        for p in paths:
            docs += TextLoader(str(p), encoding="utf-8").load()
        return docs
    print(f"[兜底] 未找到 {directory}/*.txt，改用内置模拟文档")
    return [Document(page_content=t, metadata={"source": n}) for n, t in MOCK_DOCS.items()]

def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    """② TextSplitter：按分隔符递归切块；默认无中文句读，需手动加中文标点。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    return splitter.split_documents(docs)

def build_vectorstore(chunks):
    """④ 索引 + 检索：FAISS 向量库 + Retriever(k=3)。"""
    embeddings = build_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return vector_store, retriever

def l2_to_cosine(l2):
    """归一化向量下 ||a-b||²=2-2cos，故 cos=1-d²/2；把 L2 距离换算成余弦方便对比。"""
    return max(0.0, 1.0 - (l2 * l2) / 2.0)

def warm_up(model_name):
    """预热：/v1 冷启动会 502，用原生 /api/generate 耐心等加载（同 Day 8）。"""
    print("正在预热大模型（冷加载约几秒~几十秒）...")
    try:
        r = httpx.post("http://127.0.0.1:11434/api/generate",
                       json={"model": model_name, "prompt": "你好", "stream": False},
                       timeout=180)
        print("模型预热完成" if r.status_code == 200 else f"  [预热] 状态 {r.status_code}：{r.text[:200]}")
    except Exception as e:
        print(f"  [预热] 失败：{type(e).__name__} {e}")

def generate(question, context_chunks, client, model_name):
    """⑤ 生成：拼 prompt 要求引用来源、不知道就说不（同 Day 8，带重试 + 离线兜底）。"""
    context = "\n\n".join(f"[{i + 1}] {c.page_content}" for i, c in enumerate(context_chunks))
    prompt = ("根据下面的资料回答问题。如果资料中没有答案，请回答“资料中没有相关信息”，不要编造。\n\n"
              f"资料：\n{context}\n\n问题：{question}\n\n"
              "请回答，并在末尾注明引用了哪几条资料（如 [1][2]）。")
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
            if attempt < 5:
                time.sleep(15)
    print("[提示] 大模型不可用，返回离线兜底答案（仅拼接检索结果）")
    return "（离线兜底）资料中相关内容：\n" + "\n".join(
        f"[{i + 1}] {c.page_content[:100]}" for i, c in enumerate(context_chunks))

def main():
    print("=" * 60)
    print("Day 9：LangChain 重写手写 RAG（对照 Day 8）")
    print("=" * 60)
    docs = load_documents("./docs")
    chunks = chunk_documents(docs)                 # 默认 500/50，与手写版对齐
    print(f"加载 {len(docs)} 篇文档，切出 {len(chunks)} 块")
    vector_store, retriever = build_vectorstore(chunks)

    # 本地 Ollama（同 Day 8）；127.0.0.1 避免 IPv6 解析问题
    client = OpenAI(api_key="ollama", base_url="http://127.0.0.1:11434/v1")
    model_name = "qwen2.5:7b"
    question = input("\n请输入你的问题：")
    top_docs = retriever.invoke(question)                    # 方式一：Retriever 标准接口
    scored = vector_store.similarity_search_with_score(question, k=3)  # 方式二：拿分数对比

    print("\n命中的 3 块（余弦相似度，越大越相关）：")
    for doc, l2 in scored:
        print(f"  [{l2_to_cosine(l2):.3f}] {doc.page_content[:60]}...")

    warm_up(model_name)
    print("\n回答：\n", generate(question, top_docs, client, model_name))

    print("\n" + "=" * 60)
    print("与手写版(Day 8)对照：")
    print("  分块：字符窗口 → 递归分隔符(按句子边界)｜索引：IndexFlatIP → FAISS VectorStore")
    print("  检索：index.search → retriever.invoke(标准接口)｜相似度：内积=余弦 ↔ L2 换算余弦(等价)")
    print("=" * 60)

    print("\n【自测/思考】\n"
          "1. RecursiveCharacterTextSplitter 的 separators 为什么要给中文加 。，！？\n"
          "2. 手写版用内积(IndexFlatIP)、本版用 L2 距离，为什么二者等价？前提是什么？\n"
          "3. as_retriever 相比直接调 similarity_search 有什么好处？\n"
          "4. 想换成 OpenAI 的 text-embedding-3-small，要改几行代码？体现什么设计？")

if __name__ == "__main__":
    main()
