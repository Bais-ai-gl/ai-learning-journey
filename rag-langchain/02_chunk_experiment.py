# Day 9：分块策略实验（chunk_size × overlap）
# 用法：py 02_chunk_experiment.py
# 依赖：py -m pip install langchain langchain-community langchain-text-splitters langchain-huggingface faiss-cpu sentence-transformers
#
# 目标：对比 5 组 (chunk_size, chunk_overlap) 的「分块数量」和「检索命中」，输出 markdown 表格。
# 只做 分块 → 向量化 → 检索（不生成，无需 Ollama）。

import os
# 关键修复：系统代理会拦截 httpx 的本地请求；huggingface 走国内镜像
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 内置测试语料（8 段）。
# 说明：真实的 3 篇 txt 每篇 < 256 字符，任何 chunk_size 都只切 1 块、无法体现差异，
#       故这里用更长的内置语料模拟"真实知识库"，让 chunk_size 的差别真正显现。
CORPUS_SECTIONS = [
    "星火科技成立于 2018 年，总部位于北京，在上海、深圳设有分公司。公司专注于企业级人工智能解决方案，核心产品包括智能客服系统、知识库问答平台和数据分析助手。公司拥有 200 多名员工，其中研发人员占比超过 60%。2024 年公司完成 B 轮融资，估值超过 10 亿元人民币。公司的使命是用人工智能技术帮助企业降本增效，目前已服务超过 500 家企业客户。",
    "智能客服系统是星火科技的主打产品，主要功能包括自动回答客户常见问题、7x24 小时在线服务、支持多轮对话、支持接入企业微信和网页。系统基于大语言模型和检索增强生成技术，能够根据企业知识库自动生成回答，回答准确率超过 92%。系统支持私有化部署，也提供云端 SaaS 版本，客户可以根据自身情况灵活选择。",
    "收费模式按照坐席数量和调用量计费：基础版每月每坐席 99 元，专业版每月每坐席 199 元，企业版按需定制。首次购买可享受首年 8 折优惠，续费支持月付和年付两种方式。对于超过 100 坐席的大客户，还可以申请阶梯折扣。",
    "如何申请试用？请访问官网填写试用申请表单，销售人员会在 1 个工作日内联系您。试用期 14 天，期间支持接入 10 个坐席，试用结束后可以无缝升级为正式版本，历史对话记录和数据都会完整保留。",
    "售后支持时间是每周一至周五 9 点到 18 点，紧急问题可拨打 400 热线，客服承诺 5 分钟内响应。企业版客户还享有专属技术支持和 7x24 小时值班服务。普通问题也可以通过官网在线工单提交，通常在 2 个工作日内得到回复。",
    "数据是否安全？所有数据均加密存储，公司通过 ISO27001 信息安全认证。系统支持私有化部署，数据可以不出企业内网，满足金融、政务、医疗等行业的合规要求。数据库每日自动备份，支持异地容灾。",
    "系统底层采用检索增强生成（RAG）架构，先通过向量检索找到相关资料，再交给大模型生成回答。向量化使用 bge-small-zh-v1.5 模型，生成使用 qwen2.5 大语言模型。知识库支持增量更新、标签分类和细粒度权限管理，管理员可以在后台实时查看检索命中率。",
    "商务合作请联系 sales@sparkai.com，招聘请投递 hr@sparkai.com。更多产品动态和行业案例请访问公司官网 www.sparkai.com，或关注官方微信公众号「星火科技」。公司在北京、上海、深圳均设有办公地点，欢迎预约参观。",
]

# 5 个测试问题：(问题, 命中关键词)。检索 top3 里出现关键词即判定"命中"。
TEST_QUESTIONS = [
    ("星火科技是哪一年成立的？", "2018"),
    ("智能客服系统支持接入哪些渠道？", "企业微信"),
    ("申请试用的试用期是多久？", "14 天"),
    ("专业版每月每坐席多少钱？", "199"),
    ("紧急问题可以拨打哪个电话？", "400"),
]

# 5 组实验配置 (chunk_size, chunk_overlap)
CONFIGS = [(256, 0), (512, 50), (512, 100), (1024, 50), (1024, 100)]


class HashEmbeddings:
    """离线兜底向量：字符 bigram 哈希成 256 维，仅演示流程，语义效果不如真实模型。"""
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
    """向量化：优先用 bge-small-zh-v1.5；下载失败退回哈希向量兜底。"""
    try:
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as e:
        print(f"[兜底] 加载真实 embedding 失败（{type(e).__name__}），改用哈希向量（仅演示流程）")
        return HashEmbeddings()


def load_corpus():
    """把 8 段拼成一个长文档再交给 splitter 切块（单文档切块，块数才随 chunk_size 变化）。"""
    return [Document(page_content="\n\n".join(CORPUS_SECTIONS), metadata={"source": "内置语料"})]


def run_one_config(chunk_size, chunk_overlap, docs, embeddings):
    """跑一组配置：切块 → 建库 → 5 问检索，返回 (分块数, [(top1命中, top3命中)...])"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    store = FAISS.from_documents(chunks, embeddings)
    results = []
    for q, kw in TEST_QUESTIONS:
        top1_hit = kw in store.similarity_search(q, k=1)[0].page_content
        top3_hit = any(kw in d.page_content for d in store.similarity_search(q, k=3))
        results.append((top1_hit, top3_hit))
    return len(chunks), results


def main():
    embeddings = build_embeddings()
    docs = load_corpus()

    print("| 配置(chunk/overlap) | 分块数 | Q1 | Q2 | Q3 | Q4 | Q5 | Top1命中 | Top3命中 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for size, ov in CONFIGS:
        n, results = run_one_config(size, ov, docs, embeddings)
        top1 = sum(1 for r in results if r[0])
        top3 = sum(1 for r in results if r[1])
        cells = [f"{size}/{ov}", str(n)] + ["√" if r[1] else "×" for r in results] + [f"{top1}/5", f"{top3}/5"]
        print("| " + " | ".join(cells) + " |")

    print("\n说明：Q1~Q5 列 = top3 是否命中（√=命中 ×=未命中）。")
    print("结论：见文件末尾注释「分块策略总结」。")
    print("\n提示：本语料较短，各配置命中率可能差异不大；差异主要体现在分块数量与检索粒度上。")


if __name__ == "__main__":
    main()


# ================= 分块策略总结（面试可聊） =================
# 1. chunk_size 怎么选：
#    - 小块(256)：每块信息密度高，检索定位精准，适合 FAQ、关键词问答；
#      但语义/上下文易被切断，问题需要跨块信息时容易漏。
#    - 大块(1024)：上下文完整，回答需要背景/推理时效果好；
#      但一块里混入无关信息会"稀释"关键句，降低相似度打分，且更耗 token。
#    - 经验值：中文 512 是常用起点；英文/代码可 256~512。
# 2. chunk_overlap 怎么选：
#    - overlap 让相邻块共享边界内容，避免一句话被从中间切开导致检索不到；
#    - overlap 越大冗余越多、块数越多、索引越慢；50~100 是常见区间。
# 3. 什么时候用小块 / 大块：
#    - 短问答、FAQ、精确检索        → 小块(256) + overlap 50
#    - 长文档、摘要、需要上下文的问答 → 大块(1024) + overlap 100
#    - 更高级做法：父子块(parent-child)、按语义分块、混合多粒度检索。
