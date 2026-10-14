# Day 10：RAG 评测 · 用 RAGAS 跑忠实度 / 相关性指标
# 用法：py 03_ragas_eval.py
# 依赖：py -m pip install ragas langchain-openai langchain-huggingface sentence-transformers
# 注意：ragas 0.4.3 的 evaluate() 只认「旧式单例指标」（from ragas.metrics import faithfulness）。
#       新的 ragas.metrics.collections 指标类继承不同基类，会报 TypeError，暂不能配合 evaluate() 用。

import os
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings


def build_samples():
    """构造评测样本。真实项目里这些来自你 RAG 系统的真实问答记录。"""
    return [
        SingleTurnSample(
            user_input="星火科技成立于哪一年？",
            retrieved_contexts=["星火科技成立于 2018 年，是一家专注企业级人工智能解决方案的公司。"],
            response="星火科技成立于 2018 年。",
            reference="2018 年",
        ),
        SingleTurnSample(
            user_input="星火科技的核心产品有哪些？",
            retrieved_contexts=["核心产品包括智能客服系统、知识库问答平台和数据分析助手。"],
            response="核心产品有智能客服系统、知识库问答平台和数据分析助手。",
            reference="智能客服系统、知识库问答平台、数据分析助手",
        ),
        SingleTurnSample(
            user_input="公司的使命是什么？",
            retrieved_contexts=["公司的使命是用人工智能技术帮助企业降本增效。"],
            response="帮助企业降本增效。",
            reference="用人工智能技术帮助企业降本增效",
        ),
    ]


def main():
    # 评测用 LLM（本地 Ollama）；judge 模型建议用更强的，本地 7B 也能跑通示意
    llm = LangchainLLMWrapper(ChatOpenAI(
        model="qwen2.5:7b",
        base_url="http://127.0.0.1:11434/v1",
        api_key="ollama",
        temperature=0,
    ))
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5"
    ))

    # 旧式单例指标直接传，不要实例化；llm/embeddings 通过 evaluate() 传入
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    dataset = EvaluationDataset(samples=build_samples())

    print("开始评测（本地模型，可能较慢）...")
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )

    print("\n=== RAGAS 评测结果 ===")
    print(result)
    # 常用指标解读：
    #   faithfulness    忠实度：回答是否都基于检索到的资料（越低越爱编造）
    #   answer_relevancy 答案相关性：回答是否切题
    #   context_precision 上下文精确率：检索到的资料里相关项排得靠不靠前
    #   context_recall   上下文召回率：相关资料是否被检索到


if __name__ == "__main__":
    main()

# 思考：
# 1. 为什么 RAG 要单独做评测，而不是只看「回答看着对不对」？
# 2. faithfulness 和 answer_relevancy 各回答什么问题？谁更容易被「幻觉」拉低？
# 3. 生产环境里，RAGAS 的评测数据（question/contexts/answer/reference）从哪来？
