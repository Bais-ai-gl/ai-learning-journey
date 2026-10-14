# 学习代码 · Day 对照表

> 每个代码文件开头都已标注「Day X」。下表按学习顺序排列（对应《3周冲刺计划-详细版.md》Day 1~21）。

| Day | 文件 | 内容 | 依赖 |
|---|---|---|---|
| **Day 1** | `python-basics/01_types_io.py` ~ `10_algorithm_mini.py` | Python 快扫 10 个练习 | 无 |
| **Day 2** | `pandas-practice/day2_pandas_tasks.py` | Pandas 5 任务 | pandas, matplotlib |
| **Day 3** | `llm-playground/01_api_basics.py` | 调大模型 API、temperature 实验 | openai + API Key |
| **Day 4** | `prompt-experiments/01_six_techniques.py` | Prompt 6 技巧 | openai + API Key |
| **Day 5** | `ml-basics/01_classification.py` | 逻辑回归 vs 随机森林 | scikit-learn |
| **Day 6** | `deep-learning/01_mlp_fashion.py` | PyTorch MLP 训练 Fashion-MNIST | torch |
| **Day 7** | `deep-learning/02_attention_numpy.py` | 手写单头注意力 | numpy |
| **Day 8** | `rag-handwritten/01_rag_skeleton.py` / `01_rag_solution.py` / `docs/*.txt` | 手写 RAG（骨架 + 答案 + 文档） | sentence-transformers, faiss-cpu, openai |
| **Day 9** | `rag-langchain/01_rag_langchain.py` / `02_chunk_experiment.py` | LangChain 重写 RAG + 分块实验 | langchain 系, faiss-cpu |
| **Day 10** | `rag-advanced/01_hybrid_retrieval.py` / `02_rerank.py` / `03_ragas_eval.py` | 混合检索 + 重排 + RAGAS 评测 | rank_bm25, sentence-transformers, ragas |
| **Day 11** | `agent-handwritten/01_react.py` | 手写 ReAct（Function Calling + 3 工具） | openai |
| **Day 12** | `agent-langgraph/01_agent_langgraph.py` | LangGraph 重写 ReAct | langgraph, langchain-openai |
| **Day 13** | `rag-service/main.py` | RAG 服务化（FastAPI POST /ask） | fastapi, uvicorn |
| **Day 14** | `rag-service/Dockerfile` / `requirements.txt` / `README.md` | Docker 部署 | docker |
| **Day 15** | `finetune/make_data.py` → `data.json` | LoRA 微调数据（500 条合同问答） | 无（标准库） |
| **Day 16** | `finetune/train_lora.bat` / `.sh` / `dataset_info.json` | LLaMA-Factory 微调 Qwen2.5-7B（QLoRA） | LLaMA-Factory |
| **Day 17** | `finetune/eval.py` / `Modelfile` | 微调前后对比评测 + 部署 | openai |
| **Day 18** | `notes/博客提纲.md` | 2 篇技术博客提纲 | — |
| **Day 19** | `leetcode/01_*.py` ~ `10_*.py` + `notes/八股笔记.md` | LeetCode 10 题 + 八股第一轮 | 无 |
| **Day 20** | `interview/自我介绍与项目讲述稿.md` | STAR 讲述稿 + 5 个为什么 | — |
| **Day 21** | `job-hunting/面试复盘表.md` | 复盘表 + 投递记录模板 | — |

## 依赖安装速查（都用 py，避免 WSL 别名坑）

```
py -m pip install pandas matplotlib                              # Day 2
py -m pip install openai                                         # Day 3/4/11/17
py -m pip install scikit-learn                                   # Day 5
py -m pip install torch torchvision                              # Day 6
py -m pip install sentence-transformers faiss-cpu numpy          # Day 8/10
py -m pip install langchain langchain-community langchain-text-splitters langchain-huggingface faiss-cpu  # Day 9
py -m pip install rank_bm25                                      # Day 10 混合检索
py -m pip install ragas langchain-openai                         # Day 10 评测
py -m pip install langgraph langchain-openai                     # Day 12
py -m pip install fastapi uvicorn                                # Day 13
```

## 运行方式（在对应目录下）

```
py 文件名.py
```

> - Day 3/4 需要先设置 API Key：`$env:OPENAI_API_KEY="你的key"`。
> - Day 8~13 的生成环节用本地 Ollama（无需 Key），脚本已内置 `NO_PROXY` 和 `HF_ENDPOINT` 修复。
> - Day 16（微调）需先安装 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)，本机 RTX 5070 12G 用 QLoRA（4bit）。
