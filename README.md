# ai-learning-journey

目标：月薪 2 万 AI 工程师（大模型应用方向）· 学习旅程代码仓库

## 计划与进度

- 学习计划：《AI工程师学习计划.md》（48 周完整版）、《3周冲刺计划-详细版.md》（Day 1~21）
- 阶段 0（环境准备）：✅ 完成（GitHub 推送待办，见《阶段0-打卡清单.md》）
- 3 周冲刺计划（Day 1~21）：📝 代码已全部生成，逐日运行验证中
  - 学习顺序与文件对应见 [Day-文件对照表.md](Day-文件对照表.md)

## 目录结构（按 Day 顺序）

| 目录 | 内容 | Day |
|---|---|---|
| `python-basics/` | Python 快扫 10 个练习 | 1 |
| `pandas-practice/` | Pandas 数据练习 | 2 |
| `llm-playground/` | 大模型 API 实验 | 3 |
| `prompt-experiments/` | Prompt 工程 | 4 |
| `ml-basics/` | 机器学习最小集 | 5 |
| `deep-learning/` | PyTorch + 注意力 | 6~7 |
| `rag-handwritten/` | 手写 RAG | 8 |
| `rag-langchain/` | LangChain 重写 RAG + 分块实验 | 9 |
| `rag-advanced/` | 混合检索 + 重排 + RAGAS 评测 | 10 |
| `agent-handwritten/` | 手写 ReAct | 11 |
| `agent-langgraph/` | LangGraph 版 Agent | 12 |
| `rag-service/` | FastAPI 服务 + Docker 部署 | 13~14 |
| `finetune/` | LoRA 微调（数据 + 训练 + 评测） | 15~17 |
| `leetcode/` | LeetCode 10 题 | 19 |
| `notes/` | 八股笔记、踩坑记录、博客提纲 | 18~19 |
| `interview/` | 面试讲述稿 | 20 |
| `job-hunting/` | 复盘表、投递记录 | 21 |

## 使用方式

```bash
# 首次
git clone https://github.com/<你的用户名>/ai-learning-journey.git
# 每天
git add .
git commit -m "今天做了什么"
git push
```

## 环境注意（重要）

- 本机开了系统代理（Clash/樱花猫），所有调本地 Ollama 的脚本顶部已加 `NO_PROXY` 修复；下载 Hugging Face 模型已改用 `HF_ENDPOINT=https://hf-mirror.com`。
- 本地大模型：Ollama + `qwen2.5:7b`（`http://127.0.0.1:11434/v1`）。
- 详见 [notes/rag踩坑记录.md](notes/rag踩坑记录.md)。
