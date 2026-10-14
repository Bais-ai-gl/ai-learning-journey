# RAG 服务（Day 13 项目整合 + Day 14 Docker）

把 Day 8 手写的 RAG（`rag-handwritten/01_rag_solution.py`）包装成一个 **FastAPI 服务**，
对外只暴露一个 `POST /ask` 接口：传入问题，返回「答案 + 引用的资料块」。

## 项目背景

- 手写 RAG 只能通过命令行交互式问答，不方便被其他程序调用；
- Day 13 的目标是把它「服务化」：用 FastAPI 包一层 HTTP 接口；
- Day 14 的目标是把它「容器化」：用 Dockerfile 打包成镜像，随处可跑。

## 架构（文字流程图）

```
                     POST /ask {"question": "..."}
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │   FastAPI 服务 (main.py)   │
                    │                            │
                    │  1. 问题向量化 (bge-small) │
                    │  2. FAISS 检索 top_k 块    │
                    │  3. 拼装 Prompt            │
                    │  4. 调用 Ollama 生成       │
                    └───────┬────────────┬───────┘
                            │            │
                 ┌──────────▼───┐   ┌────▼────────────┐
                 │  FAISS 索引  │   │ Ollama(qwen2.5) │
                 │ (启动时构建)  │   │  (宿主机/外部)  │
                 └──────────────┘   └─────────────────┘
```

启动时一次性完成「加载文档 → 分块 → 向量化 → 建索引 → 预热大模型」，
请求来的时候只做「向量化问题 → 检索 → 生成」，避免每次请求重复加载模型。

## 运行方式

### 前置条件

1. 已安装并启动 Ollama，且已 `ollama pull qwen2.5:7b`；
2. Python 3.11（本地运行）。

### 方式一：本地运行

```bash
cd rag-service
py -m pip install -r requirements.txt
py -m uvicorn main:app --host 127.0.0.1 --port 8000
# 或直接：py main.py
```

首次运行会从 hf-mirror 下载 embedding 模型（约 100MB），请耐心等待。

### 方式二：Docker 运行

```bash
cd rag-service
docker build -t rag-service .

# Linux：用 host.docker.internal 访问宿主机上的 Ollama
docker run -d -p 8000:8000 \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_HOST=host.docker.internal:11434 \
  rag-service

# Mac / Windows（Docker Desktop）：host.docker.internal 已内置，无需 --add-host
docker run -d -p 8000:8000 -e OLLAMA_HOST=host.docker.internal:11434 rag-service
```

## 接口说明

### POST /ask

- 入参：`{"question": "星火科技的主打产品是什么？"}`
- 出参：`{"answer": "...", "sources": ["资料块1", "资料块2"]}`

curl 示例（Windows CMD 用 `^` 换行，PowerShell 用反引号 `` ` `` 换行）：

```bash
curl -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"如何申请试用？\"}"
```

### GET /health

健康检查，返回模型就绪状态、索引块数、初始化错误信息等：

```bash
curl http://127.0.0.1:8000/health
```

## 兜底策略（错误处理）

| 情况 | 处理方式 |
|------|----------|
| embedding 模型 / 索引未加载 | 返回兜底文案，服务不崩溃 |
| 检索结果相似度低于阈值（默认 0.25） | 返回「资料中没有相关信息」 |
| 大模型未加载 / 调用超时 / 502 | 生成环节重试 5 次后返回兜底文案，并附上已检索到的资料块 |
| 空问题 | 直接返回「问题不能为空」 |

## 踩坑记录

1. **代理拦截本地 Ollama**：openai 库底层 httpx 默认走系统代理，导致 `127.0.0.1:11434` 请求被代理转发返回 502。解决：设置 `NO_PROXY=127.0.0.1,localhost,::1`（在创建客户端之前）。
2. **huggingface 下载超时**：`huggingface.co` 在国内不稳定，下载 embedding 模型会超时。解决：设置 `HF_ENDPOINT=https://hf-mirror.com`。
3. **`/v1` 冷启动返回 502**：Ollama 的 OpenAI 兼容接口在模型未加载时会直接 502。解决：启动时用原生 `/api/generate` 预热模型，让模型先加载好。
4. **`localhost` 解析成 IPv6**：Windows 上 `localhost` 可能解析成 `::1`，而 Ollama 只绑 IPv4，会连不上。解决：统一用 `127.0.0.1`。
5. **Docker 里访问不到宿主机 Ollama**：容器内 `127.0.0.1` 指向容器自己。解决：用 `OLLAMA_HOST=host.docker.internal:11434`（Linux 需加 `--add-host=host.docker.internal:host-gateway`）。
6. **faiss 与 numpy 版本**：部分 faiss 旧轮子按 numpy 1.x 编译，requirements.txt 里把 numpy 锁定 `<2.0` 更稳妥。
7. **top_k 超过总块数**：faiss 会返回 `-1` 索引，`retrieve()` 里加了 `idx < 0` 保护，避免越界报错。

## 目录结构

```
rag-service/
├── main.py           # FastAPI 服务，复用 Day 8 手写 RAG 环节
├── requirements.txt  # 运行依赖
├── Dockerfile        # 镜像构建脚本
├── README.md         # 本文档
└── docs/             # 知识库文档（与 rag-handwritten/docs 同内容）
    ├── 公司介绍.txt
    ├── 产品说明.txt
    └── 常见问题.txt
```
