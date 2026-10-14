# Day 14：把 RAG 服务打成 Docker 镜像
# 用法：
#   docker build -t rag-service .
#   docker run -d -p 8000:8000 --add-host=host.docker.internal:host-gateway -e OLLAMA_HOST=host.docker.internal:11434 rag-service

# 基础镜像：python:3.11-slim（比完整版小很多，够用）
FROM python:3.11-slim

# 设置容器内工作目录为 /app，后面的 COPY/RUN/CMD 都以此为基准
WORKDIR /app

# 只先复制 requirements.txt 并安装依赖：
# 这样能利用 Docker 的“层缓存”——代码改动时，只要依赖没变，这一层就不用重新构建。
COPY requirements.txt .

# 用清华 PyPI 镜像加速下载（国内直连 pypi 可能很慢/被墙）
# --no-cache-dir 避免把 pip 缓存打进镜像，减小体积
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 把当前目录（rag-service/）所有文件复制进容器，包括 main.py 和 docs/
COPY . .

# 声明服务监听端口（仅文档用途，实际端口由 CMD 决定）
EXPOSE 8000

# 容器启动时运行 uvicorn；--host 0.0.0.0 让容器外（宿主机）能访问
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
