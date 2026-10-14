#!/usr/bin/env bash
# ============================================================
#  Day 16：LoRA 微调 Qwen2.5-7B-Instruct（合同领域问答）· Linux / WSL2 版
#
#  用法：
#    1) git clone https://github.com/hiyouga/LLaMA-Factory.git
#    2) cd LLaMA-Factory && pip install -e ".[torch,metrics]"
#    3) 修改下方 LLAMA_FACTORY 为实际路径
#    4) 先运行  python make_data.py  生成 data.json
#    5) bash train_lora.sh
#
#  关键点：12G 显存装不下 7B 全量模型，用 4-bit 量化（QLoRA），见 --quantization_bit 4。
# ============================================================
set -e

# ===== 1. 路径配置（按实际环境改）=====
LLAMA_FACTORY="$HOME/LLaMA-Factory"           # 改成你的 LLaMA-Factory 实际路径
DATA_DIR="$(cd "$(dirname "$0")" && pwd)"     # 脚本所在目录（data.json / dataset_info.json）

# ===== 2. 从 ModelScope 下载模型（国内 CDN，速度快）=====
export USE_MODELSCOPE_HUB=1
# HF 兜底：走镜像 + 禁用 xet
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_XET=1

echo "[1/3] 切换到 LLaMA-Factory 目录..."
cd "$LLAMA_FACTORY"

echo "[2/3] 激活虚拟环境（如使用，取消注释）"
# conda activate llamafactory
# source .venv/bin/activate

echo "[3/3] 开始 LoRA 微调..."
python -m llamafactory.cli train \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset contract_qa \
    --dataset_dir "$DATA_DIR" \
    --template qwen \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --learning_rate 1e-4 \
    --num_train_epochs 3.0 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --cutoff_len 512 \
    --logging_steps 10 \
    --save_steps 200 \
    --val_size 0.1 \
    --bf16 \
    --output_dir "$DATA_DIR/output/qwen2.5-3b-contract-lora"

echo ""
echo "[完成] LoRA 适配器已保存到：$DATA_DIR/output/qwen2.5-7b-contract-lora"
echo "下一步：按 README.md「三、评测」运行 eval.py 对比微调前后效果。"
