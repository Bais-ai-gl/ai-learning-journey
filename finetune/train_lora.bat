@echo off
REM ============================================================
REM  Day 16: LoRA fine-tune Qwen2.5-7B-Instruct (contract QA) - Windows
REM  NOTE: This .bat is ASCII-only on purpose. Chinese comments in a
REM        .bat file break cmd.exe on Chinese Windows (GBK vs UTF-8).
REM
REM  Prerequisites (first time only):
REM    1) Install LLaMA-Factory:
REM         git clone https://github.com/hiyouga/LLaMA-Factory.git
REM         cd LLaMA-Factory
REM         pip install -e ".[torch,metrics]"
REM    2) Set LLAMA_FACTORY below to your real LLaMA-Factory path
REM    3) Run:  py make_data.py   (generates data.json in this folder)
REM    4) Run this script
REM
REM  Key point: RTX 5070 has only 12G VRAM, 7B full model does NOT fit.
REM        Must use 4-bit quantization (QLoRA), see --quantization_bit 4.
REM ============================================================
setlocal

REM ===== 1. Path config (CHANGE THIS to your real LLaMA-Factory path) =====
set "LLAMA_FACTORY=E:\project\ai-learning-journey\finetune\LLaMA-Factory"
REM This script's folder = finetune project dir (data.json / dataset_info.json are here)
set "DATA_DIR=%~dp0"

REM ===== 2. Download model from ModelScope (China CDN, much faster than HF) =====
REM Qwen is Alibaba's model; ModelScope is the official China source.
set "USE_MODELSCOPE_HUB=1"
REM Fallback for any HF downloads: use mirror + disable xet
set "HF_ENDPOINT=https://hf-mirror.com"
set "HF_HUB_DISABLE_XET=1"

echo [1/3] Switching to LLaMA-Factory dir...
cd /d "%LLAMA_FACTORY%" || (echo ERROR: LLaMA-Factory dir not found, fix the path above & exit /b 1)

echo [2/3] Activating env (uncomment and set your env name if using conda/venv)
REM call conda activate llamafactory
REM call .venv\Scripts\activate.bat

echo [3/3] Starting LoRA fine-tune (12G VRAM + 4bit QLoRA)...
REM ===== 3. Training command (see README.md for each param) =====
REM --dataset contract_qa looks up dataset_info.json in --dataset_dir
REM to locate data.json. No need to edit LLaMA-Factory source.
py -m llamafactory.cli train ^
    --stage sft ^
    --do_train ^
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct ^
    --dataset contract_qa ^
    --dataset_dir "%DATA_DIR%" ^
    --template qwen ^
    --finetuning_type lora ^
    --lora_target all ^
    --lora_rank 8 ^
    --lora_alpha 16 ^
    --lora_dropout 0.1 ^
    --learning_rate 1e-4 ^
    --num_train_epochs 3.0 ^
    --per_device_train_batch_size 1 ^
    --gradient_accumulation_steps 8 ^
    --lr_scheduler_type cosine ^
    --warmup_ratio 0.1 ^
    --gradient_checkpointing ^
    --cutoff_len 512 ^
    --logging_steps 10 ^
    --save_steps 200 ^
    --val_size 0.1 ^
    --bf16 ^
    --output_dir "%DATA_DIR%output\qwen2.5-3b-contract-lora"

if errorlevel 1 (
    echo.
    echo [FAILED] Training error. See README.md "common issues" (VRAM / version / mirror).
    pause
    exit /b 1
)

echo.
echo [DONE] LoRA adapter saved to: %DATA_DIR%output\qwen2.5-7b-contract-lora
echo Next: run eval.py to compare before/after fine-tuning.
pause
endlocal
