# Day 15~17：大模型 LoRA 微调（合同领域问答）

> 目标：把 Qwen2.5-7B-Instruct 在「合同/劳动法问答」数据上做 LoRA 微调，并在本地 RTX 5070（12G 显存）上跑通全流程：**准备数据 → 装 LLaMA-Factory → 训练 → 评测 → 部署（Ollama）**。

## 本目录文件一览

| 文件 | 作用 |
|---|---|
| `make_data.py` | 用 20 条模板批量生成 500 条训练数据 → `data.json` |
| `dataset_info.json` | 把 `data.json` 注册成 LLaMA-Factory 能识别的数据集 `contract_qa` |
| `train_lora.bat` | Windows 训练脚本（`py -m llamafactory.cli`） |
| `train_lora.sh` | Linux / WSL2 训练脚本（内容与 .bat 一致） |
| `eval.py` | 微调前后对比评测（20 题，走本地 Ollama） |
| `Modelfile` | 把微调产物导入 Ollama 的配置示例 |

---

## 一、准备数据

```powershell
cd E:\project\ai-learning-journey\finetune
py make_data.py
```

- 产出 `data.json`（500 条），每条格式：

  ```json
  {"instruction": "……问题……", "input": "", "output": "……标准答案……"}
  ```

- 覆盖 20 个知识点：试用期辞退赔偿、经济补偿金 N、违法解除 2N、服务期违约金、竞业限制、保密协议、合同解除、未签合同双倍工资、加班费、年休假、工伤、三期保护、仲裁时效等。
- 可扩展：同目录放一个 `seed.json`（格式同 `data.json`），脚本会先读它、再用模板补足到 500 条。
- **说明**：数据为学习演示用途，法条引用以正式法律文本为准，非法律意见。

---

## 二、安装 LLaMA-Factory 并训练

### 1. 安装

```powershell
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

> 建议新建一个 Python 3.10/3.11 虚拟环境（conda 或 venv）再装，避免污染系统环境。
> 推荐用 `uv` 或 `conda` 管理；RTX 5070 需要较新的 CUDA/`torch`（Blackwell 架构，sm_120）。

### 2. 训练

1. 打开 `train_lora.bat`，把 `LLAMA_FACTORY` 改成你 clone 出来的实际路径。
2. 确认 `make_data.py` 已运行、`data.json` 已生成。
3. 运行 `train_lora.bat`（Linux 用 `bash train_lora.sh`）。

训练完成后，LoRA 适配器（只有几十 MB，不是整模型）保存在 `output/qwen2.5-7b-contract-lora/`。

### 训练参数速查

| 参数 | 取值 | 为什么 |
|---|---|---|
| `--finetuning_type lora` | lora | 只训练少量适配器参数，12G 显存可行 |
| `--lora_rank 8` | 8 | LoRA 低秩维度，越大可训练参数越多 |
| `--lora_alpha 16` | 16 | 缩放系数，常取 `2 × rank` |
| `--quantization_bit 4` | 4 | **关键**：QLoRA，把基座量化到 4bit 才能塞进 12G |
| `--learning_rate 1e-4` | 1e-4 | LoRA 常用学习率，比全量微调(2e-5)稍大 |
| `--num_train_epochs 3.0` | 3 | 小数据多跑几轮，但过多会过拟合 |
| `--per_device_train_batch_size 1` | 1 | 单卡显存小，batch 只能开 1 |
| `--gradient_accumulation_steps 8` | 8 | 等效 batch = 1 × 8 = 8，补偿小 batch |
| `--gradient_checkpointing` | 开 | 用计算换显存，进一步省显存 |
| `--max_seq_length 512` | 512 | 问答较短，缩短序列省显存 |
| `--template qwen` | qwen | Qwen2.5 的对话模板 |

---

## 三、评测（微调前后对比）

### 1. 准备基线模型（微调前）

```powershell
ollama pull qwen2.5:7b
```

### 2. 把微调产物导入 Ollama（微调后）

训练产出的只是 LoRA 适配器，要先**合并回基座模型**并导出 GGUF，再 `ollama create`。

在 LLaMA-Factory 目录执行（合并 + 导出 GGUF，需要安装 llama.cpp）：

```powershell
py -m llamafactory.cli export `
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct `
    --adapter_name_or_path E:\project\ai-learning-journey\finetune\output\qwen2.5-7b-contract-lora `
    --template qwen `
    --export_dir E:\project\ai-learning-journey\finetune\merged `
    --export_size 7 `
    --export_legacy_format False
```

然后把 `Modelfile` 里的 `FROM` 改成导出得到的 `.gguf` 路径，执行：

```powershell
cd E:\project\ai-learning-journey\finetune
ollama create qwen2.5-contract-lora -f Modelfile
ollama list   # 确认能看到 qwen2.5-contract-lora
```

### 3. 跑评测

打开 `eval.py`，确认 `AFTER_MODEL` 与 `ollama create` 起的名字一致（默认 `qwen2.5-contract-lora`），然后：

```powershell
py eval.py            # 评测全部 20 题
py eval.py --limit 5  # 只跑前 5 题，快速试跑
```

输出：控制台精简对比表 + 5 组典型样例全文 + `eval_results.json` / `eval_report.md`。

> 提示：Qwen2.5-7B-Instruct 本身法律常识就不差，500 条小数据带来的提升可能「有但不明显」，
> 这是正常现象——**强基座 + 小数据**的收益本来就需要仔细看差异点，也是一条重要经验。

---

## 四、部署（Ollama Modelfile 示例）

见 `Modelfile`，核心内容：

```
FROM ./merged/qwen2.5-7b-contract-lora-merged.gguf
SYSTEM """你是一名熟悉《中华人民共和国劳动合同法》的法律助手……"""
PARAMETER temperature 0.3
PARAMETER num_ctx 4096
```

导入并运行：

```powershell
ollama create qwen2.5-contract-lora -f Modelfile
ollama run qwen2.5-contract-lora "试用期被辞退，公司要赔钱吗？"
```

---

## 常见坑

1. **下载模型 10060 超时**：`train_lora.bat/.sh` 里已设 `HF_ENDPOINT=https://hf-mirror.com`，国内镜像。
2. **12G 显存 OOM**：确认加了 `--quantization_bit 4`；仍不行就 `--max_seq_length 256`，或改用 Qwen2.5-3B（把 `--model_name_or_path` 换成 `Qwen/Qwen2.5-3B-Instruct`，可去掉 4bit 量化）。
3. **RTX 5070 报 bitsandbytes 相关错误**：Blackwell（sm_120）需要较新的 `bitsandbytes`（0.45+）和 `torch`（2.7+），升级依赖版本。
4. **Windows 上编译/驱动问题**：LLaMA-Factory 官方更推荐 **WSL2**，Windows 原生遇到装不上的坑时改用 WSL2 + `train_lora.sh`。
5. **`--dataset contract_qa` 找不到**：确认 `--dataset_dir` 指向本目录、`dataset_info.json` 和 `data.json` 都在里面。

---

## 学习要点（面试能聊）

- **为什么 LoRA 而不是全量微调**：12G 显存装不下 7B 全量梯度，LoRA 冻结基座、只训练低秩适配器（几十 MB），大幅省显存、省算力、易迁移。
- **QLoRA 是什么**：LoRA + 4bit 量化基座，进一步把 7B 从 ~14G 压到 ~5G，让消费级显卡也能训 7B。
- **梯度累积**：显存只够 batch=1 时，用 `gradient_accumulation_steps=8` 累积 8 步再更新，等效 batch=8，训练更稳。
- **灾难性遗忘**：微调后通用能力可能下降；数据质量差/轮数过多会加剧，评测里「微调后更差」往往就是它。
