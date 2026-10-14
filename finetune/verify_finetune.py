# 验证微调效果：直接加载「基座模型 + LoRA 适配器」做推理对比（不经过 GGUF/Ollama）
# 用法：py verify_finetune.py
import os
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
os.environ["HF_HUB_OFFLINE"] = "1"   # 离线，只用本地缓存

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 直接用 ModelScope 的本地缓存路径（避免联网下载）
BASE = "C:/Users/23829/.cache/modelscope/models/Qwen--Qwen2.5-3B-Instruct/snapshots/master"
ADAPTER = "E:/project/ai-learning-journey/finetune/output/qwen2.5-3b-contract-lora"
SYSTEM = "你是一名熟悉《中华人民共和国劳动合同法》的法律助手，回答专业、准确、简洁。"

QUESTIONS = [
    "公司以「试用期不符合录用条件」辞退了我，但从来没给过考核记录，公司需要赔偿吗？",
    "我在公司干了3年2个月，现在公司要裁员，经济补偿金怎么算？",
]


def load_model():
    print("加载基座模型 + LoRA 适配器（约 1 分钟）...")
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    return model, tokenizer


def ask(model, tokenizer, question):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.3, top_p=0.8)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


if __name__ == "__main__":
    model, tokenizer = load_model()
    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n===== 问题 {i} =====\n{q}\n")
        print(f"----- 微调后回答 -----\n{ask(model, tokenizer, q)}\n")
