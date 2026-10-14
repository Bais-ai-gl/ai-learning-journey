# Day 17：LoRA 微调 · 微调前后对比评测（合同领域问答）
# 目标：用 20 个合同领域测试题，分别问「微调前模型」和「微调后模型」，对比回答质量
# 依赖：py -m pip install openai   （Ollama 走 OpenAI 兼容接口，本地无需 API Key）
# 运行：py eval.py           （评测全部 20 题）
#       py eval.py --limit 5 （只评测前 5 题，快速试跑）
#
# 前提：
#   1) 已安装并启动 Ollama（https://ollama.com），且已拉取基线模型：ollama pull qwen2.5:7b
#   2) 微调产物已通过「合并 + GGUF 导出 + ollama create」导入 Ollama（见 README.md 四、部署）
#
# 输出：
#   - 控制台：精简对比表 + 5 组典型样例（完整回答）
#   - 文件：eval_results.json（原始数据）、eval_report.md（完整 Markdown 对比报告）

import os
import sys
import time
import json
from pathlib import Path

# 关键修复（和 Day 8 同理）：本机开了系统代理时，httpx 默认走代理会把本地 Ollama 请求
# 拦下返回 502。设置 NO_PROXY 让本地地址直连（必须在创建任何客户端之前设置）。
os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"

import httpx                       # openai 库自带，无需额外安装
from openai import OpenAI

# ================= 配置（按你的环境改） =================
BASE_URL = "http://127.0.0.1:11434/v1"   # 用 127.0.0.1 而非 localhost，避免 IPv6 坑
BEFORE_MODEL = "qwen2.5:3b"              # 微调前：Ollama 官方基线模型（3B）
AFTER_MODEL = "qwen2.5-3b-contract-lora" # 微调后：ollama create 时起的名字（占位，改这里）
# ↑ 怎么填 AFTER_MODEL：
#   用 ollama create qwen2.5-3b-contract-lora -f Modelfile 导入后，
#   用 `ollama list` 能看到的那个名字填到这里即可。

SYSTEM_PROMPT = "你是一名熟悉《中华人民共和国劳动合同法》的法律助手，回答专业、准确、简洁，尽量分点说明。"

# 5 组「典型样例」的题号（从 0 开始），评测结束后会完整打印这几组全文
SAMPLE_INDEXES = [0, 3, 7, 12, 19]

# ================= 20 个合同领域测试题（与训练数据同领域、不同问法） =================
TEST_QUESTIONS = [
    "公司以「试用期不符合录用条件」辞退了我，但从来没给过我任何考核记录，公司需要赔偿吗？",
    "我在公司干了 3 年 2 个月，现在公司要裁员，我的经济补偿金应该怎么算？",
    "劳动合同里写着「离职要赔公司 10 万违约金」，这个条款合法吗？",
    "公司让我签了竞业限制，但离职后一分钱补偿都不给，我还要继续遵守吗？",
    "我提前 30 天提交了书面辞职申请，公司就是不批不放人，我该怎么办？",
    "我的劳动合同到期了，公司说不续签了，我能拿到经济补偿吗？",
    "入职 8 个月了公司一直没跟我签书面劳动合同，我能主张什么权利？",
    "保密协议里写着「保密义务永久有效」，这个约定有效吗？",
    "法律规定的竞业限制最长期限是多久？",
    "签 1 年劳动合同，试用期约定 6 个月合法吗？",
    "法定节假日加班，加班费应该是几倍工资？",
    "累计工作满 5 年，每年能休几天带薪年假？",
    "上班路上被车撞了，交警认定对方全责，这算工伤吗？",
    "公司知道女员工怀孕了，就以「无法胜任工作」为由辞退她，合法吗？",
    "什么情况下员工可以要求签无固定期限劳动合同？",
    "公司单方面把我从北京调到深圳，我可以拒绝吗？",
    "被公司拖欠工资，申请劳动仲裁有时间限制吗？",
    "一份合法的劳动合同必须具备哪些条款？",
    "病假期间的工资怎么发？",
    "劳务派遣和劳务外包有什么区别？",
]


def warm_up(client, model):
    """预热模型：用 Ollama 原生接口加载。
    模型冷启动时 /v1 兼容接口会直接返回 502，而原生 /api/generate 会耐心等加载完。"""
    print(f"正在预热 {model}（冷加载约几秒~几十秒，请耐心等待）...")
    try:
        r = httpx.post(
            "http://127.0.0.1:11434/api/generate",
            json={"model": model, "prompt": "你好", "stream": False},
            timeout=180,
        )
        if r.status_code == 200:
            print(f"  {model} 预热完成")
        else:
            print(f"  [{model}] 预热返回状态 {r.status_code}：{r.text[:200]}")
    except Exception as e:
        print(f"  [{model}] 预热失败：{type(e).__name__} {e}")


def model_exists(model):
    """用 Ollama /api/tags 检查某个模型是否已存在（微调后模型必须先部署才能评测）。"""
    try:
        r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=10)
        if r.status_code == 200:
            names = [m.get("name", "") for m in r.json().get("models", [])]
            return model in names
    except Exception:
        pass
    return False


def generate(client, model, question):
    """调一个模型回答一个问题，失败自动重试 5 次，最终失败返回错误信息（不中断整体评测）。"""
    for attempt in range(1, 6):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0.3,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [警告] {model} 第 {attempt} 次调用失败：{type(e).__name__} {e}")
            if attempt == 5:
                return f"[调用失败] {type(e).__name__}: {e}"
            time.sleep(15)


def clip(text, width):
    """把长文本截断成 width 个字符（用于控制台表格），超长加省略号。"""
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


def print_console_table(results):
    """控制台精简对比表（答案截断，完整内容见 eval_report.md）。"""
    w_q, w_a = 18, 30
    print("\n" + "=" * 90)
    print("微调前后对比表（精简，完整见 eval_report.md）")
    print("=" * 90)
    print(f"{'#':<3}{'问题'.ljust(w_q)}  {'微调前'.ljust(w_a)}  {'微调后'.ljust(w_a)}")
    print("-" * 90)
    for i, r in enumerate(results, 1):
        q = clip(r["question"], w_q).ljust(w_q)
        before = clip(r["before"], w_a).ljust(w_a)
        after = clip(r["after"], w_a).ljust(w_a)
        print(f"{i:<3}{q}  {before}  {after}")
        print("-" * 90)


def print_samples(results):
    """打印 5 组典型样例的完整问答，方便人工对比质量。"""
    print("\n" + "=" * 90)
    print("5 组典型样例（完整回答）")
    print("=" * 90)
    for idx in SAMPLE_INDEXES:
        if idx >= len(results):
            continue
        r = results[idx]
        print(f"\n【样例 {idx + 1}】问题：{r['question']}")
        print(f"\n—— 微调前 ({BEFORE_MODEL}) ——\n{r['before']}")
        print(f"\n—— 微调后 ({AFTER_MODEL}) ——\n{r['after']}")
        print("\n" + "-" * 90)


def save_reports(results):
    """保存原始数据 (json) 和完整对比报告 (markdown)。"""
    base = Path(__file__).parent
    (base / "eval_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 微调前后对比评测报告（合同领域问答）",
        "",
        f"- 微调前模型：`{BEFORE_MODEL}`",
        f"- 微调后模型：`{AFTER_MODEL}`",
        f"- 测试题数量：{len(results)}",
        "",
        "> 说明：本报告由 eval.py 自动生成，仅供学习参考；回答质量需人工判断。",
        "",
        "## 对比表（节选）",
        "",
        "| # | 问题 | 微调前 | 微调后 |",
        "|---|---|---|---|",
    ]
    for i, r in enumerate(results, 1):
        q = r["question"].replace("|", "\\|").replace("\n", " ")
        before = r["before"].replace("|", "\\|").replace("\n", "<br>")
        after = r["after"].replace("|", "\\|").replace("\n", "<br>")
        lines.append(f"| {i} | {q} | {before} | {after} |")

    lines += ["", "## 完整回答", ""]
    for i, r in enumerate(results, 1):
        lines.append(f"### {i}. {r['question']}")
        lines.append(f"\n**微调前（{BEFORE_MODEL}）**：\n\n{r['before']}\n")
        lines.append(f"**微调后（{AFTER_MODEL}）**：\n\n{r['after']}\n")

    (base / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已保存：{base / 'eval_results.json'}、{base / 'eval_report.md'}")


def main():
    # 可选：py eval.py --limit 5 只跑前 5 题快速试跑
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    questions = TEST_QUESTIONS[:limit] if limit else TEST_QUESTIONS

    client = OpenAI(api_key="ollama", base_url=BASE_URL)

    # 前置检查：微调后模型必须已存在（先完成 Day 16 微调 + Modelfile 部署）
    if not model_exists(AFTER_MODEL):
        print(f"\n[未找到微调后模型] {AFTER_MODEL}")
        print("评测需要「微调前」和「微调后」两个模型对比，但微调后模型还没导入 Ollama。")
        print("请先完成：Day 16 微调（train_lora.bat）→ 合并导出 → ollama create 导入（见 README.md「四、部署」）。")
        print("或临时把 AFTER_MODEL 改成已存在的模型名（如 qwen2.5:7b）只试跑评测流程。")
        sys.exit(1)

    # 预热两个模型，避免第一个问题撞上冷启动 502
    warm_up(client, BEFORE_MODEL)
    warm_up(client, AFTER_MODEL)

    results = []
    print(f"\n开始评测 {len(questions)} 道题……")
    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {q}")
        before = generate(client, BEFORE_MODEL, q)
        after = generate(client, AFTER_MODEL, q)
        results.append({"question": q, "before": before, "after": after})

    print_console_table(results)
    print_samples(results)
    save_reports(results)


if __name__ == "__main__":
    main()

# ================= 自测题 =================
# 1. 为什么评测要「同领域、不同问法」出题，而不是直接拿训练数据当测试题？
#    答：拿训练原题来测等于考「背诵」，只有换一种问法才能测出模型是否真的学会了规则（泛化）。
# 2. 为什么这里设置 NO_PROXY？不设会怎样？
#    答：本机开代理时，httpx 会把 127.0.0.1 的请求也送进代理导致 502（详见 notes/rag踩坑记录.md）。
# 3. temperature 为什么设为 0.3 而不是 1.0？
#    答：评测想比的是「知识掌握程度」，低温度让输出更确定、可复现，减少随机性干扰。
# 4. 如果「微调后」模型反而答得更差，可能是什么原因？
#    答：数据量太少导致过拟合、数据里答案有误、学习率/轮数过大导致遗忘通用能力（灾难性遗忘）、
#    或基座模型本身已很强（7B Instruct 本来就会答很多法律题，小数据集提升有限）。
