#!/usr/bin/env python3
"""
Phase 3: Speculative Decoding 系统性测试
测试不同 draft/target 组合的接受率和实际吞吐量
"""
import csv
import re
import subprocess
import time
from pathlib import Path

LLAMA_SPEC = str(Path("~/Documents/Codes/llama.cpp/build/bin/llama-speculative").expanduser())
LLAMA_CLI  = str(Path("~/Documents/Codes/llama.cpp/build/bin/llama-cli").expanduser())
RESULTS_DIR = Path("~/Documents/Codes/FreeExplore/data").expanduser()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

Q_DIR = Path("~/Documents/mlx-community/Qwen3.5-4B-quants").expanduser()
MLX_DIR = Path("~/Documents/mlx-community").expanduser()

# 测试模型
MODELS = {
    "9B_Q8":    str(MLX_DIR / "Qwen3.5-9B-UD-Q8_K_XL/Qwen3.5-9B-UD-Q8_K_XL.gguf"),
    "0.8B_Q8":  str(MLX_DIR / "Qwen3.5-0.8B-GGUF-UD-Q8_K_XL/Qwen3.5-0.8B-UD-Q8_K_XL.gguf"),
    "2B_Q8":    str(MLX_DIR / "Qwen3.5-2B-GGUF-UD-Q8_K_L/Qwen3.5-2B-UD-Q8_K_XL.gguf"),
    "4B_Q8":    str(Q_DIR / "Qwen3.5-4B-Q8_0.gguf"),
    "4B_Q6":    str(Q_DIR / "Qwen3.5-4B-Q6_K.gguf"),
    "4B_Q4":    str(Q_DIR / "Qwen3.5-4B-Q4_K_M.gguf"),
    "4B_Q3":    str(Q_DIR / "Qwen3.5-4B-Q3_K_M.gguf"),
    "4B_Q2":    str(Q_DIR / "Qwen3.5-4B-Q2_K.gguf"),
    "35B_Q4":   str(MLX_DIR / "Qwen3.5-35B-A3B-GGUF-UD-Q4_K_XL/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"),
}

# 标准测试 prompt（无思维链模式的简单续写）
PROMPTS = {
    "code": "def fibonacci(n):\n    \"\"\"Return nth Fibonacci number.\"\"\"\n    if n <= 1:\n        return n\n    return",
    "text": "Large language models have transformed natural language processing by enabling",
    "math": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals",
}

N_PREDICT = 200
DRAFT_K = [4, 8]
TEMPERATURES = [0.0]  # greedy for max acceptance rate

def run_baseline(target_key: str, prompt: str, n_predict: int) -> dict | None:
    """无 SD 的基准速度"""
    target_path = MODELS.get(target_key)
    if not target_path or not Path(target_path).exists():
        return None

    cmd = [LLAMA_CLI, "-m", target_path, "-p", prompt, "-n", str(n_predict),
           "--temp", "0", "--perf", "--no-display-prompt"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr

    tg_match = re.search(r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens per second", output)
    if tg_match:
        tg_tps = float(tg_match.group(3))
        return {"type": "baseline", "target": target_key, "draft": "none",
                "draft_k": 0, "tg_tps": tg_tps, "accept_rate": None, "prompt_type": "text"}
    return None

def run_sd(target_key: str, draft_key: str, draft_k: int, prompt: str, temp: float = 0.0) -> dict | None:
    """运行推测采样并收集指标"""
    target_path = MODELS.get(target_key)
    draft_path = MODELS.get(draft_key)
    if not target_path or not draft_path:
        return None
    if not Path(target_path).exists() or not Path(draft_path).exists():
        print(f"  [SKIP] {target_key}←{draft_key}: 文件不存在")
        return None

    cmd = [LLAMA_SPEC, "-m", target_path, "-md", draft_path,
           "-p", prompt, "-n", str(N_PREDICT),
           "--draft", str(draft_k), "--temp", str(temp), "--perf"]

    print(f"[{time.strftime('%H:%M:%S')}] SD: {draft_key}→{target_key} (k={draft_k}, T={temp})...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    output = result.stdout + result.stderr

    # 保存原始输出
    fname = f"sd_{target_key}_{draft_key}_k{draft_k}.txt"
    (RESULTS_DIR / "raw" / fname).write_text(output)

    # 解析接受率
    accept_match = re.search(r"accept\s*=\s*([\d.]+)%", output)
    n_drafted_match = re.search(r"n_drafted\s*=\s*(\d+)", output)
    n_accept_match = re.search(r"n_accept\s*=\s*(\d+)", output)

    # 解析 draft 速度
    draft_eval = re.search(r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs.*?([\d.]+)\s*tokens per second", output)
    # 解析总时间（用 target 的 total time = LAST occurrence）
    # draft section first, target section last → use findall, take last
    total_time_matches = re.findall(r"total time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", output)
    total_time = None
    if total_time_matches:
        # Use last match = target model's total time
        class _M:
            def __init__(self, groups): self._g = groups
            def group(self, i): return self._g[i-1]
        total_time = _M(total_time_matches[-1])

    accept_rate = float(accept_match.group(1)) / 100 if accept_match else None
    n_drafted = int(n_drafted_match.group(1)) if n_drafted_match else None
    n_accept = int(n_accept_match.group(1)) if n_accept_match else None

    # 用 total_time 计算 effective TPS
    eff_tps = None
    if total_time:
        total_ms = float(total_time.group(1))
        n_tokens = int(total_time.group(2))
        eff_tps = n_tokens / (total_ms / 1000)

    print(f"  accept={accept_rate:.1%} n_drafted={n_drafted} n_accept={n_accept} eff_tps={eff_tps:.1f}" if eff_tps else
          f"  accept={accept_rate}")

    return {
        "type": "speculative",
        "machine": "M2Ultra_192GB",
        "target": target_key,
        "draft": draft_key,
        "draft_k": draft_k,
        "temperature": temp,
        "n_drafted": n_drafted,
        "n_accept": n_accept,
        "accept_rate": accept_rate,
        "eff_tps": eff_tps,
    }


def run_all_experiments():
    results = []
    out_csv = RESULTS_DIR / "phase3_sd_results.csv"

    # 实验矩阵
    experiments = [
        # (target, draft, k) - 系统性测试
        # 1. 不同 draft 尺寸 → 9B target
        ("9B_Q8", "0.8B_Q8", 4),
        ("9B_Q8", "0.8B_Q8", 8),
        ("9B_Q8", "2B_Q8", 4),
        ("9B_Q8", "2B_Q8", 8),
        # 2. 自推测采样：4B 不同量化
        ("4B_Q8", "4B_Q2", 4),
        ("4B_Q8", "4B_Q3", 4),
        ("4B_Q8", "4B_Q4", 4),
        ("4B_Q8", "4B_Q6", 4),
        # 3. 大 target：35B
        ("35B_Q4", "0.8B_Q8", 4),
        ("35B_Q4", "4B_Q4", 4),
    ]

    for target_key, draft_key, k in experiments:
        for prompt_name, prompt_text in PROMPTS.items():
            r = run_sd(target_key, draft_key, k, prompt_text)
            if r:
                r["prompt_type"] = prompt_name
                results.append(r)
                with open(out_csv, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)

    print(f"\n=== Phase 3 完成 {len(results)} 个实验 | 结果: {out_csv} ===")
    return results


if __name__ == "__main__":
    print("=== Phase 3: Speculative Decoding 实验 ===")
    run_all_experiments()
