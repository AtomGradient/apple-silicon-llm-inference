#!/usr/bin/env python3
"""
跨设备 Speculative Decoding 实验
测试：113 作为 target (9B Q8) + 103/104 作为 draft compute backend via RPC
"""
import csv
import re
import subprocess
import time
from pathlib import Path

LLAMA_SPEC = str(Path("~/Documents/Codes/llama.cpp/build/bin/llama-speculative").expanduser())
RESULTS_DIR = Path("~/Documents/Codes/FreeExplore/data").expanduser()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MLX_DIR = Path("~/Documents/mlx-community").expanduser()

MODELS = {
    "9B_Q8":   str(MLX_DIR / "Qwen3.5-9B-UD-Q8_K_XL/Qwen3.5-9B-UD-Q8_K_XL.gguf"),
    "0.8B_Q8": str(MLX_DIR / "Qwen3.5-0.8B-GGUF-UD-Q8_K_XL/Qwen3.5-0.8B-UD-Q8_K_XL.gguf"),
    "2B_Q8":   str(MLX_DIR / "Qwen3.5-2B-GGUF-UD-Q8_K_L/Qwen3.5-2B-UD-Q8_K_XL.gguf"),
}

PROMPTS = {
    "code": "def fibonacci(n):\n    \"\"\"Return nth Fibonacci number.\"\"\"\n    if n <= 1:\n        return n\n    return",
    "text": "Large language models have transformed natural language processing by enabling",
    "math": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals",
}

RPC_SERVERS = {
    "local":   None,                        # 本地 SD（基线对比）
    "103_M1Max": "192.168.0.103:50052",     # M1 Max RPC
    "104_M2Pro": "192.168.0.104:50052",     # M2 Pro RPC
}

N_PREDICT = 200
DRAFT_K = 4


def parse_result(output: str) -> dict:
    accept_match = re.search(r"n_accept\s*=\s*(\d+)", output)
    draft_match  = re.search(r"n_drafted\s*=\s*(\d+)", output)
    total_matches = re.findall(r"total time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", output)

    eff_tps = None
    if total_matches:
        ms, n = float(total_matches[-1][0]), int(total_matches[-1][1])
        eff_tps = n / (ms / 1000)

    n_accept = int(accept_match.group(1)) if accept_match else None
    n_drafted = int(draft_match.group(1)) if draft_match else None
    accept_rate = n_accept / n_drafted if (n_accept and n_drafted and n_drafted > 0) else None

    return {"n_drafted": n_drafted, "n_accept": n_accept,
            "accept_rate": accept_rate, "eff_tps": eff_tps}


def run_experiment(target_key, draft_key, prompt_name, prompt_text, rpc_label, rpc_server):
    target_path = MODELS[target_key]
    draft_path = MODELS[draft_key]

    cmd = [LLAMA_SPEC,
           "-m", target_path,
           "-md", draft_path,
           "-p", prompt_text,
           "-n", str(N_PREDICT),
           "--draft", str(DRAFT_K),
           "--temp", "0",
           "--perf"]

    if rpc_server:
        cmd += ["--rpc", rpc_server, "-ngld", "99"]

    tag = f"SD {draft_key}→{target_key} rpc={rpc_label} prompt={prompt_name}"
    print(f"\n[{time.strftime('%H:%M:%S')}] {tag}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout + result.stderr

        # 保存原始输出
        raw_dir = RESULTS_DIR / "raw"
        raw_dir.mkdir(exist_ok=True)
        fname = f"crossdevice_{rpc_label}_{draft_key}_{prompt_name}.txt"
        (raw_dir / fname).write_text(output)

        parsed = parse_result(output)
        if parsed["eff_tps"]:
            print(f"  accept={parsed['accept_rate']:.2%}  eff_tps={parsed['eff_tps']:.1f}")
        else:
            print(f"  PARSE ERROR. Output snippet:\n{output[-500:]}")
        return {
            "type": "cross_device_sd",
            "machine": "M2Ultra_192GB",
            "target": target_key,
            "draft": draft_key,
            "draft_k": DRAFT_K,
            "rpc_backend": rpc_label,
            "rpc_server": rpc_server or "none",
            "prompt_type": prompt_name,
            **parsed
        }
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT")
        return None


def main():
    results = []
    out_csv = RESULTS_DIR / "phase3_cross_device_sd.csv"

    # 实验矩阵: 3 RPC 后端 × 1 draft × 3 prompts
    # "local" 是本地基准（无RPC，已有数据但重跑确认一致性）
    # 103/104 是跨设备
    experiments = [
        ("9B_Q8", "0.8B_Q8", "local",     None),
        ("9B_Q8", "0.8B_Q8", "103_M1Max", "192.168.0.103:50052"),
        ("9B_Q8", "0.8B_Q8", "104_M2Pro", "192.168.0.104:50052"),
    ]

    for target_key, draft_key, rpc_label, rpc_server in experiments:
        for prompt_name, prompt_text in PROMPTS.items():
            r = run_experiment(target_key, draft_key, prompt_name, prompt_text, rpc_label, rpc_server)
            if r:
                results.append(r)
                # 每次都写 CSV
                with open(out_csv, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)

    print(f"\n=== 跨设备 SD 实验完成 {len(results)} 个 | {out_csv} ===")
    # 输出汇总
    for rpc in ["local", "103_M1Max", "104_M2Pro"]:
        rows = [r for r in results if r["rpc_backend"] == rpc and r["eff_tps"]]
        if rows:
            avg_tps = sum(r["eff_tps"] for r in rows) / len(rows)
            avg_acc = sum(r["accept_rate"] for r in rows if r["accept_rate"]) / len(rows)
            print(f"  {rpc:15s}: avg_tps={avg_tps:.1f}  avg_accept={avg_acc:.2%}")


if __name__ == "__main__":
    main()
