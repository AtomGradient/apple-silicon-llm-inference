#!/usr/bin/env python3
"""
Phase 1: llama-bench 多精度测试（正确 CSV 解析）
支持 113 (M2 Ultra) 和 103 (M1 Max)
"""

import csv
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ============================================================
# 配置
# ============================================================
LLAMA_BENCH = str(Path("~/Documents/Codes/llama.cpp/build/bin/llama-bench").expanduser())
LLAMA_QUANTIZE = str(Path("~/Documents/Codes/llama.cpp/build/bin/llama-quantize").expanduser())
RESULTS_DIR = Path("~/Documents/Codes/FreeExplore/data").expanduser()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 9B Q8 源文件（113 用于量化）
SRC_9B = Path("~/Documents/mlx-community/Qwen3.5-9B-UD-Q8_K_XL/Qwen3.5-9B-UD-Q8_K_XL.gguf").expanduser()
QUANT_DIR = Path("~/Documents/mlx-community/Qwen3.5-9B-quants").expanduser()

MACHINE_NAME = "M2Ultra_192GB"  # 在 103 上运行时改为 M1Max_32GB

# 测试参数
N_PROMPT = 512
N_GEN = 128
N_RUNS = 3

# 目标量化级别
QUANT_TYPES = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K"]

# 测试模型列表（label: path）
def get_models_113():
    models = {}
    # 9B 各量化
    models["9B_Q8"] = str(SRC_9B)
    for q in QUANT_TYPES:
        p = QUANT_DIR / f"Qwen3.5-9B-{q}.gguf"
        models[f"9B_{q}"] = str(p)
    # Draft 候选模型
    models["0.8B_Q8"] = str(Path("~/Documents/mlx-community/Qwen3.5-0.8B-GGUF-UD-Q8_K_XL/Qwen3.5-0.8B-UD-Q8_K_XL.gguf").expanduser())
    models["2B_Q8"] = str(Path("~/Documents/mlx-community/Qwen3.5-2B-GGUF-UD-Q8_K_L/Qwen3.5-2B-UD-Q8_K_XL.gguf").expanduser())
    return models

def get_models_103():
    return {
        "9B_Q8": str(Path("~/Documents/mlx-community/Qwen3.5-9B-UD-Q8_K_XL/Qwen3.5-9B-UD-Q8_K_XL.gguf").expanduser()),
        "0.6B_Q4": str(Path("~/Documents/mlx-community/Qwen3-0.6B-q4_k_m.gguf").expanduser()),
    }


# ============================================================
# 量化生成
# ============================================================
def generate_quants():
    if not SRC_9B.exists():
        print(f"[ERROR] 源文件不存在: {SRC_9B}")
        return

    QUANT_DIR.mkdir(parents=True, exist_ok=True)

    for q in QUANT_TYPES:
        out = QUANT_DIR / f"Qwen3.5-9B-{q}.gguf"
        if out.exists():
            size_gb = out.stat().st_size / 1e9
            print(f"[SKIP] {q}: 已存在 ({size_gb:.1f} GB)")
            continue

        print(f"[{time.strftime('%H:%M:%S')}] 生成 {q}...")
        result = subprocess.run(
            [LLAMA_QUANTIZE, str(SRC_9B), str(out), q],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"[ERROR] {q}: {result.stderr[-200:]}")
        else:
            size_gb = out.stat().st_size / 1e9
            print(f"[{time.strftime('%H:%M:%S')}] {q} 完成 ({size_gb:.1f} GB)")


# ============================================================
# llama-bench 执行 + 解析
# ============================================================
def run_bench(model_path: str, label: str) -> list[dict]:
    """运行 llama-bench 并返回 [pp_result, tg_result]"""
    if not Path(model_path).exists():
        print(f"  [SKIP] {label}: {model_path} 不存在")
        return []

    model_size = Path(model_path).stat().st_size

    cmd = [
        LLAMA_BENCH,
        "-m", model_path,
        "-p", str(N_PROMPT),
        "-n", str(N_GEN),
        "-r", str(N_RUNS),
        "--output", "csv",
    ]

    print(f"[{time.strftime('%H:%M:%S')}] Bench {label} ({model_size/1e9:.1f} GB)...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"  [ERROR] {result.stderr[-200:]}")
        return []

    output = result.stdout.strip()

    # 保存原始输出
    raw_path = RESULTS_DIR / "raw" / f"bench_{MACHINE_NAME}_{label}.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(output)

    # 正确解析 CSV（处理引号内的逗号）
    rows = []
    reader = csv.DictReader(io.StringIO(output))
    for row in reader:
        rows.append({
            "machine": MACHINE_NAME,
            "model_label": label,
            "n_prompt": int(row.get("n_prompt", 0)),
            "n_gen": int(row.get("n_gen", 0)),
            "avg_tps": float(row.get("avg_ts", 0)),
            "stddev_tps": float(row.get("stddev_ts", 0)),
            "avg_ns": int(row.get("avg_ns", 0)),
            "model_size_bytes": model_size,
            "model_type": row.get("model_type", ""),
            "n_threads": row.get("n_threads", ""),
            "n_gpu_layers": row.get("n_gpu_layers", ""),
        })

    for r in rows:
        phase = "pp" if r["n_prompt"] > 0 and r["n_gen"] == 0 else "tg"
        print(f"  {phase}: {r['avg_tps']:.2f} tps")

    return rows


# ============================================================
# 主函数
# ============================================================
def main():
    global MACHINE_NAME
    mode = sys.argv[1] if len(sys.argv) > 1 else "113"

    # 选择模型列表
    if mode == "103":
        MACHINE_NAME = "M1Max_32GB"

    print(f"=== Phase 1: llama-bench ({MACHINE_NAME}) ===")
    print(f"N_PROMPT={N_PROMPT}, N_GEN={N_GEN}, N_RUNS={N_RUNS}")

    if mode == "103":
        models = get_models_103()
        out_csv = RESULTS_DIR / "phase1_bench_103_m1max.csv"
    else:
        # 113 模式：先生成量化版本
        print("\n--- 量化生成阶段 ---")
        generate_quants()
        models = get_models_113()
        out_csv = RESULTS_DIR / "phase1_bench_113_m2ultra.csv"

    # 运行 bench
    print("\n--- llama-bench 测试阶段 ---")
    all_results = []
    for label, path in models.items():
        rows = run_bench(path, label)
        all_results.extend(rows)

        # 实时保存
        if all_results:
            with open(out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)

    # 打印摘要
    print(f"\n=== 完成 | 结果: {out_csv} ===")
    print(f"\n{'Model':<20} {'PP (tps)':>10} {'TG (tps)':>10} {'Size(GB)':>10}")
    print("-" * 52)

    labels = list(dict.fromkeys(r["model_label"] for r in all_results))
    for label in labels:
        rows = [r for r in all_results if r["model_label"] == label]
        pp = next((r["avg_tps"] for r in rows if r["n_prompt"] > 0 and r["n_gen"] == 0), None)
        tg = next((r["avg_tps"] for r in rows if r["n_gen"] > 0), None)
        size = rows[0]["model_size_bytes"] / 1e9 if rows else 0
        pp_s = f"{pp:.1f}" if pp else "-"
        tg_s = f"{tg:.1f}" if tg else "-"
        print(f"{label:<20} {pp_s:>10} {tg_s:>10} {size:>10.1f}")


if __name__ == "__main__":
    main()


# ============================================================
# 4B 多精度 bench（单独入口）
# ============================================================
def get_models_4B():
    src_f16 = Path("~/Documents/mlx-community/Qwen3.5-4B-F16.gguf").expanduser()
    qdir = Path("~/Documents/mlx-community/Qwen3.5-4B-quants").expanduser()
    models = {"4B_F16": str(src_f16)}
    for q in ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]:
        models[f"4B_{q}"] = str(qdir / f"Qwen3.5-4B-{q}.gguf")
    return models
