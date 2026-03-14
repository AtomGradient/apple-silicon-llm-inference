#!/usr/bin/env python3
"""
Phase 1: llama-bench 对 Qwen3.5-4B 多精度 Q2_K→F16 的测试
"""
import csv, io, subprocess, sys, time
from pathlib import Path

LLAMA_BENCH = str(Path("~/Documents/Codes/llama.cpp/build/bin/llama-bench").expanduser())
RESULTS_DIR = Path("~/Documents/Codes/FreeExplore/data").expanduser()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

QUANT_DIR = Path("~/Documents/mlx-community/Qwen3.5-4B-quants").expanduser()
SRC_F16 = Path("~/Documents/mlx-community/Qwen3.5-4B-F16.gguf").expanduser()

MACHINE = "M2Ultra_192GB"
N_PROMPT = 512
N_GEN = 128
N_RUNS = 3

MODELS = {
    "4B_F16":   str(SRC_F16),
    "4B_Q8_0":  str(QUANT_DIR / "Qwen3.5-4B-Q8_0.gguf"),
    "4B_Q6_K":  str(QUANT_DIR / "Qwen3.5-4B-Q6_K.gguf"),
    "4B_Q5_K_M": str(QUANT_DIR / "Qwen3.5-4B-Q5_K_M.gguf"),
    "4B_Q4_K_M": str(QUANT_DIR / "Qwen3.5-4B-Q4_K_M.gguf"),
    "4B_Q3_K_M": str(QUANT_DIR / "Qwen3.5-4B-Q3_K_M.gguf"),
    "4B_Q2_K":  str(QUANT_DIR / "Qwen3.5-4B-Q2_K.gguf"),
}

def run_bench(label, model_path):
    if not Path(model_path).exists():
        print(f"  [SKIP] {label}: 不存在")
        return []

    model_size = Path(model_path).stat().st_size
    print(f"[{time.strftime('%H:%M:%S')}] Bench {label} ({model_size/1e9:.2f} GB)...")

    cmd = [LLAMA_BENCH, "-m", model_path,
           "-p", str(N_PROMPT), "-n", str(N_GEN),
           "-r", str(N_RUNS), "--output", "csv"]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  [ERROR] {result.stderr[-100:]}")
        return []

    # 保存原始输出
    raw_path = RESULTS_DIR / "raw" / f"bench_4B_{MACHINE}_{label}.csv"
    raw_path.parent.mkdir(exist_ok=True)
    raw_path.write_text(result.stdout)

    rows = []
    reader = csv.DictReader(io.StringIO(result.stdout.strip()))
    for row in reader:
        rows.append({
            "machine": MACHINE, "model_label": label,
            "n_prompt": int(row.get("n_prompt", 0)),
            "n_gen": int(row.get("n_gen", 0)),
            "avg_tps": float(row.get("avg_ts", 0)),
            "stddev_tps": float(row.get("stddev_ts", 0)),
            "avg_ns": int(row.get("avg_ns", 0)),
            "model_size_bytes": model_size,
            "model_type": row.get("model_type", ""),
        })

    for r in rows:
        phase = "pp" if r["n_prompt"] > 0 and r["n_gen"] == 0 else "tg"
        print(f"  {phase}: {r['avg_tps']:.2f} tps")
    return rows

def main():
    print(f"=== Phase 1: llama-bench 4B 多精度 ({MACHINE}) ===")
    out_csv = RESULTS_DIR / "phase1_bench_4B_llamacpp.csv"
    all_rows = []

    for label, path in MODELS.items():
        rows = run_bench(label, path)
        all_rows.extend(rows)
        if all_rows:
            with open(out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
                writer.writeheader()
                writer.writerows(all_rows)

    print(f"\n=== 完成 | 结果: {out_csv} ===")
    print(f"\n{'Model':<15} {'PP(tps)':>10} {'TG(tps)':>10} {'Size(GB)':>10}")
    print("-" * 47)
    for label in MODELS:
        rows = [r for r in all_rows if r["model_label"] == label]
        pp = next((r["avg_tps"] for r in rows if r["n_prompt"] > 0 and r["n_gen"] == 0), None)
        tg = next((r["avg_tps"] for r in rows if r["n_gen"] > 0), None)
        size = rows[0]["model_size_bytes"]/1e9 if rows else 0
        print(f"{label:<15} {(f'{pp:.1f}' if pp else '-'):>10} {(f'{tg:.1f}' if tg else '-'):>10} {size:>10.2f}")

if __name__ == "__main__":
    main()
