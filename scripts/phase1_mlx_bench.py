#!/usr/bin/env python3
"""
Phase 1: MLX 框架基准测试
对 Gemma-3 QAT 多精度 + Qwen3 系列 进行 tokens/sec 测量
"""
import time, subprocess, sys, csv, os
from pathlib import Path

RESULTS_DIR = Path("~/Documents/Codes/FreeExplore/data").expanduser()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("Gemma3-4B-QAT-3bit",  "~/Documents/mlx-community/gemma-3-4b-it-qat-3bit"),
    ("Gemma3-4B-QAT-4bit",  "~/Documents/mlx-community/gemma-3-4b-it-qat-4bit"),
    ("Gemma3-4B-QAT-6bit",  "~/Documents/mlx-community/gemma-3-4b-it-qat-6bit"),
    ("Gemma3-4B-QAT-8bit",  "~/Documents/mlx-community/gemma-3-4b-it-qat-8bit"),
    ("Qwen3.5-0.8B",        "~/Documents/mlx-community/Qwen3.5-0.8B"),
    ("Qwen3.5-4B",          "~/Documents/mlx-community/Qwen3.5-4B"),
    ("Qwen3.5-9B",          "~/Documents/mlx-community/Qwen3.5-9B"),
    ("Qwen3-8B-3bit",       "~/Documents/mlx-community/Qwen3-8B-3bit"),
    ("Qwen3-8B-4bit",       "~/Documents/mlx-community/Qwen3-8B-4bit"),
    ("Qwen3-8B-6bit",       "~/Documents/mlx-community/Qwen3-8B-6bit"),
]

N_GEN = 128
N_RUNS = 3
PROMPT = "The development of large language models represents a significant breakthrough in artificial intelligence research. These models have demonstrated remarkable capabilities across a wide range of natural language processing tasks."

def model_size_mb(path_str):
    p = Path(path_str).expanduser()
    if not p.exists(): return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6

def bench_mlx(label, path_str):
    p = Path(path_str).expanduser()
    if not p.exists():
        print(f"  [SKIP] {label}: 路径不存在")
        return None
    
    size_mb = model_size_mb(path_str)
    print(f"\n[{time.strftime('%H:%M:%S')}] {label} ({size_mb:.0f} MB)")

    script = f"""
import time, mlx.core as mx
from mlx_lm import load, generate

model, tokenizer = load("{str(p)}")
prompt = "{PROMPT}"
max_tokens = {N_GEN}

# warmup
_ = generate(model, tokenizer, prompt=prompt, max_tokens=32, verbose=False)
mx.eval(mx.array([1]))

results = []
for i in range({N_RUNS}):
    t0 = time.perf_counter()
    out = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    mx.eval(mx.array([1]))
    elapsed = time.perf_counter() - t0
    tps = max_tokens / elapsed
    results.append(tps)
    print("  run%d: %.1f tps" % (i+1, tps))

avg = sum(results)/len(results)
std = (sum((x-avg)**2 for x in results)/len(results))**0.5
print("RESULT:%.4f:%.4f" % (avg, std))
"""
    try:
        env = os.environ.copy()
        r = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=600)
        output = r.stdout + r.stderr
        for line in output.split("\n"):
            if line.startswith("  run"): print(line)
            if line.startswith("RESULT:"):
                parts = line.split(":")
                avg_tps = float(parts[1])
                std_tps = float(parts[2])
                print(f"  avg: {avg_tps:.1f} ± {std_tps:.1f} tps")
                return {"machine":"M2Ultra_192GB","framework":"MLX","model":label,
                        "n_gen":N_GEN,"avg_tps":round(avg_tps,3),
                        "std_tps":round(std_tps,3),"size_mb":round(size_mb,1)}
        print(f"  [ERROR] 未找到 RESULT: {output[-200:]}")
        return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None

def main():
    print(f"=== Phase 1: MLX 基准测试 (M2Ultra 192GB) ===")
    print(f"N_GEN={N_GEN}, N_RUNS={N_RUNS}")
    
    out_csv = RESULTS_DIR / "phase1_mlx_bench_113.csv"
    results = []
    
    for label, path in MODELS:
        r = bench_mlx(label, path)
        if r:
            results.append(r)
            with open(out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=r.keys())
                writer.writeheader()
                writer.writerows(results)
    
    print(f"\n=== 完成 {len(results)}/{len(MODELS)} ===")
    print(f"结果: {out_csv}")
    print(f"\n{'Model':<25} {'TPS':>8} {'Std':>6} {'Size(MB)':>10}")
    print("-" * 51)
    for r in results:
        print(f"{r['model']:<25} {r['avg_tps']:>8.1f} {r['std_tps']:>6.1f} {r['size_mb']:>10.0f}")

if __name__ == "__main__":
    main()
