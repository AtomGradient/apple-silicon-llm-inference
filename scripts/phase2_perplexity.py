#!/usr/bin/env python3
"""
Phase 2: 困惑度（Perplexity）评估
评估不同量化精度下 Qwen3.5-4B 的质量损失
数据集: WikiText-2 (标准 LLM PPL 基准)
"""
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

LLAMA_PPL = str(Path("~/Documents/Codes/llama.cpp/build/bin/llama-perplexity").expanduser())
RESULTS_DIR = Path("~/Documents/Codes/FreeExplore/data").expanduser()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WIKITEXT_DIR = RESULTS_DIR / "wikitext"
WIKITEXT_FILE = WIKITEXT_DIR / "wiki.test.raw"

QUANT_DIR = Path("~/Documents/mlx-community/Qwen3.5-4B-quants").expanduser()
SRC_F16 = Path("~/Documents/mlx-community/Qwen3.5-4B-F16.gguf").expanduser()

MACHINE = "M2Ultra_192GB"

# 测试所有精度（高精度→低精度）
MODELS = {
    "4B_F16":    str(SRC_F16),
    "4B_Q8_0":   str(QUANT_DIR / "Qwen3.5-4B-Q8_0.gguf"),
    "4B_Q6_K":   str(QUANT_DIR / "Qwen3.5-4B-Q6_K.gguf"),
    "4B_Q5_K_M": str(QUANT_DIR / "Qwen3.5-4B-Q5_K_M.gguf"),
    "4B_Q4_K_M": str(QUANT_DIR / "Qwen3.5-4B-Q4_K_M.gguf"),
    "4B_Q3_K_M": str(QUANT_DIR / "Qwen3.5-4B-Q3_K_M.gguf"),
    "4B_Q2_K":   str(QUANT_DIR / "Qwen3.5-4B-Q2_K.gguf"),
}


def download_wikitext():
    """下载 WikiText-2 测试集"""
    WIKITEXT_DIR.mkdir(parents=True, exist_ok=True)
    if WIKITEXT_FILE.exists():
        print(f"[OK] WikiText-2 已存在: {WIKITEXT_FILE}")
        return True

    print("[下载] WikiText-2 测试集...")
    url = "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/examples/perplexity/wikitext-2-raw-v1-test.txt"

    # 尝试 curl
    r = subprocess.run(
        ["curl", "-sL", url, "-o", str(WIKITEXT_FILE)],
        capture_output=True, timeout=120
    )
    if r.returncode == 0 and WIKITEXT_FILE.exists() and WIKITEXT_FILE.stat().st_size > 1000:
        print(f"[OK] 下载成功: {WIKITEXT_FILE.stat().st_size // 1024}KB")
        return True

    # fallback: 从 HuggingFace Datasets 下载
    print("[尝试] 从 HuggingFace 下载...")
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
        WIKITEXT_FILE.write_text(text)
        print(f"[OK] HF 下载成功: {len(text)} chars")
        return True
    except Exception as e:
        print(f"[ERROR] 下载失败: {e}")
        return False


def run_perplexity(label, model_path, ctx=512) -> dict | None:
    """运行 llama-perplexity 并解析 PPL"""
    if not Path(model_path).exists():
        print(f"  [SKIP] {label}: 不存在")
        return None
    if not WIKITEXT_FILE.exists():
        print(f"  [SKIP] {label}: 数据集不存在")
        return None

    model_size = Path(model_path).stat().st_size
    print(f"[{time.strftime('%H:%M:%S')}] PPL {label} ({model_size/1e9:.2f}GB)...")

    cmd = [
        LLAMA_PPL,
        "-m", model_path,
        "-f", str(WIKITEXT_FILE),
        "-c", str(ctx),
        "--chunks", "10",   # 使用前 10 块，加快速度
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    output = result.stdout + result.stderr

    # 保存原始输出
    raw_path = RESULTS_DIR / "raw" / f"ppl_{label}.txt"
    raw_path.parent.mkdir(exist_ok=True)
    raw_path.write_text(output)

    # 解析 PPL: 格式 "Final estimate: PPL = 8.1234 +/- 0.0567"
    ppl_match = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", output)
    if not ppl_match:
        # 备用格式: "perplexity: N/N [t=Xs] 8.1234"
        matches = re.findall(r"perplexity:\s+\d+/\d+.*?\]\s+([0-9.]+)", output)
        if matches:
            ppl_val = float(matches[-1])
            print(f"  PPL = {ppl_val:.4f}")
            return {"machine": MACHINE, "model": label, "ppl": ppl_val,
                    "ppl_std": None, "model_size_bytes": model_size, "ctx": ctx}
        print(f"  [ERROR] 未找到 PPL 值\n  输出末尾: {output[-300:]}")
        return None

    ppl = float(ppl_match.group(1))
    ppl_std = float(ppl_match.group(2))
    print(f"  PPL = {ppl:.4f} ± {ppl_std:.4f}")

    return {
        "machine": MACHINE, "model": label,
        "ppl": round(ppl, 4), "ppl_std": round(ppl_std, 4),
        "model_size_bytes": model_size, "ctx": ctx,
    }


def main():
    print("=== Phase 2: Perplexity 评估 ===")

    # 下载数据集
    if not download_wikitext():
        print("[FATAL] 无法获取 WikiText-2，退出")
        sys.exit(1)

    out_csv = RESULTS_DIR / "phase2_ppl_results.csv"
    results = []

    for label, path in MODELS.items():
        r = run_perplexity(label, path)
        if r:
            results.append(r)
            with open(out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)

    print(f"\n=== 完成 {len(results)}/{len(MODELS)} | 结果: {out_csv} ===")
    print(f"\n{'Model':<15} {'PPL':>8} {'±':>8} {'Size(GB)':>10}")
    print("-" * 43)
    for r in results:
        std_s = f"{r['ppl_std']:.4f}" if r['ppl_std'] else "N/A"
        print(f"{r['model']:<15} {r['ppl']:>8.4f} {std_s:>8} {r['model_size_bytes']/1e9:>10.2f}")

if __name__ == "__main__":
    main()
