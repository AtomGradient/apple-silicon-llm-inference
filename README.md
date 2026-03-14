# Apple Silicon LLM Inference — Research Workspace

Experiment workspace for the paper:

> **Efficient On-Device LLM Inference on Apple Silicon: From Quantization to Speculative Decoding**
> AtomGradient · March 2026
> [Paper (PDF)](https://atomgradient.github.io/apple-silicon-llm-inference/paper.pdf) · [Website](https://atomgradient.github.io/apple-silicon-llm-inference/) · [GitHub](https://github.com/AtomGradient/apple-silicon-llm-inference)

---

## Hardware

| Machine | Chip | Memory | Mem BW | IP |
|---------|------|--------|--------|----|
| M2 Ultra (113) | Apple M2 Ultra | 192 GB | 800 GB/s | 192.168.0.113 |
| M1 Max (103)   | Apple M1 Max   |  32 GB | 400 GB/s | 192.168.0.103 |
| M2 Pro (104)   | Apple M2 Pro   |  32 GB | 200 GB/s | 192.168.0.104 |

SSH access from 113: `ssh alex@192.168.0.103` / `ssh atomgradient@192.168.0.104`

Python env (all machines): `source ~/Documents/mlx-community/3-11-mlx-community-env/bin/activate`

---

## Directory Structure

```
FreeExplore/
├── scripts/                        # Experiment scripts (run on 113 unless noted)
│   ├── phase1_bench_llamacpp.py    # Phase 1: llama-bench throughput (M2 Ultra)
│   ├── phase1_bench_103.sh         # Phase 1: llama-bench on M1 Max (103)
│   ├── phase1_mlx_bench.py         # Phase 1: MLX framework bench (Gemma-3 QAT)
│   ├── phase1_quantize_and_bench.sh# Phase 1: quantize 4B then bench
│   ├── phase2_perplexity.py        # Phase 2: WikiText-2 PPL evaluation
│   ├── phase3_sd_bench.py          # Phase 3: local speculative decoding
│   └── phase3_cross_device_sd.py   # Phase 3: cross-device SD via GGML_RPC
│
├── data/                           # Raw outputs and intermediate results
│   ├── phase1_bench_113_m2ultra.csv
│   ├── phase1_mlx_bench_113.csv
│   ├── phase2_ppl_results.csv
│   ├── phase3_sd_results.csv
│   ├── phase3_cross_device_sd.csv
│   └── raw/                        # Raw llama-speculative stdout/stderr
│
├── paper/                          # Paper repository (published to GitHub Pages)
│   ├── docs/
│   │   ├── paper.tex               # LaTeX source
│   │   ├── paper.pdf               # Compiled PDF
│   │   └── index.html              # Bilingual (EN/ZH) GitHub Pages site
│   └── results/
│       └── benchmark_results.json  # Source-of-truth for all benchmark numbers
│
├── phase1-quantization-bench.md    # Phase 1 notes
├── phase2-quality-eval.md          # Phase 2 notes
├── phase3-speculative-decoding.md  # Phase 3 notes
└── research-plan.md                # Original research plan
```

---

## Experiment Phases

### Phase 1 — Quantization Throughput Benchmark

**Goal:** Measure TG and PP throughput for Qwen3.5-4B across 7 GGUF quantization levels (F16 → Q2_K) on M2 Ultra; extend to M1 Max and M2 Pro for hardware comparison.

```bash
# On 113 (M2 Ultra): run full quantization ladder
python scripts/phase1_bench_llamacpp.py

# On 113: MLX bench (Gemma-3 QAT, Qwen3.5 multi-size)
python scripts/phase1_mlx_bench.py

# On 103 (M1 Max) via SSH: run hardware comparison bench
bash scripts/phase1_bench_103.sh
```

**Key parameters:** `llama-bench -p 512 -n 128 -r 4`

**Models used:**
- Qwen3.5-4B (F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K)
- Qwen3.5-0.8B / 2B / 9B / 35B-A3B (Q8_0) for multi-size comparison
- Gemma-3-4B QAT 3/4/6/8-bit (MLX)

---

### Phase 2 — Perplexity Evaluation (PTQ Quality)

**Goal:** Measure WikiText-2 perplexity for each quantization level of Qwen3.5-4B to quantify quality-compression trade-off.

```bash
python scripts/phase2_perplexity.py
```

**Key parameters:** `llama-perplexity --chunks 10 --ctx-size 512`
**Dataset:** WikiText-2 Raw v1 test split (`data/wikitext/`)

---

### Phase 3 — Speculative Decoding

**Goal 1 (local SD):** Measure throughput gain of speculative decoding with various draft/target model pairs and draft lengths (k=4, k=8).

```bash
python scripts/phase3_sd_bench.py
```

**Goal 2 (cross-device SD):** Offload draft model compute to remote machines via GGML_RPC over Gigabit LAN.

```bash
# Start no-BLAS rpc-server on 103 (M1 Max)
ssh alex@192.168.0.103 "~/Documents/mlx-community/llama.cpp/build3/bin/rpc-server -p 50052"

# Start no-BLAS rpc-server on 104 (M2 Pro)
ssh atomgradient@192.168.0.104 "~/Documents/mlx-community/llama.cpp/build2/bin/rpc-server -p 50052"

# Run cross-device SD experiment on 113
python scripts/phase3_cross_device_sd.py
```

> **Note on rpc-server builds:** Use the no-BLAS build (`GGML_BLAS=OFF`). The BLAS-enabled build silently falls back to local compute due to a BLAS/Metal backend conflict, producing invalid measurements. Verified builds: 103/build3, 104/build2.

---

## Key Results (Summary)

| Experiment | Best Config | Result |
|------------|------------|--------|
| Quantization (TG speed) | Q2_K | 72.8 tok/s (1.98× vs F16) |
| Quantization (quality) | Q8_0 | 0.18% PPL degradation |
| **Pareto-optimal** | **Q6_K** | **1.68× speed, 0.54% PPL, 59% smaller** |
| Speculative decoding | 0.8B→9B, k=4 | +25.7% throughput (53.3 tok/s) |
| Cross-device SD (RPC) | M1 Max via GGML_RPC | −79.2% vs local (not viable) |

See [`paper/results/benchmark_results.json`](paper/results/benchmark_results.json) for the complete source-of-truth dataset.

---

## Dependencies

| Tool | Version / Notes |
|------|----------------|
| llama.cpp | Built from source; binaries at `~/Documents/Codes/llama.cpp/build/bin/` (113) |
| MLX | via Python env `3-11-mlx-community-env` |
| tectonic | For compiling `paper.tex` → `paper.pdf` |
| Python | 3.11 |

Model files are stored under `~/Documents/mlx-community/` on each machine (not tracked in git).
