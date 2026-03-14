#!/bin/bash
# Phase 1: 生成多精度 GGUF 并运行 llama-bench
# 机器: 113 (M2 Ultra 192GB)

LLAMA_BIN=~/Documents/Codes/llama.cpp/build/bin
SRC_9B=~/Documents/mlx-community/Qwen3.5-9B-UD-Q8_K_XL/Qwen3.5-9B-UD-Q8_K_XL.gguf
QUANT_DIR=~/Documents/mlx-community/Qwen3.5-9B-quants
RESULTS_DIR=~/Documents/Codes/FreeExplore/data
RAW_DIR=$RESULTS_DIR/raw

mkdir -p $QUANT_DIR $RESULTS_DIR $RAW_DIR

QUANT_TYPES=("Q2_K" "Q3_K_M" "Q4_K_M" "Q5_K_M" "Q6_K")

echo "=== [Phase 1-A] 生成多精度 GGUF from Qwen3.5-9B Q8 ==="
for Q in "${QUANT_TYPES[@]}"; do
    OUT=$QUANT_DIR/Qwen3.5-9B-${Q}.gguf
    if [ ! -f "$OUT" ]; then
        echo "[$(date '+%H:%M:%S')] 生成 $Q ..."
        $LLAMA_BIN/llama-quantize "$SRC_9B" "$OUT" "$Q"
        echo "[$(date '+%H:%M:%S')] $Q 完成 -> $(du -sh $OUT | cut -f1)"
    else
        echo "[$(date '+%H:%M:%S')] $Q 已存在 -> $(du -sh $OUT | cut -f1)，跳过"
    fi
done

echo ""
echo "=== [Phase 1-B] llama-bench 基准测试 (M2 Ultra 113) ==="

# 构建测试模型列表: "quantname:filepath"
MODELS=()
MODELS+=("Q8_K_XL:$SRC_9B")
for Q in "${QUANT_TYPES[@]}"; do
    MODELS+=("${Q}:$QUANT_DIR/Qwen3.5-9B-${Q}.gguf")
done

# 同时测试 0.8B 和 2B（用于后续 draft model 参照）
MODELS+=("0.8B_Q8:~/Documents/mlx-community/Qwen3.5-0.8B-GGUF-UD-Q8_K_XL/Qwen3.5-0.8B-UD-Q8_K_XL.gguf")
MODELS+=("2B_Q8:~/Documents/mlx-community/Qwen3.5-2B-GGUF-UD-Q8_K_L/Qwen3.5-2B-UD-Q8_K_XL.gguf")

# 结果 CSV
OUT_CSV=$RESULTS_DIR/phase1_bench_113_m2ultra.csv
echo "machine,quant_label,n_prompt,n_gen,avg_tps,stddev_tps,model_size_bytes" > $OUT_CSV

for ENTRY in "${MODELS[@]}"; do
    QLABEL="${ENTRY%%:*}"
    MODEL_PATH="${ENTRY#*:}"
    # 展开 ~
    MODEL_PATH="${MODEL_PATH/#\~/$HOME}"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "[SKIP] $QLABEL: 文件不存在 $MODEL_PATH"
        continue
    fi

    MODEL_SIZE=$(stat -f%z "$MODEL_PATH" 2>/dev/null || stat -c%s "$MODEL_PATH" 2>/dev/null)
    echo "[$(date '+%H:%M:%S')] Bench $QLABEL ($(du -sh $MODEL_PATH | cut -f1))..."

    RAW=$($LLAMA_BIN/llama-bench \
        -m "$MODEL_PATH" \
        -p 512 -n 128 \
        -r 3 \
        --output csv 2>/dev/null)

    # 保存原始输出
    echo "$RAW" > $RAW_DIR/bench_113_${QLABEL}.csv

    # 提取 pp (n_prompt=512, n_gen=0) 和 tg (n_prompt=0, n_gen=128) 行
    # CSV 列: build_commit(0),...,n_prompt(32),n_gen(33),...,avg_ts(37),stddev_ts(38)
    echo "$RAW" | tail -n +2 | while IFS=',' read -ra COLS; do
        N_PROMPT="${COLS[32]}"
        N_GEN="${COLS[33]}"
        AVG_TPS="${COLS[37]}"
        STDDEV_TPS="${COLS[38]}"
        echo "M2Ultra_192GB,$QLABEL,$N_PROMPT,$N_GEN,$AVG_TPS,$STDDEV_TPS,$MODEL_SIZE" >> $OUT_CSV
    done

    echo "[$(date '+%H:%M:%S')] $QLABEL: 完成"
done

echo ""
echo "=== Phase 1-B 完成 ==="
echo "CSV 结果: $OUT_CSV"
echo ""
cat $OUT_CSV
