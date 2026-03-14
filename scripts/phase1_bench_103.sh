#!/bin/bash
# Phase 1: llama-bench on M1 Max (103) - 对比基线
# 在 103 机器上运行

LLAMA_BIN=~/Documents/Codes/llama.cpp/build/bin
RESULTS_DIR=/tmp/phase1_103
RAW_DIR=$RESULTS_DIR/raw
mkdir -p $RESULTS_DIR $RAW_DIR

# 103 上有 9B Q8 和 0.6B Q4
MODELS=()
MODELS+=("9B_Q8:~/Documents/mlx-community/Qwen3.5-9B-UD-Q8_K_XL/Qwen3.5-9B-UD-Q8_K_XL.gguf")
MODELS+=("0.6B_Q4:~/Documents/mlx-community/Qwen3-0.6B-q4_k_m.gguf")

OUT_CSV=$RESULTS_DIR/phase1_bench_103_m1max.csv
echo "machine,quant_label,n_prompt,n_gen,avg_tps,stddev_tps,model_size_bytes" > $OUT_CSV

echo "=== Phase 1: llama-bench on M1 Max (103) ==="
for ENTRY in "${MODELS[@]}"; do
    QLABEL="${ENTRY%%:*}"
    MODEL_PATH="${ENTRY#*:}"
    MODEL_PATH="${MODEL_PATH/#\~/$HOME}"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "[SKIP] $QLABEL: $MODEL_PATH 不存在"
        continue
    fi

    MODEL_SIZE=$(stat -f%z "$MODEL_PATH" 2>/dev/null || stat -c%s "$MODEL_PATH" 2>/dev/null)
    echo "[$(date '+%H:%M:%S')] Bench $QLABEL ($(du -sh $MODEL_PATH | cut -f1))..."

    RAW=$($LLAMA_BIN/llama-bench \
        -m "$MODEL_PATH" \
        -p 512 -n 128 \
        -r 3 \
        --output csv 2>/dev/null)

    echo "$RAW" > $RAW_DIR/bench_103_${QLABEL}.csv

    echo "$RAW" | tail -n +2 | while IFS=',' read -ra COLS; do
        N_PROMPT="${COLS[32]}"
        N_GEN="${COLS[33]}"
        AVG_TPS="${COLS[37]}"
        STDDEV_TPS="${COLS[38]}"
        echo "M1Max_32GB,$QLABEL,$N_PROMPT,$N_GEN,$AVG_TPS,$STDDEV_TPS,$MODEL_SIZE" >> $OUT_CSV
    done

    echo "[$(date '+%H:%M:%S')] $QLABEL 完成"
done

echo ""
echo "=== 103 Bench 完成 ==="
cat $OUT_CSV
