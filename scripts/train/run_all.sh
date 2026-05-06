#!/bin/bash
# CRBSA 三阶段训练一键脚本
# 用法: bash scripts/train/run_all.sh

set -e

MODEL="Qwen/Qwen3.6-35B-A3B"
DATA_DIR="./data/train"
OUTPUT_BASE="./outputs"

# ── Stage 1: 路由器蒸馏 ──────────────────────────
echo "=== Stage 1: Router Distillation ==="
python scripts/train/stage1_distill.py \
    --model $MODEL \
    --data-dir $DATA_DIR \
    --output-dir $OUTPUT_BASE/stage1 \
    --seq-len 131072 \
    --batch-size 1 \
    --lr 1e-4 \
    --epochs 3 \
    --log-interval 10 \
    --debug

# ── Stage 2: 截断式稀疏微调 ──────────────────────
echo "=== Stage 2: Detached Sparse Tuning ==="
python scripts/train/stage2_sft.py \
    --model $MODEL \
    --stage1-dir $OUTPUT_BASE/stage1 \
    --data-dir $DATA_DIR \
    --output-dir $OUTPUT_BASE/stage2 \
    --seq-len 131072 \
    --batch-size 1 \
    --lr 5e-5 \
    --epochs 2 \
    --log-interval 10 \
    --debug

# ── Stage 3: GRPO 强化学习 ───────────────────────
echo "=== Stage 3: GRPO RL ==="
python scripts/train/stage3_grpo.py \
    --model $MODEL \
    --output-dir $OUTPUT_BASE/stage3 \
    --seq-len 100000 \
    --epochs 1 \
    --steps-per-epoch 100 \
    --debug

echo "=== All stages complete ==="
