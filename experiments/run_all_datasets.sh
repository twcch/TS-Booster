#!/bin/bash
# Run comprehensive evaluation on all three datasets

echo "=========================================="
echo "Running Comprehensive Evaluation on All Datasets"
echo "=========================================="

# USD_TWD
echo ""
echo ">>> 1/3: USD_TWD"
uv run python experiments/comprehensive_evaluation.py \
    --experiment all \
    --dataset dataset/USD_TWD.csv \
    --epochs 100 \
    --seed 42

# JPY_TWD
echo ""
echo ">>> 2/3: JPY_TWD"
uv run python experiments/comprehensive_evaluation.py \
    --experiment all \
    --dataset dataset/JPY_TWD.csv \
    --epochs 100 \
    --seed 42

# KRW_TWD
# echo ""
# echo ">>> 3/3: KRW_TWD"
# uv run python experiments/comprehensive_evaluation.py \
#     --experiment all \
#     --dataset dataset/KRW_TWD.csv \
#     --epochs 100 \
#     --seed 42

# echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results saved in: experiments/comprehensive_results/"
ls -lh experiments/comprehensive_results/*.json
