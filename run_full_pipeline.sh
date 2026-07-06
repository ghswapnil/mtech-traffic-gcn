#!/bin/bash
echo "======================================"
echo " Starting Full Two-Stage Training Run "
echo "======================================"
echo "Stage 1: Time-Conditioned GCN (stride=1, epochs=20)"
./venv/bin/python -m src.train_stage1 --stride 1 --epochs 20

echo "======================================"
echo "Stage 2: TTM on Residuals (stride=1, epochs=20, fine-tuning TTM)"
./venv/bin/python -m src.train_stage2 --stride 1 --epochs 20 --fine-tune-ttm

echo "======================================"
echo " Full Run Complete! "
echo "======================================"
