#!/usr/bin/env bash
set -e

echo "===== 1. Download & prepare data ====="
python -m data.download

echo "===== 2. Train ====="
python train.py

echo "===== 3. Evaluate ====="
python evaluate.py --checkpoint checkpoints/vitpose_B_mix_best.pt

echo "===== 4. Visualise ====="
python visualize.py --checkpoint checkpoints/vitpose_B_mix_best.pt

echo "===== Done ====="