#!/usr/bin/env bash
set -euo pipefail

root=${LEGALPDF_TRAINING_ROOT:?set LEGALPDF_TRAINING_ROOT}
python="$root/venv/bin/python"
script=${LEGALPDF_TRAINING_TOOLS:?set LEGALPDF_TRAINING_TOOLS}/train_student.py
dataset=/mnt/f/oajd_compute_storage/ppdoclayoutv3_training_runs/ppdocv3_train1-12_project_full_20260619_01/dataset
run="$root/runs/legal25-ppdocv3-640-document-safe-e30-seed20260813-r5"
pretrain=/mnt/f/oajd_compute_storage/ppdoc_lite_models/official/PP-DocLayoutV3_legal25_pretrained.pdparams
repo="$root/PaddleDetection-2.9-dfine-amp-20260814"

if pgrep -af '[t]rain_student.py.*legal25-ppdocv3-640' >/dev/null; then
  echo 'A PP-DocLayoutV3 640 training process is already running.' >&2
  exit 1
fi
if [[ -e "$run/run_manifest.json" ]]; then
  echo "Run directory already initialized: $run" >&2
  exit 1
fi
mkdir -p "$run"
exec "$python" "$script" \
  --dataset "$dataset" \
  --annotations-dir annotations_generalization_v1 \
  --run-root "$run" \
  --model PP-DocLayoutV3 \
  --pretrain "$pretrain" \
  --mode train \
  --no-amp \
  --resolution 640 \
  --epochs 30 \
  --batch-size 1 \
  --workers 2 \
  --learning-rate 0.00005 \
  --warmup-steps 20 \
  --eval-interval 1 \
  --log-interval 10 \
  --early-stop-patience 5 \
  --early-stop-min-epoch 8 \
  --early-stop-min-delta 0.002 \
  --augmentation document-safe \
  --seed 20260813 \
  --paddledetection-root "$repo"
