#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

root=${LEGALPDF_TRAINING_ROOT:?set LEGALPDF_TRAINING_ROOT}
python="$root/venv/bin/python"
repo="$root/PaddleDetection-2.9-dfine-amp-20260814"
dataset=/mnt/f/oajd_compute_storage/ppdoclayoutv3_training_runs/ppdocv3_train1-12_project_full_20260619_01/dataset
models=/mnt/f/oajd_compute_storage/ppdoc_lite_models
run="$models/student_preflight/ppdocv3-m-cwd-smoke320-v2"
student="$models/student_preflight/PP-DocLayoutV3-M_legal25_initialized.pdparams"
teacher="$models/official/PP-DocLayoutV3_legal25_pretrained.pdparams"
recipe=${LEGALPDF_TRAINING_TOOLS:?set LEGALPDF_TRAINING_TOOLS}/train_student.py

if [[ -e "$run/run_manifest.json" ]]; then
  printf 'Smoke target already exists: %s\n' "$run" >&2
  exit 1
fi

exec "$python" "$recipe" \
  --dataset "$dataset" \
  --annotations-dir annotations_generalization_v1 \
  --run-root "$run" \
  --model PP-DocLayoutV3 \
  --pretrain "$student" \
  --distill-teacher "$teacher" \
  --distill-encoder-weight 1.0 \
  --distill-mask-weight 1.0 \
  --distill-tau 1.0 \
  --mode smoke \
  --cpu \
  --no-amp \
  --resolution 320 \
  --batch-size 1 \
  --workers 0 \
  --learning-rate 0.00005 \
  --warmup-steps 1 \
  --eval-interval 1 \
  --log-interval 1 \
  --augmentation document-safe \
  --backbone-arch M \
  --num-queries 300 \
  --paddledetection-root "$repo"
