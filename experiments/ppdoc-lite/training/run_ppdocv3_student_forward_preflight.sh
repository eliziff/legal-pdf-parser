#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

root=${LEGALPDF_TRAINING_ROOT:?set LEGALPDF_TRAINING_ROOT}
python="$root/venv/bin/python"
repo="$root/PaddleDetection-2.9-dfine-amp-20260814"
dataset=/mnt/f/oajd_compute_storage/ppdoclayoutv3_training_runs/ppdocv3_train1-12_project_full_20260619_01/dataset
preflight=/mnt/f/oajd_compute_storage/ppdoc_lite_models/student_preflight
recipe=${LEGALPDF_TRAINING_TOOLS:?set LEGALPDF_TRAINING_TOOLS}/train_student.py
forward=${LEGALPDF_TRAINING_TOOLS:?set LEGALPDF_TRAINING_TOOLS}/preflight_ppdocv3_forward.py

for size in M S; do
  lower=${size,,}
  weights="$preflight/PP-DocLayoutV3-${size}_legal25_initialized.pdparams"
  for queries in 300 100; do
    if [[ "$queries" == 300 ]]; then
      run="$preflight/ppdocv3-$lower-v1"
    else
      run="$preflight/ppdocv3-$lower-q100-v1"
      if [[ ! -e "$run/run_manifest.json" ]]; then
        "$python" "$recipe" \
          --dataset "$dataset" \
          --annotations-dir annotations_generalization_v1 \
          --run-root "$run" \
          --model PP-DocLayoutV3 \
          --pretrain "$weights" \
          --mode check \
          --no-amp \
          --resolution 640 \
          --epochs 30 \
          --batch-size 1 \
          --workers 2 \
          --learning-rate 0.00005 \
          --warmup-steps 20 \
          --eval-interval 1 \
          --augmentation document-safe \
          --backbone-arch "$size" \
          --num-queries "$queries" \
          --paddledetection-root "$repo"
      fi
    fi
    receipt="$preflight/PP-DocLayoutV3-${size}-q${queries}-cpu-forward.json"
    if [[ -e "$receipt" ]]; then
      printf 'Forward receipt already exists: %s\n' "$receipt" >&2
      exit 1
    fi
    "$python" "$forward" \
      --paddledetection-root "$repo" \
      --config "$run/config/PP-DocLayoutV3-640.yml" \
      --weights "$weights" \
      --receipt "$receipt"
  done
done
