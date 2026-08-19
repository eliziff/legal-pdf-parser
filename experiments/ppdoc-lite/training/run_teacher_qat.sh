#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 MODEL_DIR DATASET_DIR RUN_DIR" >&2
  exit 2
fi

model_dir=$1
dataset_dir=$2
run_dir=$3
train_iter=${PPDOC_QAT_TRAIN_ITER:-200}
venv=${LEGALPDF_TRAINING_ROOT:?set LEGALPDF_TRAINING_ROOT}/paddle26-venv
runner=${LEGALPDF_TRAINING_ROOT:?set LEGALPDF_TRAINING_ROOT}/venv/lib/python3.12/site-packages/paddlex/repo_manager/repos/PaddleDetection/deploy/auto_compression/run_det.py
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

mkdir -p "$run_dir/config" "$run_dir/model" "$run_dir/output"
started_at=$(date --iso-8601=seconds)
write_status() {
  local state=$1
  local exit_code=${2:-}
  {
    printf 'state=%s\n' "$state"
    printf 'started_at=%s\n' "$started_at"
    printf 'updated_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'train_iter=%s\n' "$train_iter"
    [[ -z "$exit_code" ]] || printf 'exit_code=%s\n' "$exit_code"
  } > "$run_dir/status.env"
}
finish() {
  local exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    write_status complete "$exit_code"
  else
    write_status failed "$exit_code"
  fi
}
trap finish EXIT
write_status running

cp "$model_dir/model.pdmodel" "$model_dir/model.pdiparams" "$run_dir/model/"
cp "$script_dir/qat_teacher.yml" "$run_dir/config/qat.yml"
cp "$script_dir/qat_teacher_reader.yml" "$run_dir/config/reader.yml"
sed -i \
  -e "s|__MODEL_DIR__|$run_dir/model|g" \
  -e "s|__DATASET_DIR__|$dataset_dir|g" \
  -e "s|train_iter: 200|train_iter: $train_iter|g" \
  "$run_dir/config/qat.yml" "$run_dir/config/reader.yml"

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$venv/lib/python3.10/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$run_dir"
"$venv/bin/python" -u "$runner" \
  --config "$run_dir/config/reader.yml" \
  --act_config_path "$run_dir/config/qat.yml" \
  --save_dir "$run_dir/output" \
  --devices gpu 2>&1 | tee "$run_dir/train.log"
if grep -Eiq 'loss: (nan|[-+]?inf)' "$run_dir/train.log"; then
  echo "QAT rejected: training loss became non-finite" >&2
  exit 3
fi
