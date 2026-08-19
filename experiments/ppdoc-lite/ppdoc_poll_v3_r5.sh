#!/usr/bin/env bash
set -u

run=${LEGALPDF_TRAINING_ROOT:?set LEGALPDF_TRAINING_ROOT}/runs/legal25-ppdocv3-640-document-safe-e30-seed20260813-r5
log="$run/train.log"

date -Is
nvidia-smi \
  --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv,noheader
ps -eo pid,ppid,etime,pcpu,pmem,stat,args \
  | grep -E '[t]rain_student.py.*legal25-ppdocv3|[t]ools/train.py.*PP-DocLayoutV3-640'
printf 'amp_backoffs='
grep -c 'Found inf or nan' "$log" 2>/dev/null || true
printf '%s\n' '=== recent signals ==='
grep -E 'Epoch:|Average Precision|Total sample number:|Save checkpoint:|Traceback|Error|Found inf or nan' "$log" 2>/dev/null \
  | tail -n 8
printf '%s\n' '=== durable progress ==='
grep -E '"phase"|"epoch"|"batch"|"total_batches"|"validation_ap"|"best_validation_ap"|"stale_evaluations"|"evaluation_history"|"bbox_ap"' "$run/status.json" 2>/dev/null || true
printf '%s\n' '=== checkpoints ==='
find "$run/output" -maxdepth 3 -type f -name '*.pdparams' \
  -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null \
  | sort
