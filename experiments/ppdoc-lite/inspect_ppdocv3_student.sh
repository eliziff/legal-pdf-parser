#!/usr/bin/env bash
set -u

repo=${LEGALPDF_TRAINING_ROOT:?set LEGALPDF_TRAINING_ROOT}/PaddleDetection-2.9-dfine-amp-20260814
cd "$repo"

printf '%s\n' '=== PP-DocLayoutV3 released config ==='
sed -n '1,360p' configs/layout_analysis/PP-DocLayoutV3.yaml

printf '%s\n' '=== mask RT-DETR shared base ==='
sed -n '1,360p' configs/mask_rtdetr/_base_/mask_rtdetr_r50vd.yml

printf '%s\n' '=== PP-DocLayoutV3 transformer ==='
file=$(grep -R -l -E 'class DocLayoutV3Transformer' ppdet/modeling | head -n 1)
printf 'FILE=%s\n' "$file"
sed -n '1,760p' "$file"

printf '%s\n' '=== PP-DocLayoutV3 head ==='
file=$(grep -R -l -E 'class DocLayoutV3Head' ppdet/modeling | head -n 1)
printf 'FILE=%s\n' "$file"
sed -n '1,620p' "$file"

printf '%s\n' '=== feature distillation loss ==='
file=$(grep -R -l -E 'class FGDFeatureLoss' ppdet | head -n 1)
printf 'FILE=%s\n' "$file"
sed -n '1,520p' "$file"

printf '%s\n' '=== distill_pairs references ==='
grep -R -n -E 'distill_pairs' ppdet/modeling ppdet/slim 2>/dev/null || true
