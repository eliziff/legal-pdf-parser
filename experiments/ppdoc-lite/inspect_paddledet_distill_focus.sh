#!/usr/bin/env bash
set -u

repo=${LEGALPDF_TRAINING_ROOT:?set LEGALPDF_TRAINING_ROOT}/PaddleDetection-2.9-dfine-amp-20260814
cd "$repo"

for size in s m l; do
  printf '\n=== mask-rtdetr %s ===\n' "$size"
  sed -n '1,260p' "configs/mask_rtdetr/mask_rtdetr_hgnetv2_${size}_6x_coco.yml"
done

printf '%s\n' '=== DETR architecture ==='
sed -n '1,280p' ppdet/modeling/architectures/detr.py

printf '%s\n' '=== generic distill wrapper ==='
sed -n '1,390p' ppdet/slim/distill_model.py

printf '%s\n' '=== mask hybrid encoder construction ==='
grep -R -n -A80 -B10 -E 'class MaskHybridEncoder|def from_config' \
  ppdet/modeling/necks/hybrid_encoder.py ppdet/modeling/necks 2>/dev/null \
  | head -n 420 || true

printf '%s\n' '=== official mask-rtdetr docs ==='
find configs/mask_rtdetr -maxdepth 1 -type f -iname 'README*' -print -exec sed -n '1,260p' {} \;

printf '%s\n' '=== PP-DocLayoutV3 released config ==='
sed -n '1,360p' configs/layout_analysis/PP-DocLayoutV3.yaml

printf '%s\n' '=== PP-DocLayoutV3 transformer and head registrations ==='
grep -R -n -A220 -B20 -E 'class DocLayoutV3Transformer|class DocLayoutV3Head|class DocLayoutV3PostProcess' \
  ppdet/modeling 2>/dev/null | head -n 900 || true

printf '%s\n' '=== built-in distillation hooks and feature loss ==='
grep -R -n -A140 -B20 -E 'distill_pairs|class FGDFeatureLoss|class KnowledgeDistillationKLDivLoss' \
  ppdet/modeling ppdet/slim 2>/dev/null | head -n 900 || true
