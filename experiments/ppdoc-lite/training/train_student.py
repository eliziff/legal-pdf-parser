#!/usr/bin/env python3
"""Run a pinned low-level PaddleDetection legal-layout training recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


CANDIDATES = {
    "PP-DocLayout-S": "PP-DocLayout-S.yaml",
    "PP-DocLayout-M": "PP-DocLayout-M.yaml",
    "PP-DocLayoutV3": "configs/layout_analysis/PP-DocLayoutV3.yaml",
    "DEIM-D-FINE-S": "configs/deim/deim_dfine/deim_hgnetv2_s_132e_coco.yml",
    "PP-YOLOE-S": "configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml",
    "PP-YOLOE-M": "configs/ppyoloe/ppyoloe_plus_crn_m_80e_coco.yml",
}
LOW_LEVEL_MODELS = {"DEIM-D-FINE-S", "PP-DocLayoutV3", "PP-YOLOE-S", "PP-YOLOE-M"}
MODEL_DEFAULTS = {
    "PP-DocLayout-S": {
        "resolution": 480, "epochs": 100, "batch_size": 1,
        "learning_rate": 0.0001, "warmup_steps": 100,
        "static_fraction": 0.1, "eval_interval": 1,
    },
    "PP-DocLayout-M": {
        "resolution": 640, "epochs": 100, "batch_size": 1,
        "learning_rate": 0.0001, "warmup_steps": 100,
        "static_fraction": 0.1, "eval_interval": 1,
    },
    "PP-DocLayoutV3": {
        "resolution": 640, "epochs": 30, "batch_size": 1,
        "learning_rate": 0.00005, "warmup_steps": 20,
        "static_fraction": 0.0, "eval_interval": 1,
    },
    "DEIM-D-FINE-S": {
        "resolution": 640, "epochs": 100, "batch_size": 8,
        "learning_rate": 0.0001, "warmup_steps": 100,
        "static_fraction": 0.0, "eval_interval": 5,
    },
    "PP-YOLOE-S": {
        "resolution": 640, "epochs": 80, "batch_size": 8,
        "learning_rate": 0.000125, "warmup_steps": 0,
        "static_fraction": 0.375, "eval_interval": 5,
    },
    "PP-YOLOE-M": {
        "resolution": 640, "epochs": 80, "batch_size": 4,
        "learning_rate": 0.0000625, "warmup_steps": 0,
        "static_fraction": 0.375, "eval_interval": 5,
    },
}
PPDOCV3_VARIANTS = {
    # These are PaddleDetection's released Mask RT-DETR HGNetV2 size recipes.
    # The PP-DocLayoutV3 transformer, mask head, and reading-order head stay unchanged.
    "L": {
        "use_encoder_idx": [3],
        "expansion": 1.0,
        "mask_feat_channels": [64, 64],
    },
    "M": {
        "use_encoder_idx": [2],
        "expansion": 0.5,
        "mask_feat_channels": [64, 64],
    },
    "S": {
        "use_encoder_idx": [2],
        "expansion": 0.5,
        "mask_feat_channels": [64, 32],
    },
}
EXPECTED_PAGES = {"train": 499, "val": 75, "test": 87}
EXPECTED_LABELS = (
    "abstract", "algorithm", "chart", "content", "display_formula", "doc_title",
    "figure_title", "footer", "footer_image", "footnote", "formula_number",
    "header", "header_image", "image", "number", "paragraph_title", "reference",
    "reference_content", "seal", "table", "text", "vertical_text",
    "vision_footnote", "block_quote", "byline",
)
EPOCH_RE = re.compile(r"Epoch:\s+\[(\d+)\]\s+\[\s*(\d+)/(\d+)\]")
AP_RE = re.compile(
    r"Average Precision\s+\(AP\) @\[ IoU=0\.50:0\.95 \| area=\s+all \| maxDets=100 \] = ([0-9.]+)"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def terminate_training_process(
    process: subprocess.Popen[str],
    timeout: float = 10.0,
) -> int:
    """Stop the trainer and every worker that inherited its output pipe."""
    if process.poll() is not None:
        return int(process.returncode)
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        return process.wait()


def load_annotations(dataset: Path, annotations_dir: str, split: str) -> tuple[Path, dict[str, Any]]:
    path = dataset / annotations_dir / f"instance_{split}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, json.loads(path.read_text(encoding="utf-8"))


def validate_dataset(
    dataset: Path,
    annotations_dir: str,
    *,
    require_layout_fields: bool = False,
) -> dict[str, Any]:
    if not (dataset / "images").is_dir():
        raise FileNotFoundError(dataset / "images")
    receipt: dict[str, Any] = {"annotations_dir": annotations_dir, "splits": {}}
    for split, expected_pages in EXPECTED_PAGES.items():
        path, payload = load_annotations(dataset, annotations_dir, split)
        labels = tuple(
            str(category["name"])
            for category in sorted(payload.get("categories", []), key=lambda item: int(item["id"]))
        )
        ids = tuple(
            int(category["id"])
            for category in sorted(payload.get("categories", []), key=lambda item: int(item["id"]))
        )
        if labels != EXPECTED_LABELS or ids != tuple(range(len(EXPECTED_LABELS))):
            raise ValueError(f"{split} does not implement the contiguous legal-25 contract")
        if len(payload.get("images", [])) != expected_pages:
            raise ValueError(f"{split} has {len(payload.get('images', []))} pages; expected {expected_pages}")
        if require_layout_fields:
            orders_by_image: dict[int, list[int]] = {}
            for annotation in payload.get("annotations", []):
                if not annotation.get("segmentation"):
                    raise ValueError(f"{split} annotation {annotation.get('id')} has no segmentation")
                if "read_order" not in annotation:
                    raise ValueError(f"{split} annotation {annotation.get('id')} has no read_order")
                orders_by_image.setdefault(int(annotation["image_id"]), []).append(
                    int(annotation["read_order"])
                )
            for image_id, orders in orders_by_image.items():
                if sorted(orders) != list(range(len(orders))):
                    raise ValueError(f"{split} image {image_id} has non-contiguous read_order values")
        receipt["splits"][split] = {
            "pages": expected_pages,
            "annotations": len(payload.get("annotations", [])),
            "sha256": sha256(path),
        }
    receipt["labels"] = list(EXPECTED_LABELS)
    return receipt


def prepare_smoke_dataset(
    dataset: Path,
    annotations_dir: str,
    destination: Path,
    page_count: int,
) -> Path:
    _, train = load_annotations(dataset, annotations_dir, "train")
    annotated_ids = {int(annotation["image_id"]) for annotation in train["annotations"]}
    images = sorted(
        (item for item in train["images"] if int(item["id"]) in annotated_ids),
        key=lambda item: int(item["id"]),
    )[:page_count]
    if len(images) != page_count:
        raise ValueError(f"Smoke dataset needs {page_count} annotated pages; found {len(images)}")
    image_ids = {int(image["id"]) for image in images}
    smoke = {
        **train,
        "images": images,
        "annotations": [
            annotation for annotation in train["annotations"]
            if int(annotation["image_id"]) in image_ids
        ],
    }
    annotations = destination / annotations_dir
    annotations.mkdir(parents=True, exist_ok=True)
    for split in EXPECTED_PAGES:
        write_json(annotations / f"instance_{split}.json", smoke)
    images = destination / "images"
    if images.is_symlink():
        if images.resolve() != (dataset / "images").resolve():
            raise ValueError(f"Smoke image link targets the wrong dataset: {images}")
    elif images.exists():
        raise FileExistsError(images)
    else:
        os.symlink((dataset / "images").resolve(), images, target_is_directory=True)
    return destination


def replace_transform(transforms: list[dict[str, Any]], name: str, values: dict[str, Any]) -> None:
    for transform in transforms:
        if name in transform:
            transform[name].update(values)
            return
    raise KeyError(f"Missing {name} transform")


def build_dfine_config(
    source: Path,
    destination: Path,
    dataset: Path,
    annotations_dir: str,
    pretrain: Path,
    output: Path,
    resolution: int,
    epochs: int,
    batch_size: int,
    workers: int,
    learning_rate: float,
    warmup_steps: int,
    eval_interval: int,
    log_interval: int,
    augmentation: str,
    seed: int,
) -> None:
    """Write a legal-page override on PaddleDetection's released D-FINE-S recipe."""
    scales = list(range(resolution - 64, resolution + 65, 32))
    start_decay = max(1, round(epochs * 0.6))
    last_plateau = min(5, max(1, epochs // 10))
    train_reader = ""
    if augmentation == "document-safe":
        train_reader = f"""
TrainReader:
  sample_transforms:
    - Decode: {{}}
    - RandomDistort: {{prob: 0.5}}
  batch_transforms:
    - BatchRandomResize: {{target_size: {scales}, random_size: True, random_interp: True, keep_ratio: False}}
    - NormalizeImage: {{mean: [0., 0., 0.], std: [1., 1., 1.], norm_type: none}}
    - NormalizeBox: {{}}
    - BboxXYXY2XYWH: {{}}
    - Permute: {{}}
  batch_size: {batch_size}
  shuffle: true
  drop_last: true
  collate_batch: false
  use_shared_memory: false
  mosaic_start_epoch: -1
  mosaic_epoch: -1
  transform_schedulers: []
"""
    else:
        train_reader = f"""
TrainReader:
  batch_size: {batch_size}
  use_shared_memory: false
"""
    config = f"""# Legal-layout override on PaddleDetection 2.9's official D-FINE-S recipe.
_BASE_: ['{source.as_posix()}']

epoch: {epochs}
snapshot_epoch: {eval_interval}
log_iter: {log_interval}
save_dir: {output.as_posix()}
output_eval: {(output.parent / 'eval').as_posix()}
num_classes: {len(EXPECTED_LABELS)}
worker_num: {workers}
eval_size: [{resolution}, {resolution}]
seed: {seed}
use_gpu: true
use_ema: true
use_shared_memory: false
pretrain_weights: {pretrain.as_posix()}

TrainDataset:
  name: COCODataSet
  image_dir: images
  anno_path: {annotations_dir}/instance_train.json
  dataset_dir: {dataset.as_posix()}
  data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']
  allow_empty: false
EvalDataset:
  name: COCODataSet
  image_dir: images
  anno_path: {annotations_dir}/instance_val.json
  dataset_dir: {dataset.as_posix()}
  allow_empty: true
TestDataset:
  name: ImageFolder
  image_dir: images
  anno_path: {annotations_dir}/instance_test.json
  dataset_dir: {dataset.as_posix()}
{train_reader}
EvalReader:
  sample_transforms:
    - Decode: {{}}
    - Resize: {{target_size: [{resolution}, {resolution}], keep_ratio: False, interp: 1}}
    - NormalizeImage: {{mean: [0., 0., 0.], std: [1., 1., 1.], norm_type: none}}
    - Permute: {{}}
  batch_size: {min(batch_size, 8)}
  shuffle: false
  drop_last: false
TestReader:
  inputs_def:
    image_shape: [3, {resolution}, {resolution}]
  sample_transforms:
    - Decode: {{}}
    - Resize: {{target_size: [{resolution}, {resolution}], keep_ratio: False, interp: 1}}
    - NormalizeImage: {{mean: [0., 0., 0.], std: [1., 1., 1.], norm_type: none}}
    - Permute: {{}}
  batch_size: 1
  shuffle: false
  drop_last: false

LearningRate:
  base_lr: {learning_rate:.10f}
  schedulers:
    - !CosineDecay
      start_epochs: {start_decay}
      max_epochs: {epochs}
      min_lr_ratio: 0.5
      last_plateau_epochs: {last_plateau}
    - !ExpWarmup
      steps: {warmup_steps}
"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(config, encoding="utf-8")


def build_ppyoloe_config(
    source: Path,
    destination: Path,
    dataset: Path,
    annotations_dir: str,
    pretrain: Path,
    output: Path,
    resolution: int,
    epochs: int,
    batch_size: int,
    workers: int,
    learning_rate: float,
    warmup_steps: int,
    static_assigner_epoch: int,
    eval_interval: int,
    log_interval: int,
    augmentation: str,
    seed: int,
    fixed_train_size: int | None = None,
) -> None:
    """Write a legal-page override on PaddleDetection's released PP-YOLOE+ recipe."""
    scales = (
        [fixed_train_size]
        if fixed_train_size is not None
        else list(range(resolution - 320, resolution + 129, 32))
    )
    steps_per_epoch = max(1, math.ceil(EXPECTED_PAGES["train"] / batch_size))
    warmup_epochs = math.ceil(warmup_steps / steps_per_epoch)
    sample_transforms = """    - Decode: {}
    - RandomDistort: {}
    - RandomExpand: {fill_value: [123.675, 116.28, 103.53]}
    - RandomCrop: {}
    - RandomFlip: {}
"""
    if augmentation == "document-safe":
        sample_transforms = """    - Decode: {}
    - RandomDistort: {}
    - RandomFlip: {}
"""
    config = f"""# Legal-layout override on PaddleDetection's official PP-YOLOE+ recipe.
_BASE_: ['{source.as_posix()}']

epoch: {epochs}
snapshot_epoch: {eval_interval}
log_iter: {log_interval}
save_dir: {output.as_posix()}
output_eval: {(output.parent / 'eval').as_posix()}
num_classes: {len(EXPECTED_LABELS)}
worker_num: {workers}
eval_size: [{resolution}, {resolution}]
draw_threshold: 0.10
seed: {seed}
use_gpu: true
use_ema: true
use_shared_memory: false
pretrain_weights: {pretrain.as_posix()}

PPYOLOEHead:
  static_assigner_epoch: {static_assigner_epoch}

TrainDataset:
  name: COCODataSet
  image_dir: images
  anno_path: {annotations_dir}/instance_train.json
  dataset_dir: {dataset.as_posix()}
  data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']
  allow_empty: false
EvalDataset:
  name: COCODataSet
  image_dir: images
  anno_path: {annotations_dir}/instance_val.json
  dataset_dir: {dataset.as_posix()}
  allow_empty: true
TestDataset:
  name: ImageFolder
  image_dir: images
  anno_path: {annotations_dir}/instance_test.json
  dataset_dir: {dataset.as_posix()}

TrainReader:
  sample_transforms:
{sample_transforms}  batch_transforms:
    - BatchRandomResize: {{target_size: {scales}, random_size: true, random_interp: true, keep_ratio: false}}
    - NormalizeImage: {{mean: [0., 0., 0.], std: [1., 1., 1.], norm_type: none}}
    - Permute: {{}}
    - PadGT: {{}}
  batch_size: {batch_size}
  shuffle: true
  drop_last: true
  use_shared_memory: false
  prefetch_factor: 1
  collate_batch: true
EvalReader:
  sample_transforms:
    - Decode: {{}}
    - Resize: {{target_size: [{resolution}, {resolution}], keep_ratio: false, interp: 2}}
    - NormalizeImage: {{mean: [0., 0., 0.], std: [1., 1., 1.], norm_type: none}}
    - Permute: {{}}
  batch_size: {min(batch_size, 8)}
  shuffle: false
  drop_last: false
  prefetch_factor: 1
TestReader:
  inputs_def:
    image_shape: [3, {resolution}, {resolution}]
  sample_transforms:
    - Decode: {{}}
    - Resize: {{target_size: [{resolution}, {resolution}], keep_ratio: false, interp: 2}}
    - NormalizeImage: {{mean: [0., 0., 0.], std: [1., 1., 1.], norm_type: none}}
    - Permute: {{}}
  batch_size: 1
  shuffle: false
  drop_last: false
  prefetch_factor: 1

LearningRate:
  base_lr: {learning_rate:.10f}
  schedulers:
    - name: CosineDecay
      max_epochs: {epochs}
    - name: LinearWarmup
      start_factor: 0.0
      epochs: {warmup_epochs}
"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(config, encoding="utf-8")


def build_doclayout_v3_config(
    source: Path,
    destination: Path,
    dataset: Path,
    annotations_dir: str,
    pretrain: Path,
    output: Path,
    resolution: int,
    epochs: int,
    batch_size: int,
    workers: int,
    learning_rate: float,
    warmup_steps: int,
    eval_interval: int,
    log_interval: int,
    augmentation: str,
    seed: int,
    backbone_arch: str = "L",
    num_queries: int = 300,
) -> None:
    """Write a legal PP-DocLayoutV3 recipe with a released HGNetV2 size."""
    try:
        variant = PPDOCV3_VARIANTS[backbone_arch]
    except KeyError as error:
        raise ValueError(f"Unsupported PP-DocLayoutV3 backbone: {backbone_arch}") from error
    if num_queries < 1:
        raise ValueError("PP-DocLayoutV3 query count must be positive")
    scales = list(range(resolution - 64, resolution + 65, 32))
    crop = "    - RandomCrop: {prob: 0.8, use_box_candidates: true}\n"
    if augmentation == "document-safe":
        crop = ""
    config = f"""# Legal-layout override on PaddleDetection 2.9's official PP-DocLayoutV3 recipe.
_BASE_: ['{source.as_posix()}']

epoch: {epochs}
snapshot_epoch: {eval_interval}
log_iter: {log_interval}
save_dir: {output.as_posix()}
output_eval: {(output.parent / 'eval').as_posix()}
num_classes: {len(EXPECTED_LABELS)}
worker_num: {workers}
eval_size: [{resolution}, {resolution}]
seed: {seed}
use_gpu: true
use_ema: true
use_shared_memory: false
pretrain_weights: {pretrain.as_posix()}

# Keep PP-DocLayoutV3's task-specific transformer, mask, and reading-order
# components. Only use PaddleDetection's released Mask RT-DETR HGNetV2 size
# settings for the backbone and its dynamically constructed neck.
PPHGNetV2:
  arch: '{backbone_arch}'

MaskHybridEncoder:
  use_encoder_idx: {variant['use_encoder_idx']}
  expansion: {variant['expansion']}
  mask_feat_channels: {variant['mask_feat_channels']}

DocLayoutV3Transformer:
  num_queries: {num_queries}

DocLayoutV3PostProcess:
  num_top_queries: {num_queries}

DocLayoutV3Metric:
  eval_mask: false

TrainDataset:
  name: COCOInstSegDataset
  image_dir: images
  anno_path: {annotations_dir}/instance_train.json
  dataset_dir: {dataset.as_posix()}
  data_fields: ['image', 'gt_bbox', 'gt_class', 'gt_poly', 'is_crowd', 'gt_read_order']
  allow_empty: false
EvalDataset:
  name: COCOInstSegDataset
  image_dir: images
  anno_path: {annotations_dir}/instance_val.json
  dataset_dir: {dataset.as_posix()}
  allow_empty: true
TestDataset:
  name: ImageFolder
  image_dir: images
  anno_path: {annotations_dir}/instance_test.json
  dataset_dir: {dataset.as_posix()}

TrainReader:
  sample_transforms:
    - Decode: {{}}
    - Poly2MaskPack: {{del_poly: true}}
    - RandomDistort: {{prob: 0.8}}
    - UpdateBBoxFromMask: {{}}
    - RandomExpand: {{prob: 0.5, ratio: 1.5, fill_value: [123.675, 116.28, 103.53]}}
{crop}  batch_transforms:
    - BatchRandomResize: {{target_size: {scales}, random_size: true, random_interp: true, keep_ratio: false}}
    - UnpackMask: {{}}
    - NormalizeImage: {{mean: [0., 0., 0.], std: [1., 1., 1.], norm_type: none}}
    - NormalizeBox: {{}}
    - BboxXYXY2XYWH: {{}}
    - Permute: {{}}
  batch_size: {batch_size}
  shuffle: true
  drop_last: true
  collate_batch: false
  use_shared_memory: false
EvalReader:
  sample_transforms:
    - Decode: {{}}
    - Resize: {{target_size: [{resolution}, {resolution}], keep_ratio: false, interp: 2}}
    - NormalizeImage: {{mean: [0., 0., 0.], std: [1., 1., 1.], norm_type: none}}
    - Permute: {{}}
  batch_size: 1
  shuffle: false
  drop_last: false
TestReader:
  inputs_def:
    image_shape: [3, {resolution}, {resolution}]
  sample_transforms:
    - Decode: {{}}
    - Resize: {{target_size: [{resolution}, {resolution}], keep_ratio: false, interp: 2}}
    - NormalizeImage: {{mean: [0., 0., 0.], std: [1., 1., 1.], norm_type: none}}
    - Permute: {{}}
  batch_size: 1
  shuffle: false
  drop_last: false

LearningRate:
  base_lr: {learning_rate:.10f}
  schedulers:
    - !PiecewiseDecay
      gamma: 1.0
      milestones: [{max(1, round(epochs * 0.7))}]
      use_warmup: true
    - !LinearWarmup
      start_factor: 0.001
      steps: {warmup_steps}
"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(config, encoding="utf-8")


def build_doclayout_v3_cwd_distill_config(
    source: Path,
    destination: Path,
    teacher: Path,
    resolution: int,
    encoder_weight: float,
    mask_weight: float,
    tau: float,
) -> None:
    """Write the L-teacher side of the PP-DocLayoutV3 CWD recipe."""
    if encoder_weight < 0 or mask_weight < 0 or tau <= 0:
        raise ValueError("Distillation weights must be non-negative and tau positive")
    config = f"""# PP-DocLayoutV3 L teacher for shape-aligned neck distillation.
_BASE_: ['{source.as_posix()}']

pretrain_weights: {teacher.as_posix()}
eval_size: [{resolution}, {resolution}]
num_classes: {len(EXPECTED_LABELS)}

PPHGNetV2:
  arch: 'L'

MaskHybridEncoder:
  use_encoder_idx: [3]
  expansion: 1.0
  mask_feat_channels: [64, 64]

DocLayoutV3Transformer:
  num_queries: 300

slim: Distill
slim_method: PPDocV3CWD

PPDocV3CWD:
  encoder_weight: {encoder_weight}
  mask_weight: {mask_weight}
  tau: {tau}
  normalize: true
"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(config, encoding="utf-8")


def build_config(
    model: str,
    source: Path,
    destination: Path,
    dataset: Path,
    annotations_dir: str,
    pretrain: Path,
    output: Path,
    resolution: int,
    epochs: int,
    batch_size: int,
    workers: int,
    learning_rate: float,
    warmup_steps: int,
    static_assigner_epoch: int,
    eval_interval: int,
    log_interval: int,
    augmentation: str,
    seed: int,
    backbone_arch: str = "L",
    num_queries: int = 300,
    fixed_train_size: int | None = None,
) -> dict[str, Any]:
    if model == "DEIM-D-FINE-S":
        build_dfine_config(
            source, destination, dataset, annotations_dir, pretrain, output,
            resolution, epochs, batch_size, workers, learning_rate,
            warmup_steps, eval_interval, log_interval, augmentation, seed,
        )
        return {}
    if model in {"PP-YOLOE-S", "PP-YOLOE-M"}:
        build_ppyoloe_config(
            source,
            destination,
            dataset,
            annotations_dir,
            pretrain,
            output,
            resolution,
            epochs,
            batch_size,
            workers,
            learning_rate,
            warmup_steps,
            static_assigner_epoch,
            eval_interval,
            log_interval,
            augmentation,
            seed,
            fixed_train_size,
        )
        return {}
    if model == "PP-DocLayoutV3":
        build_doclayout_v3_config(
            source, destination, dataset, annotations_dir, pretrain, output,
            resolution, epochs, batch_size, workers, learning_rate,
            warmup_steps, eval_interval, log_interval, augmentation, seed,
            backbone_arch, num_queries,
        )
        return {}
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    scales = list(range(resolution - 64, resolution + 65, 32))
    config.update(
        {
            "epoch": epochs,
            "num_classes": len(EXPECTED_LABELS),
            "worker_num": workers,
            "eval_height": resolution,
            "eval_width": resolution,
            "eval_size": [resolution, resolution],
            "save_dir": str(output),
            "output_eval": str(output.parent / "eval"),
            "snapshot_epoch": eval_interval,
            "log_iter": log_interval,
            "seed": seed,
            "use_gpu": True,
            "use_ema": True,
            "use_shared_memory": False,
            "pretrain_weights": str(pretrain),
        }
    )
    for split, dataset_key in (("train", "TrainDataset"), ("val", "EvalDataset"), ("test", "TestDataset")):
        config[dataset_key].update(
            {
                "dataset_dir": str(dataset),
                "image_dir": "images",
                "anno_path": f"{annotations_dir}/instance_{split}.json",
            }
        )
    config["TrainReader"]["batch_size"] = batch_size
    config["EvalReader"]["batch_size"] = min(batch_size, 8)
    replace_transform(
        config["TrainReader"]["batch_transforms"],
        "BatchRandomResize",
        {"target_size": scales, "random_size": True, "random_interp": True, "keep_ratio": False},
    )
    if augmentation == "document-safe":
        config["TrainReader"]["sample_transforms"] = [
            transform
            for transform in config["TrainReader"]["sample_transforms"]
            if "RandomCrop" not in transform
        ]
    replace_transform(
        config["EvalReader"]["sample_transforms"],
        "Resize",
        {"target_size": [resolution, resolution], "keep_ratio": False},
    )
    replace_transform(
        config["TestReader"]["sample_transforms"],
        "Resize",
        {"target_size": [resolution, resolution], "keep_ratio": False},
    )
    config["TestReader"]["inputs_def"]["image_shape"] = [3, resolution, resolution]
    config["PicoHeadV2"]["static_assigner_epoch"] = static_assigner_epoch
    config["LearningRate"]["base_lr"] = learning_rate
    for scheduler in config["LearningRate"]["schedulers"]:
        if scheduler["name"] == "CosineDecay":
            scheduler["max_epochs"] = epochs
        elif scheduler["name"] == "LinearWarmup":
            scheduler["steps"] = warmup_steps
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--annotations-dir", default="annotations_generalization_v1")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--pretrain", type=Path, required=True)
    parser.add_argument("--mode", choices=("check", "smoke", "train", "eval"), default="train")
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--static-assigner-epoch", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--fixed-train-size",
        type=int,
        help="PP-YOLOE preflight only: replace multiscale training with one fixed size.",
    )
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-epoch", type=int)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.002)
    parser.add_argument("--augmentation", choices=("official", "document-safe"), default="official")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--backbone-arch",
        choices=tuple(PPDOCV3_VARIANTS),
        default="L",
        help="PP-DocLayoutV3 HGNetV2 size; ignored by other model families.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=300,
        help="PP-DocLayoutV3 decoder/query count; ignored by other model families.",
    )
    parser.add_argument(
        "--distill-teacher",
        type=Path,
        help="Legal PP-DocLayoutV3 L checkpoint for shape-aligned CWD feature distillation.",
    )
    parser.add_argument("--distill-encoder-weight", type=float, default=1.0)
    parser.add_argument("--distill-mask-weight", type=float, default=1.0)
    parser.add_argument("--distill-tau", type=float, default=1.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run a bounded CPU preflight; production training defaults to GPU.",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--paddledetection-root",
        type=Path,
        help="Use an existing PaddleDetection checkout for a low-level model recipe.",
    )
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    run_root = args.run_root.resolve()
    output = run_root / "output"
    status_path = run_root / "status.json"
    run_root.mkdir(parents=True, exist_ok=True)
    dataset_receipt = validate_dataset(
        dataset,
        args.annotations_dir,
        require_layout_fields=args.model == "PP-DocLayoutV3",
    )

    if args.paddledetection_root:
        if args.model not in LOW_LEVEL_MODELS:
            raise ValueError(
                "--paddledetection-root supports only the low-level model recipes"
            )
        detection_repo = args.paddledetection_root.resolve()
        source_config = detection_repo / CANDIDATES[args.model]
    else:
        import paddlex

        paddlex_root = Path(paddlex.__file__).resolve().parent
        detection_repo = paddlex_root / "repo_manager/repos/PaddleDetection"
        if args.model in LOW_LEVEL_MODELS:
            source_config = detection_repo / CANDIDATES[args.model]
        else:
            source_config = paddlex_root / "repo_apis/PaddleDetection_api/configs" / CANDIDATES[args.model]
    trainer = detection_repo / "tools/train.py"
    for required in (source_config, trainer, args.pretrain):
        if not required.is_file():
            raise FileNotFoundError(required)

    defaults = MODEL_DEFAULTS[args.model]
    resolution = args.resolution or int(defaults["resolution"])
    batch_size = args.batch_size if args.batch_size is not None else int(defaults["batch_size"])
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else float(defaults["learning_rate"])
    )
    eval_interval = (
        args.eval_interval
        if args.eval_interval is not None
        else int(defaults["eval_interval"])
    )
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if eval_interval < 1:
        raise ValueError("evaluation interval must be positive")
    if args.fixed_train_size is not None:
        if args.model not in {"PP-YOLOE-S", "PP-YOLOE-M"}:
            raise ValueError("--fixed-train-size is supported only for PP-YOLOE")
        if args.fixed_train_size < 32:
            raise ValueError("fixed train size must be at least 32 pixels")
    if args.distill_teacher and args.model != "PP-DocLayoutV3":
        raise ValueError("Feature distillation is supported only for PP-DocLayoutV3")
    if args.distill_teacher and args.mode == "eval":
        raise ValueError("Evaluate the exported student directly, without a distillation teacher")
    if args.distill_teacher and not args.distill_teacher.is_file():
        raise FileNotFoundError(args.distill_teacher)
    if args.cpu and not args.no_amp:
        raise ValueError("CPU preflights require --no-amp")
    epochs = 1 if args.mode == "smoke" else (args.epochs or int(defaults["epochs"]))
    train_pages = batch_size if args.mode == "smoke" else EXPECTED_PAGES["train"]
    steps_per_epoch = max(1, math.ceil(train_pages / batch_size))
    if args.warmup_steps is not None:
        warmup_steps = args.warmup_steps
    elif args.mode == "smoke":
        warmup_steps = 1
    elif args.model in {"PP-YOLOE-S", "PP-YOLOE-M"}:
        # PaddleDetection's released PP-YOLOE+ recipes warm up for five epochs.
        warmup_steps = 5 * steps_per_epoch
    else:
        warmup_steps = int(defaults["warmup_steps"])
    if args.model in {"DEIM-D-FINE-S", "PP-DocLayoutV3"}:
        static_assigner_epoch = 0
    else:
        static_assigner_epoch = (
            args.static_assigner_epoch
            if args.static_assigner_epoch is not None
            else max(1, round(epochs * float(defaults["static_fraction"])))
        )
    if not 0 <= static_assigner_epoch <= epochs:
        raise ValueError("static assigner epoch must fall within the run")
    if args.early_stop_patience < 0:
        raise ValueError("early-stop patience cannot be negative")
    early_stop_min_epoch = args.early_stop_min_epoch or (
        static_assigner_epoch + eval_interval
    )
    if args.mode == "smoke":
        dataset = prepare_smoke_dataset(
            dataset,
            args.annotations_dir,
            run_root / "smoke_dataset",
            batch_size,
        )

    config_path = run_root / "config" / f"{args.model}-{resolution}.yml"
    build_config(
        args.model,
        source_config,
        config_path,
        dataset,
        args.annotations_dir,
        args.pretrain.resolve(),
        output,
        resolution,
        epochs,
        batch_size,
        args.workers,
        learning_rate,
        warmup_steps,
        static_assigner_epoch,
        eval_interval,
        args.log_interval,
        args.augmentation,
        args.seed,
        args.backbone_arch,
        args.num_queries,
        args.fixed_train_size,
    )
    slim_config_path = None
    if args.distill_teacher:
        slim_config_path = run_root / "config" / "PP-DocLayoutV3-CWD-teacher.yml"
        build_doclayout_v3_cwd_distill_config(
            config_path,
            slim_config_path,
            args.distill_teacher.resolve(),
            resolution,
            args.distill_encoder_weight,
            args.distill_mask_weight,
            args.distill_tau,
        )
    manifest = {
        "schema_version": "legalpdf.ppdoc_lite_student_run.v3",
        "model": args.model,
        "backbone_arch": args.backbone_arch if args.model == "PP-DocLayoutV3" else None,
        "num_queries": args.num_queries if args.model == "PP-DocLayoutV3" else None,
        "mode": args.mode,
        "dataset": str(dataset),
        "source_dataset": str(args.dataset.resolve()),
        "dataset_contract": dataset_receipt,
        "test_used_for_training_or_selection": False,
        "resolution": resolution,
        "epochs": epochs,
        "batch_size": batch_size,
        "workers": args.workers,
        "fixed_train_size": args.fixed_train_size,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "static_assigner_epoch": static_assigner_epoch,
        "eval_interval": eval_interval,
        "log_interval": args.log_interval,
        "early_stop": {
            "patience_evaluations": args.early_stop_patience,
            "min_epoch": early_stop_min_epoch,
            "min_delta": args.early_stop_min_delta,
        },
        "augmentation": args.augmentation,
        "seed": args.seed,
        "amp": not args.no_amp,
        "device": "cpu" if args.cpu else "gpu",
        "pretrain": {"path": str(args.pretrain.resolve()), "sha256": sha256(args.pretrain)},
        "distillation": (
            {
                "method": "PPDocV3CWD",
                "teacher": {
                    "path": str(args.distill_teacher.resolve()),
                    "sha256": sha256(args.distill_teacher),
                },
                "encoder_weight": args.distill_encoder_weight,
                "mask_weight": args.distill_mask_weight,
                "tau": args.distill_tau,
                "config": {
                    "path": str(slim_config_path),
                    "sha256": sha256(slim_config_path),
                },
            }
            if args.distill_teacher else None
        ),
        "source_config": {"path": str(source_config), "sha256": sha256(source_config)},
        "paddledetection": {
            "root": str(detection_repo),
            "trainer_sha256": sha256(trainer),
        },
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "output": str(output),
        "resume": str(args.resume.resolve()) if args.resume else None,
        "started_at": utc_now(),
    }
    write_json(run_root / "run_manifest.json", manifest)
    if args.mode == "check":
        write_json(status_path, {**manifest, "phase": "complete", "finished_at": utc_now()})
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
        return 0

    if args.mode == "eval":
        command = [
            sys.executable,
            str(detection_repo / "tools/eval.py"),
            "-c",
            str(config_path),
            "-o",
            f"weights={args.pretrain.resolve()}",
            "--classwise",
            "--output_eval",
            str(run_root / "eval"),
        ]
    else:
        command = [sys.executable, str(trainer), "-c", str(config_path), "--eval"]
        if args.cpu:
            command.extend(["-o", "use_gpu=False"])
        if slim_config_path:
            command.extend(["--slim_config", str(slim_config_path)])
        if not args.no_amp:
            command.append("--amp")
        if args.resume:
            command.extend(["--resume", str(args.resume.resolve())])
    write_json(status_path, {**manifest, "phase": "running", "command": command})
    print(json.dumps({**manifest, "command": command}, indent=2, sort_keys=True), flush=True)
    log_path = run_root / "train.log"
    process_env = os.environ.copy()
    cudnn_lib = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
        / "cudnn"
        / "lib"
    )
    if cudnn_lib.is_dir():
        existing_library_path = process_env.get("LD_LIBRARY_PATH")
        process_env["LD_LIBRARY_PATH"] = (
            f"{cudnn_lib}:{existing_library_path}"
            if existing_library_path
            else str(cudnn_lib)
        )
    progress: dict[str, Any] = {}
    best_ap = float("-inf")
    early_stop_reference_ap = float("-inf")
    stale_evaluations = 0
    evaluation_history: list[dict[str, Any]] = []
    stop_requested = False
    stopped_early = False
    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            process = subprocess.Popen(
                command,
                cwd=detection_repo,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=process_env,
                start_new_session=os.name == "posix",
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
                if epoch_match := EPOCH_RE.search(line):
                    progress.update(
                        {
                            "epoch": int(epoch_match.group(1)) + 1,
                            "batch": int(epoch_match.group(2)) + 1,
                            "total_batches": int(epoch_match.group(3)),
                        }
                    )
                    write_json(
                        status_path,
                        {**manifest, "phase": "running", "command": command, "progress": progress},
                    )
                    if stop_requested:
                        # The first logged batch of the following epoch proves the
                        # preceding evaluation and all of its callbacks/checkpoint
                        # writes completed, including the no-improvement case.
                        stopped_early = True
                        terminate_training_process(process)
                        break
                if ap_match := AP_RE.search(line):
                    ap = float(ap_match.group(1))
                    epoch = int(progress.get("epoch", 0))
                    progress["validation_ap"] = ap
                    best_ap = max(best_ap, ap)
                    evaluation_history.append({"epoch": epoch, "bbox_ap": ap})
                    if epoch < early_stop_min_epoch:
                        stale_evaluations = 0
                    elif early_stop_reference_ap == float("-inf"):
                        early_stop_reference_ap = ap
                        stale_evaluations = 0
                    elif ap > early_stop_reference_ap + args.early_stop_min_delta:
                        early_stop_reference_ap = ap
                        stale_evaluations = 0
                    else:
                        stale_evaluations += 1
                    progress.update(
                        {
                            "best_validation_ap": best_ap,
                            "early_stop_reference_ap": (
                                None
                                if early_stop_reference_ap == float("-inf")
                                else early_stop_reference_ap
                            ),
                            "stale_evaluations": stale_evaluations,
                            "evaluation_history": evaluation_history,
                        }
                    )
                    write_json(
                        status_path,
                        {**manifest, "phase": "running", "command": command, "progress": progress},
                    )
                    if (
                        args.mode == "train"
                        and args.early_stop_patience
                        and int(progress.get("epoch", 0)) >= early_stop_min_epoch
                        and stale_evaluations >= args.early_stop_patience
                    ):
                        stop_requested = True
            return_code = process.wait()
        if return_code:
            if not stopped_early:
                raise subprocess.CalledProcessError(return_code, command)
    except BaseException as exc:
        write_json(
            status_path,
            {**manifest, "phase": "failed", "finished_at": utc_now(), "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    write_json(
        status_path,
        {
            **manifest,
            "phase": "complete",
            "finished_at": utc_now(),
            "progress": progress,
            "stopped_early": stopped_early,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
