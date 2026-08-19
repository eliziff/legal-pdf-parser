from __future__ import annotations

import importlib.util
import json
import signal
from types import SimpleNamespace
from pathlib import Path


TRAINING = (
    Path(__file__).resolve().parents[2]
    / "training"
    / "train_student.py"
)
SPEC = importlib.util.spec_from_file_location("ppdoc_lite_training", TRAINING)
assert SPEC and SPEC.loader
RECIPE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECIPE)

SOURCE_MANIFEST_TOOL = TRAINING.parent / "make_ppyoloe_source_manifest.py"
SOURCE_SPEC = importlib.util.spec_from_file_location(
    "ppdoc_lite_ppyoloe_source_manifest", SOURCE_MANIFEST_TOOL
)
assert SOURCE_SPEC and SOURCE_SPEC.loader
SOURCE_MANIFEST = importlib.util.module_from_spec(SOURCE_SPEC)
SOURCE_SPEC.loader.exec_module(SOURCE_MANIFEST)


class FakeTrainingProcess:
    pid = 314
    returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = -signal.SIGTERM
        return self.returncode

    def terminate(self) -> None:
        raise AssertionError("POSIX training must terminate the whole process group")

    def kill(self) -> None:
        raise AssertionError("a clean SIGTERM must not require SIGKILL")


def test_early_stop_terminates_the_posix_training_process_group(monkeypatch) -> None:
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        RECIPE,
        "os",
        SimpleNamespace(
            name="posix",
            killpg=lambda pid, sent_signal: signals.append((pid, sent_signal)),
        ),
    )

    process = FakeTrainingProcess()
    assert RECIPE.terminate_training_process(process) == -signal.SIGTERM
    assert signals == [(process.pid, signal.SIGTERM)]


def test_dfine_fast_tier_keeps_legal_ontology_and_document_safe_transforms(
    tmp_path: Path,
) -> None:
    config = tmp_path / "run" / "config.yml"
    RECIPE.build_dfine_config(
        source=Path("/opt/PaddleDetection/configs/deim/deim_dfine/deim_hgnetv2_s_132e_coco.yml"),
        destination=config,
        dataset=Path("/data/legal25"),
        annotations_dir="annotations_generalization_v1",
        pretrain=Path("/models/deim_hgnetv2_s_132e_coco.pdparams"),
        output=tmp_path / "run" / "output",
        resolution=640,
        epochs=100,
        batch_size=8,
        workers=2,
        learning_rate=0.0001,
        warmup_steps=100,
        eval_interval=5,
        log_interval=10,
        augmentation="document-safe",
        seed=20260813,
    )

    text = config.read_text(encoding="utf-8")
    assert "num_classes: 25" in text
    assert "image_shape: [3, 640, 640]" in text
    assert "deim_hgnetv2_s_132e_coco.yml" in text
    assert "RandomDistort" in text
    assert "Mosaic" not in text
    assert "RandomCrop" not in text
    assert "RandomFlip" not in text
    assert "mosaic_start_epoch: -1" in text
    assert "mosaic_epoch: -1" in text
    assert "transform_schedulers: []" in text
    assert "instance_test.json" in text


def test_ppyoloe_m_recipe_is_the_released_bbox_family_with_legal25_overrides(
    tmp_path: Path,
) -> None:
    config = tmp_path / "run" / "config.yml"
    RECIPE.build_ppyoloe_config(
        source=Path(
            "/opt/PaddleDetection/configs/ppyoloe/ppyoloe_plus_crn_m_80e_coco.yml"
        ),
        destination=config,
        dataset=Path("/data/legal25"),
        annotations_dir="annotations_generalization_v1",
        pretrain=Path("/models/ppyoloe_plus_crn_m_80e_coco.pdparams"),
        output=tmp_path / "run" / "output",
        resolution=640,
        epochs=80,
        batch_size=4,
        workers=2,
        learning_rate=0.0000625,
        warmup_steps=625,
        static_assigner_epoch=30,
        eval_interval=5,
        log_interval=10,
        augmentation="official",
        seed=20260813,
    )

    text = config.read_text(encoding="utf-8")
    assert "ppyoloe_plus_crn_m_80e_coco.yml" in text
    assert "num_classes: 25" in text
    assert "name: COCODataSet" in text
    assert "COCOInstSegDataset" not in text
    assert "gt_read_order" not in text
    assert "static_assigner_epoch: 30" in text
    assert "target_size: [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768]" in text
    assert "base_lr: 0.0000625000" in text
    assert "epochs: 5" in text
    assert "draw_threshold: 0.10" in text
    assert "RandomExpand" in text
    assert "RandomCrop" in text
    assert "image_shape: [3, 640, 640]" in text
    assert "instance_test.json" in text
    assert text.count("prefetch_factor: 1") == 3


def test_ppyoloe_s_and_m_defaults_follow_official_linear_batch_lr_scaling() -> None:
    reference_lr = 0.001
    reference_total_batch = 8 * 8

    assert RECIPE.MODEL_DEFAULTS["PP-YOLOE-S"]["learning_rate"] == (
        reference_lr * RECIPE.MODEL_DEFAULTS["PP-YOLOE-S"]["batch_size"] / reference_total_batch
    )
    assert RECIPE.MODEL_DEFAULTS["PP-YOLOE-M"]["learning_rate"] == (
        reference_lr * RECIPE.MODEL_DEFAULTS["PP-YOLOE-M"]["batch_size"] / reference_total_batch
    )


def test_ppyoloe_export_source_manifest_hash_locks_run_checkpoint_and_config(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    checkpoint = run / "output" / "best_model.pdparams"
    config = run / "config" / "PP-YOLOE-M-640.yml"
    checkpoint.parent.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    config.write_text("num_classes: 25\n", encoding="utf-8")
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "model": "PP-YOLOE-M",
                "config": {"path": str(config)},
                "dataset_contract": {"labels": list(RECIPE.EXPECTED_LABELS)},
                "test_used_for_training_or_selection": False,
                "resolution": 640,
                "epochs": 80,
                "batch_size": 4,
                "learning_rate": 0.0000625,
                "warmup_steps": 625,
                "static_assigner_epoch": 30,
                "augmentation": "official",
                "seed": 20260813,
            }
        ),
        encoding="utf-8",
    )

    manifest = SOURCE_MANIFEST.build_manifest(
        run,
        checkpoint,
        "legal25-ppyoloe-m-best",
        "PP-YOLOE-M legal25 640",
    )

    assert manifest["labels"] == list(RECIPE.EXPECTED_LABELS)
    assert set(manifest["source_files"]) == {
        "output/best_model.pdparams",
        "config/PP-YOLOE-M-640.yml",
        "run_manifest.json",
    }
    assert manifest["training"]["model"] == "PP-YOLOE-M"


def test_doclayout_v3_640_recipe_preserves_masks_order_and_document_geometry(
    tmp_path: Path,
) -> None:
    config = tmp_path / "run" / "config.yml"
    RECIPE.build_doclayout_v3_config(
        source=Path("/opt/PaddleDetection/configs/layout_analysis/PP-DocLayoutV3.yaml"),
        destination=config,
        dataset=Path("/data/legal25"),
        annotations_dir="annotations_generalization_v1",
        pretrain=Path("/models/PP-DocLayoutV3_legal25_pretrained.pdparams"),
        output=tmp_path / "run" / "output",
        resolution=640,
        epochs=30,
        batch_size=1,
        workers=2,
        learning_rate=0.00005,
        warmup_steps=20,
        eval_interval=1,
        log_interval=10,
        augmentation="document-safe",
        seed=20260813,
    )

    text = config.read_text(encoding="utf-8")
    assert "num_classes: 25" in text
    assert "image_shape: [3, 640, 640]" in text
    assert "PP-DocLayoutV3.yaml" in text
    assert "COCOInstSegDataset" in text
    assert "gt_read_order" in text
    assert "Poly2MaskPack" in text
    assert "UnpackMask" in text
    assert "RandomDistort" in text
    assert "RandomExpand" in text
    assert "RandomCrop" not in text
    assert "instance_test.json" in text
    assert "base_lr: 0.0000500000" in text
    assert "arch: 'L'" in text
    assert "num_queries: 300" in text
    assert "use_encoder_idx: [3]" in text
    assert "expansion: 1.0" in text
    assert "mask_feat_channels: [64, 64]" in text
    assert "num_top_queries: 300" in text


def test_doclayout_v3_student_uses_released_m_neck_without_changing_task_heads(
    tmp_path: Path,
) -> None:
    config = tmp_path / "run" / "config.yml"
    RECIPE.build_doclayout_v3_config(
        source=Path("/opt/PaddleDetection/configs/layout_analysis/PP-DocLayoutV3.yaml"),
        destination=config,
        dataset=Path("/data/legal25"),
        annotations_dir="annotations_generalization_v1",
        pretrain=Path("/models/PP-DocLayoutV3-M_legal25_pretrained.pdparams"),
        output=tmp_path / "run" / "output",
        resolution=640,
        epochs=30,
        batch_size=1,
        workers=2,
        learning_rate=0.00005,
        warmup_steps=20,
        eval_interval=1,
        log_interval=10,
        augmentation="document-safe",
        seed=20260813,
        backbone_arch="M",
        num_queries=100,
    )

    text = config.read_text(encoding="utf-8")
    assert "num_classes: 25" in text
    assert "PP-DocLayoutV3.yaml" in text
    assert "arch: 'M'" in text
    assert "use_encoder_idx: [2]" in text
    assert "expansion: 0.5" in text
    assert "mask_feat_channels: [64, 64]" in text
    assert "num_queries: 100" in text
    assert "num_top_queries: 100" in text
    assert "DocLayoutV3Transformer" in text
    assert "DocLayoutV3PostProcess" in text
    assert "gt_read_order" in text
    assert "COCOInstSegDataset" in text


def test_doclayout_v3_distillation_uses_l_teacher_and_shape_aligned_cwd(
    tmp_path: Path,
) -> None:
    config = tmp_path / "PP-DocLayoutV3-CWD-teacher.yml"
    RECIPE.build_doclayout_v3_cwd_distill_config(
        source=Path("/runs/legal25-m/config/PP-DocLayoutV3-640.yml"),
        destination=config,
        teacher=Path("/models/legal25-v3-l-best.pdparams"),
        resolution=640,
        encoder_weight=1.0,
        mask_weight=0.5,
        tau=1.0,
    )

    text = config.read_text(encoding="utf-8")
    assert "slim_method: PPDocV3CWD" in text
    assert "arch: 'L'" in text
    assert "num_classes: 25" in text
    assert "num_queries: 300" in text
    assert "use_encoder_idx: [3]" in text
    assert "expansion: 1.0" in text
    assert "mask_feat_channels: [64, 64]" in text
    assert "/runs/legal25-m/config/PP-DocLayoutV3-640.yml" in text
    assert "encoder_weight: 1.0" in text
    assert "mask_weight: 0.5" in text
    assert "legal25-v3-l-best.pdparams" in text


def test_doclayout_v3_distill_patch_uses_neck_features_not_query_indices() -> None:
    patch = (
        Path(__file__).resolve().parents[2]
        / "training"
        / "paddledetection_ppdocv3_cwd_distill.patch"
    ).read_text(encoding="utf-8")

    assert "PPDocV3CWDDistillModel" in patch
    assert "student_neck = student.neck(student_backbone)" in patch
    assert "teacher_neck = self.teacher_model.neck" in patch
    assert "CWDFeatureLoss(256, 256" in patch
    assert "32, 32, normalize=normalize" in patch
    assert "student_transformer" in patch
    assert "student.detr_head" in patch
