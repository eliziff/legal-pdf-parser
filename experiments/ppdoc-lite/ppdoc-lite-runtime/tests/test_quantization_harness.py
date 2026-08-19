from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools" / "quantize.py"
SPEC = importlib.util.spec_from_file_location("ppdoc_lite_quantize", TOOLS)
assert SPEC and SPEC.loader
QUANTIZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUANTIZE)

NNCF_TOOLS = TOOLS.with_name("quantize_nncf.py")
sys.path.insert(0, str(NNCF_TOOLS.parent))
NNCF_SPEC = importlib.util.spec_from_file_location("ppdoc_lite_quantize_nncf", NNCF_TOOLS)
assert NNCF_SPEC and NNCF_SPEC.loader
QUANTIZE_NNCF = importlib.util.module_from_spec(NNCF_SPEC)
NNCF_SPEC.loader.exec_module(QUANTIZE_NNCF)


def test_class_journal_calibration_selection_is_reproducible_and_covers_rare_classes(
    tmp_path: Path,
) -> None:
    images = []
    annotations = []
    journals = ("ALTA-L-REV", "DALHOUSIE-LJ", "OSGOODE-HALL-LJ")
    category_sets = ({0}, {0, 1}, {0}, {0}, {0, 2}, {0})
    for offset, categories in enumerate(category_sets, start=1):
        journal = journals[(offset - 1) % len(journals)]
        name = f"sample__001_{journal}_article-{100 + offset}_pdf-page-1.png"
        (tmp_path / name).write_bytes(b"image")
        images.append({"id": offset, "file_name": name})
        for category in categories:
            annotations.append(
                {"id": len(annotations) + 1, "image_id": offset, "category_id": category}
            )
    annotation_path = tmp_path / "train.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": [
                    {"id": 0, "name": "text"},
                    {"id": 1, "name": "block_quote"},
                    {"id": 2, "name": "byline"},
                ],
            }
        ),
        encoding="utf-8",
    )

    first_paths, first = QUANTIZE.selected_images(
        annotation_path, tmp_path, 3, strategy="class-journal", seed=17
    )
    second_paths, second = QUANTIZE.selected_images(
        annotation_path, tmp_path, 3, strategy="class-journal", seed=999
    )

    assert first_paths == second_paths
    assert first == second
    assert first["categories"]["block_quote"] == 1
    assert first["categories"]["byline"] == 1
    assert len(first["journals"]) >= 2


def test_accuracy_metric_scores_no_detections_as_zero() -> None:
    assert QUANTIZE_NNCF.coco_metric(Path("unused.json"), [1], []) == (0.0, 0.0)


def test_quantization_feeds_follow_the_declared_graph_inputs() -> None:
    feeds = {
        "image": np.zeros((1, 3, 640, 640), dtype=np.float32),
        "im_shape": np.asarray([[640, 640]], dtype=np.float32),
        "scale_factor": np.asarray([[1, 1]], dtype=np.float32),
    }

    filtered = QUANTIZE_NNCF.filter_model_feeds(feeds, ["image"])

    assert list(filtered) == ["image"]
    assert filtered["image"] is feeds["image"]


def test_nncf_quantization_uses_the_library_fast_bias_default() -> None:
    parser = QUANTIZE_NNCF.build_parser()
    assert parser.get_default("fast_bias_correction") is True
    assert parser.get_default("ranking_workers") == 0


def test_progress_receipt_retries_a_windows_sharing_violation(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "progress.json"
    real_replace = Path.replace
    attempts = 0

    def replace_with_one_sharing_violation(source: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("simulated sharing violation")
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", replace_with_one_sharing_violation)

    QUANTIZE.write_json(destination, {"phase": "running"})

    assert attempts == 2
    assert json.loads(destination.read_text(encoding="utf-8")) == {"phase": "running"}


def test_quantization_scope_keeps_only_matching_graph_blocks() -> None:
    assert QUANTIZE.names_matching_patterns(
        ["Conv.0", "Conv.79", "Conv.80", "MatMul.0"],
        [r"Conv\.(?:[0-9]|[1-7][0-9])"],
    ) == ["Conv.0", "Conv.79"]
    assert QUANTIZE_NNCF.names_outside_patterns(
        ["Conv.0", "Conv.79", "Conv.80", "MatMul.0"],
        [r"Conv\.(?:[0-9]|[1-7][0-9])"],
    ) == ["Conv.80", "MatMul.0"]
    assert QUANTIZE_NNCF.names_after_marker(
        ["image", "Conv.0", "Relu.0", "Conv.79", "MatMul.0", "result"],
        "Conv.79",
    ) == ["MatMul.0", "result"]
