from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from ppdoc_lite.runtime import (
    PACK_FORMAT,
    ModelPack,
    PPDocLite,
    decode_ppyoloe_raw,
    postprocess_boxes,
    prepare_image,
)


LABELS = ["text", "reference", "reference_content", "image", "inline_formula"]


def test_model_pack_verifies_hash_and_rejects_escape(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"unit-model")
    manifest = {
        "format": PACK_FORMAT,
        "variant_id": "fp32",
        "labels": LABELS,
        "model": {
            "file": "model.onnx",
            "sha256": hashlib.sha256(b"unit-model").hexdigest(),
            "outputs": {"contract": "decoded_boxes", "boxes": "boxes", "counts": "counts"},
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert ModelPack.load(tmp_path).output_names == {"boxes": "boxes", "counts": "counts"}
    manifest["model"]["file"] = "../model.onnx"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        ModelPack.load(tmp_path)


def test_model_pack_verifies_companion_artifacts(tmp_path: Path) -> None:
    model = tmp_path / "model.xml"
    weights = tmp_path / "model.bin"
    model.write_bytes(b"graph")
    weights.write_bytes(b"weights")
    manifest = {
        "format": PACK_FORMAT,
        "variant_id": "openvino-int8",
        "labels": LABELS,
        "model": {
            "file": model.name,
            "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            "bytes": model.stat().st_size + weights.stat().st_size,
            "files": [
                {
                    "file": weights.name,
                    "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                }
            ],
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert ModelPack.load(tmp_path).model_bytes == 12
    weights.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="hash mismatch"):
        ModelPack.load(tmp_path)


def test_prepare_image_matches_paddlex_tensor_contract(tmp_path: Path) -> None:
    path = tmp_path / "page.png"
    source = np.zeros((20, 10, 3), dtype=np.uint8)
    source[:, :, 2] = 255
    assert cv2.imwrite(str(path), source)
    tensors, size = prepare_image(path)
    assert size == (10, 20)
    assert tensors["image"].shape == (3, 800, 800)
    assert tensors["image"].dtype == np.float32
    np.testing.assert_allclose(tensors["im_shape"], [800, 800])
    np.testing.assert_allclose(tensors["scale_factor"], [40, 80])
    assert tensors["image"][0, 0, 0] == pytest.approx(1.0)


def test_postprocess_preserves_learned_order_and_box_contract() -> None:
    raw = np.asarray(
        [
            [0, 0.90, 10.2, 10.4, 100.3, 100.2, 5],
            [2, 0.95, 120.1, 20.1, 220.2, 80.2, 2],
            [1, 0.99, 250, 20, 350, 80, 1],
            [4, 0.80, 12, 12, 40, 40, 3],
            [0, 0.05, 0, 0, 10, 10, 0],
        ],
        dtype=np.float32,
    )
    boxes = postprocess_boxes(
        raw,
        labels=LABELS,
        image_size=(400, 300),
        threshold=0.10,
        filter_overlap_boxes=True,
    )
    assert [box["label"] for box in boxes] == ["reference_content", "text"]
    assert boxes[0]["coordinate"] == [120, 20, 220, 80]
    assert boxes[0]["order"] == 1
    assert boxes[1]["order"] == 2


def test_infer_tensors_is_the_image_library_free_boundary() -> None:
    runtime = PPDocLite.__new__(PPDocLite)
    runtime.pack = SimpleNamespace(labels=LABELS)
    runtime._raw_predictions = lambda feeds: (  # type: ignore[method-assign]
        np.asarray([[0, 0.9, 1, 2, 9, 18]], dtype=np.float32),
        np.asarray([1], dtype=np.int32),
    )
    result = runtime.infer_tensors(
        {
            "image": np.zeros((1, 3, 800, 800), dtype=np.float32),
            "im_shape": np.asarray([[800, 800]], dtype=np.float32),
            "scale_factor": np.asarray([[40, 80]], dtype=np.float32),
        },
        [(10, 20)],
        image_ids=["page-1"],
    )
    assert result[0]["image"] == "page-1"
    assert result[0]["detections"][0]["box"] == [1.0, 2.0, 9.0, 18.0]


def test_ppyoloe_raw_decoder_uses_classwise_nms_and_global_score_order() -> None:
    boxes = np.asarray(
        [[[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30], [5, 5, 5, 8]]],
        dtype=np.float32,
    )
    scores = np.asarray(
        [[[0.90, 0.80, 0.70, 0.99], [0.10, 0.95, 0.05, 0.99]]],
        dtype=np.float32,
    )

    decoded, counts = decode_ppyoloe_raw(
        boxes,
        scores,
        score_threshold=0.10,
        nms_threshold=0.5,
        nms_top_k=1_000,
        keep_top_k=3,
    )

    assert counts.tolist() == [3]
    np.testing.assert_allclose(
        decoded[:, :2], [[1.0, 0.95], [0.0, 0.90], [0.0, 0.70]]
    )
    np.testing.assert_array_equal(decoded[:, 2:6], [[1, 1, 11, 11], [0, 0, 10, 10], [20, 20, 30, 30]])
