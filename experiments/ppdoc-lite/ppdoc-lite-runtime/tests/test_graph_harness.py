from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from ppdoc_lite.runtime import PACK_FORMAT, sha256_file


TOOLS = Path(__file__).resolve().parents[1] / "tools" / "graph.py"
SPEC = importlib.util.spec_from_file_location("ppdoc_lite_graph", TOOLS)
assert SPEC and SPEC.loader
GRAPH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GRAPH)
sys.modules.setdefault("graph", GRAPH)

BUILD_TOOLS = Path(__file__).resolve().parents[1] / "tools" / "build_paddle_openvino_pack.py"
BUILD_SPEC = importlib.util.spec_from_file_location("ppdoc_lite_build_openvino", BUILD_TOOLS)
assert BUILD_SPEC and BUILD_SPEC.loader
BUILD = importlib.util.module_from_spec(BUILD_SPEC)
BUILD_SPEC.loader.exec_module(BUILD)


def test_prepare_pack_accepts_the_raw_dfine_export_contract() -> None:
    args = GRAPH.build_parser().parse_args(
        [
            "prepare-pack",
            "--onnx",
            "model.onnx",
            "--source-manifest",
            "source.json",
            "--source-dir",
            "source",
            "--inference-yml",
            "infer_cfg.yml",
            "--output-dir",
            "pack",
            "--variant-id",
            "legal25-dfine-s-raw-fp32",
            "--output-contract",
            "rtdetr_raw",
        ]
    )
    assert args.output_contract == "rtdetr_raw"
    assert args.boxes_output == "fetch_name_0"
    assert args.logits_output == "fetch_name_1"


def test_prepare_openvino_pack_hash_locks_both_ir_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source_model = source / "model.onnx"
    source_model.write_bytes(b"onnx")
    source_manifest = source / "manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "format": PACK_FORMAT,
                "variant_id": "teacher-raw-fp32",
                "model": {
                    "backend": "onnx",
                    "file": source_model.name,
                    "sha256": sha256_file(source_model),
                    "inputs": ["image"],
                    "outputs": {
                        "contract": "rtdetr_raw",
                        "boxes": "boxes",
                        "logits": "logits",
                    },
                },
                "input": {"target_size": [800, 800], "mean": [0, 0, 0], "std": [1, 1, 1]},
                "labels": ["text"],
            }
        ),
        encoding="utf-8",
    )
    xml = tmp_path / "converted.xml"
    binary = tmp_path / "converted.bin"
    xml.write_bytes(b"xml")
    binary.write_bytes(b"weights")
    output = tmp_path / "pack"

    GRAPH.run_prepare_openvino_pack(
        argparse.Namespace(
            source_pack=source_manifest,
            xml=xml,
            bin=binary,
            output_dir=output,
            variant_id="teacher-openvino-fp32",
            precision="fp32",
        )
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["backend"] == "openvino"
    assert manifest["model"]["sha256"] == sha256_file(output / "model.xml")
    assert manifest["model"]["files"] == [
        {"file": "model.bin", "sha256": sha256_file(output / "model.bin")}
    ]
    assert manifest["model"]["bytes"] == len(b"xml") + len(b"weights")
    assert manifest["provenance"]["derived_from"]["variant_id"] == "teacher-raw-fp32"


def test_prepare_direct_paddle_openvino_pack_preserves_export_contract(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "epoch.pdparams"
    checkpoint.write_bytes(b"checkpoint")
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        json.dumps(
            {
                "source_id": "legal25-640-e15",
                "model_name": "PP-DocLayoutV3-L-640",
                "source_files": {checkpoint.name: sha256_file(checkpoint)},
                "labels": ["text", "footnote"],
            }
        ),
        encoding="utf-8",
    )
    inference = tmp_path / "infer_cfg.yml"
    inference.write_text(
        """draw_threshold: 0.1
Preprocess:
- type: Resize
  target_size: [640, 640]
  keep_ratio: false
  interp: 2
- type: NormalizeImage
  mean: [0.0, 0.0, 0.0]
  std: [1.0, 1.0, 1.0]
- type: Permute
label_list: [text, footnote]
""",
        encoding="utf-8",
    )
    xml = tmp_path / "converted.xml"
    binary = tmp_path / "converted.bin"
    xml.write_bytes(b"xml")
    binary.write_bytes(b"weights")
    output = tmp_path / "pack"

    GRAPH.run_prepare_paddle_openvino_pack(
        argparse.Namespace(
            xml=xml,
            bin=binary,
            inference_yml=inference,
            source_manifest=source_manifest,
            source_dir=source,
            output_dir=output,
            variant_id="legal25-ppdocv3-l640-e15-openvino-fp32",
            precision="fp32",
            inputs=["im_shape", "image", "scale_factor"],
            boxes_output="save_infer_model/scale_0.tmp_0",
            counts_output="save_infer_model/scale_1.tmp_0",
            detections_per_image=100,
            output_width=6,
        )
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["labels"] == ["text", "footnote"]
    assert manifest["input"]["target_size"] == [640, 640]
    assert manifest["model"]["backend"] == "openvino"
    assert manifest["model"]["inputs"] == ["im_shape", "image", "scale_factor"]
    assert manifest["model"]["outputs"] == {
        "contract": "decoded_boxes",
        "boxes": "save_infer_model/scale_0.tmp_0",
        "counts": "save_infer_model/scale_1.tmp_0",
    }
    assert manifest["model"]["detections_per_image"] == 100
    assert manifest["model"]["output_width"] == 6
    assert manifest["model"]["files"] == [
        {"file": "model.bin", "sha256": sha256_file(output / "model.bin")}
    ]


def test_prepare_ppyoloe_raw_pack_records_native_nms_contract(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "best_model.pdparams"
    checkpoint.write_bytes(b"checkpoint")
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        json.dumps(
            {
                "source_id": "legal25-ppyoloe-m",
                "model_name": "PP-YOLOE-M-640",
                "source_files": {checkpoint.name: sha256_file(checkpoint)},
                "labels": ["text", "footnote"],
            }
        ),
        encoding="utf-8",
    )
    inference = tmp_path / "infer_cfg.yml"
    inference.write_text(
        """draw_threshold: 0.1
NMS:
  keep_top_k: 300
Preprocess:
- type: Resize
  target_size: [640, 640]
  keep_ratio: false
  interp: 2
- type: NormalizeImage
  mean: [0.0, 0.0, 0.0]
  std: [1.0, 1.0, 1.0]
- type: Permute
label_list: [text, footnote]
""",
        encoding="utf-8",
    )
    xml = tmp_path / "model.xml"
    binary = tmp_path / "model.bin"
    xml.write_bytes(b"xml")
    binary.write_bytes(b"weights")
    output = tmp_path / "pack"

    GRAPH.run_prepare_paddle_openvino_pack(
        argparse.Namespace(
            xml=xml,
            bin=binary,
            inference_yml=inference,
            source_manifest=source_manifest,
            source_dir=source,
            output_dir=output,
            variant_id="legal25-ppyoloe-m-openvino-fp16",
            precision="fp16",
            inputs=["image", "scale_factor"],
            boxes_output="boxes",
            counts_output="unused",
            scores_output="scores",
            output_contract="ppyoloe_raw",
            nms_score_threshold=0.01,
            nms_threshold=0.7,
            nms_top_k=1_000,
            transform=BUILD.RAW_TRANSFORM,
            detections_per_image=None,
            output_width=None,
        )
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["outputs"] == {
        "contract": "ppyoloe_raw",
        "boxes": "boxes",
        "scores": "scores",
    }
    assert manifest["model"]["inputs"] == ["image", "scale_factor"]
    assert manifest["postprocess"]["model_nms"] == {
        "score_threshold": 0.01,
        "nms_threshold": 0.7,
        "nms_top_k": 1_000,
        "keep_top_k": 300,
    }
    assert manifest["provenance"]["transform"] == BUILD.RAW_TRANSFORM


def test_fast_paddle_export_selects_decoded_outputs_without_shape_inference() -> None:
    model = SimpleNamespace(
        graph=SimpleNamespace(
            output=[
                SimpleNamespace(name="boxes"),
                SimpleNamespace(name="counts"),
                SimpleNamespace(name="masks"),
            ]
        )
    )

    BUILD.select_onnx_outputs(model, ("boxes", "counts"))

    assert [value.name for value in model.graph.output] == ["boxes", "counts"]


def test_direct_paddle_pack_records_the_selected_transform(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = source / "epoch.pdparams"
    checkpoint.write_bytes(b"checkpoint")
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        json.dumps(
            {
                "source_id": "legal25-640-e15",
                "model_name": "PP-DocLayoutV3-L-640",
                "source_files": {checkpoint.name: sha256_file(checkpoint)},
                "labels": ["text"],
            }
        ),
        encoding="utf-8",
    )
    inference = tmp_path / "infer_cfg.yml"
    inference.write_text(
        """draw_threshold: 0.1
Preprocess:
- type: Resize
  target_size: [640, 640]
  keep_ratio: false
  interp: 2
- type: NormalizeImage
  mean: [0.0, 0.0, 0.0]
  std: [1.0, 1.0, 1.0]
- type: Permute
label_list: [text]
""",
        encoding="utf-8",
    )
    xml = tmp_path / "model.xml"
    binary = tmp_path / "model.bin"
    xml.write_bytes(b"xml")
    binary.write_bytes(b"weights")
    output = tmp_path / "pack"

    GRAPH.run_prepare_paddle_openvino_pack(
        argparse.Namespace(
            xml=xml,
            bin=binary,
            inference_yml=inference,
            source_manifest=source_manifest,
            source_dir=source,
            output_dir=output,
            variant_id="fast-route",
            precision="fp32",
            inputs=["im_shape", "image", "scale_factor"],
            boxes_output="boxes",
            counts_output="counts",
            transform=BUILD.TRANSFORM,
        )
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provenance"]["transform"] == BUILD.TRANSFORM
