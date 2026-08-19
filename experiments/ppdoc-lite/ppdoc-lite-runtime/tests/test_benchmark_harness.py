from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools" / "benchmark.py"
SPEC = importlib.util.spec_from_file_location("ppdoc_lite_benchmark", TOOLS)
assert SPEC and SPEC.loader
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def test_differential_reports_numeric_drift_without_hiding_contract_mismatches() -> None:
    left = {
        "variant_id": "left",
        "page_results": [
            {
                "file_name": "page.png",
                "detections": [
                    {
                        "label": "text",
                        "score": 0.9,
                        "box": [1, 2, 3, 4],
                        "raw": {"order": 1},
                    }
                ],
            }
        ],
    }
    right = {
        "variant_id": "right",
        "page_results": [
            {
                "file_name": "page.png",
                "detections": [
                    {
                        "label": "text",
                        "score": 0.89,
                        "box": [1, 2, 3.5, 4],
                        "raw": {"order": 1},
                    }
                ],
            }
        ],
    }

    result = BENCHMARK.compare_page_results(left, right, score_atol=0.02, box_atol=0.5)

    assert result["count_mismatch_pages"] == []
    assert result["label_mismatches"] == 0
    assert result["order_mismatches"] == 0
    assert result["max_abs_score_error"] > 0
    assert result["max_abs_box_error"] == 0.5
    assert not result["exact_detection_contract"]
    assert result["detection_contract_within_tolerance"]


def test_rust_detections_normalize_to_the_reference_contract() -> None:
    result = BENCHMARK.normalize_rust_detections(
        [
            {
                "label_id": 2,
                "label": "text",
                "score": 0.75,
                "bbox": [1, 2, 30, 40],
                "order": 3,
            }
        ]
    )

    assert result == [
        {
            "score": 0.75,
            "label_id": 2,
            "label": "text",
            "box": [1.0, 2.0, 30.0, 40.0],
            "raw": {"order": 3},
        }
    ]


def test_rust_command_forwards_generic_runtime_options() -> None:
    args = Namespace(
        binary=Path("legalpdf.exe"),
        model_pack=Path("pack"),
        runtime=Path("openvino_c.dll"),
        backend="openvino",
        device="CPU",
        cache_dir=Path("cache"),
        threads=4,
        threshold=0.01,
    )

    assert BENCHMARK.rust_command(args, Path("images.txt")) == [
        "legalpdf.exe",
        "ppdoc-images",
        "--list",
        "images.txt",
        "--model-pack",
        "pack",
        "--runtime",
        "openvino_c.dll",
        "--backend",
        "openvino",
        "--threads",
        "4",
        "--threshold",
        "0.01",
        "--device",
        "CPU",
        "--cache-dir",
        "cache",
    ]


def test_raw_rtdetr_decode_selects_score_and_scales_box() -> None:
    boxes = np.asarray([[[0.5, 0.5, 0.4, 0.2]]], dtype=np.float32)
    logits = np.asarray([[[-2.0, 2.0]]], dtype=np.float32)
    feeds = {
        "im_shape": np.asarray([[800, 800]], dtype=np.float32),
        "scale_factor": np.asarray([[4, 8]], dtype=np.float32),
    }

    decoded = BENCHMARK.decode_rtdetr_raw(boxes, logits, feeds, top_k=1)

    assert decoded.shape == (1, 1, 7)
    assert decoded[0, 0, 0] == 1
    np.testing.assert_allclose(decoded[0, 0, 1], 1 / (1 + np.exp(-2)))
    np.testing.assert_allclose(decoded[0, 0, 2:6], [30, 80, 70, 120])


def test_openvino_hybrid_cpu_controls_map_to_native_properties() -> None:
    assert BENCHMARK.openvino_config(
        "latency", "f32", 4, 1, "pcores", True, False
    ) == {
        "PERFORMANCE_HINT": "LATENCY",
        "INFERENCE_PRECISION_HINT": "f32",
        "INFERENCE_NUM_THREADS": 4,
        "NUM_STREAMS": 1,
        "SCHEDULING_CORE_TYPE": "PCORE_ONLY",
        "ENABLE_CPU_PINNING": True,
        "ENABLE_HYPER_THREADING": False,
    }


def test_openvino_native_decoded_contract_splits_flat_rows_by_counts() -> None:
    boxes = np.asarray(
        [
            [0, 0.9, 1, 1, 5, 5, 1],
            [1, 0.8, 2, 2, 6, 6, 1],
            [0, 0.7, 7, 7, 9, 9, 2],
        ],
        dtype=np.float32,
    )

    class Request:
        def infer(self, _feeds):
            return {"boxes": boxes, "counts": np.asarray([1, 2], dtype=np.int32)}

    engine = object.__new__(BENCHMARK.OpenVinoRaw)
    engine._request = Request()
    engine._input_names = ("im_shape", "image", "scale_factor")
    engine._output_ports = {"boxes": "boxes", "counts": "counts"}
    engine._output_contract = "decoded_boxes"
    engine._top_k = 300
    engine.pack = SimpleNamespace(labels=["text", "footnote"])
    feeds = {
        "im_shape": np.asarray([[10, 10], [10, 10]], dtype=np.float32),
        "image": np.zeros((2, 3, 10, 10), dtype=np.float32),
        "scale_factor": np.ones((2, 2), dtype=np.float32),
    }

    results = engine.infer_tensors(
        feeds,
        [(10, 10), (10, 10)],
        image_ids=["one", "two"],
        threshold=0.1,
        filter_overlap_boxes=False,
    )

    assert [len(row["detections"]) for row in results] == [1, 2]
    assert results[0]["detections"][0]["label"] == "text"
    assert [row["label"] for row in results[1]["detections"]] == ["footnote", "text"]
