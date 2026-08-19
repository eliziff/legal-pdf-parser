from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME_ROOT / "src"))

from ppdoc_lite.runtime import (
    ModelPack,
    decode_ppyoloe_raw,
    decode_rtdetr_raw,
    postprocess_boxes,
    prepare_image,
)
from quantize import prepare_output_pack, selected_images, sha256_file, write_json


def names_outside_patterns(names: Iterable[str], patterns: Sequence[str]) -> list[str]:
    compiled = [re.compile(pattern) for pattern in patterns]
    return sorted(name for name in names if not any(pattern.fullmatch(name) for pattern in compiled))


def names_after_marker(names: Iterable[str], marker: str) -> list[str]:
    ordered = list(names)
    matches = [index for index, name in enumerate(ordered) if name == marker]
    if len(matches) != 1:
        raise ValueError(f"expected one graph node named {marker!r}, found {len(matches)}")
    return sorted(ordered[matches[0] + 1 :])


def annotation_rows(annotation_path: Path, image_root: Path, limit: int = 0) -> list[dict[str, Any]]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    rows = sorted(payload["images"], key=lambda row: (str(row["file_name"]), int(row["id"])))
    if limit > 0:
        rows = rows[:limit]
    for row in rows:
        path = image_root / str(row["file_name"])
        if not path.is_file():
            path = image_root / Path(str(row["file_name"])).name
        if not path.is_file():
            raise FileNotFoundError(path)
        row["_path"] = path
    return rows


def prepare_feeds(pack: ModelPack, path: Path, image_backend: str) -> dict[str, np.ndarray]:
    config = pack.manifest.get("input", {})
    feeds, _ = prepare_image(
        path,
        config.get("target_size", [800, 800]),
        backend=image_backend,
        scale=float(config.get("scale", 1.0 / 255.0)),
        mean=config.get("mean", [0.0, 0.0, 0.0]),
        std=config.get("std", [1.0, 1.0, 1.0]),
    )
    return {name: np.expand_dims(value, 0) for name, value in feeds.items()}


def filter_model_feeds(
    feeds: dict[str, np.ndarray], input_names: Sequence[str]
) -> dict[str, np.ndarray]:
    missing = [name for name in input_names if name not in feeds]
    if missing:
        raise ValueError(f"Preprocessing did not produce model inputs: {missing}")
    return {name: feeds[name] for name in input_names}


def coco_metric(
    annotation_path: Path,
    image_ids: Sequence[int],
    predictions: list[dict[str, Any]],
) -> tuple[float, float]:
    if not predictions:
        return 0.0, 0.0
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gold = COCO(str(annotation_path))
        candidate = gold.loadRes(predictions)
        evaluation = COCOeval(gold, candidate, "bbox")
        evaluation.params.imgIds = list(image_ids)
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
    return float(evaluation.stats[0]), float(evaluation.stats[1])


class AccuracyValidator:
    def __init__(
        self,
        *,
        pack: ModelPack,
        annotations: Path,
        category_ids: dict[str, int],
        output_dir: Path,
        threads: int,
        threshold: float,
        image_backend: str,
    ) -> None:
        self.pack = pack
        self.annotations = annotations
        self.category_ids = category_ids
        self.output_dir = output_dir
        self.threads = threads
        self.threshold = threshold
        self.image_backend = image_backend
        self.call = 0

    def runner(self, model: Any):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        session = ort.InferenceSession(
            model.SerializeToString(),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        input_names = [value.name for value in session.get_inputs()]
        score_key = "scores" if self.pack.output_contract == "ppyoloe_raw" else "logits"
        output_names = [self.pack.output_names[name] for name in ("boxes", score_key)]

        def run(feeds: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            boxes, logits = session.run(
                output_names, filter_model_feeds(feeds, input_names)
            )
            return boxes, logits

        return run

    def __call__(
        self, model: Any, rows: Iterable[dict[str, Any]]
    ) -> tuple[float, list[list[np.ndarray]]]:
        self.call += 1
        run = self.runner(model)
        predictions: list[dict[str, Any]] = []
        outputs_per_item: list[list[np.ndarray]] = []
        image_ids: list[int] = []
        rows = list(rows)
        started = time.perf_counter()
        progress_path = self.output_dir / f"validation-{self.call:03d}.json"
        for index, row in enumerate(rows, 1):
            feeds = prepare_feeds(self.pack, Path(row["_path"]), self.image_backend)
            boxes, class_values = run(feeds)
            outputs_per_item.append([np.asarray(boxes), np.asarray(class_values)])
            if self.pack.output_contract == "ppyoloe_raw":
                model_nms = self.pack.manifest.get("postprocess", {}).get("model_nms", {})
                decoded, counts = decode_ppyoloe_raw(
                    boxes,
                    class_values,
                    score_threshold=max(
                        self.threshold, float(model_nms.get("score_threshold", 0.01))
                    ),
                    nms_threshold=float(model_nms.get("nms_threshold", 0.7)),
                    nms_top_k=int(model_nms.get("nms_top_k", 1_000)),
                    keep_top_k=int(
                        model_nms.get("keep_top_k")
                        or self.pack.manifest["model"].get("detections_per_image")
                        or 300
                    ),
                )
                decoded = decoded[: int(counts[0])]
            else:
                decoded = decode_rtdetr_raw(
                    boxes,
                    class_values,
                    feeds,
                    int(self.pack.manifest["model"].get("detections_per_image") or 300),
                )[0]
            detections = postprocess_boxes(
                decoded,
                labels=self.pack.labels,
                image_size=(int(row["width"]), int(row["height"])),
                threshold=self.threshold,
                filter_overlap_boxes=False,
            )
            image_id = int(row["id"])
            image_ids.append(image_id)
            for detection in detections:
                category_id = self.category_ids.get(str(detection["label"]))
                if category_id is None:
                    continue
                x0, y0, x1, y1 = [float(value) for value in detection["coordinate"]]
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "score": float(detection["score"]),
                    }
                )
            if index == 1 or index % 5 == 0 or index == len(rows):
                write_json(
                    progress_path,
                    {
                        "schema_version": "legalpdf.ppdoc_lite_nncf_validation.v1",
                        "phase": "running",
                        "validation_call": self.call,
                        "processed_pages": index,
                        "total_pages": len(rows),
                        "elapsed_seconds": time.perf_counter() - started,
                        "predictions": predictions,
                    },
                )
                print(
                    f"[ppdoc-lite nncf] validation={self.call} pages={index}/{len(rows)}",
                    flush=True,
                )
        ap, ap50 = coco_metric(self.annotations, image_ids, predictions)
        write_json(
            progress_path,
            {
                "schema_version": "legalpdf.ppdoc_lite_nncf_validation.v1",
                "phase": "complete",
                "validation_call": self.call,
                "pages": len(rows),
                "elapsed_seconds": time.perf_counter() - started,
                "ap_iou_50_95": ap,
                "ap_iou_50": ap50,
                "predictions": predictions,
            },
        )
        print(
            f"[ppdoc-lite nncf] validation={self.call} AP={ap:.6f} AP50={ap50:.6f}",
            flush=True,
        )
        # NNCF uses these per-item tensors to rank sensitive quantizer groups by
        # normalized output error when task-level AP ties (notably at AP=0).
        return ap, outputs_per_item


class OpenVinoAccuracyValidator(AccuracyValidator):
    def runner(self, model: Any):
        input_names = [port.any_name for port in model.inputs]
        score_key = "scores" if self.pack.output_contract == "ppyoloe_raw" else "logits"
        output_names = [self.pack.output_names[name] for name in ("boxes", score_key)]
        output_ports = [model.output(name) for name in output_names]

        def run(feeds: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            values = model(filter_model_feeds(feeds, input_names))
            return tuple(np.asarray(values[port]) for port in output_ports)  # type: ignore[return-value]

        return run


def prepare_openvino_output_pack(
    source: ModelPack, destination: Path, variant_id: str, model_xml: Path
) -> None:
    model_bin = model_xml.with_suffix(".bin")
    if not model_bin.is_file():
        raise FileNotFoundError(model_bin)
    manifest = json.loads(json.dumps(source.manifest))
    manifest["variant_id"] = variant_id
    manifest.pop("profile", None)
    manifest["model"].update(
        {
            "backend": "openvino",
            "file": model_xml.name,
            "sha256": sha256_file(model_xml),
            "bytes": model_xml.stat().st_size + model_bin.stat().st_size,
            "files": [
                {
                    "file": model_bin.name,
                    "sha256": sha256_file(model_bin),
                    "bytes": model_bin.stat().st_size,
                }
            ],
        }
    )
    manifest["derived_from"] = {
        "variant_id": source.variant_id,
        "model_sha256": source.manifest["model"]["sha256"],
    }
    write_json(destination / "manifest.json", manifest)


def run(args: argparse.Namespace) -> int:
    import nncf
    import onnxruntime as ort
    from nncf.quantization.advanced_parameters import RestoreMode

    source = ModelPack.load(args.source_pack)
    if source.output_contract not in {"rtdetr_raw", "ppyoloe_raw"}:
        raise ValueError("NNCF PTQ requires an rtdetr_raw or ppyoloe_raw model contract")
    calibration_paths, selection = selected_images(
        args.calibration_annotations,
        args.image_root,
        args.calibration_pages,
        strategy=args.calibration_selection,
        seed=args.calibration_seed,
    )
    validation_payload = json.loads(args.validation_annotations.read_text(encoding="utf-8"))
    validation_rows = annotation_rows(
        args.validation_annotations, args.image_root, args.validation_pages
    )
    category_ids = {
        str(row["name"]): int(row["id"])
        for row in validation_payload["categories"]
    }
    args.output_pack.mkdir(parents=True, exist_ok=True)
    intermediate_dir = args.intermediate_dir or args.output_pack / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output_pack / "quantization.json"
    receipt: dict[str, Any] = {
        "schema_version": "legalpdf.ppdoc_lite_quantization.v1",
        "phase": "running",
        "operation": "nncf_accuracy_control" if args.accuracy_control else "nncf_ptq",
        "variant_id": args.variant_id,
        "source_variant_id": source.variant_id,
        "source_sha256": sha256_file(source.model_path),
        "source_bytes": source.model_path.stat().st_size,
        "started_epoch_seconds": time.time(),
        "backend": args.backend,
        "versions": {
            "nncf": nncf.__version__,
            "onnxruntime": ort.__version__,
        },
        "calibration": {
            "annotations": str(args.calibration_annotations),
            "annotation_sha256": sha256_file(args.calibration_annotations),
            "pages": len(calibration_paths),
            "selection": selection,
        },
        "validation": {
            "annotations": str(args.validation_annotations),
            "annotation_sha256": sha256_file(args.validation_annotations),
            "pages": len(validation_rows),
        },
        "settings": {
            "preset": args.preset,
            "target_device": args.target_device,
            "max_drop": args.max_drop,
            "fast_bias_correction": args.fast_bias_correction,
            "disable_bias_correction": args.disable_bias_correction,
            "restore_mode": args.restore_mode,
            "threads": args.threads,
            "threshold": args.threshold,
            "image_backend": args.image_backend,
            "intermediate_dir": str(intermediate_dir),
            "ignored_op_types": args.ignore_op_type,
            "ignore_dynamic_rank": args.ignore_dynamic_rank,
            "quantize_name_regex": args.quantize_name_regex,
            "quantize_through": args.quantize_through,
            "accuracy_control": args.accuracy_control,
        },
    }
    write_json(receipt_path, receipt)

    calibration_rows = [{"_path": path} for path in calibration_paths]
    model_input_names = list(source.manifest["model"].get("inputs") or [])
    if not model_input_names:
        raise ValueError("Source pack does not declare model inputs")
    calibration_dataset = nncf.Dataset(
        calibration_rows,
        lambda row: filter_model_feeds(
            prepare_feeds(source, Path(row["_path"]), args.image_backend),
            model_input_names,
        ),
    )
    validation_dataset = nncf.Dataset(
        validation_rows,
        lambda row: filter_model_feeds(
            prepare_feeds(source, Path(row["_path"]), args.image_backend),
            model_input_names,
        ),
    )
    validator_type = OpenVinoAccuracyValidator if args.backend == "openvino" else AccuracyValidator
    validator = validator_type(
        pack=source,
        annotations=args.validation_annotations,
        category_ids=category_ids,
        output_dir=args.output_pack,
        threads=args.threads,
        threshold=args.threshold,
        image_backend=args.image_backend,
    )
    presets = {
        "mixed": nncf.QuantizationPreset.MIXED,
        "performance": nncf.QuantizationPreset.PERFORMANCE,
    }
    devices = {
        "any": nncf.TargetDevice.ANY,
        "cpu": nncf.TargetDevice.CPU,
        "gpu": nncf.TargetDevice.GPU,
    }
    restore_modes = {
        "activations-and-weights": RestoreMode.ACTIVATIONS_AND_WEIGHTS,
        "only-activations": RestoreMode.ONLY_ACTIVATIONS,
    }
    restorer = nncf.AdvancedAccuracyRestorerParameters(
        max_num_iterations=args.max_iterations if args.max_iterations > 0 else sys.maxsize,
        ranking_subset_size=(
            args.ranking_pages if args.ranking_pages > 0 else len(validation_rows)
        ),
        num_ranking_workers=(args.ranking_workers if args.ranking_workers > 0 else None),
        intermediate_model_dir=str(intermediate_dir),
        restore_mode=restore_modes[args.restore_mode],
    )
    started = time.perf_counter()
    try:
        if args.backend == "openvino":
            import openvino as ov

            receipt["versions"]["openvino"] = ov.__version__
            model = ov.Core().read_model(source.model_path)
        else:
            import onnx

            receipt["versions"]["onnx"] = onnx.__version__
            model = onnx.load(source.model_path)
        ignored_names: list[str] = []
        if args.ignore_dynamic_rank:
            if args.backend != "openvino":
                raise ValueError("--ignore-dynamic-rank requires --backend openvino")
            ignored_names = sorted(
                op.get_friendly_name()
                for op in model.get_ops()
                if any(output.get_partial_shape().rank.is_dynamic for output in op.outputs())
            )
            ignored_path = args.output_pack / "ignored-dynamic-rank.json"
            write_json(
                ignored_path,
                {
                    "schema_version": "legalpdf.ppdoc_lite_ignored_scope.v1",
                    "reason": "OpenVINO/NNCF cannot collect quantization statistics for dynamic-rank tensors",
                    "node_count": len(ignored_names),
                    "nodes": ignored_names,
                },
            )
            receipt["settings"]["ignored_dynamic_rank_nodes"] = len(ignored_names)
            receipt["settings"]["ignored_dynamic_rank_receipt"] = str(ignored_path)
            receipt["settings"]["ignored_dynamic_rank_sha256"] = sha256_file(ignored_path)
            write_json(receipt_path, receipt)
        if args.quantize_name_regex:
            if args.backend != "openvino":
                raise ValueError("--quantize-name-regex requires --backend openvino")
            ignored_names = sorted(
                set(ignored_names)
                | set(
                    names_outside_patterns(
                        (op.get_friendly_name() for op in model.get_ops()),
                        args.quantize_name_regex,
                    )
                )
            )
            receipt["settings"]["scope_ignored_nodes"] = len(ignored_names)
            write_json(receipt_path, receipt)
        if args.quantize_through:
            if args.backend != "openvino":
                raise ValueError("--quantize-through requires --backend openvino")
            ignored_names = sorted(
                set(ignored_names)
                | set(
                    names_after_marker(
                        (op.get_friendly_name() for op in model.get_ordered_ops()),
                        args.quantize_through,
                    )
                )
            )
            receipt["settings"]["scope_ignored_nodes"] = len(ignored_names)
            write_json(receipt_path, receipt)
        ignored_scope = (
            nncf.IgnoredScope(names=ignored_names, types=args.ignore_op_type)
            if ignored_names or args.ignore_op_type
            else None
        )
        advanced = nncf.AdvancedQuantizationParameters(
            disable_bias_correction=args.disable_bias_correction
        )
        if args.accuracy_control:
            quantized = nncf.quantize_with_accuracy_control(
                model,
                calibration_dataset,
                validation_dataset,
                validation_fn=validator,
                max_drop=args.max_drop,
                preset=presets[args.preset],
                target_device=devices[args.target_device],
                subset_size=len(calibration_paths),
                fast_bias_correction=args.fast_bias_correction,
                ignored_scope=ignored_scope,
                advanced_quantization_parameters=advanced,
                advanced_accuracy_restorer_parameters=restorer,
            )
        else:
            quantized = nncf.quantize(
                model,
                calibration_dataset,
                preset=presets[args.preset],
                target_device=devices[args.target_device],
                subset_size=len(calibration_paths),
                fast_bias_correction=args.fast_bias_correction,
                ignored_scope=ignored_scope,
                advanced_parameters=advanced,
            )
        if args.backend == "openvino":
            model_path = args.output_pack / "model.xml"
            ov.save_model(quantized, model_path, compress_to_fp16=False)
            prepare_openvino_output_pack(source, args.output_pack, args.variant_id, model_path)
        else:
            model_path = args.output_pack / "model.onnx"
            onnx.save(quantized, model_path)
            prepare_output_pack(source, args.output_pack, args.variant_id, model_path)
        receipt.update(
            {
                "phase": "complete",
                "seconds": time.perf_counter() - started,
                "output_sha256": sha256_file(model_path),
                "output_bytes": int(
                    json.loads((args.output_pack / "manifest.json").read_text(encoding="utf-8"))["model"]["bytes"]
                ),
                "validation_calls": validator.call,
            }
        )
    except BaseException as exc:
        receipt.update(
            {
                "phase": "failed",
                "seconds": time.perf_counter() - started,
                "validation_calls": validator.call,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_json(receipt_path, receipt)
        raise
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NNCF PTQ and accuracy-controlled INT8 for PPdoc raw detectors"
    )
    parser.add_argument("--source-pack", type=Path, required=True)
    parser.add_argument("--output-pack", type=Path, required=True)
    parser.add_argument("--variant-id", required=True)
    parser.add_argument("--backend", choices=("onnx", "openvino"), default="openvino")
    parser.add_argument("--calibration-annotations", type=Path, required=True)
    parser.add_argument("--validation-annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--calibration-pages", type=int, default=300)
    parser.add_argument("--validation-pages", type=int, default=0)
    parser.add_argument(
        "--calibration-selection",
        choices=("even", "random", "class-journal"),
        default="class-journal",
    )
    parser.add_argument("--calibration-seed", type=int, default=20260813)
    parser.add_argument("--preset", choices=("mixed", "performance"), default="mixed")
    parser.add_argument("--target-device", choices=("any", "cpu", "gpu"), default="cpu")
    parser.add_argument("--max-drop", type=float, default=0.01)
    parser.add_argument(
        "--fast-bias-correction", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--disable-bias-correction", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--restore-mode",
        choices=("activations-and-weights", "only-activations"),
        default="activations-and-weights",
    )
    parser.add_argument("--max-iterations", type=int, default=0)
    parser.add_argument("--ranking-pages", type=int, default=0)
    parser.add_argument("--ranking-workers", type=int, default=0)
    parser.add_argument("--intermediate-dir", type=Path)
    parser.add_argument("--ignore-op-type", action="append", default=[])
    parser.add_argument("--ignore-dynamic-rank", action="store_true")
    parser.add_argument("--quantize-name-regex", action="append", default=[])
    parser.add_argument("--quantize-through")
    parser.add_argument(
        "--accuracy-control", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--image-backend", choices=("opencv", "pillow"), default="opencv")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
