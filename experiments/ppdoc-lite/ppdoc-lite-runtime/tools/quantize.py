from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ppdoc_lite.runtime import ModelPack, prepare_image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def _journal(file_name: str) -> str:
    match = re.search(r"__(?:\d+_)?(.+?)_article-[^_]+_pdf-page-", Path(file_name).name)
    return match.group(1) if match else "unknown"


def selected_images(
    annotation_path: Path,
    image_root: Path,
    count: int,
    *,
    strategy: str,
    seed: int,
) -> tuple[list[Path], dict[str, Any]]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = sorted(payload["images"], key=lambda row: (str(row["file_name"]), int(row["id"])))
    if count < 1 or count > len(images):
        raise ValueError(f"Calibration count must be in 1..{len(images)}, got {count}")
    annotations: dict[int, set[int]] = {}
    for row in payload.get("annotations", []):
        annotations.setdefault(int(row["image_id"]), set()).add(int(row["category_id"]))
    if strategy == "even":
        indices = np.linspace(0, len(images) - 1, count, dtype=np.int64).tolist()
    elif strategy == "random":
        indices = sorted(
            np.random.default_rng(seed).choice(len(images), size=count, replace=False).tolist()
        )
    elif strategy == "class-journal":
        category_frequency: dict[int, int] = {}
        for values in annotations.values():
            for category in values:
                category_frequency[category] = category_frequency.get(category, 0) + 1
        selected: list[int] = []
        selected_set: set[int] = set()
        category_counts: dict[int, int] = {}
        journal_counts: dict[str, int] = {}
        while len(selected) < count:
            best = max(
                (index for index in range(len(images)) if index not in selected_set),
                key=lambda index: (
                    sum(
                        1.0 / max(1, category_frequency[category])
                        for category in annotations.get(int(images[index]["id"]), set())
                        if category_counts.get(category, 0) == 0
                    ),
                    1.0 / (1 + journal_counts.get(_journal(str(images[index]["file_name"])), 0)),
                    sum(
                        1.0 / ((1 + category_counts.get(category, 0)) * max(1, category_frequency[category]))
                        for category in annotations.get(int(images[index]["id"]), set())
                    ),
                    -index,
                ),
            )
            selected.append(best)
            selected_set.add(best)
            journal = _journal(str(images[best]["file_name"]))
            journal_counts[journal] = journal_counts.get(journal, 0) + 1
            for category in annotations.get(int(images[best]["id"]), set()):
                category_counts[category] = category_counts.get(category, 0) + 1
        indices = sorted(selected)
    else:
        raise ValueError(f"Unsupported calibration selection: {strategy!r}")
    paths = []
    selected_rows = []
    for index in indices:
        relative = Path(str(images[index]["file_name"]))
        path = image_root / relative
        if not path.is_file():
            path = image_root / relative.name
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
        selected_rows.append(images[index])
    category_names = {
        int(row["id"]): str(row["name"])
        for row in payload.get("categories", [])
    }
    category_counts: dict[str, int] = {}
    journal_counts: dict[str, int] = {}
    for row in selected_rows:
        journal = _journal(str(row["file_name"]))
        journal_counts[journal] = journal_counts.get(journal, 0) + 1
        for category in annotations.get(int(row["id"]), set()):
            name = category_names.get(category, str(category))
            category_counts[name] = category_counts.get(name, 0) + 1
    return paths, {
        "strategy": strategy,
        "seed": seed if strategy == "random" else None,
        "image_ids": [int(row["id"]) for row in selected_rows],
        "files": [path.name for path in paths],
        "journals": dict(sorted(journal_counts.items())),
        "categories": dict(sorted(category_counts.items())),
    }


class ImageCalibrationReader:
    def __init__(
        self,
        paths: Sequence[Path],
        pack: ModelPack,
        *,
        image_backend: str,
        progress_every: int = 8,
    ) -> None:
        self.paths = list(paths)
        self.pack = pack
        self.image_backend = image_backend
        self.progress_every = progress_every
        self._index = 0
        input_config = pack.manifest.get("input", {})
        self._target_size = tuple(input_config.get("target_size", [800, 800]))
        self._scale = float(input_config.get("scale", 1.0 / 255.0))
        self._mean = tuple(input_config.get("mean", [0.0, 0.0, 0.0]))
        self._std = tuple(input_config.get("std", [1.0, 1.0, 1.0]))

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._index >= len(self.paths):
            return None
        path = self.paths[self._index]
        feeds, _ = prepare_image(
            path,
            self._target_size,
            backend=self.image_backend,
            scale=self._scale,
            mean=self._mean,
            std=self._std,
        )
        self._index += 1
        if self._index == 1 or self._index % self.progress_every == 0 or self._index == len(self.paths):
            print(
                f"[ppdoc-lite quantize] calibration={self._index}/{len(self.paths)} image={path.name}",
                flush=True,
            )
        return {name: np.expand_dims(value, axis=0) for name, value in feeds.items()}

    def rewind(self) -> None:
        self._index = 0


def prepare_output_pack(source: ModelPack, destination: Path, variant_id: str, model: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    output_model = destination / "model.onnx"
    if model.resolve() != output_model.resolve():
        shutil.copyfile(model, output_model)
    manifest = json.loads(json.dumps(source.manifest))
    manifest["variant_id"] = variant_id
    manifest.pop("profile", None)
    manifest["model"]["file"] = output_model.name
    manifest["model"]["sha256"] = sha256_file(output_model)
    manifest["model"]["bytes"] = output_model.stat().st_size
    manifest["derived_from"] = {
        "variant_id": source.variant_id,
        "model_sha256": source.manifest["model"]["sha256"],
    }
    write_json(destination / "manifest.json", manifest)


def run_preprocess(args: argparse.Namespace) -> int:
    from onnxruntime.quantization.shape_inference import quant_pre_process

    source = ModelPack.load(args.source_pack)
    receipt = _begin_receipt(args, "preprocess")
    receipt.update(
        {
            "skip_optimization": args.skip_optimization,
            "skip_symbolic_shape": args.skip_symbolic_shape,
            "skip_onnx_shape": args.skip_onnx_shape,
        }
    )
    receipt_path = args.output_pack / "quantization.json"
    write_json(receipt_path, receipt)
    model = args.output_pack / "model.onnx"
    model.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        quant_pre_process(
            source.model_path,
            model,
            skip_optimization=args.skip_optimization,
            skip_symbolic_shape=args.skip_symbolic_shape,
            skip_onnx_shape=args.skip_onnx_shape,
        )
        prepare_output_pack(source, args.output_pack, args.variant_id, model)
        receipt.update(
            {
                "phase": "complete",
                "output_sha256": sha256_file(model),
                "output_bytes": model.stat().st_size,
                "seconds": time.perf_counter() - started,
            }
        )
    except BaseException as exc:
        receipt.update(
            {
                "phase": "failed",
                "seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_json(receipt_path, receipt)
        raise
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    return 0


def run_fix_shapes(args: argparse.Namespace) -> int:
    import onnx
    from onnxruntime.tools.make_dynamic_shape_fixed import (
        fix_output_shapes,
        make_dim_param_fixed,
        make_input_shape_fixed,
    )

    source = ModelPack.load(args.source_pack)
    receipt = _begin_receipt(args, "fix_shapes")
    receipt_path = args.output_pack / "quantization.json"
    write_json(receipt_path, receipt)
    model_path = args.output_pack / "model.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        model = onnx.load(source.model_path)
        target_height, target_width = source.manifest.get("input", {}).get(
            "target_size", [800, 800]
        )
        shapes = {
            "image": [1, 3, int(target_height), int(target_width)],
            "im_shape": [1, 2],
            "scale_factor": [1, 2],
        }
        inputs = {value.name for value in model.graph.input}
        for name, shape in shapes.items():
            if name in inputs:
                make_input_shape_fixed(model.graph, name, shape)
        fix_output_shapes(model)
        for output in model.graph.output:
            batch = output.type.tensor_type.shape.dim[0]
            if batch.dim_param:
                make_dim_param_fixed(model.graph, batch.dim_param, 1)
        onnx.checker.check_model(model)
        onnx.save(model, model_path)
        prepare_output_pack(source, args.output_pack, args.variant_id, model_path)
        receipt.update(
            {
                "phase": "complete",
                "seconds": time.perf_counter() - started,
                "input_shapes": {name: shape for name, shape in shapes.items() if name in inputs},
                "output_sha256": sha256_file(model_path),
                "output_bytes": model_path.stat().st_size,
            }
        )
    except BaseException as exc:
        receipt.update(
            {
                "phase": "failed",
                "seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_json(receipt_path, receipt)
        raise
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    return 0


def _op_types(value: str) -> list[str] | None:
    choices = {
        "all": None,
        "conv": ["Conv"],
        "linear": ["MatMul", "Gemm"],
        "conv-linear": ["Conv", "MatMul", "Gemm"],
    }
    return choices[value]


def names_matching_patterns(names: Iterable[str], patterns: Sequence[str]) -> list[str]:
    compiled = [re.compile(pattern) for pattern in patterns]
    return sorted(name for name in names if any(pattern.fullmatch(name) for pattern in compiled))


def _begin_receipt(args: argparse.Namespace, operation: str) -> dict[str, Any]:
    source = ModelPack.load(args.source_pack)
    return {
        "schema_version": "legalpdf.ppdoc_lite_quantization.v1",
        "phase": "running",
        "operation": operation,
        "variant_id": args.variant_id,
        "source_variant_id": source.variant_id,
        "source_sha256": sha256_file(source.model_path),
        "source_bytes": source.model_path.stat().st_size,
        "started_epoch_seconds": time.time(),
    }


def run_dynamic(args: argparse.Namespace) -> int:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    source = ModelPack.load(args.source_pack)
    receipt = _begin_receipt(args, "dynamic")
    write_json(args.output_pack / "quantization.json", receipt)
    model = args.output_pack / "model.onnx"
    model.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        quantize_dynamic(
            source.model_path,
            model,
            op_types_to_quantize=_op_types(args.op_types),
            per_channel=args.per_channel,
            reduce_range=args.reduce_range,
            weight_type=QuantType.QInt8 if args.weight == "s8" else QuantType.QUInt8,
        )
        prepare_output_pack(source, args.output_pack, args.variant_id, model)
        receipt.update(
            {
                "phase": "complete",
                "seconds": time.perf_counter() - started,
                "output_sha256": sha256_file(model),
                "output_bytes": model.stat().st_size,
                "op_types": args.op_types,
                "weight": args.weight,
                "per_channel": args.per_channel,
                "reduce_range": args.reduce_range,
            }
        )
    except BaseException as exc:
        receipt.update(
            {
                "phase": "failed",
                "seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_json(args.output_pack / "quantization.json", receipt)
        raise
    write_json(args.output_pack / "quantization.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    return 0


def run_static(args: argparse.Namespace) -> int:
    import onnx
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    source = ModelPack.load(args.source_pack)
    receipt = _begin_receipt(args, "static")
    paths, selection = selected_images(
        args.annotations,
        args.image_root,
        args.calibration_pages,
        strategy=args.calibration_selection,
        seed=args.calibration_seed,
    )
    receipt["calibration"] = {
        "annotations": str(args.annotations),
        "annotation_sha256": sha256_file(args.annotations),
        "pages": len(paths),
        "selection": selection,
        "method": args.calibration_method,
        "image_backend": args.image_backend,
    }
    write_json(args.output_pack / "quantization.json", receipt)
    reader = ImageCalibrationReader(paths, source, image_backend=args.image_backend)
    formats = {"qdq": QuantFormat.QDQ, "qoperator": QuantFormat.QOperator}
    quant_types = {"s8": QuantType.QInt8, "u8": QuantType.QUInt8}
    methods = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }
    model = args.output_pack / "model.onnx"
    model.parent.mkdir(parents=True, exist_ok=True)
    cache = args.calibration_cache
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        nodes_to_quantize = None
        if args.node_name_regex:
            graph = onnx.load(source.model_path, load_external_data=False)
            nodes_to_quantize = names_matching_patterns(
                (node.name for node in graph.graph.node), args.node_name_regex
            )
            if not nodes_to_quantize:
                raise ValueError("--node-name-regex matched no ONNX nodes")
        quantize_static(
            source.model_path,
            model,
            calibration_data_reader=reader,
            quant_format=formats[args.format],
            op_types_to_quantize=_op_types(args.op_types),
            nodes_to_quantize=nodes_to_quantize,
            per_channel=args.per_channel,
            reduce_range=args.reduce_range,
            activation_type=quant_types[args.activation],
            weight_type=quant_types[args.weight],
            calibrate_method=methods[args.calibration_method],
            calibration_providers=["CPUExecutionProvider"],
            calibration_cache_path=cache,
            extra_options={
                "ActivationSymmetric": args.activation == "s8",
                "WeightSymmetric": args.weight == "s8",
                "CalibTensorRangeSymmetric": args.activation == "s8",
            },
        )
        prepare_output_pack(source, args.output_pack, args.variant_id, model)
        receipt.update(
            {
                "phase": "complete",
                "seconds": time.perf_counter() - started,
                "output_sha256": sha256_file(model),
                "output_bytes": model.stat().st_size,
                "format": args.format,
                "activation": args.activation,
                "weight": args.weight,
                "op_types": args.op_types,
                "node_name_regex": args.node_name_regex,
                "nodes_to_quantize": nodes_to_quantize,
                "per_channel": args.per_channel,
                "reduce_range": args.reduce_range,
                "calibration_cache": str(cache) if cache else None,
            }
        )
    except BaseException as exc:
        receipt.update(
            {
                "phase": "failed",
                "seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_json(args.output_pack / "quantization.json", receipt)
        raise
    write_json(args.output_pack / "quantization.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    return 0


def run_int4(args: argparse.Namespace) -> int:
    source = ModelPack.load(args.source_pack)
    receipt = _begin_receipt(args, "weight_only_int4")
    write_json(args.output_pack / "quantization.json", receipt)
    model_path = args.output_pack / "model.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        from onnxruntime.quantization import quant_utils

        try:
            from onnxruntime.quantization import matmul_4bits_quantizer

            config_type = matmul_4bits_quantizer.DefaultWeightOnlyQuantConfig
            quantizer_type = matmul_4bits_quantizer.MatMul4BitsQuantizer
        except ImportError:
            # ONNX Runtime 1.28 renamed the implementation while retaining
            # the same weight-only configuration contract.
            from onnxruntime.quantization.matmul_nbits_quantizer import (
                DefaultWeightOnlyQuantConfig,
                MatMulNBitsQuantizer,
            )

            config_type = DefaultWeightOnlyQuantConfig
            quantizer_type = MatMulNBitsQuantizer

        config = config_type(
            block_size=args.block_size,
            is_symmetric=args.symmetric,
            accuracy_level=args.accuracy_level,
            quant_format=(
                quant_utils.QuantFormat.QOperator
                if args.format == "qoperator"
                else quant_utils.QuantFormat.QDQ
            ),
            op_types_to_quantize=("MatMul",),
            quant_axes=(("MatMul", 0),),
        )
        model = quant_utils.load_model_with_shape_infer(source.model_path)
        quantizer = quantizer_type(model, algo_config=config)
        quantizer.process()
        quantizer.model.save_model_to_file(model_path, False)
        prepare_output_pack(source, args.output_pack, args.variant_id, model_path)
        receipt.update(
            {
                "phase": "complete",
                "seconds": time.perf_counter() - started,
                "output_sha256": sha256_file(model_path),
                "output_bytes": model_path.stat().st_size,
                "format": args.format,
                "block_size": args.block_size,
                "symmetric": args.symmetric,
                "accuracy_level": args.accuracy_level,
            }
        )
    except BaseException as exc:
        receipt.update(
            {
                "phase": "failed",
                "seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_json(args.output_pack / "quantization.json", receipt)
        raise
    write_json(args.output_pack / "quantization.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    return 0


def add_pack_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-pack", type=Path, required=True)
    parser.add_argument("--output-pack", type=Path, required=True)
    parser.add_argument("--variant-id", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auditable PPdoc-lite quantization build tool")
    commands = parser.add_subparsers(dest="command", required=True)

    preprocess = commands.add_parser("preprocess")
    add_pack_args(preprocess)
    preprocess.add_argument("--skip-optimization", action=argparse.BooleanOptionalAction, default=False)
    preprocess.add_argument("--skip-symbolic-shape", action=argparse.BooleanOptionalAction, default=False)
    preprocess.add_argument("--skip-onnx-shape", action=argparse.BooleanOptionalAction, default=False)
    preprocess.set_defaults(handler=run_preprocess)

    fixed = commands.add_parser("fix-shapes")
    add_pack_args(fixed)
    fixed.set_defaults(handler=run_fix_shapes)

    dynamic = commands.add_parser("dynamic")
    add_pack_args(dynamic)
    dynamic.add_argument("--op-types", choices=("all", "conv", "linear", "conv-linear"), default="linear")
    dynamic.add_argument("--weight", choices=("s8", "u8"), default="s8")
    dynamic.add_argument("--per-channel", action=argparse.BooleanOptionalAction, default=False)
    dynamic.add_argument("--reduce-range", action=argparse.BooleanOptionalAction, default=False)
    dynamic.set_defaults(handler=run_dynamic)

    static = commands.add_parser("static")
    add_pack_args(static)
    static.add_argument("--annotations", type=Path, required=True)
    static.add_argument("--image-root", type=Path, required=True)
    static.add_argument("--calibration-pages", type=int, required=True)
    static.add_argument(
        "--calibration-selection",
        choices=("even", "random", "class-journal"),
        default="class-journal",
    )
    static.add_argument("--calibration-seed", type=int, default=20260813)
    static.add_argument("--calibration-cache", type=Path)
    static.add_argument("--calibration-method", choices=("minmax", "entropy", "percentile"), default="minmax")
    static.add_argument("--format", choices=("qdq", "qoperator"), default="qdq")
    static.add_argument("--activation", choices=("s8", "u8"), default="s8")
    static.add_argument("--weight", choices=("s8", "u8"), default="s8")
    static.add_argument("--op-types", choices=("all", "conv", "linear", "conv-linear"), default="conv-linear")
    static.add_argument("--node-name-regex", action="append", default=[])
    static.add_argument("--per-channel", action=argparse.BooleanOptionalAction, default=False)
    static.add_argument("--reduce-range", action=argparse.BooleanOptionalAction, default=False)
    static.add_argument("--image-backend", choices=("opencv", "pillow"), default="opencv")
    static.set_defaults(handler=run_static)

    int4 = commands.add_parser("int4")
    add_pack_args(int4)
    int4.add_argument("--format", choices=("qoperator", "qdq"), default="qoperator")
    int4.add_argument("--block-size", type=int, choices=(16, 32, 64, 128, 256), default=128)
    int4.add_argument("--symmetric", action=argparse.BooleanOptionalAction, default=True)
    int4.add_argument("--accuracy-level", type=int, choices=(0, 1, 2, 3, 4), default=4)
    int4.set_defaults(handler=run_int4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
