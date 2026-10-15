from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .model_pack import sha256_file


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _validate_recognition_source(source: Any) -> tuple[int, int, int, int]:
    """Validates the deliberately narrow version-one recognition ABI.

    The runtime rectifies stock-BLLA baseline polygons before recognition. A
    recognizer trained for axis-aligned boxes has a materially different input
    distribution, so accepting it would create a plausible-looking but false
    compatibility claim.
    """

    if getattr(source, "codec", None) is None:
        raise RuntimeError("Recognition model has no codec")
    try:
        batch, channels, height, width = (int(value) for value in source.input)
    except Exception as exc:
        raise RuntimeError("Recognition model has no valid NCHW input specification") from exc
    if height <= 0:
        raise RuntimeError(
            "The lightweight recognizer ABI requires a fixed input line height"
        )
    seg_type = getattr(source, "seg_type", None)
    if seg_type != "baselines":
        raise RuntimeError(
            "Version one only accepts recognizers trained for Kraken baseline "
            f"segmentations; model seg_type is {seg_type!r}"
        )
    return batch, channels, height, width


def _validate_stock_blla_task(task: Any) -> Any:
    """Returns the sole stock segmenter or fails on hidden model loss."""

    seg_models = list(getattr(task, "seg_models", ()))
    if len(seg_models) != 1:
        raise RuntimeError(
            "The stock BLLA pack was expected to contain one segmentation "
            f"model; found {len(seg_models)}"
        )
    # The lightweight runtime implements Kraken's deterministic polygonal
    # reading-order algorithm. Never silently discard an auxiliary neural
    # reading-order model if a future stock package begins shipping one.
    ro_models = list(getattr(task, "ro_models", ()))
    if ro_models:
        raise RuntimeError(
            "The installed stock BLLA package contains auxiliary neural "
            "reading-order models. This compiler does not yet export them; "
            "refusing to produce a geometry-incomplete pack."
        )
    return seg_models[0]


def _write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _export_onnx(
    module,
    dummy_inputs,
    output_path: Path,
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
    dynamic_axes: Mapping[str, Mapping[int, str]],
    opset: int,
) -> None:
    import torch

    module.eval().cpu()
    with torch.inference_mode():
        torch.onnx.export(
            module,
            dummy_inputs,
            output_path,
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_axes={key: dict(value) for key, value in dynamic_axes.items()},
            opset_version=opset,
            do_constant_folding=True,
            export_params=True,
            dynamo=False,
        )


def _verify_onnx(
    model_path: Path,
    inputs: Mapping[str, np.ndarray],
    torch_outputs: Sequence[np.ndarray],
    *,
    rtol: float = 2e-3,
    atol: float = 2e-4,
) -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    names = [item.name for item in session.get_outputs()]
    outputs = session.run(names, dict(inputs))
    if len(outputs) != len(torch_outputs):
        raise RuntimeError("ONNX verification returned an unexpected output count")
    for index, (expected, actual) in enumerate(zip(torch_outputs, outputs, strict=True)):
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            err_msg=f"ONNX parity failed for output {index}",
        )
    return True


def compile_stock_blla(
    output_dir: str | Path,
    *,
    dummy_height: int = 1200,
    dummy_width: int = 900,
    opset: int = 18,
    verify: bool = True,
) -> Path:
    """Compiles Kraken's bundled stock BLLA network into a runtime-only pack."""

    import torch
    from importlib import resources
    from kraken.tasks import SegmentationTaskModel

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "model.onnx"
    task = SegmentationTaskModel.load_model()
    source = _validate_stock_blla_task(task)
    source.eval().cpu()

    class Wrapper(torch.nn.Module):
        def __init__(self, model) -> None:
            super().__init__()
            self.model = model

        def forward(self, image):
            result = self.model.nn(image)
            return result[0] if isinstance(result, tuple) else result

    _, channels, height, width = (int(value) for value in source.input)
    export_height = height if height > 0 else int(dummy_height)
    export_width = width if width > 0 else int(dummy_width)
    wrapper = Wrapper(source)

    # Exercise a real multi-page batch at export time rather than merely
    # annotating a batch-one trace as dynamic. If the installed Kraken/model
    # combination cannot export a dynamic batch, retain a fully usable batch-one
    # graph and record that capability honestly in the manifest.
    dynamic_batch = True
    batch_export_fallback: str | None = None
    export_batch = 2
    dummy = torch.rand(
        (export_batch, channels, export_height, export_width), dtype=torch.float32
    )
    dynamic_axes = {
        "image": {0: "batch", 2: "height", 3: "width"},
        "heatmap": {0: "batch", 2: "heatmap_height", 3: "heatmap_width"},
    }
    try:
        _export_onnx(
            wrapper,
            dummy,
            model_path,
            input_names=["image"],
            output_names=["heatmap"],
            dynamic_axes=dynamic_axes,
            opset=opset,
        )
    except Exception as exc:
        dynamic_batch = False
        batch_export_fallback = f"{type(exc).__name__}: {exc}"
        export_batch = 1
        model_path.unlink(missing_ok=True)
        dummy = torch.rand(
            (export_batch, channels, export_height, export_width),
            dtype=torch.float32,
        )
        _export_onnx(
            wrapper,
            dummy,
            model_path,
            input_names=["image"],
            output_names=["heatmap"],
            dynamic_axes={
                "image": {2: "height", 3: "width"},
                "heatmap": {2: "heatmap_height", 3: "heatmap_width"},
            },
            opset=opset,
        )

    verified_shapes: list[list[int]] = []
    if verify:
        with torch.inference_mode():
            expected = wrapper(dummy).detach().cpu().numpy()
        if _verify_onnx(model_path, {"image": dummy.numpy()}, [expected]):
            verified_shapes.append(list(dummy.shape))

        alternate_height = max(128, export_height // 2)
        alternate_width = max(128, export_width // 2 + 17)
        if (alternate_height, alternate_width) != (export_height, export_width):
            alternate = torch.rand(
                (export_batch, channels, alternate_height, alternate_width),
                dtype=torch.float32,
            )
            with torch.inference_mode():
                expected = wrapper(alternate).detach().cpu().numpy()
            if _verify_onnx(
                model_path, {"image": alternate.numpy()}, [expected]
            ):
                verified_shapes.append(list(alternate.shape))

    model_resource = resources.files("kraken").joinpath("blla.mlmodel")
    source_hash = None
    try:
        source_hash = sha256_file(Path(str(model_resource)))
    except (FileNotFoundError, TypeError, OSError):
        pass
    metadata = source.user_metadata
    class_mapping = _jsonable(metadata.get("class_mapping", {}))
    manifest = {
        "format": "kraken-lite-model/1",
        "kind": "blla-segmentation",
        "id": "kraken-stock-blla",
        "source": {
            "engine": "kraken",
            "krakenVersion": importlib.metadata.version("kraken"),
            "model": "bundled stock blla.mlmodel",
            "sha256": source_hash,
        },
        "model": {
            "file": model_path.name,
            "sha256": sha256_file(model_path),
            "input": "image",
            "output": "heatmap",
            "opset": opset,
            "dynamic_batch": dynamic_batch,
        },
        "verification": {
            "requested": bool(verify),
            "onnxRuntimeParity": bool(verified_shapes),
            "shapes": verified_shapes,
            "batchExportFallback": batch_export_fallback,
        },
        "input": {
            "channels": channels,
            "height": height,
            "width": width,
            "padding": [0, 0, 0, 0],
            "invert": True,
            "oneChannelMode": source.one_channel_mode,
        },
        "output": {"activation": "sigmoid"},
        "segmentation": {
            "class_mapping": class_mapping,
            "bounding_regions": _jsonable(metadata.get("bounding_regions")),
            "topline": metadata.get("topline", False),
            "baseline_threshold": 0.17,
            "region_threshold": 0.5,
            "minimum_baseline_length": 5.0,
            "maximum_polygon_distance": 72.0,
            "geometry": "blla-baseline-polygon-v1",
            "cpu_batch_size": 1,
            "accelerated_batch_size": 4,
            "width_bucket": 0,
            "maximum_batch_pixels": 80000000,
        },
    }
    _write_manifest(root, manifest)
    return root


def compile_recognizer(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    dummy_width: int = 1024,
    opset: int = 18,
    verify: bool = True,
) -> Path:
    """Compiles one Kraken recognition model into the tensor-only runtime ABI."""

    import torch
    from kraken.models import load_models

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "model.onnx"
    models = [model for model in load_models(source_path) if "recognition" in model.model_type]
    if len(models) != 1:
        raise RuntimeError(f"Expected exactly one recognition model, found {len(models)}")
    source = models[0]
    _, channels, height, width = _validate_recognition_source(source)
    source.eval().cpu()

    class Wrapper(torch.nn.Module):
        def __init__(self, model) -> None:
            super().__init__()
            self.model = model

        def forward(self, image, sequence_lengths):
            result = self.model.nn(image, sequence_lengths)
            if isinstance(result, tuple):
                logits = result[0]
                output_lengths = result[1]
            else:
                logits = result
                output_lengths = None
            if output_lengths is None:
                output_lengths = torch.floor(
                    sequence_lengths.to(torch.float32)
                    * (logits.shape[-1] / image.shape[-1])
                ).to(torch.int64)
                output_lengths = output_lengths.clamp(min=1, max=logits.shape[-1])
            return logits, output_lengths

    export_width = width if width > 0 else int(dummy_width)
    wrapper = Wrapper(source)
    dynamic_batch = True
    batch_export_fallback: str | None = None
    export_batch = 2

    def make_dummy(batch_size: int):
        image = torch.rand(
            (batch_size, channels, height, export_width), dtype=torch.float32
        )
        lengths = torch.tensor(
            [max(1, export_width - 64 * index) for index in range(batch_size)],
            dtype=torch.int64,
        )
        return image, lengths

    dummy, dummy_lengths = make_dummy(export_batch)
    dynamic_axes: dict[str, dict[int, str]] = {
        "image": {0: "batch"},
        "sequence_lengths": {0: "batch"},
        "logits": {0: "batch"},
        "output_lengths": {0: "batch"},
    }
    if width <= 0:
        dynamic_axes["image"][3] = "width"
        dynamic_axes["logits"][3] = "timesteps"
    try:
        _export_onnx(
            wrapper,
            (dummy, dummy_lengths),
            model_path,
            input_names=["image", "sequence_lengths"],
            output_names=["logits", "output_lengths"],
            dynamic_axes=dynamic_axes,
            opset=opset,
        )
    except Exception as exc:
        dynamic_batch = False
        batch_export_fallback = f"{type(exc).__name__}: {exc}"
        export_batch = 1
        model_path.unlink(missing_ok=True)
        dummy, dummy_lengths = make_dummy(export_batch)
        fallback_axes: dict[str, dict[int, str]] = {}
        if width <= 0:
            fallback_axes = {
                "image": {3: "width"},
                "logits": {3: "timesteps"},
            }
        _export_onnx(
            wrapper,
            (dummy, dummy_lengths),
            model_path,
            input_names=["image", "sequence_lengths"],
            output_names=["logits", "output_lengths"],
            dynamic_axes=fallback_axes,
            opset=opset,
        )
    verified_shapes: list[list[int]] = []
    if verify:
        with torch.inference_mode():
            expected_logits, expected_lengths = wrapper(dummy, dummy_lengths)
        if _verify_onnx(
            model_path,
            {
                "image": dummy.numpy(),
                "sequence_lengths": dummy_lengths.numpy(),
            },
            [
                expected_logits.detach().cpu().numpy(),
                expected_lengths.detach().cpu().numpy(),
            ],
        ):
            verified_shapes.append(list(dummy.shape))

        if width <= 0:
            alternate_width = max(128, export_width // 2 + 31)
            if alternate_width != export_width:
                alternate = torch.rand(
                    (export_batch, channels, height, alternate_width),
                    dtype=torch.float32,
                )
                alternate_lengths = torch.tensor(
                    [
                        max(1, alternate_width - 37 * index)
                        for index in range(export_batch)
                    ],
                    dtype=torch.int64,
                )
                with torch.inference_mode():
                    expected_logits, expected_lengths = wrapper(
                        alternate, alternate_lengths
                    )
                if _verify_onnx(
                    model_path,
                    {
                        "image": alternate.numpy(),
                        "sequence_lengths": alternate_lengths.numpy(),
                    },
                    [
                        expected_logits.detach().cpu().numpy(),
                        expected_lengths.detach().cpu().numpy(),
                    ],
                ):
                    verified_shapes.append(list(alternate.shape))

    codec = _jsonable(source.codec.c2l)
    codec_path = root / "codec.json"
    codec_path.write_text(
        json.dumps(codec, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    source_path = Path(source_path)
    manifest = {
        "format": "kraken-lite-model/1",
        "kind": "recognition",
        "id": source_path.stem,
        "source": {
            "engine": "kraken",
            "krakenVersion": importlib.metadata.version("kraken"),
            "file": source_path.name,
            "sha256": sha256_file(source_path),
            "modelClass": source.__class__.__name__,
        },
        "model": {
            "file": model_path.name,
            "sha256": sha256_file(model_path),
            "input": "image",
            "output": "logits",
            "lengths_input": "sequence_lengths",
            "lengths_output": "output_lengths",
            "opset": opset,
            "dynamic_batch": dynamic_batch,
        },
        "verification": {
            "requested": bool(verify),
            "onnxRuntimeParity": bool(verified_shapes),
            "shapes": verified_shapes,
            "batchExportFallback": batch_export_fallback,
        },
        "input": {
            "channels": channels,
            "height": height,
            "width": width,
            "padding": 0 if width > 0 else 16,
            "invert": True,
            "oneChannelMode": source.one_channel_mode,
        },
        "output": {
            "activation": "logits",
            "class_axis": 1,
            "blank_label": 0,
        },
        "codec": {"file": "codec.json", "sha256": sha256_file(codec_path)},
        "recognition": {
            "seg_type": source.seg_type,
            "legacy_polygons": bool(source.use_legacy_polygons),
            "cpu_batch_size": 4,
            "accelerated_batch_size": 32,
            "width_bucket": 128,
        },
    }
    _write_manifest(root, manifest)
    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kraken-lite-compile",
        description="Compile Kraken models into dependency-light ONNX model packs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    blla = subparsers.add_parser("stock-blla")
    blla.add_argument("output")
    blla.add_argument("--dummy-height", type=int, default=1200)
    blla.add_argument("--dummy-width", type=int, default=900)
    blla.add_argument("--opset", type=int, default=18)
    blla.add_argument("--no-verify", action="store_true")

    recognition = subparsers.add_parser("recognizer")
    recognition.add_argument("source")
    recognition.add_argument("output")
    recognition.add_argument("--dummy-width", type=int, default=1024)
    recognition.add_argument("--opset", type=int, default=18)
    recognition.add_argument("--no-verify", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "stock-blla":
        compile_stock_blla(
            args.output,
            dummy_height=args.dummy_height,
            dummy_width=args.dummy_width,
            opset=args.opset,
            verify=not args.no_verify,
        )
    else:
        compile_recognizer(
            args.source,
            args.output,
            dummy_width=args.dummy_width,
            opset=args.opset,
            verify=not args.no_verify,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
