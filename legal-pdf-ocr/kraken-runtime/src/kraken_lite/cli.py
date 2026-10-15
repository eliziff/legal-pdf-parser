from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence, TextIO

from PIL import Image

from .blla import BLLASegmenter
from .compiler import compile_recognizer, compile_stock_blla
from .engine import KrakenLiteEngine
from .hardware import resolve_provider_plan
from .model_pack import available_execution_providers
from .recognition import Recognizer


def _providers(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _write(value: Any, target: str, *, text: bool = False) -> None:
    handle: TextIO
    close = False
    if target == "-":
        handle = sys.stdout
    else:
        handle = Path(target).open("w", encoding="utf-8", newline="\n")
        close = True
    try:
        if text:
            handle.write(str(value))
            if not str(value).endswith("\n"):
                handle.write("\n")
        else:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    finally:
        if close:
            handle.close()


def _load_engine(args, *, recognition: bool) -> KrakenLiteEngine:
    providers = _providers(args.providers)
    blla_path = args.blla or os.environ.get("KRAKEN_LITE_BLLA")
    if not blla_path:
        raise SystemExit(
            "No stock-BLLA model pack was provided. Use --blla or set "
            "KRAKEN_LITE_BLLA."
        )
    segmenter = BLLASegmenter.from_pack(
        blla_path,
        device=args.device,
        providers=providers,
        intra_threads=args.threads,
        allow_cpu_fallback=not args.strict_device,
        tensorrt_fp16=args.tensorrt_fp16,
        tensorrt_cache=args.tensorrt_cache,
        cuda_tf32=not args.no_tf32,
    )
    recognizer = None
    if recognition:
        recognizer_path = args.recognizer or os.environ.get("KRAKEN_LITE_RECOGNIZER")
        if not recognizer_path:
            raise SystemExit(
                "No recognition model pack was provided. Use --recognizer or set "
                "KRAKEN_LITE_RECOGNIZER."
            )
        recognizer = Recognizer.from_pack(
            recognizer_path,
            device=args.device,
            providers=providers,
            intra_threads=args.threads,
            allow_cpu_fallback=not args.strict_device,
            tensorrt_fp16=args.tensorrt_fp16,
            tensorrt_cache=args.tensorrt_cache,
            cuda_tf32=not args.no_tf32,
        )
    return KrakenLiteEngine(segmenter, recognizer)


def _run_one(args, *, recognition: bool) -> int:
    engine = _load_engine(args, recognition=recognition)
    with Image.open(args.image) as source:
        image = source.copy()
    if recognition:
        page = engine.recognize(
            image,
            text_direction=args.text_direction,
            raise_on_error=args.raise_on_error,
            batch_size=args.batch_size,
            page_batch_size=args.page_batch_size,
        )
        if args.format == "text":
            _write(page.text, args.output, text=True)
        else:
            _write(page.to_dict(), args.output)
    else:
        segmentation = engine.segment(
            image,
            text_direction=args.text_direction,
            raise_on_error=args.raise_on_error,
            batch_size=args.page_batch_size,
        )
        _write(segmentation.to_dict(), args.output)
    return 0



def _supplied_segmentations(images, pages):
    from .types import BaselineLine, Segmentation

    if not isinstance(pages, list) or len(pages) != len(images):
        raise ValueError("One supplied geometry page is required per image")
    result = []
    for image, page in zip(images, pages, strict=True):
        if not isinstance(page, dict) or not isinstance(page.get("lines"), list):
            raise ValueError("Supplied page requires lines")
        width, height = image.size
        identities = set()
        lines = []
        for line in page["lines"]:
            identity = line.get("id")
            if not isinstance(identity, str) or not identity or identity in identities:
                raise ValueError("Supplied line IDs must be unique nonempty strings")
            identities.add(identity)
            points = {}
            for key, minimum in (("baseline", 2), ("boundary", 3)):
                raw = line.get(key)
                if not isinstance(raw, list) or len(raw) < minimum:
                    raise ValueError(f"Supplied {key} has too few points")
                values = []
                for point in raw:
                    if not isinstance(point, list) or len(point) != 2:
                        raise ValueError("Geometry requires coordinate pairs")
                    x, y = point
                    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in point):
                        raise ValueError("Geometry coordinates must be finite numbers")
                    if not (0 <= x <= width and 0 <= y <= height):
                        raise ValueError("Geometry lies outside original image")
                    values.append((float(x), float(y)))
                if len(set(values)) < minimum:
                    raise ValueError("Supplied geometry is degenerate")
                points[key] = values
            lines.append(BaselineLine(id=identity, **points))
        result.append(Segmentation(width=width, height=height, lines=lines))
    return result

def _serve(args) -> int:
    has_recognizer = bool(
        args.recognizer or os.environ.get("KRAKEN_LITE_RECOGNIZER")
    )
    engine = _load_engine(args, recognition=has_recognizer)
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        request_id = None
        try:
            request = json.loads(raw)
            request_id = request.get("id")
            operation = request.get("op", "ocr")
            direction = request.get("textDirection", "horizontal-lr")
            page_batch_size = request.get(
                "pageBatchSize", args.page_batch_size
            )
            if operation in {"segment_batch", "ocr_batch", "recognize_lines_batch"}:
                paths = request["images"]
                images = []
                for path in paths:
                    with Image.open(path) as source:
                        images.append(source.copy())
                if operation == "recognize_lines_batch":
                    supplied = _supplied_segmentations(images, request.get("pages"))
                    result = [page.to_dict() for page in engine.recognize_segmentations(
                        images, supplied, batch_size=request.get("batchSize", args.batch_size)
                    )]
                elif operation == "segment_batch":
                    result = [
                        page.to_dict()
                        for page in engine.segment_many(
                            images,
                            text_direction=direction,
                            batch_size=page_batch_size,
                        )
                    ]
                else:
                    result = [
                        page.to_dict()
                        for page in engine.recognize_many(
                            images,
                            text_direction=direction,
                            batch_size=request.get(
                                "batchSize", args.batch_size
                            ),
                            page_batch_size=page_batch_size,
                        )
                    ]
            else:
                path = request["image"]
                with Image.open(path) as source:
                    image = source.copy()
                if operation == "segment":
                    result = engine.segment(
                        image,
                        text_direction=direction,
                        batch_size=page_batch_size,
                    ).to_dict()
                elif operation == "ocr":
                    result = engine.recognize(
                        image,
                        text_direction=direction,
                        batch_size=request.get("batchSize", args.batch_size),
                        page_batch_size=page_batch_size,
                    ).to_dict()
                else:
                    raise ValueError(f"Unsupported operation {operation!r}")
            response = {"id": request_id, "ok": True, "result": result}
        except Exception as exc:
            response = {
                "id": request_id,
                "ok": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


def _devices(args) -> int:
    available = available_execution_providers()
    plan = resolve_provider_plan(
        available,
        device=args.device,
        providers=_providers(args.providers),
        allow_cpu_fallback=not args.strict_device,
        tensorrt_fp16=args.tensorrt_fp16,
        tensorrt_cache=args.tensorrt_cache,
        cuda_tf32=not args.no_tf32,
    )
    _write(plan.to_dict(), args.output)
    return 0


def _add_runtime_options(
    parser: argparse.ArgumentParser, *, recognizer: bool
) -> None:
    parser.add_argument("--blla", help="Unpacked compiled stock-BLLA model pack")
    parser.add_argument(
        "--page-batch-size",
        type=int,
        default=0,
        help=(
            "BLLA page batch size; 0 chooses 4 on accelerated providers and "
            "1 on CPU. Pages are shape-bucketed before execution."
        ),
    )
    if recognizer:
        parser.add_argument("--recognizer", help="Unpacked compiled recognition pack")
        parser.add_argument(
            "--batch-size",
            type=int,
            default=0,
            help="Recognition batch size; 0 chooses 32 on accelerated providers and 4 on CPU",
        )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Execution device: auto, cpu, cuda[:N], rocm[:N], directml[:N], "
            "coreml, openvino, or tensorrt[:N]"
        ),
    )
    parser.add_argument(
        "--providers",
        help="Expert override: comma-separated ONNX Runtime provider names",
    )
    parser.add_argument(
        "--strict-device",
        action="store_true",
        help="Do not fall back to CPU when an accelerated provider cannot initialize",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--no-tf32",
        action="store_true",
        help="Disable CUDA TF32 kernels",
    )
    parser.add_argument(
        "--tensorrt-fp16",
        action="store_true",
        help="Allow TensorRT FP16 execution",
    )
    parser.add_argument(
        "--tensorrt-cache",
        help="TensorRT engine-cache directory",
    )
    parser.add_argument(
        "--text-direction",
        choices=["horizontal-lr", "horizontal-rl", "vertical-lr", "vertical-rl"],
        default="horizontal-lr",
    )
    parser.add_argument("--raise-on-error", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kraken-lite",
        description=(
            "Run Kraken's stock BLLA geometry and compiled recognition models "
            "without the Kraken training/runtime dependency graph."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    segment = commands.add_parser("segment", help="Stock-BLLA page segmentation")
    segment.add_argument("image")
    segment.add_argument("output", nargs="?", default="-")
    _add_runtime_options(segment, recognizer=False)

    ocr = commands.add_parser("ocr", help="Segment and recognize a page")
    ocr.add_argument("image")
    ocr.add_argument("output", nargs="?", default="-")
    ocr.add_argument("--format", choices=["json", "text"], default="json")
    _add_runtime_options(ocr, recognizer=True)

    serve = commands.add_parser("serve", help="Long-lived newline-delimited JSON mode")
    _add_runtime_options(serve, recognizer=True)

    devices = commands.add_parser(
        "devices", help="Show available ONNX Runtime providers and the selected plan"
    )
    devices.add_argument("--output", default="-")
    devices.add_argument("--device", default="auto")
    devices.add_argument("--providers")
    devices.add_argument("--strict-device", action="store_true")
    devices.add_argument("--no-tf32", action="store_true")
    devices.add_argument("--tensorrt-fp16", action="store_true")
    devices.add_argument("--tensorrt-cache")

    compile_blla = commands.add_parser(
        "compile-blla", help="Compile Kraken's bundled stock BLLA model"
    )
    compile_blla.add_argument("output")
    compile_blla.add_argument("--dummy-height", type=int, default=1200)
    compile_blla.add_argument("--dummy-width", type=int, default=900)
    compile_blla.add_argument("--opset", type=int, default=18)
    compile_blla.add_argument("--no-verify", action="store_true")

    compile_rec = commands.add_parser(
        "compile-recognizer", help="Compile one Kraken recognition model"
    )
    compile_rec.add_argument("source")
    compile_rec.add_argument("output")
    compile_rec.add_argument("--dummy-width", type=int, default=1024)
    compile_rec.add_argument("--opset", type=int, default=18)
    compile_rec.add_argument("--no-verify", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "segment":
            return _run_one(args, recognition=False)
        if args.command == "ocr":
            return _run_one(args, recognition=True)
        if args.command == "serve":
            return _serve(args)
        if args.command == "devices":
            return _devices(args)
        if args.command == "compile-blla":
            compile_stock_blla(
                args.output,
                dummy_height=args.dummy_height,
                dummy_width=args.dummy_width,
                opset=args.opset,
                verify=not args.no_verify,
            )
            return 0
        if args.command == "compile-recognizer":
            compile_recognizer(
                args.source,
                args.output,
                dummy_width=args.dummy_width,
                opset=args.opset,
                verify=not args.no_verify,
            )
            return 0
        raise AssertionError(args.command)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
