from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .runtime import ModelPack, PPDocLite


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
PROGRESS_SCHEMA = "legalpdf.ppdoc_lite_progress.v1"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _pack_path(args: argparse.Namespace) -> Path:
    if args.model_pack:
        return args.model_pack
    root = args.model_root or os.environ.get("PPDOC_LITE_MODELS")
    if not root:
        raise SystemExit("Pass --model-pack, --model-root, or set PPDOC_LITE_MODELS")
    return Path(root) / args.variant


def _engine(args: argparse.Namespace) -> PPDocLite:
    providers = [value.strip() for value in args.providers.split(",") if value.strip()] if args.providers else None
    return PPDocLite(
        ModelPack.load(_pack_path(args)),
        device=args.device,
        providers=providers,
        threads=args.threads,
        strict_device=args.strict_device,
        image_backend=args.image_backend,
    )


def _images(args: argparse.Namespace) -> list[Path]:
    if args.image_list:
        rows = [value.strip() for value in args.image_list.read_text(encoding="utf-8-sig").splitlines() if value.strip()]
        images = [Path(value) if Path(value).is_absolute() else args.image_dir / value for value in rows]
    else:
        images = sorted(path for path in args.image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    return images[: args.max_pages] if args.max_pages > 0 else images


def _result_path(image: Path, args: argparse.Namespace) -> Path:
    if args.output_layout == "mirror-input":
        try:
            relative = image.resolve().relative_to(args.image_dir.resolve())
        except ValueError:
            relative = Path(image.name)
        return args.output_dir / relative.with_suffix(".json")
    return args.output_dir / f"{image.stem}.json"


def _dpi_preflight(images: Sequence[Path], minimum: int) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("DPI checking is optional; install the images-pillow extra to enable it") from exc
    rows = []
    issues: dict[str, int] = {}
    for image_path in images:
        row: dict[str, Any] = {"image": str(image_path), "issues": []}
        try:
            with Image.open(image_path) as image:
                row["width"], row["height"] = image.size
                dpi = image.info.get("dpi")
                if not isinstance(dpi, (tuple, list)) or len(dpi) < 2:
                    row["issues"].append("image_dpi_missing")
                else:
                    row["dpi_x"], row["dpi_y"] = float(dpi[0]), float(dpi[1])
                    if min(row["dpi_x"], row["dpi_y"]) < minimum - 1.0:
                        row["issues"].append("image_dpi_below_minimum")
        except Exception as exc:
            row["issues"].append("image_probe_failed")
            row["error"] = f"{type(exc).__name__}: {exc}"
        for issue in row["issues"]:
            issues[issue] = issues.get(issue, 0) + 1
        rows.append(row)
    return {
        "schema_version": "legalpdf.ppdoc_lite_image_preflight.v1",
        "min_required_dpi": minimum,
        "image_count": len(rows),
        "ok": not issues,
        "issue_counts": dict(sorted(issues.items())),
        "images": rows,
    }


def _progress(path: Path, *, phase: str, started: float, total: int, processed: int, skipped: int, error: str = "") -> None:
    payload = {
        "schema_version": PROGRESS_SCHEMA,
        "phase": phase,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
        "total_images": total,
        "processed_images": processed,
        "skipped_existing_images": skipped,
        "remaining_images": max(total - processed - skipped, 0),
        "error": error,
    }
    _write_json(path, payload)
    print(
        f"[ppdoc-lite] phase={phase} processed={processed} skipped={skipped} "
        f"remaining={payload['remaining_images']} elapsed={payload['elapsed_seconds']}s",
        flush=True,
    )


def run_infer(args: argparse.Namespace) -> int:
    images = _images(args)
    if not images:
        raise RuntimeError(f"No input images found in {args.image_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = args.output_dir.parent if args.output_dir.name in {"ppdoc_raw_layout_json", "raw_layout_json"} else args.output_dir
    preflight: dict[str, Any] = {"status": "not_requested"}
    if args.check_dpi:
        preflight = _dpi_preflight(images, args.min_production_dpi)
        _write_json(summary_dir / "ppdoc_image_dpi_preflight.json", preflight)
        if not preflight["ok"] and not args.allow_low_dpi_diagnostic:
            raise RuntimeError(f"PPDoc image DPI preflight failed: {preflight['issue_counts']}")

    started = time.time()
    progress_path = summary_dir / "ppdoc_inference_progress.json"
    _progress(progress_path, phase="loading_model", started=started, total=len(images), processed=0, skipped=0)
    engine = _engine(args)
    pending = []
    rows = []
    skipped = 0
    indexes = {image: index for index, image in enumerate(images, start=1)}
    for image in images:
        target = _result_path(image, args)
        if args.resume and target.is_file():
            try:
                existing = json.loads(target.read_text(encoding="utf-8"))
                rows.append({"image": image.name, "json": str(target), "detections": len(existing.get("detections", [])), "skipped_existing": True})
                skipped += 1
                continue
            except Exception:
                pass
        pending.append(image)
    processed = 0
    _progress(progress_path, phase="started", started=started, total=len(images), processed=0, skipped=skipped)
    try:
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            for image, result in zip(
                batch,
                engine.infer(
                    batch,
                    threshold=args.threshold,
                    layout_nms=args.layout_nms,
                    filter_overlap_boxes=args.filter_overlap_boxes,
                ),
                strict=True,
            ):
                target = _result_path(image, args)
                result.update(
                    {
                        "index": indexes[image],
                        "stem": image.stem,
                        "model_source": "trained",
                        "model_dir": str(engine.pack.root),
                        "model_name": engine.pack.manifest.get("source", {}).get("model_name", ""),
                        "runtime": "ppdoc-lite-onnxruntime",
                        "runtime_variant": engine.pack.variant_id,
                        "threshold": args.threshold,
                        "layout_nms": args.layout_nms,
                        "batch_size": args.batch_size,
                        "predict_mode": "batched" if args.batch_size > 1 else "per-image",
                    }
                )
                _write_json(target, result)
                rows.append({"image": image.name, "json": str(target), "detections": len(result["detections"])})
                processed += 1
            if processed % max(1, args.progress_interval) == 0 or processed == len(pending):
                _progress(progress_path, phase="predicting", started=started, total=len(images), processed=processed, skipped=skipped)
    except Exception as exc:
        _progress(progress_path, phase="failed", started=started, total=len(images), processed=processed, skipped=skipped, error=f"{type(exc).__name__}: {exc}")
        raise
    summary = {
        "schema_version": "legalpdf.ppdoc_lite_inference.v1",
        "runtime": "ppdoc-lite-onnxruntime",
        "variant_id": engine.pack.variant_id,
        "providers": list(engine.providers),
        "model_dir": str(engine.pack.root),
        "image_count": len(rows),
        "threshold": args.threshold,
        "layout_nms": args.layout_nms,
        "batch_size": args.batch_size,
        "image_dpi_preflight": preflight,
        "skipped_existing_images": skipped,
        "elapsed_seconds": round(time.time() - started, 3),
        "images": rows,
    }
    _write_json(summary_dir / "ppdoc_inference_summary.json", summary)
    _progress(progress_path, phase="complete", started=started, total=len(images), processed=processed, skipped=skipped)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def run_serve(args: argparse.Namespace) -> int:
    engine = _engine(args)
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        request_id = None
        try:
            request = json.loads(raw)
            request_id = request.get("id")
            paths = request.get("images") or [request["image"]]
            result = engine.infer(
                paths,
                threshold=float(request.get("threshold", args.threshold)),
                layout_nms=bool(request.get("layout_nms", args.layout_nms)),
                filter_overlap_boxes=bool(
                    request.get("filter_overlap_boxes", args.filter_overlap_boxes)
                ),
            )
            response = {"id": request_id, "ok": True, "result": result if "images" in request else result[0]}
        except Exception as exc:
            response = {"id": request_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, ensure_ascii=False), flush=True)
    return 0


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-pack", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--variant", default="fp32", help="Subdirectory below --model-root")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "directml", "openvino", "coreml"), default="auto")
    parser.add_argument("--providers", default="")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--strict-device", action="store_true")
    parser.add_argument("--image-backend", choices=("opencv", "pillow"), default="opencv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ppdoc-lite", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    infer = subparsers.add_parser("infer", help="Infer an image directory and write PPdoc-compatible JSON")
    add_model_args(infer)
    infer.add_argument("--image-dir", type=Path, required=True)
    infer.add_argument("--image-list", type=Path)
    infer.add_argument("--output-dir", type=Path, required=True)
    infer.add_argument("--threshold", type=float, default=0.10)
    infer.add_argument("--layout-nms", action=argparse.BooleanOptionalAction, default=False)
    infer.add_argument("--filter-overlap-boxes", action=argparse.BooleanOptionalAction, default=True)
    infer.add_argument("--batch-size", type=int, default=1)
    infer.add_argument("--output-layout", choices=("flat", "mirror-input"), default="flat")
    infer.add_argument("--max-pages", type=int, default=0)
    infer.add_argument("--progress-interval", type=int, default=10)
    infer.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    infer.add_argument("--check-dpi", action=argparse.BooleanOptionalAction, default=False)
    infer.add_argument("--min-production-dpi", type=int, default=300)
    infer.add_argument("--allow-low-dpi-diagnostic", action="store_true")
    infer.set_defaults(handler=run_infer)

    serve = subparsers.add_parser("serve", help="Run a persistent JSONL stdio worker")
    add_model_args(serve)
    serve.add_argument("--threshold", type=float, default=0.10)
    serve.add_argument("--layout-nms", action=argparse.BooleanOptionalAction, default=False)
    serve.add_argument("--filter-overlap-boxes", action=argparse.BooleanOptionalAction, default=True)
    serve.set_defaults(handler=run_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "batch_size", 1) < 1:
        raise SystemExit("--batch-size must be positive")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
