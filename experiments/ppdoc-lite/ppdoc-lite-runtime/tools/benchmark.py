from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ppdoc_lite.runtime import (
    ModelPack,
    PPDocLite,
    decode_rtdetr_raw,
    load_rgb,
    normalized_detections,
    postprocess_boxes,
    prepare_rgb,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def percentile(values: Sequence[float], value: float) -> float | None:
    return float(np.percentile(values, value)) if values else None


def installed_distribution(name: str) -> dict[str, Any]:
    distribution = importlib.metadata.distribution(name)
    files = list(distribution.files or [])
    total = 0
    for relative in files:
        path = Path(distribution.locate_file(relative))
        try:
            total += path.stat().st_size
        except OSError:
            pass
    return {
        "name": distribution.metadata["Name"],
        "version": distribution.version,
        "installed_bytes": total,
        "requires": list(distribution.requires or []),
    }


def run_environment(args: argparse.Namespace) -> int:
    rows = []
    names = args.distribution or sorted(
        {
            str(distribution.metadata["Name"])
            for distribution in importlib.metadata.distributions()
            if distribution.metadata["Name"]
        },
        key=str.casefold,
    )
    for name in names:
        try:
            rows.append(installed_distribution(name))
        except importlib.metadata.PackageNotFoundError:
            rows.append({"name": name, "missing": True})
    payload = {
        "schema_version": "legalpdf.ppdoc_lite_environment.v1",
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "distributions": rows,
        "total_distribution_bytes": sum(int(row.get("installed_bytes") or 0) for row in rows),
    }
    if args.output:
        write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


def load_split(annotation_path: Path, image_root: Path, limit: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    gold = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = sorted(gold["images"], key=lambda row: (str(row["file_name"]), int(row["id"])))
    if limit > 0:
        images = images[:limit]
    for row in images:
        path = image_root / str(row["file_name"])
        if not path.is_file():
            fallback = image_root / Path(str(row["file_name"])).name
            if fallback.is_file():
                path = fallback
            else:
                raise FileNotFoundError(path)
        row["_path"] = str(path)
    category_ids = {str(row["name"]): int(row["id"]) for row in gold["categories"]}
    return gold, images, category_ids


def memory_rss(*, peak: bool = False) -> int | None:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        get_memory_info = kernel32.K32GetProcessMemoryInfo
        get_memory_info.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        )
        get_memory_info.restype = wintypes.BOOL
        handle = kernel32.GetCurrentProcess()
        if get_memory_info(handle, ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize if peak else counters.WorkingSetSize)
        return None
    status = Path("/proc/self/status")
    if status.is_file():
        field = "VmHWM:" if peak else "VmRSS:"
        for line in status.read_text(encoding="ascii").splitlines():
            if line.startswith(field):
                return int(line.split()[1]) * 1024
    return None


def detections_to_coco(
    result: dict[str, Any], image_id: int, category_ids: dict[str, int]
) -> list[dict[str, Any]]:
    rows = []
    for detection in result["detections"]:
        category_id = category_ids.get(str(detection["label"]))
        if category_id is None:
            continue
        x0, y0, x1, y1 = [float(value) for value in detection["box"]]
        rows.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)],
                "score": float(detection["score"]),
            }
        )
    return rows


def score_coco(annotation_path: Path, predictions: list[dict[str, Any]], image_ids: Sequence[int]) -> dict[str, Any]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("Install the benchmark extra to compute held-out COCO metrics") from exc
    gold = COCO(str(annotation_path))
    if predictions:
        candidate = gold.loadRes(predictions)
    else:
        candidate = gold.loadRes([])
    evaluation = COCOeval(gold, candidate, "bbox")
    evaluation.params.imgIds = list(image_ids)
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    names = (
        "ap_iou_50_95",
        "ap_iou_50",
        "ap_iou_75",
        "ap_small",
        "ap_medium",
        "ap_large",
        "ar_1",
        "ar_10",
        "ar_100",
        "ar_small",
        "ar_medium",
        "ar_large",
    )
    return {name: float(value) for name, value in zip(names, evaluation.stats, strict=True)}


def openvino_config(
    performance_hint: str,
    inference_precision: str,
    threads: int,
    streams: int,
    core_type: str,
    cpu_pinning: bool | None,
    hyper_threading: bool | None,
) -> dict[str, Any]:
    config: dict[str, Any] = {"PERFORMANCE_HINT": performance_hint.upper()}
    if inference_precision != "auto":
        config["INFERENCE_PRECISION_HINT"] = inference_precision
    if threads > 0:
        config["INFERENCE_NUM_THREADS"] = threads
    if streams > 0:
        config["NUM_STREAMS"] = streams
    if core_type != "auto":
        config["SCHEDULING_CORE_TYPE"] = {
            "any": "ANY_CORE",
            "pcores": "PCORE_ONLY",
            "ecores": "ECORE_ONLY",
        }[core_type]
    if cpu_pinning is not None:
        config["ENABLE_CPU_PINNING"] = cpu_pinning
    if hyper_threading is not None:
        config["ENABLE_HYPER_THREADING"] = hyper_threading
    return config


class OpenVinoRaw:
    """Native OpenVINO benchmark adapter for supported detector contracts."""

    def __init__(
        self,
        pack: ModelPack,
        *,
        device: str,
        inference_precision: str,
        performance_hint: str,
        image_backend: str,
        cache_dir: Path | None,
        threads: int,
        streams: int,
        core_type: str,
        cpu_pinning: bool | None,
        hyper_threading: bool | None,
    ) -> None:
        try:
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError("Install OpenVINO to use --device openvino-native") from exc
        if pack.output_contract not in {"decoded_boxes", "rtdetr_raw"}:
            raise ValueError(
                "Native OpenVINO requires the decoded_boxes or rtdetr_raw contract"
            )
        config = openvino_config(
            performance_hint,
            inference_precision,
            threads,
            streams,
            core_type,
            cpu_pinning,
            hyper_threading,
        )
        core = ov.Core()
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            core.set_property({"CACHE_DIR": str(cache_dir)})
        model = core.read_model(pack.model_path)
        self._compiled = core.compile_model(model, device, config)
        self.runtime_config = {}
        for key in (
            "INFERENCE_NUM_THREADS",
            "NUM_STREAMS",
            "PERFORMANCE_HINT",
            "SCHEDULING_CORE_TYPE",
            "ENABLE_CPU_PINNING",
            "ENABLE_HYPER_THREADING",
            "EXECUTION_DEVICES",
        ):
            try:
                self.runtime_config[key] = str(self._compiled.get_property(key))
            except RuntimeError:
                pass
        self._request = self._compiled.create_infer_request()
        self._input_names = tuple(port.get_any_name() for port in self._compiled.inputs)
        output_keys = (
            ("boxes", "logits")
            if pack.output_contract == "rtdetr_raw"
            else ("boxes", "counts")
        )
        self._output_ports = {
            name: self._compiled.output(pack.output_names[name]) for name in output_keys
        }
        self._output_contract = pack.output_contract
        self._top_k = int(pack.manifest["model"].get("detections_per_image") or 300)
        self._output_width = int(pack.manifest["model"].get("output_width") or 7)
        self.pack = pack
        self.providers = (f"OpenVINO:{device}:{inference_precision}",)
        input_config = pack.manifest.get("input", {})
        self._target_size = tuple(input_config.get("target_size", [800, 800]))
        self._scale = float(input_config.get("scale", 1.0 / 255.0))
        self._mean = tuple(input_config.get("mean", [0.0, 0.0, 0.0]))
        self._std = tuple(input_config.get("std", [1.0, 1.0, 1.0]))
        self.image_backend = image_backend

    def infer_tensors(
        self,
        feeds: dict[str, np.ndarray],
        image_sizes: Sequence[tuple[int, int]],
        *,
        image_ids: Sequence[str | None] | None = None,
        threshold: float = 0.10,
        layout_nms: bool = False,
        filter_overlap_boxes: bool = True,
    ) -> list[dict[str, Any]]:
        values = self._request.infer(
            {name: np.asarray(feeds[name], dtype=np.float32) for name in self._input_names}
        )
        if self._output_contract == "rtdetr_raw":
            decoded = decode_rtdetr_raw(
                np.asarray(values[self._output_ports["boxes"]]),
                np.asarray(values[self._output_ports["logits"]]),
                feeds,
                self._top_k,
            )
            counts = np.full((decoded.shape[0],), decoded.shape[1], dtype=np.int32)
            flat_boxes = decoded.reshape((-1, decoded.shape[-1]))
        else:
            flat_boxes = np.asarray(values[self._output_ports["boxes"]]).reshape(
                (-1, self._output_width)
            )
            counts = np.asarray(values[self._output_ports["counts"]]).reshape(-1)
        if len(counts) != len(image_sizes):
            raise ValueError(
                f"OpenVINO returned {len(counts)} counts for {len(image_sizes)} images"
            )
        identifiers = list(image_ids) if image_ids is not None else [None] * len(image_sizes)
        results = []
        offset = 0
        for identifier, image_size, count_value in zip(
            identifiers, image_sizes, counts, strict=True
        ):
            count = int(count_value)
            raw = flat_boxes[offset : offset + count]
            offset += count
            boxes = postprocess_boxes(
                raw,
                labels=self.pack.labels,
                image_size=image_size,
                threshold=threshold,
                layout_nms=layout_nms,
                filter_overlap_boxes=filter_overlap_boxes,
            )
            results.append(
                {
                    "image": identifier,
                    "image_size": list(image_size),
                    "detections": normalized_detections(boxes),
                    "raw_payloads": [{"input_path": identifier, "boxes": boxes}],
                }
            )
        return results

    def infer_rgb(
        self,
        images: Sequence[np.ndarray],
        *,
        image_ids: Sequence[str | None] | None = None,
        **options: Any,
    ) -> list[dict[str, Any]]:
        prepared = [
            prepare_rgb(
                image,
                self._target_size,
                backend=self.image_backend,
                scale=self._scale,
                mean=self._mean,
                std=self._std,
            )
            for image in images
        ]
        feeds = {
            name: np.stack([row[0][name] for row in prepared], axis=0)
            for name in ("im_shape", "image", "scale_factor")
        }
        return self.infer_tensors(
            feeds,
            [row[1] for row in prepared],
            image_ids=image_ids,
            **options,
        )

    def infer(self, paths: Sequence[Path], **options: Any) -> list[dict[str, Any]]:
        identifiers = [str(path) for path in paths]
        return self.infer_rgb(
            [load_rgb(path, self.image_backend) for path in paths],
            image_ids=identifiers,
            **options,
        )


def run_benchmark(args: argparse.Namespace) -> int:
    if args.image_backend == "opencv":
        import cv2

        cv2.setUseOptimized(bool(args.opencv_optimized))
    _, images, category_ids = load_split(args.annotations, args.image_root, args.limit_pages)
    paths = [Path(row["_path"]) for row in images]
    if not paths:
        raise RuntimeError("The requested split has no images")
    pack = ModelPack.load(args.model_pack)
    rss_before = memory_rss()
    load_started = time.perf_counter()
    if args.device == "openvino-native":
        engine = OpenVinoRaw(
            pack,
            device=args.openvino_device,
            inference_precision=args.inference_precision,
            performance_hint=args.performance_hint,
            image_backend=args.image_backend,
            cache_dir=args.openvino_cache_dir,
            threads=args.openvino_threads,
            streams=args.openvino_streams,
            core_type=args.openvino_core_type,
            cpu_pinning=args.openvino_cpu_pinning,
            hyper_threading=args.openvino_hyper_threading,
        )
    else:
        engine = PPDocLite(
            pack,
            device=args.device,
            threads=args.threads,
            inter_threads=args.inter_threads,
            strict_device=True,
            image_backend=args.image_backend,
            graph_optimization=args.graph_optimization,
            execution_mode=args.execution_mode,
            allow_spinning=args.allow_spinning,
            cpu_mem_arena=args.cpu_mem_arena,
            disable_prepacking=args.disable_prepacking,
        )
    load_seconds = time.perf_counter() - load_started
    rss_after_load = memory_rss()

    input_config = pack.manifest.get("input", {})

    def prepare_batch(batch_paths: Sequence[Path]) -> Any:
        if args.input_mode == "paths":
            return list(batch_paths)
        rgb = [load_rgb(path, args.image_backend) for path in batch_paths]
        if args.input_mode == "rgb":
            return rgb
        prepared = [
            prepare_rgb(
                image,
                input_config.get("target_size", [800, 800]),
                backend=args.image_backend,
                scale=float(input_config.get("scale", 1.0 / 255.0)),
                mean=input_config.get("mean", [0.0, 0.0, 0.0]),
                std=input_config.get("std", [1.0, 1.0, 1.0]),
            )
            for image in rgb
        ]
        return (
            {
                name: np.stack([row[0][name] for row in prepared], axis=0)
                for name in ("im_shape", "image", "scale_factor")
            },
            [row[1] for row in prepared],
        )

    def predict(prepared: Any, batch_paths: Sequence[Path]) -> list[dict[str, Any]]:
        options = {
            "threshold": args.threshold,
            "filter_overlap_boxes": args.filter_overlap_boxes,
        }
        if args.input_mode == "paths":
            return engine.infer(prepared, **options)
        identifiers = [str(path) for path in batch_paths]
        if args.input_mode == "rgb":
            return engine.infer_rgb(prepared, image_ids=identifiers, **options)
        feeds, image_sizes = prepared
        return engine.infer_tensors(
            feeds,
            image_sizes,
            image_ids=identifiers,
            **options,
        )

    input_preparation_seconds = 0.0
    preparation_started = time.perf_counter()
    first_input = prepare_batch(paths[:1])
    input_preparation_seconds += time.perf_counter() - preparation_started
    first_started = time.perf_counter()
    predict(first_input, paths[:1])
    first_seconds = time.perf_counter() - first_started
    first_rss = memory_rss()
    peak_rss = memory_rss(peak=True)

    warmup_batch = paths[: min(args.batch_size, len(paths))]
    for index in range(args.warmup_runs):
        predict(prepare_batch(warmup_batch), warmup_batch)
        print(f"[ppdoc-lite benchmark] warmup={index + 1}/{args.warmup_runs}", flush=True)

    batch_times: list[float] = []
    predictions: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    progress_path = args.output.with_suffix(args.output.suffix + ".progress.json")
    for offset in range(0, len(paths), args.batch_size):
        batch_paths = paths[offset : offset + args.batch_size]
        preparation_started = time.perf_counter()
        batch_input = prepare_batch(batch_paths)
        input_preparation_seconds += time.perf_counter() - preparation_started
        started = time.perf_counter()
        results = predict(batch_input, batch_paths)
        elapsed = time.perf_counter() - started
        batch_times.append(elapsed)
        seconds_per_page = elapsed / len(batch_paths)
        for image, result in zip(images[offset : offset + len(batch_paths)], results, strict=True):
            coco_rows = detections_to_coco(result, int(image["id"]), category_ids)
            predictions.extend(coco_rows)
            result_rows.append(
                {
                    "image_id": int(image["id"]),
                    "file_name": image["file_name"],
                    "inference_seconds": seconds_per_page,
                    "detections": result["detections"],
                }
            )
        measured_peak = memory_rss(peak=True)
        if measured_peak is not None:
            peak_rss = max(peak_rss or measured_peak, measured_peak)
        processed = min(offset + len(batch_paths), len(paths))
        write_json(
            progress_path,
            {
                "schema_version": "legalpdf.ppdoc_lite_benchmark_progress.v1",
                "variant_id": pack.variant_id,
                "processed_pages": processed,
                "total_pages": len(paths),
                "elapsed_inference_seconds": sum(batch_times),
                "partial_results": result_rows,
            },
        )
        print(
            f"[ppdoc-lite benchmark] variant={pack.variant_id} pages={processed}/{len(paths)} "
            f"batch_seconds={elapsed:.4f}",
            flush=True,
        )

    total_seconds = sum(batch_times)
    page_times = [value / min(args.batch_size, len(paths) - index) for index, value in zip(range(0, len(paths), args.batch_size), batch_times, strict=True)]
    metrics = score_coco(args.annotations, predictions, [int(row["id"]) for row in images]) if args.score else None
    payload = {
        "schema_version": "legalpdf.ppdoc_lite_benchmark.v1",
        "variant_id": pack.variant_id,
        "model_sha256": pack.manifest["model"]["sha256"],
        "model_bytes": pack.model_bytes,
        "providers": list(engine.providers),
        "device": args.device,
        "openvino_device": args.openvino_device if args.device == "openvino-native" else None,
        "inference_precision": args.inference_precision if args.device == "openvino-native" else None,
        "performance_hint": args.performance_hint if args.device == "openvino-native" else None,
        "openvino_cache_dir": str(args.openvino_cache_dir) if args.openvino_cache_dir else None,
        "openvino_threads": args.openvino_threads if args.device == "openvino-native" else None,
        "openvino_streams": args.openvino_streams if args.device == "openvino-native" else None,
        "openvino_core_type": args.openvino_core_type if args.device == "openvino-native" else None,
        "openvino_cpu_pinning": args.openvino_cpu_pinning if args.device == "openvino-native" else None,
        "openvino_hyper_threading": args.openvino_hyper_threading if args.device == "openvino-native" else None,
        "openvino_runtime_config": engine.runtime_config if args.device == "openvino-native" else None,
        "threads": args.threads,
        "inter_threads": args.inter_threads,
        "batch_size": args.batch_size,
        "image_backend": args.image_backend,
        "opencv_optimized": args.opencv_optimized if args.image_backend == "opencv" else None,
        "input_mode": args.input_mode,
        "graph_optimization": args.graph_optimization,
        "execution_mode": args.execution_mode,
        "allow_spinning": args.allow_spinning,
        "cpu_mem_arena": args.cpu_mem_arena,
        "disable_prepacking": args.disable_prepacking,
        "threshold": args.threshold,
        "filter_overlap_boxes": args.filter_overlap_boxes,
        "page_count": len(paths),
        "session_load_seconds": load_seconds,
        "first_page_seconds_after_load": first_seconds,
        "cold_start_through_first_page_seconds": load_seconds + first_seconds,
        "warm_total_seconds": total_seconds,
        "untimed_input_preparation_seconds": input_preparation_seconds,
        "warm_pages_per_second": len(paths) / total_seconds,
        "warm_ms_per_page_median": statistics.median(page_times) * 1000.0,
        "warm_ms_per_page_p95": percentile(page_times, 95) * 1000.0,
        "rss_before_bytes": rss_before,
        "rss_after_load_bytes": rss_after_load,
        "rss_after_first_page_bytes": first_rss,
        "peak_rss_bytes": peak_rss,
        "heldout_coco_bbox": metrics,
        "predictions": predictions,
        "page_results": result_rows,
    }
    write_json(args.output, payload)
    write_json(progress_path, {"schema_version": "legalpdf.ppdoc_lite_benchmark_progress.v1", "phase": "complete", "output": str(args.output)})
    print(json.dumps({key: value for key, value in payload.items() if key not in {"predictions", "page_results"}}, indent=2))
    return 0


def run_score(args: argparse.Namespace) -> int:
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    if isinstance(predictions, dict):
        predictions = predictions.get("predictions") or []
    gold = json.loads(args.annotations.read_text(encoding="utf-8"))
    image_ids = [int(row["id"]) for row in gold["images"]]
    metrics = score_coco(args.annotations, predictions, image_ids)
    if args.output:
        write_json(args.output, metrics)
    print(json.dumps(metrics, indent=2))
    return 0


def normalize_rust_detections(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "score": float(row["score"]),
            "label_id": int(row["label_id"]),
            "label": str(row["label"]),
            "box": [float(value) for value in row["bbox"]],
            "raw": {"order": row.get("order")},
        }
        for row in rows
    ]


def rust_command(args: argparse.Namespace, image_list: Path) -> list[str]:
    command = [
        str(args.binary),
        "ppdoc-images",
        "--list",
        str(image_list),
        "--model-pack",
        str(args.model_pack),
        "--runtime",
        str(args.runtime),
        "--backend",
        str(args.backend),
        "--threads",
        str(args.threads),
        "--threshold",
        str(args.threshold),
    ]
    if args.device:
        command.extend(("--device", str(args.device)))
    if args.cache_dir:
        command.extend(("--cache-dir", str(args.cache_dir)))
    return command


def run_rust(args: argparse.Namespace) -> int:
    pack = ModelPack.load(args.model_pack)
    _, images, category_ids = load_split(args.annotations, args.image_root, args.limit_pages)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image_list = args.output.with_suffix(args.output.suffix + ".images.txt")
    image_list.write_text(
        "\n".join(str(row["_path"]) for row in images) + "\n", encoding="utf-8"
    )
    command = rust_command(args, image_list)
    variant_id = f"{pack.variant_id}-rust-{args.backend}"
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    page_results: list[dict[str, Any]] = []
    page_seconds: list[float] = []
    identity = None
    assert process.stdout is not None
    for line in process.stdout:
        if len(page_results) >= len(images):
            process.kill()
            raise RuntimeError("Rust PPdoc emitted more receipts than requested pages")
        image = images[len(page_results)]
        receipt = json.loads(line)
        identity = str(receipt["identity"])
        page_seconds.append(float(receipt["seconds"]))
        page_results.append(
            {
                "image_id": int(image["id"]),
                "file_name": str(image["file_name"]),
                "inference_seconds": page_seconds[-1],
                "detections": normalize_rust_detections(receipt["detections"]),
            }
        )
        write_json(
            args.output,
            {
                "schema_version": "legalpdf.ppdoc_lite_benchmark.v1",
                "status": "partial",
                "variant_id": variant_id,
                "page_count": len(page_results),
                "page_results": page_results,
            },
        )
        print(
            f"[ppdoc-lite rust] pages={len(page_results)}/{len(images)} "
            f"page_seconds={page_seconds[-1]:.4f}",
            flush=True,
        )
    assert process.stderr is not None
    stderr = process.stderr.read().strip()
    code = process.wait()
    wall_seconds = time.perf_counter() - started
    if code or len(page_results) != len(images):
        detail = f": {stderr[-4000:]}" if stderr else ""
        raise RuntimeError(
            f"Rust PPdoc failed ({code}) after {len(page_results)}/{len(images)} pages{detail}"
        )
    inference_seconds = sum(page_seconds)
    predictions = [
        prediction
        for page in page_results
        for prediction in detections_to_coco(
            {"detections": page["detections"]}, int(page["image_id"]), category_ids
        )
    ]
    payload = {
        "schema_version": "legalpdf.ppdoc_lite_benchmark.v1",
        "status": "inference_complete",
        "variant_id": variant_id,
        "model_sha256": pack.manifest["model"]["sha256"],
        "model_bytes": pack.model_bytes,
        "identity": identity,
        "backend": args.backend,
        "device": args.device,
        "cache_dir": str(args.cache_dir) if args.cache_dir else None,
        "threads": args.threads,
        "threshold": args.threshold,
        "page_count": len(page_results),
        "process_wall_seconds": wall_seconds,
        "warm_total_seconds": inference_seconds,
        "startup_and_session_overhead_seconds": max(0.0, wall_seconds - inference_seconds),
        "warm_pages_per_second": len(page_results) / max(inference_seconds, 1e-9),
        "warm_ms_per_page_median": statistics.median(page_seconds) * 1000.0,
        "warm_ms_per_page_p95": percentile(page_seconds, 95) * 1000.0,
        "heldout_coco_bbox": None,
        "predictions": predictions,
        "page_results": page_results,
    }
    write_json(args.output, payload)
    metrics = (
        score_coco(args.annotations, predictions, [int(row["id"]) for row in images])
        if args.score
        else None
    )
    payload["status"] = "complete"
    payload["heldout_coco_bbox"] = metrics
    write_json(args.output, payload)
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key not in {"predictions", "page_results"}},
            indent=2,
        )
    )
    return 0


def compare_page_results(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    score_atol: float = 0.0,
    box_atol: float = 0.0,
) -> dict[str, Any]:
    left_pages = {str(row["file_name"]): row for row in left.get("page_results", [])}
    right_pages = {str(row["file_name"]): row for row in right.get("page_results", [])}
    if left_pages.keys() != right_pages.keys():
        raise ValueError("Benchmark results do not contain the same pages")
    count_mismatch_pages = []
    label_mismatches = 0
    order_mismatches = 0
    max_score_error = 0.0
    max_box_error = 0.0
    compared = 0
    for name in sorted(left_pages):
        left_rows = left_pages[name]["detections"]
        right_rows = right_pages[name]["detections"]
        if len(left_rows) != len(right_rows):
            count_mismatch_pages.append(
                {"file_name": name, "left": len(left_rows), "right": len(right_rows)}
            )
        for left_row, right_row in zip(left_rows, right_rows):
            compared += 1
            if left_row["label"] != right_row["label"]:
                label_mismatches += 1
            if left_row.get("raw", {}).get("order") != right_row.get("raw", {}).get("order"):
                order_mismatches += 1
            max_score_error = max(
                max_score_error, abs(float(left_row["score"]) - float(right_row["score"]))
            )
            max_box_error = max(
                max_box_error,
                *(abs(float(a) - float(b)) for a, b in zip(left_row["box"], right_row["box"])),
            )
    return {
        "schema_version": "legalpdf.ppdoc_lite_differential.v1",
        "left_variant_id": left.get("variant_id"),
        "right_variant_id": right.get("variant_id"),
        "pages": len(left_pages),
        "compared_detections": compared,
        "count_mismatch_pages": count_mismatch_pages,
        "label_mismatches": label_mismatches,
        "order_mismatches": order_mismatches,
        "score_atol": score_atol,
        "box_atol": box_atol,
        "max_abs_score_error": max_score_error,
        "max_abs_box_error": max_box_error,
        "exact_detection_contract": (
            not count_mismatch_pages
            and label_mismatches == 0
            and order_mismatches == 0
            and max_score_error == 0.0
            and max_box_error == 0.0
        ),
        "detection_contract_within_tolerance": (
            not count_mismatch_pages
            and label_mismatches == 0
            and order_mismatches == 0
            and max_score_error <= score_atol
            and max_box_error <= box_atol
        ),
    }


def run_compare(args: argparse.Namespace) -> int:
    payload = compare_page_results(
        json.loads(args.left.read_text(encoding="utf-8")),
        json.loads(args.right.read_text(encoding="utf-8")),
        score_atol=args.score_atol,
        box_atol=args.box_atol,
    )
    if args.output:
        write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ppdoc-lite-benchmark", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    environment = subparsers.add_parser("environment")
    environment.add_argument("--distribution", action="append", default=[])
    environment.add_argument("--output", type=Path)
    environment.set_defaults(handler=run_environment)

    benchmark = subparsers.add_parser("run")
    benchmark.add_argument("--model-pack", type=Path, required=True)
    benchmark.add_argument("--annotations", type=Path, required=True)
    benchmark.add_argument("--image-root", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument(
        "--device",
        choices=("cpu", "cuda", "directml", "openvino", "openvino-native", "coreml"),
        default="cpu",
    )
    benchmark.add_argument("--openvino-device", default="GPU")
    benchmark.add_argument(
        "--inference-precision", choices=("auto", "f32", "f16"), default="auto"
    )
    benchmark.add_argument(
        "--performance-hint", choices=("latency", "throughput"), default="latency"
    )
    benchmark.add_argument("--openvino-cache-dir", type=Path)
    benchmark.add_argument("--openvino-threads", type=int, default=0)
    benchmark.add_argument("--openvino-streams", type=int, default=0)
    benchmark.add_argument(
        "--openvino-core-type", choices=("auto", "any", "pcores", "ecores"), default="auto"
    )
    benchmark.add_argument(
        "--openvino-cpu-pinning", action=argparse.BooleanOptionalAction, default=None
    )
    benchmark.add_argument(
        "--openvino-hyper-threading", action=argparse.BooleanOptionalAction, default=None
    )
    benchmark.add_argument("--threads", type=int, default=0)
    benchmark.add_argument("--inter-threads", type=int, default=1)
    benchmark.add_argument("--batch-size", type=int, default=1)
    benchmark.add_argument("--warmup-runs", type=int, default=2)
    benchmark.add_argument("--limit-pages", type=int, default=0)
    benchmark.add_argument("--threshold", type=float, default=0.01)
    benchmark.add_argument("--filter-overlap-boxes", action=argparse.BooleanOptionalAction, default=False)
    benchmark.add_argument("--image-backend", choices=("opencv", "pillow"), default="opencv")
    benchmark.add_argument("--opencv-optimized", action=argparse.BooleanOptionalAction, default=True)
    benchmark.add_argument("--input-mode", choices=("paths", "rgb", "tensors"), default="paths")
    benchmark.add_argument(
        "--graph-optimization",
        choices=("disable", "basic", "extended", "all"),
        default="all",
    )
    benchmark.add_argument("--execution-mode", choices=("sequential", "parallel"), default="sequential")
    benchmark.add_argument("--allow-spinning", action=argparse.BooleanOptionalAction, default=True)
    benchmark.add_argument("--cpu-mem-arena", action=argparse.BooleanOptionalAction, default=True)
    benchmark.add_argument("--disable-prepacking", action=argparse.BooleanOptionalAction, default=False)
    benchmark.add_argument("--score", action=argparse.BooleanOptionalAction, default=True)
    benchmark.set_defaults(handler=run_benchmark)

    score = subparsers.add_parser("score")
    score.add_argument("--annotations", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path)
    score.set_defaults(handler=run_score)

    compare = subparsers.add_parser("compare", help="Compare two benchmark output contracts")
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    compare.add_argument("--output", type=Path)
    compare.add_argument("--score-atol", type=float, default=0.0)
    compare.add_argument("--box-atol", type=float, default=0.0)
    compare.set_defaults(handler=run_compare)

    rust = subparsers.add_parser("rust-run", help="Capture direct Rust runtime output in the benchmark contract")
    rust.add_argument("--binary", type=Path, required=True)
    rust.add_argument("--runtime", "--onnx-runtime", dest="runtime", type=Path, required=True)
    rust.add_argument(
        "--backend",
        choices=("cpu", "cuda", "tensorrt", "directml", "openvino", "onednn"),
        default="cpu",
    )
    rust.add_argument("--device")
    rust.add_argument("--cache-dir", type=Path)
    rust.add_argument("--model-pack", type=Path, required=True)
    rust.add_argument("--annotations", type=Path, required=True)
    rust.add_argument("--image-root", type=Path, required=True)
    rust.add_argument("--output", type=Path, required=True)
    rust.add_argument("--threads", type=int, default=0)
    rust.add_argument("--limit-pages", type=int, default=0)
    rust.add_argument("--threshold", type=float, default=0.01)
    rust.add_argument("--score", action=argparse.BooleanOptionalAction, default=True)
    rust.set_defaults(handler=run_rust)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "batch_size", 1) < 1:
        raise SystemExit("--batch-size must be positive")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
