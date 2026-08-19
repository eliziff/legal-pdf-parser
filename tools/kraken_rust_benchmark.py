#!/usr/bin/env python3
"""Checkpointed CER/WER, layout, order, and speed matrix for Rust Kraken-lite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from rapidfuzz.distance.Levenshtein import distance
except ImportError:
    def distance(left, right):
        if len(left) > len(right):
            left, right = right, left
        row = list(range(len(left) + 1))
        for index, value in enumerate(right, 1):
            previous, row[0] = row[0], index
            for column, candidate in enumerate(left, 1):
                old = row[column]
                row[column] = min(row[column] + 1, row[column - 1] + 1,
                                  previous + (candidate != value))
                previous = old
        return row[-1]


TIERS = ("quality", "balanced", "turbo", "extreme")
BACKENDS = ("cpu", "cuda", "tensorrt", "directml", "openvino", "onednn")


def asset(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def corpus_fingerprint(xmls: list[Path]) -> str:
    digest = hashlib.sha256()
    for index, xml in enumerate(xmls, 1):
        digest.update(index.to_bytes(8, "little"))
        for kind, path in ((b"gold", xml), (b"image", xml.with_suffix(".png"))):
            digest.update(kind)
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        if index % 50 == 0:
            print(f"fingerprint {index}/{len(xmls)}", file=sys.stderr, flush=True)
    return digest.hexdigest()


def normalized(text: str) -> str:
    text = re.sub(r"[\u00ad\u00ac]\r?\n", "", text)
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "").replace("Ã‚Â¬\n", "")
    return re.sub(r"\s+", " ", text.replace("\u00ac", "")).strip()


def truth(path: Path) -> tuple[str, list[dict]]:
    lines = []
    root = ET.parse(path).getroot()
    page = next(node for node in root.iter() if node.tag.endswith("Page"))
    page_area = int(page.get("imageWidth", "0")) * int(page.get("imageHeight", "0"))
    for fallback, line in enumerate(root.iter()):
        if not line.tag.endswith("TextLine"):
            continue
        match = re.search(r"readingOrder\s*\{index:(\d+)", line.get("custom", ""))
        text = next((node.text or "" for node in line.iter() if node.tag.endswith("Unicode")), "")
        coords = next((node.get("points", "") for node in line if node.tag.endswith("Coords")), "")
        points = [tuple(map(float, point.split(","))) for point in coords.split() if "," in point]
        bbox = ([min(x for x, _ in points), min(y for _, y in points),
                 max(x for x, _ in points), max(y for _, y in points)] if points else None)
        if bbox and (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > page_area * .7:
            bbox = None  # Silver page-level transcription, not line geometry.
        if text:
            lines.append({
                "order": int(match.group(1)) if match else 10**9 + fallback,
                "text": normalized(text),
                "bbox": bbox,
            })
    lines.sort(key=lambda line: line["order"])
    return normalized("\n".join(line["text"] for line in lines)), lines


def iou(left, right) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    union = ((left[2] - left[0]) * (left[3] - left[1])
             + (right[2] - right[0]) * (right[3] - right[1]) - intersection)
    return intersection / union if union > 0 else 0.0


def matches(gold: list[dict], predicted: list, threshold: float = .25) -> list[tuple[int, int, float]]:
    candidates = sorted(((iou(line["bbox"], box), gi, pi)
                         for gi, line in enumerate(gold) if line["bbox"]
                         for pi, box in enumerate(predicted)), reverse=True)
    used_gold, used_predicted, output = set(), set(), []
    for overlap, gi, pi in candidates:
        if overlap < threshold:
            break
        if gi not in used_gold and pi not in used_predicted:
            used_gold.add(gi); used_predicted.add(pi)
            output.append((gi, pi, overlap))
    return output


def page_stages(gold: list[dict], receipt: dict) -> dict:
    layout = receipt["layout_boxes"]
    gold_line_count = sum(line["bbox"] is not None for line in gold)
    if not gold_line_count:
        return {
            "geometry_pages": 0, "gold_lines": 0, "layout_lines": 0, "matched_lines": 0,
            "iou_sum": 0, "order_pairs": 0, "concordant_pairs": 0,
            "line_chars": 0, "line_edits": 0,
            "layout_sha256": hashlib.sha256(json.dumps(layout, separators=(",", ":")).encode()).hexdigest(),
            "text_sha256": hashlib.sha256(receipt["text"].encode()).hexdigest(),
        }
    paired = matches(gold, layout)
    sequence = [pi for gi, pi, _ in sorted(paired)]
    order_pairs = len(sequence) * (len(sequence) - 1) // 2
    concordant = sum(sequence[i] < sequence[j]
                     for i in range(len(sequence)) for j in range(i + 1, len(sequence)))
    recognized = receipt["lines"]
    line_pairs = matches(gold, [line["bbox"] for line in recognized])
    line_chars = sum(len(gold[gi]["text"]) for gi, _, _ in line_pairs)
    line_edits = sum(distance(gold[gi]["text"], normalized(recognized[pi]["text"]))
                     for gi, pi, _ in line_pairs)
    return {
        "geometry_pages": 1, "gold_lines": gold_line_count,
        "layout_lines": len(layout), "matched_lines": len(paired),
        "iou_sum": sum(overlap for _, _, overlap in paired),
        "order_pairs": order_pairs, "concordant_pairs": concordant,
        "line_chars": line_chars, "line_edits": line_edits,
        "layout_sha256": hashlib.sha256(json.dumps(layout, separators=(",", ":")).encode()).hexdigest(),
        "text_sha256": hashlib.sha256(receipt["text"].encode()).hexdigest(),
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)] if ordered else 0.0


def summarize(rows: list[dict]) -> dict:
    identities = {row["ocr_identity"] for row in rows}
    engines = {row["ocr_engine"] for row in rows}
    if len(identities) > 1 or len(engines) > 1:
        raise SystemExit("OCR engine identity changed within one benchmark run")
    chars = sum(row["gold_chars"] for row in rows)
    words = sum(row["gold_words"] for row in rows)
    stage = {key: sum(row["stages"][key] for row in rows) for key in (
        "geometry_pages", "gold_lines", "layout_lines", "matched_lines", "iou_sum", "order_pairs",
        "concordant_pairs", "line_chars", "line_edits")}
    seconds = sum(row["seconds"] for row in rows)
    return {
        "cer": sum(row["edits"] for row in rows) / max(1, chars),
        "wer": sum(row["word_edits"] for row in rows) / max(1, words),
        "pages_per_second": len(rows) / max(seconds, 1e-9),
        "seconds": seconds,
        "layout_seconds": sum(row["layout_seconds"] for row in rows),
        "recognition_seconds": sum(row["recognition_seconds"] for row in rows),
        "p50_page_seconds": percentile([row["seconds"] for row in rows], .50),
        "p95_page_seconds": percentile([row["seconds"] for row in rows], .95),
        "geometry_pages": stage["geometry_pages"],
        "layout_precision": (stage["matched_lines"] / stage["layout_lines"] if stage["layout_lines"] else None),
        "layout_recall": (stage["matched_lines"] / stage["gold_lines"] if stage["gold_lines"] else None),
        "mean_line_iou": (stage["iou_sum"] / stage["matched_lines"] if stage["matched_lines"] else None),
        "reading_order_agreement": (stage["concordant_pairs"] / stage["order_pairs"] if stage["order_pairs"] else None),
        "matched_line_cer": (stage["line_edits"] / stage["line_chars"] if stage["line_chars"] else None),
        "ocr_identity": next(iter(identities), None),
        "ocr_engine": next(iter(engines), None),
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_tier(args, tier: str, xmls: list[Path], result: dict) -> None:
    rows = result.get("tiers", {}).get(tier, {}).get("pages", [])
    if len(rows) > len(xmls):
        raise SystemExit(f"checkpoint has {len(rows)} {tier} pages; expected at most {len(xmls)}")
    if len(rows) == len(xmls):
        result["tiers"][tier] = {"status": "complete", "pages": rows, "engine": summarize(rows)}
        return
    offset = len(rows)
    command = [str(args.binary), "_kraken-images", "--list", str(args.list),
               "--kraken-model", str(args.model), "--kraken-codec", str(args.codec),
               "--onnx-runtime", str(args.runtime), "--kraken-layout", args.layout,
               "--kraken-tier", tier, "--kraken-backend", args.backend]
    if args.device:
        command += ["--kraken-device", args.device]
    if args.cpu_fallback:
        command += ["--kraken-cpu-fallback"]
    if args.low_memory:
        command += ["--kraken-low-memory"]
    if args.workers is not None:
        command += ["--kraken-workers", str(args.workers)]
    if args.threads is not None:
        command += ["--kraken-threads", str(args.threads)]
    if args.layout_workers is not None:
        command += ["--kraken-layout-workers", str(args.layout_workers)]
    if args.tesseract_library:
        command += ["--kraken-tesseract-library", str(args.tesseract_library)]
    if args.rgba_input:
        command += ["--rgba-input"]
    if args.limit or offset:
        limited = args.output.with_name(f"{args.output.stem}-{tier}-remaining.lst")
        limited.write_text("\n".join(map(str, xmls[offset:])) + "\n", encoding="utf-8")
        command[3] = str(limited)
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, text=True, encoding="utf-8",
        creationflags=(subprocess.BELOW_NORMAL_PRIORITY_CLASS
                       if os.name == "nt" and args.priority == "below-normal" else 0),
    )
    assert process.stdout is not None
    for index, line in enumerate(process.stdout, offset + 1):
        receipt = json.loads(line)
        gold_text, gold_lines = truth(xmls[index - 1])
        prediction = normalized(receipt["text"])
        stages = page_stages(gold_lines, receipt)
        rows.append({
            "image": receipt["image"], "gold_chars": len(gold_text),
            "gold_words": len(gold_text.split()), "edits": distance(gold_text, prediction),
            "word_edits": distance(gold_text.split(), prediction.split()),
            "seconds": receipt["seconds"], "layout_seconds": receipt["layout_seconds"],
            "recognition_seconds": receipt["recognition_seconds"], "stages": stages,
            "gold": gold_text, "rust_text": prediction,
            "ocr_identity": receipt["ocr_identity"], "ocr_engine": receipt["ocr_engine"],
        })
        result["tiers"][tier] = {"status": "partial", "pages": rows, "engine": summarize(rows)}
        write_json(args.output, result)
        print(f"{tier} {index}/{len(xmls)} {receipt['seconds']:.3f}s", flush=True)
    code = process.wait()
    if code or len(rows) != len(xmls):
        raise SystemExit(f"Rust Kraken {tier} failed ({code}) after {len(rows)}/{len(xmls)} pages")
    result["tiers"][tier] = {"status": "complete", "pages": rows, "engine": summarize(rows)}
    write_json(args.output, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--tesseract-library", type=Path)
    parser.add_argument("--hardware-label",
                        help="stable hardware-class label for cross-machine promotion receipts")
    parser.add_argument("--priority", choices=("normal", "below-normal"),
                        default="below-normal")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=("tesseract",),
                        default="tesseract")
    parser.add_argument("--backend", choices=BACKENDS, default="cpu")
    parser.add_argument("--device",
                        help="numeric CUDA/TensorRT/DirectML id or OpenVINO device string")
    parser.add_argument("--cpu-fallback", action="store_true",
                        help="explicitly permit unsupported accelerator nodes to run on CPU")
    parser.add_argument("--low-memory", action="store_true",
                        help="disable the CPU arena to trade throughput for lower peak RAM")
    parser.add_argument("--tiers", default="quality",
                        help="comma-separated quality,balanced,turbo,extreme or all")
    parser.add_argument("--workers", type=int,
                        help="recognition sessions (default: engine chooses for backend)")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--layout-workers", type=int)
    parser.add_argument("--rgba-input", action="store_true",
                        help="keep decoded pages RGBA until line crops are prepared")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="continue a matching partial output checkpoint")
    args = parser.parse_args()
    if args.tesseract_library is None:
        configured = os.environ.get("LEGALPDF_TESSERACT_LIBRARY")
        if not configured:
            parser.error("--tesseract-library is required for a reproducible receipt")
        args.tesseract_library = Path(configured)
    if args.backend == "cpu" and args.cpu_fallback:
        parser.error("--cpu-fallback requires an accelerator backend")
    if args.device and args.backend in {"cpu", "onednn"}:
        parser.error(f"--device does not apply to the {args.backend} backend")
    tiers = TIERS if args.tiers == "all" else tuple(args.tiers.split(","))
    if not tiers or any(tier not in TIERS for tier in tiers):
        parser.error("--tiers must contain quality,balanced,turbo,extreme or all")
    xmls = [Path(line) for line in args.list.read_text(encoding="utf-8-sig").splitlines() if line]
    if args.offset < 0 or args.offset >= len(xmls):
        parser.error("--offset must select a page in the benchmark")
    xmls = xmls[args.offset:]
    if args.limit:
        xmls = xmls[:args.limit]
    protocol = {"benchmark": asset(args.list), "corpus_sha256": corpus_fingerprint(xmls),
                "pages": len(xmls), "offset": args.offset,
                "tiers": list(tiers), "layout": args.layout, "workers": args.workers,
                "threads": args.threads, "layout_workers": args.layout_workers,
                "rgba_input": args.rgba_input,
                "backend": args.backend, "device": args.device,
                "cpu_fallback": args.cpu_fallback,
                "low_memory": args.low_memory,
                "priority": args.priority,
                "hardware": {"label": args.hardware_label,
                             "system": platform.system(), "release": platform.release(),
                             "machine": platform.machine(),
                             "processor": (platform.processor()
                                           or os.environ.get("PROCESSOR_IDENTIFIER", "")),
                             "logical_cpus": os.cpu_count()},
                "metrics": ["cer", "wer", "layout_precision", "layout_recall",
                            "mean_line_iou", "reading_order_agreement",
                            "matched_line_cer", "pages_per_second"],
                "assets": {"binary": asset(args.binary), "model": asset(args.model),
                           "codec": asset(args.codec), "runtime": asset(args.runtime),
                           "tesseract_library": asset(args.tesseract_library)},
                "normalization": "nfkc-collapse-not-soft-hyphen-v1",
                "warmup": "first page once, unscored",
                "timing": "persistent in-process OCR on already-decoded pixels"}
    if args.resume and args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
        if result.get("protocol") != protocol:
            parser.error("--resume checkpoint protocol does not match this run")
        result["status"] = "partial"
    else:
        result = {"status": "partial", "protocol": protocol, "tiers": {}}
    started = time.perf_counter()
    write_json(args.output, result)
    for tier in tiers:
        run_tier(args, tier, xmls, result)
    result["status"] = "complete"
    result["wall_seconds"] = time.perf_counter() - started
    if len(tiers) > 1:
        result["ladder"] = [{
            "from": left, "to": right,
            "cer_delta": result["tiers"][right]["engine"]["cer"] - result["tiers"][left]["engine"]["cer"],
            "speed_ratio": result["tiers"][right]["engine"]["pages_per_second"]
                           / result["tiers"][left]["engine"]["pages_per_second"],
        } for left, right in zip(tiers, tiers[1:])]
    write_json(args.output, result)
    print(json.dumps({tier: result["tiers"][tier]["engine"] for tier in tiers}, indent=2), flush=True)


if __name__ == "__main__":
    main()
