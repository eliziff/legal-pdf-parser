"""Bounded CPU benchmark for the released Docling Heron ONNX detector.

This is experiment-only. Production inference remains native Rust.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


LABELS = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
    "Document Index",
    "Code",
    "Checkbox-Selected",
    "Checkbox-Unselected",
    "Form",
    "Key-Value Region",
]
LEGAL_LABELS = {
    "Caption": "figure_title",
    "Footnote": "footnote",
    "Formula": "display_formula",
    "Page-footer": "footer",
    "Page-header": "header",
    "Picture": "image",
    "Section-header": "paragraph_title",
    "Table": "table",
    "Text": "text",
    "Title": "doc_title",
    "Document Index": "content",
    "Code": "algorithm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batches", default="1,2,4")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--annotations", type=Path)
    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        resized = rgb.resize((640, 640), Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0


def batches(values: list[np.ndarray], size: int):
    for start in range(0, len(values), size):
        yield np.stack(values[start : start + size])


def detection_counts(logits: np.ndarray, threshold: float) -> dict[str, int]:
    counts = {label: 0 for label in LABELS}
    for page in logits:
        scores = 1.0 / (1.0 + np.exp(-page.reshape(-1)))
        top_k = min(300, scores.size)
        indices = np.argpartition(scores, -top_k)[-top_k:]
        for index in indices[scores[indices] > threshold]:
            counts[LABELS[int(index) % len(LABELS)]] += 1
    return {label: count for label, count in counts.items() if count}


def main() -> None:
    args = parse_args()
    preprocess_start = time.perf_counter()
    images = [load_image(path) for path in args.images]
    preprocess_seconds = time.perf_counter() - preprocess_start

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    load_start = time.perf_counter()
    session = ort.InferenceSession(
        str(args.model),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    load_seconds = time.perf_counter() - load_start

    session.run(["logits", "pred_boxes"], {"image": images[0][None, ...]})
    results = []
    first_logits = None
    quality_logits = None
    quality_boxes = None
    for batch_size in [int(value) for value in args.batches.split(",")]:
        timings = []
        final_logits = []
        final_boxes = []
        for _ in range(args.repeats):
            run_logits = []
            run_boxes = []
            start = time.perf_counter()
            for tensor in batches(images, batch_size):
                logits, boxes = session.run(["logits", "pred_boxes"], {"image": tensor})
                run_logits.append(logits)
                run_boxes.append(boxes)
            timings.append(time.perf_counter() - start)
            final_logits = run_logits
            final_boxes = run_boxes
        logits = np.concatenate(final_logits, axis=0)
        boxes = np.concatenate(final_boxes, axis=0)
        if batch_size == 1:
            first_logits = logits
            quality_logits = logits
            quality_boxes = boxes
        results.append(
            {
                "batch_size": batch_size,
                "wall_seconds": timings,
                "median_seconds_per_page": statistics.median(timings) / len(images),
                "median_pages_per_second": len(images) / statistics.median(timings),
                "detections": detection_counts(logits, args.threshold),
                "max_abs_logit_difference_from_batch_1": (
                    None
                    if first_logits is None
                    else float(np.max(np.abs(logits - first_logits)))
                ),
            }
        )

    quality = None
    if args.annotations:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        assert quality_logits is not None and quality_boxes is not None
        gold = COCO(str(args.annotations))
        categories = {row["name"]: int(row["id"]) for row in gold.dataset["categories"]}
        images_by_name = {
            Path(row["file_name"]).name: row for row in gold.dataset["images"]
        }
        predictions = []
        image_ids = []
        for path, page_logits, page_boxes in zip(
            args.images, quality_logits, quality_boxes, strict=True
        ):
            image = images_by_name[path.name]
            image_id = int(image["id"])
            width, height = float(image["width"]), float(image["height"])
            image_ids.append(image_id)
            scores = 1.0 / (1.0 + np.exp(-page_logits.reshape(-1)))
            top_k = min(300, scores.size)
            indices = np.argpartition(scores, -top_k)[-top_k:]
            for index in indices:
                query = int(index) // len(LABELS)
                label = LABELS[int(index) % len(LABELS)]
                legal_label = LEGAL_LABELS.get(label)
                category_id = categories.get(legal_label) if legal_label else None
                if category_id is None:
                    continue
                center_x, center_y, box_width, box_height = map(float, page_boxes[query])
                x1 = max(0.0, (center_x - box_width / 2.0) * width)
                y1 = max(0.0, (center_y - box_height / 2.0) * height)
                x2 = min(width, (center_x + box_width / 2.0) * width)
                y2 = min(height, (center_y + box_height / 2.0) * height)
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(scores[index]),
                    }
                )
        candidate = gold.loadRes(predictions)
        evaluation = COCOeval(gold, candidate, "bbox")
        evaluation.params.imgIds = image_ids
        evaluation.params.catIds = [
            categories[label] for label in LEGAL_LABELS.values() if label in categories
        ]
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
        precision = evaluation.eval["precision"]
        per_class_ap50 = {}
        for category_index, category_id in enumerate(evaluation.params.catIds):
            values = precision[0, :, category_index, 0, 2]
            values = values[values > -1]
            name = next(name for name, value in categories.items() if value == category_id)
            per_class_ap50[name] = float(values.mean()) if values.size else None
        quality = {
            "mapped_categories": list(LEGAL_LABELS.values()),
            "bbox_ap": float(evaluation.stats[0]),
            "bbox_ap50": float(evaluation.stats[1]),
            "bbox_ap75": float(evaluation.stats[2]),
            "per_class_ap50": per_class_ap50,
        }

    print(
        json.dumps(
            {
                "model": str(args.model),
                "provider": session.get_providers(),
                "threads": args.threads,
                "pages": len(images),
                "preprocess_seconds_per_page": preprocess_seconds / len(images),
                "session_load_seconds": load_seconds,
                "results": results,
                "heldout_mapped_coco": quality,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
