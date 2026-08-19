"""Bounded laptop CPU benchmark for Paddle's stock PicoDet layout model.

This is experiment-only. It measures the released model in Paddle's native
inference runtime before deciding whether a portable Rust model pack is worth
shipping.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import paddle.inference as paddle_infer
from PIL import Image


LABELS = [
    "paragraph_title",
    "image",
    "text",
    "number",
    "abstract",
    "content",
    "figure_title",
    "formula",
    "table",
    "table_title",
    "reference",
    "doc_title",
    "footnote",
    "header",
    "algorithm",
    "footer",
    "seal",
]
PPDOC23_LABELS = LABELS + [
    "chart_title",
    "chart",
    "formula_number",
    "header_image",
    "footer_image",
    "aside_text",
]
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--ontology", choices=("layout17", "ppdoc23"), default="layout17")
    return parser.parse_args()


def load_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        width, height = rgb.size
        resized = rgb.resize((480, 480), Image.Resampling.BICUBIC)
        tensor = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    tensor = (tensor - MEAN) / STD
    scale = np.asarray([[480.0 / height, 480.0 / width]], dtype=np.float32)
    return tensor[None, ...], scale


def main() -> None:
    args = parse_args()
    labels = PPDOC23_LABELS if args.ontology == "ppdoc23" else LABELS
    model_file = args.model_dir / "inference.json"
    params_file = args.model_dir / "inference.pdiparams"

    preprocess_start = time.perf_counter()
    pages = [load_image(path) for path in args.images]
    preprocess_seconds = time.perf_counter() - preprocess_start

    config = paddle_infer.Config(str(model_file), str(params_file))
    config.disable_glog_info()
    config.set_cpu_math_library_num_threads(args.threads)
    config.enable_mkldnn()
    config.set_mkldnn_cache_capacity(10)
    load_start = time.perf_counter()
    predictor = paddle_infer.create_predictor(config)
    load_seconds = time.perf_counter() - load_start

    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    input_handles = {name: predictor.get_input_handle(name) for name in input_names}
    output_handles = [predictor.get_output_handle(name) for name in output_names]

    def infer(page: tuple[np.ndarray, np.ndarray]) -> list[np.ndarray]:
        image, scale_factor = page
        values = {"image": image, "scale_factor": scale_factor}
        for name, handle in input_handles.items():
            handle.copy_from_cpu(values[name])
        predictor.run()
        return [handle.copy_to_cpu() for handle in output_handles]

    infer(pages[0])
    timings = []
    final_outputs = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        final_outputs = [infer(page) for page in pages]
        timings.append(time.perf_counter() - start)

    counts: Counter[str] = Counter()
    scores = []
    invalid_boxes = 0
    output_shapes = []
    for page_outputs in final_outputs:
        output_shapes.append([list(value.shape) for value in page_outputs])
        boxes = next(
            (value for value in page_outputs if value.ndim == 2 and value.shape[-1] == 6),
            None,
        )
        if boxes is not None:
            for row in boxes:
                scores.append(float(row[1]))
                if row[4] <= row[2] or row[5] <= row[3]:
                    invalid_boxes += 1
                if row[1] < 0.5:
                    continue
                label_id = int(row[0])
                if 0 <= label_id < len(labels):
                    counts[labels[label_id]] += 1

    median = statistics.median(timings)
    quality = None
    if args.annotations:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        gold = COCO(str(args.annotations))
        categories = {row["name"]: int(row["id"]) for row in gold.dataset["categories"]}
        images_by_name = {
            Path(row["file_name"]).name: row for row in gold.dataset["images"]
        }
        predictions = []
        image_ids = []
        for path, page_outputs in zip(args.images, final_outputs, strict=True):
            image_id = int(images_by_name[path.name]["id"])
            image_ids.append(image_id)
            boxes = page_outputs[0]
            for row in boxes:
                label_id = int(row[0])
                if not 0 <= label_id < len(labels):
                    continue
                category_id = categories.get(labels[label_id])
                if category_id is None:
                    continue
                x1, y1, x2, y2 = map(float, row[2:6])
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(row[1]),
                    }
                )
        candidate = gold.loadRes(predictions)
        evaluation = COCOeval(gold, candidate, "bbox")
        evaluation.params.imgIds = image_ids
        evaluation.params.catIds = [categories[label] for label in labels if label in categories]
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
            "matched_categories": [
                label for label in labels if label in categories
            ],
            "bbox_ap": float(evaluation.stats[0]),
            "bbox_ap50": float(evaluation.stats[1]),
            "bbox_ap75": float(evaluation.stats[2]),
            "per_class_ap50": per_class_ap50,
        }
    print(
        json.dumps(
            {
                "model": str(args.model_dir),
                "runtime": "Paddle Inference CPU + oneDNN",
                "threads": args.threads,
                "pages": len(pages),
                "input_names": input_names,
                "output_names": output_names,
                "representative_output_shapes": output_shapes[0],
                "representative_rows": final_outputs[0][0][:10].tolist(),
                "preprocess_seconds_per_page": preprocess_seconds / len(pages),
                "session_load_seconds": load_seconds,
                "wall_seconds": timings,
                "median_seconds_per_page": median / len(pages),
                "median_pages_per_second": len(pages) / median,
                "confidence": {
                    "min": min(scores),
                    "median": statistics.median(scores),
                    "max": max(scores),
                    "at_least_0.5": sum(score >= 0.5 for score in scores),
                    "invalid_boxes": invalid_boxes,
                },
                "detections": dict(counts),
                "heldout_exact_label_coco": quality,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
