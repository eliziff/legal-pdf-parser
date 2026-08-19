from __future__ import annotations

import argparse
import json
import runpy
import time
from pathlib import Path

import numpy as np
from PIL import Image

from kraken_lite.fast_layout import detect_line_boxes
from kraken_lite.geometry import reading_order
from kraken_lite.parity import compare_segmentations
from kraken_lite.types import BaselineLine, Segmentation


helpers = runpy.run_path(str(Path(__file__).with_name("validate_segmentation.py")))
atomic_json = helpers["atomic_json"]
segmentation_from_dict = helpers["segmentation_from_dict"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    pages = []
    page_output_dir = args.output.parent / f"{args.output.stem}-pages"
    for index, item in enumerate(corpus, 1):
        page_id = item["id"]
        with Image.open(item["image"]) as source:
            image = source.convert("RGB")
        reference = segmentation_from_dict(
            json.loads((args.cache / f"{page_id}.json").read_text(encoding="utf-8"))
        )

        started = time.perf_counter()
        lines = []
        for line_index, box in enumerate(detect_line_boxes(image), 1):
            baseline_y = box.bottom - max(1, round(box.height * 0.25))
            lines.append(
                BaselineLine(
                    id=f"line-{line_index:06d}",
                    type="default",
                    baseline=[
                        (float(box.left), float(baseline_y)),
                        (float(box.right - 1), float(baseline_y)),
                    ],
                    boundary=[
                        (float(box.left), float(box.top)),
                        (float(box.right - 1), float(box.top)),
                        (float(box.right - 1), float(box.bottom - 1)),
                        (float(box.left), float(box.bottom - 1)),
                        (float(box.left), float(box.top)),
                    ],
                )
            )
        order = reading_order(lines, text_direction="lr")
        candidate = Segmentation(
            width=image.width,
            height=image.height,
            lines=[lines[line_index] for line_index in order],
            regions={},
            text_direction="horizontal-lr",
            model={"id": "opencv-fast-layout"},
        )
        seconds = time.perf_counter() - started
        parity = compare_segmentations(reference, candidate, image_name=page_id)
        result = parity.to_dict()
        result["candidateSeconds"] = seconds
        pages.append(result)
        atomic_json(page_output_dir / f"{page_id}.candidate.json", candidate.to_dict())
        atomic_json(page_output_dir / f"{page_id}.json", result)
        print(
            f"[{index}/{len(corpus)}] {page_id} lines={len(lines)} "
            f"recall={parity.recall:.4f} iou={parity.mean_polygon_iou:.4f} "
            f"seconds={seconds:.3f}",
            flush=True,
        )

    summary = {
        "pageCount": len(pages),
        "precision": float(np.mean([page["precision"] for page in pages])),
        "recall": float(np.mean([page["recall"] for page in pages])),
        "meanBaselineDistance": float(
            np.mean([page["meanBaselineDistance"] for page in pages])
        ),
        "meanPolygonIoU": float(np.mean([page["meanPolygonIoU"] for page in pages])),
        "readingOrderAgreement": float(
            np.mean([page["readingOrderAgreement"] for page in pages])
        ),
        "candidateSeconds": float(sum(page["candidateSeconds"] for page in pages)),
    }
    atomic_json(args.output, {"pages": pages, "summary": summary})
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
