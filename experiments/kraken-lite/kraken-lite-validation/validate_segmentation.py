from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from kraken_lite.blla import BLLASegmenter
from kraken_lite.parity import _reference_segmentation, compare_segmentations
from kraken_lite.types import BaselineLine, Region, Segmentation


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def segmentation_from_dict(value: dict) -> Segmentation:
    return Segmentation(
        width=int(value["width"]),
        height=int(value["height"]),
        text_direction=value["textDirection"],
        lines=[
            BaselineLine(
                id=line["id"],
                type=line.get("type", "default"),
                baseline=[tuple(point) for point in line["baseline"]],
                boundary=[tuple(point) for point in line["boundary"]],
                regions=list(line.get("regions", [])),
                tags=dict(line.get("tags", {})),
            )
            for line in value["lines"]
        ],
        regions={
            kind: [
                Region(
                    id=region["id"],
                    type=region.get("type", kind),
                    boundary=[tuple(point) for point in region["boundary"]],
                    tags=dict(region.get("tags", {})),
                )
                for region in regions
            ]
            for kind, regions in value.get("regions", {}).items()
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("blla_pack", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    segmenter = BLLASegmenter.from_pack(
        str(args.blla_pack), device="cpu", intra_threads=args.threads
    )
    page_output_dir = args.output.parent / f"{args.output.stem}-pages"
    pages = []
    for index, item in enumerate(corpus, 1):
        page_id = item["id"]
        image_path = Path(item["image"])
        cache_path = args.cache / f"{page_id}.json"
        page_output = page_output_dir / f"{page_id}.json"
        candidate_output = page_output_dir / f"{page_id}.candidate.json"
        if page_output.is_file() and candidate_output.is_file():
            result = json.loads(page_output.read_text(encoding="utf-8"))
            pages.append(result)
            print(f"[{index}/{len(corpus)}] {page_id} result=cached", flush=True)
            continue
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        reference_started = time.perf_counter()
        if cache_path.is_file():
            reference = segmentation_from_dict(
                json.loads(cache_path.read_text(encoding="utf-8"))
            )
            reference_state = "cached"
        else:
            reference = _reference_segmentation(image)
            atomic_json(cache_path, reference.to_dict())
            reference_state = "created"
        reference_seconds = time.perf_counter() - reference_started
        print(
            f"[{index}/{len(corpus)}] {page_id} reference={reference_state} "
            f"seconds={reference_seconds:.2f}",
            flush=True,
        )

        candidate_started = time.perf_counter()
        candidate = segmenter.segment(image, batch_size=1)
        candidate_seconds = time.perf_counter() - candidate_started
        atomic_json(candidate_output, candidate.to_dict())
        parity = compare_segmentations(reference, candidate, image_name=page_id)
        result = parity.to_dict()
        result["referenceSeconds"] = reference_seconds
        result["candidateSeconds"] = candidate_seconds
        result["seconds"] = reference_seconds + candidate_seconds
        result["reference"] = reference_state
        pages.append(result)
        atomic_json(page_output, result)
        print(
            f"[{index}/{len(corpus)}] {page_id} candidate "
            f"recall={parity.recall:.4f} iou={parity.mean_polygon_iou:.4f} "
            f"seconds={candidate_seconds:.2f}",
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
        "seconds": float(sum(page["seconds"] for page in pages)),
        "candidateSeconds": float(sum(page["candidateSeconds"] for page in pages)),
    }
    atomic_json(args.output, {"pages": pages, "summary": summary})
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
