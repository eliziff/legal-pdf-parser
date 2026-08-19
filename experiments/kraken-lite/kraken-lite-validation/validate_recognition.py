from __future__ import annotations

import argparse
import json
import runpy
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from kraken.models import load_models
from kraken_lite.geometry import rectify_line
from kraken_lite.recognition import (
    Recognizer,
    _canonical_probabilities,
    prepare_line,
)


helpers = runpy.run_path(str(Path(__file__).with_name("validate_segmentation.py")))
atomic_json = helpers["atomic_json"]
segmentation_from_dict = helpers["segmentation_from_dict"]


def edit_distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for row, left in enumerate(first, 1):
        current = [row]
        for column, right in enumerate(second, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("segmentations", type=Path)
    parser.add_argument("source_model", type=Path)
    parser.add_argument("recognizer_pack", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    models = [model for model in load_models(args.source_model) if "recognition" in model.model_type]
    if len(models) != 1:
        raise RuntimeError(f"Expected one recognition model, found {len(models)}")
    source = models[0].eval().cpu()
    candidate = Recognizer.from_pack(
        str(args.recognizer_pack), device="cpu", intra_threads=args.threads
    )
    input_name = str(candidate.manifest["model"].get("input", "image"))
    output_name = str(candidate.manifest["model"].get("output", "logits"))
    page_output_dir = args.output.parent / f"{args.output.stem}-pages"
    pages = []

    for index, item in enumerate(corpus, 1):
        page_id = item["id"]
        page_output = page_output_dir / f"{page_id}.json"
        if page_output.is_file():
            result = json.loads(page_output.read_text(encoding="utf-8"))
            pages.append(result)
            print(f"[{index}/{len(corpus)}] {page_id} result=cached", flush=True)
            continue

        with Image.open(item["image"]) as opened:
            image = opened.convert("RGB")
        segmentation = segmentation_from_dict(
            json.loads(
                (args.segmentations / f"{page_id}.candidate.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        rectifications = [
            rectify_line(image, line.baseline, line.boundary)
            for line in segmentation.lines
        ]

        started = time.perf_counter()
        exact = 0
        edits = 0
        reference_characters = 0
        max_absolute = 0.0
        absolute_sum = 0.0
        value_count = 0
        for rectification in rectifications:
            prepared = prepare_line(rectification.image, candidate.manifest)
            with torch.inference_mode():
                reference_raw = source.nn(torch.from_numpy(prepared.tensor), None)
            reference_logits = (
                reference_raw[0] if isinstance(reference_raw, tuple) else reference_raw
            ).detach().cpu().numpy()
            candidate_logits = candidate.session.run({input_name: prepared.tensor})[
                output_name
            ]
            difference = np.abs(reference_logits - candidate_logits)
            max_absolute = max(max_absolute, float(difference.max(initial=0.0)))
            absolute_sum += float(difference.sum())
            value_count += difference.size

            reference_probabilities = _canonical_probabilities(
                reference_logits, candidate.manifest
            )
            candidate_probabilities = _canonical_probabilities(
                candidate_logits, candidate.manifest
            )
            reference_result = candidate._decode(
                reference_probabilities, prepared, rectification
            )
            candidate_result = candidate._decode(
                candidate_probabilities, prepared, rectification
            )
            exact += reference_result.text == candidate_result.text
            edits += edit_distance(reference_result.text, candidate_result.text)
            reference_characters += len(reference_result.text)

        result = {
            "page": page_id,
            "lines": len(rectifications),
            "exactLines": exact,
            "referenceCharacters": reference_characters,
            "editDistance": edits,
            "maxAbsoluteLogitDifference": max_absolute,
            "meanAbsoluteLogitDifference": absolute_sum / max(1, value_count),
            "seconds": time.perf_counter() - started,
        }
        atomic_json(page_output, result)
        pages.append(result)
        print(
            f"[{index}/{len(corpus)}] {page_id} lines={len(rectifications)} "
            f"exact={exact} edits={edits} max_abs={max_absolute:.6f} "
            f"seconds={result['seconds']:.2f}",
            flush=True,
        )

    total_lines = sum(page["lines"] for page in pages)
    total_characters = sum(page["referenceCharacters"] for page in pages)
    total_edits = sum(page["editDistance"] for page in pages)
    summary = {
        "pageCount": len(pages),
        "lines": total_lines,
        "exactLines": sum(page["exactLines"] for page in pages),
        "exactLineRate": sum(page["exactLines"] for page in pages)
        / max(1, total_lines),
        "referenceCharacters": total_characters,
        "editDistance": total_edits,
        "characterErrorRate": total_edits / max(1, total_characters),
        "maxAbsoluteLogitDifference": max(
            page["maxAbsoluteLogitDifference"] for page in pages
        ),
        "meanAbsoluteLogitDifference": sum(
            page["meanAbsoluteLogitDifference"] * page["lines"] for page in pages
        )
        / max(1, total_lines),
        "seconds": sum(page["seconds"] for page in pages),
    }
    atomic_json(args.output, {"pages": pages, "summary": summary})
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
