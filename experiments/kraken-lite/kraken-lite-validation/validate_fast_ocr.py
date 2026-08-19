from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from PIL import Image

from kraken_lite.fast_layout import crop_line, detect_line_boxes, order_line_boxes
from kraken_lite.geometry import rectify_line
from kraken_lite.recognition import Recognizer


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


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


def compact(text: str) -> str:
    return "".join(text.split())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("full_segmentations", type=Path)
    parser.add_argument("recognizer_pack", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference-cache", type=Path)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    recognizer = Recognizer.from_pack(
        str(args.recognizer_pack), device="cpu", intra_threads=args.threads
    )
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    page_output_dir = args.output.parent / f"{args.output.stem}-pages"
    pages = []
    for index, item in enumerate(corpus, 1):
        page_output = page_output_dir / f"{item['id']}.json"
        if page_output.is_file():
            result = json.loads(page_output.read_text(encoding="utf-8"))
            pages.append(result)
            print(f"[{index}/{len(corpus)}] {item['id']} result=cached", flush=True)
            continue
        with Image.open(item["image"]) as source:
            image = source.convert("RGB")
        full = json.loads(
            (args.full_segmentations / f"{item['id']}.candidate.json").read_text(
                encoding="utf-8"
            )
        )
        reference_path = (
            args.reference_cache / f"{item['id']}.json"
            if args.reference_cache is not None
            else None
        )
        if reference_path is not None and reference_path.is_file():
            reference_texts = json.loads(
                reference_path.read_text(encoding="utf-8")
            )["texts"]
            full_seconds = 0.0
        else:
            started = time.perf_counter()
            full_rectifications = [
                rectify_line(
                    image,
                    [tuple(point) for point in line["baseline"]],
                    [tuple(point) for point in line["boundary"]],
                )
                for line in full["lines"]
            ]
            reference_texts = [
                result.text for result in recognizer.recognize_many(full_rectifications)
            ]
            full_seconds = time.perf_counter() - started

        started = time.perf_counter()
        boxes = order_line_boxes(detect_line_boxes(image))
        layout_seconds = time.perf_counter() - started
        fast_rectifications = [crop_line(image, box) for box in boxes]
        fast_results = recognizer.recognize_many(fast_rectifications)
        fast_seconds = time.perf_counter() - started

        full_text = "\n".join(reference_texts)
        fast_text = "\n".join(result.text for result in fast_results)
        reference = compact(full_text)
        candidate = compact(fast_text)
        edits = edit_distance(reference, candidate)
        result = {
            "page": item["id"],
            "fullLines": len(reference_texts),
            "fastLines": len(fast_results),
            "referenceCharacters": len(reference),
            "fastCharacters": len(candidate),
            "editDistance": edits,
            "differentialCharacterErrorRate": edits / max(1, len(reference)),
            "fullSeconds": full_seconds,
            "layoutSeconds": layout_seconds,
            "fastSeconds": fast_seconds,
            "fullText": full_text,
            "fastText": fast_text,
        }
        atomic_json(page_output, result)
        pages.append(result)
        print(
            f"[{index}/{len(corpus)}] {item['id']} full={len(reference_texts)} "
            f"fast={len(fast_results)} cer={result['differentialCharacterErrorRate']:.4f} "
            f"seconds={fast_seconds:.2f}",
            flush=True,
        )

    reference_characters = sum(page["referenceCharacters"] for page in pages)
    summary = {
        "pageCount": len(pages),
        "fullLines": sum(page["fullLines"] for page in pages),
        "fastLines": sum(page["fastLines"] for page in pages),
        "referenceCharacters": reference_characters,
        "fastCharacters": sum(page["fastCharacters"] for page in pages),
        "editDistance": sum(page["editDistance"] for page in pages),
        "differentialCharacterErrorRate": sum(page["editDistance"] for page in pages)
        / max(1, reference_characters),
        "fullSeconds": sum(page["fullSeconds"] for page in pages),
        "layoutSeconds": sum(page["layoutSeconds"] for page in pages),
        "fastSeconds": sum(page["fastSeconds"] for page in pages),
    }
    atomic_json(args.output, {"pages": pages, "summary": summary})
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
