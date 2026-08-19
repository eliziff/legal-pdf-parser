from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from PIL import Image

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("segmentations", type=Path)
    parser.add_argument("reference_pack", type=Path)
    parser.add_argument("candidate_pack", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    reference = Recognizer.from_pack(
        str(args.reference_pack), device="cpu", intra_threads=args.threads
    )
    candidate = Recognizer.from_pack(
        str(args.candidate_pack), device="cpu", intra_threads=args.threads
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
        segmentation = json.loads(
            (args.segmentations / f"{item['id']}.candidate.json").read_text(
                encoding="utf-8"
            )
        )
        rectifications = [
            rectify_line(
                image,
                [tuple(point) for point in line["baseline"]],
                [tuple(point) for point in line["boundary"]],
            )
            for line in segmentation["lines"]
        ]

        reference_path = args.reference_cache / f"{item['id']}.json"
        if reference_path.is_file():
            reference_texts = json.loads(reference_path.read_text(encoding="utf-8"))[
                "texts"
            ]
            reference_state = "cached"
        else:
            reference_texts = [
                result.text for result in reference.recognize_many(rectifications)
            ]
            atomic_json(reference_path, {"texts": reference_texts})
            reference_state = "created"

        started = time.perf_counter()
        candidate_texts = [
            result.text for result in candidate.recognize_many(rectifications)
        ]
        seconds = time.perf_counter() - started
        edits = sum(
            edit_distance(expected, actual)
            for expected, actual in zip(reference_texts, candidate_texts, strict=True)
        )
        characters = sum(len(text) for text in reference_texts)
        result = {
            "page": item["id"],
            "lines": len(reference_texts),
            "exactLines": sum(
                expected == actual
                for expected, actual in zip(reference_texts, candidate_texts, strict=True)
            ),
            "referenceCharacters": characters,
            "editDistance": edits,
            "characterErrorRate": edits / max(1, characters),
            "candidateSeconds": seconds,
            "reference": reference_state,
        }
        atomic_json(page_output, result)
        pages.append(result)
        print(
            f"[{index}/{len(corpus)}] {item['id']} exact={result['exactLines']}/"
            f"{result['lines']} edits={edits} seconds={seconds:.2f} "
            f"reference={reference_state}",
            flush=True,
        )

    lines = sum(page["lines"] for page in pages)
    characters = sum(page["referenceCharacters"] for page in pages)
    edits = sum(page["editDistance"] for page in pages)
    summary = {
        "pageCount": len(pages),
        "lines": lines,
        "exactLines": sum(page["exactLines"] for page in pages),
        "exactLineRate": sum(page["exactLines"] for page in pages) / max(1, lines),
        "referenceCharacters": characters,
        "editDistance": edits,
        "characterErrorRate": edits / max(1, characters),
        "candidateSeconds": sum(page["candidateSeconds"] for page in pages),
    }
    atomic_json(args.output, {"pages": pages, "summary": summary})
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
