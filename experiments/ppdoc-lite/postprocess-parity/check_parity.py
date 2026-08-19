#!/usr/bin/env python
"""Byte-for-byte differential for the Python and Rust PPDoc postprocessors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = Path(__file__).with_name("Cargo.toml")
EXPECTED_SOURCE_REVISION = "d8b25257687b3b9aad644dec42cca966b45675ff"
SOURCE_FILES = (
    "tools/ocr/layout_regioning/ppdoc/region_postprocess.py",
    "tools/escript/quote_goldset_escriptorium/escriptorium_apply_ppdoc_regions.py",
    "tools/ocr/layout_regioning/ppdoc/postprocess_ppdoc_detection_json.py",
)


class FakeLine:
    def __init__(self, line_id: str, pk: int, bbox: list[float], text: str):
        self.line_id = line_id
        self.pk = pk
        self.order = pk
        x0, y0, x1, y1 = bbox
        self.mask = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        self.text = text


def production_args() -> SimpleNamespace:
    return SimpleNamespace(
        block_quote_heuristic=True,
        block_quote_threshold=0.68,
        block_quote_min_lines=2,
        ppdoc_hard_validity_rules=True,
        ppdoc_repeat_headers_footers=True,
        ppdoc_filter_byline_edge_pages=True,
        ppdoc_byline_first_pages=3,
        ppdoc_edge_digits_to_number=True,
        ppdoc_nonnumeric_edge_numbers_to_header_footer=False,
        ppdoc_sequenced_edge_digits_to_number=True,
        ppdoc_roman_title_heuristic=True,
        ppdoc_footnote_sandwich=True,
        ppdoc_top_footnotes_to_text=True,
        ppdoc_first_page_abstract_to_abstract=False,
        ppdoc_first_page_title_byline_stack=False,
        ppdoc_full_width_block_quotes_to_text=True,
        ppdoc_footnote_tail_nonfootnotes_to_footnote=False,
        ppdoc_footnote_boundary_low_uncertainty_text_to_footnote=False,
        ppdoc_drop_overlap_regions=True,
        ppdoc_overlap_threshold=0.35,
        ppdoc_overlap_require_loser_coverage=True,
    )


def region(label: str, raw_index: int, bbox: list[float], score: float = 0.9) -> dict[str, Any]:
    return {
        "label": label,
        "score": score,
        "bbox": bbox,
        "order": raw_index,
        "raw_index": raw_index,
    }


def lines_for_regions(regions: list[dict[str, Any]], texts: dict[int, list[str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in regions:
        for offset, text in enumerate(texts.get(item["raw_index"], [])):
            x0, y0, x1, y1 = item["bbox"]
            line_y0 = y0 + 8 + offset * 16
            output.append(
                {
                    "line_id": f"l{len(output) + 1}",
                    "text": text,
                    "bbox": [x0 + 5, line_y0, x1 - 5, min(y1 - 2, line_y0 + 10)],
                }
            )
    return output


def page(page_number: int, regions: list[dict[str, Any]], texts: dict[int, list[str]]) -> dict[str, Any]:
    return {
        "page_number": page_number,
        "width": 1000.0,
        "height": 1000.0,
        "lines": lines_for_regions(regions, texts),
        "regions": regions,
    }


def targeted_cases() -> list[dict[str, Any]]:
    validity = [
        region("number", 1, [100, 450, 900, 490]),
        region("number", 2, [100, 50, 900, 80]),
        region("number", 3, [20, 50, 80, 80]),
        region("doc_title", 4, [100, 120, 900, 170]),
        region("abstract", 5, [100, 180, 900, 240]),
        region("block_quote", 6, [100, 300, 900, 360]),
        region("block_quote", 7, [100, 380, 900, 420]),
        region("text", 8, [100, 500, 900, 550]),
    ]
    validity_pages = [page(index, [], {}) for index in range(1, 4)]
    validity_pages.append(
        page(
            4,
            validity,
            {
                1: ["123"],
                2: ["12A"],
                3: ["42"],
                4: ["Later running title"],
                5: ["Late abstract"],
                6: ["Short quote line 1", "Short quote line 2"],
                7: ["PART TWO"],
                8: ["II. Background"],
            },
        )
    )
    validity_pages.extend(page(index, [], {}) for index in range(5, 7))

    repeats = []
    for index in range(1, 4):
        items = [
            region("text", 1, [100, 70, 900, 110]),
            region("text", 2, [50, 340, 120, 370]),
            region("text", 3, [700, 340, 760, 370]),
        ]
        repeats.append(
            page(
                index,
                items,
                {
                    1: [f"{722 + index} Alberta Law Review"],
                    2: [str(722 + index)],
                    3: ["99"],
                },
            )
        )

    inset = [
        region("text", 1, [100, 100, 900, 160]),
        region("text", 2, [100, 200, 900, 260]),
        region("text", 3, [135, 300, 860, 368]),
        region("text", 4, [100, 600, 900, 660]),
    ]
    full_width = [
        region("text", 1, [100, 100, 900, 160]),
        region("text", 2, [100, 200, 900, 260]),
        region("block_quote", 3, [100, 300, 900, 368]),
        region("text", 4, [100, 600, 900, 660]),
    ]
    quote_text = {
        1: ["Body paragraph before the candidate runs full measure."],
        2: ["Another full-measure body paragraph sits here."],
        3: ["candidate line 1", "candidate line 2", "candidate line 3"],
        4: ["Body paragraph after the candidate runs full measure."],
    }
    footnotes = [
        region("footnote", 1, [100, 80, 900, 120]),
        region("footnote", 2, [100, 620, 900, 650]),
        region("footnote", 3, [100, 660, 900, 690]),
        region("text", 4, [100, 700, 900, 730]),
        region("footer", 5, [100, 830, 900, 850]),
        region("footnote", 6, [100, 880, 900, 910]),
    ]
    overlap = [
        region("text", 1, [100, 600, 900, 760]),
        region("footnote", 2, [100, 620, 900, 740]),
    ]
    return [
        {"name": "validity_and_headings", "pages": validity_pages},
        {"name": "repeat_and_sequence", "pages": repeats},
        {"name": "three_line_inset", "pages": [page(1, inset, quote_text)]},
        {
            "name": "full_width_quote",
            "pages": [page(1, [], {}), page(2, [], {}), page(3, full_width, quote_text)],
        },
        {
            "name": "footnote_sandwich",
            "pages": [
                page(
                    1,
                    footnotes,
                    {
                        1: ["not really a note"],
                        2: ["1 First note"],
                        3: ["2 Second note"],
                        4: ["continued note text"],
                        5: ["Journal footer"],
                        6: ["3 Third note"],
                    },
                )
            ],
        },
        {"name": "overlap_priority", "pages": [page(1, overlap, {1: ["body text"], 2: ["1 Footnote text"]})]},
    ]


def random_cases(count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    labels = [
        "text",
        "content",
        "block_quote",
        "paragraph_title",
        "doc_title",
        "abstract",
        "byline",
        "footnote",
        "vision_footnote",
        "reference",
        "reference_content",
        "header",
        "footer",
        "number",
        "formula_number",
        "image",
        "table",
    ]
    texts = [
        "Ordinary legal prose continues across this region.",
        "II. Background and Procedural History",
        "II. the court then turned to whether the parties intended.",
        "PART TWO",
        "42",
        "12A",
        "Alberta Law Review",
        "1 First footnote text",
        "continued note text",
        "(a) a listed proposition",
        "By Jane Smith",
    ]
    cases = []
    for case_index in range(count):
        page_count = rng.randint(1, 6)
        pages = []
        repeated = f"Volume {40 + case_index % 8} Alberta Law Review"
        for page_number in range(1, page_count + 1):
            regions = []
            region_texts: dict[int, list[str]] = {}
            raw_index = 1
            if page_count >= 3:
                item = region("text", raw_index, [80, 45, 920, 85], rng.choice([0.2, 0.9]))
                regions.append(item)
                region_texts[raw_index] = [repeated]
                raw_index += 1
                item = region("text", raw_index, [450, 925, 550, 965], 0.9)
                regions.append(item)
                region_texts[raw_index] = [str(700 + page_number)]
                raw_index += 1
            for _ in range(rng.randint(4, 18)):
                column = rng.randint(0, 1)
                base_x = 60 if column == 0 else 530
                x0 = base_x + rng.randint(0, 100)
                width = rng.randint(160, 410)
                x1 = min(970, x0 + width)
                y0 = rng.randint(35, 900)
                height = rng.randint(32, 150)
                y1 = min(990, y0 + height)
                label = rng.choice(labels)
                regions.append(region(label, raw_index, [x0, y0, x1, y1], rng.choice([0.2, 0.25, 0.55, 0.9])))
                line_count = rng.randint(0, min(4, max(0, int((y1 - y0 - 12) // 16))))
                if line_count:
                    region_texts[raw_index] = [rng.choice(texts) for _ in range(line_count)]
                raw_index += 1
            pages.append(page(page_number, regions, region_texts))
        cases.append({"name": f"random_{case_index:03d}", "pages": pages})
    return cases


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def verify_source(source_root: Path) -> str:
    git_root = next(
        (candidate for candidate in (source_root, *source_root.parents) if (candidate / ".git").exists()),
        source_root,
    )
    revision = subprocess.run(
        ["git", "-c", f"safe.directory={git_root}", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    if revision != EXPECTED_SOURCE_REVISION:
        raise RuntimeError(f"expected Text-Fidelity revision {EXPECTED_SOURCE_REVISION}, found {revision}")
    clean = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={git_root}",
            "-C",
            str(source_root),
            "diff",
            "--quiet",
            "--",
            *SOURCE_FILES,
        ],
        check=False,
    )
    if clean.returncode:
        raise RuntimeError("Text-Fidelity PPDoc source files have uncommitted changes")
    return revision


def python_oracle(cases: list[dict[str, Any]], source_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sys.path.insert(0, str(source_root))
    from tools.ocr.layout_regioning.ppdoc.region_postprocess import (
        DetectionRegion,
        PageRegionPlan,
        best_region_for_line,
        block_quote_heuristic_regions,
        ordered_regions,
        ppdoc_postprocess_region_plans,
        rect_polygon,
    )

    args = production_args()
    normalized_cases = []
    outputs = []
    for case in cases:
        normalized_pages = []
        plans = []
        fake_lines_by_page = []
        page_count = len(case["pages"])
        for input_page in case["pages"]:
            regions = [
                DetectionRegion(
                    label=item["label"],
                    source_label=item["label"],
                    score=item["score"],
                    box=rect_polygon(dict(zip(("x0", "y0", "x1", "y1"), item["bbox"]))),
                    bbox=dict(zip(("x0", "y0", "x1", "y1"), item["bbox"])),
                    order=item["order"],
                    raw_index=item["raw_index"],
                )
                for item in input_page["regions"]
            ]
            regions = ordered_regions(regions, int(input_page["width"]), int(input_page["height"]))
            normalized_page = {
                **input_page,
                "regions": [
                    {
                        "label": item.label,
                        "score": item.score,
                        "bbox": [item.bbox[key] for key in ("x0", "y0", "x1", "y1")],
                        "order": item.order,
                        "raw_index": item.raw_index,
                    }
                    for item in regions
                ],
            }
            normalized_pages.append(normalized_page)
            fake_lines = [
                FakeLine(line["line_id"], index + 1, line["bbox"], line["text"])
                for index, line in enumerate(input_page["lines"])
            ]
            regions, _, _ = block_quote_heuristic_regions(
                regions,
                fake_lines,
                int(input_page["width"]),
                int(input_page["height"]),
                args,
            )
            plans.append(
                PageRegionPlan(
                    row={},
                    summary={},
                    part=None,
                    regions=regions,
                    skipped=[],
                    lines=fake_lines,
                    line_text_by_id={line.pk: line.text for line in fake_lines},
                    width=int(input_page["width"]),
                    height=int(input_page["height"]),
                    page_index=input_page["page_number"],
                    page_count=page_count,
                )
            )
            fake_lines_by_page.append(fake_lines)
        ppdoc_postprocess_region_plans(plans, args)
        page_outputs = []
        for input_page, plan, fake_lines in zip(normalized_pages, plans, fake_lines_by_page):
            blocks = [SimpleNamespace(pk=item.raw_index) for item in plan.regions]
            region_rows = list(zip(blocks, plan.regions))
            assignments = []
            for line_input, fake_line in zip(input_page["lines"], fake_lines):
                block = best_region_for_line(fake_line, region_rows)
                assigned = None if block is None else next(item for item in plan.regions if item.raw_index == block.pk)
                assignments.append(
                    {
                        "line_id": line_input["line_id"],
                        "label": None if assigned is None else assigned.label,
                        "raw_index": None if assigned is None else assigned.raw_index,
                    }
                )
            page_outputs.append(
                {
                    "page_number": input_page["page_number"],
                    "regions": [
                        {
                            "label": item.label,
                            "score": item.score,
                            "bbox": [float(item.bbox[key]) for key in ("x0", "y0", "x1", "y1")],
                            "order": item.order,
                            "raw_index": item.raw_index,
                        }
                        for item in plan.regions
                    ],
                    "assignments": assignments,
                }
            )
        normalized_cases.append({"name": case["name"], "pages": case["pages"]})
        outputs.append({"name": case["name"], "pages": page_outputs})
    return {"cases": normalized_cases}, {"cases": outputs}


def first_difference(left: Any, right: Any, path: str = "$") -> str:
    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if list(left) != list(right):
            return f"{path}: keys {list(left)} != {list(right)}"
        for key in left:
            difference = first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return ""
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = first_difference(left_item, right_item, f"{path}[{index}]")
            if difference:
                return difference
        return ""
    return "" if left == right else f"{path}: {left!r} != {right!r}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-fidelity-root", type=Path, required=True)
    parser.add_argument("--random-cases", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    source_root = args.text_fidelity_root.resolve()
    source_revision = verify_source(source_root)
    cases = targeted_cases() + random_cases(args.random_cases, args.seed)
    normalized_input, expected = python_oracle(cases, source_root)
    input_bytes = canonical_bytes(normalized_input)
    expected_bytes = canonical_bytes(expected)
    result = subprocess.run(
        [
            "cargo",
            "run",
            "--release",
            "--offline",
            "--locked",
            "--quiet",
            "--manifest-path",
            str(MANIFEST),
        ],
        cwd=ROOT,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env={**os.environ, "CARGO_TARGET_DIR": str(ROOT / "target")},
    )
    if result.returncode:
        sys.stderr.buffer.write(result.stderr)
        return result.returncode
    if result.stdout != expected_bytes:
        actual = json.loads(result.stdout)
        print(first_difference(expected, actual), file=sys.stderr)
        print(
            json.dumps(
                {
                    "status": "mismatch",
                    "cases": len(cases),
                    "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
                    "python_sha256": hashlib.sha256(expected_bytes).hexdigest(),
                    "rust_sha256": hashlib.sha256(result.stdout).hexdigest(),
                },
                sort_keys=True,
            )
        )
        return 1
    page_count = sum(len(case["pages"]) for case in normalized_input["cases"])
    region_count = sum(
        len(page["regions"])
        for case in normalized_input["cases"]
        for page in case["pages"]
    )
    line_count = sum(
        len(page["lines"])
        for case in normalized_input["cases"]
        for page in case["pages"]
    )
    print(
        json.dumps(
            {
                "status": "byte_identical",
                "source_revision": source_revision,
                "seed": args.seed,
                "cases": len(cases),
                "pages": page_count,
                "regions": region_count,
                "lines": line_count,
                "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
                "output_sha256": hashlib.sha256(expected_bytes).hexdigest(),
                "bytes": len(expected_bytes),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
