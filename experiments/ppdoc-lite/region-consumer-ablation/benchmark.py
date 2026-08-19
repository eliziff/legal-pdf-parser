#!/usr/bin/env python3
"""Replay held-out manual lines with and without source-region labels.

This measures final *line roles*.  It deliberately does not score reading order
or footnote-reference/body pairing because the manual line export contains
neither shuffled source order nor reference-anchor gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


IMAGE_KEY = re.compile(r"__\d+_(.+?)_article-(\d+)_pdf-page-(\d+)\.png$")
SCORED_TYPES = {
    "heading_line_role": ("heading", {"paragraph_title"}),
    "header_line_role": ("header", {"header"}),
    "footer_line_role": ("footer", {"footer"}),
    "footnote_line_role": ("footnote", {"footnote"}),
}


def wanted_pages(coco_path: Path) -> dict[tuple[str, int, int], tuple[int, int]]:
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    wanted = {}
    for image in coco["images"]:
        match = IMAGE_KEY.search(image["file_name"])
        if not match:
            raise ValueError(f"unrecognized image name: {image['file_name']}")
        wanted[(match.group(1), int(match.group(2)), int(match.group(3)))] = (image["width"], image["height"])
    return wanted


def manual_paths(gold_root: Path) -> list[Path]:
    primary = gold_root / "manual_materialized/full_article_input/manual_gt_lines.jsonl"
    return [primary, *sorted(gold_root.glob("pending_materialized/*/final_contract_input/manual_gt_lines.jsonl"))]


def load_lines(paths: list[Path], wanted: set[tuple[str, int, int]]) -> dict[tuple[str, int, int], list[dict]]:
    pages: dict[tuple[str, int, int], dict[str, dict]] = defaultdict(dict)
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for raw in handle:
                row = json.loads(raw)
                key = (row["dataset"], int(row["article_id"]), int(row["pdf_page"]))
                if key in wanted:
                    pages[key].setdefault(row.get("record_id") or row["line_id"], row)
    return {
        key: sorted(rows.values(), key=lambda row: (int(row["reading_order_index"]), row.get("record_id") or row["line_id"]))
        for key, rows in pages.items()
    }


def common_input(
    article_pages: list[tuple[tuple[str, int, int], list[dict]]],
    page_sizes: dict[tuple[str, int, int], tuple[int, int]],
    regions_by_id: dict[str, tuple[str, str]] | None,
) -> dict:
    pages = []
    for page_index, (key, rows) in enumerate(article_pages):
        heights = [row["line_bbox_px"]["y1"] - row["line_bbox_px"]["y0"] for row in rows]
        median_height = statistics.median(heights) or 1.0
        lines = []
        for source_index, row in enumerate(rows, 1):
            box = row["line_bbox_px"]
            text = row["raw_transcription"]
            line_id = f"{key[0]}-{key[1]}-{key[2]}-{row['line_id']}"
            assigned = regions_by_id.get(line_id) if regions_by_id is not None else None
            size = min(24.0, max(4.0, 10.0 * (box["y1"] - box["y0"]) / median_height))
            lines.append(
                {
                    "id": line_id,
                    "page_index": page_index,
                    "page_number": key[2],
                    "source_index": source_index,
                    "reading_order": source_index,
                    "block_index": source_index,
                    "text": text,
                    "bbox": [box["x0"], box["y0"], box["x1"], box["y1"]],
                    "spans": [{"id": f"{line_id}-s1", "text": text, "bbox": [box["x0"], box["y0"], box["x1"], box["y1"]], "size": size}],
                    "region_id": assigned[1] if assigned else "",
                    "region_type": assigned[0] if assigned else "unknown",
                    "source": "manual-heldout",
                }
            )
        pages.append(
            {
                "id": f"page-{page_index + 1}",
                "index": page_index,
                "number": key[2],
                "width": rows[0].get("page_width_px") or page_sizes[key][0],
                "height": rows[0].get("page_height_px") or page_sizes[key][1],
                "lines": lines,
                "regions": [],
                "source": "manual-heldout",
                "text_quality": 1.0,
            }
        )
    identity = "|".join(f"{key[0]}:{key[1]}:{key[2]}" for key, _ in article_pages)
    return {
        "schema_version": "legalpdf.common-input.v1",
        "source_name": identity,
        "source_sha256": hashlib.sha256(identity.encode()).hexdigest(),
        "pages": pages,
        "separators": [None] * len(pages),
        "metadata": {},
        "tables": [],
        "images": [],
        "diagnostics": [],
    }


def binary_metrics(gold: set[str], predicted: set[str]) -> dict[str, float | int]:
    true_positive = len(gold & predicted)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    return {
        "gold": len(gold),
        "predicted": len(predicted),
        "true_positive": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def score(results: list[dict], gold_by_id: dict[str, str]) -> dict:
    predicted: dict[str, set[str]] = {kind: set() for kind, _ in SCORED_TYPES.values()}
    for result in results:
        for page in result["derived_pages"]:
            for line in page["lines"]:
                if line["region_type"] in predicted:
                    predicted[line["region_type"]].add(line["id"])
    metrics = {}
    for metric, (predicted_label, labels) in SCORED_TYPES.items():
        gold = {line_id for line_id, label in gold_by_id.items() if label in labels}
        metrics[metric] = binary_metrics(gold, predicted[predicted_label])
    return metrics


def load_model_predictions(path: Path) -> dict[tuple[str, int, int], list[dict]]:
    with path.open(encoding="utf-16" if path.read_bytes()[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8") as handle:
        records = [json.loads(raw) for raw in handle if raw.strip()]
    predictions = {}
    for record in records:
        match = IMAGE_KEY.search(Path(record["image"]).name)
        if not match:
            raise ValueError(f"unrecognized model image: {record['image']}")
        predictions[(match.group(1), int(match.group(2)), int(match.group(3)))] = record["detections"]
    return predictions


def model_assignments(
    runner: Path,
    by_article: dict[tuple[str, int], list[tuple[tuple[str, int, int], list[dict]]]],
    page_sizes: dict[tuple[str, int, int], tuple[int, int]],
    predictions: dict[tuple[str, int, int], list[dict]],
) -> tuple[dict[tuple[str, int], dict[str, tuple[str, str]] | None], dict]:
    cases = []
    for article, article_pages in sorted(by_article.items()):
        case_pages = []
        for key, rows in sorted(article_pages, key=lambda item: item[0][2]):
            width, height = page_sizes[key]
            case_pages.append(
                {
                    "page_number": key[2],
                    "width": width,
                    "height": height,
                    "lines": [
                        {
                            "line_id": f"{key[0]}-{key[1]}-{key[2]}-{row['line_id']}",
                            "text": row["raw_transcription"],
                            "bbox": [
                                row["line_bbox_px"]["x0"],
                                row["line_bbox_px"]["y0"],
                                row["line_bbox_px"]["x1"],
                                row["line_bbox_px"]["y1"],
                            ],
                        }
                        for row in rows
                    ],
                    "regions": [
                        {"label": detection["label"], "score": detection["score"], "bbox": detection["bbox"]}
                        for detection in predictions[key]
                    ],
                }
            )
        cases.append({"name": f"{article[0]}:{article[1]}", "pages": case_pages})
    completed = subprocess.run(
        [str(runner)], input=json.dumps({"cases": cases}), check=True, capture_output=True, text=True
    )
    output = json.loads(completed.stdout)
    assignments = {}
    fail_closed = []
    matched_lines = 0
    for case in output["cases"]:
        article_key = case["name"].split(":", 1)
        article = (article_key[0], int(article_key[1]))
        rows = [assignment for page in case["pages"] for assignment in page["assignments"]]
        if any(row["label"] is None for row in rows):
            assignments[article] = None
            fail_closed.append(case["name"])
            continue
        article_assignments = {}
        for page in case["pages"]:
            for row in page["assignments"]:
                article_assignments[row["line_id"]] = (
                    row["label"],
                    f"model-p{page['page_number']}-r{row['raw_index']}",
                )
        matched_lines += len(article_assignments)
        assignments[article] = article_assignments
    return assignments, {
        "prediction_pages": len(predictions),
        "complete_articles": len(assignments) - len(fail_closed),
        "fail_closed_articles": fail_closed,
        "matched_lines_in_complete_articles": matched_lines,
    }


def run(args: argparse.Namespace) -> dict:
    wanted = wanted_pages(args.coco)
    pages = load_lines(manual_paths(args.manual_gold_root), set(wanted))
    by_article: dict[tuple[str, int], list[tuple[tuple[str, int, int], list[dict]]]] = defaultdict(list)
    gold_by_id = {}
    for key, rows in pages.items():
        by_article[key[:2]].append((key, rows))
        for row in rows:
            gold_by_id[f"{key[0]}-{key[1]}-{key[2]}-{row['line_id']}"] = row["region_type"]
    model_by_article = None
    model_summary = None
    if args.model_predictions is not None:
        predictions = load_model_predictions(args.model_predictions)
        missing = sorted(set(pages) - set(predictions))
        if missing:
            raise ValueError(f"model predictions missing {len(missing)} scored pages: {missing[:3]}")
        model_by_article, model_summary = model_assignments(
            args.postprocess_runner, by_article, wanted, predictions
        )
    args.output.mkdir(parents=True, exist_ok=True)
    arm_results: dict[str, list[dict]] = {"no_regions": [], "gold_regions": []}
    complete_model_results: dict[str, list[dict]] = {"no_regions": [], "model_regions": []}
    if model_by_article is not None:
        arm_results["model_regions"] = []
    for article, article_pages in sorted(by_article.items()):
        article_pages.sort(key=lambda item: item[0][2])
        gold_regions = {
            f"{key[0]}-{key[1]}-{key[2]}-{row['line_id']}": (row["region_type"], row["region_id"])
            for key, rows in article_pages
            for row in rows
        }
        arms = [("no_regions", None), ("gold_regions", gold_regions)]
        if model_by_article is not None:
            arms.append(("model_regions", model_by_article[article]))
        for arm, regions_by_id in arms:
            stem = f"{article[0]}-{article[1]}-{arm}"
            input_path = args.output / f"{stem}.input.json"
            result_path = args.output / f"{stem}.result.json"
            input_path.write_text(
                json.dumps(common_input(article_pages, wanted, regions_by_id), ensure_ascii=False), encoding="utf-8"
            )
            subprocess.run(
                [str(args.legalpdf), "_parity-replay", str(input_path), "--output", str(result_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            arm_results[arm].append(result)
            if (
                model_by_article is not None
                and model_by_article[article] is not None
                and arm in complete_model_results
            ):
                complete_model_results[arm].append(result)
    complete_gold = {}
    if model_by_article is not None:
        complete_ids = {
            line_id
            for article_assignments in model_by_article.values()
            if article_assignments is not None
            for line_id in article_assignments
        }
        complete_gold = {line_id: label for line_id, label in gold_by_id.items() if line_id in complete_ids}
    summary = {
        "schema_version": "legalpdf.region-consumer-ablation.v1",
        "wanted_pages": len(wanted),
        "scored_pages": len(pages),
        "article_count": len(by_article),
        "line_count": len(gold_by_id),
        "scope": {
            "scores_final_line_roles": True,
            "scores_reading_order": False,
            "scores_footnote_pairing": False,
            "heading_limit": "synthetic spans retain geometry-derived size but not original font flags",
        },
        "model": model_summary,
        "arms": {arm: score(results, gold_by_id) for arm, results in arm_results.items()},
        "model_complete_subset": (
            {
                "article_count": model_summary["complete_articles"],
                "line_count": len(complete_gold),
                "arms": {
                    arm: score(results, complete_gold) for arm, results in complete_model_results.items()
                },
            }
            if model_summary is not None
            else None
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def self_test() -> None:
    assert binary_metrics({"a", "b"}, {"b", "c"}) == {
        "gold": 2, "predicted": 2, "true_positive": 1, "precision": 0.5, "recall": 0.5, "f1": 0.5
    }
    assert IMAGE_KEY.search("x__001_APPEAL_article-42_pdf-page-7.png").groups() == ("APPEAL", "42", "7")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", type=Path)
    parser.add_argument("--manual-gold-root", type=Path)
    parser.add_argument("--legalpdf", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-predictions", type=Path)
    parser.add_argument("--postprocess-runner", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    for name in ("coco", "manual_gold_root", "legalpdf", "output"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    if (args.model_predictions is None) != (args.postprocess_runner is None):
        parser.error("--model-predictions and --postprocess-runner must be supplied together")
    run(args)


if __name__ == "__main__":
    main()
