#!/usr/bin/env python3
"""Reading-order bench: engine `_order_page` vs the TFP 661-page ordered gold.

Replays the manual gold bundle
(train1_12_full_manual_gold_ordered_db_reading_order_20260629_01, fetched as a
slim field subset to %LOCALAPPDATA%/legalpdf/gold/train1_12_ordered_661) through
the engine's page-ordering logic with PERFECT region labels (gold region types
mapped onto the engine vocabulary), isolating ordering from classification.
Lines are fed in raster order (y, x) — the OCR-lane analogue of kraken-native
input; the native content-stream lane is not exercised here, so numbers are
comparable across engine revisions on this harness, not to TFP's product
benchmarks.

Scoring imports the Text-Fidelity reference at probe time only (sanctioned:
references may be imported inside measurement harnesses, never in production
lanes): `score_sequences`/`aggregate_sequence_scores` from
tools.ocr.routing.ordered_surface_common and `column_model` from
tools.ocr.layout_regioning.ppdoc.column_order_arbiter for subset labeling.

    python -X utf8 dev/bench_order_gold.py --label baseline
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TFP_ROOT = Path(
    os.environ.get("TEXT_FIDELITY_ROOT", ROOT.parent / "Text-Fidelity-Project")
)
GOLD_DIR = (
    Path(os.environ.get("LOCALAPPDATA", ""))
    / "legalpdf"
    / "gold"
    / "train1_12_ordered_661"
)

# Gold ontology -> engine region vocabulary. Everything not banded by the
# engine (tables, quotes, titles, marginal numbers, references) is body-band
# material for ordering purposes.
REGION_MAP = {
    "header": "header",
    "footer": "footer",
    "footnote": "footnote",
    "vision_footnote": "footnote",
    "doc_title": "heading",
    "paragraph_title": "heading",
}


def load_pages(path: Path) -> dict[str, list[dict]]:
    pages: dict[str, list[dict]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for raw in fh:
            row = json.loads(raw)
            if row.get("record_kind") != "line" or row.get("skip_reason"):
                continue
            bbox = row.get("bbox") or []
            if len(bbox) != 4 or any(v is None for v in bbox):
                continue
            pages[row["page_id"]].append(row)
    return pages


def engine_order(rows: list[dict], width: float, height: float) -> list[str]:
    from legalpdf.core import _order_page
    from legalpdf.model import Line, Page

    lines = [
        Line(
            id=str(row["line_id"]),
            page_index=0,
            page_number=1,
            source_index=index,
            reading_order=index,
            block_index=1,
            text=str(row.get("normalized_transcription") or ""),
            bbox=[float(v) for v in row["bbox"]],
            region_type=REGION_MAP.get(str(row.get("region_type")), "body"),
        )
        for index, row in enumerate(rows, start=1)
    ]
    page = Page(
        id="p0001", index=0, number=1, width=width, height=height,
        lines=lines, regions=[],
    )
    _order_page(page)
    return [line.id for line in page.lines]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="run")
    parser.add_argument("--text-fidelity-root", default=str(TFP_ROOT))
    parser.add_argument(
        "--feed",
        choices=["raster", "gold"],
        default="raster",
        help=(
            "incumbent line order fed to the engine: 'raster' is the "
            "worst-case interleaved incumbent (adversarial bound); 'gold' is "
            "the best-case trustworthy incumbent (false-fire bound). Real "
            "lanes (content stream, Tesseract blocks) sit between."
        ),
    )
    args = parser.parse_args()

    gold_path = GOLD_DIR / "gold_lines_slim.jsonl.gz"
    if not gold_path.is_file():
        print(f"error: gold not fetched: {gold_path}", file=sys.stderr)
        return 2
    tfp = Path(args.text_fidelity_root)
    if not (tfp / "tools").is_dir():
        print(f"error: Text-Fidelity checkout not found: {tfp}", file=sys.stderr)
        return 2
    sys.path.insert(0, str(tfp))
    from tools.ocr.layout_regioning.ppdoc.column_order_arbiter import column_model
    from tools.ocr.routing.ordered_surface_common import (
        aggregate_sequence_scores,
        score_sequences,
    )

    pages = load_pages(gold_path)
    scores: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    page_kinds: dict[str, int] = defaultdict(int)
    for page_id, rows in sorted(pages.items()):
        if len(rows) < 2:
            continue
        width = float(rows[0].get("page_width_px") or 0)
        height = float(rows[0].get("page_height_px") or 0)
        if width <= 0 or height <= 0:
            continue
        gold_ids = [
            str(row["line_id"])
            for row in sorted(rows, key=lambda r: int(r["reading_order_index"]))
        ]
        raster = sorted(rows, key=lambda r: (float(r["bbox"][1]), float(r["bbox"][0])))
        if args.feed == "gold":
            feed = sorted(rows, key=lambda r: int(r["reading_order_index"]))
        else:
            feed = raster
        ratio_rows = [
            {
                "line_id": str(row["line_id"]),
                "source_order": index,
                "rx0": float(row["bbox"][0]) / width,
                "ry0": float(row["bbox"][1]) / height,
                "rx1": float(row["bbox"][2]) / width,
                "ry1": float(row["bbox"][3]) / height,
                "text": str(row.get("normalized_transcription") or ""),
            }
            for index, row in enumerate(feed, start=1)
        ]
        kind = str(column_model(ratio_rows)["kind"])
        page_kinds[kind] += 1

        candidates = {
            "engine": engine_order(feed, width, height),
            "raster": [str(row["line_id"]) for row in raster],
        }
        for name, ordered in candidates.items():
            score = score_sequences(gold_ids, ordered)
            scores[name][kind].append(score)
            scores[name]["all"].append(score)

    report = {"label": args.label, "page_kinds": dict(page_kinds), "candidates": {}}
    print(f"pages={sum(page_kinds.values())} kinds={dict(page_kinds)}")
    for name, by_kind in scores.items():
        report["candidates"][name] = {}
        for kind in ("all", "two_column", "margin_column", "single"):
            page_scores = by_kind.get(kind)
            if not page_scores:
                continue
            agg = aggregate_sequence_scores(page_scores)
            rate = agg["normalized_inversion_rate"]
            report["candidates"][name][kind] = {
                "pages": len(page_scores),
                "lines": agg["common_count"],
                "normalized_inversion_rate": rate,
                "exact_position_accuracy": agg["exact_position_accuracy"],
            }
            print(
                f"  {name:8s} {kind:13s} pages={len(page_scores):4d} "
                f"lines={agg['common_count']:6d} inversion={rate * 100:6.2f}% "
                f"exact={agg['exact_position_accuracy'] * 100:5.1f}%"
            )
    snapshot = GOLD_DIR / f"bench_{args.label}.json"
    snapshot.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"snapshot: {snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
