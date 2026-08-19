"""Vendored superscript splice: parity vs the Text-Fidelity checkout, splice
vectors, and the flag-absent marker lanes inside the engine."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalpdf.core import (  # noqa: E402
    _associate_detached_references,
    _label_is_typographic,
)
from legalpdf.model import Line, Page, Span  # noqa: E402
from legalpdf.superscript_splice import (  # noqa: E402
    line_median_font_size,
    splice_orphaned_superscript_markers,
)

SENTINEL_A = "# --- byte-equal payload A below (to_float); do not edit ---\n"
SENTINEL_B = "# --- byte-equal payload B below (median/marker/splice); do not edit ---\n"
A_START = "def to_float("
A_END = "\ndef bbox_px("
B_START = "def line_median_font_size("
B_END = "\ndef span_text("
TFP_ROOT = Path(
    os.environ.get(
        "TEXT_FIDELITY_ROOT",
        ROOT.parent / "Text-Fidelity-Project",
    )
)


def row(
    line_id: str,
    text: str,
    bbox: tuple[float, float, float, float],
    size: float,
    *,
    styles: list[str] | None = None,
    region: str = "b1",
) -> dict:
    return {
        "engine_id": line_id,
        "raw_transcription": text,
        "region_id": region,
        "line_bbox_px": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
        "native_pdf_median_font_size": size,
        "native_pdf_span_styles": [
            {
                "size": size,
                "styles": styles or [],
                "raw_start": 0,
                "raw_end": len(text),
                "start": 0,
                "end": len(text),
            }
        ],
    }


HOST = row("host", "The court held that the appeal must fail", (72, 100, 300, 112), 11.0)
RAISED_MARKER = row("marker", "12", (300.5, 98, 308, 106), 8.0)


class VendoredParity(unittest.TestCase):
    def test_payload_regions_match_checkout(self) -> None:
        source_path = (
            TFP_ROOT / "tools" / "galley" / "final_contract_v2" / "native_extraction.py"
        )
        if not source_path.is_file():
            self.skipTest("Text-Fidelity checkout not present on this machine")
        source = source_path.read_text(encoding="utf-8")
        region_a = source[source.index(A_START) : source.index(A_END)]
        region_b = source[source.index(B_START) : source.index(B_END)]
        vendored = (ROOT / "src" / "legalpdf" / "superscript_splice.py").read_text(
            encoding="utf-8"
        )
        payload_a = vendored[vendored.index(SENTINEL_A) + len(SENTINEL_A) :]
        self.assertEqual(region_a, payload_a[: len(region_a)], "region A drifted")
        payload_b = vendored[vendored.index(SENTINEL_B) + len(SENTINEL_B) :]
        self.assertEqual(region_b, payload_b, "region B drifted")


class SpliceVectors(unittest.TestCase):
    def test_flag_absent_raised_marker_splices_into_trailing_host(self) -> None:
        merged, count = splice_orphaned_superscript_markers(
            [dict(RAISED_MARKER), dict(HOST)], scale=1.0
        )
        self.assertEqual(1, count)
        self.assertEqual(
            ["The court held that the appeal must fail12"],
            [record["raw_transcription"] for record in merged],
        )

    def test_same_size_digit_line_abstains(self) -> None:
        paragraph_number = row("num", "12", (72, 98, 86, 112), 11.0)
        merged, count = splice_orphaned_superscript_markers(
            [dict(paragraph_number), dict(HOST)], scale=1.0
        )
        self.assertEqual(0, count)
        self.assertEqual(2, len(merged))

    def test_two_plausible_hosts_abstain(self) -> None:
        other = row("other", "A second body line on the same baseline", (72, 100, 298, 112), 11.0)
        merged, count = splice_orphaned_superscript_markers(
            [dict(other), dict(RAISED_MARKER), dict(HOST)], scale=1.0
        )
        self.assertEqual(0, count)
        self.assertEqual(3, len(merged))

    def test_median_excludes_superscript_spans(self) -> None:
        spans = [
            {"size": 11.0, "styles": [], "raw_start": 0, "raw_end": 4},
            {"size": 6.0, "styles": ["superscript"], "raw_start": 4, "raw_end": 40},
        ]
        self.assertEqual(11.0, line_median_font_size(spans))


def engine_line(
    line_id: str,
    order: int,
    text: str,
    bbox: list[float],
    size: float,
    *,
    superscript: bool = False,
) -> Line:
    return Line(
        id=line_id,
        page_index=0,
        page_number=1,
        source_index=order,
        reading_order=order,
        block_index=1,
        text=text,
        bbox=bbox,
        spans=[
            Span(
                id=f"{line_id}-s",
                text=text,
                bbox=bbox,
                size=size,
                superscript=superscript,
                start=0,
                end=len(text),
            )
        ],
        region_type="body",
    )


class EngineLanes(unittest.TestCase):
    def test_flag_absent_raised_marker_attaches_via_vendored_lane(self) -> None:
        # 8.6pt vs 11pt body = 0.78x: the primary lane's 0.75x gate rejects
        # it; the vendored peer-ratio (<=0.8x) plus raise proof accepts.
        host = engine_line(
            "host", 1, "The court held that the appeal must fail", [72, 100, 300, 112], 11.0
        )
        marker = engine_line("marker", 2, "12", [300.5, 98, 308, 106], 8.6)
        filler = [
            engine_line(f"body{index}", 3 + index, "Further body text follows here.",
                        [72, 130 + index * 20, 300, 142 + index * 20], 11.0)
            for index in range(4)
        ]
        page = Page(
            id="p0001", index=0, number=1, width=612, height=792,
            lines=[host, marker, *filler], regions=[],
        )
        _associate_detached_references(page, None)
        self.assertTrue(marker.exclude_from_body)
        self.assertEqual(1, len(host.detached_references))
        record = host.detached_references[0]
        self.assertEqual("12", record["selected_text"])
        self.assertEqual(len(host.text), record["start_offset"])
        self.assertEqual("marker", record["source_line_id"])

    def test_paragraph_number_line_stays_in_body(self) -> None:
        host = engine_line(
            "host", 1, "The court held that the appeal must fail", [72, 100, 300, 112], 11.0
        )
        number = engine_line("num", 2, "12", [72, 130, 86, 142], 11.0)
        page = Page(
            id="p0001", index=0, number=1, width=612, height=792,
            lines=[host, number], regions=[],
        )
        _associate_detached_references(page, None)
        self.assertFalse(number.exclude_from_body)
        self.assertEqual([], host.detached_references)

    def test_label_size_inference_requires_raise(self) -> None:
        # 8.2pt on a 10.5pt line: the 0.75x gates reject (8.2 > 7.875), the
        # vendored peer ratio accepts (8.2 * 1.25 = 10.25 <= 10.5) - but only
        # with the raise proof.
        raised = Line(
            id="l1", page_index=0, page_number=1, source_index=1, reading_order=1,
            block_index=1, text="12 Note body text here.", bbox=[72, 660, 300, 672],
            spans=[
                Span(id="s1", text="12", bbox=[72, 660, 82, 666], size=8.2, start=0, end=2),
                Span(id="s2", text=" Note body text here.", bbox=[84, 660, 300, 672], size=10.5, start=2, end=23),
            ],
            region_type="body",
        )
        typographic, _ = _label_is_typographic(raised, start=0, end=2, body_size=10.5)
        self.assertTrue(typographic)

        flat = Line(
            id="l2", page_index=0, page_number=1, source_index=1, reading_order=1,
            block_index=1, text="12 Note body text here.", bbox=[72, 660, 300, 672],
            spans=[
                Span(id="s1", text="12", bbox=[72, 660, 82, 672], size=8.2, start=0, end=2),
                Span(id="s2", text=" Note body text here.", bbox=[84, 660, 300, 672], size=10.5, start=2, end=23),
            ],
            region_type="body",
        )
        typographic, _ = _label_is_typographic(flat, start=0, end=2, body_size=10.5)
        self.assertFalse(typographic)


if __name__ == "__main__":
    unittest.main()
