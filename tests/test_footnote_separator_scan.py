"""Vendored footnote-separator scan: parity vs the Text-Fidelity checkout,
classification vectors, and the raster fallback inside parse_pdf."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalpdf.footnote_separator_scan import (  # noqa: E402
    classify_separator,
    scan_gray_page,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

SENTINEL = "# --- byte-equal payload below; do not edit (see header) ---\n"
REGION_START = "OTSU_MIN_THRESHOLD = 64"
REGION_END = "\n\ndef scan_page_image"
TFP_ROOT = Path(
    os.environ.get(
        "TEXT_FIDELITY_ROOT",
        ROOT.parent / "Text-Fidelity-Project",
    )
)


def rule(
    *,
    y: float = 0.70,
    x0: float = 0.08,
    x1: float = 0.40,
    darkness: float = 0.9,
    thickness: float = 0.002,
    above_ink: float = 0.0,
    below_ink: float = 0.0,
    above_block: float = 0.2,
    below_block: float = 0.2,
) -> dict:
    return {
        "y_center_ratio": y,
        "x0_ratio": x0,
        "x1_ratio": x1,
        "width_ratio": x1 - x0,
        "darkness": darkness,
        "thickness_ratio": thickness,
        "above_ink": above_ink,
        "below_ink": below_ink,
        "above_block_ink": above_block,
        "below_block_ink": below_block,
    }


class VendoredParity(unittest.TestCase):
    def test_payload_matches_checkout_region(self) -> None:
        source_path = (
            TFP_ROOT / "tools" / "ocr" / "layout_regioning" / "footnote_separator_scan.py"
        )
        if not source_path.is_file():
            self.skipTest("Text-Fidelity checkout not present on this machine")
        source = source_path.read_text(encoding="utf-8")
        region = source[source.index(REGION_START) : source.index(REGION_END)]
        if not region.endswith("\n"):
            region += "\n"
        vendored = (ROOT / "src" / "legalpdf" / "footnote_separator_scan.py").read_text(
            encoding="utf-8"
        )
        payload = vendored[vendored.index(SENTINEL) + len(SENTINEL) :]
        self.assertEqual(region, payload, "vendored detection region drifted")

class ClassifyVectors(unittest.TestCase):
    def test_single_left_anchored_rule_is_found(self) -> None:
        separators, status = classify_separator([rule()])
        self.assertEqual("found", status)
        self.assertEqual("short_rule", separators[0]["kind"])

    def test_heading_underline_rejected_by_hugging_text(self) -> None:
        separators, status = classify_separator([rule(above_ink=0.2)])
        self.assertEqual("none", status)
        self.assertEqual([], separators)

    def test_title_page_divider_rejected_without_content_below(self) -> None:
        separators, status = classify_separator([rule(below_block=0.0)])
        self.assertEqual("none", status)

    def test_column_partners_at_equal_y(self) -> None:
        left = rule(x0=0.08, x1=0.40)
        right = rule(x0=0.55, x1=0.87, y=0.705)
        separators, status = classify_separator([left, right])
        self.assertEqual("found_two_column", status)
        self.assertEqual(2, len(separators))

    def test_two_unrelated_candidates_are_ambiguous(self) -> None:
        separators, status = classify_separator([rule(y=0.5), rule(y=0.8)])
        self.assertEqual("ambiguous", status)
        self.assertEqual([], separators)

    def test_full_width_rule_is_full_kind(self) -> None:
        separators, status = classify_separator([rule(x0=0.1, x1=0.85)])
        self.assertEqual("found", status)
        self.assertEqual("full_rule", separators[0]["kind"])


@unittest.skipIf(np is None, "numpy not installed")
class ScanGrayPage(unittest.TestCase):
    def synthetic_page(self, *, with_rule: bool = True) -> "np.ndarray":
        page = np.full((1000, 800), 255, dtype=np.uint8)
        for y in range(100, 620, 4):
            page[y : y + 2, 80:720:3] = 0
        if with_rule:
            page[700:702, 64:320] = 0
        for y in range(720, 950, 4):
            page[y : y + 2, 80:720:3] = 0
        return page

    def test_rule_found_on_synthetic_page(self) -> None:
        record = scan_gray_page(self.synthetic_page())
        self.assertEqual("found", record["separator_status"])
        self.assertAlmostEqual(0.70, record["separators"][0]["y_center_ratio"], places=2)

    def test_no_rule_yields_none_status(self) -> None:
        record = scan_gray_page(self.synthetic_page(with_rule=False))
        self.assertEqual("none", record["separator_status"])


def rule_page(document: "object") -> "object":
    """One page shaped like a real note-bearing page: body block, a thin
    filled-rect separator (the Word/LibreOffice export shape), note block."""
    import fitz

    page = document.new_page(width=612, height=792)
    body = (
        "The first proposition is supported by binding authority and "
        "the second by persuasive authority from a parallel court."
    )
    for row in range(23):
        page.insert_text((72, 90 + row * 24), body[: 90 - row], fontsize=11)
    page.draw_rect(fitz.Rect(72, 650, 220, 651.4), color=0, fill=0)
    for row, note in enumerate(
        (
            "1 First footnote body with a citation and a pinpoint.",
            "2 Second footnote body continuing the note block.",
            "3 Third footnote body closing the page.",
        )
    ):
        page.insert_text((72, 668 + row * 14), note, fontsize=8)
    return page


class SeparatorLanes(unittest.TestCase):
    def test_word_export_rect_rule_found_by_vector_lane(self) -> None:
        import fitz

        from legalpdf.core import _separator_y

        with fitz.open() as document:
            found = _separator_y(rule_page(document))
        self.assertIsNotNone(found)
        self.assertAlmostEqual(650.7, found, delta=1.0)

    @unittest.skipIf(np is None, "numpy not installed")
    def test_scanned_page_rule_recovered_by_raster_scan(self) -> None:
        import fitz

        from legalpdf.core import _raster_separator_y, _separator_y

        with fitz.open() as source:
            pixmap = rule_page(source).get_pixmap(
                matrix=fitz.Matrix(2, 2), alpha=False
            )
            with fitz.open() as scanned:
                page = scanned.new_page(width=612, height=792)
                page.insert_image(page.rect, pixmap=pixmap)
                self.assertIsNone(_separator_y(page))
                recovered = _raster_separator_y(page)
                self.assertIsNotNone(recovered)
                self.assertAlmostEqual(650.7, recovered, delta=3.0)

    def test_native_pages_skip_the_raster_scan(self) -> None:
        import tempfile

        import fitz

        from legalpdf import core

        calls = []
        original = core._raster_separator_y
        core._raster_separator_y = lambda page: calls.append(1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "norule.pdf"
                with fitz.open() as document:
                    page = document.new_page(width=612, height=792)
                    page.insert_text((72, 90), "Plain body text only.", fontsize=11)
                    document.save(str(path))
                core.parse_pdf(str(path), cache_dir=Path(tmp) / "cache")
        finally:
            core._raster_separator_y = original
        self.assertEqual([], calls)


if __name__ == "__main__":
    unittest.main()
