"""Vendored column-order arbiter: parity vs the Text-Fidelity checkout,
behavior vectors, and the `_order_page` integration contract."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalpdf.column_order_arbiter import (  # noqa: E402
    arbitrate_page_order,
    column_model,
    hyphen_join_confidence,
)
from legalpdf.core import _order_page  # noqa: E402
from legalpdf.model import Line, Page, Span  # noqa: E402

SENTINEL = b"# --- byte-equal payload below; do not edit (see header) ---\n"
TFP_ROOT = Path(
    os.environ.get(
        "TEXT_FIDELITY_ROOT",
        ROOT.parent / "Text-Fidelity-Project",
    )
)


def ratio_row(line_id: str, order: int, x0: float, y0: float, x1: float, y1: float, text: str = "") -> dict:
    return {
        "line_id": line_id,
        "source_order": order,
        "rx0": x0,
        "ry0": y0,
        "rx1": x1,
        "ry1": y1,
        "text": text,
    }


def two_column_rows(*, interleaved: bool) -> list[dict]:
    """Ten aligned rows per column; interleaved=True feeds raster order."""
    rows = []
    for index in range(10):
        y0 = 0.10 + index * 0.05
        rows.append((f"L{index}", 0.08, y0, 0.45, y0 + 0.03))
        rows.append((f"R{index}", 0.55, y0, 0.92, y0 + 0.03))
    if not interleaved:
        rows.sort(key=lambda r: r[0])
    return [
        ratio_row(line_id, order, x0, y0, x1, y1)
        for order, (line_id, x0, y0, x1, y1) in enumerate(rows, start=1)
    ]


class VendoredParity(unittest.TestCase):
    def test_payload_matches_checkout_byte_for_byte(self) -> None:
        source = (
            TFP_ROOT
            / "tools"
            / "ocr"
            / "layout_regioning"
            / "ppdoc"
            / "column_order_arbiter.py"
        )
        if not source.is_file():
            self.skipTest("Text-Fidelity checkout not present on this machine")
        vendored = (ROOT / "src" / "legalpdf" / "column_order_arbiter.py").read_bytes()
        payload = vendored[vendored.index(SENTINEL) + len(SENTINEL) :]
        self.assertEqual(source.read_bytes(), payload, "vendored payload drifted")


class ArbiterVectors(unittest.TestCase):
    def test_interleaved_two_column_page_fires_column_repair(self) -> None:
        decision = arbitrate_page_order(two_column_rows(interleaved=True))
        self.assertTrue(decision["fired"])
        self.assertEqual("column_interleave_repair", decision["reason"])
        self.assertEqual(
            [f"L{index}" for index in range(10)] + [f"R{index}" for index in range(10)],
            decision["order_line_ids"],
        )

    def test_column_major_two_column_page_keeps_incumbent(self) -> None:
        decision = arbitrate_page_order(two_column_rows(interleaved=False))
        self.assertFalse(decision["fired"])
        self.assertEqual("two_column_kraken_coherent", decision["reason"])

    def test_single_column_keeps_incumbent(self) -> None:
        rows = [
            ratio_row(f"S{index}", index + 1, 0.30, 0.10 + index * 0.05, 0.70, 0.13 + index * 0.05)
            for index in range(10)
        ]
        decision = arbitrate_page_order(rows)
        self.assertFalse(decision["fired"])
        self.assertEqual([f"S{index}" for index in range(10)], decision["order_line_ids"])

    def test_column_model_sees_two_columns(self) -> None:
        self.assertEqual("two_column", column_model(two_column_rows(interleaved=True))["kind"])

    def test_hyphen_join_confidence(self) -> None:
        self.assertEqual(0.95, hyphen_join_confidence("constitu-", "tional order"))
        self.assertEqual(0.0, hyphen_join_confidence("no tail", "anything"))


def engine_line(line_id: str, order: int, text: str, bbox: list[float], region_type: str = "body") -> Line:
    return Line(
        id=line_id,
        page_index=0,
        page_number=1,
        source_index=order,
        reading_order=order,
        block_index=order,
        text=text,
        bbox=bbox,
        spans=[Span(id=f"{line_id}-s", text=text, bbox=bbox, size=10.0, start=0, end=len(text))],
        region_type=region_type,
    )


class OrderPageIntegration(unittest.TestCase):
    def make_interleaved_page(self) -> Page:
        lines = []
        order = 0
        for index in range(10):
            y0 = 80.0 + index * 40.0
            order += 1
            lines.append(engine_line(f"L{index}", order, f"left {index}", [49, y0, 275, y0 + 12]))
            order += 1
            lines.append(engine_line(f"R{index}", order, f"right {index}", [337, y0, 563, y0 + 12]))
        order += 1
        lines.append(
            engine_line("note", order, "1 Footnote body.", [72, 700, 540, 712], region_type="footnote")
        )
        return Page(id="p0001", index=0, number=1, width=612, height=792, lines=lines, regions=[])

    def test_interleaved_columns_are_repaired_and_diagnosed(self) -> None:
        page = self.make_interleaved_page()
        diagnostics = _order_page(page)
        self.assertEqual(
            [f"L{index}" for index in range(10)]
            + [f"R{index}" for index in range(10)]
            + ["note"],
            [line.id for line in page.lines],
        )
        self.assertEqual(
            list(range(1, len(page.lines) + 1)),
            [line.reading_order for line in page.lines],
        )
        self.assertIn("COLUMN_ORDER_REPAIRED", [d.code for d in diagnostics])

    def test_trustworthy_source_order_is_kept(self) -> None:
        lines = [
            engine_line(f"S{index}", index + 1, f"line {index}", [180, 80.0 + index * 40.0, 430, 92.0 + index * 40.0])
            for index in range(10)
        ]
        page = Page(id="p0001", index=0, number=1, width=612, height=792, lines=lines, regions=[])
        diagnostics = _order_page(page)
        self.assertEqual([f"S{index}" for index in range(10)], [line.id for line in page.lines])
        self.assertEqual([], diagnostics)

    def test_long_run_column_flow_keeps_order_but_warns(self) -> None:
        lines = []
        order = 0
        for block, prefix in enumerate(("A", "B", "C", "D")):
            column = 49.0 if block % 2 == 0 else 337.0
            for index in range(7):
                y0 = 80.0 + index * 40.0 + (block // 2) * 300.0
                order += 1
                lines.append(
                    engine_line(
                        f"{prefix}{index}", order, f"{prefix} {index}",
                        [column, y0, column + 226.0, y0 + 12.0],
                    )
                )
        page = Page(id="p0001", index=0, number=1, width=612, height=792, lines=lines, regions=[])
        diagnostics = _order_page(page)
        self.assertEqual([line.id for line in lines], [line.id for line in page.lines])
        self.assertIn("COLUMN_ORDER_UNCERTAIN", [d.code for d in diagnostics])


if __name__ == "__main__":
    unittest.main()
