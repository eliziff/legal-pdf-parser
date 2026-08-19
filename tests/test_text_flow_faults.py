"""Text-flow continuity fault channels (Text-Fidelity semantics over the
vendored hyphen primitives)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalpdf.core import _text_flow_faults  # noqa: E402
from legalpdf.model import Line, Page, Span  # noqa: E402


def line(line_id: str, order: int, text: str, y0: float, region: str = "body") -> Line:
    bbox = [72.0, y0, 400.0, y0 + 12.0]
    return Line(
        id=line_id,
        page_index=0,
        page_number=1,
        source_index=order,
        reading_order=order,
        block_index=1,
        text=text,
        bbox=bbox,
        spans=[Span(id=f"{line_id}-s", text=text, bbox=bbox, size=10.0, start=0, end=len(text))],
        region_type=region,
    )


def page_of(*lines: Line) -> Page:
    return Page(
        id="p0001", index=0, number=1, width=612, height=792,
        lines=list(lines), regions=[],
    )


class TextFlowFaults(unittest.TestCase):
    def test_clean_hyphen_join_is_silent(self) -> None:
        faults = _text_flow_faults(
            [page_of(
                line("a", 1, "the constitutional frame-", 100),
                line("b", 2, "work requires deference.", 120),
            )]
        )
        self.assertEqual([], faults)

    def test_join_across_body_note_boundary_marks_both_endpoints(self) -> None:
        faults = _text_flow_faults(
            [page_of(
                line("a", 1, "the constitutional frame-", 100),
                line("b", 2, "work requires deference.", 660, region="footnote"),
            )]
        )
        self.assertEqual(["REGION_BOUNDARY_FAULT"], [fault.code for fault in faults])
        self.assertEqual(["a", "b"], faults[0].line_ids)
        self.assertEqual("warning", faults[0].severity)

    def test_uncontinued_fragment_is_a_dangling_fault(self) -> None:
        faults = _text_flow_faults(
            [page_of(
                line("a", 1, "the constitutional frame-", 100),
                line("b", 2, "1982, c 11 (UK).", 120),
            )]
        )
        self.assertEqual(["DANGLING_SOFT_HYPHEN"], [fault.code for fault in faults])
        self.assertEqual(["a"], faults[0].line_ids)
        self.assertEqual("info", faults[0].severity)

    def test_page_final_fragment_is_not_dangling(self) -> None:
        faults = _text_flow_faults(
            [page_of(line("a", 1, "the constitutional frame-", 700))]
        )
        self.assertEqual([], faults)

    def test_headers_and_excluded_lines_are_invisible(self) -> None:
        marker = line("m", 2, "12", 101)
        marker.exclude_from_body = True
        faults = _text_flow_faults(
            [page_of(
                line("a", 1, "the constitutional frame-", 100),
                marker,
                line("b", 3, "work requires deference.", 120),
            )]
        )
        self.assertEqual([], faults)


if __name__ == "__main__":
    unittest.main()
