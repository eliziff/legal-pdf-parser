from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import fitz

from legalpdf.adapters import to_alr_payload, to_toa_text_units
from legalpdf.benchmark import extract_docx_gold, score_docx_gold
from legalpdf.codex_repair import (
    _codex_command,
    _schema,
    _stable_hash,
    repair_identity,
)
from legalpdf.core import (
    _assign_printed_page_labels,
    _associate_detached_references,
    _build_regions,
    _build_sections,
    _classify_page,
    _derive,
    _engine_identity,
    _extract_native_page,
    _materialize_footnotes,
    _mark_repeated_furniture,
    _order_page,
    _pair_markers,
    _separator_y,
    improve,
    lookup_footnote,
    parse_pdf,
)
from legalpdf.model import (
    Diagnostic,
    Line,
    Page,
    Paragraph,
    Span,
    Word,
)
from legalpdf.ocr import OCRLine, TesseractOCRProvider
from legalpdf.pdf_backend import _assign_block_indexes, _group_inspector_items


def synthetic_line(
    line_id: str,
    page_number: int,
    text: str,
    bbox: list[float],
    size: float,
    *,
    order: int,
    region_type: str = "unknown",
    spans: list[Span] | None = None,
) -> Line:
    return Line(
        id=line_id,
        page_index=page_number - 1,
        page_number=page_number,
        source_index=order,
        reading_order=order,
        block_index=order,
        text=text,
        bbox=bbox,
        spans=spans
        or [
            Span(
                id=f"{line_id}-span",
                text=text,
                bbox=bbox,
                size=size,
                start=0,
                end=len(text),
            )
        ],
        region_type=region_type,
    )


def pair_markers(
    lines: list[Line],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    pages = [
        Page(
            id=f"p{page_number:04d}",
            index=page_number - 1,
            number=page_number,
            width=612,
            height=792,
            lines=[
                line for line in lines if line.page_number == page_number
            ],
            regions=[],
        )
        for page_number in sorted({line.page_number for line in lines})
    ]
    return _pair_markers(pages)


def make_legal_pdf(path: Path, *, restarted: bool = True) -> None:
    with fitz.open() as document:
        page = document.new_page(width=612, height=792)
        page.insert_text(
            (72, 90),
            "The first proposition is supported by authority¹.",
            fontsize=11,
        )
        page.draw_line((72, 650), (220, 650), width=0.7)
        page.insert_text((72, 675), "1 First footnote body.", fontsize=8)
        if restarted:
            page = document.new_page(width=612, height=792)
            page.insert_text(
                (72, 90),
                "A restarted sequence supports the second proposition¹.",
                fontsize=11,
            )
            page.draw_line((72, 650), (220, 650), width=0.7)
            page.insert_text((72, 675), "1 Restarted footnote body.", fontsize=8)
        document.save(path)


def make_empty_pdf(path: Path) -> None:
    with fitz.open() as document:
        document.new_page(width=612, height=792)
        document.save(path)


def make_endnote_pdf(path: Path) -> None:
    with fitz.open() as document:
        page = document.new_page(width=612, height=792)
        page.insert_text((72, 90), "Body text preceding endnotes.", fontsize=11)
        page = document.new_page(width=612, height=792)
        page.insert_text((72, 70), "NOTES", fontsize=11)
        page.insert_text((72, 110), "1 First endnote.", fontsize=8)
        page.insert_text((72, 135), "2 Second endnote.", fontsize=8)
        page.insert_text((72, 160), "3 Third endnote.", fontsize=8)
        document.save(path)


def make_docx(path: Path) -> None:
    document_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Title</w:t></w:r></w:p>
    <w:p><w:r><w:t>A proposition under 2020 SCC 1</w:t></w:r><w:r><w:footnoteReference w:id="2"/></w:r><w:r><w:t>.</w:t></w:r></w:p>
  </w:body>
</w:document>"""
    footnotes_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:footnotes xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:footnote w:id="-1"><w:p><w:r><w:t>separator</w:t></w:r></w:p></w:footnote>
  <w:footnote w:id="2"><w:p><w:r><w:t>Footnote body.</w:t></w:r></w:p></w:footnote>
</w:footnotes>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/footnotes.xml", footnotes_xml)


def make_endnote_docx(path: Path) -> None:
    document_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>An endnote proposition</w:t></w:r><w:r><w:endnoteReference w:id="2"/></w:r><w:r><w:t>.</w:t></w:r></w:p>
  </w:body>
</w:document>"""
    endnotes_xml = """<?xml version="1.0" encoding="UTF-8"?>
<w:endnotes xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:endnote w:id="-1"><w:p><w:r><w:t>separator</w:t></w:r></w:p></w:endnote>
  <w:endnote w:id="2"><w:p><w:r><w:t>Endnote body.</w:t></w:r></w:p></w:endnote>
</w:endnotes>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/endnotes.xml", endnotes_xml)


class StubOCR:
    name = "stub"
    identity = "stub-cli-v1"

    def __init__(self) -> None:
        self.page_indexes: list[int] = []

    def extract_page(
        self,
        pdf_path: Path,
        page_index: int,
        *,
        width: float,
        height: float,
    ) -> list[OCRLine]:
        self.page_indexes.append(page_index)
        return [OCRLine("OCR recovered body text.", [72, 72, 300, 90], 0.95)]


class EngineTests(unittest.TestCase):
    def test_alr_adapter_omits_notes_without_both_body_and_reference(self) -> None:
        document = SimpleNamespace(
            document_id="doc",
            source_sha256="a" * 64,
            paragraphs=[],
            footnotes=[
                SimpleNamespace(
                    pair_id="usable",
                    body="Usable note.",
                    reference_line_id="body-line",
                    body_line_ids=["note-line"],
                ),
                SimpleNamespace(
                    pair_id="no-body",
                    body="2",
                    reference_line_id="body-line",
                    body_line_ids=[],
                ),
                SimpleNamespace(
                    pair_id="no-reference",
                    body="Unanchored note.",
                    reference_line_id=None,
                    body_line_ids=["note-line"],
                ),
            ],
            metadata={},
            lines=[],
        )

        payload = to_alr_payload(document)

        self.assertEqual({1: "Usable note."}, payload["footnotes"])
        self.assertEqual([1], payload["footnote_order"])
        self.assertEqual(1, payload["metadata"]["legalpdf_usable_footnotes"])
        self.assertEqual(
            2,
            payload["metadata"]["legalpdf_omitted_unusable_footnotes"],
        )

    def test_toa_adapter_preserves_clean_offsets_for_multiple_note_anchors(
        self,
    ) -> None:
        first_marker = "⟦FN:first⟧"
        second_marker = "⟦FN:second⟧"
        text = f"Alpha{first_marker} beta{second_marker} gamma."
        document = SimpleNamespace(
            paragraphs=[
                SimpleNamespace(
                    id="paragraph-1",
                    text=text,
                    anchors=[
                        {"pair_id": "first", "offset": text.index(first_marker)},
                        {"pair_id": "second", "offset": text.index(second_marker)},
                    ],
                )
            ],
            footnotes=[
                SimpleNamespace(pair_id="first", body="First note."),
                SimpleNamespace(pair_id="second", body="Second note."),
            ],
        )

        units = to_toa_text_units(document)

        self.assertEqual("Alpha beta gamma.", units[0]["text"])
        self.assertEqual([(1, 5), (2, 10)], units[0]["footnote_refs"])

    def test_native_extraction_normalizes_skia_zero_width_boundaries(self) -> None:
        class NativePage:
            rect = SimpleNamespace(width=612, height=792)

            @staticmethod
            def get_text(_kind: str, **_kwargs: object) -> object:
                if _kind == "words":
                    return [
                        (72, 90, 102, 105, "Words", 0, 0, 0),
                        (104, 90, 125, 105, "with", 0, 0, 1),
                        (125, 90, 165, 105, "styles", 0, 0, 2),
                        (165, 88, 170, 99, "1", 0, 0, 3),
                    ]
                return {
                    "blocks": [
                        {
                            "type": 0,
                            "lines": [
                                {
                                    "bbox": (72, 90, 180, 105),
                                    "spans": [
                                        {
                                            "text": "\ufeff\u200bWords\u200b\u200bwith\u200b",
                                            "font": "Arial",
                                            "size": 10,
                                            "flags": 0,
                                            "bbox": (72, 90, 125, 105),
                                        },
                                        {
                                            "text": "\ufeff\u200bstyles\u200b",
                                            "font": "Arial-Italic",
                                            "size": 10,
                                            "flags": 0,
                                            "bbox": (125, 90, 165, 105),
                                        },
                                        {
                                            "text": "1",
                                            "font": "Arial",
                                            "size": 6,
                                            "flags": fitz.TEXT_FONT_SUPERSCRIPT,
                                            "bbox": (165, 88, 170, 99),
                                        },
                                    ],
                                }
                            ],
                        }
                    ]
                }

        lines, _ = _extract_native_page(
            NativePage(), page_index=0, global_line_offset=0
        )

        self.assertEqual("Words with styles1", lines[0].text)
        self.assertEqual(
            ["Words with", "styles", "1"],
            [span.text for span in lines[0].spans],
        )
        self.assertEqual(
            [(0, 10), (11, 17), (17, 18)],
            [(span.start, span.end) for span in lines[0].spans],
        )
        self.assertTrue(lines[0].spans[-1].superscript)
        self.assertEqual(
            [
                ("Words", 0, 5, [72.0, 90.0, 102.0, 105.0]),
                ("with", 6, 10, [104.0, 90.0, 125.0, 105.0]),
                ("styles", 11, 17, [125.0, 90.0, 165.0, 105.0]),
                ("1", 17, 18, [165.0, 88.0, 170.0, 99.0]),
            ],
            [
                (word.text, word.start, word.end, word.bbox)
                for word in lines[0].words
            ],
        )

    def test_sections_span_to_next_heading_with_locator_aliases(self) -> None:
        paragraphs = [
            Paragraph("preamble", 0, "body", "Preamble.", ["line-0"]),
            Paragraph(
                "heading-1",
                0,
                "heading",
                "Section 7 — Rights",
                ["line-1"],
            ),
            Paragraph(
                "body-1",
                0,
                "body",
                "First body.\u27e6FN:pair-1\u27e7",
                ["line-2"],
            ),
            Paragraph("body-2", 1, "body", "Second body.", ["line-3"]),
            Paragraph(
                "heading-2",
                1,
                "heading",
                "Schedule A Forms",
                ["line-4"],
            ),
            Paragraph("body-3", 2, "body", "Last body.", ["line-5"]),
        ]

        sections = _build_sections(paragraphs)

        self.assertEqual(2, len(sections))
        self.assertEqual(
            ["heading-1", "body-1", "body-2"],
            sections[0].paragraph_ids,
        )
        self.assertEqual([0, 1], sections[0].page_indexes)
        self.assertEqual("section", sections[0].locator_kind)
        self.assertEqual("7", sections[0].locator)
        self.assertEqual(
            ["7", "section 7", "Section 7 — Rights"],
            sections[0].aliases,
        )
        self.assertEqual(
            "Section 7 — Rights\n\nFirst body.\n\nSecond body.",
            sections[0].text,
        )
        self.assertEqual(["heading-2", "body-3"], sections[1].paragraph_ids)
        self.assertEqual("schedule", sections[1].locator_kind)
        self.assertEqual(
            ["A", "schedule A", "Schedule A Forms"],
            sections[1].aliases,
        )

    def test_printed_page_labels_fail_closed_on_conflict(self) -> None:
        header = synthetic_line(
            "header",
            1,
            "iv",
            [72, 20, 90, 35],
            9,
            order=1,
            region_type="header",
        )
        footer = synthetic_line(
            "footer",
            1,
            "12",
            [72, 760, 90, 775],
            9,
            order=2,
            region_type="footer",
        )
        page = Page(
            id="p0001",
            index=0,
            number=1,
            width=612,
            height=792,
            lines=[header, footer],
            regions=[],
        )

        diagnostics = _assign_printed_page_labels([page])

        self.assertIsNone(page.printed_label)
        self.assertEqual(
            ["PRINTED_PAGE_LABEL_AMBIGUOUS"],
            [diagnostic.code for diagnostic in diagnostics],
        )
        header.text = "Running head"
        self.assertEqual([], _assign_printed_page_labels([page]))
        self.assertEqual(("12", "footer", "footer"), (
            page.printed_label,
            page.printed_label_source,
            page.printed_label_line_id,
        ))

    def test_repeated_page_numbers_do_not_consume_detached_note_labels(self) -> None:
        pages = []
        for page_number in range(1, 4):
            note_label = synthetic_line(
                f"note-{page_number}",
                page_number,
                str(page_number),
                [72, 720, 78, 730],
                6,
                order=1,
            )
            page_number_line = synthetic_line(
                f"page-{page_number}",
                page_number,
                str(page_number),
                [300, 770, 312, 782],
                9,
                order=2,
            )
            body = synthetic_line(
                f"body-{page_number}",
                page_number,
                "Ordinary body text establishes the page font size.",
                [72, 100, 320, 115],
                10,
                order=3,
            )
            pages.append(
                Page(
                    id=f"p{page_number:04d}",
                    index=page_number - 1,
                    number=page_number,
                    width=612,
                    height=792,
                    lines=[note_label, page_number_line, body],
                    regions=[],
                )
            )

        _mark_repeated_furniture(pages)

        self.assertEqual(
            ["unknown", "unknown", "unknown"],
            [page.lines[0].region_type for page in pages],
        )
        self.assertEqual(
            ["footer", "footer", "footer"],
            [page.lines[1].region_type for page in pages],
        )

    def test_detached_pure_label_starts_footnote_region(self) -> None:
        body = synthetic_line(
            "body",
            1,
            "Body text.",
            [72, 100, 180, 115],
            10,
            order=1,
        )
        label = synthetic_line(
            "label",
            1,
            "1",
            [72, 750, 78, 760],
            5,
            order=2,
        )
        note_body = synthetic_line(
            "note-body",
            1,
            "Authority supporting the proposition.",
            [90, 750, 320, 765],
            8,
            order=3,
        )
        page = Page(
            id="p0001",
            index=0,
            number=1,
            width=612,
            height=792,
            lines=[body, label, note_body],
            regions=[],
        )

        _classify_page(page, 700)

        self.assertEqual("body", body.region_type)
        self.assertEqual("footnote", label.region_type)
        self.assertEqual("footnote", note_body.region_type)

    def test_detached_superscript_becomes_zero_width_anchor_not_body(self) -> None:
        body = synthetic_line(
            "body",
            1,
            "Held.",
            [100, 90, 180, 105],
            10,
            order=1,
        )
        glyph = synthetic_line(
            "glyph",
            1,
            "1",
            [182, 88, 187, 101],
            6,
            order=2,
        )
        far_glyph = synthetic_line(
            "far-glyph",
            1,
            "2",
            [300, 88, 305, 101],
            6,
            order=3,
        )
        note = synthetic_line(
            "note",
            1,
            "1 Note text.",
            [100, 660, 220, 675],
            8,
            order=4,
            spans=[
                Span(
                    id="note-label",
                    text="1",
                    bbox=[100, 658, 105, 668],
                    size=5,
                    start=0,
                    end=1,
                ),
                Span(
                    id="note-text",
                    text="Note text.",
                    bbox=[110, 660, 220, 675],
                    size=8,
                    start=2,
                    end=12,
                ),
            ],
        )
        page = Page(
            id="p0001",
            index=0,
            number=1,
            width=612,
            height=792,
            lines=[body, glyph, far_glyph, note],
            regions=[],
        )

        _associate_detached_references(page, 650)
        _classify_page(page, 650)
        _order_page(page)
        _build_regions(page)
        paragraphs, footnotes, _, _ = _derive([page])

        self.assertTrue(glyph.exclude_from_body)
        self.assertFalse(far_glyph.exclude_from_body)
        self.assertEqual(5, body.detached_references[0]["start_offset"])
        self.assertEqual("Note text.", footnotes[0].body)
        held = next(paragraph for paragraph in paragraphs if "Held." in paragraph.text)
        self.assertEqual(
            "Held.\u27e6FN:fnv2-pair-LEGALPDF-document-000001\u27e7",
            held.text,
        )
        self.assertNotIn("glyph", {line_id for p in paragraphs for line_id in p.line_ids})

        # Codex may reclassify every source row as body; the persisted exclusion wins.
        glyph.region_type = "body"
        _build_regions(page)
        repaired_paragraphs, _, _, _ = _derive([page])
        self.assertNotIn(
            "glyph", {line_id for p in repaired_paragraphs for line_id in p.line_ids}
        )

    def test_pairing_rejects_date_tail_and_bottom_right_false_labels(self) -> None:
        body = synthetic_line(
            "body",
            1,
            "Held1",
            [72, 100, 110, 115],
            10,
            order=1,
            spans=[
                Span(
                    id="body-text",
                    text="Held",
                    bbox=[72, 100, 100, 115],
                    size=10,
                    start=0,
                    end=4,
                ),
                Span(
                    id="body-ref",
                    text="1",
                    bbox=[100, 98, 105, 108],
                    size=6,
                    superscript=True,
                    start=4,
                    end=5,
                ),
            ],
        )
        real_note = synthetic_line(
            "real-note",
            1,
            "1 Real note.",
            [72, 650, 180, 665],
            8,
            order=2,
            spans=[
                Span(
                    id="real-label",
                    text="1",
                    bbox=[72, 648, 77, 658],
                    size=5,
                    start=0,
                    end=1,
                ),
                Span(
                    id="real-text",
                    text="Real note.",
                    bbox=[82, 650, 180, 665],
                    size=8,
                    start=2,
                    end=12,
                ),
            ],
        )
        date_tail = synthetic_line(
            "date-tail",
            1,
            "19, 2024), online.",
            [72, 680, 210, 695],
            8,
            order=3,
        )
        page_number = synthetic_line(
            "page-number",
            1,
            "2 Page",
            [400, 730, 450, 742],
            5,
            order=4,
        )
        page = Page(
            id="p0001",
            index=0,
            number=1,
            width=612,
            height=792,
            lines=[body, real_note, date_tail, page_number],
            regions=[],
        )

        _classify_page(page, None)
        markers, _ = pair_markers(page.lines)

        self.assertTrue(date_tail.suppress_footnote_label)
        self.assertTrue(page_number.suppress_footnote_label)
        labels = [marker for marker in markers if marker["role"] == "fn_label"]
        self.assertEqual(["real-note"], [marker["line_id"] for marker in labels])

    def test_endnote_and_right_column_guards_preserve_numbered_body_and_notes(
        self,
    ) -> None:
        numbered = [
            synthetic_line(
                f"body-{number}",
                1,
                f"{number} Numbered body paragraph with ordinary text.",
                [72, 100 + number * 25, 300, 115 + number * 25],
                8,
                order=number,
            )
            for number in range(1, 4)
        ]
        body_page = Page(
            id="p0001",
            index=0,
            number=1,
            width=612,
            height=792,
            lines=numbered,
            regions=[],
        )
        _classify_page(body_page, None)
        self.assertTrue(all(line.region_type == "body" for line in numbered))

        continued_endnotes = [
            synthetic_line(
                "citation-before-endnotes",
                2,
                "2 U.S. 1, continued citation.",
                [72, 80, 300, 95],
                8,
                order=1,
            ),
            *[
                synthetic_line(
                    f"endnote-{number}",
                    2,
                    f"{number} Continued endnote text.",
                    [72, 90 + number * 25, 300, 105 + number * 25],
                    8,
                    order=number,
                )
                for number in range(2, 5)
            ],
        ]
        endnote_page = Page(
            id="p0002",
            index=1,
            number=2,
            width=612,
            height=792,
            lines=continued_endnotes,
            regions=[],
        )
        _classify_page(
            endnote_page,
            None,
            continuing_endnotes=True,
            expected_endnote=2,
            continuing_endnote_size=8,
        )
        self.assertTrue(
            all(
                line.region_type == "footnote"
                and line.note_region_mode == "endnote"
                for line in continued_endnotes
            )
        )
        endnote_markers, _ = pair_markers(continued_endnotes)
        self.assertEqual(
            ["endnote-2", "endnote-3", "endnote-4"],
            [
                marker["line_id"]
                for marker in endnote_markers
                if marker["role"] == "fn_label"
            ],
        )
        true_before_collision = [
            synthetic_line(
                "true-label-2",
                2,
                "2 True endnote text.",
                [72, 80, 300, 95],
                8,
                order=1,
            ),
            synthetic_line(
                "citation-after-label",
                2,
                "2 U.S. (2 Dall.) 419, citation within the note.",
                [72, 105, 350, 120],
                8,
                order=2,
            ),
            synthetic_line(
                "true-label-3",
                2,
                "3 Next endnote text.",
                [72, 130, 300, 145],
                8,
                order=3,
            ),
        ]
        collision_page = Page(
            id="p0002-collision",
            index=1,
            number=2,
            width=612,
            height=792,
            lines=true_before_collision,
            regions=[],
        )
        _classify_page(
            collision_page,
            None,
            continuing_endnotes=True,
            expected_endnote=2,
            continuing_endnote_size=8,
        )
        collision_markers, _ = pair_markers(true_before_collision)
        self.assertEqual(
            ["true-label-2", "true-label-3"],
            [
                marker["line_id"]
                for marker in collision_markers
                if marker["role"] == "fn_label"
            ],
        )

        header_and_endnotes = [
            synthetic_line(
                "unrecognized-running-header",
                2,
                "SUPREME COURT OF CANADA",
                [180, 20, 430, 35],
                8,
                order=1,
            ),
            synthetic_line(
                "header-note-2",
                2,
                "2 Continued endnote text.",
                [72, 80, 300, 95],
                8,
                order=2,
            ),
            synthetic_line(
                "header-note-3",
                2,
                "3 Next endnote text.",
                [72, 110, 300, 125],
                8,
                order=3,
            ),
        ]
        header_page = Page(
            id="p0002-header",
            index=1,
            number=2,
            width=612,
            height=792,
            lines=header_and_endnotes,
            regions=[],
        )
        _classify_page(
            header_page,
            None,
            continuing_endnotes=True,
            expected_endnote=2,
            continuing_endnote_size=8,
        )
        self.assertTrue(
            all(
                line.region_type == "footnote"
                and line.note_region_mode == "endnote"
                for line in header_and_endnotes[1:]
            )
        )
        self.assertEqual("header", header_and_endnotes[0].region_type)
        prior_note = synthetic_line(
            "prior-note",
            1,
            "1 First endnote body.",
            [72, 680, 300, 695],
            8,
            order=1,
            region_type="footnote",
        )
        prior_note.note_region_mode = "endnote"
        header_markers, _ = pair_markers(
            [prior_note, *header_and_endnotes]
        )
        header_footnotes, _, _ = _materialize_footnotes(
            [prior_note, *header_and_endnotes], header_markers
        )
        self.assertEqual("First endnote body.", header_footnotes[0].body)

        label_free = [
            synthetic_line(
                f"continued-{number}",
                3,
                "Continuation text without a repeated endnote label.",
                [72, 100 + number * 25, 300, 115 + number * 25],
                8,
                order=number,
            )
            for number in (1, 2)
        ]
        label_free_page = Page(
            id="p0003",
            index=2,
            number=3,
            width=612,
            height=792,
            lines=label_free,
            regions=[],
        )
        _classify_page(
            label_free_page,
            None,
            continuing_endnotes=True,
            expected_endnote=5,
            continuing_endnote_size=8,
        )
        self.assertTrue(
            all(line.note_region_mode == "endnote" for line in label_free)
        )

        bibliography = [
            synthetic_line(
                "bibliography-heading",
                4,
                "BIBLIOGRAPHY",
                [72, 80, 300, 95],
                8,
                order=1,
            ),
            synthetic_line(
                "bibliography-entry",
                4,
                "Smith, A Treatise on Legal Interpretation.",
                [72, 110, 400, 125],
                8,
                order=2,
            ),
        ]
        bibliography_page = Page(
            id="p0004",
            index=3,
            number=4,
            width=612,
            height=792,
            lines=bibliography,
            regions=[],
        )
        _classify_page(
            bibliography_page,
            None,
            continuing_endnotes=True,
            expected_endnote=5,
            continuing_endnote_size=8,
        )
        self.assertTrue(
            all(line.region_type != "footnote" for line in bibliography)
        )
        for heading in (
            "ACKNOWLEDGMENTS",
            "CERTIFICATE OF SERVICE",
            "ABOUT THE AUTHORS",
        ):
            post_notes = [
                synthetic_line(
                    f"{heading}-heading",
                    4,
                    heading,
                    [72, 80, 300, 95],
                    8,
                    order=1,
                ),
                synthetic_line(
                    f"{heading}-body",
                    4,
                    "Ordinary post-note section text.",
                    [72, 110, 400, 125],
                    8,
                    order=2,
                ),
            ]
            post_notes_page = Page(
                id=f"p-{heading}",
                index=3,
                number=4,
                width=612,
                height=792,
                lines=post_notes,
                regions=[],
            )
            _classify_page(
                post_notes_page,
                None,
                continuing_endnotes=True,
                expected_endnote=5,
                continuing_endnote_size=8,
            )
            self.assertTrue(
                all(line.region_type != "footnote" for line in post_notes),
                heading,
            )

        appendix = [
            synthetic_line(
                "appendix-heading",
                5,
                "APPENDIX",
                [72, 80, 300, 95],
                12,
                order=1,
            ),
            *[
                synthetic_line(
                    f"appendix-{number}",
                    5,
                    f"{number} {'Scope' if number == 5 else 'Definitions'}",
                    [72, 100 + number * 10, 300, 115 + number * 10],
                    8,
                    order=number - 3,
                )
                for number in (5, 6)
            ],
        ]
        appendix_page = Page(
            id="p0005",
            index=4,
            number=5,
            width=612,
            height=792,
            lines=appendix,
            regions=[],
        )
        _classify_page(
            appendix_page,
            None,
            continuing_endnotes=True,
            expected_endnote=5,
            continuing_endnote_size=8,
        )
        self.assertTrue(all(line.region_type != "footnote" for line in appendix))

        body = synthetic_line(
            "body",
            6,
            "Ordinary body text.",
            [72, 100, 250, 115],
            10,
            order=1,
        )
        right_note = synthetic_line(
            "right-note",
            6,
            "1 Right-column note.",
            [330, 725, 500, 740],
            8,
            order=2,
            spans=[
                Span(
                    id="right-label",
                    text="1",
                    bbox=[330, 723, 335, 733],
                    size=5,
                    start=0,
                    end=1,
                ),
                Span(
                    id="right-text",
                    text="Right-column note.",
                    bbox=[340, 725, 500, 740],
                    size=8,
                    start=2,
                    end=20,
                ),
            ],
        )
        note_page = Page(
            id="p0006",
            index=5,
            number=6,
            width=612,
            height=792,
            lines=[body, right_note],
            regions=[],
        )
        _classify_page(note_page, 700)
        markers, _ = pair_markers(note_page.lines)
        self.assertEqual("footnote", right_note.region_type)
        self.assertFalse(right_note.suppress_footnote_label)
        self.assertEqual(
            ["right-note"],
            [
                marker["line_id"]
                for marker in markers
                if marker["role"] == "fn_label"
            ],
        )

    def test_unmatched_detached_reference_is_diagnosed(self) -> None:
        body = synthetic_line(
            "body",
            1,
            "Held.",
            [100, 90, 180, 105],
            10,
            order=1,
        )
        glyph = synthetic_line(
            "glyph",
            1,
            "9",
            [182, 88, 187, 101],
            6,
            order=2,
        )
        page = Page(
            id="p0001",
            index=0,
            number=1,
            width=612,
            height=792,
            lines=[body, glyph],
            regions=[],
        )
        _associate_detached_references(page, None)
        _classify_page(page, None)
        _order_page(page)
        _build_regions(page)
        paragraphs, footnotes, diagnostics, _ = _derive(
            [page]
        )

        self.assertEqual([], footnotes)
        self.assertTrue(
            any(
                diagnostic.code == "FOOTNOTE_UNMATCHED_REFERENCE"
                for diagnostic in diagnostics
            )
        )
        self.assertNotIn("glyph", {line for p in paragraphs for line in p.line_ids})

    def test_note_order_uses_word_ink_for_raised_labels(self) -> None:
        body = synthetic_line(
            "note-body",
            1,
            "Note body.",
            [68.5, 531.9, 150.0, 539.9],
            8,
            order=1,
            region_type="footnote",
        )
        body.words = [Word("body-word", "Note", [68.5, 534.3, 90.0, 540.0], 0, 4)]
        label = synthetic_line(
            "note-label",
            1,
            "1",
            [54.1, 532.15, 56.4, 536.75],
            4.6,
            order=2,
            region_type="footnote",
        )
        label.words = [Word("label-word", "1", [54.6, 533.6, 55.8, 536.8], 0, 1)]
        page = Page(
            id="p0001",
            index=0,
            number=1,
            width=432,
            height=670,
            lines=[body, label],
            regions=[],
        )

        _order_page(page)

        self.assertEqual(["note-label", "note-body"], [line.id for line in page.lines])

    def test_pdf_backend_groups_lines_until_a_paragraph_gap(self) -> None:
        lines = [
            synthetic_line("first", 1, "First", [36, 50, 300, 60], 10, order=1),
            synthetic_line("continuation", 1, "line", [36, 62, 300, 72], 10, order=2),
            synthetic_line("next", 1, "Next", [50, 86, 300, 96], 10, order=3),
        ]

        _assign_block_indexes(lines)

        self.assertEqual([1, 1, 2], [line.block_index for line in lines])

    def test_pdf_backend_does_not_fuse_distant_columns_after_short_text(self) -> None:
        def item(text: str, x: float, width: float) -> SimpleNamespace:
            return SimpleNamespace(
                text=text,
                x=x,
                y=604.7,
                width=width,
                height=10.0,
                font_size=10.0,
                item_type="text",
            )

        groups = _group_inspector_items(
            [
                item("short left-column tail", 72.0, 213.5),
                item("18", 318.0, 19.6),
                item("Voir, dans le registre", 340.5, 120.0),
            ]
        )

        self.assertEqual(
            [["short left-column tail"], ["18", "Voir, dans le registre"]],
            [[part.text for part in group] for group in groups],
        )

    def test_bundled_pairer_keeps_global_order_and_refreshes_detached_metadata(
        self,
    ) -> None:
        body = synthetic_line(
            "body",
            1,
            "Held.",
            [72, 100, 150, 115],
            10,
            order=1,
            region_type="body",
        )
        body.detached_references.append(
            {
                "note_id": "1",
                "selected_text": "1",
                "start_offset": 5,
                "end_offset": 5,
            }
        )
        excluded = synthetic_line(
            "glyph",
            1,
            "1",
            [152, 98, 157, 108],
            6,
            order=2,
        )
        excluded.exclude_from_body = True
        label = synthetic_line(
            "label",
            1,
            "1 Note.",
            [72, 650, 140, 665],
            8,
            order=3,
            region_type="footnote",
        )
        seen_orders: list[int] = []

        def pair(rows: list[dict[str, object]]):
            seen_orders.extend(int(row["reading_order_index"]) for row in rows)
            return (
                [
                    {
                        "role": "fn_label",
                        "note_id": "1",
                        "marker_id": "label-1",
                        "materialized_pair_id": "pair-1",
                        "materialized_note_id": "1",
                        "materialized_pair_status": "label_only",
                        "reading_order_index": 3,
                        "pdf_page": 1,
                        "line_id": "label",
                        "end_offset": 1,
                        "article_sequence_context": {"endnote_mode": False},
                    }
                ],
                {"marker_count": 1, "pair_count": 0},
            )

        page = Page(
            id="p0001",
            index=0,
            number=1,
            width=612,
            height=792,
            lines=[body, excluded, label],
            regions=[],
        )
        with patch("legalpdf.core.pair_article_footnotes", side_effect=pair):
            markers, summary = _pair_markers([page])

        self.assertEqual([1, 3], seen_orders)
        self.assertEqual(2, summary["marker_count"])
        self.assertEqual(1, summary["pair_count"])
        self.assertEqual(0, summary["materialized_label_only_count"])
        self.assertEqual(
            {"fn_label": 1, "fn_ref": 1}, summary["role_counts"]
        )
        self.assertEqual(
            ["fn_label", "fn_ref"],
            sorted(marker["role"] for marker in markers),
        )

    def test_distant_endnote_reference_remains_supported(self) -> None:
        body = synthetic_line(
            "body",
            1,
            "Held1",
            [72, 100, 110, 115],
            10,
            order=1,
            region_type="body",
            spans=[
                Span(
                    id="held",
                    text="Held",
                    bbox=[72, 100, 100, 115],
                    size=10,
                    start=0,
                    end=4,
                ),
                Span(
                    id="endnote-ref",
                    text="1",
                    bbox=[100, 98, 105, 108],
                    size=6,
                    superscript=True,
                    start=4,
                    end=5,
                ),
            ],
        )
        label = synthetic_line(
            "endnote",
            3,
            "1 Endnote text.",
            [72, 110, 190, 125],
            8,
            order=2,
            region_type="footnote",
        )

        markers, _ = pair_markers([body, label])
        footnotes, _, _ = _materialize_footnotes([body, label], markers)

        self.assertEqual(1, footnotes[0].reference_page)
        self.assertEqual([3], footnotes[0].body_pages)
        self.assertEqual("Endnote text.", footnotes[0].body)

    def test_superscript_refs_attached_to_statute_citations_are_kept(self) -> None:
        text = (
            "RSA 2009,11 the Occupational Health and Safety Act,12 "
            "the Workers Compensation Act13."
        )
        spans = [
            Span(
                id="citation-1",
                text="RSA 2009,",
                bbox=[72, 100, 125, 115],
                size=10,
                start=0,
                end=9,
            ),
            Span(
                id="ref-11",
                text="11",
                bbox=[125, 98, 134, 108],
                size=6,
                superscript=True,
                start=9,
                end=11,
            ),
            Span(
                id="citation-2",
                text=" the Occupational Health and Safety Act,",
                bbox=[134, 100, 330, 115],
                size=10,
                start=11,
                end=51,
            ),
            Span(
                id="ref-12",
                text="12",
                bbox=[330, 98, 339, 108],
                size=6,
                superscript=True,
                start=51,
                end=53,
            ),
            Span(
                id="citation-3",
                text=" the Workers Compensation Act",
                bbox=[339, 100, 475, 115],
                size=10,
                start=53,
                end=82,
            ),
            Span(
                id="ref-13",
                text="13",
                bbox=[475, 98, 484, 108],
                size=6,
                superscript=True,
                start=82,
                end=84,
            ),
            Span(
                id="period",
                text=".",
                bbox=[484, 100, 488, 115],
                size=10,
                start=84,
                end=85,
            ),
        ]
        body = synthetic_line(
            "body",
            1,
            text,
            [72, 98, 488, 115],
            10,
            order=1,
            region_type="body",
            spans=spans,
        )
        body.suppress_footnote_label = True
        labels = [
            synthetic_line(
                f"label-{value}",
                1,
                f"{value} Citation {value}.",
                [72, 650 + (value - 11) * 10, 180, 658 + (value - 11) * 10],
                8,
                order=value - 9,
                region_type="footnote",
            )
            for value in (11, 12, 13)
        ]

        markers, _ = pair_markers([body, *labels])
        refs = [
            marker
            for marker in markers
            if marker["role"] == "fn_ref"
        ]

        self.assertEqual(["11", "12", "13"], [marker["note_id"] for marker in refs])

    def test_same_page_duplicate_one_is_not_a_restart(self) -> None:
        body = synthetic_line(
            "body",
            1,
            "Held1",
            [72, 100, 110, 115],
            10,
            order=1,
            region_type="body",
            spans=[
                Span(
                    id="body-text",
                    text="Held",
                    bbox=[72, 100, 100, 115],
                    size=10,
                    start=0,
                    end=4,
                ),
                Span(
                    id="body-ref",
                    text="1",
                    bbox=[100, 98, 105, 108],
                    size=6,
                    superscript=True,
                    start=4,
                    end=5,
                ),
            ],
        )
        labels = [
            synthetic_line(
                f"label-{index}",
                1,
                f"1 Candidate {index}.",
                [72, 640 + index * 20, 180, 655 + index * 20],
                8,
                order=index + 1,
                region_type="footnote",
            )
            for index in (1, 2)
        ]
        for label in labels:
            label.note_region_mode = "footnote"

        markers, summary = pair_markers([body, *labels])

        self.assertEqual(1, summary["pair_count"])
        self.assertEqual(
            1, sum(marker["role"] == "fn_label" for marker in markers)
        )

    def test_footnote_body_is_bounded_but_keeps_adjacent_continuation(self) -> None:
        first = synthetic_line(
            "note-1",
            1,
            "1 First text.",
            [72, 650, 180, 665],
            8,
            order=1,
            region_type="footnote",
        )
        unrelated = synthetic_line(
            "unrelated",
            2,
            "Unrelated footnote-region text.",
            [72, 650, 240, 665],
            8,
            order=2,
            region_type="footnote",
        )
        marker = {
            "role": "fn_label",
            "note_id": "1",
            "materialized_pair_id": "pair-1",
            "reading_order_index": 1,
            "pdf_page": 1,
            "line_id": "note-1",
            "end_offset": 1,
        }

        footnotes, _, _ = _materialize_footnotes([first, unrelated], [marker])
        self.assertEqual("First text.", footnotes[0].body)
        endnote, _, _ = _materialize_footnotes(
            [first, unrelated], [{**marker, "endnote_mode": True}]
        )
        self.assertEqual(
            "First text. Unrelated footnote-region text.", endnote[0].body
        )
        unrelated.note_region_mode = "footnote_continuation"
        continued_last, _, _ = _materialize_footnotes(
            [first, unrelated], [marker]
        )
        self.assertEqual(
            "First text. Unrelated footnote-region text.",
            continued_last[0].body,
        )
        unrelated.note_region_mode = ""

        second = synthetic_line(
            "note-2",
            2,
            "2 Second text.",
            [72, 680, 180, 695],
            8,
            order=3,
            region_type="footnote",
        )
        second_marker = {
            "role": "fn_label",
            "note_id": "2",
            "materialized_pair_id": "pair-2",
            "reading_order_index": 3,
            "pdf_page": 2,
            "line_id": "note-2",
            "end_offset": 1,
        }
        continued, _, _ = _materialize_footnotes(
            [first, unrelated, second], [marker, second_marker]
        )
        self.assertEqual(
            "First text. Unrelated footnote-region text.", continued[0].body
        )

    def test_footnote_body_stops_before_separate_lower_page_block(self) -> None:
        note = synthetic_line(
            "note",
            1,
            "1 Note text.",
            [54, 580, 180, 590],
            8,
            order=1,
            region_type="footnote",
        )
        note.region_id = "note-region"
        license_text = synthetic_line(
            "license",
            1,
            "This work is licensed separately.",
            [127, 607, 300, 617],
            8,
            order=2,
            region_type="footnote",
        )
        license_text.region_id = "license-region"
        marker = {
            "role": "fn_label",
            "note_id": "1",
            "materialized_pair_id": "pair-1",
            "reading_order_index": 1,
            "pdf_page": 1,
            "line_id": "note",
            "end_offset": 1,
        }

        footnotes, _, _ = _materialize_footnotes(
            [note, license_text], [marker]
        )

        self.assertEqual("Note text.", footnotes[0].body)

    def test_separator_prefers_rule_immediately_above_note_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "table-before-notes.pdf"
            document = fitz.open()
            page = document.new_page(width=432, height=648)
            page.insert_text((36, 400), "Body text.", fontsize=10)
            page.draw_line((31, 526), (135, 526), width=0.48)
            page.draw_line((36, 610), (180, 610), width=0.72)
            page.insert_text((54, 625), "71 Note text.", fontsize=8)
            document.save(path)
            document.close()

            with fitz.open(path) as source:
                raw_page = source.load_page(0)
                lines, _ = _extract_native_page(
                    raw_page,
                    page_index=0,
                    global_line_offset=0,
                )
                separator = _separator_y(raw_page, lines)

        self.assertIsNotNone(separator)
        self.assertAlmostEqual(610, separator or 0, delta=1)

    def test_local_parse_lookup_and_adapters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "legal.pdf"
            make_legal_pdf(pdf)
            first = parse_pdf(pdf)
            self.assertEqual(2, first.page_count)
            self.assertEqual(2, len(first.footnotes))
            self.assertFalse(first.provenance["cache_hit"])
            self.assertEqual("First footnote body.", first.footnotes[0].body)
            self.assertIn("first proposition", first.footnotes[0].sentence_proposition)
            self.assertNotIn("⟦FN:", first.footnotes[0].sentence_proposition)
            self.assertNotIn(
                "second proposition", first.footnotes[0].sentence_proposition
            )
            self.assertEqual("ambiguous", lookup_footnote(first, "1").status)
            found = lookup_footnote(first, "1", occurrence=2)
            self.assertEqual("found", found.status)
            self.assertEqual("Restarted footnote body.", found.footnote.body)

            self.assertEqual(
                {1, 2}, set(to_alr_payload(first)["footnotes"])
            )
            toa = to_toa_text_units(first)
            self.assertTrue(any(unit["kind"] == "body" for unit in toa))
            self.assertEqual(2, sum(unit["kind"] == "footnote" for unit in toa))
            self.assertEqual(
                [1, 2],
                [
                    note
                    for unit in toa
                    if unit["kind"] == "body"
                    for note, _offset in unit["footnote_refs"]
                ],
            )
            self.assertFalse(
                any("⟦FN:" in unit["text"] for unit in toa if unit["kind"] == "body")
            )

    def test_empty_pdf_requires_ocr_and_provider_is_source_neutral(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "empty.pdf"
            make_empty_pdf(pdf)
            without = parse_pdf(pdf, cache_dir=root / "cache-empty")
            self.assertEqual("ocr_required", without.status)
            with_ocr = parse_pdf(
                pdf, cache_dir=root / "cache-ocr", ocr_provider=StubOCR()
            )
            self.assertEqual("ocr", with_ocr.pages[0].source)
            self.assertIn("OCR recovered", with_ocr.text)

    def test_ocr_only_visits_low_quality_pages_in_a_mixed_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "mixed.pdf"
            with fitz.open() as document:
                page = document.new_page(width=612, height=792)
                for index in range(8):
                    page.insert_text(
                        (72, 90 + index * 18),
                        f"Native legal text line {index + 1} with reliable embedded text.",
                        fontsize=11,
                    )
                document.new_page(width=612, height=792)
                document.save(pdf)
            provider = StubOCR()
            parsed = parse_pdf(
                pdf, cache_dir=root / "cache", ocr_provider=provider
            )

        self.assertEqual([1], provider.page_indexes)
        self.assertEqual(["native", "ocr"], [page.source for page in parsed.pages])

    def test_tesseract_provider_parses_tsv_through_the_command_boundary(self) -> None:
        tsv = "\n".join(
            [
                "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
                "5\t1\t1\t1\t1\t1\t150\t200\t100\t30\t96\tOCR",
                "5\t1\t1\t1\t1\t2\t260\t200\t180\t30\t94\trecovered",
            ]
        )
        command_results = [
            subprocess.CompletedProcess(
                ["tesseract", "--version"],
                0,
                stdout="tesseract 5.3.0\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                ["tesseract", "page.png"],
                0,
                stdout=tsv,
                stderr="",
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            pdf = Path(temporary) / "empty.pdf"
            make_empty_pdf(pdf)
            with patch(
                "legalpdf.ocr.subprocess.run", side_effect=command_results
            ) as invocation:
                provider = TesseractOCRProvider(
                    language="eng", dpi=180, psm=6
                )
                lines = provider.extract_page(
                    pdf, 0, width=612, height=792
                )

        self.assertEqual(2, invocation.call_count)
        command = invocation.call_args_list[1].args[0]
        self.assertEqual("tesseract", command[0])
        self.assertIn("stdout", command)
        self.assertEqual(
            ["-l", "eng", "--dpi", "180", "--psm", "6", "tsv"],
            command[-7:],
        )
        self.assertEqual("OCR recovered", lines[0].text)
        self.assertAlmostEqual(0.95, lines[0].confidence)
        self.assertTrue(provider.name.startswith("tesseract-cli-v1:tesseract 5.3.0"))
        self.assertEqual("tesseract-cli-v1:tesseract 5.3.0", provider.identity)
        self.assertGreater(lines[0].bbox[2], lines[0].bbox[0])
        self.assertGreater(lines[0].bbox[3], lines[0].bbox[1])

    def test_codex_repair_honors_the_shared_launcher_without_calling_it(self) -> None:
        with patch.dict(
            os.environ, {"CODEX_EXEC_COMMAND": "C:/pinned/codex.cmd"}
        ), patch("legalpdf.codex_repair.shutil.which") as which:
            self.assertEqual("C:/pinned/codex.cmd", _codex_command())
            which.assert_not_called()

        with patch.dict(os.environ, {}, clear=True), patch(
            "legalpdf.codex_repair.shutil.which", return_value="codex-fallback"
        ) as which:
            self.assertEqual("codex-fallback", _codex_command())
            which.assert_called_once_with("codex")

    def test_endnote_page_is_detected_above_the_page_midpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "endnotes.pdf"
            make_endnote_pdf(pdf)
            document = parse_pdf(pdf, cache_dir=root / "cache")
            self.assertEqual(["1", "2", "3"], [note.label for note in document.footnotes])
            self.assertEqual(
                ["First endnote.", "Second endnote.", "Third endnote."],
                [
                    unit["text"]
                    for unit in to_toa_text_units(document)
                    if unit["kind"] == "footnote"
                ],
            )
            self.assertTrue(
                all(
                    line.region_type == "footnote"
                    for line in document.pages[1].lines
                    if line.text[:1].isdigit()
                )
            )

    def test_engine_identity_covers_parser_helpers_and_grammar_tables(self) -> None:
        parser_files = {
            "core.py",
            "model.py",
            "column_order_arbiter.py",
            "footnote_separator_scan.py",
            "footnote_pairing.py",
            "footnote_pairing_support.py",
            "grammar_tables.py",
            "note_crossrefs.py",
            "ocr.py",
            "pdf_backend.py",
            "superscript_splice.py",
            "data/mcgill_reporters.json",
        }
        first = _engine_identity()

        self.assertEqual(
            parser_files | {"data/legal-grammar-tables/grammar-corpus.json"},
            set(first),
        )
        self.assertEqual(
            first["data/legal-grammar-tables/grammar-corpus.json"],
            hashlib.sha256(
                (
                    Path(__file__).resolve().parents[1]
                    / "data"
                    / "legal-grammar-tables"
                    / "grammar-corpus.json"
                ).read_bytes()
            ).hexdigest(),
        )

    def test_codex_repair_is_bounded_validated_and_cached(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "legal.pdf"
            make_legal_pdf(pdf, restarted=False)
            document = parse_pdf(pdf, cache_dir=root / "local")
            document.diagnostics.append(
                Diagnostic(
                    code="COLUMN_ORDER_UNCERTAIN",
                    severity="warning",
                    message="fixture",
                    page_index=0,
                )
            )
            line_ids = [line.id for line in document.pages[0].lines]
            response = {
                "pages": [
                    {
                        "page_index": 0,
                        "regions": [
                            {"region_type": "body", "line_ids": line_ids}
                        ],
                    }
                ],
            }
            with patch(
                "legalpdf.codex_repair._invoke",
                return_value=(response, {"input_tokens": 10, "output_tokens": 5}, 0.1),
            ) as invocation:
                repaired = improve(
                    document,
                    pdf,
                    model="test-model",
                    effort="max",
                    cache_dir=root / "codex",
                )
            invocation.assert_called_once()
            self.assertEqual("applied", repaired.repairs[-1].status)
            self.assertEqual(
                [line.text for line in document.pages[0].lines],
                [line.text for line in repaired.pages[0].lines],
            )
            with patch("legalpdf.codex_repair._invoke") as invocation:
                cached = improve(
                    document,
                    pdf,
                    model="test-model",
                    effort="max",
                    cache_dir=root / "codex",
                )
            invocation.assert_not_called()
            self.assertEqual("applied", cached.repairs[-1].status)

            response_path = next((root / "codex").glob("*/response.json"))
            metadata_path = response_path.with_name("metadata.json")
            response_path.write_text("{broken", encoding="utf-8")
            with patch(
                "legalpdf.codex_repair._invoke",
                return_value=(response, {}, 0.1),
            ) as invocation:
                improve(
                    document,
                    pdf,
                    model="test-model",
                    effort="max",
                    cache_dir=root / "codex",
                )
            invocation.assert_called_once()
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual("test-model", metadata["model"])
            self.assertRegex(metadata["response_sha256"], r"^[0-9a-f]{64}$")

            tampered_response = json.loads(
                response_path.read_text(encoding="utf-8")
            )
            tampered_response["pages"][0]["regions"][0][
                "region_type"
            ] = "heading"
            response_path.write_text(
                json.dumps(tampered_response), encoding="utf-8"
            )
            with patch(
                "legalpdf.codex_repair._invoke",
                return_value=(response, {}, 0.1),
            ) as invocation:
                improve(
                    document,
                    pdf,
                    model="test-model",
                    effort="max",
                    cache_dir=root / "codex",
                )
            invocation.assert_called_once()

            invalid_response = {
                "pages": [
                    {
                        "page_index": 0,
                        "regions": [
                            {
                                "region_type": "body",
                                "line_ids": line_ids[:-1],
                            }
                        ],
                    }
                ],
            }
            response_path.write_text(
                json.dumps(invalid_response), encoding="utf-8"
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["response_sha256"] = _stable_hash(invalid_response)
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with patch(
                "legalpdf.codex_repair._invoke",
                return_value=(response, {}, 0.1),
            ) as invocation:
                improve(
                    document,
                    pdf,
                    model="test-model",
                    effort="max",
                    cache_dir=root / "codex",
                )
            invocation.assert_called_once()

            mismatches = {
                "schema_version": "legalpdf.codex.cache.v0",
                "cache_key": "0" * 64,
                "model": "mismatched-model",
                "effort": "mismatched-effort",
                "prompt_version": "mismatched-prompt",
                "response_schema_sha256": "1" * 64,
                "repairable_diagnostics_sha256": "2" * 64,
                "repairable_diagnostics": [],
                "context_radius": 0,
                "max_attempts": 2,
                "max_live_calls": 5,
                "max_scope_pages": 1,
            }
            for field, value in mismatches.items():
                with self.subTest(cache_contract_field=field):
                    metadata = json.loads(
                        metadata_path.read_text(encoding="utf-8")
                    )
                    metadata[field] = value
                    metadata_path.write_text(
                        json.dumps(metadata), encoding="utf-8"
                    )
                    with patch(
                        "legalpdf.codex_repair._invoke",
                        return_value=(response, {}, 0.1),
                    ) as invocation:
                        improve(
                            document,
                            pdf,
                            model="test-model",
                            effort="max",
                            cache_dir=root / "codex",
                        )
                    invocation.assert_called_once()

            metadata_path.write_text("{broken", encoding="utf-8")
            with patch(
                "legalpdf.codex_repair._invoke",
                return_value=(response, {}, 0.1),
            ) as invocation:
                improve(
                    document,
                    pdf,
                    model="test-model",
                    effort="max",
                    cache_dir=root / "codex",
                )
            invocation.assert_called_once()
            repaired_metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            self.assertEqual("max", repaired_metadata["effort"])

    def test_codex_repair_recovers_zero_byte_and_tampered_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "legal.pdf"
            make_legal_pdf(pdf, restarted=False)
            document = parse_pdf(pdf, cache_dir=root / "local")
            document.diagnostics.append(
                Diagnostic(
                    code="COLUMN_ORDER_UNCERTAIN",
                    severity="warning",
                    message="fixture",
                    page_index=0,
                )
            )
            response = {
                "pages": [
                    {
                        "page_index": 0,
                        "regions": [
                            {
                                "region_type": "body",
                                "line_ids": [
                                    line.id for line in document.pages[0].lines
                                ],
                            }
                        ],
                    }
                ],
            }
            identity = repair_identity()
            cache_dir = root / "codex"
            cache_dir.mkdir()
            schema_path = cache_dir / (
                f"{identity['prompt_version']}."
                f"{identity['response_schema_sha256'][:16]}.schema.json"
            )

            schema_path.write_bytes(b"")
            with patch(
                "legalpdf.codex_repair._invoke",
                return_value=(response, {}, 0.1),
            ) as invocation:
                improve(
                    document,
                    pdf,
                    model="test-model",
                    effort="max",
                    cache_dir=cache_dir,
                )
            invocation.assert_called_once()
            recovered = json.loads(schema_path.read_text(encoding="utf-8"))
            self.assertEqual(_schema(), recovered)
            self.assertEqual(
                identity["response_schema_sha256"], _stable_hash(recovered)
            )

            schema_path.write_text('{"type":"array"}', encoding="utf-8")
            with patch("legalpdf.codex_repair._invoke") as invocation:
                improve(
                    document,
                    pdf,
                    model="test-model",
                    effort="max",
                    cache_dir=cache_dir,
                )
            invocation.assert_not_called()
            recovered = json.loads(schema_path.read_text(encoding="utf-8"))
            self.assertEqual(_schema(), recovered)
            self.assertEqual(
                identity["response_schema_sha256"], _stable_hash(recovered)
            )

    def test_invalid_codex_response_never_replaces_local_parse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "legal.pdf"
            make_legal_pdf(pdf, restarted=False)
            document = parse_pdf(pdf, cache_dir=root / "local")
            document.diagnostics.append(
                Diagnostic(
                    code="COLUMN_ORDER_UNCERTAIN",
                    severity="warning",
                    message="fixture",
                    page_index=0,
                )
            )
            original = [line.text for line in document.pages[0].lines]
            bad_response = {
                "pages": [
                    {
                        "page_index": 0,
                        "regions": [{"region_type": "body", "line_ids": []}],
                    }
                ],
            }
            with patch(
                "legalpdf.codex_repair._invoke",
                return_value=(bad_response, {}, 0.1),
            ) as invocation:
                result = improve(
                    document,
                    pdf,
                    model="test-model",
                    effort="low",
                    cache_dir=root / "codex",
                )
            self.assertEqual(3, invocation.call_count)
            self.assertEqual("failed", result.repairs[-1].status)
            self.assertEqual(original, [line.text for line in result.pages[0].lines])

    def test_codex_repair_has_one_document_live_call_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "four-pages.pdf"
            with fitz.open() as source:
                for page_number in range(1, 5):
                    page = source.new_page(width=612, height=792)
                    page.insert_text(
                        (72, 90), f"Body text on page {page_number}.", fontsize=11
                    )
                source.save(pdf)
            document = parse_pdf(pdf, cache_dir=root / "local")
            repairable = {
                "COLUMN_ORDER_UNCERTAIN",
                "FOOTNOTE_UNMATCHED_LABEL",
                "FOOTNOTE_UNMATCHED_REFERENCE",
                "FOOTNOTE_REGION_UNCERTAIN",
                "TEXT_QUALITY_LOW",
            }
            document.diagnostics = [
                item for item in document.diagnostics if item.code not in repairable
            ]
            for page_index, code in enumerate(
                [
                    "COLUMN_ORDER_UNCERTAIN",
                    "FOOTNOTE_UNMATCHED_LABEL",
                    "TEXT_QUALITY_LOW",
                    "FOOTNOTE_REGION_UNCERTAIN",
                ]
            ):
                document.diagnostics.append(
                    Diagnostic(
                        code=code,
                        severity="warning",
                        message="budget fixture",
                        page_index=page_index,
                    )
                )

            with patch(
                "legalpdf.codex_repair._invoke",
                return_value=({"pages": []}, {}, 0.1),
            ) as invocation:
                result = improve(
                    document,
                    pdf,
                    model="test-model",
                    effort="max",
                    cache_dir=root / "codex",
                )

            self.assertEqual(6, invocation.call_count)
            skipped = [
                item
                for item in result.diagnostics
                if item.code == "CODEX_REPAIR_BUDGET_EXHAUSTED"
            ]
            self.assertEqual([2, 3], [item.page_index for item in skipped])
            self.assertEqual(
                ["skipped", "skipped"],
                [item.status for item in result.repairs[-2:]],
            )
            self.assertEqual(6, result.provenance["codex"]["max_live_calls"])
            self.assertEqual(6, result.provenance["codex"]["live_calls"])

    def test_long_adjacent_fault_run_uses_bounded_scopes_and_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "sixteen-pages.pdf"
            with fitz.open() as source:
                for page_number in range(1, 17):
                    page = source.new_page(width=612, height=792)
                    page.insert_text(
                        (72, 90), f"Body text on page {page_number}.", fontsize=11
                    )
                source.save(pdf)
            document = parse_pdf(pdf, cache_dir=root / "local")
            repairable = set(repair_identity()["repairable_diagnostics"])
            document.diagnostics = [
                item for item in document.diagnostics if item.code not in repairable
            ]
            document.diagnostics.extend(
                Diagnostic(
                    code="COLUMN_ORDER_UNCERTAIN",
                    severity="warning",
                    message="long adjacent fixture",
                    page_index=page_index,
                )
                for page_index in range(16)
            )
            target_scopes: list[list[int]] = []
            context_sizes: list[int] = []
            image_sizes: list[int] = []

            def valid_response(**kwargs: object) -> tuple[dict, dict, float]:
                prompt = str(kwargs["prompt"])
                context = json.loads(prompt.split("INPUT:\n", 1)[1])
                targets = [
                    page["page_index"] for page in context["pages"] if page["target"]
                ]
                target_scopes.append(targets)
                context_sizes.append(len(context["pages"]))
                image_sizes.append(len(kwargs["image_paths"]))
                return (
                    {
                        "pages": [
                            {
                                "page_index": page["page_index"],
                                "regions": [
                                    {
                                        "region_type": "body",
                                        "line_ids": [
                                            line["id"] for line in page["lines"]
                                        ],
                                    }
                                ],
                            }
                            for page in context["pages"]
                            if page["target"]
                        ]
                    },
                    {},
                    0.1,
                )

            with patch(
                "legalpdf.codex_repair._invoke", side_effect=valid_response
            ) as invocation:
                result = improve(
                    document,
                    pdf,
                    model="test-model",
                    effort="max",
                    cache_dir=root / "codex",
                )

            expected_scopes = [
                [page, page + 1] for page in range(0, document.page_count, 2)
            ]
            self.assertEqual(
                expected_scopes,
                [repair.scope_pages for repair in result.repairs],
            )
            self.assertEqual(6, invocation.call_count)
            self.assertTrue(all(len(scope) <= 2 for scope in target_scopes))
            self.assertTrue(all(size <= 4 for size in context_sizes))
            self.assertEqual(context_sizes, image_sizes)
            self.assertEqual(6, result.provenance["codex"]["live_calls"])
            self.assertEqual(2, result.provenance["codex"]["max_scope_pages"])

    def test_adjacent_pages_with_the_same_fault_share_one_codex_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "legal.pdf"
            make_legal_pdf(pdf)
            document = parse_pdf(pdf, cache_dir=root / "local")
            for page_index in (0, 1):
                document.diagnostics.append(
                    Diagnostic(
                        code="COLUMN_ORDER_UNCERTAIN",
                        severity="warning",
                        message="shared fixture",
                        page_index=page_index,
                    )
                )
            response = {
                "pages": [
                    {
                        "page_index": page.index,
                        "regions": [
                            {
                                "region_type": "body",
                                "line_ids": [line.id for line in page.lines],
                            }
                        ],
                    }
                    for page in document.pages
                ]
            }
            with patch(
                "legalpdf.codex_repair._invoke",
                return_value=(response, {}, 0.1),
            ) as invocation:
                result = improve(
                    document,
                    pdf,
                    model="test-model",
                    effort="low",
                    cache_dir=root / "codex",
                )
            invocation.assert_called_once()
            self.assertEqual([0, 1], result.repairs[-1].scope_pages)

    def test_docx_gold_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            docx = root / "gold.docx"
            make_docx(docx)
            gold = extract_docx_gold(docx)
            self.assertEqual("1", gold["footnotes"][0]["label"])
            self.assertEqual("Footnote body.", gold["footnotes"][0]["body"])
            self.assertIn("2020 SCC 1", gold["citations"])

            pdf = root / "legal.pdf"
            make_legal_pdf(pdf, restarted=False)
            document = parse_pdf(pdf, cache_dir=root / "cache")
            gold["regions"] = [
                {
                    "page_index": region.page_index,
                    "type": region.type,
                    "line_ids": region.line_ids,
                }
                for page in document.pages
                for region in page.regions
            ]
            metrics = score_docx_gold(
                gold, document, baseline_document=document
            )
            self.assertIn("cer", metrics["text"])
            self.assertIn("f1", metrics["footnotes"])
            self.assertEqual(1.0, metrics["structure"]["region_type_accuracy"])
            self.assertEqual(
                1.0, metrics["codex"]["repair"]["source_line_conservation"]
            )

    def test_docx_gold_includes_true_ooxml_endnotes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            docx = Path(temporary) / "endnotes.docx"
            make_endnote_docx(docx)

            gold = extract_docx_gold(docx)

            self.assertEqual("legalpdf.docx_gold.v2", gold["schema_version"])
            self.assertEqual({"endnote": 1}, gold["note_counts"])
            self.assertEqual("endnote", gold["footnotes"][0]["kind"])
            self.assertEqual("1", gold["footnotes"][0]["label"])
            self.assertEqual("Endnote body.", gold["footnotes"][0]["body"])
            self.assertEqual(["2"], gold["paragraphs"][0]["endnote_ids"])

    def test_docx_gold_accumulates_full_passage_between_notes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            docx = Path(temporary) / "passages.docx"
            document_xml = """<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:r><w:t>First proposition</w:t></w:r><w:r><w:footnoteReference w:id="2"/></w:r></w:p>
<w:p><w:r><w:t>Intervening first paragraph.</w:t></w:r></w:p>
<w:p><w:r><w:t>Intervening second paragraph.</w:t></w:r></w:p>
<w:p><w:r><w:t>Final proposition</w:t></w:r><w:r><w:footnoteReference w:id="3"/></w:r></w:p>
</w:body></w:document>"""
            footnotes_xml = """<w:footnotes xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:footnote w:id="2"><w:p><w:r><w:t>First note.</w:t></w:r></w:p></w:footnote>
<w:footnote w:id="3"><w:p><w:r><w:t>Second note.</w:t></w:r></w:p></w:footnote>
</w:footnotes>"""
            with zipfile.ZipFile(docx, "w") as archive:
                archive.writestr("word/document.xml", document_xml)
                archive.writestr("word/footnotes.xml", footnotes_xml)

            gold = extract_docx_gold(docx)

            self.assertEqual(
                "Intervening first paragraph.\n\n"
                "Intervening second paragraph.\n\n"
                "Final proposition",
                gold["footnotes"][1]["passage_since_prior_note"],
            )

if __name__ == "__main__":
    unittest.main()
