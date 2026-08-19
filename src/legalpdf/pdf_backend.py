from __future__ import annotations

import ctypes
import difflib
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Sequence

import pdf_inspector
import pypdfium2 as pdfium

from .footnote_separator_scan import scan_gray_page
from .model import Diagnostic, Line, Page, Span, Word


@dataclass(slots=True)
class _Glyph:
    index: int
    text: str
    bbox: list[float] | None
    font: str = ""
    size: float = 0.0
    bold: bool = False
    italic: bool = False


def identity() -> str:
    def package_version(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return "unknown"

    return (
        f"pdf-inspector {package_version('pdf-inspector')} + "
        f"PDFium/pypdfium2 {package_version('pypdfium2')}"
    )


def _rounded_bbox(value: Sequence[float]) -> list[float]:
    raw = list(value) + [0.0] * (4 - len(value))
    return [round(float(part), 3) for part in raw[:4]]


def _union_bbox(values: Iterable[Sequence[float]]) -> list[float]:
    boxes = [list(value) for value in values]
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0]
    return _rounded_bbox(
        [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]
    )


def _geometry(page: Any) -> tuple[float, float, float, float, int, float, float]:
    box_x0, box_y0, box_x1, box_y1 = map(float, page.get_bbox())
    raw_width = abs(box_x1 - box_x0)
    raw_height = abs(box_y1 - box_y0)
    rotation = int(page.get_rotation()) % 360
    width, height = map(float, page.get_size())
    return box_x0, box_y0, raw_width, raw_height, rotation, width, height


def _transform_bbox(
    geometry: tuple[float, float, float, float, int, float, float],
    value: Sequence[float],
) -> list[float]:
    box_x0, box_y0, raw_width, raw_height, rotation, width, height = geometry

    def transform(x: float, y: float) -> tuple[float, float]:
        x -= box_x0
        y -= box_y0
        if rotation == 90:
            return y, raw_width - x
        if rotation == 180:
            return raw_width - x, raw_height - y
        if rotation == 270:
            return raw_height - y, x
        return x, y

    if rotation == 0:
        return [
            max(0.0, float(value[0]) - box_x0),
            max(0.0, height - (float(value[3]) - box_y0)),
            min(width, float(value[2]) - box_x0),
            min(height, height - (float(value[1]) - box_y0)),
        ]
    points = (
        transform(float(value[0]), float(value[1])),
        transform(float(value[0]), float(value[3])),
        transform(float(value[2]), float(value[1])),
        transform(float(value[2]), float(value[3])),
    )
    x0 = max(0.0, min(point[0] for point in points))
    x1 = min(width, max(point[0] for point in points))
    bottom = max(0.0, min(point[1] for point in points))
    top = min(height, max(point[1] for point in points))
    return [x0, height - top, x1, height - bottom]


def _inspector_bbox(item: Any, page_height: float) -> list[float]:
    return _rounded_bbox(
        [
            float(item.x),
            page_height - float(item.y) - float(item.height),
            float(item.x) + float(item.width),
            page_height - float(item.y),
        ]
    )


def _normalize_text(value: Any) -> str:
    return str(value or "").replace("\ufeff", "").replace("\u200b", "")


def _quality(lines: Sequence[Line]) -> float:
    text = "\n".join(line.text for line in lines)
    replacement_share = text.count("\ufffd") / max(1, len(text))
    printable_share = sum(character.isprintable() for character in text) / max(
        1, len(text)
    )
    quantity = min(1.0, len(text.strip()) / 100.0)
    return round(
        max(
            0.0,
            quantity
            * printable_share
            * (1.0 - min(1.0, replacement_share * 20)),
        ),
        4,
    )


def _same_inspector_line(group: list[Any], item: Any) -> bool:
    previous = group[-1]
    y_gap = abs(float(group[0].y) - float(item.y))
    if (
        len(group) == 1
        and re.fullmatch(
            r"(?:\d{1,4}|[ivxlcdm]{1,8})", str(group[0].text).strip(), re.I
        )
        and float(group[0].font_size) <= float(item.font_size) * 0.75
        and float(item.x) > float(group[0].x)
    ):
        return False
    if y_gap < 3.0:
        if y_gap > 0.5 and abs(float(item.x) - float(group[0].x)) < 5.0:
            return False
        if y_gap > 0.5 and float(item.x) < float(previous.x) - 10.0:
            return False
        gap = float(item.x) - (float(previous.x) + float(previous.width))
        if gap > max(
            18.0, max(float(item.font_size), float(previous.font_size)) * 1.5
        ):
            return False
        return True
    smaller = min(float(item.font_size), float(previous.font_size))
    larger = max(float(item.font_size), float(previous.font_size))
    horizontal_gap = float(item.x) - (float(previous.x) + float(previous.width))
    return (
        smaller <= larger * 0.78
        and y_gap <= larger * 0.55
        and -2.0 <= horizontal_gap <= larger * 1.5
    )


def _group_inspector_items(items: Sequence[Any]) -> list[list[Any]]:
    groups: list[list[Any]] = []
    for item in items:
        if str(item.item_type) != "text" or not str(item.text).strip():
            continue
        if groups and _same_inspector_line(groups[-1], item):
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


def _inspector_line(
    group: list[Any],
    *,
    page_index: int,
    local_index: int,
    source_index: int,
    page_height: float,
) -> Line:
    group.sort(key=lambda item: float(item.x))
    weighted_sizes = [
        float(item.font_size)
        for item in group
        for _ in range(max(1, min(100, len(str(item.text).strip()))))
        if float(item.font_size) > 0
    ]
    body_size = statistics.median(weighted_sizes) if weighted_sizes else 0.0
    baseline = statistics.median(float(item.y) for item in group)
    line_id = f"p{page_index + 1:04d}-l{local_index:04d}"
    parts: list[str] = []
    spans: list[Span] = []
    offset = 0
    previous: Any | None = None
    for item in group:
        text = _normalize_text(item.text)
        if not text:
            continue
        if previous is not None:
            gap = float(item.x) - (float(previous.x) + float(previous.width))
            if (
                gap
                >= max(float(item.font_size), float(previous.font_size), 10.0)
                * 0.15
                and not parts[-1].endswith(" ")
                and not text.startswith(" ")
            ):
                parts.append(" ")
                offset += 1
        start = offset
        parts.append(text)
        offset += len(text)
        superscript = (
            body_size > 0
            and 0 < float(item.font_size) <= body_size * 0.78
            and float(item.y) >= baseline + max(0.5, body_size * 0.08)
            and (bool(spans) or len(group) == 1)
        )
        flags = (
            (1 if superscript else 0)
            | (2 if bool(item.is_italic) else 0)
            | (16 if bool(item.is_bold) else 0)
        )
        spans.append(
            Span(
                id=f"{line_id}-s{len(spans) + 1:03d}",
                text=text,
                bbox=_inspector_bbox(item, page_height),
                font=str(item.font or ""),
                size=float(item.font_size or 0.0),
                flags=flags,
                superscript=superscript,
                start=start,
                end=offset,
            )
        )
        previous = item
    raw_text = "".join(parts)
    leading = len(raw_text) - len(raw_text.lstrip())
    text = raw_text.strip()
    for span in spans:
        span.start = max(0, span.start - leading)
        span.end = min(len(text), max(span.start, span.end - leading))
        span.text = text[span.start : span.end]
    spans = [span for span in spans if span.text]
    return Line(
        id=line_id,
        page_index=page_index,
        page_number=page_index + 1,
        source_index=source_index,
        reading_order=source_index,
        block_index=local_index,
        text=text,
        bbox=_union_bbox(span.bbox for span in spans),
        spans=spans,
        words=[],
    )


def _assign_block_indexes(lines: Sequence[Line]) -> None:
    block_index = 0
    previous: Line | None = None
    for line in sorted(lines, key=lambda item: item.source_index):
        if previous is None:
            block_index += 1
        else:
            previous_height = max(1.0, previous.bbox[3] - previous.bbox[1])
            current_height = max(1.0, line.bbox[3] - line.bbox[1])
            line_height = max(previous_height, current_height)
            vertical_gap = line.bbox[1] - previous.bbox[3]
            if (
                vertical_gap > max(4.0, line_height * 0.65)
                or line.bbox[1] < previous.bbox[1] - line_height * 0.5
            ):
                block_index += 1
        line.block_index = block_index
        previous = line


def _font_record(
    text_page: Any,
    index: int,
    cache: dict[int, tuple[str, float, bool, bool]],
    name_buffer: Any,
    flags: Any,
) -> tuple[str, float, bool, bool]:
    obj = pdfium.raw.FPDFText_GetTextObject(text_page.raw, index)
    if not obj:
        return "", 0.0, False, False
    key = int(ctypes.cast(obj, ctypes.c_void_p).value or 0)
    if key in cache:
        return cache[key]
    pdfium.raw.FPDFText_GetFontInfo(
        text_page.raw,
        index,
        name_buffer,
        len(name_buffer),
        ctypes.byref(flags),
    )
    name = name_buffer.value.decode("utf-8", errors="replace")
    raw_size = float(pdfium.raw.FPDFText_GetFontSize(text_page.raw, index))
    matrix = pdfium.raw.FS_MATRIX()
    scale = 1.0
    if pdfium.raw.FPDFPageObj_GetMatrix(obj, ctypes.byref(matrix)):
        scale = max(math.hypot(matrix.a, matrix.b), math.hypot(matrix.c, matrix.d))
    folded = name.casefold()
    record = (
        name,
        raw_size * scale,
        int(pdfium.raw.FPDFText_GetFontWeight(text_page.raw, index)) >= 600
        or "bold" in folded,
        "italic" in folded or "oblique" in folded,
    )
    cache[key] = record
    return record


def _glyph_lines(page: Any, text_page: Any, *, styled: bool) -> list[list[_Glyph]]:
    geometry = _geometry(page)
    lines: list[list[_Glyph]] = []
    current: list[_Glyph] = []
    font_cache: dict[int, tuple[str, float, bool, bool]] = {}
    name_buffer = ctypes.create_string_buffer(256)
    flags = ctypes.c_int()
    for index in range(text_page.count_chars()):
        value = int(pdfium.raw.FPDFText_GetUnicode(text_page.raw, index))
        if value in {10, 13}:
            if current:
                lines.append(current)
                current = []
            continue
        try:
            text = chr(value)
        except ValueError:
            text = "\ufffd"
        if not text or text == "\x00":
            continue
        if styled:
            try:
                box = _transform_bbox(geometry, text_page.get_charbox(index))
            except Exception:
                box = None
            font, size, bold, italic = _font_record(
                text_page, index, font_cache, name_buffer, flags
            )
        else:
            box = None
            font, size, bold, italic = "", 0.0, False, False
        current.append(_Glyph(index, text, box, font, size, bold, italic))
    if current:
        lines.append(current)
    return lines


def _glyph_line_bbox(glyphs: Sequence[_Glyph]) -> list[float]:
    return _union_bbox(glyph.bbox for glyph in glyphs if glyph.bbox is not None)


def _range_bbox(
    text_page: Any,
    geometry: tuple[float, float, float, float, int, float, float],
    start: int,
    count: int,
) -> list[float] | None:
    if count <= 0:
        return None
    try:
        rectangle_count = text_page.count_rects(start, count)
        boxes = [
            _transform_bbox(geometry, text_page.get_rect(index))
            for index in range(rectangle_count)
        ]
    except Exception:
        return None
    return _union_bbox(boxes) if boxes else None


def _vertical_distance(first: Sequence[float], second: Sequence[float]) -> float:
    if first[3] >= second[1] and second[3] >= first[1]:
        return 0.0
    return min(abs(first[1] - second[3]), abs(second[1] - first[3]))


def _words_from_glyphs(
    line: Line,
    glyph_lines: Sequence[Sequence[_Glyph]],
    glyph_bboxes: Sequence[Sequence[float]] | None = None,
    *,
    text_page: Any | None = None,
    geometry: tuple[float, float, float, float, int, float, float] | None = None,
) -> list[Word]:
    if not glyph_lines:
        return []
    boxes = (
        list(glyph_bboxes)
        if glyph_bboxes is not None
        else [_glyph_line_bbox(glyphs) for glyphs in glyph_lines]
    )
    candidate_indexes = sorted(
        range(len(glyph_lines)),
        key=lambda index: (
            _vertical_distance(line.bbox, boxes[index]),
            abs(
                (line.bbox[1] + line.bbox[3]) / 2
                - (boxes[index][1] + boxes[index][3]) / 2
            ),
            abs(line.bbox[0] - boxes[index][0]),
        ),
    )
    best_glyphs: list[_Glyph] = []
    best_mapping: dict[int, int] = {}
    best_coverage = 0.0
    for candidate_index in candidate_indexes[:2]:
        raw_glyph_text = "".join(
            glyph.text for glyph in glyph_lines[candidate_index]
        )
        leading = len(raw_glyph_text) - len(raw_glyph_text.lstrip())
        trailing = len(raw_glyph_text.rstrip())
        glyphs = list(glyph_lines[candidate_index])[leading:trailing]
        glyph_text = raw_glyph_text.strip()
        if not glyph_text:
            continue
        matcher = difflib.SequenceMatcher(None, line.text, glyph_text, autojunk=False)
        mapped: dict[int, int] = {}
        for tag, line_start, line_end, glyph_start, glyph_end in matcher.get_opcodes():
            if tag != "equal":
                continue
            for offset in range(min(line_end - line_start, glyph_end - glyph_start)):
                mapped[line_start + offset] = glyph_start + offset
        coverage = len(mapped) / max(1, len(line.text))
        if coverage > best_coverage:
            best_glyphs = glyphs
            best_mapping = mapped
            best_coverage = coverage
        if coverage >= 0.95:
            break
    if best_coverage < 0.45:
        return []
    words: list[Word] = []
    for match in re.finditer(r"\S+", line.text):
        mapped_glyphs = [
            best_glyphs[best_mapping[index]]
            for index in range(match.start(), match.end())
            if index in best_mapping
        ]
        if not mapped_glyphs:
            continue
        if text_page is not None and geometry is not None:
            first_index = min(glyph.index for glyph in mapped_glyphs)
            last_index = max(glyph.index for glyph in mapped_glyphs)
            word_bbox = _range_bbox(
                text_page,
                geometry,
                first_index,
                last_index - first_index + 1,
            )
        else:
            boxes = [
                glyph.bbox for glyph in mapped_glyphs if glyph.bbox is not None
            ]
            word_bbox = _union_bbox(boxes) if boxes else None
        if word_bbox is None:
            continue
        words.append(
            Word(
                id=f"{line.id}-w{len(words) + 1:03d}",
                text=match.group(),
                bbox=word_bbox,
                start=match.start(),
                end=match.end(),
            )
        )
    return words


def _styled_glyph_line(
    glyphs: list[_Glyph],
    *,
    page_index: int,
    local_index: int,
    source_index: int,
) -> Line | None:
    raw_text = _normalize_text("".join(glyph.text for glyph in glyphs))
    leading = len(raw_text) - len(raw_text.lstrip())
    trailing = len(raw_text.rstrip())
    text = raw_text.strip()
    if not text:
        return None
    glyphs = glyphs[leading:trailing]
    visible = [glyph for glyph in glyphs if glyph.bbox is not None and not glyph.text.isspace()]
    if not visible:
        return None
    sizes = [glyph.size for glyph in visible if glyph.size > 0]
    body_size = statistics.median(sizes) if sizes else 0.0
    bottoms = [
        glyph.bbox[3]
        for glyph in visible
        if glyph.bbox is not None and (body_size <= 0 or glyph.size >= body_size * 0.85)
    ]
    baseline = statistics.median(bottoms) if bottoms else 0.0
    line_id = f"p{page_index + 1:04d}-l{local_index:04d}"
    spans: list[Span] = []
    start = 0
    while start < len(glyphs):
        glyph = glyphs[start]
        key = (glyph.font, round(glyph.size, 3), glyph.bold, glyph.italic)
        end = start + 1
        while end < len(glyphs):
            candidate = glyphs[end]
            if (
                candidate.font,
                round(candidate.size, 3),
                candidate.bold,
                candidate.italic,
            ) != key:
                break
            end += 1
        span_glyphs = glyphs[start:end]
        span_text = "".join(part.text for part in span_glyphs)
        boxes = [part.bbox for part in span_glyphs if part.bbox is not None]
        superscript = (
            body_size > 0
            and 0 < glyph.size <= body_size * 0.78
            and boxes
            and min(box[3] for box in boxes) <= baseline - max(0.5, body_size * 0.08)
        )
        flags = (
            (1 if superscript else 0)
            | (2 if glyph.italic else 0)
            | (16 if glyph.bold else 0)
        )
        spans.append(
            Span(
                id=f"{line_id}-s{len(spans) + 1:03d}",
                text=span_text,
                bbox=_union_bbox(boxes),
                font=glyph.font,
                size=round(glyph.size, 3),
                flags=flags,
                superscript=superscript,
                start=start,
                end=end,
            )
        )
        start = end
    line = Line(
        id=line_id,
        page_index=page_index,
        page_number=page_index + 1,
        source_index=source_index,
        reading_order=source_index,
        block_index=local_index,
        text=text,
        bbox=_union_bbox(glyph.bbox for glyph in visible if glyph.bbox is not None),
        spans=spans,
        words=[],
    )
    line.words = _words_from_glyphs(
        line,
        [glyphs],
        [_glyph_line_bbox(glyphs)],
    )
    return line


def _separator_y(page: Any, lines: Sequence[Line], width: float, height: float) -> float | None:
    candidates: list[tuple[float, float]] = []
    geometry = _geometry(page)
    try:
        objects = page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_PATH], max_depth=15)
    except Exception:
        return None
    for obj in objects:
        try:
            box = _transform_bbox(geometry, obj.get_bounds())
        except Exception:
            continue
        length = box[2] - box[0]
        thickness = box[3] - box[1]
        y = (box[1] + box[3]) / 2
        if (
            thickness <= 1.75
            and length >= width * 0.20
            and height * 0.30 <= y <= height * 0.98
        ):
            candidates.append((length, y))
    if not candidates:
        return None
    body_sizes = [
        statistics.median(span.size for span in line.spans if span.size > 0)
        for line in lines
        if line.bbox[1] < height * 0.70 and any(span.size > 0 for span in line.spans)
    ]
    body_size = statistics.median(body_sizes) if body_sizes else 0.0
    label_ys = [
        line.bbox[1]
        for line in lines
        if line.bbox[1] >= height * 0.48
        and re.match(r"\s*(?:\d+|[ivxlcdm]+)\b", line.text, re.I)
        and body_size > 0
        and min(
            (span.size for span in line.spans if span.size > 0),
            default=body_size,
        )
        <= body_size * 0.90
    ]
    if label_ys:
        above_label = [
            candidate
            for candidate in candidates
            if candidate[1] <= min(label_ys) + max(1.0, height * 0.004)
        ]
        if above_label:
            return max(above_label, key=lambda candidate: candidate[1])[1]
    conservative = [candidate for candidate in candidates if candidate[1] <= height * 0.92]
    return (
        min(conservative, key=lambda candidate: (candidate[0], candidate[1]))[1]
        if conservative
        else None
    )


def _raster_separator_y(page: Any, height: float) -> float | None:
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        bitmap = page.render(scale=2, grayscale=True)
        try:
            gray = bitmap.to_numpy()
            if gray.ndim == 3:
                gray = np.mean(gray[:, :, :3], axis=2).astype(np.uint8)
        finally:
            bitmap.close()
    except Exception:
        return None
    record = scan_gray_page(gray)
    if record.get("separator_status") not in {"found", "found_two_column"}:
        return None
    return float(record["separators"][0]["y_center_ratio"]) * height


def _ocr_lines(
    provider: Any,
    path: Path,
    page_index: int,
    width: float,
    height: float,
    global_line_offset: int,
) -> list[Line]:
    return [
        Line(
            id=f"p{page_index + 1:04d}-l{index:04d}",
            page_index=page_index,
            page_number=page_index + 1,
            source_index=global_line_offset + index,
            reading_order=global_line_offset + index,
            block_index=index,
            text=result.text.strip(),
            bbox=_rounded_bbox(result.bbox),
            spans=[],
            source="ocr",
        )
        for index, result in enumerate(
            provider.extract_page(path, page_index, width=width, height=height),
            start=1,
        )
        if result.text.strip()
    ]


def extract_pdf_pages(
    path: Path,
    *,
    ocr_provider: Any | None,
) -> tuple[list[Page], list[Diagnostic], dict[str, Any], dict[int, float | None]]:
    diagnostics: list[Diagnostic] = []
    try:
        classification = pdf_inspector.classify_pdf(str(path))
        ocr_pages = {int(index) for index in classification.pages_needing_ocr}
        classified_pages = int(classification.page_count)
    except Exception as error:
        classification = None
        ocr_pages = set()
        classified_pages = 0
        diagnostics.append(
            Diagnostic(
                code="PDF_INSPECTOR_CLASSIFICATION_FAILED",
                severity="info",
                message="The fast classifier failed; PDFium recovery was used.",
                details={"error": f"{type(error).__name__}: {error}"},
            )
        )
    try:
        raw_items = pdf_inspector.extract_text_with_positions(str(path))
    except Exception as error:
        raw_items = []
        diagnostics.append(
            Diagnostic(
                code="PDF_INSPECTOR_EXTRACTION_FAILED",
                severity="info",
                message="The primary positioned-text extractor failed; PDFium recovery was used.",
                details={"error": f"{type(error).__name__}: {error}"},
            )
        )
    items_by_page: defaultdict[int, list[Any]] = defaultdict(list)
    for item in raw_items:
        page_index = int(item.page) - 1
        if page_index >= 0:
            items_by_page[page_index].append(item)

    pages: list[Page] = []
    separators: dict[int, float | None] = {}
    global_offset = 0
    try:
        document = pdfium.PdfDocument(path)
    except Exception as error:
        raise ValueError(f"PDFium could not open the PDF: {error}") from error
    with document:
        if classified_pages and classified_pages != len(document):
            diagnostics.append(
                Diagnostic(
                    code="PDF_PAGE_COUNT_DISAGREEMENT",
                    severity="info",
                    message="The classifier and renderer reported different page counts; PDFium was used.",
                    details={
                        "pdf_inspector": classified_pages,
                        "pdfium": len(document),
                    },
                )
            )
        metadata = document.get_metadata_dict(skip_empty=True)
        for page_index in range(len(document)):
            page = document[page_index]
            try:
                width, height = map(float, page.get_size())
                rotation = int(page.get_rotation()) % 360
                geometry = _geometry(page)
                text_page = page.get_textpage()
                try:
                    glyph_lines = _glyph_lines(page, text_page, styled=False)
                    glyph_bboxes = [
                        _range_bbox(
                            text_page,
                            geometry,
                            glyphs[0].index,
                            glyphs[-1].index - glyphs[0].index + 1,
                        )
                        or [0.0, 0.0, 0.0, 0.0]
                        for glyphs in glyph_lines
                    ]
                    inspector_groups = _group_inspector_items(
                        items_by_page.get(page_index, ())
                    )
                    lines = [
                        _inspector_line(
                            group,
                            page_index=page_index,
                            local_index=index,
                            source_index=global_offset + index,
                            page_height=height,
                        )
                        for index, group in enumerate(inspector_groups, start=1)
                    ]
                    lines = [line for line in lines if line.text]
                    for line in lines:
                        line.words = _words_from_glyphs(
                            line,
                            glyph_lines,
                            glyph_bboxes,
                            text_page=text_page,
                            geometry=geometry,
                        )
                    quality = _quality(lines)
                    recover_with_pdfium = (
                        rotation != 0
                        or page_index in ocr_pages
                        or quality < 0.15
                        or not lines
                    )
                    recovered = False
                    if recover_with_pdfium:
                        styled_lines = _glyph_lines(page, text_page, styled=True)
                        candidate_lines: list[Line] = []
                        for glyphs in styled_lines:
                            line = _styled_glyph_line(
                                glyphs,
                                page_index=page_index,
                                local_index=len(candidate_lines) + 1,
                                source_index=global_offset + len(candidate_lines) + 1,
                            )
                            if line is not None:
                                candidate_lines.append(line)
                        candidate_quality = _quality(candidate_lines)
                        if candidate_lines and (
                            candidate_quality > quality
                            or rotation != 0
                            or page_index in ocr_pages
                        ):
                            lines = candidate_lines
                            quality = candidate_quality
                            recovered = True
                    _assign_block_indexes(lines)
                finally:
                    text_page.close()
                source = "native"
                if quality < 0.15 and ocr_provider is not None:
                    ocr_lines = _ocr_lines(
                        ocr_provider,
                        path,
                        page_index,
                        width,
                        height,
                        global_offset,
                    )
                    if ocr_lines:
                        lines = ocr_lines
                        quality = 0.5
                        source = "ocr"
                if quality < 0.15 or not lines:
                    diagnostics.append(
                        Diagnostic(
                            code="OCR_REQUIRED",
                            severity="warning",
                            message="Page has no reliable embedded text and no usable OCR result.",
                            page_index=page_index,
                        )
                    )
                elif recovered and page_index in ocr_pages:
                    diagnostics.append(
                        Diagnostic(
                            code="EMBEDDED_OCR_RECOVERED",
                            severity="info",
                            message="PDFium recovered a positioned embedded text layer from an OCR-classified page.",
                            page_index=page_index,
                        )
                    )
                separator = _separator_y(page, lines, width, height)
                if separator is None and source == "ocr":
                    separator = _raster_separator_y(page, height)
                separators[page_index] = separator
                pages.append(
                    Page(
                        id=f"p{page_index + 1:04d}",
                        index=page_index,
                        number=page_index + 1,
                        width=round(width, 3),
                        height=round(height, 3),
                        lines=lines,
                        regions=[],
                        source=source,
                        text_quality=quality,
                    )
                )
                global_offset += len(lines)
            finally:
                page.close()
    return pages, diagnostics, metadata, separators
