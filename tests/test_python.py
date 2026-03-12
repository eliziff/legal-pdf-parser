"""Tests for the pdf_inspector Python bindings."""

import os
import pytest
import pdf_inspector

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def fixture_path(name: str) -> str:
    return os.path.join(FIXTURES_DIR, name)


# ---------------------------------------------------------------------------
# process_pdf
# ---------------------------------------------------------------------------


class TestProcessPdf:
    def test_basic(self):
        result = pdf_inspector.process_pdf(fixture_path("thermo-freon12.pdf"))
        assert result.pdf_type == "text_based"
        assert result.page_count == 3
        assert result.confidence > 0.0
        assert result.markdown is not None
        assert len(result.markdown) > 0

    def test_result_repr(self):
        result = pdf_inspector.process_pdf(fixture_path("thermo-freon12.pdf"))
        r = repr(result)
        assert "PdfResult" in r
        assert "text_based" in r

    def test_with_pages(self):
        result = pdf_inspector.process_pdf(
            fixture_path("thermo-freon12.pdf"), pages=[1]
        )
        assert result.page_count == 3  # total pages in doc
        assert result.markdown is not None

    def test_result_fields(self):
        result = pdf_inspector.process_pdf(fixture_path("thermo-freon12.pdf"))
        # All fields should be accessible
        assert isinstance(result.pdf_type, str)
        assert isinstance(result.page_count, int)
        assert isinstance(result.processing_time_ms, int)
        assert isinstance(result.pages_needing_ocr, list)
        assert isinstance(result.confidence, float)
        assert isinstance(result.is_complex_layout, bool)
        assert isinstance(result.pages_with_tables, list)
        assert isinstance(result.pages_with_columns, list)
        assert isinstance(result.has_encoding_issues, bool)
        # title can be None or str
        assert result.title is None or isinstance(result.title, str)


# ---------------------------------------------------------------------------
# process_pdf_bytes
# ---------------------------------------------------------------------------


class TestProcessPdfBytes:
    def test_basic(self):
        with open(fixture_path("thermo-freon12.pdf"), "rb") as f:
            data = f.read()
        result = pdf_inspector.process_pdf_bytes(data)
        assert result.pdf_type == "text_based"
        assert result.markdown is not None

    def test_with_pages(self):
        with open(fixture_path("thermo-freon12.pdf"), "rb") as f:
            data = f.read()
        result = pdf_inspector.process_pdf_bytes(data, pages=[1, 2])
        assert result.markdown is not None


# ---------------------------------------------------------------------------
# detect_pdf / detect_pdf_bytes
# ---------------------------------------------------------------------------


class TestDetectPdf:
    def test_detect_file(self):
        result = pdf_inspector.detect_pdf(fixture_path("thermo-freon12.pdf"))
        assert result.pdf_type == "text_based"
        assert result.markdown is None  # detect only — no markdown
        assert result.page_count == 3

    def test_detect_bytes(self):
        with open(fixture_path("thermo-freon12.pdf"), "rb") as f:
            data = f.read()
        result = pdf_inspector.detect_pdf_bytes(data)
        assert result.pdf_type == "text_based"
        assert result.markdown is None


# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_basic(self):
        text = pdf_inspector.extract_text(fixture_path("thermo-freon12.pdf"))
        assert isinstance(text, str)
        assert len(text) > 0


# ---------------------------------------------------------------------------
# extract_text_with_positions
# ---------------------------------------------------------------------------


class TestExtractTextWithPositions:
    def test_basic(self):
        items = pdf_inspector.extract_text_with_positions(
            fixture_path("thermo-freon12.pdf")
        )
        assert len(items) > 0
        item = items[0]
        assert isinstance(item.text, str)
        assert isinstance(item.x, float)
        assert isinstance(item.y, float)
        assert isinstance(item.width, float)
        assert isinstance(item.height, float)
        assert isinstance(item.font, str)
        assert isinstance(item.font_size, float)
        assert isinstance(item.page, int)
        assert isinstance(item.is_bold, bool)
        assert isinstance(item.is_italic, bool)
        assert isinstance(item.item_type, str)

    def test_with_pages(self):
        items = pdf_inspector.extract_text_with_positions(
            fixture_path("thermo-freon12.pdf"), pages=[1]
        )
        assert len(items) > 0
        assert all(item.page == 1 for item in items)

    def test_repr(self):
        items = pdf_inspector.extract_text_with_positions(
            fixture_path("thermo-freon12.pdf")
        )
        r = repr(items[0])
        assert "TextItem" in r


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_nonexistent_file(self):
        with pytest.raises(ValueError):
            pdf_inspector.process_pdf("/nonexistent/file.pdf")

    def test_not_a_pdf(self):
        with pytest.raises(ValueError):
            pdf_inspector.process_pdf_bytes(b"this is not a pdf")

    def test_empty_bytes(self):
        with pytest.raises(ValueError):
            pdf_inspector.process_pdf_bytes(b"")


# ---------------------------------------------------------------------------
# Multiple fixtures
# ---------------------------------------------------------------------------


class TestMultipleFixtures:
    """Run basic processing on all available test fixtures."""

    @pytest.mark.parametrize(
        "filename",
        [f for f in os.listdir(FIXTURES_DIR) if f.endswith(".pdf")],
    )
    def test_process_all_fixtures(self, filename):
        result = pdf_inspector.process_pdf(fixture_path(filename))
        assert result.pdf_type in (
            "text_based",
            "scanned",
            "image_based",
            "mixed",
        )
        assert result.page_count > 0
        assert result.confidence >= 0.0
