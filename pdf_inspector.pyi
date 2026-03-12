"""Type stubs for pdf_inspector."""

from typing import Optional

class PdfResult:
    """Result of processing a PDF file."""
    pdf_type: str
    """'text_based', 'scanned', 'image_based', or 'mixed'."""
    markdown: Optional[str]
    page_count: int
    processing_time_ms: int
    pages_needing_ocr: list[int]
    title: Optional[str]
    confidence: float
    is_complex_layout: bool
    pages_with_tables: list[int]
    pages_with_columns: list[int]
    has_encoding_issues: bool

class TextItem:
    """A positioned text item extracted from a PDF."""
    text: str
    x: float
    y: float
    width: float
    height: float
    font: str
    font_size: float
    page: int
    is_bold: bool
    is_italic: bool
    item_type: str

def process_pdf(path: str, pages: Optional[list[int]] = None) -> PdfResult:
    """Process a PDF: detect type, extract text, convert to Markdown."""
    ...

def process_pdf_bytes(data: bytes, pages: Optional[list[int]] = None) -> PdfResult:
    """Process a PDF from bytes in memory."""
    ...

def detect_pdf(path: str) -> PdfResult:
    """Fast detection only — no text extraction."""
    ...

def detect_pdf_bytes(data: bytes) -> PdfResult:
    """Fast detection from bytes."""
    ...

def extract_text(path: str) -> str:
    """Extract plain text from a PDF."""
    ...

def extract_text_with_positions(path: str, pages: Optional[list[int]] = None) -> list[TextItem]:
    """Extract text with position information."""
    ...
