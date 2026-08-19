"""Local-first structural parsing for legal PDFs."""

from .core import (
    add_pdf_geometry,
    improve,
    lookup_artifact_footnote,
    lookup_footnote,
    parse_pdf,
)
from .docx_linking import (
    apply_docx_links,
    assess_route,
    plan_docx_links,
    plan_footnotes,
)
from .model import (
    Diagnostic,
    Footnote,
    FootnoteLookup,
    LegalDocument,
    Line,
    Page,
    Paragraph,
    Region,
    RepairRecord,
    Section,
    Span,
    Word,
    load_geometry_artifacts,
    load_artifacts,
    write_artifacts,
)
from .ocr import OCRLine, TesseractOCRProvider

__all__ = [
    "Diagnostic",
    "Footnote",
    "FootnoteLookup",
    "LegalDocument",
    "Line",
    "OCRLine",
    "Page",
    "Paragraph",
    "Region",
    "RepairRecord",
    "Section",
    "Span",
    "TesseractOCRProvider",
    "Word",
    "add_pdf_geometry",
    "apply_docx_links",
    "assess_route",
    "improve",
    "load_geometry_artifacts",
    "load_artifacts",
    "lookup_artifact_footnote",
    "lookup_footnote",
    "parse_pdf",
    "plan_docx_links",
    "plan_footnotes",
    "write_artifacts",
]
