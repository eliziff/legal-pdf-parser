from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

SCHEMA_VERSION = "legalpdf.document.v2"
PARSER_VERSION = "0.3.0"


@dataclass(slots=True)
class Word:
    id: str
    text: str
    bbox: list[float]
    start: int
    end: int


@dataclass(slots=True)
class Span:
    id: str
    text: str
    bbox: list[float]
    font: str = ""
    size: float = 0.0
    flags: int = 0
    superscript: bool = False
    start: int = 0
    end: int = 0


@dataclass(slots=True)
class Line:
    id: str
    page_index: int
    page_number: int
    source_index: int
    reading_order: int
    block_index: int
    text: str
    bbox: list[float]
    spans: list[Span] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)
    detached_references: list[dict[str, Any]] = field(default_factory=list)
    exclude_from_body: bool = False
    suppress_footnote_label: bool = False
    note_region_mode: str = ""
    region_id: str = ""
    region_type: str = "unknown"
    source: str = "native"


@dataclass(slots=True)
class Region:
    id: str
    page_index: int
    type: str
    line_ids: list[str]
    bbox: list[float]
    reading_order: int


@dataclass(slots=True)
class Page:
    id: str
    index: int
    number: int
    width: float
    height: float
    lines: list[Line]
    regions: list[Region]
    source: str = "native"
    text_quality: float = 1.0
    printed_label: str | None = None
    printed_label_source: str | None = None
    printed_label_line_id: str | None = None


@dataclass(slots=True)
class Paragraph:
    id: str
    page_index: int
    region_type: str
    text: str
    line_ids: list[str]
    anchors: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class Section:
    id: str
    heading_paragraph_id: str
    heading: str
    locator: str
    locator_kind: str | None
    aliases: list[str]
    text: str
    paragraph_ids: list[str]
    page_indexes: list[int]
    line_ids: list[str]
    provenance: str = "heading-region"


@dataclass(slots=True)
class Footnote:
    pair_id: str
    label: str
    occurrence: int
    restart_sequence: int
    reference_page: int | None
    body_pages: list[int]
    reference_line_id: str | None
    body_line_ids: list[str]
    body: str
    sentence_proposition: str
    passage_since_prior_note: str
    confidence: float
    provenance: str
    warnings: list[str] = field(default_factory=list)
    crossrefs: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class Diagnostic:
    code: str
    severity: str
    message: str
    page_index: int | None = None
    line_ids: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RepairRecord:
    page_index: int
    status: str
    model: str
    effort: str
    prompt_version: str
    cache_key: str
    attempts: int
    elapsed_seconds: float
    input_line_hash: str
    output_hash: str = ""
    token_usage: dict[str, int] = field(default_factory=dict)
    error: str = ""
    scope_pages: list[int] = field(default_factory=list)


@dataclass(slots=True)
class LegalDocument:
    document_id: str
    source_name: str
    source_sha256: str
    page_count: int
    status: str
    pages: list[Page]
    paragraphs: list[Paragraph]
    sections: list[Section]
    footnotes: list[Footnote]
    diagnostics: list[Diagnostic]
    repairs: list[RepairRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    parser_version: str = PARSER_VERSION

    @property
    def lines(self) -> list[Line]:
        return [line for page in self.pages for line in page.lines]

    @property
    def text(self) -> str:
        return "\n\n".join(paragraph.text for paragraph in self.paragraphs)


@dataclass(slots=True)
class FootnoteLookup:
    status: str
    query: str
    matches: list[str]
    footnote: Footnote | None = None
    proposition_mode: str = "sentence"
    proposition: str = ""
    context: str = ""


def _require_fields(value: dict[str, Any], kind: type[Any]) -> None:
    if missing := {item.name for item in fields(kind)} - value.keys():
        raise ValueError(f"{kind.__name__} value is missing fields: {', '.join(sorted(missing))}")


def _page_from_dict(value: dict[str, Any]) -> Page:
    _require_fields(value, Page)
    for line in value["lines"]:
        _require_fields(line, Line)
        for span in line["spans"]:
            _require_fields(span, Span)
        for word in line["words"]:
            _require_fields(word, Word)
    for region in value["regions"]:
        _require_fields(region, Region)
    return Page(
        id=value["id"],
        index=int(value["index"]),
        number=int(value["number"]),
        width=float(value["width"]),
        height=float(value["height"]),
        lines=[
            Line(
                **{
                    **line,
                    "spans": [Span(**span) for span in line["spans"]],
                    "words": [Word(**word) for word in line["words"]],
                }
            )
            for line in value["lines"]
        ],
        regions=[Region(**region) for region in value["regions"]],
        source=value["source"],
        text_quality=float(value["text_quality"]),
        printed_label=value["printed_label"],
        printed_label_source=value["printed_label_source"],
        printed_label_line_id=value["printed_label_line_id"],
    )
