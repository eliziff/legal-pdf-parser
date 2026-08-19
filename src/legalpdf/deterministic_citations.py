"""Exact-offset citation splitting for conservative and recall-first modes."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .anchored_scan import AnchoredPattern
from .grammar_tables import lazy_table_entry as _table

# Every name below binds the grammar-table entry its pattern was
# extracted into (tools/grammar_differential.py's MAPPING is the
# name <-> id ledger); the rf-string fragments the old compiles spliced
# (_REPORTER_WORDS, _REF_NUM/_REF_NUMS/_REF_PIN, _PIN_*) live on as the
# tables' defs (reporter_words, ref_num/ref_nums/ref_pin,
# pin_*_splitter). Handles compile on first use: eagerly compiling
# these 26 boundary-expanded entries costs ~1.8s cold at import.
_URL_RE = _table("cite.url")
_NEUTRAL_RE = _table("cite.neutral")
_REPORTER_RE = _table("cite.reporter.splitter")
_STATUTE_RE = _table("cite.statute.splitter")
_JOURNAL_RE = _table("cite.journal.splitter")
_BOOK_FRAME_RE = _table("frame.book")
_REF_TOKEN_RE = _table("ref.token")
_PURE_REF_RE = _table("ref.pure.splitter")
_LINK_ATTACHMENT_RE = _table("attach.link")
_SIGNAL_PREFIX_RE = _table("signal.prefix.splitter")
_SOURCE_SIGNAL_RE = _table("signal.source")
_QUOTED_WORK_AUTHOR_RE = _table("ref.quoted-work-author")
_TRAILING_SHORT_FORM_RE = _table("shortform.splitter")
_PAR_PIN_RE = _table("pinpoint.para.splitter")
_SEC_PIN_RE = _table("pinpoint.section.splitter")
_PAGE_PIN_RE = _table("pinpoint.page.splitter")
_EDITORIAL_BRACKET_RE = _table("bracket.editorial")
_SENTENCE_BOUNDARY_RE = _table("boundary.sentence.splitter")
_AGGRESSIVE_SIGNAL_RE = _table("signal.aggressive")
# Engine-only boundary heuristic; deliberately has no table entry.
_CASE_START_RE = re.compile(
    r"(?<!\w)(?:R\.?\s+v\.?|Reference\s+re|In\s+re|"
    r"[A-Z][A-Za-z'’().& -]{1,70}\s+(?:v\.?|c))\s+",
)
_CROSS_REFERENCE_RE = _table("ref.cross-reference")
_QUOTED_CITATION_RE = _table("cite.quoted")
_SECONDARY_CITATION_RE = _table("cite.secondary")
_LEGAL_TITLE_RE = _table("title.legal.splitter")
_NAMED_CODE_RE = _table("title.named-code")
_CONJUNCTION_RE = _table("boundary.conjunction")
_NOTE_REFERENCE_START_RE = _table("ref.note-reference")

# Anchor-windowed handles for the whole-input scans below, built lazily
# because AnchoredPattern derivation compiles its pattern and these are
# lazy table entries. Keyed by pattern identity: every wrapped pattern is
# a module-level singleton. Same matches as the bare pattern, faster on
# document-sized input (anchored_scan's differential test covers every
# table entry; texts under its window cutoff full-scan anyway).
_ANCHORED: dict[int, AnchoredPattern] = {}


def _anchored(rx: re.Pattern[str]) -> AnchoredPattern:
    handle = _ANCHORED.get(id(rx))
    if handle is None:
        handle = _ANCHORED[id(rx)] = AnchoredPattern(rx)
    return handle


@dataclass(frozen=True)
class DeterministicPart:
    start: int
    end: int
    text: str
    anchors: tuple[str, ...]


@dataclass(frozen=True)
class DeterministicSplit:
    status: str
    parts: tuple[DeterministicPart, ...] = ()
    delimiters: tuple[tuple[int, int, str], ...] = ()
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeterministicFields:
    status: str
    corrected: str
    kind: str
    link_candidate: str
    pinpoint_fragments: tuple[str, ...]
    page_pinpoints: tuple[int, ...]
    bare_citation: str
    citation_with_style: str
    short_form: str
    reasons: tuple[str, ...] = ()


def _masked_ranges(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in _anchored(_URL_RE).finditer(text)]


def _top_level_indices(text: str) -> set[int]:
    masked = _masked_ranges(text)
    masked_index = 0
    round_depth = square_depth = curly_depth = 0
    smart_quote = False
    straight_quote = False
    positions: set[int] = set()
    for index, character in enumerate(text):
        while masked_index < len(masked) and index >= masked[masked_index][1]:
            masked_index += 1
        if masked_index < len(masked) and masked[masked_index][0] <= index < masked[masked_index][1]:
            continue
        if not smart_quote and not straight_quote and not (round_depth or square_depth or curly_depth):
            positions.add(index)
        if character == "“":
            smart_quote = True
        elif character == "”":
            smart_quote = False
        elif character == '"':
            straight_quote = not straight_quote
        elif not smart_quote and not straight_quote:
            if character == "(":
                round_depth += 1
            elif character == ")" and round_depth:
                round_depth -= 1
            elif character == "[":
                square_depth += 1
            elif character == "]" and square_depth:
                square_depth -= 1
            elif character == "{":
                curly_depth += 1
            elif character == "}" and curly_depth:
                curly_depth -= 1
    return positions


def _top_level_semicolons(text: str) -> list[int]:
    top_level = _top_level_indices(text)
    return [index for index, character in enumerate(text) if character == ";" and index in top_level]


def _top_level_signals(text: str) -> list[int]:
    top_level = _top_level_indices(text)
    positions: list[int] = []
    for match in _anchored(_SOURCE_SIGNAL_RE).finditer(text):
        start = match.start("sentence") if match.group("sentence") else match.start("inline")
        if start in top_level:
            positions.append(start)
    return positions


def _anchors(text: str) -> list[tuple[int, int, str]]:
    found: list[tuple[int, int, str]] = []
    for kind, pattern in (
        ("neutral", _NEUTRAL_RE),
        ("reporter", _REPORTER_RE),
        ("statute", _STATUTE_RE),
        ("journal", _JOURNAL_RE),
        ("book", _BOOK_FRAME_RE),
        ("url", _URL_RE),
    ):
        found.extend(
            (match.start(), match.end(), kind)
            for match in _anchored(pattern).finditer(text)
        )
    found.sort()
    deduped: list[tuple[int, int, str]] = []
    for item in found:
        if deduped and item[0] < deduped[-1][1]:
            if item[1] - item[0] > deduped[-1][1] - deduped[-1][0]:
                deduped[-1] = item
            continue
        deduped.append(item)
    return deduped


def _one_anchor_cluster(text: str, anchors: list[tuple[int, int, str]]) -> bool:
    if not anchors:
        return False
    for previous, current in zip(anchors, anchors[1:]):
        gap = text[previous[1]:current[0]]
        if re.fullmatch(r"\s*,\s*", gap):
            continue
        if current[2] == "url" and _LINK_ATTACHMENT_RE.fullmatch(gap):
            continue
        return False
    return True


def _clause(start: int, end: int, text: str) -> DeterministicPart | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    value = text[start:end]
    if _PURE_REF_RE.fullmatch(value):
        return DeterministicPart(start, end, value, ("reference",))
    anchors = _anchors(value)
    if not _one_anchor_cluster(value, anchors):
        return None
    return DeterministicPart(start, end, value, tuple(item[2] for item in anchors))


def _strip_signals(text: str) -> str:
    value = text.strip()
    for _ in range(3):
        stripped = _SIGNAL_PREFIX_RE.sub("", value)
        if stripped == value:
            break
        value = stripped.strip()
    return value


def _short_form(text: str, kind: str) -> str:
    bracket = _TRAILING_SHORT_FORM_RE.search(text)
    if bracket:
        value = bracket.group(1).strip()
        if not value.isdigit() and not _EDITORIAL_BRACKET_RE.match(value):
            return value
    if _REF_TOKEN_RE.search(text):
        if re.search(r"\bibid\b", text, re.I):
            return "Ibid"
        prefix = re.split(r"\bsupra\b", _strip_signals(text), maxsplit=1, flags=re.I)[0]
        return prefix.strip(" ,;:.")
    if kind in {"journal", "book", "essay_collection", "report"}:
        quote_positions = [position for position in (text.find('"'), text.find("\u201c")) if position >= 0]
        if quote_positions:
            prefix = text[:min(quote_positions)].strip(" ,")
            if ":" in prefix:
                prefix = prefix.rsplit(":", 1)[-1].strip()
        else:
            prefix = text.split(",", 1)[0].strip()
        prefix = _strip_signals(prefix)
        authors = re.split(r"\s*(?:,|&|\band\b)\s*", prefix)
        surnames: list[str] = []
        for author in authors:
            tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’.-]*", author)
            tokens = [token for token in tokens if token.casefold() not in {"et", "al", "kc", "qc", "eds", "ed"}]
            if tokens:
                surnames.append(tokens[-1].strip("."))
        if surnames:
            return " and ".join(surnames)
    return ""


def _pin_values(value: str, *, expand_ranges: bool) -> list[str]:
    values: list[str] = []
    for item in re.split(r"\s*(?:,\s*(?:and\s+)?|\band\b|&)\s*", value):
        numbers = re.findall(r"\d+(?:\.\d+)?", item)
        if not numbers:
            continue
        values.append(numbers[0])
        if expand_ranges and len(numbers) > 1 and "." not in numbers[0] + numbers[1]:
            start = int(numbers[0])
            end_text = numbers[1]
            if len(end_text) < len(numbers[0]):
                magnitude = 10 ** len(end_text)
                end = start - (start % magnitude) + int(end_text)
                if end < start:
                    end += magnitude
            else:
                end = int(end_text)
            if 0 < end - start <= 100:
                values.extend(str(number) for number in range(start + 1, end + 1))
    return values


def _provision_values(value: str) -> list[str]:
    """Preserve rule and section identifiers instead of treating hyphens as ranges."""
    values: list[str] = []
    for item in re.split(r"\s*(?:,\s*(?:and\s+)?|\band\b|&)\s*", value):
        match = re.search(
            r"\d+(?:\.\d+)*(?:\s*\([A-Za-z0-9]+\))*"
            r"(?:\s*[-–]\s*\d+(?:\.\d+)*)?",
            item,
        )
        if match:
            values.append(re.sub(r"\s+", "", match.group(0)).replace("–", "-"))
    return values


def _pinpoints(text: str, kind: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    source_kind = (kind or "").casefold()
    case_source = source_kind in {"case", "unreported"}
    law_source = source_kind in {"statute", "regulation", "legislation"}
    unresolved_reference = source_kind in {"", "other"}

    if case_source or unresolved_reference:
        paragraph = _PAR_PIN_RE.search(text)
        if paragraph:
            values = _pin_values(paragraph.group("values"), expand_ranges=False)
            return tuple(f"par{value}" for value in values), ()
    if law_source or unresolved_reference:
        reporter_spans = [match.span() for match in _REPORTER_RE.finditer(text)]
        provision = next(
            (
                match
                for match in _SEC_PIN_RE.finditer(text)
                if not any(start <= match.start() < end for start, end in reporter_spans)
            ),
            None,
        )
        if provision:
            values = _provision_values(provision.group("values"))
            return tuple(f"sec{value}" for value in values), ()
    if not law_source:
        page = _PAGE_PIN_RE.search(text)
        if page:
            values = _pin_values(page.group("values"), expand_ranges=True)
            return (), tuple(int(value) for value in values if value.isdigit())
    return (), ()


def _kind(text: str, anchors: tuple[str, ...]) -> str:
    if _PURE_REF_RE.fullmatch(text):
        return "other"
    if _BOOK_FRAME_RE.search(text):
        return "essay_collection" if re.search(r"[\"\u201c].+?[\"\u201d]\s+in\b", text) else "book"
    if _JOURNAL_RE.search(text) and re.search(r"[\"\u201c]", text):
        return "journal"
    if "statute" in anchors:
        return "statute"
    if "neutral" in anchors or "reporter" in anchors:
        return "case"
    if "journal" in anchors:
        return "journal"
    return "other"


def _bare_citation(text: str, kind: str) -> str:
    value = _TRAILING_SHORT_FORM_RE.sub("", text).strip().rstrip(".").strip()
    if _REF_TOKEN_RE.search(value):
        return value
    if kind == "case":
        matches = [*_NEUTRAL_RE.finditer(value), *_REPORTER_RE.finditer(value)]
        if matches:
            return value[min(match.start() for match in matches):]
    if kind == "statute":
        match = _STATUTE_RE.search(value)
        if match:
            return value[match.start():]
    if kind == "journal":
        match = _JOURNAL_RE.search(value)
        if match:
            return value[match.start():]
    return value


def extract_fields(part: DeterministicPart) -> DeterministicFields:
    """Populate the full splitter tuple without treating route kind as identity."""
    text = part.text.strip()
    kind = _kind(text, part.anchors)
    styled = _strip_signals(text)
    fragments, pages = _pinpoints(styled, kind)
    direct_link = next(iter(_URL_RE.finditer(text)), None)
    link = direct_link.group(0).strip("<>.,; ") if direct_link else "other"
    reasons: list[str] = []
    if _has_embedded_source_signal(styled):
        reasons.append("embedded_second_source")
    if not styled:
        reasons.append("missing_citation_surface")
    bare = _bare_citation(styled, kind) if styled else ""
    if not bare:
        reasons.append("missing_bare_citation")
    return DeterministicFields(
        "partial" if reasons else "complete",
        text,
        kind,
        link,
        fragments,
        pages,
        bare,
        styled,
        _short_form(styled, kind),
        tuple(reasons),
    )


def extract_text_fields(text: str) -> DeterministicFields:
    value = str(text or "")
    start = len(value) - len(value.lstrip())
    end = len(value.rstrip())
    part_text = value[start:end]
    anchors = tuple(kind for _left, _right, kind in _anchors(part_text))
    return extract_fields(DeterministicPart(start, end, part_text, anchors))


def _has_embedded_source_signal(text: str) -> bool:
    signal_re = re.compile(
        r"(?:\.\s*(?:see(?:\s+also|\s+generally)?|cf\.?|compare)|"
        r"\b(?:citing|quoted?\s+in|quoting|discuss(?:ed|ing)\s+in)\b)",
        re.I,
    )
    for signal in signal_re.finditer(text):
        if signal.start() < 3:
            continue
        tail = text[signal.end():signal.end() + 320]
        if (
            _REF_TOKEN_RE.search(tail)
            or _NEUTRAL_RE.search(tail)
            or _REPORTER_RE.search(tail)
            or _STATUTE_RE.search(tail)
            or _JOURNAL_RE.search(tail)
            or _BOOK_FRAME_RE.search(tail)
        ):
            return True
    return False


def _inside_quotes(text: str, position: int) -> bool:
    inside = False
    for character in text[:position]:
        if character == "\u201c":
            inside = True
        elif character == "\u201d":
            inside = False
        elif character == '"':
            inside = not inside
    return inside


def _inside_square_brackets(text: str, position: int) -> bool:
    prefix = text[:position]
    return prefix.rfind("[") > prefix.rfind("]")


def _source_evidence(text: str) -> bool:
    return bool(
        _anchors(text)
        or _CROSS_REFERENCE_RE.search(text)
        or _QUOTED_CITATION_RE.search(text)
        or _SECONDARY_CITATION_RE.search(text)
        or _LEGAL_TITLE_RE.search(text)
        or _NAMED_CODE_RE.search(text)
        or re.search(r"^\s*(?:s(?:ection)?|r(?:ule)?|art(?:icle)?)\.?\s*\d", text, re.I)
    )


def _sentence_starts(text: str) -> list[int]:
    starts: list[int] = []
    for match in _anchored(_SENTENCE_BOUNDARY_RE).finditer(text):
        if _inside_quotes(text, match.start()):
            continue
        prefix = text[:match.start() + 1]
        if re.search(r"(?:\be\.g|\bi\.e|\bcf|\bno|\bv|\bpara|\bart|\b[A-Z])\.$", prefix, re.I):
            continue
        suffix = text[match.end():]
        if (
            re.search(
                r"\b(?:Ltd|Inc|Corp|Co|LLC|LLP)\.$",
                prefix,
                re.IGNORECASE,
            )
            and re.match(r"v\.?(?:\s|$)", suffix, re.IGNORECASE)
        ):
            continue
        starts.append(match.end())
    return starts


def _segment_start(boundaries: list[tuple[int, int, str]], position: int) -> int:
    return max(
        (right for left, right, _reason in boundaries if left < position and right <= position),
        default=0,
    )


def _segment_end(boundaries: list[tuple[int, int, str]], position: int, length: int) -> int:
    return min(
        (left for left, _right, _reason in boundaries if left > position),
        default=length,
    )


def _recall_boundaries(text: str) -> list[tuple[int, int, str]]:
    """Find boundaries between intact citations, biased strongly against misses."""
    boundaries: list[tuple[int, int, str]] = [
        (index, index + 1, "semicolon")
        for index, character in enumerate(text)
        if character == ";"
    ]

    sentence_starts = _sentence_starts(text)
    sentence_ends = [*sentence_starts[1:], len(text)]
    for start, end in zip(sentence_starts, sentence_ends):
        if text[:start].strip() and _source_evidence(text[start:end]):
            boundaries.append((start, start, "new_citation_sentence"))

    hard_positions = sorted({
        0, len(text),
        *(left for left, _right, _reason in boundaries),
        *(right for _left, right, _reason in boundaries),
    })
    for match in _anchored(_AGGRESSIVE_SIGNAL_RE).finditer(text):
        if _inside_quotes(text, match.start()):
            continue
        left = max(position for position in hard_positions if position <= match.start())
        right = min(position for position in hard_positions if position >= match.end())
        if _source_evidence(text[left:match.start()]) and _source_evidence(text[match.start():right]):
            boundaries.append((match.start(), match.start(), "source_signal"))

    case_starts = [match.start() for match in _anchored(_CASE_START_RE).finditer(text)]
    for position in case_starts[1:]:
        if text[position - 1:position] == "(" or re.search(
            r"(?:\b(?:and|or)|&)\s*$", text[max(0, position - 12):position], re.I
        ):
            continue
        start = _segment_start(boundaries, position)
        end = _segment_end(boundaries, position, len(text))
        if _source_evidence(text[start:position]) and _source_evidence(text[position:end]):
            boundaries.append((position, position, "new_case_frame"))

    author_starts = [
        match.start() for match in _anchored(_QUOTED_WORK_AUTHOR_RE).finditer(text)
    ]
    for position in author_starts[1:]:
        start = _segment_start(boundaries, position)
        end = _segment_end(boundaries, position, len(text))
        if _source_evidence(text[start:position]) and _source_evidence(text[position:end]):
            boundaries.append((position, position, "new_author_title_frame"))

    note_starts = [
        match.start() for match in _anchored(_NOTE_REFERENCE_START_RE).finditer(text)
    ]
    for position in note_starts[1:]:
        start = _segment_start(boundaries, position)
        if _source_evidence(text[start:position]):
            boundaries.append((position, position, "new_note_reference"))

    if not _anchors(text) and re.match(r"^\s*(?:see|compare|cf\.?|contra)\b", text, re.I):
        bare_short_form = re.search(
            r"\band\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]{2,50}\s*\.\s*$", text
        )
        if bare_short_form:
            boundaries.append((bare_short_form.start(), bare_short_form.start(),
                               "conjoined_short_form"))

    for match in _anchored(_CONJUNCTION_RE).finditer(text):
        if _inside_quotes(text, match.start()):
            continue
        start = _segment_start(boundaries, match.start())
        end = _segment_end(boundaries, match.start(), len(text))
        if _source_evidence(text[start:match.start()]) and _source_evidence(text[match.start():end]):
            boundaries.append((match.start(), match.start(), "conjoined_citation"))

    legal_starts = [
        match.start() for match in _anchored(_LEGAL_TITLE_RE).finditer(text)
        if not _inside_quotes(text, match.start())
        and not _inside_square_brackets(text, match.start())
    ]
    for position in legal_starts:
        start = _segment_start(boundaries, position)
        if not any(start <= prior < position for prior in legal_starts):
            continue
        end = _segment_end(boundaries, position, len(text))
        if _source_evidence(text[start:position]) and _source_evidence(text[position:end]):
            boundaries.append((position, position, "new_legal_source_frame"))

    semicolon_positions = {left for left, right, _reason in boundaries if right > left}
    deduped: dict[tuple[int, int], str] = {}
    for left, right, reason in sorted(boundaries):
        if left == right and (left in semicolon_positions or left - 1 in semicolon_positions):
            continue
        deduped.setdefault((left, right), reason)
    return [(left, right, reason) for (left, right), reason in sorted(deduped.items())]


def split_footnote_recall_first(text: str) -> DeterministicSplit:
    """Losslessly partition Free-mode text without source-type abstention."""
    if not isinstance(text, str) or not text.strip():
        return DeterministicSplit("abstain", reasons=("empty",))
    boundaries = _recall_boundaries(text)
    starts = [0, *(right for _left, right, _reason in boundaries)]
    ends = [*(left for left, _right, _reason in boundaries), len(text)]
    parts: list[DeterministicPart] = []
    for start, end in zip(starts, ends):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start >= end:
            continue
        value = text[start:end]
        anchors = tuple(kind for _left, _right, kind in _anchors(value))
        if _PURE_REF_RE.fullmatch(value):
            anchors = ("reference",)
        parts.append(DeterministicPart(start, end, value, anchors))
    if not parts:
        return DeterministicSplit("abstain", reasons=("empty_parts",))
    delimiters = tuple(
        (parts[index].end, parts[index + 1].start,
         text[parts[index].end:parts[index + 1].start])
        for index in range(len(parts) - 1)
    )
    reasons = tuple(dict.fromkeys(reason for _left, _right, reason in boundaries))
    return DeterministicSplit(
        "deterministic_complete",
        tuple(parts),
        delimiters,
        reasons or ("single_citation_or_prose",),
    )


def split_footnote(text: str) -> DeterministicSplit:
    if not isinstance(text, str) or not text.strip():
        return DeterministicSplit("abstain", reasons=("empty",))
    semicolons = _top_level_semicolons(text)
    boundary_spans: list[tuple[int, int, str]] = [
        (position, position + 1, "top_level_semicolon") for position in semicolons
    ]
    for position in _top_level_signals(text):
        ordered = sorted(boundary_spans)
        segment_start = max((right for _left, right, _reason in ordered if right <= position), default=0)
        segment_end = min((left for left, _right, _reason in ordered if left >= position), default=len(text))
        if _clause(segment_start, position, text) and _clause(position, segment_end, text):
            boundary_spans.append((position, position, "explicit_source_signal"))

    boundary_spans.sort()
    if not boundary_spans:
        if _PURE_REF_RE.fullmatch(text):
            part = _clause(0, len(text), text)
            return DeterministicSplit("deterministic_complete", (part,), reasons=("pure_reference",))
        return DeterministicSplit("abstain", reasons=("no_supported_boundary",))

    starts = [0, *(right for _left, right, _reason in boundary_spans)]
    ends = [*(left for left, _right, _reason in boundary_spans), len(text)]
    parts: list[DeterministicPart] = []
    for start, end in zip(starts, ends):
        part = _clause(start, end, text)
        if part is None:
            return DeterministicSplit("abstain", reasons=("unconsumed_or_ambiguous_clause",))
        parts.append(part)
    delimiters = tuple(
        (parts[index].end, parts[index + 1].start, text[parts[index].end:parts[index + 1].start])
        for index in range(len(parts) - 1)
    )
    used_reasons = {reason for _left, _right, reason in boundary_spans}
    reasons = tuple(
        reason for reason in ("top_level_semicolon", "explicit_source_signal")
        if reason in used_reasons
    )
    return DeterministicSplit(
        "deterministic_complete",
        tuple(parts),
        delimiters,
        reasons,
    )
