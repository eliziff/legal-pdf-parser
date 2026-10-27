//! A lookup by page, paragraph, section or footnote: the request, the units it returns, and how
//! each locator finds its units.

use super::*;

pub(crate) struct PrintedParagraphMap {
    pub(crate) paragraphs: Vec<PrintedParagraph>,
    pub(crate) status: HashMap<usize, PdfLookupStatus>,
}

/// A paragraph a PDF numbers in print ("[12]", or 12 in the margin): its number, its span of the
/// query text in UTF-16 code units, the pages it is on and its native lines.
/// `ranges` is exact membership; `start` and `end` only position its envelope.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PrintedParagraph {
    #[serde(skip)]
    pub(crate) id: String,
    pub number: usize,
    pub start: usize,
    pub end: usize,
    pub ranges: Vec<ScalarRange>,
    pub page_numbers: Vec<u32>,
    pub line_ids: Vec<String>,
}

#[derive(Clone, Serialize)]
pub struct PdfLookupRequest {
    pub locator_kind: String,
    pub locator: String,
    pub end_locator: Option<String>,
    pub context_blocks: usize,
    pub page: Option<u32>,
    pub occurrence: Option<usize>,
}

impl PdfLookupRequest {
    pub fn new(locator_kind: impl Into<String>, locator: impl Into<String>) -> Self {
        Self {
            locator_kind: locator_kind.into(),
            locator: locator.into(),
            end_locator: None,
            context_blocks: 0,
            page: None,
            occurrence: None,
        }
    }

    fn valid(&self) -> bool {
        LOCATOR_KINDS.contains(&self.locator_kind.as_str())
            && !self.locator.trim().is_empty()
            && utf16_len(&self.locator) <= 200
            && self
                .end_locator
                .as_deref()
                .is_none_or(|end| end.is_empty() || utf16_len(end) <= 200)
            && self.context_blocks <= MAX_CONTEXT
            && self.page.is_none_or(|page| page > 0)
            && self.occurrence.is_none_or(|occurrence| occurrence > 0)
    }
}

#[derive(Serialize)]
pub struct PdfLookupProposition {
    pub sentence: String,
    pub passage_since_prior_note: String,
}

#[derive(Serialize)]
pub struct PdfLookupNote {
    pub label: String,
    pub occurrence: usize,
    pub restart_sequence: usize,
    pub reference_page: Option<u32>,
    pub body_pages: Vec<u32>,
    pub warnings: Vec<String>,
}

#[derive(Serialize)]
pub struct PdfLookupUnit {
    pub id: String,
    pub kind: &'static str,
    pub locator: String,
    pub text: String,
    pub page_numbers: Vec<u32>,
    pub confidence: Option<f64>,
    pub confidence_basis: &'static str,
    pub provenance: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proposition: Option<PdfLookupProposition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<PdfLookupNote>,
}

#[derive(Serialize)]
pub struct PdfLookupPage {
    pub page_number: u32,
    pub text: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PdfLookupStatus {
    Found,
    NotFound,
    Ambiguous,
    Invalid,
    Unavailable,
}

#[derive(Serialize)]
pub struct PdfStructureLookup {
    pub schema_version: &'static str,
    pub requested: PdfLookupRequest,
    pub units: Vec<PdfLookupUnit>,
    pub before: Vec<PdfLookupUnit>,
    pub after: Vec<PdfLookupUnit>,
    pub matches: Vec<String>,
    pub pages: Vec<PdfLookupPage>,
    pub status: PdfLookupStatus,
    pub exact: bool,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub payload_sha256: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub page_text_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub(crate) fn result(
    input: &PdfLookupRequest,
    status: PdfLookupStatus,
    error: Option<String>,
) -> PdfStructureLookup {
    let mut requested = input.clone();
    if requested.end_locator.as_deref() == Some("") {
        requested.end_locator = None;
    }
    PdfStructureLookup {
        schema_version: "legalpdf.structure-lookup.v1",
        requested,
        units: Vec::new(),
        before: Vec::new(),
        after: Vec::new(),
        matches: Vec::new(),
        pages: Vec::new(),
        status,
        exact: status == PdfLookupStatus::Found,
        payload_sha256: String::new(),
        page_text_sha256: String::new(),
        error,
    }
}

pub(crate) fn nfkc_text(value: &str) -> Cow<'_, str> {
    if value.is_ascii() {
        Cow::Borrowed(value)
    } else {
        Cow::Owned(value.nfkc().collect())
    }
}

pub(crate) fn join_page_lines(lines: Vec<Line>) -> String {
    let mut output = String::with_capacity(lines.iter().map(|line| line.text.len() + 1).sum());
    for line in lines {
        let text = line.text.trim();
        if text.is_empty() {
            continue;
        }
        if output.ends_with('-') && text.chars().next().is_some_and(char::is_lowercase) {
            output.pop();
        } else if !output.is_empty() {
            output.push(' ');
        }
        output.push_str(text);
    }
    output
}

pub(crate) fn range_text(text: &ScalarText<'_>, ranges: &[ScalarRange]) -> String {
    ranges.iter().filter_map(|range| text.slice_utf16(range.start..range.end))
        .filter(|part| !part.is_empty()).collect::<Vec<_>>().join("\n")
}

pub(crate) fn rendered_slice(text: &ScalarText<'_>, document: &DocumentStructure, node: &StructureNode) -> String {
    range_text(text, &document.query_ranges(node))
}

pub(crate) fn section_nodes(document: &PdfDocument) -> Vec<&StructureNode> {
    document
        .structure
        .nodes
        .iter()
        // A heading's or a constituent document's extent is no provision: it has no identifier
        // to look up.
        .filter(|node| {
            node.kind == NodeKind::Section
                && node
                    .locator_kind
                    .as_deref()
                    .is_some_and(|kind| !kind.is_empty())
        })
        .collect()
}

pub(crate) fn nodes_by_id(document: &PdfDocument) -> HashMap<&str, &StructureNode> {
    document
        .structure
        .nodes
        .iter()
        .map(|node| (node.id.as_str(), node))
        .collect()
}

/// A declared paragraph as a lookup unit, retaining its graph identity.
pub(crate) fn printed_unit(
    text: &ScalarText<'_>,
    document: &PdfDocument,
    paragraph: &PrintedParagraph,
) -> PdfLookupUnit {
    let page = paragraph
        .page_numbers
        .first()
        .and_then(|number| document.pages.iter().find(|page| page.number == *number));
    PdfLookupUnit {
        id: paragraph.id.clone(),
        kind: "paragraph",
        locator: format!("paragraph {}", paragraph.number),
        text: range_text(text, &paragraph.ranges),
        page_numbers: paragraph.page_numbers.clone(),
        confidence: page.map(|item| item.text_quality.clamp(0.0, 1.0)),
        confidence_basis: if page.is_some() {
            "page_text_quality"
        } else {
            "unavailable"
        },
        provenance: "legalpdf:printed".to_owned(),
        proposition: None,
        note: None,
    }
}

pub(crate) fn page_unit(
    document: &DocumentStructure,
    text: &ScalarText<'_>,
    nodes: &HashMap<&str, &StructureNode>,
    page: &ProjectionPage,
) -> PdfLookupUnit {
    let text = nodes
        .get(page.id.as_str())
        .copied()
        .map(|node| rendered_slice(text, document, node))
        .unwrap_or_default();
    PdfLookupUnit {
        id: if page.id.is_empty() {
            format!("page-{}", page.number)
        } else {
            page.id.clone()
        },
        kind: "page",
        locator: format!("[page {}]", page.number),
        text,
        page_numbers: vec![page.number],
        confidence: Some(page.text_quality.clamp(0.0, 1.0)),
        confidence_basis: "page_text_quality",
        provenance: if page.source.is_empty() {
            "unknown".to_owned()
        } else {
            page.source.clone()
        },
        proposition: None,
        note: None,
    }
}

/// A page's printed transcript lines in reading order: each line number the structure labels
/// (its `LineLabel` nodes), its node's id, and the words set across its row; a number on a
/// blank line is left out.
pub(crate) fn transcript_lines(
    document: &PdfDocument,
    page: &ProjectionPage,
    geometry: &PdfTextPage,
) -> Vec<(usize, String, String)> {
    let height = |rect: &[f64; 4]| (rect[3] - rect[1]).max(0.1);
    document
        .structure
        .nodes
        .iter()
        .filter(|node| {
            node.role == Some(legal_structure_model::PrimaryRole::LineLabel)
                && node.page_indexes.contains(&page.index)
        })
        .filter_map(|node| {
            let number = node.label.as_deref()?.trim().parse().ok()?;
            let label_id = node.line_ids.first()?;
            let label = geometry.lines.iter().find(|line| &line.id == label_id)?;
            let mut row = geometry
                .lines
                .iter()
                .filter(|line| {
                    line.id != label.id
                        && line.region_type != "line_label"
                        && line.rect[3].min(label.rect[3]) - line.rect[1].max(label.rect[1])
                            > 0.5 * height(&line.rect).min(height(&label.rect))
                })
                .collect::<Vec<_>>();
            row.sort_by(|left, right| left.rect[0].total_cmp(&right.rect[0]));
            let text = row
                .iter()
                .map(|line| line.text.trim())
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            (!text.is_empty()).then(|| (number, node.id.clone(), text))
        })
        .collect()
}

pub(crate) fn footnote_unit(note: &ProjectionFootnote) -> PdfLookupUnit {
    let mut page_numbers = Vec::new();
    if let Some(page) = note.reference_page {
        page_numbers.push(page);
    }
    for page in &note.body_pages {
        if !page_numbers.contains(page) {
            page_numbers.push(*page);
        }
    }
    PdfLookupUnit {
        id: note.pair_id.clone(),
        kind: "footnote",
        locator: format!("footnote {}", note.label),
        text: note.body.trim().to_owned(),
        page_numbers,
        confidence: Some(note.confidence.clamp(0.0, 1.0)),
        confidence_basis: "footnote_pairing",
        provenance: if note.provenance.is_empty() {
            "unknown".to_owned()
        } else {
            note.provenance.clone()
        },
        proposition: Some(PdfLookupProposition {
            sentence: note.sentence_proposition.trim().to_owned(),
            passage_since_prior_note: note.passage_since_prior_note.trim().to_owned(),
        }),
        note: Some(PdfLookupNote {
            label: note.label.clone(),
            occurrence: note.occurrence,
            restart_sequence: note.restart_sequence,
            reference_page: note.reference_page,
            body_pages: note.body_pages.clone(),
            warnings: note.warnings.clone(),
        }),
    }
}

pub(crate) fn section_unit(
    document: &DocumentStructure,
    text: &ScalarText<'_>,
    pages: &HashMap<usize, &ProjectionPage>,
    section: &StructureNode,
    index: usize,
) -> PdfLookupUnit {
    let mut page_numbers = Vec::new();
    let mut confidence: Option<f64> = None;
    for page_index in &section.page_indexes {
        if let Some(page) = pages.get(page_index) {
            if !page_numbers.contains(&page.number) {
                page_numbers.push(page.number);
            }
            let quality = page.text_quality.clamp(0.0, 1.0);
            confidence = Some(confidence.map_or(quality, |value| value.min(quality)));
        }
    }
    PdfLookupUnit {
        id: if section.id.is_empty() {
            format!("section-{}", index + 1)
        } else {
            section.id.clone()
        },
        kind: "section",
        locator: section
            .label
            .as_deref()
            .unwrap_or(&section.id)
            .trim()
            .to_owned(),
        text: rendered_slice(text, document, section).trim().to_owned(),
        page_numbers,
        confidence,
        confidence_basis: if confidence.is_some() {
            "minimum_page_text_quality"
        } else {
            "unavailable"
        },
        provenance: match section.source {
            Derivation::Native => "native",
            Derivation::Heuristic => "legal-structure",
            Derivation::Model => "model",
        }
        .to_owned(),
        proposition: None,
        note: None,
    }
}

pub fn numeric_range(kind: &str, raw: &str) -> Option<(usize, usize)> {
    static RANGE: OnceLock<Regex> = OnceLock::new();
    static PAGE_PREFIX: OnceLock<Regex> = OnceLock::new();
    static PARAGRAPH_PREFIX: OnceLock<Regex> = OnceLock::new();
    static LINE_PREFIX: OnceLock<Regex> = OnceLock::new();
    let value = nfkc_text(raw);
    let value = value
        .trim()
        .trim_start_matches('#')
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim();
    let prefix = if kind == "page" {
        PAGE_PREFIX.get_or_init(|| Regex::new(r"(?i)^(?:pages?|pp?\.?)[\s:_=-]*").unwrap())
    } else if kind == "line" {
        LINE_PREFIX.get_or_init(|| Regex::new(r"(?i)^(?:lines?|ll?\.?)[\s:_=-]*").unwrap())
    } else {
        PARAGRAPH_PREFIX
            .get_or_init(|| Regex::new(r"(?i)^(?:paragraphs?|paras?|pars?)\.?[\s:_=-]*").unwrap())
    };
    let stripped = prefix.replace(value, "");
    let captures = RANGE
        .get_or_init(|| Regex::new(r"(?i)^(\d{1,6})\s*(?:-|–|—|\.\.|to)\s*(\d{1,6})$").unwrap())
        .captures(&stripped)?;
    Some((captures[1].parse().ok()?, captures[2].parse().ok()?))
}

pub(crate) fn normalize_footnote(raw: &str) -> String {
    static PREFIX: OnceLock<Regex> = OnceLock::new();
    let value = nfkc_text(raw);
    PREFIX
        .get_or_init(|| Regex::new(r"(?i)^(?:footnotes?|notes?|fn)\s*[#.:_-]?\s*").unwrap())
        .replace(value.trim(), "")
        .to_lowercase()
}

pub(crate) fn normalized_section(raw: &str) -> Option<String> {
    static PREFIX: OnceLock<Regex> = OnceLock::new();
    let value = nfkc_text(raw);
    let compact: String = PREFIX
        .get_or_init(|| Regex::new(r"(?i)^(?:ss?\.?|sections?)\s*").unwrap())
        .replace(value.trim(), "")
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    let normalized = normalize_compact_numbered_section_locator(&compact);
    (!normalized.is_empty()).then_some(normalized)
}

pub(crate) fn section_alias(raw: &str) -> String {
    normalized_section(raw).unwrap_or_else(|| {
        let value = nfkc_text(raw);
        value
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    })
}

pub(crate) fn section_matches(section: &StructureNode, requested_kind: &str, requested: &str) -> bool {
    if requested_kind != "section" && section.locator_kind.as_deref() != Some(requested_kind) {
        return false;
    }
    if requested_kind == "section"
        && !section.id.is_empty()
        && section_alias(&format!("section:{}", section.id)) == requested
    {
        return true;
    }
    section
        .label
        .iter()
        .map(String::as_str)
        .chain(section.aliases.iter().flatten().map(String::as_str))
        .any(|value| section_alias(value) == requested)
}

pub(crate) fn exact_footnotes(document: &PdfDocument, locator: &str, input: &PdfLookupRequest) -> Vec<usize> {
    let query = normalize_footnote(locator);
    document
        .footnotes
        .iter()
        .enumerate()
        .filter(|(_, note)| {
            input
                .occurrence
                .is_none_or(|number| note.occurrence == number)
                && input.page.is_none_or(|number| {
                    note.reference_page == Some(number) || note.body_pages.contains(&number)
                })
                && (normalize_footnote(&note.pair_id) == query
                    || normalize_footnote(&note.label) == query)
        })
        .map(|(index, _)| index)
        .collect()
}

pub(crate) fn finish<F>(
    document: &PdfDocument,
    input: &PdfLookupRequest,
    ordered_len: usize,
    selected_start: usize,
    selected_end: usize,
    mut unit_at: F,
) -> PdfStructureLookup
where
    F: FnMut(usize) -> PdfLookupUnit,
{
    if selected_end - selected_start + 1 > MAX_UNITS {
        return result(
            input,
            PdfLookupStatus::Invalid,
            Some(format!("Exact ranges are limited to {MAX_UNITS} units")),
        );
    }
    let before: Vec<_> = (selected_start.saturating_sub(input.context_blocks)..selected_start)
        .map(&mut unit_at)
        .collect();
    let selected: Vec<_> = (selected_start..=selected_end).map(&mut unit_at).collect();
    let after: Vec<_> = (selected_end + 1
        ..usize::min(ordered_len, selected_end + 1 + input.context_blocks))
        .map(unit_at)
        .collect();
    if selected.iter().any(|unit| unit.text.is_empty()) {
        return result(
            input,
            PdfLookupStatus::Unavailable,
            Some("The requested structural unit has no exact text".to_owned()),
        );
    }
    if before
        .iter()
        .chain(selected.iter())
        .chain(after.iter())
        .map(|unit| utf16_len(&unit.text))
        .sum::<usize>()
        > MAX_RETURN_CHARS
    {
        return result(
            input,
            PdfLookupStatus::Invalid,
            Some(format!(
                "Exact result exceeds {MAX_RETURN_CHARS} characters; request a narrower range"
            )),
        );
    }
    let page_numbers: HashSet<_> = before
        .iter()
        .chain(selected.iter())
        .chain(after.iter())
        .flat_map(|unit| unit.page_numbers.iter().copied())
        .collect();
    let pages = document
        .pages
        .iter()
        .filter(|page| page_numbers.contains(&page.number) && !page.evidence_text.is_empty())
        .map(|page| PdfLookupPage {
            page_number: page.number,
            text: page.evidence_text.clone(),
        })
        .collect();
    let matches = selected.iter().map(|unit| unit.id.clone()).collect();
    let mut lookup = result(input, PdfLookupStatus::Found, None);
    lookup.units = selected;
    lookup.before = before;
    lookup.after = after;
    lookup.matches = matches;
    lookup.pages = pages;
    let payload_sha256 = lookup_payload_sha256(&lookup);
    let page_text_sha256 = serialization_sha256(&lookup.pages);
    lookup.payload_sha256 = payload_sha256;
    lookup.page_text_sha256 = page_text_sha256;
    lookup
}

pub(crate) fn structure_lookup(document: &PdfDocument, input: &PdfLookupRequest) -> PdfStructureLookup {
    if !input.valid() {
        return result(
            input,
            PdfLookupStatus::Invalid,
            Some("Invalid or unbounded PDF locator".to_owned()),
        );
    }
    let kind = match input.locator_kind.as_str() {
        "page" | "line" | "paragraph" | "footnote" => input.locator_kind.as_str(),
        _ => "section",
    };
    let end_locator = input
        .end_locator
        .as_deref()
        .filter(|value| !value.is_empty());
    if kind == "line" {
        // A transcript's printed line numbers on one page ("p. 451, ll. 12-20"): each numbered
        // line that carries words, read across the row its number opens.
        let Some(page_number) = input.page else {
            return result(
                input,
                PdfLookupStatus::Invalid,
                Some("A line locator needs its page".to_owned()),
            );
        };
        let inline = numeric_range(kind, &input.locator);
        let start_number = inline
            .map(|item| item.0)
            .or_else(|| parse_ordinal(kind, &input.locator));
        let end_number = end_locator
            .map(|end| parse_ordinal(kind, end))
            .unwrap_or_else(|| inline.map(|item| item.1).or(start_number));
        let (Some(start_number), Some(end_number)) = (start_number, end_number) else {
            return result(
                input,
                PdfLookupStatus::Invalid,
                Some("Invalid exact range".to_owned()),
            );
        };
        if start_number > end_number {
            return result(
                input,
                PdfLookupStatus::Invalid,
                Some("Invalid exact range".to_owned()),
            );
        }
        // The page a reader cites is the one printed with that number; a PDF whose pages print
        // none is read by its own page numbers.
        let printed = page_number.to_string();
        let page = document
            .structure
            .nodes
            .iter()
            .find(|node| {
                node.label.as_deref().is_some_and(|label| label.starts_with("page"))
                    && node
                        .aliases
                        .as_ref()
                        .is_some_and(|aliases| aliases.iter().any(|alias| alias.trim() == printed))
            })
            .and_then(|node| node.page_indexes.first())
            .and_then(|index| document.pages.iter().find(|page| page.index == *index))
            .or_else(|| document.pages.iter().find(|page| page.number == page_number));
        let Some((page, geometry)) = page.and_then(|page| {
            document
                .passage_pages
                .iter()
                .find(|geometry| geometry.page_number == page.number)
                .map(|geometry| (page, geometry))
        }) else {
            return result(input, PdfLookupStatus::NotFound, None);
        };
        let lines = transcript_lines(document, page, geometry);
        if lines.is_empty() {
            return result(
                input,
                PdfLookupStatus::Unavailable,
                Some("This page prints no line numbers".to_owned()),
            );
        }
        let at = |number| lines.iter().position(|line| line.0 == number);
        let (Some(start), Some(end)) = (at(start_number), at(end_number)) else {
            return result(input, PdfLookupStatus::NotFound, None);
        };
        if end < start {
            return result(input, PdfLookupStatus::NotFound, None);
        }
        return finish(document, input, lines.len(), start, end, |index| {
            let (number, id, text) = &lines[index];
            PdfLookupUnit {
                id: id.clone(),
                kind: "line",
                locator: format!("page {}, line {number}", page.number),
                text: text.clone(),
                page_numbers: vec![page.number],
                confidence: Some(page.text_quality.clamp(0.0, 1.0)),
                confidence_basis: "page_text_quality",
                provenance: if page.source.is_empty() {
                    "unknown".to_owned()
                } else {
                    page.source.clone()
                },
                proposition: None,
                note: None,
            }
        });
    }
    if kind == "page" || kind == "paragraph" {
        let inline = numeric_range(kind, &input.locator);
        let start_number = inline
            .map(|item| item.0)
            .or_else(|| parse_ordinal(kind, &input.locator));
        let end_number = end_locator
            .map(|end| parse_ordinal(kind, end))
            .unwrap_or_else(|| inline.map(|item| item.1).or(start_number));
        let (Some(start_number), Some(end_number)) = (start_number, end_number) else {
            return result(
                input,
                PdfLookupStatus::Invalid,
                Some("Invalid exact range".to_owned()),
            );
        };
        if start_number > end_number {
            return result(
                input,
                PdfLookupStatus::Invalid,
                Some("Invalid exact range".to_owned()),
            );
        }
        if kind == "page" {
            let position = |number| {
                u32::try_from(number)
                    .ok()
                    .and_then(|number| document.pages.iter().position(|page| page.number == number))
            };
            let (Some(start), Some(end)) = (position(start_number), position(end_number)) else {
                return result(input, PdfLookupStatus::NotFound, None);
            };
            if end < start {
                return result(input, PdfLookupStatus::NotFound, None);
            }
            let text = ScalarText::new(document.structure.query_text());
            let nodes = nodes_by_id(document);
            return finish(document, input, document.pages.len(), start, end, |index| {
                page_unit(&document.structure, &text, &nodes, &document.pages[index])
            });
        }
        if end_number - start_number >= MAX_UNITS {
            return result(
                input,
                PdfLookupStatus::Invalid,
                Some("Paragraph range exceeds the lookup limit".to_owned()),
            );
        }
        let selected = match document.paragraph_range(start_number, end_number) {
            Ok(selected) => selected,
            Err(status) => return result(input, status, None),
        };
        let printed = document.printed_paragraphs();
        let start = printed
            .iter()
            .position(|p| std::ptr::eq(p, &selected[0]))
            .unwrap();
        let end = start + selected.len() - 1;
        let text = ScalarText::new(document.structure.query_text());
        return finish(document, input, printed.len(), start, end, |index| {
            printed_unit(&text, document, &printed[index])
        });
    }
    if kind == "footnote" {
        let start = exact_footnotes(document, &input.locator, input);
        let end = end_locator.map(|locator| exact_footnotes(document, locator, input));
        let end_matches = end.as_deref().unwrap_or(&start);
        if start.len() > 1 || end_matches.len() > 1 {
            let mut lookup = result(input, PdfLookupStatus::Ambiguous, None);
            for pair_id in start
                .iter()
                .chain(end.iter().flatten())
                .map(|index| document.footnotes[*index].pair_id.as_str())
            {
                if !lookup.matches.iter().any(|item| item == pair_id) {
                    lookup.matches.push(pair_id.to_owned());
                }
            }
            return lookup;
        }
        let (Some(start), Some(end)) = (start.first().copied(), end_matches.first().copied())
        else {
            return result(input, PdfLookupStatus::NotFound, None);
        };
        if end < start {
            return result(input, PdfLookupStatus::NotFound, None);
        }
        return finish(
            document,
            input,
            document.footnotes.len(),
            start,
            end,
            |index| footnote_unit(&document.footnotes[index]),
        );
    }
    if end_locator.is_some() {
        return result(
            input,
            PdfLookupStatus::Invalid,
            Some("Section ranges are not supported by this document contract".to_owned()),
        );
    }
    let sections = section_nodes(document);
    if input.locator_kind != "section"
        && !sections
            .iter()
            .any(|section| section.locator_kind.as_deref() == Some(input.locator_kind.as_str()))
    {
        return result(
            input,
            PdfLookupStatus::Unavailable,
            Some(format!(
                "No exact {} identifiers exist in this source PDF",
                input.locator_kind
            )),
        );
    }
    let requested = section_alias(&input.locator);
    let candidates: Vec<_> = sections
        .iter()
        .enumerate()
        .filter(|(_, section)| section_matches(section, &input.locator_kind, &requested))
        .map(|(index, _)| index)
        .collect();
    if candidates.len() > 1 {
        let mut lookup = result(input, PdfLookupStatus::Ambiguous, None);
        lookup.matches = candidates
            .into_iter()
            .map(|index| sections[index].id.clone())
            .collect();
        return lookup;
    }
    let Some(index) = candidates.first().copied() else {
        return result(input, PdfLookupStatus::NotFound, None);
    };
    let text = ScalarText::new(document.structure.query_text());
    let pages: HashMap<_, _> = document
        .pages
        .iter()
        .map(|page| (page.index, page))
        .collect();
    finish(document, input, sections.len(), index, index, |index| {
        section_unit(&document.structure, &text, &pages, sections[index], index)
    })
}
