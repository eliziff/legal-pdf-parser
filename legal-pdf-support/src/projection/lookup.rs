//! A lookup by page, paragraph, section or footnote: the request, the units it returns, and how
//! each locator finds its units.

use super::*;

/// A paragraph a PDF numbers in print ("[12]", or 12 in the margin): its number and its printings, the structure's
/// paragraph nodes that print it (a parallel report's columns), the first printing's first, as the structure engine
/// reads them (`paragraph_printings`).
pub type PrintedParagraph = (usize, Vec<usize>);

#[derive(Clone, Deserialize, Serialize)]
pub struct PdfLookupRequest {
    pub locator_kind: String,
    pub locator: String,
    #[serde(default)]
    pub end_locator: Option<String>,
    #[serde(default)]
    pub context_blocks: usize,
    #[serde(default)]
    pub page: Option<u32>,
    #[serde(default)]
    pub occurrence: Option<usize>,
    /// The instrument a provision is of, where the PDF prints several ("Canadian Charter of Rights and Freedoms" in
    /// "The Constitution Acts 1867 to 1982").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instrument: Option<String>,
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
            instrument: None,
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
            && self.instrument.as_deref().is_none_or(|instrument| utf16_len(instrument) <= 200)
    }
}

/// A unit a lookup returns: its node, as the model serializes it, and what no node holds: its text as the lookup reads
/// it, how far its pages' text is trusted, a numbered paragraph's other printings (a parallel report's columns), and a
/// note's count of its label's printings up to it and the page its first call stands on.
#[derive(Serialize)]
pub struct PdfLookupUnit {
    pub node: StructureNode,
    pub text: String,
    /// The pages (from 1) it stands on (`page_numbers`).
    pub pages: Vec<u32>,
    /// The spans of the document's query text its node holds (`DocumentStructure::query_ranges`).
    pub ranges: Vec<ScalarRange>,
    pub confidence: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub printings: Vec<StructureNode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub occurrence: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_page: Option<u32>,
}

impl PdfLookupUnit {
    fn new(node: &StructureNode, text: String, confidence: Option<f64>) -> Self {
        Self { node: node.clone(), text, pages: Vec::new(), ranges: Vec::new(), confidence, printings: Vec::new(), occurrence: None, reference_page: None }
    }

    /// The pages (from 1) the unit stands on: its note's first call's, then its node's and its printings'.
    pub fn page_numbers(&self) -> Vec<u32> {
        let mut pages: Vec<u32> = self.reference_page.into_iter().collect();
        for page in std::iter::once(&self.node).chain(&self.printings).flat_map(|node| &node.page_indexes) {
            let page = *page as u32 + 1;
            if !pages.contains(&page) {
                pages.push(page);
            }
        }
        pages
    }
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
    /// The kind of unit the request asks for: a page, a transcript's line, a paragraph, a note or a provision.
    pub kind: &'static str,
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
        kind: match input.locator_kind.as_str() {
            "page" => "page",
            "line" => "line",
            "paragraph" => "paragraph",
            "footnote" => "footnote",
            _ => "section",
        },
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

pub(crate) fn range_text(text: &ScalarText<'_>, ranges: &[ScalarRange]) -> String {
    ranges
        .iter()
        .filter_map(|range| text.slice_utf16(range.start..range.end))
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) fn rendered_slice(
    text: &ScalarText<'_>,
    document: &DocumentStructure,
    node: &StructureNode,
) -> String {
    range_text(text, &document.query_ranges(node))
}

/// The PDF's provisions, in the order its text prints them, within `scope` where it is given: a node located by a
/// kind. A heading's or a constituent document's extent is no provision: it has no identifier to look up; and a
/// parallel translation's provisions repeat the body's, which is read.
pub(crate) fn section_nodes(document: &PdfDocument, scope: Option<(usize, usize)>) -> Vec<&StructureNode> {
    let mut sections: Vec<&StructureNode> = document
        .structure
        .nodes
        .iter()
        .filter(|node| {
            node.kind == NodeKind::Section
                && node.locator_kind.as_deref().is_some_and(|kind| !kind.is_empty())
                && node.grammar.as_deref() != Some("translation")
                && scope.is_none_or(|(start, end)| start <= node.range.start && node.range.start < end)
        })
        .collect();
    sections.sort_by_key(|node| node.range.start);
    sections
}

/// A heading's words, without a Part's number before them or a note's number after a year, in capitals.
fn title_words(text: &str) -> Vec<String> {
    let mut words = text.split(|c: char| !c.is_alphanumeric()).filter(|word| !word.is_empty())
        .map(str::to_uppercase).collect::<Vec<_>>();
    if words.first().is_some_and(|word| word == "PART") && words.len() > 2 {
        words.drain(..2);
    }
    let year = |word: &String| word.len() == 4 && word.chars().all(|c| c.is_ascii_digit());
    if words.len() > 1 && words.last().is_some_and(|note| note.len() <= 3 && note.chars().all(|c| c.is_ascii_digit()))
        && year(&words[words.len() - 2]) {
        words.pop();
    }
    words
}

/// Where one instrument of a PDF that prints several is printed ("The Constitution Acts 1867 to 1982"): from the
/// heading that is its title ("CONSTITUTION ACT, 1982", "PART I Canadian Charter of Rights and Freedoms", a note's
/// number after it read past) to the next heading that is another Act's title in capitals ("CANADA ACT 1982", "LOI
/// CONSTITUTIONNELLE DE 1867"). A contents list that names it comes before its body: the last heading that is its title
/// opens it. None where no heading is its title, or where the PDF prints one Act.
fn instrument_scope(document: &PdfDocument, text: &ScalarText<'_>, instrument: &str) -> Option<(usize, usize)> {
    let wanted = title_words(instrument);
    if wanted.is_empty() {
        return None;
    }
    // Each heading's start and own text, in the order the text prints them.
    let mut headings: Vec<(usize, String)> = document.structure.nodes.iter()
        .filter(|node| node.kind == NodeKind::Heading || node.role == Some(legal_structure_model::PrimaryRole::Heading))
        .map(|node| (node.range.start, range_text(text, if node.ranges.is_empty() { std::slice::from_ref(&node.range) } else { &node.ranges })))
        .collect();
    headings.sort_by_key(|(start, _)| *start);
    // Another Act's title: a heading in capitals that ends in its year.
    let act_title = |text: &str| {
        let words = title_words(text);
        !text.chars().any(char::is_lowercase) && words.iter().any(|word| word == "ACT" || word == "LOI")
            && words.last().is_some_and(|year| year.len() == 4 && year.chars().all(|c| c.is_ascii_digit()))
    };
    let acts = headings.iter().filter(|(_, text)| act_title(text)).map(|(_, text)| title_words(text)).collect::<HashSet<_>>();
    if acts.len() < 2 {
        return None;
    }
    let (at, (start, _)) = headings.iter().enumerate().rev().find(|(_, (_, text))| title_words(text) == wanted)?;
    let end = headings[at + 1..].iter().find(|(_, text)| act_title(text) && title_words(text) != wanted)
        .map_or(usize::MAX, |(end, _)| *end);
    Some((*start, end))
}

/// The label a Part, Division or Schedule's heading takes in the instrument grammar ("part2", "div2", "schedA"): a
/// Part's or Division's number in figures or capital roman numerals. None for another kind; an empty label for a
/// number neither reads.
fn container_label(kind: &str, locator: &str) -> Option<String> {
    match kind {
        "part" | "division" => {
            let number = locator.trim().trim_start_matches(|c: char| c.is_alphabetic() && !"IVXLCDM".contains(c)).trim();
            let value = number.parse::<u32>().ok().or_else(|| legal_structure_model::layout::roman_value(number)
                .filter(|_| number.chars().all(|c| c.is_ascii_uppercase())));
            Some(value.map_or_else(String::new, |value| format!("{}{value}", if kind == "part" { "part" } else { "div" })))
        }
        "schedule" => Some(format!("sched{}", locator.trim().to_lowercase())),
        _ => None,
    }
}

/// A declared paragraph as a lookup unit: its first printing, with its other printings and the text of them all; trusted
/// as its first page's text is.
pub(crate) fn printed_unit(
    text: &ScalarText<'_>,
    document: &PdfDocument,
    (_, printings): &PrintedParagraph,
) -> PdfLookupUnit {
    let nodes: Vec<&StructureNode> = printings.iter().map(|&index| &document.structure.nodes[index]).collect();
    let mut ranges: Vec<ScalarRange> = nodes.iter().flat_map(|node| document.structure.query_ranges(node).into_owned()).collect();
    ScalarRange::normalize_union(&mut ranges);
    let first = nodes.iter().flat_map(|node| node.page_indexes.iter().copied()).min();
    let page = first.and_then(|index| document.page_nodes().find(|page| page_index(page) == index));
    let mut unit = PdfLookupUnit::new(nodes[0], range_text(text, &ranges),
        page.map(|page| page_confidence(&document.summary.page_text_quality, page)));
    unit.printings = nodes[1..].iter().map(|node| (*node).clone()).collect();
    unit
}

pub(crate) fn page_unit(document: &DocumentStructure, text: &ScalarText<'_>, quality: &[f64], page: &StructureNode) -> PdfLookupUnit {
    PdfLookupUnit::new(page, rendered_slice(text, document, page), Some(page_confidence(quality, page)))
}

/// A page's printed transcript lines in reading order: each line number the structure labels
/// (its `LineLabel` nodes), its node, and the words set across its row; a number on a
/// blank line is left out.
pub(crate) fn transcript_lines<'a>(
    document: &'a PdfDocument,
    page: &StructureNode,
    geometry: &Page,
) -> Vec<(usize, &'a StructureNode, String)> {
    let height = |rect: &[f64; 4]| (rect[3] - rect[1]).max(0.1);
    document
        .structure
        .nodes
        .iter()
        .filter(|node| {
            node.role == Some(legal_structure_model::PrimaryRole::LineLabel)
                && node.page_indexes.contains(&page_index(page))
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
                        && line.bbox[3].min(label.bbox[3]) - line.bbox[1].max(label.bbox[1])
                            > 0.5 * height(&line.bbox).min(height(&label.bbox))
                })
                .collect::<Vec<_>>();
            row.sort_by(|left, right| left.bbox[0].total_cmp(&right.bbox[0]));
            let text = row
                .iter()
                .map(|line| line.text.trim())
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            (!text.is_empty()).then_some((number, node, text))
        })
        .collect()
}

/// The document's notes, in order.
pub(crate) fn notes(document: &PdfDocument) -> Vec<&StructureNode> {
    document.structure.nodes.iter().filter(|node| matches!(node.kind, NodeKind::Footnote | NodeKind::Endnote)).collect()
}

/// The page (counted from 1) whose text holds `offset`.
fn page_at(document: &PdfDocument, offset: usize) -> Option<u32> {
    document.structure.nodes.iter()
        .find(|node| node.kind == NodeKind::Page && node.range.start <= offset && offset < node.range.end)
        .and_then(|node| node.page_indexes.first())
        .map(|page| *page as u32 + 1)
}

/// A note as a lookup returns it: its body, the times its label was printed up to it, and the page its first call stands
/// on.
pub(crate) fn footnote_unit(text: &ScalarText<'_>, document: &PdfDocument, notes: &[&StructureNode], at: usize) -> PdfLookupUnit {
    let note = notes[at];
    let mut unit = PdfLookupUnit::new(note, rendered_slice(text, &document.structure, note).trim().to_owned(), None);
    unit.occurrence = Some(notes[..=at].iter().filter(|other| other.label == note.label).count());
    unit.reference_page = note.calls.first().and_then(|call| page_at(document, call.start));
    unit
}

/// A provision as a lookup returns it, trusted as the least trusted of its pages.
pub(crate) fn section_unit(
    document: &DocumentStructure,
    text: &ScalarText<'_>,
    quality: &[f64],
    pages: &HashMap<usize, &StructureNode>,
    section: &StructureNode,
) -> PdfLookupUnit {
    let confidence = section.page_indexes.iter().filter_map(|index| pages.get(index))
        .map(|page| page_confidence(quality, page)).reduce(f64::min);
    PdfLookupUnit::new(section, rendered_slice(text, document, section).trim().to_owned(), confidence)
}

pub fn numeric_range(kind: &str, raw: &str) -> Option<(usize, usize)> {
    let value = nfkc_text(raw);
    let value = value
        .trim()
        .trim_start_matches('#')
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim();
    let prefix = match kind {
        "page" => legal_pdf_core::regex!(r"(?i)^(?:pages?|pp?\.?)[\s:_=-]*"),
        "line" => legal_pdf_core::regex!(r"(?i)^(?:lines?|ll?\.?)[\s:_=-]*"),
        _ => legal_pdf_core::regex!(r"(?i)^(?:paragraphs?|paras?|pars?)\.?[\s:_=-]*"),
    };
    let stripped = prefix.replace(value, "");
    let captures = legal_pdf_core::regex!(r"(?i)^(\d{1,6})\s*(?:-|–|—|\.\.|to)\s*(\d{1,6})$")
        .captures(&stripped)?;
    // An end shorter than its start keeps the start's leading figures: "142-54" runs to 154, "138-9" to 139.
    let (first, end) = (&captures[1], &captures[2]);
    let last = if end.len() < first.len() { format!("{}{end}", &first[..first.len() - end.len()]) } else { end.to_owned() };
    Some((first.parse().ok()?, last.parse().ok()?))
}

pub(crate) fn normalize_footnote(raw: &str) -> String {
    let value = nfkc_text(raw);
    legal_pdf_core::regex!(r"(?i)^(?:footnotes?|notes?|fn)\s*[#.:_-]?\s*")
        .replace(value.trim(), "")
        .to_lowercase()
}

pub(crate) fn normalized_section(raw: &str) -> Option<String> {
    let value = nfkc_text(raw);
    let compact: String = legal_pdf_core::regex!(r"(?i)^(?:ss?\.?|sections?)\s*")
        .replace(value.trim(), "")
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    let normalized = legal_structure_model::structure_analysis().numbered_section_locator(&compact);
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

pub(crate) fn section_matches(section: &StructureNode,
    requested_kind: &str,
    requested: &str,
) -> bool {
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

pub(crate) fn exact_footnotes(document: &PdfDocument, notes: &[&StructureNode], locator: &str, input: &PdfLookupRequest) -> Vec<usize> {
    let query = normalize_footnote(locator);
    (0..notes.len()).filter(|&at| {
        let note = notes[at];
        let label = note.label.as_deref().unwrap_or("");
        input.occurrence.is_none_or(|number| notes[..=at].iter().filter(|other| other.label.as_deref().unwrap_or("") == label).count() == number)
            && input.page.is_none_or(|number| note.calls.first().and_then(|call| page_at(document, call.start)) == Some(number)
                || note.page_indexes.iter().any(|page| *page as u32 + 1 == number))
            && (normalize_footnote(&note.id) == query || normalize_footnote(label) == query)
    }).collect()
}

/// Both ends of a range found, the first not after the last.
fn ordered(start: Option<usize>, end: Option<usize>) -> Option<(usize, usize)> {
    start.zip(end).filter(|(start, end)| start <= end)
}

pub(crate) fn finish<F>(
    document: &PdfDocument,
    text: &ScalarText<'_>,
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
    // Each unit with the pages it stands on.
    let mut unit_at = |index| {
        let mut unit = unit_at(index);
        unit.pages = unit.page_numbers();
        unit.ranges = document.structure.query_ranges(&unit.node).into_owned();
        unit
    };
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
        .flat_map(|unit| unit.pages.iter().copied())
        .collect();
    // The text of each page a unit is on, as a page lookup returns it (`PdfDocument::page_texts`).
    let pages = document
        .page_nodes()
        .filter(|page| page_numbers.contains(&page_number(page)))
        .filter_map(|page| {
            let text = rendered_slice(text, &document.structure, page);
            (!text.is_empty()).then_some(PdfLookupPage { page_number: page_number(page), text })
        })
        .collect();
    let matches = selected.iter().map(|unit| unit.node.id.clone()).collect();
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

pub(crate) fn structure_lookup(document: &PdfDocument,
    input: &PdfLookupRequest,
) -> PdfStructureLookup {
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
    let text = ScalarText::new(document.structure.query_text());
    let end_locator = input
        .end_locator
        .as_deref()
        .filter(|value| !value.is_empty());
    // The numbers a page, paragraph or line locator runs from and to, in order.
    let numbers = || {
        let inline = numeric_range(kind, &input.locator);
        let start_number = inline
            .map(|item| item.0)
            .or_else(|| legal_structure_model::structure_analysis().parse_ordinal(kind, &input.locator));
        let end_number = end_locator
            .map(|end| legal_structure_model::structure_analysis().parse_ordinal(kind, end))
            .unwrap_or_else(|| inline.map(|item| item.1).or(start_number));
        match (start_number, end_number) {
            (Some(start), Some(end)) if start <= end => Ok((start, end)),
            _ => Err(result(
                input,
                PdfLookupStatus::Invalid,
                Some("Invalid exact range".to_owned()),
            )),
        }
    };
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
        let (start_number, end_number) = match numbers() {
            Ok(numbers) => numbers,
            Err(invalid) => return invalid,
        };
        // The page a reader cites is the one printed with that number; a PDF whose pages print
        // none is read by its own page numbers.
        let printed = page_number.to_string();
        let page = document
            .page_nodes()
            .find(|page| page.aliases.as_ref().is_some_and(|aliases| aliases.iter().any(|alias| alias.trim() == printed)))
            .or_else(|| document.page_nodes().find(|page| self::page_number(page) == page_number));
        let passage = document.passage_pages();
        let Some((page, geometry)) = page.and_then(|page| {
            passage
                .iter()
                .find(|geometry| geometry.number == self::page_number(page))
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
        let Some((start, end)) = ordered(at(start_number), at(end_number)) else {
            return result(input, PdfLookupStatus::NotFound, None);
        };
        return finish(document, &text, input, lines.len(), start, end, |index| {
            let (_, node, words) = &lines[index];
            PdfLookupUnit::new(node, words.clone(), Some(page_confidence(&document.summary.page_text_quality, page)))
        });
    }
    if kind == "page" || kind == "paragraph" {
        let (start_number, end_number) = match numbers() {
            Ok(numbers) => numbers,
            Err(invalid) => return invalid,
        };
        if kind == "page" {
            let pages: Vec<&StructureNode> = document.page_nodes().collect();
            let position = |number| {
                u32::try_from(number)
                    .ok()
                    .and_then(|number| pages.iter().position(|page| page_number(page) == number))
            };
            let Some((start, end)) = ordered(position(start_number), position(end_number)) else {
                return result(input, PdfLookupStatus::NotFound, None);
            };
            return finish(document, &text, input, pages.len(), start, end, |index| {
                page_unit(&document.structure, &text, &document.summary.page_text_quality, pages[index])
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
        return finish(document, &text, input, printed.len(), start, end, |index| {
            printed_unit(&text, document, &printed[index])
        });
    }
    if kind == "footnote" {
        let notes = notes(document);
        let start = exact_footnotes(document, &notes, &input.locator, input);
        let end = end_locator.map(|locator| exact_footnotes(document, &notes, locator, input));
        let end_matches = end.as_deref().unwrap_or(&start);
        if start.len() > 1 || end_matches.len() > 1 {
            let mut lookup = result(input, PdfLookupStatus::Ambiguous, None);
            for pair_id in start
                .iter()
                .chain(end.iter().flatten())
                .map(|index| notes[*index].id.as_str())
            {
                if !lookup.matches.iter().any(|item| item == pair_id) {
                    lookup.matches.push(pair_id.to_owned());
                }
            }
            return lookup;
        }
        let Some((start, end)) = ordered(start.first().copied(), end_matches.first().copied())
        else {
            return result(input, PdfLookupStatus::NotFound, None);
        };
        return finish(document, &text, input, notes.len(), start, end, |index| footnote_unit(&text, document, &notes, index));
    }
    // One instrument of a PDF printing several is read on its own; a PDF printing one is read whole.
    let scope = input.instrument.as_deref().and_then(|instrument| instrument_scope(document, &text, instrument));
    let sections = section_nodes(document, scope);
    // A code's rules are its sections.
    let kind = if input.locator_kind == "rule" { "section" } else { input.locator_kind.as_str() };
    let container = container_label(kind, &input.locator).is_some();
    if kind != "section"
        && !container
        && !sections.iter().any(|section| section.locator_kind.as_deref() == Some(kind))
    {
        return result(
            input,
            PdfLookupStatus::Unavailable,
            Some(format!("No exact {kind} identifiers exist in this source PDF")),
        );
    }
    // The provisions a locator names: a Part, Division or Schedule the one its heading's label names, any other the one
    // its label or printing names.
    let named = |locator: &str| -> Vec<usize> {
        let label = container_label(kind, locator);
        let requested = section_alias(locator);
        (0..sections.len()).filter(|&index| match &label {
            Some(label) => !label.is_empty() && sections[index].label.as_deref() == Some(label.as_str()),
            None => section_matches(sections[index], kind, &requested),
        }).collect()
    };
    // A range ("49-51", "5-2-5-3", or a locator to an end locator) runs from its first provision through its last,
    // unless a provision is itself numbered so ("1-2"): its dash is the one both ends are found at.
    let (first, last) = match end_locator {
        Some(end) => (named(&input.locator), named(end)),
        None => {
            let whole = named(&input.locator);
            let dashed = || input.locator.char_indices().filter(|(_, c)| matches!(c, '-' | '\u{2013}')).find_map(|(at, dash)| {
                let (first, last) = (named(input.locator[..at].trim()), named(input.locator[at + dash.len_utf8()..].trim()));
                (!first.is_empty() && !last.is_empty()).then_some((first, last))
            });
            if whole.is_empty() { dashed().unwrap_or_default() } else { (whole.clone(), whole) }
        }
    };
    // A label the body prints twice is ambiguous, as is a range whose last provision is printed before its first.
    if first.len() > 1 || last.len() > 1 || first.first().zip(last.first()).is_some_and(|(first, last)| first > last) {
        let mut lookup = result(input, PdfLookupStatus::Ambiguous, None);
        for index in first.iter().chain(&last) {
            if !lookup.matches.contains(&sections[*index].id) {
                lookup.matches.push(sections[*index].id.clone());
            }
        }
        return lookup;
    }
    let (Some(&first), Some(&last)) = (first.first(), last.first()) else {
        return result(input, PdfLookupStatus::NotFound, None);
    };
    let pages: HashMap<_, _> = document.page_nodes().map(|page| (page_index(page), page)).collect();
    if first == last {
        return finish(document, &text, input, sections.len(), first, first, |index| {
            section_unit(&document.structure, &text, &document.summary.page_text_quality, &pages, sections[index])
        });
    }
    // A range's provisions are the outermost from its first through its last, each with what it holds.
    let spanned: HashSet<&str> = sections[first..=last].iter().map(|section| section.id.as_str()).collect();
    let span: Vec<&StructureNode> = sections[first..=last].iter().copied()
        .filter(|section| section.parent_id.as_deref().is_none_or(|parent| !spanned.contains(parent)))
        .collect();
    finish(document, &text, input, span.len(), 0, span.len() - 1, |index| {
        section_unit(&document.structure, &text, &document.summary.page_text_quality, &pages, span[index])
    })
}
