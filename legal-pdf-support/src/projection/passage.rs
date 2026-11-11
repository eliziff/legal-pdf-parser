//! Where a cited passage, and the quotations in it, stand on a PDF's pages: each page it is read on, its lines'
//! rectangles there and the start of its words, and each quotation's rectangles, line by line and page by page. Also
//! where a Word brief's citations fall on the pages of a PDF saved from it, read by the same words.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use legal_pdf_core::model::{Page, ScalarRange};
use legal_structure_model::{utf16_len, NodeKind, ScalarText};
use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;
use super::document::PdfDocument;
use super::lookup::{PdfLookupRequest, PdfLookupStatus};
use legal_structure_model::{PrimaryRole, SourceFragment};
use std::collections::BTreeMap;

type Rect = [f64; 4];

/// A passage asked for: its id, its locator (a page's is the PDF's page, from 1, unless `physical_pages` names the
/// pages), the instrument the locator is of where the PDF prints several, and the quotations to place in it.
#[derive(Clone, Deserialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
#[serde(rename_all = "camelCase")]
pub struct PassageTarget {
    pub id: String,
    #[serde(default)]
    #[cfg_attr(feature = "binding-types", ts(optional))]
    pub physical_pages: Option<Vec<u32>>,
    pub locator_kind: String,
    pub locator: String,
    /// The instrument the locator is of, where the PDF prints several ("Canadian Charter of Rights and Freedoms" in
    /// "The Constitution Acts 1867 to 1982").
    #[serde(default)]
    #[cfg_attr(feature = "binding-types", ts(optional))]
    pub instrument: Option<String>,
    /// An item of the paragraph's list the locator names ("b" of "para 12, item (b)").
    #[serde(default)]
    #[cfg_attr(feature = "binding-types", ts(optional))]
    pub item: Option<String>,
    /// The quotations to place in the passage: each whole, or the part of its text a selection names.
    #[serde(default)]
    #[cfg_attr(feature = "binding-types", ts(as = "Option<Vec<String>>", optional))]
    pub exact_quotes: Vec<String>,
    #[serde(default)]
    #[cfg_attr(feature = "binding-types", ts(as = "Option<Vec<QuoteSelection>>", optional))]
    pub quote_selections: Vec<QuoteSelection>,
}

/// A quotation's text and the part of it (UTF-16) that is placed: its words in their context.
#[derive(Clone, Deserialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
pub struct QuoteSelection {
    pub text: String,
    pub start: usize,
    pub end: usize,
}

/// Every passage asked for, placed: the rectangles are fractions of the page as it is displayed (its visible crop box,
/// rotation applied), from its top left, what marks and links are placed by.
#[derive(Serialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
#[serde(rename_all = "camelCase")]
pub struct PassageGeometry {
    pub targets: Vec<PlacedPassage>,
}

#[derive(Serialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
#[serde(rename_all = "camelCase")]
pub struct PlacedPassage {
    pub id: String,
    pub locator_kind: String,
    pub locator: String,
    pub status: PdfLookupStatus,
    /// A paragraph's: its numbers are printed on the passage, not only counted.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "binding-types", ts(optional))]
    pub printed: Option<bool>,
    pub pages: Vec<PassagePage>,
    pub quotes: Vec<PlacedQuote>,
}

/// A page as the passage is marked on it: its lines' rectangles (none for a page asked for whole), and the start of
/// its words (2,000 UTF-16 units at most). A page that cannot be read (a scan not yet recognized) is unavailable.
#[derive(Serialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
#[serde(rename_all = "camelCase")]
pub struct PassagePage {
    pub page_number: u32,
    pub width: f64,
    pub height: f64,
    #[cfg_attr(feature = "binding-types", ts(type = "\"native\" | \"unavailable\""))]
    pub source: &'static str,
    pub passage_rects: Vec<Rect>,
    pub text: String,
    /// The passage's lines on the page, as `passage_rects` and `text` read them; tools read them, readers need not.
    #[serde(skip)]
    pub lines: Vec<PassageLine>,
}

#[derive(Serialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
#[serde(rename_all = "camelCase")]
pub struct PlacedQuote {
    pub text: String,
    pub status: PdfLookupStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "binding-types", ts(optional))]
    pub page_number: Option<u32>,
    pub rects: Vec<Rect>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "binding-types", ts(optional))]
    pub fragments: Option<Vec<QuoteFragment>>,
}

/// One page's part of a quotation that crosses pages.
#[derive(Serialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
#[serde(rename_all = "camelCase")]
pub struct QuoteFragment {
    pub page_number: u32,
    pub rects: Vec<Rect>,
}

/// A page of the passage as it is read, before it is marked.
struct ReadPage<'a> {
    page: &'a Page,
    available: bool,
    lines: Vec<PassageLine>,
}

/// A line of a passage: its part of the source line (the line's id, and the passage's span of it where it holds part),
/// that part's rectangle, text and available word boxes. A recognized line may have only line geometry.
pub struct PassageLine {
    pub id: String,
    pub rect: Rect,
    pub text: String,
    pub words: Vec<(String, Rect)>,
}

/// Words as a quotation's are matched in a passage's: case, width and apostrophe forms folded.
pub fn passage_words(text: &str) -> Vec<String> {
    static WORD: OnceLock<regex::Regex> = OnceLock::new();
    let word = WORD.get_or_init(|| regex::Regex::new(r"[\p{L}\p{N}]+(?:['\u{2019}][\p{L}\p{N}]+)*").unwrap());
    let folded = text.nfkc().collect::<String>().to_lowercase();
    word.find_iter(&folded).map(|word| word.as_str().replace('\u{2019}', "'")).collect()
}

impl PdfDocument {
    /// Each passage placed on the pages it is read on, with its quotations. A scan not yet read has no geometry; a page
    /// whose own layer reads poorly (a signature's scribble over it) and waits on recognition has its lines, read and
    /// placed as the structure took them.
    pub fn passage_geometry(&self, query: &legal_document_query::DocumentQuery, targets: Vec<PassageTarget>) -> PassageGeometry {
        let pages = self.passage_pages();
        let unavailable = self.summary.pages_needing_ocr.iter().chain(&self.summary.ocr_routed_pages).copied()
            .filter(|index| pages.get(*index).is_none_or(|page| page.lines.is_empty()))
            .collect::<HashSet<_>>();
        PassageGeometry {
            targets: targets.into_iter().map(|target| self.place(query, target, &pages, &unavailable)).collect(),
        }
    }

    /// The passage's status, its spans of the text and the pages its locator names: a paragraph's every printing, a
    /// page's own pages where they are named, else what a lookup finds (none where it finds several places).
    fn passage_of(&self, query: &legal_document_query::DocumentQuery, target: &PassageTarget) -> (PdfLookupStatus, Vec<ScalarRange>, Vec<u32>) {
        if target.locator_kind == "paragraph" {
            let (status, ranges) = self.paragraph_source_ranges(&target.locator);
            let Some(item) = target.item.as_deref() else { return (status, ranges, Vec::new()) };
            // An item of the paragraph: the one list item among its printings with that label, with what is set under it.
            let items = self.structure.nodes.iter().filter(|node| node.kind == NodeKind::ListItem
                && node.label.as_deref().is_some_and(|label| label.trim_start_matches('(').trim_end_matches(['.', ')']) == item)
                && ranges.iter().any(|range| range.start <= node.range.start && node.range.start < range.end)).collect::<Vec<_>>();
            return match items[..] {
                [node] => (status, if node.extent.is_empty() { node.ranges.clone() } else { node.extent.clone() }, Vec::new()),
                [] => (PdfLookupStatus::NotFound, Vec::new(), Vec::new()),
                _ => (PdfLookupStatus::Ambiguous, Vec::new(), Vec::new()),
            };
        }
        if let Some(physical) = target.physical_pages.as_ref().filter(|_| target.locator_kind == "page") {
            let pages = physical.iter().copied().filter(|page| *page > 0 && (*page as usize) <= self.page_count()).collect::<Vec<_>>();
            return (if pages.is_empty() { PdfLookupStatus::NotFound } else { PdfLookupStatus::Found }, Vec::new(), pages);
        }
        let lookup = self.lookup(query, &PdfLookupRequest {
            instrument: target.instrument.clone(),
            ..PdfLookupRequest::new(&target.locator_kind, &target.locator)
        });
        let ranges = if lookup.status == PdfLookupStatus::Ambiguous { Vec::new() } else {
            lookup.matches.iter()
                .filter_map(|id| self.structure.nodes.iter().find(|node| &node.id == id))
                .flat_map(|node| {
                    if (matches!(node.kind, NodeKind::Heading | NodeKind::Section) || node.role == Some(PrimaryRole::Heading)) && !node.extent.is_empty() { &node.extent[..] }
                    else if !node.ranges.is_empty() { &node.ranges[..] }
                    else { std::slice::from_ref(&node.range) }
                }).copied().collect()
        };
        let pages = lookup.units.iter().flat_map(|unit| unit.pages.iter().copied()).collect();
        (lookup.status, ranges, pages)
    }

    fn place(&self, query: &legal_document_query::DocumentQuery, target: PassageTarget, pages: &[Page], unavailable: &HashSet<usize>) -> PlacedPassage {
        let fragments: &[SourceFragment] = &self.structure.source_fragments;
        let whole_page = target.locator_kind == "page";
        let (status, mut ranges, wanted) = self.passage_of(query, &target);
        ScalarRange::normalize_union(&mut ranges);
        // The passage's part of each source line, in the line's own UTF-16.
        let mut selected = HashMap::<&str, Vec<ScalarRange>>::new();
        for fragment in fragments {
            let own = selected.entry(fragment.id.as_str()).or_default();
            let start = ranges.partition_point(|range| range.end <= fragment.range.start);
            for range in ranges[start..].iter().take_while(|range| range.start < fragment.range.end) {
                let (start, end) = (range.start.max(fragment.range.start), range.end.min(fragment.range.end));
                if start < end {
                    own.push(ScalarRange { start: start - fragment.range.start, end: end - fragment.range.start });
                }
            }
            ScalarRange::normalize_union(own);
        }
        selected.retain(|_, ranges| !ranges.is_empty());
        let selected_pages = fragments.iter().filter(|fragment| selected.contains_key(fragment.id.as_str()))
            .map(|fragment| fragment.page_index as u32 + 1).collect::<HashSet<_>>();
        let read = pages.iter()
            .filter(|page| wanted.contains(&page.number) || selected_pages.contains(&page.number))
            .map(|page| {
                let available = !unavailable.contains(&((page.number - 1) as usize))
                    && (whole_page || fragments.iter()
                        .filter(|fragment| fragment.page_index as u32 + 1 == page.number && selected.contains_key(fragment.id.as_str()))
                        .all(|fragment| page.lines.iter().any(|line| line.id == fragment.id)));
                let lines = if available { read_lines(page, whole_page, &selected) } else { Vec::new() };
                ReadPage { page, available, lines }
            })
            .collect::<Vec<_>>();
        let complete = read.iter().all(|page| page.available)
            && selected_pages.iter().all(|number| read.iter().any(|page| page.page.number == *number));
        // A passage found with no page it can be read on (a text, its every page unread or holding no line of it) is
        // unavailable, and so is every quotation in it.
        let readable = if status == PdfLookupStatus::Found
            && read.iter().all(|page| !page.available || (!whole_page && page.lines.is_empty())) {
            PdfLookupStatus::Unavailable
        } else {
            status
        };
        let quotes = target.exact_quotes.into_iter()
            .map(|text| QuoteSelection { end: utf16_len(&text), text, start: 0 })
            .chain(target.quote_selections)
            .map(|quote| place_quote(&read, readable, &quote))
            .collect();
        let printed = (target.locator_kind == "paragraph").then(|| status == PdfLookupStatus::Found && complete
            && !selected_pages.is_empty() && selected_pages.iter().all(|page| !unavailable.contains(&((*page - 1) as usize))));
        let pages = read.into_iter().map(|page| {
            let words = page.lines.iter().map(|line| line.text.as_str()).collect::<Vec<_>>().join(" ");
            let mut width = 0;
            let text = words.chars().take_while(|c| { width += c.len_utf16(); width <= 2_000 }).collect::<String>();
            PassagePage {
                page_number: page.page.number,
                width: page.page.width,
                height: page.page.height,
                source: if page.available { "native" } else { "unavailable" },
                passage_rects: if whole_page { Vec::new() } else { page.lines.iter().map(|line| fraction(line.rect, page.page)).collect() },
                text,
                lines: page.lines,
            }
        }).collect();
        PlacedPassage { id: target.id, locator_kind: target.locator_kind, locator: target.locator, status: readable, printed, pages, quotes }
    }
}

/// A page's lines as the passage holds them: every line of a page asked for whole, else each line's part of the passage,
/// its rectangle the union of its words' there.
fn read_lines(page: &Page, whole_page: bool, selected: &HashMap<&str, Vec<ScalarRange>>) -> Vec<PassageLine> {
    let mut lines = Vec::new();
    for line in &page.lines {
        if whole_page {
            let words = line.words.iter().map(|word| (word.text.clone(), word.bbox)).collect::<Vec<_>>();
            let text = if words.is_empty() { line.text.clone() }
                else { words.iter().map(|(text, _)| text.as_str()).collect::<Vec<_>>().join(" ") };
            lines.push(PassageLine { id: line.id.clone(), rect: line.bbox,
                text, words });
            continue;
        }
        let source = ScalarText::new(&line.text);
        for range in selected.get(line.id.as_str()).into_iter().flatten() {
            let id = format!("{}:{}-{}", line.id, range.start, range.end);
            // A recognized line has its measured rectangle even where its recognizer supplies no word boxes.
            // It can be marked whole; a partial line still needs word witnesses.
            if line.words.is_empty() {
                // Canonical marker recovery can add a bracket to the logical line. Its range still
                // covers the whole original recognition; clip to that original for its physical reading.
                if page.source == "native" || range.start == 0 && range.end >= source.utf16_len() {
                    let text = source.slice_utf16(range.start..range.end.min(source.utf16_len())).unwrap_or_default().to_owned();
                    let words = if page.source == "native" { vec![(text.clone(), line.bbox)] } else { Vec::new() };
                    lines.push(PassageLine { id, rect: line.bbox, text, words });
                }
                continue;
            }
            // Physical word witnesses use Unicode scalars; source fragments use UTF-16.
            let words = line.words.iter().filter_map(|word| {
                let start = source.utf16_at_scalar(word.start)?.max(range.start);
                let end = source.utf16_at_scalar(word.end)?.min(range.end);
                if start >= end {
                    return None;
                }
                Some((source.slice_utf16(start..end)?.to_owned(), word.bbox))
            }).collect::<Vec<_>>();
            // A mark no word box covers costs that mark, not the page its line stands on.
            let Some(rect) = words.iter().map(|(_, rect)| *rect).reduce(union) else { continue };
            let text = words.iter().map(|(text, _)| text.as_str()).collect::<Vec<_>>().join(" ");
            lines.push(PassageLine { id, rect, text, words });
        }
    }
    lines
}

/// A rectangle as a fraction of its page as it is displayed, clamped to it.
fn fraction(rect: Rect, page: &Page) -> Rect {
    [rect[0] / page.width, rect[1] / page.height, rect[2] / page.width, rect[3] / page.height].map(|value| value.clamp(0.0, 1.0))
}

fn union(a: Rect, b: Rect) -> Rect {
    [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])]
}

/// Where a quotation stands in a passage's pages, its words read once in its context and found once among the passage's
/// words in order, a page apart at most: its rectangle on each line it crosses, page by page. A passage not found, or a
/// quotation found other than once, places none.
fn place_quote(pages: &[ReadPage], status: PdfLookupStatus, quote: &QuoteSelection) -> PlacedQuote {
    let context = ScalarText::new(&quote.text);
    let slice = |start: usize, end: usize| context.slice_utf16(start..end).unwrap_or("");
    let text = slice(quote.start, quote.end);
    let none = |status| PlacedQuote { text: text.to_owned(), status, page_number: None, rects: Vec::new(), fragments: None };
    if status != PdfLookupStatus::Found {
        return none(status);
    }
    let needle = passage_words(&quote.text);
    if needle.len() < 2 {
        return none(PdfLookupStatus::Invalid);
    }
    let (before, selected) = (passage_words(slice(0, quote.start)), passage_words(text));
    let after = passage_words(slice(quote.end, context.utf16_len()));
    if selected.is_empty() || [&before[..], &selected[..], &after[..]].concat() != needle {
        return none(PdfLookupStatus::Invalid);
    }
    // Line rectangles support a whole paragraph, without inventing boxes for words within it.
    if pages.iter().filter(|page| page.available).any(|page| page.lines.iter().any(|line| line.words.is_empty())) {
        return none(PdfLookupStatus::Unavailable);
    }
    let words = pages.iter().filter(|page| page.available)
        .flat_map(|page| page.lines.iter().flat_map(move |line| line.words.iter().flat_map(move |(text, rect)|
            passage_words(text).into_iter().map(move |value| (value, page.page.number, line.id.as_str(), *rect)))))
        .collect::<Vec<_>>();
    let hits = (0..words.len()).filter(|&start| needle.iter().enumerate().all(|(offset, value)| {
        words.get(start + offset).is_some_and(|current| &current.0 == value && (offset == 0 || current.1 <= words[start + offset - 1].1 + 1))
    })).collect::<Vec<_>>();
    if hits.len() != 1 {
        return none(if hits.is_empty() { PdfLookupStatus::NotFound } else { PdfLookupStatus::Ambiguous });
    }
    // One quotation can span pages: one mark, its fragments a page each, a line's rectangle the union of its words'.
    let mut fragments: Vec<(u32, Vec<(&str, Rect)>)> = Vec::new();
    let first = hits[0] + before.len();
    for (_, number, line, rect) in &words[first..first + selected.len()] {
        let at = fragments.iter().position(|(page, _)| page == number).unwrap_or_else(|| {
            fragments.push((*number, Vec::new()));
            fragments.len() - 1
        });
        let lines = &mut fragments[at].1;
        match lines.iter_mut().find(|(id, _)| id == line) {
            Some((_, joined)) => *joined = union(*joined, *rect),
            None => lines.push((line, *rect)),
        }
    }
    let page = |number: u32| pages.iter().find(|page| page.page.number == number).expect("a quotation's page").page;
    let fragments = fragments.into_iter()
        .map(|(page_number, lines)| QuoteFragment { page_number, rects: lines.into_iter().map(|(_, rect)| fraction(rect, page(page_number))).collect() })
        .collect::<Vec<_>>();
    PlacedQuote { text: text.to_owned(), status: PdfLookupStatus::Found, page_number: Some(fragments[0].page_number),
        rects: fragments[0].rects.clone(), fragments: Some(fragments) }
}

/// A unit of a Word brief whose citations are placed on the pages of a PDF saved from it, as its draft holds it: the
/// note it is (none for the body), its text, and its citations' ids.
#[derive(Deserialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
#[serde(rename_all = "camelCase")]
pub struct BriefUnit {
    #[cfg_attr(feature = "binding-types", ts(type = "number | null"))]
    footnote_id: Option<serde::de::IgnoredAny>,
    text: String,
    occurrence_ids: Vec<String>,
}

/// A citation of the brief (the draft's, by its id): its span of its unit's text (UTF-16) and its own text.
#[derive(Deserialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
pub struct BriefOccurrence {
    start: usize,
    end: usize,
    text: String,
}

/// The page (from 1) of each citation of a Word brief in a PDF saved from it (`pages`, each page's text), by the
/// citation's id. The brief's words are read in order, its body and its notes each onward from their last match, so a
/// repeated citation keeps its place: a citation is found with the four words before and after it, then with those
/// before or those after, from where its stream left off and then from the start; the citation alone is tried last,
/// and is not read on from, for it also matches the brief's own table of authorities, which lies past every later
/// citation. A citation that cannot be placed has none.
pub fn brief_occurrence_pages(pages: &[String], units: &[BriefUnit], occurrences: &HashMap<String, BriefOccurrence>)
    -> BTreeMap<String, u32> {
    let words: Vec<(String, u32)> = pages.iter().enumerate()
        .flat_map(|(at, text)| passage_words(text).into_iter().map(move |word| (word, at as u32 + 1))).collect();
    let mut positions = HashMap::<&str, Vec<usize>>::new();
    for (at, (word, _)) in words.iter().enumerate() {
        positions.entry(word.as_str()).or_default().push(at);
    }
    let find = |needle: &[String], from: usize| positions.get(needle.first()?.as_str())?.iter().copied()
        .find(|&at| at >= from && needle.iter().enumerate().all(|(offset, word)| words.get(at + offset).is_some_and(|(read, _)| read == word)));
    let mut cursors = [0usize; 2];
    let mut placed = BTreeMap::new();
    for unit in units {
        let text = ScalarText::new(&unit.text);
        let stream = usize::from(unit.footnote_id.is_some());
        for (id, occurrence) in unit.occurrence_ids.iter().filter_map(|id| Some((id, occurrences.get(id)?))) {
            let before = passage_words(text.slice_utf16(0..occurrence.start).unwrap_or(""));
            let before = before[before.len().saturating_sub(4)..].to_vec();
            let core = passage_words(&occurrence.text);
            let after = passage_words(text.slice_utf16(occurrence.end..text.utf16_len()).unwrap_or(""))
                .into_iter().take(4).collect::<Vec<_>>();
            let cursor = cursors[stream];
            let contexts = [([&before[..], &core[..], &after[..]].concat(), before.len()), ([&before[..], &core[..]].concat(), before.len()),
                ([&core[..], &after[..]].concat(), 0)];
            let attempts = contexts.iter().map(|(needle, skip)| (needle, *skip, cursor, false))
                .chain(contexts.iter().map(|(needle, skip)| (needle, *skip, 0, false)))
                .chain([(&core, 0, cursor, true)]);
            for (needle, skip, from, alone) in attempts {
                let Some(at) = (needle.len() > 1).then(|| find(needle, from)).flatten() else { continue };
                placed.insert(id.clone(), words[at + skip].1);
                if !alone {
                    cursors[stream] = at + needle.len();
                }
                break;
            }
        }
    }
    placed
}
