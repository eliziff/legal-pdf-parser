//! Each page's note band: which of its lines are footnotes or endnotes, read page by page through a
//! document with its endnote sequence carried from page to page. A page is read in steps:
//!
//! 1. `PageReading`: the type and geometry its notes are read against (body and note sizes, the
//!    separator rule unless a table owns it, its table cells).
//! 2. `paragraph_labels`: lines whose number is a numbered paragraph's, never a note's.
//! 3. `label_candidates`: lines that open with a note label, and those whose label is suppressed.
//! 4. `PageFacts`: what the candidates say of the page (their runs and sizes, a notes heading, a
//!    structural reset).
//! 5. `continue_endnotes`: where an endnote sequence carried from the page before resumes.
//! 6. `margin_notes`: notes set in a margin column.
//! 7. `labels`: the candidates that are notes, and whether the page is an endnote page.
//! 8. `running_heads`: an endnote page's running head.
//! 9. `note_cut` / `column_cuts`: where the band begins, across the page or per column.
//! 10. `Band::holds`: the lines the band holds; a notes heading, a table's cell, furniture, a
//!     source's heading and a numbered paragraph's line are none of them.
//!
//! The band then gives back the endnote runs the body never refers to (`verify_endnotes`), and says
//! of each page whether it read a note there and where its band begins. It reads a document's lines
//! (`Document`: its evidence and fences) on its pages and writes nothing (`read_band`).

use super::lines::{
    citation_shaped_candidate, continuing_note_heading, endnote_heading, has_prior_reference, longest_label_run,
    structural_reset_heading,
};
use legal_pdf_core::model::{Diagnostic, Line, Page};
use legal_pdf_core::median;
use crate::fences::Fences;
use crate::headings::recognized_page;
use crate::layout::{
    aligned_note_body_among, band_geometry_top, columns_of, kept_items, line_center_x, line_width,
    margin_note_model, standalone_note_label, ColumnKind, ColumnModel,
};
use crate::model::{
    Evidence, NoteLabel, PageView, Role, Slot, Slots, SourceTag, SUPPRESS_COMMA, SUPPRESS_CORNER, SUPPRESS_FOOT,
    SUPPRESS_WRAPPED,
};
use crate::text::text::all_caps;
use legal_structure_model::text::{is_note_symbol, label_prefix, line_start_label_prefix, normalize_label, LabelPrefix};
use std::collections::BTreeSet;

/// A note's kind: at its page's foot, or among the notes collected at a part's or the document's end.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum NoteKind {
    Footnote,
    Endnote,
}

/// What the band reads of a document: each note line and its kind; the lines whose number it read
/// as a numbered paragraph's; the labels it read as no note's; an endnote page's running heads; the
/// notes headings; what it read of each page (`BandPageReading`, by the pages' order); and the
/// warnings of a band drawn on weak geometry.
#[derive(Debug, Default)]
pub(crate) struct BandReading {
    pub(crate) notes: Vec<(Slot, NoteKind)>,
    pub(crate) paragraph_labels: Vec<Slot>,
    pub(crate) suppressed: Vec<Slot>,
    pub(crate) running_heads: Vec<Slot>,
    pub(crate) headings: Vec<Slot>,
    pub(crate) pages: Vec<BandPageReading>,
    pub(crate) diagnostics: Vec<Diagnostic>,
}

/// What the band read of a page: whether it read a note there (once the endnote runs the body
/// never refers to are given back), where its band begins across the page, and the label the
/// document's next note takes by the notes read before the page.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct BandPageReading {
    pub(crate) noted: bool,
    pub(crate) cut: Option<f64>,
    pub(crate) next_label: u32,
}

/// What the band reads a document's lines by: the pages' slots, the evidence (each line's type size, note label and
/// source tag), the fences (each line's role, cell, held paragraph and margin number; each page's separator, columns and
/// whether it is a contents page), which counted runs' members open a unit in their document's body type (`unit`, by
/// slot), and the document's body type size, against which notes are set small (0 where unknown: each page then reads
/// its own).
#[derive(Clone, Copy)]
pub(crate) struct Document<'a> {
    pub(crate) slots: &'a Slots,
    pub(crate) evidence: &'a Evidence,
    pub(crate) fences: &'a Fences,
    pub(crate) unit: &'a [bool],
    pub(crate) body_size: f64,
}

impl Document<'_> {
    /// A line the band never reads: a transcript's line number, a contents list's line.
    pub(crate) fn excluded(&self, at: usize) -> bool { matches!(self.fences.role[at], Role::LineLabel | Role::Contents) }
    /// A running head or foot, or a folio.
    pub(crate) fn furniture(&self, at: usize) -> bool { matches!(self.fences.role[at], Role::Furniture | Role::PageLabel) }
    /// A table's cell: body, its number no note's label.
    pub(crate) fn cell(&self, at: usize) -> bool { self.fences.cell[at] != 0 }
    /// A numbered paragraph's number, unless its page shows it a note's (`paragraph_labels`): a counted run's member
    /// opening a unit, a paragraph's held number, a margin's number.
    pub(crate) fn paragraph(&self, at: usize) -> bool {
        self.unit[at] || !self.fences.paragraphs.held[at].is_empty() || self.fences.margin_number[at].is_some()
    }
    /// Labelled a note by its source.
    pub(crate) fn noted(&self, at: usize) -> bool { self.evidence.tag[at] == SourceTag::Note }
    /// A heading by its fence or its source.
    pub(crate) fn headed(&self, at: usize) -> bool {
        self.fences.role[at] == Role::Heading || matches!(self.evidence.tag[at], SourceTag::Heading(_) | SourceTag::Title | SourceTag::ParagraphTitle)
    }
}

/// A page with its lines' measures, each by its line's place on the page: the document's lines on it
/// (`members`, in page order, and `member` by place), their labels, and the document it is read in.
struct Sheet<'a> {
    page: &'a Page,
    /// The page's slot among the document's pages.
    at: usize,
    /// The slot of its first line.
    first: usize,
    view: &'a PageView<'a>,
    members: Vec<usize>,
    member: Vec<bool>,
    labels: &'a [Option<NoteLabel>],
    document: Document<'a>,
}

impl<'a> Sheet<'a> {
    fn of(view: &'a PageView<'a>, document: Document<'a>) -> Self {
        let count = view.page.lines.len();
        let first = view.slot(0).index();
        let members: Vec<usize> = view.lines.iter().map(|&slot| view.place(slot)).collect();
        let mut member = vec![false; count];
        for &place in &members {
            member[place] = true;
        }
        Self {
            page: view.page,
            at: document.slots.page(view.slot(0)),
            first,
            view,
            members,
            member,
            labels: &document.evidence.label[first..first + count],
            document,
        }
    }

    fn line(&self, place: usize) -> &'a Line { &self.page.lines[place] }
    /// A running head or foot, a folio.
    fn furniture(&self, place: usize) -> bool { self.document.furniture(self.first + place) }
    /// A line the band never reads.
    fn excluded(&self, place: usize) -> bool { self.document.excluded(self.first + place) }
    fn cell(&self, place: usize) -> bool { self.document.cell(self.first + place) }
    fn paragraph(&self, place: usize) -> bool { self.document.paragraph(self.first + place) }
    fn noted(&self, place: usize) -> bool { self.document.noted(self.first + place) }
    fn headed(&self, place: usize) -> bool { self.document.headed(self.first + place) }
    /// The note separator rule the extraction found on the page.
    fn separator(&self) -> Option<f64> { self.document.fences.separators[self.at] }
    /// The page's columns, read on its words outside its furniture (`Fences::columns`).
    fn columns(&self) -> ColumnModel { self.document.fences.columns[self.at] }
    /// The page's words' text items.
    fn items(&self) -> &'a crate::tables::PageItems { &self.document.evidence.items[self.at] }
    /// Each of the page's lines' row (`Rows`).
    fn rows(&self) -> &'a [u32] { &self.document.evidence.pages[self.at].rows.of }
    /// Whether the page is a contents page.
    fn contents(&self) -> bool { self.document.fences.contents_pages[self.at] }

    /// The document's lines on the page, with their places.
    fn lines(&self) -> impl Iterator<Item = (usize, &'a Line)> + '_ {
        self.members.iter().map(|&place| (place, &self.page.lines[place]))
    }

    /// The document's lines on the page, each with its reference marks (`Evidence::marks`): what a label's
    /// reference is looked for in (`has_prior_reference`).
    fn marked(&self) -> impl Iterator<Item = (&'a Line, &'a [(u32, u32, u32)])> + '_ {
        let evidence: &'a Evidence = self.document.evidence;
        self.with(&evidence.marks)
    }

    /// The document's lines on the page, each with its marks set apart from its text (`Evidence::calls`): what decides
    /// that a whole count or run is notes (an endnote run kept, a page's count released) looks for a label's reference
    /// in, a scan's figures run on from its words being as often its garble as a call.
    fn called(&self) -> impl Iterator<Item = (&'a Line, &'a [(u32, u32, u32)])> + '_ {
        let evidence: &'a Evidence = self.document.evidence;
        self.with(&evidence.calls)
    }

    /// The page's member lines, each with its entry of `marks` (one per document line).
    fn with(&self, marks: &'a [Vec<(u32, u32, u32)>]) -> impl Iterator<Item = (&'a Line, &'a [(u32, u32, u32)])> + '_ {
        self.members.iter().map(move |&place| (&self.page.lines[place], marks[self.first + place].as_slice()))
    }

    /// The type sizes of the page's lines `keep` holds (by place and line), those set from 4 to 24 points.
    fn sizes_where(&self, keep: impl Fn(usize, &Line) -> bool) -> Vec<f64> {
        self.lines().filter(|&(place, line)| keep(place, line)).map(|(place, _)| self.page.lines[place].size).filter(|size| (4.0..=24.0).contains(size)).collect()
    }
}

/// The endnote sequence a run of pages carries: the label the next page's notes continue from and
/// the type size they are set in.
#[derive(Default)]
struct EndnoteRun {
    expected: Option<u32>,
    size: Option<f64>,
    previous_page: Option<usize>,
}

impl EndnoteRun {
    /// A page that does not follow the last one continues no sequence.
    fn enter(&mut self, page: usize) {
        if self.previous_page.is_some_and(|prior| prior.checked_add(1) != Some(page)) {
            (self.expected, self.size) = (None, None);
        }
        self.previous_page = Some(page);
    }

    /// After a page: its last endnote's label and its endnotes' size carry on, or the run ends.
    /// `endnotes` are the places of the page's endnote lines, in page order.
    fn leave(&mut self, sheet: &Sheet, endnotes: &[usize]) {
        if endnotes.is_empty() {
            (self.expected, self.size) = (None, None);
            return;
        }
        if let Some(last) = endnotes.iter().rev().filter_map(|&place| label_prefix(&sheet.line(place).text)?.label.parse::<u32>().ok()).next() {
            self.expected = Some(last + 1);
        }
        let size = median(endnotes.iter().map(|&place| sheet.page.lines[place].size).filter(|size| *size > 0.0).collect());
        if size > 0.0 {
            self.size = Some(size);
        }
    }
}

/// The type and geometry a page's notes are read against.
struct PageReading {
    body_size: f64,
    /// The type of the page's wide lines of words: its running text, against which a margin's
    /// notes are small.
    note_size: f64,
    tolerance: f64,
    /// Table cells, by place.
    table_cells: Vec<bool>,
    /// The page's note separator rule, unless it runs through a table.
    separator: Option<f64>,
}

fn page_reading(sheet: &Sheet, separator: Option<f64>, document_body: f64) -> PageReading {
    let page = sheet.page;
    let table_cells: Vec<bool> = (0..page.lines.len()).map(|place| sheet.member[place] && sheet.cell(place)).collect();
    let body_size = if document_body > 0.0 && sheet.members.iter().any(|&place| sheet.page.lines[place].size > 0.0) {
        document_body
    } else {
        median(sheet.sizes_where(|place, line| {
            !sheet.furniture(place) && !sheet.excluded(place) && !table_cells[place] && page.height * 0.10 <= line.bbox[1] && line.bbox[1] <= page.height * 0.75
        }))
    };
    let body_size = if body_size > 0.0 { body_size } else { 10.0 };
    let wide_sizes = sheet.sizes_where(|place, line| {
        !sheet.excluded(place) && !table_cells[place] && line_width(line) >= page.width * 0.30 && line.text.chars().filter(|character| character.is_alphabetic()).take(4).count() == 4
    });
    let note_size = if wide_sizes.len() >= 3 { median(wide_sizes) } else { body_size };
    let tolerance = separator_tolerance(page);
    let separator = separator_beside_tables(page, separator, (0..page.lines.len()).filter(|&place| table_cells[place]).map(|place| &page.lines[place]));
    PageReading { body_size, note_size, tolerance, table_cells, separator }
}

/// How far from a page's note separator a line still stands at it.
pub(crate) fn separator_tolerance(page: &Page) -> f64 { (page.height * 0.004).max(1.0) }

/// The page's note separator rule, unless it runs through the band its table cells (`cells`) fill.
pub(crate) fn separator_beside_tables<'a>(page: &Page, separator: Option<f64>, cells: impl Iterator<Item = &'a Line>) -> Option<f64> {
    let tolerance = separator_tolerance(page);
    let table_band = cells.fold(None, |band, line| {
        Some(band.map_or((line.bbox[1], line.bbox[3]), |(top, bottom): (f64, f64)| {
            (top.min(line.bbox[1]), bottom.max(line.bbox[3]))
        }))
    });
    separator.filter(|cut| table_band.is_none_or(|(top, bottom)| *cut < top - tolerance || *cut > bottom + tolerance))
}

/// The label a line's number is read as: its note label, else its whole text.
fn printed_label(line: &Line) -> String {
    label_prefix(&line.text).map_or_else(|| line.text.trim().to_owned(), |prefix| prefix.label)
}

/// The lines whose number is a numbered paragraph's, never a note's, by place. Alignment,
/// typography and a paragraph sequence are positive evidence, not a veto on the note witnesses:
/// rules, references, source regions and note continuation win.
fn paragraph_labels(sheet: &Sheet, reading: &PageReading, expected_endnote: Option<u32>) -> Vec<bool> {
    let mut found = vec![false; sheet.page.lines.len()];
    // The page's counted numbers: each one's place, its value, and whether the page shows it a note's.
    let mut counted: Vec<(usize, u32, bool)> = Vec::new();
    for (place, line) in sheet.lines().filter(|&(place, _)| sheet.paragraph(place)) {
        let label = printed_label(line);
        let shown = reading.separator.is_some_and(|cut| line.bbox[1] >= cut - reading.tolerance)
            || has_prior_reference(sheet.called(), &label, line.bbox[1])
            || sheet.noted(place);
        found[place] = !shown
            && !sheet.lines().any(|(_, heading)| heading.bbox[1] < line.bbox[1] && endnote_heading(&heading.text))
            && !expected_endnote.is_some_and(|expected| {
                label.parse::<u32>().is_ok_and(|number| number >= expected)
                    && !sheet.lines().any(|(_, heading)| heading.bbox[1] < line.bbox[1] && structural_reset_heading(&heading.text))
            });
        if let Ok(value) = normalize_label(&label).parse::<u32>() {
            counted.push((place, value, shown));
        }
    }
    // One count on a page is one thing. Where three or more of the page's numbers count on by one and the page shows
    // two of them, and a third of them or more, the notes' (notes 68 and 72 to 74 referred to above, 69 to 71 and 75
    // between and after them with their references misread), the whole count is the notes'. A page shows a number by a
    // mark set apart from its text (`Sheet::called`): a scan's figures run on from its words ("n9", "n08" for "n°" in a
    // count of sentences) show none.
    counted.sort_by_key(|&(_, value, _)| value);
    for run in counted.chunk_by(|a, b| b.1 == a.1 + 1) {
        let shown = run.iter().filter(|(_, _, shown)| *shown).count();
        if run.len() >= 3 && shown >= 2 && shown * 3 >= run.len() {
            // Only a label the band would read as a note's is released (`rejected`): one it reads as no note's stays the
            // paragraph's it was.
            let takes = |place: usize| sheet.labels[place].is_some_and(|label| !rejected(sheet, reading, place, &label, typographic(sheet, reading, place, &label)));
            run.iter().filter(|&&(place, ..)| takes(place)).for_each(|&(place, ..)| found[place] = false);
        }
    }
    found
}

/// A line that opens with a note label: its place, the label, and whether its type sets it as one.
type Candidate = (usize, LabelPrefix, bool);

/// The lines that open with a note label, and the lines whose label is no note's: a table's cells and the labels the
/// band rejects (`rejected`).
fn label_candidates(sheet: &Sheet, reading: &PageReading, paragraph: &[bool]) -> (Vec<Candidate>, Vec<usize>) {
    let (mut candidates, mut suppress) = (Vec::new(), Vec::new());
    for (place, line) in sheet.lines() {
        if paragraph[place] {
            continue;
        }
        if reading.table_cells[place] {
            suppress.push(place);
            continue;
        }
        if sheet.furniture(place) || sheet.excluded(place) {
            continue;
        }
        let (Some(label), Some(prefix)) = (sheet.labels[place], label_prefix(&line.text)) else { continue };
        let typographic = typographic(sheet, reading, place, &label);
        if rejected(sheet, reading, place, &label, typographic) {
            suppress.push(place);
            continue;
        }
        candidates.push((place, prefix, typographic));
    }
    (candidates, suppress)
}

/// Whether a line's label is set as a note's: its own type, or a lone label beside a note's body in a note's type.
fn typographic(sheet: &Sheet, reading: &PageReading, place: usize, label: &NoteLabel) -> bool {
    let body_size = reading.body_size;
    label.typographic(body_size)
        || (standalone_note_label(sheet.line(place))
            && aligned_note_body_among(&sheet.page.lines, sheet.rows(), sheet.members.iter().copied(), place).is_some_and(|body| {
                let size = sheet.page.lines[body].size;
                size > 0.0 && size <= body_size * 0.90
            }))
}

/// Whether a line's label is no note's: a page number in the corner, a number before a comma, a reference's number
/// wrapped onto the next line, a foot's number not set as a note's under the rule, or a label set larger than the body
/// (unless raised and not stopped: a scan's line is as tall as its raised label makes it, and a raised "1." in large
/// type is a plan's or a by-law's numbered paragraph).
fn rejected(sheet: &Sheet, reading: &PageReading, place: usize, label: &NoteLabel, typographic: bool) -> bool {
    let (line, body_size) = (sheet.line(place), reading.body_size);
    let size = if line.size > 0.0 { line.size } else { body_size };
    let bottom_right = reading.separator.is_none() && label.suppress & SUPPRESS_CORNER != 0;
    // A number followed by a comma is the line's text ("1991, c. 43, s. 4" in a statute's small-type history note),
    // unless it is raised above its line.
    let comma_tail = label.suppress & SUPPRESS_COMMA != 0;
    // A number opening a line under one that ends on a reference's noun ("... c. 27 (1st Supp.), s" / "13.") is that
    // reference's number, wrapped.
    let wrapped_reference = label.suppress & SUPPRESS_WRAPPED != 0;
    let below_separator = reading.separator.is_some_and(|cut| line.bbox[1] >= cut - reading.tolerance);
    let stopped = || label_prefix(&line.text).is_some_and(|prefix| line.text.chars().nth(prefix.end) == Some('.'));
    (label.suppress & SUPPRESS_FOOT != 0 && !(below_separator && typographic)) || bottom_right || comma_tail || wrapped_reference
        || size > body_size * 1.15 && (!label.raised || stopped())
}

/// What a page's label candidates and its other lines say of it before any band is drawn.
struct PageFacts {
    /// The longest run of counting labels.
    best_run: usize,
    columns: ColumnModel,
    endnote_heading_y: Option<f64>,
    /// The first line of a page under a notes heading, in source order.
    endnote_heading_source: Option<usize>,
    /// A heading above every label that starts a new part ("APPENDIX A", "BIBLIOGRAPHY", or a
    /// heading in capitals on a page with no labels): no endnote sequence runs on across it.
    structural_reset: bool,
    label_size: Option<f64>,
    /// Labels in the page's upper half set as labels or small.
    early_labels: bool,
    /// A run of three or more small labels in the page's lower half.
    lower_note_run: bool,
}

/// The columns of the page's lines `keep` holds (places), its furniture aside, read on their words'
/// text items: the page's own columns where they are all its lines.
fn columns_among(sheet: &Sheet, keep: &[bool]) -> ColumnModel {
    let page = sheet.page;
    let kept = |place: usize| keep[place] && !sheet.furniture(place);
    let places = (0..page.lines.len()).filter(|&place| kept(place));
    columns_of(&page.lines, places, &kept_items(sheet.items(), kept), page.index as u32 + 1)
}

fn page_facts(sheet: &Sheet, reading: &PageReading, candidates: &[Candidate]) -> PageFacts {
    let page = sheet.page;
    let body_size = reading.body_size;
    let numeric: Vec<u32> = candidates.iter().filter_map(|(_, prefix, _)| prefix.label.parse().ok()).collect();
    let columns = {
        let page_model = if sheet.members.len() == page.lines.len() { sheet.columns() } else { columns_among(sheet, &sheet.member) };
        if page_model.kind == ColumnKind::Two {
            page_model
        } else {
            let mut labelled = vec![false; page.lines.len()];
            for (place, _, _) in candidates {
                labelled[*place] = true;
            }
            let label_model = columns_among(sheet, &labelled);
            if label_model.kind == ColumnKind::Single { page_model } else { label_model }
        }
    };
    let first_candidate_y = candidates.first().map(|(place, _, _)| sheet.line(*place).bbox[1]);
    let minimum_label_y = candidates
        .iter()
        .map(|(place, _, _)| sheet.line(*place).bbox[1])
        .min_by(f64::total_cmp)
        .unwrap_or(page.height);
    let endnote_heading_y = sheet
        .lines()
        .filter(|(_, line)| endnote_heading(&line.text) && first_candidate_y.is_none_or(|y| line.bbox[1] < y))
        .map(|(_, line)| line.bbox[1])
        .min_by(f64::total_cmp);
    let endnote_heading_source = endnote_heading_y.is_some().then(|| {
        sheet.lines().filter(|(_, line)| endnote_heading(&line.text)).map(|(_, line)| line.source_index).min().unwrap_or(0)
    });
    let content_before = || sheet.lines().filter_map(|(place, line)| (!sheet.furniture(place) && line.bbox[1] < minimum_label_y).then_some(place));
    let first_content = content_before().min_by(|left, right| sheet.line(*left).bbox[1].total_cmp(&sheet.line(*right).bbox[1]));
    let generic_heading_reset = candidates.is_empty()
        && first_content.is_some_and(|place| {
            let line = sheet.line(place);
            let text = line.text.trim();
            let letter_count = text.chars().filter(|character| character.is_alphabetic()).count();
            line.bbox[1] >= page.height * 0.08
                && letter_count >= 4
                && text.chars().count() <= 100
                && all_caps(text)
                && !continuing_note_heading(text)
        });
    let structural_reset = generic_heading_reset
        || content_before().any(|place| {
            let line = sheet.line(place);
            (sheet.headed(place) || line.bbox[1] >= page.height * 0.08) && structural_reset_heading(&line.text)
        });
    let label_sizes: Vec<f64> = candidates.iter().map(|(place, _, _)| sheet.page.lines[*place].size).filter(|size| *size > 0.0).collect();
    let label_size = (!label_sizes.is_empty()).then(|| median(label_sizes));
    let early_labels = candidates.iter().any(|(place, _, typographic)| {
        sheet.line(*place).bbox[1] < page.height * 0.48 && (*typographic || sheet.page.lines[*place].size <= body_size * 0.90)
    });
    let lower_numeric: Vec<u32> = candidates
        .iter()
        .filter(|(place, _, _)| {
            sheet.line(*place).bbox[1] >= page.height * 0.48 && sheet.page.lines[*place].size > 0.0 && sheet.page.lines[*place].size <= body_size * 0.90
        })
        .filter_map(|(_, prefix, _)| prefix.label.parse().ok())
        .collect();
    PageFacts {
        best_run: longest_label_run(&numeric),
        columns,
        endnote_heading_y,
        endnote_heading_source,
        structural_reset,
        label_size,
        early_labels,
        lower_note_run: longest_label_run(&lower_numeric) >= 3,
    }
}

/// Where an endnote sequence carried from the page before resumes on this one.
struct Continuation {
    /// The first candidate that is a note: the one that continues the sequence, else the first.
    selected_start: usize,
    /// A page of endnotes with no label of its own (a note running on from the page before).
    label_free: bool,
    /// The sequence runs on through this page.
    continuing: bool,
}

/// The candidate that continues the carried sequence: the one numbered next, that reads as no
/// citation's number, and that the longest counting run follows. The labels before it, and other
/// citation-shaped ones numbered next, are suppressed (returned with it, by place).
fn continue_endnotes(sheet: &Sheet, candidates: &[Candidate], facts: &PageFacts, run: &EndnoteRun) -> (Continuation, Vec<usize>) {
    let page = sheet.page;
    let mut suppressed = Vec::new();
    let expected_indexes: Vec<usize> = candidates
        .iter()
        .enumerate()
        .filter_map(|(index, (_, prefix, _))| (prefix.label.parse::<u32>().ok() == run.expected).then_some(index))
        .collect();
    let noncitation: Vec<usize> = expected_indexes.iter().copied().filter(|index| !citation_shaped_candidate(page, &candidates[*index])).collect();
    let pool = if noncitation.is_empty() { &expected_indexes } else { &noncitation };
    let expected_index = pool.iter().copied().max_by_key(|index| {
        let mut sequence = 1;
        let mut prior = run.expected;
        for (_, prefix, _) in &candidates[index + 1..] {
            let Ok(current) = prefix.label.parse::<u32>() else {
                break;
            };
            if prior.is_none_or(|value| current != value + 1) {
                break;
            }
            sequence += 1;
            prior = Some(current);
        }
        (usize::from(candidates[*index].2), sequence, *index)
    });
    if run.expected.is_some() {
        if let Some(selected) = expected_index {
            suppressed.extend(candidates[..selected].iter().map(|candidate| candidate.0));
            for index in &expected_indexes {
                if *index != selected && citation_shaped_candidate(page, &candidates[*index]) {
                    suppressed.push(candidates[*index].0);
                }
            }
        }
    }
    let content_sizes: Vec<f64> = sheet.lines().filter(|&(place, _)| !sheet.furniture(place) && !sheet.excluded(place)).map(|(place, _)| sheet.page.lines[place].size).filter(|size| *size > 0.0).collect();
    let content_size = (!content_sizes.is_empty()).then(|| median(content_sizes));
    let size_matches = run.size.is_none_or(|size| content_size.is_none_or(|content| content <= size * 1.15));
    let label_free = run.expected.is_some() && candidates.is_empty() && !facts.structural_reset && size_matches;
    let continuation = Continuation {
        selected_start: if run.expected.is_some() { expected_index.unwrap_or(0) } else { 0 },
        label_free,
        continuing: run.expected.is_some() && !facts.structural_reset && size_matches && (label_free || expected_index.is_some()),
    };
    (continuation, suppressed)
}

/// Notes set in a margin column: small, narrow labels counting on (or each referred to above), and
/// the side of the page they stand on.
fn margin_notes(sheet: &Sheet, reading: &PageReading, candidates: &[Candidate]) -> Option<(ColumnModel, usize)> {
    let page = sheet.page;
    let note_size = reading.note_size;
    let margin_candidates: Vec<_> = candidates
        .iter()
        .filter_map(|(place, _, _)| (sheet.page.lines[*place].size <= note_size * 0.90 && line_width(sheet.line(*place)) <= page.width * 0.30).then_some(*place))
        .collect();
    let margin_numeric: Vec<u32> = candidates
        .iter()
        .filter(|(place, _, _)| margin_candidates.contains(place))
        .filter_map(|(_, prefix, _)| prefix.label.parse().ok())
        .collect();
    let supported: Vec<_> = candidates
        .iter()
        .filter(|(place, prefix, _)| {
            margin_candidates.contains(place) && has_prior_reference(sheet.marked(), &prefix.label, sheet.line(*place).bbox[1])
        })
        .map(|(place, _, _)| *place)
        .collect();
    let margin_labels = if longest_label_run(&margin_numeric) >= 3 { &margin_candidates } else { &supported };
    let set_aside = |place| sheet.furniture(place) || sheet.excluded(place);
    let model = margin_note_model(&page.lines, sheet.rows(), &sheet.members, set_aside, margin_labels, page.width, note_size, if margin_labels.len() >= 3 { 3 } else { 2 })?;
    let side = usize::from(margin_labels.iter().filter(|place| line_center_x(sheet.line(**place)) >= model.split_x).count() * 2 >= margin_labels.len());
    Some((model, side))
}

/// The candidates that are notes: all of them on an endnote page or in the margin column, else those
/// in the page's lower half under its separator, set as labels in a two-column page, or (with no
/// separator) set as labels, backed by a reference above, or in a run of small labels; and, with no
/// separator, a small label numbered next after the document's notes so far (`next_label`) whose
/// reference stands above it, anywhere on the page.
fn labels(sheet: &Sheet, reading: &PageReading, candidates: &[Candidate], facts: &PageFacts, endnote_page: bool, margin: Option<(ColumnModel, usize)>, mut next_label: u32) -> Vec<usize> {
    let page = sheet.page;
    let (body_size, separator) = (reading.body_size, reading.separator);
    candidates
        .iter()
        .filter_map(|(place, prefix, typographic)| {
            let line = sheet.line(*place);
            let size = sheet.page.lines[*place].size;
            let lower = line.bbox[1] >= page.height * 0.48;
            let small = size > 0.0 && size <= body_size * 0.90;
            let referenced = || has_prior_reference(sheet.marked(), &prefix.label, line.bbox[1]);
            let reference_backed = lower && size > 0.0 && (size <= body_size * 0.90 || prefix.label.chars().all(is_note_symbol)) && referenced();
            let in_margin = margin.is_some_and(|(model, side)| {
                usize::from(line_center_x(line) >= model.split_x) == side && size <= reading.note_size * 0.90 && line_width(line) <= page.width * 0.30
            });
            let value = prefix.label.parse::<u32>().ok();
            let counts_on = separator.is_none() && small && value == Some(next_label) && referenced();
            let note = endnote_page
                || in_margin
                || counts_on
                || (lower
                    && (separator.is_some_and(|cut| line.bbox[1] >= cut - reading.tolerance)
                        || (facts.columns.kind == ColumnKind::Two && *typographic)
                        || (separator.is_none()
                            && (*typographic
                                || reference_backed
                                || (facts.lower_note_run && !recognized_page(page) && small)))));
            if note {
                if let Some(value) = value.filter(|value| *value >= next_label) {
                    next_label = value + 1;
                }
            }
            note.then_some(*place)
        })
        .collect()
}

/// On an endnote page, a short line in capitals at the very top, above every label and set in from
/// the margins, is the page's running head.
fn running_heads(sheet: &Sheet, labels: &[usize]) -> Vec<usize> {
    let page = sheet.page;
    let first_label = labels.iter().map(|place| sheet.line(*place).bbox[1]).min_by(f64::total_cmp).unwrap_or(page.height);
    let (width, height) = (page.width, page.height);
    sheet
        .lines()
        .filter(|&(place, line)| {
            let text = line.text.trim();
            let letter_count = text.chars().filter(|character| character.is_alphabetic()).count();
            !sheet.furniture(place)
                && line.bbox[1] < height * 0.08
                && line.bbox[1] < first_label
                && line.bbox[0] >= width * 0.15
                && line.bbox[2] <= width * 0.85
                && letter_count >= 4
                && text.chars().count() <= 100
                && all_caps(text)
                && line_start_label_prefix(text).is_none()
        })
        .map(|(place, _)| place)
        .collect()
}

/// Where the band begins across the page: at the top of the page's running text on a page of
/// endnotes continued, at the separator when the first label sits close under it (or the small
/// type starts under it), else at the first label, warned of where that rests on weak geometry.
/// `furniture` holds the page's running heads and feet.
#[allow(clippy::too_many_arguments)]
fn note_cut(sheet: &Sheet, reading: &PageReading, labels: &[usize], contents_page: bool, endnote_page: bool, continuation: &Continuation, run: &EndnoteRun, furniture: &dyn Fn(usize) -> bool, diagnostics: &mut Vec<Diagnostic>) -> Option<f64> {
    let page = sheet.page;
    let (body_size, separator, tolerance) = (reading.body_size, reading.separator, reading.tolerance);
    let eligible_top = || {
        sheet
            .lines()
            .filter(|&(place, _)| !furniture(place) && !sheet.excluded(place))
            .map(|(_, line)| line.bbox[1])
            .min_by(f64::total_cmp)
    };
    if continuation.label_free {
        return eligible_top();
    }
    if labels.is_empty() {
        return None;
    }
    let selected_sizes: Vec<f64> = labels.iter().map(|place| sheet.page.lines[*place].size).filter(|size| *size > 0.0).collect();
    let selected_size = (!selected_sizes.is_empty()).then(|| median(selected_sizes));
    let separator_starts_small_text = separator.is_some_and(|cut| {
        !contents_page
            && sheet
                .lines()
                .filter(|&(place, line)| {
                    !reading.table_cells[place] && !sheet.excluded(place) && !furniture(place) && line.bbox[1] >= cut - tolerance
                })
                .min_by(|(_, left), (_, right)| left.bbox[1].total_cmp(&right.bbox[1]))
                .is_some_and(|(place, line)| line.bbox[1] - cut <= page.height * 0.06 && sheet.page.lines[place].size > 0.0 && sheet.page.lines[place].size <= body_size * 0.90)
    });
    let first_label = labels.iter().map(|place| sheet.line(*place).bbox[1]).min_by(f64::total_cmp).unwrap_or(page.height);
    if endnote_page && run.expected.is_some() {
        return eligible_top();
    }
    if separator.is_some_and(|cut| {
        0.0 <= first_label - cut
            && (first_label - cut <= page.height * 0.15 || (separator_starts_small_text && selected_size.is_some_and(|size| size <= body_size * 0.90)))
    }) {
        return separator;
    }
    let confident = (first_label >= page.height * 0.58 || endnote_page) && selected_size.is_some_and(|size| size <= body_size * 0.90);
    if !confident {
        diagnostics.push(Diagnostic::new("FOOTNOTE_REGION_UNCERTAIN", Some(page.index)));
    }
    Some(first_label)
}

/// Where the band begins in each column of a two-column page or beside a margin column: at each
/// column's first label (or the note body it aligns with), and under a notes heading at the
/// column's first line. `furniture` holds the page's running heads and feet.
fn column_cuts(sheet: &Sheet, labels: &[usize], columns: ColumnModel, has_margin: bool, endnote_page: bool, facts: &PageFacts, furniture: &dyn Fn(usize) -> bool) -> Option<[Option<f64>; 2]> {
    if !((columns.kind == ColumnKind::Two || has_margin) && (!endnote_page || facts.endnote_heading_y.is_some())) {
        return None;
    }
    let lines = &sheet.page.lines;
    let mut cuts = [None::<f64>; 2];
    for &place in labels {
        let label = sheet.line(place);
        let column = usize::from(line_center_x(label) >= columns.split_x);
        let label_top = aligned_note_body_among(lines, sheet.rows(), sheet.members.iter().copied(), place)
            .map(|body| band_geometry_top(&lines[body]))
            .unwrap_or_else(|| band_geometry_top(label))
            .min(band_geometry_top(label));
        cuts[column] = Some(cuts[column].map_or(label_top, |cut| cut.min(label_top)));
    }
    if !(cuts.iter().all(Option::is_some) || (columns.kind == ColumnKind::Margin && cuts.iter().any(Option::is_some))) {
        return None;
    }
    if let Some(heading_y) = facts.endnote_heading_y {
        for (column, cut) in cuts.iter_mut().enumerate() {
            if cut.is_some_and(|value| value < heading_y) {
                *cut = sheet
                    .lines()
                    .filter(|&(place, line)| {
                        !furniture(place) && !sheet.excluded(place) && usize::from(line_center_x(line) >= columns.split_x) == column
                    })
                    .map(|(_, line)| line.bbox[1])
                    .min_by(f64::total_cmp);
            }
        }
    }
    Some(cuts)
}

/// Where a page's band runs.
struct Band {
    endnote_page: bool,
    cut: Option<f64>,
    column_cuts: Option<[Option<f64>; 2]>,
    columns: ColumnModel,
    endnote_heading_source: Option<usize>,
}

impl Band {
    fn holds(&self, line: &Line) -> bool {
        self.endnote_heading_source.is_some_and(|start| self.column_cuts.is_none() && self.endnote_page && line.source_index >= start)
            || self.column_cuts.as_ref().map_or_else(
                || self.cut.is_some_and(|cut| line.bbox[1] >= cut),
                |cuts| {
                    let column = usize::from(line_center_x(line) >= self.columns.split_x);
                    cuts[column].is_some_and(|cut| line.bbox[1] >= cut)
                },
            )
    }
}

/// Endnotes are notes: the body before them refers to them. Each run of consecutive pages read as
/// endnotes keeps them only where a notes heading opens it, or where references earlier in the
/// document (raised numbers, detached references) name at least half of its note numbers; else
/// its lines are the body's, a numbered sequence the page-by-page reading took for notes. On the
/// structure gold (batches 5-7) endnote runs held 126 note lines and 5,472 lines of body; the
/// references keep 49 of the notes and give back 5,429 of the body. `notes` holds each page's note
/// lines (places, with their kind).
fn verify_endnotes(sheets: &[Sheet], notes: &mut [Vec<(usize, NoteKind)>]) {
    let endnote = |slot: usize, notes: &[Vec<(usize, NoteKind)>]| notes[slot].iter().any(|(_, kind)| *kind == NoteKind::Endnote);
    let mut slot = 0;
    while slot < sheets.len() {
        if !endnote(slot, notes) {
            slot += 1;
            continue;
        }
        let mut end = slot + 1;
        while end < sheets.len() && endnote(end, notes) && sheets[end].page.index == sheets[end - 1].page.index + 1 {
            end += 1;
        }
        let headed = sheets[slot..end].iter().flat_map(|sheet| sheet.lines()).any(|(_, line)| endnote_heading(&line.text));
        let endnotes = |at: usize| notes[at].iter().filter(|(_, kind)| *kind == NoteKind::Endnote).map(move |(place, _)| sheets[at].line(*place));
        let labels: BTreeSet<String> = (slot..end).flat_map(endnotes).filter_map(|line| label_prefix(&line.text).map(|prefix| prefix.label)).collect();
        let first_note = endnotes(slot).map(|line| line.bbox[1]).min_by(f64::total_cmp).unwrap_or(0.0);
        let referenced = |label: &&String| {
            sheets[..slot].iter().any(|sheet| has_prior_reference(sheet.called(), label, f64::INFINITY))
                || has_prior_reference(sheets[slot].called(), label, first_note)
        };
        let referenced = labels.iter().filter(referenced).count();
        if !headed && referenced * 2 < labels.len().max(1) {
            for page in &mut notes[slot..end] {
                page.retain(|(_, kind)| *kind != NoteKind::Endnote);
            }
        }
        slot = end;
    }
}

/// Every page's note band, in order, then the endnote runs the body never refers to given back.
pub(crate) fn read_band(views: &[PageView], document: Document) -> BandReading {
    let mut found = BandReading::default();
    let sheets: Vec<Sheet> = views.iter().map(|view| Sheet::of(view, document)).collect();
    let mut notes: Vec<Vec<(usize, NoteKind)>> = Vec::with_capacity(sheets.len());
    let mut run = EndnoteRun::default();
    // The label the document's next note takes, by the notes read so far.
    let mut next_label = 1;
    for sheet in &sheets {
        let page = sheet.page;
        found.pages.push(BandPageReading { next_label, ..BandPageReading::default() });
        run.enter(page.index);
        let reading = page_reading(sheet, sheet.separator(), sheet.document.body_size);
        let paragraph = paragraph_labels(sheet, &reading, run.expected);
        found.paragraph_labels.extend(sheet.members.iter().filter(|&&place| paragraph[place]).map(|&place| sheet.view.slot(place)));
        let (candidates, suppress) = label_candidates(sheet, &reading, &paragraph);
        found.suppressed.extend(suppress.into_iter().map(|place| sheet.view.slot(place)));
        let facts = page_facts(sheet, &reading, &candidates);
        let (continuation, suppress) = continue_endnotes(sheet, &candidates, &facts, &run);
        found.suppressed.extend(suppress.into_iter().map(|place| sheet.view.slot(place)));
        let margin = margin_notes(sheet, &reading, &candidates);
        let endnote_page = facts.endnote_heading_y.is_some()
            || continuation.continuing
            || (reading.separator.is_none() && margin.is_none() && !candidates.is_empty() && facts.early_labels && facts.best_run >= 3 && facts.label_size.is_some_and(|size| size <= reading.body_size * 0.90));
        let labels = labels(sheet, &reading, &candidates[continuation.selected_start..], &facts, endnote_page, margin, next_label);
        let heads = if endnote_page && !labels.is_empty() { running_heads(sheet, &labels) } else { Vec::new() };
        found.running_heads.extend(heads.iter().map(|&place| sheet.view.slot(place)));
        let furniture = |place: usize| sheet.furniture(place) || heads.contains(&place);
        let cut = note_cut(sheet, &reading, &labels, sheet.contents(), endnote_page, &continuation, &run, &furniture, &mut found.diagnostics);
        found.pages.last_mut().expect("the page's reading").cut = cut;
        let columns = margin.map_or(facts.columns, |(model, _)| model);
        let column_cuts = column_cuts(sheet, &labels, columns, margin.is_some(), endnote_page, &facts, &furniture);
        let band = Band { endnote_page, cut, column_cuts, columns, endnote_heading_source: facts.endnote_heading_source };
        // Each line's reading: a notes heading is a heading, a table's cell body text, furniture
        // stays, a source's heading survives, a numbered paragraph's line is body, and the rest are
        // note or body by the band.
        let kind = if endnote_page { NoteKind::Endnote } else { NoteKind::Footnote };
        let mut page_notes = Vec::new();
        for (place, line) in sheet.lines().filter(|&(place, _)| !sheet.excluded(place)) {
            if endnote_heading(&line.text) {
                found.headings.push(sheet.view.slot(place));
            } else if reading.table_cells[place] || furniture(place) || sheet.headed(place) {
            } else if !paragraph[place] && band.holds(line) {
                page_notes.push((place, kind));
            }
        }
        let endnotes: Vec<usize> = page_notes.iter().filter(|(_, kind)| *kind == NoteKind::Endnote).map(|(place, _)| *place).collect();
        run.leave(sheet, &endnotes);
        if let Some(last) = page_notes.iter().filter_map(|(place, _)| label_prefix(&sheet.line(*place).text)?.label.parse::<u32>().ok()).max() {
            next_label = next_label.max(last + 1);
        }
        notes.push(page_notes);
    }
    verify_endnotes(&sheets, &mut notes);
    for ((sheet, page_notes), page) in sheets.iter().zip(notes).zip(&mut found.pages) {
        page.noted = !page_notes.is_empty();
        found.notes.extend(page_notes.into_iter().map(|(place, kind)| (sheet.view.slot(place), kind)));
    }
    found
}

/// The endnote runs among a document's notes (`notes`: each page's note lines by place, with their
/// kind, by the pages' order) that the body never refers to, given back (`verify_endnotes`): the
/// notes kept.
pub(crate) fn verified_endnotes(views: &[PageView], document: Document, notes: &mut [Vec<(usize, NoteKind)>]) {
    let sheets: Vec<Sheet> = views.iter().map(|view| Sheet::of(view, document)).collect();
    verify_endnotes(&sheets, notes);
}
