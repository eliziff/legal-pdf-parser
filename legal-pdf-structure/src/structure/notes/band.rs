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
//! The band then gives back the endnote runs the body never refers to (`verify_endnotes`). It reads
//! a document's lines (`BandLines`) on its pages (`BandPage`) and writes nothing (`read_band`).
//! A line a page layout model labelled keeps that label's reading of note or text under
//! LEGALPDF_ABLATE=layout_notes.

use super::*;

/// A line the band never reads: a transcript's line number, a line held out of the body.
pub(crate) const BAND_EXCLUDED: u8 = 1;
/// A running head or foot, or a folio.
pub(crate) const BAND_FURNITURE: u8 = 1 << 1;
/// A table's cell: body, its number no note's label.
pub(crate) const BAND_CELL: u8 = 1 << 2;
/// A numbered paragraph's number, unless its page shows it a note's (`paragraph_labels`).
pub(crate) const BAND_PARAGRAPH: u8 = 1 << 3;
/// Labelled a note by its source.
pub(crate) const BAND_NOTED: u8 = 1 << 4;
/// Labelled a heading by its source.
pub(crate) const BAND_HEADED: u8 = 1 << 5;
/// Labelled by a page layout model.
pub(crate) const BAND_LAYOUT: u8 = 1 << 6;

/// A note's kind: at its page's foot, or among the notes collected at a part's or the document's end.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum NoteKind {
    Footnote,
    Endnote,
}

/// What the band reads of a document's lines beside their text and boxes, by slot: each line's type
/// size (`line_font_size`), the note label it opens with, and its `BAND_*` flags; and the document's
/// body type size, against which notes are set small (0 where unknown: each page then reads its own).
pub(crate) struct BandLines<'a> {
    pub(crate) sizes: &'a [f64],
    pub(crate) labels: &'a [Option<NoteLabel>],
    pub(crate) flags: &'a [u8],
    pub(crate) body_size: f64,
}

/// A page as the band reads it: the document's lines on it (`view`), the note separator rule the
/// extraction found on it, its columns read on its words outside its furniture (`Fences::columns`)
/// and its words' text items, and whether it is a contents page.
pub(crate) struct BandPage<'a> {
    pub(crate) view: PageView<'a>,
    pub(crate) separator: Option<f64>,
    pub(crate) columns: ColumnModel,
    pub(crate) items: &'a legal_pdf_tables::PageItems,
    pub(crate) contents: bool,
}

/// What the band reads of a document: each note line and its kind; the lines whose number it read
/// as a numbered paragraph's; the labels it read as no note's; an endnote page's running heads; the
/// notes headings; and the warnings of a band drawn on weak geometry.
#[derive(Debug, Default)]
pub(crate) struct BandReading {
    pub(crate) notes: Vec<(Slot, NoteKind)>,
    pub(crate) paragraph_labels: Vec<Slot>,
    pub(crate) suppressed: Vec<Slot>,
    pub(crate) running_heads: Vec<Slot>,
    pub(crate) headings: Vec<Slot>,
    pub(crate) diagnostics: Vec<Diagnostic>,
}

/// A page with its lines' measures, each by its line's place on the page: the document's lines on it
/// (`members`, in page order, and `member` by place), their sizes, labels and flags.
struct Sheet<'a> {
    page: &'a Page,
    view: &'a PageView<'a>,
    members: Vec<usize>,
    member: Vec<bool>,
    sizes: &'a [f64],
    labels: &'a [Option<NoteLabel>],
    flags: &'a [u8],
}

impl<'a> Sheet<'a> {
    fn of(page: &'a BandPage<'a>, lines: &'a BandLines<'a>) -> Self {
        let view = &page.view;
        let count = view.page.lines.len();
        let first = view.slot(0).index();
        let members: Vec<usize> = view.lines.iter().map(|&slot| view.place(slot)).collect();
        let mut member = vec![false; count];
        for &place in &members {
            member[place] = true;
        }
        Self {
            page: view.page,
            view,
            members,
            member,
            sizes: &lines.sizes[first..first + count],
            labels: &lines.labels[first..first + count],
            flags: &lines.flags[first..first + count],
        }
    }

    fn line(&self, place: usize) -> &'a Line {
        &self.page.lines[place]
    }

    /// The document's lines on the page, with their places.
    fn lines(&self) -> impl Iterator<Item = (usize, &'a Line)> + '_ {
        self.members.iter().map(|&place| (place, &self.page.lines[place]))
    }

    fn flagged(&self, place: usize, flag: u8) -> bool {
        self.flags[place] & flag != 0
    }

    /// A running head or foot, a folio.
    fn furniture(&self, place: usize) -> bool {
        self.flagged(place, BAND_FURNITURE)
    }

    /// A line the band never reads.
    fn excluded(&self, place: usize) -> bool {
        self.flagged(place, BAND_EXCLUDED)
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
            self.expected = None;
            self.size = None;
        }
        self.previous_page = Some(page);
    }

    /// After a page: its last endnote's label and its endnotes' size carry on, or the run ends.
    /// `endnotes` are the places of the page's endnote lines, in page order.
    fn leave(&mut self, sheet: &Sheet, endnotes: &[usize]) {
        if endnotes.is_empty() {
            self.expected = None;
            self.size = None;
            return;
        }
        if let Some(last) = endnotes
            .iter()
            .rev()
            .filter_map(|&place| label_prefix(&sheet.line(place).text)?.label.parse::<u32>().ok())
            .next()
        {
            self.expected = Some(last + 1);
        }
        let size = median(endnotes.iter().map(|&place| sheet.sizes[place]).filter(|size| *size > 0.0).collect());
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
    let table_cells: Vec<bool> = (0..page.lines.len()).map(|place| sheet.member[place] && sheet.flagged(place, BAND_CELL)).collect();
    let body_size = if document_body > 0.0 && sheet.members.iter().any(|&place| sheet.sizes[place] > 0.0) {
        document_body
    } else {
        median(
            sheet
                .lines()
                .filter(|&(place, line)| {
                    !sheet.furniture(place)
                        && !sheet.excluded(place)
                        && !table_cells[place]
                        && page.height * 0.10 <= line.bbox[1]
                        && line.bbox[1] <= page.height * 0.75
                })
                .map(|(place, _)| sheet.sizes[place])
                .filter(|size| (4.0..=24.0).contains(size))
                .collect(),
        )
    };
    let body_size = if body_size > 0.0 { body_size } else { 10.0 };
    let wide_sizes: Vec<_> = sheet
        .lines()
        .filter(|&(place, line)| {
            !sheet.excluded(place)
                && !table_cells[place]
                && line_width(line) >= page.width * 0.30
                && line.text.chars().filter(|character| character.is_alphabetic()).take(4).count() == 4
        })
        .map(|(place, _)| sheet.sizes[place])
        .filter(|size| (4.0..=24.0).contains(size))
        .collect();
    let note_size = if wide_sizes.len() >= 3 { median(wide_sizes) } else { body_size };
    let tolerance = separator_tolerance(page);
    let separator = separator_beside_tables(page, separator, (0..page.lines.len()).filter(|&place| table_cells[place]).map(|place| &page.lines[place]));
    PageReading { body_size, note_size, tolerance, table_cells, separator }
}

/// How far from a page's note separator a line still stands at it.
pub(crate) fn separator_tolerance(page: &Page) -> f64 {
    (page.height * 0.004).max(1.0)
}

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
    for (place, line) in sheet.lines() {
        let label = || printed_label(line);
        found[place] = sheet.flagged(place, BAND_PARAGRAPH)
            && !reading.separator.is_some_and(|cut| line.bbox[1] >= cut - reading.tolerance)
            && !has_prior_reference(sheet.lines().map(|(_, line)| line), &label(), line.bbox[1])
            && !sheet.flagged(place, BAND_NOTED)
            && !sheet.lines().any(|(_, heading)| heading.bbox[1] < line.bbox[1] && endnote_heading(&heading.text))
            && !expected_endnote.is_some_and(|expected| {
                label().parse::<u32>().is_ok_and(|number| number >= expected)
                    && !sheet.lines().any(|(_, heading)| heading.bbox[1] < line.bbox[1] && structural_reset_heading(&heading.text))
            });
    }
    found
}

/// A line that opens with a note label: its place, the label, and whether its type sets it as one.
type Candidate = (usize, LabelPrefix, bool);

/// The lines that open with a note label, and the lines whose label is no note's: a table's cells,
/// a page number in the corner, a number before a comma, a reference's number wrapped onto the
/// next line, or a label set larger than the body.
fn label_candidates(sheet: &Sheet, reading: &PageReading, paragraph: &[bool]) -> (Vec<Candidate>, Vec<usize>) {
    let page = sheet.page;
    let body_size = reading.body_size;
    let mut candidates = Vec::new();
    let mut suppress = Vec::new();
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
        let Some(label) = sheet.labels[place] else {
            continue;
        };
        let Some(prefix) = label_prefix(&line.text) else {
            continue;
        };
        let typographic = label.typographic(body_size)
            || (standalone_note_label(line)
                && aligned_note_body_among(&page.lines, sheet.members.iter().copied(), place).is_some_and(|body| {
                    let size = sheet.sizes[body];
                    size > 0.0 && size <= body_size * 0.90
                }));
        let size = if sheet.sizes[place] > 0.0 { sheet.sizes[place] } else { body_size };
        let bottom_right = reading.separator.is_none() && label.suppress & SUPPRESS_CORNER != 0;
        // A number followed by a comma is the line's text ("1991, c. 43, s. 4" in a statute's
        // small-type history note), unless it is raised above its line.
        let comma_tail = label.suppress & SUPPRESS_COMMA != 0;
        // A number opening a line under one that ends on a reference's noun ("... c. 27 (1st
        // Supp.), s" / "13.") is that reference's number, wrapped.
        let wrapped_reference = label.suppress & SUPPRESS_WRAPPED != 0;
        let below_separator = reading.separator.is_some_and(|cut| line.bbox[1] >= cut - reading.tolerance);
        let suppressed = (label.suppress & SUPPRESS_FOOT != 0 && !(below_separator && typographic))
            || bottom_right
            || comma_tail
            || wrapped_reference
            || size > body_size * 1.15;
        if suppressed {
            suppress.push(place);
            continue;
        }
        candidates.push((place, prefix, typographic));
    }
    (candidates, suppress)
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
fn columns_among(band: &BandPage, sheet: &Sheet, keep: &[bool]) -> ColumnModel {
    let page = sheet.page;
    let kept = |place: usize| keep[place] && !sheet.furniture(place);
    let places = (0..page.lines.len()).filter(|&place| kept(place));
    columns_of(&page.lines, places, &kept_items(band.items, kept), page.index as u32 + 1)
}

fn page_facts(band: &BandPage, sheet: &Sheet, reading: &PageReading, candidates: &[Candidate]) -> PageFacts {
    let page = sheet.page;
    let (sizes, body_size) = (sheet.sizes, reading.body_size);
    let numeric: Vec<u32> = candidates.iter().filter_map(|(_, prefix, _)| prefix.label.parse().ok()).collect();
    let columns = {
        let page_model = if sheet.members.len() == page.lines.len() { band.columns } else { columns_among(band, sheet, &sheet.member) };
        if page_model.kind == "two_column" {
            page_model
        } else {
            let mut labelled = vec![false; page.lines.len()];
            for (place, _, _) in candidates {
                labelled[*place] = true;
            }
            let label_model = columns_among(band, sheet, &labelled);
            if label_model.kind == "single" { page_model } else { label_model }
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
            (sheet.flagged(place, BAND_HEADED) || line.bbox[1] >= page.height * 0.08) && structural_reset_heading(&line.text)
        });
    let label_sizes: Vec<f64> = candidates.iter().map(|(place, _, _)| sizes[*place]).filter(|size| *size > 0.0).collect();
    let label_size = (!label_sizes.is_empty()).then(|| median(label_sizes));
    let early_labels = candidates.iter().any(|(place, _, typographic)| {
        sheet.line(*place).bbox[1] < page.height * 0.48 && (*typographic || sizes[*place] <= body_size * 0.90)
    });
    let lower_numeric: Vec<u32> = candidates
        .iter()
        .filter(|(place, _, _)| {
            sheet.line(*place).bbox[1] >= page.height * 0.48 && sizes[*place] > 0.0 && sizes[*place] <= body_size * 0.90
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
    let content_sizes: Vec<f64> = sheet
        .lines()
        .filter(|&(place, _)| !sheet.furniture(place) && !sheet.excluded(place))
        .map(|(place, _)| sheet.sizes[place])
        .filter(|size| *size > 0.0)
        .collect();
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
        .filter_map(|(place, _, _)| (sheet.sizes[*place] <= note_size * 0.90 && line_width(sheet.line(*place)) <= page.width * 0.30).then_some(*place))
        .collect();
    let margin_numeric: Vec<u32> = candidates
        .iter()
        .filter(|(place, _, _)| margin_candidates.contains(place))
        .filter_map(|(_, prefix, _)| prefix.label.parse().ok())
        .collect();
    let supported: Vec<_> = candidates
        .iter()
        .filter(|(place, prefix, _)| {
            margin_candidates.contains(place) && has_prior_reference(sheet.lines().map(|(_, line)| line), &prefix.label, sheet.line(*place).bbox[1])
        })
        .map(|(place, _, _)| *place)
        .collect();
    let margin_labels = if longest_label_run(&margin_numeric) >= 3 { &margin_candidates } else { &supported };
    let model = margin_note_model(
        &page.lines,
        &sheet.members,
        |place| sheet.furniture(place) || sheet.excluded(place),
        margin_labels,
        page.width,
        note_size,
        if margin_labels.len() >= 3 { 3 } else { 2 },
    )?;
    let side = usize::from(margin_labels.iter().filter(|place| line_center_x(sheet.line(**place)) >= model.split_x).count() * 2 >= margin_labels.len());
    Some((model, side))
}

/// The candidates that are notes: all of them on an endnote page or in the margin column, else those
/// in the page's lower half under its separator, set as labels in a two-column page, or (with no
/// separator) set as labels, backed by a reference above, or in a run of small labels; and, with no
/// separator, a small label numbered next after the document's notes so far (`next_label`) whose
/// reference stands above it, anywhere on the page.
fn labels(
    sheet: &Sheet,
    reading: &PageReading,
    candidates: &[Candidate],
    facts: &PageFacts,
    endnote_page: bool,
    margin: Option<(ColumnModel, usize)>,
    next_label: u32,
) -> Vec<usize> {
    let page = sheet.page;
    let (sizes, body_size, separator) = (sheet.sizes, reading.body_size, reading.separator);
    let mut next_label = next_label;
    candidates
        .iter()
        .filter_map(|(place, prefix, typographic)| {
            let line = sheet.line(*place);
            let size = sizes[*place];
            let lower = line.bbox[1] >= page.height * 0.48;
            let small = size > 0.0 && size <= body_size * 0.90;
            let referenced = || has_prior_reference(sheet.lines().map(|(_, line)| line), &prefix.label, line.bbox[1]);
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
                        || (facts.columns.kind == "two_column" && *typographic)
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
fn note_cut(
    sheet: &Sheet,
    reading: &PageReading,
    labels: &[usize],
    contents_page: bool,
    endnote_page: bool,
    continuation: &Continuation,
    run: &EndnoteRun,
    furniture: &dyn Fn(usize) -> bool,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<f64> {
    let page = sheet.page;
    let (sizes, body_size, separator, tolerance) = (sheet.sizes, reading.body_size, reading.separator, reading.tolerance);
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
    let selected_sizes: Vec<f64> = labels.iter().map(|place| sizes[*place]).filter(|size| *size > 0.0).collect();
    let selected_size = (!selected_sizes.is_empty()).then(|| median(selected_sizes));
    let separator_starts_small_text = separator.is_some_and(|cut| {
        !contents_page
            && sheet
                .lines()
                .filter(|&(place, line)| {
                    !reading.table_cells[place] && !sheet.excluded(place) && !furniture(place) && line.bbox[1] >= cut - tolerance
                })
                .min_by(|(_, left), (_, right)| left.bbox[1].total_cmp(&right.bbox[1]))
                .is_some_and(|(place, line)| line.bbox[1] - cut <= page.height * 0.06 && sizes[place] > 0.0 && sizes[place] <= body_size * 0.90)
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
        let mut diagnostic = Diagnostic::warning(
            "FOOTNOTE_REGION_UNCERTAIN",
            "Footnote region inferred from weak label geometry without a separator.",
            Some(page.index),
        );
        if let Some(place) = labels.iter().min_by(|left, right| sheet.line(**left).bbox[1].total_cmp(&sheet.line(**right).bbox[1])) {
            diagnostic.line_ids.push(sheet.line(*place).id.clone());
        }
        diagnostics.push(diagnostic);
    }
    Some(first_label)
}

/// Where the band begins in each column of a two-column page or beside a margin column: at each
/// column's first label (or the note body it aligns with), and under a notes heading at the
/// column's first line. `furniture` holds the page's running heads and feet.
fn column_cuts(
    sheet: &Sheet,
    labels: &[usize],
    columns: ColumnModel,
    has_margin: bool,
    endnote_page: bool,
    facts: &PageFacts,
    furniture: &dyn Fn(usize) -> bool,
) -> Option<[Option<f64>; 2]> {
    if !((columns.kind == "two_column" || has_margin) && (!endnote_page || facts.endnote_heading_y.is_some())) {
        return None;
    }
    let lines = &sheet.page.lines;
    let mut cuts = [None::<f64>; 2];
    for &place in labels {
        let label = sheet.line(place);
        let column = usize::from(line_center_x(label) >= columns.split_x);
        let label_top = aligned_note_body_among(lines, sheet.members.iter().copied(), place)
            .map(|body| band_geometry_top(&lines[body]))
            .unwrap_or_else(|| band_geometry_top(label))
            .min(band_geometry_top(label));
        cuts[column] = Some(cuts[column].map_or(label_top, |cut| cut.min(label_top)));
    }
    if !(cuts.iter().all(Option::is_some) || (columns.kind == "margin_column" && cuts.iter().any(Option::is_some))) {
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
        let referenced = labels
            .iter()
            .filter(|label| {
                sheets[..slot].iter().any(|sheet| has_prior_reference(sheet.lines().map(|(_, line)| line), label, f64::INFINITY))
                    || has_prior_reference(sheets[slot].lines().map(|(_, line)| line), label, first_note)
            })
            .count();
        if !headed && referenced * 2 < labels.len().max(1) {
            for page in &mut notes[slot..end] {
                page.retain(|(_, kind)| *kind != NoteKind::Endnote);
            }
        }
        slot = end;
    }
}

/// Every page's note band, in order, then the endnote runs the body never refers to given back.
pub(crate) fn read_band(pages: &[BandPage], lines: &BandLines) -> BandReading {
    let mut found = BandReading::default();
    let sheets: Vec<Sheet> = pages.iter().map(|page| Sheet::of(page, lines)).collect();
    let mut notes: Vec<Vec<(usize, NoteKind)>> = Vec::with_capacity(sheets.len());
    let mut run = EndnoteRun::default();
    // The label the document's next note takes, by the notes read so far.
    let mut next_label = 1;
    for (band, sheet) in pages.iter().zip(&sheets) {
        let page = sheet.page;
        run.enter(page.index);
        let reading = page_reading(sheet, band.separator, lines.body_size);
        let paragraph = paragraph_labels(sheet, &reading, run.expected);
        found.paragraph_labels.extend(sheet.members.iter().filter(|&&place| paragraph[place]).map(|&place| sheet.view.slot(place)));
        let (candidates, suppress) = label_candidates(sheet, &reading, &paragraph);
        found.suppressed.extend(suppress.into_iter().map(|place| sheet.view.slot(place)));
        let facts = page_facts(band, sheet, &reading, &candidates);
        let (continuation, suppress) = continue_endnotes(sheet, &candidates, &facts, &run);
        found.suppressed.extend(suppress.into_iter().map(|place| sheet.view.slot(place)));
        let margin = margin_notes(sheet, &reading, &candidates);
        let endnote_page = facts.endnote_heading_y.is_some()
            || continuation.continuing
            || (reading.separator.is_none()
                && margin.is_none()
                && !candidates.is_empty()
                && facts.early_labels
                && facts.best_run >= 3
                && facts.label_size.is_some_and(|size| size <= reading.body_size * 0.90));
        let labels = labels(sheet, &reading, &candidates[continuation.selected_start..], &facts, endnote_page, margin, next_label);
        let heads = if endnote_page && !labels.is_empty() { running_heads(sheet, &labels) } else { Vec::new() };
        found.running_heads.extend(heads.iter().map(|&place| sheet.view.slot(place)));
        let furniture = |place: usize| sheet.furniture(place) || heads.contains(&place);
        let cut = note_cut(sheet, &reading, &labels, band.contents, endnote_page, &continuation, &run, &furniture, &mut found.diagnostics);
        let columns = margin.map_or(facts.columns, |(model, _)| model);
        let band = Band {
            endnote_page,
            cut,
            column_cuts: column_cuts(sheet, &labels, columns, margin.is_some(), endnote_page, &facts, &furniture),
            columns,
            endnote_heading_source: facts.endnote_heading_source,
        };
        // Each line's reading: a notes heading is a heading, a table's cell body text, furniture
        // stays, a source's heading survives, a line a layout model labelled keeps its reading of
        // note or text, a numbered paragraph's line is body, and the rest are note or body by the band.
        let kind = if endnote_page { NoteKind::Endnote } else { NoteKind::Footnote };
        let mut page_notes = Vec::new();
        for (place, line) in sheet.lines() {
            if sheet.excluded(place) {
                continue;
            }
            if endnote_heading(&line.text) {
                found.headings.push(sheet.view.slot(place));
            } else if reading.table_cells[place] || furniture(place) || sheet.flagged(place, BAND_HEADED) {
            } else if legal_structure_model::ablated("layout_notes") && sheet.flagged(place, BAND_LAYOUT) {
                if sheet.flagged(place, BAND_NOTED) {
                    page_notes.push((place, kind));
                }
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
    for (sheet, page_notes) in sheets.iter().zip(notes) {
        found.notes.extend(page_notes.into_iter().map(|(place, kind)| (sheet.view.slot(place), kind)));
    }
    found
}
