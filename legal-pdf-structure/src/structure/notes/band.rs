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
//! 8. `note_cut` / `column_cuts`: where the band begins, across the page or per column.
//! 9. `assign_roles`: each line's role from the band, its claims and its source's labels.
//!
//! A line a page layout model labelled keeps that label's reading of note or text (step 9).

use super::*;

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
    fn leave(&mut self, page: &Page, line_sizes: &[f64]) {
        if !page.lines.iter().any(|line| line.note_region_mode == "endnote") {
            self.expected = None;
            self.size = None;
            return;
        }
        if let Some(last) = page
            .lines
            .iter()
            .rev()
            .filter(|line| line.note_region_mode == "endnote")
            .filter_map(|line| label_prefix(&line.text)?.label.parse::<u32>().ok())
            .next()
        {
            self.expected = Some(last + 1);
        }
        let size = median(
            page.lines
                .iter()
                .enumerate()
                .filter(|(_, line)| line.note_region_mode == "endnote")
                .map(|(index, _)| line_sizes[index])
                .filter(|size| *size > 0.0)
                .collect(),
        );
        if size > 0.0 {
            self.size = Some(size);
        }
    }
}

fn furniture(line: &Line) -> bool {
    matches!(line.region_type.as_str(), "header" | "footer")
}

/// The type and geometry a page's notes are read against.
struct PageReading {
    line_sizes: Vec<f64>,
    body_size: f64,
    /// The type of the page's wide lines of words: its running text, against which a margin's
    /// notes are small.
    note_size: f64,
    tolerance: f64,
    table_cells: HashSet<usize>,
    /// The page's note separator rule, unless it runs through a table.
    separator: Option<f64>,
}

fn page_reading(page: &Page, separator: Option<f64>, document_body: f64, evidence: &PdfPrimitiveEvidence) -> PageReading {
    let table_cells = page
        .lines
        .iter()
        .enumerate()
        .filter(|(_, line)| evidence.claims.table_cell_line_ids.contains(&line.id))
        .map(|(slot, _)| slot)
        .collect::<HashSet<_>>();
    let line_sizes: Vec<f64> = page.lines.iter().map(line_font_size).collect();
    let body_size = if document_body > 0.0 && line_sizes.iter().any(|size| *size > 0.0) {
        document_body
    } else {
        median(
            page.lines
                .iter()
                .enumerate()
                .filter(|(index, line)| {
                    !furniture(line)
                        && !line.exclude_from_body
                        && !table_cells.contains(index)
                        && page.height * 0.10 <= line.bbox[1]
                        && line.bbox[1] <= page.height * 0.75
                })
                .map(|(index, _)| line_sizes[index])
                .filter(|size| (4.0..=24.0).contains(size))
                .collect(),
        )
    };
    let body_size = if body_size > 0.0 { body_size } else { 10.0 };
    let wide_sizes: Vec<_> = page
        .lines
        .iter()
        .enumerate()
        .filter(|(index, line)| {
            !line.exclude_from_body
                && !table_cells.contains(index)
                && line_width(line) >= page.width * 0.30
                && line.text.chars().filter(|character| character.is_alphabetic()).take(4).count() == 4
        })
        .map(|(index, _)| line_sizes[index])
        .filter(|size| (4.0..=24.0).contains(size))
        .collect();
    let note_size = if wide_sizes.len() >= 3 { median(wide_sizes) } else { body_size };
    let tolerance = (page.height * 0.004).max(1.0);
    let table_band = table_cells.iter().fold(None, |band, index| {
        let line = &page.lines[*index];
        Some(band.map_or((line.bbox[1], line.bbox[3]), |(top, bottom): (f64, f64)| {
            (top.min(line.bbox[1]), bottom.max(line.bbox[3]))
        }))
    });
    let separator = separator
        .filter(|cut| table_band.is_none_or(|(top, bottom)| *cut < top - tolerance || *cut > bottom + tolerance));
    PageReading { line_sizes, body_size, note_size, tolerance, table_cells, separator }
}

/// The lines whose number is a numbered paragraph's, never a note's. Alignment, typography and a
/// paragraph sequence are positive evidence, not a veto on the note witnesses: rules, references,
/// source regions and note continuation win.
fn paragraph_labels(
    page: &Page,
    reading: &PageReading,
    paragraph_candidates: &HashSet<String>,
    evidence: &PdfPrimitiveEvidence,
    expected_endnote: Option<u32>,
) -> HashSet<String> {
    page.lines
        .iter()
        .filter(|line| {
            paragraph_candidates.contains(&line.id)
                && !reading.separator.is_some_and(|cut| line.bbox[1] >= cut - reading.tolerance)
                && !has_prior_reference(page, line.text.trim(), line.bbox[1])
                && !evidence
                    .source_regions
                    .as_ref()
                    .and_then(|regions| regions.get(&line.id))
                    .is_some_and(|region| region.contains("note"))
                && !page.lines.iter().any(|heading| heading.bbox[1] < line.bbox[1] && endnote_heading(&heading.text))
                && !expected_endnote.is_some_and(|expected| {
                    line.text.trim().parse::<u32>().is_ok_and(|number| number >= expected)
                        && !page.lines.iter().any(|heading| {
                            heading.bbox[1] < line.bbox[1] && structural_reset_heading(&heading.text)
                        })
                })
        })
        .map(|line| line.id.clone())
        .collect()
}

/// A line that opens with a note label: its slot, the label, and whether its type sets it as one.
type Candidate = (usize, LabelPrefix, bool);

/// The lines that open with a note label, and the lines whose label is no note's: a table's cells,
/// a page number in the corner, a number before a comma, a reference's number wrapped onto the
/// next line, or a label set larger than the body.
fn label_candidates(page: &Page, reading: &PageReading) -> (Vec<Candidate>, Vec<usize>) {
    let (sizes, body_size) = (&reading.line_sizes, reading.body_size);
    let mut candidates = Vec::new();
    let mut suppress = Vec::new();
    for (index, line) in page.lines.iter().enumerate() {
        if line.paragraph_label_candidate {
            continue;
        }
        if reading.table_cells.contains(&index) {
            suppress.push(index);
            continue;
        }
        if furniture(line) || line.exclude_from_body {
            continue;
        }
        let Some(prefix) = label_prefix(&line.text) else {
            continue;
        };
        let typographic = label_is_typographic(line, &prefix, sizes[index], body_size)
            || (standalone_note_label(line)
                && aligned_note_body_index(&page.lines, index).is_some_and(|body| {
                    let size = sizes[body];
                    size > 0.0 && size <= body_size * 0.90
                }));
        let size = if sizes[index] > 0.0 { sizes[index] } else { body_size };
        let bottom_right = reading.separator.is_none()
            && line.bbox[1] >= page.height * 0.91
            && line.bbox[0] >= page.width * 0.50
            && !compact_note_line(&line.text);
        // A number followed by a comma is the line's text ("1991, c. 43, s. 4" in a statute's
        // small-type history note), unless it is raised above its line.
        let comma_tail = line.text.chars().nth(prefix.end) == Some(',') && !label_is_raised(line, &prefix, sizes[index]);
        // A number opening a line under one that ends on a reference's noun ("... c. 27 (1st
        // Supp.), s" / "13.") is that reference's number, wrapped.
        let wrapped_reference =
            !label_is_raised(line, &prefix, sizes[index]) && reference_wrapped_onto(&page.lines, sizes, index);
        let below_separator = reading.separator.is_some_and(|cut| line.bbox[1] >= cut - reading.tolerance);
        let suppressed = (line.bbox[1] >= page.height * 0.94 && !(below_separator && typographic))
            || bottom_right
            || comma_tail
            || wrapped_reference
            || size > body_size * 1.15;
        if suppressed {
            suppress.push(index);
            continue;
        }
        candidates.push((index, prefix, typographic));
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

fn page_facts(page: &Page, reading: &PageReading, candidates: &[Candidate]) -> PageFacts {
    let (sizes, body_size) = (&reading.line_sizes, reading.body_size);
    let numeric: Vec<u32> = candidates.iter().filter_map(|(_, prefix, _)| prefix.label.parse().ok()).collect();
    let columns = {
        let page_model = column_model(&page.lines);
        if page_model.kind == "two_column" {
            page_model
        } else {
            let label_lines: Vec<_> = candidates.iter().map(|(index, _, _)| page.lines[*index].clone()).collect();
            let label_model = column_model(&label_lines);
            if label_model.kind == "single" { page_model } else { label_model }
        }
    };
    let first_candidate_y = candidates.first().map(|(index, _, _)| page.lines[*index].bbox[1]);
    let minimum_label_y = candidates
        .iter()
        .map(|(index, _, _)| page.lines[*index].bbox[1])
        .min_by(f64::total_cmp)
        .unwrap_or(page.height);
    let endnote_heading_y = page
        .lines
        .iter()
        .filter(|line| endnote_heading(&line.text) && first_candidate_y.is_none_or(|y| line.bbox[1] < y))
        .map(|line| line.bbox[1])
        .min_by(f64::total_cmp);
    let endnote_heading_source = endnote_heading_y.is_some().then(|| {
        page.lines.iter().filter(|line| endnote_heading(&line.text)).map(|line| line.source_index).min().unwrap_or(0)
    });
    let content_before = || {
        page.lines
            .iter()
            .enumerate()
            .filter_map(|(index, line)| (!furniture(line) && line.bbox[1] < minimum_label_y).then_some(index))
    };
    let first_content =
        content_before().min_by(|left, right| page.lines[*left].bbox[1].total_cmp(&page.lines[*right].bbox[1]));
    let generic_heading_reset = candidates.is_empty()
        && first_content.is_some_and(|index| {
            let line = &page.lines[index];
            let text = line.text.trim();
            let letter_count = text.chars().filter(|character| character.is_alphabetic()).count();
            line.bbox[1] >= page.height * 0.08
                && letter_count >= 4
                && text.chars().count() <= 100
                && all_caps(text)
                && !continuing_note_heading(text)
        });
    let structural_reset = generic_heading_reset
        || content_before().any(|index| {
            let line = &page.lines[index];
            (line.region_type == "heading" || line.bbox[1] >= page.height * 0.08)
                && structural_reset_heading(&line.text)
        });
    let label_sizes: Vec<f64> =
        candidates.iter().map(|(index, _, _)| sizes[*index]).filter(|size| *size > 0.0).collect();
    let label_size = (!label_sizes.is_empty()).then(|| median(label_sizes));
    let early_labels = candidates.iter().any(|(index, _, typographic)| {
        page.lines[*index].bbox[1] < page.height * 0.48 && (*typographic || sizes[*index] <= body_size * 0.90)
    });
    let lower_numeric: Vec<u32> = candidates
        .iter()
        .filter(|(index, _, _)| {
            page.lines[*index].bbox[1] >= page.height * 0.48
                && sizes[*index] > 0.0
                && sizes[*index] <= body_size * 0.90
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
/// citation-shaped ones numbered next, are suppressed.
fn continue_endnotes(page: &mut Page, reading: &PageReading, candidates: &[Candidate], facts: &PageFacts, run: &EndnoteRun) -> Continuation {
    let expected_indexes: Vec<usize> = candidates
        .iter()
        .enumerate()
        .filter_map(|(index, (_, prefix, _))| (prefix.label.parse::<u32>().ok() == run.expected).then_some(index))
        .collect();
    let noncitation: Vec<usize> = expected_indexes
        .iter()
        .copied()
        .filter(|index| !citation_shaped_candidate(page, &candidates[*index]))
        .collect();
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
            for candidate in &candidates[..selected] {
                page.lines[candidate.0].suppress_footnote_label = true;
            }
            for index in &expected_indexes {
                if *index != selected && citation_shaped_candidate(page, &candidates[*index]) {
                    page.lines[candidates[*index].0].suppress_footnote_label = true;
                }
            }
        }
    }
    let content_sizes: Vec<f64> = page
        .lines
        .iter()
        .enumerate()
        .filter(|(_, line)| !furniture(line) && !line.exclude_from_body)
        .map(|(index, _)| reading.line_sizes[index])
        .filter(|size| *size > 0.0)
        .collect();
    let content_size = (!content_sizes.is_empty()).then(|| median(content_sizes));
    let size_matches = run.size.is_none_or(|size| content_size.is_none_or(|content| content <= size * 1.15));
    let label_free = run.expected.is_some() && candidates.is_empty() && !facts.structural_reset && size_matches;
    Continuation {
        selected_start: if run.expected.is_some() { expected_index.unwrap_or(0) } else { 0 },
        label_free,
        continuing: run.expected.is_some()
            && !facts.structural_reset
            && size_matches
            && (label_free || expected_index.is_some()),
    }
}

/// Notes set in a margin column: small, narrow labels counting on (or each referred to above), and
/// the side of the page they stand on.
fn margin_notes(page: &Page, reading: &PageReading, candidates: &[Candidate]) -> Option<(ColumnModel, usize)> {
    let note_size = reading.note_size;
    let margin_candidates: Vec<_> = candidates
        .iter()
        .filter_map(|(index, _, _)| {
            let line = &page.lines[*index];
            (reading.line_sizes[*index] <= note_size * 0.90 && line_width(line) <= page.width * 0.30)
                .then_some(*index)
        })
        .collect();
    let margin_numeric: Vec<u32> = candidates
        .iter()
        .filter(|(index, _, _)| margin_candidates.contains(index))
        .filter_map(|(_, prefix, _)| prefix.label.parse().ok())
        .collect();
    let supported: Vec<_> = candidates
        .iter()
        .filter(|(index, prefix, _)| {
            margin_candidates.contains(index) && has_prior_reference(page, &prefix.label, page.lines[*index].bbox[1])
        })
        .map(|(index, _, _)| *index)
        .collect();
    let margin_labels = if longest_label_run(&margin_numeric) >= 3 { &margin_candidates } else { &supported };
    let model = margin_note_model(
        &page.lines,
        margin_labels,
        page.width,
        note_size,
        if margin_labels.len() >= 3 { 3 } else { 2 },
    )?;
    let side = usize::from(
        margin_labels.iter().filter(|index| line_center_x(&page.lines[**index]) >= model.split_x).count() * 2
            >= margin_labels.len(),
    );
    Some((model, side))
}

/// The candidates that are notes: all of them on an endnote page or in the margin column, else those
/// in the page's lower half under its separator, set as labels in a two-column page, or (with no
/// separator) set as labels, backed by a reference above, or in a run of small labels.
fn labels(
    page: &Page,
    reading: &PageReading,
    candidates: &[Candidate],
    facts: &PageFacts,
    endnote_page: bool,
    margin: Option<(ColumnModel, usize)>,
) -> Vec<usize> {
    let (sizes, body_size, separator) = (&reading.line_sizes, reading.body_size, reading.separator);
    candidates
        .iter()
        .filter_map(|(index, prefix, typographic)| {
            let line = &page.lines[*index];
            let size = sizes[*index];
            let lower = line.bbox[1] >= page.height * 0.48;
            let reference_backed = lower
                && size > 0.0
                && (size <= body_size * 0.90 || prefix.label.chars().all(is_note_symbol))
                && has_prior_reference(page, &prefix.label, line.bbox[1]);
            let in_margin = margin.is_some_and(|(model, side)| {
                usize::from(line_center_x(line) >= model.split_x) == side
                    && size <= reading.note_size * 0.90
                    && line_width(line) <= page.width * 0.30
            });
            (endnote_page
                || in_margin
                || (lower
                    && (separator.is_some_and(|cut| line.bbox[1] >= cut - reading.tolerance)
                        || (facts.columns.kind == "two_column" && *typographic)
                        || (separator.is_none()
                            && (*typographic
                                || reference_backed
                                || (facts.lower_note_run
                                    && !recognized_page(page)
                                    && size > 0.0
                                    && size <= body_size * 0.90))))))
                .then_some(*index)
        })
        .collect()
}

/// On an endnote page, a short line in capitals at the very top, above every label and set in from
/// the margins, is the page's running head.
fn mark_running_head(page: &mut Page, labels: &[usize]) {
    let first_label = labels.iter().map(|index| page.lines[*index].bbox[1]).min_by(f64::total_cmp).unwrap_or(page.height);
    let (width, height) = (page.width, page.height);
    for line in &mut page.lines {
        let text = line.text.trim();
        let letter_count = text.chars().filter(|character| character.is_alphabetic()).count();
        if !furniture(line)
            && line.bbox[1] < height * 0.08
            && line.bbox[1] < first_label
            && line.bbox[0] >= width * 0.15
            && line.bbox[2] <= width * 0.85
            && letter_count >= 4
            && text.chars().count() <= 100
            && all_caps(text)
            && line_start_label_prefix(text).is_none()
        {
            line.region_type = "header".to_owned();
        }
    }
}

/// Where the band begins across the page: at the top of the page's running text on a page of
/// endnotes continued, at the separator when the first label sits close under it (or the small
/// type starts under it), else at the first label, warned of where that rests on weak geometry.
fn note_cut(
    page: &Page,
    reading: &PageReading,
    labels: &[usize],
    contents_pages: &HashSet<usize>,
    endnote_page: bool,
    continuation: &Continuation,
    run: &EndnoteRun,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<f64> {
    let (sizes, body_size, separator, tolerance) = (&reading.line_sizes, reading.body_size, reading.separator, reading.tolerance);
    let eligible_top = || {
        page.lines
            .iter()
            .filter(|line| !furniture(line) && !line.exclude_from_body)
            .map(|line| line.bbox[1])
            .min_by(f64::total_cmp)
    };
    if continuation.label_free {
        return eligible_top();
    }
    if labels.is_empty() {
        return None;
    }
    let selected_sizes: Vec<f64> = labels.iter().map(|index| sizes[*index]).filter(|size| *size > 0.0).collect();
    let selected_size = (!selected_sizes.is_empty()).then(|| median(selected_sizes));
    let separator_starts_small_text = separator.is_some_and(|cut| {
        !contents_pages.contains(&page.index)
            && page
                .lines
                .iter()
                .enumerate()
                .filter(|(index, line)| {
                    !reading.table_cells.contains(index)
                        && !line.exclude_from_body
                        && !furniture(line)
                        && line.bbox[1] >= cut - tolerance
                })
                .min_by(|(_, left), (_, right)| left.bbox[1].total_cmp(&right.bbox[1]))
                .is_some_and(|(index, line)| {
                    line.bbox[1] - cut <= page.height * 0.06 && sizes[index] > 0.0 && sizes[index] <= body_size * 0.90
                })
    });
    let first_label = labels.iter().map(|index| page.lines[*index].bbox[1]).min_by(f64::total_cmp).unwrap_or(page.height);
    if endnote_page && run.expected.is_some() {
        return eligible_top();
    }
    if separator.is_some_and(|cut| {
        0.0 <= first_label - cut
            && (first_label - cut <= page.height * 0.15
                || (separator_starts_small_text && selected_size.is_some_and(|size| size <= body_size * 0.90)))
    }) {
        return separator;
    }
    let confident = (first_label >= page.height * 0.58 || endnote_page)
        && selected_size.is_some_and(|size| size <= body_size * 0.90);
    if !confident {
        let mut diagnostic = Diagnostic::warning(
            "FOOTNOTE_REGION_UNCERTAIN",
            "Footnote region inferred from weak label geometry without a separator.",
            Some(page.index),
        );
        if let Some(index) = labels.iter().min_by(|left, right| page.lines[**left].bbox[1].total_cmp(&page.lines[**right].bbox[1])) {
            diagnostic.line_ids.push(page.lines[*index].id.clone());
        }
        diagnostics.push(diagnostic);
    }
    Some(first_label)
}

/// Where the band begins in each column of a two-column page or beside a margin column: at each
/// column's first label (or the note body it aligns with), and under a notes heading at the
/// column's first line.
fn column_cuts(
    page: &Page,
    labels: &[usize],
    columns: ColumnModel,
    has_margin: bool,
    endnote_page: bool,
    facts: &PageFacts,
) -> Option<[Option<f64>; 2]> {
    if !((columns.kind == "two_column" || has_margin) && (!endnote_page || facts.endnote_heading_y.is_some())) {
        return None;
    }
    let mut cuts = [None::<f64>; 2];
    for index in labels {
        let label = &page.lines[*index];
        let column = usize::from(line_center_x(label) >= columns.split_x);
        let label_top = aligned_note_body_index(&page.lines, *index)
            .map(|body| band_geometry_top(&page.lines[body]))
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
                *cut = page
                    .lines
                    .iter()
                    .filter(|line| {
                        !furniture(line)
                            && !line.exclude_from_body
                            && usize::from(line_center_x(line) >= columns.split_x) == column
                    })
                    .map(|line| line.bbox[1])
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

/// Each line's role: a notes heading is a heading, a table's cell body text, furniture stays, a
/// source's heading survives, a line a layout model labelled keeps its reading of note or text, a
/// numbered paragraph's line is body, and the rest are note or body by the band.
fn assign_roles(page: &mut Page, reading: &PageReading, band: &Band, paragraph_labels: &HashSet<String>, evidence: &PdfPrimitiveEvidence) {
    let source_region = |id: &str| evidence.source_regions.as_ref().and_then(|regions| regions.get(id));
    let note_mode = if band.endnote_page { "endnote" } else { "footnote" };
    for (index, line) in page.lines.iter_mut().enumerate() {
        if line.exclude_from_body {
            continue;
        }
        let is_note = band.holds(line);
        if endnote_heading(&line.text) {
            line.region_type = "heading".to_owned();
            line.note_region_mode.clear();
        } else if reading.table_cells.contains(&index) {
            line.region_type = "body".to_owned();
            line.note_region_mode.clear();
        } else if furniture(line) {
            continue;
        } else if legal_pdf_core::is_heading_region(source_region(&line.id).map(String::as_str).unwrap_or(line.region_type.as_str())) {
            // Source roles survive the font-size fallback, including recognized and tagged text.
            line.region_type = "heading".to_owned();
            line.note_region_mode.clear();
        } else if legal_structure_model::ablated("layout_notes") && layout_labelled(line) {
            let note = source_region(&line.id).is_some_and(|region| region == "footnote");
            line.region_type = if note { "footnote" } else { "body" }.to_owned();
            line.note_region_mode = if note { note_mode.to_owned() } else { String::new() };
        } else if paragraph_labels.contains(&line.id) {
            line.region_type = "body".to_owned();
            line.note_region_mode.clear();
        } else if is_note {
            line.region_type = "footnote".to_owned();
            line.note_region_mode = note_mode.to_owned();
        } else {
            line.region_type = "body".to_owned();
        }
    }
}

/// Every page's note band, in order, then the endnote runs the body never refers to given back.
pub(crate) fn classify_line_roles(
    pages: &mut [Page],
    separators: &[Option<f64>],
    contents_pages: &HashSet<usize>,
    evidence: &mut PdfPrimitiveEvidence,
) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    let paragraph_candidates = numbered_paragraphs::margin_label_ids(pages);
    // Notes are set small against the document's body type, read once from all its running text,
    // not against each page's own lines: a page mostly of notes would take their size for its
    // body. A page whose lines carry no size falls back to its own.
    let document_body = body_font_size(pages.iter().flat_map(|page| &page.lines).filter(|line| {
        !furniture(line) && !line.exclude_from_body && !evidence.claims.table_cell_line_ids.contains(&line.id)
    }));
    let mut run = EndnoteRun::default();
    for (page, separator) in pages.iter_mut().zip(separators.iter().copied()) {
        run.enter(page.index);
        let reading = page_reading(page, separator, document_body, evidence);
        let paragraph_labels = paragraph_labels(page, &reading, &paragraph_candidates, evidence, run.expected);
        for line in &mut page.lines {
            line.paragraph_label_candidate = paragraph_labels.contains(&line.id);
        }
        let (candidates, suppress) = label_candidates(page, &reading);
        for index in suppress {
            page.lines[index].suppress_footnote_label = true;
        }
        let facts = page_facts(page, &reading, &candidates);
        let continuation = continue_endnotes(page, &reading, &candidates, &facts, &run);
        let margin = margin_notes(page, &reading, &candidates);
        let endnote_page = facts.endnote_heading_y.is_some()
            || continuation.continuing
            || (reading.separator.is_none()
                && margin.is_none()
                && !candidates.is_empty()
                && facts.early_labels
                && facts.best_run >= 3
                && facts.label_size.is_some_and(|size| size <= reading.body_size * 0.90));
        let labels = labels(page, &reading, &candidates[continuation.selected_start..], &facts, endnote_page, margin);
        if endnote_page && !labels.is_empty() {
            mark_running_head(page, &labels);
        }
        let cut = note_cut(page, &reading, &labels, contents_pages, endnote_page, &continuation, &run, &mut diagnostics);
        let columns = margin.map_or(facts.columns, |(model, _)| model);
        let band = Band {
            endnote_page,
            cut,
            column_cuts: column_cuts(page, &labels, columns, margin.is_some(), endnote_page, &facts),
            columns,
            endnote_heading_source: facts.endnote_heading_source,
        };
        assign_roles(page, &reading, &band, &paragraph_labels, evidence);
        run.leave(page, &reading.line_sizes);
    }
    verify_endnotes(pages);
    diagnostics
}
