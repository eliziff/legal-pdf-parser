use crate::structure::{is_note_symbol, label_prefix, scalar_suffix, MAX_SYMBOL_LABEL_LEN};
use legal_pdf_core::{line_font_size, median, union_bbox, Diagnostic, Line, Page, Region};
use regex::Regex;
use std::{
    collections::{HashMap, HashSet},
    sync::OnceLock,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OrderRepair {
    None,
    Column,
    Geometry,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ColumnModel {
    pub(super) kind: &'static str,
    pub(super) split_x: f64,
    left_count: usize,
    right_count: usize,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct OrderDecision {
    pub(super) repair: OrderRepair,
    source_switches: usize,
    strategy: &'static str,
    pub(super) reason: &'static str,
}

impl OrderDecision {
    const fn keep(reason: &'static str) -> Self {
        Self {
            repair: OrderRepair::None,
            source_switches: 0,
            strategy: "kraken-native",
            reason,
        }
    }
}

pub(super) fn line_width(line: &Line) -> f64 {
    line.bbox[2] - line.bbox[0]
}

pub(super) fn has_valid_bbox(line: &Line) -> bool {
    line.bbox.iter().all(|value| value.is_finite())
        && line.bbox[2] > line.bbox[0]
        && line.bbox[3] > line.bbox[1]
}

pub(super) fn line_center_x(line: &Line) -> f64 {
    (line.bbox[0] + line.bbox[2]) / 2.0
}

fn line_center_y(line: &Line) -> f64 {
    (line.bbox[1] + line.bbox[3]) / 2.0
}

/// A page's grid of short lines in rows and columns, figures in a fifth of its cells: a plain
/// table a recognized page's line boxes measure. Tables themselves are pdf-inspector's
/// (legal_pdf_tables).
#[derive(Default)]
struct TableEvidence {
    rows: usize,
    columns: usize,
    numeric_cells: usize,
    cells: usize,
}

impl TableEvidence {
    fn strong(&self) -> bool {
        self.rows >= 6 && self.columns >= 3 && self.numeric_cells * 5 >= self.cells
    }
}

/// Notes to a table: a line set just below where a table ends on the page that opens with a
/// symbol mark ("*", "†", "a)") over words, and the lines set close under it.
pub(super) fn table_note_lines(lines: &[Line], cells: &HashSet<usize>, ends: &[f64]) -> HashSet<usize> {
    let line_height = median(cells
        .iter()
        .map(|index| lines[*index].bbox[3] - lines[*index].bbox[1])
        .filter(|height| *height > 0.0)
        .collect());
    if ends.is_empty() || line_height <= 0.0 {
        return HashSet::new();
    }
    let anchors: Vec<_> = lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| {
            let prefix = label_prefix(&line.text)?;
            let symbolic = prefix.label.chars().all(|character| !character.is_ascii_digit());
            let tail = scalar_suffix(&line.text, prefix.end);
            (symbolic
                && tail.chars().filter(|character| character.is_alphabetic()).count() >= 4
                && !cells.contains(&index)
                && ends.iter().any(|bottom| line.bbox[1] > *bottom && line.bbox[1] <= bottom + line_height * 3.0))
                .then_some(index)
        })
        .collect();
    let mut notes: HashSet<_> = anchors.iter().copied().collect();
    for anchor in anchors {
        let mut bottom = lines[anchor].bbox[3];
        let mut following: Vec<_> = lines
            .iter()
            .enumerate()
            .filter(|(index, line)| *index != anchor && line.bbox[1] >= lines[anchor].bbox[1])
            .collect();
        following.sort_by(|(_, left), (_, right)| left.bbox[1].total_cmp(&right.bbox[1]));
        for (index, line) in following {
            if line.bbox[1] - bottom > line_height * 1.6
                || cells.contains(&index)
                || has_table_caption(std::slice::from_ref(line))
            {
                break;
            }
            notes.insert(index);
            bottom = bottom.max(line.bbox[3]);
        }
    }
    notes
}

pub(super) fn has_table_caption(lines: &[Line]) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    let regex =
        RE.get_or_init(|| Regex::new(r"(?i)^(?:table|tableau)\s+(?:\d+|[ivxlcdm]+)\b").unwrap());
    lines.iter().any(|line| regex.is_match(line.text.trim()))
}

/// The layout readings this crate lends the table detector.
pub(super) struct TableReadings;

impl legal_pdf_tables::TableLayout for TableReadings {
    /// A page's white space reads as a table only where its lines stand in a plain grid of rows,
    /// columns and figures: justified prose and a note's label beside its text are gaps between
    /// words, not columns.
    fn white_space(&self, page: &Page) -> bool {
        table_evidence(&page.lines).strong()
    }

    fn column_language(&self, lines: &[&Line], minimum: usize) -> Option<bool> {
        column_language(language_votes(lines.iter().copied()), minimum)
    }
}

fn table_evidence(lines: &[Line]) -> TableEvidence {
    let height = median(lines
        .iter()
        .filter(|line| has_valid_bbox(line))
        .map(|line| line.bbox[3] - line.bbox[1])
        .filter(|value| *value > 0.0)
        .collect());
    if height <= 0.0 {
        return TableEvidence::default();
    }
    let caption = lines
        .iter()
        .filter(|line| has_table_caption(std::slice::from_ref(line)))
        .min_by(|left, right| left.bbox[1].total_cmp(&right.bbox[1]));
    let caption_bottom = caption.map(|line| line.bbox[3]);
    let mut rows: HashMap<i64, Vec<usize>> = HashMap::new();
    for (index, line) in lines.iter().enumerate().filter(|(_, line)| {
        has_valid_bbox(line)
            && !line.exclude_from_body
            && !matches!(line.region_type.as_str(), "header" | "footer")
    }) {
        rows.entry((line_center_y(line) / (height * 0.75)).round() as i64)
            .or_default()
            .push(index);
    }
    let mut dense_rows: Vec<_> = rows
        .into_values()
        .filter(|row| {
            row.len() >= 2
                && median(row
                    .iter()
                    .map(|index| lines[*index].text.trim().chars().count() as f64)
                    .collect())
                    <= 24.0
        })
        .collect();
    dense_rows.sort_by(|left, right| {
        let center = |row: &[usize]| {
            row.iter()
                .map(|index| line_center_y(&lines[*index]))
                .min_by(f64::total_cmp)
                .unwrap_or(0.0)
        };
        center(left).total_cmp(&center(right))
    });
    if let Some(bottom) = caption_bottom {
        dense_rows.retain(|row| {
            row.iter()
                .map(|index| line_center_y(&lines[*index]))
                .min_by(f64::total_cmp)
                .is_some_and(|center| center >= bottom - height)
        });
    }
    let mut prior = None;
    dense_rows = dense_rows
        .into_iter()
        .take_while(|row| {
            let center = row
                .iter()
                .map(|index| line_center_y(&lines[*index]))
                .min_by(f64::total_cmp)
                .unwrap_or(0.0);
            let connected = prior.is_none_or(|value| center - value <= height * 6.0);
            prior = Some(center);
            connected
        })
        .collect();
    if dense_rows.len() < 3 {
        return TableEvidence::default();
    }
    let mut columns: HashMap<i64, usize> = HashMap::new();
    for row in &dense_rows {
        let mut seen = HashSet::new();
        for index in row {
            seen.insert((lines[*index].bbox[0] / (height * 2.0)).round() as i64);
        }
        for column in seen {
            *columns.entry(column).or_default() += 1;
        }
    }
    let row_count = dense_rows.len();
    let column_count = columns.values().filter(|count| **count >= 3).count();
    let compact: Vec<_> = dense_rows
        .iter()
        .flatten()
        .map(|index| lines[*index].text.trim())
        .collect();
    let numeric_cells = compact
        .iter()
        .filter(|text| {
            text.chars().any(|character| character.is_ascii_digit())
                && text.chars().all(|character| {
                    character.is_ascii_digit()
                        || character.is_whitespace()
                        || matches!(
                            character,
                            '.' | ',' | '%' | '(' | ')' | '/' | '$' | '-' | '\u{2013}' | '\u{2014}'
                        )
                })
        })
        .count();
    TableEvidence {
        rows: row_count,
        columns: column_count,
        numeric_cells,
        cells: compact.len(),
    }
}

/// Lines' words as pdf-inspector's text items, each carrying its line's slot (as its `mcid`).
fn inspector_items(lines: &[Line]) -> Vec<pdf_inspector::TextItem> {
    let height = lines.iter().map(|line| line.bbox[3]).filter(|y| y.is_finite()).fold(0.0, f64::max);
    let (mut items, origin) = legal_pdf_tables::line_items(lines, 1, height);
    for (item, (slot, _)) in items.iter_mut().zip(origin) {
        item.mcid = Some(slot as i64);
    }
    items
}

/// The page's columns as pdf-inspector finds them in its words (`detect_columns`, a projection of
/// the words across the page whose empty valleys are the gutters): two side by side, set apart at
/// the gutter between them, a pair whose narrower is under half the wider's width being a body and
/// its margin column (pdf-inspector's sidebar); more than two, a multi-column page.
pub(super) fn column_model(lines: &[Line]) -> ColumnModel {
    let regions = pdf_inspector::extractor::detect_columns(&inspector_items(lines), 1, false);
    let [left, right, ..] = regions.as_slice() else {
        return ColumnModel { kind: "single", split_x: 0.0, left_count: 0, right_count: 0 };
    };
    let split_x = f64::from(left.x_max + right.x_min) / 2.0;
    let boxed = lines.iter().filter(|line| has_valid_bbox(line) && !line.exclude_from_body);
    let left_count = boxed.clone().filter(|line| line_center_x(line) < split_x).count();
    let right_count = boxed.count() - left_count;
    let (left_width, right_width) = (left.x_max - left.x_min, right.x_max - right.x_min);
    let kind = if regions.len() > 2 {
        "multi_column"
    } else if left_width.min(right_width) >= 0.5 * left_width.max(right_width) {
        "two_column"
    } else {
        "margin_column"
    };
    ColumnModel { kind, split_x, left_count, right_count }
}

/// The lines in the order pdf-inspector reads their words (`group_into_lines_preserving_all_text`:
/// column by column, banded where the columns change down the page, a newspaper's columns read
/// through); a line it was not offered stays after the line it followed.
fn inspected_order(lines: &[Line]) -> Vec<Line> {
    let mut placed = vec![false; lines.len()];
    let mut order = Vec::with_capacity(lines.len());
    for line in pdf_inspector::extractor::group_into_lines_preserving_all_text(inspector_items(lines)) {
        for slot in line.items.iter().filter_map(|item| item.mcid).map(|slot| slot as usize) {
            if slot < lines.len() && !std::mem::replace(&mut placed[slot], true) {
                order.push(slot);
            }
        }
    }
    for slot in (0..lines.len()).filter(|&slot| !placed[slot]) {
        let at = (0..slot).rev().find_map(|prior| order.iter().position(|&placed| placed == prior)).map_or(0, |at| at + 1);
        order.insert(at, slot);
    }
    order.into_iter().map(|slot| lines[slot].clone()).collect()
}

/// English and French function words counted in some lines: (english, french).
pub(super) fn language_votes<'a>(lines: impl Iterator<Item = &'a Line>) -> (usize, usize) {
    const ENGLISH: [&str; 24] = [
        "the", "of", "and", "to", "in", "that", "is", "be", "for", "by", "with", "or", "any",
        "which", "shall", "not", "was", "this", "as", "it", "are", "from", "under", "were",
    ];
    const FRENCH: [&str; 28] = [
        "le", "la", "les", "des", "du", "de", "et", "à", "au", "aux", "que", "qui", "une", "un",
        "est", "pour", "dans", "par", "sur", "ne", "pas", "il", "ou", "sont", "cette", "ces", "l",
        "d",
    ];
    let (mut english, mut french) = (0, 0);
    for line in lines {
        for word in line
            .text
            .split(|character: char| !character.is_alphabetic())
        {
            let word = word.to_lowercase();
            english += usize::from(ENGLISH.contains(&word.as_str()));
            french += usize::from(FRENCH.contains(&word.as_str()));
        }
    }
    (english, french)
}

/// A column's language from its votes: Some(true) for French, Some(false) for English.
pub(super) fn column_language((english, french): (usize, usize), minimum: usize) -> Option<bool> {
    if french >= minimum && french >= english * 3 {
        Some(true)
    } else if english >= minimum && english >= french * 3 {
        Some(false)
    } else {
        None
    }
}

/// The lines printing the French of a parallel English and French text set column by
/// column, as the Supreme Court Reports and federal statutes are. Their headings,
/// numbers and contents repeat the English column's, which is the one read. A document
/// is parallel when some page holds an English column beside a French one; then the
/// French side of every page is the translation wherever its own words are French.
pub(super) fn translation_lines(pages: &[Page]) -> HashSet<String> {
    let body = |line: &&Line| {
        has_valid_bbox(line)
            && !line.exclude_from_body
            && !matches!(line.region_type.as_str(), "header" | "footer")
    };
    let models = pages
        .iter()
        .map(|page| {
            let model = column_model(&page.lines);
            (model.kind == "two_column").then_some(model.split_x)
        })
        .collect::<Vec<_>>();
    let side = |line: &Line, split: f64| {
        (line.bbox[2] <= split || line.bbox[0] >= split).then(|| line_center_x(line) >= split)
    };
    // Pages whose two columns are confidently one language each, and which side is French.
    let parallel = pages
        .iter()
        .zip(&models)
        .filter_map(|(page, split)| {
            let split = (*split)?;
            let votes = |right: bool| {
                language_votes(
                    page.lines
                        .iter()
                        .filter(body)
                        .filter(|line| side(line, split) == Some(right)),
                )
            };
            match (
                column_language(votes(false), 8),
                column_language(votes(true), 8),
            ) {
                (Some(false), Some(true)) => Some((page.index, split, true)),
                (Some(true), Some(false)) => Some((page.index, split, false)),
                _ => None,
            }
        })
        .collect::<Vec<_>>();
    if parallel.is_empty() {
        return HashSet::new();
    }
    let french_right = parallel.iter().filter(|(_, _, right)| *right).count() * 2 >= parallel.len();
    let split = median(parallel.iter().map(|(_, split, _)| *split).collect());
    let confirmed = parallel
        .iter()
        .filter(|(_, _, right)| *right == french_right)
        .map(|(index, _, _)| *index)
        .collect::<HashSet<_>>();
    let mut translation = HashSet::new();
    for (page, page_split) in pages.iter().zip(&models) {
        let split = page_split.unwrap_or(split);
        let mut french = page
            .lines
            .iter()
            .filter(body)
            .filter(|line| side(line, split) == Some(french_right))
            .collect::<Vec<_>>();
        // What stands apart above the English column, a cover page's file number or a
        // memo's title, is no part of the translation beside it.
        let english_top = page
            .lines
            .iter()
            .filter(body)
            .filter(|line| (line_center_x(line) >= split) != french_right)
            .map(|line| line.bbox[1])
            .min_by(f64::total_cmp);
        if let Some(top) = english_top {
            french.sort_by(|a, b| a.bbox[1].total_cmp(&b.bbox[1]));
            let apart = french
                .windows(2)
                .filter(|pair| {
                    let height = pair[0].bbox[3] - pair[0].bbox[1];
                    pair[0].bbox[3] <= top && pair[1].bbox[1] - pair[0].bbox[3] >= height * 2.0
                })
                .map(|pair| pair[0].bbox[3])
                .max_by(f64::total_cmp);
            if let Some(cut) = apart {
                french.retain(|line| line.bbox[3] > cut);
            }
        }
        if confirmed.contains(&page.index)
            || column_language(language_votes(french.iter().copied()), 3) == Some(true)
        {
            translation.extend(french.into_iter().map(|line| line.id.clone()));
        }
    }
    translation
}

pub(super) fn margin_note_model(
    lines: &[Line],
    labels: &[usize],
    page_width: f64,
    body_size: f64,
    minimum_labels: usize,
) -> Option<ColumnModel> {
    if labels.is_empty() || page_width <= 0.0 || body_size <= 0.0 {
        return None;
    }
    [false, true]
        .into_iter()
        .filter_map(|right| {
            let lane: Vec<_> = labels
                .iter()
                .copied()
                .filter(|index| (line_center_x(&lines[*index]) >= page_width / 2.0) == right)
                .collect();
            if lane.len() < minimum_labels {
                return None;
            }
            let label_set: HashSet<_> = lane.iter().copied().collect();
            let mut note_left = page_width;
            let mut note_right: f64 = 0.0;
            let mut note_top = f64::INFINITY;
            let mut note_bottom: f64 = 0.0;
            for index in &lane {
                let label = &lines[*index];
                note_left = note_left.min(label.bbox[0]);
                note_right = note_right.max(label.bbox[2]);
                note_top = note_top.min(label.bbox[1]);
                note_bottom = note_bottom.max(label.bbox[3]);
                if standalone_note_label(label) {
                    if let Some(body) = aligned_note_body_index(lines, *index) {
                        note_left = note_left.min(lines[body].bbox[0]);
                        note_right = note_right.max(lines[body].bbox[2]);
                        note_top = note_top.min(lines[body].bbox[1]);
                        note_bottom = note_bottom.max(lines[body].bbox[3]);
                    }
                }
            }
            let prose: Vec<_> = lines
                .iter()
                .enumerate()
                .filter(|(index, line)| {
                    !label_set.contains(index)
                        && has_valid_bbox(line)
                        && !line.exclude_from_body
                        && !matches!(line.region_type.as_str(), "header" | "footer")
                        && line_font_size(line) >= body_size * 0.90
                        && line_width(line) >= page_width * 0.25
                        && line.bbox[3] >= note_top
                        && line.bbox[1] <= note_bottom
                })
                .map(|(_, line)| line)
                .collect();
            if prose.len() < 3 {
                return None;
            }
            let body_left = prose
                .iter()
                .map(|line| line.bbox[0])
                .min_by(f64::total_cmp)?;
            let body_right = prose
                .iter()
                .map(|line| line.bbox[2])
                .max_by(f64::total_cmp)?;
            let gap = (body_size * 1.5).max(page_width * 0.02);
            let split_x = if note_right + gap <= body_left {
                (note_right + body_left) / 2.0
            } else if body_right + gap <= note_left {
                (body_right + note_left) / 2.0
            } else {
                return None;
            };
            Some((
                lane.len(),
                ColumnModel {
                    kind: "margin_column",
                    split_x,
                    left_count: if right { prose.len() } else { lane.len() },
                    right_count: if right { lane.len() } else { prose.len() },
                },
            ))
        })
        .max_by_key(|(count, _)| *count)
        .map(|(_, model)| model)
}

fn note_column_model(lines: &[Line]) -> ColumnModel {
    let mut model = column_model(lines);
    if model.kind != "two_column" {
        return model;
    }
    let crossing_prose = lines
        .iter()
        .filter(|line| {
            has_valid_bbox(line)
                && line
                    .text
                    .chars()
                    .filter(|character| !character.is_whitespace())
                    .count()
                    >= 8
                && line.bbox[0] < model.split_x
                && line.bbox[2] > model.split_x
        })
        .count();
    if crossing_prose > 0 {
        model.kind = "margin_column";
    }
    model
}

fn geometry_order(lines: &mut [Line]) {
    lines.sort_by(|left, right| {
        line_center_y(left)
            .total_cmp(&line_center_y(right))
            .then(line_center_x(left).total_cmp(&line_center_x(right)))
            .then(left.bbox[0].total_cmp(&right.bbox[0]))
            .then(left.id.cmp(&right.id))
    });
}

fn column_switches(lines: &[Line], model: ColumnModel, page_width: f64) -> usize {
    let mut previous = None;
    let mut switches = 0;
    for line in lines {
        if !has_valid_bbox(line) || line_width(line) / page_width > 0.55 {
            continue;
        }
        let column = line_center_x(line) >= model.split_x;
        if previous.is_some_and(|prior| prior != column) {
            switches += 1;
        }
        previous = Some(column);
    }
    switches
}

fn median_column_run(lines: &[Line], model: ColumnModel, page_width: f64) -> usize {
    let mut previous = None;
    let mut current = 0;
    let mut runs = Vec::new();
    for line in lines {
        if !has_valid_bbox(line) || line_width(line) / page_width > 0.55 {
            continue;
        }
        let column = line_center_x(line) >= model.split_x;
        if previous.is_none_or(|prior| prior == column) {
            current += 1;
        } else {
            runs.push(current);
            current = 1;
        }
        previous = Some(column);
    }
    if current > 0 {
        runs.push(current);
    }
    runs.sort_unstable();
    runs.get(runs.len() / 2).copied().unwrap_or(0)
}

fn hyphen_join_score(lines: &[Line]) -> (usize, usize) {
    let mut candidates = 0;
    let mut satisfied = 0;
    for (index, line) in lines.iter().enumerate() {
        let text = line.text.trim_end();
        let Some(last) = text
            .chars()
            .last()
            .filter(|character| matches!(character, '-' | '\u{00ad}' | '\u{00ac}'))
        else {
            continue;
        };
        let prefix = &text[..text.len() - last.len_utf8()];
        if prefix
            .chars()
            .rev()
            .take_while(|character| character.is_ascii_alphabetic())
            .count()
            < 2
        {
            continue;
        }
        candidates += 1;
        if lines.get(index + 1).is_some_and(|next| {
            next.text
                .trim_start()
                .chars()
                .take_while(|character| character.is_ascii_alphabetic())
                .count()
                >= 2
        }) {
            satisfied += 1;
        }
    }
    (candidates, satisfied)
}

fn y_regressions(lines: &[Line]) -> usize {
    let boxed: Vec<&Line> = lines.iter().filter(|line| has_valid_bbox(line)).collect();
    let tolerance = median(boxed
        .iter()
        .map(|line| (line.bbox[3] - line.bbox[1]).max(0.0))
        .collect());
    boxed
        .windows(2)
        .filter(|pair| line_center_y(pair[1]) < line_center_y(pair[0]) - tolerance)
        .count()
}

pub(super) fn arbitrate_body_order(
    lines: &mut [Line],
    page_width: f64,
    page_height: f64,
) -> OrderDecision {
    let boxed = lines.iter().filter(|line| has_valid_bbox(line)).count();
    if lines.len() < 8 || boxed * 5 < lines.len() * 4 || page_width <= 0.0 || page_height <= 0.0 {
        return OrderDecision::keep("insufficient_geometry");
    }
    if has_table_caption(lines) {
        return OrderDecision::keep("table_grid");
    }
    let model = column_model(lines);
    if model.kind == "two_column" {
        let source_switches = column_switches(lines, model, page_width);
        let source_run = median_column_run(lines, model, page_width);
        let source_hyphens = hyphen_join_score(lines);
        // The challenger: the lines in pdf-inspector's reading order; the source order stands unless
        // it fixes the columns' interleaving and breaks no hyphenated word.
        let challenger = inspected_order(lines);
        let challenger_switches = column_switches(&challenger, model, page_width);
        let challenger_hyphens = hyphen_join_score(&challenger);
        let minimum_switches = if source_hyphens.0 > 0 { 3 } else { 5 }
            .max(((model.left_count + model.right_count) as f64 * 0.10) as usize);
        if source_switches >= minimum_switches
            && source_run <= 6
            && challenger_switches <= 2
            && source_switches.saturating_sub(challenger_switches) >= 2
            && challenger_hyphens.1 >= source_hyphens.1
            && challenger_hyphens.0.saturating_sub(challenger_hyphens.1)
                <= source_hyphens.0.saturating_sub(source_hyphens.1)
        {
            lines.clone_from_slice(&challenger);
            return OrderDecision {
                repair: OrderRepair::Column,
                source_switches,
                strategy: "column-geometry",
                reason: "column_interleave_repair",
            };
        }
        return OrderDecision {
            repair: OrderRepair::None,
            source_switches,
            strategy: "kraken-native",
            reason: "two_column_kraken_coherent",
        };
    }
    if model.kind == "single" {
        let source_regressions = y_regressions(lines);
        let source_hyphens = hyphen_join_score(lines);
        let mut challenger = lines.to_vec();
        geometry_order(&mut challenger);
        let challenger_regressions = y_regressions(&challenger);
        let challenger_hyphens = hyphen_join_score(&challenger);
        let threshold = 3.max((boxed as f64 * 0.08) as usize);
        if source_regressions >= threshold
            && challenger_regressions <= source_regressions / 3
            && challenger_hyphens.1 > source_hyphens.1
            && challenger_hyphens.0.saturating_sub(challenger_hyphens.1)
                < source_hyphens.0.saturating_sub(source_hyphens.1)
        {
            lines.clone_from_slice(&challenger);
            return OrderDecision {
                repair: OrderRepair::Geometry,
                source_switches: 0,
                strategy: "geometry",
                reason: "kraken_order_scrambled",
            };
        }
    }
    OrderDecision::keep(if model.kind == "single" {
        "single_column_kraken_coherent"
    } else {
        "non_two_column"
    })
}

pub(super) fn repair_drop_caps(lines: &mut Vec<Line>) {
    let body_size = median(lines
        .iter()
        .map(line_font_size)
        .filter(|size| (4.0..=24.0).contains(size))
        .collect());
    if body_size <= 0.0 {
        return;
    }
    let mut moves = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let text = line.text.trim();
        if text.chars().count() != 1
            || !text.chars().all(char::is_alphabetic)
            || line_font_size(line) < body_size * 1.8
        {
            continue;
        }
        let target = lines
            .iter()
            .enumerate()
            .filter(|(other, candidate)| {
                *other != index
                    && line_font_size(candidate) <= body_size * 1.4
                    && candidate.bbox[0] >= line.bbox[2] - body_size
                    && candidate.bbox[0] - line.bbox[2] <= body_size * 3.0
                    && candidate.bbox[1] < line.bbox[3]
                    && candidate.bbox[3] > line.bbox[1]
            })
            .min_by(|(_, left), (_, right)| {
                left.bbox[1]
                    .total_cmp(&right.bbox[1])
                    .then(left.bbox[0].total_cmp(&right.bbox[0]))
            })
            .map(|(target, _)| target);
        if let Some(target) = target.filter(|target| *target < index) {
            moves.push((index, target));
        }
    }
    for (index, target) in moves.into_iter().rev() {
        let line = lines.remove(index);
        lines.insert(target, line);
    }
}

pub(super) fn band_geometry_top(line: &Line) -> f64 {
    line.words
        .iter()
        .map(|word| word.bbox[1])
        .min_by(f64::total_cmp)
        .unwrap_or(line.bbox[1])
}

pub(super) fn band_geometry_order(left: &Line, right: &Line) -> std::cmp::Ordering {
    band_geometry_top(left)
        .total_cmp(&band_geometry_top(right))
        .then(left.bbox[0].total_cmp(&right.bbox[0]))
}

pub(super) fn standalone_note_label(line: &Line) -> bool {
    let text = line.text.trim();
    let length = text.chars().count();
    ((1..=MAX_SYMBOL_LABEL_LEN).contains(&length) && text.chars().all(is_note_symbol))
        || ((1..=4).contains(&length) && text.chars().all(|character| character.is_ascii_digit()))
}

pub(super) fn aligned_note_body(label: &Line, body: &Line) -> bool {
    if standalone_note_label(body) || body.bbox[0] < label.bbox[0] {
        return false;
    }
    let overlap = label.bbox[3].min(body.bbox[3]) - label.bbox[1].max(body.bbox[1]);
    let minimum_height = (label.bbox[3] - label.bbox[1]).min(body.bbox[3] - body.bbox[1]);
    overlap > 0.0 && minimum_height > 0.0 && overlap / minimum_height >= 0.5
}

pub(super) fn aligned_note_body_index(lines: &[Line], label_index: usize) -> Option<usize> {
    let label = &lines[label_index];
    lines
        .iter()
        .enumerate()
        .filter(|(_, body)| aligned_note_body(label, body))
        .min_by(|(_, left), (_, right)| {
            (left.bbox[0] - label.bbox[2])
                .max(0.0)
                .total_cmp(&(right.bbox[0] - label.bbox[2]).max(0.0))
                .then(
                    (left.bbox[1] - label.bbox[1])
                        .abs()
                        .total_cmp(&(right.bbox[1] - label.bbox[1]).abs()),
                )
        })
        .map(|(index, _)| index)
}

pub(super) fn order_note_lines(lines: &mut Vec<Line>) {
    let mut tops: Vec<f64> = lines.iter().map(band_geometry_top).collect();
    for (label_index, label) in lines.iter().enumerate() {
        if !standalone_note_label(label) {
            continue;
        }
        if let Some(body_index) = aligned_note_body_index(lines, label_index) {
            tops[label_index] = tops[body_index];
        }
    }
    let columns = note_column_model(lines);
    let mut keyed: Vec<_> = std::mem::take(lines).into_iter().zip(tops).collect();
    keyed.sort_by(|(left, left_top), (right, right_top)| {
        let left_column =
            usize::from(columns.kind == "two_column" && line_center_x(left) >= columns.split_x);
        let right_column =
            usize::from(columns.kind == "two_column" && line_center_x(right) >= columns.split_x);
        left_column.cmp(&right_column).then(
            left_top
                .total_cmp(right_top)
                .then(left.bbox[0].total_cmp(&right.bbox[0])),
        )
    });
    lines.extend(keyed.into_iter().map(|(line, _)| line));
}

pub(super) fn weave_note_columns(
    mut body: Vec<Line>,
    note: Vec<Line>,
    page_width: f64,
) -> Vec<Line> {
    let note_columns = note_column_model(&note);
    let body_model = column_model(&body);
    let split_x = if note_columns.kind == "two_column" {
        note_columns.split_x
    } else if body_model.kind == "two_column" {
        body_model.split_x
    } else {
        body.extend(note);
        return body;
    };
    let note_sides = note
        .iter()
        .filter(|line| line_width(line) / page_width <= 0.55)
        .fold([0_usize; 2], |mut counts, line| {
            counts[usize::from(line_center_x(line) >= split_x)] += 1;
            counts
        });
    if note_sides.contains(&0) {
        body.extend(note);
        return body;
    }
    let (left_notes, right_notes): (Vec<_>, Vec<_>) = note
        .into_iter()
        .partition(|line| line_center_x(line) < split_x);
    let body_columns = body
        .iter()
        .filter(|line| line_width(line) / page_width <= 0.55)
        .fold([0_usize; 2], |mut counts, line| {
            counts[usize::from(line_center_x(line) >= split_x)] += 1;
            counts
        });
    if left_notes.is_empty()
        || right_notes.is_empty()
        || body_columns.iter().any(|count| *count < 3)
    {
        body.extend(left_notes);
        body.extend(right_notes);
        return body;
    }
    let insert_at = body
        .iter()
        .rposition(|line| line_width(line) / page_width <= 0.55 && line_center_x(line) < split_x)
        .map_or(0, |index| index + 1);
    let right_body = body.split_off(insert_at);
    body.extend(left_notes);
    body.extend(right_body);
    body.extend(right_notes);
    body
}

/// Restore physical header and footer bands after independent readers share a page.
pub(super) fn order_furniture(page: &mut Page) {
    let band = |line: &Line| match line.region_type.as_str() {
        "header" => 0,
        "footer" => 2,
        _ => 1,
    };
    page.lines.sort_by(|a, b| {
        band(a).cmp(&band(b)).then_with(|| {
            if band(a) == 1 {
                std::cmp::Ordering::Equal
            } else {
                band_geometry_order(a, b)
            }
        })
    });
    for (slot, line) in page.lines.iter_mut().enumerate() {
        line.reading_order = slot + 1;
    }
}

/// contents: the page is a contents list's (contents_lists).
pub(super) fn order_page(
    page: &mut Page,
    table_page: bool,
    contents: bool,
    table_notes: &HashSet<usize>,
) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    let mut header = Vec::new();
    let mut body = Vec::new();
    let mut note = Vec::new();
    let mut footer = Vec::new();
    for (index, line) in std::mem::take(&mut page.lines).into_iter().enumerate() {
        match line.region_type.as_str() {
            "header" => header.push(line),
            "footnote" if table_notes.contains(&index) => body.push(line),
            "footnote" => note.push(line),
            "footer" => footer.push(line),
            _ => body.push(line),
        }
    }
    repair_drop_caps(&mut body);
    let contents = table_page && contents && !has_table_caption(&body);
    let decision = if contents && y_regressions(&body) > 0 {
        geometry_order(&mut body);
        OrderDecision {
            repair: OrderRepair::Geometry,
            source_switches: 0,
            strategy: "table-row-geometry",
            reason: "table_source_order_scrambled",
        }
    } else if table_page {
        OrderDecision::keep("table_grid")
    } else {
        arbitrate_body_order(&mut body, page.width, page.height)
    };
    if decision.repair != OrderRepair::None {
        let mut diagnostic = Diagnostic::info(
            "COLUMN_ORDER_REPAIRED",
            format!(
                "Extraction order replaced by {}: {}.",
                decision.strategy, decision.reason
            ),
            Some(page.index),
        );
        diagnostic.line_ids = body.iter().take(20).map(|line| line.id.clone()).collect();
        diagnostics.push(diagnostic);
    } else if decision.source_switches > 2 {
        let mut diagnostic = Diagnostic::warning(
            "COLUMN_ORDER_UNCERTAIN",
            format!(
                "Two-column page keeps an order that crosses columns {} times ({}).",
                decision.source_switches, decision.reason
            ),
            Some(page.index),
        );
        diagnostic.line_ids = body.iter().take(20).map(|line| line.id.clone()).collect();
        diagnostics.push(diagnostic);
    }
    order_note_lines(&mut note);
    let mut ordered = header;
    ordered.extend(weave_note_columns(body, note, page.width));
    ordered.extend(footer);
    page.lines = ordered;
    order_furniture(page);
    diagnostics
}

/// Whether a transcript's line opens a unit of its own: a speaker's turn, or a line in capitals
/// (an examination's "BY MR. JONES:", "EXHIBIT NO. 3", "--- Upon resuming").
fn opens_turn(line: &Line) -> bool {
    let text = line.text.trim_start();
    legal_structure_model::transcript::speaker_label(text).is_some()
        || text.starts_with("--")
        || (text.chars().any(char::is_alphabetic) && !text.chars().any(char::is_lowercase))
}

/// Each page's regions: its lines in order, a region where their kind and extraction block run on.
/// On a transcript's page (one with line numbers) a line number stands apart and parts no region,
/// and its body reads by turns: a turn runs from the line that opens it to the next that opens one.
pub(super) fn build_regions(pages: &mut [Page]) {
    for page in pages {
        let transcript = page.lines.iter().any(|line| line.region_type == "line_label");
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut labels: Vec<Vec<usize>> = Vec::new();
        for index in 0..page.lines.len() {
            let line = &page.lines[index];
            if transcript && line.region_type == "line_label" {
                labels.push(vec![index]);
                continue;
            }
            if groups.last().is_some_and(|group| {
                let prior = &page.lines[*group.last().expect("non-empty group")];
                prior.region_type == line.region_type
                    && if transcript && line.region_type == "body" { !opens_turn(line) } else { prior.block_index == line.block_index }
            }) {
                groups.last_mut().expect("group exists").push(index);
            } else {
                groups.push(vec![index]);
            }
        }
        groups.extend(labels);
        groups.sort_by_key(|group| group[0]);
        page.regions.clear();
        for (region_index, indexes) in groups.into_iter().enumerate() {
            let id = format!("p{:04}-r{:04}", page.number, region_index + 1);
            let kind = page.lines[indexes[0]].region_type.clone();
            let line_ids = indexes
                .iter()
                .map(|&index| page.lines[index].id.clone())
                .collect();
            let bbox = union_bbox(indexes.iter().map(|&index| page.lines[index].bbox));
            let reading_order = indexes
                .iter()
                .map(|&index| page.lines[index].reading_order)
                .min()
                .unwrap_or(0);
            for &index in &indexes {
                page.lines[index].region_id.clone_from(&id);
            }
            page.regions.push(Region {
                id,
                page_index: page.index,
                kind,
                line_ids,
                bbox,
                reading_order,
            });
        }
    }
}

fn marker_alone() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^\s*(?:\(?(?:\d{1,3}(?:\.\d{1,3}){0,3}|[A-Za-z]{1,2}|[ivxlc]{1,6}|[IVXLC]{1,6})[.)]|\(\s*(?:\d{1,3}|[A-Za-z]{1,2}|[ivxlc]{1,6}|[IVXLC]{1,6})\s*\)|[•·▪◦●○■□➢✓\-–—*\u{f0b7}\u{f0a7}\u{f0d8}])\s*$")
            .expect("detached marker regex")
    })
}

/// A clause's decimal number standing alone, as contracts print it in a column of its own
/// ("3.05", "*1.04" for an amended clause, "10.01 (a)" with its first item).
fn decimal_alone() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^\s*\*?\d{1,3}(?:\.\d{1,3}){1,3}(?:\s*\(\s*(?:\d{1,3}|[A-Za-z]{1,2}|[ivxlc]{1,6})\s*\))?\s*$")
            .expect("decimal marker regex")
    })
}

/// A numbering marker printed on a line of its own ("11.", "(a)", "•", "3.05") belongs to the
/// line that carries its text: the nearest line to its right on the same row, within six line
/// heights. A bare decimal joins only words (a letter, bracket or quote first), never a table's
/// next figure. The marker joins the front of that line, which keeps its identity; the marker's
/// line goes. Returns the number of joins.
pub(super) fn join_detached_markers(page: &mut Page) -> usize {
    let height = |line: &Line| (line.bbox[3] - line.bbox[1]).max(0.1);
    let alone = |line: &Line| marker_alone().is_match(&line.text) || decimal_alone().is_match(&line.text);
    let mut joins: Vec<(usize, usize)> = Vec::new(); // (marker, body)
    let mut taken: HashSet<usize> = HashSet::new();
    for (m, marker) in page.lines.iter().enumerate() {
        let decimal = decimal_alone().is_match(&marker.text);
        if !has_valid_bbox(marker) || marker.text.chars().count() > 12 || !(decimal || marker_alone().is_match(&marker.text)) {
            continue;
        }
        let h = height(marker);
        let mut best: Option<(f64, usize)> = None;
        for (b, body) in page.lines.iter().enumerate() {
            if b == m || taken.contains(&b) || !has_valid_bbox(body) || alone(body) {
                continue;
            }
            if decimal && !body.text.trim_start().starts_with(|c: char| c.is_alphabetic() || matches!(c, '(' | '"' | '“' | '\'' | '‘')) {
                continue;
            }
            let overlap = body.bbox[3].min(marker.bbox[3]) - body.bbox[1].max(marker.bbox[1]);
            let gap = body.bbox[0] - marker.bbox[2];
            if overlap >= 0.5 * h.min(height(body)) && gap >= -0.25 * h && gap <= 6.0 * h {
                if best.is_none_or(|(g, _)| gap < g) {
                    best = Some((gap, b));
                }
            }
        }
        if let Some((_, b)) = best {
            taken.insert(b);
            joins.push((m, b));
        }
    }
    if joins.is_empty() {
        return 0;
    }
    let mut removed: HashSet<usize> = HashSet::new();
    let mut renamed: HashMap<String, String> = HashMap::new();
    for &(m, b) in &joins {
        let marker = page.lines[m].clone();
        let body = &mut page.lines[b];
        let prefix = marker.text.trim().to_owned();
        let shift = prefix.chars().count() + 1;
        let mut text = prefix.clone();
        text.push(' ');
        text.push_str(&body.text);
        body.text = text;
        for span in &mut body.spans {
            span.start += shift;
            span.end += shift;
        }
        for word in &mut body.words {
            word.start += shift;
            word.end += shift;
        }
        let lead = marker.text.chars().count() - marker.text.trim_start().chars().count();
        let mut marker_spans = marker.spans;
        for span in &mut marker_spans {
            span.start = span.start.saturating_sub(lead).min(prefix.chars().count());
            span.end = span.end.saturating_sub(lead).min(prefix.chars().count());
        }
        let mut marker_words = marker.words;
        for word in &mut marker_words {
            word.start = word.start.saturating_sub(lead).min(prefix.chars().count());
            word.end = word.end.saturating_sub(lead).min(prefix.chars().count());
        }
        marker_spans.append(&mut body.spans);
        body.spans = marker_spans;
        marker_words.append(&mut body.words);
        body.words = marker_words;
        body.bbox = union_bbox([marker.bbox, body.bbox]);
        removed.insert(m);
        renamed.insert(marker.id, body.id.clone());
    }
    for region in &mut page.regions {
        region.line_ids.retain(|id| !renamed.contains_key(id));
    }
    if let Some(id) = page.printed_label_line_id.as_ref().and_then(|id| renamed.get(id)) {
        page.printed_label_line_id = Some(id.clone());
    }
    let mut index = 0usize;
    page.lines.retain(|_| {
        let keep = !removed.contains(&index);
        index += 1;
        keep
    });
    removed.len()
}
