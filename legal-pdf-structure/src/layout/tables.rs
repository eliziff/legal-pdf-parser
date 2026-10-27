//! Tables found in a page's white space: rows of aligned columns pdf-inspector's grid detector
//! reads, kept where the page's evidence says they are a table (`TableReadings`).

use super::*;

/// A page's grid of short lines in rows and columns, figures in a fifth of its cells: a plain
/// table a recognized page's line boxes measure. Tables themselves are pdf-inspector's
/// (legal_pdf_tables).
#[derive(Default)]
pub(crate) struct TableEvidence {
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

pub(crate) fn has_table_caption(lines: &[Line]) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    let regex =
        RE.get_or_init(|| Regex::new(r"(?i)^(?:table|tableau)\s+(?:\d+|[ivxlcdm]+)\b").unwrap());
    lines.iter().any(|line| regex.is_match(line.text.trim()))
}

/// The layout readings this crate lends the table detector.
pub(crate) struct TableReadings;

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

pub(crate) fn table_evidence(lines: &[Line]) -> TableEvidence {
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
