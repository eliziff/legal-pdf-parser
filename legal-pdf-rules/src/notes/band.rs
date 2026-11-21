//! Where a page's note separator stands, and how far a line may stand from it.

use legal_pdf_core::model::{Line, Page};

/// How far from a page's note separator a line still stands at it.
pub fn separator_tolerance(page: &Page) -> f64 {
    (page.height * 0.004).max(1.0)
}

/// The page's note separator rule, unless it runs through the band its table cells (`cells`) fill.
pub fn separator_beside_tables<'a>(
    page: &Page,
    separator: Option<f64>,
    cells: impl Iterator<Item = &'a Line>,
) -> Option<f64> {
    let tolerance = separator_tolerance(page);
    let table_band = cells.fold(None, |band, line| {
        Some(
            band.map_or((line.bbox[1], line.bbox[3]), |(top, bottom): (f64, f64)| {
                (top.min(line.bbox[1]), bottom.max(line.bbox[3]))
            }),
        )
    });
    separator.filter(|cut| {
        table_band.is_none_or(|(top, bottom)| *cut < top - tolerance || *cut > bottom + tolerance)
    })
}
