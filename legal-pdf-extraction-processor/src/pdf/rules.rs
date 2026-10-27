//! Ruled lines and rectangles: a note separator, words underlined, and the regions rectangles box.

use super::*;

/// The page's painted rules as regions of kind `rule`, each a horizontal or vertical line
/// (`bbox` from one end to the other) in page coordinates, collinear pieces joined.
pub(crate) fn rule_regions(
    geometry: PageGeometry,
    page: &Page,
    rules: &[PdfLine],
) -> Vec<legal_pdf_core::model::Region> {
    const JOIN: f64 = 1.0;
    // (vertical, position across, start along, end along)
    let mut pieces: Vec<(bool, f64, f64, f64)> = rules
        .iter()
        .filter_map(|rule| {
            let (x1, y1) = transform_point(geometry, f64::from(rule.x1), f64::from(rule.y1));
            let (x2, y2) = transform_point(geometry, f64::from(rule.x2), f64::from(rule.y2));
            let (y1, y2) = (geometry.height - y1, geometry.height - y2);
            let vertical = (x1 - x2).abs() < (y1 - y2).abs();
            let piece = if vertical {
                (true, (x1 + x2) / 2.0, y1.min(y2), y1.max(y2))
            } else {
                (false, (y1 + y2) / 2.0, x1.min(x2), x1.max(x2))
            };
            [piece.1, piece.2, piece.3]
                .iter()
                .all(|value| value.is_finite())
                .then_some(piece)
        })
        .collect();
    pieces.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then(left.1.total_cmp(&right.1))
            .then(left.2.total_cmp(&right.2))
    });
    let mut joined: Vec<(bool, f64, f64, f64)> = Vec::new();
    for (vertical, across, start, end) in pieces {
        let mut group = joined
            .iter_mut()
            .rev()
            .take_while(|prior| prior.0 == vertical && across - prior.1 <= JOIN);
        if let Some(prior) = group.find(|prior| start <= prior.3 + JOIN && prior.2 <= end + JOIN) {
            prior.2 = prior.2.min(start);
            prior.3 = prior.3.max(end);
        } else {
            joined.push((vertical, across, start, end));
        }
    }
    joined
        .into_iter()
        .filter(|(_, _, start, end)| end - start > JOIN)
        .enumerate()
        .map(|(index, (vertical, across, start, end))| {
            let bbox = if vertical {
                [across, start, across, end]
            } else {
                [start, across, end, across]
            };
            legal_pdf_core::model::Region {
                id: format!("p{:04}-rule{:04}", page.number, index + 1),
                page_index: page.index,
                kind: "rule".to_owned(),
                line_ids: Vec::new(),
                bbox: bbox.map(round3),
                reading_order: 0,
            }
        })
        .collect()
}

/// Mark the words a horizontal rule on their page underlines: painted in the PDF, or drawn on a
/// recognized page's image. A typeset word pdf-inspector read as underlined with its text stays
/// underlined; the rule finds the underlines it misses.
pub fn mark_rule_underlines(page: &mut Page) {
    let rules = page
        .regions
        .iter()
        .filter(|region| region.kind == "rule")
        .map(|region| region.bbox)
        .filter(|[x0, y0, x1, y1]| (y1 - y0).abs() <= 2.0 && (x1 - x0).abs() > 2.0)
        .collect::<Vec<_>>();
    if rules.is_empty() {
        return;
    }
    for line in &mut page.lines {
        let height = (line.bbox[3] - line.bbox[1]).max(1.0);
        for word in &mut line.words {
            word.underlined |= rules.iter().any(|rule| {
                underlines(word.bbox, height, rule[0].min(rule[2]), rule[0].max(rule[2]), (rule[1] + rule[3]) / 2.0)
            });
        }
    }
}

/// Whether a horizontal stroke from `start` to `end` at height `at` underlines a word box: set
/// from a third of its line's height above the word's bottom to half below, across most of it.
pub(crate) fn underlines(word: [f64; 4], line_height: f64, start: f64, end: f64, at: f64) -> bool {
    let width = (word[2] - word[0]).max(1.0);
    at >= word[3] - 0.3 * line_height
        && at <= word[3] + 0.5 * line_height
        && end.min(word[2]) - start.max(word[0]) >= 0.6 * width
}

/// Add each page's painted rectangles (`re`: filled cells and shading, boxes drawn as borders) as
/// regions of kind `rect` after its rules, `bbox` the rectangle in page coordinates. A rectangle
/// covering the whole page is its background, not a drawing on it, and a rectangle painted again
/// where it already stands adds nothing: neither is kept.
pub fn add_rect_regions(pages: &mut [Page], geometries: &PageGeometryMap, rects: &[PdfRect]) {
    let mut by_page = HashMap::<u32, Vec<&PdfRect>>::new();
    for rect in rects {
        by_page.entry(rect.page).or_default().push(rect);
    }
    for page in pages {
        let (Some(&(_, geometry)), Some(rects)) =
            (geometries.get(&page.number), by_page.get(&page.number))
        else {
            continue;
        };
        let regions: Vec<_> = rects
            .iter()
            .filter_map(|rect| {
                let (x1, y1) = transform_point(geometry, f64::from(rect.x), f64::from(rect.y));
                let (x2, y2) = transform_point(
                    geometry,
                    f64::from(rect.x + rect.width),
                    f64::from(rect.y + rect.height),
                );
                let (y1, y2) = (geometry.height - y1, geometry.height - y2);
                let bbox = [x1.min(x2), y1.min(y2), x1.max(x2), y1.max(y2)].map(round3);
                let background = bbox[2] - bbox[0] >= page.width * 0.95 && bbox[3] - bbox[1] >= page.height * 0.95;
                (bbox.iter().all(|value| value.is_finite()) && !background).then_some(bbox)
            })
            .fold(Vec::<[f64; 4]>::new(), |mut kept, bbox| {
                if !kept.contains(&bbox) {
                    kept.push(bbox);
                }
                kept
            })
            .into_iter()
            .enumerate()
            .map(|(index, bbox)| legal_pdf_core::model::Region {
                id: format!("p{:04}-rect{:04}", page.number, index + 1),
                page_index: page.index,
                kind: "rect".to_owned(),
                line_ids: Vec::new(),
                bbox,
                reading_order: 0,
            })
            .collect();
        page.regions.extend(regions);
    }
}

pub(crate) fn separator_y(geometry: PageGeometry, lines: &[Line], rules: &[PdfLine]) -> Option<f64> {
    let candidates: Vec<(f64, f64)> = rules
        .iter()
        .map(|rule| {
            let first = transform_point(geometry, f64::from(rule.x1), f64::from(rule.y1));
            let second = transform_point(geometry, f64::from(rule.x2), f64::from(rule.y2));
            (
                first.0.min(second.0),
                first.0.max(second.0),
                geometry.height - (first.1 + second.1) / 2.0,
            )
        })
        .filter(|(left, right, y)| {
            right - left >= geometry.width * 0.20
                && geometry.height * 0.30 <= *y
                && *y <= geometry.height * 0.98
                // Underlines and strikeouts cross a text line's glyph band;
                // they do not separate the body from a footnote region.
                && !lines.iter().any(|line| {
                    line.bbox[1] <= *y && *y <= line.bbox[3]
                        && line.bbox[0] < *right && *left < line.bbox[2]
                })
        })
        .map(|(left, right, y)| (right - left, y))
        .collect();
    if candidates.is_empty() {
        return None;
    }
    // Prefer prose-length lines: a dense table must not redefine the body font.
    let mut sizes: Vec<_> = lines
        .iter()
        .filter(|line| line.bbox[1] < geometry.height * 0.80 && line.text.chars().count() >= 40)
        .map(model_line_font_size)
        .filter(|size| *size > 0.0)
        .collect();
    if sizes.is_empty() {
        sizes = lines
            .iter()
            .filter(|line| line.bbox[1] < geometry.height * 0.80)
            .map(model_line_font_size)
            .filter(|size| *size > 0.0)
            .collect();
    }
    let body_size = median(sizes);
    let labels: Vec<_> = lines
        .iter()
        .filter(|line| {
            (begins_with_note_label(&line.text)
                || line
                    .spans
                    .first()
                    .is_some_and(|span| span.superscript && begins_with_note_label(&span.text)))
                && body_size > 0.0
                && (model_line_font_size(line) <= body_size * 0.90
                    || line
                        .spans
                        .first()
                        .is_some_and(|span| span.size <= body_size * 0.78))
                // A detached label needs a body to its right; a footer page
                // number alone cannot establish a note region.
                && (line.text.chars().any(char::is_alphabetic)
                    || lines.iter().any(|body| {
                        body.bbox[0] >= line.bbox[2]
                            && body.bbox[1] < line.bbox[3]
                            && line.bbox[1] < body.bbox[3]
                            && body.text.chars().any(char::is_alphabetic)
                    }))
        })
        .map(|line| line.bbox[1])
        .collect();
    candidates
        .iter()
        .filter(|(_, y)| {
            labels.iter().any(|label| {
                *y <= *label + (geometry.height * 0.004).max(1.0) && *label - *y <= body_size * 3.0
            })
        })
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(_, y)| *y)
}
