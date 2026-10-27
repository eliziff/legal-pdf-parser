//! A page's reading order: the body in the order its columns read (arbitrated between the
//! source's and the geometry's), a drop cap joined to its line, furniture last, a transcript's turns
//! kept.

use super::*;

pub(crate) fn geometry_order(lines: &mut [Line]) {
    lines.sort_by(|left, right| {
        line_center_y(left)
            .total_cmp(&line_center_y(right))
            .then(line_center_x(left).total_cmp(&line_center_x(right)))
            .then(left.bbox[0].total_cmp(&right.bbox[0]))
            .then(left.id.cmp(&right.id))
    });
}

pub(crate) fn hyphen_join_score(lines: &[Line]) -> (usize, usize) {
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

pub(crate) fn y_regressions(lines: &[Line]) -> usize {
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

pub(crate) fn arbitrate_body_order(
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

pub(crate) fn repair_drop_caps(lines: &mut Vec<Line>) {
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

/// Restore physical header and footer bands after independent readers share a page.
pub(crate) fn order_furniture(page: &mut Page) {
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
pub(crate) fn order_page(
    page: &mut Page,
    table_page: bool,
    contents: bool,
) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    let mut header = Vec::new();
    let mut body = Vec::new();
    let mut note = Vec::new();
    let mut footer = Vec::new();
    for line in std::mem::take(&mut page.lines) {
        match line.region_type.as_str() {
            "header" => header.push(line),
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
pub(crate) fn opens_turn(line: &Line) -> bool {
    let text = line.text.trim_start();
    legal_structure_model::transcript::speaker_label(text).is_some()
        || text.starts_with("--")
        || (text.chars().any(char::is_alphabetic) && !text.chars().any(char::is_lowercase))
}
