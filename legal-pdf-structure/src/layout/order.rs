//! A page's reading order: the body in the order its columns read (arbitrated between the
//! source's and the geometry's), a drop cap joined to its line, furniture last, a transcript's turns
//! kept. Orders are slots of the page's lines.

use super::*;

/// The stream of a page a line reads in: its running head, its body, its note band or its foot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stream {
    Header,
    Body,
    Note,
    Footer,
}

pub(crate) fn geometry_order(lines: &[Line], members: &mut [usize]) {
    members.sort_by(|&left, &right| {
        let (left, right) = (&lines[left], &lines[right]);
        line_center_y(left)
            .total_cmp(&line_center_y(right))
            .then(line_center_x(left).total_cmp(&line_center_x(right)))
            .then(left.bbox[0].total_cmp(&right.bbox[0]))
            .then(left.id.cmp(&right.id))
    });
}

pub(crate) fn hyphen_join_score(lines: &[Line], members: &[usize]) -> (usize, usize) {
    let mut candidates = 0;
    let mut satisfied = 0;
    for (index, &slot) in members.iter().enumerate() {
        let text = lines[slot].text.trim_end();
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
        if members.get(index + 1).is_some_and(|&next| {
            lines[next]
                .text
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

pub(crate) fn y_regressions(lines: &[Line], members: &[usize]) -> usize {
    let boxed: Vec<&Line> = members.iter().map(|&slot| &lines[slot]).filter(|line| has_valid_bbox(line)).collect();
    let tolerance = median(boxed
        .iter()
        .map(|line| (line.bbox[3] - line.bbox[1]).max(0.0))
        .collect());
    boxed
        .windows(2)
        .filter(|pair| line_center_y(pair[1]) < line_center_y(pair[0]) - tolerance)
        .count()
}

/// The body's order (`body`, slots of the page's lines): the source's, unless the page's columns
/// (`columns`) read interleaved where pdf-inspector's order of the page's `items` reads them
/// through, or a single column's lines run up the page where their geometry reads down it.
pub(crate) fn arbitrate_body_order(
    page: &Page,
    body: &mut Vec<usize>,
    items: &legal_pdf_tables::PageItems,
    columns: ColumnModel,
) -> OrderDecision {
    let (lines, page_width, page_height) = (&page.lines, page.width, page.height);
    let boxed = body.iter().filter(|&&slot| has_valid_bbox(&lines[slot])).count();
    if body.len() < 8 || boxed * 5 < body.len() * 4 || page_width <= 0.0 || page_height <= 0.0 {
        return OrderDecision::keep("insufficient_geometry");
    }
    if has_table_caption(body.iter().map(|&slot| &lines[slot])) {
        return OrderDecision::keep("table_grid");
    }
    let model = columns;
    if model.kind == "two_column" {
        let source_switches = column_switches(lines, body, model, page_width);
        let source_run = median_column_run(lines, body, model, page_width);
        let source_hyphens = hyphen_join_score(lines, body);
        // The challenger: the lines in pdf-inspector's reading order; the source order stands unless
        // it fixes the columns' interleaving and breaks no hyphenated word.
        let mut kept = vec![false; lines.len()];
        for &slot in body.iter() {
            kept[slot] = true;
        }
        let challenger = inspected_order(body, kept_items(items, |slot| kept[slot]), lines.len());
        let challenger_switches = column_switches(lines, &challenger, model, page_width);
        let challenger_hyphens = hyphen_join_score(lines, &challenger);
        let (left_count, right_count) = body
            .iter()
            .map(|&slot| &lines[slot])
            .filter(|line| has_valid_bbox(line) && !line.exclude_from_body)
            .fold((0, 0), |(left, right), line| {
                if line_center_x(line) < model.split_x { (left + 1, right) } else { (left, right + 1) }
            });
        let minimum_switches = if source_hyphens.0 > 0 { 3 } else { 5 }
            .max(((left_count + right_count) as f64 * 0.10) as usize);
        if source_switches >= minimum_switches
            && source_run <= 6
            && challenger_switches <= 2
            && source_switches.saturating_sub(challenger_switches) >= 2
            && challenger_hyphens.1 >= source_hyphens.1
            && challenger_hyphens.0.saturating_sub(challenger_hyphens.1)
                <= source_hyphens.0.saturating_sub(source_hyphens.1)
        {
            *body = challenger;
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
        let source_regressions = y_regressions(lines, body);
        let source_hyphens = hyphen_join_score(lines, body);
        let mut challenger = body.clone();
        geometry_order(lines, &mut challenger);
        let challenger_regressions = y_regressions(lines, &challenger);
        let challenger_hyphens = hyphen_join_score(lines, &challenger);
        let threshold = 3.max((boxed as f64 * 0.08) as usize);
        if source_regressions >= threshold
            && challenger_regressions <= source_regressions / 3
            && challenger_hyphens.1 > source_hyphens.1
            && challenger_hyphens.0.saturating_sub(challenger_hyphens.1)
                < source_hyphens.0.saturating_sub(source_hyphens.1)
        {
            *body = challenger;
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

/// A drop cap (a single large letter beside the start of its paragraph) moved to before the line it
/// begins; `members` are slots of `lines`.
pub(crate) fn repair_drop_caps(lines: &[Line], members: &mut Vec<usize>) {
    let body_size = median(members
        .iter()
        .map(|&slot| line_font_size(&lines[slot]))
        .filter(|size| (4.0..=24.0).contains(size))
        .collect());
    if body_size <= 0.0 {
        return;
    }
    let mut moves = Vec::new();
    for (index, &slot) in members.iter().enumerate() {
        let line = &lines[slot];
        let text = line.text.trim();
        if text.chars().count() != 1
            || !text.chars().all(char::is_alphabetic)
            || line_font_size(line) < body_size * 1.8
        {
            continue;
        }
        let target = members
            .iter()
            .enumerate()
            .filter(|&(other, &candidate)| {
                let candidate = &lines[candidate];
                other != index
                    && line_font_size(candidate) <= body_size * 1.4
                    && candidate.bbox[0] >= line.bbox[2] - body_size
                    && candidate.bbox[0] - line.bbox[2] <= body_size * 3.0
                    && candidate.bbox[1] < line.bbox[3]
                    && candidate.bbox[3] > line.bbox[1]
            })
            .min_by(|&(_, &left), &(_, &right)| {
                lines[left].bbox[1]
                    .total_cmp(&lines[right].bbox[1])
                    .then(lines[left].bbox[0].total_cmp(&lines[right].bbox[0]))
            })
            .map(|(target, _)| target);
        if let Some(target) = target.filter(|target| *target < index) {
            moves.push((index, target));
        }
    }
    for (index, target) in moves.into_iter().rev() {
        let slot = members.remove(index);
        members.insert(target, slot);
    }
}

/// A page's lines in reading order, as slots of its lines: its running head, then its body in the
/// order its columns read with its note band woven in by column, then its foot; each head and foot
/// top to bottom. `stream` places each line; `items` are the page's words (`page_items`) and
/// `columns` its column model. `table_page`: a table holds some of its lines; `contents`: the page
/// is a contents list's (contents_lists).
pub(crate) fn page_order(
    page: &Page,
    stream: impl Fn(usize) -> Stream,
    items: &legal_pdf_tables::PageItems,
    columns: ColumnModel,
    table_page: bool,
    contents: bool,
) -> (Vec<usize>, Vec<Diagnostic>) {
    let lines = &page.lines;
    let number = page.index as u32 + 1;
    let mut diagnostics = Vec::new();
    let mut header = Vec::new();
    let mut body = Vec::new();
    let mut note = Vec::new();
    let mut footer = Vec::new();
    for slot in 0..lines.len() {
        match stream(slot) {
            Stream::Header => header.push(slot),
            Stream::Note => note.push(slot),
            Stream::Footer => footer.push(slot),
            Stream::Body => body.push(slot),
        }
    }
    repair_drop_caps(lines, &mut body);
    let contents = table_page && contents && !has_table_caption(body.iter().map(|&slot| &lines[slot]));
    let decision = if contents && y_regressions(lines, &body) > 0 {
        geometry_order(lines, &mut body);
        OrderDecision {
            repair: OrderRepair::Geometry,
            source_switches: 0,
            strategy: "table-row-geometry",
            reason: "table_source_order_scrambled",
        }
    } else if table_page {
        OrderDecision::keep("table_grid")
    } else {
        arbitrate_body_order(page, &mut body, items, columns)
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
        diagnostic.line_ids = body.iter().take(20).map(|&slot| lines[slot].id.clone()).collect();
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
        diagnostic.line_ids = body.iter().take(20).map(|&slot| lines[slot].id.clone()).collect();
        diagnostics.push(diagnostic);
    }
    order_note_lines(lines, &mut note, items, number);
    // A running head and foot read top to bottom.
    header.sort_by(|&left, &right| band_geometry_order(&lines[left], &lines[right]));
    footer.sort_by(|&left, &right| band_geometry_order(&lines[left], &lines[right]));
    let mut ordered = header;
    ordered.extend(weave_note_columns(lines, body, note, items, page.width, number));
    ordered.extend(footer);
    (ordered, diagnostics)
}

/// Whether a transcript's line opens a unit of its own: a speaker's turn, or a line in capitals
/// (an examination's "BY MR. JONES:", "EXHIBIT NO. 3", "--- Upon resuming").
pub(crate) fn opens_turn(line: &Line) -> bool {
    let text = line.text.trim_start();
    legal_structure_model::transcript::speaker_label(text).is_some()
        || text.starts_with("--")
        || (text.chars().any(char::is_alphabetic) && !text.chars().any(char::is_lowercase))
}
