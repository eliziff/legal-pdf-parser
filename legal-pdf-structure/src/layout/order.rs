//! A page's reading order: the body in the order its columns read (arbitrated between the
//! source's and the geometry's), a drop cap joined to its line, furniture last, a transcript's turns
//! kept. Orders are slots of the page's lines.

use super::*;

/// The stream of a page a line reads in: its running head, its body, its note band or its foot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    Header,
    Body,
    Note,
    Footer,
}

pub fn geometry_order(lines: &[Line], members: &mut [usize]) {
    members.sort_by(|&left, &right| {
        let (left, right) = (&lines[left], &lines[right]);
        line_center_y(left)
            .total_cmp(&line_center_y(right))
            .then(line_center_x(left).total_cmp(&line_center_x(right)))
            .then(left.bbox[0].total_cmp(&right.bbox[0]))
            .then(left.id.cmp(&right.id))
    });
}

/// Two columns' lines read column by column down the page, band by band: a line spanning both columns (over 55% of
/// the page's width) stands as a band of its own between the columns it parts.
pub fn column_geometry_order(lines: &[Line], members: &[usize], model: ColumnModel, page_width: f64) -> Vec<usize> {
    let mut sorted = members.to_vec();
    geometry_order(lines, &mut sorted);
    let (mut ordered, mut left, mut right) = (Vec::with_capacity(sorted.len()), Vec::new(), Vec::new());
    for slot in sorted {
        let line = &lines[slot];
        if !has_valid_bbox(line) || line_width(line) / page_width > 0.55 {
            ordered.append(&mut left);
            ordered.append(&mut right);
            ordered.push(slot);
        } else if line_center_x(line) < model.split_x {
            left.push(slot);
        } else {
            right.push(slot);
        }
    }
    ordered.append(&mut left);
    ordered.append(&mut right);
    ordered
}

/// The lines ending on a hyphen after two letters or more, and of them those the next line goes on
/// from with two letters or more.
pub fn hyphen_join_score(lines: &[Line], members: &[usize]) -> (usize, usize) {
    let letters = |text: &mut dyn Iterator<Item = char>| text.take_while(char::is_ascii_alphabetic).count();
    let (mut candidates, mut satisfied) = (0, 0);
    for (index, &slot) in members.iter().enumerate() {
        let text = lines[slot].text.trim_end();
        let Some(last) = text.chars().last().filter(|character| matches!(character, '-' | '\u{00ad}' | '\u{00ac}')) else {
            continue;
        };
        if letters(&mut text[..text.len() - last.len_utf8()].chars().rev()) < 2 {
            continue;
        }
        candidates += 1;
        satisfied += usize::from(members.get(index + 1).is_some_and(|&next| letters(&mut lines[next].text.trim_start().chars()) >= 2));
    }
    (candidates, satisfied)
}

pub fn y_regressions(lines: &[Line], members: &[usize]) -> usize {
    let boxed: Vec<&Line> = members.iter().map(|&slot| &lines[slot]).filter(|line| has_valid_bbox(line)).collect();
    let tolerance = median(boxed.iter().map(|line| (line.bbox[3] - line.bbox[1]).max(0.0)).collect());
    boxed.windows(2).filter(|pair| line_center_y(pair[1]) < line_center_y(pair[0]) - tolerance).count()
}

/// The body's order (`body`, slots of the page's lines): the source's, unless the page's columns
/// (`columns`) read interleaved where pdf-inspector's order of the page's `items` reads them
/// through, or a single column's lines run up the page where their geometry reads down it. Whether the order is
/// uncertain: the source's kept on two columns it crosses more than twice.
pub fn arbitrate_body_order(page: &Page, body: &mut Vec<usize>, items: &crate::tables::PageItems, model: ColumnModel) -> bool {
    let (lines, page_width, page_height) = (&page.lines, page.width, page.height);
    let boxed = body.iter().filter(|&&slot| has_valid_bbox(&lines[slot])).count();
    if body.len() < 8 || boxed * 5 < body.len() * 4 || page_width <= 0.0 || page_height <= 0.0 {
        return false;
    }
    if has_table_caption(body.iter().map(|&slot| &lines[slot])) {
        return false;
    }
    // Hyphenated words left broken: those ending lines, less those the next line goes on from.
    let broken = |(candidates, satisfied): (usize, usize)| candidates.saturating_sub(satisfied);
    if model.kind == ColumnKind::Two {
        let source_switches = column_switches(lines, body, model, page_width);
        let source_run = median_column_run(lines, body, model, page_width);
        let source_hyphens = hyphen_join_score(lines, body);
        // The challenger: the lines in pdf-inspector's reading order; the source order stands unless
        // it fixes the columns' interleaving and breaks no hyphenated word.
        let mut kept = vec![false; lines.len()];
        for &slot in body.iter() {
            kept[slot] = true;
        }
        let mut challenger = inspected_order(body, kept_items(items, |slot| kept[slot]), lines.len());
        // Where pdf-inspector's order still crosses the columns (columns set line for line on one baseline, which it
        // reads as one line), the columns' geometry read one after the other.
        if column_switches(lines, &challenger, model, page_width) > 2 {
            challenger = column_geometry_order(lines, body, model, page_width);
        }
        let challenger_switches = column_switches(lines, &challenger, model, page_width);
        let challenger_hyphens = hyphen_join_score(lines, &challenger);
        let boxed_body = body.iter().filter(|&&slot| has_valid_bbox(&lines[slot]) && !lines[slot].exclude_from_body).count();
        let minimum_switches = if source_hyphens.0 > 0 { 3 } else { 5 }.max((boxed_body as f64 * 0.10) as usize);
        // A source changing column at nearly every line (half its column runs a line or two long) reads the page's
        // rows across both columns: its hyphen joins pair one column's word with the other's line, and prove
        // nothing; the challenger reading the columns one after the other stands.
        let rows = source_run <= 2 && source_switches >= minimum_switches && challenger_switches * 4 <= source_switches;
        if rows
            || source_switches >= minimum_switches
                && source_run <= 6
                && challenger_switches <= 2
                && source_switches.saturating_sub(challenger_switches) >= 2
                && challenger_hyphens.1 >= source_hyphens.1
                && broken(challenger_hyphens) <= broken(source_hyphens)
        {
            *body = challenger;
            return false;
        }
        return source_switches > 2;
    }
    if model.kind != ColumnKind::Single {
        return false;
    }
    let (source_regressions, source_hyphens) = (y_regressions(lines, body), hyphen_join_score(lines, body));
    let mut challenger = body.clone();
    geometry_order(lines, &mut challenger);
    let (challenger_regressions, challenger_hyphens) = (y_regressions(lines, &challenger), hyphen_join_score(lines, &challenger));
    if source_regressions >= 3.max((boxed as f64 * 0.08) as usize)
        && challenger_regressions <= source_regressions / 3
        && challenger_hyphens.1 > source_hyphens.1
        && broken(challenger_hyphens) < broken(source_hyphens)
    {
        *body = challenger;
    }
    false
}

/// A drop cap (a single large letter beside the start of its paragraph) moved to before the line it
/// begins; `members` are slots of `lines`.
pub fn repair_drop_caps(lines: &[Line], members: &mut Vec<usize>) {
    let body_size = median(members.iter().map(|&slot| line_font_size(&lines[slot])).filter(|size| (4.0..=24.0).contains(size)).collect());
    if body_size <= 0.0 {
        return;
    }
    let mut moves = Vec::new();
    for (index, &slot) in members.iter().enumerate() {
        let line = &lines[slot];
        let text = line.text.trim();
        if text.chars().count() != 1 || !text.chars().all(char::is_alphabetic) || line_font_size(line) < body_size * 1.8 {
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
            .min_by(|&(_, &left), &(_, &right)| lines[left].bbox[1].total_cmp(&lines[right].bbox[1]).then(lines[left].bbox[0].total_cmp(&lines[right].bbox[0])))
            .map(|(target, _)| target);
        moves.extend(target.filter(|target| *target < index).map(|target| (index, target)));
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
pub fn page_order(
    page: &Page,
    stream: impl Fn(usize) -> Stream,
    line_label: impl Fn(usize) -> bool,
    items: &crate::tables::PageItems,
    columns: ColumnModel,
    table_page: bool,
    contents: bool,
) -> (Vec<usize>, Vec<Diagnostic>) {
    let lines = &page.lines;
    let number = page.index as u32 + 1;
    let (mut header, mut body, mut note, mut footer) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for slot in 0..lines.len() {
        match stream(slot) {
            Stream::Header => &mut header,
            Stream::Note => &mut note,
            Stream::Footer => &mut footer,
            Stream::Body => &mut body,
        }
        .push(slot);
    }
    repair_drop_caps(lines, &mut body);
    let contents = table_page && contents && !has_table_caption(body.iter().map(|&slot| &lines[slot]));
    let uncertain = if contents && y_regressions(lines, &body) > 0 {
        geometry_order(lines, &mut body);
        false
    } else if table_page {
        false
    } else {
        // A law report's margin numbers are no column's lines: the columns' order is read without them, and they
        // stand after it (`Fences::read_margin_numbers` reads each before its row).
        // (A transcript's line numbers are its rows': they stay where they stand.)
        let lane: Vec<usize> = margin_lane(lines, &body, page.width).into_iter().filter(|&slot| !line_label(slot)).collect();
        body.retain(|slot| !lane.contains(slot));
        let uncertain = arbitrate_body_order(page, &mut body, items, columns);
        body.extend(lane);
        uncertain
    };
    let diagnostics = uncertain.then(|| Diagnostic::new("COLUMN_ORDER_UNCERTAIN", Some(page.index))).into_iter().collect();
    order_note_lines(lines, &mut note, items, number);
    // A running head and foot read row by row.
    read_rows(lines, &mut header, |slot| band_geometry_top(&lines[slot]));
    read_rows(lines, &mut footer, |slot| band_geometry_top(&lines[slot]));
    let mut ordered = header;
    ordered.extend(weave_note_columns(lines, body, note, items, page.width, number));
    ordered.extend(footer);
    (ordered, diagnostics)
}

/// Whether a transcript's line opens a unit of its own: a speaker's turn, or a line in capitals
/// (an examination's "BY MR. JONES:", "EXHIBIT NO. 3", "--- Upon resuming").
pub fn opens_turn(line: &Line) -> bool {
    let text = line.text.trim_start();
    legal_pdf_core::layout_analysis().speaker_label(text).is_some()
        || text.starts_with("--")
        || (text.chars().any(char::is_alphabetic) && !text.chars().any(char::is_lowercase))
}
