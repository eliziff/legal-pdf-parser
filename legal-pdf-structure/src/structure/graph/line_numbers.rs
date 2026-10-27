//! A transcript's printed line numbers: read, marked as line labels, and parted from the text a
//! line fused them into.

use super::*;

/// The page's transcript line numbers, as `mark_line_numbers` claimed them.
pub(crate) fn line_labels<'a>(page: &'a Page) -> impl Iterator<Item = &'a Line> + 'a {
    page.lines.iter().filter(|line| line.region_type == "line_label")
}

/// A line's transcript line number, printed apart or read into the line it numbers.
pub(crate) fn line_number(line: &Line) -> Option<(f64, u32)> {
    arabic_page_number(&line.text).map(|number| (line.bbox[0], number)).or_else(|| fused_line_number(line))
}

/// A transcript's printed line numbers, decided for the whole document. A page whose numbers
/// count 1 to 15 down one column shows where the column stands; on every page, the numbers set
/// there that count up are its line numbers too (a short last page, numbers every fifth line).
/// A number the extraction read into its line is split off as a line of its own. Each is a
/// line_label, outside the body, before any other reading of the page.
pub(crate) fn mark_line_numbers(pages: &mut [Page]) {
    let lefts: Vec<f64> = pages
        .iter()
        .flat_map(|page| {
            let column = number_column(page, line_number);
            column.into_iter().filter_map(|slot| line_number(&page.lines[slot]).map(|(left, _)| left)).collect::<Vec<_>>()
        })
        .collect();
    if lefts.is_empty() {
        return;
    }
    let left = legal_pdf_core::upper_median(lefts);
    for page in pages.iter_mut() {
        let tolerance = page.width * 0.03;
        let mut found: Vec<(f64, u32, usize)> = page
            .lines
            .iter()
            .enumerate()
            .filter_map(|(slot, line)| {
                let (x, number) = line_number(line)?;
                ((x - left).abs() <= tolerance && number <= 40).then_some((line.bbox[1], number, slot))
            })
            .collect();
        found.sort_by(|a, b| a.0.total_cmp(&b.0));
        if found.len() < 3 || !found.windows(2).all(|pair| pair[0].1 < pair[1].1) {
            continue;
        }
        let mut slots: Vec<usize> = found.into_iter().map(|(_, _, slot)| slot).collect();
        slots.sort_unstable();
        for &slot in slots.iter().rev() {
            if arabic_page_number(&page.lines[slot].text).is_none() {
                let number = split_line_number(&mut page.lines[slot]);
                let id = page.lines[slot].id.clone();
                for region in &mut page.regions {
                    if let Some(at) = region.line_ids.iter().position(|line_id| *line_id == id) {
                        region.line_ids.insert(at, number.id.clone());
                    }
                }
                page.lines.insert(slot, number);
            } else {
                page.lines[slot].region_type = "line_label".to_owned();
                page.lines[slot].exclude_from_body = true;
            }
        }
    }
}

/// A line's printed transcript line number where the extraction read it into the line it numbers
/// ("4        Q.   What is"): its left edge and value. A number whose next word stands at least two
/// line heights away is set in the margin, not in the text.
pub(crate) fn fused_line_number(line: &Line) -> Option<(f64, u32)> {
    let [first, second, ..] = line.words.as_slice() else {
        return None;
    };
    let number = first.text.parse::<u32>().ok().filter(|_| first.text.len() <= 2)?;
    let height = (first.bbox[3] - first.bbox[1]).max(1.0);
    let set_apart = second.bbox[0] - first.bbox[2] >= 2.0 * height;
    let read = line.text.chars().skip(first.start).take(first.end.saturating_sub(first.start)).eq(first.text.chars());
    let ordered = first.end <= second.start && second.start <= line.text.chars().count();
    (set_apart && read && ordered).then_some((first.bbox[0], number))
}

/// The line's leading number as a line label of its own, the line keeping its words from the
/// next on.
pub(crate) fn split_line_number(line: &mut Line) -> Line {
    let first = line.words[0].clone();
    let cut = line.words[1].start;
    let chars = line.text.chars().collect::<Vec<_>>();
    let width = first.end - first.start;
    let mut number = line.clone();
    number.id = format!("{}-n", line.id);
    number.text.clone_from(&first.text);
    number.bbox = first.bbox;
    number.words = vec![Word { start: 0, end: width, ..first.clone() }];
    number.spans = line
        .spans
        .iter()
        .filter(|span| span.start <= first.start && first.start < span.end)
        .map(|span| Span { id: format!("{}-n", span.id), text: first.text.clone(), bbox: first.bbox, start: 0, end: width, ..span.clone() })
        .take(1)
        .collect();
    number.detached_references.clear();
    number.region_type = "line_label".to_owned();
    number.exclude_from_body = true;
    line.text = chars[cut..].iter().collect();
    line.bbox[0] = line.words[1].bbox[0];
    line.words.remove(0);
    for word in &mut line.words {
        word.start -= cut;
        word.end -= cut;
    }
    line.spans.retain(|span| span.end > cut);
    for span in &mut line.spans {
        if span.start < cut {
            span.text = chars[cut..span.end.min(chars.len())].iter().collect();
            span.bbox[0] = line.bbox[0];
            span.start = cut;
        }
        span.start -= cut;
        span.end -= cut;
    }
    for reference in &mut line.detached_references {
        reference.start_offset = reference.start_offset.saturating_sub(cut);
        reference.end_offset = reference.end_offset.saturating_sub(cut);
    }
    number
}

/// A column of small numbers counting at least 1 to 15 down the page, as `number` reads each
/// line's (left edge, value). Returns the column's line slots, or none.
pub(crate) fn number_column(page: &Page, number: impl Fn(&Line) -> Option<(f64, u32)>) -> Vec<usize> {
    const MIN_LINE_NUMBERS: u32 = 15;
    if page.width <= 0.0 {
        return Vec::new();
    }
    let mut candidates = page
        .lines
        .iter()
        .enumerate()
        .filter_map(|(slot, line)| {
            let (x, number) = number(line)?;
            (number <= 40).then_some((x, number, slot))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.0.total_cmp(&right.0));
    let tolerance = page.width * 0.03;
    let mut best = &candidates[0..0];
    let mut start = 0;
    for end in 0..candidates.len() {
        while candidates[end].0 - candidates[start].0 > tolerance {
            start += 1;
        }
        if end + 1 - start > best.len() {
            best = &candidates[start..=end];
        }
    }
    let values = best
        .iter()
        .map(|(_, number, _)| *number)
        .collect::<HashSet<_>>();
    if values.len() >= MIN_LINE_NUMBERS as usize
        && (1..=MIN_LINE_NUMBERS).all(|number| values.contains(&number))
    {
        best.iter().map(|(_, _, slot)| *slot).collect()
    } else {
        Vec::new()
    }
}
