//! Reading a PDF's print: the paragraphs a printed paragraph number addresses, in the body or
//! in the margin.

use std::collections::HashSet;

// Printed paragraph numbers are addresses, not structural ordinal positions.
// Split on the actual native lines: a prose node can contain several numbered
// paragraphs, and one printed paragraph can contain several prose nodes.
// Parallel-language columns print one paragraph number per column. A line
// belongs to the column holding its centre, which mirrored margins preserve.
fn in_column(column: [f64; 4], line: [f64; 4]) -> bool {
    let centre = (line[0] + line[2]) / 2.0;
    centre > column[0] && centre < column[2]
}

type PrintedLine<'a> = (&'a str, &'a str, u32, [f64; 4]);

/// A heading's text: a title, or a short line in sentence case that cites nothing and ends
/// without a sentence's punctuation ("Ministerial delegate's testimony").
fn heading_text(text: &str) -> bool {
    let text = text.trim();
    crate::heading_text_plausible(text) || (text.starts_with(|c: char| c.is_alphabetic() && c.is_uppercase())
        && !text.ends_with(['.', ',', ';', ':']) && text.chars().count() <= 100
        && !text.rsplit(char::is_whitespace).next().is_some_and(|word| word.chars().all(|c| c.is_ascii_digit()))
        && legal_pdf_core::structure_analysis().citations(text).is_empty())
}

/// A paragraph's own lines, up to the next heading of the outline below its first
/// ("Circumstances of the offence").
fn own<'l, 'a>(run: impl Iterator<Item = &'l PrintedLine<'a>>, headings: &HashSet<String>) -> Vec<&'l PrintedLine<'a>> {
    run.enumerate().take_while(|(index, (id, ..))| *index == 0 || !headings.contains(*id))
        .map(|(_, line)| line).collect()
}

/// `headings`: the lines of the document's outline headings. A paragraph ends where the next
/// heading or the next numbered paragraph begins.
pub fn printed_paragraph_plan<'a>(
    lines: &[(&'a str, &'a str, u32, [f64; 4])],
    paragraphs: &[Vec<String>],
    headings: &HashSet<String>,
    locator: &str,
) -> Option<(crate::PdfLookupStatus, HashSet<&'a str>)> {
    use crate::PdfLookupStatus as Status;
    let labels = lines.iter().enumerate().filter_map(|(index, (_, text, ..))| {
        let (number, rest) = text.trim_start().strip_prefix('[')?.split_once(']')?;
        if (!rest.is_empty() && !rest.starts_with(char::is_whitespace)) ||
            !number.chars().all(|c| c.is_ascii_digit()) { return None; }
        Some((number.parse::<usize>().ok()?, index))
    }).collect::<Vec<_>>();
    if labels.is_empty() { return None; }
    let range = crate::numeric_range("paragraph", locator)
        .or_else(|| crate::parse_ordinal("paragraph", locator).map(|n| (n, n)));
    let Some((from, to)) = range.filter(|(a, b)| a <= b && b - a < 100) else {
        return Some((Status::Invalid, HashSet::new()));
    };
    let mut selected = HashSet::new();
    for number in from..=to {
        let hits = labels.iter().filter(|(label, _)| *label == number)
            .map(|(_, index)| *index).collect::<Vec<_>>();
        // The same number printed once per column is one address, not two
        // paragraphs; a repeat in the same column or on another page is not.
        if hits.is_empty() || hits.iter().any(|&hit| hits.iter().any(|&other| other != hit &&
            (lines[other].2 != lines[hit].2 || in_column(lines[hit].3, lines[other].3)))) {
            return Some((if hits.is_empty() { Status::NotFound } else { Status::Ambiguous }, HashSet::new()));
        }
        for start in hits {
            // A detached number and the first line of its text share a row;
            // the column is both. A parallel translation never shares a row.
            let mut column = lines[start].3;
            if let Some(next) = lines.get(start + 1).filter(|next| next.2 == lines[start].2
                && next.3[1] < column[3] && column[1] < next.3[3]) {
                column = [column[0].min(next.3[0]), column[1], column[2].max(next.3[2]), column[3]];
            }
            if let Some((_, end)) = labels.iter()
                .find(|(_, index)| *index > start && in_column(column, lines[*index].3)) {
                let mut run = own(lines[start..*end].iter().filter(|line| in_column(column, line.3)), headings);
                // A heading the outline does not name ("Issue") stands set apart just above the next
                // paragraph, its text a heading's: it opens what follows, not this paragraph.
                let next = &lines[*end];
                let gap = |above: &[f64; 4], below: &[f64; 4]| below[1] - above[3] > 0.5 * (below[3] - below[1]).min(above[3] - above[1]);
                // Headings may stand one above another ("Submissions of the parties", "Submissions of
                // the appellant"); a heading of several lines has no gap within it.
                let (mut at, mut cut) = (run.len(), run.len());
                while at > 1 && run[at - 1].2 == next.2 && heading_text(run[at - 1].1)
                    && (at < run.len() || gap(&run[at - 1].3, &next.3)) {
                    at -= 1;
                    if gap(&run[at - 1].3, &run[at].3) { cut = at; }
                }
                run.truncate(cut);
                selected.extend(run.iter().map(|(id, ..)| *id));
            } else {
                // At EOF use the native owner, not unbounded end matter.
                let owner = paragraphs.iter().find(|ids| ids.iter().any(|id| id == lines[start].0));
                selected.extend(own(lines[start..].iter().filter(|(id, ..)|
                    owner.is_some_and(|ids| ids.iter().any(|value| value == id))), headings).iter().map(|(id, ..)| *id));
            }
        }
    }
    Some((Status::Found, selected))
}

// Detached paragraph labels sit outside the body, unlike reporter page
// numbers. Their aligned body rows bound both columns in parallel text.
// Use these native witnesses even when the structure profile has no prose
// nodes; a structural ordinal is not a substitute for a printed address.
// A heading set out to the margin ("Circumstances of the offence") is no part of the body the
// numbers stand beside, and a paragraph ends where the next heading begins.
pub fn marginal_paragraph_plan<'a>(
    pages: &'a [crate::PdfTextPage],
    headings: &HashSet<String>,
    locator: &str,
) -> Option<(crate::PdfLookupStatus, HashSet<&'a str>)> {
    use crate::PdfLookupStatus as Status;
    let body_bounds = pages.iter().map(|page| {
        page.lines.iter().filter(|line| line.rect[2] - line.rect[0] > page.width * 0.20
            && !headings.contains(&line.id) && line.text.chars().any(char::is_alphabetic))
            .map(|line| line.rect).reduce(|a, b| [a[0].min(b[0]), a[1].min(b[1]),
                a[2].max(b[2]), a[3].max(b[3])])
    }).collect::<Vec<_>>();
    let mut labels = Vec::new();
    for (page_index, page) in pages.iter().enumerate() {
        let Some(bounds) = body_bounds[page_index] else { continue };
        for line in &page.lines {
            if let Some(number) = set_apart_number(page, line) {
                labels.push((number, page_index, line.rect[1]));
                continue;
            }
            let text = line.text.trim();
            let text = text.strip_prefix('[').and_then(|s| s.strip_suffix(']')).unwrap_or(text);
            if text.is_empty() || text.len() > 5 || !text.bytes().all(|b| b.is_ascii_digit()) {
                continue;
            }
            let height = line.rect[3] - line.rect[1];
            let gap = (bounds[0] - line.rect[2]).max(line.rect[0] - bounds[2]);
            if gap < height * 0.5 || gap > height * 8.0 { continue; }
            let row = page.lines.iter().filter(|peer|
                peer.rect[2] - peer.rect[0] > page.width * 0.20
                && peer.text.chars().any(char::is_alphabetic)
                && (peer.rect[1] - line.rect[1]).abs() <= height * 0.6)
                .map(|peer| peer.rect[1]).min_by(f64::total_cmp);
            if let Some(y) = row {
                labels.push((text.parse::<usize>().ok()?, page_index, y));
            }
        }
    }
    labels.sort_by(|a, b| a.1.cmp(&b.1).then(a.2.total_cmp(&b.2)));
    // One isolated margin number could be a note. Require a numbering run.
    if !labels.windows(3).any(|w| w[0].0 + 1 == w[1].0 && w[1].0 + 1 == w[2].0) {
        return None;
    }
    let range = crate::numeric_range("paragraph", locator)
        .or_else(|| crate::parse_ordinal("paragraph", locator).map(|n| (n, n)));
    let Some((from, to)) = range.filter(|(a, b)| a <= b && b - a < 100) else {
        return Some((Status::Invalid, HashSet::new()));
    };
    let mut selected = HashSet::new();
    for number in from..=to {
        let hits = labels.iter().enumerate().filter(|(_, label)| label.0 == number)
            .map(|(index, _)| index).collect::<Vec<_>>();
        if hits.len() != 1 {
            return Some((if hits.is_empty() { Status::NotFound } else { Status::Ambiguous }, HashSet::new()));
        }
        let (_, start_page, start_y) = labels[hits[0]];
        // The last paragraph ends with its page's body; its end is otherwise unwitnessed, and no end
        // matter or numbering restart is swept in.
        let last = hits[0] + 1 == labels.len();
        let Some((end_page, end_y)) = labels.get(hits[0] + 1).filter(|next| next.0 == number + 1)
            .map(|next| (next.1, next.2))
            .or_else(|| last.then(|| body_bounds[start_page].map(|bounds|
                (start_page, (bounds[3] + 1.0).min(pages[start_page].height * 0.95)))).flatten()) else {
            return Some((Status::Unavailable, HashSet::new()));
        };
        // The first heading below the paragraph's first line, before the next paragraph.
        let stop = (start_page..=end_page).find_map(|page_index| pages[page_index].lines.iter()
            .filter(|line| headings.contains(&line.id)
                && (page_index > start_page || line.rect[1] > start_y + 0.5)
                && (page_index < end_page || line.rect[1] < end_y - 0.5))
            .map(|line| line.rect[1]).min_by(f64::total_cmp).map(|y| (page_index, y)));
        let end_page = stop.map_or(end_page, |(page, _)| page);
        for page_index in start_page..=end_page {
            let Some(bounds) = body_bounds[page_index] else {
                return Some((Status::Unavailable, HashSet::new()));
            };
            let top = if page_index == start_page { start_y - 0.5 } else { bounds[1] };
            let bottom = match stop {
                Some((page, y)) if page == page_index => y - 0.5,
                _ if page_index == end_page => end_y - 0.5,
                _ => bounds[3],
            };
            let lines = pages[page_index].lines.iter().filter(|line|
                line.rect[0] >= bounds[0] - 0.5 && line.rect[2] <= bounds[2] + 0.5
                && line.rect[1] >= top && line.rect[1] < bottom).collect::<Vec<_>>();
            // A separated heading immediately before the next numbered
            // paragraph belongs to that following section, not this passage.
            let heading_top = (page_index == end_page).then(|| lines.iter().filter(|line| {
                let Some((prefix, text)) = line.text.trim().split_once(". ") else { return false };
                if crate::enumerator_interpretations(prefix, ".").is_empty()
                    || !crate::heading_text_plausible(text) { return false; }
                let height = line.rect[3] - line.rect[1];
                let prior_bottom = lines.iter().filter(|prior| prior.rect[3] < line.rect[1])
                    .map(|prior| prior.rect[3]).max_by(f64::total_cmp);
                prior_bottom.is_some_and(|y| line.rect[1] - y > height * 0.5)
                    && end_y - line.rect[3] > height * 0.5
            }).map(|line| line.rect[1]).min_by(f64::total_cmp)).flatten();
            selected.extend(lines.iter().filter(|line|
                heading_top.is_none_or(|y| line.rect[1] < y - 0.5)).map(|line| line.id.as_str()));
        }
    }
    Some((Status::Found, selected))
}

/// A number that opens a line of the body set apart from the text after it by more than a space
/// ("1      At the conclusion of the hearing", as Westlaw prints a judgment's paragraphs): the
/// paragraph's printed number, standing bare. A number in running text ("11 of the Act") is
/// followed by an ordinary space; a page's running head and foot are no body.
fn set_apart_number(page: &crate::PdfTextPage, line: &crate::PdfTextLine) -> Option<usize> {
    let [number, next, ..] = line.words.as_slice() else { return None };
    let height = number.rect[3] - number.rect[1];
    (number.text.len() <= 4 && number.text.bytes().all(|byte| byte.is_ascii_digit()) && height > 0.0
        && next.rect[0] - number.rect[2] >= 0.75 * height
        && line.rect[1] > page.height * 0.05 && line.rect[3] < page.height * 0.95)
        .then(|| number.text.parse().ok()).flatten()
}

/// "[12]", "(12)", "12." or "12)": a printed paragraph number; a margin
/// number may also stand bare.
fn printed_label_number(text: &str, bare: bool) -> Option<usize> {
    let text = text.trim();
    let digits = text.strip_prefix('[').and_then(|rest| rest.strip_suffix(']'))
        .or_else(|| text.strip_prefix('(').and_then(|rest| rest.strip_suffix(')')))
        .or_else(|| text.strip_suffix(['.', ')']))
        .or(bare.then_some(text))?.trim();
    (!digits.is_empty() && digits.len() <= 5 && digits.bytes().all(|byte| byte.is_ascii_digit()))
        .then(|| digits.parse().ok()).flatten()
}

/// Whether the cited paragraph numbers are printed on the passage: each
/// endpoint opens one of its lines, or stands in the margin beside one.
pub fn printed_paragraph_witnessed<'a>(
    pages: impl Iterator<Item = &'a crate::PdfTextPage>,
    selected: &HashSet<&str>,
    locator: &str,
) -> bool {
    let Some((from, to)) = crate::numeric_range("paragraph", locator)
        .or_else(|| crate::parse_ordinal("paragraph", locator).map(|n| (n, n)))
    else { return false };
    let mut missing = HashSet::from([from, to]);
    for page in pages {
        let lines = page.lines.iter().filter(|line| selected.contains(line.id.as_str()))
            .collect::<Vec<_>>();
        for line in &lines {
            // A label opens the line: "[12] Text", "12. Text" or "(12) Text".
            let opening = line.words.first().map(|word| word.text.as_str()).unwrap_or_default();
            let opening = if opening == "[" || opening == "(" {
                line.words.iter().take(3).map(|word| word.text.as_str()).collect::<String>()
            } else { opening.to_owned() };
            if let Some(number) = printed_label_number(&opening, false).or_else(|| set_apart_number(page, line)) {
                missing.remove(&number);
            }
        }
        for word in page.lines.iter().flat_map(|line| &line.words) {
            let Some(number) = printed_label_number(&word.text, true).filter(|n| missing.contains(n)) else { continue };
            let [left, top, right, bottom] = word.rect;
            if top <= page.height * 0.05 || bottom >= page.height * 0.95 { continue; }
            if lines.iter().any(|line| {
                let [text_left, text_top, text_right, text_bottom] = line.rect;
                let overlap = bottom.min(text_bottom) - top.max(text_top);
                let gap = if right <= text_left { text_left - right }
                    else if left >= text_right { left - text_right } else { -1.0 };
                (0.0..=80.0).contains(&gap)
                    && overlap >= 0.45 * (bottom - top).min(text_bottom - text_top)
            }) { missing.remove(&number); }
        }
    }
    missing.is_empty()
}
