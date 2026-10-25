//! Contents lists read from their own typography, before any table claims their lines, and a word
//! index read as the contents it is. A contents page sets its entries' locators (page numbers,
//! paragraph numbers, ranges) in one column at the right of their titles, climbing down the page;
//! rows are read by their geometry, so a locator the extraction reads apart from its title, or
//! files as a page number, still stands with its row. A list of citations is a table of
//! authorities, not contents. Where nothing on the page names it contents and a table reader took
//! its lines, or where it prints no locators at all, the list stands only when its titles recur
//! later in the document, in order, on later pages: a document's contents names its own parts.
//! Measured on the structure gold's rows (batches 5-7): with the word index, precision 0.95 and
//! recall 0.89 on the gold's contents, against the readers' 0.79 and 0.08.

use super::bookmarks::normalize_title;
use super::contents::row_bands;
use super::graph::index_pages;
use legal_pdf_core::{layout_analysis, Line, Page};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

fn pattern(cell: &'static OnceLock<Regex>, source: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(source).expect("contents list regex"))
}

/// A locator alone: "12", "xiv", "p. 3", "[18] - [43]", "1 to 27".
fn locator(text: &str) -> Option<u32> {
    static RE: OnceLock<Regex> = OnceLock::new();
    pattern(&RE, r"(?i)^\s*(?:p(?:age|p?\.|ara\.?)?\s*)?\[?(\d{1,4}|[ivxlc]{1,7})\]?(?:\s*(?:[-–]|to)\s*\[?\d{1,4}\]?)?\s*$")
        .captures(text)
        .map(|found| value(&found[1]))
}

/// A locator after a leader at a title's end ("Overview ........ 3"), or after a tab or a wide
/// space the title is typeset with ("Introduction\u{2003}3").
fn leader_locator(text: &str) -> Option<u32> {
    static RE: OnceLock<Regex> = OnceLock::new();
    pattern(&RE, r"(?:(?:\.\s?){2,}|_{3,}|\u{2026}+|[\t\u{2002}\u{2003}\u{2007}])\s*(?:p(?:age|p?\.)?\s*)?\[?(\d{1,4}|[ivxlc]{1,6})\]?(?:\s*(?:[-–]|to)\s*\[?\d{1,4}\]?)?\s*$")
        .captures(text)
        .map(|found| value(&found[1]))
}

/// A page's or a list's own title: the parser's contents titles ("TABLE OF CONTENTS", "TABLE OF
/// PROVISIONS"), the same continued ("CONTENTS (cont'd)"), and an index's or a summary's ("INDEX
/// OF EXHIBITS", "I N D E X", "SOMMAIRE").
fn contents_title(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    super::contents_heading(text)
        || pattern(&RE, r"(?i)^\s*(?:(?:table\s+of\s+)?contents|table\s+des\s+mati[eè]res|index(?:\s+of\s+\w+)?|sommaire|i\s*n\s*d\s*e\s*x)\b").is_match(text)
}

/// A word index's letter: "< A >", "- B -"; a heading over its entries, not an entry.
fn index_letter(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    pattern(&RE, r"^\s*(?:<\s*[A-Z0-9]\s*>|-\s*[A-Z]\s*-)\s*$").is_match(text)
}

fn value(token: &str) -> u32 {
    if let Ok(number) = token.parse() {
        return number;
    }
    let digit = |c: char| match c.to_ascii_lowercase() { 'i' => 1, 'v' => 5, 'x' => 10, 'l' => 50, 'c' => 100, _ => 0 };
    let (mut total, mut prior) = (0i64, 0i64);
    for c in token.chars().rev() {
        let d = digit(c);
        if d < prior { total -= d } else { total += d; prior = d }
    }
    total.max(0) as u32
}

/// Mostly a citation, its leader and locator aside: an authority's entry.
fn citation(text: &str) -> bool {
    static TAIL: OnceLock<Regex> = OnceLock::new();
    let title = pattern(&TAIL, r"(?:(?:\.\s?){2,}|_{3,}|\u{2026}+|\s{2,})[\s\d,\[\]–-]*$").replace(text.trim(), "");
    let citations = layout_analysis().citations(&title);
    let cited: usize = citations.iter().map(|found| found.range.len()).sum();
    !citations.is_empty() && cited * 3 >= title.trim().len()
}

/// The locator a line ends on when a gap of its own sets it apart: its last word, a number, three
/// line heights or more from the word before it.
fn spaced_locator(line: &Line) -> Option<u32> {
    let [.., before, last] = line.words.as_slice() else { return None };
    let height = (line.bbox[3] - line.bbox[1]).max(1.0);
    (last.bbox[0] - before.bbox[2] >= 3.0 * height).then(|| locator(&last.text)).flatten()
}

/// One row of a page: its lines left to right, its text, and the locator it closes on.
struct Row<'a> {
    lines: Vec<&'a Line>,
    text: String,
    locator: Option<(u32, f64)>,
}

fn rows(page: &Page) -> Vec<Row<'_>> {
    let mut lines: Vec<&Line> = page.lines.iter().filter(|line| !line.text.trim().is_empty() && line.region_type != "line_label").collect();
    lines.sort_by(|a, b| a.bbox[1].total_cmp(&b.bbox[1]).then(a.bbox[0].total_cmp(&b.bbox[0])));
    let mut rows: Vec<Row> = Vec::new();
    for band in row_bands(&lines) {
        let text = band.iter().map(|line| line.text.trim()).collect::<Vec<_>>().join(" ");
        let last = band[band.len() - 1];
        let alone = (band.len() == 1).then(|| locator(&last.text)).flatten();
        // A locator alone on its row closes the title on the row above, set to its left.
        if let (Some(value), Some(prior)) = (alone, rows.last_mut()) {
            if prior.locator.is_none() && prior.lines[0].bbox[0] < last.bbox[0] && locator(&prior.text).is_none() {
                prior.locator = Some((value, last.bbox[2]));
                prior.lines.push(last);
                continue;
            }
        }
        let found = if band.len() > 1 { locator(&last.text) } else { None }
            .or_else(|| leader_locator(&text))
            .or_else(|| spaced_locator(last))
            .or(alone);
        rows.push(Row { locator: found.map(|value| (value, last.bbox[2])), text, lines: band });
    }
    rows
}

/// A page of a word index read by its references: most of its lines cite a transcript's
/// page:line ("153:13, 22"), a clock's time aside. Multi-column concordances, whose headwords
/// do not read alphabetically line by line, are found this way.
fn reference_page(page: &Page) -> bool {
    static REFERENCE: OnceLock<Regex> = OnceLock::new();
    static CLOCK: OnceLock<Regex> = OnceLock::new();
    let reference = pattern(&REFERENCE, r"\b\d{1,4}:\d{1,2}\b");
    let clock = pattern(&CLOCK, r"(?i)\b\d{1,2}:\d{2}\s*[ap]\.?\s?m\b");
    let lines: Vec<&Line> = page.lines.iter()
        .filter(|line| !line.exclude_from_body && !matches!(line.region_type.as_str(), "header" | "footer" | "line_label") && !line.text.trim().is_empty())
        .collect();
    let referring = lines.iter().filter(|line| reference.is_match(&line.text) && !clock.is_match(&line.text)).count();
    referring >= 10 && referring * 5 >= lines.len() * 3
}

/// Where each title recurs: every line's (page slot, line slot), by its first five words, normalized.
fn recurrences(pages: &[Page]) -> HashMap<String, Vec<(usize, usize)>> {
    let mut found: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
    for (slot, page) in pages.iter().enumerate() {
        for (at, line) in page.lines.iter().enumerate() {
            if line.exclude_from_body || matches!(line.region_type.as_str(), "header" | "footer") {
                continue;
            }
            let key = opening(&line.text);
            if !key.is_empty() {
                found.entry(key).or_default().push((slot, at));
            }
        }
    }
    found
}

/// A page's number: the label the PDF declares for it, else the folio the page-number reader finds
/// in its header or footer.
fn page_number(page: &Page, embedded: &[Option<String>]) -> Option<u32> {
    let numeric = |label: &str| {
        let label = label.trim();
        (label.chars().all(|c| c.is_ascii_digit()) || label.chars().all(|c| "ivxlcIVXLC".contains(c)))
            .then(|| value(label))
            .filter(|n| *n > 0)
    };
    embedded.get(page.index).and_then(Option::as_deref).and_then(numeric).or_else(|| {
        page.lines
            .iter()
            .filter(|line| matches!(line.region_type.as_str(), "header" | "footer"))
            .find_map(|line| super::printed_label(&line.text))
            .and_then(|label| numeric(&label))
    })
}

/// How many of a list's entries point where their titles recur: the entry's locator is the number
/// of the page its title recurs on, or the number of the paragraph printed right after it. Returns
/// (agreeing, checked).
fn agreement(
    entries: &[(&str, u32)],
    list_slot: usize,
    pages: &[Page],
    recurrences: &HashMap<String, Vec<(usize, usize)>>,
    embedded: &[Option<String>],
) -> (usize, usize) {
    static PARAGRAPH: OnceLock<Regex> = OnceLock::new();
    let paragraph = pattern(&PARAGRAPH, r"^\s*\[?(\d{1,4})[\].)]?(?:\s|$)");
    let (mut agreeing, mut checked) = (0, 0);
    for (title, at) in entries {
        let Some(&(slot, line)) = recurrences.get(&opening(title)).and_then(|found| found.iter().find(|(slot, _)| *slot > list_slot)) else {
            continue;
        };
        checked += 1;
        let page = &pages[slot];
        let on_page = page_number(page, embedded).is_some_and(|number| number.abs_diff(*at) <= 1);
        let next_paragraph = page.lines[line + 1..].iter().take(6)
            .find_map(|next| paragraph.captures(&next.text).and_then(|found| found[1].parse::<u32>().ok()));
        if on_page || next_paragraph.is_some_and(|number| number.abs_diff(*at) <= 1) {
            agreeing += 1;
        }
    }
    (agreeing, checked)
}

/// A title's first five words, its enumerator and locator aside.
fn opening(text: &str) -> String {
    static TAIL: OnceLock<Regex> = OnceLock::new();
    let text = pattern(&TAIL, r"(?:(?:\.\s?){2,}|_{3,}|\u{2026}+|\s{2,})\s*\S{1,8}\s*$").replace(text.trim(), "");
    let words: Vec<String> = text.split_whitespace().map(normalize_title).filter(|word| !word.is_empty()).collect();
    let skip = usize::from(words.len() > 2 && words[0].chars().all(|c| c.is_ascii_digit()));
    if words.len() < skip + 2 {
        return String::new();
    }
    words[skip..].iter().take(5).cloned().collect::<Vec<_>>().join(" ")
}

/// The rows of a page whose titles recur later in the document, in order: the first and last of
/// them, where at least six do and four in five of the page's short rows between them do.
fn recurring_run(rows: &[Row], list_slot: usize, pages: &[Page], recurrences: &HashMap<String, Vec<(usize, usize)>>) -> Option<(usize, usize)> {
    let short: Vec<usize> = (0..rows.len()).filter(|&r| rows[r].text.split_whitespace().count() <= 15).collect();
    // Where a part begins: a short line opening its own block, or one already read as a heading.
    let opens = |&(slot, at): &(usize, usize)| {
        let lines = &pages[slot].lines;
        let line = &lines[at];
        line.text.split_whitespace().count() <= 15
            && (line.region_type == "heading" || at == 0 || lines[at - 1].block_index != line.block_index)
    };
    let landed: Vec<(usize, (usize, usize))> = short.iter().copied().filter_map(|r| {
        recurrences.get(&opening(&rows[r].text))
            .and_then(|places| places.iter().find(|(slot, _)| *slot > list_slot).filter(|place| opens(place)))
            .map(|&place| (r, place))
    }).collect();
    let (first, last) = (landed.first()?.0, landed.last()?.0);
    let between = short.iter().filter(|&&r| r >= first && r <= last).count();
    let advancing = landed.windows(2).all(|pair| pair[1].1 > pair[0].1);
    let spread = landed.iter().map(|(_, (slot, _))| slot).collect::<HashSet<_>>().len();
    (landed.len() >= 6 && landed.len() * 5 >= between * 4 && advancing && spread >= 3).then_some((first, last))
}

/// Whether most of a list's titles recur later in the document, in order.
fn echoed(titles: &[&str], list_slot: usize, recurrences: &HashMap<String, Vec<(usize, usize)>>) -> bool {
    let mut later = Vec::new();
    for title in titles {
        let key = opening(title);
        if let Some(&found) = recurrences.get(&key).and_then(|places| places.iter().find(|(slot, _)| *slot > list_slot)) {
            later.push(found);
        }
    }
    let ordered = later.windows(2).filter(|pair| pair[1] >= pair[0]).count();
    later.len() >= 3 && later.len() * 5 >= titles.len() * 3 && ordered * 10 >= later.len().saturating_sub(1) * 9
}

/// The lists a document prints: its contents (and word index), and its tables of authorities.
#[derive(Default)]
pub(super) struct Lists {
    pub(super) contents: HashSet<String>,
    /// A list whose entries are mostly citations: the authorities a factum or a judgment relies on.
    pub(super) authorities: HashSet<String>,
}

/// Every contents line the typography and the word index find, and every table of authorities.
/// `embedded` holds the page labels the PDF declares, by page index.
pub(super) fn read(pages: &[Page], table_lines: &HashSet<String>, embedded: &[Option<String>]) -> Lists {
    let mut lists = Lists::default();
    let found = &mut lists.contents;
    let index = index_pages(pages);
    let recurrences = recurrences(pages);
    let mut continuing = false;
    let mut previous_index = None;
    for (slot, page) in pages.iter().enumerate() {
        if previous_index.is_some_and(|prior: usize| prior + 1 != page.index) {
            continuing = false;
        }
        previous_index = Some(page.index);
        // A word index is contents throughout, its letters aside.
        if index.contains(&page.index) || reference_page(page) {
            if std::env::var_os("LEGALPDF_CONTENTS_TRACE").is_some() {
                eprintln!("contents\tpage {}\tword index", page.index + 1);
            }
            found.extend(page.lines.iter()
                .filter(|line| !line.exclude_from_body && !matches!(line.region_type.as_str(), "header" | "footer") && !index_letter(&line.text))
                .map(|line| line.id.clone()));
            continuing = false;
            continue;
        }
        let rows = rows(page);
        let width = page.width.max(1.0);
        // The locators of the list: the one column most of them close on, at the right of their
        // titles (a column of numbers at the left margin is a transcript's line numbers).
        let entries: Vec<usize> = (0..rows.len())
            .filter(|&r| rows[r].locator.is_some_and(|(_, right)| rows[r].lines[0].bbox[0] + 0.15 * width < right))
            .collect();
        let headed = rows.iter().take(entries.first().map_or(4, |&first| first + 1).max(4)).any(|row| contents_title(&row.text));
        let column = entries.iter()
            .map(|&r| rows[r].locator.unwrap().1)
            .max_by_key(|&right| entries.iter().filter(|&&r| (rows[r].locator.unwrap().1 - right).abs() <= 0.015 * width).count());
        let listed: Vec<usize> = column.map_or_else(Vec::new, |column| {
            entries.iter().copied().filter(|&r| (rows[r].locator.unwrap().1 - column).abs() <= 0.015 * width).collect()
        });
        let enough = if continuing || headed { 3 } else { 4 };
        let accepted = listed.len() >= enough && {
            let numbers: Vec<u32> = listed.iter().map(|&r| rows[r].locator.unwrap().0).collect();
            let climbing = numbers.windows(2).filter(|pair| pair[1] >= pair[0]).count();
            let span = listed[listed.len() - 1] - listed[0] + 1;
            let bar = if continuing || headed { 0.3 } else { 0.6 };
            // Titles stand left of their locators: a column of numbers at the left margin is a
            // transcript's line numbers, not a list's locators.
            let titled = listed.iter().filter(|&&r| rows[r].lines[0].bbox[0] + 0.15 * width < rows[r].locator.unwrap().1).count();
            let claimed = listed.iter().filter(|&&r| rows[r].lines.iter().any(|line| table_lines.contains(&line.id))).count();
            let titles: Vec<&str> = listed.iter().map(|&r| rows[r].text.as_str()).collect();
            let pointed: Vec<(&str, u32)> = listed.iter().map(|&r| (rows[r].text.as_str(), rows[r].locator.unwrap().0)).collect();
            let (agreeing, checked) = agreement(&pointed, slot, pages, &recurrences, embedded);
            // The list points where its titles are: confirmation whatever its layout.
            let agreed = checked >= 3 && agreeing * 2 >= checked;
            climbing * 10 >= numbers.len().saturating_sub(1) * 7
                && (agreed || listed.len() as f64 >= bar * span as f64 || (listed.len() >= 6 && listed.len() as f64 >= 0.6 * bar * span as f64))
                && titled * 4 >= listed.len() * 3
                && (headed || agreed || claimed * 2 < listed.len() || echoed(&titles, slot, &recurrences))
        };
        // A list printing no locators: its titles recur later as the document's own parts, under a
        // contents title or six of them running together.
        let titled_rows: Vec<&str> = rows.iter().skip(1).filter(|row| row.text.split_whitespace().count() <= 15).map(|row| row.text.as_str()).collect();
        let run = if accepted { None } else { recurring_run(&rows, slot, pages, &recurrences) };
        let unlocated = !accepted && (run.is_some() || headed && titled_rows.len() >= 5 && echoed(&titled_rows, slot, &recurrences));
        // Entries mostly citations, their locators in one column whatever their order: a table of
        // authorities (its locators are the paragraphs citing each authority, not its place).
        let cited: Vec<usize> = entries.iter().copied()
            .filter(|&r| citation(&rows[r].text) || r > 0 && citation(&format!("{} {}", rows[r - 1].text, rows[r].text)))
            .collect();
        let authorities = !accepted && entries.len() >= 5 && cited.len() * 5 >= entries.len() * 3;
        if std::env::var_os("LEGALPDF_CONTENTS_TRACE").is_some() && (entries.len() >= 2 || headed || run.is_some()) {
            let numbers: Vec<u32> = listed.iter().map(|&r| rows[r].locator.unwrap().0).collect();
            let pointed: Vec<(&str, u32)> = listed.iter().map(|&r| (rows[r].text.as_str(), rows[r].locator.unwrap().0)).collect();
            let (agreeing, checked) = agreement(&pointed, slot, pages, &recurrences, embedded);
            let claimed = listed.iter().filter(|&&r| rows[r].lines.iter().any(|line| table_lines.contains(&line.id))).count();
            eprintln!(
                "contents\tpage {}\trows {}\tentries {}\tlisted {}\tcolumn {:.2}\tspan {}\tnumbers {:?}\theaded {headed}\tcontinuing {continuing}\tclaimed {claimed}\tagree {agreeing}/{checked}\taccepted {accepted}\tunlocated {unlocated}\tauthorities {authorities}\tfirst {:?}",
                page.index + 1, rows.len(), entries.len(), listed.len(), column.unwrap_or(0.0) / width,
                listed.last().zip(listed.first()).map_or(0, |(last, first)| last - first + 1), &numbers[..numbers.len().min(12)],
                listed.first().map(|&r| rows[r].text.chars().take(50).collect::<String>()),
            );
        }
        if accepted {
            // The list runs from its first entry (with up to two short rows above it: a column's
            // header, a wrapped title) to its last.
            let mut top = listed[0];
            while top > 0 && top + 2 > listed[0] && rows[top - 1].text.split_whitespace().count() <= 12 && !contents_title(&rows[top - 1].text) {
                top -= 1;
            }
            let span = &rows[top..=listed[listed.len() - 1]];
            // Mostly citations, entry by entry (a citation wraps over rows): a table of authorities.
            let cited = listed.iter().filter(|&&r| citation(&rows[r].text) || r > 0 && citation(&format!("{} {}", rows[r - 1].text, rows[r].text))).count();
            let target = if cited * 2 > listed.len() { &mut lists.authorities } else { &mut *found };
            for row in span {
                target.extend(row.lines.iter().map(|line| line.id.clone()));
            }
        } else if let Some((first, last)) = run {
            for row in &rows[first..=last] {
                found.extend(row.lines.iter().filter(|line| !matches!(line.region_type.as_str(), "header" | "footer")).map(|line| line.id.clone()));
            }
        } else if unlocated {
            for row in rows.iter().skip_while(|row| !contents_title(&row.text)).skip(1) {
                found.extend(row.lines.iter().filter(|line| !matches!(line.region_type.as_str(), "header" | "footer")).map(|line| line.id.clone()));
            }
        } else if authorities {
            for row in &rows[entries[0]..=entries[entries.len() - 1]] {
                lists.authorities.extend(row.lines.iter().map(|line| line.id.clone()));
            }
        }
        continuing = accepted || unlocated;
    }
    lists
}
