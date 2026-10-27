//! The document's contents lists, read once from their own typography before any table claims
//! their lines, and handed on as entries to the heading outline (`contents::contents_outline`). A
//! contents page sets its entries' locators (page numbers, paragraph numbers, ranges) in one column
//! at the right of their titles, climbing down the page; rows are read by their geometry, so a
//! locator the extraction reads apart from its title, or files as a page number, still stands with
//! its row. A list of citations is a table of authorities, not contents. Where nothing names the
//! page contents and a table reader took its lines, or where it prints no locators at all, the list
//! stands only when its titles recur later in the document, in order, on later pages, or point to
//! where they recur (its locators the page labels or the paragraph numbers there). A word index is
//! contents throughout. Measured on the structure gold (batches 5-7): contents lines 0.89 / 0.96 /
//! 0.91 F1, the gold's contents found 97% of the time.

use super::bookmarks::normalize_title;
use legal_pdf_core::{layout_analysis, Line, Page};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

fn pattern(cell: &'static OnceLock<Regex>, source: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(source).expect("contents list regex"))
}

/// A locator alone, its value and as printed: "12", "xiv", "p. 3", "[18] - [43]", "1 to 27".
fn locator(text: &str) -> Option<(u32, String)> {
    static RE: OnceLock<Regex> = OnceLock::new();
    pattern(&RE, r"(?i)^\s*(?:p(?:age|p?\.|ara\.?)?\s*)?\[?(\d{1,4}|[ivxlc]{1,7})\]?(?:\s*(?:[-–]|to)\s*\[?\d{1,4}\]?)?\s*$")
        .captures(text)
        .map(|found| (value(&found[1]), found[1].to_owned()))
}

/// A locator after a leader at a title's end ("Overview ........ 3"), or after a tab or a wide
/// space the title is typeset with ("Introduction\u{2003}3"): where the title ends, and the locator.
fn leader_locator(text: &str) -> Option<(usize, u32, String)> {
    static RE: OnceLock<Regex> = OnceLock::new();
    pattern(&RE, r"(?:(?:\.\s?){2,}|_{3,}|\u{2026}+|[\t\u{2002}\u{2003}\u{2007}])\s*(?:p(?:age|p?\.)?\s*)?\[?(\d{1,4}|[ivxlc]{1,6})\]?(?:\s*(?:[-–]|to)\s*\[?\d{1,4}\]?)?\s*$")
        .captures(text)
        .map(|found| (found.get(0).map_or(text.len(), |all| all.start()), value(&found[1]), found[1].to_owned()))
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
pub(super) fn index_letter(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    pattern(&RE, r"^\s*(?:<\s*[A-Z0-9]\s*>|-\s*[A-Z]\s*-)\s*$").is_match(text)
}

fn value(token: &str) -> u32 {
    token.parse().ok().or_else(|| legal_pdf_core::roman_value(token)).unwrap_or(0)
}

/// Mostly a citation, its leader and locator aside: an authority's entry.
fn citation(text: &str) -> bool {
    static TAIL: OnceLock<Regex> = OnceLock::new();
    let title = pattern(&TAIL, r"(?:(?:\.\s?){2,}|_{3,}|\u{2026}+|\s{2,})[\s\d,\[\]–-]*$").replace(text.trim(), "");
    let citations = layout_analysis().citations(&title);
    let cited: usize = citations.iter().map(|found| found.range.len()).sum();
    !citations.is_empty() && cited * 3 >= title.trim().len()
}

/// One row of a page: its lines left to right, its text, its title (the text its locator aside),
/// and the locator it closes on (value, right edge, as printed).
pub(super) struct Row<'a> {
    pub(super) lines: Vec<&'a Line>,
    pub(super) text: String,
    pub(super) title: String,
    pub(super) locator: Option<(u32, f64, String)>,
}

/// A row read from the lines sharing its baseline band, left to right: its locator alone in the
/// row's last cell, after a leader or a wide space, or set apart by a gap of three line heights.
pub(super) fn row_of(lines: Vec<&Line>) -> Row<'_> {
    let text = lines.iter().map(|line| line.text.trim()).collect::<Vec<_>>().join(" ");
    let last = lines[lines.len() - 1];
    let join = |part: &[&Line]| part.iter().map(|line| line.text.trim()).collect::<Vec<_>>().join(" ");
    if lines.len() > 1 {
        if let Some((value, token)) = locator(&last.text) {
            let title = join(&lines[..lines.len() - 1]);
            return Row { locator: Some((value, last.bbox[2], token)), title, text, lines };
        }
    }
    if let Some((end, value, token)) = leader_locator(&text) {
        let title = text[..end].trim_end_matches(|c: char| c == '.' || c == '_' || c == '\u{2026}' || c.is_whitespace()).to_owned();
        return Row { locator: Some((value, last.bbox[2], token)), title, text, lines };
    }
    let height = (last.bbox[3] - last.bbox[1]).max(1.0);
    if let [.., before, word] = last.words.as_slice() {
        if word.bbox[0] - before.bbox[2] >= 3.0 * height {
            if let (Some((value, token)), Some(title)) = (locator(&word.text), text.strip_suffix(word.text.trim())) {
                let title = title.trim_end().to_owned();
                return Row { locator: Some((value, last.bbox[2], token)), title, text, lines };
            }
        }
    }
    let alone = (lines.len() == 1).then(|| locator(&last.text)).flatten();
    Row { locator: alone.map(|(value, token)| (value, last.bbox[2], token)), title: if lines.len() == 1 && locator(&text).is_some() { String::new() } else { text.clone() }, text, lines }
}

/// Lines in one baseline band are one row: the bands of `lines` in their order, each read left to
/// right.
pub(super) fn row_bands<'a>(lines: &[&'a Line]) -> Vec<Vec<&'a Line>> {
    let mut bands: Vec<Vec<&Line>> = Vec::new();
    for line in lines {
        let middle = (line.bbox[1] + line.bbox[3]) / 2.0;
        match bands.last_mut().filter(|band| {
            let first = band[0];
            let height = (first.bbox[3] - first.bbox[1]).max(1.0);
            ((first.bbox[1] + first.bbox[3]) / 2.0 - middle).abs() <= height / 2.0
        }) {
            Some(band) => band.push(line),
            None => bands.push(vec![line]),
        }
    }
    for band in &mut bands {
        band.sort_by(|a, b| a.bbox[0].total_cmp(&b.bbox[0]));
    }
    bands
}

/// A page's rows top to bottom, a locator alone on its row closing the title on the row above it.
fn rows(page: &Page) -> Vec<Row<'_>> {
    let mut lines: Vec<&Line> = page.lines.iter().filter(|line| !line.text.trim().is_empty() && line.region_type != "line_label").collect();
    lines.sort_by(|a, b| a.bbox[1].total_cmp(&b.bbox[1]).then(a.bbox[0].total_cmp(&b.bbox[0])));
    let mut rows: Vec<Row> = Vec::new();
    for band in row_bands(&lines) {
        let row = row_of(band);
        if let (Some(found), Some(prior)) = (row.locator.clone().filter(|_| row.title.is_empty()), rows.last_mut()) {
            if prior.locator.is_none() && prior.lines[0].bbox[0] < row.lines[0].bbox[0] && locator(&prior.text).is_none() {
                prior.locator = Some(found);
                prior.lines.extend(row.lines);
                continue;
            }
        }
        rows.push(row);
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

/// A page's number: the label the PDF declares for it, else the folio the page-number reader reads
/// in its running head or foot (`page_label`).
fn page_number(page: &Page, embedded: &[Option<String>]) -> Option<u32> {
    let numeric = |label: &str| {
        let label = label.trim();
        (label.chars().all(|c| c.is_ascii_digit()) || legal_pdf_core::roman_value(label).is_some())
            .then(|| value(label))
            .filter(|n| *n > 0)
    };
    embedded.get(page.index).and_then(Option::as_deref).and_then(numeric).or_else(|| {
        super::page_label(page).ok().flatten().and_then(|(label, _)| numeric(&label))
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
        if let Some(&found) = recurrences.get(&opening(title)).and_then(|places| places.iter().find(|(slot, _)| *slot > list_slot)) {
            later.push(found);
        }
    }
    let ordered = later.windows(2).filter(|pair| pair[1] >= pair[0]).count();
    later.len() >= 3 && later.len() * 5 >= titles.len() * 3 && ordered * 10 >= later.len().saturating_sub(1) * 9
}

/// One entry of a contents list as the heading outline reads it: its title (the rows it wraps over
/// joined), the locator it prints, where it starts, and the page it is on.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(super) struct Entry {
    pub(super) title: String,
    pub(super) locator: Option<String>,
    pub(super) left: f64,
    pub(super) page_index: usize,
}

/// The lists a document prints: its contents (and word index), the pages wholly contents, the
/// entries of its contents lists, and its tables of authorities.
#[derive(Default)]
pub(super) struct Lists {
    pub(super) contents: HashSet<String>,
    pub(super) pages: HashSet<usize>,
    pub(super) entries: Vec<Entry>,
    /// A list whose entries are mostly citations: the authorities a factum or a judgment relies on.
    pub(super) authorities: HashSet<String>,
}

/// Every contents list and table of authorities the document prints. `embedded` holds the page
/// labels the PDF declares, by page index.
pub(super) fn read(pages: &[Page], table_lines: &HashSet<String>, embedded: &[Option<String>]) -> Lists {
    let mut lists = Lists::default();
    let index = index_pages(pages);
    let recurrences = recurrences(pages);
    let trace = std::env::var_os("LEGALPDF_CONTENTS_TRACE").is_some();
    let ids = |rows: &[Row]| rows.iter().flat_map(|row| row.lines.iter().map(|line| line.id.clone())).collect::<Vec<_>>();
    let mut continuing = false;
    let mut previous_index = None;
    for (slot, page) in pages.iter().enumerate() {
        if previous_index.is_some_and(|prior: usize| prior + 1 != page.index) {
            continuing = false;
        }
        previous_index = Some(page.index);
        // A word index is contents throughout, its letters aside.
        if index.contains(&page.index) || reference_page(page) {
            lists.contents.extend(page.lines.iter()
                .filter(|line| !line.exclude_from_body && !matches!(line.region_type.as_str(), "header" | "footer") && !index_letter(&line.text))
                .map(|line| line.id.clone()));
            lists.pages.insert(page.index);
            continuing = false;
            continue;
        }
        let rows = rows(page);
        let width = page.width.max(1.0);
        // The locators of the list: the one column most of them close on, at the right of their
        // titles (a column of numbers at the left margin is a transcript's line numbers).
        let right = |r: usize| rows[r].locator.as_ref().map_or(0.0, |(_, right, _)| *right);
        let number = |r: usize| rows[r].locator.as_ref().map_or(0, |(value, _, _)| *value);
        let entries: Vec<usize> = (0..rows.len())
            .filter(|&r| rows[r].locator.is_some() && rows[r].lines[0].bbox[0] + 0.15 * width < right(r))
            .collect();
        // Named contents above the list, or in the page's running head.
        let headed = rows.iter().take(entries.first().map_or(4, |&first| first + 1).max(4)).any(|row| contents_title(&row.text))
            || page.lines.iter().any(|line| line.region_type == "header" && contents_title(&line.text));
        let column = entries.iter().map(|&r| right(r))
            .max_by_key(|&edge| entries.iter().filter(|&&r| (right(r) - edge).abs() <= 0.015 * width).count());
        let listed: Vec<usize> = column.map_or_else(Vec::new, |column| entries.iter().copied().filter(|&r| (right(r) - column).abs() <= 0.015 * width).collect());
        let pointed: Vec<(&str, u32)> = listed.iter().map(|&r| (rows[r].title.as_str(), number(r))).collect();
        let (agreeing, checked) = if listed.is_empty() { (0, 0) } else { agreement(&pointed, slot, pages, &recurrences, embedded) };
        let accepted = listed.len() >= if continuing || headed { 3 } else { 4 } && {
            let numbers: Vec<u32> = listed.iter().map(|&r| number(r)).collect();
            let climbing = numbers.windows(2).filter(|pair| pair[1] >= pair[0]).count();
            let span = listed[listed.len() - 1] - listed[0] + 1;
            let bar = if continuing || headed { 0.3 } else { 0.6 };
            let claimed = listed.iter().filter(|&&r| rows[r].lines.iter().any(|line| table_lines.contains(&line.id))).count();
            let titles: Vec<&str> = listed.iter().map(|&r| rows[r].title.as_str()).collect();
            // The list points where its titles are: confirmation whatever its layout.
            let agreed = checked >= 3 && agreeing * 2 >= checked;
            climbing * 10 >= numbers.len().saturating_sub(1) * 7
                && (agreed || listed.len() as f64 >= bar * span as f64 || (listed.len() >= 6 && listed.len() as f64 >= 0.6 * bar * span as f64))
                && (headed || agreed || claimed * 2 < listed.len() || echoed(&titles, slot, &recurrences))
        };
        // A list printing no locators: its titles recur later as the document's own parts, under a
        // contents title or six of them running together.
        let titled_rows: Vec<&str> = rows.iter().skip(1).filter(|row| row.text.split_whitespace().count() <= 15).map(|row| row.text.as_str()).collect();
        let run = if accepted { None } else { recurring_run(&rows, slot, pages, &recurrences) };
        let unlocated = !accepted && (run.is_some() || headed && titled_rows.len() >= 5 && echoed(&titled_rows, slot, &recurrences));
        // Entries mostly citations, their locators in one column whatever their order: a table of
        // authorities (its locators are the paragraphs citing each authority, not its place).
        let cites = |r: usize| citation(&rows[r].text) || r > 0 && citation(&format!("{} {}", rows[r - 1].text, rows[r].text));
        let authorities = !accepted && entries.len() >= 5 && entries.iter().filter(|&&r| cites(r)).count() * 5 >= entries.len() * 3;
        if trace && (entries.len() >= 2 || headed || run.is_some()) {
            eprintln!(
                "contents\tpage {}\trows {}\tentries {}\tlisted {}\tagree {agreeing}/{checked}\theaded {headed}\tcontinuing {continuing}\taccepted {accepted}\tunlocated {unlocated}\tauthorities {authorities}\tfirst {:?}",
                page.index + 1, rows.len(), entries.len(), listed.len(), listed.first().map(|&r| rows[r].text.chars().take(50).collect::<String>()),
            );
        }
        if accepted {
            // The list runs from its first entry (with up to two short rows above it: a column's
            // header, a wrapped title) to its last.
            let mut top = listed[0];
            while top > 0 && top + 2 > listed[0] && rows[top - 1].text.split_whitespace().count() <= 12 && !contents_title(&rows[top - 1].text) {
                top -= 1;
            }
            let last = listed[listed.len() - 1];
            // Mostly citations, entry by entry (a citation wraps over rows): a table of authorities.
            if listed.iter().filter(|&&r| cites(r)).count() * 2 > listed.len() {
                lists.authorities.extend(ids(&rows[top..=last]));
            } else {
                lists.contents.extend(ids(&rows[top..=last]));
                if (last - top + 1) * 5 >= rows.len() * 3 {
                    lists.pages.insert(page.index);
                }
                // Each entry with the rows its title wraps over since the entry before it (up to
                // three); where it prints its own number, those rows are entries of their own (a
                // part's title over its sections).
                let mut since = listed[0];
                for &r in &listed {
                    let between: Vec<&Row> = rows[since..r].iter().filter(|row| row.locator.is_none() && !row.title.is_empty()).collect();
                    let wraps = if super::heading_rules::numbered(&rows[r].title) { 0 } else { between.len().min(3) };
                    let (apart, wrapped) = between.split_at(between.len() - wraps);
                    for row in apart {
                        lists.entries.push(Entry { title: row.title.clone(), locator: None, left: row.lines[0].bbox[0], page_index: page.index });
                    }
                    let title = wrapped.iter().map(|row| row.title.as_str()).chain([rows[r].title.as_str()]).collect::<Vec<_>>().join(" ");
                    lists.entries.push(Entry {
                        title,
                        locator: rows[r].locator.as_ref().map(|(_, _, token)| token.clone()),
                        left: wrapped.first().map_or(rows[r].lines[0].bbox[0], |row| row.lines[0].bbox[0].min(rows[r].lines[0].bbox[0])),
                        page_index: page.index,
                    });
                    since = r + 1;
                }
            }
        } else if let Some((first, last)) = run {
            for row in &rows[first..=last] {
                lists.contents.extend(row.lines.iter().filter(|line| !matches!(line.region_type.as_str(), "header" | "footer")).map(|line| line.id.clone()));
                lists.entries.push(Entry { title: row.title.clone(), locator: None, left: row.lines[0].bbox[0], page_index: page.index });
            }
        } else if unlocated {
            for row in rows.iter().skip_while(|row| !contents_title(&row.text)).skip(1) {
                lists.contents.extend(row.lines.iter().filter(|line| !matches!(line.region_type.as_str(), "header" | "footer")).map(|line| line.id.clone()));
                lists.entries.push(Entry { title: row.title.clone(), locator: None, left: row.lines[0].bbox[0], page_index: page.index });
            }
        } else if authorities {
            lists.authorities.extend(ids(&rows[entries[0]..=entries[entries.len() - 1]]));
        }
        continuing = accepted || unlocated;
    }
    lists
}

/// Word-index and concordance pages: short headwords in alphabetical order, each
/// followed by an occurrence count or `page:line` / `page/line` locators.
pub(crate) fn index_pages(pages: &[Page]) -> HashSet<usize> {
    static COUNTED: OnceLock<Regex> = OnceLock::new();
    static HEADWORD: OnceLock<Regex> = OnceLock::new();
    let counted = COUNTED.get_or_init(|| {
        Regex::new(r"\[\d{1,3}\]\s+\d{1,4}[:/]\d{1,3}").expect("index entry regex")
    });
    let headword = HEADWORD.get_or_init(|| {
        Regex::new(
            r"^\s*(\S+(?:\s\S+){0,2}?)\s+(?:[\[(]\d{1,3}[\])]\s*$|(?:[\[(]\d{1,3}[\])]\s+)?\d{1,4}[:/]\d{1,3}(?:[\s,;]|$))",
        )
        .expect("index headword regex")
    });
    pages
        .iter()
        .filter_map(|page| {
            let text = page
                .lines
                .iter()
                .map(|line| line.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let headwords = page
                .lines
                .iter()
                .filter_map(|line| headword.captures(&line.text))
                .map(|capture| {
                    capture[1]
                        .trim_start_matches(['\'', '"', '\u{2018}', '\u{201c}'])
                        .to_lowercase()
                })
                .collect::<Vec<_>>();
            let ascending = headwords
                .windows(2)
                .filter(|pair| pair[0] <= pair[1])
                .count();
            (counted.find_iter(&text).take(5).count() >= 5
                || headwords.len() >= 5 && ascending * 5 >= (headwords.len() - 1) * 4)
                .then_some(page.index)
        })
        .collect()
}
