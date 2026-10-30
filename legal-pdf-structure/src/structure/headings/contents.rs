//! The document's printed contents list read as its outline: each entry's title, the page it
//! names and its depth, for the bookmark reconciliation to find on that page. Ported from the Open
//! Access Journals Database's contents-outline witness (`toc_outline_witness.py`): an entry is
//! confirmed on its destination page, where its title is a heading already or a line that may
//! become one, and its depth stands only where the list's own hierarchy is coherent. The entries
//! are the contents lists' own (`contents_lists`).
//!
//! Resolve within local list and pagination runs using neighboring confirmed body titles.
//! Repeated titles remain unresolved unless local bounds or agreeing pagination witnesses
//! identify one actual occurrence. Depth comes from the heading ladder where the entries are
//! enumerated or all set at one indent, else from the list's indents.

use super::bookmarks::normalize_title;
use crate::structure::contents_lists::{row_bands, row_of, Entry};
use super::rules::numbered;
use crate::structure::decide::ladder_levels;
use legal_pdf_core::model::{Line, Page, PdfOutlineEntry};
use std::collections::HashMap;

/// Indents within this many points are one tier.
const INDENT_TOLERANCE: f64 = 6.0;
/// An entry's title wraps over at most this many rows.
const MAX_WRAPPED_ROWS: usize = 3;
/// The deepest level a contents list sets.
const MAX_LEVEL: usize = 6;

/// Each contents line's entry, numbered from 0 in order, of a page `width` wide whose `lines` (in
/// reading order) `listed` names as contents: a printed marker or a completed locator starts the
/// next entry, a row without one goes on with the entry its locator left open, and a row holding
/// another line closes it.
pub(crate) fn entry_rows(width: f64, lines: &[&Line], listed: impl Fn(usize) -> bool) -> Vec<Option<usize>> {
    let mut entries = vec![None; lines.len()];
    let mut next: Option<usize> = None;
    let mut open = false;
    let places: Vec<usize> = (0..lines.len()).collect();
    for band in row_bands(&places, |at| lines[at].bbox) {
        if band.iter().any(|&at| !listed(at)) {
            open = false;
            continue;
        }
        let row = row_of(band.iter().map(|&at| lines[at]).collect());
        let first = lines[band[0]].text.trim();
        let token = first.split_whitespace().next().unwrap_or("");
        let numeric = token.chars().any(|c| c.is_ascii_digit())
            && token.chars().all(|c| c.is_ascii_digit() || c == '.');
        let locator = band.len() == 1 && lines[band[0]].bbox[0] > width * 0.5 && row.locator.is_some();
        let marker = !locator
            && (numeric
                || numbered(&row.title)
                || band.len() > 1
                    && first.chars().count() == 1
                    && first.chars().all(char::is_alphabetic));
        if marker || !open {
            next = Some(next.map_or(0, |entry| entry + 1));
        }
        for &at in &band {
            entries[at] = next;
        }
        open = row.locator.is_none() && !locator;
    }
    entries
}

/// The level of each entry: where most entries print a range ("[18] - [43]"), one deeper than the
/// entries before it whose ranges hold its own (a part's paragraphs hold its sections'); where most
/// print an enumerator, or every entry starts at one indent, its level on the heading ladder the
/// headings themselves are levelled on (`heading_rules::ladder_levels`), a plain title taking the
/// first level; else its indent's tier.
fn levels(entries: &[&Entry]) -> Vec<usize> {
    let ranges: Vec<Option<(i64, i64)>> = entries.iter().map(|entry| {
        let (first, last) = entry.locator.as_deref()?.split_once('-')?;
        Some((first.parse().ok()?, last.parse().ok()?)).filter(|(first, last)| first <= last)
    }).collect();
    if ranges.iter().flatten().count() * 4 >= entries.len() * 3 {
        let mut open: Vec<(i64, i64)> = Vec::new();
        return ranges.iter().map(|range| {
            if let Some((first, last)) = *range {
                while open.last().is_some_and(|&(start, end)| first < start || last > end) {
                    open.pop();
                }
                open.push((first, last));
            }
            open.len().clamp(1, MAX_LEVEL)
        }).collect();
    }
    let rungs = ladder_levels(&entries.iter().map(|entry| entry.title.as_str()).collect::<Vec<_>>(), &[]);
    let ladder = || -> Vec<usize> { rungs.iter().map(|rung| rung.map_or(1, |(level, _)| level.min(MAX_LEVEL))).collect() };
    if rungs.iter().filter(|rung| rung.is_some()).count() * 2 >= entries.len() {
        return ladder();
    }
    let mut tiers = entries.iter().map(|entry| entry.left).collect::<Vec<_>>();
    tiers.sort_by(f64::total_cmp);
    tiers.dedup_by(|a, b| (*a - *b).abs() <= INDENT_TOLERANCE);
    if tiers.len() < 2 {
        return ladder();
    }
    entries
        .iter()
        .map(|entry| {
            let tier = tiers.iter().position(|tier| entry.left <= tier + INDENT_TOLERANCE).unwrap_or(tiers.len() - 1);
            (tier + 1).min(MAX_LEVEL)
        })
        .collect()
}

/// A locator's page number: its first run of figures ("p. 12", "12 - 14").
fn arabic(locator: Option<&str>) -> Option<i64> {
    let digits = locator?.trim_start_matches(|c: char| !c.is_ascii_digit());
    digits.split(|c: char| !c.is_ascii_digit()).next()?.parse().ok()
}
/// The page slot of every run of up to `MAX_WRAPPED_ROWS` lines set outside the contents and its
/// furniture, by the run's normalized text, each start counted once and in reading order.
fn titles_by_text(pages: &[Page], outside: &impl Fn(usize, usize) -> bool) -> HashMap<String, Vec<usize>> {
    let mut titles = HashMap::<String, Vec<usize>>::new();
    for (slot, page) in pages.iter().enumerate() {
        let norms = page.lines.iter().enumerate().map(|(at, line)| {
            outside(slot, at).then(|| normalize_title(&line.text))
        }).collect::<Vec<_>>();
        for start in 0..norms.len() {
            let mut joined = String::new();
            let mut read = Vec::<String>::new();
            for norm in &norms[start..(start + MAX_WRAPPED_ROWS).min(norms.len())] {
                let Some(norm) = norm else { break };
                joined.push_str(norm);
                if !read.contains(&joined) {
                    read.push(joined.clone());
                    titles.entry(joined.clone()).or_default().push(slot);
                }
            }
        }
    }
    titles
}

/// The contents lists' `entries` as the document's outline, each local run of entries placed by
/// its own confirmed destinations and pagination, a title found only on the lines `outside` holds
/// (by page slot and line slot): no line label, furniture, or contents list's line or page.
pub(crate) fn contents_outline_among(
    pages: &[Page],
    entries: &[Entry],
    outside: impl Fn(usize, usize) -> bool,
) -> Vec<PdfOutlineEntry> {
    // The entries listed on these pages, each with its page's slot among them.
    let slots: HashMap<usize, usize> = pages.iter().enumerate().map(|(slot, page)| (page.index, slot)).collect();
    let listed: Vec<(usize, &Entry)> = entries
        .iter()
        .filter(|entry| entry.title.chars().any(char::is_alphabetic))
        .filter_map(|entry| slots.get(&entry.page_index).map(|&slot| (slot, entry)))
        .collect();
    if listed.is_empty() {
        return Vec::new();
    }
    let titles = titles_by_text(pages, &outside);
    let mut result = Vec::new();
    let mut first = 0;
    while first < listed.len() {
        let mut last = first + 1;
        while last < listed.len() {
            let ((previous_slot, previous), (slot, current)) = (listed[last - 1], listed[last]);
            if slot > previous_slot + 1
                || arabic(previous.locator.as_deref()).zip(arabic(current.locator.as_deref())).is_some_and(|(a, b)| b < a)
            {
                break;
            }
            last += 1;
        }
        let entries: Vec<&Entry> = listed[first..last].iter().map(|(_, entry)| *entry).collect();
        let after = listed[first..last].iter().map(|(slot, _)| *slot).max().unwrap() + 1;
        let hits = entries.iter().map(|entry| {
            let wanted = normalize_title(&entry.title);
            if wanted.chars().count() < 3 { return Vec::new(); }
            titles.get(&wanted).map_or_else(Vec::new, |slots| slots.iter().copied().filter(|slot| *slot >= after).collect())
        }).collect::<Vec<Vec<usize>>>();
        // A unique printed title is a confirmed anchor. Offset agreement is local to neighbouring anchors.
        let anchors = hits.iter().enumerate().filter_map(|(i, hits)| (hits.len() == 1).then(|| (i, hits[0]))).collect::<Vec<_>>();
        let levels = levels(&entries);
        for (i, entry) in entries.iter().enumerate() {
            let before = anchors.iter().rev().find(|(index, _)| *index < i);
            let following = anchors.iter().find(|(index, _)| *index > i);
            let offset = |anchor: &(usize, usize)| arabic(entries[anchor.0].locator.as_deref())
                .map(|printed| pages[anchor.1].index as i64 - (printed - 1));
            let agreed = match (before, following) {
                (Some(left), Some(right)) => offset(left).filter(|value| Some(*value) == offset(right)),
                (Some(left), None) => offset(left).filter(|value| anchors.iter().any(|other|
                    other.0 < left.0 && offset(other) == Some(*value))),
                (None, Some(right)) => offset(right).filter(|value| anchors.iter().any(|other|
                    other.0 > right.0 && offset(other) == Some(*value))),
                _ => None,
            };
            let candidates = hits[i].iter().filter(|&&hit|
                before.is_none_or(|(_, slot)| hit >= *slot)
                    && following.is_none_or(|(_, slot)| hit <= *slot)
                    && (hits[i].len() == 1 || agreed.zip(arabic(entry.locator.as_deref())).is_none_or(|(offset, printed)|
                        pages[hit].index as i64 == printed - 1 + offset)))
                .collect::<Vec<_>>();
            if let [&target] = candidates.as_slice() {
                result.push(PdfOutlineEntry { title: entry.title.clone(), level: levels[i], page_index: pages[target].index });
            }
        }
        first = last;
    }
    result
}
