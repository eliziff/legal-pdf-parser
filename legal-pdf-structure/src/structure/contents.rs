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
use super::contents_lists::{row_bands, row_of, Entry};
use super::heading_rules::{ladder_levels, numbered};
use super::Claims;
use legal_pdf_core::model::{Page, PdfOutlineEntry};
use std::collections::{HashMap, HashSet};

/// Indents within this many points are one tier.
const INDENT_TOLERANCE: f64 = 6.0;
/// An entry's title wraps over at most this many rows.
const MAX_WRAPPED_ROWS: usize = 3;
/// The deepest level a contents list sets.
const MAX_LEVEL: usize = 6;
/// Contents subheadings use the list's own type size, independently of body prose.
pub(super) fn retain_headings(pages: &mut [Page], claims: &mut Claims) {
    for page in pages {
        let lines = page
            .lines
            .iter()
            .filter(|line| {
                !line.exclude_from_body && claims.contents_line_ids.contains(&line.id)
            })
            .collect::<Vec<_>>();
        let size = super::median(
            lines
                .iter()
                .map(|line| legal_pdf_core::line_font_size(line))
                .filter(|size| *size > 0.0)
                .collect(),
        );
        let mut headings = HashSet::new();
        for band in row_bands(&lines) {
            if row_of(band.clone()).locator.is_some() {
                continue;
            }
            headings.extend(
                band.iter()
                    .filter(|line| {
                        size > 0.0
                            && legal_pdf_core::line_font_size(line)
                                >= size * super::FONT_HEADING_SIZE_RATIO
                            && line.region_type == "heading"
                    })
                    .map(|line| line.id.clone()),
            );
        }
        for line in page
            .lines
            .iter_mut()
            .filter(|line| headings.contains(&line.id))
        {
            claims.contents_line_ids.remove(&line.id);
            line.region_type = "heading".to_owned();
        }
    }
}

/// A printed marker or completed locator starts the next entry, even when prose geometry joined it.
pub(super) fn group_rows(pages: &mut [Page], claims: &Claims) {
    for page in pages {
        if !page
            .lines
            .iter()
            .any(|line| claims.contents_line_ids.contains(&line.id))
        {
            continue;
        }
        let lines = page
            .lines
            .iter()
            .filter(|line| !line.exclude_from_body)
            .collect::<Vec<_>>();
        let mut block = page
            .lines
            .iter()
            .map(|line| line.block_index)
            .max()
            .unwrap_or(0);
        let mut open = false;
        let mut groups = HashMap::new();
        for band in row_bands(&lines) {
            if band.iter().any(|line| {
                line.region_type != "body" || !claims.contents_line_ids.contains(&line.id)
            }) {
                open = false;
                continue;
            }
            let row = row_of(band.clone());
            let first = band[0].text.trim();
            let token = first.split_whitespace().next().unwrap_or("");
            let numeric = token.chars().any(|c| c.is_ascii_digit())
                && token.chars().all(|c| c.is_ascii_digit() || c == '.');
            let locator = band.len() == 1 && band[0].bbox[0] > page.width * 0.5 && row.locator.is_some();
            let marker = !locator
                && (numeric
                    || numbered(&row.title)
                    || band.len() > 1
                        && first.chars().count() == 1
                        && first.chars().all(char::is_alphabetic));
            let continues = !marker && open;
            if !continues {
                block += 1;
            }
            for line in &band {
                groups.insert(line.id.clone(), block);
            }
            open = row.locator.is_none() && !locator;
        }
        for line in &mut page.lines {
            if let Some(&block) = groups.get(&line.id) {
                line.block_index = block;
            }
        }
    }
}

/// The level of each entry: where most entries print an enumerator, or every entry starts at one
/// indent, its level on the heading ladder the headings themselves are levelled on
/// (`heading_rules::ladder_levels`), a plain title taking the first level; else its indent's tier.
fn levels(entries: &[&Entry]) -> Vec<usize> {
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
fn titles_by_text(pages: &[Page], claims: &Claims) -> HashMap<String, Vec<usize>> {
    let mut titles = HashMap::<String, Vec<usize>>::new();
    for (slot, page) in pages.iter().enumerate() {
        let norms = page.lines.iter().map(|line| {
            let outside = !(line.exclude_from_body || matches!(line.region_type.as_str(), "header" | "footer")
                || claims.contents_line_ids.contains(&line.id)
                || claims.contents_pages.contains(&page.index));
            outside.then(|| normalize_title(&line.text))
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

/// Each local contents run uses its own confirmed destinations and pagination relationships.
pub(super) fn contents_outline(pages: &[Page], claims: &Claims) -> Vec<PdfOutlineEntry> {
    // The entries listed on these pages, each with its page's slot among them.
    let slots: HashMap<usize, usize> = pages.iter().enumerate().map(|(slot, page)| (page.index, slot)).collect();
    let listed: Vec<(usize, &Entry)> = claims
        .contents_entries
        .iter()
        .filter(|entry| entry.title.chars().any(char::is_alphabetic))
        .filter_map(|entry| slots.get(&entry.page_index).map(|&slot| (slot, entry)))
        .collect();
    if listed.is_empty() {
        return Vec::new();
    }
    let titles = titles_by_text(pages, claims);
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