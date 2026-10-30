//! The document's own bookmarks as an outline of it: which to trust, how deep each sits, and how a
//! bookmark's title is matched with the lines that print it (`decide::anchors` finds them).
//!
//! Validation, tiering, title normalization and the bookmark-frame merges are ported from
//! VectifyAI/PageIndex `pageindex/flash/embedded_toc.py` (`validate_bookmarks`,
//! `classify_bookmarks`, `_normalize_title`, `_title_template`, `_same_heading`,
//! `_find_entry_node`, `merge_bookmark_skeleton`, `merge_bookmark_tree`) at
//! 6d23caf416858f2ca136840305d1f479a86f6ef7, with Python's `difflib.SequenceMatcher.ratio`.
//! Copyright (c) 2025 Vectify AI; MIT License, reproduced at the end of this file.
//!
//! PageIndex merges page ranges. Here a bookmark is a navigation hint that becomes structure only
//! where its title is found as lines of its page: those lines become (or already are) the heading,
//! and the bookmark frame sets heading levels. Detected headings keep their own ladder below the
//! frame entry they fall under. Deliberate legal adaptations, apart from line anchoring:
//! - a `Tab N` / `Exhibit N` set is package navigation, not an enumeration to ignore;
//! - a bookmark on a contents page holds no other bookmark;
//! - constituent documents and parallel translations keep separate anchors and bookmark frames;
//! - a set whose titles are mostly passages indexes paragraphs: it promotes no line to a heading.

use legal_pdf_core::model::PdfOutlineEntry;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

/// Words that number the constituents of a filed package.
const PACKAGE_UNITS: &[&str] = &["tab", "exhibit", "schedule", "appendix", "annex", "attachment"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tier {
    Ignore,
    Skeleton,
    Full,
}

/// Same-page titles at or above this similarity refer to the same section.
const REPAIR_SIMILARITY: f64 = 0.7;
/// A title longer than this is a lead, not a heading.
pub(crate) const BACKFILL_MAX_TITLE: usize = 100;
/// A bookmark title is found on at most this many consecutive lines.
pub(crate) const MAX_TITLE_LINES: usize = 4;

const CONTENT_UNITS: &[&str] = &["chapter", "chap", "ch", "part", "pt", "section", "sec", "book", "appendix"];
const ROMAN_EXCLUDED: &[&str] = &["di", "div", "li", "liv", "mi", "mix", "xi"];

#[derive(Debug, Clone)]
pub(crate) struct Entry {
    pub(crate) title: String,
    pub(crate) level: usize,
    /// One-based physical page.
    pub(crate) page: usize,
}

/// A bookmark named for its page alone ("12", "Page 12").
pub(crate) fn generic_title() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)^(?:page\s*)?\d+$").expect("generic title"))
}

/// A numbering label left over beside a title: "1", "a", "part4", "tab12".
fn label_remainder() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"^(?:part|chapter|article|section|division|title|book|schedule|appendix|annex|tab|exhibit)?(?:[0-9]+[a-z]?|[a-z])$",
        )
        .expect("label remainder")
    })
}

fn tokens(title: &str) -> Vec<String> {
    title
        .to_lowercase()
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_owned)
        .collect()
}

fn roman(token: &str) -> Option<u32> {
    legal_pdf_core::roman_value(token).filter(|_| !ROMAN_EXCLUDED.contains(&token))
}

/// Lowercased, LaTeX dropped, punctuation stripped, roman numerals as digits.
pub(crate) fn normalize_title(title: &str) -> String {
    tokens(title)
        .into_iter()
        .map(|token| {
            roman(&token).map_or(token, |value| value.to_string())
        })
        .collect()
}

/// Every counter (digit run or roman token) collapsed to `#`.
fn title_template(title: &str) -> String {
    tokens(title)
        .into_iter()
        .map(|token| {
            if roman(&token).is_some() {
                "#".to_owned()
            } else {
                let mut out = String::new();
                let mut digits = false;
                for character in token.chars() {
                    if character.is_ascii_digit() {
                        if !digits {
                            out.push('#');
                        }
                        digits = true;
                    } else {
                        out.push(character);
                        digits = false;
                    }
                }
                out
            }
        })
        .collect()
}

/// How nearly two normalized titles are one, as the share of their character pairs they hold in
/// common (Dice's coefficient): a recognizer's slip or a dropped word leaves it high.
fn similarity(a: &str, b: &str) -> f64 {
    let pairs = |text: &str| text.chars().collect::<Vec<_>>().windows(2).map(|pair| (pair[0], pair[1])).collect::<Vec<_>>();
    let (left, right) = (pairs(a), pairs(b));
    if left.is_empty() || right.is_empty() {
        return f64::from(u8::from(a == b));
    }
    let mut counts = HashMap::<(char, char), usize>::new();
    for pair in &right {
        *counts.entry(*pair).or_default() += 1;
    }
    let common = left.iter().filter(|pair| counts.get_mut(*pair).is_some_and(|count| *count > 0 && { *count -= 1; true })).count();
    2.0 * common as f64 / (left.len() + right.len()) as f64
}

/// Entries must walk forward through the document; levels are re-stacked over the survivors.
pub(crate) fn validate(outline: &[PdfOutlineEntry], pages: usize) -> Vec<Entry> {
    let mut kept = Vec::new();
    let mut last_page = 0;
    for entry in outline {
        let page = entry.page_index + 1;
        if !(1..=pages).contains(&page) || page < last_page {
            continue;
        }
        kept.push(Entry {
            title: entry.title.clone(),
            level: entry.level,
            page,
        });
        last_page = page;
    }
    restack(&mut kept);
    kept
}

/// Repeated package labels name attached documents; ordinary chapter bookmarks do not.
pub(crate) fn package_targets(outline: &[PdfOutlineEntry], pages: usize) -> Vec<(usize, String)> {
    let entries = validate(outline, pages);
    let authorities = authorities_level(&entries);
    let entries = entries
        .into_iter()
        .filter_map(|entry| {
            let words = tokens(&entry.title);
            let unit = words.first()?;
            let counter = words.get(1)?;
            let kind = if PACKAGE_UNITS.contains(&unit.as_str())
                && (label_remainder().is_match(counter) || roman(counter).is_some())
            {
                unit.clone()
            } else if authorities == Some(entry.level)
                && legal_pdf_core::layout_analysis()
                    .numbering_marker(entry.title.split_whitespace().next()?, None)
                    .is_some()
            {
                "authority".to_owned()
            } else {
                return None;
            };
            Some((entry.page - 1, entry.level, kind))
        })
        .collect::<Vec<_>>();
    let mut families = HashMap::<(&str, usize), HashSet<usize>>::new();
    for (page, level, unit) in &entries {
        families.entry((unit, *level)).or_default().insert(*page);
    }
    entries
        .iter()
        .filter(|(page, level, unit)| *page > 0 && families[&(unit.as_str(), *level)].len() > 1)
        .map(|(page, _, unit)| (*page, unit.clone()))
        .collect()
}

pub(crate) fn restack(entries: &mut [Entry]) {
    let mut stack = Vec::<usize>::new();
    for entry in entries {
        while stack.last().is_some_and(|level| *level >= entry.level) {
            stack.pop();
        }
        stack.push(entry.level);
        entry.level = stack.len();
    }
}

pub(crate) fn classify(entries: &[Entry], pages: usize) -> Tier {
    if entries.len() < 3 {
        return Tier::Ignore;
    }
    let generic = entries
        .iter()
        .filter(|entry| generic_title().is_match(&entry.title))
        .count();
    if 2 * generic >= entries.len() {
        return Tier::Ignore;
    }
    // The most common template, the first seen among equals.
    let mut counts = Vec::<(String, usize)>::new();
    for entry in entries {
        let template = title_template(&entry.title);
        match counts.iter_mut().find(|(seen, _)| *seen == template) {
            Some((_, count)) => *count += 1,
            None => counts.push((template, 1)),
        }
    }
    let (top_template, top_count) =
        counts.iter().fold(
            &counts[0],
            |best, item| if item.1 > best.1 { item } else { best },
        );
    let unit = top_template.replace('#', "");
    let counters = top_template.matches('#').count();
    if top_template.contains('#')
        && 2 * top_count >= entries.len()
        && (counters >= 2 || !CONTENT_UNITS.contains(&unit.as_str()))
        && !(counters == 1 && PACKAGE_UNITS.contains(&unit.as_str()))
    {
        return Tier::Ignore;
    }
    let distinct = entries
        .iter()
        .map(|entry| normalize_title(&entry.title))
        .collect::<HashSet<_>>();
    if 4 * distinct.len() <= entries.len() {
        return Tier::Ignore;
    }
    let max_level = entries.iter().map(|entry| entry.level).max().unwrap_or(0);
    if max_level >= 2 && entries.len() >= 6.max(pages / 20) {
        Tier::Full
    } else {
        Tier::Skeleton
    }
}

pub(crate) fn same_title(found: &str, title: &str, heading: bool) -> bool {
    if found.is_empty() || title.is_empty() {
        return false;
    }
    if found == title {
        return true;
    }
    if heading {
        return found.ends_with(title) || title.ends_with(found) || similarity(found, title) >= REPAIR_SIMILARITY;
    }
    // Body lines carry the title whole, or with a numbering label only one side prints.
    let labelled = |long: &str, short: &str| {
        long.strip_suffix(short)
            .is_some_and(|label| label_remainder().is_match(label))
    };
    labelled(found, title) || labelled(title, found)
}

/// Whether a document's bookmarks outline its divisions at all (not generic titles, one numbered
/// template, or too few).
pub(crate) fn usable(outline: &[PdfOutlineEntry], pages: usize) -> bool {
    classify(&validate(outline, pages), pages) != Tier::Ignore
}

/// Whether a document's own bookmarks set out every level of it, so that a heading they do not
/// name is none: one tree walking forward through the pages (not a bilingual act's two, the
/// second starting over at its first page). A book of authorities' bookmarks name its tabs and the
/// passages cited in them, none of its authorities' headings.
pub(crate) fn complete(outline: &[PdfOutlineEntry], pages: usize) -> bool {
    let entries = validate(outline, pages);
    entries.len() * 10 >= outline.len() * 9 && classify(&entries, pages) == Tier::Full && authorities_level(&entries).is_none()
}

/// The level of a book or bundle of authorities' bookmark on its first page, where its bookmarks
/// open with one.
fn authorities_level(entries: &[Entry]) -> Option<usize> {
    entries
        .iter()
        .find(|entry| {
            entry.page == 1
                && tokens(&entry.title).windows(3).any(|words| words == ["book", "of", "authorities"] || words == ["bundle", "of", "authorities"])
        })
        .map(|entry| entry.level)
}

/// The document's bookmarks and its printed contents list as one outline. A listed entry the
/// bookmarks name, whatever numbering label one side prints ("1 Short Title" / "Short Title"),
/// confirms theirs; one they do not name stands after the entry before it, at its listed level
/// counted from the nearest entry before it that both name (its own listed level where none does).
pub(crate) fn merge_outlines(bookmarks: &[PdfOutlineEntry], listed: &[PdfOutlineEntry]) -> Vec<PdfOutlineEntry> {
    let mut merged = bookmarks.to_vec();
    let mut cursor: Option<usize> = None;
    let mut anchor: Option<(usize, usize)> = None;
    for entry in listed {
        let title = normalize_title(&entry.title);
        if let Some(at) = merged.iter().position(|bookmark| same_title(&normalize_title(&bookmark.title), &title, false)) {
            cursor = Some(at);
            anchor = Some((entry.level, merged[at].level));
            continue;
        }
        let mut at = cursor.map_or(0, |cursor| cursor + 1);
        while merged.get(at).is_some_and(|next| next.page_index < entry.page_index) {
            at += 1;
        }
        let level = anchor.map_or(entry.level, |(listed, bookmarked)| (bookmarked + entry.level).saturating_sub(listed).max(1));
        merged.insert(at, PdfOutlineEntry { title: entry.title.clone(), level, page_index: entry.page_index });
        cursor = Some(at);
    }
    merged
}

// PageIndex license, for the code ported above:
//
// MIT License
//
// Copyright (c) 2025 Vectify AI
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
