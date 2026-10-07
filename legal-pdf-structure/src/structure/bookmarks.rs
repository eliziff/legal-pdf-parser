//! The document's own bookmarks reconciled with the headings found in its text.
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
//! - a set whose titles are mostly passages indexes paragraphs: it promotes no line to a heading.

use super::{
    continues_prose, has_dot_leader, inline_enumerator_re, standalone_enumerator,
    PdfPrimitiveEvidence,
};
use crate::layout::build_regions;
use legal_pdf_core::model::{Page, PdfOutlineEntry};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tier {
    Ignore,
    Skeleton,
    Full,
}

/// Same-page titles at or above this similarity refer to the same section.
const REPAIR_SIMILARITY: f64 = 0.7;
/// A detected title recurring this often is a running label; one longer than this is a lead.
const BACKFILL_DUP_MIN: usize = 3;
const BACKFILL_MAX_TITLE: usize = 100;
/// A bookmark title is found on at most this many consecutive lines.
const MAX_TITLE_LINES: usize = 4;

const CONTENT_UNITS: &[&str] = &[
    "chapter", "chap", "ch", "part", "pt", "section", "sec", "book", "appendix", "unit", "lesson",
    "章", "部", "节", "篇", "卷", "第章", "第部", "第节", "第篇", "第卷",
];
/// Words that number the constituents of a filed package.
const PACKAGE_UNITS: &[&str] = &[
    "tab",
    "exhibit",
    "schedule",
    "appendix",
    "annex",
    "attachment",
];
const ROMAN_EXCLUDED: &[&str] = &["di", "div", "li", "liv", "mi", "mix", "xi"];

#[derive(Debug, Clone)]
struct Entry {
    title: String,
    level: usize,
    /// One-based physical page.
    page: usize,
}

fn generic_title() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)^(?:(?:page|slide|folie|document\s+page)\s*)?\d+$").expect("generic title")
    })
}

fn latex_span() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\$.*?\$").expect("latex span"))
}

fn roman() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^m{0,3}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})$")
            .expect("roman numeral")
    })
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
    let lowered = latex_span()
        .replace_all(&title.to_lowercase(), "")
        .into_owned();
    lowered
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_owned)
        .collect()
}

fn is_roman(token: &str) -> bool {
    roman().is_match(token) && !ROMAN_EXCLUDED.contains(&token)
}

fn roman_to_arabic(token: &str) -> String {
    let (mut total, mut prior) = (0_i64, 0_i64);
    for character in token.chars().rev() {
        let value = match character {
            'i' => 1,
            'v' => 5,
            'x' => 10,
            'l' => 50,
            'c' => 100,
            'd' => 500,
            _ => 1000,
        };
        total = if value < prior {
            total - value
        } else {
            total + value
        };
        prior = prior.max(value);
    }
    total.to_string()
}

/// Lowercased, LaTeX dropped, punctuation stripped, roman numerals as digits.
fn normalize_title(title: &str) -> String {
    tokens(title)
        .into_iter()
        .map(|token| {
            if is_roman(&token) {
                roman_to_arabic(&token)
            } else {
                token
            }
        })
        .collect()
}

/// Every counter (digit run or roman token) collapsed to `#`.
fn title_template(title: &str) -> String {
    tokens(title)
        .into_iter()
        .map(|token| {
            if is_roman(&token) {
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

/// `difflib.SequenceMatcher(None, a, b).ratio()`.
fn similarity(a: &str, b: &str) -> f64 {
    let a = a.chars().collect::<Vec<_>>();
    let b = b.chars().collect::<Vec<_>>();
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let mut b2j = HashMap::<char, Vec<usize>>::new();
    for (j, character) in b.iter().enumerate() {
        b2j.entry(*character).or_default().push(j);
    }
    if b.len() >= 200 {
        let limit = b.len() / 100 + 1;
        b2j.retain(|_, indexes| indexes.len() <= limit);
    }
    let longest = |alo: usize, ahi: usize, blo: usize, bhi: usize| {
        let (mut besti, mut bestj, mut bestsize) = (alo, blo, 0);
        let mut j2len = HashMap::<usize, usize>::new();
        for i in alo..ahi {
            let mut next = HashMap::new();
            for &j in b2j.get(&a[i]).map_or(&[][..], Vec::as_slice) {
                if j < blo {
                    continue;
                }
                if j >= bhi {
                    break;
                }
                let k = j
                    .checked_sub(1)
                    .and_then(|prior| j2len.get(&prior))
                    .copied()
                    .unwrap_or(0)
                    + 1;
                next.insert(j, k);
                if k > bestsize {
                    (besti, bestj, bestsize) = (i + 1 - k, j + 1 - k, k);
                }
            }
            j2len = next;
        }
        while besti > alo && bestj > blo && a[besti - 1] == b[bestj - 1] {
            (besti, bestj, bestsize) = (besti - 1, bestj - 1, bestsize + 1);
        }
        while besti + bestsize < ahi
            && bestj + bestsize < bhi
            && a[besti + bestsize] == b[bestj + bestsize]
        {
            bestsize += 1;
        }
        (besti, bestj, bestsize)
    };
    let mut matched = 0;
    let mut queue = vec![(0, a.len(), 0, b.len())];
    while let Some((alo, ahi, blo, bhi)) = queue.pop() {
        let (i, j, k) = longest(alo, ahi, blo, bhi);
        if k > 0 {
            matched += k;
            if alo < i && blo < j {
                queue.push((alo, i, blo, j));
            }
            if i + k < ahi && j + k < bhi {
                queue.push((i + k, ahi, j + k, bhi));
            }
        }
    }
    2.0 * matched as f64 / (a.len() + b.len()) as f64
}

/// Entries must walk forward through the document; levels are re-stacked over the survivors.
fn validate(outline: &[PdfOutlineEntry], pages: usize) -> Vec<Entry> {
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

fn restack(entries: &mut [Entry]) {
    let mut stack = Vec::<usize>::new();
    for entry in entries {
        while stack.last().is_some_and(|level| *level >= entry.level) {
            stack.pop();
        }
        stack.push(entry.level);
        entry.level = stack.len();
    }
}

fn classify(entries: &[Entry], pages: usize) -> Tier {
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

/// The source lines a bookmark's title was found on.
#[derive(Debug, Clone)]
struct Anchor {
    page_slot: usize,
    lines: Vec<usize>,
    existing: bool,
}

fn same_title(found: &str, title: &str, heading: bool) -> bool {
    if found.is_empty() || title.is_empty() {
        return false;
    }
    if found == title {
        return true;
    }
    if heading {
        return found.ends_with(title)
            || title.ends_with(found)
            || similarity(found, title) >= REPAIR_SIMILARITY;
    }
    // Body lines carry the title whole, or with a numbering label only one side prints.
    let labelled = |long: &str, short: &str| {
        long.strip_suffix(short)
            .is_some_and(|label| label_remainder().is_match(label))
    };
    labelled(found, title) || labelled(title, found)
}

pub(super) fn reconcile_bookmarks(
    pages: &mut [Page],
    outline: &[PdfOutlineEntry],
    primitives: &mut PdfPrimitiveEvidence,
) {
    let mut entries = validate(outline, pages.len());
    let tier = classify(&entries, pages.len());
    if tier == Tier::Ignore {
        return;
    }
    let slot_of_page = pages
        .iter()
        .enumerate()
        .map(|(slot, page)| (page.index, slot))
        .collect::<HashMap<_, _>>();
    let contents_pages = pages
        .iter()
        .filter(|page| {
            primitives.contents_pages.contains(&page.index)
                || page
                    .lines
                    .iter()
                    .any(|line| primitives.contents_line_ids.contains(&line.id))
        })
        .map(|page| page.index)
        .collect::<HashSet<_>>();
    // A bookmark on a contents page holds no other: its subtree rises a level.
    let mut holder: Option<usize> = None;
    for entry in &mut entries {
        if holder.is_some_and(|level| entry.level <= level) {
            holder = None;
        }
        if holder.is_some() {
            entry.level -= 1;
        } else if contents_pages.contains(&(entry.page - 1)) {
            holder = Some(entry.level);
        }
    }
    restack(&mut entries);
    let promotes = 2 * entries
        .iter()
        .filter(|entry| entry.title.chars().count() > BACKFILL_MAX_TITLE)
        .count()
        < entries.len();

    let norms = pages
        .iter()
        .map(|page| {
            page.lines
                .iter()
                .map(|line| normalize_title(&line.text))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let eligible = |page: &Page, slot: usize| {
        let line = &page.lines[slot];
        !line.exclude_from_body
            && line.note_region_mode.is_empty()
            && matches!(line.region_type.as_str(), "body" | "heading")
            && !primitives.translation_line_ids.contains(&line.id)
            && !primitives.contents_line_ids.contains(&line.id)
            && !has_dot_leader(&line.text)
    };
    let mut claimed = HashSet::<(usize, usize)>::new();
    let mut anchors = vec![None::<Anchor>; entries.len()];
    let mut last = None::<(usize, usize)>;
    for (index, entry) in entries.iter().enumerate() {
        let title = normalize_title(&entry.title);
        let Some(&target) = slot_of_page.get(&(entry.page - 1)) else {
            continue;
        };
        let window = [Some(target), target.checked_add(1), target.checked_sub(1)];
        // A title read whole wins over one read as a fragment of a longer heading.
        let found = [true, false].into_iter().find_map(|whole| {
            window
                .into_iter()
                .flatten()
                .enumerate()
                .find_map(|(order, page_slot)| {
                    let page = pages.get(page_slot)?;
                    if order > 0 && contents_pages.contains(&page.index) {
                        return None;
                    }
                    (0..page.lines.len()).find_map(|start| {
                        let block = page.lines[start].block_index;
                        let mut joined = String::new();
                        let mut heading = true;
                        for end in start..(start + MAX_TITLE_LINES).min(page.lines.len()) {
                            if claimed.contains(&(page_slot, end)) || !eligible(page, end) {
                                return None;
                            }
                            joined.push_str(&norms[page_slot][end]);
                            // One detected heading, not several read together.
                            heading &= page.lines[end].region_type == "heading"
                                && page.lines[end].block_index == block;
                            let matched = if whole {
                                joined == title
                            } else {
                                same_title(&joined, &title, heading)
                            };
                            if matched {
                                return Some(Anchor {
                                    page_slot,
                                    lines: (start..=end).collect(),
                                    existing: heading,
                                });
                            }
                        }
                        None
                    })
                })
        });
        let Some(anchor) = found else {
            continue;
        };
        let position = (anchor.page_slot, anchor.lines[0]);
        if last.is_some_and(|last| position <= last) {
            continue;
        }
        if !anchor.existing {
            let page = &pages[anchor.page_slot];
            let first = &page.lines[anchor.lines[0]];
            let end = *anchor.lines.last().unwrap();
            let text = anchor
                .lines
                .iter()
                .map(|slot| page.lines[*slot].text.trim())
                .collect::<Vec<_>>()
                .join(" ");
            let carried_on = page.lines.get(end + 1).is_some_and(|next| {
                next.region_type == "body"
                    && next.block_index == page.lines[end].block_index
                    && next.text.trim_start().starts_with(char::is_lowercase)
            });
            // A SKELETON frame inserts deeper bookmarks only within its chapters.
            let chaptered = tier == Tier::Full
                || entries
                    .iter()
                    .any(|chapter| chapter.level == 1 && chapter.page <= entry.page);
            if !promotes
                || !chaptered
                || generic_title().is_match(&entry.title)
                || text.ends_with(',')
                || text.starts_with(char::is_lowercase)
                || continues_prose(page, first)
                || carried_on
            {
                continue;
            }
        }
        last = Some(position);
        claimed.extend(anchor.lines.iter().map(|slot| (anchor.page_slot, *slot)));
        anchors[index] = Some(anchor);
    }
    if anchors.iter().all(Option::is_none) {
        return;
    }

    // An anchored title is a heading of its own. Within a heading it cuts only where a numbered
    // heading starts; otherwise it takes the whole heading.
    let levels = &primitives.heading_levels;
    for anchor in anchors.iter_mut().flatten() {
        let page = &mut pages[anchor.page_slot];
        let block = page.lines[anchor.lines[0]].block_index;
        if anchor.existing {
            let same = |slot: usize, page: &Page| {
                page.lines[slot].region_type == "heading" && page.lines[slot].block_index == block
            };
            let numbered = |slot: usize, page: &Page| {
                let text = page.lines[slot].text.trim();
                inline_enumerator_re().is_match(text) || standalone_enumerator(text)
            };
            let mut start = anchor.lines[0];
            if !numbered(start, page) {
                while start > 0 && same(start - 1, page) {
                    start -= 1;
                }
            }
            let mut end = *anchor.lines.last().unwrap();
            while end + 1 < page.lines.len() && same(end + 1, page) && !numbered(end + 1, page) {
                end += 1;
            }
            anchor.lines = (start..=end).collect();
        }
        let fresh = page
            .lines
            .iter()
            .map(|line| line.block_index)
            .max()
            .unwrap_or(0)
            + 1;
        for slot in &anchor.lines {
            let line = &mut page.lines[*slot];
            line.region_type = "heading".to_owned();
            line.block_index = fresh;
        }
    }

    // Heading units, as regions will group them, in reading order.
    struct Unit {
        position: (usize, usize),
        line_ids: Vec<String>,
        text: String,
        level: Option<usize>,
    }
    let mut units = Vec::<Unit>::new();
    let mut unit_at = HashMap::<(usize, usize), usize>::new();
    for (page_slot, page) in pages.iter().enumerate() {
        for (slot, line) in page.lines.iter().enumerate() {
            if line.region_type != "heading" {
                continue;
            }
            let continues = slot > 0
                && page.lines[slot - 1].region_type == "heading"
                && page.lines[slot - 1].block_index == line.block_index;
            if !continues {
                units.push(Unit {
                    position: (page_slot, slot),
                    line_ids: Vec::new(),
                    text: String::new(),
                    level: None,
                });
            }
            let unit = units.last_mut().expect("unit");
            unit.line_ids.push(line.id.clone());
            unit.text = format!("{} {}", unit.text, line.text.trim())
                .trim()
                .to_owned();
            if let Some(level) = levels.get(&line.id) {
                unit.level = Some(unit.level.map_or(*level, |prior: usize| prior.min(*level)));
            }
            unit_at.insert((page_slot, slot), units.len() - 1);
        }
    }
    let entry_unit = anchors
        .iter()
        .map(|anchor| {
            anchor
                .as_ref()
                .and_then(|anchor| unit_at.get(&(anchor.page_slot, anchor.lines[0])).copied())
        })
        .collect::<Vec<_>>();
    let promoted = anchors
        .iter()
        .zip(&entry_unit)
        .filter_map(|(anchor, unit)| anchor.as_ref().filter(|anchor| !anchor.existing).and(*unit))
        .collect::<HashSet<_>>();

    // The frame: every bookmark (FULL), or the top-level ones (SKELETON). Deeper SKELETON entries
    // found as existing headings stay detected; ones found in body text are inserted.
    let framed = |entry: &Entry| tier == Tier::Full || entry.level == 1;
    let mut frame_units = HashMap::<usize, usize>::new();
    let mut levels_out = HashMap::<usize, Option<usize>>::new();
    {
        let mut stack = Vec::<usize>::new();
        for (index, entry) in entries.iter().enumerate() {
            let Some(unit) = entry_unit[index].filter(|_| framed(entry)) else {
                continue;
            };
            while stack.last().is_some_and(|level| *level >= entry.level) {
                stack.pop();
            }
            stack.push(entry.level);
            frame_units.insert(index, unit);
            levels_out.insert(unit, Some(stack.len()));
        }
    }
    let inserted = entries
        .iter()
        .enumerate()
        .filter(|(index, entry)| {
            !framed(entry) && entry_unit[*index].is_some_and(|unit| promoted.contains(&unit))
        })
        .map(|(index, _)| index)
        .collect::<HashSet<_>>();
    let parents = {
        let mut stack = Vec::<usize>::new();
        entries
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                while stack
                    .last()
                    .is_some_and(|prior| entries[*prior].level >= entry.level)
                {
                    stack.pop();
                }
                let parents = stack.clone();
                stack.push(index);
                parents
            })
            .collect::<Vec<_>>()
    };
    // Where each frame entry opens: its heading, else the top of its page.
    let frame = entries
        .iter()
        .enumerate()
        .filter(|(_, entry)| framed(entry))
        .map(|(index, entry)| {
            let opens = frame_units
                .get(&index)
                .map(|unit| units[*unit].position)
                .unwrap_or((
                    slot_of_page
                        .get(&(entry.page - 1))
                        .copied()
                        .unwrap_or(usize::MAX),
                    0,
                ));
            (index, opens)
        })
        .collect::<Vec<_>>();

    let detected = (0..units.len())
        .filter(|unit| !levels_out.contains_key(unit) && !promoted.contains(unit))
        .collect::<Vec<_>>();
    let mut title_counts = HashMap::<String, usize>::new();
    for unit in &detected {
        *title_counts
            .entry(normalize_title(&units[*unit].text))
            .or_default() += 1;
    }
    let mut groups = HashMap::<usize, Vec<usize>>::new();
    for unit in detected {
        let norm = normalize_title(&units[unit].text);
        if tier == Tier::Full
            && (!norm.is_empty() && title_counts[&norm] >= BACKFILL_DUP_MIN
                || units[unit].text.chars().count() > BACKFILL_MAX_TITLE)
        {
            levels_out.insert(unit, None);
            continue;
        }
        let mut active = None;
        for (index, opens) in &frame {
            if *opens > units[unit].position {
                break;
            }
            active = Some(*index);
        }
        // Under the nearest frame entry found in the text; headings before the frame, or under
        // none found, keep their own levels.
        let anchor = active.and_then(|index| {
            std::iter::once(index)
                .chain(parents[index].iter().rev().copied())
                .find(|candidate| frame_units.contains_key(candidate))
        });
        if let Some(anchor) = anchor {
            groups.entry(anchor).or_default().push(unit);
        }
    }
    // A numbered frame heading places the detected ladder around itself: a bookmark set can
    // omit a sibling ("III. Facts" between bookmarked "II." and "IV."), which then stays one.
    for (anchor, members) in &groups {
        let base = levels_out[&frame_units[anchor]].expect("frame level");
        let top = members
            .iter()
            .filter_map(|unit| units[*unit].level)
            .min()
            .unwrap_or(1);
        let own = units[frame_units[anchor]].level;
        for unit in members {
            let level = match (units[*unit].level, own) {
                (Some(level), Some(own)) => (base + level).saturating_sub(own).max(1),
                (Some(level), None) => base + level + 1 - top,
                (None, _) => base + 1,
            };
            levels_out.insert(*unit, Some(level));
        }
    }
    let mut inserted = inserted.into_iter().collect::<Vec<_>>();
    inserted.sort_unstable();
    for index in inserted {
        let unit = entry_unit[index].expect("inserted unit");
        let parent = parents[index]
            .iter()
            .rev()
            .find_map(|parent| {
                entry_unit[*parent].and_then(|unit| levels_out.get(&unit).copied().flatten())
            })
            .or_else(|| {
                // Else the chapter whose pages it falls in.
                frame
                    .iter()
                    .take_while(|(_, opens)| opens.0 <= units[unit].position.0)
                    .last()
                    .and_then(|(chapter, _)| frame_units.get(chapter))
                    .and_then(|unit| levels_out.get(unit).copied().flatten())
            });
        levels_out.insert(unit, Some(parent.map_or(1, |level| level + 1)));
    }

    build_regions(pages);
    let levels = &mut primitives.heading_levels;
    for (unit, level) in levels_out {
        for line_id in &units[unit].line_ids {
            match level {
                Some(level) => levels.insert(line_id.clone(), level),
                None => levels.remove(line_id),
            };
        }
    }
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
