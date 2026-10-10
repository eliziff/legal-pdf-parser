//! The document's printed contents list read as its outline: each entry's title, the page it
//! names and its depth, for the bookmark reconciliation to find on that page. Ported from the Open
//! Access Journals Database's contents-outline witness (`toc_outline_witness.py`): an entry is
//! confirmed on its destination page, where its title is a heading already or a line that may
//! become one, and its depth stands only where the list's own hierarchy is coherent.
//!
//! Read over the whole document: the printed page an entry names maps to a physical page by the
//! offset the entries' titles agree on where they are found after the list; with no agreement (a
//! factum's contents names paragraphs, not pages) an entry goes to the one page that prints its
//! title. Depth comes from the list's indents, or, where every entry starts at one indent, from the
//! order in which its enumerator families first appear (PART I, then A., then 1.).

use super::bookmarks::normalize_title;
use super::graph::contents_leader_re;
use super::PdfPrimitiveEvidence;
use legal_pdf_core::model::{Line, Page, PdfOutlineEntry};
use regex::Regex;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Indents within this many points are one tier.
const INDENT_TOLERANCE: f64 = 6.0;
/// An entry's title wraps over at most this many rows.
const MAX_WRAPPED_ROWS: usize = 3;
/// The deepest level a contents list sets.
const MAX_LEVEL: usize = 6;

/// A printed locator closing a row: a page number in arabic or roman figures.
fn locator_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)^(?:p(?:age|p?\.)?\s*)?(\d{1,4}|[ivxlcdm]{1,7})$").expect("locator regex")
    })
}

/// A leader and the locator after it, at a row's end.
fn leader_tail_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?:\s*(?:\.\s?){2,}|\s*_{3,}|\s*\u{2026}+)\s*(\S{1,8})\s*$")
            .expect("leader regex")
    })
}

/// The enumerator family an entry's title opens with.
fn family(title: &str) -> &'static str {
    static FAMILIES: OnceLock<Vec<(&'static str, Regex)>> = OnceLock::new();
    let families = FAMILIES.get_or_init(|| {
        [
            ("unit", r"(?i)^(?:part|chapter|title|book|division)\b"),
            ("roman", r"^[IVXLC]{1,6}[.)]\s"),
            ("alpha", r"^[A-Z][.)]\s"),
            ("numeric", r"^\d{1,3}[.)]\s"),
            ("dotted", r"^\d{1,3}(?:\.\d{1,3})+\.?\s"),
            ("lower", r"^\(?[a-z]{1,3}[.)]\s"),
            ("paren", r"^\(\d{1,3}\)\s"),
        ]
        .into_iter()
        .map(|(name, pattern)| (name, Regex::new(pattern).expect("family regex")))
        .collect()
    });
    families
        .iter()
        .find(|(_, pattern)| pattern.is_match(title))
        .map_or("", |(name, _)| name)
}

/// A row of the list: its lines read left to right as one text, where it starts, and the locator
/// set apart at its end, if any.
struct Row {
    page_slot: usize,
    left: f64,
    title: String,
    locator: Option<String>,
}

fn rows_of(page: &Page, page_slot: usize, lines: &[&Line]) -> Vec<Row> {
    // Lines sharing a baseline band are one row: a grid sets the title and its page apart.
    let mut bands: Vec<Vec<&Line>> = Vec::new();
    for line in lines {
        let middle = (line.bbox[1] + line.bbox[3]) / 2.0;
        match bands.iter_mut().find(|band| {
            let first = band[0];
            let height = (first.bbox[3] - first.bbox[1]).max(1.0);
            ((first.bbox[1] + first.bbox[3]) / 2.0 - middle).abs() <= height / 2.0
        }) {
            Some(band) => band.push(line),
            None => bands.push(vec![line]),
        }
    }
    bands.sort_by(|a, b| a[0].bbox[1].total_cmp(&b[0].bbox[1]));
    bands
        .into_iter()
        .map(|mut band| {
            band.sort_by(|a, b| a.bbox[0].total_cmp(&b.bbox[0]));
            let left = band[0].bbox[0];
            // A locator alone at the row's right, apart from its title.
            let apart = band.len() > 1
                && locator_re().is_match(band.last().unwrap().text.trim())
                && band.last().unwrap().bbox[0] > page.width * 0.5;
            let (title_lines, locator) = if apart {
                let last = band.pop().unwrap();
                (band, Some(last.text.trim().to_owned()))
            } else {
                (band, None)
            };
            let mut title = title_lines
                .iter()
                .map(|line| line.text.trim())
                .collect::<Vec<_>>()
                .join(" ");
            let mut locator = locator;
            if locator.is_none() && contents_leader_re().is_match(&title) {
                if let Some(captures) = leader_tail_re().captures(&title) {
                    let found = captures[1].to_owned();
                    if locator_re().is_match(&found) {
                        let cut = captures.get(0).unwrap().start();
                        title.truncate(cut);
                        locator = Some(found);
                    }
                }
            }
            let title = title
                .trim_end_matches(|c: char| {
                    c == '.' || c == '_' || c == '\u{2026}' || c.is_whitespace()
                })
                .trim()
                .to_owned();
            Row {
                page_slot,
                left,
                title,
                locator,
            }
        })
        .collect()
}

/// One entry of the list: its title, the locator it prints, where it starts and on which page of
/// the list.
struct Listed {
    title: String,
    locator: String,
    left: f64,
    page_slot: usize,
}

fn listed_entries(pages: &[Page], primitives: &PdfPrimitiveEvidence) -> Vec<Listed> {
    let mut entries = Vec::new();
    for (page_slot, page) in pages.iter().enumerate() {
        let grid = primitives.contents_pages.contains(&page.index);
        let lines = page
            .lines
            .iter()
            .filter(|line| {
                !line.exclude_from_body
                    && !matches!(line.region_type.as_str(), "header" | "footer")
                    && !line.text.trim().is_empty()
                    && (grid || primitives.contents_line_ids.contains(&line.id))
            })
            .collect::<Vec<_>>();
        if lines.is_empty() {
            continue;
        }
        // Rows with no locator are a title wrapping onto the next row that has one.
        let mut pending = Vec::<Row>::new();
        for row in rows_of(page, page_slot, &lines) {
            match row.locator {
                Some(locator) => {
                    let (left, title) = match pending.first() {
                        Some(first) => (
                            first.left.min(row.left),
                            pending
                                .iter()
                                .map(|row| row.title.as_str())
                                .chain([row.title.as_str()])
                                .collect::<Vec<_>>()
                                .join(" "),
                        ),
                        None => (row.left, row.title.clone()),
                    };
                    pending.clear();
                    if title.chars().any(char::is_alphabetic) {
                        entries.push(Listed {
                            title,
                            locator,
                            left,
                            page_slot: row.page_slot,
                        });
                    }
                }
                None => {
                    pending.push(row);
                    if pending.len() > MAX_WRAPPED_ROWS {
                        pending.remove(0);
                    }
                }
            }
        }
    }
    entries
}

/// The level of each entry: its indent's tier, or where all start at one indent, the order in
/// which their enumerator families first appear; a plain title under an enumerated list takes the
/// first level.
fn levels(entries: &[Listed]) -> Vec<usize> {
    let mut tiers = entries.iter().map(|entry| entry.left).collect::<Vec<_>>();
    tiers.sort_by(f64::total_cmp);
    tiers.dedup_by(|a, b| (*a - *b).abs() <= INDENT_TOLERANCE);
    if tiers.len() > 1 {
        return entries
            .iter()
            .map(|entry| {
                let tier = tiers
                    .iter()
                    .position(|tier| entry.left <= tier + INDENT_TOLERANCE)
                    .unwrap_or(tiers.len() - 1);
                (tier + 1).min(MAX_LEVEL)
            })
            .collect();
    }
    let mut order = Vec::<&str>::new();
    entries
        .iter()
        .map(|entry| {
            let name = family(&entry.title);
            if name.is_empty() {
                return 1;
            }
            let at = order
                .iter()
                .position(|seen| *seen == name)
                .unwrap_or_else(|| {
                    order.push(name);
                    order.len() - 1
                });
            (at + 1).min(MAX_LEVEL)
        })
        .collect()
}

fn arabic(locator: &str) -> Option<i64> {
    let digits = locator.trim_start_matches(|c: char| !c.is_ascii_digit());
    digits.parse::<i64>().ok().filter(|_| !digits.is_empty())
}

/// The document's contents list as an outline, entries in the list's order; empty when the
/// document prints none.
pub(super) fn contents_outline(
    pages: &[Page],
    primitives: &PdfPrimitiveEvidence,
) -> Vec<PdfOutlineEntry> {
    let entries = listed_entries(pages, primitives);
    if entries.len() < 3 {
        return Vec::new();
    }
    let levels = levels(&entries);
    let after = entries
        .iter()
        .map(|entry| entry.page_slot)
        .max()
        .unwrap_or(0)
        + 1;
    // Where each title is printed after the list: the pages holding it as a line, or as up to three
    // consecutive lines read together.
    let found = |title: &str| {
        let wanted = normalize_title(title);
        if wanted.chars().count() < 3 {
            return Vec::new();
        }
        let mut hits = Vec::new();
        for (slot, page) in pages.iter().enumerate().skip(after) {
            let norms = page
                .lines
                .iter()
                .map(|line| normalize_title(&line.text))
                .collect::<Vec<_>>();
            let present = (0..norms.len()).any(|start| {
                let mut joined = String::new();
                (start..(start + 3).min(norms.len())).any(|end| {
                    joined.push_str(&norms[end]);
                    joined == wanted
                })
            });
            if present {
                hits.push(slot);
            }
        }
        hits
    };
    let hits = entries
        .iter()
        .map(|entry| found(&entry.title))
        .collect::<Vec<_>>();
    // The offset of physical from printed pages the found titles agree on.
    let mut votes = HashMap::<i64, usize>::new();
    for (entry, hits) in entries.iter().zip(&hits) {
        if let Some(printed) = arabic(&entry.locator) {
            for slot in hits {
                *votes
                    .entry(pages[*slot].index as i64 - (printed - 1))
                    .or_default() += 1;
            }
        }
    }
    let agreed = votes
        .into_iter()
        .filter(|(_, count)| *count >= 2)
        .max_by_key(|(offset, count)| (*count, -offset.abs()))
        .map(|(offset, _)| offset);
    entries
        .iter()
        .zip(levels)
        .zip(&hits)
        .filter_map(|((entry, level), hits)| {
            let page_index = match (agreed, arabic(&entry.locator)) {
                (Some(offset), Some(printed)) => usize::try_from(printed - 1 + offset).ok()?,
                _ if hits.len() == 1 => pages[hits[0]].index,
                _ => return None,
            };
            (page_index < pages.len()).then(|| PdfOutlineEntry {
                title: entry.title.clone(),
                level,
                page_index,
            })
        })
        .collect()
}
