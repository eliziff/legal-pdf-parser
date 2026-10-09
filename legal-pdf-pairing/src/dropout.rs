//! Note labels whose leading digits recognition dropped at the line's start ("10 Id." read
//! "0 Id.", "224 Julietta" read "24 Julietta"), restored from the note chain itself: ported
//! from the Open Access Journals Database's `eol_label_dropout.py`.
//!
//! Between two sequence-supported labels a and a + 2, a label-shaped line whose leading digits
//! are a proper suffix of a + 1 names itself: it is a + 1 with its prefix lost. The line must
//! start within a line's height of its neighbouring labels; a numbered heading or a citation
//! opening a wrapped line sits elsewhere. A page's first two and last line never count as
//! labels, so running heads and folios, which also climb by one a page, forge no chain.

use legal_pdf_core::model::Page;
use regex::Regex;
use std::sync::OnceLock;

const SUPPORT_WINDOW_LINES: usize = 150;
const SUPPORT_WINDOW_CANDIDATES: usize = 12;
const MAX_NOTE_VALUE: u32 = 899;
const X0_LOW: f64 = -0.5;
const X0_HIGH: f64 = 1.0;

fn label() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^(\d{1,3})(?:[\s.).\u{b0}]|$)").expect("label regex"))
}

/// Restore the dropped leading digits of note labels in place; returns how many lines it
/// repaired.
pub fn repair_label_dropouts(pages: &mut [Page]) -> usize {
    // Every non-empty line in print order: (page slot, line slot).
    let mut order = Vec::new();
    let mut edges = std::collections::HashSet::new();
    let mut slots = (0..pages.len()).collect::<Vec<_>>();
    slots.sort_by_key(|&slot| pages[slot].index);
    for page in slots {
        let mut lines = (0..pages[page].lines.len())
            .filter(|&line| !pages[page].lines[line].text.trim().is_empty())
            .collect::<Vec<_>>();
        lines.sort_by_key(|&line| pages[page].lines[line].reading_order);
        let first = order.len();
        order.extend(lines.iter().map(|&line| (page, line)));
        let last = order.len();
        edges.extend((first..last).take(2));
        if last > first {
            edges.insert(last - 1);
        }
    }
    let text = |position: usize| {
        let (page, line) = order[position];
        pages[page].lines[line].text.trim()
    };
    let candidates = (0..order.len())
        .filter(|position| !edges.contains(position))
        .filter_map(|position| {
            let value = label().captures(text(position))?[1].parse::<u32>().ok()?;
            (value > 0 && value <= MAX_NOTE_VALUE).then_some((position, value))
        })
        .collect::<Vec<_>>();
    let supported = candidates
        .iter()
        .enumerate()
        .filter(|(index, (position, value))| {
            let low = index.saturating_sub(SUPPORT_WINDOW_CANDIDATES);
            let high = (index + SUPPORT_WINDOW_CANDIDATES + 1).min(candidates.len());
            candidates[low..high].iter().any(|(other, other_value)| {
                other.abs_diff(*position) <= SUPPORT_WINDOW_LINES
                    && ((other < position && *other_value + 1 == *value)
                        || (other > position && *other_value == value + 1))
            })
        })
        .map(|(_, candidate)| *candidate)
        .collect::<Vec<_>>();
    let mut repairs = Vec::new();
    for pair in supported.windows(2) {
        let ((before, value), (after, next)) = (pair[0], pair[1]);
        if next != value + 2 || after <= before + 1 {
            continue;
        }
        let expected = (value + 1).to_string();
        for position in before + 1..after {
            let Some(kept) = label().captures(text(position)).map(|c| c[1].to_owned()) else {
                continue;
            };
            if kept == expected || !expected.ends_with(&kept) {
                continue;
            }
            let bbox = |position: usize| {
                let (page, line) = order[position];
                pages[page].lines[line].bbox
            };
            let (fired, neighbour) = (bbox(position), bbox(before)[0].min(bbox(after)[0]));
            let height = fired[3] - fired[1];
            let delta = fired[0] - neighbour;
            if height > 0.0 && (X0_LOW * height..=X0_HIGH * height).contains(&delta) {
                repairs.push((position, expected[..expected.len() - kept.len()].to_owned()));
            }
        }
    }
    for (position, prefix) in &repairs {
        let (page, line) = order[*position];
        let line = &mut pages[page].lines[line];
        let lead = line.text.chars().take_while(|c| c.is_whitespace()).count();
        let at = line.text.char_indices().nth(lead).map_or(line.text.len(), |(at, _)| at);
        line.text.insert_str(at, prefix);
        let shift = prefix.chars().count();
        for word in &mut line.words {
            if word.start == lead {
                word.text.insert_str(0, prefix);
                word.end += shift;
            } else if word.start > lead {
                word.start += shift;
                word.end += shift;
            }
        }
    }
    repairs.len()
}
