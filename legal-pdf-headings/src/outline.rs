//! PageIndex Flash's heading depth and outline stack.
//!
//! Ported from VectifyAI/PageIndex `pageindex/flash/outline_assembly/style_context.py`
//! (`compare_heading_depth`, `_apply_heading_to_state`), `selection.py` (`find_parent_heading`,
//! the font-distance rule of `should_reject_heading`) at 6d23caf416858f2ca136840305d1f479a86f6ef7.
//! Copyright (c) 2025 Vectify AI; MIT License, reproduced in `mod.rs`.

use super::detect::Block;
use super::stats::DocStats;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

/// Upstream types: 0 plain, 1 numbered, 11 introduction; 2 is a PDF Hn, 12 set apart by type.
#[derive(Debug, Clone)]
pub(super) struct Candidate {
    pub(super) block: usize,
    pub(super) kind: u8,
    /// The numbering ladder's level or an explicit PDF Hn level.
    pub(super) level: Option<usize>,
}

pub(super) struct Context<'a> {
    pub(super) doc: &'a [DocStats; 2],
    pub(super) blocks: &'a [Block],
    /// Position of each block within its page's reading order.
    pub(super) positions: &'a [usize],
    pub(super) centered: &'a [bool],
}

impl Context<'_> {
    pub(super) fn order(&self, left: usize, right: usize) -> Ordering {
        (self.blocks[left].page, self.positions[left])
            .cmp(&(self.blocks[right].page, self.positions[right]))
    }

    /// Upstream `compare_heading_depth` with the keyword clique absent: 1 when `a` is shallower
    /// (may hold `b`), -1 when deeper, 0 when they are siblings. Numbered headings compare by
    /// the native ladder's level rather than upstream's numbering length. Upstream ranks an
    /// unnumbered heading above any numbered one; legal numbering marks parts and sections at
    /// every depth, so type decides between a numbered and an unnumbered heading instead.
    pub(super) fn compare(&self, a: &Candidate, b: &Candidate) -> i8 {
        if a.kind == b.kind && matches!(a.kind, 1 | 2) {
            return match (a.level, b.level) {
                (Some(left), Some(right)) if left == right => 0,
                (Some(left), Some(right)) if left < right => 1,
                _ => -1,
            };
        }
        if a.kind == 11 && b.level == Some(1) {
            return -1;
        }
        let (left, right) = (&self.blocks[a.block].flash, &self.blocks[b.block].flash);
        let styles_known = !self.blocks[a.block].fontless && !self.blocks[b.block].fontless;
        // Sizes compare on one body-relative scale; unknown font styles supply no contrast.
        let score = |block: usize| {
            let block = &self.blocks[block];
            let (font, glyph) = (self.doc[0].body_font_size, self.doc[1].body_font_size);
            let scale = if block.fontless && font > 0.0 && glyph > 0.0 {
                font / glyph
            } else {
                1.0
            };
            block.flash.heading_score(scale, styles_known)
        };
        let (score, other) = (score(a.block), score(b.block));
        if (score - other).abs() > 0.9 {
            return if score > other { 1 } else { -1 };
        }
        let (left_caps, right_caps) = (left.caps_heavy(), right.caps_heavy());
        if left_caps != right_caps {
            return if left_caps { 1 } else { -1 };
        }
        let (left_centered, right_centered) = (self.centered[a.block], self.centered[b.block]);
        if left_centered != right_centered {
            return if left_centered { 1 } else { -1 };
        }
        let (left_italic, right_italic) = (left.italic > 0.99, right.italic > 0.99);
        if styles_known && left_italic != right_italic {
            return if left_italic { -1 } else { 1 };
        }
        if left_caps {
            return 0;
        }
        let (left_bold, right_bold) = (left.bold > 0.5, right.bold > 0.5);
        if styles_known && left_bold != right_bold {
            return if left_bold { 1 } else { -1 };
        }
        0
    }

    /// Upstream `find_parent_heading`: pop the stack until a heading that holds `candidate`.
    pub(super) fn parent(
        &self,
        stack: &mut Vec<usize>,
        candidates: &[Candidate],
        candidate: &Candidate,
    ) -> Option<usize> {
        while let Some(&top) = stack.last() {
            if self.compare(&candidates[top], candidate) == 1 {
                return Some(top);
            }
            stack.pop();
        }
        None
    }

}

/// The font-distance rule of upstream `should_reject_heading`: a plain heading whose first-line
/// font style no other candidate shares within 0.9pt stands alone and is no heading, unless it
/// is in capitals close above the body it heads.
/// Each candidate's fonts for [`stands_alone`]: the style (font and weight, as a number standing
/// for its name) and size of each distinct style key among its first line's lettered spans.
pub(super) fn candidate_fonts(context: &Context<'_>, candidates: &[Candidate]) -> Vec<Vec<(usize, f64)>> {
    let mut styles = HashMap::new();
    candidates
        .iter()
        .map(|candidate| {
            let flash = &context.blocks[candidate.block].flash;
            let mut seen = HashSet::new();
            let mut entries = Vec::new();
            if let Some(line) = flash.lines.first() {
                for span in &line.spans {
                    if span.stats.letters() == 0 || span.font_name.is_empty() {
                        continue;
                    }
                    if seen.insert(span.style_key()) {
                        let next = styles.len();
                        entries.push((*styles.entry(span.font_style()).or_insert(next), span.size));
                    }
                }
            }
            entries
        })
        .collect()
}

pub(super) fn stands_alone(
    context: &Context<'_>,
    candidates: &[Candidate],
    fonts: &[Vec<(usize, f64)>],
    index: usize,
    body_below_gap: Option<f64>,
) -> bool {
    let own = &fonts[index];
    // Missing font evidence cannot establish an isolated font style.
    if own.is_empty() {
        return false;
    }
    let mut distance = f64::INFINITY;
    for (other_index, other) in fonts.iter().enumerate() {
        if other_index == index {
            continue;
        }
        for &(style, size) in other {
            for &(own_style, own_size) in own {
                if own_style == style {
                    distance = distance.min((own_size - size).abs());
                }
            }
        }
    }
    if distance <= 0.9 {
        return false;
    }
    if distance.is_infinite() {
        return true;
    }
    let flash = &context.blocks[candidates[index].block].flash;
    !(flash.caps_heavy() && body_below_gap.is_some_and(|gap| gap < 5.0 * flash.rect.height()))
}
