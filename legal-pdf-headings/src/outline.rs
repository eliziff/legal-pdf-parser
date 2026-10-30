//! PageIndex Flash's heading depth and outline stack.
//!
//! Ported from VectifyAI/PageIndex `pageindex/flash/outline_assembly/style_context.py`
//! (`compare_heading_depth`, `_apply_heading_to_state`) and `selection.py` (`find_parent_heading`)
//! at 6d23caf416858f2ca136840305d1f479a86f6ef7.
//! Copyright (c) 2025 Vectify AI; MIT License, reproduced in `mod.rs`.

use super::detect::Block;
use super::stats::DocStats;
use std::cmp::Ordering;

/// Upstream types: 0 plain, 1 numbered; 2 is a PDF Hn, 12 set apart by type.
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
