//! PageIndex Flash's heading depth, body-heading cliques and outline stack.
//!
//! Ported from VectifyAI/PageIndex `pageindex/flash/outline_assembly/style_context.py`
//! (`compare_heading_depth`, `_apply_heading_to_state`), `selection.py` (`find_parent_heading`,
//! the font-distance rule of `should_reject_heading`) and `cliques.py` (`CliqueTreeBuilder`,
//! `block_style_signature`, `is_member_of_tree`, `can_share_heading_style`,
//! `detect_body_headings`) at 6d23caf416858f2ca136840305d1f479a86f6ef7.
//! Copyright (c) 2025 Vectify AI; MIT License, reproduced in `mod.rs`.

use super::detect::{Block, FlashPage, Neighbors};
use super::model::y_overlaps;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

/// Upstream types kept here: 0 plain, 1 numbered, 11 introduction keyword.
#[derive(Debug, Clone)]
pub(super) struct Candidate {
    pub(super) block: usize,
    pub(super) kind: u8,
    /// The native ladder's level for a numbered heading; upstream's numbering length.
    pub(super) level: Option<usize>,
}

impl Candidate {
    fn numbered(&self) -> bool {
        self.level.is_some()
    }
}

pub(super) struct Context<'a> {
    pub(super) blocks: &'a [Block],
    pub(super) pages: &'a [FlashPage],
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
        if a.kind == 1 && b.kind == 1 {
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
        let (score, other) = (left.heading_score(), right.heading_score());
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
        if left_italic != right_italic {
            return if left_italic { -1 } else { 1 };
        }
        if left_caps {
            return 0;
        }
        let (left_bold, right_bold) = (left.bold > 0.5, right.bold > 0.5);
        if left_bold != right_bold {
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

    fn signature(&self, block: usize) -> String {
        let flash = &self.blocks[block].flash;
        format!("{} {}", flash.dominant_style(), flash.caps_heavy())
    }
}

struct TreeNode {
    candidate: Option<usize>,
    parent: usize,
    children: Vec<usize>,
    next_sibling: Option<usize>,
    previous_sibling: Option<usize>,
}

/// Upstream `CliqueTreeBuilder`: candidates nested by depth comparison, at most eight deep.
struct CliqueTree {
    nodes: Vec<TreeNode>,
    cursor: usize,
}

impl CliqueTree {
    fn build(
        context: &Context<'_>,
        candidates: &[Candidate],
        order: impl Iterator<Item = usize>,
    ) -> CliqueTree {
        let mut tree = CliqueTree {
            nodes: vec![TreeNode {
                candidate: None,
                parent: 0,
                children: Vec::new(),
                next_sibling: None,
                previous_sibling: None,
            }],
            cursor: 0,
        };
        let mut depth = 0;
        for candidate in order {
            loop {
                if tree.cursor == 0 {
                    tree.append(0, candidate);
                    depth += 1;
                    break;
                }
                let top = tree.nodes[tree.cursor].candidate.expect("non-root node");
                let comparison = context.compare(&candidates[top], &candidates[candidate]);
                if comparison < 0 {
                    tree.cursor = tree.nodes[tree.cursor].parent;
                    depth -= 1;
                } else {
                    if comparison > 0 && depth < 8 {
                        tree.append(tree.cursor, candidate);
                        depth += 1;
                    } else {
                        let parent = tree.nodes[tree.cursor].parent;
                        tree.append(parent, candidate);
                    }
                    break;
                }
            }
        }
        tree
    }

    fn append(&mut self, parent: usize, candidate: usize) {
        let index = self.nodes.len();
        let last = self.nodes[parent].children.last().copied();
        self.nodes.push(TreeNode {
            candidate: Some(candidate),
            parent,
            children: Vec::new(),
            next_sibling: None,
            previous_sibling: last,
        });
        if let Some(last) = last {
            self.nodes[last].next_sibling = Some(index);
        }
        self.nodes[parent].children.push(index);
        self.cursor = index;
    }

    fn next(&self, node: usize) -> Option<usize> {
        if let Some(&child) = self.nodes[node].children.first() {
            return Some(child);
        }
        if let Some(sibling) = self.nodes[node].next_sibling {
            return Some(sibling);
        }
        let mut ancestor = self.nodes[node].parent;
        loop {
            if ancestor == 0 {
                return None;
            }
            if let Some(sibling) = self.nodes[ancestor].next_sibling {
                return Some(sibling);
            }
            ancestor = self.nodes[ancestor].parent;
        }
    }

    fn deepest_last(&self, mut node: usize) -> usize {
        while let Some(&last) = self.nodes[node].children.last() {
            node = last;
        }
        node
    }
}

/// Upstream `detect_body_headings`: blocks set in the style of three or more candidates, set
/// apart from their neighbors, that fit under a candidate of that style.
pub(super) fn body_headings(
    context: &Context<'_>,
    candidates: &[Candidate],
    neighbors: &[Vec<Neighbors>],
) -> Vec<usize> {
    let mut found = Vec::new();
    if candidates.is_empty() {
        return found;
    }
    let claimed = candidates
        .iter()
        .map(|candidate| candidate.block)
        .collect::<HashSet<_>>();
    let mut counts = HashMap::<String, usize>::new();
    for candidate in candidates.iter().filter(|candidate| !candidate.numbered()) {
        *counts
            .entry(context.signature(candidate.block))
            .or_default() += 1;
    }
    let forward = CliqueTree::build(context, candidates, 0..candidates.len());
    let backward = CliqueTree::build(context, candidates, (0..candidates.len()).rev());
    let mut ahead = 0;
    let mut behind = backward.cursor;
    let precedes = |candidate: usize, block: usize| {
        context.order(candidates[candidate].block, block) == Ordering::Less
    };
    for (page_index, page) in context.pages.iter().enumerate() {
        for (position, &block) in page.blocks.iter().enumerate() {
            while let Some(next) = forward.next(ahead) {
                match forward.nodes[next].candidate {
                    Some(candidate) if precedes(candidate, block) => ahead = next,
                    _ => break,
                }
            }
            while let Some(candidate) = backward.nodes[behind].candidate {
                if !precedes(candidate, block) {
                    break;
                }
                behind = match backward.nodes[behind].previous_sibling {
                    Some(sibling) => backward.deepest_last(sibling),
                    None => backward.nodes[behind].parent,
                };
                if behind == 0 {
                    break;
                }
            }
            if claimed.contains(&block) || forward.nodes[ahead].candidate.is_none() {
                continue;
            }
            let entry = &context.blocks[block];
            let flash = &entry.flash;
            if flash.char_count() == 0
                || flash.skew > 1.0
                || (flash.char_count() <= 1 && flash.stats.first != 4)
                || flash.line_count() >= 5
                || entry.heading
                || entry.numbered
                || (flash.stats.counts[2] == 0 && flash.stats.counts[4] == 0)
                || (flash.bold > 0.1 && flash.bold < 0.9)
            {
                continue;
            }
            let style = flash.dominant_style();
            let first = flash.first_span().map(|span| span.style_key());
            let last = flash.last_span().map(|span| span.style_key());
            if first.as_deref() != Some(style) && last.as_deref() != Some(style) {
                continue;
            }
            if style == page.stats.dominant_style {
                continue;
            }
            let near = &neighbors[page_index][position];
            let above = near.above.map(|slot| page.blocks[slot]);
            if above.is_some_and(|above| {
                context.blocks[above].flash.rect.bottom - flash.rect.top < 0.3 * flash.size
                    && flash.line_count() > 1
            }) {
                continue;
            }
            let signature = context.signature(block);
            let below = near.below.map(|slot| page.blocks[slot]);
            if above.is_some_and(|above| context.signature(above) == signature)
                || below.is_some_and(|below| context.signature(below) == signature)
            {
                continue;
            }
            let shares = |other: Option<usize>| {
                let Some(other) = other else {
                    return false;
                };
                let other_flash = &context.blocks[other].flash;
                if !y_overlaps(&flash.rect, &other_flash.rect)
                    || other_flash.dominant_style() != style
                {
                    return false;
                }
                let other_position = context.positions[other];
                let own_below = near.below.map(|slot| page.blocks[slot]);
                let other_below = neighbors[page_index][other_position]
                    .below
                    .map(|slot| page.blocks[slot]);
                !(own_below != other_below
                    && own_below.is_some_and(|slot| context.blocks[slot].is_body)
                    && other_below.is_some_and(|slot| context.blocks[slot].is_body))
            };
            let previous = position.checked_sub(1).map(|slot| page.blocks[slot]);
            let next = page.blocks.get(position + 1).copied();
            if shares(previous) || shares(next) {
                continue;
            }
            if counts.get(&signature).copied().unwrap_or(0) < 3 {
                continue;
            }
            let tokens = flash.tokens();
            let (mut words, mut lowercase) = (0usize, 0usize);
            for token in &tokens {
                if token.kind != 2 || token.text.chars().count() < 5 {
                    continue;
                }
                words += 1;
                if token.first == 3 {
                    lowercase += 1;
                }
            }
            let sentence_like = lowercase as f64 >= 2f64.max(words as f64 / 2.0);
            let member = |tree: &CliqueTree, node: usize| {
                member_of_tree(
                    context,
                    candidates,
                    tree,
                    node,
                    block,
                    &signature,
                    sentence_like,
                )
            };
            if member(&forward, ahead) || member(&backward, behind) {
                found.push(block);
            }
        }
    }
    found
}

/// Upstream `is_member_of_tree`: an unnumbered ancestor candidate shares the block's signature.
fn member_of_tree(
    context: &Context<'_>,
    candidates: &[Candidate],
    tree: &CliqueTree,
    mut node: usize,
    block: usize,
    signature: &str,
    sentence_like: bool,
) -> bool {
    loop {
        if node == 0 {
            return false;
        }
        let Some(candidate) = tree.nodes[node].candidate else {
            return false;
        };
        let parent_block = candidates[candidate].block;
        let parent = &context.blocks[parent_block].flash;
        if candidates[candidate].numbered()
            || context.signature(parent_block) != signature
            || (sentence_like && parent.sentence_like())
            || context.blocks[block].flash.stats.info_weight()
                >= 100f64.max(4.0 * parent.stats.info_weight())
        {
            node = tree.nodes[node].parent;
            continue;
        }
        return true;
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
                    if span.stats.letters() == 0 {
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
