//! PageIndex Flash's style-based heading candidates over native blocks.
//!
//! Ported from VectifyAI/PageIndex `pageindex/flash/classification/body_text.py`
//! (`is_body_paragraph`), `heading_detection/neighbors.py` (`PageNeighborMap`),
//! `heading_detection/text_checks.py` (`vertically_close`, `has_substantive_content`),
//! `heading_detection/detectors.py` (`is_too_wide_for_heading`, `passes_neighbor_check`),
//! `heading_detection/page_scan.py` (`scan_page_headings`, `filter_page_candidates`) and
//! `heading_detection/style_detectors.py` (`detect_font_heading`) at
//! 6d23caf416858f2ca136840305d1f479a86f6ef7. Copyright (c) 2025 Vectify AI; MIT License,
//! reproduced in `mod.rs`.

use super::model::{x_aligned, y_overlaps, FlashBlock, Rect};
use super::stats::{DocStats, PageStats};
use std::collections::HashSet;

/// One native block (a body or heading region) with the classification Flash reads.
#[derive(Debug, Clone)]
pub(super) struct Block {
    pub(super) flash: FlashBlock,
    pub(super) page: usize,
    pub(super) line_slots: Vec<usize>,
    pub(super) heading: bool,
    pub(super) numbered: bool,
    pub(super) is_body: bool,
}

impl Block {
    pub(super) fn rect(&self) -> &Rect {
        &self.flash.rect
    }
}

#[derive(Debug, Clone)]
pub(super) struct FlashPage {
    pub(super) width: f64,
    pub(super) height: f64,
    pub(super) stats: PageStats,
    /// Indexes into the document's blocks, in reading order.
    pub(super) blocks: Vec<usize>,
    pub(super) body_styles: HashSet<String>,
}

/// Lines of at least 40 characters of ordinary weight and size that read as running text.
pub(super) fn is_body_paragraph(doc: &DocStats, page: &FlashPage, block: &FlashBlock) -> bool {
    if block.fill < 0.6 {
        return false;
    }
    let weight = block.stats.info_weight();
    let lines = block.line_count() as f64;
    let sentence_punct = f64::from(block.stats.counts[6]);
    let per_line = if lines != 0.0 {
        weight / lines
    } else if weight > 0.0 {
        f64::INFINITY
    } else {
        f64::NAN
    };
    if per_line < 15.0
        || (lines >= 10.0 && per_line < 20.0)
        || (lines >= 10.0 && per_line < 25.0 && sentence_punct < lines / 8.0)
        || (lines >= 20.0 && per_line < 40.0 && sentence_punct < lines / 20.0)
    {
        return false;
    }
    let width = block.rect.width();
    if lines >= 4.0 {
        let short = block
            .lines
            .iter()
            .filter(|line| {
                line.rect.width() < 0.75 * width && !line.text.trim_start().starts_with('\u{2022}')
            })
            .count() as f64;
        if short >= lines / 2.0 && sentence_punct < lines / 8.0 {
            return false;
        }
    }
    if width < page.width / 7.0 {
        return false;
    }
    let chars = f64::from(block.char_count());
    if chars < 40.0
        || (lines >= 3.0 && block.alignment_code() == 3)
        || f64::from(block.stats.letters()) < 0.1 * chars
    {
        return false;
    }
    let size = block.size;
    let body = page.stats.font_size.min(doc.body_font_size);
    let minimum = doc.body_font_size.min(page.height.max(page.width) / 60.0);
    let minimum = (0.7 * minimum).min(minimum - 3.0);
    if size < body - 2.0 || size < minimum {
        return false;
    }
    if chars >= 250.0 && lines >= 4.0 {
        return true;
    }
    if width < page.width / 5.0 || size < body - 0.5 {
        return false;
    }
    if chars >= 100.0 && lines >= 2.0 && sentence_punct >= 2.0 {
        return true;
    }
    if (chars >= 100.0 || block.stats.last == 6)
        && (size >= page.stats.font_size - 0.5 || size > doc.body_font_size - 0.1)
    {
        return block
            .first_span()
            .is_some_and(|span| span.font_name == page.stats.dominant_font)
            || block
                .last_span()
                .is_some_and(|span| span.font_name == page.stats.dominant_font);
    }
    false
}

#[derive(Debug, Clone, Default)]
pub(super) struct Neighbors {
    /// Some bucket the block spans is crossed by a body paragraph.
    pub(super) body_crossed: bool,
    pub(super) above: Option<usize>,
    pub(super) body_above: Option<usize>,
    pub(super) below: Option<usize>,
    pub(super) body_below: Option<usize>,
}

/// Each block's nearest neighbors above and below through the horizontal buckets it spans.
pub(super) fn neighbor_map(page: &FlashPage, blocks: &[Block]) -> Vec<Neighbors> {
    let bucket = 5f64.max(page.width / 300.0);
    let count = (page.width / bucket).floor().max(0.0) as usize;
    let span = |rect: &Rect| {
        let clamp = |value: f64| value.max(0.0).min(count.saturating_sub(1) as f64);
        (
            clamp((rect.left / bucket).floor()) as usize,
            clamp((rect.right / bucket).ceil()) as usize,
        )
    };
    let mut map = vec![Neighbors::default(); page.blocks.len()];
    let mut marked = vec![false; count];
    for &block in &page.blocks {
        if blocks[block].is_body {
            let (start, end) = span(blocks[block].rect());
            for slot in start..end.min(count) {
                marked[slot] = true;
            }
        }
    }
    let mut recent_any: Vec<Option<usize>> = vec![None; count];
    let mut recent_body: Vec<Option<usize>> = vec![None; count];
    let mut pending: Vec<Vec<usize>> = vec![Vec::new(); count];
    for (position, &block) in page.blocks.iter().enumerate() {
        let current = &blocks[block];
        if current.flash.char_count() == 0 || current.flash.skew > 1.0 {
            continue;
        }
        let (start, end) = span(current.rect());
        for slot in start..end.min(count) {
            map[position].body_crossed |= marked[slot];
            if let Some(body) = recent_body[slot] {
                let closer = map[position].body_above.is_none_or(|prior| {
                    blocks[page.blocks[body]].rect().bottom
                        < blocks[page.blocks[prior]].rect().bottom
                });
                if closer {
                    map[position].body_above = Some(body);
                }
            }
            if let Some(previous) = recent_any[slot].replace(position) {
                let closer = map[position].above.is_none_or(|prior| {
                    blocks[page.blocks[previous]].rect().bottom
                        < blocks[page.blocks[prior]].rect().bottom
                });
                if closer {
                    map[position].above = Some(previous);
                }
                let replace = map[previous]
                    .below
                    .is_none_or(|prior| current.rect().top > blocks[page.blocks[prior]].rect().top);
                if replace {
                    map[previous].below = Some(position);
                }
            }
            if current.is_body {
                for waiting in std::mem::take(&mut pending[slot]) {
                    let replace = map[waiting].body_below.is_none_or(|prior| {
                        current.rect().top > blocks[page.blocks[prior]].rect().top
                    });
                    if replace {
                        map[waiting].body_below = Some(position);
                    }
                }
                recent_body[slot] = Some(position);
            }
            pending[slot].push(position);
        }
    }
    map
}

/// `b` sits within two font sizes of `a`, or within five when aligned with it.
pub(super) fn vertically_close(a: Option<&Block>, b: &Block) -> bool {
    let Some(a) = a else {
        return false;
    };
    let gap = if a.rect().top > b.rect().top {
        a.rect().bottom - b.rect().top
    } else {
        b.rect().bottom - a.rect().top
    };
    gap < 2.0 * b.flash.size || (x_aligned(a.rect(), b.rect(), 1.0) && gap < 5.0 * b.flash.size)
}

pub(super) struct Scan<'a> {
    pub(super) doc: &'a DocStats,
    pub(super) pages: &'a [FlashPage],
    pub(super) blocks: &'a [Block],
    pub(super) page: usize,
    pub(super) neighbors: Vec<Neighbors>,
    /// Blocks already proposed on this page.
    pub(super) proposed: HashSet<usize>,
}

impl Scan<'_> {
    fn at(&self, position: Option<usize>) -> Option<&Block> {
        position.map(|position| &self.blocks[self.pages[self.page].blocks[position]])
    }

    fn too_wide(&self, block: &Block) -> bool {
        let width = block.rect().width();
        let page = &self.pages[self.page];
        if width > 0.7 * page.width / 2.0 || width > 0.7 * self.doc.line_width {
            return true;
        }
        let count = (self.page.saturating_sub(1)..=(self.page + 1).min(self.pages.len() - 1))
            .filter(|index| width > 0.7 * self.pages[*index].stats.line_width)
            .count();
        count >= 2
    }

    /// True when a narrow block beside others on its row is a column or table fragment.
    fn neighbor_rejects(&self, position: usize) -> bool {
        let page = &self.pages[self.page];
        let block = &self.blocks[page.blocks[position]];
        if self.too_wide(block) {
            return false;
        }
        let previous = position
            .checked_sub(1)
            .map(|slot| &self.blocks[page.blocks[slot]]);
        let overlap = previous.is_some_and(|other| y_overlaps(block.rect(), other.rect()));
        if overlap && previous.is_some_and(|other| self.too_wide(other)) {
            return false;
        }
        let next = page
            .blocks
            .get(position + 1)
            .map(|&slot| &self.blocks[slot]);
        let next_overlap = next.is_some_and(|other| y_overlaps(block.rect(), other.rect()));
        if next_overlap && next.is_some_and(|other| self.too_wide(other)) {
            return false;
        }
        if !overlap && !next_overlap {
            return false;
        }
        let below = self.at(self.neighbors[position].below);
        if below.is_some_and(|other| self.too_wide(other)) {
            return false;
        }
        if below.is_some_and(|other| {
            !other.is_body
                && other.flash.line_count() > 3
                && other.rect().height() > 0.8 * other.rect().width()
        }) {
            return true;
        }
        let first_line = &block.flash.lines[0];
        let threshold = 4.0
            * if first_line.char_count() == 0 {
                0.0
            } else {
                first_line.rect.width() / f64::from(first_line.char_count())
            };
        let body_below = self.at(self.neighbors[position].body_below);
        if let Some(other) = body_below {
            if self.neighbors[position].below != self.neighbors[position].body_below
                && x_aligned(block.rect(), other.rect(), threshold)
                && self.too_wide(other)
            {
                return false;
            }
        }
        if let Some(other) = self.at(self.neighbors[position].body_above) {
            if x_aligned(block.rect(), other.rect(), threshold) && self.too_wide(other) {
                return false;
            }
        }
        true
    }

    /// Upstream `scan_page_headings` for blocks no numbering or keyword claims, then
    /// `detect_font_heading`.
    pub(super) fn candidate(&mut self, position: usize) -> bool {
        let page = &self.pages[self.page];
        let block = &self.blocks[page.blocks[position]];
        let flash = &block.flash;
        if flash.char_count() == 0 || flash.skew > 1.0 || block.heading || block.numbered {
            return false;
        }
        let neighbors = &self.neighbors[position];
        let above = self.at(neighbors.above);
        if above.is_some_and(|above| {
            let (outer, inner) = (above.rect(), block.rect());
            outer.left <= inner.left
                && outer.right >= inner.right
                && outer.top >= inner.top
                && outer.bottom <= inner.bottom
        }) {
            return false;
        }
        if flash.char_count() <= 1 && flash.stats.first != 4 {
            return false;
        }
        let rows = flash.line_count();
        let chars = flash.char_count();
        if chars >= 200 {
            return false;
        }
        if chars >= 100
            && rows > 1
            && flash.stats.counts[6].saturating_sub(flash.lines[0].stats.counts[6]) > 1
        {
            return false;
        }
        let size = flash.size;
        let lots_caps =
            f64::from(flash.stats.counts[2]) >= 3f64.max(f64::from(flash.stats.letters()) / 2.0);
        if rows > 4 || (rows >= 3 && !(size >= 1.5 * page.stats.font_size || lots_caps)) {
            return false;
        }
        if flash.fill < 0.5 * self.doc.upper_fill || flash.stats.letters() == 0 {
            return false;
        }
        if flash.bold < 0.1 && !lots_caps && size < self.doc.body_font_size - 2.0 {
            return false;
        }
        if self.neighbor_rejects(position) {
            return false;
        }
        let top_gap = above.map_or(f64::INFINITY, |above| {
            above.rect().bottom - block.rect().top
        });
        let previous = position
            .checked_sub(1)
            .map(|slot| &self.blocks[page.blocks[slot]]);
        let next = page
            .blocks
            .get(position + 1)
            .map(|&slot| &self.blocks[slot]);
        // Headers are not native blocks: a block's neighbors are never one.
        if !neighbors.body_crossed
            && (flash.heading_score() < page.stats.font_size + 1.5
                || above.is_some()
                || previous.is_some()
                || next.is_some_and(|next| !(next.rect().top < block.rect().bottom - size)))
        {
            return false;
        }
        if flash.bold < 0.1
            && !lots_caps
            && flash
                .first_span()
                .is_some_and(|span| span.font_name == page.stats.dominant_font)
            && size < self.doc.body_font_size - 0.5
        {
            return false;
        }
        let below = self.at(neighbors.below);
        if below.is_some_and(|below| below.flash.skew > 1.0) || top_gap < 0.0 {
            return false;
        }
        let below_gap = below.map_or(f64::INFINITY, |below| {
            block.rect().bottom - below.rect().top
        });
        if below_gap < -0.9 * block.rect().height() {
            return false;
        }
        let line_gap = page.stats.line_gap - page.stats.font_size;
        if top_gap < line_gap && size < page.stats.font_size - 1.0 {
            return false;
        }
        if chars >= 120 || top_gap <= line_gap - 0.1 {
            return false;
        }
        if flash.stats.info_weight() <= 3.0 || substantive_content(flash) {
            return false;
        }
        self.font_heading(position)
    }

    /// Upstream `detect_font_heading`.
    fn font_heading(&self, position: usize) -> bool {
        let page = &self.pages[self.page];
        let block = &self.blocks[page.blocks[position]];
        let flash = &block.flash;
        let rect = block.rect();
        let neighbors = &self.neighbors[position];
        let above = self.at(neighbors.above);
        let below = self.at(neighbors.below);
        let body_above = self.at(neighbors.body_above);
        let body_below = self.at(neighbors.body_below);
        let top_gap = above.map_or(f64::INFINITY, |above| above.rect().bottom - rect.top);
        let below_gap = below.map_or(f64::INFINITY, |below| rect.bottom - below.rect().top);
        let doc_size = self.doc.body_font_size;
        if !(vertically_close(body_above, block)
            || vertically_close(body_below, block)
            || (rect.bottom >= 0.8 * page.height
                && flash.size >= doc_size + 1.0
                && flash.bold > 0.9
                && above.is_none()))
        {
            return false;
        }
        let far = 10.0 * flash.size.min(page.stats.line_gap);
        if top_gap.is_finite() && top_gap > far {
            return false;
        }
        if let Some(below) = below {
            if below.flash.fill < 0.67 * self.doc.upper_fill
                || (flash.char_count() < 30
                    && below.flash.char_count() < 300
                    && below.flash.fill < 0.8 * self.doc.upper_fill)
            {
                return false;
            }
        }
        if flash.tokens().last().is_some_and(|token| {
            matches!(
                token.text.as_str(),
                "," | "\u{ff0c}" | "\u{3001}" | "\u{fe50}" | "\u{fe51}"
            )
        }) {
            return false;
        }
        // Branch 1: a tall, large title.
        if below_gap.is_finite()
            && below_gap > 0.0
            && flash.line_height >= doc_size + 2.0
            && flash.size >= doc_size + 1.5
            && flash.size >= page.stats.font_size + 0.5
            && above.is_none_or(|above| {
                flash.line_height >= above.flash.line_height && flash.size >= above.flash.size
            })
            && below.is_some_and(|below| {
                flash.line_height >= below.flash.line_height && flash.size >= below.flash.size
            })
        {
            return true;
        }
        let caps = flash.caps_heavy();
        let dominant = flash.dominant_style();
        if (page.body_styles.contains(dominant)
            && !caps
            && (page.stats.dominant_style == dominant
                || (flash.bold < 0.9
                    && flash.italic < 0.9
                    && flash.alignment_code() != 3
                    && !flash.sentence_like())))
            || (above.is_some()
                && below.is_some_and(|below| {
                    flash.size <= below.flash.size && top_gap < below_gap / 4.0
                }))
        {
            return false;
        }
        // Branch 3: a larger size than the paragraph it heads.
        if below_gap.is_finite()
            && below_gap > 0.0
            && flash.size >= doc_size + 0.5
            && below.is_some_and(|below| {
                flash.line_height >= below.flash.line_height
                    && flash.size >= below.flash.size
                    && below.flash.size >= page.stats.font_size - 0.5
                    && rect.width() < 0.95 * below.rect().width()
                    && below.rect().width() >= 0.25 * page.width
            })
        {
            return true;
        }
        let line_height = page.stats.line_gap - page.stats.font_size;
        // Branch 4: a moderate gap before a body paragraph in smaller type.
        if below_gap > line_height
            && below_gap < 5.0 * line_height
            && above.is_none_or(|above| flash.size >= above.flash.size + 0.5)
            && below.is_some_and(|below| {
                flash.size >= below.flash.size + 0.5
                    && flash.size >= doc_size - 0.5
                    && rect.width() < 0.95 * below.rect().width()
                    && below.is_body
                    && below.flash.size >= page.stats.font_size - 0.5
                    && below.flash.stats.first != 1
            })
        {
            return true;
        }
        let sentence_ends = flash.stats.counts[6];
        let spaces = flash.stats.counts[10];
        let ratio = if spaces != 0 {
            f64::from(sentence_ends + flash.stats.counts[8]) / f64::from(spaces)
        } else if sentence_ends + flash.stats.counts[8] > 0 {
            f64::INFINITY
        } else {
            f64::NAN
        };
        if sentence_ends > 1 && ratio > 0.3 {
            return false;
        }
        let symbols = f64::from(flash.stats.punctuation());
        let letters = f64::from(flash.stats.letters());
        let symbol_ratio = if letters != 0.0 {
            symbols / letters
        } else if symbols > 0.0 {
            f64::INFINITY
        } else {
            f64::NAN
        };
        if (symbols >= 5.0 && symbol_ratio > 0.2)
            || top_gap < 0.2 * flash.size
            || top_gap < flash.size.min(0.7 * below_gap)
        {
            return false;
        }
        let capital = flash.stats.first == 2;
        let lowered = if caps {
            -0.2 * flash.last_span().map_or(0.0, |span| span.rect.height())
        } else {
            0.0
        };
        let aligned = |other: &Block| x_aligned(rect, other.rect(), 1f64.max(rect.width() / 10.0));
        // Branch A: set apart in bold or capitals beside an aligned body paragraph.
        let neighbor_cue = below_gap.is_finite()
            && below_gap > lowered
            && flash.size >= page.stats.font_size - 0.1
            && below.is_some_and(|below| {
                flash.size >= below.flash.size - 0.1
                    && flash.size >= doc_size - 0.5
                    && ((below.is_body
                        && rect.width() < 0.95 * below.rect().width()
                        && aligned(below)
                        && below_gap < 6.0 * rect.height())
                        || (top_gap.is_finite()
                            && above.is_some_and(|above| {
                                above.is_body
                                    && rect.width() < 0.95 * above.rect().width()
                                    && aligned(above)
                                    && top_gap < 6.0 * rect.height()
                            })))
                    && (capital || caps)
                    && ((flash.bold > below.flash.bold
                        && flash.bold > 0.5
                        && (!below.flash.first_span().is_some_and(|span| span.bold)
                            || above.is_some_and(|above| flash.bold > above.flash.bold)))
                        || caps)
            });
        let difference_style = below.is_some_and(|below| {
            below.is_body
                && below.flash.size > page.stats.font_size - 0.5
                && below_gap > 0.0
                && below_gap < 3.0 * rect.height()
                && dominant != below.flash.dominant_style()
        });
        let style_change_cue = below.is_some_and(|below| {
            difference_style
                && above.is_some_and(|above| {
                    above.is_body
                        && top_gap > 0.0
                        && top_gap < 3.0 * rect.height()
                        && capital
                        && above.flash.dominant_style() == below.flash.dominant_style()
                })
        });
        if neighbor_cue || style_change_cue {
            return true;
        }
        let top = above.is_none();
        let body_dominant =
            |other: &Block| other.flash.dominant_style() == page.stats.dominant_style;
        let c1 = top
            && capital
            && difference_style
            && below_gap < rect.height()
            && below.is_some_and(body_dominant);
        let c2 = top
            && capital
            && match (below, body_below) {
                (Some(below), Some(body)) => {
                    neighbors.below != neighbors.body_below
                        && below.rect().bottom - body.rect().top < below.flash.size
                        && body_dominant(body)
                        && dominant != page.stats.dominant_style
                        && (flash.size >= below.flash.size + 0.5
                            || (caps && !below.flash.stats.upper_dominant()))
                }
                _ => false,
            };
        let c3 = above.is_some_and(|above| {
            (above.heading
                || neighbors
                    .above
                    .is_some_and(|slot| self.proposed.contains(&slot)))
                && (above.flash.size >= flash.size + 0.5
                    || (above.flash.stats.upper_dominant() && !caps))
                && capital
                && difference_style
                && below.is_some_and(body_dominant)
        });
        c1 || c2 || c3
    }
}

/// Upstream `has_substantive_content` without line-relative raised or lowered tokens, which
/// native spans do not mark: digits and formula symbols count for, words against.
fn substantive_content(block: &FlashBlock) -> bool {
    let mut score = 0.0;
    for token in block.tokens() {
        if token.kind == 1 {
            score += 1.0;
            continue;
        }
        if let Some(weight) = formula_weight(&token.text) {
            score += weight;
            continue;
        }
        if token.kind == 6 {
            score += 5.0;
            continue;
        }
        let length = token.text.chars().count() as f64;
        if length <= 3.0 && token.first != 4 {
            continue;
        }
        // Upstream doubles a bold token's length; tokens here carry the block's weight.
        let length = if block.bold > 0.5 {
            2.0 * length
        } else {
            length
        };
        match token.first {
            4 => score -= 2.0 * length,
            2 => score -= length,
            3 => score -= 0.5 * length,
            _ => {}
        }
    }
    score >= 5.0
}

fn formula_weight(text: &str) -> Option<f64> {
    Some(match text {
        "=" => 10.0,
        "{" | "}" | "+" => 5.0,
        "/" | "*" => 3.0,
        "-" | "~" | "[" | "]" | "(" | ")" => 1.0,
        _ => return None,
    })
}
