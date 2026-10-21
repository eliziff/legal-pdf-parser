//! Headings set apart by type alone, and the depth of every heading, as PageIndex Flash reads
//! them from layout statistics.
//!
//! Upstream: VectifyAI/PageIndex `pageindex/flash` at 6d23caf416858f2ca136840305d1f479a86f6ef7
//! (MIT, notice below). Flash's live route is `main.extract_toc` → `assemble_outline` →
//! `build_doc_heading_candidates` / `detect_body_headings` → the outline stack. The port reads
//! native lines and regions instead of Flash's own lines and blocks; each submodule names the
//! upstream functions it ports. Deliberate adaptations:
//! - The native ladder supplies numbered headings and their depth; unresolved numbered blocks
//!   can still be read by typography. Flash's numbered, labeled, chapter and caption detectors
//!   are not ported.
//! - The keyword tables are not ported except the introduction keywords: the section and
//!   abstract tables index academic and medical papers.
//! - A heading the native rules already accepted is kept; only new candidates face the
//!   font-distance rejection.
//! - Native line boxes rise to the font's ascent; Flash's glyph boxes stop near the capitals.
//! - A new heading is in the page's own type size or larger, opens with a capital or a number,
//!   is no quotation, citation or lead-in, and does not sit beside running text.
//! - Lines above the last party label on a page (`APPLICANT`, `— Respondent`) are the case
//!   caption, never headings; a heading of a few stray characters holds no section; a
//!   capitalized title wrapped onto a second line is one heading.
//! - Numbered and unnumbered headings compare by type, not numbering (see `outline.rs`).

mod detect;
mod model;
mod outline;
mod stats;

use detect::{is_body_paragraph, neighbor_map, Block, FlashPage, Scan};
use legal_pdf_core::model::Page;
use legal_pdf_core::layout_analysis;
use model::{x_aligned_center_close, FlashBlock, FlashLine};
use outline::{body_headings, candidate_fonts, stands_alone, Candidate, Context};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

pub fn party_label() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        let ordinal = r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|\d+(?:st|nd|rd|th))";
        let role = format!(
            r"(?:{ordinal}(?:\s*(?:,|and|&|to)\s*{ordinal})*\s+)?(?:named\s+)?(?:applicants?|respondents?|appellants?|appellees?|plaintiffs?|defendants?|petitioners?|claimants?|complainants?|prosecutors?|accused|interven[eo]rs?|amic(?:us|i)\s+curiae|(?:notice|third|interested|intervening|affected)\s+(?:part(?:y|ies)|persons?))"
        );
        // A label stands on its own line, or past a column gap after the party's name.
        Regex::new(&format!(
            r"(?i)(?:^[\-\u{{2013}}\u{{2014}}\u{{2015}}\u{{2026}}.\s]*|\s{{3,}})(?:the\s+)?{role}(?:\s*(?:/|,|and|&|\()\s*(?:the\s+)?{role})*[\s.,:;)]*$"
        ))
        .expect("party label")
    })
}

/// A capitalized heading wrapped onto the next line keeps one block: the same type, set at the
/// line's leading, the first line ending no sentence and the second opening no numbered heading.
fn join_wrapped_headings(
    pages: &mut [Page],
    levels: &HashMap<String, usize>,
    tagged_levels: &HashMap<String, usize>,
    counted: &[Vec<bool>],
    translation: &dyn Fn(&str) -> bool,
) {
    let style = |line: &legal_pdf_core::model::Line| {
        line.spans.first().map(|span| {
            (
                span.font.clone(),
                (span.size * 10.0).round() as i64,
                span.flags & 16,
            )
        })
    };
    for (page_slot, page) in pages.iter_mut().enumerate() {
        for slot in 1..page.lines.len() {
            let (prior, line) = (&page.lines[slot - 1], &page.lines[slot]);
            let height = prior.bbox[3] - prior.bbox[1];
            let gap = line.bbox[1] - prior.bbox[3];
            let centered =
                ((prior.bbox[0] + prior.bbox[2]) - (line.bbox[0] + line.bbox[2])).abs() / 2.0 < 2.0;
            let joined = prior.region_type == "heading"
                && line.region_type == "heading"
                && counted[page_slot][slot - 1]
                && counted[page_slot][slot]
                && translation(&prior.id) == translation(&line.id)
                && prior.block_index != line.block_index
                && style(prior).is_some()
                && style(prior) == style(line)
                && tagged_levels.get(&prior.id) == tagged_levels.get(&line.id)
                && model::CharStats::of(&prior.text).caps_heavy()
                && model::CharStats::of(&line.text).caps_heavy()
                && !prior.text.trim_end().ends_with(['.', '?', '!', ':', ';'])
                && !levels.contains_key(&line.id)
                && gap >= -1.0
                && gap < height
                && (centered || (prior.bbox[0] - line.bbox[0]).abs() < 2.0);
            if joined {
                let (from, to) = (line.block_index, prior.block_index);
                for later in page.lines[slot..]
                    .iter_mut()
                    .take_while(|later| later.block_index == from)
                {
                    later.block_index = to;
                }
            }
        }
    }
}

/// The introduction section names upstream's `introduction_dict` lists.
const INTRODUCTION: &[&str] = &[
    "introduction",
    "introductions",
    "introducción",
    "introdução",
    "einleitung",
    "introduzione",
    "inledningen",
    "esittely",
    "indledningen",
    "pendahuluan",
    "giriş",
    "introducerea",
    "introducció",
    "úvod",
    "wstęp",
    "inleiding",
    "inngangur",
    "sissejuhatus",
    "ievads",
    "įvadas",
    "bevezetés",
    "giới thiệu",
    "εισαγωγή",
    "简介",
    "簡介",
    "序章",
    "소개",
    "введение",
    "вступ",
    "увядзенне",
    "въведение",
];

fn introduction(block: &FlashBlock) -> bool {
    let tokens = block.tokens();
    let words = tokens
        .iter()
        .take_while(|token| token.kind == 2)
        .map(|token| token.text.to_lowercase())
        .collect::<Vec<_>>();
    if words.is_empty() || !INTRODUCTION.contains(&words.join(" ").as_str()) {
        return false;
    }
    // The whole block, or all but a closing punctuation mark.
    tokens.len() == words.len()
        || (tokens.len() == words.len() + 1 && matches!(tokens[words.len()].kind, 3 | 4 | 5))
}

/// Bold or capitals can set a title apart without a change in font size.
fn set_apart_by_type(block: &FlashBlock, next: Option<&FlashBlock>) -> bool {
    let stats = &block.stats;
    let total = f64::from(stats.total.max(1));
    if block.bold >= 0.95 {
        let figures = f64::from(stats.counts[1]) / total > 0.41;
        return f64::from(stats.punctuation()) / total <= 0.58
            && (!figures
                || next.is_none_or(|next| {
                    next.lines
                        .first()
                        .is_none_or(|line| line.char_count() <= 62)
                }));
    }
    stats.letters() > 0 && f64::from(stats.counts[2]) >= 0.78 * f64::from(stats.letters())
}

/// A line that is a citation and little else ("[2026] SGCA 26") names a case, not a section.
fn citation_only(block: &FlashBlock) -> bool {
    let text = block
        .lines
        .iter()
        .map(|line| line.text.trim())
        .collect::<Vec<_>>()
        .join(" ");
    // The layout reading of citations: the full citator would resolve each one, at a cost
    // every short block on a page of authorities pays.
    let spans = layout_analysis().protected_citation_spans(&text);
    if spans.is_empty() {
        return false;
    }
    text.chars()
        .enumerate()
        .filter(|(at, character)| {
            character.is_alphabetic() && !spans.iter().any(|(start, end)| start <= at && at < end)
        })
        .count()
        < 3
}

/// What the structure stage has read that the heading reader needs.
pub struct HeadingEvidence<'a> {
    /// Line id to level of every heading accepted so far; the reader adds and removes.
    pub levels: &'a mut HashMap<String, usize>,
    /// Lines that can be no heading: a contents row, a table cell.
    pub excluded: &'a dyn Fn(&str) -> bool,
    /// Parallel text whose numbering and parent stack must remain separate.
    pub translation: &'a dyn Fn(&str) -> bool,
    /// Contents pages, none of whose lines is a heading.
    pub contents_pages: &'a HashSet<usize>,
    /// Does a line open with a heading-like number ("1.", "(a)", "IV")? This is no role decision.
    pub numbered: &'a dyn Fn(&str) -> bool,
    /// Original source roles retain explicit PDF Hn levels after local classification.
    pub source_regions: Option<&'a HashMap<String, String>>,
}

/// Read the headings set apart by type alone and the depth of every heading. Line roles and
/// `evidence.levels` change; the caller rebuilds the pages' regions.
pub fn reconcile_styled_headings(pages: &mut [Page], evidence: HeadingEvidence) {
    let tagged_levels = evidence
        .source_regions
        .into_iter()
        .flatten()
        .filter(|(_, role)| legal_pdf_core::is_heading_role(role))
        .filter_map(|(id, role)| {
            role.strip_prefix('H')?
                .parse()
                .ok()
                .map(|level| (id.clone(), level))
        })
        .collect::<HashMap<String, usize>>();
    // The case caption: lines down to a page's last party label are no headings.
    let mut caption = HashSet::<String>::new();
    for page in pages.iter_mut() {
        let Some(last) = page
            .lines
            .iter()
            .rposition(|line| !line.exclude_from_body && party_label().is_match(line.text.trim()))
        else {
            continue;
        };
        for line in &mut page.lines[..=last] {
            caption.insert(line.id.clone());
            if line.region_type == "heading" && model::CharStats::of(&line.text).caps_heavy() {
                line.region_type = "body".to_owned();
                evidence.levels.remove(&line.id);
            }
        }
    }
    // Every typography reader uses the same population, including its neighbors and style seeds.
    let counted = pages
        .iter()
        .map(|page| {
            page.lines
                .iter()
                .map(|line| {
                    !line.exclude_from_body
                        && line.note_region_mode.is_empty()
                        && matches!(line.region_type.as_str(), "body" | "heading")
                        && !caption.contains(&line.id)
                        && !(evidence.excluded)(&line.id)
                        && !evidence.contents_pages.contains(&page.index)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    join_wrapped_headings(
        pages,
        evidence.levels,
        &tagged_levels,
        &counted,
        evidence.translation,
    );

    let flash_lines = pages
        .iter()
        .map(|page| {
            page.lines
                .iter()
                .map(|line| FlashLine::of(line, page.height))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let page_stats = pages
        .iter()
        .enumerate()
        .map(|(page_slot, page)| {
            let text = flash_lines[page_slot]
                .iter()
                .zip(&counted[page_slot])
                .filter_map(|(line, &keep)| keep.then_some(line));
            stats::page_stats(text, page.width)
        })
        .collect::<Vec<_>>();
    let doc = stats::doc_stats(&page_stats);

    // Blocks: runs of body or heading lines sharing a block, as regions group them.
    let mut blocks = Vec::<Block>::new();
    let mut flash_pages = Vec::<FlashPage>::new();
    for (page_slot, page) in pages.iter().enumerate() {
        let mut ids = Vec::new();
        let mut slot = 0;
        while slot < page.lines.len() {
            let line = &page.lines[slot];
            if !counted[page_slot][slot] {
                slot += 1;
                continue;
            }
            let mut end = slot + 1;
            while end < page.lines.len()
                && page.lines[end].region_type == line.region_type
                && page.lines[end].block_index == line.block_index
                && counted[page_slot][end]
                && (evidence.translation)(&page.lines[end].id) == (evidence.translation)(&line.id)
                && tagged_levels.get(&page.lines[end].id) == tagged_levels.get(&line.id)
            {
                end += 1;
            }
            let mut flash = FlashBlock::new();
            for index in slot..end {
                flash.add_line(flash_lines[page_slot][index].clone());
            }
            let lines = &page.lines[slot..end];
            let first = line.text.trim();
            let heading = line.region_type == "heading";
            blocks.push(Block {
                flash,
                page: page_slot,
                line_slots: (slot..end).collect(),
                heading,
                numbered: lines
                    .iter()
                    .any(|line| evidence.levels.contains_key(&line.id))
                    || (evidence.numbered)(first),
                is_body: false,
                translation: (evidence.translation)(&line.id),
            });
            ids.push(blocks.len() - 1);
            slot = end;
        }
        flash_pages.push(FlashPage {
            width: page.width,
            height: page.height,
            stats: page_stats[page_slot].clone(),
            blocks: ids,
            body_styles: HashSet::new(),
        });
    }
    for page in &mut flash_pages {
        for &block in &page.blocks {
            let body =
                !blocks[block].heading && is_body_paragraph(&doc, page, &blocks[block].flash);
            blocks[block].is_body = body;
            if body {
                page.body_styles
                    .insert(blocks[block].flash.dominant_style().to_owned());
            }
        }
    }
    // A new heading is a short block set in the page's own type size or larger, opens with a
    // capital or a number, and stands on rows of its own beside no running text. A colon-ending
    // title needs the stronger type witness below.
    // The citation grammar's answer for each text, read once: a document repeats its running heads
    // and titles page after page.
    let mut eligible = vec![true; blocks.len()];
    let mut type_candidates = vec![false; blocks.len()];
    let mut cited: HashMap<String, bool> = HashMap::new();
    for page in &flash_pages {
        for (position, &block) in page.blocks.iter().enumerate() {
            let flash = &blocks[block].flash;
            let text = flash
                .lines
                .iter()
                .map(|line| line.text.trim())
                .collect::<Vec<_>>()
                .join(" ");
            if flash.line_count() > 4 || text.chars().count() > 300 {
                eligible[block] = false;
                continue;
            }
            let next = page.blocks[position + 1..]
                .iter()
                .find(|&&next| blocks[next].translation == blocks[block].translation)
                .map(|&next| &blocks[next].flash);
            type_candidates[block] = set_apart_by_type(flash, next);
            let opens = text
                .chars()
                .find(|character| character.is_alphanumeric())
                .is_some_and(|character| character.is_uppercase() || character.is_numeric());
            let ends = text
                .trim_end_matches(|character: char| {
                    character.is_whitespace() || "\"')]\u{201d}\u{2019}".contains(character)
                })
                .chars()
                .last();
            let quoted = text.starts_with(['"', '\u{201c}', '\u{2018}'])
                || text.ends_with(".\u{201d}")
                || text.ends_with("\u{201d}.");
            // The type and wording tests are cheap; the layout tests below read the whole page.
            if flash.size < page.stats.font_size - 0.5
                || !opens
                || quoted
                || matches!(ends, Some(',' | ';'))
                || (ends == Some(':') && !type_candidates[block])
            {
                eligible[block] = false;
                continue;
            }
            // Running text in the next column is not beside a block set in a column of its own,
            // one where running text stands above or below it clear of that next column.
            let column = |rect: &model::Rect| {
                page.blocks.iter().any(|&own| {
                    let own = (&blocks[own].flash, blocks[own].is_body);
                    own.1
                        && own.0.line_count() >= 2
                        && !model::y_overlaps(&own.0.rect, &flash.rect)
                        && own.0.rect.left < flash.rect.right
                        && flash.rect.left < own.0.rect.right
                        && (own.0.rect.right < rect.left || own.0.rect.left > rect.right)
                })
            };
            let beside = || {
                page.blocks.iter().any(|&other| {
                    let rect = &blocks[other].flash.rect;
                    other != block
                        && blocks[other].is_body
                        && model::y_overlaps(rect, &flash.rect)
                        && (rect.right < flash.rect.left || rect.left > flash.rect.right)
                        && !column(rect)
                })
            };
            // Indented under the margin of the running text in its column, off the page's centre:
            // quoted material, whose headings are the quoted document's, not this one's.
            let margin = page
                .blocks
                .iter()
                .map(|&other| &blocks[other])
                .filter(|other| {
                    other.is_body
                        && other.flash.line_count() >= 2
                        && other.flash.rect.left < flash.rect.right
                        && flash.rect.left < other.flash.rect.right
                })
                .map(|other| other.flash.rect.left)
                .fold(f64::INFINITY, f64::min);
            // A line whose sentence runs on into the block set close below it is that block's
            // opening words, set apart in bold, not its heading.
            let runs_on = page.blocks.get(position + 1).is_some_and(|&next| {
                let next = &blocks[next].flash;
                next.lines
                    .first()
                    .is_some_and(|line| line.text.trim_start().starts_with(char::is_lowercase))
                    && flash.rect.bottom - next.rect.top < 0.5 * flash.size
                    && model::left_aligned(&flash.rect, &next.rect, 2.0)
            });
            let indented = margin.is_finite()
                && flash.rect.left > margin + 1.5 * flash.size
                && (flash.rect.center_x() - page.width / 2.0).abs() >= 0.05 * page.width;
            // The citation grammar is costly: it reads only blocks every layout test admits. A
            // citation carries a number; a cue word alone ("people", "re") names no authority.
            let analysis = layout_analysis();
            eligible[block] = !indented
                && !runs_on
                && !beside()
                && !*cited.entry(text.clone()).or_insert_with(|| {
                    (analysis.has_citation_cue(&text) && text.chars().any(|c| c.is_ascii_digit()))
                        || analysis.has_citation_signal(&text)
                })
                && !citation_only(flash);
        }
    }
    let mut positions = vec![0; blocks.len()];
    let mut centered = vec![false; blocks.len()];
    for page in &flash_pages {
        for (position, &block) in page.blocks.iter().enumerate() {
            positions[block] = position;
            let next = page
                .blocks
                .get(position + 1)
                .map(|&next| &blocks[next].flash.rect);
            centered[block] = x_aligned_center_close(&blocks[block].flash, page.width, next);
        }
    }
    let neighbors = flash_pages
        .iter()
        .map(|page| neighbor_map(page, &blocks))
        .collect::<Vec<_>>();

    let mut candidates = Vec::<Candidate>::new();
    let mut proposed = Vec::<usize>::new();
    for (page_index, page) in flash_pages.iter().enumerate() {
        let mut scan = Scan {
            doc: &doc,
            pages: &flash_pages,
            blocks: &blocks,
            page: page_index,
            neighbors: neighbors[page_index].clone(),
            proposed: HashSet::new(),
        };
        let mut found = Vec::new();
        for (position, &block) in page.blocks.iter().enumerate() {
            if blocks[block].heading {
                let tagged_level = blocks[block]
                    .line_slots
                    .first()
                    .and_then(|slot| tagged_levels.get(&pages[page_index].lines[*slot].id))
                    .copied();
                let numbered_level = blocks[block]
                    .line_slots
                    .iter()
                    .filter_map(|slot| evidence.levels.get(&pages[page_index].lines[*slot].id))
                    .min()
                    .copied();
                candidates.push(Candidate {
                    block,
                    kind: if tagged_level.is_some() {
                        2
                    } else if numbered_level.is_some() {
                        1
                    } else {
                        0
                    },
                    level: tagged_level.or(numbered_level),
                });
                continue;
            }
            if !eligible[block] {
                continue;
            }
            let rows = blocks[block].flash.line_count();
            let kind = if rows <= 2
                && !blocks[block].numbered
                && neighbors[page_index][position].body_crossed
                && introduction(&blocks[block].flash)
            {
                Some(11)
            } else if type_candidates[block] {
                Some(12)
            } else if scan.candidate(position) {
                Some(0)
            } else {
                None
            };
            if let Some(kind) = kind {
                scan.proposed.insert(position);
                found.push(Candidate {
                    block,
                    kind,
                    level: None,
                });
            }
        }
        // Apply the page cap to each parallel sequence independently.
        for translation in [false, true] {
            let own = found
                .iter()
                .filter(|candidate| blocks[candidate.block].translation == translation);
            if own.clone().count() < 20 {
                proposed.extend(own.clone().map(|candidate| candidate.block));
                candidates.extend(own.cloned());
            }
        }
    }
    let context = Context {
        blocks: &blocks,
        pages: &flash_pages,
        positions: &positions,
        centered: &centered,
    };
    candidates.sort_by(|left, right| context.order(left.block, right.block));
    for block in body_headings(&context, &candidates, &neighbors) {
        if eligible[block] {
            proposed.push(block);
            candidates.push(Candidate {
                block,
                kind: 0,
                level: None,
            });
        }
    }
    candidates.sort_by(|left, right| context.order(left.block, right.block));
    let proposed = proposed.into_iter().collect::<HashSet<_>>();
    // Every candidate's fonts, read once for all the proposed headings they are compared with.
    let fonts = std::cell::OnceCell::new();
    let rejected = (0..candidates.len())
        .filter(|index| {
            let candidate = &candidates[*index];
            proposed.contains(&candidate.block) && candidate.kind == 0 && {
                let block = &blocks[candidate.block];
                let near = &neighbors[block.page][positions[candidate.block]];
                let gap = near.body_below.map(|slot| {
                    block.flash.rect.bottom
                        - blocks[flash_pages[block.page].blocks[slot]].flash.rect.top
                });
                let fonts = fonts.get_or_init(|| candidate_fonts(&context, &candidates));
                stands_alone(&context, &candidates, fonts, *index, gap)
            }
        })
        .collect::<HashSet<_>>();
    // A heading of a few stray characters ("l ¦ I") holds no section.
    let stray = |candidate: &Candidate| {
        matches!(candidate.kind, 0 | 12) && blocks[candidate.block].flash.stats.info_weight() <= 3.0
    };
    for candidate in candidates.iter().filter(|candidate| stray(candidate)) {
        let block = &blocks[candidate.block];
        for &slot in &block.line_slots {
            evidence.levels.remove(&pages[block.page].lines[slot].id);
        }
    }
    let accepted = candidates
        .iter()
        .enumerate()
        .filter(|(index, candidate)| !rejected.contains(index) && !stray(candidate))
        .map(|(_, candidate)| candidate.clone())
        .collect::<Vec<_>>();

    // Parallel sequences retain independent open ancestors.
    let mut stacks = [Vec::<usize>::new(), Vec::<usize>::new()];
    let mut levels = vec![0usize; accepted.len()];
    for index in 0..accepted.len() {
        let stack = &mut stacks[usize::from(blocks[accepted[index].block].translation)];
        let parent = context.parent(stack, &accepted, &accepted[index]);
        let inferred = parent.map_or(1, |parent| levels[parent].saturating_add(1));
        levels[index] = if accepted[index].kind == 2 {
            accepted[index].level.unwrap_or(inferred)
        } else {
            inferred
        };
        stack.push(index);
    }
    // Region assembly groups lines by their block. Give each accepted heading and each inner
    // ladder step its own block, retaining wrapped lines at one level together.
    let mut next_blocks = pages
        .iter()
        .map(|page| {
            page.lines
                .iter()
                .map(|line| line.block_index)
                .max()
                .unwrap_or(0)
                + 1
        })
        .collect::<Vec<_>>();
    for (candidate, level) in accepted.iter().zip(&levels) {
        let block = &blocks[candidate.block];
        let mut block_index = next_blocks[block.page];
        next_blocks[block.page] += 1;
        let mut prior_level = None;
        for &slot in &block.line_slots {
            let line = &mut pages[block.page].lines[slot];
            if proposed.contains(&candidate.block) {
                line.region_type = "heading".to_owned();
            }
            let step = (candidate.kind != 2)
                .then_some(candidate.level)
                .flatten()
                .and_then(|top| {
                    evidence
                        .levels
                        .get(&line.id)
                        .map(|own| own.saturating_sub(top))
                });
            let own_level = level.saturating_add(step.unwrap_or(0));
            if prior_level.is_some_and(|prior| prior != own_level) {
                block_index = next_blocks[block.page];
                next_blocks[block.page] += 1;
            }
            prior_level = Some(own_level);
            line.block_index = block_index;
            evidence.levels.insert(line.id.clone(), own_level);
        }
    }
}

// PageIndex license, for the code ported in this module:
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
