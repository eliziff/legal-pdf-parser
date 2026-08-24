//! Shared structure derivation for aligned page and line evidence.

mod graph;

use crate::layout::*;
pub use graph::PdfTextIndex;
use graph::{contents_leader_re, map_note_pairs, native_graph_parts, PdfResolutionInput};
#[cfg(test)]
use graph::{contents_row, index_pages};
use legal_pdf_core::model::{
    DetachedReference, Diagnostic, Footnote, FootnoteCrossref, LegalDocument, Line, Page,
    Paragraph, ParagraphAnchor, PdfPairingAudit,
};
#[cfg(test)]
use legal_pdf_core::model::{NotePairClaim, NotePairKind, Span};
use legal_pdf_core::{line_font_size, Anchor, Error, Result};
use legal_pdf_support::{
    enumerator_interpretations, has_citation_signal, heading_text_plausible, parse_heading_ladder,
    EnumeratorInterpretation, HeadingAction, HeadingFamilyStats, HeadingLadderStatus,
};
use legal_structure::{
    last_scalars, normalize_decimal_digit, normalize_note_symbol, resolve_structure_graph,
    utf16_len, DocumentStructure, NodeKind, ResolutionRuleV2, ScalarRange, ScalarText,
};
#[cfg(test)]
use legal_structure::{CandidateGrammar, CandidateObservationV2};
use regex::Regex;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::OnceLock;

const HARD_DIAGNOSTICS: &[&str] = &[
    "COLUMN_ORDER_UNCERTAIN",
    "FOOTNOTE_UNMATCHED_LABEL",
    "FOOTNOTE_UNMATCHED_REFERENCE",
    "FOOTNOTE_REGION_UNCERTAIN",
    "TEXT_QUALITY_LOW",
];
pub(super) const MAX_SYMBOL_LABEL_LEN: usize = 8;

#[derive(Debug, Clone)]
pub(super) struct LabelPrefix {
    pub(super) label: String,
    start: usize,
    pub(super) end: usize,
}

#[derive(Debug, Default)]
struct PdfPrimitiveEvidence {
    source_regions: Option<HashMap<String, String>>,
    contents_pages: HashSet<usize>,
    table_cell_line_ids: HashSet<String>,
    table_note_line_ids: HashSet<String>,
    heading_levels: HashMap<String, usize>,
}

#[derive(Debug)]
struct PdfPreparation {
    diagnostics: Vec<Diagnostic>,
    primitives: PdfPrimitiveEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructureIdentity {
    pub document_id: String,
    pub source_sha256: String,
}

pub struct StructureOutput {
    pub paragraphs: Vec<Paragraph>,
    pub footnotes: Vec<Footnote>,
    pub diagnostics: Vec<Diagnostic>,
    pub pairing_audit: Option<PdfPairingAudit>,
    pub structure_graph: DocumentStructure,
}

pub struct StructureReplay {
    pub prepared_pages: Vec<Page>,
    pub derived: StructureOutput,
}

fn normalize_label(value: &str) -> String {
    let translated: String = value
        .trim()
        .chars()
        .map(|character| {
            normalize_decimal_digit(character).unwrap_or_else(|| normalize_note_symbol(character))
        })
        .collect();
    translated
        .parse::<u64>()
        .map_or(translated, |number| number.to_string())
}

pub(super) fn scalar_suffix(value: &str, start: usize) -> &str {
    if value.is_ascii() {
        return &value[start.min(value.len())..];
    }
    let end = value.len();
    let byte = value.char_indices().nth(start).map_or(end, |at| at.0);
    &value[byte..]
}

#[cfg(test)]
pub(super) fn char_slice(value: &str, start: usize, end: usize) -> &str {
    let length = value.chars().count();
    ScalarText::new(value)
        .slice(start.min(length)..end.min(length))
        .expect("valid scalar range")
}

pub(super) fn is_note_symbol(character: char) -> bool {
    matches!(
        character,
        '*' | '∗' | '\u{f02a}' | '†' | '‡' | '§' | '¶' | '#'
    )
}

fn line_start_label_prefix(text: &str) -> Option<LabelPrefix> {
    let leading = text
        .chars()
        .take_while(|character| character.is_whitespace())
        .count();
    let trimmed = text.trim_start();
    let symbols: String = trimmed
        .chars()
        .take(MAX_SYMBOL_LABEL_LEN)
        .take_while(|character| is_note_symbol(*character))
        .collect();
    let token = if symbols.is_empty() {
        trimmed
            .chars()
            .take(4)
            .take_while(char::is_ascii_digit)
            .collect()
    } else {
        symbols
    };
    if token.is_empty() {
        return None;
    }
    let token_chars = token.chars().count();
    let remainder = scalar_suffix(trimmed, token_chars);
    let embedded_endnote = format!("endnote {token}");
    let embedded_chars = embedded_endnote.chars().count();
    if remainder
        .get(..embedded_endnote.len())
        .is_some_and(|value| value.eq_ignore_ascii_case(&embedded_endnote))
    {
        return Some(LabelPrefix {
            label: normalize_label(&token),
            start: leading,
            end: leading + token_chars + embedded_chars,
        });
    }
    let mut remainder_chars = remainder.chars();
    let valid_boundary = match remainder_chars.next() {
        Some(character) if character.is_whitespace() => true,
        Some(character) if ".)],:;-".contains(character) => remainder_chars
            .next()
            .is_none_or(|next| next.is_whitespace()),
        _ => false,
    };
    if !valid_boundary {
        return None;
    }
    let label = normalize_label(&token);
    Some(LabelPrefix {
        label,
        start: leading,
        end: leading + token_chars,
    })
}

pub(super) fn label_prefix(text: &str) -> Option<LabelPrefix> {
    if let Some(prefix) = line_start_label_prefix(text) {
        return Some(prefix);
    }
    let leading = text
        .chars()
        .take_while(|character| character.is_whitespace())
        .count();
    let stripped = text.trim();
    let pure = ((1..=MAX_SYMBOL_LABEL_LEN).contains(&stripped.chars().count())
        && stripped.chars().all(is_note_symbol))
        || ((1..=4).contains(&stripped.chars().count())
            && stripped.chars().all(|character| character.is_ascii_digit()));
    pure.then(|| LabelPrefix {
        label: normalize_label(stripped),
        start: leading,
        end: leading + stripped.chars().count(),
    })
}

pub(super) fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn upper_quartile(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    values[(values.len() * 3 / 4).min(values.len() - 1)]
}

fn label_is_typographic(line: &Line, prefix: &LabelPrefix, line_size: f64, body_size: f64) -> bool {
    let spans = || {
        line.spans
            .iter()
            .filter(|span| span.start < prefix.end && span.end > prefix.start)
    };
    let label_size = spans()
        .filter_map(|span| (span.size > 0.0).then_some(span.size))
        .min_by(f64::total_cmp)
        .unwrap_or(line_size);
    let height = line.bbox[3] - line.bbox[1];
    spans().any(|span| {
        span.superscript
            || (line_size > 0.0 && span.size > 0.0 && span.size <= line_size * 0.75)
            || (line_size > 0.0
                && span.size > 0.0
                && span.size * 1.25 <= line_size
                && height > 0.0
                && span.bbox[3] <= line.bbox[3] - 0.25 * height)
    }) || (label_size > 0.0 && label_size <= body_size * 0.75)
}

fn normalize_furniture(text: &str) -> String {
    static DIGITS: OnceLock<Regex> = OnceLock::new();
    let text = DIGITS
        .get_or_init(|| Regex::new(r"\d+").unwrap())
        .replace_all(text, "#");
    let mut normalized = String::with_capacity(text.len());
    let mut pending_space = false;
    for character in text.chars().flat_map(char::to_lowercase) {
        if character.is_whitespace() {
            pending_space = !normalized.is_empty();
        } else {
            if pending_space {
                normalized.push(' ');
                pending_space = false;
            }
            normalized.push(character);
        }
    }
    normalized
}

fn compact_note_line(text: &str) -> bool {
    let Some(prefix) = line_start_label_prefix(text) else {
        return false;
    };
    scalar_suffix(text, prefix.end)
        .trim_start_matches(|character: char| {
            character.is_whitespace() || ".)],:;-".contains(character)
        })
        .chars()
        .filter(|character| character.is_alphabetic())
        .take(4)
        .count()
        == 4
}

fn standalone_enumerator_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^\s*([IVXLCDM]{1,7}|[A-Za-z]|\d{1,3}|\d{1,2}(?:\.\d{1,2}){1,3})([.)])\s*$")
            .unwrap()
    })
}

fn inline_enumerator_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"^\s*([IVXLCDM]{1,7}|[A-Za-z]|\d{1,3}|\d{1,2}(?:\.\d{1,2}){1,3})([.)])\s+(\S.*)$",
        )
        .unwrap()
    })
}

fn standalone_enumerator(text: &str) -> bool {
    standalone_enumerator_re().is_match(text)
}

const FOLIO_MIN_SEQUENCE_PAGES: usize = 4;
const FOLIO_EDGE_MAX_FRAC: f64 = 0.25;
const FOLIO_BOTTOM_MIN_FRAC: f64 = 0.92;
const FURNITURE_Y_TOLERANCE_FRAC: f64 = 0.015;
const FURNITURE_X_TOLERANCE_FRAC: f64 = 0.04;

#[derive(Clone, Copy)]
struct FolioCandidate {
    page_slot: usize,
    line_slot: usize,
    page_number: u32,
    printed_number: u32,
    right: bool,
    y_ratio: f64,
    x_ratio: f64,
}

#[derive(Clone, Copy)]
struct FurnitureHit {
    page_slot: usize,
    line_slot: usize,
    page_number: u32,
    y_ratio: f64,
    x_ratio: f64,
}

fn arabic_page_number(value: &str) -> Option<u32> {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*-?\s*(\d{1,4})\s*-?\s*$").unwrap())
        .captures(value)
        .and_then(|capture| capture.get(1))
        .and_then(|value| value.as_str().parse().ok())
}

/// Text-Fidelity's alternating-folio witness. A singleton Arabic number at
/// each outer bottom edge is furniture only when at least four consecutive
/// PDF pages carry a +1 printed sequence, alternate sides, and remain aligned
/// within each page parity.
fn alternating_folios(pages: &[Page]) -> HashSet<(usize, usize)> {
    let mut singletons = Vec::new();
    for (page_slot, page) in pages.iter().enumerate() {
        if page.width <= 0.0 || page.height <= 0.0 {
            continue;
        }
        let mut candidates = page
            .lines
            .iter()
            .enumerate()
            .filter_map(|(line_slot, line)| {
                let printed_number = arabic_page_number(&line.text)?;
                let center_x = (line.bbox[0] + line.bbox[2]) / 2.0;
                let x_ratio = center_x / page.width;
                let right = if x_ratio < FOLIO_EDGE_MAX_FRAC {
                    false
                } else if x_ratio > 1.0 - FOLIO_EDGE_MAX_FRAC {
                    true
                } else {
                    return None;
                };
                (line.bbox[1] > page.height * FOLIO_BOTTOM_MIN_FRAC).then_some(FolioCandidate {
                    page_slot,
                    line_slot,
                    page_number: page.number,
                    printed_number,
                    right,
                    y_ratio: line.bbox[1] / page.height,
                    x_ratio,
                })
            });
        if let Some(candidate) = candidates.next() {
            if candidates.next().is_none() {
                singletons.push(candidate);
            }
        }
    }
    singletons.sort_by_key(|candidate| candidate.page_number);

    fn aligned(run: &[FolioCandidate], candidate: FolioCandidate) -> bool {
        let previous = run.last().expect("nonempty folio run");
        if candidate.page_number != previous.page_number + 1
            || candidate.printed_number != previous.printed_number + 1
            || candidate.right == previous.right
            || run.iter().any(|old| {
                old.page_number % 2 == candidate.page_number % 2 && old.right != candidate.right
            })
        {
            return false;
        }
        let mut y_min = candidate.y_ratio;
        let mut y_max = candidate.y_ratio;
        let mut side_x_min = candidate.x_ratio;
        let mut side_x_max = candidate.x_ratio;
        for old in run {
            y_min = y_min.min(old.y_ratio);
            y_max = y_max.max(old.y_ratio);
            if old.right == candidate.right {
                side_x_min = side_x_min.min(old.x_ratio);
                side_x_max = side_x_max.max(old.x_ratio);
            }
        }
        y_max - y_min <= FURNITURE_Y_TOLERANCE_FRAC
            && side_x_max - side_x_min <= FURNITURE_X_TOLERANCE_FRAC
    }

    let mut admitted = HashSet::new();
    let mut run = Vec::new();
    let flush = |run: &mut Vec<FolioCandidate>, admitted: &mut HashSet<(usize, usize)>| {
        if run.len() >= FOLIO_MIN_SEQUENCE_PAGES {
            admitted.extend(
                run.iter()
                    .map(|candidate| (candidate.page_slot, candidate.line_slot)),
            );
        }
        run.clear();
    };
    for candidate in singletons {
        if !run.is_empty() && !aligned(&run, candidate) {
            flush(&mut run, &mut admitted);
        }
        run.push(candidate);
    }
    flush(&mut run, &mut admitted);
    admitted
}

fn aligned_furniture(hits: &[FurnitureHit], minimum: usize) -> HashSet<(usize, usize)> {
    fn unique_page_count<'a>(
        hits: impl IntoIterator<Item = &'a FurnitureHit>,
        seen: &mut [u32],
        generation: &mut u32,
    ) -> usize {
        *generation = generation.wrapping_add(1);
        if *generation == 0 {
            seen.fill(0);
            *generation = 1;
        }
        let mut count = 0;
        for hit in hits {
            if seen[hit.page_slot] != *generation {
                seen[hit.page_slot] = *generation;
                count += 1;
            }
        }
        count
    }

    let mut best = Vec::with_capacity(hits.len());
    let mut best_page_count = 0;
    let mut y_cluster = Vec::with_capacity(hits.len());
    let mut parity_hits = Vec::with_capacity(hits.len());
    let mut trial = Vec::with_capacity(hits.len());
    let mut seen_pages = vec![0_u32; hits.iter().map(|hit| hit.page_slot).max().unwrap_or(0) + 1];
    let mut x_counts = BTreeMap::new();
    let mut generation = 0;
    for y_start in hits {
        y_cluster.clear();
        y_cluster.extend(hits.iter().copied().filter(|hit| {
            hit.y_ratio >= y_start.y_ratio
                && hit.y_ratio - y_start.y_ratio <= FURNITURE_Y_TOLERANCE_FRAC
        }));
        trial.clear();
        for parity in [0, 1] {
            x_counts.clear();
            parity_hits.clear();
            parity_hits.extend(
                y_cluster
                    .iter()
                    .copied()
                    .filter(|hit| hit.page_number % 2 == parity),
            );
            let mut selected_x = None;
            let mut selected_count = 0;
            for x_start in &parity_hits {
                let x = x_start.x_ratio.to_bits();
                let count = *x_counts.entry(x).or_insert_with(|| {
                    unique_page_count(
                        parity_hits.iter().filter(|hit| {
                            hit.x_ratio >= x_start.x_ratio
                                && hit.x_ratio - x_start.x_ratio <= FURNITURE_X_TOLERANCE_FRAC
                        }),
                        &mut seen_pages,
                        &mut generation,
                    )
                });
                // Iterator::max_by_key chooses the last equal maximum.
                if selected_x.is_none() || count >= selected_count {
                    selected_x = Some(x_start.x_ratio);
                    selected_count = count;
                }
            }
            if let Some(x_start) = selected_x {
                trial.extend(parity_hits.iter().copied().filter(|hit| {
                    hit.x_ratio >= x_start && hit.x_ratio - x_start <= FURNITURE_X_TOLERANCE_FRAC
                }));
            }
        }
        let trial_page_count = unique_page_count(&trial, &mut seen_pages, &mut generation);
        if trial_page_count > best_page_count {
            best.clear();
            best.extend_from_slice(&trial);
            best_page_count = trial_page_count;
        }
    }
    if best_page_count >= minimum {
        best.into_iter()
            .map(|hit| (hit.page_slot, hit.line_slot))
            .collect()
    } else {
        HashSet::new()
    }
}

fn mark_repeated_furniture(pages: &mut [Page]) {
    let mut candidates: HashMap<(bool, String), Vec<FurnitureHit>> = HashMap::new();
    legal_pdf_support::profile::measure("furniture.candidates", || {
        for (page_slot, page) in pages.iter().enumerate() {
            for (line_slot, line) in page.lines.iter().enumerate() {
                if page.width <= 0.0 || page.height <= 0.0 {
                    continue;
                }
                let edge = if line.bbox[3] < page.height * 0.12 {
                    Some(true)
                } else if line.bbox[1] > page.height * 0.90 {
                    Some(false)
                } else {
                    None
                };
                let Some(top) = edge else {
                    continue;
                };
                let normalized = normalize_furniture(&line.text);
                if !normalized.is_empty() {
                    candidates
                        .entry((top, normalized))
                        .or_default()
                        .push(FurnitureHit {
                            page_slot,
                            line_slot,
                            page_number: page.number,
                            y_ratio: line.bbox[1] / page.height,
                            x_ratio: (line.bbox[0] + line.bbox[2]) / (2.0 * page.width),
                        });
                }
            }
        }
    });
    // Preserve the old engine's two-page/parity behavior, but cap the witness
    // at four pages as Text-Fidelity does once alignment is also required.
    let minimum = 4_usize.min(2_usize.max(((pages.len() as f64) * 0.35).ceil() as usize));
    // A cover-page title can repeat the running title away from the header
    // baseline. Admit the stable occurrence cluster, not every matching string.
    let repeated: HashSet<(usize, usize)> =
        legal_pdf_support::profile::measure("furniture.alignment", || {
            candidates
                .into_values()
                .flat_map(|hits| aligned_furniture(&hits, minimum))
                .collect()
        });
    let sequence_folios =
        legal_pdf_support::profile::measure("furniture.folios", || alternating_folios(pages));
    legal_pdf_support::profile::measure("furniture.apply", || {
        for (page_slot, page) in pages.iter_mut().enumerate() {
            let line_sizes: Vec<_> = page.lines.iter().map(line_font_size).collect();
            let body_size = upper_quartile(
                page.lines
                    .iter()
                    .enumerate()
                    .filter(|(_, line)| {
                        line.bbox[1] >= page.height * 0.10 && line.bbox[1] <= page.height * 0.75
                    })
                    .map(|(index, _)| line_sizes[index])
                    .filter(|size| (4.0..=24.0).contains(size))
                    .collect(),
            );
            let body_size = if body_size > 0.0 { body_size } else { 10.0 };
            let attached_labels: HashSet<usize> = page
                .lines
                .iter()
                .enumerate()
                .filter(|(_, label)| {
                    standalone_note_label(label)
                        && page.lines.iter().enumerate().any(|(body_index, body)| {
                            !repeated.contains(&(page_slot, body_index))
                                && aligned_note_body(label, body)
                        })
                })
                .map(|(index, _)| index)
                .collect();
            let attached_enumerators: HashSet<usize> = page
                .lines
                .iter()
                .enumerate()
                .filter(|(_, label)| {
                    standalone_enumerator(&label.text)
                        && page.lines.iter().any(|body| aligned_note_body(label, body))
                })
                .map(|(index, _)| index)
                .collect();
            // Repeated shortforms such as "Ibid." remain note bodies when a small
            // detached label shares their baseline. Repetition alone is not enough
            // to turn a source-backed legal citation into running furniture.
            let citation_note_lines: HashSet<usize> = page
                .lines
                .iter()
                .enumerate()
                .filter_map(|(label_index, label)| {
                    if !standalone_note_label(label)
                        || label.bbox[1] <= page.height * 0.86
                        || sequence_folios.contains(&(page_slot, label_index))
                    {
                        return None;
                    }
                    let body_index = aligned_note_body_index(&page.lines, label_index)?;
                    let body = &page.lines[body_index];
                    (body.bbox[1] > page.height * 0.86
                        && line_sizes[body_index] < body_size * 0.90
                        && has_citation_signal(&body.text))
                    .then_some([label_index, body_index])
                })
                .flatten()
                .collect();
            for (index, line) in page.lines.iter_mut().enumerate() {
                let line_size = line_sizes[index];
                let at_top = line.bbox[3] < page.height * 0.12;
                let at_bottom = line.bbox[1] > page.height * 0.90;
                let page_number_at_top = line.bbox[3] < page.height * 0.14;
                let compact_note =
                    at_bottom && line_size < body_size * 0.90 && compact_note_line(&line.text);
                if sequence_folios.contains(&(page_slot, index)) {
                    line.region_type = "footer".to_owned();
                } else if repeated.contains(&(page_slot, index))
                    && (at_top || at_bottom)
                    && (line_size >= body_size * 0.75 || normalize_furniture(&line.text) != "#")
                    && !compact_note
                    && !attached_labels.contains(&index)
                    && !attached_enumerators.contains(&index)
                    && !citation_note_lines.contains(&index)
                {
                    line.region_type = if at_top { "header" } else { "footer" }.to_owned();
                } else if (line.bbox[3] < page.height * 0.14 || line.bbox[1] > page.height * 0.86)
                    && footer_page_number(&line.text)
                    && line_size >= body_size * 0.75
                    && !attached_labels.contains(&index)
                    && !citation_note_lines.contains(&index)
                {
                    line.region_type = if page_number_at_top {
                        "header"
                    } else {
                        "footer"
                    }
                    .to_owned();
                }
            }
        }
    });
}

fn printed_label(value: &str) -> Option<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)^(?:page\s+)?(\d{1,6}|[ivxlcdm]{1,12})$").unwrap())
        .captures(value.trim())
        .and_then(|capture| capture.get(1))
        .map(|value| value.as_str().to_owned())
}

fn footer_page_number(value: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)^\s*-?\s*(?:page\s+)?[ivxlcdm\d]+\s*-?\s*$").unwrap())
        .is_match(value.trim())
}

fn standalone_reference(text: &str) -> Option<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^(?:\d{1,4}|[*\u{2020}\u{2021}\u{00a7}\u{00b6}#])$").unwrap())
        .is_match(text)
        .then(|| normalize_label(text))
}

fn detached_reference_target(
    reference_index: usize,
    lines: &[Line],
    body_size: f64,
) -> Option<(usize, usize)> {
    let reference = &lines[reference_index];
    let reference_x = (reference.bbox[0] + reference.bbox[2]) / 2.0;
    lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| {
            if index == reference_index
                || line.exclude_from_body
                || matches!(line.region_type.as_str(), "header" | "footer")
                || line_font_size(line) < body_size * 0.80
            {
                return None;
            }
            let y_distance = (line.bbox[1] - reference.bbox[1]).abs();
            if y_distance > ((line.bbox[3] - line.bbox[1]) * 0.20).max(2.0) {
                return None;
            }
            let (distance, offset) = line
                .spans
                .iter()
                .flat_map(|span| [(span.bbox[0], span.start), (span.bbox[2], span.end)])
                .map(|(x, offset)| ((reference_x - x).abs(), offset))
                .min_by(|left, right| left.0.total_cmp(&right.0).then(right.1.cmp(&left.1)))?;
            (distance <= body_size.max(6.0)).then_some((
                (
                    distance,
                    y_distance,
                    line.source_index.abs_diff(reference.source_index),
                ),
                index,
                offset,
            ))
        })
        .min_by(|left, right| {
            left.0
                 .0
                .total_cmp(&right.0 .0)
                .then(left.0 .1.total_cmp(&right.0 .1))
                .then(left.0 .2.cmp(&right.0 .2))
        })
        .map(|(_, index, offset)| (index, offset))
}

fn associate_detached_references(pages: &mut [Page], separators: &[Option<f64>]) {
    for page in pages {
        let body_size = median(
            page.lines
                .iter()
                .filter(|line| {
                    !matches!(line.region_type.as_str(), "header" | "footer")
                        && line.bbox[1] < page.height * 0.75
                })
                .map(line_font_size)
                .filter(|size| (7.0..=20.0).contains(size))
                .collect(),
        );
        let body_size = if body_size > 0.0 { body_size } else { 10.0 };
        let note_cut = separators
            .get(page.index)
            .copied()
            .flatten()
            .unwrap_or(page.height * 0.88);
        // The oracle consumes accepted markers one at a time. Preserve that
        // mutation order so a consumed row cannot become a later host.
        for marker_index in 0..page.lines.len() {
            let Some(label) = standalone_reference(page.lines[marker_index].text.trim()) else {
                continue;
            };
            let size = line_font_size(&page.lines[marker_index]);
            if matches!(
                page.lines[marker_index].region_type.as_str(),
                "header" | "footer"
            ) || !(0.0 < size && size <= body_size * 0.75)
                || page.lines[marker_index].bbox[1] >= note_cut
            {
                continue;
            }
            let Some((host_index, offset)) =
                detached_reference_target(marker_index, &page.lines, body_size)
            else {
                continue;
            };
            let source_line_id = page.lines[marker_index].id.clone();
            let selected_text = page.lines[marker_index].text.trim().to_owned();
            page.lines[host_index]
                .detached_references
                .push(DetachedReference {
                    note_id: label,
                    selected_text,
                    start_offset: offset,
                    end_offset: offset,
                    source_line_id,
                });
            page.lines[marker_index].exclude_from_body = true;
        }
        associate_spliced_markers(page, note_cut);
    }
}

fn orphaned_marker(line: &Line) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    !line.spans.is_empty()
        && RE
            .get_or_init(|| Regex::new(r"^\d{1,4}$").unwrap())
            .is_match(&line.text)
}

fn spliced_marker_host(marker: &Line, host: Option<&Line>) -> bool {
    let Some(host) = host else {
        return false;
    };
    if orphaned_marker(host) || marker.block_index != host.block_index {
        return false;
    }
    let marker_height = marker.bbox[3] - marker.bbox[1];
    let host_height = host.bbox[3] - host.bbox[1];
    if marker_height <= 0.0 || host_height <= 0.0 {
        return false;
    }
    let overlap = marker.bbox[3].min(host.bbox[3]) - marker.bbox[1].max(host.bbox[1]);
    if overlap.max(0.0) / marker_height < 0.5
        || marker.bbox[0] < host.bbox[0] - 12.0
        || marker.bbox[0] > host.bbox[2] + 12.0
    {
        return false;
    }
    if marker.spans.iter().all(|span| span.superscript) {
        return true;
    }
    let marker_size = line_font_size(marker);
    let host_size = line_font_size(host);
    marker_size > 0.0
        && host_size >= 1.25 * marker_size
        && marker.bbox[3] <= host.bbox[3] - 0.25 * host_height
}

fn associate_spliced_markers(page: &mut Page, note_cut: f64) {
    let eligible: Vec<usize> = page
        .lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| {
            (!matches!(line.region_type.as_str(), "header" | "footer") && !line.exclude_from_body)
                .then_some(index)
        })
        .collect();
    if eligible.len() < 2 {
        return;
    }
    let mut claims = Vec::<(usize, usize)>::new();
    for (position, marker_index) in eligible.iter().copied().enumerate() {
        let marker = &page.lines[marker_index];
        if !orphaned_marker(marker) || marker.bbox[1] >= note_cut {
            continue;
        }
        let previous = position
            .checked_sub(1)
            .and_then(|index| eligible.get(index))
            .copied();
        let next = eligible.get(position + 1).copied();
        let previous_ok = spliced_marker_host(marker, previous.map(|index| &page.lines[index]));
        let next_ok = spliced_marker_host(marker, next.map(|index| &page.lines[index]));
        if previous_ok == next_ok {
            continue;
        }
        claims.push((
            marker_index,
            if previous_ok {
                previous.unwrap()
            } else {
                next.unwrap()
            },
        ));
    }
    let mut host_counts = HashMap::<usize, usize>::new();
    for (_, host) in &claims {
        *host_counts.entry(*host).or_default() += 1;
    }
    for (marker_index, host_index) in claims {
        if host_counts.get(&host_index) != Some(&1) {
            continue;
        }
        let value = normalize_label(page.lines[marker_index].text.trim());
        let selected_text = page.lines[marker_index].text.trim().to_owned();
        let source_line_id = page.lines[marker_index].id.clone();
        let offset = page.lines[host_index].text.chars().count();
        page.lines[host_index]
            .detached_references
            .push(DetachedReference {
                note_id: value,
                selected_text,
                start_offset: offset,
                end_offset: offset,
                source_line_id,
            });
        page.lines[marker_index].exclude_from_body = true;
    }
}

fn all_caps(text: &str) -> bool {
    let mut letters = text.chars().filter(|character| character.is_alphabetic());
    letters.next().is_some_and(|first| {
        first.is_uppercase() && letters.all(|character| character.is_uppercase())
    })
}

fn has_word_character(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\w").unwrap()).is_match(text)
}

/// Text-Fidelity's native-PDF body-font estimator: the largest rounded font
/// bucket carrying at least ten percent of document character mass. Smaller
/// quote/note fonts can be common without redefining the body face, while
/// headings normally lack enough mass to become a substantial bucket.
fn article_body_font_size(pages: &[Page]) -> f64 {
    let mut mass = HashMap::<i64, usize>::new();
    for line in pages.iter().flat_map(|page| &page.lines).filter(|line| {
        !line.exclude_from_body && matches!(line.region_type.as_str(), "text" | "body")
    }) {
        let size = line_font_size(line);
        if size > 0.0 {
            *mass.entry((size * 100.0).round() as i64).or_default() += line.text.chars().count();
        }
    }
    let total: usize = mass.values().sum();
    mass.into_iter()
        .filter(|(_, chars)| total > 0 && chars.saturating_mul(10) >= total)
        .map(|(size, _)| size as f64 / 100.0)
        .max_by(f64::total_cmp)
        .unwrap_or(0.0)
}

/// Region-dependent Text-Fidelity lanes are fail-closed. A complete set of
/// non-unknown line labels may come from PPDoc or any MLLM; consumers depend
/// on the region contract, not the provider identity. The snapshot survives
/// the engine's later ordering and normalized-label passes.
fn source_region_contract(pages: &[Page]) -> Option<HashMap<String, String>> {
    let mut regions = HashMap::new();
    for line in pages
        .iter()
        .flat_map(|page| &page.lines)
        .filter(|line| !line.exclude_from_body && !line.text.trim().is_empty())
    {
        let region = line.region_type.trim().to_ascii_lowercase();
        if line.id.is_empty()
            || matches!(region.as_str(), "" | "unknown" | "unknown_region")
            || regions.insert(line.id.clone(), region).is_some()
        {
            return None;
        }
    }
    (!regions.is_empty()).then_some(regions)
}

#[cfg(test)]
fn source_regions_available(pages: &[Page]) -> bool {
    source_region_contract(pages).is_some()
}

fn heading_source_eligible(regions: &HashMap<String, String>, line: &Line) -> bool {
    regions.get(&line.id).is_some_and(|region| {
        matches!(
            region.as_str(),
            "text" | "body" | "paragraph_title" | "heading"
        )
    })
}

struct HeadingCandidate<'a> {
    page_slot: usize,
    line_slot: usize,
    joined_line_slot: Option<usize>,
    text: &'a str,
    interpretations: Vec<EnumeratorInterpretation>,
}

#[derive(Clone, Copy)]
struct HeadingDecision {
    page_slot: usize,
    line_slot: usize,
    joined_line_slot: Option<usize>,
    text_plausible: bool,
    level: Option<usize>,
    action: HeadingAction,
    coherent_family: bool,
    footnote_suspect: bool,
}

fn heading_candidates<'a>(
    pages: &'a [Page],
    primitives: &PdfPrimitiveEvidence,
) -> Vec<HeadingCandidate<'a>> {
    let regions = primitives.source_regions.as_ref().expect("source regions");
    let inline = inline_enumerator_re();
    let standalone = standalone_enumerator_re();
    let mut candidates = Vec::new();
    for (page_slot, page) in pages.iter().enumerate() {
        if primitives.contents_pages.contains(&page.index) {
            continue;
        }
        for (line_slot, line) in page.lines.iter().enumerate().filter(|(_, line)| {
            !line.exclude_from_body
                && matches!(line.region_type.as_str(), "body" | "heading")
                && heading_source_eligible(regions, line)
        }) {
            if let Some(capture) = inline.captures(line.text.trim()) {
                let value = capture.get(1).unwrap().as_str();
                let punct = capture.get(2).unwrap().as_str();
                let text = capture.get(3).unwrap().as_str().trim();
                let interpretations = enumerator_interpretations(value, punct);
                if heading_text_plausible(text) && !interpretations.is_empty() {
                    candidates.push(HeadingCandidate {
                        page_slot,
                        line_slot,
                        joined_line_slot: None,
                        text,
                        interpretations,
                    });
                }
                continue;
            }
            let Some(capture) = standalone.captures(line.text.trim()) else {
                continue;
            };
            let follower = ((line_slot + 1)..(line_slot + 3).min(page.lines.len())).find(|index| {
                let line = &page.lines[*index];
                !line.text.trim().is_empty()
                    && !line.exclude_from_body
                    && matches!(line.region_type.as_str(), "body" | "heading")
                    && heading_source_eligible(regions, line)
            });
            let Some(follower_slot) = follower else {
                continue;
            };
            let text = page.lines[follower_slot].text.trim();
            let value = capture.get(1).unwrap().as_str();
            let punct = capture.get(2).unwrap().as_str();
            let interpretations = enumerator_interpretations(value, punct);
            if heading_text_plausible(text) && !interpretations.is_empty() {
                candidates.push(HeadingCandidate {
                    page_slot,
                    line_slot,
                    joined_line_slot: Some(follower_slot),
                    text,
                    interpretations,
                });
            }
        }
    }
    candidates
}

fn bold_char_share(line: &Line) -> f64 {
    let mut total = 0_usize;
    let mut bold = 0_usize;
    for span in &line.spans {
        let chars = span.text.chars().count();
        total += chars;
        if span.flags & 16 != 0
            || span
                .font
                .as_bytes()
                .windows(4)
                .any(|value| value.eq_ignore_ascii_case(b"bold"))
        {
            bold += chars;
        }
    }
    if total == 0 {
        0.0
    } else {
        bold as f64 / total as f64
    }
}

fn sentence_ended(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r#"[.!?][\"')\]]*$"#).unwrap())
        .is_match(text.trim_end())
}

fn bracket_excess(text: &str) -> usize {
    let mut stack = Vec::new();
    for character in text.chars() {
        match character {
            '(' => stack.push(')'),
            '[' => stack.push(']'),
            '{' => stack.push('}'),
            '\u{201c}' => stack.push('\u{201d}'),
            ')' | ']' | '}' | '\u{201d}' if stack.last() == Some(&character) => {
                stack.pop();
            }
            _ => {}
        }
    }
    stack.len()
}

fn closes_before_opening(text: &str) -> bool {
    text.chars()
        .take(48)
        .find_map(|character| match character {
            ')' | ']' | '}' | '\u{201d}' => Some(true),
            '(' | '[' | '{' | '\u{201c}' => Some(false),
            _ => None,
        })
        .unwrap_or(false)
}

fn starts_note_or_list(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"^\s*(?:(?:[-*â€¢])|(?:\(?\d{1,4}[.)])|(?:\(?[A-Za-z][.)])|(?:[IVXLCDMivxlcdm]{1,8}[.)])|(?:[*â€ â€¡Â§]))\s+\S",
        )
        .unwrap()
    })
    .is_match(text)
}

fn body_flow_edge(previous: &Line, current: &Line) -> bool {
    let previous_text = previous.text.trim();
    let current_text = current.text.trim();
    if previous_text.is_empty() || current_text.is_empty() {
        return false;
    }
    if hyphen_fragment_tail(previous_text) && hyphen_continuation(current_text) {
        return true;
    }
    if starts_note_or_list(current_text) {
        return false;
    }
    if !sentence_ended(previous_text) && current_text.chars().next().is_some_and(char::is_lowercase)
    {
        return true;
    }
    bracket_excess(previous_text) > 0 && closes_before_opening(current_text)
}

fn has_body_flow(
    page: &Page,
    line_slot: usize,
    structural: &HashSet<(usize, usize)>,
    page_slot: usize,
) -> bool {
    [
        ("incoming", line_slot.checked_sub(1)),
        ("outgoing", line_slot.checked_add(1)),
    ]
    .into_iter()
    .any(|(direction, neighbor_slot)| {
        let Some(neighbor_slot) = neighbor_slot.filter(|index| *index < page.lines.len()) else {
            return false;
        };
        if structural.contains(&(page_slot, neighbor_slot))
            || page.lines[neighbor_slot].region_type != "body"
        {
            return false;
        }
        if direction == "incoming" {
            body_flow_edge(&page.lines[neighbor_slot], &page.lines[line_slot])
        } else {
            body_flow_edge(&page.lines[line_slot], &page.lines[neighbor_slot])
        }
    })
}

fn coherent_heading_family(family: &HeadingFamilyStats) -> bool {
    family.count >= 2 && family.violations == 0 && family.level_votes.len() == 1
}

fn wrapped_heading_continuation(
    page: &Page,
    heading_slot: usize,
    structural: &HashSet<(usize, usize)>,
    page_slot: usize,
    source_regions: &HashMap<String, String>,
) -> Option<usize> {
    static INLINE: OnceLock<Regex> = OnceLock::new();
    if heading_slot + 2 >= page.lines.len() {
        return None;
    }
    let heading = &page.lines[heading_slot];
    let continuation = &page.lines[heading_slot + 1];
    let following = &page.lines[heading_slot + 2];
    let heading_text = heading.text.trim();
    let continuation_text = continuation.text.trim();
    let caps_head = heading_text.split_whitespace().count() >= 2
        && heading_text
            .chars()
            .filter(|character| character.is_alphabetic())
            .count()
            >= 8
        && !heading_text.chars().any(char::is_lowercase)
        && !heading_text.ends_with(['.', '?', '!', ';', ':']);
    let wrap_capable = INLINE
        .get_or_init(|| {
            Regex::new(
                r"^\s*(?:[IVXLCDM]{1,7}|[A-Za-z]|\d{1,3}|\d{1,2}(?:\.\d{1,2}){1,3})[.)]\s+\S",
            )
            .unwrap()
        })
        .is_match(heading_text)
        || (caps_head && !continuation_text.chars().any(char::is_lowercase));
    let continuation_source = source_regions.get(&continuation.id).map(String::as_str);
    let following_source = source_regions.get(&following.id).map(String::as_str);
    if !wrap_capable
        || structural.contains(&(page_slot, heading_slot + 1))
        || !matches!(
            continuation_source,
            Some("text" | "body" | "paragraph_title" | "heading" | "block_quote")
        )
        || !matches!(following_source, Some("text" | "body"))
        || !matches!(continuation.region_type.as_str(), "body" | "heading")
        || following.region_type != "body"
        || continuation_text.is_empty()
        || !(1..=12).contains(&continuation_text.split_whitespace().count())
        || sentence_ended(continuation_text)
        || has_citation_signal(continuation_text)
        || starts_note_or_list(continuation_text)
        || !has_valid_bbox(heading)
        || !has_valid_bbox(continuation)
        || !has_valid_bbox(following)
    {
        return None;
    }
    let heading_size = line_font_size(heading);
    let continuation_size = line_font_size(continuation);
    let heading_height = (heading.bbox[3] - heading.bbox[1]).max(1.0);
    let continuation_height = (continuation.bbox[3] - continuation.bbox[1]).max(1.0);
    let x0_delta = continuation.bbox[0] - heading.bbox[0];
    let internal_gap = continuation.bbox[1] - heading.bbox[3];
    let internal_step = continuation.bbox[1] - heading.bbox[1];
    let following_step = following.bbox[1] - continuation.bbox[1];
    (heading_size > 0.0
        && continuation_size > 0.0
        && (continuation_size - heading_size).abs() <= (heading_size * 0.02).max(0.1)
        && (bold_char_share(continuation) - bold_char_share(heading)).abs() <= 0.1
        && (continuation_height - heading_height).abs() <= heading_height * 0.05
        && (-3.0..=48.0_f64.max(heading_height * 1.75)).contains(&x0_delta)
        && (-heading_height * 0.2..=(heading_height * 0.8).max(6.0)).contains(&internal_gap)
        && following_step >= internal_step + (continuation_height * 0.5).max(8.0))
    .then_some(heading_slot + 1)
}

fn titlecase_ratio(text: &str) -> f64 {
    let mut words = 0;
    let titlecase = text
        .split_whitespace()
        .filter(|word| word.chars().any(char::is_alphabetic))
        .inspect(|_| words += 1)
        .filter(|word| word.chars().next().is_some_and(char::is_uppercase))
        .count();
    if words == 0 {
        0.0
    } else {
        titlecase as f64 / words as f64
    }
}

fn heading_style_corroborated(text: &str) -> bool {
    let text = text.trim();
    if text.is_empty() || text.chars().count() > 120 || has_citation_signal(text) {
        return false;
    }
    if text
        .chars()
        .find(|character| character.is_alphabetic())
        .is_some_and(char::is_lowercase)
    {
        return false;
    }
    if text
        .chars()
        .filter(|character| character.is_alphabetic())
        .count()
        >= 4
        && all_caps(text)
    {
        return true;
    }
    if let Some(capture) = inline_enumerator_re().captures(text) {
        if heading_text_plausible(capture.get(3).unwrap().as_str()) && !text.ends_with('.') {
            return true;
        }
    }
    let words = text
        .split_whitespace()
        .filter(|word| word.chars().any(char::is_alphabetic))
        .count();
    (1..=9).contains(&words) && titlecase_ratio(text) >= 0.60 && !sentence_ended(text)
}

fn caps_warble(text: &str) -> bool {
    let (mut letters, mut uppercase, mut lowercase) = (0_usize, 0_usize, false);
    for character in text.chars().filter(|character| character.is_alphabetic()) {
        letters += 1;
        uppercase += usize::from(character.is_uppercase());
        lowercase |= character.is_lowercase();
    }
    uppercase > 0 && lowercase && uppercase as f64 / letters as f64 >= 0.65
}

fn bilateral_body_geometry(page: &Page, line_slot: usize) -> bool {
    let line = &page.lines[line_slot];
    let height = (line.bbox[3] - line.bbox[1]).max(1.0);
    [line_slot.checked_sub(1), line_slot.checked_add(1)]
        .into_iter()
        .all(|neighbor_slot| {
            let Some(neighbor_slot) = neighbor_slot.filter(|index| *index < page.lines.len())
            else {
                return false;
            };
            let neighbor = &page.lines[neighbor_slot];
            if neighbor.region_type != "body" {
                return false;
            }
            let neighbor_height = (neighbor.bbox[3] - neighbor.bbox[1]).max(1.0);
            let gap = if neighbor_slot < line_slot {
                line.bbox[1] - neighbor.bbox[3]
            } else {
                neighbor.bbox[1] - line.bbox[3]
            };
            (line.bbox[0] - neighbor.bbox[0]).abs() <= 10.0
                && -height * 0.5 <= gap
                && gap <= (height * 1.4).max(14.0)
                && (0.75..=1.35).contains(&(height / neighbor_height))
        })
}

fn demote_false_headings(
    pages: &mut [Page],
    decisions: &[HeadingDecision],
    source_regions: &HashMap<String, String>,
) {
    let mut by_line = HashMap::<(usize, usize), &HeadingDecision>::new();
    for decision in decisions {
        by_line.insert((decision.page_slot, decision.line_slot), decision);
        if let Some(line) = decision.joined_line_slot {
            by_line.insert((decision.page_slot, line), decision);
        }
    }
    for (page_slot, page) in pages.iter_mut().enumerate() {
        for line_slot in 0..page.lines.len() {
            if page.lines[line_slot].region_type != "heading" {
                continue;
            }
            if !source_regions
                .get(&page.lines[line_slot].id)
                .is_some_and(|region| matches!(region.as_str(), "paragraph_title" | "heading"))
            {
                continue;
            }
            let text = page.lines[line_slot].text.trim();
            if caps_warble(text) {
                continue;
            }
            let decision = by_line.get(&(page_slot, line_slot));
            if decision.is_some_and(|decision| decision.coherent_family) {
                continue;
            }
            let style = heading_style_corroborated(text);
            let grammar_negative = decision.map_or(!style, |decision| {
                !style
                    || matches!(
                        decision.action,
                        HeadingAction::IllegalRestart | HeadingAction::Violation
                    )
            });
            if !grammar_negative {
                continue;
            }
            let words = text
                .split_whitespace()
                .filter(|word| word.chars().any(char::is_alphabetic))
                .count();
            let starts_lowercase = text
                .chars()
                .find(|character| character.is_alphabetic())
                .is_some_and(char::is_lowercase);
            let terminal = words >= 4 && sentence_ended(text);
            let long_prose = words >= 12 || text.chars().count() > 110;
            let prose_case = words >= 4 && titlecase_ratio(text) < 0.45;
            let citation = has_citation_signal(text);
            let strong_prose = terminal || long_prose || prose_case || citation;
            if !strong_prose {
                continue;
            }
            let flow = [line_slot.checked_sub(1), line_slot.checked_add(1)]
                .into_iter()
                .any(|neighbor_slot| {
                    let Some(neighbor_slot) =
                        neighbor_slot.filter(|index| *index < page.lines.len())
                    else {
                        return false;
                    };
                    if page.lines[neighbor_slot].region_type != "body" {
                        return false;
                    }
                    if neighbor_slot < line_slot {
                        body_flow_edge(&page.lines[neighbor_slot], &page.lines[line_slot])
                    } else {
                        body_flow_edge(&page.lines[line_slot], &page.lines[neighbor_slot])
                    }
                });
            let soft_hyphen_flow = [line_slot.checked_sub(1), line_slot.checked_add(1)]
                .into_iter()
                .any(|neighbor_slot| {
                    let Some(neighbor_slot) =
                        neighbor_slot.filter(|index| *index < page.lines.len())
                    else {
                        return false;
                    };
                    if page.lines[neighbor_slot].region_type != "body" {
                        return false;
                    }
                    let (previous, current) = if neighbor_slot < line_slot {
                        (&page.lines[neighbor_slot], &page.lines[line_slot])
                    } else {
                        (&page.lines[line_slot], &page.lines[neighbor_slot])
                    };
                    hyphen_fragment_tail(&previous.text) && hyphen_continuation(&current.text)
                });
            let independent_shape = starts_lowercase || long_prose || prose_case || citation;
            let broader_continuity = bilateral_body_geometry(page, line_slot)
                || soft_hyphen_flow
                || (terminal && independent_shape);
            if (flow && starts_lowercase) || broader_continuity {
                page.lines[line_slot].region_type = "body".to_owned();
            }
        }
    }
}

fn apply_text_fidelity_headings(
    pages: &mut [Page],
    body_size: f64,
    primitives: &PdfPrimitiveEvidence,
) -> HashMap<String, usize> {
    let Some(source_regions) = primitives.source_regions.as_ref() else {
        return HashMap::new();
    };
    let toc_leader = contents_leader_re();
    for page in pages.iter_mut() {
        if primitives.contents_pages.contains(&page.index)
            || page
                .lines
                .iter()
                .filter(|line| toc_leader.is_match(&line.text))
                .take(5)
                .count()
                >= 5
        {
            continue;
        }
        for line in &mut page.lines {
            let text = line.text.trim();
            let letters = text
                .chars()
                .filter(|character| character.is_alphabetic())
                .count();
            let source_mutable = source_regions
                .get(&line.id)
                .is_some_and(|region| matches!(region.as_str(), "text" | "body"));
            if line.region_type == "body"
                && source_mutable
                && (8..=70).contains(&text.chars().count())
                && letters >= 4
                && !text.chars().any(char::is_lowercase)
                && !text.ends_with(['.', '?', '!', ';', ':', ','])
                && heading_text_plausible(text)
            {
                line.region_type = "heading".to_owned();
            }
        }
    }

    let (ladder_clean, decisions) = {
        let candidates = heading_candidates(pages, primitives);
        let parsed = parse_heading_ladder(
            candidates
                .iter()
                .map(|candidate| candidate.interpretations.as_slice()),
        );
        let decisions = candidates
            .iter()
            .zip(&parsed.assignments)
            .map(|(candidate, assignment)| {
                let family = parsed.families.get(assignment.family);
                HeadingDecision {
                    page_slot: candidate.page_slot,
                    line_slot: candidate.line_slot,
                    joined_line_slot: candidate.joined_line_slot,
                    text_plausible: heading_text_plausible(candidate.text),
                    level: assignment.level,
                    action: assignment.action,
                    coherent_family: family.is_some_and(coherent_heading_family),
                    footnote_suspect: family.is_some_and(|family| family.footnote_suspect),
                }
            })
            .collect::<Vec<_>>();
        (parsed.status == HeadingLadderStatus::ParsedClean, decisions)
    };
    demote_false_headings(pages, &decisions, source_regions);
    let mut heading_levels = decisions
        .iter()
        .flat_map(|decision| {
            [Some(decision.line_slot), decision.joined_line_slot]
                .into_iter()
                .flatten()
                .filter_map(move |line| {
                    decision
                        .level
                        .map(|level| (decision.page_slot, line, level))
                })
        })
        .filter_map(|(page, line, level)| {
            pages
                .get(page)
                .and_then(|page| page.lines.get(line))
                .map(|line| (line.id.clone(), level))
        })
        .collect::<HashMap<_, _>>();
    let structural: HashSet<(usize, usize)> = decisions
        .iter()
        .flat_map(|decision| {
            [
                Some((decision.page_slot, decision.line_slot)),
                decision
                    .joined_line_slot
                    .map(|line| (decision.page_slot, line)),
            ]
            .into_iter()
            .flatten()
        })
        .collect();

    for decision in &decisions {
        let page_slot = decision.page_slot;
        let marker_slot = decision.line_slot;
        let target_slot = decision.joined_line_slot.unwrap_or(marker_slot);
        if page_slot >= pages.len() || target_slot >= pages[page_slot].lines.len() {
            continue;
        }
        if !source_regions
            .get(&pages[page_slot].lines[target_slot].id)
            .is_some_and(|region| matches!(region.as_str(), "text" | "body"))
        {
            continue;
        }
        let target_is_heading = pages[page_slot].lines[target_slot].region_type == "heading";
        if target_is_heading {
            continue;
        }
        if !ladder_clean
            || !decision.text_plausible
            || matches!(
                decision.action,
                HeadingAction::IllegalRestart | HeadingAction::Violation
            )
        {
            continue;
        }
        if decision.footnote_suspect || decision.level.unwrap_or(0) == 0 {
            continue;
        }
        let page = &pages[page_slot];
        if has_body_flow(page, target_slot, &structural, page_slot) {
            continue;
        }
        let target = &page.lines[target_slot];
        let visual = bold_char_share(target) >= 0.60
            || (body_size > 0.0 && line_font_size(target) >= body_size * 1.02);
        if !visual && !decision.coherent_family {
            continue;
        }
        pages[page_slot].lines[target_slot].region_type = "heading".to_owned();
        if target_slot != marker_slot && marker_slot < pages[page_slot].lines.len() {
            let block = pages[page_slot].lines[target_slot].block_index;
            pages[page_slot].lines[marker_slot].region_type = "heading".to_owned();
            pages[page_slot].lines[marker_slot].block_index = block;
        }
        if let Some(continuation_slot) = wrapped_heading_continuation(
            &pages[page_slot],
            target_slot,
            &structural,
            page_slot,
            source_regions,
        ) {
            let block = pages[page_slot].lines[target_slot].block_index;
            pages[page_slot].lines[continuation_slot].region_type = "heading".to_owned();
            pages[page_slot].lines[continuation_slot].block_index = block;
            if let Some(level) = decision.level {
                heading_levels.insert(pages[page_slot].lines[continuation_slot].id.clone(), level);
            }
        }
    }
    let accepted = pages
        .iter()
        .flat_map(|page| &page.lines)
        .filter(|line| line.region_type == "heading")
        .map(|line| line.id.as_str())
        .collect::<HashSet<_>>();
    heading_levels.retain(|line_id, _| accepted.contains(line_id.as_str()));
    heading_levels
}

fn endnote_heading(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)^(?:end)?notes?$").unwrap())
        .is_match(text.trim())
}

fn continuing_note_heading(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)^(?:endnotes?|footnotes?|notes?)(?:\s*\(?(?:continued|cont'd)\)?)?$")
            .unwrap()
    })
    .is_match(text.trim())
}

fn structural_reset_heading(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)^(?:appendix|annex|schedule|part|chapter|bibliography|references|works\s+cited|table\s+of\s+authorities|index|acknowledg(?:e)?ments|certificate\s+of\s+service|about\s+the\s+authors?)(?:\s+[\w.-]+)?$",
        )
        .unwrap()
    })
    .is_match(text.trim())
}

fn citation_shaped_candidate(page: &Page, candidate: &(usize, LabelPrefix, bool)) -> bool {
    let line = &page.lines[candidate.0];
    let tail = scalar_suffix(&line.text, candidate.1.end);
    citation_shaped_tail(tail)
}

fn longest_label_run(values: &[u32]) -> usize {
    let mut runs = vec![1; values.len()];
    for index in 0..values.len() {
        for prior in 0..index {
            if (1..=3).contains(&values[index].saturating_sub(values[prior])) {
                runs[index] = runs[index].max(runs[prior] + 1);
            }
        }
    }
    runs.into_iter().max().unwrap_or(0)
}

fn has_prior_reference(page: &Page, label: &str, label_y: f64) -> bool {
    let matches_label = |value: &str| {
        let value = value.trim();
        value == label || normalize_label(value) == label
    };
    page.lines.iter().any(|line| {
        line.bbox[1] < label_y
            && (line
                .spans
                .iter()
                .any(|span| span.superscript && matches_label(&span.text))
                || line
                    .detached_references
                    .iter()
                    .any(|reference| matches_label(&reference.note_id)))
    })
}

fn classify_pages_with_source(
    pages: &mut [Page],
    separators: &[Option<f64>],
    evidence: &mut PdfPrimitiveEvidence,
) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    let article_body_size = if evidence.source_regions.is_some() {
        article_body_font_size(pages)
    } else {
        0.0
    };
    let mut expected_endnote: Option<u32> = None;
    let mut continuing_size: Option<f64> = None;
    let mut continuing_table = false;
    let table_pages: Vec<_> = pages
        .iter()
        .zip(separators.iter().copied())
        .map(|(page, separator)| {
            let continuation = continuing_table;
            let table = table_evidence(&page.lines, page.width);
            evidence
                .contents_pages
                .extend(table.contents.then_some(page.index));
            let caption = has_table_caption(&page.lines);
            let is_table = caption
                || strong_table_evidence(&table, &page.lines)
                || continuation && table.continuation_on_page(page.height);
            continuing_table = is_table
                && table.reaches_page_bottom(page.height)
                && (caption || table.continuation());
            let mut cells = if is_table {
                table.expanded_lines(&page.lines, page.height, continuation, separator)
            } else {
                HashSet::new()
            };
            let notes = table.table_note_lines(&page.lines, &cells);
            cells.retain(|index| !notes.contains(index));
            (is_table, cells, notes)
        })
        .collect();

    for (page, (_, table_cells, table_notes)) in pages.iter().zip(&table_pages) {
        evidence.table_cell_line_ids.extend(
            table_cells
                .iter()
                .filter_map(|index| page.lines.get(*index))
                .map(|line| line.id.clone()),
        );
        evidence.table_note_line_ids.extend(
            table_notes
                .iter()
                .filter_map(|index| page.lines.get(*index))
                .map(|line| line.id.clone()),
        );
    }

    for (page, (table_page, table_cells, table_notes)) in pages.iter_mut().zip(table_pages) {
        let line_sizes: Vec<f64> = page.lines.iter().map(line_font_size).collect();
        let body_size = median(
            page.lines
                .iter()
                .enumerate()
                .filter(|(index, line)| {
                    !matches!(line.region_type.as_str(), "header" | "footer")
                        && !line.exclude_from_body
                        && !table_cells.contains(index)
                        && !table_notes.contains(index)
                        && page.height * 0.10 <= line.bbox[1]
                        && line.bbox[1] <= page.height * 0.75
                })
                .map(|(index, _)| line_sizes[index])
                .filter(|size| (4.0..=24.0).contains(size))
                .collect(),
        );
        let body_size = if body_size > 0.0 { body_size } else { 10.0 };
        let wide_sizes: Vec<_> = page
            .lines
            .iter()
            .enumerate()
            .filter(|(index, line)| {
                !line.exclude_from_body
                    && !table_cells.contains(index)
                    && !table_notes.contains(index)
                    && line_width(line) >= page.width * 0.30
                    && line
                        .text
                        .chars()
                        .filter(|character| character.is_alphabetic())
                        .take(4)
                        .count()
                        == 4
            })
            .map(|(index, _)| line_sizes[index])
            .filter(|size| (4.0..=24.0).contains(size))
            .collect();
        let note_size = if wide_sizes.len() >= 3 {
            median(wide_sizes)
        } else {
            body_size
        };
        let tolerance = (page.height * 0.004).max(1.0);
        let table_band = table_cells.iter().fold(None, |band, index| {
            let line = &page.lines[*index];
            Some(
                band.map_or((line.bbox[1], line.bbox[3]), |(top, bottom): (f64, f64)| {
                    (top.min(line.bbox[1]), bottom.max(line.bbox[3]))
                }),
            )
        });
        let separator = separators.get(page.index).copied().flatten().filter(|cut| {
            table_band
                .is_none_or(|(top, bottom)| *cut < top - tolerance || *cut > bottom + tolerance)
        });
        let mut candidates = Vec::new();
        let mut suppress = Vec::new();
        for (index, line) in page.lines.iter().enumerate() {
            if table_cells.contains(&index) {
                suppress.push(index);
                continue;
            }
            if (matches!(line.region_type.as_str(), "header" | "footer")
                && !table_notes.contains(&index))
                || line.exclude_from_body
            {
                continue;
            }
            let Some(prefix) = label_prefix(&line.text) else {
                continue;
            };
            if table_notes.contains(&index) {
                candidates.push((index, prefix, true));
                continue;
            }
            let typographic = label_is_typographic(line, &prefix, line_sizes[index], body_size)
                || (standalone_note_label(line)
                    && aligned_note_body_index(&page.lines, index).is_some_and(|body| {
                        let size = line_sizes[body];
                        size > 0.0 && size <= body_size * 0.90
                    }));
            let size = {
                let size = line_sizes[index];
                if size > 0.0 {
                    size
                } else {
                    body_size
                }
            };
            let bottom_right = separator.is_none()
                && line.bbox[1] >= page.height * 0.91
                && line.bbox[0] >= page.width * 0.50
                && !compact_note_line(&line.text);
            let comma_tail = line.text.chars().nth(prefix.end) == Some(',') && !typographic;
            let below_separator = separator.is_some_and(|cut| line.bbox[1] >= cut - tolerance);
            let suppressed = (line.bbox[1] >= page.height * 0.94
                && !(below_separator && typographic))
                || bottom_right
                || comma_tail
                || size > body_size * 1.15;
            if suppressed {
                suppress.push(index);
                continue;
            }
            candidates.push((index, prefix, typographic));
        }
        for index in suppress {
            page.lines[index].suppress_footnote_label = true;
        }

        let numeric: Vec<u32> = candidates
            .iter()
            .filter_map(|(_, prefix, _)| prefix.label.parse().ok())
            .collect();
        let best_run = longest_label_run(&numeric);
        let page_columns = {
            let page_model = column_model(&page.lines, page.width);
            if page_model.kind == "two_column" {
                page_model
            } else {
                let label_lines: Vec<_> = candidates
                    .iter()
                    .map(|(index, _, _)| page.lines[*index].clone())
                    .collect();
                let label_model = column_model(&label_lines, page.width);
                if label_model.kind == "single" {
                    page_model
                } else {
                    label_model
                }
            }
        };
        let first_candidate_y = candidates
            .first()
            .map(|(index, _, _)| page.lines[*index].bbox[1]);
        let minimum_label_y = candidates
            .iter()
            .map(|(index, _, _)| page.lines[*index].bbox[1])
            .min_by(f64::total_cmp);
        let endnote_heading_y = page
            .lines
            .iter()
            .filter(|line| {
                endnote_heading(&line.text) && first_candidate_y.is_none_or(|y| line.bbox[1] < y)
            })
            .map(|line| line.bbox[1])
            .min_by(f64::total_cmp);
        let has_endnote_heading = endnote_heading_y.is_some();
        let endnote_heading_source = has_endnote_heading.then(|| {
            page.lines
                .iter()
                .filter(|line| endnote_heading(&line.text))
                .map(|line| line.source_index)
                .min()
                .unwrap_or(0)
        });
        let minimum_label_y = minimum_label_y.unwrap_or(page.height);
        let content_before = || {
            page.lines.iter().enumerate().filter_map(|(index, line)| {
                (!matches!(line.region_type.as_str(), "header" | "footer")
                    && line.bbox[1] < minimum_label_y)
                    .then_some(index)
            })
        };
        let first_content = content_before()
            .min_by(|left, right| page.lines[*left].bbox[1].total_cmp(&page.lines[*right].bbox[1]));
        let generic_heading_reset = candidates.is_empty()
            && first_content.is_some_and(|index| {
                let line = &page.lines[index];
                let text = line.text.trim();
                let letter_count = text
                    .chars()
                    .filter(|character| character.is_alphabetic())
                    .count();
                line.bbox[1] >= page.height * 0.08
                    && letter_count >= 4
                    && text.chars().count() <= 100
                    && all_caps(text)
                    && !continuing_note_heading(text)
            });
        let structural_reset = generic_heading_reset
            || content_before().any(|index| {
                let line = &page.lines[index];
                (line.region_type == "heading" || line.bbox[1] >= page.height * 0.08)
                    && structural_reset_heading(&line.text)
            });
        let label_sizes: Vec<f64> = candidates
            .iter()
            .map(|(index, _, _)| line_sizes[*index])
            .filter(|size| *size > 0.0)
            .collect();
        let label_size = (!label_sizes.is_empty()).then(|| median(label_sizes));
        let early_labels = candidates.iter().any(|(index, _, typographic)| {
            page.lines[*index].bbox[1] < page.height * 0.48
                && (*typographic || line_sizes[*index] <= body_size * 0.90)
        });
        let lower_numeric: Vec<u32> = candidates
            .iter()
            .filter(|(index, _, _)| {
                page.lines[*index].bbox[1] >= page.height * 0.48
                    && line_sizes[*index] > 0.0
                    && line_sizes[*index] <= body_size * 0.90
            })
            .filter_map(|(_, prefix, _)| prefix.label.parse().ok())
            .collect();
        let lower_note_run = longest_label_run(&lower_numeric) >= 3;
        let expected_indexes: Vec<usize> = candidates
            .iter()
            .enumerate()
            .filter_map(|(index, (_, prefix, _))| {
                (prefix.label.parse::<u32>().ok() == expected_endnote).then_some(index)
            })
            .collect();
        let noncitation: Vec<usize> = expected_indexes
            .iter()
            .copied()
            .filter(|index| !citation_shaped_candidate(page, &candidates[*index]))
            .collect();
        let pool = if noncitation.is_empty() {
            &expected_indexes
        } else {
            &noncitation
        };
        let expected_index = pool.iter().copied().max_by_key(|index| {
            let mut sequence = 1;
            let mut prior = expected_endnote;
            for (_, prefix, _) in &candidates[index + 1..] {
                let Ok(current) = prefix.label.parse::<u32>() else {
                    break;
                };
                if prior.is_none_or(|value| current != value + 1) {
                    break;
                }
                sequence += 1;
                prior = Some(current);
            }
            (usize::from(candidates[*index].2), sequence, *index)
        });
        if expected_endnote.is_some() {
            if let Some(selected) = expected_index {
                for candidate in &candidates[..selected] {
                    page.lines[candidate.0].suppress_footnote_label = true;
                }
                for index in &expected_indexes {
                    if *index != selected && citation_shaped_candidate(page, &candidates[*index]) {
                        page.lines[candidates[*index].0].suppress_footnote_label = true;
                    }
                }
            }
        }
        let selected_start = if expected_endnote.is_some() {
            expected_index.unwrap_or(0)
        } else {
            0
        };
        let content_sizes: Vec<f64> = page
            .lines
            .iter()
            .enumerate()
            .filter(|(_, line)| {
                !matches!(line.region_type.as_str(), "header" | "footer") && !line.exclude_from_body
            })
            .map(|(index, _)| line_sizes[index])
            .filter(|size| *size > 0.0)
            .collect();
        let content_size = (!content_sizes.is_empty()).then(|| median(content_sizes));
        let continuation_size_matches = continuing_size
            .is_none_or(|size| content_size.is_none_or(|content| content <= size * 1.15));
        let label_free_continuation = expected_endnote.is_some()
            && candidates.is_empty()
            && !structural_reset
            && continuation_size_matches;
        let continuing_endnotes = expected_endnote.is_some()
            && !structural_reset
            && continuation_size_matches
            && (label_free_continuation || expected_index.is_some());
        let margin_candidates: Vec<_> = candidates
            .iter()
            .filter_map(|(index, _, _)| {
                let line = &page.lines[*index];
                (line_sizes[*index] <= note_size * 0.90 && line_width(line) <= page.width * 0.30)
                    .then_some(*index)
            })
            .collect();
        let margin_numeric: Vec<u32> = candidates
            .iter()
            .filter(|(index, _, _)| margin_candidates.contains(index))
            .filter_map(|(_, prefix, _)| prefix.label.parse().ok())
            .collect();
        let supported_margin_candidates: Vec<_> = candidates
            .iter()
            .filter(|(index, prefix, _)| {
                margin_candidates.contains(index)
                    && has_prior_reference(page, &prefix.label, page.lines[*index].bbox[1])
            })
            .map(|(index, _, _)| *index)
            .collect();
        let margin_labels = if longest_label_run(&margin_numeric) >= 3 {
            &margin_candidates
        } else {
            &supported_margin_candidates
        };
        let margin_model = margin_note_model(
            &page.lines,
            margin_labels,
            page.width,
            note_size,
            if margin_labels.len() >= 3 { 3 } else { 2 },
        );
        let margin_side = margin_model.map(|model| {
            usize::from(
                margin_labels
                    .iter()
                    .filter(|index| line_center_x(&page.lines[**index]) >= model.split_x)
                    .count()
                    * 2
                    >= margin_labels.len(),
            )
        });
        let endnote_page = has_endnote_heading
            || continuing_endnotes
            || (separator.is_none()
                && margin_model.is_none()
                && !candidates.is_empty()
                && early_labels
                && best_run >= 3
                && label_size.is_some_and(|size| size <= body_size * 0.90));
        let labels: Vec<usize> = candidates[selected_start..]
            .iter()
            .filter_map(|(index, prefix, typographic)| {
                let size = line_sizes[*index];
                let reference_backed = page.lines[*index].bbox[1] >= page.height * 0.48
                    && size > 0.0
                    && size <= body_size * 0.90
                    && has_prior_reference(page, &prefix.label, page.lines[*index].bbox[1]);
                let in_margin = margin_model.zip(margin_side).is_some_and(|(model, side)| {
                    usize::from(line_center_x(&page.lines[*index]) >= model.split_x) == side
                        && size <= note_size * 0.90
                        && line_width(&page.lines[*index]) <= page.width * 0.30
                });
                (endnote_page
                    || in_margin
                    || (page.lines[*index].bbox[1] >= page.height * 0.48
                        && (separator
                            .is_some_and(|cut| page.lines[*index].bbox[1] >= cut - tolerance)
                            || (page_columns.kind == "two_column" && *typographic)
                            || (separator.is_none()
                                && (*typographic
                                    || reference_backed
                                    || (lower_note_run
                                        && page.lines[*index].bbox[1] >= page.height * 0.48
                                        && size > 0.0
                                        && size <= body_size * 0.90))))))
                    .then_some(*index)
            })
            .collect();
        if endnote_page && !labels.is_empty() {
            let first_label = labels
                .iter()
                .map(|index| page.lines[*index].bbox[1])
                .min_by(f64::total_cmp)
                .unwrap_or(page.height);
            for line in &mut page.lines {
                let text = line.text.trim();
                let letter_count = text
                    .chars()
                    .filter(|character| character.is_alphabetic())
                    .count();
                if !matches!(line.region_type.as_str(), "header" | "footer")
                    && line.bbox[1] < page.height * 0.08
                    && line.bbox[1] < first_label
                    && line.bbox[0] >= page.width * 0.15
                    && line.bbox[2] <= page.width * 0.85
                    && letter_count >= 4
                    && text.chars().count() <= 100
                    && all_caps(text)
                    && line_start_label_prefix(text).is_none()
                {
                    line.region_type = "header".to_owned();
                }
            }
        }
        let eligible_top = || {
            page.lines
                .iter()
                .filter(|line| {
                    !matches!(line.region_type.as_str(), "header" | "footer")
                        && !line.exclude_from_body
                })
                .map(|line| line.bbox[1])
                .min_by(f64::total_cmp)
        };
        let selected_sizes: Vec<f64> = labels
            .iter()
            .map(|index| line_sizes[*index])
            .filter(|size| *size > 0.0)
            .collect();
        let selected_size = (!selected_sizes.is_empty()).then(|| median(selected_sizes));
        let separator_starts_small_text = separator.is_some_and(|cut| {
            !evidence.contents_pages.contains(&page.index)
                && page
                    .lines
                    .iter()
                    .enumerate()
                    .filter(|(index, line)| {
                        !table_cells.contains(index)
                            && !table_notes.contains(index)
                            && !line.exclude_from_body
                            && !matches!(line.region_type.as_str(), "header" | "footer")
                            && line.bbox[1] >= cut - tolerance
                    })
                    .min_by(|(_, left), (_, right)| left.bbox[1].total_cmp(&right.bbox[1]))
                    .is_some_and(|(index, line)| {
                        line.bbox[1] - cut <= page.height * 0.06
                            && line_sizes[index] > 0.0
                            && line_sizes[index] <= body_size * 0.90
                    })
        });
        let note_cut = if label_free_continuation {
            eligible_top()
        } else if labels.is_empty() {
            None
        } else {
            let first_label = labels
                .iter()
                .map(|index| page.lines[*index].bbox[1])
                .min_by(f64::total_cmp)
                .unwrap_or(page.height);
            if endnote_page && expected_endnote.is_some() {
                eligible_top()
            } else if separator.is_some_and(|cut| {
                0.0 <= first_label - cut
                    && (first_label - cut <= page.height * 0.15
                        || (separator_starts_small_text
                            && selected_size.is_some_and(|size| size <= body_size * 0.90)))
            }) {
                separator
            } else {
                let confident = (first_label >= page.height * 0.58 || endnote_page)
                    && selected_size.is_some_and(|size| size <= body_size * 0.90);
                if !confident {
                    let mut diagnostic = Diagnostic::warning(
                        "FOOTNOTE_REGION_UNCERTAIN",
                        "Footnote region inferred from weak label geometry without a separator.",
                        Some(page.index),
                    );
                    if let Some(index) = labels.iter().min_by(|left, right| {
                        page.lines[**left].bbox[1].total_cmp(&page.lines[**right].bbox[1])
                    }) {
                        diagnostic.line_ids.push(page.lines[*index].id.clone());
                    }
                    diagnostics.push(diagnostic);
                }
                Some(first_label)
            }
        };
        let note_columns = margin_model.unwrap_or(page_columns);
        let column_cuts = if (note_columns.kind == "two_column" || margin_model.is_some())
            && (!endnote_page || has_endnote_heading)
        {
            let mut cuts = [None::<f64>; 2];
            for index in &labels {
                let label = &page.lines[*index];
                let column = usize::from(line_center_x(label) >= note_columns.split_x);
                let label_top = aligned_note_body_index(&page.lines, *index)
                    .map(|body| band_geometry_top(&page.lines[body]))
                    .unwrap_or_else(|| band_geometry_top(label))
                    .min(band_geometry_top(label));
                cuts[column] = Some(cuts[column].map_or(label_top, |cut| cut.min(label_top)));
            }
            if cuts.iter().all(Option::is_some)
                || (note_columns.kind == "margin_column" && cuts.iter().any(Option::is_some))
            {
                if let Some(heading_y) = endnote_heading_y {
                    for (column, cut) in cuts.iter_mut().enumerate() {
                        if cut.is_some_and(|value| value < heading_y) {
                            *cut = page
                                .lines
                                .iter()
                                .filter(|line| {
                                    !matches!(line.region_type.as_str(), "header" | "footer")
                                        && !line.exclude_from_body
                                        && usize::from(line_center_x(line) >= note_columns.split_x)
                                            == column
                                })
                                .map(|line| line.bbox[1])
                                .min_by(f64::total_cmp);
                        }
                    }
                }
                Some(cuts)
            } else {
                None
            }
        } else {
            None
        };
        for (index, line) in page.lines.iter_mut().enumerate() {
            if line.exclude_from_body {
                continue;
            }
            let size = line_sizes[index];
            let is_note = endnote_heading_source.is_some_and(|start| {
                column_cuts.is_none() && endnote_page && line.source_index >= start
            }) || column_cuts.as_ref().map_or_else(
                || note_cut.is_some_and(|cut| line.bbox[1] >= cut),
                |cuts| {
                    let column = usize::from(line_center_x(line) >= note_columns.split_x);
                    cuts[column].is_some_and(|cut| line.bbox[1] >= cut)
                },
            );
            if endnote_heading(&line.text) {
                line.region_type = "heading".to_owned();
                line.note_region_mode.clear();
            } else if table_notes.contains(&index) {
                line.region_type = "footnote".to_owned();
                line.note_region_mode = "footnote".to_owned();
            } else if table_cells.contains(&index) {
                line.region_type = "body".to_owned();
                line.note_region_mode.clear();
            } else if matches!(line.region_type.as_str(), "header" | "footer") {
                continue;
            } else if is_note {
                line.region_type = "footnote".to_owned();
                line.note_region_mode =
                    if endnote_page { "endnote" } else { "footnote" }.to_owned();
            } else if line.text.chars().count() <= 180
                && size
                    >= (if article_body_size > 0.0 {
                        article_body_size
                    } else {
                        body_size
                    }) * 1.18
                && has_word_character(&line.text)
            {
                line.region_type = "heading".to_owned();
            } else {
                line.region_type = "body".to_owned();
            }
        }
        if !page
            .lines
            .iter()
            .any(|line| line.note_region_mode == "endnote")
        {
            expected_endnote = None;
            continuing_size = None;
        } else {
            if let Some(last) = page
                .lines
                .iter()
                .rev()
                .filter(|line| line.note_region_mode == "endnote")
                .filter_map(|line| label_prefix(&line.text)?.label.parse::<u32>().ok())
                .next()
            {
                expected_endnote = Some(last + 1);
            }
            let size = median(
                page.lines
                    .iter()
                    .enumerate()
                    .filter(|(_, line)| line.note_region_mode == "endnote")
                    .map(|(index, _)| line_sizes[index])
                    .filter(|size| *size > 0.0)
                    .collect(),
            );
            if size > 0.0 {
                continuing_size = Some(size);
            }
        }
        diagnostics.extend(order_page(page, table_page, &table_notes));
        build_regions(std::slice::from_mut(page));
    }
    let heading_levels = apply_text_fidelity_headings(pages, article_body_size, evidence);
    evidence.heading_levels = heading_levels;
    build_regions(pages);
    diagnostics
}

#[cfg(test)]
fn classify_pages(pages: &mut [Page], separators: &[Option<f64>]) -> Vec<Diagnostic> {
    let mut evidence = PdfPrimitiveEvidence {
        source_regions: source_region_contract(pages),
        ..PdfPrimitiveEvidence::default()
    };
    classify_pages_with_source(pages, separators, &mut evidence)
}

fn assign_printed_page_labels(pages: &mut [Page]) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    for page in pages {
        page.printed_label = None;
        page.printed_label_source = None;
        page.printed_label_line_id = None;
        let candidates: Vec<(String, usize)> = page
            .lines
            .iter()
            .enumerate()
            .filter(|(_, line)| matches!(line.region_type.as_str(), "header" | "footer"))
            .filter_map(|(index, line)| printed_label(&line.text).map(|label| (label, index)))
            .collect();
        let labels: BTreeSet<String> = candidates
            .iter()
            .map(|(label, _)| label.to_lowercase())
            .collect();
        if labels.len() > 1 {
            let mut diagnostic = Diagnostic::info(
                "PRINTED_PAGE_LABEL_AMBIGUOUS",
                "Conflicting header/footer page labels were left unresolved.",
                Some(page.index),
            );
            diagnostic.line_ids = candidates
                .iter()
                .map(|(_, index)| page.lines[*index].id.clone())
                .collect();
            diagnostic.details.insert(
                "candidates".to_owned(),
                json!(candidates
                    .iter()
                    .map(|(label, _)| label)
                    .collect::<Vec<_>>()),
            );
            diagnostics.push(diagnostic);
            continue;
        }
        let chosen = candidates.iter().min_by_key(|(_, index)| {
            (
                page.lines[*index].region_type != "footer",
                page.lines[*index].reading_order,
            )
        });
        if let Some((label, index)) = chosen {
            let line = &page.lines[*index];
            page.printed_label = Some(label.clone());
            page.printed_label_source = Some(line.region_type.clone());
            page.printed_label_line_id = Some(line.id.clone());
        }
    }
    diagnostics
}

fn citation_shaped_tail(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"^\s+(?:\[\d{4}\]\s+)?(?:[A-Z][A-Za-z0-9.&'-]*\s+){1,4}(?:\([^\)\r\n]{1,40}\)\s+)?\d+\b",
        )
        .unwrap()
    })
    .is_match(text)
}

fn join_lines(lines: &[&Line]) -> (String, Vec<Option<(usize, usize)>>) {
    let mut text = String::new();
    let mut offsets = Vec::with_capacity(lines.len());
    let mut scalar_cursor = 0;
    for line in lines {
        let value = line.text.trim();
        if value.is_empty() {
            offsets.push(None);
            continue;
        }
        if text
            .chars()
            .next_back()
            .is_some_and(|character| matches!(character, '-' | '\u{00ad}' | '\u{00ac}'))
            && value.chars().next().is_some_and(char::is_lowercase)
        {
            text.pop();
            scalar_cursor -= 1;
        } else if !text.is_empty() {
            text.push(' ');
            scalar_cursor += 1;
        }
        let start = scalar_cursor;
        text.push_str(value);
        scalar_cursor += value.chars().count();
        offsets.push(Some((start, scalar_cursor)));
    }
    (text, offsets)
}

fn build_paragraphs(pages: &[Page], anchors: &HashMap<String, Vec<Anchor>>) -> Vec<Paragraph> {
    let mut paragraphs = Vec::new();
    for page in pages {
        let line_by_id: HashMap<&str, &Line> = page
            .lines
            .iter()
            .map(|line| (line.id.as_str(), line))
            .collect();
        for region in &page.regions {
            if !matches!(region.kind.as_str(), "body" | "heading") {
                continue;
            }
            let lines: Vec<&Line> = region
                .line_ids
                .iter()
                .filter_map(|id| line_by_id.get(id.as_str()).copied())
                .filter(|line| !line.exclude_from_body)
                .collect();
            let (text, line_offsets) = join_lines(&lines);
            if text.is_empty() {
                continue;
            }
            let mut events = Vec::new();
            for (line, offset) in lines.iter().zip(&line_offsets) {
                let Some((base, _)) = offset else {
                    continue;
                };
                for anchor in anchors.get(&line.id).into_iter().flatten() {
                    events.push((
                        *base + anchor.start,
                        *base + anchor.end,
                        anchor.pair_id.as_str(),
                        anchor.label.as_str(),
                    ));
                }
            }
            events.sort_by_key(|event| (event.0, event.1));
            let (rendered, output_anchors) = if events.is_empty() {
                (text, Vec::new())
            } else {
                let coordinates = ScalarText::new(&text);
                let text_len = coordinates.len();
                let mut rendered = String::with_capacity(text.len());
                let mut output_anchors = Vec::with_capacity(events.len());
                let (mut cursor, mut rendered_len) = (0, 0);
                for (start, end, pair_id, label) in events {
                    let start = start.max(cursor).min(text_len);
                    let end = end.max(start).min(text_len);
                    rendered.push_str(coordinates.slice(cursor..start).unwrap());
                    rendered_len += start - cursor;
                    let offset = rendered_len;
                    rendered.push_str("⟦FN:");
                    rendered.push_str(&pair_id);
                    rendered.push('⟧');
                    rendered_len += 5 + pair_id.chars().count();
                    output_anchors.push(ParagraphAnchor {
                        pair_id: pair_id.to_owned(),
                        label: label.to_owned(),
                        offset,
                    });
                    cursor = end;
                }
                rendered.push_str(coordinates.slice(cursor..text_len).unwrap());
                (rendered, output_anchors)
            };
            paragraphs.push(Paragraph {
                id: format!("para-{:06}", paragraphs.len() + 1),
                page_index: page.index,
                region_type: region.kind.clone(),
                text: rendered,
                line_ids: lines.iter().map(|line| line.id.clone()).collect(),
                anchors: output_anchors,
            });
        }
    }
    paragraphs
}

fn marker_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"⟦FN:[^⟧]+⟧").unwrap())
}

fn clean_markers(value: &str) -> String {
    marker_re().replace_all(value, "").trim().to_owned()
}

fn sentence_boundaries(chars: &[char]) -> Vec<(usize, usize)> {
    let mut boundaries = Vec::<(usize, usize)>::new();
    let mut index = 0;
    while index < chars.len() {
        if matches!(chars[index], '.' | '?' | '!') {
            let start = index;
            index += 1;
            while chars
                .get(index)
                .is_some_and(|character| ['"', '\'', '”', '’', ')', ']'].contains(character))
            {
                index += 1;
            }
            let marker_follows = chars
                .get(index..)
                .is_some_and(|tail| tail.starts_with(&['⟦', 'F', 'N', ':']));
            if index == chars.len()
                || chars
                    .get(index)
                    .is_some_and(|character| character.is_whitespace())
                || marker_follows
            {
                boundaries.push((start, index));
            }
            continue;
        }
        index += 1;
    }
    boundaries
}

fn sentence_at_boundaries(chars: &[char], boundaries: &[(usize, usize)], offset: usize) -> String {
    let previous = &boundaries[..boundaries.partition_point(|(_, end)| *end <= offset)];
    let (start, end) = if previous.last().is_some_and(|(_, end)| {
        chars[*end..offset.min(chars.len())]
            .iter()
            .all(|character| character.is_whitespace())
    }) {
        (
            if previous.len() > 1 {
                previous[previous.len() - 2].1
            } else {
                0
            },
            previous.last().map_or(chars.len(), |(_, end)| *end),
        )
    } else {
        let start = previous.last().map_or(0, |(_, end)| *end);
        let end = boundaries
            .iter()
            .find(|(boundary_start, _)| *boundary_start >= offset)
            .map_or(chars.len(), |(_, boundary_end)| *boundary_end);
        (start, end)
    };
    clean_markers(&chars[start..end].iter().collect::<String>())
}

#[cfg(test)]
fn sentence_at(text: &str, offset: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    sentence_at_boundaries(&chars, &sentence_boundaries(&chars), offset)
}

fn attach_propositions(footnotes: &mut [Footnote], paragraphs: &[Paragraph]) {
    let by_pair: HashMap<String, usize> = footnotes
        .iter()
        .enumerate()
        .map(|(index, footnote)| (footnote.pair_id.clone(), index))
        .collect();
    let mut previous_tail = String::new();
    for paragraph in paragraphs {
        let chars: Vec<char> = paragraph.text.chars().collect();
        let boundaries = sentence_boundaries(&chars);
        let mut previous_offset = 0;
        for anchor in &paragraph.anchors {
            let Some(&index) = by_pair.get(&anchor.pair_id) else {
                continue;
            };
            let offset = anchor.offset;
            footnotes[index].sentence_proposition =
                sentence_at_boundaries(&chars, &boundaries, offset);
            let passage_text: String = chars
                [previous_offset.min(chars.len())..offset.min(chars.len())]
                .iter()
                .collect();
            let passage = clean_markers(&passage_text);
            footnotes[index].passage_since_prior_note = if passage.is_empty() {
                previous_tail.clone()
            } else {
                passage
            };
            previous_offset = offset + 5 + anchor.pair_id.chars().count();
        }
        previous_tail = chars[previous_offset.min(chars.len())..]
            .iter()
            .collect::<String>()
            .trim()
            .to_owned();
    }
}

fn infer_note_region_modes(pages: &mut [Page]) {
    let mut prior_note_page = false;
    let mut active_endnotes = false;
    let mut expected_endnote = None::<u32>;
    for page in pages {
        let footnote_indexes: Vec<usize> = page
            .lines
            .iter()
            .enumerate()
            .filter_map(|(index, line)| (line.region_type == "footnote").then_some(index))
            .collect();
        let heading = page.lines.iter().any(|line| endnote_heading(&line.text));
        let explicit_endnote = footnote_indexes
            .iter()
            .any(|&index| page.lines[index].note_region_mode == "endnote");
        let numbers: Vec<u32> = footnote_indexes
            .iter()
            .filter_map(|&index| line_start_label_prefix(&page.lines[index].text))
            .filter_map(|prefix| normalize_label(&prefix.label).parse().ok())
            .collect();
        let continues_endnotes = active_endnotes
            && (numbers.is_empty()
                || expected_endnote.is_some_and(|expected| numbers[0] == expected));
        if !footnote_indexes.is_empty() && (heading || explicit_endnote || continues_endnotes) {
            for &index in &footnote_indexes {
                page.lines[index].note_region_mode = "endnote".to_owned();
            }
            active_endnotes = true;
            if let Some(number) = numbers.last() {
                expected_endnote = number.checked_add(1);
            }
        } else if !footnote_indexes.is_empty() {
            let explicit_footnote = footnote_indexes
                .iter()
                .any(|&index| page.lines[index].note_region_mode == "footnote");
            let mode = if explicit_footnote {
                "footnote"
            } else if prior_note_page {
                "footnote_continuation"
            } else {
                ""
            };
            if !mode.is_empty() {
                for &index in &footnote_indexes {
                    if page.lines[index].note_region_mode.is_empty() {
                        page.lines[index].note_region_mode = mode.to_owned();
                    }
                }
            }
            active_endnotes = false;
            expected_endnote = None;
        } else {
            active_endnotes = false;
            expected_endnote = None;
        }
        prior_note_page = !footnote_indexes.is_empty();
    }
}

fn crossref_shortform(text: &str, byte_start: usize) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    let pattern = RE.get_or_init(|| {
        Regex::new(
            r"(?:\[|^|[^\p{L}\p{M}\p{N}_])([A-Z][\w.'’&-]*(?:\s+(?:[A-Z][\w.'’&-]*|v\.?|c\.?|de|du|and|&)){0,5})\]?[,:]?\s*$",
        )
        .unwrap()
    });
    let Some(capture) = pattern.captures(last_scalars(&text[..byte_start], 70)) else {
        return String::new();
    };
    let short = capture
        .get(1)
        .map_or("", |value| value.as_str())
        .trim()
        .trim_end_matches([',', '.', ';', ':'])
        .to_owned();
    if "see in the but and also supra infra ibid at"
        .split(' ')
        .any(|word| short.eq_ignore_ascii_case(word))
    {
        String::new()
    } else {
        short
    }
}

fn line_family(line: &Line) -> &'static str {
    match line.region_type.as_str() {
        "footnote" => "note",
        "heading" => "heading",
        _ => "body",
    }
}

fn hyphen_fragment_tail(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[A-Za-z]{2,}[-\u{00ac}\u{00ad}]$").unwrap())
        .is_match(text.trim_end())
}

fn hyphen_continuation(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[A-Za-z]{2,}").unwrap())
        .is_match(text.trim_start())
}

fn text_flow_faults(pages: &[Page]) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    for page in pages {
        let mut eligible = page.lines.iter().filter(|line| {
            !matches!(line.region_type.as_str(), "header" | "footer") && !line.exclude_from_body
        });
        let Some(mut previous) = eligible.next() else {
            continue;
        };
        for current in eligible {
            if hyphen_fragment_tail(&previous.text) {
                let continues = hyphen_continuation(&current.text);
                if continues && line_family(previous) != line_family(current) {
                    let mut diagnostic = Diagnostic::warning(
                        "REGION_BOUNDARY_FAULT",
                        format!(
                            "A hyphenated word spans the {}/{} boundary; either a region label or the order is wrong.",
                            line_family(previous),
                            line_family(current)
                        ),
                        Some(page.index),
                    );
                    diagnostic.line_ids = vec![previous.id.clone(), current.id.clone()];
                    diagnostics.push(diagnostic);
                } else if !continues && line_family(previous) == line_family(current) {
                    let mut diagnostic = Diagnostic::info(
                        "DANGLING_SOFT_HYPHEN",
                        "A line ends mid-word but the next eligible line does not continue it.",
                        Some(page.index),
                    );
                    diagnostic.line_ids.push(previous.id.clone());
                    diagnostics.push(diagnostic);
                }
            }
            previous = current;
        }
    }
    diagnostics
}

fn unmatched_reference_diagnostics(
    pages: &[Page],
    footnotes: &[Footnote],
    anchors: &HashMap<String, Vec<Anchor>>,
) -> Vec<Diagnostic> {
    let labels: HashSet<&str> = footnotes.iter().map(|note| note.label.as_str()).collect();
    let primary_lines: HashSet<&str> = footnotes
        .iter()
        .filter_map(|note| note.reference_line_id.as_deref())
        .collect();
    let mut diagnostics = Vec::new();
    for line in pages.iter().flat_map(|page| page.lines.iter()) {
        for detached in &line.detached_references {
            let label = normalize_label(&detached.note_id);
            let start = detached.start_offset;
            let end = detached.end_offset;
            let paired = anchors.get(&line.id).is_some_and(|entries| {
                entries.iter().any(|anchor| {
                    anchor.start == start && anchor.end == end && anchor.label == label
                })
            });
            if !paired {
                let mut diagnostic = Diagnostic::warning(
                    "FOOTNOTE_UNMATCHED_REFERENCE",
                    format!("Detached superscript '{label}' has no paired label."),
                    Some(line.page_index),
                );
                diagnostic.line_ids.push(line.id.clone());
                if !detached.source_line_id.is_empty() {
                    diagnostic.line_ids.push(detached.source_line_id.clone());
                }
                diagnostic.details.insert("label".to_owned(), json!(label));
                diagnostics.push(diagnostic);
            }
        }
        if line.exclude_from_body
            || !matches!(line.region_type.as_str(), "body" | "heading")
            || primary_lines.contains(line.id.as_str())
        {
            continue;
        }
        if line.spans.iter().any(|span| {
            span.superscript && labels.contains(normalize_label(span.text.trim()).as_str())
        }) {
            let mut diagnostic = Diagnostic::warning(
                "FOOTNOTE_UNMATCHED_REFERENCE",
                "A superscript resembling a known note label was not paired.",
                Some(line.page_index),
            );
            diagnostic.line_ids.push(line.id.clone());
            diagnostics.push(diagnostic);
        }
    }
    diagnostics
}

fn attach_crossrefs(footnotes: &mut [Footnote], diagnostics: &mut Vec<Diagnostic>) {
    static RE: OnceLock<Regex> = OnceLock::new();
    let pattern = RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:(supra|infra),?\s+(?:foot)?notes?|(op)\.?\s*cit\.?,?\s+(?:foot)?notes?|(see)\s+(?:also\s+)?footnote)\s+(\d{1,3})\b",
        )
        .unwrap()
    });
    let mut by_number: HashMap<u32, Vec<(usize, String)>> = HashMap::new();
    for footnote in footnotes.iter() {
        if let Ok(number) = footnote.label.parse::<u32>() {
            by_number
                .entry(number)
                .or_default()
                .push((footnote.restart_sequence, footnote.pair_id.clone()));
        }
    }
    for footnote in footnotes {
        let (mut prior_byte, mut prior_scalar) = (0, 0);
        for capture in pattern.captures_iter(&footnote.body) {
            let Some(found) = capture.get(0) else {
                continue;
            };
            let number = capture
                .get(4)
                .and_then(|value| value.as_str().parse::<u32>().ok())
                .unwrap_or(0);
            let candidates = by_number
                .get(&number)
                .map(Vec::as_slice)
                .unwrap_or_default();
            let target_pair_id = if candidates.len() == 1 {
                candidates[0].1.clone()
            } else {
                let mut scoped = candidates
                    .iter()
                    .filter(|(restart, _)| *restart == footnote.restart_sequence);
                match (scoped.next(), scoped.next()) {
                    (Some((_, pair_id)), None) => pair_id.clone(),
                    _ => String::new(),
                }
            };
            let kind = if let Some(value) = capture.get(1) {
                value.as_str().to_lowercase()
            } else if capture.get(2).is_some() {
                "op_cit".to_owned()
            } else {
                "see_footnote".to_owned()
            };
            let start = prior_scalar + footnote.body[prior_byte..found.start()].chars().count();
            let end = start + found.as_str().chars().count();
            (prior_byte, prior_scalar) = (found.end(), end);
            let record = FootnoteCrossref {
                source_pair_id: footnote.pair_id.clone(),
                kind,
                number,
                shortform: crossref_shortform(&footnote.body, found.start()),
                start,
                end,
                resolved: !candidates.is_empty(),
                target_pair_id,
                target_count: candidates.len(),
            };
            if candidates.is_empty() {
                let mut diagnostic = Diagnostic::info(
                    "NOTE_CROSSREF_UNRESOLVED",
                    format!(
                        "Note {} references {} note {}, which no paired note carries - a pairing-quality witness.",
                        footnote.label, record.kind, number
                    ),
                    footnote.reference_page.map(|page| page as usize),
                );
                diagnostic
                    .details
                    .insert("crossref".to_owned(), json!(&record));
                diagnostics.push(diagnostic);
            }
            footnote.crossrefs.push(record);
        }
    }
}

fn prepare_pages(pages: &mut [Page], separators: &[Option<f64>]) -> Result<PdfPreparation> {
    if separators.len() != pages.len() {
        return Err(Error::Message(
            "common input must contain one separator value per page".to_owned(),
        ));
    }
    let mut primitives = PdfPrimitiveEvidence {
        source_regions: legal_pdf_support::profile::measure("prepare.source_regions", || {
            source_region_contract(pages)
        }),
        ..PdfPrimitiveEvidence::default()
    };
    legal_pdf_support::profile::measure("prepare.furniture", || mark_repeated_furniture(pages));
    legal_pdf_support::profile::measure("prepare.detached_references", || {
        associate_detached_references(pages, separators)
    });
    let mut diagnostics = legal_pdf_support::profile::measure("prepare.classify", || {
        classify_pages_with_source(pages, separators, &mut primitives)
    });
    diagnostics.extend(legal_pdf_support::profile::measure(
        "prepare.printed_labels",
        || assign_printed_page_labels(pages),
    ));
    Ok(PdfPreparation {
        diagnostics,
        primitives,
    })
}

fn derive_prepared(
    pages: &mut [Page],
    mut prepared: PdfPreparation,
    identity: StructureIdentity,
    include_pairing_audit: bool,
) -> Result<StructureOutput> {
    legal_pdf_support::profile::measure("derive.note_regions", || infer_note_region_modes(pages));
    prepared
        .diagnostics
        .extend(legal_pdf_support::profile::measure(
            "derive.text_flow",
            || text_flow_faults(pages),
        ));
    let resolution = legal_pdf_support::profile::measure("derive.structure_candidates", || {
        PdfResolutionInput::from_pages(pages, &prepared.primitives)
    });
    let pairing =
        legal_pdf_pairing::pair_footnotes(pages, &resolution.citation_spans, include_pairing_audit);
    let (note_pairs, graph_diagnostics) = map_note_pairs(&resolution.index, &pairing.pair_claims)?;
    let pairing_audit = pairing.pairing_audit;
    let anchors = pairing.anchors;
    let mut footnotes = pairing.footnotes;
    let mut diagnostics = prepared.diagnostics;
    diagnostics.extend(pairing.diagnostics);
    diagnostics.extend(legal_pdf_support::profile::measure(
        "derive.unmatched_references",
        || unmatched_reference_diagnostics(pages, &footnotes, &anchors),
    ));
    let paragraphs = legal_pdf_support::profile::measure("derive.paragraphs", || {
        build_paragraphs(pages, &anchors)
    });
    legal_pdf_support::profile::measure("derive.propositions", || {
        attach_propositions(&mut footnotes, &paragraphs)
    });
    legal_pdf_support::profile::measure("derive.crossrefs", || {
        attach_crossrefs(&mut footnotes, &mut diagnostics)
    });
    let nodes = native_graph_parts(&resolution.index, pages, &paragraphs)?;
    let structure_graph = legal_pdf_support::profile::measure("derive.structure_graph", || {
        resolve_structure_graph(
            identity.document_id,
            resolution.index.text(),
            Some(identity.source_sha256),
            nodes,
            &resolution.runs,
            &resolution.evidence,
            &note_pairs,
            graph_diagnostics,
        )
    })
    .map_err(|error| Error::Message(error.to_string()))?;
    Ok(StructureOutput {
        paragraphs,
        footnotes,
        diagnostics,
        pairing_audit,
        structure_graph,
    })
}

pub fn derive(
    pages: &mut [Page],
    separators: &[Option<f64>],
    identity: StructureIdentity,
) -> Result<StructureOutput> {
    let prepared = prepare_pages(pages, separators)?;
    derive_prepared(pages, prepared, identity, false)
}

pub fn replay(
    pages: &mut [Page],
    separators: &[Option<f64>],
    identity: StructureIdentity,
) -> Result<StructureReplay> {
    let _profile = legal_pdf_support::profile::scope("structure_replay");
    let prepared = prepare_pages(pages, separators)?;
    let prepared_pages = pages.to_vec();
    let derived = derive_prepared(pages, prepared, identity, true)?;
    Ok(StructureReplay {
        prepared_pages,
        derived,
    })
}

pub fn status(diagnostics: &[Diagnostic], pages: &[Page]) -> String {
    let ocr_pages: HashSet<Option<usize>> = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code == "OCR_REQUIRED")
        .map(|diagnostic| diagnostic.page_index)
        .collect();
    if !pages.is_empty() && ocr_pages.len() == pages.len() {
        return "ocr_required".to_owned();
    }
    if diagnostics.iter().any(|diagnostic| {
        matches!(diagnostic.severity.as_str(), "warning" | "error")
            && (diagnostic.code == "OCR_REQUIRED"
                || HARD_DIAGNOSTICS.contains(&diagnostic.code.as_str()))
    }) {
        "degraded".to_owned()
    } else {
        "ready".to_owned()
    }
}

fn validate_page_records(pages: &[Page]) -> Result<HashSet<&str>> {
    let mut ids = HashSet::new();
    let mut span_ids = HashSet::new();
    let mut word_ids = HashSet::new();
    for page in pages {
        if page.lines.iter().any(|line| !ids.insert(line.id.as_str())) {
            return Err(Error::Message(
                "document contains duplicate line IDs".to_owned(),
            ));
        }
        let mut regions_by_line = HashMap::new();
        let duplicate_region_line = page.regions.iter().any(|region| {
            region
                .line_ids
                .iter()
                .any(|line| regions_by_line.insert(line.as_str(), region).is_some())
        });
        if duplicate_region_line
            || regions_by_line.len() != page.lines.len()
            || page
                .lines
                .iter()
                .any(|line| !regions_by_line.contains_key(line.id.as_str()))
        {
            return Err(Error::Message(format!(
                "page {} region coverage is incomplete",
                page.number
            )));
        }
        for line in &page.lines {
            if !line
                .spans
                .iter()
                .all(|span| span_ids.insert(span.id.as_str()))
            {
                return Err(Error::Message(
                    "document contains duplicate span IDs".to_owned(),
                ));
            }
            if !line
                .words
                .iter()
                .all(|word| word_ids.insert(word.id.as_str()))
            {
                return Err(Error::Message(
                    "document contains duplicate word IDs".to_owned(),
                ));
            }
            let region = regions_by_line
                .get(line.id.as_str())
                .copied()
                .ok_or_else(|| {
                    Error::Message(format!("line {} has no containing region", line.id))
                })?;
            if line.region_id != region.id || line.region_type != region.kind {
                return Err(Error::Message(format!(
                    "page {} line/region annotations disagree for {}",
                    page.number, line.id
                )));
            }
            let mut prior_end = 0;
            let scalar_boundaries = (!line.text.is_ascii()).then(|| {
                line.text
                    .char_indices()
                    .map(|(index, _)| index)
                    .chain(std::iter::once(line.text.len()))
                    .collect::<Vec<_>>()
            });
            let scalar_len = scalar_boundaries
                .as_ref()
                .map_or(line.text.len(), |boundaries| boundaries.len() - 1);
            for word in &line.words {
                let text = scalar_boundaries.as_ref().map_or_else(
                    || line.text.get(word.start..word.end),
                    |boundaries| {
                        boundaries
                            .get(word.start)
                            .zip(boundaries.get(word.end))
                            .and_then(|(&start, &end)| line.text.get(start..end))
                    },
                );
                if word.start < prior_end
                    || word.end <= word.start
                    || word.end > scalar_len
                    || text != Some(word.text.as_str())
                {
                    return Err(Error::Message(format!(
                        "line {} contains invalid word geometry",
                        line.id
                    )));
                }
                prior_end = word.end;
            }
        }
        let printed = (
            page.printed_label.as_deref(),
            page.printed_label_source.as_deref(),
            page.printed_label_line_id.as_deref(),
        );
        if printed.0.is_some() || printed.1.is_some() || printed.2.is_some() {
            let (Some(label), Some(source), Some(line_id)) = printed else {
                return Err(Error::Message(format!(
                    "page {} has incomplete printed-label provenance",
                    page.number
                )));
            };
            let valid = page.lines.iter().any(|line| {
                line.id == line_id
                    && line.region_type == source
                    && matches!(source, "header" | "footer")
                    && printed_label(line.text.trim()).as_deref() == Some(label)
            });
            if !valid {
                return Err(Error::Message(format!(
                    "page {} has invalid printed-label provenance",
                    page.number
                )));
            }
        }
    }
    Ok(ids)
}

pub fn validate_document(document: &LegalDocument) -> Result<()> {
    if document.page_count != document.pages.len() {
        return Err(Error::Message(
            "document page_count does not match the page collection".to_owned(),
        ));
    }
    validate_pdf_components(
        &document.document_id,
        &document.source_sha256,
        &document.pages,
        &document.paragraphs,
        &document.footnotes,
        &document.structure_graph,
    )
}

pub fn validate_pdf_components(
    document_id: &str,
    source_sha256: &str,
    pages: &[Page],
    paragraphs: &[Paragraph],
    footnotes: &[Footnote],
    structure_graph: &DocumentStructure,
) -> Result<()> {
    let known_lines = validate_page_records(pages)?;
    let known_pages = pages.iter().map(|page| page.index).collect::<HashSet<_>>();
    let mut pair_ids = HashSet::new();
    for footnote in footnotes {
        if !pair_ids.insert(&footnote.pair_id) {
            return Err(Error::Message(
                "document contains duplicate footnote pair IDs".to_owned(),
            ));
        }
        if footnote
            .reference_line_id
            .as_ref()
            .is_some_and(|line| !known_lines.contains(line.as_str()))
            || footnote
                .body_line_ids
                .iter()
                .any(|line| !known_lines.contains(line.as_str()))
        {
            return Err(Error::Message(format!(
                "footnote {} contains an unknown source line",
                footnote.pair_id
            )));
        }
    }
    let mut paragraph_ids = HashSet::new();
    for paragraph in paragraphs {
        if !paragraph_ids.insert(&paragraph.id)
            || paragraph
                .line_ids
                .iter()
                .any(|line| !known_lines.contains(line.as_str()))
        {
            return Err(Error::Message(format!(
                "paragraph {} is invalid",
                paragraph.id
            )));
        }
    }
    if structure_graph.document_id != document_id
        || structure_graph.source_sha256.as_deref() != Some(source_sha256)
    {
        return Err(Error::Message(
            "structure graph identity disagrees with the document".to_owned(),
        ));
    }
    let (text_length, line_count) = pages
        .iter()
        .flat_map(|page| &page.lines)
        .fold((0, 0_usize), |(units, count), line| {
            (units + utf16_len(&line.text), count + 1)
        });
    let text_length = text_length + line_count.saturating_sub(1);
    let mut node_ids = HashSet::new();
    for node in &structure_graph.nodes {
        if node.id.is_empty()
            || !node_ids.insert(node.id.as_str())
            || node.range.start > node.range.end
            || node.range.end > text_length
            || node
                .page_indexes
                .iter()
                .any(|page| !known_pages.contains(page))
            || node
                .line_ids
                .iter()
                .any(|line| !known_lines.contains(line.as_str()))
            || (node.kind == NodeKind::Section
                && (node.locator_kind.as_deref().is_none_or(str::is_empty)
                    || node
                        .proof
                        .as_ref()
                        .is_none_or(|proof| proof.rule != ResolutionRuleV2::HierarchySection)))
        {
            return Err(Error::Message(format!(
                "structure node {} is invalid",
                node.id
            )));
        }
    }
    if structure_graph.nodes.iter().any(|node| {
        node.parent_id
            .as_deref()
            .is_some_and(|parent| !node_ids.contains(parent))
    }) {
        return Err(Error::Message(
            "structure graph contains an unknown parent".to_owned(),
        ));
    }
    for note in &structure_graph.notes {
        let valid_range = |range: ScalarRange| range.start <= range.end && range.end <= text_length;
        if !node_ids.contains(note.node_id.as_str())
            || !valid_range(note.label_range)
            || !valid_range(note.body_range)
            || note
                .primary_reference
                .is_some_and(|range| !valid_range(range))
            || note.references.iter().any(|reference| {
                !valid_range(reference.range)
                    || reference
                        .page_indexes
                        .iter()
                        .any(|page| !known_pages.contains(page))
                    || reference
                        .line_ids
                        .iter()
                        .any(|line| !known_lines.contains(line.as_str()))
            })
        {
            return Err(Error::Message(format!(
                "structure note {} is invalid",
                note.id
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
