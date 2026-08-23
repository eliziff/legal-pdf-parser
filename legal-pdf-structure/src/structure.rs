//! Shared structure derivation for aligned page and line evidence.

use legal_pdf_core::model::{
    Diagnostic, Footnote, LegalDocument, Line, NotePairClaim, NotePairKind, Page, Paragraph,
    PdfPageIdentity, PdfPairingAudit, PdfSourceExtent, PdfSourceMap, PdfSourceSpan, Region, Span,
};
use legal_pdf_core::{line_font_size, union_bbox, Anchor, Error, PairingOutput, Result};
use legal_pdf_support::{
    enumerator_interpretations, has_citation_signal, heading_text_plausible, parse_heading_ladder,
    protected_citation_spans,
};
use legal_structure::{
    detect_structure_candidate_runs, resolve_structure_graph, CandidateEvidenceV2,
    CandidateGrammar, CandidateObservationV2, Derivation, DiagnosticSeverity, DocumentStructure,
    NodeKind, NoteBodyV2, NoteKindV2, NotePairClaimV2, ResolutionRuleV2, ScalarRange,
    StructureCandidateRun, StructureDiagnostic, StructureNode, TextAnchorV2,
};
use regex::Regex;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::OnceLock;

const HARD_DIAGNOSTICS: &[&str] = &[
    "COLUMN_ORDER_UNCERTAIN",
    "FOOTNOTE_UNMATCHED_LABEL",
    "FOOTNOTE_UNMATCHED_REFERENCE",
    "FOOTNOTE_REGION_UNCERTAIN",
    "TEXT_QUALITY_LOW",
];
const MAX_SYMBOL_LABEL_LEN: usize = 8;

#[derive(Debug, Clone)]
struct LabelPrefix {
    label: String,
    start: usize,
    end: usize,
}

#[derive(Debug, Default)]
struct PdfPrimitiveEvidence {
    source_regions: Option<HashMap<String, String>>,
    table_cell_line_ids: HashSet<String>,
    table_note_line_ids: HashSet<String>,
    heading_levels: HashMap<String, usize>,
}

#[derive(Debug)]
pub struct PdfPreparation {
    diagnostics: Vec<Diagnostic>,
    primitives: PdfPrimitiveEvidence,
    resolution: Option<PdfResolutionInput>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructureIdentity {
    pub document_id: String,
    pub source_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedPdfLine {
    pub page_id: String,
    pub page_index: usize,
    pub line_id: String,
    pub range: ScalarRange,
}

#[derive(Debug, Clone)]
pub struct PdfTextIndex {
    text: String,
    lines: Vec<IndexedPdfLine>,
    line_slots: HashMap<String, usize>,
    page_ranges: HashMap<usize, ScalarRange>,
}

#[derive(Debug)]
struct PdfResolutionInput {
    index: PdfTextIndex,
    runs: Vec<StructureCandidateRun>,
    evidence: Vec<CandidateEvidenceV2>,
    citation_spans: BTreeMap<String, Vec<PdfSourceSpan>>,
}

impl PdfTextIndex {
    pub fn from_pages(pages: &[Page]) -> Self {
        let mut ordered_pages = pages.iter().collect::<Vec<_>>();
        ordered_pages.sort_by_key(|page| page.index);
        let mut text = String::new();
        let mut lines = Vec::new();
        let mut line_slots = HashMap::new();
        let mut page_ranges = HashMap::new();
        let mut scalar_cursor = 0;
        for page in ordered_pages {
            let mut ordered_lines = page.lines.iter().collect::<Vec<_>>();
            ordered_lines.sort_by(|left, right| {
                left.reading_order
                    .cmp(&right.reading_order)
                    .then_with(|| left.id.cmp(&right.id))
            });
            let mut page_start = None;
            let mut page_end = scalar_cursor;
            for line in ordered_lines {
                if !lines.is_empty() {
                    text.push('\n');
                    scalar_cursor += 1;
                }
                let start = scalar_cursor;
                text.push_str(&line.text);
                scalar_cursor += line.text.chars().count();
                let range = ScalarRange {
                    start,
                    end: scalar_cursor,
                };
                page_start.get_or_insert(start);
                page_end = range.end;
                let slot = lines.len();
                line_slots.insert(line.id.clone(), slot);
                lines.push(IndexedPdfLine {
                    page_id: page.id.clone(),
                    page_index: page.index,
                    line_id: line.id.clone(),
                    range,
                });
            }
            page_ranges.insert(
                page.index,
                ScalarRange {
                    start: page_start.unwrap_or(scalar_cursor),
                    end: page_end,
                },
            );
        }
        Self {
            text,
            lines,
            line_slots,
            page_ranges,
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn lines(&self) -> &[IndexedPdfLine] {
        &self.lines
    }

    pub fn line(&self, line_id: &str) -> Option<&IndexedPdfLine> {
        self.line_slots
            .get(line_id)
            .and_then(|slot| self.lines.get(*slot))
    }

    fn line_at(&self, at: usize) -> Option<&IndexedPdfLine> {
        let slot = self
            .lines
            .partition_point(|line| line.range.start <= at)
            .checked_sub(1)?;
        self.lines.get(slot).filter(|line| {
            (line.range.start <= at && at < line.range.end)
                || (line.range.start == line.range.end && line.range.start == at)
        })
    }

    fn overlapping_lines(&self, range: ScalarRange) -> &[IndexedPdfLine] {
        let start = self
            .lines
            .partition_point(|line| line.range.end <= range.start);
        let end = self
            .lines
            .partition_point(|line| line.range.start < range.end);
        &self.lines[start.min(end)..end]
    }

    pub fn page_range(&self, page_index: usize) -> Option<ScalarRange> {
        self.page_ranges.get(&page_index).copied()
    }

    pub fn global_range(&self, line_id: &str, start: usize, end: usize) -> Option<ScalarRange> {
        let line = self.line(line_id)?;
        let length = line.range.end - line.range.start;
        (start <= end && end <= length).then_some(ScalarRange {
            start: line.range.start + start,
            end: line.range.start + end,
        })
    }

    pub fn line_ids(&self, range: ScalarRange) -> Vec<String> {
        self.overlapping_lines(range)
            .iter()
            .map(|line| line.line_id.clone())
            .collect()
    }

    pub fn page_indexes(&self, range: ScalarRange) -> Vec<usize> {
        self.overlapping_lines(range)
            .iter()
            .fold(Vec::new(), |mut pages, line| {
                if !pages.contains(&line.page_index) {
                    pages.push(line.page_index);
                }
                pages
            })
    }

    fn range_for_line_ids<'a>(
        &self,
        line_ids: impl IntoIterator<Item = &'a String>,
    ) -> Option<ScalarRange> {
        line_ids
            .into_iter()
            .filter_map(|line_id| self.line(line_id))
            .fold(None, |range, line| {
                Some(range.map_or(line.range, |range: ScalarRange| ScalarRange {
                    start: range.start.min(line.range.start),
                    end: range.end.max(line.range.end),
                }))
            })
    }

    fn page_indexes_for_line_ids<'a>(
        &self,
        line_ids: impl IntoIterator<Item = &'a String>,
    ) -> Vec<usize> {
        line_ids.into_iter().fold(Vec::new(), |mut pages, line_id| {
            if let Some(page_index) = self.line(line_id).map(|line| line.page_index) {
                if !pages.contains(&page_index) {
                    pages.push(page_index);
                }
            }
            pages
        })
    }
}

pub struct StructureOutput {
    pub paragraphs: Vec<Paragraph>,
    pub footnotes: Vec<Footnote>,
    pub diagnostics: Vec<Diagnostic>,
    pub pairing_audit: PdfPairingAudit,
    pub pdf_source_map: PdfSourceMap,
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
        .map(|character| match character {
            '⁰' => '0',
            '¹' => '1',
            '²' => '2',
            '³' => '3',
            '⁴' => '4',
            '⁵' => '5',
            '⁶' => '6',
            '⁷' => '7',
            '⁸' => '8',
            '⁹' => '9',
            '∗' | '\u{f02a}' => '*',
            other => other,
        })
        .collect();
    translated
        .parse::<u64>()
        .map_or(translated, |number| number.to_string())
}

fn char_to_byte(value: &str, character_offset: usize) -> usize {
    value
        .char_indices()
        .nth(character_offset)
        .map_or(value.len(), |(index, _)| index)
}

fn char_slice(value: &str, start: usize, end: usize) -> &str {
    let start = char_to_byte(value, start.min(value.chars().count()));
    let end = char_to_byte(value, end.min(value.chars().count()));
    &value[start..end]
}

fn is_note_symbol(character: char) -> bool {
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
    let remainder = char_slice(trimmed, token_chars, trimmed.chars().count());
    let embedded_endnote = format!("endnote {token}");
    let embedded_chars = embedded_endnote.chars().count();
    if remainder
        .to_ascii_lowercase()
        .starts_with(&embedded_endnote.to_ascii_lowercase())
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

fn label_prefix(text: &str) -> Option<LabelPrefix> {
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

fn median(mut values: Vec<f64>) -> f64 {
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
    let spans: Vec<&Span> = line
        .spans
        .iter()
        .filter(|span| span.start < prefix.end && span.end > prefix.start)
        .collect();
    let label_size = spans
        .iter()
        .filter_map(|span| (span.size > 0.0).then_some(span.size))
        .min_by(f64::total_cmp)
        .unwrap_or(line_size);
    let height = line.bbox[3] - line.bbox[1];
    spans.iter().any(|span| {
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
    let digits = DIGITS.get_or_init(|| Regex::new(r"\d+").unwrap());
    if digits.is_match(&normalized) {
        digits.replace_all(&normalized, "#").into_owned()
    } else {
        normalized
    }
}

fn compact_note_line(text: &str) -> bool {
    let Some(prefix) = line_start_label_prefix(text) else {
        return false;
    };
    char_slice(text, prefix.end, text.chars().count())
        .trim_start_matches(|character: char| {
            character.is_whitespace() || ".)],:;-".contains(character)
        })
        .chars()
        .filter(|character| character.is_alphabetic())
        .take(4)
        .count()
        == 4
}

fn standalone_enumerator(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^\s*(?:[IVXLCDM]{1,7}|[A-Za-z]|\d{1,3}|\d{1,2}(?:\.\d{1,2}){1,3})[.)]\s*$")
            .unwrap()
    })
    .is_match(text)
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
        let candidates = page
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
            })
            .collect::<Vec<_>>();
        if candidates.len() == 1 {
            singletons.push(candidates[0]);
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
    let normalized_lines: Vec<Vec<String>> =
        legal_pdf_support::profile::measure("furniture.normalize", || {
            pages
                .iter()
                .map(|page| {
                    page.lines
                        .iter()
                        .map(|line| normalize_furniture(&line.text))
                        .collect()
                })
                .collect()
        });
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
                let normalized = &normalized_lines[page_slot][line_slot];
                if let Some(top) = edge.filter(|_| !normalized.is_empty()) {
                    candidates
                        .entry((top, normalized.clone()))
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
                let normalized = &normalized_lines[page_slot][index];
                let line_size = line_sizes[index];
                let at_top = line.bbox[3] < page.height * 0.12;
                let at_bottom = line.bbox[1] > page.height * 0.90;
                let page_number_at_top = line.bbox[3] < page.height * 0.14;
                let plausible_number = normalized != "#" || line_size >= body_size * 0.75;
                let compact_note =
                    at_bottom && line_size < body_size * 0.90 && compact_note_line(&line.text);
                if sequence_folios.contains(&(page_slot, index)) {
                    line.region_type = "footer".to_owned();
                } else if repeated.contains(&(page_slot, index))
                    && (at_top || at_bottom)
                    && plausible_number
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
            page.lines[host_index].detached_references.push(json!({
                "note_id": label,
                "selected_text": selected_text,
                "start_offset": offset,
                "end_offset": offset,
                "source_line_id": source_line_id,
            }));
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
        page.lines[host_index].detached_references.push(json!({
            "note_id": value,
            "selected_text": selected_text,
            "start_offset": offset,
            "end_offset": offset,
            "source_line_id": source_line_id,
        }));
        page.lines[marker_index].exclude_from_body = true;
    }
}

fn all_caps(text: &str) -> bool {
    let letters: Vec<char> = text
        .chars()
        .filter(|character| character.is_alphabetic())
        .collect();
    !letters.is_empty() && letters.iter().all(|character| character.is_uppercase())
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

fn heading_candidates(pages: &[Page], regions: &HashMap<String, String>) -> Vec<Value> {
    static INLINE: OnceLock<Regex> = OnceLock::new();
    static STANDALONE: OnceLock<Regex> = OnceLock::new();
    let inline = INLINE.get_or_init(|| {
        Regex::new(
            r"^\s*([IVXLCDM]{1,7}|[A-Za-z]|\d{1,3}|\d{1,2}(?:\.\d{1,2}){1,3})([.)])\s+(\S.*)$",
        )
        .unwrap()
    });
    let standalone = STANDALONE.get_or_init(|| {
        Regex::new(r"^\s*([IVXLCDM]{1,7}|[A-Za-z]|\d{1,3}|\d{1,2}(?:\.\d{1,2}){1,3})([.)])\s*$")
            .unwrap()
    });
    let mut candidates = Vec::new();
    for (page_slot, page) in pages.iter().enumerate() {
        if contents_grid(&page.lines, page.width) {
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
                    candidates.push(json!({
                        "page_slot": page_slot,
                        "line_slot": line_slot,
                        "kind": "enumerator",
                        "joined": false,
                        "value_text": value,
                        "punct": punct,
                        "text": text,
                        "interpretations": interpretations,
                    }));
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
                candidates.push(json!({
                    "page_slot": page_slot,
                    "line_slot": line_slot,
                    "joined_line_slot": follower_slot,
                    "kind": "enumerator",
                    "joined": true,
                    "value_text": value,
                    "punct": punct,
                    "text": text,
                    "interpretations": interpretations,
                }));
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
        if span.flags & 16 != 0 || span.font.to_ascii_lowercase().contains("bold") {
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

fn coherent_heading_family(family: &Value) -> bool {
    family.get("count").and_then(Value::as_u64).unwrap_or(0) >= 2
        && family
            .get("violations")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            == 0
        && family
            .get("level_votes")
            .and_then(Value::as_object)
            .is_some_and(|votes| votes.len() == 1)
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
    let words = text
        .split_whitespace()
        .filter(|word| word.chars().any(char::is_alphabetic))
        .collect::<Vec<_>>();
    if words.is_empty() {
        0.0
    } else {
        words
            .iter()
            .filter(|word| word.chars().next().is_some_and(char::is_uppercase))
            .count() as f64
            / words.len() as f64
    }
}

fn heading_style_corroborated(text: &str) -> bool {
    static INLINE: OnceLock<Regex> = OnceLock::new();
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
    if let Some(capture) = INLINE
        .get_or_init(|| {
            Regex::new(
                r"^\s*(?:[IVXLCDM]{1,7}|[A-Za-z]|\d{1,3}|\d{1,2}(?:\.\d{1,2}){1,3})[.)]\s+(\S.*)$",
            )
            .unwrap()
        })
        .captures(text)
    {
        if heading_text_plausible(capture.get(1).unwrap().as_str()) && !text.ends_with('.') {
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
    let letters = text
        .chars()
        .filter(|character| character.is_alphabetic())
        .collect::<Vec<_>>();
    !letters.is_empty()
        && letters.iter().any(|character| character.is_uppercase())
        && letters.iter().any(|character| character.is_lowercase())
        && letters
            .iter()
            .filter(|character| character.is_uppercase())
            .count() as f64
            / letters.len() as f64
            >= 0.65
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
    parsed: &Value,
    source_regions: &HashMap<String, String>,
) {
    let assignments = parsed
        .get("assignments")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let families = parsed.get("families").and_then(Value::as_object);
    let mut by_line = HashMap::<(usize, usize), Value>::new();
    for assignment in assignments {
        let Some(page) = assignment.get("page_slot").and_then(Value::as_u64) else {
            continue;
        };
        for key in ["line_slot", "joined_line_slot"] {
            if let Some(line) = assignment.get(key).and_then(Value::as_u64) {
                by_line.insert((page as usize, line as usize), assignment.clone());
            }
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
            let assignment = by_line.get(&(page_slot, line_slot));
            let family = assignment
                .and_then(|assignment| assignment.get("family"))
                .and_then(Value::as_str)
                .and_then(|family| families.and_then(|families| families.get(family)));
            if family.is_some_and(coherent_heading_family) {
                continue;
            }
            let style = heading_style_corroborated(text);
            let grammar_negative = assignment.map_or(!style, |assignment| {
                !style
                    || matches!(
                        assignment.get("action").and_then(Value::as_str),
                        Some("illegal_restart" | "violation")
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
    source_regions: Option<&HashMap<String, String>>,
) -> HashMap<String, usize> {
    let Some(source_regions) = source_regions else {
        return HashMap::new();
    };
    static TOC_LEADER: OnceLock<Regex> = OnceLock::new();
    let toc_leader =
        TOC_LEADER.get_or_init(|| Regex::new(r"(?:\. ){3,}|\.{4,}").expect("TOC leader regex"));
    for page in pages.iter_mut() {
        if contents_grid(&page.lines, page.width)
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

    let candidates = heading_candidates(pages, source_regions);
    let parsed = parse_heading_ladder(&candidates);
    demote_false_headings(pages, &parsed, source_regions);
    let ladder_clean = parsed.get("status").and_then(Value::as_str) == Some("parsed_clean");
    let families = parsed.get("families").and_then(Value::as_object);
    let assignments = parsed
        .get("assignments")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut heading_levels = assignments
        .iter()
        .flat_map(|assignment| {
            let page = assignment
                .get("page_slot")
                .and_then(Value::as_u64)
                .map(|value| value as usize);
            let marker = assignment
                .get("line_slot")
                .and_then(Value::as_u64)
                .map(|value| value as usize);
            let joined = assignment
                .get("joined_line_slot")
                .and_then(Value::as_u64)
                .map(|value| value as usize);
            let level = assignment
                .get("level")
                .and_then(Value::as_u64)
                .map(|value| value as usize);
            [marker, joined]
                .into_iter()
                .flatten()
                .filter_map(move |line| page.zip(level).map(|(page, level)| (page, line, level)))
        })
        .filter_map(|(page, line, level)| {
            pages
                .get(page)
                .and_then(|page| page.lines.get(line))
                .map(|line| (line.id.clone(), level))
        })
        .collect::<HashMap<_, _>>();
    let structural: HashSet<(usize, usize)> = assignments
        .iter()
        .flat_map(|assignment| {
            let page = assignment.get("page_slot").and_then(Value::as_u64);
            let line = assignment.get("line_slot").and_then(Value::as_u64);
            let joined = assignment.get("joined_line_slot").and_then(Value::as_u64);
            [
                page.zip(line)
                    .map(|(page, line)| (page as usize, line as usize)),
                page.zip(joined)
                    .map(|(page, line)| (page as usize, line as usize)),
            ]
            .into_iter()
            .flatten()
        })
        .collect();

    for assignment in assignments {
        let Some(page_slot) = assignment
            .get("page_slot")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
        else {
            continue;
        };
        let Some(marker_slot) = assignment
            .get("line_slot")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
        else {
            continue;
        };
        let target_slot = assignment
            .get("joined_line_slot")
            .and_then(Value::as_u64)
            .map_or(marker_slot, |value| value as usize);
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
            || !heading_text_plausible(assignment.get("text").and_then(Value::as_str).unwrap_or(""))
            || matches!(
                assignment.get("action").and_then(Value::as_str),
                Some("illegal_restart" | "violation")
            )
        {
            continue;
        }
        let family_name = assignment
            .get("family")
            .and_then(Value::as_str)
            .unwrap_or("");
        let family = families.and_then(|families| families.get(family_name));
        if family.is_some_and(|family| {
            family
                .get("footnote_suspect")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        }) || assignment.get("level").and_then(Value::as_u64).unwrap_or(0) == 0
        {
            continue;
        }
        let page = &pages[page_slot];
        if has_body_flow(page, target_slot, &structural, page_slot) {
            continue;
        }
        let target = &page.lines[target_slot];
        let visual = bold_char_share(target) >= 0.60
            || (body_size > 0.0 && line_font_size(target) >= body_size * 1.02);
        if !visual && !family.is_some_and(coherent_heading_family) {
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
            if let Some(level) = assignment.get("level").and_then(Value::as_u64) {
                heading_levels.insert(
                    pages[page_slot].lines[continuation_slot].id.clone(),
                    level as usize,
                );
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
    let tail = char_slice(&line.text, candidate.1.end, line.text.chars().count());
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
    page.lines.iter().any(|line| {
        line.bbox[1] < label_y
            && (line
                .spans
                .iter()
                .any(|span| span.superscript && normalize_label(span.text.trim()) == label)
                || line.detached_references.iter().any(|reference| {
                    normalize_label(reference["note_id"].as_str().unwrap_or_default()) == label
                }))
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
            let evidence = table_evidence(&page.lines, page.width);
            let caption = has_table_caption(&page.lines);
            let is_table = caption
                || strong_table_evidence(&evidence, &page.lines)
                || continuation && evidence.continuation_on_page(page.height);
            continuing_table = is_table
                && evidence.reaches_page_bottom(page.height)
                && (caption || evidence.continuation());
            let mut cells = if is_table {
                evidence.expanded_lines(&page.lines, page.height, continuation, separator)
            } else {
                HashSet::new()
            };
            let notes = evidence.table_note_lines(&page.lines, &cells);
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
            let comma_tail =
                char_slice(&line.text, prefix.end, prefix.end + 1) == "," && !typographic;
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
        let content_before: Vec<usize> = page
            .lines
            .iter()
            .enumerate()
            .filter_map(|(index, line)| {
                (!matches!(line.region_type.as_str(), "header" | "footer")
                    && line.bbox[1] < minimum_label_y)
                    .then_some(index)
            })
            .collect();
        let first_content = content_before
            .iter()
            .copied()
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
            || content_before.iter().any(|index| {
                let line = &page.lines[*index];
                (line.region_type == "heading" || line.bbox[1] >= page.height * 0.08)
                    && structural_reset_heading(&line.text)
            });
        let label_sizes: Vec<f64> = candidates
            .iter()
            .map(|(index, _, _)| line_sizes[*index])
            .filter(|size| *size > 0.0)
            .collect();
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
        let continuation_size_matches = continuing_size.is_none_or(|size| {
            content_sizes.is_empty() || median(content_sizes.clone()) <= size * 1.15
        });
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
        let mut margin_runs = vec![1; margin_numeric.len()];
        for index in 0..margin_numeric.len() {
            for prior in 0..index {
                if (1..=3).contains(&margin_numeric[index].saturating_sub(margin_numeric[prior])) {
                    margin_runs[index] = margin_runs[index].max(margin_runs[prior] + 1);
                }
            }
        }
        let supported_margin_candidates: Vec<_> = candidates
            .iter()
            .filter(|(index, prefix, _)| {
                margin_candidates.contains(index)
                    && has_prior_reference(page, &prefix.label, page.lines[*index].bbox[1])
            })
            .map(|(index, _, _)| *index)
            .collect();
        let margin_labels = if margin_runs.into_iter().max().unwrap_or(1) >= 3 {
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
                && !label_sizes.is_empty()
                && median(label_sizes.clone()) <= body_size * 0.90);
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
        let separator_starts_small_text = separator.is_some_and(|cut| {
            if contents_grid(&page.lines, page.width) {
                return false;
            }
            page.lines
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
                            && !selected_sizes.is_empty()
                            && median(selected_sizes.clone()) <= body_size * 0.90))
            }) {
                separator
            } else {
                let confident = (first_label >= page.height * 0.58 || endnote_page)
                    && !selected_sizes.is_empty()
                    && median(selected_sizes.clone()) <= body_size * 0.90;
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
        let endnote_lines: Vec<&Line> = page
            .lines
            .iter()
            .filter(|line| line.note_region_mode == "endnote")
            .collect();
        if endnote_lines.is_empty() {
            expected_endnote = None;
            continuing_size = None;
        } else {
            if let Some(last) = endnote_lines
                .iter()
                .filter_map(|line| label_prefix(&line.text)?.label.parse::<u32>().ok())
                .next_back()
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
    evidence.heading_levels =
        apply_text_fidelity_headings(pages, article_body_size, evidence.source_regions.as_ref());
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OrderRepair {
    None,
    Column,
    Geometry,
}

#[derive(Debug, Clone, Copy)]
struct ColumnModel {
    kind: &'static str,
    split_x: f64,
    left_count: usize,
    right_count: usize,
}

#[derive(Debug, Clone, Copy)]
struct OrderDecision {
    repair: OrderRepair,
    source_switches: usize,
    strategy: &'static str,
    reason: &'static str,
}

impl OrderDecision {
    const fn keep(reason: &'static str) -> Self {
        Self {
            repair: OrderRepair::None,
            source_switches: 0,
            strategy: "kraken-native",
            reason,
        }
    }
}

fn line_width(line: &Line) -> f64 {
    line.bbox[2] - line.bbox[0]
}

fn has_valid_bbox(line: &Line) -> bool {
    line.bbox.iter().all(|value| value.is_finite())
        && line.bbox[2] > line.bbox[0]
        && line.bbox[3] > line.bbox[1]
}

fn line_center_x(line: &Line) -> f64 {
    (line.bbox[0] + line.bbox[2]) / 2.0
}

fn line_center_y(line: &Line) -> f64 {
    (line.bbox[1] + line.bbox[3]) / 2.0
}

fn p50(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[derive(Default)]
struct TableEvidence {
    lines: HashSet<usize>,
    rows: usize,
    columns: usize,
    numeric_cells: usize,
    cells: usize,
    top: f64,
    bottom: f64,
    left: f64,
    right: f64,
    line_height: f64,
}

impl TableEvidence {
    fn strong(&self) -> bool {
        self.rows >= 6 && self.columns >= 3 && self.numeric_cells * 5 >= self.cells
    }

    fn continuation(&self) -> bool {
        self.rows >= 6 && self.columns >= 2 && self.numeric_cells * 10 >= self.cells
    }

    fn reaches_page_bottom(&self, page_height: f64) -> bool {
        page_height > 0.0 && self.bottom >= page_height * 0.70
    }

    fn continuation_on_page(&self, page_height: f64) -> bool {
        page_height > 0.0 && self.top <= page_height * 0.30 && self.continuation()
    }

    fn expanded_lines(
        &self,
        lines: &[Line],
        page_height: f64,
        continuation: bool,
        separator: Option<f64>,
    ) -> HashSet<usize> {
        if self.lines.is_empty() {
            return HashSet::new();
        }
        let top = if continuation {
            lines
                .iter()
                .filter(|line| {
                    has_valid_bbox(line)
                        && !line.exclude_from_body
                        && line.bbox[1] >= page_height * 0.08
                        && line.bbox[1] <= self.top
                })
                .map(|line| line.bbox[1])
                .min_by(f64::total_cmp)
                .unwrap_or(self.top)
        } else {
            self.top
        };
        let table_size = p50(self
            .lines
            .iter()
            .map(|index| line_font_size(&lines[*index]))
            .filter(|size| *size > 0.0)
            .collect());
        let bottom = separator
            .filter(|cut| {
                *cut > top
                    && lines
                        .iter()
                        .filter(|line| has_valid_bbox(line) && line.bbox[1] >= *cut)
                        .min_by(|left, right| left.bbox[1].total_cmp(&right.bbox[1]))
                        .is_some_and(|next| {
                            let gap = next.bbox[1] - cut;
                            gap >= self.line_height * 1.5
                                && (gap >= self.line_height * 4.0
                                    || (table_size > 0.0
                                        && line_font_size(next) <= table_size * 0.90))
                        })
            })
            .unwrap_or(self.bottom);
        lines
            .iter()
            .enumerate()
            .filter_map(|(index, line)| {
                (has_valid_bbox(line)
                    && !line.exclude_from_body
                    && line.bbox[3] >= top - self.line_height
                    && line.bbox[1] <= bottom
                    && line.bbox[2] >= self.left
                    && line.bbox[0] <= self.right)
                    .then_some(index)
            })
            .collect()
    }

    fn table_note_lines(&self, lines: &[Line], cells: &HashSet<usize>) -> HashSet<usize> {
        let anchors: Vec<_> = lines
            .iter()
            .enumerate()
            .filter_map(|(index, line)| {
                let prefix = label_prefix(&line.text)?;
                let symbolic = prefix
                    .label
                    .chars()
                    .all(|character| !character.is_ascii_digit());
                let tail = char_slice(&line.text, prefix.end, line.text.chars().count());
                (symbolic
                    && tail
                        .chars()
                        .filter(|character| character.is_alphabetic())
                        .count()
                        >= 4
                    && ((!self.lines.contains(&index) && cells.contains(&index))
                        || (line.bbox[1] > self.bottom
                            && line.bbox[1] <= self.bottom + self.line_height * 3.0)))
                    .then_some(index)
            })
            .collect();
        let mut notes: HashSet<_> = anchors.iter().copied().collect();
        for anchor in anchors {
            let mut bottom = lines[anchor].bbox[3];
            let mut following: Vec<_> = lines
                .iter()
                .enumerate()
                .filter(|(index, line)| *index != anchor && line.bbox[1] >= lines[anchor].bbox[1])
                .collect();
            following.sort_by(|(_, left), (_, right)| left.bbox[1].total_cmp(&right.bbox[1]));
            for (index, line) in following {
                if line.bbox[1] - bottom > self.line_height * 1.6
                    || cells.contains(&index)
                    || has_table_caption(std::slice::from_ref(line))
                {
                    break;
                }
                notes.insert(index);
                bottom = bottom.max(line.bbox[3]);
            }
        }
        notes
    }
}

fn strong_table_evidence(evidence: &TableEvidence, lines: &[Line]) -> bool {
    let mut prior = None;
    let mut run = 0;
    let mut longest = 0;
    for line in lines {
        let Some(prefix) = label_prefix(&line.text) else {
            continue;
        };
        let Ok(label) = prefix.label.parse::<u32>() else {
            continue;
        };
        if !char_slice(&line.text, prefix.end, line.text.chars().count())
            .chars()
            .any(char::is_alphabetic)
        {
            continue;
        }
        run = if prior.is_some_and(|value| (1..=3).contains(&label.saturating_sub(value))) {
            run + 1
        } else {
            1
        };
        longest = longest.max(run);
        prior = Some(label);
    }
    evidence.strong()
        && longest < 3
        && lines
            .iter()
            .enumerate()
            .filter(|(index, line)| {
                standalone_note_label(line) && aligned_note_body_index(lines, *index).is_some()
            })
            .take(3)
            .count()
            < 3
}

fn has_table_caption(lines: &[Line]) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    let regex =
        RE.get_or_init(|| Regex::new(r"(?i)^(?:table|tableau)\s+(?:\d+|[ivxlcdm]+)\b").unwrap());
    lines.iter().any(|line| regex.is_match(line.text.trim()))
}

fn contents_grid(lines: &[Line], page_width: f64) -> bool {
    if page_width <= 0.0 {
        return false;
    }
    let mut locators: Vec<_> = lines
        .iter()
        .filter_map(|line| {
            let value = line.text.trim().parse::<u32>().ok()?;
            (has_valid_bbox(line) && line.bbox[0] >= page_width * 0.72).then_some((line, value))
        })
        .filter(|(locator, _)| {
            lines.iter().any(|peer| {
                let overlap = locator.bbox[3].min(peer.bbox[3]) - locator.bbox[1].max(peer.bbox[1]);
                let height = (locator.bbox[3] - locator.bbox[1]).min(peer.bbox[3] - peer.bbox[1]);
                peer.bbox[0] < page_width * 0.68
                    && peer.text.chars().any(char::is_alphabetic)
                    && height > 0.0
                    && overlap / height >= 0.5
            })
        })
        .collect();
    if locators.len() < 6 {
        return false;
    }
    locators.sort_by(|(left, _), (right, _)| band_geometry_order(left, right));
    let monotone = locators
        .windows(2)
        .filter(|pair| pair[0].1 <= pair[1].1)
        .count();
    monotone * 5 >= (locators.len() - 1) * 4
}

fn table_evidence(lines: &[Line], page_width: f64) -> TableEvidence {
    let height = p50(lines
        .iter()
        .filter(|line| has_valid_bbox(line))
        .map(|line| line.bbox[3] - line.bbox[1])
        .filter(|value| *value > 0.0)
        .collect());
    if height <= 0.0 {
        return TableEvidence::default();
    }
    let caption = lines
        .iter()
        .filter(|line| has_table_caption(std::slice::from_ref(line)))
        .min_by(|left, right| left.bbox[1].total_cmp(&right.bbox[1]));
    let caption_bottom = caption.map(|line| line.bbox[3]);
    let mut rows: HashMap<i64, Vec<usize>> = HashMap::new();
    for (index, line) in lines.iter().enumerate().filter(|(_, line)| {
        has_valid_bbox(line)
            && !line.exclude_from_body
            && !matches!(line.region_type.as_str(), "header" | "footer")
    }) {
        rows.entry((line_center_y(line) / (height * 0.75)).round() as i64)
            .or_default()
            .push(index);
    }
    let mut dense_rows: Vec<_> = rows
        .into_values()
        .filter(|row| {
            row.len() >= 2
                && p50(row
                    .iter()
                    .map(|index| lines[*index].text.trim().chars().count() as f64)
                    .collect())
                    <= 24.0
        })
        .collect();
    dense_rows.sort_by(|left, right| {
        let center = |row: &[usize]| {
            row.iter()
                .map(|index| line_center_y(&lines[*index]))
                .min_by(f64::total_cmp)
                .unwrap_or(0.0)
        };
        center(left).total_cmp(&center(right))
    });
    if let Some(bottom) = caption_bottom {
        dense_rows.retain(|row| {
            row.iter()
                .map(|index| line_center_y(&lines[*index]))
                .min_by(f64::total_cmp)
                .is_some_and(|center| center >= bottom - height)
        });
    }
    if !contents_grid(lines, page_width) {
        let mut prior = None;
        dense_rows = dense_rows
            .into_iter()
            .take_while(|row| {
                let center = row
                    .iter()
                    .map(|index| line_center_y(&lines[*index]))
                    .min_by(f64::total_cmp)
                    .unwrap_or(0.0);
                let connected = prior.is_none_or(|value| center - value <= height * 6.0);
                prior = Some(center);
                connected
            })
            .collect();
    }
    if dense_rows.len() < 3 {
        return TableEvidence::default();
    }
    let mut columns: HashMap<i64, usize> = HashMap::new();
    for row in &dense_rows {
        let mut seen = HashSet::new();
        for index in row {
            seen.insert((lines[*index].bbox[0] / (height * 2.0)).round() as i64);
        }
        for column in seen {
            *columns.entry(column).or_default() += 1;
        }
    }
    let row_count = dense_rows.len();
    let column_count = columns.values().filter(|count| **count >= 3).count();
    let compact: Vec<_> = dense_rows
        .iter()
        .flatten()
        .map(|index| lines[*index].text.trim())
        .collect();
    let numeric_cells = compact
        .iter()
        .filter(|text| {
            text.chars().any(|character| character.is_ascii_digit())
                && text.chars().all(|character| {
                    character.is_ascii_digit()
                        || character.is_whitespace()
                        || matches!(
                            character,
                            '.' | ',' | '%' | '(' | ')' | '/' | '$' | '-' | '\u{2013}' | '\u{2014}'
                        )
                })
        })
        .count();
    let dense_lines: HashSet<_> = dense_rows.into_iter().flatten().collect();
    TableEvidence {
        top: dense_lines
            .iter()
            .map(|index| lines[*index].bbox[1])
            .min_by(f64::total_cmp)
            .unwrap_or(0.0),
        bottom: dense_lines
            .iter()
            .map(|index| lines[*index].bbox[3])
            .max_by(f64::total_cmp)
            .unwrap_or(0.0),
        left: dense_lines
            .iter()
            .map(|index| lines[*index].bbox[0])
            .min_by(f64::total_cmp)
            .unwrap_or(0.0),
        right: dense_lines
            .iter()
            .map(|index| lines[*index].bbox[2])
            .max_by(f64::total_cmp)
            .unwrap_or(0.0),
        line_height: height,
        lines: dense_lines,
        rows: row_count,
        columns: column_count,
        numeric_cells,
        cells: compact.len(),
    }
}

fn column_model(lines: &[Line], page_width: f64) -> ColumnModel {
    column_model_with_furniture(lines, page_width, false)
}

fn margin_note_model(
    lines: &[Line],
    labels: &[usize],
    page_width: f64,
    body_size: f64,
    minimum_labels: usize,
) -> Option<ColumnModel> {
    if labels.is_empty() || page_width <= 0.0 || body_size <= 0.0 {
        return None;
    }
    [false, true]
        .into_iter()
        .filter_map(|right| {
            let lane: Vec<_> = labels
                .iter()
                .copied()
                .filter(|index| (line_center_x(&lines[*index]) >= page_width / 2.0) == right)
                .collect();
            if lane.len() < minimum_labels {
                return None;
            }
            let label_set: HashSet<_> = lane.iter().copied().collect();
            let mut note_left = page_width;
            let mut note_right: f64 = 0.0;
            let mut note_top = f64::INFINITY;
            let mut note_bottom: f64 = 0.0;
            for index in &lane {
                let label = &lines[*index];
                note_left = note_left.min(label.bbox[0]);
                note_right = note_right.max(label.bbox[2]);
                note_top = note_top.min(label.bbox[1]);
                note_bottom = note_bottom.max(label.bbox[3]);
                if standalone_note_label(label) {
                    if let Some(body) = aligned_note_body_index(lines, *index) {
                        note_left = note_left.min(lines[body].bbox[0]);
                        note_right = note_right.max(lines[body].bbox[2]);
                        note_top = note_top.min(lines[body].bbox[1]);
                        note_bottom = note_bottom.max(lines[body].bbox[3]);
                    }
                }
            }
            let prose: Vec<_> = lines
                .iter()
                .enumerate()
                .filter(|(index, line)| {
                    !label_set.contains(index)
                        && has_valid_bbox(line)
                        && !line.exclude_from_body
                        && !matches!(line.region_type.as_str(), "header" | "footer")
                        && line_font_size(line) >= body_size * 0.90
                        && line_width(line) >= page_width * 0.25
                        && line.bbox[3] >= note_top
                        && line.bbox[1] <= note_bottom
                })
                .map(|(_, line)| line)
                .collect();
            if prose.len() < 3 {
                return None;
            }
            let body_left = prose
                .iter()
                .map(|line| line.bbox[0])
                .min_by(f64::total_cmp)?;
            let body_right = prose
                .iter()
                .map(|line| line.bbox[2])
                .max_by(f64::total_cmp)?;
            let gap = (body_size * 1.5).max(page_width * 0.02);
            let split_x = if note_right + gap <= body_left {
                (note_right + body_left) / 2.0
            } else if body_right + gap <= note_left {
                (body_right + note_left) / 2.0
            } else {
                return None;
            };
            Some((
                lane.len(),
                ColumnModel {
                    kind: "margin_column",
                    split_x,
                    left_count: if right { prose.len() } else { lane.len() },
                    right_count: if right { lane.len() } else { prose.len() },
                },
            ))
        })
        .max_by_key(|(count, _)| *count)
        .map(|(_, model)| model)
}

fn column_model_with_furniture(
    lines: &[Line],
    page_width: f64,
    ignore_centered_furniture: bool,
) -> ColumnModel {
    let single = ColumnModel {
        kind: "single",
        split_x: 0.0,
        left_count: 0,
        right_count: 0,
    };
    if page_width <= 0.0 || lines.is_empty() {
        return single;
    }
    let boxed: Vec<&Line> = lines
        .iter()
        .filter(|line| {
            has_valid_bbox(line)
                && !line.exclude_from_body
                && !matches!(line.region_type.as_str(), "header" | "footer")
        })
        .collect();
    let centered_band = |line: &Line| {
        let width_ratio = line_width(line) / page_width;
        ignore_centered_furniture
            && width_ratio <= 0.30
            && (line_center_x(line) / page_width - 0.5).abs() <= 0.12
    };
    let inference_lines: Vec<&Line> = boxed
        .iter()
        .copied()
        .filter(|line| !centered_band(line))
        .collect();
    let candidates: Vec<&Line> = inference_lines
        .iter()
        .copied()
        .filter(|line| line_width(line) / page_width <= 0.55)
        .collect();
    if candidates.len() < 6
        || (!inference_lines.is_empty()
            && 1.0 - candidates.len() as f64 / inference_lines.len() as f64 > 0.40)
    {
        return single;
    }
    let mut centers: Vec<f64> = candidates.iter().map(|line| line_center_x(line)).collect();
    centers.sort_by(f64::total_cmp);
    centers
        .windows(2)
        .filter_map(|pair| {
            let left = pair[0];
            let right = pair[1];
            let center_gap = pair[1] - pair[0];
            let initial_split = (left + right) / 2.0;
            if center_gap / page_width < 0.12
                || !(0.25..=0.75).contains(&(initial_split / page_width))
            {
                return None;
            }
            let (left_lines, right_lines): (Vec<_>, Vec<_>) = candidates
                .iter()
                .copied()
                .partition(|line| line_center_x(line) < initial_split);
            if left_lines.len() < 3 || right_lines.len() < 3 {
                return None;
            }
            let (split_x, gap, imbalance) = if ignore_centered_furniture {
                let left_edge = left_lines
                    .iter()
                    .map(|line| line.bbox[2])
                    .max_by(f64::total_cmp)
                    .unwrap_or(initial_split);
                let right_edge = right_lines
                    .iter()
                    .map(|line| line.bbox[0])
                    .min_by(f64::total_cmp)
                    .unwrap_or(initial_split);
                if right_edge <= left_edge {
                    return None;
                }
                (
                    (left_edge + right_edge) / 2.0,
                    right_edge - left_edge,
                    left_lines.len().abs_diff(right_lines.len()),
                )
            } else {
                (initial_split, center_gap, 0)
            };
            let split_ratio = (split_x / page_width * 10_000.0).round_ties_even() / 10_000.0;
            let vertical_extent = |values: &[&Line]| {
                (
                    values
                        .iter()
                        .map(|line| line.bbox[1])
                        .min_by(f64::total_cmp)
                        .unwrap_or(0.0),
                    values
                        .iter()
                        .map(|line| line.bbox[3])
                        .max_by(f64::total_cmp)
                        .unwrap_or(0.0),
                )
            };
            let left_y = vertical_extent(&left_lines);
            let right_y = vertical_extent(&right_lines);
            let span = left_y.1.max(right_y.1) - left_y.0.min(right_y.0);
            let overlap = left_y.1.min(right_y.1) - left_y.0.max(right_y.0);
            if span <= 0.0 || overlap.max(0.0) / span < 0.30 {
                return None;
            }
            let crossings = candidates
                .iter()
                .filter(|line| line.bbox[0] < split_x && line.bbox[2] > split_x)
                .count();
            let left_width = p50(left_lines
                .iter()
                .map(|line| line_width(line) / page_width)
                .collect());
            let right_width = p50(right_lines
                .iter()
                .map(|line| line_width(line) / page_width)
                .collect());
            let width_ratio = left_width.min(right_width) / left_width.max(right_width);
            Some((
                crossings,
                imbalance,
                gap,
                ColumnModel {
                    kind: if (crossings == 0 && (0.40..=0.60).contains(&split_ratio))
                        || width_ratio >= 0.60
                    {
                        "two_column"
                    } else {
                        "margin_column"
                    },
                    split_x,
                    left_count: left_lines.len(),
                    right_count: right_lines.len(),
                },
            ))
        })
        .min_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then(left.1.cmp(&right.1))
                .then_with(|| right.2.total_cmp(&left.2))
        })
        .map(|(_, _, _, model)| model)
        .unwrap_or(single)
}

fn note_column_model(lines: &[Line], page_width: f64) -> ColumnModel {
    let mut model = column_model(lines, page_width);
    if model.kind != "two_column" {
        return model;
    }
    let crossing_prose = lines
        .iter()
        .filter(|line| {
            has_valid_bbox(line)
                && line
                    .text
                    .chars()
                    .filter(|character| !character.is_whitespace())
                    .count()
                    >= 8
                && line.bbox[0] < model.split_x
                && line.bbox[2] > model.split_x
        })
        .count();
    if crossing_prose > 0 {
        model.kind = "margin_column";
    }
    model
}

fn geometry_order(lines: &mut [Line]) {
    lines.sort_by(|left, right| {
        line_center_y(left)
            .total_cmp(&line_center_y(right))
            .then(line_center_x(left).total_cmp(&line_center_x(right)))
            .then(left.bbox[0].total_cmp(&right.bbox[0]))
            .then(left.id.cmp(&right.id))
    });
}

fn column_order(lines: &mut [Line], split_x: f64) {
    let spans = |line: &Line| {
        let width = line_width(line);
        width > 0.0
            && line.bbox[0] <= split_x - width * 0.20
            && line.bbox[2] >= split_x + width * 0.20
    };
    let bounds = |right: bool| {
        let mut values = lines
            .iter()
            .filter(|line| {
                has_valid_bbox(line) && !spans(line) && (line_center_x(line) >= split_x) == right
            })
            .map(line_center_y);
        let first = values.next()?;
        Some(values.fold((first, first), |(top, bottom), value| {
            (top.min(value), bottom.max(value))
        }))
    };
    let Some((left, right)) = bounds(false).zip(bounds(true)) else {
        geometry_order(lines);
        return;
    };
    let overlap = (left.0.max(right.0), left.1.min(right.1));
    let mut anchors: Vec<_> = lines
        .iter()
        .filter(|line| spans(line) && (overlap.0..=overlap.1).contains(&line_center_y(line)))
        .map(line_center_y)
        .collect();
    anchors.sort_by(f64::total_cmp);
    lines.sort_by(|left, right| {
        let left_y = line_center_y(left);
        let right_y = line_center_y(right);
        let band = |y: f64| usize::from(y >= overlap.0) + usize::from(y > overlap.1);
        let left_band = band(left_y);
        let right_band = band(right_y);
        let band_order = left_band.cmp(&right_band);
        if band_order != std::cmp::Ordering::Equal || left_band != 1 {
            return band_order
                .then(left_y.total_cmp(&right_y))
                .then(left.bbox[0].total_cmp(&right.bbox[0]))
                .then(left.id.cmp(&right.id));
        }
        let left_anchor = spans(left);
        let right_anchor = spans(right);
        let left_segment = anchors.partition_point(|anchor| *anchor < left_y);
        let right_segment = anchors.partition_point(|anchor| *anchor < right_y);
        let left_column = usize::from(line_center_x(left) >= split_x);
        let right_column = usize::from(line_center_x(right) >= split_x);
        left_segment
            .cmp(&right_segment)
            .then(left_anchor.cmp(&right_anchor))
            .then_with(|| {
                if left_anchor {
                    std::cmp::Ordering::Equal
                } else {
                    left_column.cmp(&right_column)
                }
            })
            .then(left_y.total_cmp(&right_y))
            .then(line_center_x(left).total_cmp(&line_center_x(right)))
            .then(left.bbox[0].total_cmp(&right.bbox[0]))
            .then(left.id.cmp(&right.id))
    });
}

fn column_switches(lines: &[Line], model: ColumnModel, page_width: f64) -> usize {
    let mut previous = None;
    let mut switches = 0;
    for line in lines {
        if !has_valid_bbox(line) || line_width(line) / page_width > 0.55 {
            continue;
        }
        let column = line_center_x(line) >= model.split_x;
        if previous.is_some_and(|prior| prior != column) {
            switches += 1;
        }
        previous = Some(column);
    }
    switches
}

fn median_column_run(lines: &[Line], model: ColumnModel, page_width: f64) -> usize {
    let mut previous = None;
    let mut current = 0;
    let mut runs = Vec::new();
    for line in lines {
        if !has_valid_bbox(line) || line_width(line) / page_width > 0.55 {
            continue;
        }
        let column = line_center_x(line) >= model.split_x;
        if previous.is_none_or(|prior| prior == column) {
            current += 1;
        } else {
            runs.push(current);
            current = 1;
        }
        previous = Some(column);
    }
    if current > 0 {
        runs.push(current);
    }
    runs.sort_unstable();
    runs.get(runs.len() / 2).copied().unwrap_or(0)
}

fn hyphen_join_score(lines: &[Line]) -> (usize, usize) {
    let mut candidates = 0;
    let mut satisfied = 0;
    for (index, line) in lines.iter().enumerate() {
        let text = line.text.trim_end();
        let Some(last) = text
            .chars()
            .last()
            .filter(|character| matches!(character, '-' | '\u{00ad}' | '\u{00ac}'))
        else {
            continue;
        };
        let prefix = &text[..text.len() - last.len_utf8()];
        if prefix
            .chars()
            .rev()
            .take_while(|character| character.is_ascii_alphabetic())
            .count()
            < 2
        {
            continue;
        }
        candidates += 1;
        if lines.get(index + 1).is_some_and(|next| {
            next.text
                .trim_start()
                .chars()
                .take_while(|character| character.is_ascii_alphabetic())
                .count()
                >= 2
        }) {
            satisfied += 1;
        }
    }
    (candidates, satisfied)
}

fn y_regressions(lines: &[Line]) -> usize {
    let boxed: Vec<&Line> = lines.iter().filter(|line| has_valid_bbox(line)).collect();
    let tolerance = p50(boxed
        .iter()
        .map(|line| (line.bbox[3] - line.bbox[1]).max(0.0))
        .collect());
    boxed
        .windows(2)
        .filter(|pair| line_center_y(pair[1]) < line_center_y(pair[0]) - tolerance)
        .count()
}

fn arbitrate_body_order(lines: &mut [Line], page_width: f64, page_height: f64) -> OrderDecision {
    let boxed = lines.iter().filter(|line| has_valid_bbox(line)).count();
    if lines.len() < 8 || boxed * 5 < lines.len() * 4 || page_width <= 0.0 || page_height <= 0.0 {
        return OrderDecision::keep("insufficient_geometry");
    }
    if has_table_caption(lines) {
        return OrderDecision::keep("table_grid");
    }
    let mut model = column_model(lines, page_width);
    if model.kind != "two_column" {
        let alternative = column_model_with_furniture(lines, page_width, true);
        if alternative.kind == "two_column" {
            model = alternative;
        }
    }
    if model.kind == "two_column" {
        let source_switches = column_switches(lines, model, page_width);
        let source_run = median_column_run(lines, model, page_width);
        let source_hyphens = hyphen_join_score(lines);
        let mut challenger = lines.to_vec();
        column_order(&mut challenger, model.split_x);
        let challenger_switches = column_switches(&challenger, model, page_width);
        let challenger_hyphens = hyphen_join_score(&challenger);
        let minimum_switches = if source_hyphens.0 > 0 { 3 } else { 5 }
            .max(((model.left_count + model.right_count) as f64 * 0.10) as usize);
        if source_switches >= minimum_switches
            && source_run <= 6
            && challenger_switches <= 2
            && source_switches.saturating_sub(challenger_switches) >= 2
            && challenger_hyphens.1 >= source_hyphens.1
            && challenger_hyphens.0.saturating_sub(challenger_hyphens.1)
                <= source_hyphens.0.saturating_sub(source_hyphens.1)
        {
            lines.clone_from_slice(&challenger);
            return OrderDecision {
                repair: OrderRepair::Column,
                source_switches,
                strategy: "column-geometry",
                reason: "column_interleave_repair",
            };
        }
        return OrderDecision {
            repair: OrderRepair::None,
            source_switches,
            strategy: "kraken-native",
            reason: "two_column_kraken_coherent",
        };
    }
    if model.kind == "single" {
        let source_regressions = y_regressions(lines);
        let source_hyphens = hyphen_join_score(lines);
        let mut challenger = lines.to_vec();
        geometry_order(&mut challenger);
        let challenger_regressions = y_regressions(&challenger);
        let challenger_hyphens = hyphen_join_score(&challenger);
        let threshold = 3.max((boxed as f64 * 0.08) as usize);
        if source_regressions >= threshold
            && challenger_regressions <= source_regressions / 3
            && challenger_hyphens.1 > source_hyphens.1
            && challenger_hyphens.0.saturating_sub(challenger_hyphens.1)
                < source_hyphens.0.saturating_sub(source_hyphens.1)
        {
            lines.clone_from_slice(&challenger);
            return OrderDecision {
                repair: OrderRepair::Geometry,
                source_switches: 0,
                strategy: "geometry",
                reason: "kraken_order_scrambled",
            };
        }
    }
    OrderDecision::keep(if model.kind == "single" {
        "single_column_kraken_coherent"
    } else {
        "non_two_column"
    })
}

fn repair_drop_caps(lines: &mut Vec<Line>) {
    let body_size = p50(lines
        .iter()
        .map(line_font_size)
        .filter(|size| (4.0..=24.0).contains(size))
        .collect());
    if body_size <= 0.0 {
        return;
    }
    let mut moves = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let text = line.text.trim();
        if text.chars().count() != 1
            || !text.chars().all(char::is_alphabetic)
            || line_font_size(line) < body_size * 1.8
        {
            continue;
        }
        let target = lines
            .iter()
            .enumerate()
            .filter(|(other, candidate)| {
                *other != index
                    && line_font_size(candidate) <= body_size * 1.4
                    && candidate.bbox[0] >= line.bbox[2] - body_size
                    && candidate.bbox[0] - line.bbox[2] <= body_size * 3.0
                    && candidate.bbox[1] < line.bbox[3]
                    && candidate.bbox[3] > line.bbox[1]
            })
            .min_by(|(_, left), (_, right)| {
                left.bbox[1]
                    .total_cmp(&right.bbox[1])
                    .then(left.bbox[0].total_cmp(&right.bbox[0]))
            })
            .map(|(target, _)| target);
        if let Some(target) = target.filter(|target| *target < index) {
            moves.push((index, target));
        }
    }
    for (index, target) in moves.into_iter().rev() {
        let line = lines.remove(index);
        lines.insert(target, line);
    }
}

fn band_geometry_top(line: &Line) -> f64 {
    line.words
        .iter()
        .map(|word| word.bbox[1])
        .min_by(f64::total_cmp)
        .unwrap_or(line.bbox[1])
}

fn band_geometry_order(left: &Line, right: &Line) -> std::cmp::Ordering {
    band_geometry_top(left)
        .total_cmp(&band_geometry_top(right))
        .then(left.bbox[0].total_cmp(&right.bbox[0]))
}

fn standalone_note_label(line: &Line) -> bool {
    let text = line.text.trim();
    let length = text.chars().count();
    ((1..=MAX_SYMBOL_LABEL_LEN).contains(&length) && text.chars().all(is_note_symbol))
        || ((1..=4).contains(&length) && text.chars().all(|character| character.is_ascii_digit()))
}

fn aligned_note_body(label: &Line, body: &Line) -> bool {
    if standalone_note_label(body) || body.bbox[0] < label.bbox[0] {
        return false;
    }
    let overlap = label.bbox[3].min(body.bbox[3]) - label.bbox[1].max(body.bbox[1]);
    let minimum_height = (label.bbox[3] - label.bbox[1]).min(body.bbox[3] - body.bbox[1]);
    overlap > 0.0 && minimum_height > 0.0 && overlap / minimum_height >= 0.5
}

fn aligned_note_body_index(lines: &[Line], label_index: usize) -> Option<usize> {
    let label = &lines[label_index];
    lines
        .iter()
        .enumerate()
        .filter(|(_, body)| aligned_note_body(label, body))
        .min_by(|(_, left), (_, right)| {
            (left.bbox[0] - label.bbox[2])
                .max(0.0)
                .total_cmp(&(right.bbox[0] - label.bbox[2]).max(0.0))
                .then(
                    (left.bbox[1] - label.bbox[1])
                        .abs()
                        .total_cmp(&(right.bbox[1] - label.bbox[1]).abs()),
                )
        })
        .map(|(index, _)| index)
}

fn order_note_lines(lines: &mut Vec<Line>, page_width: f64) {
    let mut tops: Vec<f64> = lines.iter().map(band_geometry_top).collect();
    for (label_index, label) in lines.iter().enumerate() {
        if !standalone_note_label(label) {
            continue;
        }
        if let Some(body_index) = aligned_note_body_index(lines, label_index) {
            tops[label_index] = tops[body_index];
        }
    }
    let columns = note_column_model(lines, page_width);
    let mut keyed: Vec<_> = std::mem::take(lines).into_iter().zip(tops).collect();
    keyed.sort_by(|(left, left_top), (right, right_top)| {
        let left_column =
            usize::from(columns.kind == "two_column" && line_center_x(left) >= columns.split_x);
        let right_column =
            usize::from(columns.kind == "two_column" && line_center_x(right) >= columns.split_x);
        left_column.cmp(&right_column).then(
            left_top
                .total_cmp(right_top)
                .then(left.bbox[0].total_cmp(&right.bbox[0])),
        )
    });
    lines.extend(keyed.into_iter().map(|(line, _)| line));
}

fn weave_note_columns(mut body: Vec<Line>, note: Vec<Line>, page_width: f64) -> Vec<Line> {
    let note_columns = note_column_model(&note, page_width);
    let body_model = column_model(&body, page_width);
    let split_x = if note_columns.kind == "two_column" {
        note_columns.split_x
    } else if body_model.kind == "two_column" {
        body_model.split_x
    } else {
        body.extend(note);
        return body;
    };
    let note_sides = note
        .iter()
        .filter(|line| line_width(line) / page_width <= 0.55)
        .fold([0_usize; 2], |mut counts, line| {
            counts[usize::from(line_center_x(line) >= split_x)] += 1;
            counts
        });
    if note_sides.contains(&0) {
        body.extend(note);
        return body;
    }
    let (left_notes, right_notes): (Vec<_>, Vec<_>) = note
        .into_iter()
        .partition(|line| line_center_x(line) < split_x);
    let body_columns = body
        .iter()
        .filter(|line| line_width(line) / page_width <= 0.55)
        .fold([0_usize; 2], |mut counts, line| {
            counts[usize::from(line_center_x(line) >= split_x)] += 1;
            counts
        });
    if left_notes.is_empty()
        || right_notes.is_empty()
        || body_columns.iter().any(|count| *count < 3)
    {
        body.extend(left_notes);
        body.extend(right_notes);
        return body;
    }
    let insert_at = body
        .iter()
        .rposition(|line| line_width(line) / page_width <= 0.55 && line_center_x(line) < split_x)
        .map_or(0, |index| index + 1);
    let right_body = body.split_off(insert_at);
    body.extend(left_notes);
    body.extend(right_body);
    body.extend(right_notes);
    body
}

fn order_page(page: &mut Page, table_page: bool, table_notes: &HashSet<usize>) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    let mut header = Vec::new();
    let mut body = Vec::new();
    let mut note = Vec::new();
    let mut footer = Vec::new();
    for (index, line) in std::mem::take(&mut page.lines).into_iter().enumerate() {
        match line.region_type.as_str() {
            "header" => header.push(line),
            "footnote" if table_notes.contains(&index) => body.push(line),
            "footnote" => note.push(line),
            "footer" => footer.push(line),
            _ => body.push(line),
        }
    }
    header.sort_by(band_geometry_order);
    repair_drop_caps(&mut body);
    let contents = table_page && !has_table_caption(&body) && contents_grid(&body, page.width);
    let decision = if contents && y_regressions(&body) > 0 {
        geometry_order(&mut body);
        OrderDecision {
            repair: OrderRepair::Geometry,
            source_switches: 0,
            strategy: "table-row-geometry",
            reason: "table_source_order_scrambled",
        }
    } else if table_page {
        OrderDecision::keep("table_grid")
    } else {
        arbitrate_body_order(&mut body, page.width, page.height)
    };
    if decision.repair != OrderRepair::None {
        let mut diagnostic = Diagnostic::info(
            "COLUMN_ORDER_REPAIRED",
            format!(
                "Extraction order replaced by {}: {}.",
                decision.strategy, decision.reason
            ),
            Some(page.index),
        );
        diagnostic.line_ids = body.iter().take(20).map(|line| line.id.clone()).collect();
        diagnostics.push(diagnostic);
    } else if decision.source_switches > 2 {
        let mut diagnostic = Diagnostic::warning(
            "COLUMN_ORDER_UNCERTAIN",
            format!(
                "Two-column page keeps an order that crosses columns {} times ({}).",
                decision.source_switches, decision.reason
            ),
            Some(page.index),
        );
        diagnostic.line_ids = body.iter().take(20).map(|line| line.id.clone()).collect();
        diagnostics.push(diagnostic);
    }
    order_note_lines(&mut note, page.width);
    footer.sort_by(band_geometry_order);
    let mut ordered = header;
    ordered.extend(weave_note_columns(body, note, page.width));
    ordered.extend(footer);
    for (index, line) in ordered.iter_mut().enumerate() {
        line.reading_order = index + 1;
    }
    page.lines = ordered;
    diagnostics
}

#[cfg(test)]
fn order_pages(pages: &mut [Page]) -> Vec<Diagnostic> {
    pages
        .iter_mut()
        .flat_map(|page| {
            let evidence = table_evidence(&page.lines, page.width);
            let table_page =
                has_table_caption(&page.lines) || strong_table_evidence(&evidence, &page.lines);
            order_page(page, table_page, &HashSet::new())
        })
        .collect()
}

fn build_regions(pages: &mut [Page]) {
    for page in pages {
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for index in 0..page.lines.len() {
            if groups.last().is_some_and(|group| {
                let prior = &page.lines[*group.last().expect("non-empty group")];
                prior.region_type == page.lines[index].region_type
                    && prior.block_index == page.lines[index].block_index
            }) {
                groups.last_mut().expect("group exists").push(index);
            } else {
                groups.push(vec![index]);
            }
        }
        page.regions.clear();
        for (region_index, indexes) in groups.into_iter().enumerate() {
            let id = format!("p{:04}-r{:04}", page.number, region_index + 1);
            let kind = page.lines[indexes[0]].region_type.clone();
            let line_ids = indexes
                .iter()
                .map(|&index| page.lines[index].id.clone())
                .collect();
            let bbox = union_bbox(indexes.iter().map(|&index| page.lines[index].bbox));
            let reading_order = indexes
                .iter()
                .map(|&index| page.lines[index].reading_order)
                .min()
                .unwrap_or(0);
            for &index in &indexes {
                page.lines[index].region_id.clone_from(&id);
            }
            page.regions.push(Region {
                id,
                page_index: page.index,
                kind,
                line_ids,
                bbox,
                reading_order,
            });
        }
    }
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

fn join_lines(lines: &[&Line]) -> (String, HashMap<String, (usize, usize)>) {
    let mut text = String::new();
    let mut offsets = HashMap::new();
    for line in lines {
        let value = line.text.trim();
        if value.is_empty() {
            continue;
        }
        if text
            .chars()
            .next_back()
            .is_some_and(|character| matches!(character, '-' | '\u{00ad}' | '\u{00ac}'))
            && value.chars().next().is_some_and(char::is_lowercase)
        {
            text.pop();
        } else if !text.is_empty() {
            text.push(' ');
        }
        let start = text.chars().count();
        text.push_str(value);
        offsets.insert(line.id.clone(), (start, start + value.chars().count()));
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
        let mut regions: Vec<&Region> = page.regions.iter().collect();
        regions.sort_by_key(|region| region.reading_order);
        for region in regions {
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
            for line in &lines {
                let Some((base, _)) = line_offsets.get(&line.id) else {
                    continue;
                };
                for anchor in anchors.get(&line.id).into_iter().flatten() {
                    events.push((
                        base + anchor.start,
                        base + anchor.end,
                        anchor.pair_id.clone(),
                        anchor.label.clone(),
                    ));
                }
            }
            events.sort_by_key(|event| (event.0, event.1));
            let mut rendered = String::new();
            let mut output_anchors = Vec::new();
            let mut cursor = 0;
            for (start, end, pair_id, label) in events {
                let start = start.max(cursor).min(text.chars().count());
                let end = end.max(start).min(text.chars().count());
                rendered.push_str(char_slice(&text, cursor, start));
                let offset = rendered.chars().count();
                rendered.push_str(&format!("⟦FN:{pair_id}⟧"));
                output_anchors.push(json!({
                    "pair_id": pair_id,
                    "label": label,
                    "offset": offset,
                }));
                cursor = end;
            }
            rendered.push_str(char_slice(&text, cursor, text.chars().count()));
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

fn sentence_at(text: &str, offset: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
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
    let previous: Vec<(usize, usize)> = boundaries
        .iter()
        .copied()
        .filter(|(_, end)| *end <= offset)
        .collect();
    let (start, end) = if previous.last().is_some_and(|(_, end)| {
        char_slice(text, *end, offset.min(chars.len()))
            .trim()
            .is_empty()
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
    clean_markers(char_slice(text, start, end))
}

fn attach_propositions(footnotes: &mut [Footnote], paragraphs: &[Paragraph]) {
    let by_pair: HashMap<String, usize> = footnotes
        .iter()
        .enumerate()
        .map(|(index, footnote)| (footnote.pair_id.clone(), index))
        .collect();
    let mut previous_tail = String::new();
    for paragraph in paragraphs {
        let mut anchors = paragraph.anchors.clone();
        anchors.sort_by_key(|anchor| anchor.get("offset").and_then(Value::as_u64).unwrap_or(0));
        let mut previous_offset = 0;
        for anchor in anchors {
            let Some(pair_id) = anchor.get("pair_id").and_then(Value::as_str) else {
                continue;
            };
            let Some(&index) = by_pair.get(pair_id) else {
                continue;
            };
            let offset = anchor.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
            footnotes[index].sentence_proposition = sentence_at(&paragraph.text, offset);
            let passage = clean_markers(char_slice(&paragraph.text, previous_offset, offset));
            footnotes[index].passage_since_prior_note = if passage.is_empty() {
                previous_tail.clone()
            } else {
                passage
            };
            previous_offset = offset + format!("⟦FN:{pair_id}⟧").chars().count();
        }
        previous_tail = char_slice(
            &paragraph.text,
            previous_offset,
            paragraph.text.chars().count(),
        )
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
    let prefix = &text[..byte_start];
    let window_start = prefix.chars().count().saturating_sub(70);
    let start = char_to_byte(prefix, window_start);
    let Some(capture) = pattern.captures(&prefix[start..]) else {
        return String::new();
    };
    let short = capture
        .get(1)
        .map_or("", |value| value.as_str())
        .trim()
        .trim_end_matches([',', '.', ';', ':'])
        .to_owned();
    if matches!(
        short.to_lowercase().as_str(),
        "see" | "in" | "the" | "but" | "and" | "also" | "supra" | "infra" | "ibid" | "at"
    ) {
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
        let eligible: Vec<&Line> = page
            .lines
            .iter()
            .filter(|line| {
                !matches!(line.region_type.as_str(), "header" | "footer") && !line.exclude_from_body
            })
            .collect();
        for pair in eligible.windows(2) {
            let previous = pair[0];
            let current = pair[1];
            if !hyphen_fragment_tail(&previous.text) {
                continue;
            }
            if hyphen_continuation(&current.text) {
                if line_family(previous) != line_family(current) {
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
                }
            } else if line_family(previous) == line_family(current) {
                let mut diagnostic = Diagnostic::info(
                    "DANGLING_SOFT_HYPHEN",
                    "A line ends mid-word but the next eligible line does not continue it.",
                    Some(page.index),
                );
                diagnostic.line_ids.push(previous.id.clone());
                diagnostics.push(diagnostic);
            }
        }
    }
    diagnostics
}

fn unmatched_reference_diagnostics(
    pages: &[Page],
    footnotes: &[Footnote],
    anchors: &HashMap<String, Vec<Anchor>>,
) -> Vec<Diagnostic> {
    let labels: HashSet<String> = footnotes.iter().map(|note| note.label.clone()).collect();
    let primary_lines: HashSet<&str> = footnotes
        .iter()
        .filter_map(|note| note.reference_line_id.as_deref())
        .collect();
    let mut diagnostics = Vec::new();
    for line in pages.iter().flat_map(|page| page.lines.iter()) {
        for detached in &line.detached_references {
            let label = normalize_label(
                detached
                    .get("note_id")
                    .and_then(Value::as_str)
                    .unwrap_or(""),
            );
            let start = detached
                .get("start_offset")
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .unwrap_or(0);
            let end = detached
                .get("end_offset")
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .unwrap_or(start);
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
                if let Some(source) = detached.get("source_line_id").and_then(Value::as_str) {
                    if !source.is_empty() {
                        diagnostic.line_ids.push(source.to_owned());
                    }
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
        if line
            .spans
            .iter()
            .any(|span| span.superscript && labels.contains(&normalize_label(span.text.trim())))
        {
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
    let mut by_number: HashMap<String, Vec<(usize, String)>> = HashMap::new();
    for footnote in footnotes.iter() {
        if let Ok(number) = footnote.label.parse::<u32>() {
            by_number
                .entry(number.to_string())
                .or_default()
                .push((footnote.restart_sequence, footnote.pair_id.clone()));
        }
    }
    for footnote in footnotes {
        for capture in pattern.captures_iter(&footnote.body) {
            let Some(found) = capture.get(0) else {
                continue;
            };
            let number = capture
                .get(4)
                .and_then(|value| value.as_str().parse::<u32>().ok())
                .unwrap_or(0);
            let number_key = number.to_string();
            let candidates = by_number.get(&number_key).cloned().unwrap_or_default();
            let scoped: Vec<&(usize, String)> = candidates
                .iter()
                .filter(|(restart, _)| *restart == footnote.restart_sequence)
                .collect();
            let target_pair_id = if candidates.len() == 1 {
                candidates[0].1.clone()
            } else if scoped.len() == 1 {
                scoped[0].1.clone()
            } else {
                String::new()
            };
            let kind = if let Some(value) = capture.get(1) {
                value.as_str().to_lowercase()
            } else if capture.get(2).is_some() {
                "op_cit".to_owned()
            } else {
                "see_footnote".to_owned()
            };
            let record = json!({
                "source_pair_id": footnote.pair_id,
                "kind": kind,
                "number": number,
                "shortform": crossref_shortform(&footnote.body, found.start()),
                "start": footnote.body[..found.start()].chars().count(),
                "end": footnote.body[..found.end()].chars().count(),
                "resolved": !candidates.is_empty(),
                "target_pair_id": target_pair_id,
                "target_count": candidates.len(),
            });
            footnote.crossrefs.push(record.clone());
            if candidates.is_empty() {
                let mut diagnostic = Diagnostic::info(
                    "NOTE_CROSSREF_UNRESOLVED",
                    format!(
                        "Note {} references {} note {}, which no paired note carries - a pairing-quality witness.",
                        footnote.label, kind, number
                    ),
                    footnote.reference_page.map(|page| page as usize),
                );
                diagnostic.details.insert("crossref".to_owned(), record);
                diagnostics.push(diagnostic);
            }
        }
    }
}

fn add_observation(
    observations: &mut Vec<CandidateObservationV2>,
    observation: CandidateObservationV2,
) {
    if !observations.contains(&observation) {
        observations.push(observation);
    }
}

fn visual_source_region(region: &str) -> bool {
    matches!(
        region,
        "table"
            | "table_cell"
            | "form"
            | "figure"
            | "image"
            | "chart"
            | "formula"
            | "separator"
            | "visual"
    )
}

fn contents_row(text: &str) -> bool {
    static LEADER: OnceLock<Regex> = OnceLock::new();
    LEADER
        .get_or_init(|| Regex::new(r"(?:\. ){3,}|\.{4,}").expect("contents leader regex"))
        .is_match(text)
}

fn transcript_line_number_pages(pages: &[Page]) -> HashSet<usize> {
    const MIN_LINE_NUMBERS: u32 = 15;
    pages
        .iter()
        .filter_map(|page| {
            if page.width <= 0.0 {
                return None;
            }
            let mut candidates = page
                .lines
                .iter()
                .filter_map(|line| {
                    let number = arabic_page_number(&line.text)?;
                    (number <= 40).then_some((line.bbox[0], number))
                })
                .collect::<Vec<_>>();
            candidates.sort_by(|left, right| left.0.total_cmp(&right.0));
            let tolerance = page.width * 0.03;
            let mut best = &candidates[0..0];
            let mut start = 0;
            for end in 0..candidates.len() {
                while candidates[end].0 - candidates[start].0 > tolerance {
                    start += 1;
                }
                if end + 1 - start > best.len() {
                    best = &candidates[start..=end];
                }
            }
            let values = best
                .iter()
                .map(|(_, number)| *number)
                .collect::<HashSet<_>>();
            (values.len() >= MIN_LINE_NUMBERS as usize
                && (1..=MIN_LINE_NUMBERS).all(|number| values.contains(&number)))
            .then_some(page.index)
        })
        .collect()
}

fn index_pages(pages: &[Page]) -> HashSet<usize> {
    static ENTRY: OnceLock<Regex> = OnceLock::new();
    let entry = ENTRY
        .get_or_init(|| Regex::new(r"\[\d{1,3}\]\s+\d{1,3}:\d{1,3}").expect("index entry regex"));
    pages
        .iter()
        .filter_map(|page| {
            let text = page
                .lines
                .iter()
                .map(|line| line.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            (entry.find_iter(&text).take(5).count() >= 5).then_some(page.index)
        })
        .collect()
}

impl PdfResolutionInput {
    fn from_pages(pages: &[Page], primitives: &PdfPrimitiveEvidence) -> Self {
        let index = PdfTextIndex::from_pages(pages);
        let runs = detect_structure_candidate_runs(index.text());
        let contents_pages = pages
            .iter()
            .filter(|page| contents_grid(&page.lines, page.width))
            .map(|page| page.index)
            .collect::<HashSet<_>>();
        let transcript_line_number_pages = transcript_line_number_pages(pages);
        let index_pages = index_pages(pages);
        let by_line = pages
            .iter()
            .flat_map(|page| {
                page.lines
                    .iter()
                    .map(move |line| (line.id.as_str(), (page, line)))
            })
            .collect::<HashMap<_, _>>();
        let mut flow_lines = HashSet::new();
        for page in pages {
            let mut lines = page.lines.iter().collect::<Vec<_>>();
            lines.sort_by(|left, right| {
                left.reading_order
                    .cmp(&right.reading_order)
                    .then_with(|| left.id.cmp(&right.id))
            });
            for pair in lines.windows(2) {
                if pair.iter().all(|line| {
                    !line.exclude_from_body
                        && line.note_region_mode.is_empty()
                        && line.region_type == "body"
                }) && body_flow_edge(pair[0], pair[1])
                {
                    flow_lines.insert(pair[0].id.as_str());
                    flow_lines.insert(pair[1].id.as_str());
                }
            }
        }
        let citation_spans = by_line
            .iter()
            .map(|(line_id, (_, line))| {
                ((*line_id).to_owned(), protected_citation_spans(&line.text))
            })
            .collect::<HashMap<_, _>>();
        let mut evidence = Vec::new();
        for run in &runs {
            let mut list_candidates = HashSet::new();
            if !matches!(run.grammar, CandidateGrammar::Numeric) {
                for candidate in &run.markers {
                    let Some(indexed) = index.line_at(candidate.marker_range.start) else {
                        continue;
                    };
                    let Some((page, line)) = by_line.get(indexed.line_id.as_str()) else {
                        continue;
                    };
                    let aligned_siblings = run
                        .markers
                        .iter()
                        .filter(|sibling| {
                            if sibling.id == candidate.id || sibling.level != candidate.level {
                                return false;
                            }
                            let Some(sibling_indexed) = index.line_at(sibling.marker_range.start)
                            else {
                                return false;
                            };
                            let Some((sibling_page, sibling_line)) =
                                by_line.get(sibling_indexed.line_id.as_str())
                            else {
                                return false;
                            };
                            line.region_type == "body"
                                && sibling_line.region_type == "body"
                                && !line.exclude_from_body
                                && !sibling_line.exclude_from_body
                                && (line.bbox[0] - sibling_line.bbox[0]).abs()
                                    <= page.width.max(sibling_page.width).max(1.0) * 0.008
                        })
                        .count();
                    let list_context = candidate.parent_candidate_id.is_some()
                        || (run.grammar == CandidateGrammar::Enumerator
                            && run.rooted
                            && run.consecutive);
                    if aligned_siblings >= 1 && list_context {
                        list_candidates.insert(candidate.id.as_str());
                    }
                }
            }
            for candidate in &run.markers {
                let line_ids = index.line_ids(candidate.range);
                let page_indexes = index.page_indexes(candidate.range);
                let marker_line_ids = index.line_ids(candidate.marker_range);
                let mut observations = Vec::new();

                let body_prose = line_ids.iter().take(3).any(|line_id| {
                    let Some((_, line)) = by_line.get(line_id.as_str()) else {
                        return false;
                    };
                    if line.exclude_from_body
                        || !line.note_region_mode.is_empty()
                        || line.region_type != "body"
                    {
                        return false;
                    }
                    let start = index.line(line_id).map_or(0, |indexed| {
                        candidate
                            .content_start
                            .saturating_sub(indexed.range.start)
                            .min(line.text.chars().count())
                    });
                    let tail = char_slice(&line.text, start, line.text.chars().count());
                    tail.chars()
                        .filter(|character| character.is_alphabetic())
                        .count()
                        >= 8
                        && (flow_lines.contains(line.id.as_str())
                            || tail.split_whitespace().take(3).count() == 3)
                });
                if body_prose {
                    add_observation(&mut observations, CandidateObservationV2::BodyProseFlow);
                }
                if marker_line_ids.iter().any(|line_id| {
                    by_line.get(line_id.as_str()).is_some_and(|(_, line)| {
                        line.region_type == "heading"
                            || primitives
                                .source_regions
                                .as_ref()
                                .and_then(|regions| regions.get(&line.id))
                                .is_some_and(|region| {
                                    matches!(region.as_str(), "heading" | "paragraph_title")
                                })
                    })
                }) {
                    add_observation(&mut observations, CandidateObservationV2::SectionHeading);
                }
                if list_candidates.contains(candidate.id.as_str()) && body_prose {
                    add_observation(&mut observations, CandidateObservationV2::ListItemLayout);
                }
                let marker_is_cross_reference = marker_line_ids.iter().any(|line_id| {
                    let Some(indexed) = index.line(line_id) else {
                        return false;
                    };
                    let local = ScalarRange {
                        start: candidate
                            .marker_range
                            .start
                            .saturating_sub(indexed.range.start),
                        end: candidate
                            .marker_range
                            .end
                            .saturating_sub(indexed.range.start)
                            .min(indexed.range.end - indexed.range.start),
                    };
                    citation_spans.get(line_id.as_str()).is_some_and(|spans| {
                        spans
                            .iter()
                            .any(|(start, end)| *start < local.end && local.start < *end)
                    })
                });
                if marker_is_cross_reference {
                    add_observation(&mut observations, CandidateObservationV2::CrossReference);
                }
                let table_or_form = marker_line_ids.iter().any(|line_id| {
                    primitives.table_cell_line_ids.contains(line_id)
                        || by_line.get(line_id.as_str()).is_some_and(|(_, line)| {
                            primitives
                                .source_regions
                                .as_ref()
                                .and_then(|regions| regions.get(&line.id))
                                .is_some_and(|region| visual_source_region(region))
                        })
                });
                if table_or_form {
                    add_observation(&mut observations, CandidateObservationV2::TableOrForm);
                }
                let contents = line_ids.iter().take(3).any(|line_id| {
                    by_line.get(line_id.as_str()).is_some_and(|(page, line)| {
                        contents_pages.contains(&page.index) || contents_row(&line.text)
                    })
                });
                if contents {
                    add_observation(&mut observations, CandidateObservationV2::ContentsRow);
                }
                let marker_pages = marker_line_ids
                    .iter()
                    .filter_map(|line_id| by_line.get(line_id.as_str()).map(|(page, _)| page.index))
                    .collect::<HashSet<_>>();
                if marker_pages.iter().any(|page| index_pages.contains(page)) {
                    add_observation(&mut observations, CandidateObservationV2::IndexRow);
                }
                if marker_pages
                    .iter()
                    .any(|page| transcript_line_number_pages.contains(page))
                {
                    add_observation(
                        &mut observations,
                        CandidateObservationV2::TranscriptLineNumber,
                    );
                }
                let furniture = marker_line_ids.iter().any(|line_id| {
                    by_line.get(line_id.as_str()).is_some_and(|(_, line)| {
                        matches!(line.region_type.as_str(), "header" | "footer")
                            || (line.exclude_from_body
                                && !primitives.table_cell_line_ids.contains(&line.id)
                                && !primitives.table_note_line_ids.contains(&line.id))
                    })
                });
                if furniture {
                    add_observation(&mut observations, CandidateObservationV2::Furniture);
                }
                evidence.push(CandidateEvidenceV2 {
                    candidate_id: candidate.id.clone(),
                    page_indexes,
                    line_ids,
                    observations,
                });
            }
        }
        Self {
            index,
            runs,
            evidence,
            citation_spans: citation_spans
                .into_iter()
                .map(|(line_id, spans)| {
                    (
                        line_id,
                        spans
                            .into_iter()
                            .map(|(start, end)| PdfSourceSpan { start, end })
                            .collect(),
                    )
                })
                .collect(),
        }
    }
}

fn pdf_source_map(
    index: &PdfTextIndex,
    pages: &[Page],
    structure: &mut DocumentStructure,
    protected_citation_spans: BTreeMap<String, Vec<PdfSourceSpan>>,
) -> PdfSourceMap {
    PdfSourceMap {
        pages: pages
            .iter()
            .map(|page| PdfPageIdentity {
                physical_index: page.index,
                physical_number: page.number,
                printed_folio: page.printed_label.clone(),
            })
            .collect(),
        nodes: structure
            .nodes
            .iter_mut()
            .map(|node| PdfSourceExtent {
                id: node.id.clone(),
                page_indexes: std::mem::take(&mut node.page_indexes),
                line_ids: std::mem::take(&mut node.line_ids),
            })
            .collect(),
        note_references: structure
            .notes
            .iter()
            .flat_map(|note| {
                note.references
                    .iter()
                    .enumerate()
                    .map(move |(reference, value)| PdfSourceExtent {
                        id: format!("{}:reference:{reference}", note.id),
                        page_indexes: index.page_indexes(value.range),
                        line_ids: index.line_ids(value.range),
                    })
            })
            .collect(),
        protected_citation_spans,
        table_ids: Vec::new(),
        image_ids: Vec::new(),
    }
}

fn map_note_pairs(
    index: &PdfTextIndex,
    pairs: &[NotePairClaim],
) -> Result<(Vec<NotePairClaimV2>, Vec<StructureDiagnostic>)> {
    let mut claims = Vec::new();
    let mut diagnostics = Vec::new();
    let mut pair_ids = HashSet::new();
    for pair in pairs {
        if pair.pair_id.is_empty() || !pair_ids.insert(pair.pair_id.as_str()) {
            return Err(Error::Message(format!(
                "paired note '{}' has an empty or duplicate pair id",
                pair.pair_id
            )));
        }
        let label_line = index.line(&pair.label_anchor.line_id).ok_or_else(|| {
            Error::Message(format!(
                "paired note {} has an unknown label line",
                pair.pair_id
            ))
        })?;
        let label_range = index
            .global_range(
                &pair.label_anchor.line_id,
                pair.label_anchor.start,
                pair.label_anchor.end,
            )
            .ok_or_else(|| {
                Error::Message(format!(
                    "paired note {} has an invalid label range",
                    pair.pair_id
                ))
            })?;
        let references = pair
            .reference_anchors
            .iter()
            .map(|anchor| {
                let line = index.line(&anchor.line_id).ok_or_else(|| {
                    Error::Message(format!(
                        "paired note {} has an unknown reference line",
                        pair.pair_id
                    ))
                })?;
                let range = index
                    .global_range(&anchor.line_id, anchor.start, anchor.end)
                    .ok_or_else(|| {
                        Error::Message(format!(
                            "paired note {} has an invalid reference range",
                            pair.pair_id
                        ))
                    })?;
                Ok(TextAnchorV2 {
                    range,
                    page_index: line.page_index,
                    line_id: anchor.line_id.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if pair
            .body_line_ids
            .iter()
            .any(|line_id| index.line(line_id).is_none())
        {
            return Err(Error::Message(format!(
                "paired note {} has an unknown body line",
                pair.pair_id
            )));
        }
        let body_range = index.range_for_line_ids(&pair.body_line_ids);
        if pair.reference_anchors.is_empty()
            || body_range.is_none()
            || body_range.is_some_and(|range| range.start == range.end)
            || label_range.start == label_range.end
            || references
                .iter()
                .any(|anchor| anchor.range.start == anchor.range.end)
        {
            diagnostics.push(StructureDiagnostic {
                code: "note_pair_unmaterialized".to_owned(),
                severity: DiagnosticSeverity::Info,
                ranges: Vec::new(),
                node_ids: Vec::new(),
            });
            continue;
        }
        claims.push(NotePairClaimV2 {
            pair_id: pair.pair_id.clone(),
            kind: match pair.kind {
                NotePairKind::Footnote => NoteKindV2::Footnote,
                NotePairKind::Endnote => NoteKindV2::Endnote,
            },
            label: TextAnchorV2 {
                range: label_range,
                page_index: label_line.page_index,
                line_id: pair.label_anchor.line_id.clone(),
            },
            body: NoteBodyV2 {
                range: body_range.unwrap(),
                page_indexes: index.page_indexes_for_line_ids(&pair.body_line_ids),
                line_ids: pair.body_line_ids.clone(),
            },
            references,
        });
    }
    Ok((claims, diagnostics))
}

fn native_graph_parts(
    index: &PdfTextIndex,
    pages: &[Page],
    paragraphs: &[Paragraph],
) -> Result<Vec<StructureNode>> {
    const ORIGIN: &str = "legalpdf.pdf-structure.v2";
    let mut nodes = Vec::new();
    for page in pages {
        let range = index.page_range(page.index).ok_or_else(|| {
            Error::Message(format!("page {} is absent from the text index", page.index))
        })?;
        nodes.push(StructureNode {
            id: page.id.clone(),
            kind: NodeKind::Page,
            range,
            rendered_range: None,
            origin_id: ORIGIN.to_owned(),
            source: Derivation::Native,
            label: page.printed_label.clone(),
            locator_kind: None,
            aliases: None,
            parent_id: None,
            anchor: page.printed_label_line_id.clone(),
            content_start: None,
            marker_range: None,
            page_indexes: vec![page.index],
            line_ids: index
                .lines()
                .iter()
                .filter(|line| line.page_index == page.index)
                .map(|line| line.line_id.clone())
                .collect(),
            grammar: None,
            proof: None,
        });
    }
    for paragraph in paragraphs {
        let range = index
            .range_for_line_ids(&paragraph.line_ids)
            .ok_or_else(|| {
                Error::Message(format!("paragraph {} has no indexed lines", paragraph.id))
            })?;
        let heading = paragraph.region_type == "heading";
        nodes.push(StructureNode {
            id: paragraph.id.clone(),
            kind: if heading {
                NodeKind::Heading
            } else {
                NodeKind::Prose
            },
            range,
            rendered_range: None,
            origin_id: ORIGIN.to_owned(),
            source: if heading {
                Derivation::Heuristic
            } else {
                Derivation::Native
            },
            label: heading.then(|| char_slice(index.text(), range.start, range.end).to_owned()),
            locator_kind: None,
            aliases: None,
            parent_id: None,
            anchor: None,
            content_start: None,
            marker_range: None,
            page_indexes: index.page_indexes_for_line_ids(&paragraph.line_ids),
            line_ids: paragraph.line_ids.clone(),
            grammar: heading.then(|| "accepted_heading".to_owned()),
            proof: None,
        });
    }
    Ok(nodes)
}

pub fn prepare_pages(pages: &mut [Page], separators: &[Option<f64>]) -> PdfPreparation {
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
    PdfPreparation {
        diagnostics,
        primitives,
        resolution: None,
    }
}

pub fn prepare_derivation(pages: &mut [Page], mut prepared: PdfPreparation) -> PdfPreparation {
    legal_pdf_support::profile::measure("derive.note_regions", || infer_note_region_modes(pages));
    prepared
        .diagnostics
        .extend(legal_pdf_support::profile::measure(
            "derive.text_flow",
            || text_flow_faults(pages),
        ));
    prepared.resolution = Some(legal_pdf_support::profile::measure(
        "derive.structure_candidates",
        || PdfResolutionInput::from_pages(pages, &prepared.primitives),
    ));
    prepared
}

pub fn finish_derivation(
    pages: &mut [Page],
    mut prepared: PdfPreparation,
    pairing: PairingOutput,
    identity: StructureIdentity,
) -> Result<StructureOutput> {
    let resolution = prepared.resolution.take().ok_or_else(|| {
        Error::Message("PDF structure candidates were not prepared before pairing".to_owned())
    })?;
    let (note_pairs, graph_diagnostics) = map_note_pairs(&resolution.index, &pairing.pair_claims)?;
    let pairing_summary = pairing.summary;
    let markers = pairing.markers;
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
    let mut structure_graph = legal_pdf_support::profile::measure("derive.structure_graph", || {
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
    let pdf_source_map = pdf_source_map(
        &resolution.index,
        pages,
        &mut structure_graph,
        resolution.citation_spans,
    );
    Ok(StructureOutput {
        paragraphs,
        footnotes,
        diagnostics,
        pairing_audit: PdfPairingAudit {
            markers,
            pairing_summary,
        },
        pdf_source_map,
        structure_graph,
    })
}

pub fn validate_input(pages: &[Page], separators: &[Option<f64>]) -> Result<()> {
    (separators.len() == pages.len())
        .then_some(())
        .ok_or_else(|| {
            Error::Message("common input must contain one separator value per page".to_owned())
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

fn validate_page_records(pages: &[Page]) -> Result<HashSet<String>> {
    let mut ids = HashSet::new();
    let mut span_ids = HashSet::new();
    let mut word_ids = HashSet::new();
    for page in pages {
        let page_ids: HashSet<&str> = page.lines.iter().map(|line| line.id.as_str()).collect();
        if page_ids.len() != page.lines.len()
            || page.lines.iter().any(|line| !ids.insert(line.id.clone()))
        {
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
            || regions_by_line.len() != page_ids.len()
            || page_ids
                .iter()
                .any(|line| !regions_by_line.contains_key(line))
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
                .all(|span| span_ids.insert(span.id.clone()))
            {
                return Err(Error::Message(
                    "document contains duplicate span IDs".to_owned(),
                ));
            }
            if !line
                .words
                .iter()
                .all(|word| word_ids.insert(word.id.clone()))
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
    let known_lines = validate_page_records(&document.pages)?;
    let mut block_ids = HashSet::new();
    for (id, page_index, page_number, bbox) in document
        .tables
        .iter()
        .map(|table| (&table.id, table.page_index, table.page_number, table.bbox))
        .chain(
            document
                .images
                .iter()
                .map(|image| (&image.id, image.page_index, image.page_number, image.bbox)),
        )
    {
        let page = document.pages.get(page_index);
        if !block_ids.insert(id)
            || page.is_none_or(|page| page.number != page_number)
            || bbox.iter().any(|value| !value.is_finite())
            || bbox[0] < 0.0
            || bbox[1] < 0.0
            || bbox[2] < bbox[0]
            || bbox[3] < bbox[1]
            || page.is_some_and(|page| bbox[2] > page.width + 0.01 || bbox[3] > page.height + 0.01)
        {
            return Err(Error::Message(format!(
                "document visual block {id} is invalid"
            )));
        }
    }
    let mut pair_ids = HashSet::new();
    for footnote in &document.footnotes {
        if !pair_ids.insert(&footnote.pair_id) {
            return Err(Error::Message(
                "document contains duplicate footnote pair IDs".to_owned(),
            ));
        }
        if footnote
            .reference_line_id
            .as_ref()
            .is_some_and(|line| !known_lines.contains(line))
            || footnote
                .body_line_ids
                .iter()
                .any(|line| !known_lines.contains(line))
        {
            return Err(Error::Message(format!(
                "footnote {} contains an unknown source line",
                footnote.pair_id
            )));
        }
    }
    let mut paragraph_ids = HashSet::new();
    for paragraph in &document.paragraphs {
        if !paragraph_ids.insert(&paragraph.id)
            || paragraph
                .line_ids
                .iter()
                .any(|line| !known_lines.contains(line))
        {
            return Err(Error::Message(format!(
                "paragraph {} is invalid",
                paragraph.id
            )));
        }
    }
    if document.structure_graph.document_id != document.document_id
        || document.structure_graph.source_sha256.as_deref() != Some(&document.source_sha256)
    {
        return Err(Error::Message(
            "structure graph identity disagrees with the document".to_owned(),
        ));
    }
    let text_length = document
        .pages
        .iter()
        .map(|page| {
            page.lines
                .iter()
                .map(|line| line.text.encode_utf16().count())
                .sum::<usize>()
        })
        .sum::<usize>()
        + document.line_count().saturating_sub(1);
    let mut node_ids = HashSet::new();
    for node in &document.structure_graph.nodes {
        if node.id.is_empty()
            || !node_ids.insert(node.id.as_str())
            || node.range.start > node.range.end
            || node.range.end > text_length
            || node.line_ids.iter().any(|line| !known_lines.contains(line))
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
    if document.structure_graph.nodes.iter().any(|node| {
        node.parent_id
            .as_deref()
            .is_some_and(|parent| !node_ids.contains(parent))
    }) {
        return Err(Error::Message(
            "structure graph contains an unknown parent".to_owned(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use legal_pdf_core::model::Word;

    #[test]
    fn pdf_text_index_preserves_exact_lines_and_scalar_offsets() {
        let pages: Vec<Page> = serde_json::from_value(json!([
            {
                "id": "p2",
                "index": 1,
                "number": 2,
                "width": 100.0,
                "height": 100.0,
                "lines": [{
                    "id": "excluded",
                    "page_index": 1,
                    "page_number": 2,
                    "source_index": 0,
                    "reading_order": 0,
                    "block_index": 0,
                    "text": "EXCLUDED",
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                    "exclude_from_body": true
                }],
                "regions": []
            },
            {
                "id": "p1",
                "index": 0,
                "number": 1,
                "width": 100.0,
                "height": 100.0,
                "lines": [
                    {
                        "id": "unicode",
                        "page_index": 0,
                        "page_number": 1,
                        "source_index": 1,
                        "reading_order": 1,
                        "block_index": 0,
                        "text": "\u{1f600}e\u{301}",
                        "bbox": [0.0, 10.0, 10.0, 20.0]
                    },
                    {
                        "id": "alpha",
                        "page_index": 0,
                        "page_number": 1,
                        "source_index": 0,
                        "reading_order": 0,
                        "block_index": 0,
                        "text": "\u{3b1}",
                        "bbox": [0.0, 0.0, 10.0, 10.0]
                    }
                ],
                "regions": []
            }
        ]))
        .unwrap();

        let index = PdfTextIndex::from_pages(&pages);

        assert_eq!(index.text(), "\u{3b1}\n\u{1f600}e\u{301}\nEXCLUDED");
        assert_eq!(
            index.line("alpha").unwrap().range,
            ScalarRange { start: 0, end: 1 }
        );
        assert_eq!(
            index.line("unicode").unwrap().range,
            ScalarRange { start: 2, end: 5 }
        );
        assert_eq!(
            index.global_range("unicode", 0, 1),
            Some(ScalarRange { start: 2, end: 3 })
        );
        assert_eq!(
            index.line_ids(ScalarRange { start: 0, end: 5 }),
            ["alpha", "unicode"]
        );
        assert_eq!(index.page_range(0), Some(ScalarRange { start: 0, end: 5 }));
        assert_eq!(index.page_range(1), Some(ScalarRange { start: 6, end: 14 }));
    }

    #[test]
    fn pdf_adapter_maps_raw_numeric_candidates_to_exact_line_ids_once() {
        let pages: Vec<Page> = serde_json::from_value(json!([{
            "id": "page-1",
            "index": 0,
            "number": 1,
            "width": 600.0,
            "height": 800.0,
            "lines": [
                {
                    "id": "line-1",
                    "page_index": 0,
                    "page_number": 1,
                    "source_index": 0,
                    "reading_order": 0,
                    "block_index": 0,
                    "text": "1. First paragraph has prose.",
                    "bbox": [72.0, 100.0, 400.0, 112.0],
                    "region_type": "body"
                },
                {
                    "id": "line-2",
                    "page_index": 0,
                    "page_number": 1,
                    "source_index": 1,
                    "reading_order": 1,
                    "block_index": 1,
                    "text": "2. Second paragraph has prose.",
                    "bbox": [72.0, 120.0, 410.0, 132.0],
                    "region_type": "body"
                }
            ],
            "regions": []
        }]))
        .unwrap();

        let adapter = PdfResolutionInput::from_pages(&pages, &PdfPrimitiveEvidence::default());
        let run = adapter
            .runs
            .iter()
            .find(|run| run.grammar == CandidateGrammar::Numeric)
            .expect("numeric candidate run");
        let mapped = run
            .markers
            .iter()
            .map(|candidate| {
                adapter
                    .evidence
                    .iter()
                    .find(|evidence| evidence.candidate_id == candidate.id)
                    .expect("mapped candidate")
            })
            .collect::<Vec<_>>();

        assert_eq!(mapped[0].line_ids, ["line-1"]);
        assert_eq!(mapped[1].line_ids, ["line-2"]);
        assert!(mapped.iter().all(|evidence| evidence
            .observations
            .contains(&CandidateObservationV2::BodyProseFlow)));
    }

    #[test]
    fn pdf_adapter_abstains_on_contents_rows_and_transcript_line_columns() {
        assert!(contents_row("1.1 Background ........ 3"));
        assert!(!contents_row("1.1 Background and application"));

        let mut contents = test_page(
            (1..=3)
                .map(|number| {
                    let mut line = test_line(
                        &format!("{number}. Topic heading ........ {}", number + 4),
                        [72.0, 80.0 + f64::from(number) * 20.0, 500.0, 94.0],
                        vec![],
                    );
                    line.region_type = "body".to_owned();
                    line
                })
                .collect(),
        );
        contents.index = 7;
        contents.number = 8;
        for line in &mut contents.lines {
            line.page_index = 7;
            line.page_number = 8;
        }
        let contents_adapter =
            PdfResolutionInput::from_pages(&[contents], &PdfPrimitiveEvidence::default());
        assert!(contents_adapter.evidence.iter().any(|item| item
            .observations
            .contains(&CandidateObservationV2::ContentsRow)));

        let mut transcript = test_page(
            (1..=25)
                .map(|number| {
                    test_line(
                        &number.to_string(),
                        [112.0, 60.0 + f64::from(number) * 20.0, 140.0, 74.0],
                        vec![],
                    )
                })
                .chain((1..=25).map(|number| {
                    let mut line = test_line(
                        &format!("Counsel continues speaking on transcript line {number}."),
                        [154.0, 60.0 + f64::from(number) * 20.0, 500.0, 74.0],
                        vec![],
                    );
                    line.region_type = "body".to_owned();
                    line
                }))
                .collect(),
        );
        transcript.index = 3;
        transcript.number = 4;
        for line in &mut transcript.lines {
            line.page_index = 3;
            line.page_number = 4;
        }
        let transcript_adapter =
            PdfResolutionInput::from_pages(&[transcript], &PdfPrimitiveEvidence::default());
        assert!(!transcript_adapter.evidence.is_empty());
        assert!(transcript_adapter.evidence.iter().all(|item| item
            .observations
            .contains(&CandidateObservationV2::TranscriptLineNumber)));

        let index = test_page(
            (1..=5)
                .flat_map(|number| {
                    [
                        test_line(
                            &format!("term-{number}"),
                            [72.0, 80.0 + f64::from(number) * 20.0, 150.0, 94.0],
                            vec![],
                        ),
                        test_line(
                            &format!("[{number}] {}:{}", number + 20, number + 1),
                            [180.0, 80.0 + f64::from(number) * 20.0, 300.0, 94.0],
                            vec![],
                        ),
                    ]
                })
                .collect(),
        );
        assert!(index_pages(&[index]).contains(&0));
    }

    #[test]
    fn typed_note_pairs_keep_every_exact_reference_anchor() {
        let pages: Vec<Page> = serde_json::from_value(json!([{
            "id": "page-1",
            "index": 0,
            "number": 1,
            "width": 600.0,
            "height": 800.0,
            "lines": [
                {
                    "id": "reference",
                    "page_index": 0,
                    "page_number": 1,
                    "source_index": 0,
                    "reading_order": 0,
                    "block_index": 0,
                    "text": "x¹ y¹",
                    "bbox": [72.0, 100.0, 200.0, 112.0]
                },
                {
                    "id": "label",
                    "page_index": 0,
                    "page_number": 1,
                    "source_index": 1,
                    "reading_order": 1,
                    "block_index": 1,
                    "text": "1 Note body",
                    "bbox": [72.0, 700.0, 300.0, 712.0]
                }
            ],
            "regions": []
        }]))
        .unwrap();
        let index = PdfTextIndex::from_pages(&pages);
        let (pairs, diagnostics) = map_note_pairs(
            &index,
            &[NotePairClaim {
                pair_id: "pair-1".to_owned(),
                label: "1".to_owned(),
                kind: NotePairKind::Footnote,
                label_anchor: legal_pdf_core::SourceAnchor {
                    line_id: "label".to_owned(),
                    start: 0,
                    end: 1,
                },
                reference_anchors: vec![
                    legal_pdf_core::SourceAnchor {
                        line_id: "reference".to_owned(),
                        start: 1,
                        end: 2,
                    },
                    legal_pdf_core::SourceAnchor {
                        line_id: "reference".to_owned(),
                        start: 4,
                        end: 5,
                    },
                ],
                body_line_ids: vec!["label".to_owned()],
            }],
        )
        .unwrap();

        assert!(diagnostics.is_empty());
        assert_eq!(pairs[0].label.range, ScalarRange { start: 6, end: 7 });
        assert_eq!(pairs[0].references.len(), 2);
        assert_eq!(
            pairs[0]
                .references
                .iter()
                .map(|reference| reference.range)
                .collect::<Vec<_>>(),
            [
                ScalarRange { start: 1, end: 2 },
                ScalarRange { start: 4, end: 5 }
            ]
        );
        assert_eq!(pairs[0].body.line_ids, ["label"]);
    }

    #[test]
    fn incomplete_pairer_products_abstain_without_losing_the_footnote_product() {
        let pages: Vec<Page> = serde_json::from_value(json!([{
            "id": "page-1", "index": 0, "number": 1, "width": 600.0, "height": 800.0,
            "lines": [
                {"id":"reference","page_index":0,"page_number":1,"source_index":0,"reading_order":0,"block_index":0,"text":"1","bbox":[0.0,0.0,1.0,1.0]},
                {"id":"empty","page_index":0,"page_number":1,"source_index":1,"reading_order":1,"block_index":1,"text":"","bbox":[0.0,2.0,1.0,3.0]},
                {"id":"body","page_index":0,"page_number":1,"source_index":2,"reading_order":2,"block_index":2,"text":"1 body","bbox":[0.0,4.0,10.0,5.0]}
            ], "regions": []
        }])).unwrap();
        let index = PdfTextIndex::from_pages(&pages);
        let mut pairs = (0..346)
            .map(|number| NotePairClaim {
                pair_id: format!("no-body-{number:03}"),
                label: "1".to_owned(),
                kind: NotePairKind::Footnote,
                label_anchor: legal_pdf_core::SourceAnchor {
                    line_id: "body".to_owned(),
                    start: 0,
                    end: 1,
                },
                reference_anchors: vec![legal_pdf_core::SourceAnchor {
                    line_id: "reference".to_owned(),
                    start: 0,
                    end: 1,
                }],
                body_line_ids: Vec::new(),
            })
            .collect::<Vec<_>>();
        pairs.push(NotePairClaim {
            pair_id: "zero-label".to_owned(),
            label: "1".to_owned(),
            kind: NotePairKind::Footnote,
            label_anchor: legal_pdf_core::SourceAnchor {
                line_id: "empty".to_owned(),
                start: 0,
                end: 0,
            },
            reference_anchors: vec![legal_pdf_core::SourceAnchor {
                line_id: "reference".to_owned(),
                start: 0,
                end: 1,
            }],
            body_line_ids: vec!["body".to_owned()],
        });
        pairs.push(NotePairClaim {
            pair_id: "zero-reference".to_owned(),
            label: "1".to_owned(),
            kind: NotePairKind::Footnote,
            label_anchor: legal_pdf_core::SourceAnchor {
                line_id: "body".to_owned(),
                start: 0,
                end: 1,
            },
            reference_anchors: vec![legal_pdf_core::SourceAnchor {
                line_id: "empty".to_owned(),
                start: 0,
                end: 0,
            }],
            body_line_ids: vec!["body".to_owned()],
        });

        let (claims, diagnostics) = map_note_pairs(&index, &pairs).unwrap();

        assert!(claims.is_empty());
        assert_eq!(diagnostics.len(), 348);
        assert!(diagnostics
            .iter()
            .all(|item| item.code == "note_pair_unmaterialized"));
        assert_eq!(diagnostics[346].candidate_ids, ["zero-label"]);
        assert_eq!(diagnostics[347].candidate_ids, ["zero-reference"]);
    }

    fn test_line(text: &str, bbox: [f64; 4], spans: Vec<Span>) -> Line {
        Line {
            id: String::new(),
            page_index: 0,
            page_number: 1,
            source_index: 0,
            reading_order: 0,
            block_index: 0,
            text: text.to_owned(),
            bbox,
            spans,
            words: vec![],
            detached_references: vec![],
            exclude_from_body: false,
            suppress_footnote_label: false,
            note_region_mode: String::new(),
            region_id: String::new(),
            region_type: "unknown".to_owned(),
            source: "native".to_owned(),
        }
    }

    fn sized_line(text: &str, bbox: [f64; 4], size: f64) -> Line {
        test_line(
            text,
            bbox,
            vec![Span {
                id: String::new(),
                text: text.to_owned(),
                bbox,
                font: String::new(),
                size,
                flags: 0,
                superscript: false,
                start: 0,
                end: text.chars().count(),
            }],
        )
    }

    #[test]
    fn paragraph_join_consumes_every_source_hyphen_marker() {
        for marker in ['-', '\u{00ad}', '\u{00ac}'] {
            let first = test_line(&format!("judg{marker}"), [0.0, 0.0, 10.0, 10.0], vec![]);
            let second = test_line("ment", [0.0, 12.0, 10.0, 22.0], vec![]);
            assert_eq!(join_lines(&[&first, &second]).0, "judgment");
        }
    }

    fn test_page(mut lines: Vec<Line>) -> Page {
        for (index, line) in lines.iter_mut().enumerate() {
            line.id = format!("p0001-l{:04}", index + 1);
            line.source_index = index + 1;
            line.reading_order = index + 1;
        }
        Page {
            id: "p0001".to_owned(),
            index: 0,
            number: 1,
            width: 600.0,
            height: 800.0,
            lines,
            regions: vec![],
            source: "native".to_owned(),
            text_quality: 1.0,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        }
    }

    fn mark_source_body(lines: &mut [Line]) {
        for line in lines {
            line.region_type = "body".to_owned();
        }
    }

    #[test]
    fn note_order_puts_a_raised_label_before_its_nearby_text() {
        for marker in ["17", "**"] {
            let mut label = test_line(marker, [39.8, 420.55, 49.8, 430.56], vec![]);
            label.words.push(Word {
                id: String::new(),
                text: marker.to_owned(),
                bbox: [39.8, 420.61, 49.8, 426.72],
                start: 0,
                end: marker.chars().count(),
            });
            let body = test_line(
                "Dominique Moran, Carceral Geography",
                [57.5, 420.50, 363.7, 430.56],
                vec![],
            );
            let mut lines = vec![body, label];

            order_note_lines(&mut lines, 612.0);

            assert_eq!(lines[0].text, marker);
        }
    }

    #[test]
    fn note_order_puts_a_detached_label_before_a_distant_same_row_fragment() {
        let mut label = test_line("68", [95.8, 554.3, 105.9, 566.1], vec![]);
        label.words.push(Word {
            id: String::new(),
            text: "68".to_owned(),
            bbox: [95.8, 555.3, 105.9, 562.0],
            start: 0,
            end: 2,
        });
        let body = test_line("of", [411.7, 554.3, 422.3, 566.1], vec![]);
        let mut lines = vec![body, label];

        order_note_lines(&mut lines, 612.0);

        assert_eq!(lines[0].text, "68");
    }

    #[test]
    fn note_order_reads_two_columns_column_by_column() {
        let mut lines = Vec::new();
        for row in 0..3 {
            lines.push(test_line(
                &format!("right {row}"),
                [
                    340.0,
                    400.0 + row as f64 * 12.0,
                    540.0,
                    410.0 + row as f64 * 12.0,
                ],
                vec![],
            ));
            lines.push(test_line(
                &format!("left {row}"),
                [
                    70.0,
                    400.0 + row as f64 * 12.0,
                    270.0,
                    410.0 + row as f64 * 12.0,
                ],
                vec![],
            ));
        }

        order_note_lines(&mut lines, 612.0);

        assert_eq!(
            lines
                .iter()
                .map(|line| line.text.as_str())
                .collect::<Vec<_>>(),
            ["left 0", "left 1", "left 2", "right 0", "right 1", "right 2"]
        );
    }

    #[test]
    fn note_number_margin_is_not_a_text_column() {
        let mut lines = Vec::new();
        for row in 0..6 {
            let y = 400.0 + row as f64 * 12.0;
            lines.push(test_line(
                "citation body",
                [105.0, y, 405.0, y + 10.0],
                vec![],
            ));
            lines.push(test_line(
                &(row + 1).to_string(),
                [50.0, y, 60.0, y + 8.0],
                vec![],
            ));
        }
        assert_eq!(column_model(&lines, 612.0).kind, "margin_column");

        order_note_lines(&mut lines, 612.0);

        for (row, pair) in lines.chunks_exact(2).enumerate() {
            assert_eq!(pair[0].text, (row + 1).to_string());
            assert_eq!(pair[1].text, "citation body");
        }
    }

    #[test]
    fn hanging_citation_fragments_are_not_a_second_note_column() {
        let mut lines = vec![
            test_line("43", [95.0, 100.0, 109.0, 110.0], vec![]),
            test_line("Fashion ID GmbH", [275.0, 100.0, 422.0, 110.0], vec![]),
            test_line(
                "continuation across the row",
                [95.0, 112.0, 422.0, 122.0],
                vec![],
            ),
            test_line("See also", [95.0, 124.0, 132.0, 134.0], vec![]),
            test_line("Wirtschaftsakademie", [249.0, 124.0, 422.0, 134.0], vec![]),
            test_line("C-", [95.0, 136.0, 105.0, 146.0], vec![]),
            test_line("Jehovan todistajat", [340.0, 136.0, 422.0, 146.0], vec![]),
            test_line("C-25/17", [95.0, 148.0, 205.0, 158.0], vec![]),
        ];

        assert_ne!(
            column_model_with_furniture(&lines, 600.0, false).kind,
            "two_column"
        );
        order_note_lines(&mut lines, 600.0);

        assert_eq!(
            lines
                .iter()
                .map(|line| line.text.as_str())
                .collect::<Vec<_>>(),
            [
                "43",
                "Fashion ID GmbH",
                "continuation across the row",
                "See also",
                "Wirtschaftsakademie",
                "C-",
                "Jehovan todistajat",
                "C-25/17",
            ]
        );
    }

    #[test]
    fn column_model_ignores_far_edge_furniture() {
        let mut lines = Vec::new();
        for row in 0..3 {
            lines.push(test_line(
                "left",
                [
                    70.0,
                    100.0 + row as f64 * 12.0,
                    270.0,
                    110.0 + row as f64 * 12.0,
                ],
                vec![],
            ));
            lines.push(test_line(
                "right",
                [
                    340.0,
                    100.0 + row as f64 * 12.0,
                    540.0,
                    110.0 + row as f64 * 12.0,
                ],
                vec![],
            ));
        }
        let mut page_number = test_line("27", [580.0, 760.0, 595.0, 770.0], vec![]);
        page_number.region_type = "footer".to_owned();
        lines.push(page_number);

        assert_eq!(column_model(&lines, 612.0).kind, "two_column");
    }

    #[test]
    fn column_model_prefers_a_clear_page_gutter_over_larger_internal_gaps() {
        let mut lines = Vec::new();
        for row in 0..8 {
            let y = 100.0 + row as f64 * 12.0;
            if row < 4 {
                lines.push(test_line("left text", [54.0, y, 380.0, y + 10.0], vec![]));
            }
            lines.push(test_line("page", [408.0, y, 422.0, y + 10.0], vec![]));
            lines.push(test_line("right text", [540.0, y, 900.0, y + 10.0], vec![]));
            if row < 3 {
                lines.push(test_line("short", [540.0, y, 560.0, y + 10.0], vec![]));
            }
        }

        let model = column_model(&lines, 972.0);

        assert_eq!(model.kind, "two_column");
        assert!((450.0..520.0).contains(&model.split_x));
    }

    #[test]
    fn centered_title_furniture_does_not_hide_the_page_gutter() {
        let mut lines = vec![
            test_line("volume and page", [265.0, 30.0, 342.0, 40.0], vec![]),
            test_line("author", [275.0, 70.0, 335.0, 80.0], vec![]),
        ];
        for row in 0..5 {
            let y = 100.0 + row as f64 * 12.0;
            lines.push(test_line("left", [70.0, y, 290.0, y + 10.0], vec![]));
            lines.push(test_line("right", [324.0, y, 542.0, y + 10.0], vec![]));
        }

        let model = column_model_with_furniture(&lines, 612.0, true);

        assert_eq!(model.kind, "two_column");
    }

    #[test]
    fn table_grid_is_not_rewritten_as_columns() {
        let mut lines = vec![test_line(
            "Table 2. Results",
            [60.0, 70.0, 200.0, 80.0],
            vec![],
        )];
        for row in 0..4 {
            let y = 100.0 + row as f64 * 14.0;
            for (column, x) in [60.0, 180.0, 300.0, 420.0].into_iter().enumerate() {
                lines.push(test_line(
                    &format!("r{row}c{column}"),
                    [x, y, x + 50.0, y + 10.0],
                    vec![],
                ));
            }
        }

        assert_eq!(table_evidence(&lines, 600.0).lines.len(), 16);
        assert_eq!(
            arbitrate_body_order(&mut lines, 600.0, 800.0).reason,
            "table_grid"
        );
    }

    #[test]
    fn textual_table_caption_does_not_force_geometry_order() {
        let mut lines = vec![test_line(
            "Table 1. Contents",
            [60.0, 70.0, 200.0, 80.0],
            vec![],
        )];
        for row in 0..6 {
            let y = 100.0 + row as f64 * 14.0;
            lines.extend([
                test_line("I", [60.0, y, 70.0, y + 10.0], vec![]),
                test_line("Section title", [100.0, y, 280.0, y + 10.0], vec![]),
                test_line("Appendix", [360.0, y, 430.0, y + 10.0], vec![]),
            ]);
        }

        assert!(!contents_grid(&lines, 600.0));
    }

    #[test]
    fn margin_notes_do_not_cut_off_the_main_text_column() {
        let mut lines = Vec::new();
        for row in 0..12 {
            let y = 400.0 + row as f64 * 24.0;
            lines.push(sized_line(
                if row == 6 {
                    "1988 to a pair of inventors in the main text"
                } else {
                    "Main-column prose continues below the adjacent notes"
                },
                [145.0, y, 425.0, y + 10.0],
                9.0,
            ));
        }
        for row in 0..6 {
            let y = 500.0 + row as f64 * 24.0;
            lines.push(sized_line(
                &format!("{} Citation text", row + 1),
                [37.0, y, 110.0, y + 8.0],
                7.0,
            ));
        }
        lines.push(sized_line(
            "25 Margin citation",
            [427.0, 730.0, 505.0, 738.0],
            7.0,
        ));
        assert_eq!(column_model(&lines, 540.0).kind, "margin_column");
        let mut pages = vec![Page {
            id: "p0001".to_owned(),
            index: 0,
            number: 1,
            width: 540.0,
            height: 792.0,
            lines,
            regions: vec![],
            source: "native".to_owned(),
            text_quality: 1.0,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        }];

        classify_pages(&mut pages, &[None]);

        assert!(pages[0]
            .lines
            .iter()
            .filter(|line| line.text.starts_with("Main-column"))
            .all(|line| line.region_type == "body"));
        assert!(pages[0]
            .lines
            .iter()
            .find(|line| line.text.starts_with("1988"))
            .is_some_and(|line| line.region_type == "body"));
        assert!(pages[0]
            .lines
            .iter()
            .filter(|line| line.text.ends_with("Citation text"))
            .all(|line| line.region_type == "footnote"));
    }

    #[test]
    fn an_early_right_margin_note_lane_is_not_body_prose() {
        let mut lines = Vec::new();
        for row in 0..8 {
            let y = 250.0 + row as f64 * 32.0;
            lines.push(sized_line(
                "Main-column prose remains independent of its notes",
                [70.0, y, 410.0, y + 10.0],
                9.0,
            ));
        }
        for row in 0..5 {
            let y = 320.0 + row as f64 * 24.0;
            lines.push(sized_line(
                &format!("{} Margin citation", row + 20),
                [427.0, y, 505.0, y + 8.0],
                7.0,
            ));
        }
        let mut pages = vec![Page {
            id: "p0001".to_owned(),
            index: 0,
            number: 1,
            width: 540.0,
            height: 792.0,
            lines,
            regions: vec![],
            source: "native".to_owned(),
            text_quality: 1.0,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        }];

        classify_pages(&mut pages, &[None]);

        assert!(pages[0]
            .lines
            .iter()
            .filter(|line| line.text.starts_with("Main-column"))
            .all(|line| line.region_type == "body"));
        assert!(pages[0]
            .lines
            .iter()
            .filter(|line| line.text.ends_with("Margin citation"))
            .all(|line| line.region_type == "footnote"));
    }

    #[test]
    fn embedded_contents_locators_are_not_document_headings() {
        let mut pages = vec![Page {
            id: "p0001".to_owned(),
            index: 0,
            number: 1,
            width: 600.0,
            height: 800.0,
            lines: ["I", "II", "III", "A"]
                .into_iter()
                .enumerate()
                .map(|(row, label)| {
                    let y = 100.0 + row as f64 * 20.0;
                    sized_line(
                        &format!("{label}. Section title\u{2003}{}", row + 10),
                        [60.0, y, 500.0, y + 12.0],
                        10.0,
                    )
                })
                .collect(),
            regions: vec![],
            source: "native".to_owned(),
            text_quality: 1.0,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        }];

        classify_pages(&mut pages, &[None]);

        assert!(pages[0].lines.iter().all(|line| line.region_type == "body"));
    }

    #[test]
    fn contents_grid_does_not_scramble_prose_or_hide_an_author_note() {
        let mut lines = vec![
            sized_line("ARTICLE TITLE", [84.0, 30.0, 400.0, 44.0], 14.0),
            sized_line("AUTHOR", [84.0, 70.0, 180.0, 81.0], 10.0),
            sized_line("ABSTRACT", [84.0, 420.0, 140.0, 430.0], 8.0),
        ];
        for row in 0..4 {
            let y = 445.0 + row as f64 * 14.0;
            lines.push(sized_line(
                "The abstract is ordinary prose, not a heading.",
                [84.0, y, 420.0, y + 11.0],
                10.0,
            ));
        }
        for row in 0..6 {
            let y = 110.0 + row as f64 * 35.0;
            lines.extend([
                sized_line("I", [84.0, y, 94.0, y + 9.0], 8.0),
                sized_line("SECTION", [120.0, y, 220.0, y + 9.0], 8.0),
                sized_line(&(168 + row).to_string(), [404.0, y, 420.0, y + 9.0], 8.0),
            ]);
        }
        lines.extend([
            sized_line("*", [84.0, 570.0, 90.0, 579.0], 8.0),
            sized_line(
                "Author affiliation and acknowledgments",
                [102.0, 570.0, 400.0, 579.0],
                8.0,
            ),
        ]);
        assert!(contents_grid(&lines, 486.0));
        let mut pages = vec![Page {
            id: "p0001".to_owned(),
            index: 0,
            number: 1,
            width: 486.0,
            height: 702.0,
            lines,
            regions: vec![],
            source: "native".to_owned(),
            text_quality: 1.0,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        }];

        classify_pages(&mut pages, &[Some(220.0)]);

        let order = |text: &str| {
            pages[0]
                .lines
                .iter()
                .find(|line| line.text == text)
                .map(|line| line.reading_order)
                .unwrap()
        };
        assert!(order("SECTION") < order("ABSTRACT"));
        assert!(pages[0]
            .lines
            .iter()
            .filter(|line| line.text.starts_with("The abstract"))
            .all(|line| line.region_type == "body"));
        assert!(pages[0]
            .lines
            .iter()
            .find(|line| line.text == "*")
            .is_some_and(|line| line.region_type == "footnote"));
        assert!(pages[0]
            .lines
            .iter()
            .find(|line| line.text.starts_with("Author affiliation"))
            .is_some_and(|line| line.region_type == "footnote"));
    }

    #[test]
    fn two_column_note_prose_is_not_a_table_grid() {
        let mut lines = Vec::new();
        for row in 0..4 {
            let y = 100.0 + row as f64 * 14.0;
            lines.extend([
                test_line(&(row + 1).to_string(), [50.0, y, 60.0, y + 10.0], vec![]),
                test_line(
                    "A full citation body with enough prose to be a note",
                    [70.0, y, 280.0, y + 10.0],
                    vec![],
                ),
                test_line(&(row + 5).to_string(), [330.0, y, 340.0, y + 10.0], vec![]),
                test_line(
                    "Another complete citation body in the right column",
                    [350.0, y, 570.0, y + 10.0],
                    vec![],
                ),
            ]);
        }

        let evidence = table_evidence(&lines, 600.0);
        assert!(!evidence.strong());
        assert!(!evidence.continuation());
    }

    #[test]
    fn attached_note_sequence_is_not_a_table_grid() {
        let mut lines = Vec::new();
        for row in 0..8 {
            let y = 500.0 + row as f64 * 12.0;
            lines.extend([
                test_line(
                    &format!("{} Citation text", row + 91),
                    [60.0, y, 260.0, y + 10.0],
                    vec![],
                ),
                test_line("20", [330.0, y, 370.0, y + 10.0], vec![]),
                test_line("40", [430.0, y, 470.0, y + 10.0], vec![]),
            ]);
        }

        let evidence = table_evidence(&lines, 600.0);

        assert!(!strong_table_evidence(&evidence, &lines));
    }

    #[test]
    fn table_numbers_do_not_start_a_footnote_region() {
        let mut lines = vec![sized_line(
            "Table 2. Results",
            [60.0, 70.0, 200.0, 80.0],
            10.0,
        )];
        for row in 0..4 {
            let y = 100.0 + row as f64 * 14.0;
            let texts = [
                "Income".to_owned(),
                "Low".to_owned(),
                (30 + row).to_string(),
                (20 + row).to_string(),
            ];
            for (text, x) in texts.into_iter().zip([60.0, 160.0, 260.0, 360.0]) {
                lines.push(sized_line(&text, [x, y, x + 50.0, y + 9.0], 8.0));
            }
        }
        lines.push(sized_line(
            "† Table-specific note",
            [60.0, 160.0, 240.0, 169.0],
            8.0,
        ));
        lines.push(sized_line(
            "2004 study of the following issue",
            [60.0, 190.0, 240.0, 200.0],
            8.0,
        ));
        let mut pages = vec![test_page(lines)];

        classify_pages(&mut pages, &[Some(130.0)]);

        assert!(pages[0]
            .lines
            .iter()
            .filter(|line| line.text != "† Table-specific note")
            .all(|line| line.region_type != "footnote"));
        assert!(pages[0]
            .lines
            .iter()
            .find(|line| line.text == "† Table-specific note")
            .is_some_and(|line| line.region_type == "footnote" && !line.suppress_footnote_label));
        let order = |text: &str| {
            pages[0]
                .lines
                .iter()
                .find(|line| line.text == text)
                .map(|line| line.reading_order)
                .unwrap()
        };
        assert!(order("† Table-specific note") < order("2004 study of the following issue"));
        assert!(pages[0]
            .lines
            .iter()
            .find(|line| line.text == "2004 study of the following issue")
            .is_some_and(|line| line.region_type == "body"));
    }

    #[test]
    fn separator_keeps_an_unlabelled_note_continuation_with_the_notes() {
        let mut lines = (0..12)
            .map(|row| {
                sized_line(
                    "Ordinary main text continues above the footnotes",
                    [
                        100.0,
                        100.0 + row as f64 * 24.0,
                        480.0,
                        111.0 + row as f64 * 24.0,
                    ],
                    10.5,
                )
            })
            .collect::<Vec<_>>();
        lines.extend([
            sized_line(
                "Continuation of the preceding note below the separator",
                [130.0, 462.0, 480.0, 472.0],
                9.0,
            ),
            sized_line(
                "2) for fraudulent and wrongful trading",
                [130.0, 578.0, 390.0, 588.0],
                9.0,
            ),
        ]);
        for row in 0..4 {
            let y = 590.0 + row as f64 * 12.0;
            lines.push(sized_line(
                &format!("{} Citation", row + 28),
                [137.0, y, 300.0, y + 8.0],
                6.0,
            ));
        }
        let mut pages = vec![Page {
            id: "p0001".to_owned(),
            index: 0,
            number: 1,
            width: 612.0,
            height: 792.0,
            lines,
            regions: vec![],
            source: "native".to_owned(),
            text_quality: 1.0,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        }];

        let diagnostics = classify_pages(&mut pages, &[Some(451.0)]);

        assert!(pages[0]
            .lines
            .iter()
            .find(|line| line.text.starts_with("Continuation"))
            .is_some_and(|line| line.region_type == "footnote"));
        assert!(!diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "FOOTNOTE_REGION_UNCERTAIN"));
    }

    #[test]
    fn captioned_tables_continue_across_pages_without_becoming_notes() {
        let page = |index: usize, mut lines: Vec<Line>| {
            for line in &mut lines {
                line.page_index = index;
                line.page_number = (index + 1) as u32;
            }
            Page {
                id: format!("p{:04}", index + 1),
                index,
                number: (index + 1) as u32,
                width: 600.0,
                height: 800.0,
                lines,
                regions: vec![],
                source: "native".to_owned(),
                text_quality: 1.0,
                printed_label: None,
                printed_label_source: None,
                printed_label_line_id: None,
            }
        };
        let mut continuation = Vec::new();
        for row in 0..8 {
            let y = 100.0 + row as f64 * 14.0;
            continuation.extend([
                sized_line("Province", [60.0, y, 120.0, y + 9.0], 8.0),
                sized_line(&(20 + row).to_string(), [240.0, y, 280.0, y + 9.0], 8.0),
                sized_line(&(40 + row).to_string(), [360.0, y, 400.0, y + 9.0], 8.0),
            ]);
        }
        let mut sparse_cell = sized_line(
            "continued text in a tall cell",
            [360.0, 107.0, 500.0, 116.0],
            8.0,
        );
        sparse_cell.region_type = "footer".to_owned();
        continuation.push(sparse_cell);
        for row in 0..6 {
            let y = 600.0 + row as f64 * 12.0;
            continuation.extend([
                sized_line(&(row + 1).to_string(), [60.0, y, 70.0, y + 8.0], 7.0),
                sized_line("Genuine footnote", [85.0, y, 260.0, y + 8.0], 7.0),
            ]);
        }
        let mut pages = vec![
            page(0, {
                let mut first = vec![sized_line(
                    "Table 1: Provincial results",
                    [60.0, 580.0, 240.0, 590.0],
                    10.0,
                )];
                for row in 0..6 {
                    let y = 600.0 + row as f64 * 14.0;
                    first.extend([
                        sized_line("Province", [60.0, y, 120.0, y + 9.0], 8.0),
                        sized_line(&(20 + row).to_string(), [240.0, y, 280.0, y + 9.0], 8.0),
                        sized_line(&(40 + row).to_string(), [360.0, y, 400.0, y + 9.0], 8.0),
                    ]);
                }
                first
            }),
            page(1, continuation),
        ];

        classify_pages(&mut pages, &[None, Some(580.0)]);

        assert!(pages[1]
            .lines
            .iter()
            .find(|line| line.text == "continued text in a tall cell")
            .is_some_and(|line| line.region_type == "body" && line.note_region_mode.is_empty()));
        assert!(pages[1]
            .lines
            .iter()
            .filter(|line| line.text == "Genuine footnote")
            .all(|line| line.region_type == "footnote"));
    }

    #[test]
    fn bottom_footnotes_do_not_extend_a_table_onto_the_next_page() {
        let page = |index: usize, lines: Vec<Line>| Page {
            id: format!("p{:04}", index + 1),
            index,
            number: (index + 1) as u32,
            width: 600.0,
            height: 800.0,
            lines,
            regions: vec![],
            source: "native".to_owned(),
            text_quality: 1.0,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        };
        let mut second = (0..8)
            .map(|row| {
                sized_line(
                    "Ordinary prose on the page after a completed table",
                    [
                        60.0,
                        100.0 + row as f64 * 16.0,
                        500.0,
                        111.0 + row as f64 * 16.0,
                    ],
                    10.0,
                )
            })
            .collect::<Vec<_>>();
        for row in 0..6 {
            let y = 600.0 + row as f64 * 12.0;
            second.extend([
                sized_line(&(row + 1).to_string(), [60.0, y, 70.0, y + 8.0], 7.0),
                sized_line("Citation text", [85.0, y, 300.0, y + 8.0], 7.0),
            ]);
        }
        let mut pages = vec![
            page(
                0,
                vec![sized_line(
                    "Table 1: Results",
                    [60.0, 600.0, 240.0, 610.0],
                    10.0,
                )],
            ),
            page(1, second),
        ];

        classify_pages(&mut pages, &[None, Some(580.0)]);

        assert!(pages[1]
            .lines
            .iter()
            .filter(|line| line.text == "Citation text")
            .all(|line| line.region_type == "footnote"));
        assert!(pages[1]
            .lines
            .iter()
            .filter(|line| line.text.starts_with("Ordinary prose"))
            .all(|line| line.region_type == "body"));
    }

    #[test]
    fn a_table_ending_midpage_does_not_mark_the_next_page_as_a_continuation() {
        let page = |index: usize, lines: Vec<Line>| Page {
            id: format!("p{:04}", index + 1),
            index,
            number: (index + 1) as u32,
            width: 600.0,
            height: 800.0,
            lines,
            regions: vec![],
            source: "native".to_owned(),
            text_quality: 1.0,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        };
        let mut first = vec![sized_line(
            "Table 1: Results",
            [60.0, 70.0, 240.0, 80.0],
            10.0,
        )];
        for row in 0..6 {
            let y = 100.0 + row as f64 * 14.0;
            first.extend([
                sized_line("Province", [60.0, y, 120.0, y + 9.0], 8.0),
                sized_line(&(20 + row).to_string(), [240.0, y, 280.0, y + 9.0], 8.0),
                sized_line(&(40 + row).to_string(), [360.0, y, 400.0, y + 9.0], 8.0),
            ]);
        }
        let mut second = (0..8)
            .map(|row| {
                sized_line(
                    "Ordinary prose on the following page",
                    [
                        60.0,
                        100.0 + row as f64 * 16.0,
                        500.0,
                        111.0 + row as f64 * 16.0,
                    ],
                    10.0,
                )
            })
            .collect::<Vec<_>>();
        for row in 0..6 {
            let y = 600.0 + row as f64 * 12.0;
            second.extend([
                sized_line(&(row + 1).to_string(), [60.0, y, 70.0, y + 8.0], 7.0),
                sized_line("Footnote text", [85.0, y, 300.0, y + 8.0], 7.0),
            ]);
        }
        let mut pages = vec![page(0, first), page(1, second)];

        classify_pages(&mut pages, &[None, Some(580.0)]);

        assert!(pages[1]
            .lines
            .iter()
            .filter(|line| line.text == "Footnote text")
            .all(|line| line.region_type == "footnote"));
    }

    #[test]
    fn detached_drop_cap_moves_before_its_paragraph_line() {
        let mut lines = vec![
            sized_line("crucial opening line", [101.0, 360.0, 400.0, 372.0], 10.0),
            sized_line("continuation", [68.0, 374.0, 400.0, 386.0], 10.0),
            sized_line("A", [68.0, 350.0, 101.0, 407.0], 48.0),
        ];

        repair_drop_caps(&mut lines);

        assert_eq!(lines[0].text, "A");
        assert_eq!(lines[1].text, "crucial opening line");
    }

    #[test]
    fn two_column_pages_keep_each_columns_notes_after_its_body() {
        let lines = |prefix: &str, y: f64| {
            (0..3)
                .flat_map(|row| {
                    [
                        test_line(
                            &format!("{prefix} left {row}"),
                            [
                                70.0,
                                y + row as f64 * 12.0,
                                270.0,
                                y + 10.0 + row as f64 * 12.0,
                            ],
                            vec![],
                        ),
                        test_line(
                            &format!("{prefix} right {row}"),
                            [
                                340.0,
                                y + row as f64 * 12.0,
                                540.0,
                                y + 10.0 + row as f64 * 12.0,
                            ],
                            vec![],
                        ),
                    ]
                })
                .collect::<Vec<_>>()
        };
        let mut body = lines("body", 100.0);
        column_order(&mut body, 305.0);
        let mut notes: Vec<_> = (0..3)
            .flat_map(|row| {
                [
                    test_line(
                        &format!("note left {row}"),
                        [
                            70.0,
                            700.0 + row as f64 * 12.0,
                            270.0,
                            710.0 + row as f64 * 12.0,
                        ],
                        vec![],
                    ),
                    test_line(
                        &format!("note right {row}"),
                        [
                            340.0,
                            760.0 + row as f64 * 12.0,
                            540.0,
                            770.0 + row as f64 * 12.0,
                        ],
                        vec![],
                    ),
                ]
            })
            .collect();
        column_order(&mut notes, 305.0);

        let result = weave_note_columns(body, notes, 612.0);

        assert_eq!(
            result
                .iter()
                .map(|line| line.text.as_str())
                .collect::<Vec<_>>(),
            [
                "body left 0",
                "body left 1",
                "body left 2",
                "note left 0",
                "note left 1",
                "note left 2",
                "body right 0",
                "body right 1",
                "body right 2",
                "note right 0",
                "note right 1",
                "note right 2",
            ]
        );

        let mut body = lines("body", 100.0);
        column_order(&mut body, 305.0);
        let result = weave_note_columns(
            body,
            vec![
                test_line("*", [54.0, 700.0, 64.0, 710.0], vec![]),
                test_line("full-width note", [72.0, 700.0, 540.0, 710.0], vec![]),
            ],
            612.0,
        );
        assert_eq!(result[result.len() - 2].text, "*");
        assert_eq!(result[result.len() - 1].text, "full-width note");
    }

    #[test]
    fn labels_normalize_unicode_superscripts() {
        assert_eq!(normalize_label("⁰¹²"), "12");
        assert_eq!(label_prefix("  12. Note").unwrap().label, "12");
        assert_eq!(label_prefix("2024 decision").unwrap().label, "2024");
        assert!(line_start_label_prefix("12").is_none());
        assert_eq!(label_prefix("12").unwrap().label, "12");
        assert_eq!(label_prefix("**** Note").unwrap().label, "****");
        let embedded = line_start_label_prefix("2endnote 2This is a note").unwrap();
        assert_eq!(embedded.label, "2");
        assert_eq!(
            char_slice("2endnote 2This is a note", embedded.end, 25),
            "This is a note"
        );
        assert!(label_prefix("*Not a note").is_none());
        assert!(line_start_label_prefix("3.2. Good neighbours").is_none());
    }

    #[test]
    fn compact_note_bodies_are_not_repeated_footers() {
        let page = |index: usize, label: usize| {
            let mut body = test_line("Body prose", [72.0, 100.0, 300.0, 110.0], vec![]);
            body.spans.push(Span {
                id: String::new(),
                text: body.text.clone(),
                bbox: body.bbox,
                font: String::new(),
                size: 10.0,
                flags: 0,
                superscript: false,
                start: 0,
                end: body.text.chars().count(),
            });
            let text = format!("{label}. Ibid at {label}.");
            let mut note = test_line(&text, [72.0, 730.0, 170.0, 737.0], vec![]);
            note.spans.push(Span {
                id: String::new(),
                text: note.text.clone(),
                bbox: note.bbox,
                font: String::new(),
                size: 7.0,
                flags: 0,
                superscript: false,
                start: 0,
                end: note.text.chars().count(),
            });
            Page {
                id: format!("p{:04}", index + 1),
                index,
                number: u32::try_from(index + 1).unwrap(),
                width: 612.0,
                height: 792.0,
                lines: vec![body, note],
                regions: vec![],
                source: "native".to_owned(),
                text_quality: 1.0,
                printed_label: None,
                printed_label_source: None,
                printed_label_line_id: None,
            }
        };
        let mut pages = vec![page(0, 33), page(1, 46), page(2, 57)];

        mark_repeated_furniture(&mut pages);

        assert!(pages
            .iter()
            .all(|page| page.lines[1].region_type == "unknown"));
    }

    #[test]
    fn repeated_detached_citation_shortforms_remain_note_lines() {
        let mut pages = (0..4)
            .map(|index| {
                let mut body = sized_line(
                    "Ordinary article body establishes the document font.",
                    [72.0, 300.0, 500.0, 312.0],
                    10.0,
                );
                body.id = format!("body-{index}");
                let mut label =
                    sized_line(&(51 + index).to_string(), [50.0, 730.0, 56.0, 736.0], 5.0);
                label.id = format!("label-{index}");
                let mut note = sized_line("Ibid.", [72.0, 730.0, 92.0, 738.0], 8.0);
                note.id = format!("note-{index}");
                Page {
                    id: format!("p{:04}", index + 1),
                    index,
                    number: u32::try_from(index + 1).unwrap(),
                    width: 612.0,
                    height: 792.0,
                    lines: vec![body, label, note],
                    regions: vec![],
                    source: "native".to_owned(),
                    text_quality: 1.0,
                    printed_label: None,
                    printed_label_source: None,
                    printed_label_line_id: None,
                }
            })
            .collect::<Vec<_>>();

        mark_repeated_furniture(&mut pages);

        assert!(pages.iter().all(|page| page.lines[1..]
            .iter()
            .all(|line| line.region_type == "unknown")));
    }

    #[test]
    fn attached_top_note_labels_are_not_repeated_headers() {
        let page = |index: usize, label: usize| {
            let marker = test_line(&label.to_string(), [40.0, 70.0, 52.0, 80.0], vec![]);
            let body = test_line(
                if index == 0 {
                    "First note"
                } else {
                    "Second note"
                },
                [60.0, 70.0, 180.0, 82.0],
                vec![],
            );
            Page {
                id: format!("p{:04}", index + 1),
                index,
                number: u32::try_from(index + 1).unwrap(),
                width: 612.0,
                height: 792.0,
                lines: vec![marker, body],
                regions: vec![],
                source: "native".to_owned(),
                text_quality: 1.0,
                printed_label: None,
                printed_label_source: None,
                printed_label_line_id: None,
            }
        };
        let mut pages = vec![page(0, 41), page(1, 42)];

        mark_repeated_furniture(&mut pages);

        assert!(pages
            .iter()
            .all(|page| page.lines[0].region_type == "unknown"));
    }

    #[test]
    fn repeated_top_paragraph_enumerators_are_not_headers() {
        let mut pages = (0..5)
            .map(|index| Page {
                id: format!("p{:04}", index + 1),
                index,
                number: u32::try_from(index + 1).unwrap(),
                width: 612.0,
                height: 792.0,
                lines: vec![
                    sized_line(&format!("{}.", 18 + index), [40.0, 72.0, 52.0, 82.0], 10.0),
                    sized_line(
                        "Paragraph text begins on the same baseline.",
                        [60.0, 72.0, 500.0, 84.0],
                        10.0,
                    ),
                ],
                regions: vec![],
                source: "native".to_owned(),
                text_quality: 1.0,
                printed_label: None,
                printed_label_source: None,
                printed_label_line_id: None,
            })
            .collect::<Vec<_>>();

        mark_repeated_furniture(&mut pages);

        assert!(pages
            .iter()
            .all(|page| page.lines[0].region_type == "unknown"));
    }

    #[test]
    fn attached_page_numbers_remain_repeated_headers() {
        let page = |index: usize, number: usize| {
            let marker_text = number.to_string();
            let marker = sized_line(&marker_text, [40.0, 40.0, 52.0, 50.0], 8.0);
            let heading = sized_line("Journal title", [60.0, 40.0, 180.0, 52.0], 8.0);
            Page {
                id: format!("p{:04}", index + 1),
                index,
                number: u32::try_from(index + 1).unwrap(),
                width: 612.0,
                height: 792.0,
                lines: vec![marker, heading],
                regions: vec![],
                source: "native".to_owned(),
                text_quality: 1.0,
                printed_label: None,
                printed_label_source: None,
                printed_label_line_id: None,
            }
        };
        let mut pages = vec![page(0, 240), page(1, 242)];

        mark_repeated_furniture(&mut pages);

        assert!(pages
            .iter()
            .all(|page| page.lines.iter().all(|line| line.region_type == "header")));
    }

    #[test]
    fn repeated_edge_text_does_not_sweep_in_a_geometry_outlier() {
        let mut pages = (0..4)
            .map(|index| {
                let top = if index == 3 { 70.0 } else { 20.0 };
                let mut header =
                    sized_line("ALBERTA LAW REVIEW", [100.0, top, 400.0, top + 10.0], 8.0);
                header.id = format!("header-{index}");
                let mut body = sized_line(
                    "Unique body prose remains body evidence.",
                    [72.0, 300.0, 500.0, 312.0],
                    10.0,
                );
                body.id = format!("body-{index}");
                Page {
                    id: format!("p{:04}", index + 1),
                    index,
                    number: u32::try_from(index + 1).unwrap(),
                    width: 612.0,
                    height: 792.0,
                    lines: vec![header, body],
                    regions: vec![],
                    source: "native".to_owned(),
                    text_quality: 1.0,
                    printed_label: None,
                    printed_label_source: None,
                    printed_label_line_id: None,
                }
            })
            .collect::<Vec<_>>();

        mark_repeated_furniture(&mut pages);

        assert!(pages[..3]
            .iter()
            .all(|page| page.lines[0].region_type == "header"));
        assert_eq!(pages[3].lines[0].region_type, "unknown");
    }

    #[test]
    fn repeated_edge_text_uses_the_stable_cluster_not_a_title_outlier() {
        let mut pages = (0..5)
            .map(|index| {
                let mut header = sized_line("CIRCULAR PRIORITIES", [100.0, 30.0, 300.0, 40.0], 8.0);
                header.id = format!("header-{index}");
                let mut lines = vec![header];
                if index == 0 {
                    let mut title =
                        sized_line("CIRCULAR PRIORITIES", [100.0, 70.0, 300.0, 84.0], 14.0);
                    title.id = "article-title".to_owned();
                    lines.push(title);
                }
                Page {
                    id: format!("p{:04}", index + 1),
                    index,
                    number: u32::try_from(index + 1).unwrap(),
                    width: 612.0,
                    height: 792.0,
                    lines,
                    regions: vec![],
                    source: "native".to_owned(),
                    text_quality: 1.0,
                    printed_label: None,
                    printed_label_source: None,
                    printed_label_line_id: None,
                }
            })
            .collect::<Vec<_>>();

        mark_repeated_furniture(&mut pages);

        assert!(pages
            .iter()
            .all(|page| page.lines[0].region_type == "header"));
        assert_eq!(pages[0].lines[1].region_type, "unknown");
    }

    #[test]
    fn alternating_sequential_bottom_folios_override_same_row_footer_text() {
        let mut pages = (0..5)
            .map(|index| {
                let x = if index % 2 == 0 { 570.0 } else { 20.0 };
                let mut folio =
                    sized_line(&(51 + index).to_string(), [x, 730.0, x + 20.0, 740.0], 8.0);
                folio.id = format!("folio-{index}");
                let mut footer = sized_line(
                    "Same-baseline journal footer",
                    [80.0, 730.0, 430.0, 740.0],
                    8.0,
                );
                footer.id = format!("footer-{index}");
                let mut body = sized_line(
                    "Body prose stays ordinary text.",
                    [72.0, 300.0, 500.0, 312.0],
                    10.0,
                );
                body.id = format!("body-{index}");
                Page {
                    id: format!("p{:04}", index + 1),
                    index,
                    number: u32::try_from(index + 1).unwrap(),
                    width: 612.0,
                    height: 792.0,
                    lines: vec![body, footer, folio],
                    regions: vec![],
                    source: "native".to_owned(),
                    text_quality: 1.0,
                    printed_label: None,
                    printed_label_source: None,
                    printed_label_line_id: None,
                }
            })
            .collect::<Vec<_>>();

        mark_repeated_furniture(&mut pages);

        assert!(pages
            .iter()
            .all(|page| page.lines[2].region_type == "footer"));
        assert!(pages
            .iter()
            .all(|page| page.lines[0].region_type == "unknown"));
    }

    #[test]
    fn propositions_remove_durable_markers() {
        let text = "First rule. Second rule⟦FN:pair⟧ continues.";
        assert_eq!(sentence_at(text, 23), "Second rule continues.");
        assert_eq!(
            sentence_at("It was so held.” Next point.", 18),
            "Next point."
        );
        assert_eq!(
            sentence_at("The proceeding ended.⟦FN:12⟧", 21),
            "The proceeding ended."
        );
        assert_eq!(sentence_at("R v X at para.20", 15), "R v X at para.20");
    }

    #[test]
    fn interleaved_columns_are_repaired_to_column_order() {
        let mut title = test_line("full-width title", [50.0, 20.0, 550.0, 30.0], vec![]);
        title.id = "title".to_owned();
        let mut author = test_line("author", [360.0, 50.0, 500.0, 60.0], vec![]);
        author.id = "author".to_owned();
        let mut lines = vec![title, author];
        for row in 0..6 {
            let y = 100.0 + row as f64 * 12.0;
            let mut left = test_line("left column prose", [60.0, y, 240.0, y + 10.0], vec![]);
            left.id = format!("left-{row}");
            let mut right = test_line("right column prose", [360.0, y, 540.0, y + 10.0], vec![]);
            right.id = format!("right-{row}");
            lines.extend([left, right]);
        }
        let decision = arbitrate_body_order(&mut lines, 600.0, 800.0);
        assert_eq!(decision.repair, OrderRepair::Column);
        assert_eq!(lines[0].id, "title");
        assert_eq!(lines[1].id, "author");
        assert!(lines[2..8].iter().all(|line| line.bbox[0] < 300.0));
        assert!(lines[8..].iter().all(|line| line.bbox[0] > 300.0));
    }

    #[test]
    fn misplaced_preamble_alone_does_not_justify_a_column_repair() {
        let mut lines = Vec::new();
        for (x, name) in [(60.0, "left"), (360.0, "right")] {
            for row in 0..3 {
                let y = 100.0 + row as f64 * 12.0;
                lines.push(test_line(name, [x, y, x + 180.0, y + 10.0], vec![]));
            }
        }
        lines.push(test_line("title", [50.0, 20.0, 550.0, 30.0], vec![]));
        lines.push(test_line("author", [360.0, 50.0, 500.0, 60.0], vec![]));

        let decision = arbitrate_body_order(&mut lines, 600.0, 800.0);

        assert_eq!(decision.repair, OrderRepair::None);
        assert_eq!(lines[0].text, "left");
    }

    #[test]
    fn endnotes_read_columns_in_sequence() {
        let mut lines = Vec::new();
        for row in 0..6 {
            let y = 100.0 + row as f64 * 12.0;
            let mut left = test_line(
                &format!("{} left endnote prose", row + 1),
                [60.0, y, 240.0, y + 10.0],
                vec![],
            );
            left.id = format!("left-{row}");
            left.region_type = "footnote".to_owned();
            left.note_region_mode = "endnote".to_owned();
            let mut right = test_line(
                &format!("{} right endnote prose", row + 7),
                [360.0, y, 540.0, y + 10.0],
                vec![],
            );
            right.id = format!("right-{row}");
            right.region_type = "footnote".to_owned();
            right.note_region_mode = "endnote".to_owned();
            lines.extend([left, right]);
        }
        let mut pages = vec![Page {
            id: "p0001".to_owned(),
            index: 0,
            number: 1,
            width: 600.0,
            height: 800.0,
            lines,
            regions: Vec::new(),
            source: "native".to_owned(),
            text_quality: 1.0,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        }];

        let diagnostics = order_pages(&mut pages);

        assert!(diagnostics.is_empty());
        assert_eq!(pages[0].lines[0].id, "left-0");
        assert_eq!(pages[0].lines[1].id, "left-1");
        assert_eq!(pages[0].lines[6].id, "right-0");
    }

    #[test]
    fn detached_reference_fits_the_word_gap_not_the_note_margin() {
        let host = test_line(
            "higher. They",
            [335.0, 93.9, 396.0, 103.9],
            vec![
                Span {
                    id: "left".to_owned(),
                    text: "higher.".to_owned(),
                    bbox: [335.0, 93.9, 366.6, 103.9],
                    font: String::new(),
                    size: 10.0,
                    flags: 0,
                    superscript: false,
                    start: 0,
                    end: 7,
                },
                Span {
                    id: "right".to_owned(),
                    text: "They".to_owned(),
                    bbox: [377.8, 93.9, 396.0, 103.9],
                    font: String::new(),
                    size: 10.0,
                    flags: 0,
                    superscript: false,
                    start: 8,
                    end: 12,
                },
            ],
        );
        let inline_marker = test_line("40", [367.5, 94.2, 373.2, 100.0], vec![]);
        let margin_label = test_line("40", [54.1, 94.2, 59.8, 100.0], vec![]);

        assert_eq!(
            detached_reference_target(0, &[inline_marker, host.clone()], 10.0),
            Some((1, 7))
        );
        assert_eq!(
            detached_reference_target(0, &[margin_label, host], 10.0),
            None
        );
    }

    #[test]
    fn endnote_mode_carries_to_the_next_numbered_note_page() {
        let mut first = test_line("1 First note", [60.0, 100.0, 300.0, 110.0], vec![]);
        first.region_type = "footnote".to_owned();
        first.note_region_mode = "endnote".to_owned();
        let mut second = test_line("2 Second note", [60.0, 100.0, 300.0, 110.0], vec![]);
        second.region_type = "footnote".to_owned();
        second.page_index = 1;
        second.page_number = 2;
        let mut pages = vec![
            Page {
                id: "p0001".to_owned(),
                index: 0,
                number: 1,
                width: 600.0,
                height: 800.0,
                lines: vec![first],
                regions: vec![],
                source: "native".to_owned(),
                text_quality: 1.0,
                printed_label: None,
                printed_label_source: None,
                printed_label_line_id: None,
            },
            Page {
                id: "p0002".to_owned(),
                index: 1,
                number: 2,
                width: 600.0,
                height: 800.0,
                lines: vec![second],
                regions: vec![],
                source: "native".to_owned(),
                text_quality: 1.0,
                printed_label: None,
                printed_label_source: None,
                printed_label_line_id: None,
            },
        ];

        infer_note_region_modes(&mut pages);

        assert_eq!(pages[1].lines[0].note_region_mode, "endnote");
    }

    #[test]
    fn endnote_heading_uses_a_separate_cut_for_each_column() {
        let mut body = test_line(
            "Article body before the notes",
            [60.0, 100.0, 280.0, 110.0],
            vec![],
        );
        body.source_index = 1;
        let mut notes = test_line("Notes", [60.0, 340.0, 120.0, 360.0], vec![]);
        notes.source_index = 2;
        let mut first = test_line("*", [60.0, 365.0, 70.0, 375.0], vec![]);
        first.source_index = 3;
        let first_body = test_line("First note", [80.0, 365.0, 280.0, 375.0], vec![]);
        let continuation = test_line(
            "Continuation from the prior note",
            [350.0, 105.0, 570.0, 115.0],
            vec![],
        );
        let eighth = test_line("8", [330.0, 130.0, 340.0, 140.0], vec![]);
        let eighth_body = test_line("Eighth note", [350.0, 130.0, 570.0, 140.0], vec![]);
        let right_tail = test_line("More note text", [350.0, 300.0, 570.0, 310.0], vec![]);
        let mut pages = vec![Page {
            id: "p0001".to_owned(),
            index: 0,
            number: 1,
            width: 612.0,
            height: 792.0,
            lines: vec![
                body,
                notes,
                first,
                first_body,
                continuation,
                eighth,
                eighth_body,
                right_tail,
            ],
            regions: vec![],
            source: "native".to_owned(),
            text_quality: 1.0,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        }];

        classify_pages(&mut pages, &[None]);

        let by_text: HashMap<_, _> = pages[0]
            .lines
            .iter()
            .map(|line| (line.text.as_str(), line))
            .collect();
        assert_eq!(by_text["Article body before the notes"].region_type, "body");
        assert!(by_text["Notes"].note_region_mode.is_empty());
        assert_eq!(by_text["Notes"].region_type, "heading");
        assert_eq!(
            by_text["Continuation from the prior note"].note_region_mode,
            "endnote"
        );
        assert_eq!(by_text["First note"].note_region_mode, "endnote");
    }

    #[test]
    fn an_early_body_number_does_not_turn_bottom_footnotes_into_endnotes() {
        let mut lines = vec![
            sized_line(
                "19. The ordinary paragraph continues",
                [60.0, 180.0, 400.0, 192.0],
                10.0,
            ),
            sized_line("More body prose", [60.0, 200.0, 400.0, 212.0], 10.0),
        ];
        for number in 13..=18 {
            let y = 560.0 + f64::from(number - 13) * 20.0;
            lines.push(sized_line(
                &format!("{number} Citation text"),
                [60.0, y, 400.0, y + 9.0],
                7.0,
            ));
        }
        let mut pages = vec![Page {
            id: "p0001".to_owned(),
            index: 0,
            number: 1,
            width: 600.0,
            height: 800.0,
            lines,
            regions: vec![],
            source: "native".to_owned(),
            text_quality: 1.0,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        }];

        classify_pages(&mut pages, &[None]);

        assert!(pages[0]
            .lines
            .iter()
            .filter(|line| line.region_type == "footnote")
            .all(|line| line.note_region_mode == "footnote"));
    }

    #[test]
    fn a_compact_lower_sequence_is_footnotes_without_a_drawn_rule() {
        let mut lines: Vec<_> = (0..10)
            .map(|row| {
                let y = 120.0 + row as f64 * 20.0;
                sized_line("Ordinary article body", [60.0, y, 500.0, y + 12.0], 10.0)
            })
            .collect();
        for number in 91..=95 {
            let y = 560.0 + f64::from(number - 91) * 20.0;
            if number == 93 {
                lines.push(sized_line(
                    "2021) 35 at 41).",
                    [60.0, y - 10.0, 180.0, y - 1.0],
                    8.5,
                ));
            }
            lines.push(sized_line(
                &format!("{number} Citation text"),
                [60.0, y, 500.0, y + 10.0],
                8.5,
            ));
        }
        let mut pages = vec![Page {
            id: "p0001".to_owned(),
            index: 0,
            number: 1,
            width: 600.0,
            height: 800.0,
            lines,
            regions: vec![],
            source: "native".to_owned(),
            text_quality: 1.0,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        }];

        classify_pages(&mut pages, &[None]);

        assert!(pages[0]
            .lines
            .iter()
            .filter(|line| line.text.starts_with('9'))
            .all(|line| line.region_type == "footnote" && line.note_region_mode == "footnote"));
    }

    #[test]
    fn a_single_lower_note_is_backed_by_its_superscript_reference() {
        let mut body = sized_line(
            "Ordinary article body with a reference",
            [60.0, 120.0, 500.0, 132.0],
            10.0,
        );
        body.spans.push(Span {
            id: String::new(),
            text: "21".to_owned(),
            bbox: [400.0, 116.0, 410.0, 124.0],
            font: String::new(),
            size: 6.0,
            flags: 0,
            superscript: true,
            start: body.text.chars().count(),
            end: body.text.chars().count(),
        });
        let mut pages = vec![Page {
            id: "p0001".to_owned(),
            index: 0,
            number: 1,
            width: 600.0,
            height: 800.0,
            lines: vec![
                body,
                sized_line(
                    "More ordinary article body",
                    [60.0, 140.0, 500.0, 152.0],
                    10.0,
                ),
                sized_line("Still more article body", [60.0, 160.0, 500.0, 172.0], 10.0),
                sized_line("21 Citation text", [60.0, 400.0, 500.0, 410.0], 8.5),
                sized_line("Citation continuation", [75.0, 412.0, 500.0, 422.0], 8.5),
            ],
            regions: vec![],
            source: "native".to_owned(),
            text_quality: 1.0,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        }];

        classify_pages(&mut pages, &[None]);

        assert!(pages[0].lines[3..]
            .iter()
            .all(|line| line.region_type == "footnote" && line.note_region_mode == "footnote"));
    }

    #[test]
    fn two_reference_backed_margin_notes_do_not_cut_through_main_prose() {
        let mut lines: Vec<_> = (0..10)
            .map(|row| {
                let y = 380.0 + row as f64 * 24.0;
                sized_line(
                    "Main-column prose continues beside the margin notes",
                    [145.0, y, 425.0, y + 10.0],
                    9.0,
                )
            })
            .collect();
        for (line, label) in lines.iter_mut().take(2).zip(["39", "40"]) {
            line.spans.push(Span {
                id: String::new(),
                text: label.to_owned(),
                bbox: [400.0, line.bbox[1] - 4.0, 410.0, line.bbox[1] + 4.0],
                font: String::new(),
                size: 6.0,
                flags: 0,
                superscript: true,
                start: line.text.chars().count(),
                end: line.text.chars().count(),
            });
        }
        lines.extend([
            sized_line("39 First margin note", [37.0, 500.0, 110.0, 508.0], 7.0),
            sized_line("40 Second margin note", [37.0, 550.0, 110.0, 558.0], 7.0),
        ]);
        let mut pages = vec![Page {
            id: "p0001".to_owned(),
            index: 0,
            number: 1,
            width: 540.0,
            height: 792.0,
            lines,
            regions: vec![],
            source: "native".to_owned(),
            text_quality: 1.0,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        }];

        classify_pages(&mut pages, &[None]);

        assert!(pages[0]
            .lines
            .iter()
            .filter(|line| line.text.starts_with("Main-column"))
            .all(|line| line.region_type == "body"));
        assert!(pages[0]
            .lines
            .iter()
            .filter(|line| line.text.starts_with("39 ") || line.text.starts_with("40 "))
            .all(|line| line.region_type == "footnote"));
    }

    #[test]
    fn smaller_quoted_text_does_not_make_normal_prose_a_heading() {
        let mut lines = Vec::new();
        for row in 0..20 {
            let y = 100.0 + row as f64 * 12.0;
            lines.push(sized_line(
                "Indented quoted passage",
                [90.0, y, 500.0, y + 10.0],
                9.0,
            ));
        }
        for row in 0..10 {
            let y = 350.0 + row as f64 * 14.0;
            lines.push(sized_line(
                "Normal narrative prose continues here.",
                [60.0, y, 520.0, y + 12.0],
                11.0,
            ));
        }
        lines.push(sized_line("IV. Rulings", [60.0, 510.0, 220.0, 526.0], 14.0));
        for (index, line) in lines.iter_mut().enumerate() {
            line.region_type = if index < 20 {
                "block_quote"
            } else if index < 30 {
                "body"
            } else {
                "heading"
            }
            .to_owned();
        }
        let mut pages = vec![test_page(lines)];

        classify_pages(&mut pages, &[None]);

        assert!(pages[0]
            .lines
            .iter()
            .filter(|line| line.text.starts_with("Normal"))
            .all(|line| line.region_type == "body"));
        assert_eq!(pages[0].lines.last().unwrap().region_type, "heading");
    }

    #[test]
    fn region_dependent_lanes_require_a_complete_source_contract() {
        let mut pages = vec![test_page(vec![
            sized_line("Known body", [60.0, 100.0, 300.0, 112.0], 11.0),
            sized_line("Unknown peer", [60.0, 120.0, 300.0, 132.0], 11.0),
        ])];
        pages[0].lines[0].region_type = "body".to_owned();
        assert!(!source_regions_available(&pages));

        pages[0].lines[1].region_type = "text".to_owned();
        assert!(source_regions_available(&pages));
    }

    #[test]
    fn source_roles_admit_display_headings_without_promoting_authors() {
        let mut pages = vec![test_page(vec![
            sized_line(
                "CONSTITUTIONAL PRINCIPLES",
                [60.0, 100.0, 360.0, 112.0],
                11.0,
            ),
            sized_line("JANE EXAMPLE", [60.0, 125.0, 240.0, 137.0], 11.0),
            sized_line(
                "Ordinary narrative text ends here.",
                [60.0, 160.0, 520.0, 172.0],
                11.0,
            ),
        ])];
        pages[0].lines[0].region_type = "text".to_owned();
        pages[0].lines[1].region_type = "author".to_owned();
        pages[0].lines[2].region_type = "text".to_owned();

        classify_pages(&mut pages, &[None]);

        assert_eq!(pages[0].lines[0].region_type, "heading");
        assert_eq!(pages[0].lines[1].region_type, "body");
    }

    #[test]
    fn clean_repeated_heading_grammar_promotes_nested_ladders_without_visual_tuning() {
        let mut lines = (0..12)
            .map(|row| {
                let y = 80.0 + row as f64 * 18.0;
                sized_line(
                    "Ordinary narrative text ends here.",
                    [60.0, y, 520.0, y + 12.0],
                    11.0,
                )
            })
            .collect::<Vec<_>>();
        for (row, text) in [
            "I. First Part",
            "A. First Issue",
            "B. Second Issue",
            "II. Second Part",
        ]
        .into_iter()
        .enumerate()
        {
            let y = 330.0 + row as f64 * 25.0;
            lines.push(sized_line(text, [60.0, y, 280.0, y + 12.0], 11.0));
        }
        mark_source_body(&mut lines);
        let mut pages = vec![test_page(lines)];

        classify_pages(&mut pages, &[None]);

        assert!(pages[0]
            .lines
            .iter()
            .filter(|line| line.text.contains("Part") || line.text.contains("Issue"))
            .all(|line| line.region_type == "heading"));
    }

    #[test]
    fn source_regions_allow_same_style_wrapped_heading_continuations() {
        let mut lines = (0..10)
            .map(|row| {
                let y = 80.0 + row as f64 * 18.0;
                sized_line(
                    "Ordinary narrative text ends here.",
                    [60.0, y, 520.0, y + 12.0],
                    11.0,
                )
            })
            .collect::<Vec<_>>();
        let mut heading = sized_line(
            "I. A Complete Account Of",
            [60.0, 300.0, 340.0, 312.0],
            12.0,
        );
        heading.spans[0].flags = 16;
        let mut continuation =
            sized_line("The Governing Framework", [80.0, 313.0, 360.0, 325.0], 12.0);
        continuation.spans[0].flags = 16;
        lines.extend([
            heading,
            continuation,
            sized_line(
                "Ordinary prose begins after the display heading.",
                [60.0, 345.0, 520.0, 357.0],
                11.0,
            ),
        ]);
        mark_source_body(&mut lines);
        let mut pages = vec![test_page(lines)];

        classify_pages(&mut pages, &[None]);

        let heading_lines = pages[0]
            .lines
            .iter()
            .filter(|line| line.text.starts_with("I.") || line.text == "The Governing Framework")
            .collect::<Vec<_>>();
        assert_eq!(heading_lines.len(), 2);
        assert!(heading_lines
            .iter()
            .all(|line| line.region_type == "heading"));
        assert_eq!(heading_lines[0].region_id, heading_lines[1].region_id);
    }

    #[test]
    fn dirty_heading_ladder_abstains_instead_of_promoting_examples() {
        let mut lines = (0..10)
            .map(|row| {
                let y = 80.0 + row as f64 * 18.0;
                sized_line(
                    "Ordinary narrative text ends here.",
                    [60.0, y, 520.0, y + 12.0],
                    11.0,
                )
            })
            .collect::<Vec<_>>();
        lines.push(sized_line(
            "I. First Part",
            [60.0, 300.0, 280.0, 312.0],
            11.0,
        ));
        lines.push(sized_line(
            "I. Duplicate Part",
            [60.0, 325.0, 300.0, 337.0],
            11.0,
        ));
        mark_source_body(&mut lines);
        let mut pages = vec![test_page(lines)];

        classify_pages(&mut pages, &[None]);

        assert!(pages[0]
            .lines
            .iter()
            .filter(|line| line.text.starts_with("I."))
            .all(|line| line.region_type == "body"));
    }

    #[test]
    fn long_numeric_ladder_is_not_promoted_as_document_headings() {
        let mut lines = (0..10)
            .map(|row| {
                let y = 80.0 + row as f64 * 18.0;
                sized_line(
                    "Ordinary narrative text ends here.",
                    [60.0, y, 520.0, y + 12.0],
                    11.0,
                )
            })
            .collect::<Vec<_>>();
        lines.push(sized_line(
            "15. Historical Note",
            [60.0, 300.0, 280.0, 312.0],
            11.0,
        ));
        lines.push(sized_line(
            "16. Further Note",
            [60.0, 325.0, 280.0, 337.0],
            11.0,
        ));
        mark_source_body(&mut lines);
        let mut pages = vec![test_page(lines)];

        classify_pages(&mut pages, &[None]);

        assert!(pages[0]
            .lines
            .iter()
            .filter(|line| line.text.starts_with("15.") || line.text.starts_with("16."))
            .all(|line| line.region_type != "heading"));
    }

    #[test]
    fn body_flow_vetoes_a_visually_bold_false_heading() {
        let mut lines = (0..10)
            .map(|row| {
                let y = 80.0 + row as f64 * 18.0;
                sized_line(
                    "Ordinary narrative text ends here.",
                    [60.0, y, 520.0, y + 12.0],
                    11.0,
                )
            })
            .collect::<Vec<_>>();
        let mut candidate = sized_line("I. This Is Actually", [60.0, 300.0, 280.0, 312.0], 11.0);
        candidate.spans[0].flags = 16;
        lines.push(candidate);
        lines.push(sized_line(
            "continued prose from the same sentence.",
            [60.0, 314.0, 520.0, 326.0],
            11.0,
        ));
        mark_source_body(&mut lines);
        let mut pages = vec![test_page(lines)];

        classify_pages(&mut pages, &[None]);

        assert_eq!(
            pages[0]
                .lines
                .iter()
                .find(|line| line.text.starts_with("I."))
                .unwrap()
                .region_type,
            "body"
        );
    }

    #[test]
    fn citation_and_destination_shapes_never_enter_the_heading_ladder() {
        let mut lines = (0..10)
            .map(|row| {
                let y = 80.0 + row as f64 * 18.0;
                sized_line(
                    "Ordinary narrative text ends here.",
                    [60.0, y, 520.0, y + 12.0],
                    11.0,
                )
            })
            .collect::<Vec<_>>();
        for (row, text) in ["I. Example v Sample 123", "II. Destination 42"]
            .into_iter()
            .enumerate()
        {
            let mut line = sized_line(
                text,
                [
                    60.0,
                    300.0 + row as f64 * 25.0,
                    320.0,
                    312.0 + row as f64 * 25.0,
                ],
                11.0,
            );
            line.spans[0].flags = 16;
            lines.push(line);
        }
        mark_source_body(&mut lines);
        let mut pages = vec![test_page(lines)];

        classify_pages(&mut pages, &[None]);

        assert!(pages[0]
            .lines
            .iter()
            .filter(|line| line.text.starts_with("I.") || line.text.starts_with("II."))
            .all(|line| line.region_type == "body"));
    }

    #[test]
    fn endnote_sequence_includes_column_continuations_above_the_next_label() {
        let mut first = test_line("1", [60.0, 100.0, 70.0, 110.0], vec![]);
        first.source_index = 2;
        let mut first_body = test_line("First note", [80.0, 100.0, 300.0, 110.0], vec![]);
        first_body.source_index = 3;
        let mut second_lines = Vec::new();
        for (label, x, y) in [
            (2, 60.0, 100.0),
            (3, 60.0, 200.0),
            (4, 60.0, 300.0),
            (5, 330.0, 100.0),
            (6, 330.0, 200.0),
            (7, 330.0, 300.0),
        ] {
            let mut marker = test_line(&label.to_string(), [x, y, x + 10.0, y + 10.0], vec![]);
            let mut body = test_line("Note body", [x + 20.0, y, x + 240.0, y + 10.0], vec![]);
            marker.page_index = 1;
            marker.page_number = 2;
            body.page_index = 1;
            body.page_number = 2;
            second_lines.extend([marker, body]);
        }
        let mut continuation = test_line(
            "Continuation from the prior column",
            [350.0, 70.0, 570.0, 80.0],
            vec![],
        );
        continuation.page_index = 1;
        continuation.page_number = 2;
        second_lines.insert(6, continuation);
        let mut pages = vec![
            Page {
                id: "p0001".to_owned(),
                index: 0,
                number: 1,
                width: 600.0,
                height: 800.0,
                lines: vec![
                    test_line("Notes", [60.0, 70.0, 120.0, 85.0], vec![]),
                    first,
                    first_body,
                ],
                regions: vec![],
                source: "native".to_owned(),
                text_quality: 1.0,
                printed_label: None,
                printed_label_source: None,
                printed_label_line_id: None,
            },
            Page {
                id: "p0002".to_owned(),
                index: 1,
                number: 2,
                width: 600.0,
                height: 800.0,
                lines: second_lines,
                regions: vec![],
                source: "native".to_owned(),
                text_quality: 1.0,
                printed_label: None,
                printed_label_source: None,
                printed_label_line_id: None,
            },
        ];

        classify_pages(&mut pages, &[None, Some(90.0)]);

        assert!(pages[1]
            .lines
            .iter()
            .all(|line| line.note_region_mode == "endnote"));
    }

    #[test]
    fn crossref_shortform_uses_python_word_boundaries_at_join_controls() {
        let text = "\u{200c}Quebec Water Policy, supra note 3";
        let start = text.find("supra").unwrap();
        assert_eq!(crossref_shortform(text, start), "Quebec Water Policy");

        let text = "\u{200c}Godin, supra note 41";
        let start = text.find("supra").unwrap();
        assert_eq!(crossref_shortform(text, start), "Godin");
    }
}
