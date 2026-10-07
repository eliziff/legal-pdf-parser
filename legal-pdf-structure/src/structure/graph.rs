use super::{
    arabic_page_number, body_flow_edge, scalar_suffix, sentence_ended, starts_note_or_list,
    PdfPrimitiveEvidence,
};
use legal_pdf_core::model::{Line, NotePairClaim, NotePairKind, Page, Paragraph, PdfSourceSpan};
use legal_pdf_core::structure_analysis;
use legal_pdf_core::{Error, Result};
use legal_pdf_support::protected_citation_spans;
use legal_structure_model::{
    CandidateEvidenceV2, CandidateGrammar, CandidateObservationV2, Derivation, DiagnosticSeverity,
    NodeKind, NoteBodyV2, NoteKindV2, NotePairClaimV2, ScalarRange, ScalarText,
    StructureCandidateRun, StructureDiagnostic, StructureMarkerCandidate, StructureNode,
    TextAnchorV2,
};
use regex::Regex;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::OnceLock;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct IndexedPdfLine {
    page_index: usize,
    line_id: String,
    pub(super) range: ScalarRange,
}

#[derive(Debug, Clone)]
pub struct PdfTextIndex {
    text: String,
    lines: Vec<IndexedPdfLine>,
    line_slots: HashMap<String, usize>,
    page_ranges: HashMap<usize, ScalarRange>,
}

#[derive(Debug)]
pub(super) struct PdfResolutionInput {
    pub(super) index: PdfTextIndex,
    pub(super) runs: Vec<StructureCandidateRun>,
    pub(super) evidence: Vec<CandidateEvidenceV2>,
    pub(super) citation_spans: BTreeMap<String, Vec<PdfSourceSpan>>,
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

    pub(super) fn line(&self, line_id: &str) -> Option<&IndexedPdfLine> {
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

    pub(super) fn page_range(&self, page_index: usize) -> Option<ScalarRange> {
        self.page_ranges.get(&page_index).copied()
    }

    pub(super) fn global_range(
        &self,
        line_id: &str,
        start: usize,
        end: usize,
    ) -> Option<ScalarRange> {
        let line = self.line(line_id)?;
        let length = line.range.end - line.range.start;
        (start <= end && end <= length).then_some(ScalarRange {
            start: line.range.start + start,
            end: line.range.start + end,
        })
    }

    pub(super) fn line_ids(&self, range: ScalarRange) -> Vec<String> {
        self.overlapping_lines(range)
            .iter()
            .map(|line| line.line_id.clone())
            .collect()
    }

    fn page_indexes(&self, range: ScalarRange) -> Vec<usize> {
        self.overlapping_lines(range)
            .iter()
            .fold(Vec::new(), |mut pages, line| {
                if pages.last() != Some(&line.page_index) {
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

pub(super) fn contents_row(text: &str) -> bool {
    contents_leader_re().is_match(text)
}

pub(super) fn contents_leader_re() -> &'static Regex {
    static LEADER: OnceLock<Regex> = OnceLock::new();
    // A leader runs to the line's end or to the locator that ends it: dots inside a line are an
    // ellipsis. A rule is a leader only before a locator; alone it is a form's blank.
    LEADER.get_or_init(|| {
        Regex::new(r"(?:(?:\. ){3,}|\.{4,})\s*\S{0,6}\s*$|_{4,}\s*\d{1,4}\s*$")
            .expect("contents leader regex")
    })
}

fn transcript_line_number_pages(pages: &[Page]) -> HashSet<usize> {
    pages
        .iter()
        .filter(|page| !line_number_column(page).is_empty())
        .map(|page| page.index)
        .collect()
}

/// A transcript's printed line numbers: a column of small numbers counting at least
/// 1 to 15 down the page. Returns the column's line slots, or none.
pub(super) fn line_number_column(page: &Page) -> Vec<usize> {
    const MIN_LINE_NUMBERS: u32 = 15;
    if page.width <= 0.0 {
        return Vec::new();
    }
    let mut candidates = page
        .lines
        .iter()
        .enumerate()
        .filter_map(|(slot, line)| {
            let number = arabic_page_number(&line.text)?;
            (number <= 40).then_some((line.bbox[0], number, slot))
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
        .map(|(_, number, _)| *number)
        .collect::<HashSet<_>>();
    if values.len() >= MIN_LINE_NUMBERS as usize
        && (1..=MIN_LINE_NUMBERS).all(|number| values.contains(&number))
    {
        best.iter().map(|(_, _, slot)| *slot).collect()
    } else {
        Vec::new()
    }
}

/// Word-index and concordance pages: short headwords in alphabetical order, each
/// followed by an occurrence count or `page:line` / `page/line` locators.
pub(super) fn index_pages(pages: &[Page]) -> HashSet<usize> {
    static COUNTED: OnceLock<Regex> = OnceLock::new();
    static HEADWORD: OnceLock<Regex> = OnceLock::new();
    let counted = COUNTED.get_or_init(|| {
        Regex::new(r"\[\d{1,3}\]\s+\d{1,4}[:/]\d{1,3}").expect("index entry regex")
    });
    let headword = HEADWORD.get_or_init(|| {
        Regex::new(
            r"^\s*(\S+(?:\s\S+){0,2}?)\s+(?:[\[(]\d{1,3}[\])]\s*$|(?:[\[(]\d{1,3}[\])]\s+)?\d{1,4}[:/]\d{1,3}(?:[\s,;]|$))",
        )
        .expect("index headword regex")
    });
    pages
        .iter()
        .filter_map(|page| {
            let text = page
                .lines
                .iter()
                .map(|line| line.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let headwords = page
                .lines
                .iter()
                .filter_map(|line| headword.captures(&line.text))
                .map(|capture| {
                    capture[1]
                        .trim_start_matches(['\'', '"', '\u{2018}', '\u{201c}'])
                        .to_lowercase()
                })
                .collect::<Vec<_>>();
            let ascending = headwords
                .windows(2)
                .filter(|pair| pair[0] <= pair[1])
                .count();
            (counted.find_iter(&text).take(5).count() >= 5
                || headwords.len() >= 5 && ascending * 5 >= (headwords.len() - 1) * 4)
                .then_some(page.index)
        })
        .collect()
}

/// A body line that can carry a numbered item's text: not a running head or folio, nor a note.
fn item_text_line(line: &Line) -> bool {
    !line.exclude_from_body
        && line.note_region_mode.is_empty()
        && !matches!(line.region_type.as_str(), "header" | "footer")
}

/// Whether a number opening a line is a sentence's words wrapped onto it ("under / Article
/// 40.4.2°", "made on / 11 August"): the body line before runs the full measure to the
/// text's right edge and stops short of a sentence's or a clause's end.
fn wrapped_into(
    index: &PdfTextIndex,
    by_line: &HashMap<&str, (&Page, &Line)>,
    measures: &HashMap<usize, (f64, f64)>,
    candidate: &StructureMarkerCandidate,
) -> bool {
    let before = index.overlapping_lines(ScalarRange {
        start: 0,
        end: candidate.marker_range.start,
    });
    let Some(prior) = before
        .iter()
        .rev()
        .filter(|indexed| indexed.range.end <= candidate.marker_range.start)
        .filter_map(|indexed| by_line.get(indexed.line_id.as_str()).map(|(_, line)| *line))
        .find(|line| item_text_line(line) && !line.text.trim().is_empty())
    else {
        return false;
    };
    let Some(&(left, right)) = measures.get(&prior.page_index) else {
        return false;
    };
    let text = prior.text.trim_end();
    let last = text.split_whitespace().last().unwrap_or_default();
    let height = (prior.bbox[3] - prior.bbox[1]).max(1.0);
    prior.bbox[0] < right - 0.85 * (right - left)
        && prior.bbox[2] >= right - 2.0 * height
        && text.ends_with(|c: char| c.is_alphanumeric())
        && !matches!(last, "and" | "or")
}

/// A number alone on its line whose next line is set out left of it is no item's: a folio, a
/// column of figures.
fn stands_apart(marker: &Line, next: &Line) -> bool {
    let below = next.bbox[1] >= marker.bbox[3] - 0.5 * (marker.bbox[3] - marker.bbox[1]);
    below && next.bbox[0] < marker.bbox[0] - (next.bbox[3] - next.bbox[1])
}

/// Where a subsection's text ends when its number stands apart from it (see `stands_apart`).
fn apart_end(
    index: &PdfTextIndex,
    by_line: &HashMap<&str, (&Page, &Line)>,
    candidate: &StructureMarkerCandidate,
) -> Option<usize> {
    let (_, marker) = by_line.get(
        index
            .line_at(candidate.marker_range.start)?
            .line_id
            .as_str(),
    )?;
    let (indexed, line) = index
        .overlapping_lines(candidate.range)
        .iter()
        .filter(|indexed| indexed.range.end > candidate.content_start)
        .filter_map(|indexed| Some((indexed, by_line.get(indexed.line_id.as_str())?.1)))
        .find(|(_, line)| item_text_line(line) && !line.text.trim().is_empty())?;
    stands_apart(marker, line).then_some(indexed.range.start)
}

/// Where a numbered item's text ends by the page's layout, when that is before the next number.
/// It ends at a heading, at a printed block of its own set apart from the item's margin and
/// the column its lines wrap to (a centred title, a signature, a cover page), and after a
/// finished sentence at a block in line with them (a jurat, an unnumbered paragraph). A block
/// set in at full measure (a quotation), one opening with its own marker, an unfinished
/// sentence, the text after a quotation and a full line at the top of the next page carry the
/// item on.
fn layout_end(
    index: &PdfTextIndex,
    by_line: &HashMap<&str, (&Page, &Line)>,
    measures: &HashMap<usize, (f64, f64)>,
    candidate: &StructureMarkerCandidate,
) -> Option<usize> {
    let (_, marker) = by_line.get(
        index
            .line_at(candidate.marker_range.start)?
            .line_id
            .as_str(),
    )?;
    let (left, _) = measures.get(&marker.page_index)?;
    let offset = marker.bbox[0] - left;
    let mut previous: Option<&Line> = None;
    let mut quoted = false;
    // Where the item's own lines wrap to, from the page's left edge.
    let mut hang = None;
    for indexed in index.overlapping_lines(candidate.range) {
        if indexed.range.end <= candidate.content_start {
            continue;
        }
        let Some(&(_, line)) = by_line.get(indexed.line_id.as_str()) else {
            continue;
        };
        if line.region_type == "heading" && indexed.range.start > candidate.content_start {
            return Some(indexed.range.start);
        }
        if !item_text_line(line) || line.text.trim().is_empty() {
            continue;
        }
        let Some(prior) = previous.replace(line) else {
            if stands_apart(marker, line) {
                return Some(indexed.range.start);
            }
            continue;
        };
        let new_page = line.page_index != prior.page_index;
        let Some(&(left, right)) = measures.get(&line.page_index) else {
            continue;
        };
        if !new_page && line.block_index == prior.block_index {
            if line.bbox[1] >= prior.bbox[3] {
                hang.get_or_insert(line.bbox[0] - left);
            }
            continue;
        }
        let height = (line.bbox[3] - line.bbox[1]).max(1.0);
        let margin = left + offset;
        let full = line.bbox[2] - line.bbox[0] >= 0.6 * (right - left);
        let text = line.text.trim();
        if starts_note_or_list(text) || (line.bbox[0] > margin + height && full) {
            quoted |= line.bbox[0] > margin + height;
            continue;
        }
        let aligned = (line.bbox[0] - margin).abs() <= height
            || hang.is_some_and(|hang| (line.bbox[0] - left - hang).abs() <= height);
        let carried =
            aligned && (!sentence_ended(prior.text.trim()) || (full && (quoted || new_page)));
        quoted = false;
        if !carried {
            return Some(indexed.range.start);
        }
    }
    None
}

impl PdfResolutionInput {
    pub(super) fn from_pages(pages: &[Page], primitives: &PdfPrimitiveEvidence) -> Self {
        let index = PdfTextIndex::from_pages(pages);
        let mut runs = structure_analysis().detect_structure_candidate_runs(index.text());
        let by_line = pages
            .iter()
            .flat_map(|page| {
                page.lines
                    .iter()
                    .map(move |line| (line.id.as_str(), (page, line)))
            })
            .collect::<HashMap<_, _>>();
        let measures = pages
            .iter()
            .filter_map(|page| {
                let body = page.lines.iter().filter(|line| item_text_line(line));
                let left = body.clone().map(|line| line.bbox[0]).reduce(f64::min)?;
                let right = body.map(|line| line.bbox[2]).reduce(f64::max)?;
                Some((page.index, (left, right)))
            })
            .collect::<HashMap<_, _>>();
        // The numbered paragraphs' numbers: a count from 1 set in the body, not the folios.
        let mut paragraph_starts = runs
            .iter()
            .filter(|run| run.grammar == CandidateGrammar::Numeric && run.rooted && run.consecutive)
            .flat_map(|run| &run.markers)
            .filter(|candidate| {
                index
                    .line_at(candidate.marker_range.start)
                    .and_then(|indexed| by_line.get(indexed.line_id.as_str()))
                    .is_some_and(|(_, line)| item_text_line(line))
            })
            .map(|candidate| candidate.range.start)
            .collect::<Vec<_>>();
        paragraph_starts.sort_unstable();
        // A list item or a subsection ends where the numbered paragraphs resume. A section's and
        // its subsections' extent is otherwise their grammar's; a numbered paragraph's or list
        // item's ends where the layout ends it.
        let mut ended = HashSet::new();
        for run in &mut runs {
            let hierarchy = run.grammar == CandidateGrammar::Hierarchy;
            for candidate in &mut run.markers {
                if run.grammar != CandidateGrammar::Numeric
                    && !(hierarchy && candidate.parent_candidate_id.is_none())
                {
                    let next =
                        paragraph_starts.partition_point(|start| *start <= candidate.range.start);
                    if let Some(&end) = paragraph_starts
                        .get(next)
                        .filter(|end| **end < candidate.range.end)
                    {
                        candidate.range.end = end.max(candidate.content_start);
                    }
                }
                if hierarchy {
                    if let Some(end) = candidate
                        .parent_candidate_id
                        .as_ref()
                        .and_then(|_| apart_end(&index, &by_line, candidate))
                    {
                        candidate.range.end = end;
                        ended.insert(candidate.id.clone());
                    }
                    continue;
                }
                if let Some(end) = layout_end(&index, &by_line, &measures, candidate) {
                    candidate.range.end = end;
                    ended.insert(candidate.id.clone());
                }
            }
        }
        let transcript_line_number_pages = transcript_line_number_pages(pages);
        let index_pages = index_pages(pages);
        let mut flow_lines = HashSet::new();
        for page in pages {
            for pair in page.lines.windows(2) {
                if pair.iter().all(|line| {
                    !line.exclude_from_body
                        && line.note_region_mode.is_empty()
                        && line.region_type == "body"
                }) && body_flow_edge(&pair[0], &pair[1])
                {
                    flow_lines.insert(pair[0].id.as_str());
                    flow_lines.insert(pair[1].id.as_str());
                }
            }
        }
        let citation_spans = by_line
            .iter()
            .map(|(line_id, (_, line))| {
                (
                    (*line_id).to_owned(),
                    protected_citation_spans(&line.text)
                        .into_iter()
                        .map(|(start, end)| PdfSourceSpan { start, end })
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut evidence = Vec::new();
        for run in &runs {
            let mut list_candidates = HashSet::new();
            if !matches!(run.grammar, CandidateGrammar::Numeric) {
                let marker_sources = run
                    .markers
                    .iter()
                    .map(|candidate| {
                        index
                            .line_at(candidate.marker_range.start)
                            .and_then(|line| by_line.get(line.line_id.as_str()).copied())
                    })
                    .collect::<Vec<_>>();
                for (marker_slot, candidate) in run.markers.iter().enumerate() {
                    let list_context = candidate.parent_candidate_id.is_some()
                        || (run.grammar == CandidateGrammar::Enumerator
                            && run.rooted
                            && run.consecutive);
                    if !list_context {
                        continue;
                    }
                    let Some((page, line)) = marker_sources[marker_slot] else {
                        continue;
                    };
                    let aligned_sibling =
                        run.markers
                            .iter()
                            .zip(&marker_sources)
                            .any(|(sibling, sibling_source)| {
                                if sibling.id == candidate.id || sibling.level != candidate.level {
                                    return false;
                                }
                                let Some((sibling_page, sibling_line)) = *sibling_source else {
                                    return false;
                                };
                                line.region_type == "body"
                                    && sibling_line.region_type == "body"
                                    && !line.exclude_from_body
                                    && !sibling_line.exclude_from_body
                                    && (line.bbox[0] - sibling_line.bbox[0]).abs()
                                        <= page.width.max(sibling_page.width).max(1.0) * 0.008
                            });
                    if aligned_sibling {
                        list_candidates.insert(candidate.id.as_str());
                    }
                }
            }
            for candidate in &run.markers {
                let candidate_lines = index.overlapping_lines(candidate.range);
                // A running head, folio or print margin inside the range is the page's, not
                // the item's; an item set wholly in them keeps them, and is excluded as furniture.
                let running = |line_id: &str| {
                    by_line.get(line_id).is_some_and(|(_, line)| {
                        matches!(line.region_type.as_str(), "header" | "footer")
                    })
                };
                let line_ids: Vec<String> = if candidate_lines
                    .iter()
                    .all(|line| running(&line.line_id))
                {
                    candidate_lines
                        .iter()
                        .map(|line| line.line_id.clone())
                        .collect()
                } else {
                    candidate_lines
                        .iter()
                        .filter(|line| !running(&line.line_id))
                        .map(|line| line.line_id.clone())
                        .collect()
                };
                let page_indexes = index.page_indexes(candidate.range);
                let marker_lines = index.overlapping_lines(candidate.marker_range);
                let mut observations = Vec::new();

                // The words after the marker on the item's first body lines.
                let body_tails = candidate_lines.iter().take(3).filter_map(|indexed| {
                    let (_, line) = by_line.get(indexed.line_id.as_str())?;
                    if line.exclude_from_body
                        || !line.note_region_mode.is_empty()
                        || line.region_type != "body"
                    {
                        return None;
                    }
                    let start = candidate
                        .content_start
                        .saturating_sub(indexed.range.start)
                        .min(line.text.chars().count());
                    Some((line, scalar_suffix(&line.text, start)))
                });
                let letters = |tail: &str| tail.chars().filter(|c| c.is_alphabetic()).count();
                let body_prose = body_tails.clone().any(|(line, tail)| {
                    letters(tail) >= 8
                        && (flow_lines.contains(line.id.as_str())
                            || tail.split_whitespace().take(3).count() == 3)
                });
                // A list item may be a few words.
                let item_words = body_tails
                    .clone()
                    .any(|(_, tail)| letters(tail) >= 4 && tail.split_whitespace().count() >= 2);
                if body_prose {
                    add_observation(&mut observations, CandidateObservationV2::BodyProseFlow);
                }
                if marker_lines.iter().any(|indexed| {
                    by_line
                        .get(indexed.line_id.as_str())
                        .is_some_and(|(_, line)| {
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
                if list_candidates.contains(candidate.id.as_str()) && item_words {
                    add_observation(&mut observations, CandidateObservationV2::ListItemLayout);
                }
                let marker_is_cross_reference = marker_lines.iter().any(|indexed| {
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
                    citation_spans
                        .get(indexed.line_id.as_str())
                        .is_some_and(|spans| {
                            spans
                                .iter()
                                .any(|span| span.start < local.end && local.start < span.end)
                        })
                });
                let opening = run.grammar == CandidateGrammar::Hierarchy
                    && candidate.parent_candidate_id.is_none();
                if marker_is_cross_reference
                    || (opening && wrapped_into(&index, &by_line, &measures, candidate))
                {
                    add_observation(&mut observations, CandidateObservationV2::CrossReference);
                }
                let table_or_form = marker_lines.iter().any(|indexed| {
                    primitives
                        .table_cell_line_ids
                        .contains(indexed.line_id.as_str())
                        || by_line
                            .get(indexed.line_id.as_str())
                            .is_some_and(|(_, line)| {
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
                let contents = candidate_lines.iter().take(3).any(|indexed| {
                    by_line
                        .get(indexed.line_id.as_str())
                        .is_some_and(|(page, line)| {
                            primitives.contents_pages.contains(&page.index)
                                || primitives.contents_line_ids.contains(&line.id)
                                || contents_row(&line.text)
                        })
                });
                if contents {
                    add_observation(&mut observations, CandidateObservationV2::ContentsRow);
                }
                if marker_lines
                    .iter()
                    .any(|line| index_pages.contains(&line.page_index))
                {
                    add_observation(&mut observations, CandidateObservationV2::IndexRow);
                }
                if marker_lines
                    .iter()
                    .any(|line| transcript_line_number_pages.contains(&line.page_index))
                {
                    add_observation(
                        &mut observations,
                        CandidateObservationV2::TranscriptLineNumber,
                    );
                }
                let furniture = marker_lines.iter().any(|indexed| {
                    by_line
                        .get(indexed.line_id.as_str())
                        .is_some_and(|(_, line)| {
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
        // A list ends where the layout ended one of its items, or a numbered paragraph began,
        // before its next item.
        let runs = runs
            .into_iter()
            .flat_map(|run| {
                if run.grammar != CandidateGrammar::Enumerator {
                    return vec![run];
                }
                let mut parts = Vec::<StructureCandidateRun>::new();
                for candidate in run.markers {
                    let continued = parts.last().is_some_and(|part| {
                        let prior = part.markers.last().expect("a part has an item");
                        let between =
                            paragraph_starts.partition_point(|start| *start <= prior.range.start);
                        !ended.contains(&prior.id)
                            && paragraph_starts
                                .get(between)
                                .is_none_or(|start| *start >= candidate.range.start)
                    });
                    match parts.last_mut() {
                        Some(part) if continued => {
                            part.range.end = part.range.end.max(candidate.range.end);
                            part.markers.push(candidate);
                        }
                        _ => parts.push(StructureCandidateRun {
                            id: format!("{}.{}", run.id, parts.len() + 1),
                            grammar: run.grammar,
                            range: candidate.range,
                            rooted: run.rooted,
                            consecutive: run.consecutive,
                            markers: vec![candidate],
                        }),
                    }
                }
                parts
            })
            .collect();
        Self {
            index,
            runs,
            evidence,
            citation_spans,
        }
    }
}

pub(super) fn map_note_pairs(
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

pub(super) fn native_graph_parts(
    index: &PdfTextIndex,
    pages: &[Page],
    paragraphs: &[Paragraph],
    primitives: &PdfPrimitiveEvidence,
) -> Result<Vec<StructureNode>> {
    let heading_levels = &primitives.heading_levels;
    // A passage wholly in a contents list, or wholly a parallel translation, says so: a
    // reader of the text's own structure skips it.
    let role = |line_ids: &[String]| {
        let all =
            |test: &dyn Fn(&String) -> bool| !line_ids.is_empty() && line_ids.iter().all(test);
        if all(&|id| {
            primitives.contents_line_ids.contains(id)
                || index
                    .line(id)
                    .is_some_and(|line| primitives.contents_pages.contains(&line.page_index))
        }) {
            Some("contents")
        } else if all(&|id| primitives.translation_line_ids.contains(id)) {
            Some("translation")
        } else {
            None
        }
    };
    const ORIGIN: &str = "legalpdf.pdf-structure.v2";
    let mut nodes = Vec::new();
    let text = ScalarText::new(index.text());
    for page in pages {
        let range = index.page_range(page.index).ok_or_else(|| {
            Error::Message(format!("page {} is absent from the text index", page.index))
        })?;
        let mut node = StructureNode::new(
            page.id.clone(),
            NodeKind::Page,
            range,
            ORIGIN,
            Derivation::Native,
            None,
        );
        node.label = Some(format!("page{}", page.number));
        node.aliases = page.printed_label.clone().map(|label| vec![label]);
        node.anchor.clone_from(&page.printed_label_line_id);
        node.page_indexes.push(page.index);
        node.line_ids = index.line_ids(range);
        nodes.push(node);
    }
    let mut heading_stack: Vec<(usize, String)> = Vec::new();
    for paragraph in paragraphs {
        let range = index
            .range_for_line_ids(&paragraph.line_ids)
            .ok_or_else(|| {
                Error::Message(format!("paragraph {} has no indexed lines", paragraph.id))
            })?;
        let heading = paragraph.region_type == "heading";
        let mut node = StructureNode::new(
            paragraph.id.clone(),
            if heading {
                NodeKind::Heading
            } else {
                NodeKind::Prose
            },
            range,
            ORIGIN,
            if heading {
                Derivation::Heuristic
            } else {
                Derivation::Native
            },
            None,
        );
        node.label = heading.then(|| {
            text.slice(range.start..range.end)
                .expect("indexed heading range must be in bounds")
                .to_owned()
        });
        node.page_indexes = index.page_indexes_for_line_ids(&paragraph.line_ids);
        node.line_ids.clone_from(&paragraph.line_ids);
        node.grammar = if heading {
            Some("accepted_heading".to_owned())
        } else {
            role(&paragraph.line_ids).map(str::to_owned)
        };
        if heading {
            if let Some(level) = paragraph
                .line_ids
                .iter()
                .filter_map(|id| heading_levels.get(id))
                .min()
            {
                while heading_stack
                    .last()
                    .is_some_and(|(prior, _)| prior >= level)
                {
                    heading_stack.pop();
                }
                node.parent_id = heading_stack.last().map(|(_, id)| id.clone());
                heading_stack.push((*level, node.id.clone()));
            }
        }
        nodes.push(node);
    }
    // A table, its rows and their cells, addressed as a word processor's are. A blank
    // cell is an empty range where its text would stand.
    for (slot, table) in primitives.tables.iter().enumerate() {
        let table_id = format!("table:{}", slot + 1);
        let line_ids: Vec<String> = table.line_ids().into_iter().cloned().collect();
        let range = index
            .range_for_line_ids(&line_ids)
            .ok_or_else(|| Error::Message(format!("{table_id} has no indexed lines")))?;
        let mut node = StructureNode::new(
            table_id.clone(),
            NodeKind::Table,
            range,
            ORIGIN,
            Derivation::Heuristic,
            None,
        );
        node.label = Some(table_id.clone());
        node.page_indexes = index.page_indexes_for_line_ids(&line_ids);
        node.line_ids = line_ids;
        nodes.push(node);
        let mut cursor = range.start;
        for (row_slot, row) in table.rows.iter().enumerate() {
            let row_id = format!("{table_id}/row:{}", row_slot + 1);
            let cells: Vec<StructureNode> = row
                .cells
                .iter()
                .map(|cell| {
                    let line_ids = cell.line_ids();
                    let range = cell
                        .parts
                        .iter()
                        .filter_map(|part| match part.chars {
                            Some((start, end)) => index.global_range(&part.line_id, start, end),
                            None => index.line(&part.line_id).map(|line| line.range),
                        })
                        .reduce(|left, right| ScalarRange {
                            start: left.start.min(right.start),
                            end: left.end.max(right.end),
                        })
                        .unwrap_or(ScalarRange {
                            start: cursor,
                            end: cursor,
                        });
                    cursor = range.end;
                    let mut node = StructureNode::new(
                        format!("{row_id}/col:{}", cell.column + 1),
                        NodeKind::Cell,
                        range,
                        ORIGIN,
                        Derivation::Heuristic,
                        Some(row_id.clone()),
                    );
                    node.label = Some(node.id.clone());
                    node.row_span = (cell.row_span > 1).then_some(cell.row_span);
                    node.column_span = (cell.column_span > 1).then_some(cell.column_span);
                    node.page_indexes = index.page_indexes_for_line_ids(&line_ids);
                    node.line_ids = line_ids;
                    node
                })
                .collect();
            let row_range = ScalarRange {
                start: cells
                    .iter()
                    .map(|cell| cell.range.start)
                    .min()
                    .unwrap_or(cursor),
                end: cells
                    .iter()
                    .map(|cell| cell.range.end)
                    .max()
                    .unwrap_or(cursor),
            };
            let mut node = StructureNode::new(
                row_id.clone(),
                NodeKind::Row,
                row_range,
                ORIGIN,
                Derivation::Heuristic,
                Some(table_id.clone()),
            );
            node.label = Some(row_id);
            node.line_ids = cells
                .iter()
                .flat_map(|cell| cell.line_ids.clone())
                .collect();
            node.page_indexes = index.page_indexes_for_line_ids(&node.line_ids);
            nodes.push(node);
            nodes.extend(cells);
        }
    }
    Ok(nodes)
}
