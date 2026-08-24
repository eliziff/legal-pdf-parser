use super::{arabic_page_number, body_flow_edge, scalar_suffix, PdfPrimitiveEvidence};
use legal_pdf_core::model::{NotePairClaim, NotePairKind, Page, Paragraph, PdfSourceSpan};
use legal_pdf_core::{Error, Result};
use legal_pdf_support::protected_citation_spans;
use legal_structure::{
    detect_structure_candidate_runs, CandidateEvidenceV2, CandidateGrammar, CandidateObservationV2,
    Derivation, DiagnosticSeverity, NodeKind, NoteBodyV2, NoteKindV2, NotePairClaimV2, ScalarRange,
    ScalarText, StructureCandidateRun, StructureDiagnostic, StructureNode, TextAnchorV2,
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
    LEADER.get_or_init(|| Regex::new(r"(?:\. ){3,}|\.{4,}").expect("contents leader regex"))
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

pub(super) fn index_pages(pages: &[Page]) -> HashSet<usize> {
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
    pub(super) fn from_pages(pages: &[Page], primitives: &PdfPrimitiveEvidence) -> Self {
        let index = PdfTextIndex::from_pages(pages);
        let runs = detect_structure_candidate_runs(index.text());
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
                let line_ids = candidate_lines
                    .iter()
                    .map(|line| line.line_id.clone())
                    .collect();
                let page_indexes = index.page_indexes(candidate.range);
                let marker_lines = index.overlapping_lines(candidate.marker_range);
                let mut observations = Vec::new();

                let body_prose = candidate_lines.iter().take(3).any(|indexed| {
                    let Some((_, line)) = by_line.get(indexed.line_id.as_str()) else {
                        return false;
                    };
                    if line.exclude_from_body
                        || !line.note_region_mode.is_empty()
                        || line.region_type != "body"
                    {
                        return false;
                    }
                    let start = candidate
                        .content_start
                        .saturating_sub(indexed.range.start)
                        .min(line.text.chars().count());
                    let tail = scalar_suffix(&line.text, start);
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
                if list_candidates.contains(candidate.id.as_str()) && body_prose {
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
                if marker_is_cross_reference {
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
) -> Result<Vec<StructureNode>> {
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
        node.grammar = heading.then(|| "accepted_heading".to_owned());
        nodes.push(node);
    }
    Ok(nodes)
}
