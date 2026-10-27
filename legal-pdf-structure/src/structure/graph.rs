use super::{
    arabic_page_number, body_flow_edge, continues_prose, scalar_suffix, sentence_ended,
    starts_note_or_list, DocumentContext, PdfPrimitiveEvidence,
};
use legal_pdf_core::model::{Line, NotePairClaim, NotePairKind, Page, Paragraph, PdfSourceSpan, Span, Word};
use legal_structure_model::structure_analysis;
use legal_pdf_core::{Error, Result};
use legal_structure_model::{
    CandidateEvidenceV2, CandidateGrammar, CandidateObservationV2, Derivation, DiagnosticSeverity,
    NodeKind, NoteBodyV2, NoteKindV2, NotePairClaimV2, ScalarRange, ScalarText,
    StructureCandidateRun, StructureDiagnostic, StructureMarkerCandidate, StructureNode,
    TextAnchorV2,
};
use regex::Regex;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::OnceLock;

mod text_index;
pub use text_index::*;
mod line_numbers;
pub(crate) use line_numbers::*;
mod nodes;
pub(crate) use nodes::*;

type BodyMeasures = HashMap<(DocumentContext, usize), (f64, f64)>;

#[derive(Debug)]
pub(super) struct PdfResolutionInput {
    pub(super) index: PdfTextIndex,
    pub(super) runs: Vec<StructureCandidateRun>,
    pub(super) evidence: Vec<CandidateEvidenceV2>,
    pub(super) citation_spans: BTreeMap<String, Vec<PdfSourceSpan>>,
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

/// A heading's printed number before its title: "1.", "1.1", "I.", "(a)", "ii.", a bare "1", or a
/// numbered word ("PART 1", "Section 1.01"). A number alone on its line is no marker of a title.
fn heading_marker_re() -> &'static Regex {
    static MARKER: OnceLock<Regex> = OnceLock::new();
    MARKER.get_or_init(|| {
        Regex::new(concat!(
            r"^\s*((?i:PART|PARTIE|ARTICLE|SECTION|CHAPTER|CHAPITRE|SCHEDULE|ANNEXE|GC)\s+[0-9IVXLC]{1,6}(?:\.[0-9]{1,3})*\.?",
            r"|\(?[0-9]{1,3}(?:\.[0-9]{1,3})*[.)]?|\(?[IVXLCivxlc]{1,6}[.)]|\(?[A-Za-z][.)])\s+\S"
        ))
        .expect("heading marker regex")
    })
}

pub(super) use crate::paragraphs::contents_leader_re;

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
    measures: &BodyMeasures,
    candidate: &StructureMarkerCandidate,
    primitives: &PdfPrimitiveEvidence,
) -> bool {
    let context = context_at(index, primitives, candidate.marker_range.start);
    let before = index.overlapping_lines(ScalarRange {
        start: 0,
        end: candidate.marker_range.start,
    });
    let Some(prior) = before
        .iter()
        .rev()
        .filter(|indexed| indexed.range.end <= candidate.marker_range.start)
        .filter_map(|indexed| by_line.get(indexed.line_id.as_str()).map(|(_, line)| *line))
        .find(|line| {
            primitives.contexts.reading_context(&line.id) == context
                && item_text_line(line)
                && !line.text.trim().is_empty()
        })
    else {
        return false;
    };
    let Some(&(left, right)) = measures.get(&(context, prior.page_index)) else {
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
    primitives: &PdfPrimitiveEvidence,
) -> Option<usize> {
    let (_, marker) = by_line.get(
        index
            .line_at(candidate.marker_range.start)?
            .line_id
            .as_str(),
    )?;
    let context = primitives.contexts.reading_context(&marker.id);
    let (indexed, line) = index
        .overlapping_lines(candidate.range)
        .iter()
        .filter(|indexed| {
            indexed.range.end > candidate.content_start
                && primitives.contexts.reading_context(&indexed.line_id) == context
        })
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
    measures: &BodyMeasures,
    candidate: &StructureMarkerCandidate,
    primitives: &PdfPrimitiveEvidence,
) -> Option<usize> {
    let (_, marker) = by_line.get(
        index
            .line_at(candidate.marker_range.start)?
            .line_id
            .as_str(),
    )?;
    let context = primitives.contexts.reading_context(&marker.id);
    let (left, _) = measures.get(&(context, marker.page_index))?;
    let offset = marker.bbox[0] - left;
    let mut previous: Option<&Line> = None;
    let mut quoted = false;
    // Where the item's own lines wrap to, from the page's left edge.
    let mut hang = None;
    for indexed in index.overlapping_lines(candidate.range) {
        if primitives.contexts.reading_context(&indexed.line_id) != context {
            continue;
        }
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
        let Some(&(left, right)) = measures.get(&(context, line.page_index)) else {
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
        // A line stopped well short of the measure with a blank line under it runs on to
        // nothing: a placeholder, a label, a form's blank.
        let parted = !new_page
            && prior.bbox[2] < right - 0.25 * (right - left)
            && line.bbox[1] - prior.bbox[3] > height;
        let carried = aligned
            && ((!sentence_ended(prior.text.trim()) && !parted) || (full && (quoted || new_page)));
        quoted = false;
        if !carried {
            return Some(indexed.range.start);
        }
    }
    None
}

fn context_at(
    index: &PdfTextIndex,
    primitives: &PdfPrimitiveEvidence,
    at: usize,
) -> DocumentContext {
    index
        .line_at(at)
        .map_or((0, false), |line| primitives.contexts.reading_context(&line.line_id))
}

fn read_candidates(
    index: &PdfTextIndex,
    text: &ScalarText<'_>,
    primitives: &PdfPrimitiveEvidence,
) -> (
    Vec<StructureCandidateRun>,
    Vec<(DocumentContext, usize, usize)>,
) {
    let mut runs = Vec::new();
    let mut citations = Vec::new();
    // Citations are read once over the whole text (the citation parser is the stage's costliest
    // reader) and given to the context whose line holds their start; a reading context sees the
    // same citations it read from its own view, save one straddling a context boundary.
    let mut whole_citations: Vec<(usize, usize)> =
        legal_pdf_support::profile::measure("candidates.citations", || {
            let mut found: Vec<(usize, usize)> = legal_pdf_core::layout_analysis().protected_citation_spans(text.value);
            found.extend(structure_analysis().citations(text.value).into_iter().map(|citation| {
                (
                    text.scalar_at_byte(citation.core_range.start).expect("citation start"),
                    text.scalar_at_byte(citation.core_range.end).expect("citation end"),
                )
            }));
            found
        });
    whole_citations.sort_unstable();
    let mut read = |coordinates: &ScalarText<'_>,
                    offset: usize,
                    context: DocumentContext,
                    ordinal: Option<usize>,
                    lines: Option<&[&IndexedPdfLine]>| {
        let prefix = ordinal.map_or_else(String::new, |slot| format!("context-{slot:06}-"));
        let mut found = legal_pdf_support::profile::measure("candidates.detect", || structure_analysis().detect_structure_candidate_runs(coordinates.value));
        let shift = |range: &mut ScalarRange| {
            range.start += offset;
            range.end += offset;
        };
        for run in &mut found {
            run.id.insert_str(0, &prefix);
            shift(&mut run.range);
            for marker in &mut run.markers {
                marker.id.insert_str(0, &prefix);
                if let Some(parent) = &mut marker.parent_candidate_id {
                    parent.insert_str(0, &prefix);
                }
                shift(&mut marker.range);
                shift(&mut marker.marker_range);
                marker.content_start += offset;
            }
        }
        runs.extend(found);
        let _ = offset;
        match lines {
            None => citations.extend(whole_citations.iter().map(|&(start, end)| (context, start, end))),
            Some(lines) => {
                for line in lines {
                    let from = whole_citations.partition_point(|(start, _)| *start < line.range.start);
                    citations.extend(
                        whole_citations[from..]
                            .iter()
                            .take_while(|(start, _)| *start < line.range.end)
                            .map(|&(start, end)| (context, start, end)),
                    );
                }
            }
        }
    };
    if primitives.contexts.reading_lines().is_empty() && primitives.contexts.translation_line_ids.is_empty() {
        read(text, 0, (0, false), None, None);
    } else {
        let mut readings = BTreeMap::<DocumentContext, Vec<&IndexedPdfLine>>::new();
        for line in &index.lines {
            readings
                .entry(primitives.contexts.reading_context(&line.line_id))
                .or_default()
                .push(line);
        }
        for (ordinal, (context, lines)) in readings.into_iter().enumerate() {
            let offset = lines[0].range.start;
            let mut view = String::new();
            let mut cursor = offset;
            let view_started = std::time::Instant::now();
            for line in &lines {
                // Blank foreign text without moving source scalar offsets or line boundaries.
                view.extend(
                    text.slice(cursor..line.range.start)
                        .expect("indexed gap")
                        .chars()
                        .map(|c| if c.is_whitespace() { c } else { ' ' }),
                );
                view.push_str(
                    text.slice(line.range.start..line.range.end)
                        .expect("indexed line"),
                );
                cursor = line.range.end;
            }
            if std::env::var_os("LEGALPDF_PHASE_TIMING").is_some() {
                eprintln!("phase	candidates.context_view	{}", view_started.elapsed().as_micros());
            }
            let scalar = legal_pdf_support::profile::measure("candidates.context_text", || ScalarText::new(&view));
            legal_pdf_support::profile::measure("candidates.context_read", || read(&scalar, offset, context, Some(ordinal), Some(&lines)));
        }
    }
    runs.sort_by_key(|run| {
        let grammar = match run.grammar {
            CandidateGrammar::Numeric => 0,
            CandidateGrammar::Hierarchy => 1,
            CandidateGrammar::Enumerator => 2,
        };
        (run.range.start, grammar, run.range.end)
    });
    (runs, citations)
}

impl PdfResolutionInput {
    pub(super) fn from_pages(pages: &[Page], primitives: &PdfPrimitiveEvidence) -> Self {
        let index = PdfTextIndex::from_pages(pages);
        let text = ScalarText::new(index.text());
        let (mut runs, citation_ranges) = legal_pdf_support::profile::measure("candidates.grammar_runs", || read_candidates(&index, &text, primitives));
        let by_line = pages
            .iter()
            .flat_map(|page| {
                page.lines
                    .iter()
                    .map(move |line| (line.id.as_str(), (page, line)))
            })
            .collect::<HashMap<_, _>>();
        // OCR keeps an isolated marker and its body as separate physical fragments.
        // Use the source producer's association without changing its paragraph/note role.
        for run in &mut runs {
            for candidate in &mut run.markers {
                let body = index.line_at(candidate.marker_range.start)
                    .and_then(|indexed| by_line.get(indexed.line_id.as_str()))
                    .and_then(|(_, line)| line.ocr.as_ref())
                    .and_then(|ocr| ocr.marker.as_ref())
                    .and_then(|marker| marker.body_line_id.as_deref())
                    .and_then(|id| index.line(id));
                if let Some(body) = body.filter(|body| {
                    candidate.marker_range.end <= body.range.start
                        && body.range.start < candidate.range.end
                        && primitives.contexts.reading_context(&body.line_id)
                            == context_at(&index, primitives, candidate.marker_range.start)
                }) {
                    candidate.content_start = body.range.start;
                }
            }
        }
        let mut measures = BodyMeasures::new();
        for page in pages {
            for line in page.lines.iter().filter(|line| item_text_line(line)) {
                let bounds = measures
                    .entry((primitives.contexts.reading_context(&line.id), page.index))
                    .or_insert((line.bbox[0], line.bbox[2]));
                bounds.0 = bounds.0.min(line.bbox[0]);
                bounds.1 = bounds.1.max(line.bbox[2]);
            }
        }
        // Every run owns its reading context; quoted text is not foreign to its own run.
        let context = legal_pdf_witnesses::Context {
            documents: Some(primitives.contexts.reading_lines()),
            quoted: None,
            table_cells: Some(&primitives.claims.table_cell_line_ids),
            contents: Some(&primitives.claims.contents_line_ids),
            translation: Some(&primitives.contexts.translation_line_ids),
            layout: primitives.source_regions.as_ref(),
        };
        // Existing composition rules retain their counted-line feature population.
        let table = legal_pdf_support::profile::measure("candidates.witnesses", || legal_pdf_witnesses::witnesses(pages, &context));
        legal_pdf_witnesses::dump(&table, "candidates");
        let rows = table.by_id();
        let compositions = legal_pdf_witnesses::compositions();
        let composed = |name: &str, line_id: &str| {
            compositions
                .decision(name)
                .and_then(|decision| rows.get(line_id).and_then(|row| decision.holds(row)))
                == Some(true)
        };
        // Each reading's numbered paragraphs keep a count separate from enclosing text.
        let mut paragraph_starts = BTreeMap::<DocumentContext, Vec<usize>>::new();
        for candidate in runs
            .iter()
            .filter(|run| run.grammar == CandidateGrammar::Numeric && run.rooted && run.consecutive)
            .flat_map(|run| &run.markers)
            .filter(|candidate| {
                index
                    .line_at(candidate.marker_range.start)
                    .and_then(|indexed| by_line.get(indexed.line_id.as_str()))
                    .is_some_and(|(_, line)| item_text_line(line))
            })
        {
            paragraph_starts
                .entry(context_at(&index, primitives, candidate.marker_range.start))
                .or_default()
                .push(candidate.range.start);
        }
        for starts in paragraph_starts.values_mut() {
            starts.sort_unstable();
        }
        // A list item or a subsection ends where the numbered paragraphs resume; only a numbered
        // provision may hold them. A section's and its subsections' extent is otherwise their
        // grammar's; a numbered paragraph's or list item's ends where the layout ends it.
        let mut ended = HashSet::new();
        for run in &mut runs {
            let hierarchy = run.grammar == CandidateGrammar::Hierarchy;
            for candidate in &mut run.markers {
                let context = context_at(&index, primitives, candidate.marker_range.start);
                let paragraph_starts = paragraph_starts
                    .get(&context)
                    .map_or(&[][..], |starts| starts.as_slice());
                let label = candidate.label.trim_start_matches('(').to_lowercase();
                let provision = hierarchy
                    && candidate.parent_candidate_id.is_none()
                    && (label.starts_with(|character: char| character.is_ascii_digit())
                        || [
                            "part", "section", "article", "division", "chapter", "schedule",
                        ]
                        .iter()
                        .any(|word| label.starts_with(word)));
                if run.grammar != CandidateGrammar::Numeric && !provision {
                    let next =
                        paragraph_starts.partition_point(|start| *start <= candidate.range.start);
                    if let Some(&end) = paragraph_starts
                        .get(next)
                        .filter(|end| **end < candidate.range.end)
                    {
                        candidate.range.end = end.max(candidate.content_start);
                    }
                    // Nor does a heading belong to one.
                    if let Some(heading) = hierarchy
                        .then(|| index.overlapping_lines(candidate.range))
                        .into_iter()
                        .flatten()
                        .filter(|indexed| {
                            indexed.range.start > candidate.content_start
                                && primitives.contexts.reading_context(&indexed.line_id) == context
                        })
                        .find(|indexed| {
                            by_line
                                .get(indexed.line_id.as_str())
                                .is_some_and(|(_, line)| line.region_type == "heading")
                        })
                    {
                        candidate.range.end = heading.range.start;
                    }
                }
                if hierarchy {
                    if let Some(end) = candidate
                        .parent_candidate_id
                        .as_ref()
                        .and_then(|_| apart_end(&index, &by_line, candidate, primitives))
                    {
                        candidate.range.end = end;
                        ended.insert(candidate.id.clone());
                    }
                    continue;
                }
                if let Some(end) = layout_end(&index, &by_line, &measures, candidate, primitives) {
                    candidate.range.end = end;
                    ended.insert(candidate.id.clone());
                }
            }
        }
        let mut flow_lines = HashSet::new();
        for page in pages {
            for pair in page.lines.windows(2) {
                if pair.iter().all(|line| {
                    !line.exclude_from_body
                        && line.note_region_mode.is_empty()
                        && line.region_type == "body"
                }) && primitives.contexts.reading_context(&pair[0].id)
                    == primitives.contexts.reading_context(&pair[1].id)
                    && body_flow_edge(&pair[0], &pair[1])
                {
                    flow_lines.insert(pair[0].id.as_str());
                    flow_lines.insert(pair[1].id.as_str());
                }
            }
        }
        // A quoted decision's own numbers (`foreign_numbered_paragraphs`) number nothing here,
        // whichever source read them as a run.
        if !primitives.foreign_numbered_line_ids.is_empty() {
            for run in &mut runs {
                let parents: HashMap<String, Option<String>> = run.markers.iter()
                    .map(|candidate| (candidate.id.clone(), candidate.parent_candidate_id.clone())).collect();
                run.markers.retain(|candidate| {
                    !index
                        .line_at(candidate.marker_range.start)
                        .is_some_and(|line| primitives.foreign_numbered_line_ids.contains(&line.line_id))
                });
                // A child of a removed marker hangs from its nearest surviving ancestor, else roots the run.
                let kept: HashSet<String> = run.markers.iter().map(|candidate| candidate.id.clone()).collect();
                for candidate in &mut run.markers {
                    while let Some(parent) = candidate.parent_candidate_id.clone().filter(|parent| !kept.contains(parent)) {
                        candidate.parent_candidate_id = parents.get(&parent).cloned().flatten();
                    }
                }
            }
            runs.retain(|run| !run.markers.is_empty());
        }
        // Project context-scoped citation ranges back onto their own source lines.
        // Citations by where they start, each with the furthest end so far, so a line finds the
        // ones overlapping it without reading every citation.
        let mut by_start = (0..citation_ranges.len()).collect::<Vec<_>>();
        by_start.sort_by_key(|&slot| citation_ranges[slot].1);
        let reach = by_start
            .iter()
            .scan(0, |end, &slot| {
                *end = (*end).max(citation_ranges[slot].2);
                Some(*end)
            })
            .collect::<Vec<_>>();
        let citation_spans = index
            .lines
            .iter()
            .map(|indexed| {
                let before =
                    by_start.partition_point(|&slot| citation_ranges[slot].1 < indexed.range.end);
                let reached = reach[..before].partition_point(|end| *end <= indexed.range.start);
                let mut overlapping = by_start[reached..before]
                    .iter()
                    .copied()
                    .filter(|&slot| {
                        citation_ranges[slot].0 == primitives.contexts.reading_context(&indexed.line_id)
                            && indexed.range.start < citation_ranges[slot].2
                    })
                    .collect::<Vec<_>>();
                overlapping.sort_unstable();
                let spans = overlapping
                    .into_iter()
                    .map(|slot| &citation_ranges[slot])
                    .map(|(_, start, end)| PdfSourceSpan {
                        start: start.saturating_sub(indexed.range.start),
                        end: (*end).min(indexed.range.end) - indexed.range.start,
                    })
                    .collect::<Vec<_>>();
                (indexed.line_id.clone(), spans)
            })
            .collect::<BTreeMap<_, _>>();
        let source_role = |id: &str| {
            let (_, line) = by_line.get(id)?;
            if line.exclude_from_body
                || !line.note_region_mode.is_empty()
                || line.region_type != "body"
            {
                return None;
            }
            primitives
                .source_regions
                .as_ref()?
                .get(id)
                .map(String::as_str)
        };
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
                                if sibling.id == candidate.id
                                    || sibling.level != candidate.level
                                    || sibling.parent_candidate_id != candidate.parent_candidate_id
                                {
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
                let context = context_at(&index, primitives, candidate.marker_range.start);
                let candidate_lines = index
                    .overlapping_lines(candidate.range)
                    .iter()
                    .filter(|line| primitives.contexts.reading_context(&line.line_id) == context)
                    .collect::<Vec<_>>();
                // A running head, folio or print margin inside the range is the page's, not
                // the item's; an item set wholly in them keeps them, and is excluded as furniture.
                let running = |line_id: &str| {
                    by_line.get(line_id).is_some_and(|(_, line)| {
                        matches!(line.region_type.as_str(), "header" | "footer")
                    })
                };
                let line_ids: Vec<String> =
                    if candidate_lines.iter().all(|line| running(&line.line_id)) {
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
                let page_indexes = index.page_indexes_for_line_ids(&line_ids);
                let marker_lines = index
                    .overlapping_lines(candidate.marker_range)
                    .iter()
                    .filter(|line| primitives.contexts.reading_context(&line.line_id) == context)
                    .collect::<Vec<_>>();
                let marker_line = marker_lines.first().map(|indexed| indexed.line_id.as_str()).unwrap_or("");
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
                                    .is_some_and(|region| legal_pdf_core::is_heading_region(region))
                        })
                }) {
                    add_observation(&mut observations, CandidateObservationV2::SectionHeading);
                }
                if !marker_lines.is_empty()
                    && marker_lines
                        .iter()
                        .all(|line| source_role(&line.line_id) == Some("LI"))
                {
                    add_observation(&mut observations, CandidateObservationV2::SourceListItem);
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
                // Bare period markers can be words in wrapped prose or a case citation.
                let prose_continuation = run.grammar == CandidateGrammar::Enumerator
                    && candidate.label.ends_with('.')
                    && marker_lines.iter().any(|indexed| {
                        by_line
                            .get(indexed.line_id.as_str())
                            .is_some_and(|(page, line)| {
                                continues_prose(page, line)
                                    || candidate.label == "v." && legal_pdf_core::layout_analysis().has_citation_signal(&line.text)
                            })
                    });
                if marker_is_cross_reference
                    || prose_continuation
                    || (opening && wrapped_into(&index, &by_line, &measures, candidate, primitives))
                {
                    add_observation(&mut observations, CandidateObservationV2::CrossReference);
                }
                let table_or_form = marker_lines.iter().any(|indexed| {
                    primitives
                        .claims.table_cell_line_ids
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
                            primitives.claims.contents_pages.contains(&page.index)
                                || primitives.claims.contents_line_ids.contains(&line.id)
                                || contents_row(&line.text)
                        })
                });
                if contents {
                    add_observation(&mut observations, CandidateObservationV2::ContentsRow);
                }
                let furniture = marker_lines.iter().any(|indexed| {
                    by_line
                        .get(indexed.line_id.as_str())
                        .is_some_and(|(_, line)| {
                            matches!(line.region_type.as_str(), "header" | "footer")
                                || (line.exclude_from_body
                                    && !primitives.claims.table_cell_line_ids.contains(&line.id))
                        })
                });
                if furniture {
                    add_observation(&mut observations, CandidateObservationV2::Furniture);
                }
                if composed("list_item", marker_line) {
                    add_observation(&mut observations, CandidateObservationV2::ComposedListItem);
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
                        let context = context_at(&index, primitives, prior.marker_range.start);
                        let paragraph_starts = paragraph_starts
                            .get(&context)
                            .map_or(&[][..], |starts| starts.as_slice());
                        let between =
                            paragraph_starts.partition_point(|start| *start <= prior.range.start);
                        !ended.contains(&prior.id)
                            && context_at(&index, primitives, candidate.marker_range.start)
                                == context
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
