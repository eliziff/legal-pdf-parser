//! Sections spanning each leveled heading's extent, so that the structure engine's
//! smallest-container rule places the numbered paragraphs and lists under their heading.
//!
//! A heading's section runs from the heading to the line before the next heading at its level or
//! above. It carries no label or alias: it is never addressable and never answers a numbered
//! pinpoint ("section 5", "s. 3(2)"). A text whose running text carries provisions the engine
//! reads as sections gets none: its provisions are its sections, and their identities and
//! parentage stay as they are.
//! A section stops where a new filing's caption begins: a page carrying party labels opens
//! another document.

use super::flash::party_label;
use super::graph::PdfTextIndex;
use super::PdfPrimitiveEvidence;
use legal_pdf_core::model::Page;
use legal_structure_model::{
    CandidateEvidenceV2, CandidateGrammar, CandidateObservationV2, Derivation, NodeKind,
    ScalarRange, ScalarText, StructureCandidateRun, StructureNode,
};
use std::collections::{HashMap, HashSet};

const ORIGIN: &str = "legalpdf.pdf-structure.v2";

pub(super) fn nest_headings(
    nodes: &mut Vec<StructureNode>,
    index: &PdfTextIndex,
    runs: &[StructureCandidateRun],
    evidence: &[CandidateEvidenceV2],
    pages: &[Page],
    primitives: &PdfPrimitiveEvidence,
) {
    // Provisions in the running text that the engine will read as sections (a rooted,
    // consecutive statute numbering opening prose or a heading) make the text a statute's; one
    // alone may be a date's day ("13 May 2026").
    let observed = evidence
        .iter()
        .map(|item| (item.candidate_id.as_str(), item))
        .collect::<HashMap<_, _>>();
    let notes = pages
        .iter()
        .flat_map(|page| &page.lines)
        .filter(|line| !line.note_region_mode.is_empty())
        .map(|line| line.id.as_str())
        .collect::<HashSet<_>>();
    let text = ScalarText::new(index.text());
    let paragraph_numbers = runs
        .iter()
        .filter(|run| run.grammar == CandidateGrammar::Numeric && run.rooted && run.consecutive)
        .flat_map(|run| run.markers.iter().map(|marker| marker.marker_range.start))
        .collect::<HashSet<_>>();
    let statute = runs
        .iter()
        .filter(|run| run.grammar == CandidateGrammar::Hierarchy && run.rooted && run.consecutive)
        .flat_map(|run| &run.markers)
        // A lettered item ("(a)") divides a paragraph, and a number that opens a numbered
        // paragraph is the paragraph's; a provision is otherwise numbered or named.
        .filter(|marker| !paragraph_numbers.contains(&marker.marker_range.start))
        .filter(|marker| {
            text.slice(marker.marker_range.start..marker.marker_range.end)
                .is_some_and(|marker| {
                    let marker = marker.trim().to_lowercase();
                    let digits = marker.trim_start_matches('(');
                    digits.starts_with(|character: char| character.is_ascii_digit())
                        || [
                            "part", "section", "article", "division", "chapter", "schedule",
                        ]
                        .iter()
                        .any(|word| marker.starts_with(word))
                })
        })
        .filter(|marker| {
            observed.get(marker.id.as_str()).is_some_and(|item| {
                item.observations.iter().any(|observation| {
                    matches!(
                        observation,
                        CandidateObservationV2::BodyProseFlow
                            | CandidateObservationV2::SectionHeading
                    )
                }) && !item.observations.iter().any(|observation| {
                    matches!(
                        observation,
                        CandidateObservationV2::CrossReference
                            | CandidateObservationV2::Furniture
                            | CandidateObservationV2::TableOrForm
                            | CandidateObservationV2::ContentsRow
                            | CandidateObservationV2::IndexRow
                            | CandidateObservationV2::TranscriptLineNumber
                    )
                }) && !item.line_ids.iter().any(|id| notes.contains(id.as_str()))
            })
        })
        .take(3)
        .count()
        == 3;
    if statute {
        return;
    }
    let page_of = pages
        .iter()
        .flat_map(|page| {
            page.lines
                .iter()
                .map(move |line| (line.id.as_str(), page.index))
        })
        .collect::<HashMap<_, _>>();
    // Where each later filing begins.
    let captions = pages
        .iter()
        .filter(|page| {
            page.lines
                .iter()
                .any(|line| !line.exclude_from_body && party_label().is_match(line.text.trim()))
        })
        .filter_map(|page| index.page_range(page.index).map(|range| range.start))
        .collect::<Vec<_>>();
    let end = index.text().chars().count();

    let headings = nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| node.kind == NodeKind::Heading)
        .filter_map(|(slot, node)| {
            let level = node
                .line_ids
                .iter()
                .filter_map(|id| primitives.heading_levels.get(id))
                .min()?;
            Some((slot, *level, node.range.start))
        })
        .collect::<Vec<_>>();
    let mut sections = Vec::new();
    // Open sections: (level, section id, end).
    let mut open: Vec<(usize, String, usize)> = Vec::new();
    for (position, &(slot, level, start)) in headings.iter().enumerate() {
        while open
            .last()
            .is_some_and(|(prior, _, until)| *prior >= level || *until <= start)
        {
            open.pop();
        }
        let next = headings[position + 1..]
            .iter()
            .find(|(_, other, _)| *other <= level)
            .map_or(end, |(_, _, other)| *other);
        let boundary = captions
            .iter()
            .copied()
            .filter(|caption| *caption > start)
            .min()
            .unwrap_or(end);
        let until = next.min(boundary);
        let line_ids = index.line_ids(ScalarRange { start, end: until });
        let Some(range) = line_ids
            .iter()
            .filter_map(|id| index.line(id).map(|line| line.range))
            .filter(|range| range.end <= until || range.start == start)
            .fold(None, |range: Option<ScalarRange>, line| {
                Some(range.map_or(line, |range| ScalarRange {
                    start: range.start.min(line.start),
                    end: range.end.max(line.end),
                }))
            })
        else {
            continue;
        };
        let line_ids = line_ids
            .into_iter()
            .filter(|id| {
                index
                    .line(id)
                    .is_some_and(|line| line.range.end <= range.end)
            })
            .collect::<Vec<_>>();
        // The extent runs up to the next heading, separator included: a numbered paragraph the
        // engine ends where that heading begins lies within it.
        let range = ScalarRange {
            start: range.start,
            end: range.end.max(until),
        };
        // A heading under none sits on its page, so the engine does not read the section it
        // opens as its container; a heading under another keeps that heading as its parent.
        if nodes[slot].parent_id.is_none() {
            let page = nodes[slot]
                .line_ids
                .first()
                .and_then(|id| page_of.get(id.as_str()))
                .and_then(|page| pages.iter().find(|candidate| candidate.index == *page));
            nodes[slot].parent_id = page.map(|page| page.id.clone());
        }
        let id = format!("heading-section-{:06}", sections.len() + 1);
        let mut section = StructureNode::new(
            id.clone(),
            NodeKind::Section,
            range,
            ORIGIN,
            Derivation::Heuristic,
            None,
        );
        section.parent_id = open.last().map(|(_, id, _)| id.clone());
        section.grammar = Some("heading_section".to_owned());
        let mut seen = HashSet::new();
        section.page_indexes = line_ids
            .iter()
            .filter_map(|id| page_of.get(id.as_str()).copied())
            .filter(|page| seen.insert(*page))
            .collect();
        section.line_ids = line_ids;
        sections.push(section);
        open.push((level, id, range.end));
    }
    nodes.extend(sections);
}
