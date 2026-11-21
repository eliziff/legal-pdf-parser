//! What a note says of the text around it: its proposition (the sentence it attaches to), its
//! cross-references to other notes, and the references no note answers.

use crate::model::DocumentContext;
use crate::model::Decided;
use crate::text::text_index::PdfTextIndex;
use legal_pdf_core::model::{Diagnostic, Line, LineZone};
use legal_pdf_core::AuthorityTextUnit;
use legal_structure_model::text::normalize_label;
use legal_structure_model::{DocumentStructure, NodeKind, StructureNode};
use std::collections::HashSet;

/// Each note's sentence and the passage since the call before it, read off each constituent document's body units by
/// the model's one proposition reader (their text, their notes' marks left out, and where each mark stood:
/// `footnote_refs`, each a note's ordinal and a UTF-16 offset). Each is a property of the note's node ("sentence",
/// "passage"); a note called more than once has neither.
pub fn attach_propositions(
    structure: &mut DocumentStructure,
    units: &[AuthorityTextUnit],
    decided: &Decided,
) {
    let ordered: Vec<usize> = structure
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| matches!(node.kind, NodeKind::Footnote | NodeKind::Endnote))
        .map(|(at, _)| at)
        .collect();
    let mut documents = Vec::<(DocumentContext, Vec<&AuthorityTextUnit>)>::new();
    for unit in units.iter().filter(|unit| unit.kind == "body") {
        let context = unit
            .source
            .as_ref()
            .and_then(|source| source.line_ids.first())
            .map_or((0, false), |id| decided.document_context(id));
        match documents.iter_mut().find(|(other, _)| *other == context) {
            Some((_, group)) => group.push(unit),
            None => documents.push((context, vec![unit])),
        }
    }
    for (_, group) in documents {
        let texts: Vec<&str> = group.iter().map(|unit| unit.text.as_str()).collect();
        // A mark inside a character's surrogate pair stands nowhere.
        let calls: Vec<(usize, u64, usize)> = group
            .iter()
            .enumerate()
            .flat_map(|(at, unit)| {
                let text: Vec<u16> = unit.text.encode_utf16().collect();
                unit.footnote_refs
                    .iter()
                    .filter(move |&&(_, offset)| {
                        !text
                            .get(offset)
                            .is_some_and(|unit| (0xdc00..0xe000).contains(unit))
                    })
                    .filter_map(move |&(ordinal, offset)| {
                        u64::try_from(ordinal)
                            .ok()
                            .map(|ordinal| (at, ordinal, offset))
                    })
            })
            .collect();
        for note in legal_structure_model::propositions::propositions(&texts, &calls).notes {
            let Some(&at) = usize::try_from(note.note)
                .ok()
                .and_then(|ordinal| ordinal.checked_sub(1))
                .and_then(|at| ordered.get(at))
            else {
                continue;
            };
            let properties = &mut structure.nodes[at].properties;
            properties.insert("sentence".to_owned(), note.sentence_text);
            properties.insert("passage".to_owned(), note.passage_text);
        }
    }
}
/// A detached superscript no note's call stands on, and a body superscript that reads as a known note's label but no
/// call stands on.
pub fn unmatched_reference_diagnostics<'a>(
    lines: impl Iterator<Item = (&'a Line, LineZone)>,
    notes: &[StructureNode],
    index: &PdfTextIndex,
    decided: &Decided,
) -> Vec<Diagnostic> {
    let context_of = |note: &StructureNode| {
        note.line_ids
            .first()
            .map_or((0, false), |id| decided.document_context(id))
    };
    let labels: HashSet<_> = notes
        .iter()
        .map(|note| (context_of(note), note.label.as_deref().unwrap_or("")))
        .collect();
    let calls: HashSet<(usize, usize)> = notes
        .iter()
        .flat_map(|note| &note.calls)
        .map(|call| (call.start, call.end))
        .collect();
    let first_calls: HashSet<usize> = notes
        .iter()
        .filter_map(|note| note.calls.first())
        .map(|call| call.start)
        .collect();
    let mut diagnostics = Vec::new();
    for (line, zone) in lines {
        // A detached reference's characters (as recognition places it) stand in the line's UTF-16 code units.
        let units = |characters: usize| {
            (characters <= line.text.chars().count()).then(|| {
                line.text
                    .chars()
                    .take(characters)
                    .map(char::len_utf16)
                    .sum::<usize>()
            })
        };
        for detached in &line.detached_references {
            let placed = units(detached.start_offset)
                .zip(units(detached.end_offset))
                .and_then(|(start, end)| index.global_range(&line.id, start, end));
            if !placed.is_some_and(|range| calls.contains(&(range.start, range.end))) {
                diagnostics.push(Diagnostic::new(
                    "FOOTNOTE_UNMATCHED_REFERENCE",
                    Some(line.page_index),
                ));
            }
        }
        let holds_first_call = index.line(&line.id).is_some_and(|placed| {
            first_calls
                .iter()
                .any(|&start| placed.range.start <= start && start < placed.range.end)
        });
        if !matches!(zone, LineZone::Body | LineZone::Heading) || holds_first_call {
            continue;
        }
        let context = decided.document_context(&line.id);
        if line.spans.iter().any(|span| {
            span.superscript
                && labels.contains(&(context, normalize_label(span.text.trim()).as_str()))
        }) {
            diagnostics.push(Diagnostic::new(
                "FOOTNOTE_UNMATCHED_REFERENCE",
                Some(line.page_index),
            ));
        }
    }
    diagnostics
}
