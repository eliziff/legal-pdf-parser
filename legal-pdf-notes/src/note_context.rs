//! What a note says of the text around it: its proposition (the sentence it attaches to), its
//! cross-references to other notes, and the references no note answers.

use legal_pdf_core::model::{Diagnostic, Page};
use legal_pdf_core::AuthorityTextUnit;
use legal_pdf_text::text_index::PdfTextIndex;
use legal_structure_model::{DocumentStructure, NodeKind, ScalarText, StructureNode};
use legal_pdf_structure_model::{Decided, DocumentContext};
use legal_structure_model::text::normalize_label;
use std::collections::{HashMap, HashSet};

/// Each sentence's end in `chars`: its stop and the quotes or brackets closing it, where a space, a
/// note's marker or the text's end follows.
pub fn sentence_boundaries(chars: &[char]) -> Vec<(usize, usize)> {
    let mut boundaries = Vec::<(usize, usize)>::new();
    let mut index = 0;
    while index < chars.len() {
        if !matches!(chars[index], '.' | '?' | '!') {
            index += 1;
            continue;
        }
        let start = index;
        index += 1;
        while chars.get(index).is_some_and(|character| ['"', '\'', '”', '’', ')', ']'].contains(character)) {
            index += 1;
        }
        if chars.get(index).is_none_or(|character| character.is_whitespace()) || chars[index..].starts_with(&['⟦', 'F', 'N', ':']) {
            boundaries.push((start, index));
        }
    }
    boundaries
}

pub fn sentence_at_boundaries(chars: &[char], boundaries: &[(usize, usize)], offset: usize) -> String {
    let previous = &boundaries[..boundaries.partition_point(|(_, end)| *end <= offset)];
    let (start, end) = match previous.last() {
        // A reference after a sentence's stop is that sentence's.
        Some(&(_, end)) if chars[end..offset.min(chars.len())].iter().all(|character| character.is_whitespace()) => {
            (if previous.len() > 1 { previous[previous.len() - 2].1 } else { 0 }, end)
        }
        last => (
            last.map_or(0, |(_, end)| *end),
            boundaries.iter().find(|(boundary_start, _)| *boundary_start >= offset).map_or(chars.len(), |(_, boundary_end)| *boundary_end),
        ),
    };
    chars[start..end].iter().collect::<String>().trim().to_owned()
}

/// The document's notes in order, each its node.
pub fn notes(structure: &DocumentStructure) -> impl Iterator<Item = &StructureNode> {
    structure.nodes.iter().filter(|node| matches!(node.kind, NodeKind::Footnote | NodeKind::Endnote))
}

/// A note's body as the rendered text reads it: its stretches, a space between.
pub fn note_text(structure: &DocumentStructure, node: &StructureNode) -> String {
    let text = ScalarText::new(structure.query_text());
    node.rendered_membership.iter().filter_map(|range| text.slice_utf16(range.start..range.end)).collect::<Vec<_>>().join(" ")
}

/// Each note's sentence and the passage since the note before it, read off the body authority units: their text, their
/// notes' marks left out, and where each mark stood (`footnote_refs`, each a note's ordinal and a UTF-16 offset). Each is
/// a property of the note's node ("sentence", "passage").
pub fn attach_propositions(structure: &mut DocumentStructure, units: &[AuthorityTextUnit], decided: &Decided) {
    let ordered: Vec<usize> = structure.nodes.iter().enumerate()
        .filter(|(_, node)| matches!(node.kind, NodeKind::Footnote | NodeKind::Endnote)).map(|(at, _)| at).collect();
    let mut tails = HashMap::<DocumentContext, String>::new();
    for unit in units.iter().filter(|unit| unit.kind == "body") {
        let context = unit.source.as_ref().and_then(|source| source.line_ids.first()).map_or((0, false), |id| decided.document_context(id));
        let previous_tail = tails.entry(context).or_default();
        let coordinates = ScalarText::new(&unit.text);
        let chars: Vec<char> = unit.text.chars().collect();
        let boundaries = sentence_boundaries(&chars);
        let mut previous_offset = 0;
        for &(ordinal, offset) in &unit.footnote_refs {
            let Some(offset) = coordinates.scalar_at_utf16(offset) else { continue };
            let Some(&at) = usize::try_from(ordinal).ok().and_then(|ordinal| ordinal.checked_sub(1)).and_then(|at| ordered.get(at)) else { continue };
            let passage = chars[previous_offset.min(offset)..offset].iter().collect::<String>().trim().to_owned();
            let properties = &mut structure.nodes[at].properties;
            properties.insert("sentence".to_owned(), sentence_at_boundaries(&chars, &boundaries, offset).trim().to_owned());
            properties.insert("passage".to_owned(), if passage.is_empty() { previous_tail.clone() } else { passage });
            previous_offset = offset;
        }
        *previous_tail = chars[previous_offset.min(chars.len())..].iter().collect::<String>().trim().to_owned();
    }
}

/// A detached superscript no note's call stands on, and a body superscript that reads as a known note's label but no
/// call stands on.
pub fn unmatched_reference_diagnostics(pages: &[Page], notes: &[StructureNode], index: &PdfTextIndex, decided: &Decided) -> Vec<Diagnostic> {
    let context_of = |note: &StructureNode| note.line_ids.first().map_or((0, false), |id| decided.document_context(id));
    let labels: HashSet<_> = notes.iter().map(|note| (context_of(note), note.label.as_deref().unwrap_or(""))).collect();
    let calls: HashSet<(usize, usize)> = notes.iter().flat_map(|note| &note.calls).map(|call| (call.start, call.end)).collect();
    let first_calls: HashSet<usize> = notes.iter().filter_map(|note| note.calls.first()).map(|call| call.start).collect();
    let mut diagnostics = Vec::new();
    for line in pages.iter().flat_map(|page| page.lines.iter()) {
        for detached in &line.detached_references {
            let placed = index.global_range(&line.id, detached.start_offset, detached.end_offset);
            if !placed.is_some_and(|range| calls.contains(&(range.start, range.end))) {
                diagnostics.push(Diagnostic::new("FOOTNOTE_UNMATCHED_REFERENCE", Some(line.page_index)));
            }
        }
        let holds_first_call = index.line(&line.id).is_some_and(|placed| first_calls.iter().any(|&start| placed.range.start <= start && start < placed.range.end));
        if line.exclude_from_body || !matches!(line.region_type.as_str(), "body" | "heading") || holds_first_call {
            continue;
        }
        let context = decided.document_context(&line.id);
        if line.spans.iter().any(|span| span.superscript && labels.contains(&(context, normalize_label(span.text.trim()).as_str()))) {
            diagnostics.push(Diagnostic::new("FOOTNOTE_UNMATCHED_REFERENCE", Some(line.page_index)));
        }
    }
    diagnostics
}

