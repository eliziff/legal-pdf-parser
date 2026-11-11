//! What a note says of the text around it: its proposition (the sentence it attaches to), its
//! cross-references to other notes, and the references no note answers.

use legal_pdf_core::model::{Diagnostic, Line, LineZone};
use legal_pdf_core::AuthorityTextUnit;
use crate::text::text_index::PdfTextIndex;
use legal_structure_model::{DocumentStructure, NodeKind, StructureNode};
use crate::model::{Decided, DocumentContext};
use legal_structure_model::text::normalize_label;
use std::collections::{HashMap, HashSet};

/// Whether a UTF-16 code unit is a white space character (half a surrogate pair never is).
fn blank(unit: u16) -> bool {
    char::from_u32(u32::from(unit)).is_some_and(char::is_whitespace)
}

/// The text of UTF-16 code units, trimmed.
fn read(units: &[u16]) -> String {
    String::from_utf16_lossy(units).trim().to_owned()
}

/// Each sentence's end in `units` (a text's UTF-16 code units): its stop and the quotes or brackets
/// closing it, where a space, a note's marker or the text's end follows.
pub(crate) fn sentence_boundaries(units: &[u16]) -> Vec<(usize, usize)> {
    fn is(unit: Option<&u16>, characters: &[char]) -> bool {
        unit.is_some_and(|&unit| characters.iter().any(|&c| u32::from(unit) == u32::from(c)))
    }
    let marker: Vec<u16> = "\u{27e6}FN:".encode_utf16().collect();
    let mut boundaries = Vec::<(usize, usize)>::new();
    let mut index = 0;
    while index < units.len() {
        if !is(units.get(index), &['.', '?', '!']) {
            index += 1;
            continue;
        }
        let start = index;
        index += 1;
        while is(units.get(index), &['"', '\'', '\u{201d}', '\u{2019}', ')', ']']) {
            index += 1;
        }
        if units.get(index).is_none_or(|&unit| blank(unit)) || units[index..].starts_with(&marker) {
            boundaries.push((start, index));
        }
    }
    boundaries
}

pub(crate) fn sentence_at_boundaries(units: &[u16], boundaries: &[(usize, usize)], offset: usize) -> String {
    let previous = &boundaries[..boundaries.partition_point(|(_, end)| *end <= offset)];
    let (start, end) = match previous.last() {
        // A reference after a sentence's stop is that sentence's.
        Some(&(_, end)) if units[end..offset.min(units.len())].iter().all(|&unit| blank(unit)) => {
            (if previous.len() > 1 { previous[previous.len() - 2].1 } else { 0 }, end)
        }
        last => (
            last.map_or(0, |(_, end)| *end),
            boundaries.iter().find(|(boundary_start, _)| *boundary_start >= offset).map_or(units.len(), |(_, boundary_end)| *boundary_end),
        ),
    };
    read(&units[start..end])
}

/// Each note's sentence and the passage since the note before it, read off the body authority units: their text, their
/// notes' marks left out, and where each mark stood (`footnote_refs`, each a note's ordinal and a UTF-16 offset). Each is
/// a property of the note's node ("sentence", "passage").
pub(crate) fn attach_propositions(structure: &mut DocumentStructure, units: &[AuthorityTextUnit], decided: &Decided) {
    let ordered: Vec<usize> = structure.nodes.iter().enumerate()
        .filter(|(_, node)| matches!(node.kind, NodeKind::Footnote | NodeKind::Endnote)).map(|(at, _)| at).collect();
    let mut tails = HashMap::<DocumentContext, String>::new();
    for unit in units.iter().filter(|unit| unit.kind == "body") {
        let context = unit.source.as_ref().and_then(|source| source.line_ids.first()).map_or((0, false), |id| decided.document_context(id));
        let previous_tail = tails.entry(context).or_default();
        let units: Vec<u16> = unit.text.encode_utf16().collect();
        let boundaries = sentence_boundaries(&units);
        let mut previous_offset = 0;
        for &(ordinal, offset) in &unit.footnote_refs {
            // (A mark past the text, or inside a character's surrogate pair, stands nowhere.)
            if offset > units.len() || units.get(offset).is_some_and(|unit| (0xdc00..0xe000).contains(unit)) {
                continue;
            }
            let Some(&at) = usize::try_from(ordinal).ok().and_then(|ordinal| ordinal.checked_sub(1)).and_then(|at| ordered.get(at)) else { continue };
            let passage = read(&units[previous_offset.min(offset)..offset]);
            let properties = &mut structure.nodes[at].properties;
            properties.insert("sentence".to_owned(), sentence_at_boundaries(&units, &boundaries, offset));
            properties.insert("passage".to_owned(), if passage.is_empty() { previous_tail.clone() } else { passage });
            previous_offset = offset;
        }
        *previous_tail = read(&units[previous_offset.min(units.len())..]);
    }
}

/// A detached superscript no note's call stands on, and a body superscript that reads as a known note's label but no
/// call stands on.
pub(crate) fn unmatched_reference_diagnostics<'a>(lines: impl Iterator<Item = (&'a Line, LineZone)>, notes: &[StructureNode], index: &PdfTextIndex, decided: &Decided) -> Vec<Diagnostic> {
    let context_of = |note: &StructureNode| note.line_ids.first().map_or((0, false), |id| decided.document_context(id));
    let labels: HashSet<_> = notes.iter().map(|note| (context_of(note), note.label.as_deref().unwrap_or(""))).collect();
    let calls: HashSet<(usize, usize)> = notes.iter().flat_map(|note| &note.calls).map(|call| (call.start, call.end)).collect();
    let first_calls: HashSet<usize> = notes.iter().filter_map(|note| note.calls.first()).map(|call| call.start).collect();
    let mut diagnostics = Vec::new();
    for (line, zone) in lines {
        // A detached reference's characters (as recognition places it) stand in the line's UTF-16 code units.
        let units = |characters: usize| (characters <= line.text.chars().count()).then(|| line.text.chars().take(characters).map(char::len_utf16).sum::<usize>());
        for detached in &line.detached_references {
            let placed = units(detached.start_offset).zip(units(detached.end_offset)).and_then(|(start, end)| index.global_range(&line.id, start, end));
            if !placed.is_some_and(|range| calls.contains(&(range.start, range.end))) {
                diagnostics.push(Diagnostic::new("FOOTNOTE_UNMATCHED_REFERENCE", Some(line.page_index)));
            }
        }
        let holds_first_call = index.line(&line.id).is_some_and(|placed| first_calls.iter().any(|&start| placed.range.start <= start && start < placed.range.end));
        if !matches!(zone, LineZone::Body | LineZone::Heading) || holds_first_call {
            continue;
        }
        let context = decided.document_context(&line.id);
        if line.spans.iter().any(|span| span.superscript && labels.contains(&(context, normalize_label(span.text.trim()).as_str()))) {
            diagnostics.push(Diagnostic::new("FOOTNOTE_UNMATCHED_REFERENCE", Some(line.page_index)));
        }
    }
    diagnostics
}
