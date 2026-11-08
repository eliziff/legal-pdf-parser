//! What a note says of the text around it: its proposition (the sentence it attaches to), its
//! cross-references to other notes, and the references no note answers.

use legal_pdf_core::model::{Diagnostic, Page};
use legal_pdf_text::text_index::PdfTextIndex;
use legal_structure_model::{DocumentStructure, NodeKind, ScalarText, StructureNode};
use legal_pdf_core::regex;
use legal_pdf_structure_model::{Decided, DocumentContext};
use legal_structure_model::text::normalize_label;
use regex::Regex;
use serde_json::json;
use std::collections::{HashMap, HashSet};

pub fn marker_re() -> &'static Regex {
    regex!(r"⟦FN:[^⟧]+⟧")
}

pub fn clean_markers(value: &str) -> String {
    marker_re().replace_all(value, "").trim().to_owned()
}

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
    clean_markers(&chars[start..end].iter().collect::<String>())
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
pub fn attach_propositions(structure: &mut DocumentStructure, units: &[serde_json::Value], decided: &Decided) {
    let ordered: Vec<usize> = structure.nodes.iter().enumerate()
        .filter(|(_, node)| matches!(node.kind, NodeKind::Footnote | NodeKind::Endnote)).map(|(at, _)| at).collect();
    let mut tails = HashMap::<DocumentContext, String>::new();
    for unit in units.iter().filter(|unit| unit["kind"] == "body") {
        let context = unit["source"]["lineIds"].get(0).and_then(serde_json::Value::as_str).map_or((0, false), |id| decided.document_context(id));
        let previous_tail = tails.entry(context).or_default();
        let text = unit["text"].as_str().unwrap_or_default();
        let coordinates = ScalarText::new(text);
        let chars: Vec<char> = text.chars().collect();
        let boundaries = sentence_boundaries(&chars);
        let mut previous_offset = 0;
        for reference in unit["footnote_refs"].as_array().into_iter().flatten() {
            let (Some(ordinal), Some(offset)) = (reference[0].as_u64(), reference[1].as_u64()) else { continue };
            let Some(offset) = coordinates.scalar_at_utf16(offset as usize) else { continue };
            let Some(&at) = (ordinal as usize).checked_sub(1).and_then(|at| ordered.get(at)) else { continue };
            let passage = clean_markers(&chars[previous_offset.min(offset)..offset].iter().collect::<String>());
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
    let labels: HashSet<_> = notes.iter().map(|note| (context_of(note), note.printed_label())).collect();
    let calls: HashSet<(usize, usize)> = notes.iter().flat_map(|note| &note.calls).map(|call| (call.start, call.end)).collect();
    let first_calls: HashSet<usize> = notes.iter().filter_map(|note| note.calls.first()).map(|call| call.start).collect();
    let mut diagnostics = Vec::new();
    for line in pages.iter().flat_map(|page| page.lines.iter()) {
        for detached in &line.detached_references {
            let label = normalize_label(&detached.note_id);
            let placed = index.global_range(&line.id, detached.start_offset, detached.end_offset);
            if !placed.is_some_and(|range| calls.contains(&(range.start, range.end))) {
                let mut diagnostic =
                    Diagnostic::warning("FOOTNOTE_UNMATCHED_REFERENCE", format!("Detached superscript '{label}' has no paired label."), Some(line.page_index));
                diagnostic.line_ids.push(line.id.clone());
                if !detached.source_line_id.is_empty() {
                    diagnostic.line_ids.push(detached.source_line_id.clone());
                }
                diagnostic.details.insert("label".to_owned(), json!(label));
                diagnostics.push(diagnostic);
            }
        }
        let holds_first_call = index.line(&line.id).is_some_and(|placed| first_calls.iter().any(|&start| placed.range.start <= start && start < placed.range.end));
        if line.exclude_from_body || !matches!(line.region_type.as_str(), "body" | "heading") || holds_first_call {
            continue;
        }
        let context = decided.document_context(&line.id);
        if line.spans.iter().any(|span| span.superscript && labels.contains(&(context, normalize_label(span.text.trim()).as_str()))) {
            let message = "A superscript resembling a known note label was not paired.";
            let mut diagnostic = Diagnostic::warning("FOOTNOTE_UNMATCHED_REFERENCE", message, Some(line.page_index));
            diagnostic.line_ids.push(line.id.clone());
            diagnostics.push(diagnostic);
        }
    }
    diagnostics
}

/// A note's reference to another note ("supra note 12", "op. cit. note 4", "see footnote 3") that names no note of its
/// number, or more than one: one in the note's own numbering, else the only one of the number.
pub fn note_crossrefs(structure: &DocumentStructure, decided: &Decided) -> Vec<Diagnostic> {
    let pattern = regex!(r"(?i)\b(?:(supra|infra),?\s+(?:foot)?notes?|(op)\.?\s*cit\.?,?\s+(?:foot)?notes?|(see)\s+(?:also\s+)?footnote)\s+(\d{1,3})\b");
    let context_of = |node: &StructureNode| node.line_ids.first().map_or((0, false), |id| decided.document_context(id));
    let mut by_number: HashMap<(DocumentContext, u32), Vec<&str>> = HashMap::new();
    for node in notes(structure) {
        if let Ok(number) = node.printed_label().parse::<u32>() {
            by_number.entry((context_of(node), number)).or_default().push(node.series.as_str());
        }
    }
    let mut diagnostics = Vec::new();
    for node in notes(structure) {
        let body = note_text(structure, node);
        for capture in pattern.captures_iter(&body) {
            let number = capture.get(4).and_then(|value| value.as_str().parse::<u32>().ok()).unwrap_or(0);
            let candidates = by_number.get(&(context_of(node), number)).map(Vec::as_slice).unwrap_or_default();
            let scoped = candidates.iter().filter(|series| candidates.len() == 1 || **series == node.series).count();
            if scoped == 1 {
                continue;
            }
            let kind = match (capture.get(1), capture.get(2)) {
                (Some(value), _) => value.as_str().to_lowercase(),
                (None, Some(_)) => "op_cit".to_owned(),
                (None, None) => "see_footnote".to_owned(),
            };
            let why = if candidates.is_empty() { "which no paired note carries - a pairing-quality witness" } else { "which has multiple possible targets" };
            let message = format!("Note {} references {kind} note {number}, {why}.", node.printed_label());
            let mut diagnostic = Diagnostic::info("NOTE_CROSSREF_UNRESOLVED", message, node.page_indexes.first().copied());
            let found = capture.get(0).map_or(0, |found| found.start());
            diagnostic.details.insert("crossref".to_owned(), json!({"source_pair_id": node.id, "kind": kind, "number": number,
                "shortform": legal_structure_model::structure_analysis().crossref_short_form(&body, found), "target_count": candidates.len()}));
            diagnostics.push(diagnostic);
        }
    }
    diagnostics
}
