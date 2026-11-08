//! What a note says of the text around it: its proposition (the sentence it attaches to), its
//! cross-references to other notes, and the references no note answers.

use legal_pdf_core::model::{Diagnostic, Footnote, FootnoteCrossref, Page, Paragraph};
use legal_pdf_core::{regex, Anchor};
use legal_pdf_structure_model::{Decided, DocumentContext};
use legal_structure_model::text::normalize_label;
use legal_structure_model::StructureAnalysis;
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

pub fn attach_propositions(footnotes: &mut [Footnote], paragraphs: &[Paragraph], decided: &Decided) {
    let by_pair: HashMap<String, usize> = footnotes.iter().enumerate().map(|(index, footnote)| (footnote.pair_id.clone(), index)).collect();
    let mut tails = HashMap::<DocumentContext, String>::new();
    for paragraph in paragraphs {
        let context = paragraph.line_ids.first().map_or((0, false), |id| decided.document_context(id));
        let previous_tail = tails.entry(context).or_default();
        let chars: Vec<char> = paragraph.text.chars().collect();
        let boundaries = sentence_boundaries(&chars);
        let clip = |at: usize| at.min(chars.len());
        let mut previous_offset = 0;
        for anchor in &paragraph.anchors {
            let Some(&index) = by_pair.get(&anchor.pair_id) else {
                continue;
            };
            let offset = anchor.offset;
            footnotes[index].sentence_proposition = sentence_at_boundaries(&chars, &boundaries, offset);
            let passage = clean_markers(&chars[clip(previous_offset)..clip(offset)].iter().collect::<String>());
            footnotes[index].passage_since_prior_note = if passage.is_empty() { previous_tail.clone() } else { passage };
            previous_offset = offset + 5 + anchor.pair_id.chars().count();
        }
        *previous_tail = chars[clip(previous_offset)..].iter().collect::<String>().trim().to_owned();
    }
}

pub fn unmatched_reference_diagnostics(pages: &[Page], footnotes: &[Footnote], anchors: &HashMap<String, Vec<Anchor>>, decided: &Decided) -> Vec<Diagnostic> {
    let labels: HashSet<_> = footnotes.iter().map(|note| (decided.note_context(note), note.label.as_str())).collect();
    let primary_lines: HashSet<&str> = footnotes.iter().filter_map(|note| note.reference_line_id.as_deref()).collect();
    let mut diagnostics = Vec::new();
    for line in pages.iter().flat_map(|page| page.lines.iter()) {
        for detached in &line.detached_references {
            let label = normalize_label(&detached.note_id);
            let (start, end) = (detached.start_offset, detached.end_offset);
            let paired = anchors.get(&line.id).is_some_and(|entries| entries.iter().any(|anchor| anchor.start == start && anchor.end == end && anchor.label == label));
            if !paired {
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
        if line.exclude_from_body || !matches!(line.region_type.as_str(), "body" | "heading") || primary_lines.contains(line.id.as_str()) {
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

pub fn attach_crossrefs(footnotes: &mut [Footnote], diagnostics: &mut Vec<Diagnostic>, decided: &Decided) {
    let pattern = regex!(r"(?i)\b(?:(supra|infra),?\s+(?:foot)?notes?|(op)\.?\s*cit\.?,?\s+(?:foot)?notes?|(see)\s+(?:also\s+)?footnote)\s+(\d{1,3})\b");
    let mut by_number: HashMap<(DocumentContext, u32), Vec<(usize, String)>> = HashMap::new();
    for footnote in footnotes.iter() {
        if let Ok(number) = footnote.label.parse::<u32>() {
            by_number.entry((decided.note_context(footnote), number)).or_default().push((footnote.restart_sequence, footnote.pair_id.clone()));
        }
    }
    for footnote in footnotes {
        let context = decided.note_context(footnote);
        let (mut prior_byte, mut prior_scalar) = (0, 0);
        for capture in pattern.captures_iter(&footnote.body) {
            let Some(found) = capture.get(0) else {
                continue;
            };
            let number = capture.get(4).and_then(|value| value.as_str().parse::<u32>().ok()).unwrap_or(0);
            let candidates = by_number.get(&(context, number)).map(Vec::as_slice).unwrap_or_default();
            // One note of the number, or one in the note's own sequence.
            let mut scoped = candidates.iter().filter(|(restart, _)| candidates.len() == 1 || *restart == footnote.restart_sequence);
            let target_pair_id = match (scoped.next(), scoped.next()) {
                (Some((_, pair_id)), None) => pair_id.clone(),
                _ => String::new(),
            };
            let kind = match (capture.get(1), capture.get(2)) {
                (Some(value), _) => value.as_str().to_lowercase(),
                (None, Some(_)) => "op_cit".to_owned(),
                (None, None) => "see_footnote".to_owned(),
            };
            let start = prior_scalar + footnote.body[prior_byte..found.start()].chars().count();
            let end = start + found.as_str().chars().count();
            (prior_byte, prior_scalar) = (found.end(), end);
            let record = FootnoteCrossref {
                source_pair_id: footnote.pair_id.clone(),
                kind,
                number,
                shortform: legal_structure_model::structure_analysis().crossref_short_form(&footnote.body, found.start()),
                start,
                end,
                resolved: !target_pair_id.is_empty(),
                target_pair_id,
                target_count: candidates.len(),
            };
            if !record.resolved {
                let why = if candidates.is_empty() { "which no paired note carries - a pairing-quality witness" } else { "which has multiple possible targets" };
                let message = format!("Note {} references {} note {}, {why}.", footnote.label, record.kind, number);
                let mut diagnostic = Diagnostic::info("NOTE_CROSSREF_UNRESOLVED", message, footnote.reference_page.map(|page| page as usize));
                diagnostic.details.insert("crossref".to_owned(), json!(&record));
                diagnostics.push(diagnostic);
            }
            footnote.crossrefs.push(record);
        }
    }
}
