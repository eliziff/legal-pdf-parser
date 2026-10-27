//! What a note says of the text around it: its proposition (the sentence it attaches to), its
//! cross-references to other notes, and the references no note answers.

use super::*;

pub(super) fn marker_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"⟦FN:[^⟧]+⟧").unwrap())
}

pub(super) fn clean_markers(value: &str) -> String {
    marker_re().replace_all(value, "").trim().to_owned()
}

pub(super) fn sentence_boundaries(chars: &[char]) -> Vec<(usize, usize)> {
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
    boundaries
}

pub(super) fn sentence_at_boundaries(chars: &[char], boundaries: &[(usize, usize)], offset: usize) -> String {
    let previous = &boundaries[..boundaries.partition_point(|(_, end)| *end <= offset)];
    let (start, end) = if previous.last().is_some_and(|(_, end)| {
        chars[*end..offset.min(chars.len())]
            .iter()
            .all(|character| character.is_whitespace())
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
    clean_markers(&chars[start..end].iter().collect::<String>())
}

pub(super) fn attach_propositions(
    footnotes: &mut [Footnote],
    paragraphs: &[Paragraph],
    primitives: &PdfPrimitiveEvidence,
) {
    let by_pair: HashMap<String, usize> = footnotes
        .iter()
        .enumerate()
        .map(|(index, footnote)| (footnote.pair_id.clone(), index))
        .collect();
    let mut tails = HashMap::<DocumentContext, String>::new();
    for paragraph in paragraphs {
        let context = paragraph
            .line_ids
            .first()
            .map_or((0, false), |id| primitives.contexts.document_context(id));
        let previous_tail = tails.entry(context).or_default();
        let chars: Vec<char> = paragraph.text.chars().collect();
        let boundaries = sentence_boundaries(&chars);
        let mut previous_offset = 0;
        for anchor in &paragraph.anchors {
            let Some(&index) = by_pair.get(&anchor.pair_id) else {
                continue;
            };
            let offset = anchor.offset;
            footnotes[index].sentence_proposition =
                sentence_at_boundaries(&chars, &boundaries, offset);
            let passage_text: String = chars
                [previous_offset.min(chars.len())..offset.min(chars.len())]
                .iter()
                .collect();
            let passage = clean_markers(&passage_text);
            footnotes[index].passage_since_prior_note = if passage.is_empty() {
                previous_tail.clone()
            } else {
                passage
            };
            previous_offset = offset + 5 + anchor.pair_id.chars().count();
        }
        *previous_tail = chars[previous_offset.min(chars.len())..]
            .iter()
            .collect::<String>()
            .trim()
            .to_owned();
    }
}

pub(super) fn unmatched_reference_diagnostics(
    pages: &[Page],
    footnotes: &[Footnote],
    anchors: &HashMap<String, Vec<Anchor>>,
    primitives: &PdfPrimitiveEvidence,
) -> Vec<Diagnostic> {
    let labels: HashSet<_> = footnotes
        .iter()
        .map(|note| {
            let context = primitives.contexts.note_context(note);
            (context, note.label.as_str())
        })
        .collect();
    let primary_lines: HashSet<&str> = footnotes
        .iter()
        .filter_map(|note| note.reference_line_id.as_deref())
        .collect();
    let mut diagnostics = Vec::new();
    for line in pages.iter().flat_map(|page| page.lines.iter()) {
        for detached in &line.detached_references {
            let label = normalize_label(&detached.note_id);
            let start = detached.start_offset;
            let end = detached.end_offset;
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
                if !detached.source_line_id.is_empty() {
                    diagnostic.line_ids.push(detached.source_line_id.clone());
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
        if line.spans.iter().any(|span| {
            span.superscript
                && labels.contains(&(
                    primitives.contexts.document_context(&line.id),
                    normalize_label(span.text.trim()).as_str(),
                ))
        }) {
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

pub(super) fn attach_crossrefs(
    footnotes: &mut [Footnote],
    diagnostics: &mut Vec<Diagnostic>,
    primitives: &PdfPrimitiveEvidence,
) {
    static RE: OnceLock<Regex> = OnceLock::new();
    let pattern = RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:(supra|infra),?\s+(?:foot)?notes?|(op)\.?\s*cit\.?,?\s+(?:foot)?notes?|(see)\s+(?:also\s+)?footnote)\s+(\d{1,3})\b",
        )
        .unwrap()
    });
    let mut by_number: HashMap<(DocumentContext, u32), Vec<(usize, String)>> = HashMap::new();
    for footnote in footnotes.iter() {
        if let Ok(number) = footnote.label.parse::<u32>() {
            by_number
                .entry((primitives.contexts.note_context(footnote), number))
                .or_default()
                .push((footnote.restart_sequence, footnote.pair_id.clone()));
        }
    }
    for footnote in footnotes {
        let context = primitives.contexts.note_context(footnote);
        let (mut prior_byte, mut prior_scalar) = (0, 0);
        for capture in pattern.captures_iter(&footnote.body) {
            let Some(found) = capture.get(0) else {
                continue;
            };
            let number = capture
                .get(4)
                .and_then(|value| value.as_str().parse::<u32>().ok())
                .unwrap_or(0);
            let candidates = by_number
                .get(&(context, number))
                .map(Vec::as_slice)
                .unwrap_or_default();
            let target_pair_id = if candidates.len() == 1 {
                candidates[0].1.clone()
            } else {
                let mut scoped = candidates
                    .iter()
                    .filter(|(restart, _)| *restart == footnote.restart_sequence);
                match (scoped.next(), scoped.next()) {
                    (Some((_, pair_id)), None) => pair_id.clone(),
                    _ => String::new(),
                }
            };
            let kind = if let Some(value) = capture.get(1) {
                value.as_str().to_lowercase()
            } else if capture.get(2).is_some() {
                "op_cit".to_owned()
            } else {
                "see_footnote".to_owned()
            };
            let start = prior_scalar + footnote.body[prior_byte..found.start()].chars().count();
            let end = start + found.as_str().chars().count();
            (prior_byte, prior_scalar) = (found.end(), end);
            let record = FootnoteCrossref {
                source_pair_id: footnote.pair_id.clone(),
                kind,
                number,
                shortform: layout_analysis().crossref_short_form(&footnote.body, found.start()),
                start,
                end,
                resolved: !target_pair_id.is_empty(),
                target_pair_id,
                target_count: candidates.len(),
            };
            if !record.resolved {
                let mut diagnostic = Diagnostic::info(
                    "NOTE_CROSSREF_UNRESOLVED",
                    if candidates.is_empty() {
                        format!(
                        "Note {} references {} note {}, which no paired note carries - a pairing-quality witness.",
                        footnote.label, record.kind, number
                    )
                    } else {
                        format!(
                            "Note {} references {} note {}, which has multiple possible targets.",
                            footnote.label, record.kind, number
                        )
                    },
                    footnote.reference_page.map(|page| page as usize),
                );
                diagnostic
                    .details
                    .insert("crossref".to_owned(), json!(&record));
                diagnostics.push(diagnostic);
            }
            footnote.crossrefs.push(record);
        }
    }
}
