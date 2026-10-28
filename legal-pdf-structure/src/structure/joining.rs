//! Paragraphs assembled from lines: words a line breaks rejoined, and each paragraph's source runs.

use super::*;

/// A document's own dictionary for rejoining words its lines break: its words and its hyphenated
/// compounds (pdf-inspector's `document_vocabulary`).
pub(super) type Vocabulary = (HashSet<String>, HashSet<String>);

pub(super) fn vocabulary(pages: &[Page]) -> Vocabulary {
    let text: Vec<&str> = pages.iter().flat_map(|page| &page.lines).map(|line| line.text.trim()).collect();
    pdf_inspector::markdown::document_vocabulary(&text.join(" "))
}

/// A paragraph's lines joined into its text, with the source run of each. A word broken over two
/// lines is rejoined as pdf-inspector rejoins it: plain where the document writes it whole
/// ("de- fendant"), with its hyphen where it writes the compound or the continuation is
/// capitalized ("six- month", "Third- Party"), and otherwise plain before a lowercase continuation
/// ("judg- ment"), the paragraph's lines being one column's; a soft hyphen always joins.
pub(super) fn join_lines(lines: &[&Line], vocabulary: &Vocabulary) -> (String, Vec<legal_pdf_core::model::SourceTextSpan>) {
    use pdf_inspector::markdown::{join_decision, Join};
    let mut text = String::new();
    let mut source_spans: Vec<legal_pdf_core::model::SourceTextSpan> = Vec::new();
    for line in lines {
        let value = line.text.trim();
        if value.is_empty() {
            continue;
        }
        let first: &str = value.split(|c: char| !c.is_alphabetic()).next().unwrap_or("");
        let join = match text.chars().next_back() {
            Some('\u{00ad}') if !first.is_empty() => Join::Plain,
            Some(hyphen @ ('-' | '\u{00ac}')) => {
                let before = &text[..text.len() - hyphen.len_utf8()];
                let last = before.rsplit(|c: char| !c.is_alphabetic()).next().unwrap_or("");
                let decision = if last.chars().count() >= 2 && first.chars().count() >= 2 {
                    join_decision(last, first, &vocabulary.0, &vocabulary.1)
                } else {
                    Join::Keep
                };
                // Without the document's evidence, a lowercase continuation in one column's
                // paragraph is the word going on.
                match decision {
                    Join::Keep if !last.is_empty() && first.chars().next().is_some_and(char::is_lowercase) => Join::Plain,
                    decision => decision,
                }
            }
            _ => Join::Keep,
        };
        match join {
            Join::Plain => {
                text.pop();
                if let Some(last) = source_spans.last_mut() { last.end -= 1; last.source_end -= 1; }
                source_spans.retain(|span| span.start < span.end);
            }
            Join::Hyphen => {}
            Join::Keep if !text.is_empty() => text.push(' '),
            Join::Keep => {}
        }
        let output_start = legal_structure_model::utf16_len(&text);
        let source_start = legal_structure_model::utf16_len(&line.text[..line.text.len() - line.text.trim_start().len()]);
        let length = legal_structure_model::utf16_len(value);
        source_spans.push(legal_pdf_core::model::SourceTextSpan {start:output_start,end:output_start+length,
            line_id:line.id.clone(),source_start,source_end:source_start+length});
        text.push_str(value);
    }
    (text, source_spans)
}

pub(super) fn build_paragraphs_with_context(
    pages: &[Page],
    anchors: &HashMap<String, Vec<Anchor>>,
    primitives: &PdfPrimitiveEvidence,
) -> Vec<Paragraph> {
    let vocabulary = vocabulary(pages);
    let context = |id: &str| primitives.contexts.reading_context(id);
    let mut regions: Vec<(&Page, &str, Vec<&Line>)> = Vec::new();
    let mut tails = HashMap::<DocumentContext, usize>::new();
    for page in pages {
        let line_by_id: HashMap<&str, &Line> = page
            .lines
            .iter()
            .map(|line| (line.id.as_str(), line))
            .collect();
        let mut seen = HashSet::new();
        for region in &page.regions {
            if !matches!(region.kind.as_str(), "body" | "heading" | "caption" | "marginal_note") {
                continue;
            }
            let lines: Vec<&Line> = region
                .line_ids
                .iter()
                .filter_map(|id| line_by_id.get(id.as_str()).copied())
                .filter(|line| !line.exclude_from_body)
                .collect();
            for lines in lines.chunk_by(|left, right| context(&left.id) == context(&right.id)) {
                let key = context(&lines[0].id);
                let first = seen.insert(key);
                // The other column's intervening regions do not close this context's paragraph.
                if let Some(&slot) = tails.get(&key) {
                    let (_, kind, previous) = &mut regions[slot];
                    if first
                        && region.kind == "body"
                        && *kind == "body"
                        && crate::paragraphs::continues_onto(previous, lines)
                    {
                        previous.extend_from_slice(lines);
                        continue;
                    }
                }
                tails.insert(key, regions.len());
                regions.push((page, region.kind.as_str(), lines.to_vec()));
            }
        }
    }
    let mut paragraphs = Vec::new();
    for (page, kind, lines) in regions {
        let (text, source_spans) = join_lines(&lines, &vocabulary);
        if text.is_empty() {
            continue;
        }
        let mut events = Vec::new();
        let joined_coordinates = ScalarText::new(&text);
        for line in &lines {
            let coordinates = ScalarText::new(&line.text);
            for anchor in anchors.get(&line.id).into_iter().flatten() {
                let start = coordinates.utf16(anchor.start);
                let end = coordinates.utf16(anchor.end);
                let Some(run) = source_spans.iter().find(|span| span.line_id == line.id
                    && span.source_start <= start && end <= span.source_end) else { continue; };
                let start = joined_coordinates.scalar_at_utf16(run.start + start - run.source_start).unwrap();
                let end = joined_coordinates.scalar_at_utf16(run.start + end - run.source_start).unwrap();
                events.push((start,end,anchor.pair_id.as_str(),anchor.label.as_str()));
            }
        }
        events.sort_by_key(|event| (event.0, event.1));
        let (rendered, output_anchors, source_spans) = if events.is_empty() {
            (text, Vec::new(), source_spans)
        } else {
            let coordinates = ScalarText::new(&text);
            let text_len = coordinates.len();
            let mut rendered = String::with_capacity(text.len());
            let mut output_anchors = Vec::with_capacity(events.len());
            let mut retained_spans = Vec::new();
            let (mut cursor, mut rendered_len) = (0, 0);
            for (start, end, pair_id, label) in events {
                let start = start.max(cursor).min(text_len);
                let end = end.max(start).min(text_len);
                let output_start = legal_structure_model::utf16_len(&rendered);
                retained_spans.extend(source_spans.iter().filter_map(|span| span.retain(coordinates.utf16(cursor),coordinates.utf16(start),output_start)));
                rendered.push_str(coordinates.slice(cursor..start).unwrap());
                rendered_len += start - cursor;
                let offset = rendered_len;
                rendered.push_str("⟦FN:");
                rendered.push_str(&pair_id);
                rendered.push('⟧');
                rendered_len += 5 + pair_id.chars().count();
                output_anchors.push(ParagraphAnchor {
                    pair_id: pair_id.to_owned(),
                    label: label.to_owned(),
                    offset,
                });
                cursor = end;
            }
            let output_start = legal_structure_model::utf16_len(&rendered);
            retained_spans.extend(source_spans.iter().filter_map(|span| span.retain(coordinates.utf16(cursor),coordinates.utf16(text_len),output_start)));
            rendered.push_str(coordinates.slice(cursor..text_len).unwrap());
            (rendered, output_anchors, retained_spans)
        };
        paragraphs.push(Paragraph {
            source_spans,
            id: format!("para-{:06}", paragraphs.len() + 1),
            page_index: page.index,
            region_type: kind.to_owned(),
            text: rendered,
            line_ids: lines.iter().map(|line| line.id.clone()).collect(),
            anchors: output_anchors,
        });
    }
    paragraphs
}
