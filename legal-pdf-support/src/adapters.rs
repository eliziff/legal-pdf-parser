use legal_pdf_core::model::{Footnote, LegalDocument, Paragraph};
use legal_pdf_core::{Error, Result};
use legal_structure_model::ScalarText;
use serde_json::{json, Map, Value};
use std::collections::HashMap;

fn replace_first(value: &mut String, needle: &str, replacement: &str) -> Option<usize> {
    let byte = value.find(needle)?;
    let offset = value[..byte].chars().count();
    value.replace_range(byte..byte + needle.len(), replacement);
    Some(offset)
}

pub fn to_alr_payload(document: &LegalDocument) -> Value {
    let usable: Vec<_> = document
        .footnotes
        .iter()
        .filter(|note| note.reference_line_id.is_some() && !note.body_line_ids.is_empty())
        .collect();
    let internal_by_pair: HashMap<&str, usize> = usable
        .iter()
        .enumerate()
        .map(|(index, note)| (note.pair_id.as_str(), index + 1))
        .collect();
    let paragraphs: Vec<Value> = document
        .paragraphs
        .iter()
        .map(|paragraph| {
            let mut text = paragraph.text.clone();
            let mut anchors = Vec::new();
            for anchor in &paragraph.anchors {
                let pair_id = anchor.pair_id.as_str();
                let Some(internal) = internal_by_pair.get(pair_id) else {
                    continue;
                };
                let marker = format!("⟦FN:{pair_id}⟧");
                let replacement = format!("⟦FN:{internal}⟧");
                let offset = replace_first(&mut text, &marker, &replacement).unwrap_or(0);
                anchors.push(json!({
                    "footnote_id": internal,
                    "offset": offset,
                    "pair_id": pair_id,
                }));
            }
            json!({
                "style_id": Value::Null,
                "style_name": if paragraph.region_type == "heading" { Value::String("Heading".to_owned()) } else { Value::Null },
                "effective_indent_left": Value::Null,
                "text": text,
                "anchors": anchors,
            })
        })
        .collect();
    let footnotes: Map<String, Value> = usable
        .iter()
        .enumerate()
        .map(|(index, note)| ((index + 1).to_string(), Value::String(note.body.clone())))
        .collect();
    json!({
        "schema_version": "legalpdf.adapter.alr.v1",
        "paragraphs": paragraphs,
        "footnotes": footnotes,
        "footnote_order": (1..=usable.len()).collect::<Vec<_>>(),
        "source_kind": "PDF",
        "metadata": {
            "legalpdf_document_id": document.document_id,
            "legalpdf_source_sha256": document.source_sha256,
            "pdf_line_count": document.line_count(),
            "legalpdf_usable_footnotes": usable.len(),
            "legalpdf_omitted_unusable_footnotes": document.footnotes.len() - usable.len(),
        },
    })
}

pub fn to_toa_text_units(document: &LegalDocument) -> Result<Vec<Value>> {
    to_toa_text_units_from_parts(&document.paragraphs, &document.footnotes)
}

pub(crate) fn to_toa_text_units_from_parts(
    paragraphs: &[Paragraph],
    footnotes: &[Footnote],
) -> Result<Vec<Value>> {
    let internal_by_pair: HashMap<&str, usize> = footnotes
        .iter()
        .enumerate()
        .map(|(index, note)| (note.pair_id.as_str(), index + 1))
        .collect();
    let mut units = Vec::with_capacity(paragraphs.len() + footnotes.len());
    for (ordinal, paragraph) in paragraphs.iter().enumerate() {
        let coordinates = ScalarText::new(&paragraph.text);
        let mut anchors = paragraph.anchors.iter().collect::<Vec<_>>();
        anchors.sort_by_key(|anchor| anchor.offset);
        let mut rendered = String::new();
        let mut references = Vec::new();
        let mut cursor = 0;
        let mut clean_length = 0;
        for anchor in anchors {
            let pair_id = &anchor.pair_id;
            let marker = format!("⟦FN:{pair_id}⟧");
            let start = anchor.offset;
            let byte_start = coordinates.byte_at_scalar(start).ok_or_else(|| {
                Error::Message(format!("Invalid footnote anchor in {}", paragraph.id))
            })?;
            if !paragraph.text[byte_start..].starts_with(&marker) {
                return Err(Error::Message(format!(
                    "Invalid footnote anchor in {}",
                    paragraph.id
                )));
            }
            let byte_cursor = coordinates.byte_at_scalar(cursor).ok_or_else(|| {
                Error::Message(format!("Invalid footnote anchor in {}", paragraph.id))
            })?;
            let segment = &paragraph.text[byte_cursor..byte_start];
            rendered.push_str(segment);
            clean_length += legal_structure_model::utf16_len(segment);
            let internal = internal_by_pair.get(pair_id.as_str()).ok_or_else(|| {
                Error::Message(format!(
                    "Unknown footnote pair {pair_id} in {}",
                    paragraph.id
                ))
            })?;
            references.push(json!([internal, clean_length]));
            cursor = start + marker.chars().count();
        }
        let byte_cursor = coordinates.byte_at_scalar(cursor).ok_or_else(|| {
            Error::Message(format!("Invalid footnote anchor in {}", paragraph.id))
        })?;
        rendered.push_str(&paragraph.text[byte_cursor..]);
        units.push(json!({
            "key": format!("body:{ordinal}"),
            "kind": "body",
            "ordinal": ordinal,
            "footnote_id": Value::Null,
            "page_numbers": [paragraph.page_index + 1],
            "text": rendered,
            "footnote_refs": references,
        }));
    }
    units.extend(footnotes.iter().enumerate().map(|(index, note)| {
        let ordinal = index + 1;
        let mut pages = note.body_pages.clone();
        if pages.is_empty() {
            pages.extend(note.reference_page);
        }
        pages.sort_unstable();
        pages.dedup();
        let number = note.label.parse::<u32>().ok();
        // A note with its own mark (an author's "*") keeps the mark, as the
        // document prints it and as Word's text does: it takes no number.
        let text = if number.is_some() || note.label.is_empty() || note.body == note.label {
            note.body.clone()
        } else {
            format!("{} {}", note.label, note.body)
        };
        json!({
            "key": format!("footnote:{ordinal}"),
            "kind": "footnote",
            "ordinal": ordinal,
            "footnote_id": ordinal,
            "note_number": number,
            "restart_sequence": note.restart_sequence,
            "page_numbers": pages,
            "text": text,
            "footnote_refs": [],
        })
    }));
    join_split_words(&mut units);
    Ok(units)
}

/// A word a hyphen splits where no compound is written ("Companies' Cr-editors Arrangement Act",
/// "Cred-⏎itors") is one word when the document writes it whole elsewhere ("Creditors"): a line
/// break's hyphen, or a recognizer's, read into the text. The hyphen goes, with a line break after
/// it; a compound whose second part opens with a capital or a digit ("Non-Renewable", "C-36"),
/// or that the document never writes whole, keeps it. The note markers' offsets move with the text.
fn join_split_words(units: &mut [Value]) {
    let texts = units.iter().filter_map(|unit| unit["text"].as_str()).collect::<Vec<_>>();
    let mut whole = HashMap::<String, usize>::new();
    for word in texts.iter().flat_map(|text| text.split(|character: char| !character.is_alphabetic())) {
        if !word.is_empty() { *whole.entry(word.to_lowercase()).or_default() += 1; }
    }
    for unit in units.iter_mut() {
        let Some(text) = unit["text"].as_str() else { continue };
        let characters = text.char_indices().collect::<Vec<_>>();
        let mut removed = Vec::<(usize, usize)>::new(); // (utf16 offset, utf16 units removed)
        let mut output = String::with_capacity(text.len());
        let (mut copied, mut utf16, mut at) = (0usize, 0usize, 0usize);
        while at < characters.len() {
            let (byte, character) = characters[at];
            if character != '-' || at == 0 || !characters[at - 1].1.is_alphabetic() { at += 1; continue; }
            // The word's first part runs back to its start; the second opens after the hyphen
            // and any line break.
            let first = characters[..at].iter().rev().take_while(|(_, c)| c.is_alphabetic()).count();
            let mut next = at + 1;
            while next < characters.len() && matches!(characters[next].1, ' ' | '\n' | '\r') { next += 1; }
            let gap = &characters[at + 1..next];
            let broken = gap.is_empty() || gap.iter().any(|(_, c)| *c == '\n');
            let second = characters[next..].iter().take_while(|(_, c)| c.is_alphabetic()).count();
            if !broken || second < 2 || !characters[next].1.is_lowercase()
                || at >= first + 1 && characters[at - first - 1].1 == '-' {
                at += 1; continue;
            }
            let start_byte = characters[at - first].0;
            let end_byte = characters.get(next + second).map_or(text.len(), |(b, _)| *b);
            let joined = text[start_byte..end_byte].chars().filter(|c| c.is_alphabetic()).collect::<String>().to_lowercase();
            if whole.get(&joined).copied().unwrap_or(0) == 0 { at += 1; continue; }
            let cut_end = characters[next].0;
            output.push_str(&text[copied..byte]);
            utf16 += text[copied..byte].encode_utf16().count();
            let cut = text[byte..cut_end].encode_utf16().count();
            removed.push((utf16, cut));
            copied = cut_end;
            at = next;
        }
        if removed.is_empty() { continue; }
        output.push_str(&text[copied..]);
        if let Some(references) = unit["footnote_refs"].as_array_mut() {
            for reference in references.iter_mut() {
                let Some(offset) = reference.get(1).and_then(Value::as_u64) else { continue };
                let shift = removed.iter().filter(|(at, _)| (*at as u64) < offset).map(|(_, cut)| *cut as u64).sum::<u64>();
                reference[1] = json!(offset - shift);
            }
        }
        unit["text"] = Value::String(output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use legal_pdf_core::model::{
        Footnote, Paragraph, ParagraphAnchor, PARSER_VERSION, SCHEMA_VERSION,
    };

    fn note(pair_id: &str, body: &str, usable: bool) -> Footnote {
        Footnote {
            pair_id: pair_id.to_owned(),
            label: "1".to_owned(),
            occurrence: 1,
            restart_sequence: 1,
            reference_page: Some(1),
            body_pages: vec![1],
            reference_line_id: usable.then(|| "body-line".to_owned()),
            body_line_ids: usable
                .then(|| vec!["note-line".to_owned()])
                .unwrap_or_default(),
            body: body.to_owned(),
            sentence_proposition: String::new(),
            passage_since_prior_note: String::new(),
            confidence: 1.0,
            provenance: "deterministic".to_owned(),
            warnings: vec![],
            crossrefs: vec![],
        }
    }

    fn document() -> LegalDocument {
        let first = "first";
        let second = "second";
        let first_marker = format!("⟦FN:{first}⟧");
        let second_marker = format!("⟦FN:{second}⟧");
        let text = format!("Alpha{first_marker} beta{second_marker} gamma.");
        LegalDocument {
            document_id: "doc".to_owned(),
            source_name: "source.pdf".to_owned(),
            source_sha256: "00".repeat(32),
            page_count: 1,
            status: "ready".to_owned(),
            pages: vec![],
            paragraphs: vec![Paragraph {
                id: "paragraph-1".to_owned(),
                page_index: 0,
                region_type: "body".to_owned(),
                anchors: vec![
                    ParagraphAnchor {
                        pair_id: first.to_owned(),
                        label: "1".to_owned(),
                        offset: 5,
                    },
                    ParagraphAnchor {
                        pair_id: second.to_owned(),
                        label: "2".to_owned(),
                        offset: 5 + first_marker.chars().count() + 5,
                    },
                ],
                text,
                line_ids: vec!["body-line".to_owned()],
            }],
            footnotes: vec![
                note(first, "First note.", true),
                note(second, "Second note.", true),
                note("omitted", "No reference.", false),
            ],
            structure_graph: serde_json::from_value(serde_json::json!({
                "schema_version": "legalpdf.document-structure.v1", "document_id": "doc",
                "offset_unit": "utf16", "text": "", "text_sha256": "00",
                "scope": {"kind": "complete"}, "origins": [], "nodes": [], "diagnostics": []
            }))
            .unwrap(),
            diagnostics: vec![],
            metadata: Map::new(),
            provenance: Map::new(),
            schema_version: SCHEMA_VERSION.to_owned(),
            parser_version: PARSER_VERSION.to_owned(),
        }
    }

    #[test]
    fn adapters_preserve_oracle_numbering_and_clean_offsets() {
        let document = document();
        let alr = to_alr_payload(&document);
        assert_eq!(alr["footnote_order"], json!([1, 2]));
        assert_eq!(alr["metadata"]["legalpdf_omitted_unusable_footnotes"], 1);
        assert_eq!(
            alr["paragraphs"][0]["text"],
            "Alpha⟦FN:1⟧ beta⟦FN:2⟧ gamma."
        );

        let toa = to_toa_text_units(&document).unwrap();
        assert_eq!(toa[0]["text"], "Alpha beta gamma.");
        assert_eq!(toa[0]["footnote_refs"], json!([[1, 5], [2, 10]]));
        assert_eq!(toa[0]["page_numbers"], json!([1]));
        assert_eq!(toa[1]["page_numbers"], json!([1]));
    }

    #[test]
    fn toa_reference_offsets_are_javascript_utf16() {
        let mut document = document();
        let marker = "⟦FN:first⟧";
        document.paragraphs[0].text = format!("😀{marker}");
        document.paragraphs[0].anchors = vec![ParagraphAnchor {
            pair_id: "first".to_owned(),
            label: "1".to_owned(),
            offset: 1,
        }];

        let toa = to_toa_text_units(&document).unwrap();
        assert_eq!(toa[0]["text"], "😀");
        assert_eq!(toa[0]["footnote_refs"], json!([[1, 2]]));
    }

    #[test]
    fn a_note_with_its_own_mark_keeps_it_and_takes_no_number() {
        let mut document = document();
        document.footnotes[0].label = "*".to_owned();
        document.footnotes[0].body = "The author thanks the archivists.".to_owned();

        let toa = to_toa_text_units(&document).unwrap();
        assert_eq!(toa[1]["text"], "* The author thanks the archivists.");
        assert_eq!(toa[1]["note_number"], Value::Null);
        assert_eq!(toa[2]["text"], "Second note.");
        assert_eq!(toa[2]["note_number"], 1);
    }

    #[test]
    fn toa_footnotes_prefer_body_pages_and_fallback_to_the_reference_page() {
        let mut document = document();
        document.footnotes[0].body_pages = vec![4, 2, 4];
        document.footnotes[1].body_pages.clear();
        document.footnotes[1].reference_page = Some(3);

        let toa = to_toa_text_units(&document).unwrap();
        assert_eq!(toa[1]["page_numbers"], json!([2, 4]));
        assert_eq!(toa[2]["page_numbers"], json!([3]));
    }
}
