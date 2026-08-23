use legal_pdf_core::model::LegalDocument;
use legal_pdf_core::{Error, Result};
use serde_json::{json, Map, Value};
use std::collections::HashMap;

fn anchor_pair_id(anchor: &Value) -> Result<String> {
    anchor
        .get("pair_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| Error::Message("footnote anchor has no pair_id".to_owned()))
}

fn anchor_offset(anchor: &Value) -> Result<usize> {
    anchor
        .get("offset")
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .ok_or_else(|| Error::Message("footnote anchor has no offset".to_owned()))
}

fn char_to_byte(value: &str, index: usize) -> Option<usize> {
    if index == value.chars().count() {
        Some(value.len())
    } else {
        value.char_indices().nth(index).map(|(offset, _)| offset)
    }
}

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
                let Some(pair_id) = anchor.get("pair_id").and_then(Value::as_str) else {
                    continue;
                };
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
            "pairing_summary": document.metadata.get("pairing").cloned().unwrap_or_else(|| json!({})),
            "pdf_line_count": document.line_count(),
            "legalpdf_usable_footnotes": usable.len(),
            "legalpdf_omitted_unusable_footnotes": document.footnotes.len() - usable.len(),
        },
    })
}

pub fn to_toa_text_units(document: &LegalDocument) -> Result<Vec<Value>> {
    let internal_by_pair: HashMap<&str, usize> = document
        .footnotes
        .iter()
        .enumerate()
        .map(|(index, note)| (note.pair_id.as_str(), index + 1))
        .collect();
    let mut units = Vec::with_capacity(document.paragraphs.len() + document.footnotes.len());
    for (ordinal, paragraph) in document.paragraphs.iter().enumerate() {
        let mut anchors = paragraph.anchors.iter().collect::<Vec<_>>();
        anchors.sort_by_key(|anchor| anchor_offset(anchor).unwrap_or(usize::MAX));
        let mut rendered = String::new();
        let mut references = Vec::new();
        let mut cursor = 0;
        let mut clean_length = 0;
        for anchor in anchors {
            let pair_id = anchor_pair_id(anchor)?;
            let marker = format!("⟦FN:{pair_id}⟧");
            let start = anchor_offset(anchor)?;
            let byte_start = char_to_byte(&paragraph.text, start).ok_or_else(|| {
                Error::Message(format!("Invalid footnote anchor in {}", paragraph.id))
            })?;
            if !paragraph.text[byte_start..].starts_with(&marker) {
                return Err(Error::Message(format!(
                    "Invalid footnote anchor in {}",
                    paragraph.id
                )));
            }
            let byte_cursor = char_to_byte(&paragraph.text, cursor).ok_or_else(|| {
                Error::Message(format!("Invalid footnote anchor in {}", paragraph.id))
            })?;
            let segment = &paragraph.text[byte_cursor..byte_start];
            rendered.push_str(segment);
            clean_length += segment.chars().count();
            let internal = internal_by_pair.get(pair_id.as_str()).ok_or_else(|| {
                Error::Message(format!(
                    "Unknown footnote pair {pair_id} in {}",
                    paragraph.id
                ))
            })?;
            references.push(json!([internal, clean_length]));
            cursor = start + marker.chars().count();
        }
        let byte_cursor = char_to_byte(&paragraph.text, cursor).ok_or_else(|| {
            Error::Message(format!("Invalid footnote anchor in {}", paragraph.id))
        })?;
        rendered.push_str(&paragraph.text[byte_cursor..]);
        units.push(json!({
            "key": format!("body:{ordinal}"),
            "kind": "body",
            "ordinal": ordinal,
            "footnote_id": Value::Null,
            "text": rendered,
            "footnote_refs": references,
        }));
    }
    units.extend(document.footnotes.iter().enumerate().map(|(index, note)| {
        let ordinal = index + 1;
        json!({
            "key": format!("footnote:{ordinal}"),
            "kind": "footnote",
            "ordinal": ordinal,
            "footnote_id": ordinal,
            "text": note.body,
            "footnote_refs": [],
        })
    }));
    Ok(units)
}

#[cfg(test)]
mod tests {
    use super::*;
    use legal_pdf_core::model::{
        Footnote, GraphStatus, Paragraph, DocumentStructure, PARSER_VERSION, SCHEMA_VERSION,
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

    fn structure_graph() -> DocumentStructure {
        DocumentStructure::from_parts(
            "doc".to_owned(),
            "",
            Some("00".repeat(32)),
            GraphStatus::Complete,
            vec![],
            vec![],
            vec![],
            vec![],
        )
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
                    json!({"pair_id": first, "offset": 5}),
                    json!({"pair_id": second, "offset": 5 + first_marker.chars().count() + 5}),
                ],
                text,
                line_ids: vec!["body-line".to_owned()],
            }],
            footnotes: vec![
                note(first, "First note.", true),
                note(second, "Second note.", true),
                note("omitted", "No reference.", false),
            ],
            tables: vec![],
            images: vec![],
            structure_graph: structure_graph(),
            pdf_source_map: Default::default(),
            pairing_audit: None,
            diagnostics: vec![],
            repairs: vec![],
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
    }
}
