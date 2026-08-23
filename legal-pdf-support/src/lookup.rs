use legal_pdf_core::model::{Footnote, FootnoteLookup, LegalDocument};
use legal_pdf_core::{Error, Result};

pub fn normalize_decimal_digit(character: char) -> Option<char> {
    Some(match character {
        '⁰' => '0',
        '¹' => '1',
        '²' => '2',
        '³' => '3',
        '⁴' => '4',
        '⁵' => '5',
        '⁶' => '6',
        '⁷' => '7',
        '⁸' => '8',
        '⁹' => '9',
        value if value.is_ascii_digit() => value,
        _ => return None,
    })
}

pub fn normalize_note_symbol(character: char) -> char {
    match character {
        '∗' | '\u{f02a}' => '*',
        other => other,
    }
}

pub(crate) fn normal_label(value: &str) -> String {
    let translated: String = value
        .trim()
        .chars()
        .map(|character| normalize_decimal_digit(character).unwrap_or(character))
        .collect();
    if !translated.is_empty()
        && translated
            .chars()
            .all(|character| character.is_ascii_digit())
    {
        translated
            .parse::<u128>()
            .map_or(translated.clone(), |number| number.to_string())
    } else {
        translated
    }
}

pub(crate) fn validate_proposition_mode(mode: &str) -> Result<()> {
    if matches!(mode, "sentence" | "passage_since_prior_note") {
        Ok(())
    } else {
        Err(Error::Message(format!(
            "Unknown proposition mode: {mode:?}"
        )))
    }
}

fn result_for_matches(
    document: &LegalDocument,
    query: String,
    matches: Vec<&Footnote>,
    proposition_mode: &str,
) -> FootnoteLookup {
    if matches.is_empty() {
        return FootnoteLookup {
            status: "not_found".to_owned(),
            query,
            matches: vec![],
            footnote: None,
            proposition_mode: "sentence".to_owned(),
            proposition: String::new(),
            context: String::new(),
        };
    }
    if matches.len() > 1 {
        return FootnoteLookup {
            status: "ambiguous".to_owned(),
            query,
            matches: matches
                .into_iter()
                .map(|footnote| footnote.pair_id.clone())
                .collect(),
            footnote: None,
            proposition_mode: "sentence".to_owned(),
            proposition: String::new(),
            context: String::new(),
        };
    }

    let footnote = matches[0];
    let proposition = if proposition_mode == "sentence" {
        footnote.sentence_proposition.clone()
    } else {
        footnote.passage_since_prior_note.clone()
    };
    let context = footnote
        .reference_line_id
        .as_deref()
        .and_then(|line_id| {
            document
                .paragraphs
                .iter()
                .find(|paragraph| paragraph.line_ids.iter().any(|value| value == line_id))
        })
        .map_or_else(String::new, |paragraph| {
            paragraph.text.chars().take(2_000).collect()
        });
    FootnoteLookup {
        status: "found".to_owned(),
        query,
        matches: vec![footnote.pair_id.clone()],
        footnote: Some(footnote.clone()),
        proposition_mode: proposition_mode.to_owned(),
        proposition,
        context,
    }
}

pub fn lookup_footnote(
    document: &LegalDocument,
    label_or_pair_id: &str,
    page: Option<u32>,
    occurrence: Option<usize>,
    proposition_mode: &str,
) -> Result<FootnoteLookup> {
    validate_proposition_mode(proposition_mode)?;
    let query = label_or_pair_id.trim().to_owned();
    let label = normal_label(&query);
    let matches = document
        .footnotes
        .iter()
        .filter(|footnote| footnote.pair_id == query || footnote.label == label)
        .filter(|footnote| {
            page.is_none_or(|number| {
                footnote.reference_page == Some(number) || footnote.body_pages.contains(&number)
            })
        })
        .filter(|footnote| occurrence.is_none_or(|number| footnote.occurrence == number))
        .collect();
    Ok(result_for_matches(
        document,
        query,
        matches,
        proposition_mode,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use legal_pdf_core::model::{Paragraph, PARSER_VERSION, SCHEMA_VERSION};
    use serde_json::Map;

    fn note(pair_id: &str, occurrence: usize, reference_page: u32) -> Footnote {
        Footnote {
            pair_id: pair_id.to_owned(),
            label: "1".to_owned(),
            occurrence,
            restart_sequence: occurrence,
            reference_page: Some(reference_page),
            body_pages: vec![reference_page + 1],
            reference_line_id: Some(format!("line-{occurrence}")),
            body_line_ids: vec![format!("note-{occurrence}")],
            body: format!("Note {occurrence}."),
            sentence_proposition: format!("Sentence {occurrence}."),
            passage_since_prior_note: format!("Passage {occurrence}."),
            confidence: 1.0,
            provenance: "deterministic".to_owned(),
            warnings: vec![],
            crossrefs: vec![],
        }
    }

    fn document() -> LegalDocument {
        LegalDocument {
            document_id: "doc".to_owned(),
            source_name: "source.pdf".to_owned(),
            source_sha256: "00".repeat(32),
            page_count: 2,
            status: "ready".to_owned(),
            pages: vec![],
            paragraphs: vec![Paragraph {
                id: "paragraph".to_owned(),
                page_index: 0,
                region_type: "body".to_owned(),
                text: "Context for the second reference.".to_owned(),
                line_ids: vec!["line-2".to_owned()],
                anchors: vec![],
            }],
            footnotes: vec![note("pair-1", 1, 1), note("pair-2", 2, 1)],
            tables: vec![],
            images: vec![],
            structure_graph: serde_json::from_value(serde_json::json!({
                "schema_version": "legalpdf.document-structure.v1", "document_id": "doc",
                "offset_unit": "utf16", "text": "", "text_sha256": "00",
                "scope": {"kind": "complete"}, "origins": [], "nodes": [], "diagnostics": []
            }))
            .unwrap(),
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
    fn lookup_matches_oracle_ambiguity_hints_and_context() {
        let document = document();
        assert_eq!(
            lookup_footnote(&document, "¹", None, None, "sentence")
                .unwrap()
                .status,
            "ambiguous"
        );
        let found =
            lookup_footnote(&document, "1", Some(2), Some(2), "passage_since_prior_note").unwrap();
        assert_eq!(found.status, "found");
        assert_eq!(found.matches, ["pair-2"]);
        assert_eq!(found.proposition, "Passage 2.");
        assert_eq!(found.context, "Context for the second reference.");
    }
}
