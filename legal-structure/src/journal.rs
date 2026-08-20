use crate::source_doc::BlockFieldOrder;
use crate::{
    create_source_doc, EngineError, SourceDoc, SourceDocBlock, SourceDocKind, SourceDocOrigin,
    SourceDocProvider,
};
use serde::Deserialize;
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::io::BufRead;

#[derive(Clone)]
pub struct JournalPageLabel {
    pub label: String,
    pub pdf_page: usize,
}

#[derive(Deserialize)]
struct Page {
    article_id: Option<usize>,
    text: String,
    pdf_page: Option<usize>,
    #[serde(default)]
    regions: Vec<PageRegion>,
    #[serde(default)]
    annotations: Vec<Annotation>,
}

#[derive(Deserialize)]
struct PageRegion {
    order: Option<f64>,
    #[serde(rename = "type")]
    kind: Option<String>,
    text: String,
    #[serde(default)]
    lines: Vec<PageLine>,
}

#[derive(Deserialize)]
struct PageLine {
    codex_text_order: Option<usize>,
}

#[derive(Deserialize)]
struct Annotation {
    pair_id: Option<String>,
    pair_status: Option<String>,
    taxonomy_name: Option<String>,
    note_id: Option<Value>,
    selected_text: Option<Value>,
    start_line_order: Option<usize>,
}

#[derive(Clone, Copy)]
struct Region {
    start: usize,
    end: usize,
    pdf_page: Option<usize>,
}

struct Title {
    start: usize,
    label: Option<String>,
    aliases: Vec<String>,
}

fn utf16_len(value: &str) -> usize {
    value.encode_utf16().count()
}

fn public_label(prefix: &str, value: &str) -> String {
    let value = value
        .parse::<u64>()
        .map_or_else(|_| value.to_owned(), |value| value.to_string());
    format!("{prefix}{value}")
}

fn title(value: &str) -> (Option<String>, Vec<String>) {
    let compact = value.split_whitespace().collect::<Vec<_>>().join(" ");
    let numbered = compact.split_once('.').and_then(|(label, rest)| {
        (!rest.is_empty()
            && rest.starts_with(char::is_whitespace)
            && (label.len() == 1 && label.bytes().all(|byte| byte.is_ascii_uppercase())
                || !label.is_empty()
                    && label.bytes().all(|byte| {
                        matches!(byte, b'I' | b'V' | b'X' | b'L' | b'C' | b'D' | b'M')
                    })))
        .then_some((label, rest.trim()))
    });
    let name = numbered.map_or(compact.as_str(), |(_, rest)| rest);
    let mut normalized = String::new();
    let mut separated = true;
    for character in name.chars().flat_map(char::to_lowercase) {
        if character.is_alphanumeric() {
            normalized.push(character);
            separated = false;
        } else if !separated {
            normalized.push(' ');
            separated = true;
        }
    }
    let label = numbered.map(|(label, _)| label.to_owned());
    let mut aliases = label.iter().cloned().collect::<Vec<_>>();
    let normalized = normalized.trim();
    if !normalized.is_empty() {
        aliases.push(format!("sectitle:{normalized}"));
    }
    (label, aliases)
}

fn value_text(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(value)) => value.clone(),
        Some(Value::Number(value)) => value.to_string(),
        Some(Value::Bool(value)) => value.to_string(),
        _ => String::new(),
    }
}

pub fn journal_source_doc(
    article_id: usize,
    url: Option<String>,
    reader: impl BufRead,
    page_labels: &[JournalPageLabel],
) -> Result<SourceDoc, EngineError> {
    let mut text = String::new();
    let mut blocks = Vec::new();
    let mut titles = Vec::new();
    let mut paired_refs = HashSet::new();
    let mut notes = Vec::<(String, String, Option<Region>)>::new();
    let mut offset = 0;
    let mut paragraphs = 0;
    let mut pages = 0;

    for line in reader.lines() {
        let line = line.map_err(EngineError::source)?;
        if line.trim().is_empty() {
            continue;
        }
        let page: Page = serde_json::from_str(&line).map_err(EngineError::source)?;
        if page
            .article_id
            .filter(|value| *value > 0)
            .is_some_and(|value| value != article_id)
        {
            return Err(EngineError::source(
                "journal page belongs to another article",
            ));
        }
        if pages > 0 {
            text.push('\n');
            offset += 1;
        }
        pages += 1;
        let page_start = offset;
        text.push_str(&page.text);
        offset += utf16_len(&page.text);
        let pdf_page = page.pdf_page.filter(|value| *value > 0);
        if let Some(pdf_page) = pdf_page {
            let label = page_labels
                .iter()
                .find(|value| value.pdf_page == pdf_page)
                .map(|value| value.label.trim())
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .unwrap_or_else(|| pdf_page.to_string());
            let mut block = SourceDocBlock::new(
                SourceDocKind::Page,
                public_label("page", &label),
                page_start,
                offset,
                SourceDocOrigin::Native,
            )
            .with_field_order(BlockFieldOrder::Native);
            block.anchor = Some(format!("page={pdf_page}"));
            block.aliases.push(label);
            blocks.push(block);
        }

        let mut footnotes = HashMap::new();
        let mut cursor = 0;
        let mut regions = page.regions.into_iter().enumerate().collect::<Vec<_>>();
        regions.sort_by(|(left_index, left), (right_index, right)| {
            left.order
                .unwrap_or(*left_index as f64)
                .partial_cmp(&right.order.unwrap_or(*right_index as f64))
                .unwrap_or(Ordering::Equal)
        });
        for (_, region) in regions {
            if region.text.is_empty() {
                continue;
            }
            let Some(found) = page.text[cursor..].find(&region.text) else {
                continue;
            };
            let start_byte = cursor + found;
            cursor = start_byte + region.text.len();
            let placed = Region {
                start: page_start + utf16_len(&page.text[..start_byte]),
                end: page_start + utf16_len(&page.text[..cursor]),
                pdf_page,
            };
            paragraphs += 1;
            blocks.push(SourceDocBlock::new(
                SourceDocKind::Paragraph,
                format!("par{paragraphs}"),
                placed.start,
                placed.end,
                SourceDocOrigin::Native,
            ));
            if region.kind.as_deref() == Some("paragraph_title") {
                let (label, aliases) = title(&region.text);
                titles.push(Title {
                    start: placed.start,
                    label,
                    aliases,
                });
            }
            if region.kind.as_deref() == Some("footnote") {
                for line in region.lines {
                    if let Some(order) = line.codex_text_order.filter(|value| *value > 0) {
                        footnotes.entry(order).or_insert(placed);
                    }
                }
            }
        }
        for annotation in page.annotations {
            let pair = annotation.pair_id.as_deref().unwrap_or_default();
            if pair.is_empty() || annotation.pair_status.as_deref() != Some("paired") {
                continue;
            }
            match annotation.taxonomy_name.as_deref() {
                Some("fn_ref") => {
                    paired_refs.insert(pair.to_owned());
                }
                Some("fn_label") => {
                    let note = value_text(
                        annotation
                            .note_id
                            .as_ref()
                            .or(annotation.selected_text.as_ref()),
                    )
                    .trim()
                    .to_owned();
                    if let (false, Some(order)) = (
                        note.is_empty(),
                        annotation.start_line_order.filter(|value| *value > 0),
                    ) {
                        notes.push((pair.to_owned(), note, footnotes.get(&order).copied()));
                    }
                }
                _ => {}
            }
        }
    }
    if pages == 0 || text.trim().is_empty() {
        return Err(EngineError::source("journal export has no usable pages"));
    }

    for (index, title) in titles.iter().enumerate() {
        let mut block = SourceDocBlock::new(
            SourceDocKind::Section,
            title.label.as_ref().map_or_else(
                || format!("secTitle{}", index + 1),
                |label| format!("sec{label}"),
            ),
            title.start,
            titles.get(index + 1).map_or(offset, |value| value.start),
            SourceDocOrigin::Native,
        )
        .with_field_order(BlockFieldOrder::AliasesBeforeOrigin);
        block.aliases.clone_from(&title.aliases);
        blocks.push(block);
    }
    let mut used_pairs = HashSet::new();
    for (pair, note, region) in notes {
        if !paired_refs.contains(&pair) || !used_pairs.insert(pair) {
            continue;
        }
        let Some(region) = region else { continue };
        let mut block = SourceDocBlock::new(
            SourceDocKind::Footnote,
            public_label("fn", &note),
            region.start,
            region.end,
            SourceDocOrigin::Native,
        )
        .with_field_order(BlockFieldOrder::AliasesAnchorBeforeOrigin);
        block.aliases.push(note);
        block.anchor = region.pdf_page.map(|page| format!("page={page}"));
        blocks.push(block);
    }
    blocks.sort_by_key(|block| (block.start, block.end));
    Ok(create_source_doc(
        Some(SourceDocProvider::Journal),
        article_id.to_string(),
        url,
        None,
        text,
        blocks,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn authoritative_pages_only_add_geometry_prose_slices() {
        let pages = concat!(
            r#"{"article_id":1,"text":"TITLE\nBody\n1 Note","pdf_page":1,"regions":[{"order":0,"type":"paragraph_title","text":"TITLE"},{"order":1,"type":"paragraph","text":"Body"},{"order":2,"type":"footnote","text":"1 Note","lines":[{"codex_text_order":7}]}],"annotations":[{"pair_id":"p","pair_status":"paired","taxonomy_name":"fn_ref"},{"pair_id":"p","pair_status":"paired","taxonomy_name":"fn_label","note_id":"1","start_line_order":7}]}"#,
            "\n",
        );
        let doc = journal_source_doc(
            1,
            None,
            Cursor::new(pages),
            &[JournalPageLabel {
                label: "7".into(),
                pdf_page: 1,
            }],
        )
        .unwrap();
        assert_eq!(doc.text, "TITLE\nBody\n1 Note");
        assert_eq!(
            doc.blocks
                .iter()
                .map(|block| block.label.as_str())
                .collect::<Vec<_>>(),
            ["par1", "page7", "secTitle1", "par2", "par3", "fn1"]
        );
    }
}
