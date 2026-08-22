use crate::source_doc::BlockFieldOrder;
use crate::{
    create_source_doc, utf16_len, EngineError, ScalarText, SourceDoc, SourceDocBlock,
    SourceDocKind, SourceDocOrigin, SourceDocProvider,
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
    article_id: Option<Value>,
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
    #[serde(default)]
    text: String,
    #[serde(default)]
    lines: Vec<PageLine>,
}

#[derive(Deserialize)]
struct PageLine {
    codex_text_order: Option<usize>,
    #[serde(default)]
    text: String,
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

fn page_marker(line: &str) -> Option<&str> {
    let line = line.strip_suffix('\n').unwrap_or(line);
    let line = line.strip_suffix('\r').unwrap_or(line);
    let label = line.strip_prefix("[page ")?.strip_suffix(']')?.trim();
    (!label.is_empty() && label.len() <= 40 && !label.contains(']')).then_some(label)
}

fn positive_usize(value: Option<&Value>) -> Option<usize> {
    let value = match value? {
        Value::Number(value) => value.as_u64()?,
        Value::String(value) => value.trim().parse().ok()?,
        _ => return None,
    };
    (value > 0 && value <= 9_007_199_254_740_991)
        .then_some(value)
        .and_then(|value| value.try_into().ok())
}

pub fn journal_text_source_doc(
    article_id: usize,
    url: Option<String>,
    text: String,
    page_labels: &[JournalPageLabel],
) -> Result<SourceDoc, EngineError> {
    // Marker lines are removed first. Page offsets therefore address the clean
    // rendered text, retaining every CR/LF code unit on non-marker lines.
    let mut starts = Vec::new();
    let mut clean = String::with_capacity(text.len());
    let mut page_cursor = 0;
    for line in text.split_inclusive('\n') {
        if let Some(label) = page_marker(line) {
            let row = page_labels[page_cursor..]
                .iter()
                .position(|row| row.label.trim() == label)
                .map(|index| page_cursor + index);
            let pdf_page = row.map(|index| {
                page_cursor = index + 1;
                page_labels[index].pdf_page
            });
            starts.push((label.to_owned(), pdf_page, utf16_len(&clean)));
        } else {
            clean.push_str(line);
        }
    }
    let mut blocks = Vec::with_capacity(starts.len());
    for (index, (label, pdf_page, start)) in starts.iter().enumerate() {
        let mut block = SourceDocBlock::new(
            SourceDocKind::Page,
            public_label("page", label),
            *start,
            starts
                .get(index + 1)
                .map_or_else(|| utf16_len(&clean), |value| value.2),
            SourceDocOrigin::Native,
        )
        .with_field_order(BlockFieldOrder::EndLast);
        block.anchor = pdf_page.map(|pdf_page| format!("page={pdf_page}"));
        block.aliases.push(label.clone());
        blocks.push(block);
    }
    Ok(create_source_doc(
        Some(SourceDocProvider::Journal),
        article_id.to_string(),
        url,
        None,
        clean,
        blocks,
    ))
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
        if positive_usize(page.article_id.as_ref()).is_some_and(|value| value != article_id) {
            return Err(EngineError::source(
                "journal page belongs to another article",
            ));
        }
        if pages > 0 {
            text.push('\n');
            offset += 1;
        }
        pages += 1;
        // JSON page text is appended unchanged, with one synthetic LF between
        // pages. Region matches are byte offsets in the original page string
        // and are converted exactly into that final rendered UTF-16 plane.
        let page_start = offset;
        let page_coordinates = ScalarText::new(&page.text);
        text.push_str(&page.text);
        offset += page_coordinates.utf16_len();
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
        for (_, mut region) in regions {
            if region.text.is_empty() {
                region.text = region
                    .lines
                    .iter()
                    .map(|line| line.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
            }
            if region.text.is_empty() {
                continue;
            }
            let Some(found) = page.text[cursor..].find(&region.text) else {
                continue;
            };
            let start_byte = cursor + found;
            cursor = start_byte + region.text.len();
            let placed = Region {
                start: page_start
                    + page_coordinates
                        .utf16_at_byte(start_byte)
                        .expect("matched journal region starts at a UTF-8 boundary"),
                end: page_start
                    + page_coordinates
                        .utf16_at_byte(cursor)
                        .expect("matched journal region ends at a UTF-8 boundary"),
                pdf_page,
            };
            if region.kind.as_deref() == Some("text") {
                paragraphs += 1;
                blocks.push(SourceDocBlock::new(
                    SourceDocKind::Paragraph,
                    format!("par{paragraphs}"),
                    placed.start,
                    placed.end,
                    SourceDocOrigin::Native,
                ));
            }
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
    fn authoritative_pages_preserve_every_native_region() {
        let pages = concat!(
            r#"{"article_id":"1","text":"TITLE\nBody\n7\n1 Note","pdf_page":1,"regions":[{"order":0,"type":"paragraph_title","text":"TITLE"},{"order":1,"type":"text","lines":[{"text":"Body"}]},{"order":2,"type":"number","text":"7"},{"order":3,"type":"footnote","text":"1 Note","lines":[{"codex_text_order":7}]}],"annotations":[{"pair_id":"p","pair_status":"paired","taxonomy_name":"fn_ref"},{"pair_id":"p","pair_status":"paired","taxonomy_name":"fn_label","note_id":"1","start_line_order":7}]}"#,
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
        assert_eq!(doc.text, "TITLE\nBody\n7\n1 Note");
        assert_eq!(
            doc.blocks
                .iter()
                .filter(|block| block.kind == SourceDocKind::Paragraph)
                .map(|block| (block.label.as_str(), block.origin))
                .collect::<Vec<_>>(),
            [("par1", SourceDocOrigin::Native)]
        );
        for label in ["page7", "secTitle1", "fn1"] {
            assert!(doc.blocks.iter().any(|block| block.label == label));
        }
    }

    #[test]
    fn plain_text_uses_only_page_markers() {
        let text = "[page 9]\nFirst page.\n\n[page x]\nSecond page.";
        let doc = journal_text_source_doc(
            3,
            Some("https://example.test/article".into()),
            text.into(),
            &[
                JournalPageLabel {
                    label: "9".into(),
                    pdf_page: 4,
                },
                JournalPageLabel {
                    label: "x".into(),
                    pdf_page: 5,
                },
            ],
        )
        .unwrap();
        assert_eq!(doc.url.as_deref(), Some("https://example.test/article"));
        assert_eq!(doc.text, "First page.\n\nSecond page.");
        assert_eq!(
            doc.blocks
                .iter()
                .map(|block| (
                    block.label.as_str(),
                    block.start,
                    block.end,
                    block.anchor.as_deref(),
                    block.origin,
                ))
                .collect::<Vec<_>>(),
            [
                (
                    "page9",
                    0,
                    "First page.\n\n".len(),
                    Some("page=4"),
                    SourceDocOrigin::Native,
                ),
                (
                    "pagex",
                    "First page.\n\n".len(),
                    "First page.\n\nSecond page.".len(),
                    Some("page=5"),
                    SourceDocOrigin::Native,
                ),
            ]
        );
        assert!(doc
            .blocks
            .iter()
            .all(|block| block.kind == SourceDocKind::Page));
    }

    #[test]
    fn plain_text_page_offsets_use_clean_text_utf16_coordinates() {
        let doc = journal_text_source_doc(
            4,
            None,
            "[page 1]\r\n\u{1f9ab}e\u{301}\r\n[page 2]\nZ".to_owned(),
            &[
                JournalPageLabel {
                    label: "1".to_owned(),
                    pdf_page: 1,
                },
                JournalPageLabel {
                    label: "2".to_owned(),
                    pdf_page: 2,
                },
            ],
        )
        .unwrap();
        assert_eq!(doc.text, "\u{1f9ab}e\u{301}\r\nZ");
        assert_eq!(
            doc.blocks
                .iter()
                .map(|block| (block.start, block.end))
                .collect::<Vec<_>>(),
            [(0, 6), (6, 7)]
        );
    }

    #[test]
    fn json_regions_convert_original_page_bytes_to_rendered_utf16() {
        let pages = concat!(
            r#"{"article_id":"5","text":"\ud83e\uddab\nBody","pdf_page":1,"regions":[{"order":0,"type":"text","text":"Body"}]}"#,
            "\n",
        );
        let doc = journal_source_doc(
            5,
            None,
            Cursor::new(pages),
            &[JournalPageLabel {
                label: "1".to_owned(),
                pdf_page: 1,
            }],
        )
        .unwrap();
        let paragraph = doc
            .blocks
            .iter()
            .find(|block| block.kind == SourceDocKind::Paragraph)
            .expect("native paragraph");
        assert_eq!((paragraph.start, paragraph.end), (3, 7));
    }
}
