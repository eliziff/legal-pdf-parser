//! The documents a PDF carries after its own: an exhibit behind its certificate, each earlier
//! message an email quotes below the latest, a schedule or annex, a form after the
//! instructions for it. Each is a section from its opening line to the next one's, which the
//! heading sections stop at and nest under, so a constituent's own headings and numbering stay
//! its own.
//!
//! Only a line that by itself opens another document counts: an exhibit's certificate, a
//! message's header or forwarding mark, a page headed "SCHEDULE" or by a new form's title. A
//! page break, a numbering restart or a change of type alone opens none. A constituent carries
//! no label: it is never a pinpoint's target.

use super::{arabic_page_number, bookmarks, graph::PdfTextIndex};
use legal_pdf_core::model::{Line, Page, PdfOutlineEntry};
use legal_pdf_headings::party_label;
use legal_structure_model::{Derivation, NodeKind, ScalarRange, StructureNode};
use regex::Regex;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::OnceLock;

const ORIGIN: &str = "legalpdf.pdf-structure.v2";

/// The grammar of a constituent's section; its kind follows ("constituent_exhibit").
pub(super) const GRAMMAR: &str = "constituent_";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct Start {
    pub(super) line_id: String,
    kind: String,
}

fn exhibit_cover() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r#"(?i)^\s*(?:this\s+is\s+)?exhibit\s+(?:no\.?\s*)?["'“”‘’`]{0,2}\s*[a-z0-9]{1,4}\s*["'“”‘’`]{0,2}\s+(?:referred|to\s+the|mentioned|in\s+the|attached)|(?i)^\s*this\s+is\s+the\s+exhibit\s+marked"#,
        )
        .expect("exhibit cover regex")
    })
}

/// "-----Original Message-----", "Begin forwarded message:", as recognition reads them too
/// ("Original Message---t-").
fn forwarded_mark(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    text.chars().count() <= 40
        && RE
            .get_or_init(|| {
                Regex::new(
                    r"(?i)^\W*(?:-{2,}\W*)?(?:original|forwarded)\s+message\W*\w?\W*$|^\s*begin\s+forwarded\s+message:?\s*$",
                )
                .expect("forwarded mark regex")
            })
            .is_match(text)
}

/// A message header's field: "From:", "Sent:", "To:"…
fn header_field(text: &str) -> Option<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)^\s*(from|sent|date|to|cc|subject)\s*:").expect("header field regex")
    })
    .captures(text)
    .map(|captures| captures[1].to_lowercase())
}

/// A page headed by its own title as another document: "SCHEDULE", "ANNEX B".
fn appended_title(text: &str) -> Option<&'static str> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let captures = RE
        .get_or_init(|| {
            Regex::new(
                r#"^\s*(SCHEDULE|ANNEX|APPENDIX|ATTACHMENT|Schedule|Annex|Appendix|Attachment)(?:\s+["“]?[A-Z0-9]{1,4}["”]?)?\s*$"#,
            )
            .expect("appended title regex")
        })
        .captures(text)?;
    Some(match captures[1].to_lowercase().as_str() {
        "schedule" => "schedule",
        _ => "annex",
    })
}

/// A fresh court caption distinguishes a mid-page appendix from a reference to one.
fn court_caption<'a>(lines: impl Iterator<Item = &'a str>) -> bool {
    let mut dated = false;
    let mut filed = false;
    for text in lines {
        let Some((field, _)) = text.split_once(':') else {
            continue;
        };
        match field.trim().to_ascii_lowercase().as_str() {
            "date" => dated = true,
            "docket" | "registry" => filed = true,
            _ => {}
        }
    }
    dated && filed
}

/// A form's title: "Form F28.02A: Request for a Summary Judgment".
fn form_title(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^\s*Form\s+[A-Z]{0,3}\d{1,3}[A-Z0-9.]*\s*[:\-–—]\s*\S")
            .expect("form title regex")
    })
    .is_match(text)
}

/// A line of running content rather than a reference number or a field: three words or more.
fn content(line: &Line) -> bool {
    header_field(&line.text).is_none()
        && line
            .text
            .split_whitespace()
            .filter(|word| word.chars().any(char::is_alphabetic))
            .count()
            >= 3
}

pub(super) fn starts(pages: &[Page]) -> Vec<Start> {
    let mut ordered = pages.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|page| page.index);
    // Each body line, with its page and its place on that page.
    let body = ordered
        .iter()
        .enumerate()
        .flat_map(|(page, sheet)| {
            let mut lines = sheet
                .lines
                .iter()
                .filter(|line| {
                    line.note_region_mode.is_empty()
                        && !line.exclude_from_body
                        && !matches!(
                            line.region_type.to_ascii_lowercase().as_str(),
                            "header"
                                | "footer"
                                | "footnote"
                                | "endnote"
                                | "table"
                                | "table_cell"
                                | "contents"
                        )
                })
                .collect::<Vec<_>>();
            lines.sort_by(|left, right| {
                left.reading_order
                    .cmp(&right.reading_order)
                    .then_with(|| left.id.cmp(&right.id))
            });
            let count = lines.len();
            lines
                .into_iter()
                .enumerate()
                .map(move |(place, line)| (line, page, place, count))
        })
        .collect::<Vec<_>>();
    // Where each constituent opens, and what it is. The package's own document opens before
    // any of its content, so a mark there opens nothing new.
    let mut starts = Vec::new();
    let mut forms = HashSet::new();
    let mut slot = 0;
    // Whether the package's own content has begun.
    let mut opened = false;
    // The page's first body line, and whether a caption's party labels have been read on it.
    let (mut page_first, mut captioned) = (0, false);
    while slot < body.len() {
        let (line, page, place, count) = body[slot];
        if place == 0 {
            (page_first, captioned) = (slot, false);
        }
        let mark = forwarded_mark(&line.text);
        let after_header = slot > 0
            && (header_field(&body[slot - 1].0.text).is_some()
                || forwarded_mark(&body[slot - 1].0.text));
        if mark || (header_field(&line.text).is_some() && !after_header) {
            // A message's header: its forwarding mark and its fields, one under another.
            let mut end = slot + 1;
            let mut fields = header_field(&line.text).into_iter().collect::<HashSet<_>>();
            while end < body.len()
                && end - slot < 10
                && (header_field(&body[end].0.text).is_some()
                    || (end + 1 < body.len() && header_field(&body[end + 1].0.text).is_some()))
            {
                fields.extend(header_field(&body[end].0.text));
                end += 1;
            }
            if (mark || (fields.contains("from") && fields.len() >= 3)) && opened {
                starts.push((slot, "email"));
            }
            opened |= body[slot..end].iter().any(|(line, ..)| content(line));
            slot = end;
            continue;
        }
        let kind = if exhibit_cover().is_match(&line.text) && (place < 4 || count <= 12) {
            // An exhibit's certificate heads its page or stands on a cover page; one in running
            // text is a reference to the exhibit.
            Some("exhibit")
        } else if let Some(kind) = appended_title(&line.text) {
            (page > 0
                && (place < 2
                    || court_caption(
                        body[slot + 1..]
                            .iter()
                            .take(10)
                            .map(|(line, ..)| line.text.as_str()),
                    )))
            .then_some(kind)
        } else if place < 2 {
            // A form's title heads each of its pages; the form opens on the first.
            let new_form = form_title(&line.text)
                && forms.insert(line.text.trim().chars().take(24).collect::<String>());
            (page > 0 && new_form).then_some("form")
        } else {
            None
        };
        if let Some(kind) = kind.filter(|_| opened) {
            // An exhibit's certificate under a caption of its own opens at the caption.
            let first = if kind == "exhibit" && captioned {
                page_first
            } else {
                slot
            };
            starts.push((first, kind));
        }
        captioned |= party_label().is_match(line.text.trim());
        opened |= content(line);
        slot += 1;
    }
    // A certificate can backdate its opening to a caption already claimed by another mark.
    starts.sort_by_key(|(slot, _)| *slot);
    starts.dedup_by_key(|(slot, _)| *slot);
    starts
        .into_iter()
        .map(|(slot, kind)| Start {
            line_id: body[slot].0.id.clone(),
            kind: kind.to_owned(),
        })
        .collect()
}

/// Package bookmarks supply opening pages before the readers calibrate each document.
pub(super) fn bookmarked_starts(pages: &[Page], outline: &[PdfOutlineEntry]) -> Vec<Start> {
    let by_index = pages
        .iter()
        .map(|page| (page.index, page))
        .collect::<HashMap<_, _>>();
    bookmarks::package_targets(outline, pages.len())
        .into_iter()
        .filter_map(|(page, kind)| {
            let line = by_index
                .get(&page)?
                .lines
                .iter()
                .filter(|line| {
                    !line.exclude_from_body
                        && line.note_region_mode.is_empty()
                        && !matches!(
                            line.region_type.as_str(),
                            "header" | "footer" | "footnote" | "endnote"
                        )
                        // Version stamps contain letters but supply no word of an opening.
                        && line.text.split(|c: char| !c.is_alphanumeric()).any(|word| {
                            !word.is_empty() && word.chars().all(char::is_alphabetic)
                        })
                })
                .min_by(|a, b| {
                    a.reading_order
                        .cmp(&b.reading_order)
                        .then_with(|| a.id.cmp(&b.id))
                })?;
            Some(Start {
                line_id: line.id.clone(),
                kind,
            })
        })
        .collect()
}

/// A visible opening wins over a bookmark, including an exhibit's preceding certificate page.
pub(super) fn reconcile_starts(pages: &[Page], bookmarked: &mut Vec<Start>) {
    let mut lexical = starts(pages);
    if bookmarked.is_empty() {
        *bookmarked = lexical;
        return;
    }
    let positions = pages
        .iter()
        .flat_map(|page| {
            page.lines
                .iter()
                .map(move |line| (line.id.as_str(), (page.index, line.reading_order)))
        })
        .collect::<HashMap<_, _>>();
    let lexical_pages = lexical
        .iter()
        .map(|start| positions[start.line_id.as_str()].0)
        .collect::<HashSet<_>>();
    let certificates = lexical
        .iter()
        .filter(|start| start.kind == "exhibit")
        .map(|start| positions[start.line_id.as_str()].0)
        .collect::<HashSet<_>>();
    lexical.extend(std::mem::take(bookmarked).into_iter().filter(|start| {
        let page = positions[start.line_id.as_str()].0;
        !lexical_pages.contains(&page)
            && !(start.kind == "exhibit"
                && page
                    .checked_sub(1)
                    .is_some_and(|prior| certificates.contains(&prior)))
    }));
    lexical.sort_by_key(|start| positions[start.line_id.as_str()]);
    lexical.dedup_by(|a, b| a.line_id == b.line_id);
    *bookmarked = lexical;
}

/// The same openings scope inference and the final constituent nodes.
pub(super) fn document_lines(pages: &[Page], starts: &[Start]) -> HashMap<String, usize> {
    if starts.is_empty() {
        return HashMap::new();
    }
    let starts = starts
        .iter()
        .enumerate()
        .map(|(slot, start)| (start.line_id.as_str(), slot + 1))
        .collect::<HashMap<_, _>>();
    let mut ordered = pages.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|page| page.index);
    let mut document = 0;
    let mut documents = HashMap::new();
    for page in ordered {
        let mut lines = page.lines.iter().collect::<Vec<_>>();
        lines.sort_by(|left, right| {
            left.reading_order
                .cmp(&right.reading_order)
                .then_with(|| left.id.cmp(&right.id))
        });
        // A footer painted before the opening still belongs to the document on this page.
        if let Some(first) = lines
            .iter()
            .position(|line| starts.contains_key(line.id.as_str()))
        {
            if lines[..first].iter().all(|line| {
                line.exclude_from_body
                    || matches!(line.region_type.as_str(), "header" | "footer")
                    || (arabic_page_number(&line.text).is_some()
                        && (line.bbox[3] < page.height * 0.15 || line.bbox[1] > page.height * 0.85))
            }) {
                document = starts[lines[first].id.as_str()];
            }
        }
        for line in lines {
            if let Some(&start) = starts.get(line.id.as_str()) {
                document = start;
            }
            if document != 0 {
                documents.insert(line.id.clone(), document);
            }
        }
    }
    documents
}

/// Run the existing readers on logical documents, including each parallel sequence.
/// Lines move into page views and back; their text, ids, anchors and physical positions stay put.
pub(super) fn within_document_contexts(
    pages: &mut [Page],
    documents: &HashMap<String, usize>,
    translation: &HashSet<String>,
    mut read: impl FnMut(&mut [Page]),
) {
    if translation.is_empty() && documents.is_empty() {
        read(pages);
        return;
    }
    #[derive(Default)]
    struct Reading {
        pages: Vec<Page>,
        positions: Vec<(usize, Vec<usize>)>,
    }
    let mut readings = BTreeMap::<(usize, bool), Reading>::new();
    let mut ordered = (0..pages.len()).collect::<Vec<_>>();
    ordered.sort_by_key(|&slot| pages[slot].index);
    for page_slot in ordered {
        let page = &mut pages[page_slot];
        for (slot, line) in std::mem::take(&mut page.lines).into_iter().enumerate() {
            let context = (
                documents.get(&line.id).copied().unwrap_or(0),
                translation.contains(&line.id),
            );
            let reading = readings.entry(context).or_default();
            if reading
                .positions
                .last()
                .is_none_or(|(prior, _)| *prior != page_slot)
            {
                reading.pages.push(Page {
                    id: page.id.clone(),
                    index: page.index,
                    number: page.number,
                    width: page.width,
                    height: page.height,
                    lines: Vec::new(),
                    // Initial classification also needs the page's painted table rules.
                    regions: page
                        .regions
                        .iter()
                        .filter(|region| region.kind == "rule")
                        .cloned()
                        .collect(),
                    source: page.source.clone(),
                    text_quality: page.text_quality,
                    printed_label: page.printed_label.clone(),
                    printed_label_source: page.printed_label_source.clone(),
                    printed_label_line_id: page.printed_label_line_id.clone(),
                });
                reading.positions.push((page_slot, Vec::new()));
            }
            reading.pages.last_mut().unwrap().lines.push(line);
            reading.positions.last_mut().unwrap().1.push(slot);
        }
    }
    let mut restored = (0..pages.len()).map(|_| Vec::new()).collect::<Vec<_>>();
    for (context, mut reading) in readings {
        read(&mut reading.pages);
        for (page, (page_slot, slots)) in reading.pages.into_iter().zip(reading.positions) {
            restored[page_slot].extend(
                page.lines
                    .into_iter()
                    .zip(slots)
                    .map(|(line, slot)| (slot, context, line)),
            );
        }
    }
    for (page, mut lines) in pages.iter_mut().zip(restored) {
        lines.sort_by_key(|(slot, _, _)| *slot);
        // Block ids are page-local. Separate readers must not produce colliding groups.
        let mut blocks = HashMap::new();
        page.lines = lines
            .into_iter()
            .enumerate()
            .map(|(slot, (_, context, mut line))| {
                let next = blocks.len();
                line.block_index = *blocks.entry((context, line.block_index)).or_insert(next);
                line.reading_order = slot + 1;
                line
            })
            .collect();
    }
}

pub(super) fn constituents(
    index: &PdfTextIndex,
    pages: &[Page],
    starts: &[Start],
) -> Vec<StructureNode> {
    let starts = starts
        .iter()
        .filter_map(|start| Some((index.line(&start.line_id)?.range.start, start.kind.as_str())))
        .collect::<Vec<_>>();
    let end = index.text().chars().count();
    let page_of = pages
        .iter()
        .flat_map(|page| {
            page.lines
                .iter()
                .map(move |line| (line.id.as_str(), page.index))
        })
        .collect::<HashMap<_, _>>();
    starts
        .iter()
        .enumerate()
        .map(|(position, &(start, kind))| {
            let until = starts.get(position + 1).map_or(end, |(next, _)| *next);
            let line_ids = index
                .line_ids(ScalarRange { start, end: until })
                .into_iter()
                .filter(|id| index.line(id).is_some_and(|line| line.range.end <= until))
                .collect::<Vec<_>>();
            let mut node = StructureNode::new(
                format!("constituent-{:06}", position + 1),
                NodeKind::Section,
                ScalarRange { start, end: until },
                ORIGIN,
                Derivation::Heuristic,
                None,
            );
            node.grammar = Some(format!("{GRAMMAR}{kind}"));
            let mut seen = HashSet::new();
            node.page_indexes = line_ids
                .iter()
                .filter_map(|id| page_of.get(id.as_str()).copied())
                .filter(|page| seen.insert(*page))
                .collect();
            node.line_ids = line_ids;
            node
        })
        .collect()
}
