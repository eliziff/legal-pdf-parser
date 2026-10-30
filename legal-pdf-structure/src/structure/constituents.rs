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

use super::{arabic_page_number, graph::PdfTextIndex, headings, party_label, PageView, Slot};
use legal_pdf_core::model::{Line, Page, PdfOutlineEntry};
use legal_structure_model::{Derivation, NodeKind, ScalarRange, StructureNode};
use regex::Regex;
use std::collections::{HashMap, HashSet};
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
pub(super) fn appended_title(text: &str) -> Option<&'static str> {
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

/// Where each constituent document opens, read over the pages' lines (`views`) that `set_aside`
/// leaves as body.
pub(super) fn starts(views: &[PageView], set_aside: impl Fn(Slot) -> bool) -> Vec<Start> {
    let mut ordered = views.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|view| view.page.index);
    // Each body line, with its page and its place on that page.
    let body = ordered
        .iter()
        .enumerate()
        .flat_map(|(page, view)| {
            let mut lines = view
                .iter()
                .filter(|(slot, _)| !set_aside(*slot))
                .map(|(_, line)| line)
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

/// Package bookmarks supply opening pages before the readers calibrate each document; a line
/// `set_aside` holds (a transcript's line number, furniture, a note) opens none.
pub(super) fn bookmarked_starts(views: &[PageView], outline: &[PdfOutlineEntry], set_aside: impl Fn(Slot) -> bool) -> Vec<Start> {
    let by_index = views
        .iter()
        .map(|view| (view.page.index, view))
        .collect::<HashMap<_, _>>();
    headings::bookmarks::package_targets(outline, views.len())
        .into_iter()
        .filter_map(|(page, kind)| {
            let (_, line) = by_index
                .get(&page)?
                .iter()
                .filter(|(slot, line)| {
                    !set_aside(*slot)
                        // Version stamps contain letters but supply no word of an opening.
                        && line.text.split(|c: char| !c.is_alphanumeric()).any(|word| {
                            !word.is_empty() && word.chars().all(char::is_alphabetic)
                        })
                })
                .min_by(|(_, a), (_, b)| {
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
pub(super) fn reconcile_starts(views: &[PageView], bookmarked: &mut Vec<Start>, set_aside: impl Fn(Slot) -> bool) {
    let mut lexical = starts(views, set_aside);
    if bookmarked.is_empty() {
        *bookmarked = lexical;
        return;
    }
    let positions = views
        .iter()
        .flat_map(|view| {
            view.iter()
                .map(move |(_, line)| (line.id.as_str(), (view.page.index, line.reading_order)))
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

/// The constituent document each line reads in (1 for the first opening `starts` names, 0 for the
/// packet's own, which is left out), page by page in reading order: a line of the page's furniture
/// (`furniture`) or its folio before the opening still belongs to the document on that page.
pub(super) fn documents(views: &[PageView], starts: &[Start], furniture: impl Fn(Slot) -> bool) -> Vec<(Slot, usize)> {
    if starts.is_empty() {
        return Vec::new();
    }
    let starts = starts
        .iter()
        .enumerate()
        .map(|(slot, start)| (start.line_id.as_str(), slot + 1))
        .collect::<HashMap<_, _>>();
    let mut ordered = views.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|view| view.page.index);
    let mut document = 0;
    let mut documents = Vec::new();
    for view in ordered {
        let page = view.page;
        let mut lines = view.iter().collect::<Vec<_>>();
        lines.sort_by(|(_, left), (_, right)| {
            left.reading_order
                .cmp(&right.reading_order)
                .then_with(|| left.id.cmp(&right.id))
        });
        // A footer painted before the opening still belongs to the document on this page.
        if let Some(first) = lines
            .iter()
            .position(|(_, line)| starts.contains_key(line.id.as_str()))
        {
            if lines[..first].iter().all(|&(slot, line)| {
                furniture(slot)
                    || (arabic_page_number(&line.text).is_some()
                        && (line.bbox[3] < page.height * 0.15 || line.bbox[1] > page.height * 0.85))
            }) {
                document = starts[lines[first].1.id.as_str()];
            }
        }
        for (slot, line) in lines {
            if let Some(&start) = starts.get(line.id.as_str()) {
                document = start;
            }
            if document != 0 {
                documents.push((slot, document));
            }
        }
    }
    documents
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
