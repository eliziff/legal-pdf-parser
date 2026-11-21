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

// The crates this stage reads, by the names its paths use.
use legal_pdf_structure_model as model;
use legal_pdf_rules::text;

use crate::model::constituents::Start;
use crate::model::{PageView, Slot};
use crate::text::text::{arabic_page_number, party_label};
use crate::text::text_index::PdfTextIndex;
use legal_pdf_core::model::{Line, Page, PdfOutlineEntry};
use legal_pdf_core::regex;
use legal_structure_model::{Derivation, NodeKind, ScalarRange, StructureNode};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

const ORIGIN: &str = "legalpdf.pdf-structure.v2";

/// The grammar of a constituent's section; its kind follows ("constituent_exhibit").
pub const GRAMMAR: &str = "constituent_";



/// Lines in reading order, their ids breaking ties.
fn reading(left: &Line, right: &Line) -> Ordering {
    left.reading_order
        .cmp(&right.reading_order)
        .then_with(|| left.id.cmp(&right.id))
}

/// "-----Original Message-----", "Begin forwarded message:", as recognition reads them too
/// ("Original Message---t-").
fn forwarded_mark(text: &str) -> bool {
    text.chars().count() <= 40
        && regex!(r"(?i)^\W*(?:-{2,}\W*)?(?:original|forwarded)\s+message\W*\w?\W*$|^\s*begin\s+forwarded\s+message:?\s*$").is_match(text)
}

/// A message header's field: "From:", "Sent:", "To:"…
fn header_field(text: &str) -> Option<String> {
    regex!(r"(?i)^\s*(from|sent|date|to|cc|subject)\s*:")
        .captures(text)
        .map(|captures| captures[1].to_lowercase())
}

/// A page headed by its own title as another document: "SCHEDULE", "ANNEX B".
pub fn appended_title(text: &str) -> Option<&'static str> {
    let captures = regex!(r#"^\s*(SCHEDULE|ANNEX|APPENDIX|ATTACHMENT|Schedule|Annex|Appendix|Attachment)(?:\s+["“]?[A-Z0-9]{1,4}["”]?)?\s*$"#).captures(text)?;
    Some(if captures[1].eq_ignore_ascii_case("schedule") {
        "schedule"
    } else {
        "annex"
    })
}

/// A fresh court caption distinguishes a mid-page appendix from a reference to one.
fn court_caption<'a>(lines: impl Iterator<Item = &'a str>) -> bool {
    let fields = lines
        .filter_map(|text| Some(text.split_once(':')?.0.trim().to_ascii_lowercase()))
        .collect::<HashSet<_>>();
    fields.contains("date") && (fields.contains("docket") || fields.contains("registry"))
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
pub fn starts(views: &[PageView], set_aside: impl Fn(Slot) -> bool) -> Vec<Start> {
    let mut ordered = views.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|view| view.page.index);
    // Each body line, with its page, its place on that page and the page's count of them.
    let mut body = Vec::new();
    for (page, view) in ordered.iter().enumerate() {
        let mut lines = view
            .iter()
            .filter(|(slot, _)| !set_aside(*slot))
            .map(|(_, line)| line)
            .collect::<Vec<_>>();
        lines.sort_by(|left, right| reading(left, right));
        let count = lines.len();
        body.extend(
            lines
                .into_iter()
                .enumerate()
                .map(|(place, line)| (line, page, place, count)),
        );
    }
    let field = |slot: usize| header_field(&body[slot].0.text);
    // Where each constituent opens, and what it is. The package's own document opens before
    // any of its content, so a mark there opens nothing new.
    let (mut starts, mut forms, mut slot) = (Vec::new(), HashSet::new(), 0);
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
        let after_header =
            slot > 0 && (field(slot - 1).is_some() || forwarded_mark(&body[slot - 1].0.text));
        if mark || (field(slot).is_some() && !after_header) {
            // A message's header: its forwarding mark and its fields, one under another.
            let mut end = slot + 1;
            let mut fields = field(slot).into_iter().collect::<HashSet<_>>();
            while end < body.len()
                && end - slot < 10
                && (field(end).is_some() || (end + 1 < body.len() && field(end + 1).is_some()))
            {
                fields.extend(field(end));
                end += 1;
            }
            if (mark || (fields.contains("from") && fields.len() >= 3)) && opened {
                starts.push((slot, "email"));
            }
            opened |= body[slot..end].iter().any(|(line, ..)| content(line));
            slot = end;
            continue;
        }
        let exhibit = r#"(?i)^\s*(?:this\s+is\s+)?exhibit\s+(?:no\.?\s*)?["'“”‘’`]{0,2}\s*[a-z0-9]{1,4}\s*["'“”‘’`]{0,2}\s+(?:referred|to\s+the|mentioned|in\s+the|attached)|(?i)^\s*this\s+is\s+the\s+exhibit\s+marked"#;
        let kind = if regex!(exhibit).is_match(&line.text) && (place < 4 || count <= 12) {
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
            // A form's title heads each of its pages ("Form F28.02A: Request for a Summary
            // Judgment"); the form opens on the first.
            let new_form = regex!(r"^\s*Form\s+[A-Z]{0,3}\d{1,3}[A-Z0-9.]*\s*[:\-–—]\s*\S")
                .is_match(&line.text)
                && forms.insert(line.text.trim().chars().take(24).collect::<String>());
            (page > 0 && new_form).then_some("form")
        } else {
            None
        };
        if let Some(kind) = kind.filter(|_| opened) {
            // An exhibit's certificate under a caption of its own opens at the caption.
            starts.push((
                if kind == "exhibit" && captioned {
                    page_first
                } else {
                    slot
                },
                kind,
            ));
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
        .map(|(slot, kind)| Start::new(body[slot].0.id.clone(), kind))
        .collect()
}

/// Package bookmarks supply opening pages before the readers calibrate each document; a line
/// `set_aside` holds (a transcript's line number, furniture, a note) opens none.
pub fn bookmarked_starts(
    views: &[PageView],
    outline: &[PdfOutlineEntry],
    set_aside: impl Fn(Slot) -> bool,
) -> Vec<Start> {
    let by_index = views
        .iter()
        .map(|view| (view.page.index, view))
        .collect::<HashMap<_, _>>();
    // Version stamps contain letters but supply no word of an opening.
    let worded = |line: &Line| {
        line.text
            .split(|c: char| !c.is_alphanumeric())
            .any(|word| !word.is_empty() && word.chars().all(char::is_alphabetic))
    };
    legal_pdf_rules::headings::bookmarks::package_targets(outline, views.len())
        .into_iter()
        .filter_map(|(page, kind)| {
            let (_, line) = by_index
                .get(&page)?
                .iter()
                .filter(|(slot, line)| !set_aside(*slot) && worded(line))
                .min_by(|(_, a), (_, b)| reading(a, b))?;
            Some(Start {
                line_id: line.id.clone(),
                kind,
            })
        })
        .collect()
}

/// A visible opening wins over a bookmark, including an exhibit's preceding certificate page.
pub fn reconcile_starts(
    views: &[PageView],
    bookmarked: &mut Vec<Start>,
    set_aside: impl Fn(Slot) -> bool,
) {
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
    let page = |start: &Start| positions[start.line_id.as_str()].0;
    let lexical_pages = lexical.iter().map(page).collect::<HashSet<_>>();
    let certificates = lexical
        .iter()
        .filter(|start| start.kind == "exhibit")
        .map(page)
        .collect::<HashSet<_>>();
    lexical.extend(std::mem::take(bookmarked).into_iter().filter(|start| {
        !lexical_pages.contains(&page(start))
            && !(start.kind == "exhibit"
                && page(start)
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
pub fn documents(
    views: &[PageView],
    starts: &[Start],
    furniture: impl Fn(Slot) -> bool,
) -> Vec<(Slot, usize)> {
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
    let (mut document, mut documents) = (0, Vec::new());
    for view in ordered {
        let height = view.page.height;
        let mut lines = view.iter().collect::<Vec<_>>();
        lines.sort_by(|(_, left), (_, right)| reading(left, right));
        // A footer painted before the opening still belongs to the document on this page.
        if let Some(first) = lines
            .iter()
            .position(|(_, line)| starts.contains_key(line.id.as_str()))
        {
            let folio = |line: &Line| {
                arabic_page_number(&line.text).is_some()
                    && (line.bbox[3] < height * 0.15 || line.bbox[1] > height * 0.85)
            };
            if lines[..first]
                .iter()
                .all(|&(slot, line)| furniture(slot) || folio(line))
            {
                document = starts[lines[first].1.id.as_str()];
            }
        }
        for (slot, line) in lines {
            document = starts.get(line.id.as_str()).copied().unwrap_or(document);
            if document != 0 {
                documents.push((slot, document));
            }
        }
    }
    documents
}

pub fn constituents(
    index: &PdfTextIndex,
    pages: &[Page],
    starts: &[Start],
) -> Vec<StructureNode> {
    let starts = starts
        .iter()
        .filter_map(|start| Some((index.line(&start.line_id)?.range.start, start.kind.as_str())))
        .collect::<Vec<_>>();
    let end = legal_structure_model::utf16_len(index.text());
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
            let range = ScalarRange {
                start,
                end: starts.get(position + 1).map_or(end, |(next, _)| *next),
            };
            let line_ids = index
                .line_ids(range)
                .into_iter()
                .filter(|id| {
                    index
                        .line(id)
                        .is_some_and(|line| line.range.end <= range.end)
                })
                .collect::<Vec<_>>();
            let mut node = StructureNode::new(
                format!("constituent-{:06}", position + 1),
                NodeKind::Section,
                range,
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
