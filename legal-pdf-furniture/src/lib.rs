//! A document's furniture: running heads and feet repeated page after page, or set apart at a
//! short document's edges; the rows of its folios (the page map's, `legal_pdf_pages::page_map`);
//! side stamps and print margins.

// The crates this stage reads, by the names its paths use.
use legal_pdf_structure_model as model;
use legal_pdf_rules as rules;
use legal_pdf_rules::text;
use legal_pdf_layout as layout;

use crate::layout::{aligned_note_body, aligned_note_body_among, standalone_note_label};
use crate::model::{line_center_x};
use crate::model::{Edge, PageView, Rows, Rule, Slot};
use crate::text::text::standalone_paragraph_number;
use legal_pdf_core::model::{Line, Page};
use legal_pdf_core::profile::measure;
use legal_pdf_core::{quantile, regex};
use legal_pdf_pages::{EdgeLine, PageEvidence};
pub use legal_pdf_rules::notes::lines::compact_note_line;
use std::collections::{HashMap, HashSet};

const FURNITURE_Y_TOLERANCE_FRAC: f64 = 0.015;
const FURNITURE_X_TOLERANCE_FRAC: f64 = 0.04;

/// A line's text as repetition compares it: its figures each one `#`, lower case, its spaces single.
fn normalize_furniture(text: &str) -> String {
    let text = regex!(r"\d+")
        .replace_all(text, "#")
        .chars()
        .flat_map(char::to_lowercase)
        .collect::<String>();
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}


/// A line where repetition reads it: its page, its place on the page, and its top's and middle's
/// fractions of the page's height and width.
#[derive(Clone, Copy)]
struct Hit {
    page_slot: usize,
    line_slot: usize,
    page_number: u32,
    y_ratio: f64,
    x_ratio: f64,
}

/// What each page of a document tells the page map of its number (`legal_pdf_pages::page_map`): the label the PDF
/// declares for it (`embedded`, by page index), its size, and its lines by line slot on the page, those `line_label` holds
/// (by page slot in `pages` and line slot) left blank: a transcript's line numbers number its lines, not the page.
pub fn page_evidence(
    pages: &[Page],
    embedded: &[Option<String>],
    line_label: impl Fn(usize, usize) -> bool,
) -> Vec<PageEvidence> {
    let count = pages
        .iter()
        .map(|page| page.index + 1)
        .max()
        .unwrap_or(0)
        .max(embedded.len());
    let mut evidence = vec![PageEvidence::default(); count];
    for (page_slot, page) in pages.iter().enumerate() {
        let lines = page.lines.iter().enumerate();
        evidence[page.index] = PageEvidence {
            embedded: None,
            width: page.width,
            height: page.height,
            lines: lines
                .map(|(at, line)| EdgeLine {
                    text: if line_label(page_slot, at) {
                        String::new()
                    } else {
                        line.text.clone()
                    },
                    rect: line.bbox,
                })
                .collect(),
        };
    }
    for (page, label) in evidence.iter_mut().zip(embedded) {
        page.embedded.clone_from(label);
    }
    evidence
}

/// The hits of one repeated text that stand at one place: the cluster of tops spanning the most
/// pages, each page parity's lines in it at the place across the page that spans the most of that
/// parity's pages; none unless it spans `minimum` pages.
fn aligned_furniture(hits: &[Hit], minimum: usize) -> HashSet<(usize, usize)> {
    let pages = |hits: &mut dyn Iterator<Item = &Hit>| {
        hits.map(|hit| hit.page_slot).collect::<HashSet<_>>().len()
    };
    let within =
        |value: f64, start: f64, tolerance: f64| value >= start && value - start <= tolerance;
    let (mut best, mut best_pages) = (Vec::new(), 0);
    // (Hits at one height read one cluster: the first of them reads it for all.)
    let mut heights = HashSet::new();
    for y_start in hits
        .iter()
        .filter(|hit| heights.insert(hit.y_ratio.to_bits()))
    {
        let mut trial = Vec::new();
        for parity in [0, 1] {
            let parity_hits = hits
                .iter()
                .filter(|hit| {
                    within(hit.y_ratio, y_start.y_ratio, FURNITURE_Y_TOLERANCE_FRAC)
                        && hit.page_number % 2 == parity
                })
                .collect::<Vec<_>>();
            let column = |x_start: f64| {
                parity_hits
                    .iter()
                    .copied()
                    .filter(move |hit| within(hit.x_ratio, x_start, FURNITURE_X_TOLERANCE_FRAC))
            };
            let mut counts = HashMap::new();
            // Iterator::max_by_key chooses the last equal maximum.
            let x_start = parity_hits
                .iter()
                .max_by_key(|start| {
                    *counts
                        .entry(start.x_ratio.to_bits())
                        .or_insert_with(|| pages(&mut column(start.x_ratio)))
                })
                .map(|start| start.x_ratio);
            trial.extend(x_start.into_iter().flat_map(column).copied());
        }
        let trial_pages = pages(&mut trial.iter());
        if trial_pages > best_pages {
            (best, best_pages) = (trial, trial_pages);
        }
    }
    if best_pages >= minimum {
        best.into_iter()
            .map(|hit| (hit.page_slot, hit.line_slot))
            .collect()
    } else {
        HashSet::new()
    }
}

// Rotated running stamps live at the side edges, outside the header/footer bands.
// Keep each edge separate so repeated body text cannot supply its witnesses.
fn furniture_edge(line: &Line, width: f64, height: f64) -> Option<u8> {
    let rotated = line.bbox[3] - line.bbox[1] > 3.0 * (line.bbox[2] - line.bbox[0]);
    if rotated && line.bbox[2] < width * 0.08 {
        Some(2)
    } else if rotated && line.bbox[0] > width * 0.92 {
        Some(3)
    } else if line.bbox[1] < height * 0.12 {
        Some(0)
    } else {
        (line.bbox[3] > height * 0.90).then_some(1)
    }
}

/// The lines a document too short to repeat its running heads and feet (under three pages) shows
/// as furniture by place and type alone, as pdf-inspector reads a page's edges
/// (strip_header_footer_lines: an isolated edge block of a line or two, set smaller than the body,
/// no marker or numbered row): each (view, line slot on the page).
fn edge_furniture(views: &[PageView]) -> HashSet<(usize, usize)> {
    use pdf_inspector::types::TextLine;
    let mut lines: Vec<TextLine> = Vec::new();
    for (page_slot, view) in views.iter().enumerate() {
        let mut member = vec![false; view.page.lines.len()];
        for &slot in &view.lines {
            member[view.place(slot)] = true;
        }
        let crate::model::PageItems { items, origin } =
            legal_pdf_rules::tables::items::line_items(&view.page.lines, page_slot as u32 + 1, view.page.height);
        for (mut item, (slot, _)) in items
            .into_iter()
            .zip(origin)
            .filter(|(_, (slot, _))| member[*slot])
        {
            item.mcid = Some(slot as i64);
            match lines.last_mut() {
                Some(line) if line.page == item.page && line.items[0].mcid == item.mcid => {
                    line.items.push(item)
                }
                _ => lines.push(TextLine {
                    y: item.y,
                    page: item.page,
                    adaptive_threshold: 0.1,
                    items: vec![item],
                }),
            }
        }
    }
    let key = |line: &TextLine| {
        (
            line.page as usize - 1,
            line.items[0].mcid.unwrap_or(-1) as usize,
        )
    };
    let offered: HashSet<(usize, usize)> = lines.iter().map(key).collect();
    let kept: HashSet<(usize, usize)> =
        pdf_inspector::markdown::strip_header_footer_lines(lines, views.len() as u32)
            .iter()
            .map(key)
            .collect();
    offered.difference(&kept).copied().collect()
}

/// The edge a line's frame row stands at: one of its page's first or last rows that stand apart
/// from the body (`Rows::frame`).
fn frame_edge(rows: &Rows, line_slot: usize) -> Option<Edge> {
    let row = rows.of[line_slot];
    if row == u32::MAX {
        None
    } else if row < rows.frame[0] {
        Some(Edge::Top)
    } else {
        (row + rows.frame[1] >= rows.count).then_some(Edge::Bottom)
    }
}

/// A division's own heading, printed alone on its line: "PART 50", "FORMULE 18", "ARTICLE 47".
fn division_marker(text: &str) -> bool {
    regex!(r"(?i)^\s*(?:part|partie|form|formule|article|schedule|chapter|chapitre)\s+(?:\d{1,4}(?:\.\d{1,3})*[a-z]?|[ivxlcdm]{1,7}|[a-z])\.?\s*$").is_match(text)
}

/// A statute's running head naming the sections on its page: "Section 63", "Sections 12–14",
/// "Articles 5-7"; the first number it names.
fn section_head(text: &str) -> Option<u32> {
    regex!(r"(?i)^\s*(?:sections?|articles?)\s+(\d{1,4})[a-z]?(?:\.\d{1,3})*(?:\s*(?:-|–|—|to|and|&)\s*\d{1,4}[a-z]?(?:\.\d{1,3})*)?\s*$")
        .captures(text)
        .and_then(|found| found[1].parse().ok())
}

/// The lines at a page's edges (the head's and foot's bands, or its first and last rows) that name
/// the sections on their page (`section_head`): each line's slot, its edge, and the first section
/// it names. They are furniture where the statute's own sections bear them out.
pub fn section_heads(views: &[PageView], rows: &[&Rows]) -> Vec<(Slot, Edge, u32)> {
    let mut heads = Vec::new();
    for (view, rows) in views.iter().zip(rows) {
        let height = view.page.height;
        for (slot, line) in view.iter() {
            let row = rows.of[view.place(slot)];
            let edge = if row == 0 || line.bbox[1] < height * 0.12 {
                Edge::Top
            } else if Some(row) == rows.count.checked_sub(1) || line.bbox[3] > height * 0.90 {
                Edge::Bottom
            } else {
                continue;
            };
            heads.extend(section_head(&line.text).map(|number| (slot, edge, number)));
        }
    }
    heads
}

/// The document's furniture, read once over its pages (`views`, each page's `rows`): running
/// heads and feet repeated page after page at the page's edges (its head's and foot's bands, or its
/// first and last rows where they stand apart from the body), or set apart at a short document's
/// edges; the lines printing the pages' numbers (`folio`: the page map's, `legal_pdf_pages::page_map`);
/// document identifiers at the edges; a print driver's margins; the small lines a running head
/// stacks; and the lines sharing a row with a running head or a folio.
/// Page-and-line locators with no word among them ("84:2 85:2, 12") are a running head only on half
/// the document's pages or more: a word index's recur at one place on its own pages. A
/// paragraph's or provision's marker alone on its line beside its text ("(1)", "15(4)", "12" at a
/// page's top) is no running head or page number, though it recurs at one place. A division's
/// heading alone on its line ("PART 50") whose text no other line prints, and on a row of no other
/// furniture, is no running head. Words the body prints away from the edges on as many pages as
/// they recur at one place on the edge are the body's, no running head. Lines `line_label` holds are a transcript's line numbers, never
/// furniture. Each line read as furniture, with its edge and the rule that read it.
pub fn read(
    views: &[PageView],
    rows: &[&Rows],
    line_label: impl Fn(Slot) -> bool,
    folio: impl Fn(Slot) -> bool,
) -> Vec<(Slot, Edge, Rule)> {
    let mut candidates: HashMap<(u8, String), Vec<Hit>> = HashMap::new();
    // The pages printing each text (as repetition compares it) away from their edges.
    let mut body: HashMap<String, HashSet<usize>> = HashMap::new();
    measure("furniture.candidates", || {
        for (page_slot, view) in views.iter().enumerate() {
            let page = view.page;
            if page.width <= 0.0 || page.height <= 0.0 {
                continue;
            }
            for (slot, line) in view.iter() {
                let line_slot = view.place(slot);
                let edge = furniture_edge(line, page.width, page.height).or_else(|| {
                    frame_edge(rows[page_slot], line_slot)
                        .map(|edge| u8::from(edge == Edge::Bottom))
                });
                let normalized = normalize_furniture(&line.text);
                let Some(edge) = edge else {
                    body.entry(normalized).or_default().insert(page_slot);
                    continue;
                };
                if !normalized.is_empty() {
                    let (y_ratio, x_ratio) = (
                        line.bbox[1] / page.height,
                        (line.bbox[0] + line.bbox[2]) / (2.0 * page.width),
                    );
                    candidates.entry((edge, normalized)).or_default().push(Hit {
                        page_slot,
                        line_slot,
                        page_number: page.number,
                        y_ratio,
                        x_ratio,
                    });
                }
            }
        }
    });
    // Preserve the old engine's two-page/parity behavior, but cap the witness
    // at four pages as Text-Fidelity does once alignment is also required.
    let minimum = 4_usize.min(2_usize.max(((views.len() as f64) * 0.35).ceil() as usize));
    // Page-and-line locators with no word among them ("84:2 85:2, 12", "30:11", "22/8 24/19") are a
    // word index's where they recur only on its pages; a running head of that shape (a volume and
    // issue, "42:2") stands on half the document's pages or more.
    let figures = |key: &str| {
        !key.chars().any(char::is_alphabetic) && (key.contains("#:#") || key.contains("#/#"))
    };
    // A cover-page title can repeat the running title away from the header
    // baseline. Admit the stable occurrence cluster, not every matching string.
    // A running head's words are printed at the page's edge; words the body prints on at least as
    // many pages are the body's, where its flow reaches the edge (a statute's sections set in one
    // form of words, "shall record the reasons ...", the first lines of page after page).
    let mut repeated: HashSet<(usize, usize)> = measure("furniture.alignment", || {
        candidates
            .into_iter()
            .flat_map(|((_, key), hits)| {
                let has_even = hits.iter().any(|hit| hit.page_number % 2 == 0);
                let has_odd = hits.iter().any(|hit| hit.page_number % 2 == 1);
                let single_parity = has_even ^ has_odd;
                let required = if figures(&key) {
                    minimum.max(views.len().div_ceil(2))
                } else if single_parity && views.len() >= 4 {
                    let parity_total = if has_even {
                        views.len() / 2
                    } else {
                        views.len().div_ceil(2)
                    };
                    3_usize.min(2_usize.max(((parity_total as f64) * 0.5).ceil() as usize))
                } else {
                    minimum
                };
                let kept = aligned_furniture(&hits, required);
                let pages = kept
                    .iter()
                    .map(|&(page_slot, _)| page_slot)
                    .collect::<HashSet<_>>()
                    .len();
                let printed = if key.chars().any(char::is_alphabetic) {
                    body.get(&key).map_or(0, HashSet::len)
                } else {
                    0
                };
                if pages > 0 && printed >= pages {
                    HashSet::new()
                } else {
                    kept
                }
            })
            .collect()
    });
    // A document too short for repetition shows its furniture at its pages' edges.
    if views.len() < 3 {
        repeated.extend(edge_furniture(views));
    }
    // Each view's lines read as furniture, by line slot on the page.
    let mut found: Vec<Vec<Option<(Edge, Rule)>>> = views
        .iter()
        .map(|view| vec![None; view.page.lines.len()])
        .collect();
    measure("furniture.apply", || {
        for (page_slot, view) in views.iter().enumerate() {
            let (page, found) = (view.page, &mut found[page_slot]);
            let is_repeated = |index: usize| repeated.contains(&(page_slot, index));
            let is_folio = |index: usize| folio(view.slot(index));
            let page_rows = &rows[page_slot].of;
            let members: Vec<usize> = view.lines.iter().map(|&slot| view.place(slot)).collect();
            let body_size = quantile(
                members
                    .iter()
                    .filter(|&&index| {
                        (page.height * 0.10..=page.height * 0.75)
                            .contains(&page.lines[index].bbox[1])
                    })
                    .map(|&index| page.lines[index].size)
                    .filter(|size| (4.0..=24.0).contains(size))
                    .collect(),
                0.75,
            );
            let body_size = if body_size > 0.0 { body_size } else { 10.0 };
            // The lines no running head or page number takes, though they recur at one place.
            // A number set in the running head at the same place page after page is a folio, though
            // the head's title shares its row ("584  r. v. mabior  [2012] 2 S.C.R.").
            let mut attached: HashSet<usize> = members
                .iter()
                .copied()
                .filter(|&index| {
                    let label = &page.lines[index];
                    !(is_repeated(index) && label.bbox[1] < page.height * 0.12)
                        && standalone_note_label(label)
                        && members.iter().any(|&body| {
                            !is_repeated(body)
                                && aligned_note_body(&page.lines, page_rows, index, body)
                        })
                })
                .collect();
            // A printed paragraph number or a provision's marker opening the body under a running
            // title sits at the same place on every page, like a folio; its text shares its row. A
            // heading's number ("1.", "IV.") holds its row's text whatever repeats there; a
            // paragraph's or provision's ("[12]", "(1)", "15(4)", "(b)", "12") holds words no
            // other page repeats there, as no folio beside its running title does.
            attached.extend(members.iter().copied().filter(|&index| {
                let label = &page.lines[index];
                let heading = label.text.split_whitespace().count() == 1
                    && crate::rules::evidence::enumerated(&label.text);
                (heading || structure_marker(&label.text))
                    && members.iter().any(|&body_index| {
                        let body = &page.lines[body_index];
                        (heading
                            || !is_repeated(body_index)
                                && body.text.chars().any(char::is_alphabetic))
                            && aligned_note_body(&page.lines, page_rows, index, body_index)
                    })
            }));
            // Repeated shortforms such as "Ibid." remain note bodies when a small
            // detached label shares their baseline. Repetition alone is not enough
            // to turn a source-backed legal citation into running furniture.
            attached.extend(
                members
                    .iter()
                    .filter_map(|&index| {
                        let label = &page.lines[index];
                        if !standalone_note_label(label)
                            || label.bbox[1] <= page.height * 0.86
                            || is_folio(index)
                        {
                            return None;
                        }
                        let body_index = aligned_note_body_among(
                            &page.lines,
                            page_rows,
                            members.iter().copied(),
                            index,
                        )?;
                        let body = &page.lines[body_index];
                        (body.bbox[1] > page.height * 0.86
                            && page.lines[body_index].size < body_size * 0.90
                            && legal_structure_lines::analysis::has_citation_signal(&body.text))
                        .then_some([index, body_index])
                    })
                    .flatten(),
            );
            for (&slot, &index) in view.lines.iter().zip(&members) {
                // A transcript's line number is claimed already, though its digits repeat.
                if line_label(slot) {
                    continue;
                }
                let line = &page.lines[index];
                let line_size = page.lines[index].size;
                let frame = frame_edge(rows[page_slot], index);
                let (banded_top, banded_bottom) = (
                    line.bbox[1] < page.height * 0.12,
                    line.bbox[3] > page.height * 0.90,
                );
                let at_top = banded_top || frame == Some(Edge::Top);
                let at_bottom = banded_bottom || frame == Some(Edge::Bottom);
                let edge = |top: bool| if top { Edge::Top } else { Edge::Bottom };
                let compact_note =
                    at_bottom && line_size < body_size * 0.90 && compact_note_line(&line.text);
                found[index] = if is_repeated(index)
                    && furniture_edge(line, page.width, page.height).is_some_and(|edge| edge >= 2)
                {
                    Some((Edge::Top, Rule::RunningHead))
                } else if is_folio(index) {
                    Some((edge(line.bbox[1] + line.bbox[3] < page.height), Rule::Folio))
                } else if is_repeated(index)
                    && (at_top || at_bottom)
                    && (line_size >= body_size * 0.75 || normalize_furniture(&line.text) != "#")
                    && !compact_note
                    && !attached.contains(&index)
                {
                    Some((
                        edge(at_top),
                        if banded_top || banded_bottom {
                            Rule::RunningHead
                        } else {
                            Rule::FrameRow
                        },
                    ))
                } else if (at_top || at_bottom) && document_id_line(&line.text) {
                    Some((edge(at_top), Rule::DocumentId))
                } else if (at_bottom || line.bbox[1] > page.height * 0.85)
                    && copyright_footer_line(&line.text)
                {
                    Some((Edge::Bottom, Rule::PrintMargin))
                } else {
                    None
                };
            }
            // A bare number with a line of the page's prose on its row is the margin's, not the page's: a law report
            // prints its paragraph numbers beside the row each opens on, and at the page's first or last row they read
            // as a repeated edge number ("#"), in either margin ("Although ..." / "Bien que ..." / "25"). A folio or a
            // page number stands beside furniture or alone; the page's folio the page map read stands after its row's
            // words too (a scan's running head read as one line of prose: "[1996] I R.C.S. CHABLIS TEXTILES ... Le juge
            // Gonthier" / "167"), and is a paragraph's number only where the prose it opens follows it ("4" / "The
            // Union has traditionally ...").
            let row_of = &rows[page_slot].of;
            let letters = |index: usize| {
                page.lines[index]
                    .text
                    .chars()
                    .filter(|c| c.is_alphabetic())
                    .count()
            };
            let row_mates = |index: usize| {
                (0..page.lines.len())
                    .filter(move |&other| other != index && row_of[other] == row_of[index])
            };
            let opens = |index: usize, other: usize| {
                !is_folio(index) || page.lines[other].bbox[0] >= page.lines[index].bbox[0]
            };
            // (Figures too long for a folio stand so in a table's row, its words in cells: "(7)  15050040" / "James Bingham en
            // 1862" / "11749133" / "9 févr. 2001" / "& Timber Corporation".)
            let margin: Vec<usize> = (0..page.lines.len())
                .filter(|&index| {
                    let text = page.lines[index].text.trim();
                    found[index].is_some()
                        && !text.is_empty()
                        && text.bytes().all(|byte| byte.is_ascii_digit())
                        && row_of[index] != u32::MAX
                        && (text.len() <= 4
                            && row_mates(index).any(|other| {
                                found[other].is_none()
                                    && letters(other) >= 20
                                    && opens(index, other)
                            })
                            || text.len() > 4
                                && row_mates(index)
                                    .filter(|&other| found[other].is_none())
                                    .map(letters)
                                    .sum::<usize>()
                                    >= 20)
                })
                .collect();
            for index in margin {
                found[index] = None;
            }
            for (index, edge) in print_margins(view, page_rows, &line_label) {
                found[index] = Some((edge, Rule::PrintMargin));
            }
            extend_running_heads(page, &members, found, body_size, |index| {
                line_label(view.slot(index))
            });
            extend_bottom_footers(page, &members, found, body_size, |index| {
                line_label(view.slot(index))
            });
        }
    });
    // A division's heading the edge bands took for a running head: its own text, alone on its line,
    // read at the edges on no other page (its number changes from page to page, as a running head's
    // does not; a contents list may print it too), and on no row of furniture but other such
    // headings ("PART 50" beside "PARTIE 50").
    let mut printed: HashMap<&str, usize> = HashMap::new();
    for (view, found) in views.iter().zip(&found) {
        for (_, line) in view.iter().filter(|(slot, line)| {
            found[view.place(*slot)].is_some() && division_marker(&line.text)
        }) {
            *printed.entry(line.text.trim()).or_default() += 1;
        }
    }
    for ((view, rows), found) in views.iter().zip(rows).zip(&mut found) {
        let heading: Vec<bool> = (0..found.len())
            .map(|index| {
                let text = &view.page.lines[index].text;
                found[index].is_some()
                    && division_marker(text)
                    && printed.get(text.trim()) == Some(&1)
            })
            .collect();
        for index in (0..found.len()).filter(|&index| heading[index]) {
            if !(0..found.len()).any(|other| {
                found[other].is_some() && !heading[other] && rows.of[other] == rows.of[index]
            }) {
                found[index] = None;
            }
        }
    }
    // A line on the row of a running head or a folio is the head's or foot's. (TEMP lever "rowmate-prose": not a line of
    // prose, a provision's number and seven words or more, that a scan sets close under the head: "31. The size of the
    // house drain shall be".)
    let rowmate_prose = legal_pdf_core::lever("rowmate-prose");
    // (TEMP lever "rowmate-sentence": nor a sentence running on from the page before, numbered or not, "plication shall
    // be made in writing": five words or more, three of them lower case. A running head is short and set in capitals or
    // title case.)
    let rowmate_sentence = legal_pdf_core::lever("rowmate-sentence");
    let prose = |text: &str| {
        let lower = text.split_whitespace().filter(|word| word.chars().all(char::is_lowercase)).count();
        (rowmate_prose
            && text.split_whitespace().count() >= 7
            && text.trim_start().starts_with(|c: char| c.is_ascii_digit())
            && lower >= 3)
            || (rowmate_sentence && text.split_whitespace().count() >= 5 && lower >= 3)
    };
    for ((view, rows), found) in views.iter().zip(rows).zip(&mut found) {
        let heads: Vec<(u32, Edge)> = found
            .iter()
            .enumerate()
            .filter_map(|(index, read)| {
                read.filter(|(_, rule)| {
                    matches!(rule, Rule::RunningHead | Rule::FrameRow | Rule::Folio)
                })
                .map(|(edge, _)| (rows.of[index], edge))
            })
            .filter(|(row, _)| *row != u32::MAX)
            .collect();
        for &slot in &view.lines {
            let index = view.place(slot);
            if found[index].is_none() && !line_label(slot) && !prose(&view.line(slot).text) {
                if let Some(&(_, edge)) = heads.iter().find(|(row, _)| *row == rows.of[index]) {
                    found[index] = Some((edge, Rule::RowMate));
                }
            }
        }
    }
    views
        .iter()
        .zip(found)
        .flat_map(|(view, found)| {
            view.lines.iter().filter_map(move |&slot| {
                found[view.place(slot)].map(|(edge, rule)| (slot, edge, rule))
            })
        })
        .collect()
}

/// A running head can be a stack of small-type lines whose lower lines name the current
/// heading or sections and so change from page to page. A line in the top band set in a
/// running-head line's type, flush with it and directly above or below it, is part of it.
/// `members` are the page's lines read, `found` the furniture read of them by line slot.
fn extend_running_heads(
    page: &Page,
    members: &[usize],
    found: &mut [Option<(Edge, Rule)>],
    body_size: f64,
    line_label: impl Fn(usize) -> bool,
) {
    let top_band = page.height * 0.12;
    extend_stacked_band(
        page,
        members,
        found,
        body_size,
        line_label,
        (Edge::Top, Rule::RunningHead),
        |read| read.is_some_and(|(edge, _)| edge == Edge::Top),
        |anchor, candidate, height| {
            anchor.bbox[3] < top_band
                && candidate.bbox[3] < top_band
                && (candidate.bbox[1] - anchor.bbox[3]).max(anchor.bbox[1] - candidate.bbox[3])
                    <= height * 0.5
        },
    );
}

/// Absorb wrapped continuation lines of a publisher/copyright bottom footer in small type
/// strictly below the initial footer line inside the bottom band (`y > 0.85 * height`).
fn extend_bottom_footers(
    page: &Page,
    members: &[usize],
    found: &mut [Option<(Edge, Rule)>],
    body_size: f64,
    line_label: impl Fn(usize) -> bool,
) {
    let bottom_band = page.height * 0.85;
    extend_stacked_band(
        page,
        members,
        found,
        body_size,
        line_label,
        (Edge::Bottom, Rule::PrintMargin),
        |read| read.is_some_and(|(edge, rule)| edge == Edge::Bottom && rule == Rule::PrintMargin),
        |anchor, candidate, height| {
            anchor.bbox[1] > bottom_band
                && candidate.bbox[1] > bottom_band
                && candidate.bbox[1] >= anchor.bbox[1]
                && candidate.bbox[1] - anchor.bbox[3] <= height * 0.5
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn extend_stacked_band(
    page: &Page,
    members: &[usize],
    found: &mut [Option<(Edge, Rule)>],
    body_size: f64,
    line_label: impl Fn(usize) -> bool,
    tag: (Edge, Rule),
    is_anchor: impl Fn(Option<(Edge, Rule)>) -> bool,
    adjacent: impl Fn(&Line, &Line, f64) -> bool,
) {
    let tolerance = page.width * FURNITURE_X_TOLERANCE_FRAC;
    loop {
        let joined: Vec<usize> = members
            .iter()
            .copied()
            .filter(|&index| {
                let (line, size) = (&page.lines[index], page.lines[index].size);
                !is_anchor(found[index])
                    && found[index].is_none()
                    && !line_label(index)
                    && size > 0.0
                    && size < body_size * 0.90
                    && members.iter().any(|&anchor_index| {
                        let anchor = &page.lines[anchor_index];
                        let height =
                            (anchor.bbox[3] - anchor.bbox[1]).max(line.bbox[3] - line.bbox[1]);
                        is_anchor(found[anchor_index])
                            && (anchor.size - size).abs() <= 0.5
                            && ((anchor.bbox[0] - line.bbox[0]).abs() <= tolerance
                                || (line_center_x(anchor) - line_center_x(line)).abs() <= tolerance)
                            && adjacent(anchor, line, height)
                    })
            })
            .collect();
        if joined.is_empty() {
            return;
        }
        for index in joined {
            found[index] = Some(tag);
        }
    }
}

/// Whether a line at the bottom of a page is publisher/copyright boilerplate rather than prose or a note.
fn copyright_footer_line(value: &str) -> bool {
    let lower = value.trim().to_ascii_lowercase();
    const BOILERPLATE_CUES: [&str; 9] = [
        "author",
        "press",
        "journal",
        "association",
        "society",
        "university",
        "all rights reserved",
        "creative commons",
        "doi.org",
    ];
    if lower.starts_with('©')
        || lower.starts_with("copyright ©")
        || lower.starts_with("copyright (c)")
        || lower.starts_with("published online by ")
        || lower.starts_with("downloaded from ")
        || lower.starts_with("this is an open access article")
    {
        return true;
    }
    if lower.starts_with("published by ")
        && [
            "university press",
            "on behalf of",
            "©",
            "all rights reserved",
            "creative commons",
        ]
        .iter()
        .any(|cue| lower.contains(cue))
    {
        return true;
    }
    (lower.contains('©') || lower.starts_with("copyright "))
        && BOILERPLATE_CUES.iter().any(|cue| lower.contains(cue))
}

/// A browser or print driver sets the page's address and the time it was printed in the
/// outer margins, with the page title and "1 of 3" on the same rows. They are furniture on
/// a single page too, where nothing repeats to show it. Each such line of the view, by line slot
/// on the page, with its edge; a line `line_label` holds is none.
fn print_margins(
    view: &PageView,
    rows: &[u32],
    line_label: impl Fn(Slot) -> bool,
) -> Vec<(usize, Edge)> {
    let stamp = regex!(
        r"(?i)(?:^|\s)(?:https?://|www\.)\S+|(?:^|\s)\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4},?\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:[ap]\.?\s?m\.?)?"
    );
    let (height, band) = (view.page.height, view.page.height * 0.06);
    let edge = |line: &Line| {
        if line.bbox[1] <= band {
            Some(Edge::Top)
        } else {
            (line.bbox[3] >= height - band).then_some(Edge::Bottom)
        }
    };
    // Each stamp's row on the page (`Rows`) and its edge.
    let stamped: Vec<(Edge, u32)> = view
        .iter()
        .filter(|(slot, line)| {
            stamp.is_match(line.text.trim()) && rows[view.place(*slot)] != u32::MAX
        })
        .filter_map(|(slot, line)| edge(line).map(|kind| (kind, rows[view.place(slot)])))
        .collect();
    if stamped.is_empty() {
        return Vec::new();
    }
    view.iter()
        .filter(|(slot, _)| !line_label(*slot))
        .filter_map(|(slot, line)| {
            let kind = edge(line)?;
            stamped
                .contains(&(kind, rows[view.place(slot)]))
                .then_some((view.place(slot), kind))
        })
        .collect()
}

/// A document management identifier a firm prints in the margin ("MT DOCS 20632860v5",
/// "33312.140943.JMS3.15349802.2", "43882055_21NATDOCS"): digits with at most capitals, dots,
/// underscores and a version suffix, and no words.
fn document_id_line(value: &str) -> bool {
    let text = value.trim();
    let digits = text.chars().filter(char::is_ascii_digit).count();
    let visible = text.chars().filter(|c| !c.is_whitespace()).count();
    // (Eight figures: "1974.14" is a year and a note's call.)
    digits >= 8
        && digits * 5 >= visible * 2
        && text.chars().count() <= 40
        && regex!(r"^[A-Z0-9][A-Z0-9 ._/-]*(?:v\d{1,3})?$").is_match(text)
}

/// A paragraph's or provision's marker alone on its line: "[12]", "(1)", "(b)", "(iv)", "3.",
/// "1.2.3", "12", "15(4)", "29(2)(a)".
fn structure_marker(text: &str) -> bool {
    standalone_paragraph_number(text)
        || crate::text::paragraphs::marker_only_re().is_match(text)
        || regex!(r"^\s*\d{1,4}[A-Za-z]?(?:\.\d{1,3})*(?:\([0-9A-Za-z]{1,4}\))*\.?\s*$")
            .is_match(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use legal_citations as _;
    use legal_citations_data as _;

    fn line_box(text: &str, size: f64, bbox: [f64; 4]) -> Line {
        Line {
            text: text.to_owned(),
            size,
            bbox,
            ..Default::default()
        }
    }

    #[test]
    fn copyright_footer_line_matches_boilerplate_and_ignores_legal_prose() {
        assert!(copyright_footer_line(
            "© The Author(s), 2024. Published by Cambridge University Press"
        ));
        assert!(copyright_footer_line(
            "Downloaded from https://www.cambridge.org/core/terms"
        ));
        assert!(copyright_footer_line(
            "Published online by Cambridge University Press"
        ));
        assert!(!copyright_footer_line(
            "12 See Copyright Act, RSC 1985, c C-42, s 3."
        ));
        assert!(!copyright_footer_line(
            "Published by Carswell in 1998 as part of the treatise."
        ));
    }

    #[test]
    fn single_parity_running_headers_are_detected_on_three_even_pages() {
        let mut pages = Vec::new();
        for n in 1..=10u32 {
            let mut lines = vec![
                line_box("First line of ordinary journal prose.", 11.0, [72.0, 150.0, 520.0, 162.0]),
                line_box("Second line of ordinary journal prose.", 11.0, [72.0, 166.0, 520.0, 178.0]),
            ];
            if matches!(n, 4 | 6 | 8) {
                lines.insert(
                    0,
                    line_box(
                        &format!("{n} \u{f06e} APPEAL VOLUME 12"),
                        9.0,
                        [72.0, 42.0, 260.0, 52.0],
                    ),
                );
            }
            pages.push(Page {
                index: (n - 1) as usize,
                number: n,
                width: 612.0,
                height: 792.0,
                lines,
                ..Default::default()
            });
        }
        let rows_vec: Vec<Rows> = pages
            .iter()
            .map(|page| Rows {
                of: (0..page.lines.len() as u32).collect(),
                count: page.lines.len() as u32,
                frame: [0, 0],
            })
            .collect();
        let row_refs: Vec<&Rows> = rows_vec.iter().collect();
        let views = PageView::whole(&pages);
        let hits = read(&views, &row_refs, |_| false, |_| false);
        let top_hits: Vec<usize> = hits
            .iter()
            .filter_map(|&(slot, edge, rule)| {
                let page_index = pages
                    .iter()
                    .position(|p| p.lines.iter().any(|l| std::ptr::eq(l, views.iter().flat_map(|v| v.lines.iter().map(move |&s| (v, s))).find(|(_, s)| *s == slot).unwrap().0.line(slot))))?;
                (edge == Edge::Top && rule == Rule::RunningHead).then_some(page_index + 1)
            })
            .collect();
        assert_eq!(top_hits, vec![4, 6, 8]);
    }
}
