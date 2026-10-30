//! A document's furniture: running heads and feet repeated page after page, or set apart at a
//! short document's edges; folios and the printed page labels they carry; side stamps and print
//! margins.

use super::*;

pub(super) fn normalize_furniture(text: &str) -> String {
    static DIGITS: OnceLock<Regex> = OnceLock::new();
    let text = DIGITS
        .get_or_init(|| Regex::new(r"\d+").unwrap())
        .replace_all(text, "#");
    let mut normalized = String::with_capacity(text.len());
    let mut pending_space = false;
    for character in text.chars().flat_map(char::to_lowercase) {
        if character.is_whitespace() {
            pending_space = !normalized.is_empty();
        } else {
            if pending_space {
                normalized.push(' ');
                pending_space = false;
            }
            normalized.push(character);
        }
    }
    normalized
}

pub(super) fn compact_note_line(text: &str) -> bool {
    let Some(prefix) = line_start_label_prefix(text) else {
        return false;
    };
    scalar_suffix(text, prefix.end)
        .trim_start_matches(|character: char| {
            character.is_whitespace() || ".)],:;-".contains(character)
        })
        .chars()
        .filter(|character| character.is_alphabetic())
        .take(4)
        .count()
        == 4
}

pub(super) const FOLIO_MIN_SEQUENCE_PAGES: usize = 4;

pub(super) const FOLIO_EDGE_MAX_FRAC: f64 = 0.25;

pub(super) const FOLIO_BOTTOM_MIN_FRAC: f64 = 0.92;

pub(super) const FURNITURE_Y_TOLERANCE_FRAC: f64 = 0.015;

pub(super) const FURNITURE_X_TOLERANCE_FRAC: f64 = 0.04;

#[derive(Clone, Copy)]
pub(super) struct FolioCandidate {
    page_slot: usize,
    line_slot: usize,
    page_number: u32,
    printed_number: u32,
    /// The bottom edge's left (-1), centre (0) or right (1).
    side: i8,
    y_ratio: f64,
    x_ratio: f64,
}

#[derive(Clone, Copy)]
pub(super) struct FurnitureHit {
    page_slot: usize,
    line_slot: usize,
    page_number: u32,
    y_ratio: f64,
    x_ratio: f64,
}

pub(super) fn arabic_page_number(value: &str) -> Option<u32> {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*-?\s*(\d{1,4})\s*-?\s*$").unwrap())
        .captures(value)
        .and_then(|capture| capture.get(1))
        .and_then(|value| value.as_str().parse().ok())
}

/// The page numbers the document prints at its pages' edges, wherever in the margin it sets them: a
/// bare number on its page's top or bottom row (`rows`, by view) is a folio when the number less
/// the page's place in the document is the same on at least three such pages (or half of a shorter
/// document's). Returns each folio, by (view, line slot on the page), with whether it heads its
/// page.
pub(super) fn edge_folios(views: &[PageView], rows: &[&Rows]) -> HashMap<(usize, usize), bool> {
    let mut by_offset = HashMap::<i64, Vec<(usize, usize, bool)>>::new();
    for (page_slot, view) in views.iter().enumerate() {
        let rows = rows[page_slot];
        let last = rows.count.checked_sub(1);
        for (slot, line) in view.iter() {
            let line_slot = view.place(slot);
            let row = rows.of[line_slot];
            let top = row == 0;
            if !(top || Some(row) == last) {
                continue;
            }
            if let Some(printed) = arabic_page_number(&line.text) {
                by_offset
                    .entry(i64::from(printed) - i64::from(view.page.number))
                    .or_default()
                    .push((page_slot, line_slot, top));
            }
        }
    }
    let needed = 3.min(views.len().div_ceil(2)).max(2);
    by_offset
        .into_values()
        .filter(|folios| folios.iter().map(|folio| folio.0).collect::<HashSet<_>>().len() >= needed)
        .flatten()
        .map(|(page_slot, line_slot, top)| ((page_slot, line_slot), top))
        .collect()
}

/// Text-Fidelity's alternating-folio witness. A singleton Arabic number at
/// each outer bottom edge (its foot's band, or its last row) is furniture only when at least four
/// consecutive PDF pages carry a +1 printed sequence, alternate sides, and remain aligned
/// within each page parity. A number centred at the bottom is furniture on the
/// same sequence without alternating, whatever its type: a statute sets its
/// folio as small as its notes.
pub(super) fn alternating_folios(views: &[PageView], rows: &[&Rows]) -> HashSet<(usize, usize)> {
    let mut singletons = Vec::new();
    for (page_slot, view) in views.iter().enumerate() {
        let page = view.page;
        if page.width <= 0.0 || page.height <= 0.0 {
            continue;
        }
        let rows = rows[page_slot];
        let last = rows.count.checked_sub(1);
        let mut candidates = view
            .iter()
            .filter_map(|(slot, line)| {
                let line_slot = view.place(slot);
                let printed_number = arabic_page_number(&line.text)?;
                let center_x = (line.bbox[0] + line.bbox[2]) / 2.0;
                let x_ratio = center_x / page.width;
                let side = if x_ratio < FOLIO_EDGE_MAX_FRAC {
                    -1
                } else if x_ratio > 1.0 - FOLIO_EDGE_MAX_FRAC {
                    1
                } else if (x_ratio - 0.5).abs() <= FURNITURE_X_TOLERANCE_FRAC {
                    0
                } else {
                    return None;
                };
                (line.bbox[1] > page.height * FOLIO_BOTTOM_MIN_FRAC || Some(rows.of[line_slot]) == last).then_some(FolioCandidate {
                    page_slot,
                    line_slot,
                    page_number: page.number,
                    printed_number,
                    side,
                    y_ratio: line.bbox[1] / page.height,
                    x_ratio,
                })
            });
        if let Some(candidate) = candidates.next() {
            if candidates.next().is_none() {
                singletons.push(candidate);
            }
        }
    }
    singletons.sort_by_key(|candidate| candidate.page_number);

    fn aligned(run: &[FolioCandidate], candidate: FolioCandidate) -> bool {
        let previous = run.last().expect("nonempty folio run");
        let centred = candidate.side == 0 || previous.side == 0;
        if candidate.page_number != previous.page_number + 1
            || candidate.printed_number != previous.printed_number + 1
            || if centred {
                candidate.side != previous.side
            } else {
                candidate.side == previous.side
            }
            || run.iter().any(|old| {
                old.page_number % 2 == candidate.page_number % 2 && old.side != candidate.side
            })
        {
            return false;
        }
        let mut y_min = candidate.y_ratio;
        let mut y_max = candidate.y_ratio;
        let mut side_x_min = candidate.x_ratio;
        let mut side_x_max = candidate.x_ratio;
        for old in run {
            y_min = y_min.min(old.y_ratio);
            y_max = y_max.max(old.y_ratio);
            if old.side == candidate.side {
                side_x_min = side_x_min.min(old.x_ratio);
                side_x_max = side_x_max.max(old.x_ratio);
            }
        }
        y_max - y_min <= FURNITURE_Y_TOLERANCE_FRAC
            && side_x_max - side_x_min <= FURNITURE_X_TOLERANCE_FRAC
    }

    let mut admitted = HashSet::new();
    let mut run = Vec::new();
    let flush = |run: &mut Vec<FolioCandidate>, admitted: &mut HashSet<(usize, usize)>| {
        if run.len() >= FOLIO_MIN_SEQUENCE_PAGES {
            admitted.extend(
                run.iter()
                    .map(|candidate| (candidate.page_slot, candidate.line_slot)),
            );
        }
        run.clear();
    };
    for candidate in singletons {
        if !run.is_empty() && !aligned(&run, candidate) {
            flush(&mut run, &mut admitted);
        }
        run.push(candidate);
    }
    flush(&mut run, &mut admitted);
    admitted
}

pub(super) fn aligned_furniture(hits: &[FurnitureHit], minimum: usize) -> HashSet<(usize, usize)> {
    fn unique_page_count<'a>(
        hits: impl IntoIterator<Item = &'a FurnitureHit>,
        seen: &mut [u32],
        generation: &mut u32,
    ) -> usize {
        *generation = generation.wrapping_add(1);
        if *generation == 0 {
            seen.fill(0);
            *generation = 1;
        }
        let mut count = 0;
        for hit in hits {
            if seen[hit.page_slot] != *generation {
                seen[hit.page_slot] = *generation;
                count += 1;
            }
        }
        count
    }

    let mut best = Vec::with_capacity(hits.len());
    let mut best_page_count = 0;
    let mut y_cluster = Vec::with_capacity(hits.len());
    let mut parity_hits = Vec::with_capacity(hits.len());
    let mut trial = Vec::with_capacity(hits.len());
    let mut seen_pages = vec![0_u32; hits.iter().map(|hit| hit.page_slot).max().unwrap_or(0) + 1];
    let mut x_counts = BTreeMap::new();
    let mut generation = 0;
    for y_start in hits {
        y_cluster.clear();
        y_cluster.extend(hits.iter().copied().filter(|hit| {
            hit.y_ratio >= y_start.y_ratio
                && hit.y_ratio - y_start.y_ratio <= FURNITURE_Y_TOLERANCE_FRAC
        }));
        trial.clear();
        for parity in [0, 1] {
            x_counts.clear();
            parity_hits.clear();
            parity_hits.extend(
                y_cluster
                    .iter()
                    .copied()
                    .filter(|hit| hit.page_number % 2 == parity),
            );
            let mut selected_x = None;
            let mut selected_count = 0;
            for x_start in &parity_hits {
                let x = x_start.x_ratio.to_bits();
                let count = *x_counts.entry(x).or_insert_with(|| {
                    unique_page_count(
                        parity_hits.iter().filter(|hit| {
                            hit.x_ratio >= x_start.x_ratio
                                && hit.x_ratio - x_start.x_ratio <= FURNITURE_X_TOLERANCE_FRAC
                        }),
                        &mut seen_pages,
                        &mut generation,
                    )
                });
                // Iterator::max_by_key chooses the last equal maximum.
                if selected_x.is_none() || count >= selected_count {
                    selected_x = Some(x_start.x_ratio);
                    selected_count = count;
                }
            }
            if let Some(x_start) = selected_x {
                trial.extend(parity_hits.iter().copied().filter(|hit| {
                    hit.x_ratio >= x_start && hit.x_ratio - x_start <= FURNITURE_X_TOLERANCE_FRAC
                }));
            }
        }
        let trial_page_count = unique_page_count(&trial, &mut seen_pages, &mut generation);
        if trial_page_count > best_page_count {
            best.clear();
            best.extend_from_slice(&trial);
            best_page_count = trial_page_count;
        }
    }
    if best_page_count >= minimum {
        best.into_iter()
            .map(|hit| (hit.page_slot, hit.line_slot))
            .collect()
    } else {
        HashSet::new()
    }
}

// Rotated running stamps live at the side edges, outside the header/footer bands.
// Keep each edge separate so repeated body text cannot supply its witnesses.
pub(super) fn furniture_edge(line: &Line, width: f64, height: f64) -> Option<u8> {
    let rotated = line.bbox[3] - line.bbox[1] > 3.0 * (line.bbox[2] - line.bbox[0]);
    if rotated && line.bbox[2] < width * 0.08 {
        Some(2)
    } else if rotated && line.bbox[0] > width * 0.92 {
        Some(3)
    } else if line.bbox[1] < height * 0.12 {
        Some(0)
    } else if line.bbox[3] > height * 0.90 {
        Some(1)
    } else {
        None
    }
}

/// The lines a document too short to repeat its running heads and feet (under three pages) shows
/// as furniture by place and type alone, as pdf-inspector reads a page's edges
/// (strip_header_footer_lines: an isolated edge block of a line or two, set smaller than the body,
/// no marker or numbered row): each (view, line slot on the page).
pub(super) fn edge_furniture(views: &[PageView]) -> HashSet<(usize, usize)> {
    use pdf_inspector::types::TextLine;
    let mut lines: Vec<TextLine> = Vec::new();
    for (page_slot, view) in views.iter().enumerate() {
        let mut member = vec![false; view.page.lines.len()];
        for &slot in &view.lines {
            member[view.place(slot)] = true;
        }
        let legal_pdf_tables::PageItems { items, origin } =
            legal_pdf_tables::line_items(&view.page.lines, page_slot as u32 + 1, view.page.height);
        for (mut item, (slot, _)) in items.into_iter().zip(origin).filter(|(_, (slot, _))| member[*slot]) {
            item.mcid = Some(slot as i64);
            match lines.last_mut() {
                Some(line) if line.page == item.page && line.items[0].mcid == item.mcid => line.items.push(item),
                _ => lines.push(TextLine { y: item.y, page: item.page, adaptive_threshold: 0.1, items: vec![item] }),
            }
        }
    }
    let key = |line: &TextLine| (line.page as usize - 1, line.items[0].mcid.unwrap_or(-1) as usize);
    let offered: HashSet<(usize, usize)> = lines.iter().map(key).collect();
    let kept: HashSet<(usize, usize)> =
        pdf_inspector::markdown::strip_header_footer_lines(lines, views.len() as u32).iter().map(key).collect();
    offered.difference(&kept).copied().collect()
}

/// The edge of its page a line of furniture stands at: the running head's or the foot's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum Edge {
    Top,
    Bottom,
}

/// The edge a line's frame row stands at: one of its page's first or last rows that stand apart
/// from the body (`Rows::frame`).
fn frame_edge(rows: &Rows, line_slot: usize) -> Option<Edge> {
    let row = rows.of[line_slot];
    if row == u32::MAX {
        None
    } else if row < rows.frame[0] {
        Some(Edge::Top)
    } else if row + rows.frame[1] >= rows.count {
        Some(Edge::Bottom)
    } else {
        None
    }
}

/// A division's own heading, printed alone on its line: "PART 50", "FORMULE 18", "ARTICLE 47".
fn division_marker(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)^\s*(?:part|partie|form|formule|article|schedule|chapter|chapitre)\s+(?:\d{1,4}(?:\.\d{1,3})*[a-z]?|[ivxlcdm]{1,7}|[a-z])\.?\s*$")
            .unwrap()
    })
    .is_match(text)
}

/// A statute's running head naming the sections on its page: "Section 63", "Sections 12–14",
/// "Articles 5-7"; the first number it names.
fn section_head(text: &str) -> Option<u32> {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)^\s*(?:sections?|articles?)\s+(\d{1,4})[a-z]?(?:\.\d{1,3})*(?:\s*(?:-|–|—|to|and|&)\s*\d{1,4}[a-z]?(?:\.\d{1,3})*)?\s*$")
            .unwrap()
    })
    .captures(text)
    .and_then(|found| found[1].parse().ok())
}

/// The lines at a page's edges (the head's and foot's bands, or its first and last rows) that name
/// the sections on their page (`section_head`): each line's slot, its edge, and the first section
/// it names. They are furniture where the statute's own sections bear them out.
pub(super) fn section_heads(views: &[PageView], rows: &[&Rows]) -> Vec<(Slot, Edge, u32)> {
    let mut heads = Vec::new();
    for (view, rows) in views.iter().zip(rows) {
        let page = view.page;
        let last = rows.count.checked_sub(1);
        for (slot, line) in view.iter() {
            let row = rows.of[view.place(slot)];
            let edge = if row == 0 || line.bbox[1] < page.height * 0.12 {
                Edge::Top
            } else if Some(row) == last || line.bbox[3] > page.height * 0.90 {
                Edge::Bottom
            } else {
                continue;
            };
            if let Some(number) = section_head(&line.text) {
                heads.push((slot, edge, number));
            }
        }
    }
    heads
}

/// The document's furniture, read once over its pages (`views`, each page's `rows`): running
/// heads and feet repeated page after page at the page's edges (its head's and foot's bands, or its
/// first and last rows where they stand apart from the body), or set apart at a short document's
/// edges; folios; document identifiers and page numbers at the edges; a print driver's margins; the
/// small lines a running head stacks; and the lines sharing a row with a running head or a folio.
/// Page-and-line locators with no word among them ("84:2 85:2, 12") are a running head only on half
/// the document's pages or more: a word index's recur at one place on its own pages. A
/// paragraph's or provision's marker alone on its line beside its text ("(1)", "15(4)", "12" at a
/// page's top) is no running head or page number, though it recurs at one place. A division's
/// heading alone on its line ("PART 50") whose text no other line prints, and on a row of no other
/// furniture, is no running head. Lines `line_label` holds are a transcript's line numbers, never
/// furniture. Each line read as furniture, with its edge and the rule that read it.
pub(super) fn read(views: &[PageView], rows: &[&Rows], line_label: impl Fn(Slot) -> bool) -> Vec<(Slot, Edge, Rule)> {
    // LEGALPDF_ABLATE=furniture leaves the repeated-furniture reader out, for its measure under the tagger.
    if legal_structure_model::ablated("furniture") {
        return Vec::new();
    }
    let mut candidates: HashMap<(u8, String), Vec<FurnitureHit>> = HashMap::new();
    legal_pdf_support::profile::measure("furniture.candidates", || {
        for (page_slot, view) in views.iter().enumerate() {
            let page = view.page;
            if page.width <= 0.0 || page.height <= 0.0 {
                continue;
            }
            for (slot, line) in view.iter() {
                let line_slot = view.place(slot);
                let edge = furniture_edge(line, page.width, page.height).or_else(|| {
                    frame_edge(rows[page_slot], line_slot).map(|edge| u8::from(edge == Edge::Bottom))
                });
                let Some(top) = edge else {
                    continue;
                };
                let normalized = normalize_furniture(&line.text);
                if !normalized.is_empty() {
                    candidates
                        .entry((top, normalized))
                        .or_default()
                        .push(FurnitureHit {
                            page_slot,
                            line_slot,
                            page_number: page.number,
                            y_ratio: line.bbox[1] / page.height,
                            x_ratio: (line.bbox[0] + line.bbox[2]) / (2.0 * page.width),
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
    let figures = |key: &str| !key.chars().any(char::is_alphabetic) && (key.contains("#:#") || key.contains("#/#"));
    // A cover-page title can repeat the running title away from the header
    // baseline. Admit the stable occurrence cluster, not every matching string.
    let repeated: HashSet<(usize, usize)> =
        legal_pdf_support::profile::measure("furniture.alignment", || {
            candidates
                .into_iter()
                .flat_map(|((_, key), hits)| {
                    let needed = if figures(&key) { minimum.max(views.len().div_ceil(2)) } else { minimum };
                    aligned_furniture(&hits, needed)
                })
                .collect()
        });
    // A document too short for repetition shows its furniture at its pages' edges.
    let repeated = if views.len() < 3 {
        repeated.into_iter().chain(edge_furniture(views)).collect()
    } else {
        repeated
    };
    let sequence_folios =
        legal_pdf_support::profile::measure("furniture.folios", || alternating_folios(views, rows));
    let edge_folios = edge_folios(views, rows);
    // Each view's lines read as furniture, by line slot on the page.
    let mut found: Vec<Vec<Option<(Edge, Rule)>>> = views.iter().map(|view| vec![None; view.page.lines.len()]).collect();
    legal_pdf_support::profile::measure("furniture.apply", || {
        for (page_slot, view) in views.iter().enumerate() {
            let page = view.page;
            let found = &mut found[page_slot];
            let members: Vec<usize> = view.lines.iter().map(|&slot| view.place(slot)).collect();
            let line_sizes: Vec<_> = page.lines.iter().map(line_font_size).collect();
            // Without a sequence witness, a folio must sit beyond the remaining body text.
            let (top, bottom) = members
                .iter()
                .map(|&index| (index, &page.lines[index]))
                .filter(|(index, line)| {
                    !line.text.trim().is_empty()
                        && !repeated.contains(&(page_slot, *index))
                        && !sequence_folios.contains(&(page_slot, *index))
                        && !edge_folios.contains_key(&(page_slot, *index))
                })
                .fold(
                    (f64::INFINITY, f64::NEG_INFINITY),
                    |(top, bottom), (_, line)| (top.min(line.bbox[1]), bottom.max(line.bbox[3])),
                );
            let body_size = quantile(
                members
                    .iter()
                    .filter(|&&index| {
                        let line = &page.lines[index];
                        line.bbox[1] >= page.height * 0.10 && line.bbox[1] <= page.height * 0.75
                    })
                    .map(|&index| line_sizes[index])
                    .filter(|size| (4.0..=24.0).contains(size))
                    .collect(),
                0.75,
            );
            let body_size = if body_size > 0.0 { body_size } else { 10.0 };
            // A number set in the running head at the same place page after page is a folio, though
            // the head's title shares its row ("584  r. v. mabior  [2012] 2 S.C.R.").
            let attached_labels: HashSet<usize> = members
                .iter()
                .copied()
                .filter(|&label_index| {
                    let label = &page.lines[label_index];
                    !(repeated.contains(&(page_slot, label_index)) && label.bbox[1] < page.height * 0.12)
                        && standalone_note_label(label)
                        && members.iter().any(|&body_index| {
                            !repeated.contains(&(page_slot, body_index))
                                && aligned_note_body(label, &page.lines[body_index])
                        })
                })
                .collect();
            // A printed paragraph number or a provision's marker opening the body under a running
            // title sits at the same place on every page, like a folio; its text shares its row. A
            // heading's number ("1.", "IV.") holds its row's text whatever repeats there; a
            // paragraph's or provision's ("[12]", "(1)", "15(4)", "(b)", "12") holds words no
            // other page repeats there, as no folio beside its running title does.
            let attached_enumerators: HashSet<usize> = members
                .iter()
                .copied()
                .filter(|&label_index| {
                    let label = &page.lines[label_index];
                    let heading = heading_standalone_re().is_match(&label.text);
                    (heading || structure_marker(&label.text))
                        && members.iter().any(|&body_index| {
                            let body = &page.lines[body_index];
                            (heading || !repeated.contains(&(page_slot, body_index)) && body.text.chars().any(char::is_alphabetic))
                                && aligned_note_body(label, body)
                        })
                })
                .collect();
            // Repeated shortforms such as "Ibid." remain note bodies when a small
            // detached label shares their baseline. Repetition alone is not enough
            // to turn a source-backed legal citation into running furniture.
            let citation_note_lines: HashSet<usize> = members
                .iter()
                .filter_map(|&label_index| {
                    let label = &page.lines[label_index];
                    if !standalone_note_label(label)
                        || label.bbox[1] <= page.height * 0.86
                        || sequence_folios.contains(&(page_slot, label_index))
                    {
                        return None;
                    }
                    let body_index = aligned_note_body_among(&page.lines, members.iter().copied(), label_index)?;
                    let body = &page.lines[body_index];
                    (body.bbox[1] > page.height * 0.86
                        && line_sizes[body_index] < body_size * 0.90
                        && layout_analysis().has_citation_signal(&body.text))
                    .then_some([label_index, body_index])
                })
                .flatten()
                .collect();
            for (&slot, &index) in view.lines.iter().zip(&members) {
                // A transcript's line number is claimed already, though its digits repeat.
                if line_label(slot) {
                    continue;
                }
                let line = &page.lines[index];
                let line_size = line_sizes[index];
                let frame = frame_edge(rows[page_slot], index);
                let banded_top = line.bbox[1] < page.height * 0.12;
                let banded_bottom = line.bbox[3] > page.height * 0.90;
                let at_top = banded_top || frame == Some(Edge::Top);
                let at_bottom = banded_bottom || frame == Some(Edge::Bottom);
                let page_number_at_top = line.bbox[3] < page.height * 0.14;
                let edge = |top: bool| if top { Edge::Top } else { Edge::Bottom };
                let compact_note =
                    at_bottom && line_size < body_size * 0.90 && compact_note_line(&line.text);
                found[index] = if repeated.contains(&(page_slot, index))
                    && furniture_edge(line, page.width, page.height).is_some_and(|edge| edge >= 2)
                {
                    Some((Edge::Top, Rule::RunningHead))
                } else if sequence_folios.contains(&(page_slot, index)) {
                    Some((Edge::Bottom, Rule::Folio))
                } else if let Some(&top) = edge_folios.get(&(page_slot, index)) {
                    Some((edge(top), Rule::Folio))
                } else if repeated.contains(&(page_slot, index))
                    && (at_top || at_bottom)
                    && (line_size >= body_size * 0.75 || normalize_furniture(&line.text) != "#")
                    && !compact_note
                    && !attached_labels.contains(&index)
                    && !attached_enumerators.contains(&index)
                    && !citation_note_lines.contains(&index)
                {
                    Some((edge(at_top), if banded_top || banded_bottom { Rule::RunningHead } else { Rule::FrameRow }))
                } else if (at_top || at_bottom) && document_id_line(&line.text) {
                    Some((edge(at_top), Rule::DocumentId))
                } else if (line.bbox[3] < page.height * 0.14 || line.bbox[1] > page.height * 0.86)
                    && footer_page_number(&line.text)
                    && if page_number_at_top {
                        line.bbox[1] - top < (line.bbox[3] - line.bbox[1]) * 0.5
                    } else {
                        bottom - line.bbox[3] < (line.bbox[3] - line.bbox[1]) * 0.5
                    }
                    && line_size >= body_size * 0.75
                    && !attached_labels.contains(&index)
                    && !attached_enumerators.contains(&index)
                    && !citation_note_lines.contains(&index)
                {
                    Some((edge(page_number_at_top), Rule::PageNumber))
                } else {
                    None
                };
            }
            for (index, edge) in print_margins(view, &line_label) {
                found[index] = Some((edge, Rule::PrintMargin));
            }
            extend_running_heads(page, &members, found, &line_sizes, body_size, |index| line_label(view.slot(index)));
        }
    });
    // A division's heading the edge bands took for a running head: its own text, alone on its line,
    // read at the edges on no other page (its number changes from page to page, as a running head's
    // does not; a contents list may print it too), and on no row of furniture but other such
    // headings ("PART 50" beside "PARTIE 50").
    let mut printed: HashMap<&str, usize> = HashMap::new();
    for (view, found) in views.iter().zip(&found) {
        for (_, line) in view.iter().filter(|(slot, line)| found[view.place(*slot)].is_some() && division_marker(&line.text)) {
            *printed.entry(line.text.trim()).or_default() += 1;
        }
    }
    for (page_slot, view) in views.iter().enumerate() {
        let rows = rows[page_slot];
        let found = &mut found[page_slot];
        let heading: Vec<bool> = (0..found.len())
            .map(|index| {
                let text = &view.page.lines[index].text;
                found[index].is_some() && division_marker(text) && printed.get(text.trim()) == Some(&1)
            })
            .collect();
        for index in (0..found.len()).filter(|&index| heading[index]) {
            let row = rows.of[index];
            if !(0..found.len()).any(|other| found[other].is_some() && !heading[other] && rows.of[other] == row) {
                found[index] = None;
            }
        }
    }
    // A line on the row of a running head or a folio is the head's or foot's.
    for (page_slot, view) in views.iter().enumerate() {
        let rows = rows[page_slot];
        let found = &mut found[page_slot];
        let heads: Vec<(u32, Edge)> = found
            .iter()
            .enumerate()
            .filter_map(|(index, read)| {
                read.filter(|(_, rule)| matches!(rule, Rule::RunningHead | Rule::FrameRow | Rule::Folio | Rule::PageNumber))
                    .map(|(edge, _)| (rows.of[index], edge))
            })
            .filter(|(row, _)| *row != u32::MAX)
            .collect();
        for &slot in &view.lines {
            let index = view.place(slot);
            if found[index].is_some() || line_label(slot) {
                continue;
            }
            if let Some(&(_, edge)) = heads.iter().find(|(row, _)| *row == rows.of[index]) {
                found[index] = Some((edge, Rule::RowMate));
            }
        }
    }
    views
        .iter()
        .zip(found)
        .flat_map(|(view, found)| {
            view.lines.iter().filter_map(move |&slot| found[view.place(slot)].map(|(edge, rule)| (slot, edge, rule)))
        })
        .collect()
}

/// A running head can be a stack of small-type lines whose lower lines name the current
/// heading or sections and so change from page to page. A line in the top band set in a
/// running-head line's type, flush with it and directly above or below it, is part of it.
/// `members` are the page's lines read, `found` the furniture read of them by line slot.
pub(super) fn extend_running_heads(
    page: &Page,
    members: &[usize],
    found: &mut [Option<(Edge, Rule)>],
    line_sizes: &[f64],
    body_size: f64,
    line_label: impl Fn(usize) -> bool,
) {
    let top_band = page.height * 0.12;
    let tolerance = page.width * FURNITURE_X_TOLERANCE_FRAC;
    let header = |found: &[Option<(Edge, Rule)>], index: usize| found[index].is_some_and(|(edge, _)| edge == Edge::Top);
    loop {
        let joined: Vec<usize> = members
            .iter()
            .copied()
            .filter(|&index| {
                let line = &page.lines[index];
                let size = line_sizes[index];
                !header(found, index)
                    && !line_label(index)
                    && line.bbox[3] < top_band
                    && size > 0.0
                    && size < body_size * 0.90
                    && members.iter().any(|&head_index| {
                        let head = &page.lines[head_index];
                        let head_size = line_sizes[head_index];
                        let height = (head.bbox[3] - head.bbox[1]).max(line.bbox[3] - line.bbox[1]);
                        header(found, head_index)
                            && head.bbox[3] < top_band
                            && (head_size - size).abs() <= 0.5
                            && ((head.bbox[0] - line.bbox[0]).abs() <= tolerance
                                || (line_center_x(head) - line_center_x(line)).abs() <= tolerance)
                            && (line.bbox[1] - head.bbox[3]).max(head.bbox[1] - line.bbox[3])
                                <= height * 0.5
                    })
            })
            .collect();
        if joined.is_empty() {
            return;
        }
        for index in joined {
            found[index] = Some((Edge::Top, Rule::RunningHead));
        }
    }
}

pub(super) fn printed_label(value: &str) -> Option<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)^(?:(?:page\s+)?(\d{1,6}|[ivxlcdm]{1,12})(?:\s*(?:of|/|of\s*/\s*de|de|sur)\s*\d{1,6})?|-\s*(\d{1,6})\s*-)$").unwrap()
    })
    .captures(value.trim())
    .and_then(|capture| capture.get(1).or_else(|| capture.get(2)))
    .map(|value| value.as_str().to_owned())
}

/// A browser or print driver sets the page's address and the time it was printed in the
/// outer margins, with the page title and "1 of 3" on the same rows. They are furniture on
/// a single page too, where nothing repeats to show it. Each such line of the view, by line slot
/// on the page, with its edge; a line `line_label` holds is none.
pub(super) fn print_margins(view: &PageView, line_label: impl Fn(Slot) -> bool) -> Vec<(usize, Edge)> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let page = view.page;
    let stamp = RE.get_or_init(|| {
        Regex::new(
            r"(?i)(?:^|\s)(?:https?://|www\.)\S+|(?:^|\s)\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4},?\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:[ap]\.?\s?m\.?)?",
        )
        .unwrap()
    });
    let band = page.height * 0.06;
    let edge = |line: &Line| {
        if line.bbox[1] <= band {
            Some(Edge::Top)
        } else if line.bbox[3] >= page.height - band {
            Some(Edge::Bottom)
        } else {
            None
        }
    };
    let rows: Vec<(Edge, f64, f64)> = view
        .iter()
        .filter(|(_, line)| stamp.is_match(line.text.trim()))
        .filter_map(|(_, line)| edge(line).map(|kind| (kind, line.bbox[1], line.bbox[3])))
        .collect();
    if rows.is_empty() {
        return Vec::new();
    }
    view.iter()
        .filter(|(slot, _)| !line_label(*slot))
        .filter_map(|(slot, line)| {
            let kind = edge(line)?;
            let height = (line.bbox[3] - line.bbox[1]).max(1.0);
            rows.iter()
                .any(|(row, top, bottom)| *row == kind && line.bbox[3].min(*bottom) - line.bbox[1].max(*top) > height * 0.5)
                .then_some((view.place(slot), kind))
        })
        .collect()
}

/// A document management identifier a firm prints in the margin ("MT DOCS 20632860v5",
/// "33312.140943.JMS3.15349802.2", "43882055_21NATDOCS"): digits with at most capitals, dots,
/// underscores and a version suffix, and no words.
pub(super) fn document_id_line(value: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    let text = value.trim();
    let digits = text.chars().filter(char::is_ascii_digit).count();
    let visible = text.chars().filter(|c| !c.is_whitespace()).count();
    digits >= 6
        && digits * 5 >= visible * 2
        && text.chars().count() <= 40
        && RE
            .get_or_init(|| Regex::new(r"^[A-Z0-9][A-Z0-9 ._/-]*(?:v\d{1,3})?$").unwrap())
            .is_match(text)
}

pub(super) fn footer_page_number(value: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)^\s*-?\s*(?:page\s+)?[ivxlcdm\d]+(?:\s*(?:of|/|of\s*/\s*de|de|sur)\s*\d+)?\s*-?\s*$")
            .unwrap()
    })
        .is_match(value.trim())
}

/// The printed labels a page's furniture (`furniture`: line slots on the page, each with its edge)
/// prints, each with the slot of the line printing it: the folios its running head and foot print
/// (`printed_label`), the foot's first, then in reading order.
pub(super) fn page_labels_among(page: &Page, furniture: impl Iterator<Item = (usize, Edge)>) -> Vec<(String, usize)> {
    let mut candidates: Vec<(String, usize, bool)> = furniture
        .filter_map(|(index, edge)| Some((printed_label(&page.lines[index].text)?, index, edge == Edge::Bottom)))
        .collect();
    candidates.sort_by_key(|&(_, index, foot)| (!foot, page.lines[index].reading_order));
    candidates.into_iter().map(|(label, index, _)| (label, index)).collect()
}

/// A paragraph's or provision's marker alone on its line: "[12]", "(1)", "(b)", "(iv)", "3.",
/// "1.2.3", "12", "15(4)", "29(2)(a)".
fn structure_marker(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    standalone_paragraph_number(text)
        || crate::paragraphs::marker_only_re().is_match(text)
        || RE
            .get_or_init(|| Regex::new(r"^\s*\d{1,4}[A-Za-z]?(?:\.\d{1,3})*(?:\([0-9A-Za-z]{1,4}\))*\.?\s*$").unwrap())
            .is_match(text)
}
