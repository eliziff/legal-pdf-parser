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

/// Recognition reads a running head a little differently page to page ("of the united
/// states.", "of the united sta#es."). A reading joins the commonest repeated reading of
/// the same edge within a fifth of its characters; position alignment then decides.
pub(super) fn join_recognized_readings(
    candidates: &mut HashMap<(u8, String), Vec<FurnitureHit>>,
    recognized: &HashSet<(u8, String)>,
) {
    fn distance(a: &[char], b: &[char]) -> usize {
        let mut row: Vec<usize> = (0..=b.len()).collect();
        for (i, x) in a.iter().enumerate() {
            let mut diagonal = row[0];
            row[0] = i + 1;
            for (j, y) in b.iter().enumerate() {
                let above = row[j + 1];
                row[j + 1] = (diagonal + usize::from(x != y)).min(row[j] + 1).min(above + 1);
                diagonal = above;
            }
        }
        row[b.len()]
    }
    let mut keys: Vec<&(u8, String)> = recognized.iter().collect();
    keys.sort_by(|a, b| candidates[*b].len().cmp(&candidates[*a].len()).then(a.cmp(b)));
    let mut anchors: Vec<((u8, String), Vec<char>)> = Vec::new();
    let mut moves = Vec::new();
    for key in keys {
        let chars: Vec<char> = key.1.chars().collect();
        let limit = chars.len() / 5;
        let anchor = (chars.len() >= 6)
            .then(|| {
                anchors.iter().find(|(anchor, text)| {
                    anchor.0 == key.0
                        && text.len().abs_diff(chars.len()) <= limit
                        && distance(text, &chars) <= limit
                })
            })
            .flatten();
        match anchor {
            Some((anchor, _)) => moves.push((key.clone(), anchor.clone())),
            None if candidates[key].len() >= 2 => anchors.push((key.clone(), chars)),
            None => {}
        }
    }
    for (from, to) in moves {
        if let Some(hits) = candidates.remove(&from) {
            candidates.entry(to).or_default().extend(hits);
        }
    }
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
/// bare number that is its page's lowest line (or highest) is a folio when the number less the
/// page's place in the document is the same on at least three such pages (or half of a shorter
/// document's). Returns each folio with whether it heads its page.
pub(super) fn edge_folios(pages: &[Page]) -> HashMap<(usize, usize), bool> {
    let mut by_offset = HashMap::<i64, Vec<(usize, usize, bool)>>::new();
    for (page_slot, page) in pages.iter().enumerate() {
        let lines = || page.lines.iter().enumerate().filter(|(_, line)| !line.text.trim().is_empty());
        let lowest = lines().max_by(|a, b| a.1.bbox[1].total_cmp(&b.1.bbox[1]));
        let highest = lines().min_by(|a, b| a.1.bbox[1].total_cmp(&b.1.bbox[1]));
        for (edge, top) in [(lowest, false), (highest, true)] {
            let Some((line_slot, line)) = edge else {
                continue;
            };
            if let Some(printed) = arabic_page_number(&line.text) {
                by_offset
                    .entry(i64::from(printed) - i64::from(page.number))
                    .or_default()
                    .push((page_slot, line_slot, top));
            }
        }
    }
    let needed = 3.min(pages.len().div_ceil(2)).max(2);
    by_offset
        .into_values()
        .filter(|folios| folios.iter().map(|folio| folio.0).collect::<HashSet<_>>().len() >= needed)
        .flatten()
        .map(|(page_slot, line_slot, top)| ((page_slot, line_slot), top))
        .collect()
}

/// Text-Fidelity's alternating-folio witness. A singleton Arabic number at
/// each outer bottom edge is furniture only when at least four consecutive
/// PDF pages carry a +1 printed sequence, alternate sides, and remain aligned
/// within each page parity. A number centred at the bottom is furniture on the
/// same sequence without alternating, whatever its type: a statute sets its
/// folio as small as its notes.
pub(super) fn alternating_folios(pages: &[Page]) -> HashSet<(usize, usize)> {
    let mut singletons = Vec::new();
    for (page_slot, page) in pages.iter().enumerate() {
        if page.width <= 0.0 || page.height <= 0.0 {
            continue;
        }
        let mut candidates = page
            .lines
            .iter()
            .enumerate()
            .filter_map(|(line_slot, line)| {
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
                (line.bbox[1] > page.height * FOLIO_BOTTOM_MIN_FRAC).then_some(FolioCandidate {
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
/// no marker or numbered row): each (page slot, line slot).
pub(super) fn edge_furniture(pages: &[Page]) -> HashSet<(usize, usize)> {
    use pdf_inspector::types::TextLine;
    let mut lines: Vec<TextLine> = Vec::new();
    for (page_slot, page) in pages.iter().enumerate() {
        let (items, origin) = legal_pdf_tables::line_items(&page.lines, page_slot as u32 + 1, page.height);
        for (mut item, (slot, _)) in items.into_iter().zip(origin) {
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
        pdf_inspector::markdown::strip_header_footer_lines(lines, pages.len() as u32).iter().map(key).collect();
    offered.difference(&kept).copied().collect()
}

pub(super) fn mark_repeated_furniture(pages: &mut [Page]) {
    let mut candidates: HashMap<(u8, String), Vec<FurnitureHit>> = HashMap::new();
    let mut recognized = HashSet::new();
    legal_pdf_support::profile::measure("furniture.candidates", || {
        for (page_slot, page) in pages.iter().enumerate() {
            for (line_slot, line) in page.lines.iter().enumerate() {
                if page.width <= 0.0 || page.height <= 0.0 {
                    continue;
                }
                let Some(top) = furniture_edge(line, page.width, page.height) else {
                    continue;
                };
                let normalized = normalize_furniture(&line.text);
                if !normalized.is_empty() {
                    if line.source == "ocr" {
                        recognized.insert((top, normalized.clone()));
                    }
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
    join_recognized_readings(&mut candidates, &recognized);
    // Preserve the old engine's two-page/parity behavior, but cap the witness
    // at four pages as Text-Fidelity does once alignment is also required.
    let minimum = 4_usize.min(2_usize.max(((pages.len() as f64) * 0.35).ceil() as usize));
    // A cover-page title can repeat the running title away from the header
    // baseline. Admit the stable occurrence cluster, not every matching string.
    let repeated: HashSet<(usize, usize)> =
        legal_pdf_support::profile::measure("furniture.alignment", || {
            candidates
                .into_values()
                .flat_map(|hits| aligned_furniture(&hits, minimum))
                .collect()
        });
    // A document too short for repetition shows its furniture at its pages' edges.
    let repeated = if pages.len() < 3 {
        repeated.into_iter().chain(edge_furniture(pages)).collect()
    } else {
        repeated
    };
    let sequence_folios =
        legal_pdf_support::profile::measure("furniture.folios", || alternating_folios(pages));
    let edge_folios = edge_folios(pages);
    legal_pdf_support::profile::measure("furniture.apply", || {
        for (page_slot, page) in pages.iter_mut().enumerate() {
            let line_sizes: Vec<_> = page.lines.iter().map(line_font_size).collect();
            // Without a sequence witness, a folio must sit beyond the remaining body text.
            let (top, bottom) = page
                .lines
                .iter()
                .enumerate()
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
                page.lines
                    .iter()
                    .enumerate()
                    .filter(|(_, line)| {
                        line.bbox[1] >= page.height * 0.10 && line.bbox[1] <= page.height * 0.75
                    })
                    .map(|(index, _)| line_sizes[index])
                    .filter(|size| (4.0..=24.0).contains(size))
                    .collect(),
                0.75,
            );
            let body_size = if body_size > 0.0 { body_size } else { 10.0 };
            // A number set in the running head at the same place page after page is a folio, though
            // the head's title shares its row ("584  r. v. mabior  [2012] 2 S.C.R.").
            let attached_labels: HashSet<usize> = page
                .lines
                .iter()
                .enumerate()
                .filter(|(label_index, label)| {
                    !(repeated.contains(&(page_slot, *label_index)) && label.bbox[1] < page.height * 0.12)
                        && standalone_note_label(label)
                        && page.lines.iter().enumerate().any(|(body_index, body)| {
                            !repeated.contains(&(page_slot, body_index))
                                && aligned_note_body(label, body)
                        })
                })
                .map(|(index, _)| index)
                .collect();
            // A printed paragraph number opening the body under a running title sits
            // at the same place on every page, like a folio; its text shares its row.
            let attached_enumerators: HashSet<usize> = page
                .lines
                .iter()
                .enumerate()
                .filter(|(_, label)| {
                    let paragraph = standalone_paragraph_number(&label.text);
                    (heading_standalone_re().is_match(&label.text) || paragraph)
                        && page.lines.iter().enumerate().any(|(body_index, body)| {
                            (!paragraph || !repeated.contains(&(page_slot, body_index)))
                                && aligned_note_body(label, body)
                        })
                })
                .map(|(index, _)| index)
                .collect();
            // Repeated shortforms such as "Ibid." remain note bodies when a small
            // detached label shares their baseline. Repetition alone is not enough
            // to turn a source-backed legal citation into running furniture.
            let citation_note_lines: HashSet<usize> = page
                .lines
                .iter()
                .enumerate()
                .filter_map(|(label_index, label)| {
                    if !standalone_note_label(label)
                        || label.bbox[1] <= page.height * 0.86
                        || sequence_folios.contains(&(page_slot, label_index))
                    {
                        return None;
                    }
                    let body_index = aligned_note_body_index(&page.lines, label_index)?;
                    let body = &page.lines[body_index];
                    (body.bbox[1] > page.height * 0.86
                        && line_sizes[body_index] < body_size * 0.90
                        && layout_analysis().has_citation_signal(&body.text))
                    .then_some([label_index, body_index])
                })
                .flatten()
                .collect();
            for (index, line) in page.lines.iter_mut().enumerate() {
                // A transcript's line number is claimed already, though its digits repeat.
                if line.region_type == "line_label" {
                    continue;
                }
                let line_size = line_sizes[index];
                let at_top = line.bbox[1] < page.height * 0.12;
                let at_bottom = line.bbox[3] > page.height * 0.90;
                let page_number_at_top = line.bbox[3] < page.height * 0.14;
                let compact_note =
                    at_bottom && line_size < body_size * 0.90 && compact_note_line(&line.text);
                if repeated.contains(&(page_slot, index))
                    && furniture_edge(line, page.width, page.height).is_some_and(|edge| edge >= 2)
                {
                    line.region_type = "header".to_owned();
                } else if sequence_folios.contains(&(page_slot, index)) {
                    line.region_type = "footer".to_owned();
                } else if let Some(&top) = edge_folios.get(&(page_slot, index)) {
                    line.region_type = if top { "header" } else { "footer" }.to_owned();
                } else if repeated.contains(&(page_slot, index))
                    && (at_top || at_bottom)
                    && (line_size >= body_size * 0.75 || normalize_furniture(&line.text) != "#")
                    && !compact_note
                    && !attached_labels.contains(&index)
                    && !attached_enumerators.contains(&index)
                    && !citation_note_lines.contains(&index)
                {
                    line.region_type = if at_top { "header" } else { "footer" }.to_owned();
                } else if (at_top || at_bottom) && document_id_line(&line.text) {
                    line.region_type = if at_top { "header" } else { "footer" }.to_owned();
                } else if (line.bbox[3] < page.height * 0.14 || line.bbox[1] > page.height * 0.86)
                    && footer_page_number(&line.text)
                    && if page_number_at_top {
                        line.bbox[1] - top < (line.bbox[3] - line.bbox[1]) * 0.5
                    } else {
                        bottom - line.bbox[3] < (line.bbox[3] - line.bbox[1]) * 0.5
                    }
                    && line_size >= body_size * 0.75
                    && !attached_labels.contains(&index)
                    && !citation_note_lines.contains(&index)
                {
                    line.region_type = if page_number_at_top {
                        "header"
                    } else {
                        "footer"
                    }
                    .to_owned();
                }
            }
            mark_print_margins(page);
            extend_running_heads(page, &line_sizes, body_size);
        }
    });
}

/// A running head can be a stack of small-type lines whose lower lines name the current
/// heading or sections and so change from page to page. A line in the top band set in a
/// running-head line's type, flush with it and directly above or below it, is part of it.
pub(super) fn extend_running_heads(page: &mut Page, line_sizes: &[f64], body_size: f64) {
    let top_band = page.height * 0.12;
    let tolerance = page.width * FURNITURE_X_TOLERANCE_FRAC;
    loop {
        let joined: Vec<usize> = (0..page.lines.len())
            .filter(|index| {
                let line = &page.lines[*index];
                let size = line_sizes[*index];
                !matches!(line.region_type.as_str(), "header" | "line_label")
                    && line.bbox[3] < top_band
                    && size > 0.0
                    && size < body_size * 0.90
                    && page.lines.iter().zip(line_sizes).any(|(head, head_size)| {
                        let height = (head.bbox[3] - head.bbox[1]).max(line.bbox[3] - line.bbox[1]);
                        head.region_type == "header"
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
            page.lines[index].region_type = "header".to_owned();
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
/// a single page too, where nothing repeats to show it.
pub(super) fn mark_print_margins(page: &mut Page) {
    static RE: OnceLock<Regex> = OnceLock::new();
    let stamp = RE.get_or_init(|| {
        Regex::new(
            r"(?i)(?:^|\s)(?:https?://|www\.)\S+|(?:^|\s)\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4},?\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:[ap]\.?\s?m\.?)?",
        )
        .unwrap()
    });
    let band = page.height * 0.06;
    let edge = |line: &Line| {
        if line.bbox[1] <= band {
            Some("header")
        } else if line.bbox[3] >= page.height - band {
            Some("footer")
        } else {
            None
        }
    };
    let rows: Vec<(&'static str, f64, f64)> = page
        .lines
        .iter()
        .filter(|line| stamp.is_match(line.text.trim()))
        .filter_map(|line| edge(line).map(|kind| (kind, line.bbox[1], line.bbox[3])))
        .collect();
    for line in page.lines.iter_mut().filter(|line| line.region_type != "line_label") {
        let Some(kind) = edge(line) else { continue };
        let height = (line.bbox[3] - line.bbox[1]).max(1.0);
        if rows.iter().any(|(row, top, bottom)| {
            *row == kind && line.bbox[3].min(*bottom) - line.bbox[1].max(*top) > height * 0.5
        }) {
            line.region_type = kind.to_owned();
        }
    }
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

/// A page's printed label and the slot of the line printing it: the folio its running head or foot
/// prints (printed_label), the foot's first; none where it prints none. Where its head and foot
/// print different labels, every candidate.
pub(super) fn page_label(page: &Page) -> std::result::Result<Option<(String, usize)>, Vec<(String, usize)>> {
    let candidates: Vec<(String, usize)> = page
        .lines
        .iter()
        .enumerate()
        .filter(|(_, line)| matches!(line.region_type.as_str(), "header" | "footer"))
        .filter_map(|(index, line)| printed_label(&line.text).map(|label| (label, index)))
        .collect();
    let labels: BTreeSet<String> = candidates.iter().map(|(label, _)| label.to_lowercase()).collect();
    if labels.len() > 1 {
        return Err(candidates);
    }
    Ok(candidates.into_iter().min_by_key(|(_, index)| {
        (page.lines[*index].region_type != "footer", page.lines[*index].reading_order)
    }))
}

pub(super) fn assign_printed_page_labels(pages: &mut [Page]) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    for page in pages {
        page.printed_label = None;
        page.printed_label_source = None;
        page.printed_label_line_id = None;
        let chosen = match page_label(page) {
            Ok(chosen) => chosen,
            Err(candidates) => {
                let mut diagnostic = Diagnostic::info(
                    "PRINTED_PAGE_LABEL_AMBIGUOUS",
                    "Conflicting header/footer page labels were left unresolved.",
                    Some(page.index),
                );
                diagnostic.line_ids = candidates
                    .iter()
                    .map(|(_, index)| page.lines[*index].id.clone())
                    .collect();
                diagnostic.details.insert(
                    "candidates".to_owned(),
                    json!(candidates
                        .iter()
                        .map(|(label, _)| label)
                        .collect::<Vec<_>>()),
                );
                diagnostics.push(diagnostic);
                continue;
            }
        };
        if let Some((label, index)) = chosen {
            let line = &page.lines[index];
            page.printed_label = Some(label);
            page.printed_label_source = Some(line.region_type.clone());
            page.printed_label_line_id = Some(line.id.clone());
        }
    }
    diagnostics
}
