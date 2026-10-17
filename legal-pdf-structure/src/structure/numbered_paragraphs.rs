//! Reading a PDF's print: the paragraphs a printed paragraph number addresses, in the body or
//! in the margin.

use std::collections::HashSet;

// Printed paragraph numbers are addresses, not structural ordinal positions.
// Split on the actual native lines: a prose node can contain several numbered
// paragraphs, and one printed paragraph can contain several prose nodes.
// Parallel-language columns print one paragraph number per column. A line
// belongs to the column holding its centre, which mirrored margins preserve.
fn in_column(column: [f64; 4], line: [f64; 4]) -> bool {
    let centre = (line[0] + line[2]) / 2.0;
    centre > column[0] && centre < column[2]
}

type PrintedLine<'a> = (&'a str, &'a str, u32, [f64; 4]);

/// A heading's text: a title, or a short line in sentence case that cites nothing and ends
/// without a sentence's punctuation ("Ministerial delegate's testimony").
fn heading_text(text: &str) -> bool {
    let text = text.trim();
    legal_pdf_support::heading_text_plausible(text)
        || (text.starts_with(|c: char| c.is_alphabetic() && c.is_uppercase())
            && !text.ends_with(['.', ',', ';', ':'])
            && text.chars().count() <= 100
            && !text
                .rsplit(char::is_whitespace)
                .next()
                .is_some_and(|word| word.chars().all(|c| c.is_ascii_digit()))
            && legal_pdf_core::structure_analysis()
                .citations(text)
                .is_empty())
}

/// A paragraph's own lines, up to the next heading of the outline below its first
/// ("Circumstances of the offence").
fn own<'l, 'a>(
    run: impl Iterator<Item = &'l PrintedLine<'a>>,
    headings: &HashSet<String>,
) -> Vec<&'l PrintedLine<'a>> {
    run.enumerate()
        .take_while(|(index, (id, ..))| *index == 0 || !headings.contains(*id))
        .map(|(_, line)| line)
        .collect()
}

/// `headings`: the lines of the document's outline headings. A paragraph ends where the next
/// heading or the next numbered paragraph begins.
fn printed_paragraph_plan<'a>(
    lines: &[(&'a str, &'a str, u32, [f64; 4])],
    paragraphs: &[Vec<String>],
    headings: &HashSet<String>,
    locator: &str,
) -> Option<(legal_pdf_support::PdfLookupStatus, HashSet<&'a str>)> {
    use legal_pdf_support::PdfLookupStatus as Status;
    let labels = lines
        .iter()
        .enumerate()
        .filter_map(|(index, (_, text, ..))| Some((bracket_label(text)?, index)))
        .collect::<Vec<_>>();
    if labels.is_empty() {
        return None;
    }
    let range = legal_pdf_support::numeric_range("paragraph", locator)
        .or_else(|| legal_pdf_support::parse_ordinal("paragraph", locator).map(|n| (n, n)));
    let Some((from, to)) = range.filter(|(a, b)| a <= b && b - a < 100) else {
        return Some((Status::Invalid, HashSet::new()));
    };
    let mut selected = HashSet::new();
    for number in from..=to {
        let hits = labels
            .iter()
            .filter(|(label, _)| *label == number)
            .map(|(_, index)| *index)
            .collect::<Vec<_>>();
        // The same number printed once per column is one address, not two
        // paragraphs; a repeat in the same column or on another page is not.
        if hits.is_empty()
            || hits.iter().any(|&hit| {
                hits.iter().any(|&other| {
                    other != hit
                        && (lines[other].2 != lines[hit].2
                            || in_column(lines[hit].3, lines[other].3))
                })
            })
        {
            return Some((
                if hits.is_empty() {
                    Status::NotFound
                } else {
                    Status::Ambiguous
                },
                HashSet::new(),
            ));
        }
        for start in hits {
            // A detached number and the first line of its text share a row;
            // the column is both. A parallel translation never shares a row.
            let mut column = lines[start].3;
            if let Some(next) = lines.get(start + 1).filter(|next| {
                next.2 == lines[start].2 && next.3[1] < column[3] && column[1] < next.3[3]
            }) {
                column = [
                    column[0].min(next.3[0]),
                    column[1],
                    column[2].max(next.3[2]),
                    column[3],
                ];
            }
            if let Some((_, end)) = labels
                .iter()
                .find(|(_, index)| *index > start && in_column(column, lines[*index].3))
            {
                let mut run = own(
                    lines[start..*end]
                        .iter()
                        .filter(|line| in_column(column, line.3)),
                    headings,
                );
                // A heading the outline does not name ("Issue") stands set apart just above the next
                // paragraph, its text a heading's: it opens what follows, not this paragraph.
                let next = &lines[*end];
                let gap = |above: &[f64; 4], below: &[f64; 4]| {
                    below[1] - above[3] > 0.5 * (below[3] - below[1]).min(above[3] - above[1])
                };
                // Headings may stand one above another ("Submissions of the parties", "Submissions of
                // the appellant"); a heading of several lines has no gap within it.
                let (mut at, mut cut) = (run.len(), run.len());
                while at > 1
                    && run[at - 1].2 == next.2
                    && heading_text(run[at - 1].1)
                    && (at < run.len() || gap(&run[at - 1].3, &next.3))
                {
                    at -= 1;
                    if gap(&run[at - 1].3, &run[at].3) {
                        cut = at;
                    }
                }
                run.truncate(cut);
                selected.extend(run.iter().map(|(id, ..)| *id));
            } else {
                // At EOF use the native owner, not unbounded end matter.
                let owner = paragraphs
                    .iter()
                    .find(|ids| ids.iter().any(|id| id == lines[start].0));
                selected.extend(
                    own(
                        lines[start..].iter().filter(|(id, ..)| {
                            owner.is_some_and(|ids| ids.iter().any(|value| value == id))
                        }),
                        headings,
                    )
                    .iter()
                    .map(|(id, ..)| *id),
                );
            }
        }
    }
    Some((Status::Found, selected))
}

// Detached paragraph labels sit outside the body, unlike reporter page
// numbers. Their aligned body rows bound both columns in parallel text.
// Classification supplies the eligible body lines; note and furniture lines never enter here.
// A heading set out to the margin ("Circumstances of the offence") is no part of the body the
// numbers stand beside, and a paragraph ends where the next heading begins.
fn marginal_paragraph_plan<'a>(
    pages: &'a [legal_pdf_support::PdfTextPage],
    headings: &HashSet<String>,
    locator: &str,
) -> Option<(legal_pdf_support::PdfLookupStatus, HashSet<&'a str>)> {
    use legal_pdf_support::PdfLookupStatus as Status;
    let (body_bounds, labels) = margin_labels(pages, headings);
    // One isolated margin number could be a note. Require a numbering run.
    if !numbering_run(&labels) {
        return None;
    }
    let range = legal_pdf_support::numeric_range("paragraph", locator)
        .or_else(|| legal_pdf_support::parse_ordinal("paragraph", locator).map(|n| (n, n)));
    let Some((from, to)) = range.filter(|(a, b)| a <= b && b - a < 100) else {
        return Some((Status::Invalid, HashSet::new()));
    };
    let mut selected = HashSet::new();
    for number in from..=to {
        let hits = labels
            .iter()
            .enumerate()
            .filter(|(_, label)| label.0 == number)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if hits.len() != 1 {
            return Some((
                if hits.is_empty() {
                    Status::NotFound
                } else {
                    Status::Ambiguous
                },
                HashSet::new(),
            ));
        }
        let (_, start_page, start_y) = labels[hits[0]];
        // The last paragraph ends with its page's body; its end is otherwise unwitnessed, and no end
        // matter or numbering restart is swept in.
        let last = hits[0] + 1 == labels.len();
        let Some((end_page, end_y)) = labels
            .get(hits[0] + 1)
            .filter(|next| next.0 == number + 1)
            .map(|next| (next.1, next.2))
            .or_else(|| {
                last.then(|| {
                    body_bounds[start_page].map(|bounds| {
                        (
                            start_page,
                            (bounds[3] + 1.0).min(pages[start_page].height * 0.95),
                        )
                    })
                })
                .flatten()
            })
        else {
            return Some((Status::Unavailable, HashSet::new()));
        };
        // The first heading below the paragraph's first line, before the next paragraph.
        let stop = (start_page..=end_page).find_map(|page_index| {
            pages[page_index]
                .lines
                .iter()
                .filter(|line| {
                    headings.contains(&line.id)
                        && (page_index > start_page || line.rect[1] > start_y + 0.5)
                        && (page_index < end_page || line.rect[1] < end_y - 0.5)
                })
                .map(|line| line.rect[1])
                .min_by(f64::total_cmp)
                .map(|y| (page_index, y))
        });
        let end_page = stop.map_or(end_page, |(page, _)| page);
        for page_index in start_page..=end_page {
            let Some(bounds) = body_bounds[page_index] else {
                return Some((Status::Unavailable, HashSet::new()));
            };
            let top = if page_index == start_page {
                start_y - 0.5
            } else {
                bounds[1]
            };
            let bottom = match stop {
                Some((page, y)) if page == page_index => y - 0.5,
                _ if page_index == end_page => end_y - 0.5,
                _ => bounds[3],
            };
            let lines = pages[page_index]
                .lines
                .iter()
                .filter(|line| {
                    line.rect[0] >= bounds[0] - 0.5
                        && line.rect[2] <= bounds[2] + 0.5
                        && line.rect[1] >= top
                        && line.rect[1] < bottom
                })
                .collect::<Vec<_>>();
            // A separated heading immediately before the next numbered
            // paragraph belongs to that following section, not this passage.
            let heading_top = (page_index == end_page)
                .then(|| {
                    lines
                        .iter()
                        .filter(|line| {
                            let Some((prefix, text)) = line.text.trim().split_once(". ") else {
                                return false;
                            };
                            if legal_pdf_support::enumerator_interpretations(prefix, ".").is_empty()
                                || !legal_pdf_support::heading_text_plausible(text)
                            {
                                return false;
                            }
                            let height = line.rect[3] - line.rect[1];
                            let prior_bottom = lines
                                .iter()
                                .filter(|prior| prior.rect[3] < line.rect[1])
                                .map(|prior| prior.rect[3])
                                .max_by(f64::total_cmp);
                            prior_bottom.is_some_and(|y| line.rect[1] - y > height * 0.5)
                                && end_y - line.rect[3] > height * 0.5
                        })
                        .map(|line| line.rect[1])
                        .min_by(f64::total_cmp)
                })
                .flatten();
            selected.extend(
                lines
                    .iter()
                    .filter(|line| heading_top.is_none_or(|y| line.rect[1] < y - 0.5))
                    .map(|line| line.id.as_str()),
            );
        }
    }
    Some((Status::Found, selected))
}

/// Each page's body bounds, and the paragraph numbers standing in the margin or set apart at the
/// start of a body line, in reading order: each number, its page index and the top of its row.
#[allow(clippy::type_complexity)]
fn margin_labels(
    pages: &[legal_pdf_support::PdfTextPage],
    headings: &HashSet<String>,
) -> (Vec<Option<[f64; 4]>>, Vec<(usize, usize, f64)>) {
    let body_bounds = pages
        .iter()
        .map(|page| {
            page.lines
                .iter()
                .filter(|line| {
                    line.rect[2] - line.rect[0] > page.width * 0.20
                        && !headings.contains(&line.id)
                        && line.text.chars().any(char::is_alphabetic)
                })
                .map(|line| line.rect)
                .reduce(|a, b| {
                    [
                        a[0].min(b[0]),
                        a[1].min(b[1]),
                        a[2].max(b[2]),
                        a[3].max(b[3]),
                    ]
                })
        })
        .collect::<Vec<_>>();
    let fixed = fixed_number_gaps(pages);
    let mut labels = Vec::new();
    for (page_index, page) in pages.iter().enumerate() {
        let Some(bounds) = body_bounds[page_index] else {
            continue;
        };
        for line in &page.lines {
            if let Some(number) = set_apart_number(page, line, &fixed) {
                labels.push((number, page_index, line.rect[1]));
                continue;
            }
            let text = line.text.trim();
            let text = text
                .strip_prefix('[')
                .and_then(|s| s.strip_suffix(']'))
                .unwrap_or(text);
            if text.is_empty() || text.len() > 5 || !text.bytes().all(|b| b.is_ascii_digit()) {
                continue;
            }
            let height = line.rect[3] - line.rect[1];
            let gap = (bounds[0] - line.rect[2]).max(line.rect[0] - bounds[2]);
            if gap < height * 0.5 || gap > height * 8.0 {
                continue;
            }
            let row = page
                .lines
                .iter()
                .filter(|peer| {
                    peer.rect[2] - peer.rect[0] > page.width * 0.20
                        && peer.text.chars().any(char::is_alphabetic)
                        && (peer.rect[1] - line.rect[1]).abs() <= height * 0.6
                })
                .map(|peer| peer.rect[1])
                .min_by(f64::total_cmp);
            if let (Some(y), Ok(number)) = (row, text.parse::<usize>()) {
                labels.push((number, page_index, y));
            }
        }
    }
    labels.sort_by(|a, b| a.1.cmp(&b.1).then(a.2.total_cmp(&b.2)));
    (body_bounds, labels)
}

/// Three numbers in a row: a numbering run, not an isolated number that could be a note's.
fn numbering_run(labels: &[(usize, usize, f64)]) -> bool {
    labels
        .windows(3)
        .any(|w| w[0].0 + 1 == w[1].0 && w[1].0 + 1 == w[2].0)
}

/// "[12]" opening a line, before its words or alone: a printed paragraph number in the body.
fn bracket_label(text: &str) -> Option<usize> {
    let (number, rest) = text.trim_start().strip_prefix('[')?.split_once(']')?;
    if (!rest.is_empty() && !rest.starts_with(char::is_whitespace))
        || number.is_empty()
        || !number.chars().all(|c| c.is_ascii_digit())
    {
        return None;
    }
    number.parse().ok()
}

/// Enumerate the printed addresses using the same selectors for body and margin numbers.
fn printed_paragraphs<'a>(
    pages: &'a [legal_pdf_support::PdfTextPage],
    lines: &[PrintedLine<'a>],
    paragraphs: &[Vec<String>],
    headings: &HashSet<String>,
) -> Vec<(usize, legal_pdf_support::PdfLookupStatus, HashSet<&'a str>)> {
    let (_, margin) = margin_labels(pages, headings);
    let mut numbers = lines
        .iter()
        .filter_map(|(_, text, ..)| bracket_label(text))
        .chain(margin.iter().map(|label| label.0))
        .collect::<Vec<_>>();
    numbers.sort_unstable();
    numbers.dedup();
    numbers
        .into_iter()
        .filter_map(|number| {
            let locator = number.to_string();
            let (status, lines) = marginal_paragraph_plan(pages, headings, &locator)
                .or_else(|| printed_paragraph_plan(lines, paragraphs, headings, &locator))?;
            Some((number, status, lines))
        })
        .collect()
}

/// A number that opens a line of the body set apart from the text after it by more than a space
/// ("1      At the conclusion of the hearing", as Westlaw prints a judgment's paragraphs), by the
/// fixed separation a numbering run keeps (`fixed`), or by more than that and more than the line's
/// own spaces (Lexis's first paragraph, "1   When Robert King"): the paragraph's printed number,
/// standing bare. A number in running text ("11 of the Act") is followed by the line's ordinary
/// space; a page's running head and foot are no body.
fn set_apart_number(
    page: &legal_pdf_support::PdfTextPage,
    line: &legal_pdf_support::PdfTextLine,
    fixed: &[f64],
) -> Option<usize> {
    let (number, gap) = opening_number(page, line)?;
    let height = line.words[0].rect[3] - line.words[0].rect[1];
    let mut spaces = line.words[1..]
        .windows(2)
        .map(|pair| pair[1].rect[0] - pair[0].rect[2])
        .filter(|space| *space > 0.0)
        .collect::<Vec<_>>();
    spaces.sort_by(f64::total_cmp);
    let wider = spaces.len() >= 3 && gap > 1.2 * spaces[spaces.len() * 3 / 4];
    (gap >= 0.75 * height
        || fixed
            .iter()
            .any(|width| (gap - width).abs() <= 0.05 || wider && gap > *width))
    .then_some(number)
}

/// A number of up to four digits that opens a body line, and the space after it. A note's number
/// ("6 Some words that", set smaller than the body) opens no paragraph.
fn opening_number(
    page: &legal_pdf_support::PdfTextPage,
    line: &legal_pdf_support::PdfTextLine,
) -> Option<(usize, f64)> {
    let [number, next, ..] = line.words.as_slice() else {
        return None;
    };
    let height = number.rect[3] - number.rect[1];
    let gap = next.rect[0] - number.rect[2];
    (number.text.len() <= 4
        && number.text.bytes().all(|byte| byte.is_ascii_digit())
        && height > 0.0
        && height >= 0.9 * body_height(page)
        && gap >= 0.3 * height
        && line.rect[1] > page.height * 0.05
        && line.rect[3] < page.height * 0.95)
        .then(|| number.text.parse().ok().map(|value| (value, gap)))
        .flatten()
}

/// The height most of a page's lines are set at: its body text's.
fn body_height(page: &legal_pdf_support::PdfTextPage) -> f64 {
    let mut heights = page
        .lines
        .iter()
        .map(|line| line.rect[3] - line.rect[1])
        .filter(|height| *height > 0.0)
        .collect::<Vec<_>>();
    heights.sort_by(f64::total_cmp);
    heights.get(heights.len() / 2).copied().unwrap_or_default()
}

/// The separations after a paragraph number that a numbering run keeps unchanged: Lexis sets each
/// number off by two unstretched spaces ("19  The standard of review") while justification
/// stretches the line's own. Three numbers in a row, each set off at least that far and two of
/// them by just that, to a twentieth of a point, make it the run's; running text's numbers take
/// each line's own stretched space.
fn fixed_number_gaps<'a>(
    pages: impl IntoIterator<Item = &'a legal_pdf_support::PdfTextPage>,
) -> Vec<f64> {
    let openings = pages
        .into_iter()
        .flat_map(|page| {
            page.lines
                .iter()
                .filter_map(move |line| opening_number(page, line))
        })
        .collect::<Vec<_>>();
    let mut fixed = Vec::<f64>::new();
    for &(_, gap) in &openings {
        if fixed.iter().any(|width| (gap - width).abs() <= 0.05) {
            continue;
        }
        let run = openings
            .iter()
            .filter(|(_, other)| *other >= gap - 0.05)
            .collect::<Vec<_>>();
        if run.windows(3).any(|w| {
            w[0].0 + 1 == w[1].0
                && w[1].0 + 1 == w[2].0
                && w.iter()
                    .filter(|(_, other)| (other - gap).abs() <= 0.05)
                    .count()
                    >= 2
        }) {
            fixed.push(gap);
        }
    }
    fixed
}

/// Feed printed layout identities into the same resolver as text grammar candidates. `quoted`
/// holds the lines set as display quotations.
pub(super) fn candidates(
    index: &super::graph::PdfTextIndex,
    pages: &[legal_pdf_core::model::Page],
    primitives: &super::PdfPrimitiveEvidence,
    quoted: &HashSet<String>,
) -> (
    Vec<legal_structure_model::StructureCandidateRun>,
    Vec<legal_structure_model::CandidateEvidenceV2>,
) {
    use legal_pdf_support::{PdfLookupStatus, PdfTextLine, PdfTextPage, PdfTextWord};
    use legal_structure_model::{
        CandidateEvidenceV2, CandidateGrammar, CandidateObservationV2, StructureCandidateRun,
        StructureMarkerCandidate,
    };
    let eligible = |line: &legal_pdf_core::model::Line| {
        !line.exclude_from_body
            && !matches!(
                line.region_type.as_str(),
                "header" | "footer" | "footnote" | "endnote" | "note"
            )
            && !quoted.contains(&line.id)
            && !primitives.contents_line_ids.contains(&line.id)
    };
    let pages_original = pages;
    let pages = pages
        .iter()
        .map(|page| PdfTextPage {
            page_number: page.number,
            source: page.source.clone(),
            regions: page.regions.clone(),
            width: page.width,
            height: page.height,
            lines: page
                .lines
                .iter()
                .filter(|line| eligible(line))
                .map(|line| PdfTextLine {
                    id: line.id.clone(),
                    text: line.text.clone(),
                    rect: line.bbox,
                    spans:line.spans.clone(),ocr:line.ocr.clone(),
                    region_id:line.region_id.clone(),region_type:line.region_type.clone(),source:line.source.clone(),
                    words: line
                        .words
                        .iter()
                        .map(|word| PdfTextWord {
                            id:word.id.clone(),start:Some(word.start),end:Some(word.end),inferred_geometry:false,
                            text: word.text.clone(),
                            rect: word.bbox,
                            underlined: word.underlined,
                        })
                        .collect(),
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    let paragraphs = super::build_paragraphs(pages_original, &std::collections::HashMap::new());
    let headings = paragraphs
        .iter()
        .filter(|paragraph| paragraph.region_type == "heading")
        .flat_map(|paragraph| paragraph.line_ids.iter().cloned())
        .collect::<HashSet<_>>();
    let groups = paragraphs
        .iter()
        .map(|paragraph| paragraph.line_ids.clone())
        .collect::<Vec<_>>();
    let lines = pages
        .iter()
        .flat_map(|page| {
            page.lines.iter().map(move |line| {
                (
                    line.id.as_str(),
                    line.text.as_str(),
                    page.page_number,
                    line.rect,
                )
            })
        })
        .collect::<Vec<_>>();
    let mut runs = Vec::new();
    let mut evidence = Vec::new();
    for (number, status, selected) in printed_paragraphs(&pages, &lines, &groups, &headings) {
        let selections = if status == PdfLookupStatus::Found {
            vec![selected]
        } else if status == PdfLookupStatus::Ambiguous {
            // Distinct occurrences remain distinct graph identities; lookup reports ambiguity.
            lines
                .iter()
                .filter(|(_, text, ..)| bracket_label(text) == Some(number))
                .map(|(id, ..)| {
                    groups
                        .iter()
                        .find(|group| group.iter().any(|value| value == id))
                        .map(|group| group.iter().map(String::as_str).collect())
                        .unwrap_or_else(|| HashSet::from([*id]))
                })
                .collect()
        } else {
            continue;
        };
        for selected in selections {
            let ids = lines
                .iter()
                .filter(|(id, ..)| selected.contains(id))
                .map(|(id, ..)| (*id).to_owned())
                .collect::<Vec<_>>();
            let Some(range) = index.range_for_line_ids(&ids) else {
                continue;
            };
            let first = lines.iter().find(|(id, ..)| selected.contains(id));
            let marker = lines.iter().find(|(id, text, page, rect)| {
                (bracket_label(text) == Some(number) || text.trim() == number.to_string())
                    && (selected.contains(id)
                        || first.is_some_and(|(_, _, first_page, first_rect)| {
                            page == first_page && rect[1] < first_rect[3] && first_rect[1] < rect[3]
                        }))
            });
            let Some((marker_id, marker_text, ..)) = marker else {
                continue;
            };
            let Some(marker_range) = index.range_for_line_ids([&marker_id.to_string()]) else {
                continue;
            };
            let start = marker_range.start;
            let range = legal_structure_model::ScalarRange {
                start: range.start.min(start),
                end: range.end.max(marker_range.end),
            };
            let id = format!("layout-paragraph-{}", runs.len() + 1);
            evidence.push(CandidateEvidenceV2 {
                candidate_id: id.clone(),
                page_indexes: index.page_indexes_for_line_ids(&ids),
                line_ids: ids,
                observations: vec![
                    CandidateObservationV2::ComposedParagraph,
                    CandidateObservationV2::BodyProseFlow,
                ],
            });
            runs.push(StructureCandidateRun {
                id: id.clone(),
                grammar: CandidateGrammar::Numeric,
                range,
                rooted: false,
                consecutive: false,
                markers: vec![StructureMarkerCandidate {
                    id,
                    range,
                    marker_range: legal_structure_model::ScalarRange {
                        start,
                        end: start
                            + marker_text
                                .split_whitespace()
                                .next()
                                .unwrap_or("")
                                .chars()
                                .count(),
                    },
                    label: number.to_string(),
                    grammar_value: number.to_string(),
                    parent_candidate_id: None,
                    level: 0,
                    content_start: start
                        + marker_text
                            .split_whitespace()
                            .next()
                            .unwrap_or("")
                            .chars()
                            .count(),
                }],
            });
        }
    }
    (runs, evidence)
}

/// Detached paragraph labels share a body row and a numbering sequence, not a note band.
pub(super) fn margin_label_ids(pages: &[legal_pdf_core::model::Page]) -> HashSet<String> {
    use legal_pdf_support::{PdfTextLine, PdfTextPage};
    let witnesses = pages
        .iter()
        .map(|page| PdfTextPage {
            page_number: page.number,
            source: page.source.clone(),
            regions: page.regions.clone(),
            width: page.width,
            height: page.height,
            lines: page
                .lines
                .iter()
                .filter(|line| {
                    !line.exclude_from_body
                        && !matches!(line.region_type.as_str(), "header" | "footer")
                })
                .map(|line| PdfTextLine {
                    id: line.id.clone(),
                    text: line.text.clone(),
                    rect: line.bbox,
                    spans:line.spans.clone(),ocr:line.ocr.clone(),
                    region_id:line.region_id.clone(),region_type:line.region_type.clone(),source:line.source.clone(),
                    words: Vec::new(),
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    let (_, labels) = margin_labels(&witnesses, &HashSet::new());
    if !numbering_run(&labels) {
        return HashSet::new();
    }
    labels
        .iter()
        .flat_map(|(number, page_index, y)| {
            let page = &pages[*page_index];
            let body_size = super::median(
                page.lines
                    .iter()
                    .filter(|line| {
                        line.bbox[1] < page.height * 0.60
                            && line.text.chars().filter(|c| c.is_alphabetic()).count() >= 8
                    })
                    .map(legal_pdf_core::line_font_size)
                    .filter(|size| *size > 0.0)
                    .collect(),
            );
            page.lines
                .iter()
                .filter(move |line| {
                    let text = line
                        .text
                        .trim()
                        .trim_start_matches('[')
                        .trim_end_matches(']');
                    text.parse::<usize>().ok() == Some(*number)
                        && (line.bbox[1] - y).abs() <= (line.bbox[3] - line.bbox[1]) * 0.6
                        && page.lines.iter().any(|body| {
                            let size = legal_pdf_core::line_font_size(body);
                            body.bbox[2] - body.bbox[0] > page.width * 0.20
                                && body.text.chars().any(char::is_alphabetic)
                                && (body.bbox[1] - y).abs() < 1.0
                                && size >= body_size * 0.98
                                && size > legal_pdf_core::line_font_size(line) * 1.1
                        })
                })
                .map(|line| line.id.clone())
        })
        .collect()
}
