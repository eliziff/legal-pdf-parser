//! Case captions from their opening metadata and aligned filing labels.

use super::{heading_line_excluded, PdfPrimitiveEvidence};
use legal_pdf_core::{layout_analysis, line_font_size, Line, Page};
use legal_pdf_headings::party_label;
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Field {
    File,
    Court,
    Place,
    Parties,
    Matter,
    Document,
    Contact,
    Date,
    Docket,
    Registry,
    Citation,
    Bench,
}

fn field(text: &str) -> Option<Field> {
    let labelled = text.contains(':');
    let key = text.split_once(':').map_or(text, |(key, _)| key);
    let key = key.trim().trim_end_matches('.').to_ascii_lowercase();
    Some(match key.as_str() {
        "court file no" | "court file number" | "court file numbers" | "file no" => Field::File,
        "court" => Field::Court,
        "judicial centre" | "judicial center" => Field::Place,
        "matters" | "re" => Field::Matter,
        "between" | "not parties to the appeal" | "not parties to appeal" => Field::Parties,
        "document" | "title of proceeding" | "title of proceedings" | "style of cause" => {
            Field::Document
        }
        "address for service" => Field::Contact,
        "date" | "heard" | "judgment" if labelled => Field::Date,
        "docket" | "no. on list" | "no on list" if labelled => Field::Docket,
        "registry" if labelled => Field::Registry,
        "citation" if labelled => Field::Citation,
        "before" | "before justice" | "coram" | "the court" if labelled => Field::Bench,
        _ if (key.starts_with("dated ") || key.starts_with("filed "))
            && text.chars().any(|c| c.is_ascii_digit()) =>
        {
            Field::Date
        }
        _ if party_label().is_match(text.trim().trim_end_matches('/')) => Field::Parties,
        _ => return None,
    })
}

fn closing(text: &str) -> bool {
    let text = text.trim().to_ascii_lowercase();
    let key = text.split_once(':').map_or(text.as_str(), |(key, _)| key);
    matches!(
        key.trim(),
        "counsel" | "appearances" | "solicitors" | "participant information"
    ) || (!text.contains(':')
        && matches!(
            text.as_str(),
            "memorandum of judgment" | "judgment" | "endorsement"
        ))
        || text.starts_with("reasons for judgment")
        || text.starts_with("reasons for decision")
}

fn preamble(text: &str) -> bool {
    let text = text.trim().to_ascii_lowercase();
    text == "in bankruptcy"
        || [
            "in the matter of ",
            "and in the matter of ",
            "appeal from the ",
        ]
        .iter()
        .any(|prefix| text.starts_with(prefix))
}

fn judicial(text: &str) -> bool {
    let text = text.trim();
    let lowered = text.to_ascii_lowercase();
    let text = if lowered.starts_with("the ") {
        &text[4..]
    } else {
        text
    };
    let lowered = text.to_ascii_lowercase();
    [
        "honourable ",
        "honorable ",
        "hon. ",
        "chief justice ",
        "chief judge ",
        "justice ",
        "judge ",
        "mr. justice ",
        "madam justice ",
    ]
    .iter()
    .any(|prefix| {
        lowered.starts_with(prefix)
            && text[prefix.len()..]
                .trim_start()
                .chars()
                .next()
                .is_some_and(char::is_uppercase)
    })
}

fn bench_end(lines: &[&Line], slot: usize) -> Option<usize> {
    let mut end = slot;
    for line in &lines[slot + 1..] {
        if !judicial(&line.text) {
            break;
        }
        end += 1;
    }
    (end > slot
        || lines[slot]
            .text
            .split_once(':')
            .is_some_and(|(_, value)| judicial(value)))
    .then_some(end)
}

fn numbered_start(marker: &str) -> bool {
    layout_analysis()
        .numbering_marker(marker, None)
        .is_some_and(|marker| marker.ordinal == 1)
}

fn reasons(line: &Line) -> bool {
    let value = line.text.trim();
    if value.starts_with('[') && value.ends_with(']') && numbered_start(value) {
        return true;
    }
    let Some((marker, text)) = value.split_once(char::is_whitespace) else {
        return false;
    };
    text.trim().chars().count() >= 24 && numbered_start(marker)
}

/// A dedicated citation line can name the case; a reference inside another field cannot.
fn citation_title(line: &Line) -> Option<bool> {
    let key = field(&line.text);
    if key.is_some() && key != Some(Field::Citation) {
        return None;
    }
    let text = if key == Some(Field::Citation) {
        line.text.split_once(':')?.1.trim()
    } else {
        line.text.trim()
    };
    if !text.chars().any(|c| c.is_ascii_digit()) {
        return None;
    }
    let citations = layout_analysis().citations(text);
    if citations.is_empty()
        || text
            .char_indices()
            .filter(|(at, c)| {
                c.is_alphabetic() && !citations.iter().any(|span| span.range.contains(at))
            })
            .count()
            >= 3
    {
        return None;
    }
    Some(
        citations
            .iter()
            .any(|span| span.range.start < span.core_range.start),
    )
}

/// Called within one constituent and language, after table and furniture classification.
pub(super) fn classify(pages: &mut [Page], evidence: &mut PdfPrimitiveEvidence) {
    let eligible = |line: &Line| {
        !line.exclude_from_body
            && line.note_region_mode.is_empty()
            && matches!(line.region_type.as_str(), "body" | "heading")
            && !heading_line_excluded(evidence, &line.id)
    };
    let window = pages
        .iter()
        .take(2)
        .flat_map(|page| &page.lines)
        .filter(|line| eligible(line))
        .take_while(|line| !reasons(line))
        .collect::<Vec<_>>();
    let end = window
        .iter()
        .position(|line| closing(&line.text))
        .unwrap_or(window.len());
    let opening = &window[..end];
    let fields = opening
        .iter()
        .enumerate()
        .filter_map(|(slot, line)| field(&line.text).map(|field| (slot, field)))
        .collect::<Vec<_>>();
    let mut assigned = HashMap::<String, usize>::new();
    let dated = fields.iter().any(|(_, field)| *field == Field::Date);
    let filed = fields
        .iter()
        .any(|(_, field)| matches!(field, Field::File | Field::Docket | Field::Registry));
    let structured = fields.iter().any(|(_, field)| *field == Field::File)
        && fields
            .iter()
            .any(|(_, field)| matches!(field, Field::Court | Field::Place | Field::Document));
    let cited = opening
        .iter()
        .take(8)
        .enumerate()
        .find_map(|(slot, line)| citation_title(line).map(|named| (slot, named)));
    let appendix = opening
        .first()
        .is_some_and(|line| super::constituents::appended_title(&line.text).is_some());
    if dated && filed && !structured && (cited.is_some() || appendix) {
        let last = fields
            .iter()
            .filter(|(_, field)| *field != Field::Bench)
            .map(|(slot, _)| *slot)
            .chain(
                opening
                    .iter()
                    .enumerate()
                    .filter(|(_, line)| preamble(&line.text))
                    .map(|(slot, _)| slot),
            )
            .max();
        if let Some(mut last) = last {
            if preamble(&opening[last].text) {
                while last + 1 < opening.len()
                    && field(&opening[last + 1].text).is_none()
                    && !opening[last + 1].text.contains(':')
                {
                    last += 1;
                }
            }
            let start = if appendix {
                1
            } else {
                let (slot, named) = cited.unwrap();
                fields
                    .iter()
                    .find(|(at, field)| *at <= slot && *field == Field::Citation)
                    .map_or(slot + usize::from(named), |(at, _)| *at)
            }
            .min(
                fields
                    .iter()
                    .filter(|(_, field)| {
                        matches!(
                            field,
                            Field::File
                                | Field::Date
                                | Field::Docket
                                | Field::Registry
                                | Field::Citation
                        )
                    })
                    .map(|(slot, _)| *slot)
                    .min()
                    .unwrap(),
            );
            assigned.extend(
                opening[start..=last]
                    .iter()
                    .map(|line| (line.id.clone(), line.block_index)),
            );
        }
    }
    if let Some(page) = pages.first() {
        let lines = opening
            .iter()
            .copied()
            .filter(|line| line.page_index == page.index)
            .collect::<Vec<_>>();
        let fields = lines
            .iter()
            .filter_map(|line| field(&line.text).map(|field| (*line, field)))
            .collect::<Vec<_>>();
        let filed = fields.iter().any(|(_, field)| *field == Field::File);
        let labelled = fields
            .iter()
            .any(|(_, field)| matches!(field, Field::Court | Field::Place | Field::Document));
        if filed && labelled {
            let mut next = page
                .lines
                .iter()
                .map(|line| line.block_index)
                .max()
                .unwrap_or(0)
                + 1;
            for &(label, kind) in &fields {
                if kind == Field::Bench
                    && bench_end(
                        &lines,
                        lines.iter().position(|line| line.id == label.id).unwrap(),
                    )
                    .is_none()
                {
                    continue;
                }
                let tagged = label.spans.iter().any(|span| span.structure_role.is_some());
                let belongs = |line: &&Line| {
                    !tagged || line.spans.iter().any(|span| span.structure_role.is_some())
                };
                let until = fields
                    .iter()
                    .map(|(line, _)| line.bbox[1])
                    .filter(|y| *y > label.bbox[1] + 0.5 * (label.bbox[3] - label.bbox[1]))
                    .min_by(f64::total_cmp);
                let bounded = until.is_some();
                let until = until.unwrap_or(page.height);
                let height = (label.bbox[3] - label.bbox[1]).max(1.0);
                let value = lines
                    .iter()
                    .copied()
                    .filter(belongs)
                    .filter(|line| {
                        line.bbox[0] > label.bbox[2] + 6.0
                            && (line.bbox[1] - label.bbox[1]).abs() < height / 2.0
                    })
                    .min_by(|a, b| a.bbox[0].total_cmp(&b.bbox[0]));
                assigned.insert(label.id.clone(), next);
                // Read wrapped labels and values in their own columns; drawing order may differ.
                for left in std::iter::once(label.bbox[0]).chain(value.map(|line| line.bbox[0])) {
                    let mut column = lines
                        .iter()
                        .copied()
                        .filter(belongs)
                        .filter(|line| {
                            label.bbox[1] - 1.0 <= line.bbox[1]
                                && line.bbox[1] < until
                                && if left == label.bbox[0] {
                                    (line.bbox[0] - left).abs() <= 6.0
                                } else {
                                    // Centred values can wrap to the left of their first line.
                                    line.bbox[0] > label.bbox[2] + 6.0
                                }
                        })
                        .collect::<Vec<_>>();
                    column.sort_by(|a, b| crate::layout::band_geometry_order(a, b));
                    let mut prior = label;
                    for line in column {
                        let size = line_font_size(prior).max(prior.bbox[3] - prior.bbox[1]);
                        if !bounded && line.bbox[1] - prior.bbox[1] > 2.0 * size {
                            break;
                        }
                        assigned.insert(line.id.clone(), next);
                        prior = line;
                    }
                }
                next += 1;
            }
        }
    }
    // A bench or hearing date can follow counsel without turning the intervening text into a caption.
    if structured || dated && filed && (cited.is_some() || appendix) {
        let tagged = fields.iter().any(|(slot, _)| {
            opening[*slot]
                .spans
                .iter()
                .any(|span| span.structure_role.is_some())
        });
        for (slot, line) in window.iter().enumerate() {
            if tagged && !line.spans.iter().any(|span| span.structure_role.is_some()) {
                continue;
            }
            let kind = field(&line.text);
            if kind == Some(Field::Bench) {
                if let Some(last) = bench_end(&window, slot) {
                    for line in &window[slot..=last] {
                        assigned.entry(line.id.clone()).or_insert(line.block_index);
                    }
                }
            } else if matches!(kind, Some(Field::Date | Field::Docket)) {
                let value = line
                    .text
                    .split_once(':')
                    .map_or(line.text.as_str(), |(_, value)| value)
                    .trim();
                let next = window.get(slot + 1).filter(|next| {
                    let advance = next.bbox[1] - line.bbox[1];
                    let height = line.bbox[3] - line.bbox[1];
                    next.page_index == line.page_index
                        && -0.5 * height <= advance
                        && advance <= 2.0 * height
                });
                if !value.is_empty() && value.chars().any(|c| c.is_ascii_digit()) {
                    assigned.entry(line.id.clone()).or_insert(line.block_index);
                } else if value.is_empty() {
                    if let Some(next) = next.filter(|next| {
                        super::date_line(&next.text)
                            || kind == Some(Field::Docket)
                                && next.text.split_whitespace().count() == 1
                                && next.text.chars().any(|c| c.is_ascii_digit())
                    }) {
                        assigned.entry(line.id.clone()).or_insert(line.block_index);
                        assigned.entry(next.id.clone()).or_insert(line.block_index);
                    }
                }
            }
        }
    }
    for line in pages.iter_mut().flat_map(|page| &mut page.lines) {
        if let Some(&block) = assigned.get(&line.id) {
            line.region_type = "caption".to_owned();
            line.block_index = block;
            evidence.heading_levels.remove(&line.id);
        }
    }
}
