//! Each line's role on its page: body, note, heading, caption, furniture claimed already, read
//! with the page's note band and the references and markers detached from their lines; and the
//! endnotes a document's body refers to.

use super::*;

pub(super) fn standalone_reference(text: &str) -> Option<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^(?:\d{1,4}|[*\u{2020}\u{2021}\u{00a7}\u{00b6}#])$").unwrap())
        .is_match(text)
        .then(|| normalize_label(text))
}

pub(super) fn detached_reference_target(
    reference_index: usize,
    lines: &[Line],
    body_size: f64,
) -> Option<(usize, usize)> {
    let reference = &lines[reference_index];
    let reference_x = (reference.bbox[0] + reference.bbox[2]) / 2.0;
    lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| {
            if index == reference_index
                || line.exclude_from_body
                || matches!(line.region_type.as_str(), "header" | "footer")
                || line_font_size(line) < body_size * 0.80
            {
                return None;
            }
            let y_distance = (line.bbox[1] - reference.bbox[1]).abs();
            if y_distance > ((line.bbox[3] - line.bbox[1]) * 0.20).max(2.0) {
                return None;
            }
            let (distance, offset) = line
                .spans
                .iter()
                .flat_map(|span| [(span.bbox[0], span.start), (span.bbox[2], span.end)])
                .map(|(x, offset)| ((reference_x - x).abs(), offset))
                .min_by(|left, right| left.0.total_cmp(&right.0).then(right.1.cmp(&left.1)))?;
            (distance <= body_size.max(6.0)).then_some((
                (
                    distance,
                    y_distance,
                    line.source_index.abs_diff(reference.source_index),
                ),
                index,
                offset,
            ))
        })
        .min_by(|left, right| {
            left.0
                 .0
                .total_cmp(&right.0 .0)
                .then(left.0 .1.total_cmp(&right.0 .1))
                .then(left.0 .2.cmp(&right.0 .2))
        })
        .map(|(_, index, offset)| (index, offset))
}

pub(super) fn associate_detached_references(pages: &mut [Page], separators: &[Option<f64>]) {
    for (page, separator) in pages.iter_mut().zip(separators) {
        let body_size = median(
            page.lines
                .iter()
                .filter(|line| {
                    !matches!(line.region_type.as_str(), "header" | "footer")
                        && line.bbox[1] < page.height * 0.75
                })
                .map(line_font_size)
                .filter(|size| (7.0..=20.0).contains(size))
                .collect(),
        );
        let body_size = if body_size > 0.0 { body_size } else { 10.0 };
        let note_cut = separator.unwrap_or(page.height * 0.88);
        // The oracle consumes accepted markers one at a time. Preserve that
        // mutation order so a consumed row cannot become a later host.
        for marker_index in 0..page.lines.len() {
            let Some(label) = standalone_reference(page.lines[marker_index].text.trim()) else {
                continue;
            };
            let size = line_font_size(&page.lines[marker_index]);
            if page.lines[marker_index].exclude_from_body
                || matches!(
                    page.lines[marker_index].region_type.as_str(),
                    "header" | "footer"
                )
                || !(0.0 < size && size <= body_size * 0.75)
                || page.lines[marker_index].bbox[1] >= note_cut
            {
                continue;
            }
            let Some((host_index, offset)) =
                detached_reference_target(marker_index, &page.lines, body_size)
            else {
                continue;
            };
            let source_line_id = page.lines[marker_index].id.clone();
            let selected_text = page.lines[marker_index].text.trim().to_owned();
            page.lines[host_index]
                .detached_references
                .push(DetachedReference {
                    note_id: label,
                    selected_text,
                    start_offset: offset,
                    end_offset: offset,
                    source_line_id,
                });
            page.lines[marker_index].exclude_from_body = true;
        }
        associate_spliced_markers(page, note_cut);
    }
}

pub(super) fn orphaned_marker(line: &Line) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    !line.spans.is_empty()
        && RE
            .get_or_init(|| Regex::new(r"^\d{1,4}$").unwrap())
            .is_match(&line.text)
}

pub(super) fn spliced_marker_host(marker: &Line, host: Option<&Line>) -> bool {
    let Some(host) = host else {
        return false;
    };
    if orphaned_marker(host) || marker.block_index != host.block_index {
        return false;
    }
    let marker_height = marker.bbox[3] - marker.bbox[1];
    let host_height = host.bbox[3] - host.bbox[1];
    if marker_height <= 0.0 || host_height <= 0.0 {
        return false;
    }
    let overlap = marker.bbox[3].min(host.bbox[3]) - marker.bbox[1].max(host.bbox[1]);
    if overlap.max(0.0) / marker_height < 0.5
        || marker.bbox[0] < host.bbox[0] - 12.0
        || marker.bbox[0] > host.bbox[2] + 12.0
    {
        return false;
    }
    if marker.spans.iter().all(|span| span.superscript) {
        return true;
    }
    let marker_size = line_font_size(marker);
    let host_size = line_font_size(host);
    marker_size > 0.0
        && host_size >= 1.25 * marker_size
        && marker.bbox[3] <= host.bbox[3] - 0.25 * host_height
}

pub(super) fn associate_spliced_markers(page: &mut Page, note_cut: f64) {
    let eligible: Vec<usize> = page
        .lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| {
            (!matches!(line.region_type.as_str(), "header" | "footer") && !line.exclude_from_body)
                .then_some(index)
        })
        .collect();
    if eligible.len() < 2 {
        return;
    }
    let mut claims = Vec::<(usize, usize)>::new();
    for (position, marker_index) in eligible.iter().copied().enumerate() {
        let marker = &page.lines[marker_index];
        if !orphaned_marker(marker) || marker.bbox[1] >= note_cut {
            continue;
        }
        let previous = position
            .checked_sub(1)
            .and_then(|index| eligible.get(index))
            .copied();
        let next = eligible.get(position + 1).copied();
        let previous_ok = spliced_marker_host(marker, previous.map(|index| &page.lines[index]));
        let next_ok = spliced_marker_host(marker, next.map(|index| &page.lines[index]));
        if previous_ok == next_ok {
            continue;
        }
        claims.push((
            marker_index,
            if previous_ok {
                previous.unwrap()
            } else {
                next.unwrap()
            },
        ));
    }
    let mut host_counts = HashMap::<usize, usize>::new();
    for (_, host) in &claims {
        *host_counts.entry(*host).or_default() += 1;
    }
    for (marker_index, host_index) in claims {
        if host_counts.get(&host_index) != Some(&1) {
            continue;
        }
        let value = normalize_label(page.lines[marker_index].text.trim());
        let selected_text = page.lines[marker_index].text.trim().to_owned();
        let source_line_id = page.lines[marker_index].id.clone();
        let offset = page.lines[host_index].text.chars().count();
        page.lines[host_index]
            .detached_references
            .push(DetachedReference {
                note_id: value,
                selected_text,
                start_offset: offset,
                end_offset: offset,
                source_line_id,
            });
        page.lines[marker_index].exclude_from_body = true;
    }
}

/// "Notes", "Footnotes", "Endnotes": a heading, capitalized; a lowercase "note" alone on its line
/// is a sentence's word wrapped ("... on the information or indictment, / note").
pub(super) fn endnote_heading(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    let text = text.trim();
    text.starts_with(char::is_uppercase)
        && RE.get_or_init(|| Regex::new(r"(?i)^(?:end|foot)?notes?$").unwrap()).is_match(text)
}

pub(super) fn continuing_note_heading(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)^(?:endnotes?|footnotes?|notes?)(?:\s*\(?(?:continued|cont'd)\)?)?$")
            .unwrap()
    })
    .is_match(text.trim())
}

pub(super) fn structural_reset_heading(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)^(?:appendix|annex|schedule|part|chapter|bibliography|references|works\s+cited|table\s+of\s+authorities|index|acknowledg(?:e)?ments|certificate\s+of\s+service|about\s+the\s+authors?)(?:\s+[\w.-]+)?$",
        )
        .unwrap()
    })
    .is_match(text.trim())
}

pub(super) fn citation_shaped_candidate(page: &Page, candidate: &(usize, LabelPrefix, bool)) -> bool {
    let line = &page.lines[candidate.0];
    let tail = scalar_suffix(&line.text, candidate.1.end);
    layout_analysis().is_citation_shaped_tail(tail)
}

pub(super) fn longest_label_run(values: &[u32]) -> usize {
    let mut runs = vec![1; values.len()];
    for index in 0..values.len() {
        for prior in 0..index {
            if (1..=3).contains(&values[index].saturating_sub(values[prior])) {
                runs[index] = runs[index].max(runs[prior] + 1);
            }
        }
    }
    runs.into_iter().max().unwrap_or(0)
}

pub(super) fn has_prior_reference(page: &Page, label: &str, label_y: f64) -> bool {
    let matches_label = |value: &str| {
        let value = value.trim();
        value == label || normalize_label(value) == label
    };
    page.lines.iter().any(|line| {
        line.bbox[1] < label_y
            && (line
                .spans
                .iter()
                .any(|span| span.superscript && matches_label(&span.text))
                || line
                    .detached_references
                    .iter()
                    .any(|reference| matches_label(&reference.note_id))
                || (label.chars().all(is_note_symbol)
                    && line
                        .text
                        .trim_end()
                        .strip_suffix(label)
                        .and_then(|before| before.chars().next_back())
                        .is_some_and(char::is_alphanumeric)))
    })
}

pub(super) fn classify_line_roles(
    pages: &mut [Page],
    separators: &[Option<f64>],
    contents_pages: &HashSet<usize>,
    evidence: &mut PdfPrimitiveEvidence,
) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    let paragraph_candidates = numbered_paragraphs::margin_label_ids(pages);
    // Notes are set small against the document's body type, read once from all its running text,
    // not against each page's own lines: a page mostly of notes would take their size for its
    // body. A page whose lines carry no size falls back to its own.
    let document_body = body_font_size(pages.iter().flat_map(|page| &page.lines).filter(|line| {
        !matches!(line.region_type.as_str(), "header" | "footer")
            && !line.exclude_from_body
            && !evidence.claims.table_cell_line_ids.contains(&line.id)
            && !evidence.claims.table_note_line_ids.contains(&line.id)
    }));
    let mut expected_endnote: Option<u32> = None;
    let mut continuing_size: Option<f64> = None;
    let mut previous_page: Option<usize> = None;
    for (page, separator) in pages.iter_mut().zip(separators.iter().copied()) {
        let table_cells = page
            .lines
            .iter()
            .enumerate()
            .filter(|(_, line)| evidence.claims.table_cell_line_ids.contains(&line.id))
            .map(|(slot, _)| slot)
            .collect::<HashSet<_>>();
        let table_notes = page
            .lines
            .iter()
            .enumerate()
            .filter(|(_, line)| evidence.claims.table_note_line_ids.contains(&line.id))
            .map(|(slot, _)| slot)
            .collect::<HashSet<_>>();
        if previous_page.is_some_and(|prior| prior.checked_add(1) != Some(page.index)) {
            expected_endnote = None;
            continuing_size = None;
        }
        previous_page = Some(page.index);
        let line_sizes: Vec<f64> = page.lines.iter().map(line_font_size).collect();
        let body_size = if document_body > 0.0 && line_sizes.iter().any(|size| *size > 0.0) {
            document_body
        } else {
            median(
                page.lines
                    .iter()
                    .enumerate()
                    .filter(|(index, line)| {
                        !matches!(line.region_type.as_str(), "header" | "footer")
                            && !line.exclude_from_body
                            && !table_cells.contains(index)
                            && !table_notes.contains(index)
                            && page.height * 0.10 <= line.bbox[1]
                            && line.bbox[1] <= page.height * 0.75
                    })
                    .map(|(index, _)| line_sizes[index])
                    .filter(|size| (4.0..=24.0).contains(size))
                    .collect(),
            )
        };
        let body_size = if body_size > 0.0 { body_size } else { 10.0 };
        let wide_sizes: Vec<_> = page
            .lines
            .iter()
            .enumerate()
            .filter(|(index, line)| {
                !line.exclude_from_body
                    && !table_cells.contains(index)
                    && !table_notes.contains(index)
                    && line_width(line) >= page.width * 0.30
                    && line
                        .text
                        .chars()
                        .filter(|character| character.is_alphabetic())
                        .take(4)
                        .count()
                        == 4
            })
            .map(|(index, _)| line_sizes[index])
            .filter(|size| (4.0..=24.0).contains(size))
            .collect();
        let note_size = if wide_sizes.len() >= 3 {
            median(wide_sizes)
        } else {
            body_size
        };
        let tolerance = (page.height * 0.004).max(1.0);
        let table_band = table_cells.iter().fold(None, |band, index| {
            let line = &page.lines[*index];
            Some(
                band.map_or((line.bbox[1], line.bbox[3]), |(top, bottom): (f64, f64)| {
                    (top.min(line.bbox[1]), bottom.max(line.bbox[3]))
                }),
            )
        });
        let separator = separator.filter(|cut| {
            table_band
                .is_none_or(|(top, bottom)| *cut < top - tolerance || *cut > bottom + tolerance)
        });
        // Alignment, typography and a paragraph sequence are positive evidence, not a veto
        // on the note witnesses. Rules, references, source regions and note continuation win.
        let paragraph_labels =
            page.lines
                .iter()
                .enumerate()
                .filter(|(index, line)| {
                    paragraph_candidates.contains(&line.id)
                        && !table_notes.contains(index)
                        && !separator.is_some_and(|cut| line.bbox[1] >= cut - tolerance)
                        && !has_prior_reference(page, line.text.trim(), line.bbox[1])
                        && !evidence
                            .source_regions
                            .as_ref()
                            .and_then(|regions| regions.get(&line.id))
                            .is_some_and(|region| region.contains("note"))
                        && !page.lines.iter().any(|heading| {
                            heading.bbox[1] < line.bbox[1] && endnote_heading(&heading.text)
                        })
                        && !expected_endnote.is_some_and(|expected| {
                            line.text
                                .trim()
                                .parse::<u32>()
                                .is_ok_and(|number| number >= expected)
                                && !page.lines.iter().any(|heading| {
                                    heading.bbox[1] < line.bbox[1]
                                        && structural_reset_heading(&heading.text)
                                })
                        })
                })
                .map(|(_, line)| line.id.clone())
                .collect::<HashSet<_>>();
        for line in &mut page.lines {
            line.paragraph_label_candidate = paragraph_labels.contains(&line.id);
        }
        let mut candidates = Vec::new();
        let mut suppress = Vec::new();
        for (index, line) in page.lines.iter().enumerate() {
            if line.paragraph_label_candidate {
                continue;
            }
            if table_cells.contains(&index) {
                suppress.push(index);
                continue;
            }
            if (matches!(line.region_type.as_str(), "header" | "footer")
                && !table_notes.contains(&index))
                || line.exclude_from_body
            {
                continue;
            }
            let Some(prefix) = label_prefix(&line.text) else {
                continue;
            };
            if table_notes.contains(&index) {
                candidates.push((index, prefix, true));
                continue;
            }
            let typographic = label_is_typographic(line, &prefix, line_sizes[index], body_size)
                || (standalone_note_label(line)
                    && aligned_note_body_index(&page.lines, index).is_some_and(|body| {
                        let size = line_sizes[body];
                        size > 0.0 && size <= body_size * 0.90
                    }));
            let size = {
                let size = line_sizes[index];
                if size > 0.0 {
                    size
                } else {
                    body_size
                }
            };
            let bottom_right = separator.is_none()
                && line.bbox[1] >= page.height * 0.91
                && line.bbox[0] >= page.width * 0.50
                && !compact_note_line(&line.text);
            // A number followed by a comma is the line's text ("1991, c. 43, s. 4" in a
            // statute's small-type history note), unless it is raised above its line.
            let comma_tail = line.text.chars().nth(prefix.end) == Some(',')
                && !label_is_raised(line, &prefix, line_sizes[index]);
            // A number opening a line under one that ends on a reference's noun ("... c. 27
            // (1st Supp.), s" / "13.") is that reference's number, wrapped.
            let wrapped_reference = !label_is_raised(line, &prefix, line_sizes[index])
                && reference_wrapped_onto(&page.lines, &line_sizes, index);
            let below_separator = separator.is_some_and(|cut| line.bbox[1] >= cut - tolerance);
            let suppressed = (line.bbox[1] >= page.height * 0.94
                && !(below_separator && typographic))
                || bottom_right
                || comma_tail
                || wrapped_reference
                || size > body_size * 1.15;
            if suppressed {
                suppress.push(index);
                continue;
            }
            candidates.push((index, prefix, typographic));
        }
        for index in suppress {
            page.lines[index].suppress_footnote_label = true;
        }

        let numeric: Vec<u32> = candidates
            .iter()
            .filter_map(|(_, prefix, _)| prefix.label.parse().ok())
            .collect();
        let best_run = longest_label_run(&numeric);
        let page_columns = {
            let page_model = column_model(&page.lines);
            if page_model.kind == "two_column" {
                page_model
            } else {
                let label_lines: Vec<_> = candidates
                    .iter()
                    .map(|(index, _, _)| page.lines[*index].clone())
                    .collect();
                let label_model = column_model(&label_lines);
                if label_model.kind == "single" {
                    page_model
                } else {
                    label_model
                }
            }
        };
        let first_candidate_y = candidates
            .first()
            .map(|(index, _, _)| page.lines[*index].bbox[1]);
        let minimum_label_y = candidates
            .iter()
            .map(|(index, _, _)| page.lines[*index].bbox[1])
            .min_by(f64::total_cmp);
        let endnote_heading_y = page
            .lines
            .iter()
            .filter(|line| {
                endnote_heading(&line.text) && first_candidate_y.is_none_or(|y| line.bbox[1] < y)
            })
            .map(|line| line.bbox[1])
            .min_by(f64::total_cmp);
        let has_endnote_heading = endnote_heading_y.is_some();
        let endnote_heading_source = has_endnote_heading.then(|| {
            page.lines
                .iter()
                .filter(|line| endnote_heading(&line.text))
                .map(|line| line.source_index)
                .min()
                .unwrap_or(0)
        });
        let minimum_label_y = minimum_label_y.unwrap_or(page.height);
        let content_before = || {
            page.lines.iter().enumerate().filter_map(|(index, line)| {
                (!matches!(line.region_type.as_str(), "header" | "footer")
                    && line.bbox[1] < minimum_label_y)
                    .then_some(index)
            })
        };
        let first_content = content_before()
            .min_by(|left, right| page.lines[*left].bbox[1].total_cmp(&page.lines[*right].bbox[1]));
        let generic_heading_reset = candidates.is_empty()
            && first_content.is_some_and(|index| {
                let line = &page.lines[index];
                let text = line.text.trim();
                let letter_count = text
                    .chars()
                    .filter(|character| character.is_alphabetic())
                    .count();
                line.bbox[1] >= page.height * 0.08
                    && letter_count >= 4
                    && text.chars().count() <= 100
                    && all_caps(text)
                    && !continuing_note_heading(text)
            });
        let structural_reset = generic_heading_reset
            || content_before().any(|index| {
                let line = &page.lines[index];
                (line.region_type == "heading" || line.bbox[1] >= page.height * 0.08)
                    && structural_reset_heading(&line.text)
            });
        let label_sizes: Vec<f64> = candidates
            .iter()
            .map(|(index, _, _)| line_sizes[*index])
            .filter(|size| *size > 0.0)
            .collect();
        let label_size = (!label_sizes.is_empty()).then(|| median(label_sizes));
        let early_labels = candidates.iter().any(|(index, _, typographic)| {
            page.lines[*index].bbox[1] < page.height * 0.48
                && (*typographic || line_sizes[*index] <= body_size * 0.90)
        });
        let lower_numeric: Vec<u32> = candidates
            .iter()
            .filter(|(index, _, _)| {
                page.lines[*index].bbox[1] >= page.height * 0.48
                    && line_sizes[*index] > 0.0
                    && line_sizes[*index] <= body_size * 0.90
            })
            .filter_map(|(_, prefix, _)| prefix.label.parse().ok())
            .collect();
        let lower_note_run = longest_label_run(&lower_numeric) >= 3;
        let expected_indexes: Vec<usize> = candidates
            .iter()
            .enumerate()
            .filter_map(|(index, (_, prefix, _))| {
                (prefix.label.parse::<u32>().ok() == expected_endnote).then_some(index)
            })
            .collect();
        let noncitation: Vec<usize> = expected_indexes
            .iter()
            .copied()
            .filter(|index| !citation_shaped_candidate(page, &candidates[*index]))
            .collect();
        let pool = if noncitation.is_empty() {
            &expected_indexes
        } else {
            &noncitation
        };
        let expected_index = pool.iter().copied().max_by_key(|index| {
            let mut sequence = 1;
            let mut prior = expected_endnote;
            for (_, prefix, _) in &candidates[index + 1..] {
                let Ok(current) = prefix.label.parse::<u32>() else {
                    break;
                };
                if prior.is_none_or(|value| current != value + 1) {
                    break;
                }
                sequence += 1;
                prior = Some(current);
            }
            (usize::from(candidates[*index].2), sequence, *index)
        });
        if expected_endnote.is_some() {
            if let Some(selected) = expected_index {
                for candidate in &candidates[..selected] {
                    page.lines[candidate.0].suppress_footnote_label = true;
                }
                for index in &expected_indexes {
                    if *index != selected && citation_shaped_candidate(page, &candidates[*index]) {
                        page.lines[candidates[*index].0].suppress_footnote_label = true;
                    }
                }
            }
        }
        let selected_start = if expected_endnote.is_some() {
            expected_index.unwrap_or(0)
        } else {
            0
        };
        let content_sizes: Vec<f64> = page
            .lines
            .iter()
            .enumerate()
            .filter(|(_, line)| {
                !matches!(line.region_type.as_str(), "header" | "footer") && !line.exclude_from_body
            })
            .map(|(index, _)| line_sizes[index])
            .filter(|size| *size > 0.0)
            .collect();
        let content_size = (!content_sizes.is_empty()).then(|| median(content_sizes));
        let continuation_size_matches = continuing_size
            .is_none_or(|size| content_size.is_none_or(|content| content <= size * 1.15));
        let label_free_continuation = expected_endnote.is_some()
            && candidates.is_empty()
            && !structural_reset
            && continuation_size_matches;
        let continuing_endnotes = expected_endnote.is_some()
            && !structural_reset
            && continuation_size_matches
            && (label_free_continuation || expected_index.is_some());
        let margin_candidates: Vec<_> = candidates
            .iter()
            .filter_map(|(index, _, _)| {
                let line = &page.lines[*index];
                (line_sizes[*index] <= note_size * 0.90 && line_width(line) <= page.width * 0.30)
                    .then_some(*index)
            })
            .collect();
        let margin_numeric: Vec<u32> = candidates
            .iter()
            .filter(|(index, _, _)| margin_candidates.contains(index))
            .filter_map(|(_, prefix, _)| prefix.label.parse().ok())
            .collect();
        let supported_margin_candidates: Vec<_> = candidates
            .iter()
            .filter(|(index, prefix, _)| {
                margin_candidates.contains(index)
                    && has_prior_reference(page, &prefix.label, page.lines[*index].bbox[1])
            })
            .map(|(index, _, _)| *index)
            .collect();
        let margin_labels = if longest_label_run(&margin_numeric) >= 3 {
            &margin_candidates
        } else {
            &supported_margin_candidates
        };
        let margin_model = margin_note_model(
            &page.lines,
            margin_labels,
            page.width,
            note_size,
            if margin_labels.len() >= 3 { 3 } else { 2 },
        );
        let margin_side = margin_model.map(|model| {
            usize::from(
                margin_labels
                    .iter()
                    .filter(|index| line_center_x(&page.lines[**index]) >= model.split_x)
                    .count()
                    * 2
                    >= margin_labels.len(),
            )
        });
        let endnote_page = has_endnote_heading
            || continuing_endnotes
            || (separator.is_none()
                && margin_model.is_none()
                && !candidates.is_empty()
                && early_labels
                && best_run >= 3
                && label_size.is_some_and(|size| size <= body_size * 0.90));
        let labels: Vec<usize> = candidates[selected_start..]
            .iter()
            .filter_map(|(index, prefix, typographic)| {
                let size = line_sizes[*index];
                let reference_backed = page.lines[*index].bbox[1] >= page.height * 0.48
                    && size > 0.0
                    && (size <= body_size * 0.90 || prefix.label.chars().all(is_note_symbol))
                    && has_prior_reference(page, &prefix.label, page.lines[*index].bbox[1]);
                let in_margin = margin_model.zip(margin_side).is_some_and(|(model, side)| {
                    usize::from(line_center_x(&page.lines[*index]) >= model.split_x) == side
                        && size <= note_size * 0.90
                        && line_width(&page.lines[*index]) <= page.width * 0.30
                });
                (endnote_page
                    || in_margin
                    || (page.lines[*index].bbox[1] >= page.height * 0.48
                        && (separator
                            .is_some_and(|cut| page.lines[*index].bbox[1] >= cut - tolerance)
                            || (page_columns.kind == "two_column" && *typographic)
                            || (separator.is_none()
                                && (*typographic
                                    || reference_backed
                                    || (lower_note_run
                                        && !recognized_page(page)
                                        && page.lines[*index].bbox[1] >= page.height * 0.48
                                        && size > 0.0
                                        && size <= body_size * 0.90))))))
                    .then_some(*index)
            })
            .collect();
        if endnote_page && !labels.is_empty() {
            let first_label = labels
                .iter()
                .map(|index| page.lines[*index].bbox[1])
                .min_by(f64::total_cmp)
                .unwrap_or(page.height);
            for line in &mut page.lines {
                let text = line.text.trim();
                let letter_count = text
                    .chars()
                    .filter(|character| character.is_alphabetic())
                    .count();
                if !matches!(line.region_type.as_str(), "header" | "footer")
                    && line.bbox[1] < page.height * 0.08
                    && line.bbox[1] < first_label
                    && line.bbox[0] >= page.width * 0.15
                    && line.bbox[2] <= page.width * 0.85
                    && letter_count >= 4
                    && text.chars().count() <= 100
                    && all_caps(text)
                    && line_start_label_prefix(text).is_none()
                {
                    line.region_type = "header".to_owned();
                }
            }
        }
        let eligible_top = || {
            page.lines
                .iter()
                .filter(|line| {
                    !matches!(line.region_type.as_str(), "header" | "footer")
                        && !line.exclude_from_body
                })
                .map(|line| line.bbox[1])
                .min_by(f64::total_cmp)
        };
        let selected_sizes: Vec<f64> = labels
            .iter()
            .map(|index| line_sizes[*index])
            .filter(|size| *size > 0.0)
            .collect();
        let selected_size = (!selected_sizes.is_empty()).then(|| median(selected_sizes));
        let separator_starts_small_text = separator.is_some_and(|cut| {
            !contents_pages.contains(&page.index)
                && page
                    .lines
                    .iter()
                    .enumerate()
                    .filter(|(index, line)| {
                        !table_cells.contains(index)
                            && !table_notes.contains(index)
                            && !line.exclude_from_body
                            && !matches!(line.region_type.as_str(), "header" | "footer")
                            && line.bbox[1] >= cut - tolerance
                    })
                    .min_by(|(_, left), (_, right)| left.bbox[1].total_cmp(&right.bbox[1]))
                    .is_some_and(|(index, line)| {
                        line.bbox[1] - cut <= page.height * 0.06
                            && line_sizes[index] > 0.0
                            && line_sizes[index] <= body_size * 0.90
                    })
        });
        let note_cut = if label_free_continuation {
            eligible_top()
        } else if labels.is_empty() {
            None
        } else {
            let first_label = labels
                .iter()
                .map(|index| page.lines[*index].bbox[1])
                .min_by(f64::total_cmp)
                .unwrap_or(page.height);
            if endnote_page && expected_endnote.is_some() {
                eligible_top()
            } else if separator.is_some_and(|cut| {
                0.0 <= first_label - cut
                    && (first_label - cut <= page.height * 0.15
                        || (separator_starts_small_text
                            && selected_size.is_some_and(|size| size <= body_size * 0.90)))
            }) {
                separator
            } else {
                let confident = (first_label >= page.height * 0.58 || endnote_page)
                    && selected_size.is_some_and(|size| size <= body_size * 0.90);
                if !confident {
                    let mut diagnostic = Diagnostic::warning(
                        "FOOTNOTE_REGION_UNCERTAIN",
                        "Footnote region inferred from weak label geometry without a separator.",
                        Some(page.index),
                    );
                    if let Some(index) = labels.iter().min_by(|left, right| {
                        page.lines[**left].bbox[1].total_cmp(&page.lines[**right].bbox[1])
                    }) {
                        diagnostic.line_ids.push(page.lines[*index].id.clone());
                    }
                    diagnostics.push(diagnostic);
                }
                Some(first_label)
            }
        };
        let note_columns = margin_model.unwrap_or(page_columns);
        let column_cuts = if (note_columns.kind == "two_column" || margin_model.is_some())
            && (!endnote_page || has_endnote_heading)
        {
            let mut cuts = [None::<f64>; 2];
            for index in &labels {
                let label = &page.lines[*index];
                let column = usize::from(line_center_x(label) >= note_columns.split_x);
                let label_top = aligned_note_body_index(&page.lines, *index)
                    .map(|body| band_geometry_top(&page.lines[body]))
                    .unwrap_or_else(|| band_geometry_top(label))
                    .min(band_geometry_top(label));
                cuts[column] = Some(cuts[column].map_or(label_top, |cut| cut.min(label_top)));
            }
            if cuts.iter().all(Option::is_some)
                || (note_columns.kind == "margin_column" && cuts.iter().any(Option::is_some))
            {
                if let Some(heading_y) = endnote_heading_y {
                    for (column, cut) in cuts.iter_mut().enumerate() {
                        if cut.is_some_and(|value| value < heading_y) {
                            *cut = page
                                .lines
                                .iter()
                                .filter(|line| {
                                    !matches!(line.region_type.as_str(), "header" | "footer")
                                        && !line.exclude_from_body
                                        && usize::from(line_center_x(line) >= note_columns.split_x)
                                            == column
                                })
                                .map(|line| line.bbox[1])
                                .min_by(f64::total_cmp);
                        }
                    }
                }
                Some(cuts)
            } else {
                None
            }
        } else {
            None
        };
        for (index, line) in page.lines.iter_mut().enumerate() {
            if line.exclude_from_body {
                continue;
            }
            let is_note = endnote_heading_source.is_some_and(|start| {
                column_cuts.is_none() && endnote_page && line.source_index >= start
            }) || column_cuts.as_ref().map_or_else(
                || note_cut.is_some_and(|cut| line.bbox[1] >= cut),
                |cuts| {
                    let column = usize::from(line_center_x(line) >= note_columns.split_x);
                    cuts[column].is_some_and(|cut| line.bbox[1] >= cut)
                },
            );
            if endnote_heading(&line.text) {
                line.region_type = "heading".to_owned();
                line.note_region_mode.clear();
            } else if table_notes.contains(&index) {
                line.region_type = "footnote".to_owned();
                line.note_region_mode = "footnote".to_owned();
            } else if table_cells.contains(&index) {
                line.region_type = "body".to_owned();
                line.note_region_mode.clear();
            } else if matches!(line.region_type.as_str(), "header" | "footer") {
                continue;
            } else if legal_pdf_core::is_heading_region(
                evidence
                    .source_regions
                    .as_ref()
                    .and_then(|regions| regions.get(&line.id))
                    .map(String::as_str)
                    .unwrap_or(line.region_type.as_str()),
            ) {
                // Source roles survive the font-size fallback, including recognized and tagged text.
                line.region_type = "heading".to_owned();
                line.note_region_mode.clear();
            } else if paragraph_labels.contains(&line.id) {
                line.region_type = "body".to_owned();
                line.note_region_mode.clear();
            } else if is_note {
                line.region_type = "footnote".to_owned();
                line.note_region_mode =
                    if endnote_page { "endnote" } else { "footnote" }.to_owned();
            } else {
                line.region_type = "body".to_owned();
            }
        }
        if !page
            .lines
            .iter()
            .any(|line| line.note_region_mode == "endnote")
        {
            expected_endnote = None;
            continuing_size = None;
        } else {
            if let Some(last) = page
                .lines
                .iter()
                .rev()
                .filter(|line| line.note_region_mode == "endnote")
                .filter_map(|line| label_prefix(&line.text)?.label.parse::<u32>().ok())
                .next()
            {
                expected_endnote = Some(last + 1);
            }
            let size = median(
                page.lines
                    .iter()
                    .enumerate()
                    .filter(|(_, line)| line.note_region_mode == "endnote")
                    .map(|(index, _)| line_sizes[index])
                    .filter(|size| *size > 0.0)
                    .collect(),
            );
            if size > 0.0 {
                continuing_size = Some(size);
            }
        }
    }
    verify_endnotes(pages);
    diagnostics
}

/// Endnotes are notes: the body before them refers to them. Each run of consecutive pages read as
/// endnotes keeps them only where a notes heading opens it, or where references earlier in the
/// document (raised numbers, detached references) name at least half of its note numbers; else
/// its lines are the body's, a numbered sequence the page-by-page reading took for notes. On the
/// structure gold (batches 5-7) endnote runs held 126 note lines and 5,472 lines of body; the
/// references keep 49 of the notes and give back 5,429 of the body.
pub(super) fn verify_endnotes(pages: &mut [Page]) {
    let endnote = |page: &Page| page.lines.iter().any(|line| line.note_region_mode == "endnote");
    let mut slot = 0;
    while slot < pages.len() {
        if !endnote(&pages[slot]) {
            slot += 1;
            continue;
        }
        let mut end = slot + 1;
        while end < pages.len() && endnote(&pages[end]) && pages[end].index == pages[end - 1].index + 1 {
            end += 1;
        }
        let run = &pages[slot..end];
        let headed = run.iter().flat_map(|page| &page.lines).any(|line| endnote_heading(&line.text));
        let labels: BTreeSet<String> = run
            .iter()
            .flat_map(|page| &page.lines)
            .filter(|line| line.note_region_mode == "endnote")
            .filter_map(|line| label_prefix(&line.text).map(|prefix| prefix.label))
            .collect();
        let first_note = pages[slot]
            .lines
            .iter()
            .filter(|line| line.note_region_mode == "endnote")
            .map(|line| line.bbox[1])
            .min_by(f64::total_cmp)
            .unwrap_or(0.0);
        let referenced = labels
            .iter()
            .filter(|label| {
                pages[..slot].iter().any(|page| has_prior_reference(page, label, f64::INFINITY))
                    || has_prior_reference(&pages[slot], label, first_note)
            })
            .count();
        if !headed && referenced * 2 < labels.len().max(1) {
            for line in pages[slot..end].iter_mut().flat_map(|page| &mut page.lines) {
                if line.note_region_mode == "endnote" {
                    line.region_type = "body".to_owned();
                    line.note_region_mode.clear();
                }
            }
        }
        slot = end;
    }
}

pub(super) fn infer_note_region_modes(pages: &mut [Page]) {
    let mut prior_note_page = false;
    let mut active_endnotes = false;
    let mut expected_endnote = None::<u32>;
    let mut previous_page = None;
    for page in pages {
        if previous_page.is_some_and(|prior: usize| prior.checked_add(1) != Some(page.index)) {
            prior_note_page = false;
            active_endnotes = false;
            expected_endnote = None;
        }
        previous_page = Some(page.index);
        let footnote_indexes: Vec<usize> = page
            .lines
            .iter()
            .enumerate()
            .filter_map(|(index, line)| (line.region_type == "footnote").then_some(index))
            .collect();
        let heading = page.lines.iter().any(|line| endnote_heading(&line.text));
        let explicit_endnote = footnote_indexes
            .iter()
            .any(|&index| page.lines[index].note_region_mode == "endnote");
        let numbers: Vec<u32> = footnote_indexes
            .iter()
            .filter_map(|&index| line_start_label_prefix(&page.lines[index].text))
            .filter_map(|prefix| normalize_label(&prefix.label).parse().ok())
            .collect();
        let continues_endnotes = active_endnotes
            && (numbers.is_empty()
                || expected_endnote.is_some_and(|expected| numbers[0] == expected));
        if !footnote_indexes.is_empty() && (heading || explicit_endnote || continues_endnotes) {
            for &index in &footnote_indexes {
                page.lines[index].note_region_mode = "endnote".to_owned();
            }
            active_endnotes = true;
            if let Some(number) = numbers.last() {
                expected_endnote = number.checked_add(1);
            }
        } else if !footnote_indexes.is_empty() {
            let explicit_footnote = footnote_indexes
                .iter()
                .any(|&index| page.lines[index].note_region_mode == "footnote");
            let mode = if explicit_footnote {
                "footnote"
            } else if prior_note_page {
                "footnote_continuation"
            } else {
                ""
            };
            if !mode.is_empty() {
                for &index in &footnote_indexes {
                    if page.lines[index].note_region_mode.is_empty() {
                        page.lines[index].note_region_mode = mode.to_owned();
                    }
                }
            }
            active_endnotes = false;
            expected_endnote = None;
        } else {
            active_endnotes = false;
            expected_endnote = None;
        }
        prior_note_page = !footnote_indexes.is_empty();
    }
}
