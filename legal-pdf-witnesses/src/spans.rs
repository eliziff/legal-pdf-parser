//! Native boundary/role prediction over source starts, including starts inside physical lines.
use crate::witness::{local_span_values, Table, NAMES};
use legal_pdf_core::model::Page;
use legal_structure_model::{
    DocumentStructure, PrimaryRole, ScalarRange, ScalarText, SourceFragment,
};
use std::collections::BTreeMap;

pub const SPAN_SCHEMA: &str = "source-span-role-v7";
pub const ROLE_NAMES: &[&str] = &[
    "no_boundary",
    "text",
    "heading",
    "caption",
    "instruction",
    "headnote",
    "marginal_note",
    "furniture",
    "page_label",
    "close_unowned",
];
pub(crate) const START_NAMES: &[&str] = &[
    "line_start",
    "typography_start",
    "marker_start",
    "quotation_boundary",
    "line_fraction",
    "remaining_fraction",
    "type_known",
    "line_end",
    "word_boundary",
];

#[derive(Debug, serde::Serialize)]
pub struct SpanProposal {
    pub id: String,
    /// Unicode scalar source membership, before public UTF-16 projection.
    pub range: ScalarRange,
    pub role: PrimaryRole,
    pub line_ids: Vec<String>,
    pub page_indexes: Vec<usize>,
}

#[derive(serde::Serialize)]
pub struct SpanTrainingRow {
    pub id: String,
    pub start: usize,
    pub features: Vec<f32>,
    pub class: usize,
}

#[derive(serde::Serialize)]
pub struct SpanTrainingRows {
    pub schema: &'static str,
    pub feature_names: Vec<String>,
    pub classes: &'static [&'static str],
    pub rows: Vec<SpanTrainingRow>,
    pub unavailable_gold_starts: Vec<usize>,
    pub unavailable_gold_ends: Vec<usize>,
    pub unannotated_starts: Vec<usize>,
}

pub(crate) struct Candidate {
    pub(crate) id: String,
    pub(crate) start: usize,
    pub(crate) marker: Option<String>,
    pub(crate) numbering: Vec<f32>,
    pub(crate) features: Vec<f32>,
}

fn role(class: usize) -> Option<PrimaryRole> {
    Some(match class {
        1 => PrimaryRole::Text,
        2 => PrimaryRole::Heading,
        3 => PrimaryRole::Caption,
        4 => PrimaryRole::Instruction,
        5 => PrimaryRole::Headnote,
        6 => PrimaryRole::MarginalNote,
        7 => PrimaryRole::Furniture,
        8 => PrimaryRole::PageLabel,
        _ => return None,
    })
}

pub fn feature_names() -> Vec<String> {
    START_NAMES
        .iter()
        .map(|name| format!("start_{name}"))
        .chain(crate::numbering::SOURCE_NAMES.iter().map(|name| format!("numbering_{name}")))
        .chain(crate::profile::NAMES.iter().map(|name| (*name).to_owned()))
        .chain(NAMES.iter().map(|name| format!("source_{name}")))
        .chain(NAMES.iter().map(|name| format!("previous_{name}")))
        .chain(NAMES.iter().map(|name| format!("next_{name}")))
        .collect()
}

fn candidates(
    text: &str,
    pages: &[Page],
    source: &Table,
    fragments: &[SourceFragment],
    profile: Option<legal_structure_model::DetectionProfile>,
) -> Result<Vec<Candidate>, String> {
    let _ = pages;
    source_candidates(text, source, fragments, profile)
}

/// Shared physical source boundaries, independent of selected roles and scopes.
pub(crate) fn source_candidates(
    text: &str,
    source: &Table,
    fragments: &[SourceFragment],
    profile: Option<legal_structure_model::DetectionProfile>,
) -> Result<Vec<Candidate>, String> {
    let coordinates = ScalarText::new(text);
    let mut bound = source.clone();
    crate::witness::bind_source_fragments(&mut bound, text, fragments)?;
    let rows = bound.by_id();
    let mut starts = BTreeMap::<usize, (usize, bool, bool, bool, bool)>::new();
    let mut ranges = Vec::new();
    for (position, fragment) in fragments.iter().enumerate() {
        let start = coordinates
            .scalar_at_utf16(fragment.range.start)
            .ok_or("Invalid physical fragment start")?;
        let end = coordinates
            .scalar_at_utf16(fragment.range.end)
            .ok_or("Invalid physical fragment end")?;
        if start > end {
            return Err("Reversed physical fragment".into());
        }
        ranges.push(ScalarRange { start, end });
        let Some(line) = rows.get(fragment.id.as_str()) else {
            return Err(format!("Physical fragment {} is absent", fragment.id));
        };
        if coordinates.slice(start..end) != Some(line.text.as_str()) {
            return Err(format!(
                "Corrected physical text differs for {}",
                fragment.id
            ));
        }
        starts.insert(start, (position, false, false, false, false));
        starts
            .entry(end)
            .or_insert((position, false, false, false, false));
        if line.text.trim().is_empty() { continue; }
        for &(a, b, _) in &line.words {
            if a <= b && b <= end - start {
                for local in [a, b] {
                    starts.entry(start + local)
                        .or_insert((position, false, false, false, true)).4 = true;
                }
            }
        }
        // Punctuation supplies candidate coordinates, never an ownership verdict.
        for (offset, character) in line.text.chars().enumerate() {
            if matches!(character, '"' | '\'' | '“' | '”' | '‘' | '’' | '«' | '»') {
                for local in [offset, offset + 1] {
                    if local < end - start {
                        starts
                            .entry(start + local)
                            .or_insert((position, false, false, true, false))
                            .3 = true;
                    }
                }
            }
        }
        for span in &line.spans {
            for local in [span.start, span.end] {
                if local > 0 && local <= end - start {
                    starts
                        .entry(start + local)
                        .or_insert((position, true, false, false, false))
                        .1 = true;
                }
            }
        }
    }
    let mut markers = BTreeMap::<usize, String>::new();
    for marker in legal_pdf_core::structure_analysis()
        .detect_structure_candidate_runs(text)
        .into_iter()
        .flat_map(|run| run.markers)
    {
        if let Some(position) = ranges.iter().position(|range| {
            range.start <= marker.marker_range.start && marker.marker_range.start < range.end
        }) {
            if let Some(text) = coordinates.slice(marker.marker_range.start..marker.marker_range.end) {
                markers.entry(marker.marker_range.start).and_modify(|prior| {
                    if text.len() > prior.len() { *prior = text.to_owned(); }
                }).or_insert_with(|| text.to_owned());
            }
            starts
                .entry(marker.marker_range.start)
                .or_insert((position, false, true, false, false))
                .2 = true;
        }
    }
    let numbering = crate::numbering::source_values(&markers);
    let zeros = vec![0.0; NAMES.len()];
    starts
        .into_iter()
        .map(|(start, (position, typography, marker, quotation, word))| {
            let fragment = &fragments[position];
            let range = ranges[position];
            let row = rows.get(fragment.id.as_str()).copied();
            let body = coordinates
                .slice(start..range.end)
                .ok_or("Candidate starts outside source fragment")?;
            let local = start - range.start;
            let total = (range.end - range.start).max(1) as f32;
            let source_values = local_span_values(row, body, local, range.end - range.start);
            let mut features = vec![
                if local == 0 { 1.0 } else { 0.0 },
                if typography { 1.0 } else { 0.0 },
                if marker { 1.0 } else { 0.0 },
                if quotation { 1.0 } else { 0.0 },
                local as f32 / total,
                (range.end - start) as f32 / total,
                if source_values[crate::witness::index("size").unwrap()] > 0.0 {
                    1.0
                } else {
                    0.0
                },
                if start == range.end { 1.0 } else { 0.0 },
                if word { 1.0 } else { 0.0 },
            ];
            let number_values = numbering.get(&start).cloned()
                .unwrap_or_else(|| vec![0.0; crate::numbering::SOURCE_NAMES.len()]);
            features.extend_from_slice(&number_values);
            features.extend(crate::profile::values(profile));
            features.extend(source_values);
            for neighbour in [
                position.checked_sub(1),
                (position + 1 < fragments.len()).then_some(position + 1),
            ] {
                features.extend(
                    neighbour
                        .and_then(|position| rows.get(fragments[position].id.as_str()))
                        .map_or(&zeros, |row| &row.values),
                );
            }
            Ok(Candidate {
                id: format!("{}@{}", fragment.id, local),
                start,
                marker: markers.get(&start).cloned(),
                numbering: number_values,
                features,
            })
        })
        .collect()
}

pub fn predict_spans(
    text: &str,
    pages: &[Page],
    source: &Table,
    fragments: &[SourceFragment],
    profile: Option<legal_structure_model::DetectionProfile>,
) -> Result<Option<Vec<SpanProposal>>, String> {
    let Some(model) = crate::model::span_model() else {
        return Ok(None);
    };
    let coordinates = ScalarText::new(text);
    let mut selected = Vec::new();
    for candidate in candidates(text, pages, source, fragments, profile)? {
        let scores = model.scores(&candidate.features);
        let class = scores
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map_or(0, |(class, _)| class);
        if class != 0 {
            selected.push((candidate, role(class)));
        }
    }
    let mut proposals = Vec::new();
    for (position, (candidate, role)) in selected.iter().enumerate() {
        let Some(role) = role else {
            continue;
        };
        let end = selected
            .get(position + 1)
            .map_or(coordinates.len(), |(next, _)| next.start);
        if end <= candidate.start {
            continue;
        }
        let mut line_ids = Vec::new();
        let mut page_indexes = Vec::new();
        for fragment in fragments {
            let start = coordinates
                .scalar_at_utf16(fragment.range.start)
                .ok_or("Invalid physical source start")?;
            let stop = coordinates
                .scalar_at_utf16(fragment.range.end)
                .ok_or("Invalid physical source end")?;
            if start < end && candidate.start < stop {
                line_ids.push(fragment.id.clone());
                if !page_indexes.contains(&fragment.page_index) {
                    page_indexes.push(fragment.page_index);
                }
            }
        }
        proposals.push(SpanProposal {
            id: format!("span:{}", candidate.id),
            range: ScalarRange {
                start: candidate.start,
                end,
            },
            role: *role,
            line_ids,
            page_indexes,
        });
    }
    Ok(Some(proposals))
}

/// Only annotated membership supplies negative boundaries. Unannotated source is not a negative class.
pub fn span_training_rows(
    structure: DocumentStructure,
    pages: &[Page],
    source: &Table,
) -> Result<SpanTrainingRows, String> {
    if structure.offset_unit.as_ref() != "utf16" {
        return Err("Gold must use explicit UTF-16 export coordinates".into());
    }
    let coordinates = ScalarText::new(&structure.text);
    let mut expected = BTreeMap::<usize, usize>::new();
    let mut coverage = Vec::new();
    for node in &structure.nodes {
        let Some(role_value) = node.role else {
            continue;
        };
        let class = (1..ROLE_NAMES.len())
            .find(|&class| role(class) == Some(role_value))
            .ok_or("Unknown primary role")?;
        for part in if node.ranges.is_empty() {
            std::slice::from_ref(&node.range)
        } else {
            &node.ranges
        } {
            let part = ScalarRange {
                start: coordinates
                    .scalar_at_utf16(part.start)
                    .ok_or("Invalid gold membership start")?,
                end: coordinates
                    .scalar_at_utf16(part.end)
                    .ok_or("Invalid gold membership end")?,
            };
            if expected
                .insert(part.start, class)
                .is_some_and(|prior| prior != class && prior != 9)
            {
                return Err("Different gold roles share one structural start".into());
            }
            coverage.push(part);
        }
    }
    let ends = coverage
        .iter()
        .map(|part| part.end)
        .collect::<std::collections::BTreeSet<_>>();
    for end in &ends {
        expected.entry(*end).or_insert(9);
    }
    let mut rows = Vec::new();
    let mut unannotated_starts = Vec::new();
    let candidates = candidates(
        &structure.text,
        pages,
        source,
        &structure.source_fragments,
        structure.profile,
    )?;
    let starts = candidates
        .iter()
        .map(|candidate| candidate.start)
        .collect::<std::collections::HashSet<_>>();
    let unavailable_gold_ends = ends
        .iter()
        .filter(|end| !starts.contains(end))
        .copied()
        .collect();
    let unavailable_gold_starts = expected
        .keys()
        .filter(|start| !starts.contains(start) && expected[start] != 9)
        .copied()
        .collect();
    for candidate in candidates {
        let class = expected.get(&candidate.start).copied().or_else(|| {
            coverage
                .iter()
                .any(|range| range.start <= candidate.start && candidate.start < range.end)
                .then_some(0)
        });
        if let Some(class) = class {
            rows.push(SpanTrainingRow {
                id: candidate.id,
                start: candidate.start,
                features: candidate.features,
                class,
            });
        } else {
            unannotated_starts.push(candidate.start);
        }
    }
    Ok(SpanTrainingRows {
        schema: SPAN_SCHEMA,
        feature_names: feature_names(),
        classes: ROLE_NAMES,
        rows,
        unavailable_gold_starts,
        unavailable_gold_ends,
        unannotated_starts,
    })
}
