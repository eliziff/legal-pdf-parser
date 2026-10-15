//! Exact fragment membership is inferred after scope selection, before outline assembly.
use crate::witness::{Table, NAMES};
use legal_structure_model::{
    DocumentStructure, NodeKind, ScalarRange, ScalarText, SourceFragment, StructureNode,
};
use std::collections::HashMap;

pub const MEMBERSHIP_SCHEMA: &str = "source-fragment-membership-v4";
const PAIR_NAMES: &[&str] = &[
    "new_unit",
    "fragments_since_seen",
    "source_gap",
    "pages_apart",
    "same_style",
    "size_delta",
    "indent_delta",
    "previous_ends_sentence",
    "previous_ends_hyphen",
    "current_starts_lowercase",
    "current_marked",
    "candidate_marked",
    "same_marker",
    "role_text",
    "role_heading",
    "role_caption",
    "role_instruction",
    "role_headnote",
    "role_marginal_note",
    "role_furniture",
    "role_page_label",
];
pub fn feature_names() -> Vec<String> {
    PAIR_NAMES
        .iter()
        .map(|name| (*name).to_owned())
        .chain(crate::profile::NAMES.iter().map(|name| (*name).to_owned()))
        .chain(NAMES.iter().map(|name| format!("source_{name}")))
        .chain(NAMES.iter().map(|name| format!("candidate_{name}")))
        .collect()
}
#[derive(serde::Serialize)]
pub struct MembershipTrainingRow {
    pub unit: String,
    pub candidate: Option<String>,
    pub features: Vec<f32>,
    pub positive: bool,
}
#[derive(serde::Serialize)]
pub struct MembershipTrainingRows {
    pub schema: &'static str,
    pub feature_names: Vec<String>,
    pub rows: Vec<MembershipTrainingRow>,
}
fn atomic(node: &StructureNode) -> bool {
    node.role.is_some()
        && matches!(
            node.kind,
            NodeKind::Heading
                | NodeKind::Paragraph
                | NodeKind::Prose
                | NodeKind::ListItem
                | NodeKind::Section
        )
}
fn parts(node: &StructureNode) -> &[ScalarRange] {
    if node.ranges.is_empty() {
        std::slice::from_ref(&node.range)
    } else {
        &node.ranges
    }
}

/// Keep each logical gold identity separately from its chronological physical fragments.
pub(crate) fn gold_fragments(
    structure: &DocumentStructure,
) -> Result<(Vec<StructureNode>, HashMap<String, String>), String> {
    if structure.offset_unit.as_ref() != "utf16" {
        return Err("Gold must use explicit UTF-16 coordinates".into());
    }
    let coordinates = ScalarText::new(&structure.text);
    let scalar = |range: ScalarRange| -> Result<ScalarRange, String> {
        Ok(ScalarRange {
            start: coordinates
                .scalar_at_utf16(range.start)
                .ok_or("Invalid gold fragment start")?,
            end: coordinates
                .scalar_at_utf16(range.end)
                .ok_or("Invalid gold fragment end")?,
        })
    };
    let physical = structure
        .source_fragments
        .iter()
        .map(|fragment| Ok((fragment, scalar(fragment.range)?)))
        .collect::<Result<Vec<_>, String>>()?;
    let mut nodes = Vec::new();
    let mut identities = HashMap::new();
    for node in structure.nodes.iter().filter(|node| atomic(node)) {
        let mut own = parts(node)
            .iter()
            .copied()
            .map(scalar)
            .collect::<Result<Vec<_>, _>>()?;
        own.sort_by_key(|part| (part.start, part.end));
        for (ordinal, part) in own.into_iter().enumerate() {
            if part.start >= part.end {
                return Err("Empty gold membership fragment".into());
            }
            let mut fragment = node.clone();
            fragment.id = format!("{}@{}", node.id, part.start);
            fragment.range = part;
            fragment.ranges = vec![part];
            fragment.heading_range = (fragment.kind == NodeKind::Heading).then_some(part);
            fragment.marker_range = node
                .marker_range
                .map(scalar)
                .transpose()?
                .filter(|range| part.start <= range.start && range.end <= part.end);
            fragment.content_start = node
                .content_start
                .and_then(|start| coordinates.scalar_at_utf16(start))
                .filter(|start| part.start <= *start && *start < part.end);
            if ordinal != 0 {
                fragment.label = None;
                fragment.locator_kind = None;
            }
            fragment.line_ids.clear();
            fragment.page_indexes.clear();
            for (source, range) in &physical {
                if range.start < part.end && part.start < range.end {
                    fragment.line_ids.push(source.id.clone());
                    if !fragment.page_indexes.contains(&source.page_index) {
                        fragment.page_indexes.push(source.page_index);
                    }
                }
            }
            identities.insert(fragment.id.clone(), node.id.clone());
            nodes.push(fragment);
        }
    }
    nodes.sort_by_key(|node| (node.range.start, node.range.end));
    Ok((nodes, identities))
}
fn flag(value: bool) -> f32 {
    if value {
        1.0
    } else {
        0.0
    }
}

fn select(
    nodes: &[StructureNode],
    text: &str,
    source: &Table,
    fragments: &[SourceFragment],
    profile: Option<legal_structure_model::DetectionProfile>,
    gold: Option<&HashMap<String, String>>,
) -> Result<(HashMap<String, String>, Vec<MembershipTrainingRow>), String> {
    let model = crate::model::membership_model();
    if gold.is_none() && model.is_none() {
        return Ok((HashMap::new(), Vec::new()));
    }
    let mut table = source.clone();
    crate::witness::bind_source_fragments(&mut table, text, fragments)?;
    let rows = table.by_id();
    let coordinates = ScalarText::new(text);
    let markers = legal_pdf_core::structure_analysis()
        .detect_structure_candidate_runs(text)
        .into_iter()
        .flat_map(|run| run.markers)
        .filter_map(|candidate| {
            let printed =
                coordinates.slice(candidate.marker_range.start..candidate.marker_range.end)?;
            legal_pdf_core::structure_analysis()
                .numbering_marker(printed, None)
                .map(|marker| (candidate.marker_range.start, marker))
        })
        .collect::<HashMap<_, _>>();
    let mut order = nodes.iter().filter(|node| atomic(node)).collect::<Vec<_>>();
    order.sort_by_key(|node| (node.range.start, node.range.end));
    // Each logical unit keeps its stable first anchor and most recent observed fragment.
    let mut units = Vec::<(&StructureNode, &StructureNode, usize, Vec<f32>)>::new();
    let mut links = HashMap::new();
    let mut emitted = Vec::new();
    let zeros = vec![0.0; NAMES.len()];
    for (position, node) in order.into_iter().enumerate() {
        let evidence = node
            .line_ids
            .first()
            .and_then(|id| rows.get(id.as_str()))
            .copied();
        let values = crate::witness::node_values(evidence, node, &coordinates);
        let body = crate::witness::membership_text(node, &coordinates).unwrap_or_default();
        let marker = markers.get(&node.range.start);
        let mut candidates = vec![None];
        candidates.extend(
            units
                .iter()
                .enumerate()
                .filter(|(_, (first, last, _, _))| {
                    first.role == node.role
                        && first.ownership == node.ownership
                        && last.range.end <= node.range.start
                })
                .map(|(index, _)| Some(index)),
        );
        let expected = gold.map(|identities| {
            units
                .iter()
                .position(|(first, _, _, _)| identities.get(&first.id) == identities.get(&node.id))
        });
        if let Some(Some(expected)) = expected {
            if !candidates.contains(&Some(expected)) {
                return Err("One gold logical unit crosses role or ownership identities".into());
            }
        }
        let mut best = (None, f64::NEG_INFINITY);
        for candidate in candidates {
            let prior = candidate.map(|index| &units[index]);
            let previous = prior.map_or(&zeros, |(_, _, _, values)| values);
            let prior_body = prior
                .map(|(_, last, _, _)| {
                    crate::witness::membership_text(last, &coordinates).unwrap_or_default()
                })
                .unwrap_or_default();
            let prior_marker = prior.and_then(|(first, _, _, _)| markers.get(&first.range.start));
            let at = |name| crate::witness::index(name).unwrap();
            let mut features =
                vec![
                    flag(prior.is_none()),
                    prior.map_or(0.0, |(_, _, seen, _)| (position - seen) as f32),
                    prior.map_or(0.0, |(_, last, _, _)| {
                        node.range.start.saturating_sub(last.range.end) as f32
                    }),
                    prior.map_or(0.0, |(_, last, _, _)| {
                        node.page_indexes
                            .first()
                            .copied()
                            .unwrap_or(0)
                            .abs_diff(last.page_indexes.last().copied().unwrap_or(0))
                            as f32
                    }),
                    flag(
                        prior.is_some()
                            && values[at("size")] == previous[at("size")]
                            && values[at("bold")] == previous[at("bold")],
                    ),
                    values[at("size")] - previous[at("size")],
                    values[at("indent")] - previous[at("indent")],
                    flag(prior_body.trim_end().ends_with(['.', '!', '?'])),
                    flag(prior_body.trim_end().ends_with(['-', '‐', '‑'])),
                    flag(
                        body.trim_start()
                            .chars()
                            .next()
                            .is_some_and(char::is_lowercase),
                    ),
                    flag(marker.is_some()),
                    flag(prior_marker.is_some()),
                    flag(marker.as_ref().zip(prior_marker.as_ref()).is_some_and(
                        |(left, right)| {
                            left.family == right.family
                                && left.ordinal == right.ordinal
                                && left.prefix == right.prefix
                        },
                    )),
                ];
            for role in [
                legal_structure_model::PrimaryRole::Text,
                legal_structure_model::PrimaryRole::Heading,
                legal_structure_model::PrimaryRole::Caption,
                legal_structure_model::PrimaryRole::Instruction,
                legal_structure_model::PrimaryRole::Headnote,
                legal_structure_model::PrimaryRole::MarginalNote,
                legal_structure_model::PrimaryRole::Furniture,
                legal_structure_model::PrimaryRole::PageLabel,
            ] {
                features.push(flag(node.role == Some(role)));
            }
            features.extend(crate::profile::values(profile));
            features.extend_from_slice(&values);
            features.extend_from_slice(previous);
            if let Some(expected) = expected {
                emitted.push(MembershipTrainingRow {
                    unit: node.id.clone(),
                    candidate: prior.map(|(first, _, _, _)| first.id.clone()),
                    features,
                    positive: expected == candidate,
                });
            } else if let Some(model) = model {
                let score = model.scores(&features)[0];
                if score > best.1 {
                    best = (candidate, score);
                }
            }
        }
        let chosen = expected.unwrap_or(best.0);
        if let Some(index) = chosen {
            links.insert(node.id.clone(), units[index].0.id.clone());
            units[index].1 = node;
            units[index].2 = position;
            units[index].3 = values;
        } else {
            units.push((node, node, position, values));
        }
    }
    Ok((links, emitted))
}

/// Membership neither asserts a manuscript join nor deletes a printed hyphen.
pub fn infer_membership(
    nodes: &mut Vec<StructureNode>,
    text: &str,
    source: &Table,
    fragments: &[SourceFragment],
    profile: Option<legal_structure_model::DetectionProfile>,
) -> Result<(), String> {
    let (links, _) = select(nodes, text, source, fragments, profile, None)?;
    if links.is_empty() {
        return Ok(());
    }
    let coordinates = ScalarText::new(text);
    let indexes = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.id.clone(), index))
        .collect::<HashMap<_, _>>();
    for (fragment, anchor) in &links {
        let fragment = nodes[indexes[fragment]].clone();
        let target = &mut nodes[indexes[anchor]];
        if target.ranges.is_empty() {
            target.ranges.push(target.range);
        }
        target.ranges.extend_from_slice(parts(&fragment));
        target.ranges.sort_by_key(|range| (range.start, range.end));
        target.range.end = target.range.end.max(fragment.range.end);
        for id in fragment.line_ids {
            if !target.line_ids.contains(&id) {
                target.line_ids.push(id);
            }
        }
        for page in fragment.page_indexes {
            if !target.page_indexes.contains(&page) {
                target.page_indexes.push(page);
            }
        }
        if target.kind == NodeKind::Heading {
            target.heading_range = None;
            target.label = crate::witness::membership_text(target, &coordinates);
        }
    }
    nodes.retain(|node| !links.contains_key(&node.id));
    for node in nodes {
        if links.values().any(|anchor| anchor == &node.id) {
            node.line_ids.clear();
            node.page_indexes.clear();
            for fragment in fragments {
                let start = coordinates
                    .scalar_at_utf16(fragment.range.start)
                    .ok_or("Invalid physical source start")?;
                let end = coordinates
                    .scalar_at_utf16(fragment.range.end)
                    .ok_or("Invalid physical source end")?;
                if node
                    .ranges
                    .iter()
                    .any(|part| start < part.end && part.start < end)
                {
                    node.line_ids.push(fragment.id.clone());
                    if !node.page_indexes.contains(&fragment.page_index) {
                        node.page_indexes.push(fragment.page_index);
                    }
                }
            }
        }
        if let Some(parent) = node.parent_id.as_mut() {
            if let Some(anchor) = links.get(parent) {
                *parent = anchor.clone();
            }
        }
    }
    Ok(())
}

pub fn membership_training_rows(
    structure: DocumentStructure,
    source: &Table,
) -> Result<MembershipTrainingRows, String> {
    let (nodes, identities) = gold_fragments(&structure)?;
    let (_, rows) = select(
        &nodes,
        &structure.text,
        source,
        &structure.source_fragments,
        structure.profile,
        Some(&identities),
    )?;
    Ok(MembershipTrainingRows {
        schema: MEMBERSHIP_SCHEMA,
        feature_names: feature_names(),
        rows,
    })
}
