//! One sequential attachment decision over the active ancestors of each source scope.
//! The source span, never its first line, identifies a unit. Root is a scored candidate.

use crate::witness::{index, Table};
use legal_structure_model::{
    NodeKind, NumberingMarker, ScalarRange, ScalarText, StructureNode, StructureOwnership,
};
use std::collections::HashMap;

pub const PAIR_NAMES: &[&str] = &[
    "root",
    "u_heading",
    "u_list",
    "u_paragraph",
    "u_section",
    "c_heading",
    "c_list",
    "c_paragraph",
    "c_section",
    "same_family",
    "u_ordinal",
    "c_ordinal",
    "successor",
    "same_prefix",
    "parent_prefix",
    "u_restart",
    "u_marked",
    "c_marked",
    "u_style_rank",
    "c_style_rank",
    "same_style",
    "u_style_share",
    "c_style_share",
    "size_ratio",
    "indent_delta",
    "distance",
    "pages_apart",
    "open_steps",
    "c_lead_in",
    "u_quoted",
    "u_constituent",
    "u_stream",
    "u_contents_match",
    "c_contents_match",
    "u_marker_depth",
    "c_marker_depth",
    "u_navigation_confirmed",
    "c_navigation_confirmed",
    "confirmed_navigation_parent",
    "profile_known",
    "profile_case_rooted_complete",
    "profile_case_contiguous_complete",
    "profile_case_lossy",
    "profile_legislation",
    "profile_instrument",
    "profile_journal",
];

/// The feature layout and root semantics are part of the trained model's contract.
pub const PARENT_SCHEMA: &str = "scoped-span-root-v6";

type ScopeKey = (String, Option<String>, Option<String>);

struct Unit {
    node: usize,
    scope: ScopeKey,
    marker: Option<NumberingMarker>,
    style: (i32, bool, bool),
    rank: f32,
    share: f32,
    source_values: Vec<f32>,
}

#[derive(serde::Serialize)]
pub struct Pair {
    pub unit: String,
    pub candidate: Option<String>,
    pub range: ScalarRange,
    pub ownership: StructureOwnership,
    pub values: Vec<f32>,
    pub unit_values: Vec<f32>,
    pub candidate_values: Vec<f32>,
    pub positive: Option<bool>,
}

fn unit_value(unit: Option<&Unit>, name: &str) -> f32 {
    unit.and_then(|unit| index(name).and_then(|at| unit.source_values.get(at)))
        .copied()
        .unwrap_or(0.0)
}

fn flag(value: bool) -> f32 {
    if value {
        1.0
    } else {
        0.0
    }
}

fn heading(node: &StructureNode) -> bool {
    node.role == Some(legal_structure_model::PrimaryRole::Heading) || node.kind == NodeKind::Heading
}

pub(crate) fn outline_unit(node: &StructureNode) -> bool {
    if matches!(
        node.role,
        Some(
            legal_structure_model::PrimaryRole::Furniture
                | legal_structure_model::PrimaryRole::PageLabel
        )
    ) {
        return false;
    }
    matches!(
        node.kind,
        NodeKind::Heading | NodeKind::ListItem | NodeKind::Paragraph | NodeKind::Prose
    ) || node.kind == NodeKind::Section && node.marker_range.is_some()
}

/// Attach the materialized source units before final graph projection. The old rootless model is
/// deliberately incompatible: new weights need independently revised, frozen scope/span gold.
pub fn assemble_outline(
    nodes: &mut [StructureNode],
    document_id: &str,
    text: &str,
    table: &Table,
    navigation: &HashMap<String, Option<String>>,
    profile: Option<legal_structure_model::DetectionProfile>,
) -> Vec<Pair> {
    assemble_outline_inner(nodes, document_id, text, table, navigation, profile, None)
}

fn assemble_outline_inner(
    nodes: &mut [StructureNode],
    document_id: &str,
    text: &str,
    table: &Table,
    navigation: &HashMap<String, Option<String>>,
    profile: Option<legal_structure_model::DetectionProfile>,
    gold_parents: Option<&HashMap<String, Option<String>>>,
) -> Vec<Pair> {
    let rows = table
        .rows
        .iter()
        .enumerate()
        .map(|(index, row)| (row.id.as_str(), index))
        .collect::<HashMap<_, _>>();
    let coordinates = ScalarText::new(text);
    let mut units = nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| outline_unit(node))
        .map(|(node_index, node)| {
            let ownership = node
                .ownership
                .clone()
                .unwrap_or_else(|| StructureOwnership {
                    document_id: document_id.to_owned(),
                    quotation_id: None,
                    reading_stream_id: None,
                });
            let row = node
                .line_ids
                .first()
                .and_then(|id| rows.get(id.as_str()))
                .copied();
            let evidence = row.map(|index| &table.rows[index]);
            let source_values = crate::witness::node_values(evidence, node, &coordinates);
            Unit {
                node: node_index,
                scope: (
                    ownership.document_id,
                    ownership.quotation_id,
                    ownership.reading_stream_id,
                ),
                marker: None,
                style: (
                    (source_values[index("size").unwrap()] * 100.0).round() as i32,
                    source_values[index("bold").unwrap()] >= 0.5,
                    source_values[index("italic").unwrap()] >= 0.5,
                ),
                rank: 0.0,
                share: 0.0,
                source_values,
            }
        })
        .collect::<Vec<_>>();
    units.sort_by_key(|unit| {
        (
            nodes[unit.node].range.start,
            nodes[unit.node].range.end,
            nodes[unit.node].id.clone(),
        )
    });
    let mut styles = HashMap::<ScopeKey, HashMap<(i32, bool, bool), usize>>::new();
    let mut previous_marker = HashMap::<ScopeKey, NumberingMarker>::new();
    for unit in &mut units {
        *styles
            .entry(unit.scope.clone())
            .or_default()
            .entry(unit.style)
            .or_default() += 1;
        let node = &nodes[unit.node];
        let marker = node
            .marker_range
            .and_then(|range| coordinates.slice(range.start..range.end))
            .or(node.label.as_deref());
        unit.marker = marker.and_then(|marker| {
            crate::numbering::parse(Some(marker), previous_marker.get(&unit.scope))
        });
        if let Some(marker) = &unit.marker {
            previous_marker.insert(unit.scope.clone(), marker.clone());
        }
    }
    for unit in &mut units {
        let counts = &styles[&unit.scope];
        let total = counts.values().sum::<usize>().max(1);
        let mut ladder = counts.keys().copied().collect::<Vec<_>>();
        ladder.sort_unstable_by(|left, right| right.cmp(left));
        unit.rank = ladder
            .iter()
            .position(|style| style == &unit.style)
            .unwrap_or(0) as f32;
        unit.share = *counts.get(&unit.style).unwrap_or(&0) as f32 / total as f32;
    }
    let model = if gold_parents.is_none() {
        crate::model::parent_model()
    } else {
        None
    };
    let zeros = vec![0.0; crate::witness::NAMES.len()];
    let mut stacks = HashMap::<ScopeKey, Vec<usize>>::new();
    let mut chosen: Vec<Option<usize>> = vec![None; units.len()];
    let unit_ids = units
        .iter()
        .enumerate()
        .map(|(position, unit)| (nodes[unit.node].id.clone(), position))
        .collect::<HashMap<_, _>>();
    let mut heading_targets = HashMap::<&str, Option<usize>>::new();
    for (position, unit) in units
        .iter()
        .enumerate()
        .filter(|(_, unit)| heading(&nodes[unit.node]))
    {
        heading_targets.insert(nodes[unit.node].id.as_str(), Some(position));
        for line in &nodes[unit.node].line_ids {
            heading_targets
                .entry(line.as_str())
                .and_modify(|prior| *prior = None)
                .or_insert(Some(position));
        }
    }
    let navigation_claim = |node: &StructureNode| {
        navigation.get(&node.id).or_else(|| {
            node.line_ids
                .iter()
                .filter(|line| {
                    heading_targets
                        .get(line.as_str())
                        .is_some_and(Option::is_some)
                })
                .find_map(|line| navigation.get(line))
        })
    };
    let mut pairs = Vec::new();
    let dump = gold_parents.is_some() || std::env::var_os("LEGALPDF_WITNESS_DUMP").is_some();
    for (position, unit) in units.iter().enumerate() {
        let stack = stacks.entry(unit.scope.clone()).or_default();
        let node = &nodes[unit.node];
        let deterministic = stack
            .iter()
            .rposition(|&candidate| {
                let parent = &units[candidate];
                let parent_node = &nodes[parent.node];
                if heading(node) {
                    return heading(parent_node) && parent.rank < unit.rank;
                }
                if let (Some(marker), Some(other)) = (&unit.marker, &parent.marker) {
                    if marker.family == other.family && marker.prefix == other.prefix {
                        return false;
                    }
                    let mut parent_components = other.prefix.clone();
                    parent_components.push(other.ordinal);
                    if marker.prefix.starts_with(&parent_components) {
                        return true;
                    }
                }
                heading(parent_node) || unit_value(Some(parent), "end_colon") >= 0.5
            })
            .map(|index| stack[index]);
        // A sibling closes itself and its descendants while retaining its enclosing ancestors.
        let deterministic = unit
            .marker
            .as_ref()
            .and_then(|marker| {
                stack
                    .iter()
                    .rev()
                    .find(|&&candidate| {
                        units[candidate].marker.as_ref().is_some_and(|other| {
                            marker.family == other.family && marker.prefix == other.prefix
                        })
                    })
                    .map(|&sibling| chosen[sibling])
            })
            .unwrap_or(deterministic);
        let navigation_parent = navigation_claim(node).and_then(|parent| match parent {
            None => Some(None),
            Some(line) => heading_targets
                .get(line.as_str())
                .copied()
                .flatten()
                .filter(|&parent| parent < position && units[parent].scope == unit.scope)
                .map(Some),
        });
        let deterministic = navigation_parent
            .filter(|parent| parent.is_none_or(|parent| stack.contains(&parent)))
            .unwrap_or(deterministic);
        let mut best = None::<(Option<usize>, f64)>;
        for candidate in std::iter::once(None)
            .chain(stack.iter().rev().copied().map(Some))
            .filter(|_| model.is_some() || dump)
        {
            let parent = candidate.map(|index| &units[index]);
            let parent_node = parent.map(|unit| &nodes[unit.node]);
            let marker = unit.marker.as_ref();
            let parent_marker = parent.and_then(|unit| unit.marker.as_ref());
            let numbering = crate::numbering::Relation::between(marker, parent_marker);
            let is_parent = |kind| parent_node.is_some_and(|node| node.kind == kind);
            let mut values = vec![
                flag(candidate.is_none()),
                flag(heading(node)),
                flag(node.kind == NodeKind::ListItem),
                flag(node.kind == NodeKind::Paragraph || node.kind == NodeKind::Prose),
                flag(node.kind == NodeKind::Section),
                flag(parent_node.is_some_and(heading)),
                flag(is_parent(NodeKind::ListItem)),
                flag(is_parent(NodeKind::Paragraph) || is_parent(NodeKind::Prose)),
                flag(is_parent(NodeKind::Section)),
                flag(numbering.same_family),
                marker.map_or(0.0, |m| m.ordinal as f32),
                parent_marker.map_or(0.0, |m| m.ordinal as f32),
                flag(numbering.successor),
                flag(numbering.same_prefix),
                flag(numbering.parent_prefix),
                flag(marker.is_some_and(|m| m.ordinal == 1)),
                flag(marker.is_some()),
                flag(parent_marker.is_some()),
                unit.rank,
                parent.map_or(0.0, |p| p.rank),
                flag(parent.is_some_and(|p| p.style == unit.style)),
                unit.share,
                parent.map_or(0.0, |p| p.share),
                if unit_value(parent, "size") > 0.0 {
                    unit_value(Some(unit), "size") / unit_value(parent, "size")
                } else {
                    0.0
                },
                unit_value(Some(unit), "indent") - unit_value(parent, "indent"),
                candidate.map_or(0.0, |c| (position - c) as f32),
                (unit_value(Some(unit), "page_frac") - unit_value(parent, "page_frac")).abs(),
                candidate
                    .and_then(|c| stack.iter().rev().position(|&p| p == c))
                    .map_or(stack.len() as f32, |d| d as f32),
                unit_value(parent, "end_colon"),
                flag(unit.scope.1.is_some()),
                flag(unit.scope.0 != document_id),
                flag(unit.scope.2.is_some()),
                unit_value(Some(unit), "in_contents"),
                unit_value(parent, "in_contents"),
                marker.map_or(0.0, |m| (m.prefix.len() + 1) as f32),
                parent_marker.map_or(0.0, |m| (m.prefix.len() + 1) as f32),
                flag(navigation_parent.is_some()),
                flag(parent_node.is_some_and(|node| navigation_claim(node).is_some())),
                flag(navigation_parent == Some(candidate)),
            ];
            values.extend(crate::profile::values(profile));
            let unit_values = unit.source_values.clone();
            let candidate_values =
                parent.map_or_else(|| zeros.clone(), |parent| parent.source_values.clone());
            if let Some(model) = model {
                let mut features = values.clone();
                features.extend_from_slice(&unit_values);
                features.extend_from_slice(&candidate_values);
                let score = model.scores(&features)[0];
                if best.is_none_or(|(_, prior)| score > prior) {
                    best = Some((candidate, score));
                }
            }
            if dump {
                pairs.push(Pair {
                    unit: node.id.clone(),
                    candidate: parent_node.map(|node| node.id.clone()),
                    range: node.range,
                    ownership: StructureOwnership {
                        document_id: unit.scope.0.clone(),
                        quotation_id: unit.scope.1.clone(),
                        reading_stream_id: unit.scope.2.clone(),
                    },
                    values,
                    unit_values,
                    candidate_values,
                    positive: gold_parents.and_then(|parents| parents.get(&node.id)).map(
                        |parent| parent.as_deref() == parent_node.map(|node| node.id.as_str()),
                    ),
                });
            }
        }
        let parent = if let Some(parents) = gold_parents {
            parents
                .get(&node.id)
                .and_then(|parent| parent.as_ref())
                .and_then(|id| unit_ids.get(id))
                .copied()
                .filter(|&parent| parent < position && units[parent].scope == unit.scope)
        } else {
            best.map_or(deterministic, |(parent, _)| parent)
        };
        chosen[position] = parent;
        if let Some(parent) = parent {
            if let Some(at) = stack.iter().position(|&p| p == parent) {
                stack.truncate(at + 1);
            } else {
                // Gold can attach outside the inference stack. Preserve its history without inventing a positive row.
                stack.clear();
                let mut ancestor = Some(parent);
                while let Some(at) = ancestor {
                    stack.push(at);
                    ancestor = chosen[at];
                }
                stack.reverse();
            }
        } else {
            stack.clear();
        }
        stack.push(position);
    }
    for (position, unit) in units.iter().enumerate() {
        nodes[unit.node].parent_id =
            chosen[position].map(|parent| nodes[units[parent].node].id.clone());
        nodes[unit.node].ownership = Some(StructureOwnership {
            document_id: unit.scope.0.clone(),
            quotation_id: unit.scope.1.clone(),
            reading_stream_id: unit.scope.2.clone(),
        });
    }
    let ownership = nodes
        .iter()
        .map(|node| (node.id.clone(), (node.kind, node.ownership.clone())))
        .collect::<HashMap<_, _>>();
    for node in nodes {
        if node
            .parent_id
            .as_ref()
            .is_some_and(|parent| ownership.get(parent).is_none_or(|(kind, owner)|
                !node.kind.is_grid_child_of(*kind) && node.ownership.as_ref().zip(owner.as_ref())
                    .is_some_and(|(owner, parent_owner)| owner != parent_owner)))
        {
            node.parent_id = None;
        }
    }
    pairs
}

#[derive(serde::Serialize)]
pub struct OutlineTrainingRows {
    pub schema: &'static str,
    pub feature_names: Vec<String>,
    pub rows: Vec<Pair>,
    pub unavailable_parent_units: Vec<String>,
}

/// Feature generation on fixed, corrected source. Gold parents supply history but remain unchanged;
/// non-stack gold relationships are reported instead of becoming all-negative training queries.
pub fn outline_training_rows(
    structure: legal_structure_model::DocumentStructure,
    table: &Table,
    navigation: &HashMap<String, Option<String>>,
) -> Result<OutlineTrainingRows, String> {
    if structure.offset_unit.as_ref() != "utf16" {
        return Err("Gold structure must use explicit UTF-16 export coordinates".into());
    }
    let coordinates = ScalarText::new(&structure.text);
    let mut nodes = structure.nodes;
    let convert = |range: &mut ScalarRange| -> Result<(), String> {
        range.start = coordinates
            .scalar_at_utf16(range.start)
            .ok_or("Gold span starts inside a UTF-16 scalar")?;
        range.end = coordinates
            .scalar_at_utf16(range.end)
            .ok_or("Gold span ends inside a UTF-16 scalar")?;
        Ok(())
    };
    for node in &mut nodes {
        convert(&mut node.range)?;
        for range in &mut node.ranges {
            convert(range)?;
        }
        if let Some(range) = &mut node.marker_range {
            convert(range)?;
        }
        if let Some(range) = &mut node.heading_range {
            convert(range)?;
        }
        if outline_unit(node) && node.ownership.is_none() {
            return Err(format!(
                "Gold node {} lacks explicit scope ownership",
                node.id
            ));
        }
    }
    let parents = nodes
        .iter()
        .filter(|node| outline_unit(node))
        .map(|node| (node.id.clone(), node.parent_id.clone()))
        .collect::<HashMap<_, _>>();
    let mut table = table.clone();
    crate::witness::bind_source_fragments(
        &mut table,
        &structure.text,
        &structure.source_fragments,
    )?;
    let rows = assemble_outline_inner(
        &mut nodes,
        &structure.document_id,
        &structure.text,
        &table,
        navigation,
        structure.profile,
        Some(&parents),
    );
    let mut found = HashMap::<String, bool>::new();
    for row in &rows {
        *found.entry(row.unit.clone()).or_default() |= row.positive == Some(true);
    }
    let mut unavailable_parent_units = found
        .into_iter()
        .filter_map(|(unit, positive)| (!positive).then_some(unit))
        .collect::<Vec<_>>();
    unavailable_parent_units.sort();
    let feature_names = PAIR_NAMES
        .iter()
        .map(|name| format!("pair_{name}"))
        .chain(crate::witness::NAMES.iter().map(|name| format!("u_{name}")))
        .chain(crate::witness::NAMES.iter().map(|name| format!("c_{name}")))
        .collect();
    Ok(OutlineTrainingRows {
        schema: PARENT_SCHEMA,
        feature_names,
        rows,
        unavailable_parent_units,
    })
}
