//! Each unit's parent, chosen in reading order among the open units of its own context: a heading
//! nests under a more prominent heading, a numbered unit under the unit its numbering descends
//! from, other units under the nearest open heading or lead-in, and the printed contents or
//! bookmarks place a heading where they name its parent.

use crate::witness::{index, node_values, Table};
use legal_structure_model::{NodeKind, NumberingMarker, ScalarText, StructureNode};
use std::collections::HashMap;

/// Where a unit's outline lives: its document, and the quotation and note or table cell holding it.
pub type Context = (String, Option<String>, Option<String>);

struct Unit {
    node: usize,
    context: Context,
    marker: Option<NumberingMarker>,
    style: (i32, bool, bool),
    rank: f32,
    lead_in: bool,
}

fn heading(node: &StructureNode) -> bool {
    node.role == Some(legal_structure_model::PrimaryRole::Heading) || node.kind == NodeKind::Heading
}

fn outline_unit(node: &StructureNode) -> bool {
    if matches!(node.role, Some(legal_structure_model::PrimaryRole::Furniture | legal_structure_model::PrimaryRole::PageLabel)) {
        return false;
    }
    matches!(node.kind, NodeKind::Heading | NodeKind::ListItem | NodeKind::Paragraph | NodeKind::Prose)
        || node.kind == NodeKind::Section && node.marker_range.is_some()
}

/// Set every outline unit's parent. `contexts` holds each node's context; a parent link between
/// contexts is dropped, except a row's to its table and a cell's to its row.
pub fn assemble_outline(
    nodes: &mut [StructureNode],
    contexts: &[Context],
    text: &str,
    table: &Table,
    navigation: &HashMap<String, Option<String>>,
) {
    let rows = table.rows.iter().enumerate().map(|(index, row)| (row.id.as_str(), index)).collect::<HashMap<_, _>>();
    let coordinates = ScalarText::new(text);
    let at = |name: &str| index(name).unwrap();
    let mut units = nodes.iter().enumerate().filter(|(_, node)| outline_unit(node)).map(|(node_index, node)| {
        let evidence = node.line_ids.first().and_then(|id| rows.get(id.as_str())).map(|&index| &table.rows[index]);
        let values = node_values(evidence, node, &coordinates);
        Unit {
            node: node_index,
            context: contexts[node_index].clone(),
            marker: None,
            style: ((values[at("size")] * 100.0).round() as i32, values[at("bold")] >= 0.5, values[at("italic")] >= 0.5),
            rank: 0.0,
            lead_in: values[at("end_colon")] >= 0.5,
        }
    }).collect::<Vec<_>>();
    units.sort_by_key(|unit| (nodes[unit.node].range.start, nodes[unit.node].range.end, nodes[unit.node].id.clone()));
    let mut styles = HashMap::<Context, Vec<(i32, bool, bool)>>::new();
    let mut previous_marker = HashMap::<Context, NumberingMarker>::new();
    for unit in &mut units {
        styles.entry(unit.context.clone()).or_default().push(unit.style);
        let node = &nodes[unit.node];
        let marker = node.marker_range.and_then(|range| coordinates.slice(range.start..range.end)).or(node.label.as_deref());
        unit.marker = marker.and_then(|marker| legal_pdf_core::structure_analysis().numbering_marker(marker, previous_marker.get(&unit.context)));
        if let Some(marker) = &unit.marker {
            previous_marker.insert(unit.context.clone(), marker.clone());
        }
    }
    // A unit's rank is its type's place among its context's types, most prominent first.
    for ladder in styles.values_mut() {
        ladder.sort_unstable_by(|left, right| right.cmp(left));
        ladder.dedup();
    }
    for unit in &mut units {
        unit.rank = styles[&unit.context].iter().position(|style| style == &unit.style).unwrap_or(0) as f32;
    }
    let mut heading_targets = HashMap::<&str, Option<usize>>::new();
    for (position, unit) in units.iter().enumerate().filter(|(_, unit)| heading(&nodes[unit.node])) {
        heading_targets.insert(nodes[unit.node].id.as_str(), Some(position));
        for line in &nodes[unit.node].line_ids {
            heading_targets.entry(line.as_str()).and_modify(|prior| *prior = None).or_insert(Some(position));
        }
    }
    let navigation_claim = |node: &StructureNode| navigation.get(&node.id).or_else(|| node.line_ids.iter()
        .filter(|line| heading_targets.get(line.as_str()).is_some_and(Option::is_some))
        .find_map(|line| navigation.get(line)));
    let mut stacks = HashMap::<Context, Vec<usize>>::new();
    let mut chosen: Vec<Option<usize>> = vec![None; units.len()];
    for (position, unit) in units.iter().enumerate() {
        let stack = stacks.entry(unit.context.clone()).or_default();
        let node = &nodes[unit.node];
        let mut parent = stack.iter().rposition(|&candidate| {
            let parent = &units[candidate];
            if heading(node) {
                return heading(&nodes[parent.node]) && parent.rank < unit.rank;
            }
            if let (Some(marker), Some(other)) = (&unit.marker, &parent.marker) {
                if marker.family == other.family && marker.prefix == other.prefix {
                    return false;
                }
                let mut components = other.prefix.clone();
                components.push(other.ordinal);
                if marker.prefix.starts_with(&components) {
                    return true;
                }
            }
            heading(&nodes[parent.node]) || parent.lead_in
        }).map(|index| stack[index]);
        // A sibling closes itself and its descendants while retaining its enclosing ancestors.
        if let Some(sibling) = unit.marker.as_ref().and_then(|marker| stack.iter().rev().find(|&&candidate|
            units[candidate].marker.as_ref().is_some_and(|other| marker.family == other.family && marker.prefix == other.prefix))) {
            parent = chosen[*sibling];
        }
        let navigation_parent = navigation_claim(node).and_then(|claimed| match claimed {
            None => Some(None),
            Some(line) => heading_targets.get(line.as_str()).copied().flatten()
                .filter(|&parent| parent < position && units[parent].context == unit.context).map(Some),
        });
        if let Some(claimed) = navigation_parent.filter(|claimed| claimed.is_none_or(|parent| stack.contains(&parent))) {
            parent = claimed;
        }
        chosen[position] = parent;
        match parent.map(|parent| (parent, stack.iter().position(|&p| p == parent))) {
            Some((_, Some(at))) => stack.truncate(at + 1),
            // A parent off the stack reopens its own ancestry.
            Some((parent, None)) => {
                stack.clear();
                let mut ancestor = Some(parent);
                while let Some(at) = ancestor {
                    stack.push(at);
                    ancestor = chosen[at];
                }
                stack.reverse();
            }
            None => stack.clear(),
        }
        stack.push(position);
    }
    for (position, unit) in units.iter().enumerate() {
        nodes[unit.node].parent_id = chosen[position].map(|parent| nodes[units[parent].node].id.clone());
    }
    let positions = nodes.iter().enumerate().map(|(position, node)| (node.id.clone(), position)).collect::<HashMap<_, _>>();
    for position in 0..nodes.len() {
        let Some(parent) = nodes[position].parent_id.as_ref().map(|id| positions.get(id).copied()) else { continue };
        let grid = parent.is_some_and(|parent| matches!((nodes[position].kind, nodes[parent].kind),
            (NodeKind::Cell, NodeKind::Row) | (NodeKind::Row, NodeKind::Table)));
        if !grid && parent.is_none_or(|parent| contexts[parent] != contexts[position]) {
            nodes[position].parent_id = None;
        }
    }
}
