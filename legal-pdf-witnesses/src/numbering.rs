//! Shared-grammar numbering evidence before ownership and hierarchy decisions.
use legal_structure_model::NumberingMarker;
use std::collections::BTreeMap;

pub(crate) const SOURCE_NAMES: &[&str] = &[
    "marker", "ordinal", "prefix_depth", "prefix_first", "prefix_last", "sequence_occurrences",
    "previous_ordinal", "previous_successor", "previous_restart",
    "next_ordinal", "next_successor", "next_restart",
];
pub(crate) const PAIR_NAMES: &[&str] = &[
    "unit_marked", "candidate_marked", "unit_ordinal", "candidate_ordinal",
    "unit_prefix_depth", "candidate_prefix_depth", "same_family", "same_prefix",
    "parent_prefix", "successor", "restart",
];
fn flag(value: bool) -> f32 { if value { 1.0 } else { 0.0 } }

pub(crate) fn parse(text: Option<&str>, previous: Option<&NumberingMarker>) -> Option<NumberingMarker> {
    text.and_then(|text| legal_pdf_core::structure_analysis().numbering_marker(text, previous))
}

pub(crate) struct Relation {
    pub same_family: bool,
    pub same_prefix: bool,
    pub parent_prefix: bool,
    pub successor: bool,
    pub restart: bool,
}
impl Relation {
    pub fn between(marker: Option<&NumberingMarker>, previous: Option<&NumberingMarker>) -> Self {
        let same_family = marker.zip(previous).is_some_and(|(left, right)| left.family == right.family);
        let same_prefix = marker.zip(previous).is_some_and(|(left, right)| left.prefix == right.prefix);
        let parent_prefix = marker.zip(previous).is_some_and(|(left, right)| {
            let mut prefix = right.prefix.clone();
            prefix.push(right.ordinal);
            left.prefix.starts_with(&prefix)
        });
        let successor = same_family && same_prefix && marker.zip(previous)
            .is_some_and(|(left, right)| right.ordinal.checked_add(1) == Some(left.ordinal));
        let restart = marker.is_some_and(|marker| marker.ordinal == 1)
            || same_family && same_prefix && marker.zip(previous).is_some_and(|(left, right)| left.ordinal <= right.ordinal);
        Self { same_family, same_prefix, parent_prefix, successor, restart }
    }
}

pub(crate) fn pair_values(marker: Option<&NumberingMarker>, previous: Option<&NumberingMarker>) -> Vec<f32> {
    let relation = Relation::between(marker, previous);
    vec![flag(marker.is_some()), flag(previous.is_some()), marker.map_or(0.0, |marker| marker.ordinal as f32),
        previous.map_or(0.0, |marker| marker.ordinal as f32), marker.map_or(0.0, |marker| marker.prefix.len() as f32),
        previous.map_or(0.0, |marker| marker.prefix.len() as f32), flag(relation.same_family), flag(relation.same_prefix),
        flag(relation.parent_prefix), flag(relation.successor), flag(relation.restart)]
}

/// Source neighbours are evidence, not a declaration that they share an ownership scope.
pub(crate) fn source_values(markers: &BTreeMap<usize, String>) -> BTreeMap<usize, Vec<f32>> {
    let mut previous = None;
    let parsed = markers.iter().filter_map(|(start, text)| {
        let marker = parse(Some(text), previous.as_ref())?;
        previous = Some(marker.clone());
        Some((*start, marker))
    }).collect::<Vec<_>>();
    let mut sequences = BTreeMap::<(String, Vec<u32>), Vec<usize>>::new();
    for (index, (_, marker)) in parsed.iter().enumerate() {
        sequences.entry((marker.family.clone(), marker.prefix.clone())).or_default().push(index);
    }
    let mut values = BTreeMap::new();
    for sequence in sequences.values() {
        for (position, index) in sequence.iter().enumerate() {
            let (start, marker) = &parsed[*index];
            let previous = position.checked_sub(1).map(|position| &parsed[sequence[position]].1);
            let next = sequence.get(position + 1).map(|index| &parsed[*index].1);
            let before = Relation::between(Some(marker), previous);
            let after = Relation::between(next, Some(marker));
            values.insert(*start, vec![1.0, marker.ordinal as f32, marker.prefix.len() as f32,
                marker.prefix.first().copied().unwrap_or(0) as f32, marker.prefix.last().copied().unwrap_or(0) as f32,
                sequence.len() as f32, previous.map_or(0.0, |marker| marker.ordinal as f32), flag(before.successor),
                flag(before.restart), next.map_or(0.0, |marker| marker.ordinal as f32), flag(after.successor), flag(after.restart)]);
        }
    }
    values
}
