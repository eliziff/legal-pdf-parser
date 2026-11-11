//! A judgment's numbered paragraphs, written from what the reading decided once (`Fences::read_paragraphs`): each
//! paragraph the spans of its lines it holds, in each language that prints it. Nothing is read here.

use std::collections::{BTreeMap, HashMap, HashSet};
use legal_pdf_core::model::Page;
use legal_structure_model::{Derivation, NodeKind, ScalarRange, StructureNode};
use crate::model::{Decided, Held};
use legal_structure_model::{DocumentStructure, SourceFragment};

/// Write the numbered paragraphs the reading decided. A document it read none in keeps what its pages read.
pub(crate) fn read_judgment_paragraphs(structure: &mut DocumentStructure, pages: &[Page], decided: &Decided, apart: &HashSet<&str>) {
    // A line's span is of its text; where the structure's text of the line differs in length, the whole line.
    let lengths: HashMap<&str, usize> = pages.iter().flat_map(|page| &page.lines).map(|line| (line.id.as_str(), line.text.encode_utf16().count())).collect();
    let mut printed: [BTreeMap<usize, Vec<ScalarRange>>; 2] = Default::default();
    // (`apart`: lines a model reads as no paragraph's, a running head or a note the rules let through on a scan.)
    for fragment in structure.source_fragments.iter().filter(|fragment| !apart.contains(fragment.id.as_str())) {
        let exact = lengths.get(fragment.id.as_str()) == Some(&(fragment.range.end - fragment.range.start));
        for &Held { translation, number, start, end } in decided.held(&fragment.id) {
            let range = if exact {
                ScalarRange { start: fragment.range.start + start as usize, end: fragment.range.start + end as usize }
            } else {
                fragment.range
            };
            printed[usize::from(translation)].entry(number as usize).or_default().push(range);
        }
    }
    // A table a paragraph holds whole is written as its own node: the paragraph's extent takes it.
    for node in structure.nodes.iter().filter(|node| node.kind == NodeKind::Table) {
        if let Some(&Held { translation, number, .. }) = node.line_ids.first().and_then(|id| decided.held(id).first()) {
            if node.line_ids.iter().all(|id| decided.held(id).iter().any(|held| held.number == number)) {
                printed[usize::from(translation)].entry(number as usize).or_default().push(node.range);
            }
        }
    }
    let fragments = structure.source_fragments.clone();
    for (translation, paragraphs) in printed.into_iter().enumerate() {
        if !paragraphs.is_empty() {
            write_paragraphs(structure, &fragments, decided, translation == 1, paragraphs);
        }
    }
}

/// One language's paragraphs as the structure's numbered paragraph nodes: a node the page reading found for a
/// number takes its extent; a number it found none for gets one; a node numbered off the count is no numbered paragraph.
fn write_paragraphs(
    structure: &mut DocumentStructure,
    fragments: &[SourceFragment],
    decided: &Decided,
    translation: bool,
    mut paragraphs: BTreeMap<usize, Vec<ScalarRange>>,
) {
    for ranges in paragraphs.values_mut() {
        ScalarRange::normalize_union(ranges);
    }
    let number_of = |label: Option<&str>| label.and_then(|label| label.parse::<usize>().ok());
    let in_language = |node: &StructureNode| !node.line_ids.is_empty() && node.line_ids.iter().all(|id| decided.translated(id)) == translation;
    let mut placed = HashSet::new();
    for node in structure.nodes.iter_mut().filter(|node| node.kind == NodeKind::Paragraph) {
        let Some(number) = number_of(node.label.as_deref()) else { continue };
        if !in_language(node) {
            continue;
        }
        match paragraphs.get(&number) {
            Some(ranges) if placed.insert(number) => node.extent = ranges.clone(),
            _ => node.label = None,
        }
    }
    let lines_of = |ranges: &[ScalarRange]| {
        fragments.iter()
            .filter(|fragment| ranges.iter().any(|range| range.start < fragment.range.end && fragment.range.start < range.end))
            .collect::<Vec<_>>()
    };
    for (number, ranges) in paragraphs.into_iter().filter(|(number, _)| !placed.contains(number)) {
        let printing = lines_of(&ranges);
        let (Some(first), Some(last)) = (ranges.first(), ranges.last()) else { continue };
        let mut node = StructureNode::new(
            format!("judgment-paragraph-{}-{number:06}", if translation { "translation" } else { "text" }),
            NodeKind::Paragraph,
            ScalarRange { start: first.start, end: last.end },
            "legalpdf.judgment-reading.v1",
            Derivation::Heuristic,
            None,
        );
        node.label = Some(number.to_string());
        node.ranges = ranges.clone();
        node.extent = ranges;
        node.line_ids = printing.iter().map(|fragment| fragment.id.clone()).collect();
        node.page_indexes = printing.iter().map(|fragment| fragment.page_index).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        node.grammar = translation.then(|| "translation".to_owned());
        structure.nodes.push(node);
    }
}
