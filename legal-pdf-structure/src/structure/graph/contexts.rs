//! Each node's exact membership and role, and the context its outline lives in.

use super::*;
use crate::structure::{quotations, PdfPrimitiveEvidence};

/// Settle each node's exact membership and role, and give the context its outline lives in: the
/// constituent document holding it, and its quotation, note, table cell or parallel text.
pub(crate) fn outline_contexts(
    nodes: &mut [legal_structure_model::StructureNode], document_id: &str,
    quotes: &quotations::Reading,
    index: &PdfTextIndex,
    primitives: &PdfPrimitiveEvidence,
) -> Vec<legal_pdf_witnesses::OutlineContext> {
    use legal_structure_model::PrimaryRole;
    let translation = &primitives.contexts.translation_line_ids;
    // Establish exact membership before comparing containers. An envelope may cross
    // intervening columns, notes, quotations, or constituent documents.
    for node in nodes.iter_mut() {
        if node.ranges.is_empty() { node.ranges.push(node.range); }
        if let Some(marker) = node.marker_range {
            node.ranges.push(marker);
        }
        node.ranges.sort_by_key(|range| (range.start, range.end));
        let mut merged: Vec<legal_structure_model::ScalarRange> = Vec::new();
        for range in node.ranges.drain(..) {
            if let Some(last) = merged.last_mut().filter(|last| range.start <= last.end) {
                last.end = last.end.max(range.end);
            } else {
                merged.push(range);
            }
        }
        node.ranges = merged;
        // Keep owned source fragments when the envelope spans intervening columns.
        if node.kind != NodeKind::Page && !node.line_ids.is_empty() {
            let mut source = node
                .line_ids
                .iter()
                .filter_map(|id| index.line(id).map(|line| line.range))
                .collect::<Vec<_>>();
            if let Some(marker) = node.marker_range {
                let ids = index.line_ids(marker);
                if let Some(first) = ids.first() {
                    let context = primitives.contexts.reading_context(first);
                    source.extend(
                        ids.iter()
                            .filter(|id| primitives.contexts.reading_context(id) == context)
                            .filter_map(|id| index.line(id))
                            .map(|line| ScalarRange {
                                start: marker.start.max(line.range.start),
                                end: marker.end.min(line.range.end),
                            })
                            .filter(|part| part.start < part.end),
                    );
                }
            }
            let source = source_membership(source);
            let mut owned = Vec::new();
            let mut at = 0;
            for range in &node.ranges {
                while at < source.len() && source[at].end <= range.start {
                    at += 1;
                }
                for part in source[at..]
                    .iter()
                    .take_while(|part| part.start < range.end)
                {
                    let start = range.start.max(part.start);
                    let end = range.end.min(part.end);
                    if start < end {
                        owned.push(ScalarRange { start, end });
                    }
                }
            }
            node.ranges = owned;
        }
        let mut seen_lines = HashSet::new();
        node.line_ids = node.ranges.iter().flat_map(|range| index.line_ids(*range))
            .filter(|id| seen_lines.insert(id.clone())).collect();
        node.page_indexes = index.page_indexes_for_line_ids(&node.line_ids);
        if let (Some(first), Some(last)) = (node.ranges.first(), node.ranges.last()) {
            node.range.start = first.start;
            node.range.end = last.end;
        }
        if let Some(start) = &mut node.content_start {
            if let Some(part) = node.ranges.iter().find(|part| *start < part.end) {
                *start = (*start).max(part.start);
            }
        }
    }
    let constituents = nodes.iter().filter(|node| node.grammar.as_deref().is_some_and(|grammar| grammar.starts_with("constituent_")))
        .map(|node| (node.id.clone(), node.ranges.clone())).collect::<Vec<_>>();
    let cells = nodes.iter().filter(|node| matches!(node.kind, NodeKind::Cell | NodeKind::Footnote | NodeKind::Endnote))
        .map(|node| (node.id.clone(), node.ranges.clone())).collect::<Vec<_>>();
    let contains = |outer: &[legal_structure_model::ScalarRange], inner: &[legal_structure_model::ScalarRange]| {
        !inner.is_empty() && inner.iter().all(|part| outer.iter().any(|range|
            range.start <= part.start && part.end <= range.end))
    };
    nodes
        .iter_mut()
        .map(|node| {
            if node.kind == NodeKind::Heading {
                node.role = Some(PrimaryRole::Heading);
                node.heading_range = (node.ranges.len() == 1).then_some(node.range);
            } else if matches!(
                node.kind,
                NodeKind::Paragraph | NodeKind::ListItem | NodeKind::Prose
            ) {
                node.role.get_or_insert(PrimaryRole::Text);
            }
            let smallest = |holders: &[(String, Vec<legal_structure_model::ScalarRange>)]| {
                holders
                    .iter()
                    .filter(|(id, ranges)| id != &node.id && contains(ranges, &node.ranges))
                    .min_by_key(|(_, ranges)| {
                        ranges
                            .iter()
                            .map(|range| range.end - range.start)
                            .sum::<usize>()
                    })
                    .map(|(id, _)| id.clone())
            };
            // A page holds whatever ends on it (a blank page sits at the edge of a note) rather than
            // belonging to its quotation or note: it names only its document.
            let unit = node.kind != NodeKind::Page;
            let quotation = if node.kind == NodeKind::Quote {
                quotes
                    .introductions
                    .get(&node.id)
                    .and_then(|id| quotes.scopes.get(id))
                    .cloned()
            } else {
                node.line_ids
                    .first()
                    .filter(|_| unit)
                    .and_then(|id| quotes.scopes.get(id))
                    .filter(|id| {
                        node.line_ids
                            .iter()
                            .all(|line| quotes.scopes.get(line) == Some(*id))
                    })
                    .cloned()
            };
            let translated = unit
                && !node.line_ids.is_empty()
                && node.line_ids.iter().all(|id| translation.contains(id));
            let container = smallest(&cells)
                .filter(|_| unit)
                .map(|id| format!("object:{id}"))
                .or_else(|| translated.then(|| "translation".to_owned()));
            (
                smallest(&constituents).unwrap_or_else(|| document_id.to_owned()),
                quotation,
                container,
            )
        })
        .collect()
}
