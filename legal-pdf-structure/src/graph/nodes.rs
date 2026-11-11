//! The structure graph's nodes from the parse: its paragraphs and headings.

use std::collections::BTreeMap;
use legal_pdf_core::{Error, Result};
use legal_structure_model::{Derivation, NodeKind, ScalarRange, StructureNode};
use crate::model::Decided;
use crate::text::text_index::{source_membership, PdfTextIndex};
use super::{heading_marker_alone_re, heading_marker_re};
use legal_structure_model::utf16_len;

/// A heading's printed number, its first line `line` starting at `start` in the text and the heading holding `lines`
/// lines: before its title on that line, or that whole line where the number stands alone over its title ("II." over
/// "Background Facts").
pub(crate) fn heading_marker(start: usize, line: &str, lines: usize) -> Option<ScalarRange> {
    let found = heading_marker_re().captures(line).or_else(|| (lines > 1).then(|| heading_marker_alone_re().captures(line)).flatten())?.get(1)?;
    Some(ScalarRange { start: start + utf16_len(&line[..found.start()]), end: start + utf16_len(&line[..found.end()]) })
}

/// The ranges of the lines `line_ids` names, as the index holds them.
fn line_ranges(index: &PdfTextIndex, line_ids: &[String]) -> Vec<ScalarRange> {
    line_ids.iter().filter_map(|id| index.line(id).map(|line| line.range)).collect()
}

/// Each quotation the fences read, a node over its lines.
pub(crate) fn quotation_nodes(index: &PdfTextIndex, decided: &Decided, origin: &'static str) -> Result<Vec<StructureNode>> {
    let mut quoted = BTreeMap::<&str, Vec<String>>::new();
    for line in &index.lines {
        if let Some(id) = decided.quotation(&line.id) {
            quoted.entry(id).or_default().push(line.id.clone());
        }
    }
    quoted
        .into_iter()
        .map(|(id, line_ids)| {
            let range = index.range_for_line_ids(&line_ids).ok_or_else(|| Error::Message(format!("quotation {id} has no indexed lines")))?;
            let mut node = StructureNode::new(id.to_owned(), NodeKind::Quote, range, origin, Derivation::Heuristic, None);
            node.ranges = source_membership(line_ranges(index, &line_ids));
            node.page_indexes = index.page_indexes_for_line_ids(&line_ids);
            node.line_ids = line_ids;
            node.grammar = Some("display_quotation".to_owned());
            Ok(node)
        })
        .collect()
}

/// Each table (with its place among the document's tables), its rows and their cells, addressed as
/// a word processor's are; a blank cell is an empty range where its text would stand.
pub(crate) fn table_nodes<'a>(index: &PdfTextIndex, tables: impl IntoIterator<Item = (usize, &'a crate::tables::PdfTable)>, origin: &'static str) -> Result<Vec<StructureNode>> {
    let mut nodes = Vec::new();
    for (slot, table) in tables {
        let table_id = format!("table:{}", slot + 1);
        let line_ids: Vec<String> = table.line_ids().into_iter().cloned().collect();
        let range = index.range_for_line_ids(&line_ids).ok_or_else(|| Error::Message(format!("{table_id} has no indexed lines")))?;
        let mut node = StructureNode::new(table_id.clone(), NodeKind::Table, range, origin, Derivation::Heuristic, None);
        node.table_number = Some(slot + 1);
        node.header_rows = Some(table.rows.iter().enumerate().filter_map(|(index, row)| row.header.then_some(index + 1)).collect());
        node.label = Some(table_id.clone());
        node.page_indexes = index.page_indexes_for_line_ids(&line_ids);
        node.line_ids = line_ids;
        let table_node_index = nodes.len();
        nodes.push(node);
        let mut table_ranges = Vec::new();
        let mut cursor = range.start;
        for (row_slot, row) in table.rows.iter().enumerate() {
            let row_id = format!("{table_id}/row:{}", row_slot + 1);
            // Cells a row places in one column are one cell's text: one node, so one id.
            let mut columns: Vec<(usize, Vec<&_>)> = Vec::new();
            for cell in &row.cells {
                match columns.iter_mut().find(|(column, _)| *column == cell.column) {
                    Some((_, same)) => same.push(cell),
                    None => columns.push((cell.column, vec![cell])),
                }
            }
            let cells: Vec<StructureNode> = columns
                .into_iter()
                .map(|(column, same)| {
                    let mut line_ids: Vec<String> = Vec::new();
                    for id in same.iter().flat_map(|one| one.line_ids()) {
                        if !line_ids.contains(&id) {
                            line_ids.push(id);
                        }
                    }
                    let row_span = same.iter().map(|one| one.row_span).max().unwrap_or(1);
                    let column_span = same.iter().map(|one| one.column_span).max().unwrap_or(1);
                    let parts = same.iter().flat_map(|one| one.parts.iter()).filter_map(|part| match part.chars {
                        Some((start, end)) => index.global_range(&part.line_id, start, end),
                        None => index.line(&part.line_id).map(|line| line.range),
                    });
                    let ranges = source_membership(parts.collect());
                    let range = ranges
                        .iter()
                        .copied()
                        .reduce(|left, right| ScalarRange { start: left.start.min(right.start), end: left.end.max(right.end) })
                        .unwrap_or(ScalarRange { start: cursor, end: cursor });
                    cursor = range.end;
                    let mut node = StructureNode::new(format!("{row_id}/col:{}", column + 1), NodeKind::Cell, range, origin, Derivation::Heuristic, Some(row_id.clone()));
                    node.ranges = ranges;
                    node.label = Some(node.id.clone());
                    node.row = Some(row_slot + 1);
                    node.column = Some(column + 1);
                    node.row_span = (row_span > 1).then_some(row_span);
                    node.column_span = (column_span > 1).then_some(column_span);
                    node.page_indexes = index.page_indexes_for_line_ids(&line_ids);
                    node.line_ids = line_ids;
                    node
                })
                .collect();
            let row_range = ScalarRange {
                start: cells.iter().map(|cell| cell.range.start).min().unwrap_or(cursor),
                end: cells.iter().map(|cell| cell.range.end).max().unwrap_or(cursor),
            };
            let mut node = StructureNode::new(row_id.clone(), NodeKind::Row, row_range, origin, Derivation::Heuristic, Some(table_id.clone()));
            node.ranges = source_membership(cells.iter().flat_map(|cell| cell.ranges.iter().copied()).collect());
            table_ranges.extend(node.ranges.iter().copied());
            node.row = Some(row_slot + 1);
            node.label = Some(row_id);
            node.line_ids = cells.iter().flat_map(|cell| cell.line_ids.clone()).collect();
            node.page_indexes = index.page_indexes_for_line_ids(&node.line_ids);
            nodes.push(node);
            nodes.extend(cells);
        }
        let table_node = &mut nodes[table_node_index];
        table_node.ranges = source_membership(table_ranges);
        if let (Some(first), Some(last)) = (table_node.ranges.first(), table_node.ranges.last()) {
            table_node.range = ScalarRange { start: first.start, end: last.end };
        }
    }
    Ok(nodes)
}
