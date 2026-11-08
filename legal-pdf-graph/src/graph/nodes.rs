//! The structure graph's nodes from the parse: its paragraphs, headings, notes and their pairs.

use super::*;

pub fn map_note_pairs(
    index: &PdfTextIndex,
    pairs: &[NotePairClaim],
) -> Result<(Vec<NotePairClaimV2>, Vec<StructureDiagnostic>)> {
    let mut claims = Vec::new();
    let mut diagnostics = Vec::new();
    let mut pair_ids = HashSet::new();
    for pair in pairs {
        if pair.pair_id.is_empty() || !pair_ids.insert(pair.pair_id.as_str()) {
            return Err(Error::Message(format!(
                "paired note '{}' has an empty or duplicate pair id",
                pair.pair_id
            )));
        }
        let label_line = index.line(&pair.label_anchor.line_id).ok_or_else(|| {
            Error::Message(format!(
                "paired note {} has an unknown label line",
                pair.pair_id
            ))
        })?;
        let label_range = index
            .global_range(
                &pair.label_anchor.line_id,
                pair.label_anchor.start,
                pair.label_anchor.end,
            )
            .ok_or_else(|| {
                Error::Message(format!(
                    "paired note {} has an invalid label range",
                    pair.pair_id
                ))
            })?;
        let references = pair
            .reference_anchors
            .iter()
            .map(|anchor| {
                let line = index.line(&anchor.line_id).ok_or_else(|| {
                    Error::Message(format!(
                        "paired note {} has an unknown reference line",
                        pair.pair_id
                    ))
                })?;
                let range = index
                    .global_range(&anchor.line_id, anchor.start, anchor.end)
                    .ok_or_else(|| {
                        Error::Message(format!(
                            "paired note {} has an invalid reference range",
                            pair.pair_id
                        ))
                    })?;
                Ok(TextAnchorV2 {
                    range,
                    page_index: line.page_index,
                    line_id: anchor.line_id.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if pair
            .body_line_ids
            .iter()
            .any(|line_id| index.line(line_id).is_none())
        {
            return Err(Error::Message(format!(
                "paired note {} has an unknown body line",
                pair.pair_id
            )));
        }
        let body_range = index.range_for_line_ids(&pair.body_line_ids);
        // A note whose label the sequence supports is the document's note even where no
        // reference to it could be read, among notes whose references mostly were: a form's
        // numbered fields, a caption's column of section signs or a table's numbered rows pair
        // with nothing, or with a stray number or two.
        let support = &pair.support;
        let label_only = support.supports_label_only();
        if (pair.reference_anchors.is_empty() && !label_only)
            || body_range.is_none()
            || body_range.is_some_and(|range| range.start == range.end)
            || label_range.start == label_range.end
            || references
                .iter()
                .any(|anchor| anchor.range.start == anchor.range.end)
        {
            diagnostics.push(StructureDiagnostic {
                code: "note_pair_unmaterialized".to_owned(),
                severity: DiagnosticSeverity::Info,
                ranges: Vec::new(),
                node_ids: Vec::new(),
            });
            continue;
        }
        claims.push(NotePairClaimV2 {
            pair_id: pair.pair_id.clone(),
            support: pair.support.clone(),
            kind: match pair.kind {
                NotePairKind::Footnote => NoteKindV2::Footnote,
                NotePairKind::Endnote => NoteKindV2::Endnote,
            },
            label: TextAnchorV2 {
                range: label_range,
                page_index: label_line.page_index,
                line_id: pair.label_anchor.line_id.clone(),
            },
            body: NoteBodyV2 {
                range: body_range.unwrap(),
                ranges: pair.body_line_ids.iter().filter_map(|id| index.line(id).map(|line| line.range)).collect(),
                page_indexes: index.page_indexes_for_line_ids(&pair.body_line_ids),
                line_ids: pair.body_line_ids.clone(),
            },
            references,
        });
    }
    Ok((claims, diagnostics))
}

/// A heading's printed number, its first line `line` starting at `start` in the text and the heading holding `lines`
/// lines: before its title on that line, or that whole line where the number stands alone over its title ("II." over
/// "Background Facts").
pub fn heading_marker(start: usize, line: &str, lines: usize) -> Option<ScalarRange> {
    let found = heading_marker_re()
        .captures(line)
        .or_else(|| (lines > 1).then(|| heading_marker_alone_re().captures(line)).flatten())?
        .get(1)?;
    Some(ScalarRange {
        start: start + line[..found.start()].chars().count(),
        end: start + line[..found.end()].chars().count(),
    })
}

/// Each quotation the fences read, a node over its lines.
pub fn quotation_nodes(index: &PdfTextIndex, decided: &Decided, origin: &'static str) -> Result<Vec<StructureNode>> {
    let mut nodes = Vec::new();
    let mut quoted = BTreeMap::<&str, Vec<String>>::new();
    for line in &index.lines {
        if let Some(id) = decided.quotation(&line.line_id) {
            quoted.entry(id).or_default().push(line.line_id.clone());
        }
    }
    for (id, line_ids) in quoted {
        let range = index
            .range_for_line_ids(&line_ids)
            .ok_or_else(|| Error::Message(format!("quotation {id} has no indexed lines")))?;
        let mut node = StructureNode::new(
            id.to_owned(),
            NodeKind::Quote,
            range,
            origin,
            Derivation::Heuristic,
            None,
        );
        node.ranges = source_membership(
            line_ids
                .iter()
                .filter_map(|id| index.line(id).map(|line| line.range))
                .collect(),
        );
        node.page_indexes = index.page_indexes_for_line_ids(&line_ids);
        node.line_ids = line_ids;
        node.grammar = Some("display_quotation".to_owned());
        nodes.push(node);
    }
    Ok(nodes)
}

/// Each table (with its place among the document's tables), its rows and their cells, addressed as
/// a word processor's are; a blank cell is an empty range where its text would stand.
pub fn table_nodes<'a>(
    index: &PdfTextIndex,
    tables: impl IntoIterator<Item = (usize, &'a legal_pdf_tables::PdfTable)>,
    origin: &'static str,
) -> Result<Vec<StructureNode>> {
    let mut nodes = Vec::new();
    for (slot, table) in tables {
        let table_id = format!("table:{}", slot + 1);
        let line_ids: Vec<String> = table.line_ids().into_iter().cloned().collect();
        let range = index
            .range_for_line_ids(&line_ids)
            .ok_or_else(|| Error::Message(format!("{table_id} has no indexed lines")))?;
        let mut node = StructureNode::new(
            table_id.clone(),
            NodeKind::Table,
            range,
            origin,
            Derivation::Heuristic,
            None,
        );
        node.table_number = Some(slot + 1);
        node.header_rows = Some(table.rows.iter().enumerate()
            .filter_map(|(index, row)| row.header.then_some(index + 1)).collect());
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
                .map(|(_, same)| {
                    let cell = same[0];
                    let mut line_ids: Vec<String> = Vec::new();
                    for one in &same {
                        for id in one.line_ids() {
                            if !line_ids.contains(&id) {
                                line_ids.push(id);
                            }
                        }
                    }
                    let row_span = same.iter().map(|one| one.row_span).max().unwrap_or(1);
                    let column_span = same.iter().map(|one| one.column_span).max().unwrap_or(1);
                    let ranges = source_membership(same
                        .iter()
                        .flat_map(|one| one.parts.iter())
                        .filter_map(|part| match part.chars {
                            Some((start, end)) => index.global_range(&part.line_id, start, end),
                            None => index.line(&part.line_id).map(|line| line.range),
                        })
                        .collect());
                    let range = ranges.iter().copied().reduce(|left, right| ScalarRange {
                            start: left.start.min(right.start),
                            end: left.end.max(right.end),
                        })
                        .unwrap_or(ScalarRange {
                            start: cursor,
                            end: cursor,
                        });
                    cursor = range.end;
                    let mut node = StructureNode::new(
                        format!("{row_id}/col:{}", cell.column + 1),
                        NodeKind::Cell,
                        range,
                        origin,
                        Derivation::Heuristic,
                        Some(row_id.clone()),
                    );
                    node.ranges = ranges;
                    node.label = Some(node.id.clone());
                    node.row = Some(row_slot + 1);
                    node.column = Some(cell.column + 1);
                    node.row_span = (row_span > 1).then_some(row_span);
                    node.column_span = (column_span > 1).then_some(column_span);
                    node.page_indexes = index.page_indexes_for_line_ids(&line_ids);
                    node.line_ids = line_ids;
                    node
                })
                .collect();
            let row_range = ScalarRange {
                start: cells
                    .iter()
                    .map(|cell| cell.range.start)
                    .min()
                    .unwrap_or(cursor),
                end: cells
                    .iter()
                    .map(|cell| cell.range.end)
                    .max()
                    .unwrap_or(cursor),
            };
            let mut node = StructureNode::new(
                row_id.clone(),
                NodeKind::Row,
                row_range,
                origin,
                Derivation::Heuristic,
                Some(table_id.clone()),
            );
            node.ranges = source_membership(cells.iter().flat_map(|cell| cell.ranges.iter().copied()).collect());
            table_ranges.extend(node.ranges.iter().copied());
            node.row = Some(row_slot + 1);
            node.label = Some(row_id);
            node.line_ids = cells
                .iter()
                .flat_map(|cell| cell.line_ids.clone())
                .collect();
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
