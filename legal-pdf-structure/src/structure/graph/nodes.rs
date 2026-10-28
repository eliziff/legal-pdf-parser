//! The structure graph's nodes from the parse: its paragraphs, headings, notes and their pairs.

use super::*;

pub(crate) fn map_note_pairs(
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

pub(crate) fn native_graph_parts(
    index: &PdfTextIndex,
    pages: &[Page],
    paragraphs: &[Paragraph],
    primitives: &PdfPrimitiveEvidence,
) -> Result<Vec<StructureNode>> {
    // A passage wholly in a contents list, or wholly a parallel translation, says so: a
    // reader of the text's own structure skips it. A transcript's line numbers are no part of
    // what a passage is.
    let labelled: HashSet<&str> = pages.iter().flat_map(|page| &page.lines)
        .filter(|line| line.region_type == "line_label").map(|line| line.id.as_str()).collect();
    let role = |line_ids: &[String]| {
        let all =
            |test: &dyn Fn(&String) -> bool| !line_ids.is_empty() && line_ids.iter().all(test);
        let numbered = |id: &String| labelled.contains(id.as_str());
        let listed = |id: &String| {
            primitives.claims.contents_line_ids.contains(id)
                || index
                    .line(id)
                    .is_some_and(|line| primitives.claims.contents_pages.contains(&line.page_index))
        };
        // Nearly all of it listed: a stray line the paragraph reader took in (a folio it did not
        // know for one) does not undo a list.
        let counted = line_ids.iter().filter(|id| !numbered(id)).count();
        let in_list = line_ids.iter().filter(|id| listed(id)).count();
        if counted > 0 && in_list * 5 >= counted * 4 {
            Some("contents")
        } else if all(&|id| primitives.contexts.translation_line_ids.contains(id)) {
            Some("translation")
        } else {
            None
        }
    };
    const ORIGIN: &str = "legalpdf.pdf-structure.v2";
    let mut nodes = Vec::new();
    let text = ScalarText::new(index.text());
    for page in pages {
        let range = index.page_range(page.index).ok_or_else(|| {
            Error::Message(format!("page {} is absent from the text index", page.index))
        })?;
        let mut node = StructureNode::new(
            page.id.clone(),
            NodeKind::Page,
            range,
            ORIGIN,
            Derivation::Native,
            None,
        );
        node.label = Some(format!("page{}", page.number));
        node.aliases = page.printed_label.clone().map(|label| vec![label]);
        node.anchor.clone_from(&page.printed_label_line_id);
        node.page_indexes.push(page.index);
        node.line_ids = index.line_ids(range);
        nodes.push(node);
        // A transcript's printed line numbers: with the page's label, the page and line a reader
        // cites ("p. 451, l. 12").
        for line in line_labels(page) {
            let Some(indexed) = index.line(&line.id) else {
                continue;
            };
            let mut node = StructureNode::new(
                format!("{}:line", line.id),
                NodeKind::Prose,
                indexed.range,
                ORIGIN,
                Derivation::Heuristic,
                None,
            );
            node.role = Some(legal_structure_model::PrimaryRole::LineLabel);
            node.label = Some(line.text.trim().to_owned());
            node.page_indexes.push(page.index);
            node.line_ids = vec![line.id.clone()];
            nodes.push(node);
        }
        // The page's running heads, footers and print margins, and its printed page number.
        for line in page.lines.iter().filter(|line| matches!(line.region_type.as_str(), "header" | "footer")) {
            let Some(indexed) = index.line(&line.id) else {
                continue;
            };
            let folio = page.printed_label_line_id.as_deref() == Some(line.id.as_str());
            let mut node = StructureNode::new(
                format!("{}:{}", line.id, if folio { "page" } else { "furniture" }),
                NodeKind::Prose,
                indexed.range,
                ORIGIN,
                Derivation::Heuristic,
                None,
            );
            node.role = Some(if folio {
                legal_structure_model::PrimaryRole::PageLabel
            } else {
                legal_structure_model::PrimaryRole::Furniture
            });
            node.page_indexes.push(page.index);
            node.line_ids = vec![line.id.clone()];
            nodes.push(node);
        }
    }
    for paragraph in paragraphs {
        let range = index
            .range_for_line_ids(&paragraph.line_ids)
            .ok_or_else(|| {
                Error::Message(format!("paragraph {} has no indexed lines", paragraph.id))
            })?;
        let heading = paragraph.region_type == "heading";
        // A region of its own, not the body's: a caption, a marginal note, the document's title.
        let region_role = match paragraph.region_type.as_str() {
            "caption" => Some(legal_structure_model::PrimaryRole::Caption),
            "marginal_note" => Some(legal_structure_model::PrimaryRole::MarginalNote),
            "title" => Some(legal_structure_model::PrimaryRole::Title),
            _ => None,
        };
        let mut node = StructureNode::new(
            paragraph.id.clone(),
            if heading {
                NodeKind::Heading
            } else {
                NodeKind::Prose
            },
            range,
            ORIGIN,
            if heading || region_role.is_some() {
                Derivation::Heuristic
            } else {
                Derivation::Native
            },
            None,
        );
        node.role = region_role;
        node.ranges = source_membership(paragraph.line_ids.iter()
            .filter_map(|id| index.line(id).map(|line| line.range)).collect());
        node.label = heading.then(|| node.ranges.iter().map(|part|
            text.slice(part.start..part.end).expect("indexed heading part must be in bounds")
        ).collect::<Vec<_>>().join(" "));
        if let Some(first) = node.ranges.first().filter(|_| heading) {
            let line = text.slice(first.start..first.end).expect("indexed heading part must be in bounds");
            if let Some(found) = heading_marker_re().captures(line).and_then(|c| c.get(1)) {
                node.marker_range = Some(ScalarRange {
                    start: first.start + line[..found.start()].chars().count(),
                    end: first.start + line[..found.end()].chars().count(),
                });
            }
        }
        node.page_indexes = index.page_indexes_for_line_ids(&paragraph.line_ids);
        node.line_ids.clone_from(&paragraph.line_ids);
        node.grammar = if heading {
            Some("accepted_heading".to_owned())
        } else {
            role(&paragraph.line_ids).map(str::to_owned)
        };
        nodes.push(node);
    }
    // A table, its rows and their cells, addressed as a word processor's are. A blank
    // cell is an empty range where its text would stand.
    for (slot, table) in primitives.claims.tables.iter().enumerate() {
        let table_id = format!("table:{}", slot + 1);
        let line_ids: Vec<String> = table.line_ids().into_iter().cloned().collect();
        let range = index
            .range_for_line_ids(&line_ids)
            .ok_or_else(|| Error::Message(format!("{table_id} has no indexed lines")))?;
        let mut node = StructureNode::new(
            table_id.clone(),
            NodeKind::Table,
            range,
            ORIGIN,
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
                        ORIGIN,
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
                ORIGIN,
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
    let mut quoted = BTreeMap::<&str, Vec<String>>::new();
    for line in &index.lines {
        let mut scope = primitives
            .quotations
            .scopes
            .get(&line.line_id)
            .map(String::as_str);
        while let Some(id) = scope {
            quoted.entry(id).or_default().push(line.line_id.clone());
            scope = primitives
                .quotations
                .introductions
                .get(id)
                .and_then(|intro| primitives.quotations.scopes.get(intro))
                .map(String::as_str);
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
            ORIGIN,
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
