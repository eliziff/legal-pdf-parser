//! The transition parser's reading as the document consumers read: its structure graph, built from
//! the moves and the state they move, nothing from the stage parser's decisions. Each unit is a node:
//! a numbered paragraph a `Paragraph` (its label and marker), a provision a `Section` (located by its
//! number), a heading a `Heading`, a list item a `ListItem`, any other body text `Prose` (a quotation
//! also a `Quote` over it, a contents list's lines under the `contents` grammar), a note a `Footnote`
//! with its pair (its label, its body, the reference marks it answered), and a line set apart a
//! `Prose` of its role (furniture, a page or transcript line number, a title, a marginal note, a
//! caption, a headnote). Parents come from the state: a heading under the nearest heading above it in
//! its document; a unit counting on a marker under the unit of the frame below its own on the stack,
//! else under the open heading; a quotation under the unit before it.

use super::*;
use legal_structure_model::{Derivation, DocumentStructure, Note, NoteKindV2, NoteReference, Origin, PrimaryRole, Scope, StructureNode};


struct Unit {
    region: u8,
    lines: Vec<usize>,
    marker: Option<(usize, usize)>,
    label: Option<String>,
    parent: Option<usize>,
    references: Vec<(usize, u32, u32)>,
}

/// The opening token's characters in a line's text: its first and past-last, leading spaces skipped;
/// with `word`, the token after it too ("PART II").
fn opening(text: &str, word: bool) -> Option<(usize, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let start = chars.iter().position(|c| !c.is_whitespace())?;
    let mut end = start + chars[start..].iter().take_while(|c| !c.is_whitespace()).count();
    if word {
        let next = end + chars[end..].iter().take_while(|c| c.is_whitespace()).count();
        end = next + chars[next..].iter().take_while(|c| !c.is_whitespace()).count();
    }
    Some((start, end))
}

/// The document's structure graph from its lines (`table`, `views`) and the moves they made.
pub(crate) fn structure_graph(
    pages: &mut [Page],
    slots: &Slots,
    evidence: &Evidence,
    table: &LineTable,
    views: &[LineView],
    moves: &[Move],
    identity: &StructureIdentity,
) -> Result<DocumentStructure> {
    // The text runs in the parser's reading order, and each page's regions are its lines' regions as
    // the parser reads them: furniture and page labels the page's header above its middle, its
    // footer below.
    for (k, &slot) in table.order.iter().enumerate() {
        let height = pages[slots.page(slot)].height.max(1.0);
        let line = slots.line_mut(pages, slot);
        line.reading_order = k as _;
        let low = (line.bbox[1] + line.bbox[3]) / 2.0 > height / 2.0;
        line.region_type = match REGIONS[usize::from(moves[k].region)] {
            "line_label" => "line_label",
            "furniture" | "page_label" if low => "footer",
            "furniture" | "page_label" => "header",
            "note" | "note_quote" => "footnote",
            "heading" => "heading",
            "title" => "title",
            "marginal_note" => "marginal_note",
            _ => "body",
        }
        .to_owned();
    }
    build_regions(pages);
    let index = PdfTextIndex::from_pages(pages);
    let text_of = |k: usize| slots.line(pages, table.order[k]).text.clone();
    let id_of = |k: usize| slots.line(pages, table.order[k]).id.clone();
    // Units, each line joining the unit open in its stream or opening one.
    let mut state = State::default();
    let mut units: Vec<Unit> = Vec::new();
    let mut open: HashMap<(u16, usize), usize> = HashMap::new();
    // Per context, each open frame's unit (beside the state's stack); per document, each open heading.
    let mut frames: HashMap<u16, Vec<(String, usize)>> = HashMap::new();
    let mut headings: HashMap<u16, Vec<(i8, usize)>> = HashMap::new();
    let mut last_body: HashMap<u16, usize> = HashMap::new();
    for (k, (line, chosen)) in views.iter().zip(moves).enumerate() {
        let made = state.apply(line, chosen);
        if chosen.region == 0 {
            continue;
        }
        let key = (line.ctx, stream_of(chosen.region));
        let continues = !chosen.opens && open.get(&key).is_some_and(|&u| units[u].region == chosen.region);
        let u = if continues {
            open[&key]
        } else {
            let text = text_of(k);
            let reading = chosen.reading.and_then(|r| line.readings.get(usize::from(r)));
            let marker = if chosen.region == NOTE {
                evidence.label[table.order[k].index()].and_then(|label| opening(&text, false).map(|(s, _)| (s, s + usize::from(label.end).max(1))))
            } else {
                reading.and_then(|r| opening(&text, r.family.matches(':').count() > 1))
            };
            let label = reading.map(|r| {
                let mut parts: Vec<String> = r.prefix.iter().map(u32::to_string).collect();
                parts.push(r.value.to_string());
                parts.join(".")
            }).or_else(|| chosen.note.map(|v| v.to_string()));
            let parent = match chosen.region {
                HEADING => {
                    let open_headings = headings.entry(line.document).or_default();
                    open_headings.retain(|&(level, _)| level < chosen.level);
                    open_headings.last().map(|&(_, u)| u)
                }
                NOTE | LINE_LABEL | FURNITURE | PAGE_LABEL | MARGINAL_NOTE => None,
                QUOTE => last_body.get(&line.document).copied(),
                _ => {
                    let below = reading.and_then(|r| {
                        let stack = frames.entry(line.ctx).or_default();
                        let depth = stack.iter().rposition(|(family, _)| *family == r.family).unwrap_or(stack.len());
                        depth.checked_sub(1).map(|d| stack[d].1)
                    });
                    below.or_else(|| headings.get(&line.document).and_then(|h| h.last()).map(|&(_, u)| u))
                }
            };
            units.push(Unit { region: chosen.region, lines: Vec::new(), marker, label, parent, references: Vec::new() });
            let u = units.len() - 1;
            if let Some(r) = reading {
                let stack = frames.entry(line.ctx).or_default();
                match stack.iter().rposition(|(family, _)| *family == r.family) {
                    Some(depth) => {
                        stack.truncate(depth);
                        stack.push((r.family.clone(), u));
                    }
                    None => stack.push((r.family.clone(), u)),
                }
            }
            if chosen.region == HEADING {
                headings.entry(line.document).or_default().push((chosen.level, u));
            }
            if RANKED_BODY.contains(&chosen.region) && chosen.region != QUOTE {
                last_body.insert(line.document, u);
            }
            open.insert(key, u);
            u
        };
        units[u].lines.push(k);
        // The reference mark the note this line opens answered.
        if let Some((place, start, end)) = made.paired {
            units[u].references.push((place as usize, start, end));
        }
    }
    // Nodes, their ranges in scalar characters of the text.
    let line_range = |k: usize| index.line(&id_of(k)).map(|l| l.range);
    let mut nodes: Vec<StructureNode> = Vec::new();
    let mut notes: Vec<Note> = Vec::new();
    let node_id = |u: usize| format!("transition-{:06}", u + 1);
    for page in pages.iter() {
        let Some(range) = index.page_range(page.index) else { continue };
        let mut node = StructureNode::new(page.id.clone(), NodeKind::Page, range, ORIGIN, Derivation::Heuristic, None);
        node.label = Some(format!("page{}", page.index + 1));
        node.page_indexes = vec![page.index];
        node.line_ids = page.lines.iter().map(|l| l.id.clone()).collect();
        nodes.push(node);
    }
    for (u, unit) in units.iter().enumerate() {
        let ranges: Vec<ScalarRange> = unit.lines.iter().filter_map(|&k| line_range(k)).collect();
        let Some(envelope) = ranges.iter().map(|r| r.start).min().zip(ranges.iter().map(|r| r.end).max()) else { continue };
        let range = ScalarRange { start: envelope.0, end: envelope.1 };
        let region = REGIONS[usize::from(unit.region)];
        let (kind, role) = match region {
            "heading" => (NodeKind::Heading, None),
            "text" if unit.marker.is_some() => (NodeKind::Paragraph, None),
            "provision" => (NodeKind::Section, None),
            "list_item" => (NodeKind::ListItem, None),
            "note" | "note_quote" => (NodeKind::Footnote, None),
            "title" => (NodeKind::Prose, Some(PrimaryRole::Title)),
            "caption" => (NodeKind::Prose, Some(PrimaryRole::Caption)),
            "headnote" => (NodeKind::Prose, Some(PrimaryRole::Headnote)),
            "marginal_note" => (NodeKind::Prose, Some(PrimaryRole::MarginalNote)),
            "furniture" => (NodeKind::Prose, Some(PrimaryRole::Furniture)),
            "page_label" => (NodeKind::Prose, Some(PrimaryRole::PageLabel)),
            "line_label" => (NodeKind::Prose, Some(PrimaryRole::LineLabel)),
            _ => (NodeKind::Prose, None),
        };
        let mut node = StructureNode::new(node_id(u), kind, range, ORIGIN, Derivation::Heuristic, unit.parent.map(node_id));
        node.role = role;
        node.ranges = ranges.clone();
        node.line_ids = unit.lines.iter().map(|&k| id_of(k)).collect();
        node.page_indexes = unit.lines.iter().filter_map(|&k| index.line(&id_of(k)).map(|l| l.page_index)).collect();
        node.page_indexes.dedup();
        let marker_range = unit.marker.and_then(|(start, end)| index.global_range(&id_of(unit.lines[0]), start, end));
        if kind != NodeKind::Footnote {
            node.marker_range = marker_range;
            node.label = unit.label.clone();
        }
        match kind {
            NodeKind::Heading => node.heading_range = Some(range),
            NodeKind::Section => node.locator_kind = Some("section".to_owned()),
            _ => {}
        }
        if region == "contents" {
            node.grammar = Some("contents".to_owned());
        }
        nodes.push(node);
        if region == "quote" && unit.lines.len() <= 60 {
            let mut quote = StructureNode::new(format!("{}-quote", node_id(u)), NodeKind::Quote, range, ORIGIN, Derivation::Heuristic, Some(node_id(u)));
            quote.ranges = ranges.clone();
            quote.line_ids = unit.lines.iter().map(|&k| id_of(k)).collect();
            nodes.push(quote);
        }
        if kind == NodeKind::Footnote {
            let label_range = marker_range.unwrap_or(ScalarRange { start: range.start, end: range.start });
            let references: Vec<NoteReference> = unit.references.iter().filter_map(|&(k, start, end)| {
                let range = index.global_range(&id_of(k), start as usize, end as usize)?;
                Some(NoteReference { range, anchor: None, page_indexes: index.line(&id_of(k)).map(|l| vec![l.page_index]).unwrap_or_default(), line_ids: vec![id_of(k)] })
            }).collect();
            notes.push(Note {
                id: format!("{}-note", node_id(u)),
                node_id: Some(node_id(u)),
                kind: NoteKindV2::Footnote,
                label_range,
                body_range: range,
                label_ranges: vec![label_range],
                body_ranges: ranges,
                primary_reference: references.first().map(|r| r.range),
                references,
            });
        }
    }
    let text = index.text().to_owned();
    let digest = {
        use sha2::Digest;
        format!("{:x}", sha2::Sha256::digest(text.as_bytes()))
    };
    let mut structure = DocumentStructure::from_scalar_parts(
        identity.document_id.clone(),
        "local-pdf".to_owned(),
        text,
        digest,
        Some(identity.source_sha256.clone()),
        Scope::complete(),
        vec![Origin { id: ORIGIN.to_owned() }],
        nodes,
        notes,
        Vec::new(),
    );
    legal_structure_model::assemble_in_place(&mut structure).map_err(Error::Message)?;
    structure.profile = identity.profile;
    structure.source_fragments = index.source_fragments();
    structure.refresh_revision().map_err(|e| Error::Message(e.to_string()))?;
    Ok(structure)
}
