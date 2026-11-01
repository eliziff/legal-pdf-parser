//! The transition parser's reading as the document consumers read: its structure graph, built from
//! the moves and the state they move, nothing from the stage parser's decisions. Each unit is a node:
//! a numbered paragraph a `Paragraph` (its label and marker), a provision a `Section` (located by its
//! number, its path from the outermost provision its alias), a heading a `Heading`, a list item a
//! `ListItem`, any other body text `Prose` (a quotation also a `Quote` over it, a contents list's lines
//! under the `contents` grammar, a transcript's turn opening on its speaker's label), a note a
//! `Footnote` with its pair (its label, its body, the reference marks it answered), a line set apart a
//! `Prose` of its role (furniture, a page label, a transcript's line number each a line labelled by its
//! text, a title, a marginal note, a caption, a headnote), and the tables the parser read as tables
//! with their rows and cells. Each page carries its printed label as its alias. Parents follow the
//! gold's own hierarchy rule (`labels::assemble`).

use super::*;
use legal_structure_model::{Derivation, DocumentStructure, Note, NoteKindV2, NoteReference, Origin, PrimaryRole, Scope, StructureNode};


struct Unit {
    region: u8,
    lines: Vec<usize>,
    marker: Option<(usize, usize)>,
    label: Option<String>,
    /// A heading's level, a body unit's depth (0 for neither).
    level: usize,
    depth: usize,
    starts_document: bool,
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

/// The document's structure graph from its lines (`table`, `views`) and the moves they made, with the
/// paragraphs and footnotes the projection reads.
pub(crate) fn structure_graph(
    pages: &mut [Page],
    slots: &Slots,
    evidence: &Evidence,
    fences: &Fences,
    table: &LineTable,
    views: &[LineView],
    moves: &[Move],
    identity: &StructureIdentity,
) -> Result<(DocumentStructure, Vec<Paragraph>, Vec<Footnote>)> {
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
    let texts: Vec<String> = table.order.iter().map(|&slot| slots.line(pages, slot).text.clone()).collect();
    let ids: Vec<String> = table.order.iter().map(|&slot| slots.line(pages, slot).id.clone()).collect();
    let text_of = |k: usize| texts[k].clone();
    let id_of = |k: usize| ids[k].clone();
    // Units, each line joining the unit open in its stream, resuming the one it held before, or opening
    // one. Their parents follow the gold's own rule (`labels::assemble`): from each unit's region, its
    // opening text, a heading's level, a body unit's depth among the body units open on the stack, and
    // whether it opens a constituent document.
    let mut state = State::default();
    let mut units: Vec<Unit> = Vec::new();
    let mut open: HashMap<(u16, usize), usize> = HashMap::new();
    // The last unit of each region in each context's stream, which a line may resume.
    let mut region_unit: HashMap<(u16, usize, u8), usize> = HashMap::new();
    // The lines the parser opened a constituent document on; a start on a line of no region passes to
    // the next unit.
    let mut started: Vec<usize> = Vec::new();
    let mut pending_start = false;
    for (k, (line, chosen)) in views.iter().zip(moves).enumerate() {
        let made = state.apply(line, chosen);
        if chosen.starts {
            started.push(k);
            pending_start = true;
        }
        if chosen.region == 0 {
            continue;
        }
        let key = (state.ctx(line, false), stream_of(chosen.region));
        // A transcript's line number and a page's label are a line each.
        let single = matches!(chosen.region, LINE_LABEL | PAGE_LABEL);
        let continues = !single && !chosen.opens && open.get(&key).is_some_and(|&u| units[u].region == chosen.region);
        let resumed = (!single && chosen.resume).then(|| region_unit.get(&(key.0, key.1, chosen.region)).copied()).flatten();
        let u = if continues {
            open[&key]
        } else if let Some(u) = resumed {
            open.insert(key, u);
            u
        } else {
            let text = text_of(k);
            let reading = chosen.reading.and_then(|r| line.readings.get(usize::from(r)));
            let marker = if chosen.region == NOTE {
                evidence.label[table.order[k].index()].and_then(|label| opening(&text, false).map(|(s, _)| (s, s + usize::from(label.end).max(1))))
            } else {
                reading.and_then(|r| opening(&text, r.family.matches(':').count() > 1))
            };
            // A unit's label is its marker as printed ("12", "(a)", "[4]"), a note's its number, a line
            // number's or page label's its text.
            let printed = |(start, end): (usize, usize)| text.chars().skip(start).take(end - start).collect::<String>().trim().trim_end_matches('.').to_owned();
            let label = if single {
                Some(text.trim().to_owned())
            } else if chosen.region == NOTE {
                chosen.note.map(|v| v.to_string())
            } else {
                marker.map(printed)
            };
            let level = if chosen.opens && chosen.region == HEADING { chosen.level.max(1) as usize } else { 0 };
            let depth = if chosen.opens && RANKED_BODY.contains(&chosen.region) {
                state.outlines.get(&state.outline_key(line, false)).map_or(1, |outline| outline.body_depth.max(1))
            } else {
                0
            };
            units.push(Unit { region: chosen.region, lines: Vec::new(), marker, label, level, depth, starts_document: std::mem::take(&mut pending_start), references: Vec::new() });
            let u = units.len() - 1;
            open.insert(key, u);
            region_unit.insert((key.0, key.1, chosen.region), u);
            u
        };
        units[u].lines.push(k);
        // The reference mark the note this line opens answered.
        if let Some((place, start, end)) = made.paired {
            units[u].references.push((place as usize, start, end));
        }
    }
    let parents = {
        let regions: Vec<&str> = units.iter().map(|unit| REGIONS[usize::from(unit.region)]).collect();
        // A unit's opening text: its first lines joined, a marker often standing on a line of its own.
        let texts: Vec<String> = units.iter().map(|unit| unit.lines.iter().take(3).map(|&k| text_of(k)).collect::<Vec<_>>().join(" ")).collect();
        let levels: Vec<usize> = units.iter().map(|unit| unit.level).collect();
        let depths: Vec<usize> = units.iter().map(|unit| unit.depth).collect();
        let starts: Vec<bool> = units.iter().map(|unit| unit.starts_document).collect();
        legal_pdf_tagger::labels::assemble(&regions, &texts, &levels, &depths, &starts)
    };
    // What the lines tell a reader of the pages: a line number is no part of the body; a note's lines
    // on the pages after its first continue it.
    for unit in &units {
        let first_page = unit.lines.first().map(|&k| slots.page(table.order[k]));
        for &k in &unit.lines {
            let slot = table.order[k];
            let later = Some(slots.page(slot)) != first_page;
            let line = slots.line_mut(pages, slot);
            line.exclude_from_body = unit.region == LINE_LABEL;
            line.note_region_mode = match (unit.region, later) {
                (NOTE, true) => "footnote_continuation",
                (NOTE, false) => "footnote",
                _ => "",
            }
            .to_owned();
        }
    }
    // Each page's printed label: of the lines the parser read as the page's label, the one the
    // folios' offset names, else the first whose text reads as a label (`furniture::printed_label`).
    for (page_slot, page) in pages.iter_mut().enumerate() {
        let labelled: Vec<(String, usize)> = units
            .iter()
            .filter(|unit| unit.region == PAGE_LABEL)
            .flat_map(|unit| unit.lines.iter().copied())
            .filter(|&k| slots.page(table.order[k]) == page_slot)
            .filter_map(|k| {
                let line = (table.order[k].0 - slots.on(page_slot).start) as usize;
                Some((crate::structure::furniture::printed_label(&page.lines[line].text)?, line))
            })
            .collect();
        let folio = fences.folios[page_slot].as_ref().map(|(slot, _)| (slot.0 - slots.on(page_slot).start) as usize);
        let chosen = labelled.iter().find(|(_, line)| Some(*line) == folio).or(labelled.first());
        page.printed_label = chosen.map(|(label, _)| label.clone());
        page.printed_label_source = chosen.map(|&(_, line)| page.lines[line].region_type.clone());
        page.printed_label_line_id = chosen.map(|&(_, line)| page.lines[line].id.clone());
    }
    // A provision's path from the outermost provision it stands in: "12(1)(a)".
    let section_path = |u: usize| {
        let mut chain = vec![u];
        while let Some(parent) = parents[*chain.last().unwrap()].filter(|&p| REGIONS[usize::from(units[p].region)] == "provision") {
            chain.push(parent);
        }
        let labels: Vec<&str> = chain.iter().rev().filter_map(|&u| units[u].label.as_deref()).collect();
        (labels.len() > 1).then(|| {
            labels.iter().enumerate().map(|(at, label)| {
                if at == 0 || label.starts_with('(') { (*label).to_owned() } else { format!("({label})") }
            }).collect::<String>()
        })
    };
    // Nodes, their ranges in scalar characters of the text.
    let line_range = |k: usize| index.line(&id_of(k)).map(|l| l.range);
    let mut nodes: Vec<StructureNode> = Vec::new();
    let mut notes: Vec<Note> = Vec::new();
    let node_id = |u: usize| format!("transition-{:06}", u + 1);
    for page in pages.iter() {
        let Some(range) = index.page_range(page.index) else { continue };
        let mut node = StructureNode::new(page.id.clone(), NodeKind::Page, range, ORIGIN, Derivation::Heuristic, None);
        node.label = Some(format!("page{}", page.number));
        node.aliases = page.printed_label.clone().map(|label| vec![label]);
        node.anchor.clone_from(&page.printed_label_line_id);
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
        let mut node = StructureNode::new(node_id(u), kind, range, ORIGIN, Derivation::Heuristic, parents[u].map(node_id));
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
            NodeKind::Section => {
                node.locator_kind = Some("section".to_owned());
                node.aliases = section_path(u).map(|path| vec![path]);
            }
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
    // The tables, rows and cells the table reader found, of those whose lines the parser read as a
    // table's for the most part.
    let table_region: HashSet<String> = units.iter().filter(|unit| REGIONS[usize::from(unit.region)] == "table")
        .flat_map(|unit| unit.lines.iter().map(|&k| id_of(k))).collect();
    let kept = fences.tables.iter().enumerate().filter(|(_, table)| {
        let ids = table.line_ids();
        ids.iter().filter(|id| table_region.contains(id.as_str())).count() * 2 > ids.len()
    });
    nodes.extend(crate::structure::graph::table_nodes(&index, kept, ORIGIN)?);
    // The constituent documents the parser opened, each of the kind the constituents reader names its
    // opening line, else a document.
    let starts: Vec<constituents::Start> = started.iter().map(|&k| {
        let id = id_of(k);
        let kind = fences.starts.iter().find(|start| start.line_id == id).map_or("document", |start| start.kind()).to_owned();
        constituents::Start::new(id, &kind)
    }).collect();
    nodes.extend(constituents::constituents(&index, pages, &starts));
    // The paragraphs and footnotes the projection reads (authority units, lookups): every body unit's text,
    // each line's span in it and where its paired notes' reference marks stand; every note's.
    let page_of = |k: usize| slots.page(table.order[k]);
    // A unit's lines joined as the stage's paragraphs join them (hyphenation by the document's words).
    let vocabulary = crate::structure::joining::vocabulary(pages);
    let joined = |unit: &Unit| {
        let lines: Vec<&Line> = unit.lines.iter().map(|&k| slots.line(pages, table.order[k])).collect();
        crate::structure::joining::join_lines(&lines, &vocabulary)
    };
    // Each paired reference mark, by its line: its note's pair and label, where it stands in the line.
    let mut marks: HashMap<String, Vec<legal_pdf_core::Anchor>> = HashMap::new();
    for (u, unit) in units.iter().enumerate().filter(|(_, unit)| unit.region == NOTE) {
        for &(k, start, end) in &unit.references {
            marks.entry(id_of(k)).or_default().push(legal_pdf_core::Anchor {
                pair_id: format!("{}-note", node_id(u)),
                label: unit.label.clone().unwrap_or_default(),
                start: start as usize,
                end: end as usize,
            });
        }
    }
    let mut paragraphs = Vec::new();
    let mut footnotes = Vec::new();
    for (u, unit) in units.iter().enumerate() {
        let region = REGIONS[usize::from(unit.region)];
        if matches!(region, "furniture" | "page_label" | "line_label" | "marginal_note" | "none") {
            continue;
        }
        let (text, spans) = joined(unit);
        let pages_of: Vec<u32> = {
            let mut found: Vec<u32> = unit.lines.iter().map(|&k| pages[page_of(k)].number).collect();
            found.dedup();
            found
        };
        if unit.region == NOTE {
            let reference = unit.references.first().map(|&(k, _, _)| k);
            footnotes.push(Footnote {
                source_spans: spans,
                pair_id: format!("{}-note", node_id(u)),
                label: unit.label.clone().unwrap_or_default(),
                occurrence: 1,
                restart_sequence: 0,
                reference_page: reference.map(|k| pages[page_of(k)].number),
                body_pages: pages_of,
                reference_line_id: reference.map(id_of),
                body_line_ids: unit.lines.iter().map(|&k| id_of(k)).collect(),
                body: text,
                sentence_proposition: String::new(),
                passage_since_prior_note: String::new(),
                confidence: 1.0,
                provenance: ORIGIN.to_owned(),
                warnings: Vec::new(),
                crossrefs: Vec::new(),
            });
            continue;
        }
        // The paragraph's text as the stage writes it: each paired reference mark its note's marker.
        let lines: Vec<&Line> = unit.lines.iter().map(|&k| slots.line(pages, table.order[k])).collect();
        let Some((text, anchors, spans)) = crate::structure::joining::anchored_text(&lines, &vocabulary, &marks) else { continue };
        paragraphs.push(Paragraph {
            source_spans: spans,
            id: node_id(u),
            page_index: pages[page_of(unit.lines[0])].index,
            region_type: if unit.region == HEADING { "heading".to_owned() } else { "body".to_owned() },
            text,
            line_ids: unit.lines.iter().map(|&k| id_of(k)).collect(),
            anchors,
        });
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
    Ok((structure, paragraphs, footnotes))
}
