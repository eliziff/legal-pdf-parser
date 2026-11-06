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
//! gold's own hierarchy rule (the engine's `assemble_parents`).

use super::*;
use legal_structure_model::{Derivation, DocumentStructure, Note, NoteKindV2, NotePairClaimV2, NoteReference, Origin, PrimaryRole, Scope, StructureDiagnostic, StructureNode};

/// What tier 0 settled before its moves (`tier0`): its lines are written and their regions built, the text indexed, and
/// its notes the pairer's: their claims, their reference marks by line, the projection's notes, the claims left aside;
/// and the run member each unit opens on, by the unit's first line (`runs`).
pub(crate) struct Settled<'a> {
    pub(crate) index: &'a PdfTextIndex,
    pub(crate) claims: Vec<NotePairClaimV2>,
    pub(crate) anchors: &'a HashMap<String, Vec<legal_pdf_core::Anchor>>,
    pub(crate) footnotes: Vec<Footnote>,
    pub(crate) diagnostics: Vec<StructureDiagnostic>,
    pub(crate) runs: &'a Runs,
    pub(crate) opened: HashMap<usize, (u32, u32)>,
}

struct Unit {
    region: u8,
    lines: Vec<usize>,
    marker: Option<(usize, usize)>,
    /// The line `marker` stands on: the unit's first, or the margin's line numbering it.
    marker_line: usize,
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

/// A table line's cells where no reader found them: its spans in order, a new cell where the gap to the
/// next is wider than two of the line's characters; each cell's first and past-last character and its
/// left edge.
fn gap_cells(line: &Line) -> Vec<(usize, usize, f64)> {
    let chars: Vec<char> = line.text.chars().collect();
    let width = line.bbox[2] - line.bbox[0];
    let glyph = if chars.is_empty() { 0.0 } else { width / chars.len() as f64 };
    let mut cells: Vec<(usize, usize, f64, f64)> = Vec::new();
    let mut cursor = 0;
    for span in &line.spans {
        let text: Vec<char> = span.text.trim().chars().collect();
        if text.is_empty() || text.len() > chars.len() {
            continue;
        }
        let Some(at) = (cursor..=chars.len().saturating_sub(text.len())).find(|&at| chars[at..at + text.len()] == text[..]) else { continue };
        cursor = at + text.len();
        match cells.last_mut() {
            Some(last) if span.bbox[0] - last.2 <= glyph * 2.0 => {
                last.1 = cursor;
                last.2 = span.bbox[2];
            }
            _ => cells.push((at, cursor, span.bbox[2], span.bbox[0])),
        }
    }
    cells.into_iter().map(|(start, end, _, left)| (start, end, left)).collect()
}

/// Two gap cells whose left edges stand this near (points) are in one column.
const COLUMN_TOLERANCE: f64 = 12.0;

/// The figures a line's text opens on: what they read, and their first and past-last character.
pub(super) fn leading_figures(text: &str) -> Option<(u32, (usize, usize))> {
    let chars: Vec<char> = text.chars().collect();
    let start = chars.iter().position(|c| !c.is_whitespace())?;
    let length = chars[start..].iter().take_while(|c| c.is_ascii_digit()).count();
    let figures: String = chars[start..start + length].iter().collect();
    Some((figures.parse().ok()?, (start, start + length)))
}

/// The document's structure graph from its lines (`table`, `views`) and the moves they made, with the
/// paragraphs and footnotes the projection reads. With `settled`, what tier 0 wrote and paired before its moves stands.
pub(crate) fn structure_graph(
    pages: &mut [Page],
    slots: &Slots,
    evidence: &Evidence,
    fences: &Fences,
    table: &LineTable,
    views: &[LineView],
    moves: &[Move],
    identity: &StructureIdentity,
    mut settled: Option<Settled>,
) -> Result<(DocumentStructure, Vec<Paragraph>, Vec<Footnote>)> {
    let written = settled.is_some();
    if !written {
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
        // The reading's order of each page's lines, where the parse reads them in its own.
        for (k, chosen) in moves.iter().enumerate() {
            if let Some(rank) = chosen.rank {
                slots.line_mut(pages, table.order[k]).reading_order = rank as usize;
            }
        }
        build_regions(pages);
    }
    let built;
    let index = match &settled {
        Some(read) => read.index,
        None => {
            built = PdfTextIndex::from_pages(pages);
            &built
        }
    };
    let texts: Vec<String> = table.order.iter().map(|&slot| slots.line(pages, slot).text.clone()).collect();
    let ids: Vec<String> = table.order.iter().map(|&slot| slots.line(pages, slot).id.clone()).collect();
    let text_of = |k: usize| texts[k].clone();
    let id_of = |k: usize| ids[k].clone();
    // Units, each line joining the unit open in its stream, resuming the one it held before, or opening
    // one. Their parents follow the gold's own rule (the engine's `assemble_parents`): from each unit's region, its
    // opening text, a heading's level, a body unit's depth among the body units open on the stack, and
    // whether it opens a constituent document.
    let mut state = State::default();
    let mut units: Vec<Unit> = Vec::new();
    let mut streams = Streams::default();
    // Each unit by its first line.
    let mut unit_at: HashMap<usize, usize> = HashMap::new();
    // The lines the parser opened a constituent document on; a start on a line of no region passes to the next unit.
    let mut started: Vec<usize> = Vec::new();
    for (k, (line, chosen)) in views.iter().zip(moves).enumerate() {
        let made = state.apply(line, chosen);
        if chosen.starts {
            started.push(k);
        }
        if chosen.region == 0 {
            continue;
        }
        let key = (state.ctx(line, false), stream_of(chosen.region));
        let single = matches!(chosen.region, LINE_LABEL | PAGE_LABEL);
        let stands = streams.stands(key, chosen);
        let held = match stands {
            Stands::GoesOn(first) | Stands::Resumes(first) => unit_at.get(&first).copied(),
            Stands::Opens => None,
        };
        let u = if let Some(u) = held {
            u
        } else {
            // A unit counting on the margin's number prints its marker on the margin's line.
            let marker_line = match (chosen.reading, line.margin) {
                (Some(MARGIN), Some((margin, _))) => margin as usize,
                _ => k,
            };
            let text = text_of(marker_line);
            let reading = line.reading(chosen.reading);
            // A note's label is the number its move counts on: where the label reader found it, else the
            // figures its opening token starts on ("27" of "27Ibid"); a note counting on none has none.
            let marker = if chosen.region == NOTE {
                chosen.note.and_then(|value| {
                    let found = evidence.label[table.order[k].index()].filter(|label| label.value == value);
                    found.and_then(|label| opening(&text, false).map(|(s, _)| (s, s + usize::from(label.end).max(1))))
                        .or_else(|| leading_figures(&text).filter(|&(figures, _)| figures == value).map(|(_, range)| range))
                })
            } else if chosen.reading == Some(OPENING) {
                opening(&text, false)
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
            let depth = if chosen.opens && RANKED_BODY.contains(&chosen.region) { usize::from(chosen.depth.max(1)) } else { 0 };
            let starts_document = started.last().is_some_and(|&line| units.last().is_none_or(|unit: &Unit| unit.lines[0] < line));
            units.push(Unit { region: chosen.region, lines: Vec::new(), marker, marker_line, label, level, depth, starts_document, references: Vec::new() });
            unit_at.insert(k, units.len() - 1);
            units.len() - 1
        };
        units[u].lines.push(k);
        streams.step(key, chosen.region, if held.is_some() { stands } else { Stands::Opens }, k);
        // The reference mark the note this line opens answered.
        for (place, start, end) in made.paired.into_iter().chain(made.also.into_iter().flatten()) {
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
        legal_structure_model::structure_analysis().assemble_parents(&regions, &texts, &levels, &depths, &starts)
    };
    if !written {
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
    let line_range = |k: usize| index.line(&ids[k]).map(|l| l.range);
    let mut nodes: Vec<StructureNode> = Vec::new();
    let mut notes: Vec<Note> = Vec::new();
    // A written page's line set apart keeps the id its line's kind gives it, as the projection reads such lines.
    let folios: HashSet<&str> = pages.iter().filter_map(|page| page.printed_label_line_id.as_deref()).collect();
    let unit_ids: Vec<String> = units.iter().enumerate().map(|(u, unit)| {
        let id = ids[unit.lines[0]].as_str();
        match REGIONS[usize::from(unit.region)] {
            "line_label" if written => format!("{id}:line"),
            "furniture" | "page_label" if written => format!("{id}:{}", if folios.contains(id) { "page" } else { "furniture" }),
            _ => format!("transition-{:06}", u + 1),
        }
    }).collect();
    let node_id = |u: usize| unit_ids[u].clone();
    let nothing_decided = Decision::default();
    let decided = Decided::of(pages, slots, fences, &nothing_decided);
    let page_of_line = |k: usize| slots.page(table.order[k]);
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
        // A table's units are its rows, built with the table below; where the fences' tables are written, its text stays
        // the body's beside them.
        if unit.region == TABLE && !written {
            continue;
        }
        // The pairer's notes are written from its claims below.
        if written && unit.region == NOTE {
            continue;
        }
        let ranges: Vec<ScalarRange> = unit.lines.iter().filter_map(|&k| line_range(k)).collect();
        let Some(envelope) = ranges.iter().map(|r| r.start).min().zip(ranges.iter().map(|r| r.end).max()) else { continue };
        let range = ScalarRange { start: envelope.0, end: envelope.1 };
        let region = REGIONS[usize::from(unit.region)];
        // The member a settled unit opens on; a heading numbering a provision is that provision, a heading still.
        let member = settled.as_ref().and_then(|read| {
            let &(run, at) = read.opened.get(&unit.lines[0])?;
            Some(&read.runs.runs[run as usize].members[at as usize])
        });
        let (kind, role) = match region {
            "heading" if member.is_some_and(|member| member.kind == Some(legal_structure_model::UnitRole::Section)) => (NodeKind::Section, Some(PrimaryRole::Heading)),
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
        node.page_indexes = unit.lines.iter().filter_map(|&k| index.line(&ids[k]).map(|l| l.page_index)).collect();
        node.page_indexes.dedup();
        let marker_range = unit.marker.and_then(|(start, end)| index.global_range(&id_of(unit.marker_line), start, end));
        if kind != NodeKind::Footnote {
            node.marker_range = marker_range;
            // A numbered paragraph is addressed by its number ("par12" for "[12]" or "12."), as every reader labels it.
            node.label = if kind == NodeKind::Paragraph {
                unit.label.as_deref().map(|label| label.trim_matches(|c: char| !c.is_ascii_digit()))
                    .filter(|digits| !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())).map(|digits| format!("par{digits}"))
            } else {
                unit.label.clone()
            };
        }
        match kind {
            NodeKind::Heading => {
                node.heading_range = Some(range);
                if written {
                    let lines: Vec<&str> = unit.lines.iter().map(|&k| texts[k].as_str()).collect();
                    node.label = Some(lines.join(" "));
                    node.marker_range = ranges.first().and_then(|first| crate::structure::graph::heading_marker(first.start, lines[0], lines.len()));
                }
            }
            NodeKind::Section => {
                node.locator_kind = Some("section".to_owned());
                node.aliases = section_path(u).map(|path| vec![path]);
            }
            _ => {}
        }
        // A settled unit is addressed as the run it opens on reads it: a paragraph by its count, a provision by its
        // grammar's value (a clause, a part, an article), its printing an alias; its marker and text where they stand.
        if let Some(member) = member {
            let scalar = |place: Place| index.line(&ids[fences.place[place.slot.index()] as usize]).map(|line| line.range.start + place.offset as usize);
            if let Some((start, end)) = scalar(member.marker[0]).zip(scalar(member.marker[1])) {
                node.marker_range = Some(ScalarRange { start, end: end.max(start) });
            }
            node.content_start = scalar(member.content);
            match kind {
                NodeKind::Paragraph => node.label = Some(format!("par{}", member.grammar_value)),
                NodeKind::ListItem => node.label = Some(member.label.clone()),
                NodeKind::Section => {
                    node.label = Some(member.grammar_value.clone());
                    node.locator_kind = Some(legal_structure_model::provision_locator(&member.grammar_value, usize::from(member.level)).to_owned());
                    let printed = (member.label != member.grammar_value).then(|| member.label.clone());
                    node.aliases = Some(printed.into_iter().chain(node.aliases.take().into_iter().flatten()).collect()).filter(|aliases: &Vec<String>| !aliases.is_empty());
                    if role == Some(PrimaryRole::Heading) {
                        node.heading_range = Some(range);
                    }
                }
                _ => {}
            }
        }
        // Its region as the tagger states one (`region:<name>`), which readers take for the unit's kind; a
        // contents list's lines under the `contents` grammar.
        // A numbered paragraph wholly in the translation prints that paragraph in the other language
        // (`paragraph_printings`), as the stage's nodes say.
        node.grammar = Some(if region == "contents" {
            "contents".to_owned()
        } else if (kind == NodeKind::Paragraph || written && kind == NodeKind::Prose) && unit.lines.iter().all(|&k| views[k].translation) {
            "translation".to_owned()
        } else {
            format!("region:{region}")
        });
        // Its level or depth as the parser reads it, the outline's own: a heading two levels under the
        // one before stays two under, though no heading stands between them in the tree.
        if unit.level > 0 {
            node.markup_attributes.insert("heading_level".to_owned(), unit.level.to_string());
        }
        if unit.depth > 0 {
            node.markup_attributes.insert("depth".to_owned(), unit.depth.to_string());
        }
        nodes.push(node);
        if region == "quote" && unit.lines.len() <= 60 && !written {
            let mut quote = StructureNode::new(format!("{}-quote", node_id(u)), NodeKind::Quote, range, ORIGIN, Derivation::Heuristic, Some(node_id(u)));
            quote.ranges = ranges.clone();
            quote.line_ids = unit.lines.iter().map(|&k| id_of(k)).collect();
            nodes.push(quote);
        }
        if kind == NodeKind::Footnote {
            let label_range = marker_range.unwrap_or(ScalarRange { start: range.start, end: range.start });
            let references: Vec<NoteReference> = unit.references.iter().filter_map(|&(k, start, end)| {
                let range = index.global_range(&id_of(k), start as usize, end as usize)?;
                Some(NoteReference { range, anchor: None, page_indexes: index.line(&ids[k]).map(|l| vec![l.page_index]).unwrap_or_default(), line_ids: vec![id_of(k)] })
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
    // The tables the parser read: a run of table units no other body unit breaks (notes and the page's
    // furniture break none) is one table, one grid a page linked to the next page's; each unit a row, its
    // cells the table reader's on its lines by their column (a line of one cell under a row of several
    // goes on in the row's last column), a row the reader has no cells on one cell.
    let mut reader_cells: HashMap<String, Vec<(usize, Option<(usize, usize)>)>> = HashMap::new();
    for found in &fences.tables {
        for row in &found.rows {
            for cell in &row.cells {
                for part in &cell.parts {
                    reader_cells.entry(part.line_id.clone()).or_default().push((cell.column, part.chars));
                }
            }
        }
    }
    // The pairer's notes: each claim its node and its note, over the lines it claims, whatever the moves made of them.
    for (n, claim) in settled.iter().flat_map(|read| &read.claims).enumerate() {
        let id = format!("transition-note-{:06}", n + 1);
        let kind = if claim.kind == NoteKindV2::Endnote { NodeKind::Endnote } else { NodeKind::Footnote };
        let mut node = StructureNode::new(id.clone(), kind, claim.body.range, ORIGIN, Derivation::Heuristic, None);
        node.ranges = claim.body.ranges.clone();
        node.line_ids = claim.body.line_ids.clone();
        node.page_indexes = claim.body.page_indexes.clone();
        node.grammar = Some("region:note".to_owned());
        nodes.push(node);
        notes.push(Note {
            id: claim.pair_id.clone(),
            node_id: Some(id),
            kind: claim.kind,
            label_range: claim.label.range,
            body_range: claim.body.range,
            label_ranges: vec![claim.label.range],
            body_ranges: claim.body.ranges.clone(),
            primary_reference: claim.references.first().map(|reference| reference.range),
            references: claim.references.iter().map(|reference| NoteReference {
                range: reference.range,
                anchor: None,
                page_indexes: vec![reference.page_index],
                line_ids: vec![reference.line_id.clone()],
            }).collect(),
        });
    }
    if written {
        nodes.extend(crate::structure::graph::table_nodes(index, fences.tables.iter().enumerate(), ORIGIN)?);
        nodes.extend(crate::structure::graph::quotation_nodes(index, &decided, ORIGIN)?);
    }
    let mut runs: Vec<Vec<usize>> = Vec::new();
    let breaks = |unit: &Unit| !matches!(REGIONS[usize::from(unit.region)], "table" | "note" | "note_quote" | "furniture" | "page_label" | "line_label" | "marginal_note");
    for u in (0..units.len()).filter(|&u| !written && units[u].region == TABLE) {
        match runs.last_mut().filter(|run: &&mut Vec<usize>| !units[run[run.len() - 1]..u].iter().any(breaks)) {
            Some(run) => run.push(u),
            None => runs.push(vec![u]),
        }
    }
    let mut continuations = Vec::new();
    for (t, run) in runs.iter().enumerate() {
        let mut chain: Vec<usize> = Vec::new();
        let mut by_page: Vec<(usize, Vec<usize>)> = Vec::new();
        for &u in run {
            let page = page_of_line(units[u].lines[0]);
            match by_page.iter_mut().find(|(p, _)| *p == page) {
                Some((_, rows)) => rows.push(u),
                None => by_page.push((page, vec![u])),
            }
        }
        for (g, (_, rows)) in by_page.iter().enumerate() {
            let table_id = format!("transition-table-{:04}-{:03}", t + 1, g + 1);
            let mut table_ranges: Vec<ScalarRange> = Vec::new();
            let mut grid_nodes: Vec<StructureNode> = Vec::new();
            // The grid's columns where the reader found none: the left edges the rows' gap cells stand at,
            // near ones one column.
            let mut lefts: Vec<f64> = rows.iter().flat_map(|&u| units[u].lines.iter().copied())
                .filter(|&k| !reader_cells.contains_key(&id_of(k)))
                .flat_map(|k| gap_cells(slots.line(pages, table.order[k])).into_iter().map(|(_, _, left)| left))
                .collect();
            lefts.sort_by(f64::total_cmp);
            lefts.dedup_by(|later, earlier| (*later - *earlier).abs() <= COLUMN_TOLERANCE);
            for (r, &u) in rows.iter().enumerate() {
                let unit = &units[u];
                let row_ranges: Vec<ScalarRange> = unit.lines.iter().filter_map(|&k| line_range(k)).collect();
                let Some(hull) = row_ranges.iter().map(|x| x.start).min().zip(row_ranges.iter().map(|x| x.end).max()) else { continue };
                let row_id = node_id(u);
                let mut row = StructureNode::new(row_id.clone(), NodeKind::Row, ScalarRange { start: hull.0, end: hull.1 }, ORIGIN, Derivation::Heuristic, Some(table_id.clone()));
                row.role = Some(PrimaryRole::Text);
                row.grammar = Some("region:table".to_owned());
                row.ranges = row_ranges.clone();
                row.row = Some(r + 1);
                row.line_ids = unit.lines.iter().map(|&k| id_of(k)).collect();
                row.page_indexes = unit.lines.iter().filter_map(|&k| index.line(&ids[k]).map(|l| l.page_index)).collect();
                row.page_indexes.dedup();
                // The row's cells by column.
                let columns = reader_cells.get(&id_of(unit.lines[0])).map_or(0, |cells| cells.iter().map(|(c, _)| c + 1).max().unwrap_or(0));
                let mut cells: std::collections::BTreeMap<usize, Vec<ScalarRange>> = std::collections::BTreeMap::new();
                for &k in &unit.lines {
                    let id = id_of(k);
                    match reader_cells.get(&id) {
                        Some(found) => {
                            let wrapped = columns >= 2 && found.iter().map(|(c, _)| *c).collect::<HashSet<_>>().len() == 1;
                            for &(column, chars) in found {
                                let range = match chars {
                                    Some((start, end)) => index.global_range(&id, start, end),
                                    None => line_range(k),
                                };
                                if let Some(range) = range {
                                    cells.entry(if wrapped { columns - 1 } else { column }).or_default().push(range);
                                }
                            }
                        }
                        // No reader cells: the line's own columns, split where a wide gap parts its spans,
                        // each in the grid's column its left edge stands in.
                        None => {
                            let parts = gap_cells(slots.line(pages, table.order[k]));
                            if parts.len() >= 2 {
                                for (start, end, left) in parts {
                                    let column = lefts.iter().position(|&x| (x - left).abs() <= COLUMN_TOLERANCE).unwrap_or(lefts.len().saturating_sub(1));
                                    cells.entry(column).or_default().extend(index.global_range(&id, start, end));
                                }
                            } else {
                                cells.entry(if columns >= 2 { columns - 1 } else { 0 }).or_default().extend(line_range(k));
                            }
                        }
                    }
                }
                for (column, mut ranges) in cells {
                    ranges.sort_by_key(|x| x.start);
                    let Some(&first) = ranges.first() else { continue };
                    let end = ranges.iter().map(|x| x.end).max().unwrap_or(first.end);
                    let mut cell = StructureNode::new(format!("{row_id}-c{:03}", column + 1), NodeKind::Cell, ScalarRange { start: first.start, end }, ORIGIN, Derivation::Heuristic, Some(row_id.clone()));
                    cell.ranges = ranges;
                    cell.row = Some(r + 1);
                    cell.column = Some(column + 1);
                    cell.page_indexes = row.page_indexes.clone();
                    grid_nodes.push(cell);
                }
                table_ranges.extend(row_ranges);
                grid_nodes.push(row);
            }
            let (Some(start), Some(end)) = (table_ranges.iter().map(|x| x.start).min(), table_ranges.iter().map(|x| x.end).max()) else { continue };
            let mut table_node = StructureNode::new(table_id, NodeKind::Table, ScalarRange { start, end }, ORIGIN, Derivation::Heuristic, parents[rows[0]].map(node_id));
            table_node.ranges = table_ranges;
            table_node.table_number = Some(t + 1);
            table_node.line_ids = rows.iter().flat_map(|&u| units[u].lines.iter().map(|&k| id_of(k))).collect();
            table_node.page_indexes = index.page_indexes_for_line_ids(&table_node.line_ids);
            chain.push(start);
            nodes.push(table_node);
            nodes.extend(grid_nodes);
        }
        for pair in chain.windows(2) {
            continuations.push(legal_structure_model::SourceContinuation { kind: legal_structure_model::ContinuationKind::Table, from: pair[0], to: pair[1], separator: None });
        }
    }
    // Text the parser read as no region's lies outside the body, as the gold leaves it: furniture.
    for (k, chosen) in moves.iter().enumerate() {
        if chosen.region != 0 || texts[k].trim().is_empty() {
            continue;
        }
        let Some(range) = line_range(k) else { continue };
        let mut node = StructureNode::new(format!("transition-none-{:06}", k + 1), NodeKind::Prose, range, ORIGIN, Derivation::Heuristic, None);
        node.role = Some(PrimaryRole::Furniture);
        node.grammar = Some("region:furniture".to_owned());
        node.ranges = vec![range];
        node.page_indexes = index.line(&ids[k]).map(|l| vec![l.page_index]).unwrap_or_default();
        node.line_ids = vec![id_of(k)];
        nodes.push(node);
    }
    // The constituent documents the parser opened, each of the kind the constituents reader names its
    // opening line, else a document.
    let starts: Vec<constituents::Start> = started.iter().map(|&k| {
        let id = id_of(k);
        let kind = fences.starts.iter().find(|start| start.line_id == id).map_or("document", |start| start.kind()).to_owned();
        constituents::Start::new(id, &kind)
    }).collect();
    nodes.extend(constituents::constituents(index, pages, &starts));
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
    let mut paired: HashMap<String, Vec<legal_pdf_core::Anchor>> = HashMap::new();
    for (u, unit) in units.iter().enumerate().filter(|(_, unit)| unit.region == NOTE) {
        for &(k, start, end) in &unit.references {
            paired.entry(id_of(k)).or_default().push(legal_pdf_core::Anchor {
                pair_id: format!("{}-note", node_id(u)),
                label: unit.label.clone().unwrap_or_default(),
                start: start as usize,
                end: end as usize,
            });
        }
    }
    let marks = settled.as_ref().map_or(&paired, |read| read.anchors);
    let mut paragraphs = Vec::new();
    let mut footnotes = Vec::new();
    for (u, unit) in units.iter().enumerate() {
        let region = REGIONS[usize::from(unit.region)];
        if matches!(region, "furniture" | "page_label" | "line_label" | "none") || region == "marginal_note" && !written || unit.region == NOTE && written {
            continue;
        }
        if unit.region == NOTE {
            let (text, spans) = joined(unit);
            let pages_of: Vec<u32> = {
                let mut found: Vec<u32> = unit.lines.iter().map(|&k| pages[page_of(k)].number).collect();
                found.dedup();
                found
            };
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
        let Some((text, anchors, spans)) = crate::structure::joining::anchored_text(&lines, &vocabulary, marks) else { continue };
        paragraphs.push(Paragraph {
            source_spans: spans,
            id: node_id(u),
            page_index: pages[page_of(unit.lines[0])].index,
            region_type: if unit.region == HEADING || written && matches!(region, "title" | "marginal_note") { region.to_owned() } else { "body".to_owned() },
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
        settled.as_mut().map_or_else(Vec::new, |read| std::mem::take(&mut read.diagnostics)),
    );
    if !continuations.is_empty() {
        structure.continuations = Some(continuations);
    }
    legal_structure_model::assemble_in_place(&mut structure).map_err(Error::Message)?;
    // A judgment's numbered paragraphs are t0's (`read_judgment_paragraphs`): each of its numbers the unit carrying it,
    // with the extent the reading holds; a number no unit carries a node of its own; a number off its count none. The
    // reading is the fences', all `Decided` reads for it. On a line a model reads as a provision the number is the
    // provision's: a contract's clauses count as paragraphs do, and no paragraph stands over them. Tier 0's rules read
    // the same numbers, and nothing here overrules them.
    structure.source_fragments = index.source_fragments();
    crate::structure::graph::read_judgment_paragraphs(&mut structure, pages, &decided);
    if !written {
        let provisions: HashSet<&str> = units.iter().filter(|unit| REGIONS[usize::from(unit.region)] == "provision").flat_map(|unit| unit.lines.iter().map(|&k| ids[k].as_str())).collect();
        structure.nodes.retain(|node| !node.id.starts_with("judgment-paragraph-") || !node.line_ids.first().is_some_and(|id| provisions.contains(id.as_str())));
    }
    structure.profile = identity.profile;
    structure.refresh_revision().map_err(|e| Error::Message(e.to_string()))?;
    Ok((structure, paragraphs, settled.map_or(footnotes, |read| read.footnotes)))
}
