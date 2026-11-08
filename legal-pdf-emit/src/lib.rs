//! The transition parser's reading as the document consumers read: its structure graph, built from
//! the moves and the state they move, nothing from the stage parser's decisions. Each unit is a node:
//! a numbered paragraph a `Paragraph` (its label and marker), a provision a `Section` (located by its
//! number, its path from the outermost provision its alias), a heading a `Heading`, a list item a
//! `ListItem`, any other body text `Prose` (a quotation also a `Quote` over it, a contents list's lines
//! under the `contents` grammar, a transcript's turn opening on its speaker's label), a note a
//! `Footnote` with its pair (its label, its body, the reference marks it answered), a line set apart a
//! `Prose` of its role (furniture, a page label, a transcript's line number each a line labelled by its
//! text, a title, a marginal note, a caption, a headnote), and the tables the parser read as tables
//! with their rows and cells. Each page carries its printed label as its alias. Parents are the
//! outline's (`assemble_outline`), and a provision's path alias follows them.

use legal_pdf_constituents as constituents;
use legal_pdf_core::model::Page;
use legal_pdf_core::{Error, Result};
use legal_pdf_fences::Fences;
use legal_pdf_machine::lines::LineTable;
use legal_pdf_machine::machine::{stream_of, LineView, Move, Stands, State, Streams, MARGIN, OPENING, RANKED_BODY};
use legal_pdf_text::joining::{seam, split_words, Join};
use legal_pdf_structure_model::regions::{HEADING, LINE_LABEL, NOTE, NOTE_QUOTE, PAGE_LABEL, REGIONS, TABLE};
use legal_pdf_structure_model::{Decided, Evidence, Place, Runs, Slots, StructureIdentity, TRANSITION_ORIGIN as ORIGIN};
use legal_pdf_text::text_index::PdfTextIndex;
use legal_structure_model::{ContinuationKind, NodeKind, ScalarRange, SourceContinuation, SourceJoin};
use std::collections::{HashMap, HashSet};

mod rendered;

use legal_structure_model::{Derivation, DocumentStructure, Origin, PrimaryRole, StructureNode};

/// What tier 0 settled before its moves (`tier0`): its lines are written and their regions built, the text indexed, and
/// its notes the pairer's: their nodes, the projection's notes, the claims left aside; and the runs its moves open on.
pub struct Settled<'a> {
    pub index: &'a PdfTextIndex,
    pub notes: Vec<StructureNode>,
    pub runs: &'a Runs,
}

struct Unit {
    region: u8,
    lines: Vec<usize>,
    marker: Option<(usize, usize)>,
    /// The line `marker` stands on: the unit's first, or the margin's line numbering it.
    marker_line: usize,
    label: Option<String>,
    /// A heading's level (0 for none).
    level: usize,
    references: Vec<(usize, u32, u32)>,
    /// The unit's last line with text, and the page it holds lines on now with where its first there starts.
    last: Option<usize>,
    page_start: Option<(usize, usize)>,
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

/// The figures a line's text opens on: what they read, and their first and past-last character.
pub fn leading_figures(text: &str) -> Option<(u32, (usize, usize))> {
    let chars: Vec<char> = text.chars().collect();
    let start = chars.iter().position(|c| !c.is_whitespace())?;
    let length = chars[start..].iter().take_while(|c| c.is_ascii_digit()).count();
    let figures: String = chars[start..start + length].iter().collect();
    Some((figures.parse().ok()?, (start, start + length)))
}

/// The document's structure from its lines (`table`) and the moves they make, in one walk: each line's move is
/// DOC's reading where it read one (`doc`), but for the notes, which are tier 0's (`tier0`), a body unit tier 0 opens
/// taking its numbering's depth (`State::stack_depth`). As a line joins its unit the walk writes the seam with the unit's
/// line before it (`joining::seam`): a dropped hyphen a manuscript edit, a join with no space a join, a page crossed a
/// continuation; and each hyphen inside a line that splits a word (`joining::split_words`) an edit. What tier 0 wrote and
/// paired before its moves (`settled`) stands. With the authority units read off the rendered text, and the notes.
pub fn structure_graph(pages: &[Page],
    slots: &Slots,
    evidence: &Evidence,
    fences: &Fences,
    decided: &Decided,
    table: &LineTable,
    tier0: &[Move],
    doc: Option<Vec<Move>>,
    identity: &StructureIdentity,
    mut settled: Settled,
) -> Result<(DocumentStructure, Vec<legal_pdf_core::AuthorityTextUnit>)> {
    let index = settled.index;
    let views = LineView::all(pages, slots, evidence, fences, table, None);
    let texts: Vec<String> = fences.order.iter().map(|&slot| slots.line(pages, slot).text.clone()).collect();
    let ids: Vec<String> = fences.order.iter().map(|&slot| slots.line(pages, slot).id.clone()).collect();
    let text_of = |k: usize| texts[k].clone();
    let id_of = |k: usize| ids[k].clone();
    // Each line's source fragment (UTF-16) and page, what a seam is written in.
    let source_fragments = index.source_fragments();
    let fragment: HashMap<&str, (ScalarRange, usize)> = source_fragments.iter().map(|f| (f.id.as_str(), (f.range, f.page_index))).collect();
    let vocabulary = legal_pdf_text::joining::vocabulary(pages);
    let mut edits: HashMap<String, Vec<(usize, usize, String)>> = HashMap::new();
    let mut joins: Vec<SourceJoin> = Vec::new();
    let mut continuations: Vec<SourceContinuation> = Vec::new();
    let place_of: HashMap<&str, usize> = ids.iter().enumerate().map(|(k, id)| (id.as_str(), k)).collect();
    let note_move = |m: &Move| matches!(m.region, NOTE | NOTE_QUOTE);
    let doc = doc.filter(|read| read.len() == tier0.len());
    let mut moves: Vec<Move> = Vec::with_capacity(tier0.len());
    // Units, each line joining the unit open in its stream, resuming the one it held before, or opening one.
    let mut state = State::default();
    let mut units: Vec<Unit> = Vec::new();
    let mut streams = Streams::default();
    // Each unit by its first line.
    let mut unit_at: HashMap<usize, usize> = HashMap::new();
    // The lines the parser opened a constituent document on; a start on a line of no region passes to the next unit.
    let mut started: Vec<usize> = Vec::new();
    for (k, line) in views.iter().enumerate() {
        let read = doc.as_ref().and_then(|read| read.get(k)).filter(|read| !note_move(read) && !note_move(&tier0[k])).cloned();
        let walked = read.is_none();
        let mut chosen = read.unwrap_or_else(|| tier0[k].clone());
        if walked && chosen.opens && RANKED_BODY.contains(&chosen.region) {
            chosen.depth = state.stack_depth(line, &chosen).min(9) as u8;
        }
        moves.push(chosen);
        let chosen = &moves[k];
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
            let marker_line = match (chosen.reading, line.margin()) {
                (Some(MARGIN), Some((margin, _))) => margin as usize,
                _ => k,
            };
            let text = text_of(marker_line);
            let reading = line.reading(chosen.reading);
            // A note's label is the number its move counts on: where the label reader found it, else the
            // figures its opening token starts on ("27" of "27Ibid"); a note counting on none has none.
            let marker = if chosen.region == NOTE {
                chosen.note.and_then(|value| {
                    let found = evidence.label[fences.order[k].index()].filter(|label| label.value == value);
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
            units.push(Unit { region: chosen.region, lines: Vec::new(), marker, marker_line, label, level, references: Vec::new(), last: None, page_start: None });
            unit_at.insert(k, units.len() - 1);
            units.len() - 1
        };
        units[u].lines.push(k);
        let written = texts[k].trim();
        if chosen.region != NOTE && !matches!(REGIONS[usize::from(chosen.region)], "furniture" | "page_label" | "line_label") && !written.is_empty() {
            if let Some(&(range, page)) = fragment.get(ids[k].as_str()) {
                let unit = &mut units[u];
                if let Some(j) = unit.last {
                    let before = texts[j].trim();
                    let separator = match seam(before, written, &vocabulary) {
                        Join::Plain => {
                            let at = texts[j].chars().take_while(|c| c.is_whitespace()).count() + before.chars().count() - 1;
                            edits.entry(ids[j].clone()).or_default().push((at, at + 1, String::new()));
                            ""
                        }
                        Join::Hyphen => "",
                        Join::Keep => " ",
                    };
                    if separator.is_empty() {
                        joins.push(SourceJoin { from: fragment[ids[j].as_str()].0.start, to: range.start, separator: String::new() });
                    }
                    if let Some((_, from)) = unit.page_start.filter(|&(held, _)| held != page) {
                        let kind = if unit.region == HEADING { ContinuationKind::Heading } else { ContinuationKind::Paragraph };
                        continuations.push(SourceContinuation { kind, from, to: range.start, separator: Some(separator.to_owned()) });
                    }
                }
                if unit.page_start.is_none_or(|(held, _)| held != page) {
                    unit.page_start = Some((page, range.start));
                }
                unit.last = Some(k);
            }
            for at in split_words(&texts[k], &vocabulary) {
                edits.entry(ids[k].clone()).or_default().push((at, at + 1, String::new()));
            }
        }
        streams.step(key, chosen.region, if held.is_some() { stands } else { Stands::Opens }, k);
        // The reference mark the note this line opens answered.
        for (place, start, end) in made.paired.into_iter().chain(made.also.into_iter().flatten()) {
            units[u].references.push((place as usize, start, end));
        }
    }
    // Nodes, their ranges in scalar characters of the text; their parents are the outline's (`assemble_outline`).
    let line_range = |k: usize| index.line(&ids[k]).map(|l| l.range);
    let mut nodes: Vec<StructureNode> = Vec::new();
    // A written page's line set apart keeps the id its line's kind gives it, as the projection reads such lines.
    let folios: HashSet<&str> = pages.iter().filter_map(|page| page.printed_label_line_id.as_deref()).collect();
    let unit_ids: Vec<String> = units.iter().enumerate().map(|(u, unit)| {
        let id = ids[unit.lines[0]].as_str();
        match REGIONS[usize::from(unit.region)] {
            "line_label" => format!("{id}:line"),
            "furniture" | "page_label" => format!("{id}:{}", if folios.contains(id) { "page" } else { "furniture" }),
            _ => format!("transition-{:06}", u + 1),
        }
    }).collect();
    let node_id = |u: usize| unit_ids[u].clone();
    // Each page a node: its text's range (an empty one where its text would open, where it prints none), its index, and
    // where its text was read from.
    let mut page_end = 0;
    for page in pages.iter() {
        let range = index.page_range(page.index).unwrap_or(ScalarRange { start: page_end, end: page_end });
        page_end = range.end;
        let mut node = StructureNode::new(page.id.clone(), NodeKind::Page, range, ORIGIN, Derivation::Native, None);
        node.label = Some(format!("page{}", page.number));
        node.aliases = page.printed_label.clone().map(|label| vec![label]);
        node.anchor.clone_from(&page.printed_label_line_id);
        node.line_ids = page.lines.iter().map(|l| l.id.clone()).collect();
        node.page_indexes = vec![page.index];
        node.properties.insert("source".to_owned(), page.source.clone());
        nodes.push(node);
    }
    for (u, unit) in units.iter().enumerate() {
        // The pairer's notes are written from its claims below.
        if unit.region == NOTE {
            continue;
        }
        let ranges: Vec<ScalarRange> = unit.lines.iter().filter_map(|&k| line_range(k)).collect();
        let Some(envelope) = ranges.iter().map(|r| r.start).min().zip(ranges.iter().map(|r| r.end).max()) else { continue };
        let range = ScalarRange { start: envelope.0, end: envelope.1 };
        let region = if unit.region == TABLE { "text" } else { REGIONS[usize::from(unit.region)] };
        // The member a settled unit opens on (tier 0's move at its first line); a heading numbering a provision is that
        // provision, a heading still.
        let opening = &tier0[unit.lines[0]];
        let member = opening.member.map(|(run, at)| &settled.runs.runs[run as usize].members[at as usize]);
        let (kind, role) = match region {
            "heading" if member.is_some_and(|member| member.kind == Some(legal_structure_model::UnitRole::Section)) => (NodeKind::Section, Some(PrimaryRole::Heading)),
            "heading" => (NodeKind::Heading, None),
            "text" if unit.marker.is_some() => (NodeKind::Paragraph, None),
            "provision" => (NodeKind::Section, Some(PrimaryRole::Text)),
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
        // A unit is the document's own, as tier 0's are; a heading is read from its look.
        let derivation = if kind != NodeKind::Heading { Derivation::Native } else { Derivation::Heuristic };
        let mut node = StructureNode::new(node_id(u), kind, range, ORIGIN, derivation, None);
        node.role = role;
        node.ranges = ranges.clone();
        node.line_ids = unit.lines.iter().map(|&k| id_of(k)).collect();
        let marker_range = unit.marker.and_then(|(start, end)| index.global_range(&id_of(unit.marker_line), start, end));
        let scalar = |place: Place| index.line(&ids[fences.place[place.slot.index()] as usize]).map(|line| line.range.start + place.offset as usize);
        let lines: Vec<&str> = unit.lines.iter().map(|&k| texts[k].as_str()).collect();
        // A settled unit is addressed as the run it opens on reads it: a paragraph by its count, a provision by its
        // grammar's value (a clause, a part, an article), its printing an alias; its marker and text where they stand.
        // Its marker is the run member's where the unit opens on one, else a heading's number as its text reads it, else
        // the unit's own (a note's none); its label a heading's text, else the member's, else the unit's own, a numbered
        // paragraph's its number ("par12" for "[12]" or "12."), as every reader labels it.
        node.marker_range = member.and_then(|member| scalar(member.marker[0]).zip(scalar(member.marker[1])))
            .map(|(start, end)| ScalarRange { start, end: end.max(start) })
            .or_else(|| match kind {
                NodeKind::Footnote => None,
                NodeKind::Heading => ranges.first().and_then(|first| legal_pdf_graph::graph::heading_marker(first.start, lines[0], lines.len())),
                _ => marker_range,
            });
        node.label = match (kind, member) {
            (NodeKind::Heading, _) => Some(lines.join(" ")),
            (NodeKind::Footnote, _) => None,
            (NodeKind::Paragraph, Some(member)) => Some(format!("par{}", member.grammar_value)),
            (NodeKind::ListItem, Some(member)) => Some(member.label.clone()),
            (NodeKind::Section, Some(member)) => Some(member.grammar_value.clone()),
            (NodeKind::Paragraph, None) => unit.label.as_deref().map(|label| label.trim_matches(|c: char| !c.is_ascii_digit()))
                .filter(|digits| !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())).map(|digits| format!("par{digits}")),
            _ => unit.label.clone(),
        };
        // (A heading's heading range is its reading context's, `outline_contexts`.)
        if kind == NodeKind::Section {
            node.locator_kind = Some("section".to_owned());
        }
        if let Some(member) = member {
            node.content_start = scalar(member.content);
            match kind {
                NodeKind::Paragraph => node.aliases = Some(vec![member.label.clone()]),
                NodeKind::Section => {
                    node.locator_kind = Some(legal_structure_model::provision_locator(&member.grammar_value, usize::from(member.level)).to_owned());
                    let printed = (member.label != member.grammar_value).then(|| member.label.clone());
                    node.aliases = printed.map(|printed| vec![printed]);
                    if role == Some(PrimaryRole::Heading) {
                        node.heading_range = Some(range);
                    }
                }
                _ => {}
            }
        }
        // A contents list's lines are under the `contents` grammar. A numbered paragraph wholly in the translation prints
        // that paragraph in the other language (`paragraph_printings`), as the stage's nodes say.
        node.grammar = if region == "contents" {
            Some("contents".to_owned())
        } else if matches!(kind, NodeKind::Paragraph | NodeKind::Prose | NodeKind::Section) && unit.lines.iter().all(|&k| views[k].translation()) {
            Some("translation".to_owned())
        } else {
            None
        };
        nodes.push(node);
        // A further marker on the unit's first line opens a unit under it, to the unit's end, the unit before it left with what
        // stands before that marker: "2(1) The Court has ...", the section "2" and its subsection.
        let mut holder = nodes.len() - 1;
        for &(run, at) in &opening.within {
            let member = &settled.runs.runs[run as usize].members[at as usize];
            let scalar = |place: Place| index.line(&ids[fences.place[place.slot.index()] as usize]).map(|line| line.range.start + place.offset as usize);
            let (Some((start, end)), Some(role)) = (scalar(member.marker[0]).zip(scalar(member.marker[1])), member.kind) else { continue };
            let kind = match role {
                legal_structure_model::UnitRole::Section => NodeKind::Section,
                legal_structure_model::UnitRole::Item => NodeKind::ListItem,
                legal_structure_model::UnitRole::Paragraph => NodeKind::Paragraph,
            };
            let above = &mut nodes[holder];
            above.range.end = start;
            above.ranges = above.ranges.iter().filter(|part| part.start < start).map(|part| ScalarRange { start: part.start, end: part.end.min(start) }).collect();
            above.line_ids.truncate(1);
            let mut own = StructureNode::new(format!("{}-{start}", node_id(u)), kind, ScalarRange { start, end: range.end }, ORIGIN, Derivation::Native, Some(above.id.clone()));
            own.role = Some(PrimaryRole::Text);
            own.ranges = ranges.iter().filter(|part| part.end > start).map(|part| ScalarRange { start: part.start.max(start), end: part.end }).collect();
            own.line_ids = unit.lines.iter().map(|&k| id_of(k)).collect();
            own.marker_range = Some(ScalarRange { start, end: end.max(start) });
            own.grammar.clone_from(&nodes[holder].grammar);
            own.content_start = scalar(member.content);
            own.label = Some(if kind == NodeKind::Section { member.grammar_value.clone() } else { member.label.clone() });
            if kind == NodeKind::Section {
                own.locator_kind = Some(legal_structure_model::provision_locator(&member.grammar_value, usize::from(member.level)).to_owned());
            }
            nodes.push(own);
            holder = nodes.len() - 1;
        }
        if kind == NodeKind::Footnote {
            let calls = unit.references.iter().filter_map(|&(k, start, end)| index.global_range(&id_of(k), start as usize, end as usize)).collect();
            if let Some(note) = nodes.iter_mut().rev().find(|node| node.id == node_id(u)) {
                note.marker_range.get_or_insert(ScalarRange { start: range.start, end: range.start });
                note.calls = calls;
            }
        }
    }
    // The pairer's notes, over the lines they claim, whatever the moves made of them: a note over a page goes on with a
    // space, and a hyphen inside its line that splits a word is an edit.
    for note in &settled.notes {
        let mut held: Option<(usize, usize)> = None;
        for id in &note.line_ids {
            let Some(&(range, page)) = fragment.get(id.as_str()) else { continue };
            if let Some((_, from)) = held.filter(|&(on, _)| on != page) {
                continuations.push(SourceContinuation { kind: ContinuationKind::Note, from, to: range.start, separator: Some(" ".to_owned()) });
            }
            if held.is_none_or(|(on, _)| on != page) {
                held = Some((page, range.start));
            }
            let Some(&k) = place_of.get(id.as_str()) else { continue };
            for at in split_words(&texts[k], &vocabulary) {
                edits.entry(id.clone()).or_default().push((at, at + 1, String::new()));
            }
        }
    }
    nodes.append(&mut settled.notes);
    nodes.extend(legal_pdf_graph::graph::table_nodes(index, fences.tables.iter().enumerate(), ORIGIN)?);
    nodes.extend(legal_pdf_graph::graph::quotation_nodes(index, decided, ORIGIN)?);
    // Text the parser read as no region's lies outside the body, as the gold leaves it: furniture.
    for (k, chosen) in moves.iter().enumerate() {
        if chosen.region != 0 || texts[k].trim().is_empty() {
            continue;
        }
        let Some(range) = line_range(k) else { continue };
        let mut node = StructureNode::new(format!("transition-none-{:06}", k + 1), NodeKind::Prose, range, ORIGIN, Derivation::Heuristic, None);
        node.role = Some(PrimaryRole::Furniture);
        node.ranges = vec![range];
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
    let mut structure = DocumentStructure::new(identity.document_id.clone(), "local-pdf", index.text().to_owned(), ORIGIN);
    structure.origins.push(Origin { id: legal_structure_model::NOTE_PAIRING_ORIGIN.to_owned() });
    structure.source_sha256 = Some(identity.source_sha256.clone());
    DocumentStructure::project_scalar_parts(&legal_structure_model::ScalarText::new(&structure.text), &mut nodes);
    structure.nodes = nodes;
    // The outline is tier 0's: each constituent document's and language's own stacks, a quotation within what quotes it.
    let levels: HashMap<&str, usize> = units.iter().filter(|unit| unit.level > 0).flat_map(|unit| unit.lines.iter().map(|&k| (ids[k].as_str(), unit.level))).collect();
    let contexts = legal_pdf_graph::graph::outline_contexts(&mut structure.nodes, &identity.document_id, index, decided);
    legal_pdf_graph::graph::assemble_outline(&mut structure.nodes, &contexts, index.text(), &|id| levels.get(id).copied());
    // A provision's path from the outermost provision it stands in ("12(1)(a)"), by the parents the outline kept, after
    // its printing's alias where it has one: each unit's printed label along the chain.
    let label_of: HashMap<&str, &str> = units.iter().enumerate().filter_map(|(u, unit)| Some((unit_ids[u].as_str(), unit.label.as_deref()?))).collect();
    let provisions: HashSet<&str> = units.iter().enumerate().filter(|(_, unit)| REGIONS[usize::from(unit.region)] == "provision").map(|(u, _)| unit_ids[u].as_str()).collect();
    let parent_of: HashMap<String, String> = structure.nodes.iter().filter_map(|node| Some((node.id.clone(), node.parent_id.clone()?))).collect();
    for node in structure.nodes.iter_mut().filter(|node| node.kind == NodeKind::Section && label_of.contains_key(node.id.as_str())) {
        let mut chain = vec![node.id.as_str()];
        while let Some(parent) = parent_of.get(*chain.last().unwrap()).map(String::as_str).filter(|parent| provisions.contains(parent) && !chain.contains(parent)) {
            chain.push(parent);
        }
        let labels: Vec<&str> = chain.iter().rev().filter_map(|id| label_of.get(id).copied()).collect();
        if labels.len() > 1 {
            let path = labels.iter().enumerate().map(|(at, label)| if at == 0 || label.starts_with('(') { (*label).to_owned() } else { format!("({label})") }).collect::<String>();
            node.aliases.get_or_insert_with(Vec::new).push(path);
        }
    }
    legal_structure_model::assemble_in_place(&mut structure).map_err(Error::Message)?;
    // A judgment's numbered paragraphs are t0's (`read_judgment_paragraphs`): each of its numbers the unit carrying it,
    // with the extent the reading holds; a number no unit carries a node of its own; a number off its count none. The
    // reading is the fences', all `Decided` reads for it. On a line a model reads as a provision the number is the
    // provision's: a contract's clauses count as paragraphs do, and no paragraph stands over them. Tier 0's rules read
    // the same numbers, and nothing here overrules them.
    structure.source_fragments = source_fragments;
    for (line, written) in edits {
        let held = structure.manuscript_edits.entry(line).or_default();
        for edit in written {
            if !held.iter().any(|other| other.0 < edit.1 && edit.0 < other.1 || *other == edit) {
                held.push(edit);
            }
        }
        held.sort_by_key(|edit| (edit.0, edit.1));
    }
    structure.joins.extend(joins);
    structure.continuations = Some(continuations);
    legal_pdf_graph::graph::read_judgment_paragraphs(&mut structure, pages, decided, &HashSet::new());
    structure.profile = identity.profile;
    let runs = rendered::render(pages, &mut structure)?;
    let units = rendered::authority_units(&structure, &runs);
    legal_pdf_notes::note_context::attach_propositions(&mut structure, &units, decided);
    structure.refresh_revision().map_err(|e| Error::Message(e.to_string()))?;
    Ok((structure, units))
}
