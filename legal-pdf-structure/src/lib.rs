//! The structure stage: a PDF's extracted pages and lines in, its paragraphs, notes, headings and structure graph out.
//! Each stage is a module over the contract (`model`); `derive` calls them in order:
//!
//! 0. The pages normalized (`evidence::normalize`): a transcript's line numbers parted from the lines
//!    they were read into, a table's rulings cleared from under its words, each line given its slot.
//! 1. Measured (`Evidence`): each page's words, rows and note separator; each line's type, its source's tag, its marker,
//!    note label, reference tokens, words and citations; the document's bookmarks and whether it is an instrument.
//! 2. Fenced and ordered (`Fences`): a transcript's line numbers, the pages' numbers and the lines printing them (the
//!    page map's, `legal_pdf_pages::page_map`), furniture (`furniture`), each
//!    page's columns, contents lists (`headings`), tables, the note band under a separator, the body's type,
//!    the constituent documents (`constituents`), the reading order, quotations and reading contexts, and each
//!    line's continuation of the one before.
//! 3. Numbered (`Runs`): each reading context's numbering runs, read and resolved by the engine.
//! 4. Notes (`Notes`): each document's note band and the lanes that take more lines for notes (`notes`).
//! 5. Decided (`tier0`): every rule heard on every line (`Decision::hear`), then one walk in reading order deciding each
//!    line, its block and the unit it opens or goes on in (`settle`), then the title and the headings' levels
//!    (`Decision::finish`). Paragraph numbers are fixed input. Tier 1 (DOC, with `encoder`) reads over tier 0's
//!    reading (`transition`), its moves in tier 0's place.
//! 6. Written once (`write_lines`): each line's place in its page's reading order; and read by slot, each line's zone.
//! 7. Derived from the decision: notes paired (the engine's pairer, over `note_lines`), laid over the units the moves
//!    make (a line a note holds is the note's; a note band's line none holds is the body's), propositions and
//!    cross-references (`note_context`), the units emitted as the structure graph.
//!
//! Measure a change with the grader's digest (`structure-gold digest`): every parse hashed, two builds compared.

mod constituents;
mod decide;
mod rendered;
mod evidence;
mod fences;
mod furniture;
mod graph;
mod headings;
mod layout;
mod machine;
mod model;
mod notes;
mod runs;
mod tables;
mod text;
mod tier0;
mod transition;

use evidence::normalize;
use model::Evidence;
use fences::Fences;
use model::{Runs, Slots};
use decide::{note_lines, write_lines, write_trace, zones};
use text::text_index::PdfTextIndex;
use notes::note_context::unmatched_reference_diagnostics;

use machine::lines::LineTable;
use machine::machine::LineView;
use std::path::Path;
pub use transition::Doc;
pub use furniture::page_evidence;

use legal_pdf_core::model::{Diagnostic, LineZone, Page, PdfOutlineEntry};
use legal_structure_model::StructureNode;
use legal_pdf_core::{Error, Result};
use legal_structure_model::DocumentStructure;
use crate::machine::machine::{Move, MARGIN, OPENING};
use crate::text::joining::{seam, split_words};
use pdf_inspector::markdown::Join;
use crate::model::regions::{HEADING, LINE_LABEL, NOTE, NOTE_QUOTE, PAGE_LABEL, REGIONS, TABLE, TEXT};
use crate::model::{Decided, Place, TRANSITION_ORIGIN};
use legal_structure_model::{utf16_len, ContinuationKind, NodeKind, ScalarRange, ScalarText, SourceContinuation, SourceJoin};
use std::collections::{HashMap, HashSet};



use legal_structure_model::{Derivation, Origin, PrimaryRole};

struct Unit {
    region: u8,
    lines: Vec<usize>,
    marker: Option<(usize, usize)>,
    /// The line `marker` stands on: the unit's first, or the margin's line numbering it.
    marker_line: usize,
    label: Option<String>,
    /// A heading's level (0 for none).
    level: usize,
    /// The unit's last line with text, and the page it holds lines on now with where its first there starts.
    last: Option<usize>,
    page_start: Option<(usize, usize)>,
}

/// The opening token in a line's text, in UTF-16 code units: its first and past-last, leading spaces
/// skipped; with `word`, the token after it too ("PART II").
fn opening(text: &str, word: bool) -> Option<(usize, usize)> {
    let past = |from: usize, blank: bool| from + text[from..].find(|c: char| c.is_whitespace() != blank).unwrap_or(text.len() - from);
    let start = past(0, true);
    if start == text.len() {
        return None;
    }
    let mut end = past(start, false);
    if word {
        end = past(past(end, true), false);
    }
    Some((utf16_len(&text[..start]), utf16_len(&text[..end])))
}

/// The figures a line's text opens on: what they read, and their first and past-last UTF-16 code unit.
pub(crate) fn leading_figures(text: &str) -> Option<(u32, (usize, usize))> {
    let start = text.find(|c: char| !c.is_whitespace())?;
    let length = text[start..].bytes().take_while(u8::is_ascii_digit).count();
    let at = utf16_len(&text[..start]);
    Some((text[start..start + length].parse().ok()?, (at, at + length)))
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructureIdentity {
    pub profile: Option<legal_structure_model::DetectionProfile>,
    pub source_sha256: String,
    /// The page labels the PDF declares (`/PageLabels`), by page index; empty where it has none.
    pub page_labels: Vec<Option<String>>,
}

impl StructureIdentity {
    /// The document's id: its source's hash, named.
    pub(crate) fn document_id(&self) -> String {
        format!("doc-{}", &self.source_sha256[..20])
    }
}
pub struct StructureOutput {
    /// The authority units, read off the rendered text (`emit`).
    pub authority_text_units: Vec<legal_pdf_core::AuthorityTextUnit>,
    pub diagnostics: Vec<Diagnostic>,
    pub structure_graph: DocumentStructure,
}

/// A line of the note band that no note holds (the pairer made no note of its label, or bounded the note's body short of
/// it) is the body's, and joins its paragraph rather than standing alone: on the gold, 6,825 such lines were no notes,
/// 2,146 were.
fn give_back(zones: &mut [LineZone], slots: &Slots, notes: &[StructureNode]) {
    let mut held = vec![false; zones.len()];
    for slot in notes.iter().flat_map(|note| &note.line_ids).filter_map(|id| slots.of_id(id)) {
        held[slot.index()] = true;
    }
    for (zone, held) in zones.iter_mut().zip(held) {
        if matches!(zone, LineZone::Note(_)) && !held {
            *zone = LineZone::Body;
        }
    }
}
/// The document's structure, its moves DOC's where `doc` is given (tier 1), else tier 0's. With `trace`, the decision is also
/// written to that folder (`decide::trace`); the parse is the same.
pub fn derive(pages: &mut [Page],
    separators: &[Option<f64>],
    outline: &[PdfOutlineEntry],
    identity: StructureIdentity,
    trace: Option<&std::path::Path>,
    doc: Option<&Doc>,
) -> Result<StructureOutput> {
    use legal_pdf_core::profile::measure;
    if separators.len() != pages.len() {
        return Err(Error::Message(
            "common input must contain one separator value per page".to_owned(),
        ));
    }
    // The document's type, where known (the caller's profile): an instrument numbers its provisions.
    let instrument = identity.profile == Some(legal_structure_model::DetectionProfile::Instrument);
    let (mut slots, normalized) = measure("derive.normalize", || normalize(pages));
    // Each page's measures and columns, read once for every stage that reads the pages before they are written.
    let geometry = layout::Geometry::of(pages);
    let mut evidence = measure("derive.measure", || {
        crate::evidence::measure(pages, &slots, normalized, separators, outline, instrument)
    });
    let mut fences = measure("derive.fences", || Fences::read(pages, &slots, &evidence, outline, &identity.page_labels));
    let runs = measure("derive.runs", || crate::runs::read(pages, &slots, &mut evidence, &mut fences, &geometry))?;
    measure("derive.machine", || output(pages, &mut slots, &evidence, fences, &runs, &geometry, outline, &identity, trace, doc))
}

/// The parse: tier 0's decision (`tier0::decide`, its one walk), DOC's moves over it where DOC is given
/// (`transition::over_tier0`), the lines written, the notes paired, and the units the moves make, the pairer's notes laid
/// over them, written as the structure graph the consumers read.
#[allow(clippy::too_many_arguments)]
fn output(pages: &mut [Page],
    slots: &mut Slots,
    evidence: &Evidence,
    mut fences: Fences,
    runs: &Runs,
    geometry: &layout::Geometry,
    outline: &[PdfOutlineEntry],
    identity: &StructureIdentity,
    trace: Option<&Path>,
    doc: Option<&Doc>,
) -> Result<StructureOutput> {
    let mut diagnostics = std::mem::take(&mut fences.diagnostics);
    // One line table for both tiers: tier 0 reads it, and the walk measures it for DOC.
    let table = LineTable::plain(pages, slots, evidence, &fences, &runs.before, transition::confusions());
    let mut notes = decide::Notes::read(pages, slots, evidence, &fences, runs);
    // The fences' roles are the decision's to fill from here.
    let fenced = (std::mem::take(&mut fences.role), std::mem::take(&mut fences.rule));
    let views = LineView::all(evidence, &fences, &table);
    let (mut decision, blocks) = tier0::decide(pages, slots, evidence, &fences, runs, geometry, outline, &notes, &views, fenced, trace.is_some());
    // DOC's reading where it is given, read off the pages as they came, before this reading writes and orders them.
    let doc = transition::over_tier0(pages, slots, evidence, &fences, geometry, &identity.source_sha256, &notes, &decision, &table, &views, doc);
    diagnostics.append(&mut notes.diagnostics);
    if let Some(folder) = trace {
        write_trace(folder, &identity.source_sha256, pages, slots, evidence, &fences, runs, &notes, &decision, &table.readings, &table.references);
    }
    write_lines(pages, slots, &fences, &decision);
    let mut zones = zones(slots, evidence, &fences, &notes, &decision);
    // The lines are read in reading order from here on, the pairer's and the output's: each page's put in it once.
    slots.read_in_order(pages, &fences.order);
    let slots: &Slots = slots;
    let decided = Decided::of(pages, slots, &fences);
    let pages: &[Page] = pages;
    // A line joins the one before it where evidence reads that line's marker standing beside it (`Evidence::beside`).
    let index = PdfTextIndex::from_pages(pages, |before, line| slots.of_id(&before.id).and_then(|at| evidence.beside[at.index()]).is_some_and(|beside| slots.of_id(&line.id) == Some(beside)));
    let citations = graph::citation_spans(pages, slots, evidence);
    let lines = note_lines(pages, slots, &fences, &notes, &decision, &zones, &blocks, &citations);
    let pairing = legal_structure::note_pairing::pair_footnotes(&lines, false, &|at, start, end| index.global_range(lines[at].id, start, end));
    let notes = pairing.notes;
    diagnostics.extend(pairing.diagnostics);
    give_back(&mut zones, slots, &notes);
    diagnostics.extend(unmatched_reference_diagnostics(fences.order.iter().map(|&slot| (slots.line(pages, slot), zones[slot.index()])), &notes, &index, &decided));
    // The units the moves make: each line opening one or in the one its move names; as a line joins its unit the seam
    // with the unit's line before it is written (`joining::seam`): a dropped hyphen a manuscript edit, a join with no
    // space a join, a page crossed a continuation; and each hyphen inside a line that splits a word
    // (`joining::split_words`) an edit. A line's move is DOC's where it read one and neither reads the line as a note's,
    // the run members tier 0 opens on kept: DOC reads the same table, so its reading of a marker names one of this
    // reading's. The pairer's notes lie over the units: a line a note holds is the note's and no unit's, and a line of
    // the note band no note holds is the body's, a paragraph of its own from each note label.
    let (index, fences, decided) = (&index, &fences, &decided);
    let moves = &mut decision.moves;
    let mut notes = notes;
    let note_move = |m: &Move| matches!(m.region, NOTE | NOTE_QUOTE);
    for (record, read) in moves.iter_mut().zip(doc.iter().flatten()) {
        if !note_move(read) && !note_move(record) {
            *record = Move { role: record.role, rule: record.rule, joined: record.joined, unit: record.unit, member: record.member.take(), within: std::mem::take(&mut record.within), ..read.clone() };
        }
    }
    let moves = &*moves;
    let texts: Vec<String> = fences.order.iter().map(|&slot| slots.line(pages, slot).text.clone()).collect();
    let ids: Vec<String> = fences.order.iter().map(|&slot| slots.line(pages, slot).id.clone()).collect();
    // Each line's source fragment (UTF-16) and page, what a seam is written in.
    let fragment = |id: &str| index.line(id).map(|f| (f.range, f.page_index));
    let vocabulary = crate::text::joining::vocabulary(pages);
    let mut edits: HashMap<String, Vec<(usize, usize, String)>> = HashMap::new();
    let mut joins: Vec<SourceJoin> = Vec::new();
    let mut continuations: Vec<SourceContinuation> = Vec::new();
    let place_of: HashMap<&str, usize> = ids.iter().enumerate().map(|(k, id)| (id.as_str(), k)).collect();
    let mut noted = vec![false; ids.len()];
    for k in notes.iter().flat_map(|note| &note.line_ids).filter_map(|id| place_of.get(id.as_str())) {
        noted[*k] = true;
    }
    let mut units: Vec<Unit> = Vec::new();
    // Each unit by its first line.
    let mut unit_at: HashMap<usize, usize> = HashMap::new();
    // The body unit the note band's lines no note holds go on in.
    let mut given: Option<usize> = None;
    // The lines the parser opened a constituent document on; a start on a line of no region passes to the next unit.
    let mut started: Vec<usize> = Vec::new();
    for (k, line) in views.iter().enumerate() {
        let chosen = &moves[k];
        if chosen.starts {
            started.push(k);
        }
        let back = chosen.region == NOTE;
        if chosen.region == 0 || noted[k] {
            given = None;
            continue;
        }
        let region = if back { TEXT } else { chosen.region };
        let single = matches!(region, LINE_LABEL | PAGE_LABEL);
        let held = if back {
            given.filter(|_| evidence.label[fences.order[k].index()].is_none())
        } else {
            chosen.resumes.and_then(|first| unit_at.get(&(first as usize)).copied())
        };
        let u = if let Some(u) = held {
            u
        } else {
            // A unit counting on the margin's number prints its marker on the margin's line.
            let marker_line = match (chosen.reading, line.margin()) {
                (Some(MARGIN), Some((margin, _))) if !back => margin as usize,
                _ => k,
            };
            let text = texts[marker_line].clone();
            let marker = if back {
                None
            } else if chosen.reading == Some(OPENING) {
                opening(&text, false)
            } else {
                line.reading(chosen.reading).and_then(|r| opening(&text, r.family.matches(':').count() > 1))
            };
            // A unit's label is its marker as printed ("12", "(a)", "[4]"), a line number's or page label's its text.
            let printed = |(start, end): (usize, usize)| ScalarText::new(&text).slice_utf16(start..end).unwrap_or_default().trim().trim_end_matches('.').to_owned();
            let label = if single { Some(text.trim().to_owned()) } else { marker.map(printed) };
            let level = if chosen.opens && region == HEADING { chosen.level.max(1) as usize } else { 0 };
            units.push(Unit { region, lines: Vec::new(), marker, marker_line, label, level, last: None, page_start: None });
            unit_at.insert(k, units.len() - 1);
            units.len() - 1
        };
        given = back.then_some(u);
        units[u].lines.push(k);
        let written = texts[k].trim();
        if !matches!(REGIONS[usize::from(region)], "furniture" | "page_label" | "line_label") && !written.is_empty() {
            if let Some((range, page)) = fragment(&ids[k]) {
                let unit = &mut units[u];
                if let Some(j) = unit.last {
                    let before = texts[j].trim();
                    let separator = match seam(before, written, &vocabulary) {
                        Join::Plain => {
                            // The dropped hyphen's UTF-16 offset in its line, `before` being the line trimmed.
                            let at = utf16_len(&texts[j][..texts[j].len() - texts[j].trim_start().len()]) + utf16_len(before) - 1;
                            edits.entry(ids[j].clone()).or_default().push((at, at + 1, String::new()));
                            ""
                        }
                        Join::Hyphen => "",
                        Join::Keep => " ",
                    };
                    if separator.is_empty() {
                        joins.push(SourceJoin { from: fragment(&ids[j]).expect("a unit's last line is indexed").0.start, to: range.start, separator: String::new() });
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
    }
    // Nodes, their ranges in UTF-16 code units of the text, as the index places the lines; their parents are the
    // outline's (`assemble_outline`).
    let line_range = |k: usize| index.line(&ids[k]).map(|l| l.range);
    // Where a run member's place stands in the text.
    let placed = |place: Place| index.line(&ids[fences.place[place.slot.index()] as usize]).map(|line| line.range.start + place.offset as usize);
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
        let mut node = StructureNode::new(page.id.clone(), NodeKind::Page, range, TRANSITION_ORIGIN, Derivation::Native, None);
        node.label = Some(page.number.to_string());
        node.aliases = page.printed_label.clone().map(|label| vec![label]);
        node.anchor.clone_from(&page.printed_label_line_id);
        node.line_ids = page.lines.iter().map(|l| l.id.clone()).collect();
        node.page_indexes = vec![page.index];
        node.properties.insert("source".to_owned(), page.source.clone());
        nodes.push(node);
    }
    for (u, unit) in units.iter().enumerate() {
        let ranges: Vec<ScalarRange> = unit.lines.iter().filter_map(|&k| line_range(k)).collect();
        let Some(envelope) = ranges.iter().map(|r| r.start).min().zip(ranges.iter().map(|r| r.end).max()) else { continue };
        let range = ScalarRange { start: envelope.0, end: envelope.1 };
        let region = if unit.region == TABLE { "text" } else { REGIONS[usize::from(unit.region)] };
        // The member a settled unit opens on (tier 0's move at its first line); a heading numbering a provision is that
        // provision, a heading still.
        let opening = &moves[unit.lines[0]];
        let member = opening.member.map(|(run, at)| &runs.runs[run as usize].members[at as usize]);
        let (kind, role) = match region {
            "heading" if member.is_some_and(|member| member.kind == Some(legal_structure_model::UnitRole::Section)) => (NodeKind::Section, Some(PrimaryRole::Heading)),
            "heading" => (NodeKind::Heading, None),
            "text" if unit.marker.is_some() => (NodeKind::Paragraph, None),
            "provision" => (NodeKind::Section, Some(PrimaryRole::Text)),
            "list_item" => (NodeKind::ListItem, None),
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
        let mut node = StructureNode::new(node_id(u), kind, range, TRANSITION_ORIGIN, derivation, None);
        node.role = role;
        node.ranges = ranges.clone();
        node.line_ids = unit.lines.iter().map(|&k| ids[k].clone()).collect();
        let marker_range = unit.marker.and_then(|(start, end)| index.global_range(&ids[unit.marker_line], start, end));
        let lines: Vec<&str> = unit.lines.iter().map(|&k| texts[k].as_str()).collect();
        // A settled unit is addressed as the run it opens on reads it: a paragraph by its count, a provision by its
        // grammar's value (a clause, a part, an article), its printing an alias; its marker and text where they stand.
        // Its marker is the run member's where the unit opens on one, else a heading's number as its text reads it, else
        // the unit's own (a note's none); its label a heading's text, else the member's, else the unit's own, a numbered
        // paragraph's its number ("12" for "[12]" or "12."), as the document prints it.
        node.marker_range = member.and_then(|member| placed(member.marker[0]).zip(placed(member.marker[1])))
            .map(|(start, end)| ScalarRange { start, end: end.max(start) })
            .or_else(|| match kind {
                NodeKind::Heading => ranges.first().and_then(|first| crate::graph::heading_marker(first.start, lines[0], lines.len())),
                _ => marker_range,
            });
        node.label = match (kind, member) {
            (NodeKind::Heading, _) => Some(lines.join(" ")),
            (NodeKind::Paragraph, Some(member)) => Some(member.grammar_value.clone()),
            (NodeKind::ListItem, Some(member)) => Some(member.label.clone()),
            (NodeKind::Section, Some(member)) => Some(member.grammar_value.clone()),
            (NodeKind::Paragraph, None) => unit.label.as_deref().map(|label| label.trim_matches(|c: char| !c.is_ascii_digit()))
                .filter(|digits| !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())).map(str::to_owned),
            _ => unit.label.clone(),
        };
        // (A heading's heading range is its reading context's, `outline_contexts`.)
        if kind == NodeKind::Section {
            node.locator_kind = Some("section".to_owned());
        }
        if let Some(member) = member {
            node.content_start = placed(member.content);
            match kind {
                NodeKind::Paragraph => node.aliases = Some(vec![member.label.clone()]),
                NodeKind::Section => {
                    node.locator_kind = Some(legal_structure_model::provision_locator(member.locator_kind.as_deref(), usize::from(member.level)).to_owned());
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
            let member = &runs.runs[run as usize].members[at as usize];
            let (Some((start, end)), Some(role)) = (placed(member.marker[0]).zip(placed(member.marker[1])), member.kind) else { continue };
            let kind = match role {
                legal_structure_model::UnitRole::Section => NodeKind::Section,
                legal_structure_model::UnitRole::Item => NodeKind::ListItem,
                legal_structure_model::UnitRole::Paragraph => NodeKind::Paragraph,
            };
            let above = &mut nodes[holder];
            above.range.end = start;
            above.ranges = above.ranges.iter().filter(|part| part.start < start).map(|part| ScalarRange { start: part.start, end: part.end.min(start) }).collect();
            above.line_ids.truncate(1);
            let mut own = StructureNode::new(format!("{}-{start}", node_id(u)), kind, ScalarRange { start, end: range.end }, TRANSITION_ORIGIN, Derivation::Native, Some(above.id.clone()));
            own.role = Some(PrimaryRole::Text);
            own.ranges = ranges.iter().filter(|part| part.end > start).map(|part| ScalarRange { start: part.start.max(start), end: part.end }).collect();
            own.line_ids = unit.lines.iter().map(|&k| ids[k].clone()).collect();
            own.marker_range = Some(ScalarRange { start, end: end.max(start) });
            own.grammar.clone_from(&nodes[holder].grammar);
            own.content_start = placed(member.content);
            own.label = Some(if kind == NodeKind::Section { member.grammar_value.clone() } else { member.label.clone() });
            if kind == NodeKind::Section {
                own.locator_kind = Some(legal_structure_model::provision_locator(member.locator_kind.as_deref(), usize::from(member.level)).to_owned());
            }
            nodes.push(own);
            holder = nodes.len() - 1;
        }
    }
    // The pairer's notes, over the lines they claim, whatever the moves made of them: a note over a page goes on with a
    // space, and a hyphen inside its line that splits a word is an edit.
    for note in &notes {
        let mut held: Option<(usize, usize)> = None;
        for id in &note.line_ids {
            let Some((range, page)) = fragment(id) else { continue };
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
    nodes.append(&mut notes);
    nodes.extend(crate::graph::table_nodes(index, fences.tables.iter().enumerate(), TRANSITION_ORIGIN)?);
    nodes.extend(crate::graph::quotation_nodes(index, decided, TRANSITION_ORIGIN)?);
    // Text the parser read as no region's lies outside the body, as the gold leaves it: furniture.
    for (k, chosen) in moves.iter().enumerate() {
        if chosen.region != 0 || noted[k] || texts[k].trim().is_empty() {
            continue;
        }
        let Some(range) = line_range(k) else { continue };
        let mut node = StructureNode::new(format!("transition-none-{:06}", k + 1), NodeKind::Prose, range, TRANSITION_ORIGIN, Derivation::Heuristic, None);
        node.role = Some(PrimaryRole::Furniture);
        node.ranges = vec![range];
        node.line_ids = vec![ids[k].clone()];
        nodes.push(node);
    }
    // The constituent documents the parser opened, each of the kind the constituents reader names its
    // opening line, else a document.
    let starts: Vec<constituents::Start> = started.iter().map(|&k| {
        let id = ids[k].clone();
        let kind = fences.starts.iter().find(|start| start.line_id == id).map_or("document", |start| start.kind()).to_owned();
        constituents::Start::new(id, &kind)
    }).collect();
    nodes.extend(constituents::constituents(index, pages, &starts));
    let mut structure = DocumentStructure::new(identity.document_id(), "local-pdf", index.text().to_owned(), TRANSITION_ORIGIN);
    structure.origins.push(Origin { id: legal_structure_model::NOTE_PAIRING_ORIGIN.to_owned() });
    structure.source_sha256 = Some(identity.source_sha256.clone());
    structure.nodes = nodes;
    // The outline is tier 0's: each constituent document's and language's own stacks, a quotation within what quotes it.
    let levels: HashMap<&str, usize> = units.iter().filter(|unit| unit.level > 0).flat_map(|unit| unit.lines.iter().map(|&k| (ids[k].as_str(), unit.level))).collect();
    let contexts = crate::graph::outline_contexts(&mut structure.nodes, &structure.document_id.clone(), index, decided);
    crate::graph::assemble_outline(&mut structure.nodes, &contexts, index.text(), &|id| levels.get(id).copied());
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
    structure.source_fragments = index.lines.clone();
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
    crate::graph::read_judgment_paragraphs(&mut structure, pages, decided, &HashSet::new());
    structure.profile = identity.profile;
    let runs = rendered::render(pages, &mut structure)?;
    let units = rendered::authority_units(&structure, &runs);
    crate::notes::note_context::attach_propositions(&mut structure, &units, decided);
    structure.refresh_revision().map_err(|e| Error::Message(e.to_string()))?;
    Ok(StructureOutput { authority_text_units: units, diagnostics, structure_graph: structure })
}

