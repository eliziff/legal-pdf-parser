//! The decision written to the pages, once: each line's region, note mode, whether it is held out of
//! the body (a transcript's line number), whether its number counts a numbered paragraph (no note's
//! label then), whether its opening number is no note's label (a heading's, or one the note band
//! refused), its place in its page's reading order and its block; each page's printed label and the
//! line printing it; and each page's lines put in their reading order. Each heading's trail names
//! the rule that decided it.
//!
//! A block is one printed paragraph or unit: it starts where a unit opens (its marker and the line
//! beside it one start), where a heading, title or marginal note starts (its wrapped lines going on
//! with it), at each note's label, at each contents entry, where the role or the reading context
//! changes, in a document's front (its first two pages, and its lines before its first unit) at
//! each line of its caption standing on its own (a field and the value on its row, a label, a
//! court's name, a jurat) and at the line after such a line or a party's role or a date where that
//! line does not go on from it, and where the page's geometry parts two paragraphs
//! (`paragraphs_of`) and no continuation carries the line on from the one before. A table's cell
//! holds its own blocks: a paragraph never runs from one cell into the next, though a cell may hold
//! several.

use super::*;
use crate::paragraphs::{measure_of, paragraphs_of, Measure};
use legal_structure_model::UnitRole;

/// The kinds of line a block holds only one of.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Body,
    Heading,
    Title,
    MarginalNote,
    Note,
    Contents,
    Table,
    Edge,
}

fn class(role: Role) -> Class {
    match role {
        Role::Heading => Class::Heading,
        Role::Title => Class::Title,
        Role::MarginalNote => Class::MarginalNote,
        Role::Note => Class::Note,
        Role::Contents => Class::Contents,
        Role::Table => Class::Table,
        Role::LineLabel | Role::PageLabel | Role::Furniture => Class::Edge,
        Role::Open | Role::Body | Role::ListItem | Role::Quote => Class::Body,
    }
}

/// The region a line of a role is: a page's head or foot by its edge.
fn region(role: Role, edge: Option<Edge>) -> &'static str {
    match role {
        Role::LineLabel => "line_label",
        Role::PageLabel | Role::Furniture if edge == Some(Edge::Bottom) => "footer",
        Role::PageLabel | Role::Furniture => "header",
        Role::Note => "footnote",
        Role::Heading => "heading",
        Role::Title => "title",
        Role::MarginalNote => "marginal_note",
        Role::Open | Role::Body | Role::ListItem | Role::Quote | Role::Contents | Role::Table => "body",
    }
}

/// Writes the document's decision (`decision`) to its pages, measured (`evidence`), fenced and
/// ordered (`fences`), its notes read (`notes`), and puts each page's lines in reading order.
/// Returns each heading line's trail ("+anchor"), and that of each line a rule took from the
/// headings ("-marginal_note"), by line id.
pub(crate) fn write(
    pages: &mut [Page],
    slots: &Slots,
    evidence: &Evidence,
    fences: &Fences,
    notes: &Notes,
    decision: &Decision,
) -> HashMap<String, String> {
    let count = slots.len();
    // A page's note lines that go on with the last note of the page before: its first, and those
    // after it up to the next label.
    let mut continuation = vec![false; count];
    for segment in notes.segments.iter().filter(|segment| segment.continued) {
        let Some((first, rest)) = segment.lines.split_first() else { continue };
        continuation[first.index()] = true;
        for &slot in rest.iter().take_while(|slot| evidence.label[slot.index()].is_none()) {
            continuation[slot.index()] = true;
        }
    }
    let blocks = blocks(pages, slots, evidence, fences, decision);
    for (page_slot, page) in pages.iter_mut().enumerate() {
        let range = slots.on(page_slot);
        for (place, &slot) in fences.order[range.start as usize..range.end as usize].iter().enumerate() {
            page.lines[(slot.0 - range.start) as usize].reading_order = place + 1;
        }
        for (line, slot) in page.lines.iter_mut().zip(range.clone().map(Slot)) {
            let at = slot.index();
            let role = decision.role[at];
            line.region_type = region(role, fences.edge[at]).to_owned();
            line.note_region_mode = match (role, notes.note[at]) {
                (Role::Note, _) if continuation[at] => "footnote_continuation",
                (Role::Note, Some(NoteKind::Endnote)) => "endnote",
                (Role::Note, _) => "footnote",
                _ => "",
            }
            .to_owned();
            line.exclude_from_body = role == Role::LineLabel;
            line.paragraph_label_candidate = matches!(decision.unit[at], Some((_, UnitRole::Paragraph))) || notes.paragraph_label[at];
            line.suppress_footnote_label = notes.suppressed[at] || role == Role::Heading;
            if let Some(block) = blocks[at] {
                line.block_index = block;
            }
        }
        match &fences.folios[page_slot] {
            Some((slot, label)) => {
                let line = &page.lines[(slot.0 - range.start) as usize];
                page.printed_label = Some(label.clone());
                page.printed_label_source = Some(line.region_type.clone());
                page.printed_label_line_id = Some(line.id.clone());
            }
            None => {
                page.printed_label = None;
                page.printed_label_source = None;
                page.printed_label_line_id = None;
            }
        }
    }
    // A heading's trail names the rule that made it one; a title's, a marginal note's and a line
    // its source tagged a heading name the rule that took it from the headings.
    let trail = (0..count)
        .filter_map(|at| {
            let (role, rule) = (decision.role[at], decision.rule[at]);
            let tagged = matches!(evidence.tag[at], SourceTag::Heading(_) | SourceTag::Title | SourceTag::ParagraphTitle);
            let mark = match role {
                Role::Heading => '+',
                Role::Title | Role::MarginalNote => '-',
                _ if tagged => '-',
                _ => return None,
            };
            Some((slots.line(pages, Slot(at as u32)).id.clone(), format!("{mark}{}", rule.name())))
        })
        .collect();
    for (page_slot, page) in pages.iter_mut().enumerate() {
        let range = slots.on(page_slot);
        let mut lines: Vec<Option<Line>> = std::mem::take(&mut page.lines).into_iter().map(Some).collect();
        page.lines = fences.order[range.start as usize..range.end as usize]
            .iter()
            .filter_map(|slot| lines[(slot.0 - range.start) as usize].take())
            .collect();
    }
    trail
}

/// Whether two lines of a page stand on one row.
fn same_row(evidence: &Evidence, slots: &Slots, a: Slot, b: Slot) -> bool {
    let row = evidence.row(slots, a);
    row != u32::MAX && slots.page(a) == slots.page(b) && evidence.row(slots, b) == row
}

/// Each line's block, by slot (none for a line label, folio or furniture, which keeps its own),
/// numbered on each page after the page's own blocks.
fn blocks(pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, decision: &Decision) -> Vec<Option<usize>> {
    let count = slots.len();
    let role = &decision.role;
    let counted = |slot: Slot| matches!(class(role[slot.index()]), Class::Body | Class::Contents | Class::Table);
    let group = |slot: Slot| (fences.document[slot.index()], fences.translation[slot.index()]);
    // Each document and language's measure (`measure_of`), on its counted lines page by page.
    let mut lines_of: Vec<((u16, bool), Vec<Vec<&Line>>)> = Vec::new();
    for page_slot in 0..pages.len() {
        let range = slots.on(page_slot);
        for &slot in &fences.order[range.start as usize..range.end as usize] {
            if !counted(slot) {
                continue;
            }
            let key = group(slot);
            let at = lines_of.iter().position(|(seen, _)| *seen == key).unwrap_or_else(|| {
                lines_of.push((key, Vec::new()));
                lines_of.len() - 1
            });
            let pages_of = &mut lines_of[at].1;
            pages_of.resize_with(page_slot + 1, Vec::new);
            pages_of[page_slot].push(slots.line(pages, slot));
        }
    }
    let measures: Vec<((u16, bool), Measure)> = lines_of.iter().map(|(key, lines)| (*key, measure_of(lines))).collect();
    // Each document's front: its first page, and the place of its first unit in the reading order.
    let documents = usize::from(fences.document.iter().copied().max().unwrap_or(0)) + 1;
    let (mut opening, mut first_unit) = (vec![usize::MAX; documents], vec![u32::MAX; documents]);
    for (place, &slot) in fences.order.iter().enumerate() {
        let document = usize::from(fences.document[slot.index()]);
        opening[document] = opening[document].min(slots.page(slot));
        if decision.unit[slot.index()].is_some() {
            first_unit[document] = first_unit[document].min(place as u32);
        }
    }
    // A caption's line of front matter (`Rule::FrontMatter`) in its document's front: a field, a
    // label, a court's name or a jurat, which stands on its own (`opens`), or one of those, a
    // party's role or a date, after which the next line stands on its own (`closes`). A signature,
    // an author's line or a lead-in's colon is no heading, and still the paragraph's it stands in.
    let caption = |slot: Slot, bits: u32| {
        let at = slot.index();
        let document = usize::from(fences.document[at]);
        decision.rule[at] == Rule::FrontMatter
            && (slots.page(slot) < opening[document].saturating_add(2) || fences.place[at] < first_unit[document])
            && (evidence.words[at] & bits != 0 || bits & WORD_FIELD != 0 && crate::paragraphs::field_label(slots.line(pages, slot)))
    };
    let opens = |slot: Slot| {
        caption(slot, WORD_FIELD | WORD_LABEL | WORD_COURT | WORD_JURAT)
            && evidence.words[slot.index()] & WORD_PARTY == 0
            && fences.continues[slot.index()].is_none()
    };
    let closes = |slot: Slot| caption(slot, WORD_FIELD | WORD_LABEL | WORD_COURT | WORD_JURAT | WORD_PARTY | WORD_DATE);
    let mut blocks = vec![None; count];
    let mut paragraph = vec![None; count];
    let mut entry = vec![None; count];
    for (page_slot, page) in pages.iter().enumerate() {
        let range = slots.on(page_slot);
        let order = &fences.order[range.start as usize..range.end as usize];
        // The printed paragraphs of each document and language on the page.
        for &(key, measure) in &measures {
            let lines: Vec<Slot> = order.iter().copied().filter(|&slot| group(slot) == key).collect();
            if !lines.iter().any(|&slot| counted(slot)) {
                continue;
            }
            let read: Vec<&Line> = lines.iter().map(|&slot| slots.line(pages, slot)).collect();
            let flags: Vec<bool> = lines.iter().map(|&slot| counted(slot)).collect();
            for (slot, number) in lines.into_iter().zip(paragraphs_of(&read, &flags, measure)) {
                paragraph[slot.index()] = number;
            }
        }
        // Its contents list's entries.
        if order.iter().any(|&slot| role[slot.index()] == Role::Contents) {
            let lines: Vec<Slot> = order.iter().copied().filter(|&slot| role[slot.index()] != Role::LineLabel).collect();
            let read: Vec<&Line> = lines.iter().map(|&slot| slots.line(pages, slot)).collect();
            let entries = headings::contents::entry_rows(page.width, &read, |at| role[lines[at].index()] == Role::Contents);
            for (slot, number) in lines.into_iter().zip(entries) {
                entry[slot.index()] = number;
            }
        }
        let mut next = page.lines.iter().map(|line| line.block_index).max().unwrap_or(0);
        let mut previous: Option<Slot> = None;
        for &slot in order {
            let at = slot.index();
            let kind = class(role[at]);
            if kind == Class::Edge {
                previous = Some(slot);
                continue;
            }
            let start = previous.is_none_or(|prior| {
                let before = prior.index();
                let free = !decision.joined[at] && fences.continues[at].is_none();
                class(role[before]) != kind
                    || fences.ctx[before] != fences.ctx[at]
                    || match kind {
                        Class::Body => {
                            decision.unit[at].is_some()
                                || (opens(slot) || closes(prior) && free) && !decision.joined[at] && !same_row(evidence, slots, prior, slot)
                                || free && paragraph[at] != paragraph[before]
                        }
                        Class::Heading | Class::Title | Class::MarginalNote => !decision.joined[at],
                        Class::Note => evidence.label[at].is_some(),
                        Class::Contents => entry[at] != entry[before],
                        Class::Table => paragraph[at] != paragraph[before] || fences.cell[at] != fences.cell[before],
                        Class::Edge => true,
                    }
            });
            if start {
                next += 1;
            }
            blocks[at] = Some(next);
            previous = Some(slot);
        }
    }
    blocks
}
