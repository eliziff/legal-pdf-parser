//! The decision as the readers after it take it: written to the pages once (each line's place in
//! its page's reading order, whether it is held out of the body as a transcript's line number, each
//! page's printed label and the line printing it), and read by slot (each line's zone and block, and
//! the lines as the note pairer reads them).
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
use legal_pdf_text::paragraphs::{measure_of, paragraphs_of, Measure};
use legal_structure_model::UnitRole;
use legal_pdf_core::model::{LineZone, NoteLine, NoteMode, NotePage};

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

/// `write`, each page's lines left in their places.
pub fn write_lines(pages: &mut [Page], slots: &Slots, fences: &Fences, decision: &Decision) {
    for (page_slot, page) in pages.iter_mut().enumerate() {
        let range = slots.on(page_slot);
        for (place, &slot) in fences.order[range.start as usize..range.end as usize].iter().enumerate() {
            page.lines[(slot.0 - range.start) as usize].reading_order = place + 1;
        }
        for (line, slot) in page.lines.iter_mut().zip(range.clone().map(Slot)) {
            line.exclude_from_body = decision.role[slot.index()] == Role::LineLabel;
        }
        match &fences.folios[page_slot] {
            Some((slot, label)) => {
                page.printed_label = Some(label.clone());
                page.printed_label_line_id = Some(page.lines[(slot.0 - range.start) as usize].id.clone());
            }
            None => {
                page.printed_label = None;
                page.printed_label_line_id = None;
            }
        }
    }
}

/// Each line's zone, by slot: its decided role's, a note's in the band of its kind (a page's note lines that go on with
/// the last note of the page before, its first and those after it up to the next label, a continuation's).
pub fn zones(slots: &Slots, evidence: &Evidence, fences: &Fences, notes: &Notes, decision: &Decision) -> Vec<LineZone> {
    let mut continuation = vec![false; slots.len()];
    for segment in notes.segments.iter().filter(|segment| segment.continued) {
        let Some((first, rest)) = segment.lines.split_first() else { continue };
        continuation[first.index()] = true;
        for &slot in rest.iter().take_while(|slot| evidence.label[slot.index()].is_none()) {
            continuation[slot.index()] = true;
        }
    }
    (0..slots.len())
        .map(|at| {
            let mode = match notes.note[at] {
                _ if continuation[at] => NoteMode::Continuation,
                Some(NoteKind::Endnote) => NoteMode::Endnote,
                _ => NoteMode::Footnote,
            };
            legal_pdf_structure_model::regions::zone_of(decision.role[at], fences.edge[at], mode)
        })
        .collect()
}

/// The document's lines as the note pairer reads them, in reading order (the pages read in it): each line's zone and
/// block, by slot, whether its opening number is no note's label (a heading's, or one the note band refused) or counts a
/// numbered paragraph (which pairing may overturn), its constituent document and language, and its citations (by line
/// id).
#[allow(clippy::too_many_arguments)]
pub fn note_lines<'a>(
    pages: &'a [Page],
    slots: &Slots,
    fences: &Fences,
    notes: &Notes,
    decision: &Decision,
    zones: &[LineZone],
    blocks: &[usize],
    citations: &'a BTreeMap<String, Vec<legal_pdf_core::model::ScalarRange>>,
) -> Vec<NoteLine<'a>> {
    fences
        .order
        .iter()
        .map(|&slot| {
            let at = slot.index();
            let line = slots.line(pages, slot);
            let context = fences.at(at);
            NoteLine {
                id: &line.id,
                text: &line.text,
                layout: Some(line),
                page: NotePage::of(&pages[slots.page(slot)]),
                zone: zones[at],
                block: Some(blocks[at]),
                label_suppressed: notes.suppressed[at] || decision.role[at] == Role::Heading,
                paragraph_label: matches!(decision.unit[at], Some((_, UnitRole::Paragraph))) || notes.paragraph_label[at],
                part: 2 * usize::from(context.document) + usize::from(context.translation),
                citations: citations.get(&line.id).map_or(&[], Vec::as_slice),
            }
        })
        .collect()
}
/// Whether two lines of a page stand on one row.
fn same_row(evidence: &Evidence, slots: &Slots, a: Slot, b: Slot) -> bool {
    let row = evidence.row(slots, a);
    row != u32::MAX && slots.page(a) == slots.page(b) && evidence.row(slots, b) == row
}

/// Each line's block, by slot, numbered on each page after the page's own blocks; a line label, folio or furniture keeps
/// its own.
pub fn blocks(pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, decision: &Decision) -> Vec<usize> {
    let count = slots.len();
    let role = &decision.role;
    let counted = |slot: Slot| matches!(class(role[slot.index()]), Class::Body | Class::Contents | Class::Table);
    let group = |slot: Slot| (fences.at(slot.index()).document, fences.at(slot.index()).translation);
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
    let documents = fences.documents();
    let (mut opening, mut first_unit) = (vec![usize::MAX; documents], vec![u32::MAX; documents]);
    for (place, &slot) in fences.order.iter().enumerate() {
        let document = usize::from(fences.at(slot.index()).document);
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
        let document = usize::from(fences.at(at).document);
        decision.rule[at] == Rule::FrontMatter
            && (slots.page(slot) < opening[document].saturating_add(2) || fences.place[at] < first_unit[document])
            && (evidence.words[at] & bits != 0 || bits & WORD_FIELD != 0 && legal_pdf_text::paragraphs::field_label(slots.line(pages, slot)))
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
            // The line a margin's paragraph number numbers goes on in the number's block.
            let start = previous.is_none_or(|prior| {
                let before = prior.index();
                if fences.margin_number[before] == Some(slot) {
                    return false;
                }
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
    blocks.into_iter().enumerate().map(|(at, block)| block.unwrap_or_else(|| slots.line(pages, Slot(at as u32)).block_index)).collect()
}
