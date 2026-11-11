//! The decision as the readers after it take it: written to the pages once (each line's place in
//! its page's reading order, whether it is held out of the body as a transcript's line number, each
//! page's printed label and the line printing it), and read by slot (each line's zone, and the lines as the note pairer
//! reads them); and what parts the lines into blocks (`Paragraphs`), which the walk numbers as it decides them.
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

use std::collections::BTreeMap;
use legal_pdf_core::model::{Line, Page};
use crate::headings;
use crate::fences::Fences;
use crate::model::{
    Evidence, Role, Rule, Slot, Slots, WORD_COURT, WORD_DATE, WORD_FIELD, WORD_JURAT, WORD_LABEL, WORD_PARTY,
};
use crate::notes::band::NoteKind;
use crate::text::paragraphs::{measure_of, paragraphs_of, Measure};
use super::notes::Notes;
use super::rules::Decision;
use crate::machine::machine::Move;
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
pub(crate) fn write_lines(pages: &mut [Page], slots: &Slots, fences: &Fences, decision: &Decision) {
    for (page_slot, page) in pages.iter_mut().enumerate() {
        let range = slots.on(page_slot);
        for (place, &slot) in fences.order[range.start as usize..range.end as usize].iter().enumerate() {
            page.lines[(slot.0 - range.start) as usize].reading_order = place + 1;
        }
        for (line, slot) in page.lines.iter_mut().zip(range.clone().map(Slot)) {
            line.exclude_from_body = decision.line(slot.index()).role == Role::LineLabel;
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
pub(crate) fn zones(slots: &Slots, evidence: &Evidence, fences: &Fences, notes: &Notes, decision: &Decision) -> Vec<LineZone> {
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
            crate::model::regions::zone_of(decision.line(at).role, fences.edge[at], mode)
        })
        .collect()
}

/// The document's lines as the note pairer reads them, in reading order (the pages read in it): each line's zone and
/// block, by slot, whether its opening number is no note's label (a heading's, or one the note band refused) or counts a
/// numbered paragraph (which pairing may overturn), its constituent document and language, and its citations (by line
/// id).
#[allow(clippy::too_many_arguments)]
pub(crate) fn note_lines<'a>(
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
                label_suppressed: notes.suppressed[at] || decision.line(at).role == Role::Heading,
                paragraph_label: matches!(decision.line(at).unit, Some((_, UnitRole::Paragraph))) || notes.paragraph_label[at],
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

/// What parts a page's lines into blocks, read before the walk decides them (`tier0::settle`, which numbers the blocks
/// as it goes): each line's printed paragraph (`paragraphs_of`) among its document and language's counted lines and its
/// contents entry (`entry_rows`), over the roles the rules first gave the lines; and each document's first page.
pub(crate) struct Paragraphs {
    paragraph: Vec<Option<usize>>,
    entry: Vec<Option<usize>>,
    opening: Vec<usize>,
}

impl Paragraphs {
pub(crate) fn read(pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, role: impl Fn(usize) -> Role) -> Self {
    let count = slots.len();
    let counted = |slot: Slot| matches!(class(role(slot.index())), Class::Body | Class::Contents | Class::Table);
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
    let mut opening = vec![usize::MAX; fences.documents()];
    for &slot in &fences.order {
        let document = usize::from(fences.at(slot.index()).document);
        opening[document] = opening[document].min(slots.page(slot));
    }
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
            let rows: Vec<u32> = lines.iter().map(|&slot| evidence.row(slots, slot)).collect();
            for (slot, number) in lines.into_iter().zip(paragraphs_of(&read, &rows, &flags, measure)) {
                paragraph[slot.index()] = number;
            }
        }
        // Its contents list's entries.
        if order.iter().any(|&slot| role(slot.index()) == Role::Contents) {
            let lines: Vec<Slot> = order.iter().copied().filter(|&slot| role(slot.index()) != Role::LineLabel).collect();
            let read: Vec<&Line> = lines.iter().map(|&slot| slots.line(pages, slot)).collect();
            let rows: Vec<u32> = lines.iter().map(|&slot| evidence.row(slots, slot)).collect();
            let entries = headings::contents::entry_rows(page.width, &read, &rows, |at| role(lines[at].index()) == Role::Contents);
            for (slot, number) in lines.into_iter().zip(entries) {
                entry[slot.index()] = number;
            }
        }
    }
    Self { paragraph, entry, opening }
}

/// Whether a line (its slot, its move, and whether it stands before its document's first unit) starts a block after
/// `prior`, the line before it on its page (likewise); neither a line label, folio or furniture, which keeps its own.
/// A caption's line of front matter (`Rule::FrontMatter`) in its document's front (its first two pages, its lines
/// before its first unit): a field, a label, a court's name or a jurat stands on its own (`opens`), and the line after
/// one of those, a party's role or a date stands on its own (`closes`). A signature, an author's line or a lead-in's
/// colon is no heading, and still the paragraph's it stands in. The line a margin's paragraph number numbers goes on in
/// the number's block.
pub(crate) fn starts(&self, pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, prior: (Slot, &Move, bool), line: (Slot, &Move, bool)) -> bool {
    let caption = |(slot, decided, front): (Slot, &Move, bool), bits: u32| {
        let at = slot.index();
        decided.rule == Rule::FrontMatter
            && (slots.page(slot) < self.opening[usize::from(fences.at(at).document)].saturating_add(2) || front)
            && (evidence.words[at] & bits != 0 || bits & WORD_FIELD != 0 && crate::text::paragraphs::field_label(slots.line(pages, slot)))
    };
    let (slot, decided, _) = line;
    let (before, at) = (prior.0.index(), slot.index());
    if fences.margin_number[before] == Some(slot) {
        return false;
    }
    let kind = class(decided.role);
    let free = !decided.joined && fences.continues[at].is_none();
    let opens = || caption(line, WORD_FIELD | WORD_LABEL | WORD_COURT | WORD_JURAT) && evidence.words[at] & WORD_PARTY == 0 && fences.continues[at].is_none();
    let closes = || caption(prior, WORD_FIELD | WORD_LABEL | WORD_COURT | WORD_JURAT | WORD_PARTY | WORD_DATE);
    class(prior.1.role) != kind
        || fences.ctx[before] != fences.ctx[at]
        || match kind {
            Class::Body => {
                decided.unit.is_some()
                    || (opens() || closes() && free) && !decided.joined && !same_row(evidence, slots, prior.0, slot)
                    || free && self.paragraph[at] != self.paragraph[before]
            }
            Class::Heading | Class::Title | Class::MarginalNote => !decided.joined,
            Class::Note => evidence.label[at].is_some(),
            Class::Contents => self.entry[at] != self.entry[before],
            Class::Table => self.paragraph[at] != self.paragraph[before] || fences.cell[at] != fences.cell[before],
            Class::Edge => true,
        }
}
}