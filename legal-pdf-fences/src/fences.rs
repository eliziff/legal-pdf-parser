//! What a page's geometry and a document's own lists settle before any run of numbers is read, and
//! the order the lines read in. Each fence takes only the lines no fence before it took, in this
//! order: a transcript's line numbers; the lines printing the pages' numbers (the page map's); the
//! furniture at the pages' edges; then each
//! page's columns, read on the words the fences left; the parallel translation's lines; the
//! contents lists (their title a heading; a table of authorities' lines left to the body); the
//! tables' cells. Then the
//! note separator clear of the tables and the band under it, whose table cells are its notes' where
//! a note label stands in it; the body's type; each line's constituent document, and each
//! document's own furniture; whether each document is a transcript, and the lines beside its number
//! column, whose testimony no table holds; the reading order page by page; the quotations a
//! document's own count does not hold, each with the editorial lines set under it; the reading
//! contexts; and each line's continuation of the line before it.

use crate::model::fences::Fences;
use crate::model::{Context, Evidence, PageView, Role, Rows, Rule, Slot, Slots};
use legal_pdf_core::model::{Page, PdfOutlineEntry};

pub mod band;
pub mod columns;
pub mod contents;
pub mod continuations;
pub mod documents;
pub mod furniture;
pub mod order;
pub mod quotations;
pub mod tables;



/// Puts the line in the context `change` makes of its own, the document's contexts holding each once.
/// Puts the line in the context `change` makes of its own, the document's contexts holding each once.
pub fn recontext(fences: &mut Fences, slot: Slot, change: impl FnOnce(&mut Context)) {
    let mut context = fences.at(slot.index());
    change(&mut context);
    let known = fences
        .contexts
        .iter()
        .position(|other| *other == context)
        .unwrap_or_else(|| {
            fences.contexts.push(context);
            fences.contexts.len() - 1
        });
    fences.ctx[slot.index()] = known as u16;
}

/// Gives an open line a role, by a rule; a line some fence took already keeps its own.
/// Gives an open line a role, by a rule; a line some fence took already keeps its own.
pub fn set(fences: &mut Fences, slot: Slot, role: Role, rule: Rule) -> bool {
    let open = fences.role[slot.index()] == Role::Open;
    if open {
        fences.role[slot.index()] = role;
        fences.rule[slot.index()] = rule;
    }
    open
}

/// The fences, contexts and reading order of the document's pages, measured (`evidence`), its
/// bookmarks `outline` and the page labels it declares (`page_labels`, by page index).
/// The fences, contexts and reading order of the document's pages, measured (`evidence`), its
/// bookmarks `outline` and the page labels it declares (`page_labels`, by page index).
pub fn read(
    pages: &[Page],
    slots: &Slots,
    evidence: &Evidence,
    outline: &[PdfOutlineEntry],
    page_labels: &[Option<String>],
) -> Fences {
    let count = slots.len();
    let mut fences = Fences {
        role: vec![Role::Open; count],
        rule: vec![Rule::None; count],
        edge: vec![None; count],
        cell: vec![0; count],
        band: vec![false; count],
        ctx: vec![0; count],
        contexts: vec![Context::default()],
        continues: vec![None; count],
        ..Fences::default()
    };
    let views = legal_pdf_core::profile::tally("fences.views", || PageView::whole(pages));
    let rows: Vec<&Rows> = evidence.pages.iter().map(|page| &page.rows).collect();
    // A transcript's line numbers.
    for slot in (0..count).filter(|&slot| evidence.line_number[slot]) {
        crate::find::set(&mut fences, Slot(slot as u32), Role::LineLabel, Rule::LineNumber);
    }
    legal_pdf_core::profile::tally("fences.read_page_numbers", || {
        crate::find::read_page_numbers(&mut fences, pages, slots, evidence, page_labels)
    });
    legal_pdf_core::profile::tally("fences.read_furniture", || {
        crate::find::read_furniture(&mut fences, evidence, &views, &rows)
    });
    let items = legal_pdf_core::profile::tally("fences.read_columns", || {
        crate::find::read_columns(&mut fences, pages, slots, evidence)
    });
    legal_pdf_core::profile::tally("fences.read_translation", || {
        crate::find::read_translation(&mut fences, pages, slots, evidence, &views)
    });
    let (tables, in_table, listed) =
        legal_pdf_core::profile::tally("fences.find_tables", || {
            crate::find::find_tables(&fences, pages, slots, &items)
        });
    let listed_contents = legal_pdf_core::profile::tally("fences.read_contents", || {
        crate::find::read_contents(&mut fences, 
            pages,
            slots,
            evidence,
            page_labels,
            &views,
            &in_table,
            &listed,
        )
    });
    legal_pdf_core::profile::tally("fences.read_tables", || {
        crate::find::read_tables(&mut fences, slots, tables, &listed_contents)
    });
    let mut voided =
        legal_pdf_core::profile::tally("fences.free_referenced_note_cells", || {
            crate::find::free_referenced_note_cells(&mut fences, pages, slots, evidence)
        });
    legal_pdf_core::profile::tally("fences.read_note_band", || {
        crate::find::read_note_band(&mut fences, pages, slots, evidence)
    });
    legal_pdf_core::profile::tally("fences.read_body_type", || {
        crate::find::read_body_type(&mut fences, pages, slots, evidence)
    });
    legal_pdf_core::profile::tally("fences.clear_false_bands", || {
        crate::find::clear_false_bands(&mut fences, pages, slots, evidence)
    });
    voided |= legal_pdf_core::profile::tally("fences.free_noted_cells", || {
        crate::find::free_noted_cells(&mut fences, pages, slots, evidence)
    });
    legal_pdf_core::profile::tally("fences.read_documents", || {
        crate::find::read_documents(&mut fences, &views, outline)
    });
    if !fences.starts.is_empty() {
        legal_pdf_core::profile::tally("fences.read_document_furniture", || {
            crate::find::read_document_furniture(&mut fences, pages, slots, evidence, &rows)
        });
    }
    legal_pdf_core::profile::tally("fences.read_transcripts", || {
        crate::find::read_transcripts(&mut fences, slots, evidence)
    });
    voided |= legal_pdf_core::profile::tally("fences.free_spoken_cells", || {
        crate::find::free_spoken_cells(&mut fences, slots, evidence)
    });
    if voided {
        legal_pdf_core::profile::tally("fences.prune_tables", || crate::find::prune_tables(&mut fences, slots));
    }
    legal_pdf_core::profile::tally("fences.read_order", || {
        crate::find::read_order(&mut fences, pages, slots, evidence)
    });
    legal_pdf_core::profile::tally("fences.read_margin_numbers", || {
        crate::find::read_margin_numbers(&mut fences, pages, slots)
    });
    crate::find::read_margin_labels(&mut fences, pages, slots);
    fences
}

/// The document's paragraph numbers, read once its misread markers are restored (`Runs::read`), the margin's again
/// where any was. The judgment's paragraphs are counted on the lines' continuations, and the printed numbers the
/// count does not take are another writing's, quoted: their passages are contexts of their own, read with the lines again.
/// The document's paragraph numbers, read once its misread markers are restored (`Runs::read`), the margin's again
/// where any was. The judgment's paragraphs are counted on the lines' continuations, and the printed numbers the
/// count does not take are another writing's, quoted: their passages are contexts of their own, read with the lines again.
pub fn read_numbers(fences: &mut Fences, pages: &[Page],
    slots: &Slots,
    evidence: &Evidence,
    repaired: bool,
    judgment_paragraphs: crate::JudgmentParagraphs,
) {
    use legal_pdf_core::profile::tally;
    if repaired {
        tally("numbers.margin", || {
            crate::find::read_margin_numbers(fences, pages, slots);
            crate::find::read_margin_labels(fences, pages, slots);
        });
    }
    tally("numbers.continuations", || crate::find::read_continuations(fences, pages, slots));
    tally("numbers.paragraphs", || crate::find::read_paragraphs(fences, pages, slots, evidence, judgment_paragraphs));
    if tally("numbers.quotations", || crate::find::read_quotations(fences, pages, slots)) {
        tally("numbers.continuations", || crate::find::read_continuations(fences, pages, slots));
    }
}

