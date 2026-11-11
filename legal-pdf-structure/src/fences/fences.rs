//! What a page's geometry and a document's own lists settle before any run of numbers is read, and
//! the order the lines read in. Each fence takes only the lines no fence before it took, in this
//! order: a transcript's line numbers; the furniture and folios at the pages' edges; then each
//! page's columns, read on the words the fences left; the parallel translation's lines; the
//! contents lists (their title a heading; a table of authorities' lines left to the body); the
//! tables' cells. Then the
//! note separator clear of the tables and the band under it, whose table cells are its notes' where
//! a note label stands in it; the body's type; each line's constituent document, and each
//! document's own furniture; whether each document is a transcript, and the lines beside its number
//! column, whose testimony no table holds; the reading order page by page; the quotations a
//! document's own count does not hold, each with the editorial lines set under it; the reading
//! contexts; and each line's continuation of the line before it.

use legal_pdf_core::model::{Diagnostic, Page, PdfOutlineEntry};
use crate::constituents;
use crate::headings::contents_lists;
use crate::layout::ColumnModel;
use crate::model::{Context, Edge, PageView, Paragraphs, Role, Rule, Slot, Slots};
use crate::model::evidence::{Evidence, Rows};

pub(crate) mod band;
pub(crate) mod columns;
pub(crate) mod contents;
pub(crate) mod continuations;
pub(crate) mod documents;
pub(crate) mod furniture;
pub(crate) mod order;
pub(crate) mod quotations;
pub(crate) mod tables;

use crate::model::Decided;

/// Why a line goes on with the one before it in reading order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) enum Continuation {
    /// A word hyphenated across, an unfinished sentence going on in lower case, or a bracket closed
    /// (`body_flow_edge`).
    Flow,
    /// The line above ends on an enactment's noun its number completes (`reference_wrapped_onto`).
    Reference,
    /// A full line ending no sentence whose sentence this line finishes (`carries_a_sentence_onto`).
    Sentence,
    /// Set at its paragraph's leading under the line before it, a full line ending no sentence
    /// (`continues_prose_from`).
    Prose,
}

/// The fences, contexts and reading order, by slot.
#[derive(Debug, Default, serde::Serialize)]
pub(crate) struct Fences {
    pub(crate) role: Vec<Role>,
    pub(crate) rule: Vec<Rule>,
    /// The edge a furniture or page label line stands at.
    pub(crate) edge: Vec<Option<Edge>>,
    /// Each page's columns, read on the words of its lines no line-number or furniture fence took
    /// (`columns_of`), by page slot.
    pub(crate) columns: Vec<ColumnModel>,
    /// Each constituent document is a transcript, by document: its lines are numbered, or a tenth
    /// of them or more (twenty or more) open on a speaker's label.
    pub(crate) transcripts: Vec<bool>,
    /// Each transcript whose lines are testimony throughout, by document: a tenth of them or more
    /// (twenty or more) open on a speaker's label, or its line numbers stand on half its pages.
    pub(crate) testimony: Vec<bool>,
    /// A transcript's line beside its number column: on a row holding a line number.
    pub(crate) numbered_row: Vec<bool>,
    /// Each page's printed label and the slot of the line printing it, by page slot.
    pub(crate) folios: Vec<Option<(Slot, String)>>,
    /// The printed page number less the page's own, where most folios agree on one.
    pub(crate) folio_offset: Option<i64>,
    /// Edge lines naming their page's sections ("Section 63"), with their edge and the first
    /// section named, for the statute's spine to bear out (`confirm_section_heads`).
    pub(crate) section_heads: Vec<(Slot, Edge, u32)>,
    pub(crate) contents_entries: Vec<contents_lists::Entry>,
    /// Pages wholly a contents list's, by page slot.
    pub(crate) contents_pages: Vec<bool>,
    /// The document's tables, a contents list read as one aside.
    pub(crate) tables: Vec<crate::tables::PdfTable>,
    /// The cell each line stands in (1 for the tables' first cell; 0 for none), a line parted
    /// across cells standing in the first.
    pub(crate) cell: Vec<u32>,
    /// Each page's note separator, unless it runs through a table.
    pub(crate) separators: Vec<Option<f64>>,
    /// Under its page's note separator, and no line label, furniture or folio.
    pub(crate) band: Vec<bool>,
    /// Each page's band holds a note label no reason suppresses, by page slot: its table cells are
    /// its notes', no table's.
    pub(crate) noted: Vec<bool>,
    /// The body's type: the commonest face (an index into the evidence's faces) and half-point size
    /// of the lines no fence took outside the note band, by characters.
    pub(crate) body: Option<(u16, u16)>,
    pub(crate) starts: Vec<constituents::Start>,
    /// Each line's reading context (its document, language and quotation), an index into `contexts`: what the fences
    /// read of each, as they read it.
    pub(crate) ctx: Vec<u16>,
    pub(crate) contexts: Vec<Context>,
    /// A paragraph number printed in the margin (`read_margin_numbers`): the body line it numbers.
    pub(crate) margin_number: Vec<Option<Slot>>,
    /// The line opening the same paragraph in the other language, on a margin number's row.
    pub(crate) margin_twin: Vec<Option<Slot>>,
    /// The document's numbered paragraphs (`read_paragraphs`).
    pub(crate) paragraphs: Paragraphs,
    /// Every line in reading order, page by page.
    pub(crate) order: Vec<Slot>,
    /// Each line's place in `order`.
    pub(crate) place: Vec<u32>,
    /// How each line no fence took goes on with the line before it in its reading context and
    /// stream (the band's or the body's), in reading order.
    pub(crate) continues: Vec<Option<Continuation>>,
    pub(crate) diagnostics: Vec<Diagnostic>,
}

impl Fences {
    /// Puts the line in the context `change` makes of its own, the document's contexts holding each once.
    fn recontext(&mut self, slot: Slot, change: impl FnOnce(&mut Context)) {
        let mut context = self.at(slot.index());
        change(&mut context);
        let known = self.contexts.iter().position(|other| *other == context).unwrap_or_else(|| {
            self.contexts.push(context);
            self.contexts.len() - 1
        });
        self.ctx[slot.index()] = known as u16;
    }

    /// The reading context of the line at slot index `at`.
    pub(crate) fn at(&self, at: usize) -> Context {
        self.contexts[usize::from(self.ctx[at])]
    }

    /// The number of constituent documents: one past the last any line stands in.
    pub(crate) fn documents(&self) -> usize {
        usize::from((0..self.ctx.len()).map(|at| self.at(at).document).max().unwrap_or(0)) + 1
    }

    /// Gives an open line a role, by a rule; a line some fence took already keeps its own.
    fn set(&mut self, slot: Slot, role: Role, rule: Rule) -> bool {
        let open = self.role[slot.index()] == Role::Open;
        if open {
            self.role[slot.index()] = role;
            self.rule[slot.index()] = rule;
        }
        open
    }

    /// Whether the slot's constituent document is a transcript.
    pub(crate) fn transcript(&self, slot: Slot) -> bool {
        self.transcripts.get(usize::from(self.at(slot.index()).document)).copied().unwrap_or(false)
    }

    /// Whether the slot's constituent document is a transcript's testimony throughout (`testimony`).
    pub(crate) fn testimony(&self, slot: Slot) -> bool {
        self.testimony.get(usize::from(self.at(slot.index()).document)).copied().unwrap_or(false)
    }

    /// The fences, contexts and reading order of the document's pages, measured (`evidence`), its
    /// bookmarks `outline` and the page labels it declares (`page_labels`, by page index).
    pub(crate) fn read(pages: &[Page], slots: &Slots, evidence: &Evidence, outline: &[PdfOutlineEntry], page_labels: &[Option<String>]) -> Self {
        let count = slots.len();
        let mut fences = Self {
            role: vec![Role::Open; count],
            rule: vec![Rule::None; count],
            edge: vec![None; count],
            cell: vec![0; count],
            band: vec![false; count],
            ctx: vec![0; count],
            contexts: vec![Context::default()],
            continues: vec![None; count],
            ..Self::default()
        };
        let views = PageView::whole(pages);
        let rows: Vec<&Rows> = evidence.pages.iter().map(|page| &page.rows).collect();
        // A transcript's line numbers.
        for slot in (0..count).filter(|&slot| evidence.line_number[slot]) {
            fences.set(Slot(slot as u32), Role::LineLabel, Rule::LineNumber);
        }
        fences.read_furniture(pages, slots, evidence, &views, &rows);
        let items = fences.read_columns(pages, slots, evidence);
        fences.read_translation(pages, slots, evidence, &views);
        let (tables, in_table, listed) = fences.find_tables(pages, slots, &items);
        let listed_contents = fences.read_contents(pages, slots, evidence, page_labels, &views, &in_table, &listed);
        fences.read_tables(slots, tables, &listed_contents);
        let mut voided = fences.free_referenced_note_cells(pages, slots, evidence);
        fences.read_note_band(pages, slots, evidence);
        fences.read_body_type(pages, slots, evidence);
        fences.clear_false_bands(pages, slots, evidence);
        voided |= fences.free_noted_cells(pages, slots, evidence);
        fences.read_documents(&views, outline);
        if !fences.starts.is_empty() {
            fences.read_document_furniture(pages, slots, evidence, &rows);
        }
        fences.read_transcripts(slots, evidence);
        voided |= fences.free_spoken_cells(slots, evidence);
        if voided {
            fences.prune_tables(slots);
        }
        fences.read_order(pages, slots, evidence);
        fences.read_margin_numbers(pages, slots);
        fences
    }

    /// The document's paragraph numbers, read once its misread markers are restored (`Runs::read`), the margin's again
    /// where any was. The judgment's paragraphs are counted on the lines' continuations, and the printed numbers the
    /// count does not take are another writing's, quoted: their passages are contexts of their own, read with the lines again.
    pub(crate) fn read_numbers(&mut self, pages: &[Page], slots: &Slots, evidence: &Evidence, repaired: bool) {
        if repaired {
            self.read_margin_numbers(pages, slots);
        }
        self.read_continuations(pages, slots);
        self.read_paragraphs(pages, slots, evidence);
        if self.read_quotations(pages, slots) {
            self.read_continuations(pages, slots);
        }
    }
}

/// The decision as the derivation reads it, from the fences (`Decided`).
pub(crate) fn decided<'a>(pages: &[Page], slots: &'a Slots, fences: &'a Fences) -> Decided<'a> {
    Decided::of(pages, slots, &fences.order, &fences.ctx, &fences.contexts, &fences.paragraphs.held)
}
