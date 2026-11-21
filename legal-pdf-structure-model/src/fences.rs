//! The fences: each line's role, rule, edge and reading context, the document's reading order, its tables' cells and its
//! note bands, as the fences stage leaves them for every stage after it.

use crate::{ColumnModel, Context, Edge, Paragraphs, Role, Rule, Slot};
use legal_pdf_core::model::Diagnostic;

/// Why a line goes on with the one before it in reading order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Continuation {
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
    /// A page's first line of its stream, going on with the paragraph the page before ended, as the two pages set the
    /// stream's lines (`continues_onto`).
    Page,
}

/// The fences, contexts and reading order, by slot.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Fences {
    pub role: Vec<Role>,
    pub rule: Vec<Rule>,
    /// The edge a furniture or page label line stands at.
    pub edge: Vec<Option<Edge>>,
    /// Each page's columns, read on the words of its lines no line-number or furniture fence took
    /// (`columns_of`), by page slot.
    pub columns: Vec<ColumnModel>,
    /// Each constituent document is a transcript, by document: its lines are numbered, or a tenth
    /// of them or more (twenty or more) open on a speaker's label.
    pub transcripts: Vec<bool>,
    /// Each transcript whose lines are testimony throughout, by document: a tenth of them or more
    /// (twenty or more) open on a speaker's label, or its line numbers stand on half its pages.
    pub testimony: Vec<bool>,
    /// A transcript's line beside its number column: on a row holding a line number.
    pub numbered_row: Vec<bool>,
    /// Each page's number as the page map reads it and the slot of the line printing it, where the page prints it, by
    /// page slot.
    pub folios: Vec<Option<(String, Option<Slot>)>>,
    /// Edge lines naming their page's sections ("Section 63"), with their edge and the first
    /// section named, for the statute's spine to bear out (`confirm_section_heads`).
    pub section_heads: Vec<(Slot, Edge, u32)>,
    pub contents_entries: Vec<crate::contents_lists::Entry>,
    /// Pages wholly a contents list's, by page slot.
    pub contents_pages: Vec<bool>,
    /// The document's tables, a contents list read as one aside.
    pub tables: Vec<crate::tables::PdfTable>,
    /// The cell each line stands in (1 for the tables' first cell; 0 for none), a line parted
    /// across cells standing in the first.
    pub cell: Vec<u32>,
    /// Each page's note separator, unless it runs through a table.
    pub separators: Vec<Option<f64>>,
    /// Under its page's note separator, and no line label, furniture or folio.
    pub band: Vec<bool>,
    /// Each page's band holds a note label no reason suppresses, by page slot: its table cells are
    /// its notes', no table's.
    pub noted: Vec<bool>,
    /// The body's type: the commonest face (an index into the evidence's faces) and half-point size
    /// of the lines no fence took outside the note band, by characters.
    pub body: Option<(u16, u16)>,
    pub starts: Vec<crate::constituents::Start>,
    /// Each line's reading context (its document, language and quotation), an index into `contexts`: what the fences
    /// read of each, as they read it.
    pub ctx: Vec<u16>,
    pub contexts: Vec<Context>,
    /// A paragraph number printed in the margin (`read_margin_numbers`): the body line it numbers.
    pub margin_number: Vec<Option<Slot>>,
    /// The line opening the same paragraph in the other language, on a margin number's row.
    pub margin_twin: Vec<Option<Slot>>,
    /// The document's numbered paragraphs (`read_paragraphs`).
    pub paragraphs: Paragraphs,
    /// Every line in reading order, page by page.
    pub order: Vec<Slot>,
    /// Each line's place in `order`.
    pub place: Vec<u32>,
    /// How each line no fence took goes on with the line before it in its reading context and
    /// stream (the band's or the body's), in reading order.
    pub continues: Vec<Option<Continuation>>,
    pub diagnostics: Vec<Diagnostic>,
}

impl Fences {
    /// The reading context of the line at slot index `at`.
    pub fn at(&self, at: usize) -> Context {
        self.contexts[usize::from(self.ctx[at])]
    }
    /// The number of constituent documents: one past the last any line stands in.
    pub fn documents(&self) -> usize {
        usize::from(
            (0..self.ctx.len())
                .map(|at| self.at(at).document)
                .max()
                .unwrap_or(0),
        ) + 1
    }
    /// Whether the slot's constituent document is a transcript.
    pub fn transcript(&self, slot: Slot) -> bool {
        self.transcripts
            .get(usize::from(self.at(slot.index()).document))
            .copied()
            .unwrap_or(false)
    }
    /// Whether the slot's constituent document is a transcript's testimony throughout (`testimony`).
    pub fn testimony(&self, slot: Slot) -> bool {
        self.testimony
            .get(usize::from(self.at(slot.index()).document))
            .copied()
            .unwrap_or(false)
    }
}
