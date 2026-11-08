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

use super::*;
use legal_pdf_tables::PageItems;

pub mod band;
pub mod columns;
pub mod contents;
pub mod contexts;
pub mod continuations;
pub mod documents;
pub mod furniture;
pub mod order;
pub mod quotations;
pub mod tables;

use legal_pdf_structure_model::Decided;

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
}

/// A reading context: a constituent document of the packet (0 for its own), its parallel
/// translation or not, and a quotation in it (0 for none).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Context {
    pub document: u16,
    pub translation: bool,
    pub quotation: u16,
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
    /// Each page's printed label and the slot of the line printing it, by page slot.
    pub folios: Vec<Option<(Slot, String)>>,
    /// The printed page number less the page's own, where most folios agree on one.
    pub folio_offset: Option<i64>,
    /// Edge lines naming their page's sections ("Section 63"), with their edge and the first
    /// section named, for the statute's spine to bear out (`confirm_section_heads`).
    pub section_heads: Vec<(Slot, Edge, u32)>,
    pub contents_entries: Vec<contents_lists::Entry>,
    /// Pages wholly a contents list's, by page slot.
    pub contents_pages: Vec<bool>,
    /// The document's tables, a contents list read as one aside.
    pub tables: Vec<legal_pdf_tables::PdfTable>,
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
    pub translation: Vec<bool>,
    pub starts: Vec<constituents::Start>,
    pub document: Vec<u16>,
    pub quotation: Vec<u16>,
    /// Each line's reading context, an index into `contexts`.
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
    pub fn transcript(&self, slot: Slot) -> bool {
        self.transcripts.get(usize::from(self.document[slot.index()])).copied().unwrap_or(false)
    }

    /// Whether the slot's constituent document is a transcript's testimony throughout (`testimony`).
    pub fn testimony(&self, slot: Slot) -> bool {
        self.testimony.get(usize::from(self.document[slot.index()])).copied().unwrap_or(false)
    }

    /// The fences, contexts and reading order of the document's pages, measured (`evidence`), its
    /// bookmarks `outline` and the page labels it declares (`page_labels`, by page index).
    pub fn read(pages: &[Page], slots: &Slots, evidence: &Evidence, outline: &[PdfOutlineEntry], page_labels: &[Option<String>]) -> Self {
        let count = slots.len();
        let mut fences = Self {
            role: vec![Role::Open; count],
            rule: vec![Rule::None; count],
            edge: vec![None; count],
            cell: vec![0; count],
            band: vec![false; count],
            translation: vec![false; count],
            document: vec![0; count],
            quotation: vec![0; count],
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
        fences.read_note_band(pages, slots, evidence);
        fences.read_body_type(pages, slots, evidence);
        fences.clear_false_bands(pages, slots, evidence);
        let mut voided = fences.free_noted_cells(pages, slots, evidence);
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
    pub fn read_numbers(&mut self, pages: &[Page], slots: &Slots, evidence: &Evidence, repaired: bool) {
        if repaired {
            self.read_margin_numbers(pages, slots);
        }
        self.read_contexts();
        self.read_continuations(pages, slots, evidence);
        self.read_paragraphs(pages, slots, evidence);
        if self.read_quotations(pages, slots) {
            self.read_contexts();
            self.read_continuations(pages, slots, evidence);
        }
    }

    /// The slot's reading context.
    pub fn context(&self, slot: Slot) -> Context {
        self.contexts[usize::from(self.ctx[slot.index()])]
    }
}

/// The decision as the derivation reads it, from the fences (`Decided`).
pub fn decided<'a>(pages: &[Page], slots: &'a Slots, fences: &'a Fences) -> Decided<'a> {
    Decided::of(pages, slots, &fences.order, &fences.document, &fences.translation, &fences.ctx, &fences.quotation, &fences.paragraphs.held)
}
