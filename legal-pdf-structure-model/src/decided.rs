//! The decision as what comes after it reads it, by line id (`Decided`).

use crate::{Held, Slot, Slots};
use legal_pdf_core::model::{Footnote, Page};

/// A document or embedded reading and its parallel sequence; primary text is (0, false).
pub type DocumentContext = (usize, bool);

/// The decision as the derivation reads it, by line id: each line's document and language, its
/// reading context, role, quotation and heading level; and the document's contents pages and
/// tables.
pub struct Decided<'a> {
    slots: &'a Slots,
    /// Each line's constituent document, language (the translation's column), reading context and quotation, and the
    /// numbered paragraphs it holds, by slot (the fences' readings).
    document: &'a [u16],
    translation: &'a [bool],
    ctx: &'a [u16],
    quotation: &'a [u16],
    held: &'a [Vec<Held>],
    /// Each quotation's id ("quotation:" and its first line's id), by its number less one.
    quotations: Vec<String>,
}

impl<'a> Decided<'a> {
    /// The decision on the pages as `write` finds them, before their lines move into reading order: the fences'
    /// readings of each line, in reading order (`order`).
    #[allow(clippy::too_many_arguments)]
    pub fn of(pages: &[Page], slots: &'a Slots, order: &[Slot], document: &'a [u16], translation: &'a [bool], ctx: &'a [u16],
        quotation: &'a [u16], held: &'a [Vec<Held>]) -> Self {
        let count = quotation.iter().copied().max().unwrap_or(0);
        let mut first: Vec<Option<Slot>> = vec![None; usize::from(count)];
        for &slot in order {
            let quotation = usize::from(quotation[slot.index()]);
            if quotation > 0 && first[quotation - 1].is_none() {
                first[quotation - 1] = Some(slot);
            }
        }
        let quotations = first
            .into_iter()
            .map(|slot| slot.map_or_else(String::new, |slot| format!("quotation:{}", slots.line(pages, slot).id)))
            .collect();
        Self { slots, document, translation, ctx, quotation, held, quotations }
    }

    fn slot(&self, id: &str) -> Option<Slot> {
        self.slots.of_id(id)
    }

    /// The line's constituent document and language.
    pub fn document_context(&self, id: &str) -> DocumentContext {
        self.slot(id).map_or((0, false), |slot| {
            (usize::from(self.document[slot.index()]), self.translation[slot.index()])
        })
    }

    /// The line's reading context (its document, language and quotation) and language.
    pub fn reading_context(&self, id: &str) -> DocumentContext {
        self.slot(id).map_or((0, false), |slot| {
            (usize::from(self.ctx[slot.index()]), self.translation[slot.index()])
        })
    }

    /// The document context of a note's first line.
    pub fn note_context(&self, note: &Footnote) -> DocumentContext {
        note.body_line_ids
            .first()
            .map_or((0, false), |id| self.document_context(id))
    }

    /// The line is in the French column of a parallel English and French text.
    pub fn translated(&self, id: &str) -> bool {
        self.slot(id).is_some_and(|slot| self.translation[slot.index()])
    }

    /// The numbered paragraphs the line's text holds: each one's language, number and UTF-16 span of the line.
    pub fn held(&self, id: &str) -> &[Held] {
        self.slot(id).map_or(&[], |slot| self.held[slot.index()].as_slice())
    }

    /// The id of the quotation holding the line.
    pub fn quotation(&self, id: &str) -> Option<&str> {
        let quotation = usize::from(self.quotation[self.slot(id)?.index()]);
        (quotation > 0).then(|| self.quotations[quotation - 1].as_str())
    }
}
