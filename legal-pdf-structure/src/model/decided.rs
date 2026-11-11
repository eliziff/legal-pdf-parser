//! The decision as what comes after it reads it, by line id (`Decided`).

use crate::fences::Fences;
use crate::model::{Held, Slot, Slots};
use legal_pdf_core::model::Page;

/// A document or embedded reading and its parallel sequence; primary text is (0, false).
pub(crate) type DocumentContext = (usize, bool);

/// A reading context: a constituent document of the packet (0 for its own), its parallel
/// translation or not, and a quotation in it (0 for none).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub(crate) struct Context {
    pub(crate) document: u16,
    pub(crate) translation: bool,
    pub(crate) quotation: u16,
}

/// The fences' readings of each line as the derivation reads them, by line id once the lines are in reading order:
/// its document and language, its reading context and quotation, and the numbered paragraphs it holds.
pub(crate) struct Decided<'a> {
    slots: &'a Slots,
    fences: &'a Fences,
    /// Each quotation's id ("quotation:" and its first line's id), by its number less one.
    quotations: Vec<String>,
}

impl<'a> Decided<'a> {
    /// The fences' readings, the pages as `write` finds them, before their lines move into reading order.
    pub(crate) fn of(pages: &[Page], slots: &'a Slots, fences: &'a Fences) -> Self {
        let count = fences.contexts.iter().map(|context| context.quotation).max().unwrap_or(0);
        let mut first: Vec<Option<Slot>> = vec![None; usize::from(count)];
        for &slot in &fences.order {
            let quotation = usize::from(fences.at(slot.index()).quotation);
            if quotation > 0 && first[quotation - 1].is_none() {
                first[quotation - 1] = Some(slot);
            }
        }
        let quotations = first
            .into_iter()
            .map(|slot| slot.map_or_else(String::new, |slot| format!("quotation:{}", slots.line(pages, slot).id)))
            .collect();
        Self { slots, fences, quotations }
    }

    /// The line's reading context.
    fn context(&self, id: &str) -> Option<Context> {
        self.slots.of_id(id).map(|slot| self.fences.at(slot.index()))
    }

    /// The line's constituent document and language.
    pub(crate) fn document_context(&self, id: &str) -> DocumentContext {
        self.context(id).map_or((0, false), |context| (usize::from(context.document), context.translation))
    }

    /// The line's reading context (its document, language and quotation) and language.
    pub(crate) fn reading_context(&self, id: &str) -> DocumentContext {
        self.slots.of_id(id).map_or((0, false), |slot| (usize::from(self.fences.ctx[slot.index()]), self.fences.at(slot.index()).translation))
    }

    /// The line is in the French column of a parallel English and French text.
    pub(crate) fn translated(&self, id: &str) -> bool {
        self.context(id).is_some_and(|context| context.translation)
    }

    /// The numbered paragraphs the line's text holds: each one's language, number and UTF-16 span of the line.
    pub(crate) fn held(&self, id: &str) -> &[Held] {
        self.slots.of_id(id).map_or(&[], |slot| self.fences.paragraphs.held[slot.index()].as_slice())
    }

    /// The id of the quotation holding the line.
    pub(crate) fn quotation(&self, id: &str) -> Option<&str> {
        let quotation = usize::from(self.context(id)?.quotation);
        (quotation > 0).then(|| self.quotations[quotation - 1].as_str())
    }
}
