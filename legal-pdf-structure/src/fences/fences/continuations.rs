//! How each line goes on with the line before it, and the statute's running heads its spine bears out.
use legal_pdf_core::model::Page;
use crate::headings::{body_flow_edge, carries_a_sentence_onto, continues_prose_from};
use crate::model::{Evidence, Role, Rule, Slot, Slots};
use crate::notes::reference_wrapped_onto;
use super::{Continuation, Fences};

impl Fences {
    /// How each line no fence took goes on with the line before it, in its reading context and
    /// stream, in reading order: on one page by any continuation, across a page by the text's flow.
    pub(crate) fn read_continuations(&mut self, pages: &[Page], slots: &Slots) {
        self.continues = vec![None; slots.len()];
        let mut last: Vec<Option<Slot>> = vec![None; self.contexts.len() * 2];
        for &slot in &self.order {
            if self.role[slot.index()] != Role::Open {
                continue;
            }
            let band = self.band[slot.index()];
            let stream = usize::from(self.ctx[slot.index()]) * 2 + usize::from(band);
            let Some(prior) = last[stream].replace(slot) else {
                continue;
            };
            let (line, before) = (slots.line(pages, slot), slots.line(pages, prior));
            let page_slot = slots.page(slot);
            self.continues[slot.index()] = if body_flow_edge(before, line) {
                Some(Continuation::Flow)
            } else if slots.page(prior) != page_slot {
                None
            } else {
                let page = &pages[page_slot];
                let range = slots.on(page_slot);
                let start = range.start as usize;
                let (at, prior_at) = (slot.index() - start, prior.index() - start);
                let body = |other: usize| self.role[start + other] == Role::Open && self.band[start + other] == band;
                if reference_wrapped_onto(&page.lines, at) {
                    Some(Continuation::Reference)
                } else if carries_a_sentence_onto(page, prior_at, at, body) {
                    Some(Continuation::Sentence)
                } else if continues_prose_from(page, prior_at, at, body) {
                    Some(Continuation::Prose)
                } else {
                    None
                }
            };
        }
    }

    /// The statute's running heads naming their page's sections, where its spine bears them out
    /// (`spine`: a section numbered so opens on that page slot), on their page or the one before;
    /// and the lines on their rows. The lines' continuations are read again without them.
    pub(crate) fn confirm_section_heads(&mut self, pages: &[Page], slots: &Slots, evidence: &Evidence, spine: impl Fn(usize, u32) -> bool) {
        let mut fenced = false;
        for (slot, edge, number) in std::mem::take(&mut self.section_heads) {
            let page = slots.page(slot);
            if !(spine(page, number) || page > 0 && spine(page - 1, number)) || !self.set(slot, Role::Furniture, Rule::SectionHead) {
                continue;
            }
            fenced = true;
            self.edge[slot.index()] = Some(edge);
            self.band[slot.index()] = false;
            let row = evidence.row(slots, slot);
            for mate in slots.on(page).map(Slot) {
                if row != u32::MAX && evidence.row(slots, mate) == row && self.set(mate, Role::Furniture, Rule::RowMate) {
                    self.edge[mate.index()] = Some(edge);
                    self.band[mate.index()] = false;
                }
            }
        }
        if fenced {
            self.read_continuations(pages, slots);
        }
    }
}
