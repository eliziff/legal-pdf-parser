//! How each line goes on with the line before it, and the statute's running heads its spine bears out.
use crate::model::fences::{Continuation, Fences};
use crate::rules::headings::native::{body_flow_edge, carries_a_sentence_onto, continues_prose_from};
use crate::model::{Evidence, Role, Rule, Slot, Slots};
use crate::rules::notes::lines::reference_wrapped_onto;
use crate::text::paragraphs::continues_onto;
use legal_pdf_core::model::Page;

/// How each line no fence took goes on with the line before it, in its reading context and
/// stream, in reading order: on one page by any continuation, across a page by the text's flow or
/// as the two pages set the stream's lines (`continues_onto`).
/// How each line no fence took goes on with the line before it, in its reading context and
/// stream, in reading order: on one page by any continuation, across a page by the text's flow or
/// as the two pages set the stream's lines (`continues_onto`).
pub fn read_continuations(fences: &mut Fences, pages: &[Page], slots: &Slots) {
    // Each stream's lines, page by page.
    let mut streams: Vec<Vec<Vec<Slot>>> = vec![Vec::new(); fences.contexts.len() * 2];
    for &slot in fences
        .order
        .iter()
        .filter(|slot| fences.role[slot.index()] == Role::Open)
    {
        let runs = &mut streams
            [usize::from(fences.ctx[slot.index()]) * 2 + usize::from(fences.band[slot.index()])];
        match runs.last_mut() {
            Some(run) if slots.page(run[0]) == slots.page(slot) => run.push(slot),
            _ => runs.push(vec![slot]),
        }
    }
    let mut continues = vec![None; slots.len()];
    let read = |run: &[Slot]| {
        run.iter()
            .map(|&slot| slots.line(pages, slot))
            .collect::<Vec<_>>()
    };
    for runs in &streams {
        for (k, run) in runs.iter().enumerate() {
            for (j, &slot) in run.iter().enumerate() {
                let Some(prior) = (if j > 0 {
                    Some(run[j - 1])
                } else {
                    k.checked_sub(1).map(|k| runs[k][runs[k].len() - 1])
                }) else {
                    continue;
                };
                let (line, before) = (slots.line(pages, slot), slots.line(pages, prior));
                continues[slot.index()] = if body_flow_edge(before, line) {
                    Some(Continuation::Flow)
                } else if j == 0 {
                    continues_onto(&read(&runs[k - 1]), &read(run))
                        .then_some(Continuation::Page)
                } else {
                    let (page, start, band) = (
                        &pages[slots.page(slot)],
                        slots.on(slots.page(slot)).start as usize,
                        fences.band[slot.index()],
                    );
                    let (at, prior_at) = (slot.index() - start, prior.index() - start);
                    let body = |other: usize| {
                        fences.role[start + other] == Role::Open
                            && fences.band[start + other] == band
                    };
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
    }
    fences.continues = continues;
}

/// The statute's running heads naming their page's sections, where its spine bears them out
/// (`spine`: a section numbered so opens on that page slot), on their page or the one before;
/// and the lines on their rows. The lines' continuations are read again without them.
/// The statute's running heads naming their page's sections, where its spine bears them out
/// (`spine`: a section numbered so opens on that page slot), on their page or the one before;
/// and the lines on their rows. The lines' continuations are read again without them.
pub fn confirm_section_heads(fences: &mut Fences, pages: &[Page],
    slots: &Slots,
    evidence: &Evidence,
    spine: impl Fn(usize, u32) -> bool,
) {
    let mut fenced = false;
    for (slot, edge, number) in std::mem::take(&mut fences.section_heads) {
        let page = slots.page(slot);
        if !(spine(page, number) || page > 0 && spine(page - 1, number))
            || !crate::find::set(fences, slot, Role::Furniture, Rule::SectionHead)
        {
            continue;
        }
        fenced = true;
        fences.edge[slot.index()] = Some(edge);
        fences.band[slot.index()] = false;
        let row = evidence.row(slots, slot);
        for mate in slots.on(page).map(Slot) {
            if row != u32::MAX
                && evidence.row(slots, mate) == row
                && crate::find::set(fences, mate, Role::Furniture, Rule::RowMate)
            {
                fences.edge[mate.index()] = Some(edge);
                fences.band[mate.index()] = false;
            }
        }
    }
    if fenced {
        crate::find::read_continuations(fences, pages, slots);
    }
}

