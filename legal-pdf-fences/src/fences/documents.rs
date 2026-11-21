//! Each line's constituent document, and which documents are transcripts.
use crate::model::fences::Fences;
use crate::model::{Evidence, PageView, Role, Slot, Slots, WORD_SPEAKER};
use legal_pdf_core::model::PdfOutlineEntry;

/// Each line's constituent document: the openings the document's bookmarks name, and those its
/// own lines show, read on the lines no fence took and the note band left.
/// Each line's constituent document: the openings the document's bookmarks name, and those its
/// own lines show, read on the lines no fence took and the note band left.
pub fn read_documents(fences: &mut Fences, views: &[PageView], outline: &[PdfOutlineEntry]) {
    let (role, band) = (&fences.role, &fences.band);
    let set_aside = |slot: Slot| role[slot.index()] != Role::Open || band[slot.index()];
    let mut starts = crate::constituents::bookmarked_starts(views, outline, set_aside);
    crate::constituents::reconcile_starts(views, &mut starts, set_aside);
    let documents = crate::constituents::documents(views, &starts, |slot| {
        matches!(
            role[slot.index()],
            Role::LineLabel | Role::Furniture | Role::PageLabel
        )
    });
    for (slot, document) in documents {
        crate::find::recontext(fences, slot, |context| {
            context.document = document.min(usize::from(u16::MAX)) as u16
        });
    }
    fences.starts = starts;
}

/// Which constituent documents are transcripts (`transcripts`) and which of them are testimony
/// throughout (`testimony`), and each transcript's lines beside its number column
/// (`numbered_row`).
/// Which constituent documents are transcripts (`transcripts`) and which of them are testimony
/// throughout (`testimony`), and each transcript's lines beside its number column
/// (`numbered_row`).
pub fn read_transcripts(fences: &mut Fences, slots: &Slots, evidence: &Evidence) {
    let count = slots.len();
    let documents = fences.documents();
    let (mut numbered, mut speakers, mut lines) = (
        vec![0usize; documents],
        vec![0usize; documents],
        vec![0usize; documents],
    );
    // Each document's pages, and those printing its line numbers: (count, last page met).
    let (mut pages, mut numbered_pages) = (
        vec![(0usize, usize::MAX); documents],
        vec![(0usize, usize::MAX); documents],
    );
    let meet = |met: &mut (usize, usize), page: usize| {
        if met.1 != page {
            *met = (met.0 + 1, page);
        }
    };
    for slot in 0..count {
        let document = usize::from(fences.at(slot).document);
        let page = slots.page(Slot(slot as u32));
        meet(&mut pages[document], page);
        if evidence.line_number[slot] {
            numbered[document] += 1;
            meet(&mut numbered_pages[document], page);
        } else {
            lines[document] += 1;
            speakers[document] += usize::from(evidence.words[slot] & WORD_SPEAKER != 0);
        }
    }
    let spoken = |document: usize| {
        speakers[document] >= 20 && speakers[document] * 10 >= lines[document]
    };
    fences.transcripts = (0..documents)
        .map(|document| numbered[document] > 0 || spoken(document))
        .collect();
    // A contents page's locators or a table's row numbers read as line numbers make a document a
    // transcript, not its testimony.
    fences.testimony = (0..documents)
        .map(|document| {
            spoken(document) || numbered_pages[document].0 * 2 >= pages[document].0.max(1)
        })
        .collect();
    fences.numbered_row = vec![false; count];
    for page_slot in 0..evidence.pages.len() {
        let range = slots.on(page_slot);
        let rows = &evidence.pages[page_slot].rows;
        let numbered: Vec<u32> = range
            .clone()
            .filter(|&slot| fences.role[slot as usize] == Role::LineLabel)
            .map(|slot| rows.of[(slot - range.start) as usize])
            .filter(|row| *row != u32::MAX)
            .collect();
        if numbered.is_empty() {
            continue;
        }
        for slot in range.clone() {
            let row = rows.of[(slot - range.start) as usize];
            fences.numbered_row[slot as usize] =
                row != u32::MAX && numbered.contains(&row) && fences.transcript(Slot(slot));
        }
    }
}

