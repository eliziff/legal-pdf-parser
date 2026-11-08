//! Pair within logical documents, then retain one PDF-wide identity for each materialized note.

use legal_pdf_structure_model::{Decided, DocumentContext};
use legal_pdf_text::text_index::PdfTextIndex;
use legal_pdf_core::model::{Line, Page, ScalarRange};
use legal_structure_model::NotePairing;
use std::collections::{BTreeMap, HashMap};

/// A page holding none of its lines yet.
fn without_lines(page: &Page) -> Page {
    Page {
        id: page.id.clone(),
        index: page.index,
        number: page.number,
        width: page.width,
        height: page.height,
        lines: Vec::new(),
        source: page.source.clone(),
        text_quality: page.text_quality,
        printed_label: page.printed_label.clone(),
        printed_label_source: page.printed_label_source.clone(),
        printed_label_line_id: page.printed_label_line_id.clone(),
        ..Default::default()
    }
}

/// The document's notes paired on the pages as the decision wrote them (their regions and note modes), each constituent
/// document and language on its own lines, each note a node placed in the document's text; the pages' lines are left as
/// they stand. A note's id is the document's, and its numbering's restarts count on across the documents.
pub fn pair_notes(pages: &mut [Page],
    decided: &Decided,
    citations: &BTreeMap<String, Vec<ScalarRange>>,
    index: &PdfTextIndex,
) -> NotePairing {
    let place = |id: &str, start: usize, end: usize| index.global_range(id, start, end);
    let contexts: Vec<Vec<DocumentContext>> = pages
        .iter()
        .map(|page| page.lines.iter().map(|line| decided.document_context(&line.id)).collect())
        .collect();
    let first = contexts.iter().flatten().next().copied();
    let mut pairing = if contexts.iter().flatten().all(|context| Some(*context) == first) {
        legal_structure_model::structure_analysis().pair_footnotes(pages, citations, false, &place)
    } else {
        // Each document and language's lines, moved into pages of their own and back after.
        let mut readings = BTreeMap::<DocumentContext, Vec<Page>>::new();
        let mut ordered: Vec<usize> = (0..pages.len()).collect();
        ordered.sort_by_key(|&slot| pages[slot].index);
        for &page_slot in &ordered {
            let page = &mut pages[page_slot];
            let lines = std::mem::take(&mut page.lines);
            for (line, context) in lines.into_iter().zip(&contexts[page_slot]) {
                let reading = readings.entry(*context).or_default();
                if reading.last().is_none_or(|last| last.index != page.index) {
                    reading.push(without_lines(page));
                }
                reading.last_mut().expect("a page for the line").lines.push(line);
            }
        }
        let mut result = NotePairing { notes: Vec::new(), diagnostics: Vec::new(), audit: None };
        let mut sequence_offset = 0;
        for document in readings.values() {
            let mut part = legal_structure_model::structure_analysis().pair_footnotes(document, citations, false, &place);
            let mut sequences = 0;
            for (slot, note) in part.notes.iter_mut().enumerate() {
                let id = format!("fnv2-pair-LEGALPDF-document-{:06}", result.notes.len() + slot + 1);
                note.id = id;
                let sequence = note.series.parse::<usize>().unwrap_or(0);
                sequences = sequences.max(sequence);
                note.series = (sequence + sequence_offset).to_string();
            }
            sequence_offset += sequences;
            result.notes.extend(part.notes);
            result.diagnostics.extend(part.diagnostics);
        }
        // Each page's lines back in their places, each context's in the order they were taken.
        let mut taken: BTreeMap<DocumentContext, std::vec::IntoIter<Line>> = readings
            .into_iter()
            .map(|(context, pages)| (context, pages.into_iter().flat_map(|page| page.lines).collect::<Vec<_>>().into_iter()))
            .collect();
        for &page_slot in &ordered {
            pages[page_slot].lines = contexts[page_slot]
                .iter()
                .map(|context| taken.get_mut(context).and_then(Iterator::next).expect("the context's line"))
                .collect();
        }
        result
    };
    // The notes in the order they are printed: by where each label stands.
    pairing.notes.sort_by_key(|note| note.marker_range.map_or((usize::MAX, 0), |label| (label.start, label.end)));
    pairing
}
