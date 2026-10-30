//! Pair within logical documents, then retain one PDF-wide identity for each materialized note.

use super::{Decided, DocumentContext, PdfTextIndex};
use legal_pdf_core::model::{Line, Page, PairingOutput, PdfSourceSpan};
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
        regions: Vec::new(),
        source: page.source.clone(),
        text_quality: page.text_quality,
        printed_label: page.printed_label.clone(),
        printed_label_source: page.printed_label_source.clone(),
        printed_label_line_id: page.printed_label_line_id.clone(),
    }
}

/// The document's notes paired on the pages as the decision wrote them (their regions and note
/// modes), each constituent document and language on its own lines; the pages' lines are left as
/// they stand.
pub(super) fn pair_notes(
    pages: &mut [Page],
    decided: &Decided,
    citations: &BTreeMap<String, Vec<PdfSourceSpan>>,
    index: &PdfTextIndex,
) -> PairingOutput {
    let analysis = legal_pdf_core::layout_analysis();
    let contexts: Vec<Vec<DocumentContext>> = pages
        .iter()
        .map(|page| page.lines.iter().map(|line| decided.document_context(&line.id)).collect())
        .collect();
    let first = contexts.iter().flatten().next().copied();
    if contexts.iter().flatten().all(|context| Some(*context) == first) {
        return analysis.pair_footnotes(pages, citations, false);
    }
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
    let mut result = PairingOutput {
        footnotes: Vec::new(),
        diagnostics: Vec::new(),
        anchors: HashMap::new(),
        pair_claims: Vec::new(),
        pairing_audit: None,
    };
    let mut sequence_offset = 0;
    for document in readings.values() {
        let mut part = analysis.pair_footnotes(document, citations, false);
        let sequences = part
            .footnotes
            .iter()
            .map(|note| note.restart_sequence)
            .max()
            .unwrap_or(0);
        let mut ids = HashMap::new();
        for (slot, note) in part.footnotes.iter_mut().enumerate() {
            let id = format!(
                "fnv2-pair-LEGALPDF-document-{:06}",
                result.footnotes.len() + slot + 1
            );
            ids.insert(std::mem::replace(&mut note.pair_id, id.clone()), id);
            note.restart_sequence += sequence_offset;
        }
        for anchors in part.anchors.values_mut() {
            for anchor in anchors {
                anchor.pair_id = ids[&anchor.pair_id].clone();
            }
        }
        for claim in &mut part.pair_claims {
            claim.pair_id = ids[&claim.pair_id].clone();
            claim.support.sequence_id += sequence_offset;
        }
        for diagnostic in &mut part.diagnostics {
            if let Some(id) = diagnostic.details.get_mut("pair_id") {
                if let Some(mapped) = id.as_str().and_then(|id| ids.get(id)) {
                    *id = serde_json::Value::String(mapped.clone());
                }
            }
        }
        sequence_offset += sequences;
        result.footnotes.extend(part.footnotes);
        result.diagnostics.extend(part.diagnostics);
        result.anchors.extend(part.anchors);
        result.pair_claims.extend(part.pair_claims);
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
    // Public display occurrences follow physical reading order, independently of inference scope.
    result.footnotes.sort_by_key(|note| {
        note.body_line_ids
            .first()
            .and_then(|id| index.line(id))
            .map_or(usize::MAX, |line| line.range.start)
    });
    result.pair_claims.sort_by_key(|claim| {
        index
            .line(&claim.label_anchor.line_id)
            .map_or((usize::MAX, 0), |line| {
                (line.range.start, claim.label_anchor.start)
            })
    });
    let mut occurrences = HashMap::<String, usize>::new();
    for note in &mut result.footnotes {
        let occurrence = occurrences.entry(note.label.clone()).or_default();
        *occurrence += 1;
        note.occurrence = *occurrence;
    }
    result
}
