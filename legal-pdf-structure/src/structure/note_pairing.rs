//! Pair within logical documents, then retain one PDF-wide identity for each materialized note.

use super::{constituents, PdfPrimitiveEvidence, PdfTextIndex};
use legal_pdf_core::model::{Page, PairingOutput, PdfSourceSpan};
use std::collections::{BTreeMap, HashMap};

pub(super) fn pair_notes(
    pages: &mut [Page],
    primitives: &PdfPrimitiveEvidence,
    citations: &BTreeMap<String, Vec<PdfSourceSpan>>,
    index: &PdfTextIndex,
) -> PairingOutput {
    let analysis = legal_pdf_core::layout_analysis();
    if primitives.document_line_ids.is_empty() && primitives.translation_line_ids.is_empty() {
        return analysis.pair_footnotes(pages, citations, false);
    }
    let mut result = PairingOutput {
        footnotes: Vec::new(),
        diagnostics: Vec::new(),
        anchors: HashMap::new(),
        pair_claims: Vec::new(),
        pairing_audit: None,
    };
    let mut sequence_offset = 0;
    constituents::within_document_contexts(
        pages,
        &primitives.document_line_ids,
        &primitives.translation_line_ids,
        |document| {
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
        },
    );
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
