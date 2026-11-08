//! Diagnostics of a parse: text-flow faults, its status, and the checks every page and component
//! passes.

use legal_pdf_constituents as constituents;
use legal_pdf_core::model::{Diagnostic, Line, Page};
use legal_pdf_core::{Error, Result};
use legal_pdf_furniture::printed_label;
use legal_pdf_structure_model::TRANSITION_ORIGIN;
use legal_pdf_text::text::{hyphen_continuation, hyphen_fragment_tail};
use legal_structure_model::{utf16_len, DocumentStructure, NodeKind, ResolutionRuleV2, ScalarRange};
use std::collections::{HashMap, HashSet};

const HARD_DIAGNOSTICS: &[&str] = &["COLUMN_ORDER_UNCERTAIN", "FOOTNOTE_UNMATCHED_LABEL", "FOOTNOTE_UNMATCHED_REFERENCE", "FOOTNOTE_REGION_UNCERTAIN", "TEXT_QUALITY_LOW"];

fn invalid<T>(message: impl Into<String>) -> Result<T> {
    Err(Error::Message(message.into()))
}

pub fn line_family(line: &Line) -> &'static str {
    match line.region_type.as_str() {
        "footnote" => "note",
        "heading" => "heading",
        _ => "body",
    }
}

pub fn text_flow_faults(pages: &[Page]) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    for page in pages {
        let eligible = page.lines.iter().filter(|line| !matches!(line.region_type.as_str(), "header" | "footer") && !line.exclude_from_body).collect::<Vec<_>>();
        for pair in eligible.windows(2) {
            let (previous, current) = (pair[0], pair[1]);
            if !hyphen_fragment_tail(&previous.text) {
                continue;
            }
            let (continues, same) = (hyphen_continuation(&current.text), line_family(previous) == line_family(current));
            if continues && !same {
                let message = format!(
                    "A hyphenated word spans the {}/{} boundary; either a region label or the order is wrong.",
                    line_family(previous),
                    line_family(current)
                );
                let mut diagnostic = Diagnostic::warning("REGION_BOUNDARY_FAULT", message, Some(page.index));
                diagnostic.line_ids = vec![previous.id.clone(), current.id.clone()];
                diagnostics.push(diagnostic);
            } else if !continues && same {
                let message = "A line ends mid-word but the next eligible line does not continue it.";
                let mut diagnostic = Diagnostic::info("DANGLING_SOFT_HYPHEN", message, Some(page.index));
                diagnostic.line_ids.push(previous.id.clone());
                diagnostics.push(diagnostic);
            }
        }
    }
    diagnostics
}

pub fn status(diagnostics: &[Diagnostic], pages: &[Page]) -> String {
    let ocr = |diagnostic: &&Diagnostic| diagnostic.code == "OCR_REQUIRED";
    let ocr_pages: HashSet<Option<usize>> = diagnostics.iter().filter(ocr).map(|diagnostic| diagnostic.page_index).collect();
    if !pages.is_empty() && ocr_pages.len() == pages.len() {
        "ocr_required"
    } else if diagnostics.iter().any(|diagnostic| {
        matches!(diagnostic.severity.as_str(), "warning" | "error") && (ocr(&diagnostic) || HARD_DIAGNOSTICS.contains(&diagnostic.code.as_str()))
    }) {
        "degraded"
    } else {
        "ready"
    }
    .to_owned()
}

pub fn validate_page_records(pages: &[Page]) -> Result<HashSet<&str>> {
    let (mut ids, mut span_ids, mut word_ids) = (HashSet::new(), HashSet::new(), HashSet::new());
    for page in pages {
        if page.lines.iter().any(|line| !ids.insert(line.id.as_str())) {
            return invalid("document contains duplicate line IDs");
        }
        let mut regions_by_line = HashMap::new();
        let duplicate_region_line =
            page.regions.iter().any(|region| region.line_ids.iter().any(|line| regions_by_line.insert(line.as_str(), region).is_some()));
        if duplicate_region_line
            || regions_by_line.len() != page.lines.len()
            || page.lines.iter().any(|line| !regions_by_line.contains_key(line.id.as_str()))
        {
            return invalid(format!("page {} region coverage is incomplete", page.number));
        }
        for line in &page.lines {
            if !line.spans.iter().all(|span| span_ids.insert(span.id.as_str())) {
                return invalid("document contains duplicate span IDs");
            }
            if !line.words.iter().all(|word| word_ids.insert(word.id.as_str())) {
                return invalid("document contains duplicate word IDs");
            }
            let Some(region) = regions_by_line.get(line.id.as_str()) else {
                return invalid(format!("line {} has no containing region", line.id));
            };
            if line.region_id != region.id || line.region_type != region.kind {
                return invalid(format!("page {} line/region annotations disagree for {}", page.number, line.id));
            }
            // Each character's byte offset, and the text's end.
            let boundaries = line.text.char_indices().map(|(index, _)| index).chain([line.text.len()]).collect::<Vec<_>>();
            let mut prior_end = 0;
            for word in &line.words {
                let text = boundaries.get(word.start).zip(boundaries.get(word.end)).and_then(|(&start, &end)| line.text.get(start..end));
                if word.start < prior_end || word.end <= word.start || word.end > boundaries.len() - 1 || text != Some(word.text.as_str()) {
                    return invalid(format!("line {} contains invalid word geometry", line.id));
                }
                prior_end = word.end;
            }
        }
        match (page.printed_label.as_deref(), page.printed_label_source.as_deref(), page.printed_label_line_id.as_deref()) {
            (None, None, None) => {}
            (Some(label), Some(source), Some(line_id)) => {
                let valid = page.lines.iter().any(|line| {
                    line.id == line_id && line.region_type == source && matches!(source, "header" | "footer") && printed_label(line.text.trim()).as_deref() == Some(label)
                });
                if !valid {
                    return invalid(format!("page {} has invalid printed-label provenance", page.number));
                }
            }
            _ => return invalid(format!("page {} has incomplete printed-label provenance", page.number)),
        }
    }
    Ok(ids)
}

pub fn validate_pdf_components(
    document_id: &str,
    source_sha256: &str,
    pages: &[Page],
    structure_graph: &DocumentStructure,
) -> Result<()> {
    let known_lines = validate_page_records(pages)?;
    let known_pages = pages.iter().map(|page| page.index).collect::<HashSet<_>>();
    let unknown_line = |line: &String| !known_lines.contains(line.as_str());
    let unknown_page = |page: &usize| !known_pages.contains(page);
    if structure_graph.document_id != document_id || structure_graph.source_sha256.as_deref() != Some(source_sha256) {
        return invalid("structure graph identity disagrees with the document");
    }
    // The structure's ranges stand in its text.
    let text_length = utf16_len(&structure_graph.text);
    let mut node_ids = HashSet::new();
    for node in &structure_graph.nodes {
        let grammar = node.grammar.as_deref();
        let unproven_section = node.kind == NodeKind::Section
            && grammar != Some("heading_section")
            && !grammar.is_some_and(|grammar| grammar.starts_with(constituents::GRAMMAR))
            && (node.locator_kind.as_deref().is_none_or(str::is_empty)
                || (node.origin_id != TRANSITION_ORIGIN && node.proof.as_ref().is_none_or(|proof| proof.rule != ResolutionRuleV2::HierarchySection)));
        if node.id.is_empty()
            || !node_ids.insert(node.id.as_str())
            || node.range.start > node.range.end
            || node.range.end > text_length
            || node.page_indexes.iter().any(unknown_page)
            || node.line_ids.iter().any(unknown_line)
            || unproven_section
        {
            return invalid(format!("structure node {} is invalid", node.id));
        }
    }
    if structure_graph.nodes.iter().any(|node| node.parent_id.as_deref().is_some_and(|parent| !node_ids.contains(parent))) {
        return invalid("structure graph contains an unknown parent");
    }
    let valid_range = |range: &ScalarRange| range.start <= range.end && range.end <= text_length;
    for note in structure_graph.nodes.iter().filter(|node| matches!(node.kind, legal_structure_model::NodeKind::Footnote | legal_structure_model::NodeKind::Endnote)) {
        if !note.marker_range.iter().chain(&note.calls).all(valid_range) {
            return invalid(format!("structure note {} is invalid", note.id));
        }
    }
    Ok(())
}
