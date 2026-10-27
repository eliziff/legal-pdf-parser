//! Diagnostics of a parse: text-flow faults, its status, and the checks every page and component
//! passes.

use super::*;

pub(super) fn line_family(line: &Line) -> &'static str {
    match line.region_type.as_str() {
        "footnote" => "note",
        "heading" => "heading",
        _ => "body",
    }
}

pub(super) fn hyphen_fragment_tail(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[A-Za-z]{2,}[-\u{00ac}\u{00ad}]$").unwrap())
        .is_match(text.trim_end())
}

pub(super) fn hyphen_continuation(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[A-Za-z]{2,}").unwrap())
        .is_match(text.trim_start())
}

pub(super) fn text_flow_faults(pages: &[Page]) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    for page in pages {
        let mut eligible = page.lines.iter().filter(|line| {
            !matches!(line.region_type.as_str(), "header" | "footer") && !line.exclude_from_body
        });
        let Some(mut previous) = eligible.next() else {
            continue;
        };
        for current in eligible {
            if hyphen_fragment_tail(&previous.text) {
                let continues = hyphen_continuation(&current.text);
                if continues && line_family(previous) != line_family(current) {
                    let mut diagnostic = Diagnostic::warning(
                        "REGION_BOUNDARY_FAULT",
                        format!(
                            "A hyphenated word spans the {}/{} boundary; either a region label or the order is wrong.",
                            line_family(previous),
                            line_family(current)
                        ),
                        Some(page.index),
                    );
                    diagnostic.line_ids = vec![previous.id.clone(), current.id.clone()];
                    diagnostics.push(diagnostic);
                } else if !continues && line_family(previous) == line_family(current) {
                    let mut diagnostic = Diagnostic::info(
                        "DANGLING_SOFT_HYPHEN",
                        "A line ends mid-word but the next eligible line does not continue it.",
                        Some(page.index),
                    );
                    diagnostic.line_ids.push(previous.id.clone());
                    diagnostics.push(diagnostic);
                }
            }
            previous = current;
        }
    }
    diagnostics
}

pub fn status(diagnostics: &[Diagnostic], pages: &[Page]) -> String {
    let ocr_pages: HashSet<Option<usize>> = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code == "OCR_REQUIRED")
        .map(|diagnostic| diagnostic.page_index)
        .collect();
    if !pages.is_empty() && ocr_pages.len() == pages.len() {
        return "ocr_required".to_owned();
    }
    if diagnostics.iter().any(|diagnostic| {
        matches!(diagnostic.severity.as_str(), "warning" | "error")
            && (diagnostic.code == "OCR_REQUIRED"
                || HARD_DIAGNOSTICS.contains(&diagnostic.code.as_str()))
    }) {
        "degraded".to_owned()
    } else {
        "ready".to_owned()
    }
}

pub(super) fn validate_page_records(pages: &[Page]) -> Result<HashSet<&str>> {
    let mut ids = HashSet::new();
    let mut span_ids = HashSet::new();
    let mut word_ids = HashSet::new();
    for page in pages {
        if page.lines.iter().any(|line| !ids.insert(line.id.as_str())) {
            return Err(Error::Message(
                "document contains duplicate line IDs".to_owned(),
            ));
        }
        let mut regions_by_line = HashMap::new();
        let duplicate_region_line = page.regions.iter().any(|region| {
            region
                .line_ids
                .iter()
                .any(|line| regions_by_line.insert(line.as_str(), region).is_some())
        });
        if duplicate_region_line
            || regions_by_line.len() != page.lines.len()
            || page
                .lines
                .iter()
                .any(|line| !regions_by_line.contains_key(line.id.as_str()))
        {
            return Err(Error::Message(format!(
                "page {} region coverage is incomplete",
                page.number
            )));
        }
        for line in &page.lines {
            if !line
                .spans
                .iter()
                .all(|span| span_ids.insert(span.id.as_str()))
            {
                return Err(Error::Message(
                    "document contains duplicate span IDs".to_owned(),
                ));
            }
            if !line
                .words
                .iter()
                .all(|word| word_ids.insert(word.id.as_str()))
            {
                return Err(Error::Message(
                    "document contains duplicate word IDs".to_owned(),
                ));
            }
            let region = regions_by_line
                .get(line.id.as_str())
                .copied()
                .ok_or_else(|| {
                    Error::Message(format!("line {} has no containing region", line.id))
                })?;
            if line.region_id != region.id || line.region_type != region.kind {
                return Err(Error::Message(format!(
                    "page {} line/region annotations disagree for {}",
                    page.number, line.id
                )));
            }
            let mut prior_end = 0;
            let scalar_boundaries = (!line.text.is_ascii()).then(|| {
                line.text
                    .char_indices()
                    .map(|(index, _)| index)
                    .chain(std::iter::once(line.text.len()))
                    .collect::<Vec<_>>()
            });
            let scalar_len = scalar_boundaries
                .as_ref()
                .map_or(line.text.len(), |boundaries| boundaries.len() - 1);
            for word in &line.words {
                let text = scalar_boundaries.as_ref().map_or_else(
                    || line.text.get(word.start..word.end),
                    |boundaries| {
                        boundaries
                            .get(word.start)
                            .zip(boundaries.get(word.end))
                            .and_then(|(&start, &end)| line.text.get(start..end))
                    },
                );
                if word.start < prior_end
                    || word.end <= word.start
                    || word.end > scalar_len
                    || text != Some(word.text.as_str())
                {
                    return Err(Error::Message(format!(
                        "line {} contains invalid word geometry",
                        line.id
                    )));
                }
                prior_end = word.end;
            }
        }
        let printed = (
            page.printed_label.as_deref(),
            page.printed_label_source.as_deref(),
            page.printed_label_line_id.as_deref(),
        );
        if printed.0.is_some() || printed.1.is_some() || printed.2.is_some() {
            let (Some(label), Some(source), Some(line_id)) = printed else {
                return Err(Error::Message(format!(
                    "page {} has incomplete printed-label provenance",
                    page.number
                )));
            };
            let valid = page.lines.iter().any(|line| {
                line.id == line_id
                    && line.region_type == source
                    && matches!(source, "header" | "footer")
                    && printed_label(line.text.trim()).as_deref() == Some(label)
            });
            if !valid {
                return Err(Error::Message(format!(
                    "page {} has invalid printed-label provenance",
                    page.number
                )));
            }
        }
    }
    Ok(ids)
}

pub fn validate_document(document: &LegalDocument) -> Result<()> {
    if document.page_count != document.pages.len() {
        return Err(Error::Message(
            "document page_count does not match the page collection".to_owned(),
        ));
    }
    validate_pdf_components(
        &document.document_id,
        &document.source_sha256,
        &document.pages,
        &document.paragraphs,
        &document.footnotes,
        &document.structure_graph,
    )
}

pub fn validate_pdf_components(
    document_id: &str,
    source_sha256: &str,
    pages: &[Page],
    paragraphs: &[Paragraph],
    footnotes: &[Footnote],
    structure_graph: &DocumentStructure,
) -> Result<()> {
    let known_lines = validate_page_records(pages)?;
    let known_pages = pages.iter().map(|page| page.index).collect::<HashSet<_>>();
    let mut pair_ids = HashSet::new();
    for footnote in footnotes {
        if !pair_ids.insert(&footnote.pair_id) {
            return Err(Error::Message(
                "document contains duplicate footnote pair IDs".to_owned(),
            ));
        }
        if footnote
            .reference_line_id
            .as_ref()
            .is_some_and(|line| !known_lines.contains(line.as_str()))
            || footnote
                .body_line_ids
                .iter()
                .any(|line| !known_lines.contains(line.as_str()))
        {
            return Err(Error::Message(format!(
                "footnote {} contains an unknown source line",
                footnote.pair_id
            )));
        }
    }
    let mut paragraph_ids = HashSet::new();
    for paragraph in paragraphs {
        if !paragraph_ids.insert(&paragraph.id)
            || paragraph
                .line_ids
                .iter()
                .any(|line| !known_lines.contains(line.as_str()))
        {
            return Err(Error::Message(format!(
                "paragraph {} is invalid",
                paragraph.id
            )));
        }
    }
    if structure_graph.document_id != document_id
        || structure_graph.source_sha256.as_deref() != Some(source_sha256)
    {
        return Err(Error::Message(
            "structure graph identity disagrees with the document".to_owned(),
        ));
    }
    let (text_length, line_count) = pages
        .iter()
        .flat_map(|page| &page.lines)
        .fold((0, 0_usize), |(units, count), line| {
            (units + utf16_len(&line.text), count + 1)
        });
    let text_length = text_length + line_count.saturating_sub(1);
    let mut node_ids = HashSet::new();
    for node in &structure_graph.nodes {
        if node.id.is_empty()
            || !node_ids.insert(node.id.as_str())
            || node.range.start > node.range.end
            || node.range.end > text_length
            || node
                .page_indexes
                .iter()
                .any(|page| !known_pages.contains(page))
            || node
                .line_ids
                .iter()
                .any(|line| !known_lines.contains(line.as_str()))
            || (node.kind == NodeKind::Section
                && node.grammar.as_deref() != Some("heading_section")
                && !node
                    .grammar
                    .as_deref()
                    .is_some_and(|grammar| grammar.starts_with(constituents::GRAMMAR))
                && (node.locator_kind.as_deref().is_none_or(str::is_empty)
                    || node
                        .proof
                        .as_ref()
                        .is_none_or(|proof| proof.rule != ResolutionRuleV2::HierarchySection)))
        {
            return Err(Error::Message(format!(
                "structure node {} is invalid",
                node.id
            )));
        }
    }
    if structure_graph.nodes.iter().any(|node| {
        node.parent_id
            .as_deref()
            .is_some_and(|parent| !node_ids.contains(parent))
    }) {
        return Err(Error::Message(
            "structure graph contains an unknown parent".to_owned(),
        ));
    }
    for note in &structure_graph.notes {
        let valid_range = |range: ScalarRange| range.start <= range.end && range.end <= text_length;
        if note.node_id.as_deref().is_some_and(|id| !node_ids.contains(id))
            || !valid_range(note.label_range)
            || !valid_range(note.body_range)
            || note.label_ranges.iter().chain(&note.body_ranges).any(|range| !valid_range(*range))
            || note
                .primary_reference
                .is_some_and(|range| !valid_range(range))
            || note.references.iter().any(|reference| {
                !valid_range(reference.range)
                    || reference
                        .page_indexes
                        .iter()
                        .any(|page| !known_pages.contains(page))
                    || reference
                        .line_ids
                        .iter()
                        .any(|line| !known_lines.contains(line.as_str()))
            })
        {
            return Err(Error::Message(format!(
                "structure note {} is invalid",
                note.id
            )));
        }
    }
    Ok(())
}
