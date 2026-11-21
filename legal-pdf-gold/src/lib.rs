//! What the structure gold, its stores and the corpus tools ask of a parse, over the parser's own path (the extraction
//! and recognition caches, the structure stage): no engine links this, and an edit here recompiles no parser.
#![allow(clippy::too_many_arguments)]

use legal_pdf_core::read_gzip_json;
use legalpdf::{corpus_structure, derive_document, kept_extraction, prepared_path, CachedExtraction,
    ExtractedPdf, PdfDocument, Result, Error, EXTRACTION_CACHE_SCHEMA};
#[cfg(feature = "ocr")]
use legalpdf::{recognized_extraction, OcrOptions, OcrProvider, PdfRequest, Providers};
#[cfg(feature = "ppdoc")]
use legalpdf::{PPDocLayout, PPDocOptions};
use legal_pdf_structure::StructureOutput;
use legal_structure_model::{serialization_sha256, DocumentStructure, ScalarText};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// The structure gold's inputs: `pdf`'s pages exactly as a parse of this build feeds its structure
/// stage (`recognized_extraction`), cached under `cache_dir` as a parse caches them. Recognition is
/// Indra's where LEGALPDF_INDRA_MODEL_PACK and LEGALPDF_ONNXRUNTIME name it, through the product's
/// recognition cache (a page read before by the same OCR code is not read again); the layout model
/// PP-DocLayout's where LEGALPDF_PPDOC_MODEL_PACK names a pack. The gold is aligned to whatever
/// lines these are, so nothing here is chosen by the gold.
#[cfg(feature = "ocr")]
pub fn gold_structure_inputs(pdf: &Path, source_hash: &str, cache_dir: &Path) -> Result<ExtractedPdf> {
    // Read back where kept: the PDF is read only to extract it.
    let identity = gold_inputs_identity();
    if let Some(kept) = kept_extraction(&prepared_path(cache_dir, &identity, source_hash)) {
        return Ok(kept);
    }
    let pdf = &fs::read(pdf).map_err(|source| Error::Io { path: pdf.to_owned(), source })?;
    let mut options = PdfRequest::default();
    options.cache_dir = Some(cache_dir.to_owned());
    // Always the recognizer (LEGALPDF_INDRA_MODEL_PACK, LEGALPDF_ONNXRUNTIME), else a failure: a scan read without one is empty pages.
    options.ocr = Some(OcrOptions::default());
    // The product's layout model (OpenVINO), on the same pack and runtime: LEGALPDF_PPDOC_DEVICE its device.
    #[cfg(feature = "ppdoc")]
    {
        options.ppdoc = std::env::var_os("LEGALPDF_PPDOC_MODEL_PACK").map(|pack| PPDocOptions {
            model_pack: Some(PathBuf::from(&pack)),
            runtime: std::env::var_os("LEGALPDF_OPENVINO_RUNTIME").map(PathBuf::from),
            threads: std::env::var("LEGALPDF_PPDOC_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(1),
            device: std::env::var("LEGALPDF_PPDOC_DEVICE").ok(),
            ..PPDocOptions::default()
        });
    }
    let ocr = options.ocr.as_ref().map(OcrProvider::prepare).transpose()?;
    let providers = Providers {
        ocr_identity: ocr.as_ref().map(|provider| (provider.name().to_owned(), provider.identity().to_owned())),
        ocr,
        #[cfg(feature = "ppdoc")]
        ppdoc: options.ppdoc.as_ref().map(PPDocLayout::new).transpose()?,
    };
    recognized_extraction(pdf, source_hash, &options, Some(cache_dir), providers, None, false, &identity)
}


/// The structure gold's parses resumed from their dump entries (`legal_pdf_structure::resume_many`): each source's pages
/// and outline from its extraction as kept under `cache_dir` (never the PDF), its entry's kept tier 0 laid over them.
/// Each source's structure; a failure where its extraction is not kept.
#[cfg(feature = "ocr")]
pub fn gold_structure_resume(entries: Vec<(String, legal_pdf_structure::Entry)>, cache_dir: &Path) -> Vec<Result<StructureOutput>> {
    let identity = gold_inputs_identity();
    let mut out: Vec<Option<Result<StructureOutput>>> = Vec::with_capacity(entries.len());
    let mut present = Vec::new();
    for (sha256, entry) in entries {
        match kept_extraction(&prepared_path(cache_dir, &identity, &sha256)) {
            Some(extracted) => {
                out.push(None);
                present.push((sha256, entry, extracted.pages, extracted.outline));
            }
            None => out.push(Some(Err(Error::Message(format!("{sha256}: its extraction is not kept"))))),
        }
    }
    let mut resumed = legal_pdf_structure::resume_many(present, None).into_iter();
    out.into_iter().map(|done| done.unwrap_or_else(|| resumed.next().unwrap_or_else(|| Err(Error::Message("a resumed parse went missing".to_owned()))))).collect()
}


/// What the structure gold's inputs are keyed by: the extraction schema and the models the
/// environment names. Work after extraction never re-extracts the gold; recognition is cached on
/// its own, and a change to it that should reach the inputs bumps the version here.
pub fn gold_inputs_identity() -> String {
    let models = ["LEGALPDF_INDRA_MODEL_PACK", "LEGALPDF_ONNXRUNTIME", "LEGALPDF_PPDOC_MODEL_PACK"].map(|name| std::env::var(name).ok());
    // v3: a scan's note labels recovered in extraction (paragraph_recovery's Family::Note).
    let identity = serialization_sha256(&("legalpdf.gold-inputs.v3", EXTRACTION_CACHE_SCHEMA, models));
    identity[..16.min(identity.len())].to_owned()
}

/// The structure gold's replay: a recognized extraction through the structure stage, uncached; its dump entry written to
/// `dump` where named (`legal_pdf_structure::Entry`).
pub fn gold_structure_document(extracted: ExtractedPdf, source_hash: &str, dump: Option<legal_pdf_structure::Keep<'_>>) -> Result<PdfDocument> {
    derive_document(extracted, source_hash, "structure-gold", None, None, dump)
}


/// Where the structure gold's inputs keep a source's extraction (`gold_structure_inputs`): under `cache_dir`, by the
/// gold inputs' identity. Its reader is `legal_pdf_core::read_gzip_packed`.
pub fn gold_inputs_file(cache_dir: &Path, source_hash: &str) -> PathBuf {
    prepared_path(cache_dir, &gold_inputs_identity(), source_hash)
}


fn structure_examples(graph: &DocumentStructure) -> Value {
    use legal_structure_model::NodeKind;
    let text = ScalarText::new(&graph.text);
    let mut examples = BTreeMap::<String, Vec<Value>>::new();
    for kind in [NodeKind::Heading, NodeKind::Section, NodeKind::Paragraph, NodeKind::ListItem, NodeKind::Footnote, NodeKind::Endnote] {
        let nodes = graph.nodes.iter().filter(|node| node.kind == kind).collect::<Vec<_>>();
        if nodes.is_empty() {
            continue;
        }
        let mut slots = vec![0, nodes.len() / 2, nodes.len() - 1];
        slots.dedup();
        let example = |slot: usize| {
            let node = nodes[slot];
            let words = text.slice_utf16(node.range.start..node.range.end).unwrap_or_default().split_whitespace().collect::<Vec<_>>().join(" ");
            json!({
                "id": node.id,
                "text": words.chars().take(240).collect::<String>(),
                "label": node.label,
                "locator_kind": node.locator_kind,
                "parent_id": node.parent_id,
                "page_indexes": node.page_indexes,
                "line_ids": node.line_ids,
            })
        };
        examples.insert(kind.name().to_owned(), slots.into_iter().map(example).collect());
    }
    json!(examples)
}


pub fn corpus_check_cached_extraction(input: impl AsRef<Path>) -> Result<Value> {
    let cached: CachedExtraction = read_gzip_json(input.as_ref())?;
    if cached.schema_version != EXTRACTION_CACHE_SCHEMA || cached.source_sha256.len() != 64 || cached.extraction.pages.len() != cached.extraction.separators.len() {
        return Err(Error::Message("invalid extraction cache".to_owned()));
    }
    let page_count = cached.extraction.pages.len();
    let line_count = cached.extraction.pages.iter().map(|page| page.lines.len()).sum::<usize>();
    let source_sha256 = cached.source_sha256.clone();
    let (extracted, graph) = corpus_structure(&source_sha256, cached.extraction)?;
    let graph = &graph;
    // Each document's nodes, a line each, for reading two engines' graphs side by side.
    if let Some(directory) = std::env::var_os("LEGALPDF_CORPUS_NODES") {
        let line = |node: &legal_structure_model::StructureNode| {
            format!("{} {} {:?} {:?}\n", serde_json::to_value(node.kind).unwrap_or_default(), node.label.as_deref().unwrap_or("_"), node.range, node.parent_id)
        };
        let lines = graph.nodes.iter().map(line).collect::<String>();
        std::fs::write(Path::new(&directory).join(format!("{}.nodes.txt", &source_sha256[..16])), lines).map_err(|error| Error::Message(error.to_string()))?;
    }
    let count = |names: &mut dyn Iterator<Item = String>| names.fold(BTreeMap::<String, usize>::new(), |mut counts, name| {
        *counts.entry(name).or_default() += 1;
        counts
    });
    let by_kind = count(&mut graph.nodes.iter().map(|node| node.kind.name().to_owned()));
    let sections = graph.nodes.iter().filter(|node| node.kind == legal_structure_model::NodeKind::Section);
    let sections_by_locator_kind = count(&mut sections.map(|node| node.locator_kind.clone().unwrap_or_else(|| "unknown".to_owned())));
    let heading_derived_sections = graph
        .nodes
        .iter()
        .filter(|node| node.kind == legal_structure_model::NodeKind::Section && node.grammar.as_deref() == Some("accepted_heading"))
        .count();
    let structure = json!({
        "schema_version": graph.schema_version,
        "node_count": graph.nodes.len(),
        "note_count": graph.nodes.iter().filter(|node| matches!(node.kind, legal_structure_model::NodeKind::Footnote | legal_structure_model::NodeKind::Endnote)).count(),
        "by_kind": by_kind,
        "sections_by_locator_kind": sections_by_locator_kind,
        "heading_derived_section_count": heading_derived_sections,
        "graph_sha256": serialization_sha256(graph),
        "examples": structure_examples(graph),
    });
    Ok(json!({
        "source_sha256": source_sha256,
        "page_count": page_count,
        "line_count": line_count,
        "product_sha256": serialization_sha256(&(&extracted.pages, graph, &extracted.diagnostics)),
        "structure": structure,
    }))
}
