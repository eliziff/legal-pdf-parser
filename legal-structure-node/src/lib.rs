use legal_structure::{
    a2aj_document_structure, analyze_instrument, analyze_native_markup, derive_document_structure,
    docx_structure_lint, journal_document_structure, journal_text_document_structure,
    normalize_source_doc_locator, parse_address, phrase_spans, project_document_structure_view,
    quote_words, text_fragment_directives, tokenize_source_text, A2ajInput, AuthoritativeTableCell,
    DocumentInput, DocumentStructure, FollowDirection, InstrumentCrossReferenceGraph,
    JournalPageLabel, NativeMarkupInput, PhraseOptions, SourceDoc, SourceDocKind, SourceDocOrigin,
    SourceDocQuery,
};
use napi::{
    bindgen_prelude::{AsyncTask, Buffer, External, ExternalRef},
    Env, Error, Task, Unknown,
};
use napi_derive::napi;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::sync::OnceLock;

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum StructureRequest {
    Evidence {
        input: DocumentInput,
    },
    Instrument {
        text: String,
        id: String,
        #[serde(default)]
        table_cells: Vec<AuthoritativeTableCell>,
        reconstruct_lineation: bool,
    },
    A2aj {
        input: A2ajInput,
    },
    NativeMarkup {
        input: NativeMarkupInput,
    },
    Journal {
        article_id: usize,
        url: Option<String>,
        filename: Option<String>,
        text: Option<String>,
        #[serde(default)]
        page_rows: Vec<serde_json::Value>,
    },
}

fn page_labels(rows: Vec<serde_json::Value>) -> Vec<JournalPageLabel> {
    rows.into_iter()
        .filter_map(|row| {
            let pdf_page = row.get("pdf_page")?.as_u64()?.try_into().ok()?;
            let label = match row.get("page_label")? {
                serde_json::Value::String(value) => value.clone(),
                value @ (serde_json::Value::Number(_) | serde_json::Value::Bool(_)) => {
                    value.to_string()
                }
                _ => return None,
            };
            Some(JournalPageLabel { label, pdf_page })
        })
        .collect()
}

enum NativeProduct {
    Structure(DocumentStructure),
    Docx {
        structure: DocumentStructure,
        table_cells: Vec<AuthoritativeTableCell>,
    },
    Pdf(legalpdf::PdfDocumentResult),
}

pub struct NativeDocument {
    product: NativeProduct,
    source_doc: OnceLock<SourceDocQuery>,
}

impl NativeDocument {
    fn source_doc(&self) -> &SourceDoc {
        self.source_doc_query().document()
    }

    fn source_doc_query(&self) -> &SourceDocQuery {
        self.source_doc.get_or_init(|| {
            SourceDocQuery::new(project_document_structure_view(
                self.structure()
                    .expect("PDF documents initialize their SourceDoc"),
            ))
        })
    }

    fn cross_references(&self) -> Option<&InstrumentCrossReferenceGraph> {
        match &self.product {
            NativeProduct::Structure(structure) => structure.cross_references.as_ref(),
            NativeProduct::Docx { structure, .. } => structure.cross_references.as_ref(),
            NativeProduct::Pdf(document) => document.cross_references(),
        }
    }

    fn structure(&self) -> Option<&DocumentStructure> {
        match &self.product {
            NativeProduct::Structure(structure) | NativeProduct::Docx { structure, .. } => {
                Some(structure)
            }
            NativeProduct::Pdf(_) => None,
        }
    }
}

#[derive(Serialize)]
struct StructureSnapshot<'a> {
    structure: &'a DocumentStructure,
}

#[derive(Serialize)]
struct SourceDocAnchor<'a> {
    kind: SourceDocKind,
    label: &'a str,
    start: usize,
    end: usize,
    #[serde(rename = "parentLabel", skip_serializing_if = "Option::is_none")]
    parent_label: Option<&'a str>,
}

fn js_value(env: Env, value: &impl Serialize) -> napi::Result<Unknown<'static>> {
    env.to_js_value(value)
}

fn source_doc_kind(value: &str) -> napi::Result<SourceDocKind> {
    match value {
        "paragraph" => Ok(SourceDocKind::Paragraph),
        "page" => Ok(SourceDocKind::Page),
        "section" => Ok(SourceDocKind::Section),
        "footnote" => Ok(SourceDocKind::Footnote),
        "table" => Ok(SourceDocKind::Table),
        "row" => Ok(SourceDocKind::Row),
        "cell" => Ok(SourceDocKind::Cell),
        _ => Err(Error::from_reason("invalid SourceDoc block kind")),
    }
}

fn follow_direction(value: &str) -> napi::Result<FollowDirection> {
    match value {
        "none" => Ok(FollowDirection::None),
        "out" => Ok(FollowDirection::Out),
        "in" => Ok(FollowDirection::In),
        "both" => Ok(FollowDirection::Both),
        _ => Err(Error::from_reason("invalid reference direction")),
    }
}

fn analyze_request(request: serde_json::Value) -> napi::Result<NativeDocument> {
    let request: StructureRequest =
        serde_json::from_value(request).map_err(|error| Error::from_reason(error.to_string()))?;
    let structure = match request {
        StructureRequest::Evidence { input } => {
            derive_document_structure(input).map_err(native_error)?
        }
        StructureRequest::Instrument {
            text,
            id,
            table_cells,
            reconstruct_lineation,
        } => analyze_instrument(&text, id, &table_cells, reconstruct_lineation)
            .map_err(native_error)?,
        StructureRequest::A2aj { input } => a2aj_document_structure(input).map_err(native_error)?,
        StructureRequest::NativeMarkup { input } => {
            analyze_native_markup(input).map_err(native_error)?
        }
        StructureRequest::Journal {
            article_id,
            url,
            filename,
            text,
            page_rows,
        } => {
            let labels = page_labels(page_rows);
            if let Some(filename) = filename {
                let file =
                    File::open(filename).map_err(|error| Error::from_reason(error.to_string()))?;
                journal_document_structure(article_id, url, BufReader::new(file), &labels)
            } else if let Some(text) = text {
                journal_text_document_structure(article_id, url, text, &labels)
            } else {
                return Err(Error::from_reason(
                    "journal request requires filename or text",
                ));
            }
            .map_err(native_error)?
        }
    };
    Ok(NativeDocument {
        product: NativeProduct::Structure(structure),
        source_doc: OnceLock::new(),
    })
}

pub struct DeriveDocumentTask {
    request: serde_json::Value,
}

impl Task for DeriveDocumentTask {
    type Output = NativeDocument;
    type JsValue = ExternalRef<NativeDocument>;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        analyze_request(std::mem::take(&mut self.request))
    }

    fn resolve(&mut self, env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        ExternalRef::new(&env, output)
    }
}

#[napi(js_name = "deriveDocumentStructure")]
pub fn derive_document_structure_node(request: serde_json::Value) -> AsyncTask<DeriveDocumentTask> {
    AsyncTask::new(DeriveDocumentTask { request })
}

pub struct DeriveDocxDocumentTask {
    bytes: Vec<u8>,
    id: String,
}

#[napi(object)]
pub struct DocxSupraReasonsNode {
    #[napi(js_name = "restarted_numbering")]
    pub restarted_numbering: bool,
    #[napi(js_name = "unsafe_or_split_fields")]
    pub unsafe_or_split_fields: u32,
}

#[napi(object)]
pub struct DocxSupraCleanupNode {
    pub bytes: Buffer,
    pub detected: u32,
    pub converted: u32,
    #[napi(js_name = "already_linked")]
    pub already_linked: u32,
    #[napi(js_name = "review_required")]
    pub review_required: u32,
    #[napi(js_name = "bookmarks_added")]
    pub bookmarks_added: u32,
    pub reasons: DocxSupraReasonsNode,
}

impl From<legalpdf::DocxSupraCleanup> for DocxSupraCleanupNode {
    fn from(result: legalpdf::DocxSupraCleanup) -> Self {
        Self {
            bytes: result.bytes.into(),
            detected: result.detected as u32,
            converted: result.converted as u32,
            already_linked: result.already_linked as u32,
            review_required: result.review_required as u32,
            bookmarks_added: result.bookmarks_added as u32,
            reasons: DocxSupraReasonsNode {
                restarted_numbering: result.restarted_numbering,
                unsafe_or_split_fields: result.unsafe_or_split_fields as u32,
            },
        }
    }
}

pub struct FixDocxSuprasTask {
    bytes: Vec<u8>,
}

fn docx_supra_bytes(bytes: Buffer) -> napi::Result<Vec<u8>> {
    if bytes.is_empty() || bytes.len() > legalpdf::MAX_DOCX_SUPRA_BYTES {
        return Err(Error::from_reason(
            "DOCX is empty or exceeds the read limit",
        ));
    }
    Ok(bytes.to_vec())
}

impl Task for FixDocxSuprasTask {
    type Output = legalpdf::DocxSupraCleanup;
    type JsValue = DocxSupraCleanupNode;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        legalpdf::fix_docx_supra_cross_references(&self.bytes)
            .map_err(|error| Error::from_reason(error.to_string()))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output.into())
    }
}

#[napi(js_name = "fixDocxSupraCrossReferences")]
pub fn fix_docx_supra_cross_references_node(
    bytes: Buffer,
) -> napi::Result<AsyncTask<FixDocxSuprasTask>> {
    Ok(AsyncTask::new(FixDocxSuprasTask {
        bytes: docx_supra_bytes(bytes)?,
    }))
}

pub struct HasDocxSuprasTask {
    bytes: Vec<u8>,
}

impl Task for HasDocxSuprasTask {
    type Output = bool;
    type JsValue = bool;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        legalpdf::has_docx_supra_references(&self.bytes)
            .map_err(|error| Error::from_reason(error.to_string()))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output)
    }
}

#[napi(js_name = "hasDocxSupraReferences")]
pub fn has_docx_supra_references_node(bytes: Buffer) -> napi::Result<AsyncTask<HasDocxSuprasTask>> {
    Ok(AsyncTask::new(HasDocxSuprasTask {
        bytes: docx_supra_bytes(bytes)?,
    }))
}

impl Task for DeriveDocxDocumentTask {
    type Output = NativeDocument;
    type JsValue = ExternalRef<NativeDocument>;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        let (structure, table_cells) =
            legalpdf::analyze_docx_bytes(&self.bytes, std::mem::take(&mut self.id))
                .map_err(|error| Error::from_reason(error.to_string()))?;
        Ok(NativeDocument {
            product: NativeProduct::Docx {
                structure,
                table_cells,
            },
            source_doc: OnceLock::new(),
        })
    }

    fn resolve(&mut self, env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        ExternalRef::new(&env, output)
    }
}

#[napi(js_name = "deriveDocxDocument")]
pub fn derive_docx_document_node(bytes: Buffer, id: String) -> AsyncTask<DeriveDocxDocumentTask> {
    AsyncTask::new(DeriveDocxDocumentTask {
        bytes: bytes.to_vec(),
        id,
    })
}

pub struct DerivePdfDocumentTask {
    request: serde_json::Value,
}

impl Task for DerivePdfDocumentTask {
    type Output = NativeDocument;
    type JsValue = ExternalRef<NativeDocument>;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        if self.request.get("kind").and_then(serde_json::Value::as_str) != Some("pdf") {
            return Err(Error::from_reason(
                "PDF document request has an invalid kind",
            ));
        }
        let pairing_audit = self
            .request
            .get("pairing_audit")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        legalpdf::derive_pdf_document(&self.request, pairing_audit)
            .map(|(document, source_doc)| NativeDocument {
                product: NativeProduct::Pdf(document),
                source_doc: OnceLock::from(SourceDocQuery::new(source_doc)),
            })
            .map_err(|error| Error::from_reason(error.to_string()))
    }

    fn resolve(&mut self, env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        ExternalRef::new(&env, output)
    }
}

#[napi(js_name = "derivePdfDocument")]
pub fn derive_pdf_document_node(request: serde_json::Value) -> AsyncTask<DerivePdfDocumentTask> {
    AsyncTask::new(DerivePdfDocumentTask { request })
}

#[napi(js_name = "documentSnapshot")]
pub fn document_snapshot_node(
    env: Env,
    document: &External<NativeDocument>,
) -> napi::Result<Unknown<'static>> {
    match &document.product {
        NativeProduct::Structure(structure) => js_value(env, &StructureSnapshot { structure }),
        NativeProduct::Docx { structure, .. } => js_value(env, &StructureSnapshot { structure }),
        NativeProduct::Pdf(document) => js_value(env, &legalpdf::pdf_document_snapshot(document)),
    }
}

#[napi(js_name = "pdfDocumentSummary")]
pub fn pdf_document_summary_node(
    env: Env,
    document: &External<NativeDocument>,
) -> napi::Result<Unknown<'static>> {
    let NativeProduct::Pdf(document) = &document.product else {
        return Err(Error::from_reason("PDF summary requires a PDF document"));
    };
    js_value(env, &legalpdf::pdf_document_summary(document))
}

#[napi(js_name = "documentCitedAuthorities")]
pub fn document_cited_authorities_node(
    env: Env,
    document: &External<NativeDocument>,
) -> napi::Result<Unknown<'static>> {
    match &document.product {
        NativeProduct::Structure(structure) => js_value(env, &structure.cited_authorities),
        NativeProduct::Docx { structure, .. } => js_value(env, &structure.cited_authorities),
        NativeProduct::Pdf(_) => js_value(env, &Vec::<serde_json::Value>::new()),
    }
}

#[napi(js_name = "docxStructureLint")]
pub fn docx_structure_lint_node(
    env: Env,
    document: &External<NativeDocument>,
) -> napi::Result<Unknown<'static>> {
    let Some(structure) = document.structure() else {
        return Err(Error::from_reason(
            "DOCX lint requires a structured document",
        ));
    };
    js_value(env, &docx_structure_lint(structure).map_err(native_error)?)
}

#[napi(js_name = "docxTableCells")]
pub fn docx_table_cells_node(
    env: Env,
    document: &External<NativeDocument>,
) -> napi::Result<Unknown<'static>> {
    let NativeProduct::Docx { table_cells, .. } = &document.product else {
        return Err(Error::from_reason(
            "DOCX table cells require a DOCX document",
        ));
    };
    js_value(env, table_cells)
}

#[napi(js_name = "sourceDocSnapshot")]
pub fn source_doc_snapshot_node(
    env: Env,
    document: &External<NativeDocument>,
) -> napi::Result<Unknown<'static>> {
    js_value(env, document.source_doc())
}

#[napi(js_name = "sourceDocText")]
pub fn source_doc_text_node(document: &External<NativeDocument>) -> String {
    document.source_doc().text.clone()
}

#[napi(js_name = "sourceDocTextBytes")]
pub fn source_doc_text_bytes_node(document: &External<NativeDocument>) -> u32 {
    document.source_doc().text.len() as u32
}

#[napi(js_name = "sourceDocRevision")]
pub fn source_doc_revision_node(document: &External<NativeDocument>) -> String {
    document.source_doc().revision.clone()
}

#[napi(js_name = "sourceDocAnchors")]
pub fn source_doc_anchors_node(
    env: Env,
    document: &External<NativeDocument>,
) -> napi::Result<Unknown<'static>> {
    js_value(
        env,
        &document
            .source_doc()
            .blocks
            .iter()
            .map(|block| SourceDocAnchor {
                kind: block.kind,
                label: &block.label,
                start: block.start,
                end: block.end,
                parent_label: block.parent_label.as_deref(),
            })
            .collect::<Vec<_>>(),
    )
}

#[napi(js_name = "normalizeSourceDocLocator")]
pub fn normalize_source_doc_locator_node(kind: String, locator: String) -> napi::Result<String> {
    Ok(normalize_source_doc_locator(
        source_doc_kind(&kind)?,
        &locator,
    ))
}

#[napi(js_name = "tokenizeSourceText")]
pub fn tokenize_source_text_node(env: Env, text: String) -> napi::Result<Unknown<'static>> {
    js_value(env, &tokenize_source_text(&text))
}

#[napi(js_name = "sourceDocQuoteWords")]
pub fn source_doc_quote_words_node(text: String) -> Vec<String> {
    quote_words(&text)
}

#[napi(js_name = "lookupSourceDoc")]
pub fn lookup_source_doc_node(
    env: Env,
    document: &External<NativeDocument>,
    kind: String,
    locator: String,
    context_blocks: u32,
) -> napi::Result<Unknown<'static>> {
    js_value(
        env,
        &document.source_doc_query().lookup(
            source_doc_kind(&kind)?,
            &locator,
            context_blocks as usize,
        ),
    )
}

#[napi(js_name = "readSourceDocRange")]
pub fn read_source_doc_range_node(
    env: Env,
    document: &External<NativeDocument>,
    kind: String,
    from: String,
    to: String,
    context_blocks: u32,
) -> napi::Result<Unknown<'static>> {
    js_value(
        env,
        &document.source_doc_query().read_range(
            source_doc_kind(&kind)?,
            &from,
            &to,
            context_blocks as usize,
        ),
    )
}

#[napi(js_name = "sourceDocSmallestContainingBlock")]
pub fn source_doc_smallest_containing_block_node(
    env: Env,
    document: &External<NativeDocument>,
    start: u32,
    end: u32,
) -> napi::Result<Unknown<'static>> {
    js_value(
        env,
        &document
            .source_doc_query()
            .smallest_containing_block(start as usize, end as usize),
    )
}

#[napi(js_name = "textPhraseSpans")]
pub fn text_phrase_spans_node(
    env: Env,
    text: String,
    words: Vec<String>,
    start: Option<u32>,
    end: Option<u32>,
    same_line: Option<bool>,
    limit: Option<u32>,
) -> napi::Result<Unknown<'static>> {
    js_value(
        env,
        &phrase_spans(
            &text,
            &words,
            PhraseOptions {
                start: start.map(|value| value as usize),
                end: end.map(|value| value as usize),
                same_line: same_line.unwrap_or(false),
                limit: limit.map(|value| value as usize),
            },
        ),
    )
}

#[napi(js_name = "textFragmentDirectives")]
pub fn text_fragment_directives_node(
    block_text: String,
    quotes: Vec<String>,
    page_scoped: bool,
    document_text: Option<String>,
    document: Option<&External<NativeDocument>>,
) -> Vec<String> {
    document.map_or_else(
        || text_fragment_directives(&block_text, document_text.as_deref(), &quotes, page_scoped),
        |document| {
            document
                .source_doc_query()
                .text_fragment_directives(&block_text, &quotes, page_scoped)
        },
    )
}

#[napi(js_name = "sourceDocParagraphRangeDirective")]
pub fn source_doc_paragraph_range_directive_node(
    document: &External<NativeDocument>,
    start: String,
    end: String,
) -> Option<String> {
    document
        .source_doc_query()
        .paragraph_range_directive(&start, &end)
}

#[napi(js_name = "sourceDocPageMap")]
pub fn source_doc_page_map_node(
    env: Env,
    document: &External<NativeDocument>,
) -> napi::Result<Unknown<'static>> {
    js_value(env, &document.source_doc_query().page_map())
}

#[napi(js_name = "resolveSourceDocPage")]
pub fn resolve_source_doc_page_node(
    env: Env,
    document: &External<NativeDocument>,
    requested: String,
) -> napi::Result<Unknown<'static>> {
    js_value(env, &document.source_doc_query().resolve_page(&requested))
}

#[napi(js_name = "lookupStructureBlock")]
pub fn lookup_structure_block_node(
    env: Env,
    document: &External<NativeDocument>,
    locator: String,
    context_blocks: u32,
) -> napi::Result<Unknown<'static>> {
    js_value(
        env,
        &document
            .source_doc_query()
            .structure_block(&locator, context_blocks as usize),
    )
}

#[napi(js_name = "parseDocumentAddress")]
pub fn parse_document_address_node(env: Env, spec: String) -> napi::Result<Unknown<'static>> {
    js_value(env, &parse_address(&spec))
}

#[napi(js_name = "graphScope")]
pub fn graph_scope_node(
    env: Env,
    document: &External<NativeDocument>,
    seed_label: String,
    follow: String,
    depth: u32,
    include_descendants: bool,
    include_units: bool,
) -> napi::Result<Unknown<'static>> {
    let follow = follow_direction(&follow)?;
    js_value(
        env,
        &document
            .cross_references()
            .filter(|graph| !graph.document_abstained)
            .and_then(|graph| {
                document.source_doc_query().graph_scope(
                    graph,
                    &seed_label,
                    follow,
                    depth as usize,
                    include_descendants,
                    include_units,
                )
            }),
    )
}

#[napi(js_name = "sourceDocHasOrigin")]
pub fn source_doc_has_origin_node(
    document: &External<NativeDocument>,
    origin: String,
) -> napi::Result<bool> {
    let origin = match origin.as_str() {
        "native" => SourceDocOrigin::Native,
        "heuristic" => SourceDocOrigin::Heuristic,
        _ => return Err(Error::from_reason("invalid SourceDoc origin")),
    };
    Ok(document
        .source_doc()
        .blocks
        .iter()
        .any(|block| block.origin == origin))
}

#[napi(js_name = "queryPdfDocument")]
pub fn query_pdf_document_node(
    env: Env,
    document: &External<NativeDocument>,
    query: serde_json::Value,
) -> napi::Result<Unknown<'static>> {
    let NativeProduct::Pdf(pdf) = &document.product else {
        return Err(Error::from_reason("PDF query requires a PDF document"));
    };
    let result = legalpdf::query_pdf_document(pdf, &query)
        .map_err(|error| Error::from_reason(error.to_string()))?;
    js_value(env, &result)
}

macro_rules! js_task {
    ($name:ident { $($field:ident: $ty:ty),+ $(,)? }, $compute:expr) => {
        pub struct $name { $($field: $ty),+ }
        impl Task for $name {
            type Output = serde_json::Value;
            type JsValue = Unknown<'static>;
            fn compute(&mut self) -> napi::Result<Self::Output> { $compute(self).map_err(native_error) }
            fn resolve(&mut self, env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
                js_value(env, &output)
            }
        }
    };
}

js_task!(
    DeleteAndRenumberTask {
        source: String,
        target: String,
        reconstruct_lineation: bool
    },
    |task: &mut DeleteAndRenumberTask| {
        legal_structure::delete_provision_and_renumber_siblings(
            &task.source,
            &task.target,
            task.reconstruct_lineation,
        )
    }
);

#[napi(js_name = "deleteProvisionAndRenumberSiblings")]
pub fn delete_provision_and_renumber_siblings_node(
    source: String,
    target: String,
    reconstruct_lineation: Option<bool>,
) -> AsyncTask<DeleteAndRenumberTask> {
    AsyncTask::new(DeleteAndRenumberTask {
        source,
        target,
        reconstruct_lineation: reconstruct_lineation.unwrap_or(true),
    })
}

js_task!(
    ConsolidateAmendmentTask {
        source: String,
        amendment: String,
        reconstruct_lineation: bool
    },
    |task: &mut ConsolidateAmendmentTask| {
        legal_structure::consolidate_amendment(
            &task.source,
            &task.amendment,
            task.reconstruct_lineation,
        )
    }
);

#[napi(js_name = "consolidateAmendment")]
pub fn consolidate_amendment_node(
    source: String,
    amendment: String,
    reconstruct_lineation: Option<bool>,
) -> AsyncTask<ConsolidateAmendmentTask> {
    AsyncTask::new(ConsolidateAmendmentTask {
        source,
        amendment,
        reconstruct_lineation: reconstruct_lineation.unwrap_or(true),
    })
}

fn native_error(error: legal_structure::EngineError) -> Error {
    Error::from_reason(error.to_string())
}
