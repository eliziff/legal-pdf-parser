use legal_structure::{
    a2aj_document_structure, analyze_docx, analyze_instrument, analyze_native_markup,
    derive_document_structure, journal_document_structure, journal_text_document_structure,
    project_document_structure_view, A2ajInput, AuthoritativeTableCell, DocumentInput,
    DocumentStructure, JournalPageLabel, NativeMarkupInput, SourceDoc,
};
use napi::{
    bindgen_prelude::{AsyncTask, Buffer, External, ExternalRef},
    Env, Error, Task,
};
use napi_derive::napi;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum StructureRequest {
    Evidence {
        input: DocumentInput,
        #[serde(default)]
        source_doc: bool,
    },
    Instrument {
        text: String,
        id: String,
        #[serde(default)]
        table_cells: Vec<AuthoritativeTableCell>,
        reconstruct_lineation: bool,
        #[serde(default)]
        source_doc: bool,
    },
    A2aj {
        input: A2ajInput,
        #[serde(default)]
        source_doc: bool,
    },
    NativeMarkup {
        input: NativeMarkupInput,
        #[serde(default)]
        source_doc: bool,
    },
    Journal {
        article_id: usize,
        url: Option<String>,
        filename: Option<String>,
        text: Option<String>,
        #[serde(default)]
        page_rows: Vec<serde_json::Value>,
        #[serde(default)]
        source_doc: bool,
    },
    Docx {
        id: String,
        paragraphs: Vec<String>,
        #[serde(default)]
        table_cells: Vec<AuthoritativeTableCell>,
        #[serde(default)]
        source_doc: bool,
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

#[derive(Serialize)]
struct StructureResponse {
    structure: DocumentStructure,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_doc: Option<SourceDoc>,
}

fn analyze_request(request: serde_json::Value) -> napi::Result<serde_json::Value> {
    let request: StructureRequest =
        serde_json::from_value(request).map_err(|error| Error::from_reason(error.to_string()))?;
    let (structure, source_doc) = match request {
        StructureRequest::Evidence { input, source_doc } => (
            derive_document_structure(input).map_err(native_error)?,
            source_doc,
        ),
        StructureRequest::Instrument {
            text,
            id,
            table_cells,
            reconstruct_lineation,
            source_doc,
        } => (
            analyze_instrument(&text, id, &table_cells, reconstruct_lineation)
                .map_err(native_error)?,
            source_doc,
        ),
        StructureRequest::A2aj { input, source_doc } => (
            a2aj_document_structure(input).map_err(native_error)?,
            source_doc,
        ),
        StructureRequest::NativeMarkup { input, source_doc } => (
            analyze_native_markup(input).map_err(native_error)?,
            source_doc,
        ),
        StructureRequest::Journal {
            article_id,
            url,
            filename,
            text,
            page_rows,
            source_doc,
        } => {
            let labels = page_labels(page_rows);
            let structure = if let Some(filename) = filename {
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
            .map_err(native_error)?;
            (structure, source_doc)
        }
        StructureRequest::Docx {
            id,
            paragraphs,
            table_cells,
            source_doc,
        } => (
            analyze_docx(id, paragraphs, &table_cells).map_err(native_error)?,
            source_doc,
        ),
    };
    let source_doc = source_doc.then(|| {
        let mut document = project_document_structure_view(&structure);
        document.text.clear();
        document
    });
    serde_json::to_value(StructureResponse {
        structure,
        source_doc,
    })
    .map_err(|error| Error::from_reason(error.to_string()))
}

pub struct DeriveDocumentTask {
    request: serde_json::Value,
}

impl Task for DeriveDocumentTask {
    type Output = Vec<u8>;
    type JsValue = Buffer;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        serde_json::to_vec(&analyze_request(std::mem::take(&mut self.request))?)
            .map_err(|error| Error::from_reason(error.to_string()))
    }

    fn resolve(&mut self, _: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output.into())
    }
}

#[napi(js_name = "deriveDocumentStructure")]
pub fn derive_document_structure_node(
    request: serde_json::Value,
) -> AsyncTask<DeriveDocumentTask> {
    AsyncTask::new(DeriveDocumentTask { request })
}

pub struct DerivePdfDocumentTask {
    request: serde_json::Value,
}

impl Task for DerivePdfDocumentTask {
    type Output = legalpdf::PdfDocumentResult;
    type JsValue = ExternalRef<legalpdf::PdfDocumentResult>;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        if self.request.get("kind").and_then(serde_json::Value::as_str) != Some("pdf") {
            return Err(Error::from_reason("PDF document request has an invalid kind"));
        }
        let source_doc = self
            .request
            .get("source_doc")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let pairing_audit = self
            .request
            .get("pairing_audit")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        legalpdf::derive_pdf_document(&self.request, source_doc, pairing_audit)
            .map_err(|error| Error::from_reason(error.to_string()))
    }

    fn resolve(&mut self, env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        ExternalRef::new(&env, output)
    }
}

#[napi(js_name = "derivePdfDocument")]
pub fn derive_pdf_document_node(
    request: serde_json::Value,
) -> AsyncTask<DerivePdfDocumentTask> {
    AsyncTask::new(DerivePdfDocumentTask { request })
}

#[napi(js_name = "pdfDocumentSnapshot")]
pub fn pdf_document_snapshot_node(
    document: &External<legalpdf::PdfDocumentResult>,
) -> napi::Result<Buffer> {
    serde_json::to_vec(&legalpdf::pdf_document_snapshot(document))
        .map(Buffer::from)
        .map_err(|error| Error::from_reason(error.to_string()))
}

#[napi(js_name = "queryPdfDocument")]
pub fn query_pdf_document_node(
    document: &External<legalpdf::PdfDocumentResult>,
    query: serde_json::Value,
) -> napi::Result<Buffer> {
    let result = legalpdf::query_pdf_document(document, &query)
        .map_err(|error| Error::from_reason(error.to_string()))?;
    serde_json::to_vec(&result)
        .map(Buffer::from)
        .map_err(|error| Error::from_reason(error.to_string()))
}

fn native_error(error: legal_structure::EngineError) -> Error {
    Error::from_reason(error.to_string())
}
