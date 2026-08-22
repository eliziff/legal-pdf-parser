use legal_structure::{
    a2aj_source_doc, compose, derive_instrument_structure, derive_structure_evidence,
    instrument_lineation_hypotheses, journal_source_doc, journal_text_source_doc,
    native_markup_source_doc, A2ajInput, DocumentInput, InstrumentReferenceEvidence,
    JournalPageLabel, NativeMarkupInput, SourceDoc, SourceDocBlock,
};
use napi::Error;
use napi_derive::napi;
use std::fs::File;
use std::io::BufReader;

#[napi(js_name = "sourceDocVersion")]
pub fn source_doc_version() -> u32 {
    legal_structure::SOURCE_DOC_VERSION
}

#[napi(js_name = "instrumentLineationHypotheses")]
pub fn instrument_lineation_hypotheses_node(text: String) -> Vec<String> {
    instrument_lineation_hypotheses(&text)
}

#[napi(js_name = "deriveInstrumentStructure")]
pub fn derive_instrument_structure_node(
    text: String,
    documents: Vec<serde_json::Value>,
    references: Vec<serde_json::Value>,
) -> napi::Result<serde_json::Value> {
    let documents = documents
        .into_iter()
        .map(|document| {
            DocumentInput::try_from(document).map_err(|error| Error::from_reason(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let references = references
        .into_iter()
        .map(serde_json::from_value::<InstrumentReferenceEvidence>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| Error::from_reason(error.to_string()))?;
    let (selected, graph, contents) = derive_instrument_structure(&text, documents, &references)
        .map_err(|error| Error::from_reason(error.to_string()))?;
    Ok(serde_json::json!({ "selected": selected, "graph": graph, "contents": contents }))
}

#[napi(js_name = "deriveStructures")]
pub fn derive_structures(
    documents: Vec<serde_json::Value>,
) -> napi::Result<Vec<serde_json::Value>> {
    documents
        .into_iter()
        .map(|document| {
            let input = DocumentInput::try_from(document)
                .map_err(|error| Error::from_reason(error.to_string()))?;
            let graph = derive_structure_evidence(input)
                .map_err(|error| Error::from_reason(error.to_string()))?;
            serde_json::to_value(graph).map_err(|error| Error::from_reason(error.to_string()))
        })
        .collect()
}

fn journal_document(request: &serde_json::Value) -> napi::Result<SourceDoc> {
    let article_id = request
        .get("article_id")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| Error::from_reason("journal article_id must be a positive integer"))?;
    let url = request
        .get("url")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let page_rows = request
        .get("page_rows")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Error::from_reason("journal page_rows must be an array"))?;
    let labels = page_rows
        .iter()
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
        .collect::<Vec<_>>();
    if let Some(filename) = request.get("filename").and_then(serde_json::Value::as_str) {
        let file = File::open(filename).map_err(|error| Error::from_reason(error.to_string()))?;
        return journal_source_doc(article_id as usize, url, BufReader::new(file), &labels)
            .map_err(|error| Error::from_reason(error.to_string()));
    }
    if let Some(text) = request.get("text").and_then(serde_json::Value::as_str) {
        return journal_text_source_doc(article_id as usize, url, text.to_owned(), &labels)
            .map_err(|error| Error::from_reason(error.to_string()));
    }
    Err(Error::from_reason(
        "journal request requires filename or text",
    ))
}

fn derive_source_doc(request: &serde_json::Value) -> napi::Result<SourceDoc> {
    let kind = request
        .get("kind")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| Error::from_reason("SourceDoc request kind is required"))?;
    Ok(match kind {
        "a2aj" => {
            let input = serde_json::from_value::<A2ajInput>(
                request
                    .get("input")
                    .cloned()
                    .ok_or_else(|| Error::from_reason("A2AJ input is required"))?,
            )
            .map_err(|error| Error::from_reason(error.to_string()))?;
            a2aj_source_doc(input).map_err(|error| Error::from_reason(error.to_string()))?
        }
        "evidence" => {
            let mut input = DocumentInput::try_from(
                request
                    .get("input")
                    .cloned()
                    .ok_or_else(|| Error::from_reason("evidence input is required"))?,
            )
            .map_err(|error| Error::from_reason(error.to_string()))?;
            let original_values = request.get("original_claims").cloned().unwrap_or_default();
            let mut originals = serde_json::from_value::<
                std::collections::HashMap<String, SourceDocBlock>,
            >(original_values.clone())
            .map_err(|error| Error::from_reason(error.to_string()))?;
            let orders = request
                .get("original_claim_orders")
                .and_then(serde_json::Value::as_object);
            for (id, block) in &mut originals {
                if let Some(fields) = orders
                    .and_then(|orders| orders.get(id))
                    .and_then(serde_json::Value::as_array)
                {
                    let fields = fields
                        .iter()
                        .filter_map(serde_json::Value::as_str)
                        .map(str::to_owned)
                        .collect::<Vec<_>>();
                    block.preserve_field_order(&fields);
                }
            }
            input.set_original_claims(originals);
            compose(input).map_err(|error| Error::from_reason(error.to_string()))?
        }
        "native_markup" => {
            let input = serde_json::from_value::<NativeMarkupInput>(
                request
                    .get("input")
                    .cloned()
                    .ok_or_else(|| Error::from_reason("native-markup input is required"))?,
            )
            .map_err(|error| Error::from_reason(error.to_string()))?;
            native_markup_source_doc(input)
                .map_err(|error| Error::from_reason(error.to_string()))?
        }
        "journal" => journal_document(&request)?,
        _ => return Err(Error::from_reason("unsupported SourceDoc request kind")),
    })
}

fn source_doc(request: serde_json::Value) -> napi::Result<serde_json::Value> {
    let input_text = request
        .get("input")
        .and_then(|input| input.get("text"))
        .and_then(serde_json::Value::as_str);
    let document = derive_source_doc(&request)?;
    let index = document.index.entries();
    let value = document
        .json_value(input_text != Some(document.text.as_str()))
        .map_err(|error| Error::from_reason(error.to_string()))?;
    Ok(serde_json::json!({ "document": value, "index": index }))
}

#[napi(js_name = "sourceDocs")]
pub fn source_docs(requests: Vec<serde_json::Value>) -> napi::Result<Vec<serde_json::Value>> {
    requests.into_iter().map(source_doc).collect()
}
