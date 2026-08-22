use super::{
    derive_native_structure_evidence, DocumentInput, EngineError, StructureGraphV2,
    EVIDENCE_SCHEMA, MAX_BYTES, MAX_DOCUMENTS, RESULT_SCHEMA, SIDECAR_PROTOCOL,
};
#[cfg(feature = "structure-inference")]
use super::derive_structure_evidence;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::io::{BufRead, Write};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeriveBatch {
    #[serde(rename = "type")]
    kind: String,
    request_id: String,
    documents: Vec<Value>,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    code: &'a str,
    message: &'a str,
}

#[derive(Serialize)]
struct ItemError<'a> {
    id: &'a str,
    ok: bool,
    error: ErrorBody<'a>,
}

#[derive(Serialize)]
struct ItemResult<'a> {
    id: &'a str,
    ok: bool,
    result: StructureGraphV2,
}

fn io<T>(value: std::io::Result<T>) -> Result<T, EngineError> {
    value.map_err(|error| EngineError {
        code: "sidecar_io",
        message: error.to_string(),
    })
}

fn json_error(error: serde_json::Error) -> EngineError {
    EngineError {
        code: "sidecar_json",
        message: error.to_string(),
    }
}

fn read_line(reader: &mut impl BufRead, line: &mut Vec<u8>) -> Result<usize, EngineError> {
    loop {
        let buffer = io(reader.fill_buf())?;
        if buffer.is_empty() {
            return Ok(line.len());
        }
        let used = buffer
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(buffer.len(), |index| index + 1);
        if line.len() + used > MAX_BYTES + 1 {
            return Err(EngineError::invalid("oversized sidecar line"));
        }
        line.extend_from_slice(&buffer[..used]);
        reader.consume(used);
        if line.last() == Some(&b'\n') {
            return Ok(line.len());
        }
    }
}

fn sidecar_with(
    reader: &mut impl BufRead,
    writer: &mut impl Write,
    derive: fn(DocumentInput) -> Result<StructureGraphV2, EngineError>,
    capabilities: &[&str],
) -> Result<(), EngineError> {
    let executable = std::env::current_exe().map_err(|error| EngineError {
        code: "sidecar_identity",
        message: error.to_string(),
    })?;
    let engine_sha256 = format!(
        "{:x}",
        Sha256::digest(std::fs::read(executable).map_err(|error| EngineError {
            code: "sidecar_identity",
            message: error.to_string()
        })?)
    );
    serde_json::to_writer(&mut *writer, &json!({ "type": "hello", "protocol": SIDECAR_PROTOCOL, "evidence_schema": EVIDENCE_SCHEMA,
        "result_schema": RESULT_SCHEMA, "engine_sha256": engine_sha256, "capabilities": capabilities,
        "max_documents": MAX_DOCUMENTS, "max_bytes": MAX_BYTES })).map_err(json_error)?;
    io(writer.write_all(b"\n"))?;
    io(writer.flush())?;
    loop {
        let mut line = Vec::new();
        if read_line(reader, &mut line)? == 0 {
            return Err(EngineError::invalid("sidecar received unexpected EOF"));
        }
        if line.last() != Some(&b'\n')
            || line.len() - 1 > MAX_BYTES
            || line[..line.len() - 1].contains(&b'\r')
        {
            return Err(EngineError::invalid("invalid sidecar line"));
        }
        line.pop();
        let batch: DeriveBatch = serde_json::from_slice(&line).map_err(json_error)?;
        if batch.kind != "derive_batch"
            || batch.request_id.is_empty()
            || batch.documents.is_empty()
            || batch.documents.len() > MAX_DOCUMENTS
        {
            return Err(EngineError::invalid("invalid derive_batch envelope"));
        }
        let ids = batch
            .documents
            .iter()
            .map(|value| {
                value
                    .get("document_id")
                    .and_then(Value::as_str)
                    .filter(|id| !id.is_empty())
                    .map(str::to_owned)
                    .ok_or_else(|| EngineError::invalid("derive_batch document ID is missing"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if ids.iter().collect::<HashSet<_>>().len() != ids.len() {
            return Err(EngineError::invalid(
                "derive_batch document IDs are duplicated",
            ));
        }
        io(writer.write_all(b"{\"type\":\"result_batch\",\"request_id\":"))?;
        serde_json::to_writer(&mut *writer, &batch.request_id).map_err(json_error)?;
        io(writer.write_all(b",\"items\":["))?;
        for (index, (id, value)) in ids.iter().zip(batch.documents).enumerate() {
            if index > 0 {
                io(writer.write_all(b","))?;
            }
            match DocumentInput::try_from(value).and_then(derive) {
                Ok(result) => serde_json::to_writer(
                    &mut *writer,
                    &ItemResult {
                        id,
                        ok: true,
                        result,
                    },
                )
                .map_err(json_error)?,
                Err(error) => serde_json::to_writer(
                    &mut *writer,
                    &ItemError {
                        id,
                        ok: false,
                        error: ErrorBody {
                            code: error.code,
                            message: &error.message,
                        },
                    },
                )
                .map_err(json_error)?,
            }
        }
        io(writer.write_all(b"]}\n"))?;
        io(writer.flush())?;
    }
}

#[cfg(feature = "structure-inference")]
pub fn sidecar(reader: &mut impl BufRead, writer: &mut impl Write) -> Result<(), EngineError> {
    sidecar_with(
        reader,
        writer,
        derive_structure_evidence,
        &["native_claims", "raw_recovery"],
    )
}

pub fn native_sidecar(
    reader: &mut impl BufRead,
    writer: &mut impl Write,
) -> Result<(), EngineError> {
    sidecar_with(
        reader,
        writer,
        derive_native_structure_evidence,
        &["native_claims"],
    )
}

#[cfg(feature = "structure-inference")]
pub fn stdio_sidecar() -> Result<(), EngineError> {
    sidecar(&mut std::io::stdin().lock(), &mut std::io::stdout().lock())
}

pub fn native_stdio_sidecar() -> Result<(), EngineError> {
    native_sidecar(&mut std::io::stdin().lock(), &mut std::io::stdout().lock())
}
