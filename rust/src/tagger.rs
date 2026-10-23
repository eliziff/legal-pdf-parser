//! The structure tagger in the parse: `LEGALPDF_TAGGER_MODEL` names the folder holding
//! `tagger.onnx` and `tagger.json` (export_tagger.py); unset, the parse is tier 0 alone. The ONNX
//! Runtime library comes from `LEGALPDF_ONNX_RUNTIME` when no OCR provider initialized it. One
//! session serves every parse in turn.
use crate::{Error, Result};
use legal_pdf_core::model::Page;
use legal_structure_model::DocumentStructure;
use legal_structure_tagger::onnx::Tagger;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

fn configured() -> Result<Option<&'static Mutex<Tagger>>> {
    static TAGGER: OnceLock<Option<Mutex<Tagger>>> = OnceLock::new();
    static FAILURE: OnceLock<String> = OnceLock::new();
    if let Some(found) = TAGGER.get() {
        return Ok(found.as_ref());
    }
    let Some(dir) = std::env::var_os("LEGALPDF_TAGGER_MODEL").map(PathBuf::from).filter(|d| !d.as_os_str().is_empty()) else {
        return Ok(TAGGER.get_or_init(|| None).as_ref());
    };
    if let Some(error) = FAILURE.get() {
        return Err(Error::Message(error.clone()));
    }
    let loaded = (|| -> std::result::Result<Tagger, String> {
        if let Some(runtime) = std::env::var_os("LEGALPDF_ONNX_RUNTIME").filter(|r| !r.is_empty()) {
            // Idempotent for the path the OCR provider chose; an error names the conflict.
            legal_pdf_core::init_ort_runtime(Path::new(&runtime), None).map_err(|e| e.to_string())?;
        }
        let threads = std::env::var("LEGALPDF_TAGGER_THREADS").ok().and_then(|t| t.parse().ok()).unwrap_or(2);
        Tagger::load(&dir, threads)
    })();
    match loaded {
        Ok(tagger) => Ok(TAGGER.get_or_init(|| Some(Mutex::new(tagger))).as_ref()),
        Err(error) => {
            let message = format!("structure tagger {}: {error}", dir.display());
            let _ = FAILURE.set(message.clone());
            Err(Error::Message(message))
        }
    }
}

/// The graph re-labelled by the configured tagger; a no-op without one.
pub(crate) fn retag(structure: &mut DocumentStructure, pages: &[Page], source_sha256: &str) -> Result<()> {
    let Some(tagger) = configured()? else { return Ok(()) };
    let mut tagger = tagger.lock().map_err(|_| Error::Message("structure tagger poisoned".to_owned()))?;
    tagger.retag(structure, pages, source_sha256, true).map_err(Error::Message)
}
