//! The structure tagger in the parse: the request's `tagger` section names the folder holding
//! `tagger.onnx` and `tagger.json` (export_tagger.py), or `LEGALPDF_TAGGER_MODEL` does where a
//! request names none (the structure gold's grader); unset, the parse is tier 0 alone. The ONNX
//! Runtime library is the one the OCR provider is configured with (`LEGALPDF_ONNXRUNTIME`), or
//! `LEGALPDF_ONNX_RUNTIME` where no provider is. One session serves every parse in turn, so one
//! process runs one model: a second model pack is refused by name.
use crate::{Error, Result};
use legal_pdf_core::model::Page;
use legal_structure_model::DocumentStructure;
use legal_structure_tagger::onnx::Tagger;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// The request's tagger settings: the model pack's folder, the ONNX Runtime library (the one the
/// OCR provider runs on; without it the loader's fallback finds whatever `onnxruntime.dll` is on
/// the path, which may be another version) and the session's intra-op threads.
#[derive(Clone, Debug, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaggerOptions {
    pub model_pack: Option<PathBuf>,
    pub runtime: Option<PathBuf>,
    pub threads: Option<usize>,
}

/// The model pack a parse runs: the request's, else the one LEGALPDF_TAGGER_MODEL names.
fn model_pack(options: Option<&TaggerOptions>) -> Option<PathBuf> {
    options
        .and_then(|options| options.model_pack.clone())
        .or_else(|| std::env::var_os("LEGALPDF_TAGGER_MODEL").map(PathBuf::from))
        .filter(|dir| !dir.as_os_str().is_empty())
}

/// What a parse's structure depends on through the tagger: the model and its export beside it,
/// hashed once per process; none where no tagger runs. Part of the parse cache key.
pub(crate) fn identity(options: Option<&TaggerOptions>) -> Result<Option<String>> {
    let Some(dir) = model_pack(options) else { return Ok(None) };
    let onnx = legal_pdf_core::provider_asset_sha256(&dir.join("tagger.onnx"))?;
    let meta = legal_pdf_core::provider_asset_sha256(&dir.join("tagger.json"))?;
    Ok(Some(format!("{}:{}", &onnx[..16], &meta[..16])))
}

fn configured(options: Option<&TaggerOptions>) -> Result<Option<&'static Mutex<Tagger>>> {
    static TAGGER: OnceLock<Option<(PathBuf, Mutex<Tagger>)>> = OnceLock::new();
    static FAILURE: OnceLock<String> = OnceLock::new();
    let dir = model_pack(options);
    if let Some(found) = TAGGER.get() {
        return match (found, &dir) {
            (Some((loaded, _)), Some(dir)) if loaded != dir => Err(Error::Message(format!(
                "structure tagger already loaded from {}; refusing {}", loaded.display(), dir.display()
            ))),
            (Some((_, tagger)), Some(_)) => Ok(Some(tagger)),
            _ => Ok(None),
        };
    }
    let Some(dir) = dir else {
        return Ok(TAGGER.get_or_init(|| None).as_ref().map(|(_, tagger)| tagger));
    };
    if let Some(error) = FAILURE.get() {
        return Err(Error::Message(error.clone()));
    }
    let loaded = (|| -> std::result::Result<Tagger, String> {
        // Idempotent for the path already initialized; an error names the conflict.
        let runtime = options
            .and_then(|options| options.runtime.clone())
            .or_else(|| ["LEGALPDF_ONNX_RUNTIME", "LEGALPDF_ONNXRUNTIME"].iter()
                .find_map(|name| std::env::var_os(name).filter(|r| !r.is_empty()).map(PathBuf::from)));
        if let Some(runtime) = runtime {
            legal_pdf_core::init_ort_runtime(&runtime, None).map_err(|e| e.to_string())?;
        }
        let threads = options
            .and_then(|options| options.threads)
            .or_else(|| std::env::var("LEGALPDF_TAGGER_THREADS").ok().and_then(|t| t.parse().ok()))
            .unwrap_or(2);
        Tagger::load(&dir, threads)
    })();
    match loaded {
        Ok(tagger) => Ok(TAGGER.get_or_init(|| Some((dir, Mutex::new(tagger)))).as_ref().map(|(_, tagger)| tagger)),
        Err(error) => {
            let message = format!("structure tagger {}: {error}", dir.display());
            let _ = FAILURE.set(message.clone());
            Err(Error::Message(message))
        }
    }
}

/// The graph re-labelled by the configured tagger; a no-op without one.
pub(crate) fn retag(structure: &mut DocumentStructure, pages: &[Page], source_sha256: &str, options: Option<&TaggerOptions>) -> Result<()> {
    let Some(tagger) = configured(options)? else { return Ok(()) };
    let mut tagger = tagger.lock().map_err(|_| Error::Message("structure tagger poisoned".to_owned()))?;
    tagger.retag(structure, pages, source_sha256, false).map_err(Error::Message)
}
