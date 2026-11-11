use crate::indra::IndraTier;
use legal_pdf_core::OrtBackend;
use legal_pdf_core::{pack_file, provider_asset_sha256, Error, Result};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct Asset {
    path: String,
    sha256: String,
}

#[derive(Deserialize)]
struct Model {
    cpu: Asset,
    gpu: Asset,
    codec: Asset,
}

#[derive(Deserialize)]
struct Tier {
    model: String,
    #[serde(rename = "detectorSide")]
    detector_side: usize,
}

#[derive(Deserialize)]
struct Manifest {
    recognizer: String,
    input_height: usize,
    tiers: BTreeMap<String, Tier>,
    models: BTreeMap<String, Model>,
    detector: Asset,
}

#[derive(Clone)]
pub(crate) struct ModelPack {
    pub model: PathBuf,
    pub codec: PathBuf,
    pub detector: PathBuf,
    pub detector_side: usize,
    pub orientation: Option<PathBuf>,
}

impl ModelPack {
    pub fn open(root: &Path, tier: IndraTier, backend: OrtBackend) -> Result<Self> {
        let root = fs::canonicalize(root).map_err(|source| Error::io(root, source))?;
        let path = root.join("manifest.json");
        if fs::metadata(&path)
            .map_err(|source| Error::io(&path, source))?
            .len()
            > 2 * 1024 * 1024
        {
            return Err(Error::Message("Indra manifest exceeds 2 MiB".to_owned()));
        }
        let manifest: Manifest =
            serde_json::from_slice(&fs::read(&path).map_err(|source| Error::io(&path, source))?)
                .map_err(|error| {
                    Error::Message(format!("invalid Indra model manifest: {error}"))
                })?;
        if manifest.recognizer != "indra" || manifest.input_height != 32 {
            return Err(Error::Message(
                "Indra model pack must declare height-32 Indra recognition".to_owned(),
            ));
        }
        let selected = manifest
            .tiers
            .get(tier.name())
            .ok_or_else(|| Error::Message(format!("Indra pack has no {} tier", tier.name())))?;
        if !(128..=4096).contains(&selected.detector_side) || selected.detector_side % 32 != 0 {
            return Err(Error::Message(
                "Indra detector size must be a multiple of 32 between 128 and 4096".to_owned(),
            ));
        }
        let model = manifest.models.get(&selected.model).ok_or_else(|| {
            Error::Message("Indra tier refers to a missing recognizer".to_owned())
        })?;
        let accelerated = matches!(
            backend,
            OrtBackend::Cuda | OrtBackend::TensorRt | OrtBackend::DirectMl
        );
        Ok(Self {
            model: checked_asset(&root, if accelerated { &model.gpu } else { &model.cpu })?,
            codec: checked_asset(&root, &model.codec)?,
            detector: checked_asset(&root, &manifest.detector)?,
            detector_side: selected.detector_side,
            orientation: Some(root.join("orientation.onnx")).filter(|path| path.is_file()),
        })
    }
}

fn checked_asset(root: &Path, asset: &Asset) -> Result<PathBuf> {
    let path = pack_file(root, &asset.path, "Indra pack asset")?;
    if provider_asset_sha256(&path)? != asset.sha256 {
        return Err(Error::Message(format!(
            "Indra asset failed hash verification: {}",
            asset.path
        )));
    }
    Ok(path)
}
