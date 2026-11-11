use sha2::{Digest, Sha256};
use std::{
    env, fs,
    path::{Path, PathBuf},
};

fn files(path: &Path, out: &mut Vec<PathBuf>) {
    if path.file_name().is_some_and(|name| name == "target") {
        return;
    }
    if path.is_file() {
        out.push(path.to_owned());
    } else if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            files(&entry.path(), out);
        }
    }
}

/// legal-layout's sources, wherever Cargo has the crate (legal-structure-model says through its `links` metadata).
fn layout_sources() -> PathBuf {
    PathBuf::from(env::var_os("DEP_LEGAL_STRUCTURE_MODEL_LAYOUT").expect("legal-structure-model publishes legal-layout's sources"))
}

/// The name a file is hashed under: its path in this checkout, or in legal-layout's under `legal-layout/`.
fn key_of(root: &Path, layout: &Path, file: &Path) -> String {
    let key = match file.strip_prefix(layout) {
        Ok(rest) => Path::new("legal-layout").join(rest),
        Err(_) => file.strip_prefix(root).unwrap_or(file).to_owned(),
    };
    key.to_string_lossy().replace('\\', "/")
}

fn fingerprint(root: &Path, inputs: &[PathBuf], features: &[String]) -> String {
    let layout = layout_sources();
    let mut paths = Vec::new();
    for input in inputs {
        println!("cargo:rerun-if-changed={}", input.display());
        let mut found = Vec::new();
        files(input, &mut found);
        paths.extend(found.into_iter().map(|file| (key_of(root, &layout, &file), file)));
    }
    paths.sort();
    paths.dedup();
    let mut digest = Sha256::new();
    for (relative, path) in paths {
        digest.update(relative.as_bytes());
        digest.update([0]);
        // A file's line endings are the checkout's, not the code's: a working tree with CRLF and a
        // git archive with LF read the same.
        let bytes = fs::read(path).unwrap();
        let lf: Vec<u8> = bytes.iter().enumerate()
            .filter(|&(k, &b)| !(b == b'\r' && bytes.get(k + 1) == Some(&b'\n')))
            .map(|(_, &b)| b)
            .collect();
        digest.update(lf);
        digest.update([0]);
    }
    for feature in features {
        digest.update(feature.as_bytes());
        digest.update([0]);
    }
    format!("{:x}", digest.finalize())
}

/// pdf-inspector's sources, "bin" and the names in skip aside.
fn inspector_inputs(root: &Path, skip: &[&str]) -> Vec<PathBuf> {
    let inspector = root.join("vendor/pdf-inspector");
    let mut inputs = vec![
        inspector.join("Cargo.toml"),
        inspector.join("external/bcmaps"),
    ];
    for entry in fs::read_dir(inspector.join("src")).unwrap().flatten() {
        let name = entry.file_name();
        if name != "bin" && !skip.iter().any(|skipped| name == *skipped) {
            inputs.push(entry.path());
        }
    }
    inputs
}

/// What the extraction runs of pdf-inspector: not its Markdown, its table detectors (the document
/// stage reads tables), its vision models or its Python bindings.
const NOT_EXTRACTION: &[&str] = &["markdown", "tables", "vision", "python.rs"];

fn crate_inputs(root: &Path, name: &str) -> Vec<PathBuf> {
    let owner = root.join(format!("legal-pdf-{name}"));
    let mut inputs = vec![owner.join("Cargo.toml"), owner.join("src")];
    if owner.join("build.rs").is_file() {
        inputs.push(owner.join("build.rs"));
    }
    inputs
}

/// What reads a page's text: the OCR crate's manifest and its top-level sources, but not its
/// re-exports (lib.rs), its raster measures (raster.rs, keyed on their own) or its command-line tool
/// (bin); and the page renderer it reads.
fn recognition_sources(root: &Path) -> Vec<PathBuf> {
    let owner = root.join("legal-pdf-ocr");
    let mut inputs = vec![owner.join("Cargo.toml"), root.join("legal-pdf-core/src/render.rs")];
    for entry in fs::read_dir(owner.join("src")).unwrap().flatten() {
        let path = entry.path();
        let name = entry.file_name();
        if path.is_file() && path.extension().is_some_and(|ext| ext == "rs") && name != "lib.rs" && name != "raster.rs" {
            inputs.push(path);
        }
    }
    inputs
}

/// What measures a page's render beyond its text: the raster module and the shared render helpers.
fn raster_sources(root: &Path) -> Vec<PathBuf> {
    ["legal-pdf-ocr/Cargo.toml", "legal-pdf-ocr/src/raster.rs", "legal-pdf-ocr/src/ocr.rs", "legal-pdf-ocr/src/separator.rs", "legal-pdf-core/src/ocr_contract.rs", "legal-pdf-core/src/render.rs"]
        .iter()
        .map(|name| root.join(name))
        .collect()
}

// Raw extraction and OCR pages do not depend on DOCX, structure inference or
// document projection. Their hashes follow the code and contracts they use.
fn stage_inputs(root: &Path, indra: bool) -> [Vec<PathBuf>; 4] {
    let paths = |names: &[&str]| names.iter().map(|name| root.join(name)).collect::<Vec<_>>();
    let mut extraction = paths(&[
        "legal-pdf-core/Cargo.toml",
        "legal-pdf-core/src/lib.rs",
        "legal-pdf-core/src/model.rs",
        "legal-pdf-core/src/ocr_contract.rs",
    ]);
    // The pages' types and the measures extraction takes, not the readers' own measures.
    let layout = layout_sources();
    extraction.extend(["Cargo.toml", "src/lib.rs", "src/stats.rs"].map(|name| layout.join(name)));
    extraction.extend(inspector_inputs(root, NOT_EXTRACTION));
    extraction.extend(crate_inputs(root, "extraction-processor"));
    let mut recognition = paths(&[
        "legal-pdf-core/Cargo.toml",
        "legal-pdf-core/src/ocr_contract.rs",
        "legal-pdf-core/src/asset.rs",
        "legal-pdf-core/src/ort_backend.rs",
        "legal-pdf-core/src/ort_runtime.rs",
    ]);
    recognition.extend(recognition_sources(root));
    if indra {
        recognition.push(root.join("rust/native/gpu_adapter.cpp"));
    }
    let mut document = paths(&[
        "rust/src/contract.rs",
        "rust/src/engine.rs",
        "rust/src/supplied_ocr.rs",
        "legal-pdf-core/Cargo.toml",
        "legal-pdf-core/src/lib.rs",
        "legal-pdf-core/src/model.rs",
        "legal-pdf-core/src/analysis.rs",
        "legal-pdf-core/src/ocr_contract.rs",
        "legal-pdf-support/Cargo.toml",
        "legal-pdf-support/src/lib.rs",
        "legal-pdf-support/src/projection.rs",
        "legal-pdf-support/src/projection",
    ]);
    // The structure stage reads pdf-inspector's layout too (columns, furniture, leaders, words'
    // joins, heading tiers, tables).
    document.extend(inspector_inputs(root, &[]));
    // The structure stage, one crate: an edit to it is a new structure (and a new reader and document key).
    document.extend(crate_inputs(root, "structure"));
    let layout = paths(&[
        "legal-pdf-core/Cargo.toml",
        "legal-pdf-core/src/model.rs",
        "legal-pdf-core/src/asset.rs",
        "legal-pdf-core/src/ort_backend.rs",
        "legal-pdf-core/src/ort_runtime.rs",
        "legal-pdf-support/Cargo.toml",
        "legal-pdf-support/src/ppdoc.rs",
        "legal-pdf-support/src/ppdoc_openvino.rs",
        "legal-pdf-support/src/ppdoc_postprocess.rs",
    ]);
    [extraction, recognition, document, layout]
}

fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let enabled = |feature: &str| env::var_os(format!("CARGO_FEATURE_{feature}")).is_some();
    let pdf = enabled("PDF");
    let mut features: Vec<_> = env::vars_os()
        .filter_map(|(name, _)| {
            name.to_str()?
                .strip_prefix("CARGO_FEATURE_")
                .map(str::to_owned)
        })
        .collect();
    features.sort();
    if pdf {
        for ((name, active, stage_features), inputs) in [
            ("EXTRACTION", true, vec![]),
            (
                "OCR",
                enabled("OCR"),
                features
                    .iter()
                    .filter(|name| ["OCR", "INDRA"].contains(&name.as_str()))
                    .cloned()
                    .collect(),
            ),
            ("DOCUMENT", true, vec![]),
            (
                "LAYOUT",
                enabled("PPDOC"),
                features.iter().filter(|name| name.as_str() == "PPDOC").cloned().collect(),
            ),
        ]
        .into_iter()
        .zip(stage_inputs(&root, enabled("INDRA")))
        {
            let inputs = if active { inputs } else { vec![] };
            println!(
                "cargo:rustc-env=LEGAL_PDF_{name}_SHA256={}",
                fingerprint(&root, &inputs, &stage_features)
            );
        }
        let raster = if enabled("OCR") { raster_sources(&root) } else { Vec::new() };
        println!("cargo:rustc-env=LEGAL_PDF_RASTER_SHA256={}", fingerprint(&root, &raster, &[]));
    }
}
