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

fn fingerprint(root: &Path, inputs: &[PathBuf], features: &[String]) -> String {
    let mut paths = Vec::new();
    for input in inputs {
        println!("cargo:rerun-if-changed={}", input.display());
        files(input, &mut paths);
    }
    paths.sort();
    paths.dedup();
    let mut digest = Sha256::new();
    for path in paths {
        let relative = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        digest.update(relative.as_bytes());
        digest.update([0]);
        digest.update(fs::read(path).unwrap());
        digest.update([0]);
    }
    for feature in features {
        digest.update(feature.as_bytes());
        digest.update([0]);
    }
    format!("{:x}", digest.finalize())
}

fn inspector_inputs(root: &Path) -> Vec<PathBuf> {
    let inspector = root.join("vendor/pdf-inspector");
    let mut inputs = vec![
        inspector.join("Cargo.toml"),
        inspector.join("external/bcmaps"),
    ];
    for entry in fs::read_dir(inspector.join("src")).unwrap().flatten() {
        if entry.file_name() != "bin" {
            inputs.push(entry.path());
        }
    }
    inputs
}

fn crate_inputs(root: &Path, name: &str) -> Vec<PathBuf> {
    let owner = root.join(format!("legal-pdf-{name}"));
    let mut inputs = vec![owner.join("Cargo.toml"), owner.join("src")];
    if owner.join("build.rs").is_file() {
        inputs.push(owner.join("build.rs"));
    }
    inputs
}

// Raw extraction and OCR pages do not depend on DOCX, structure inference or
// document projection. Their hashes follow the code and contracts they use.
fn stage_inputs(root: &Path, kraken: bool) -> [Vec<PathBuf>; 4] {
    let paths = |names: &[&str]| names.iter().map(|name| root.join(name)).collect::<Vec<_>>();
    let mut extraction = paths(&[
        "legal-pdf-core/Cargo.toml",
        "legal-pdf-core/src/lib.rs",
        "legal-pdf-core/src/model.rs",
        "legal-pdf-core/src/ocr_contract.rs",
    ]);
    extraction.extend(inspector_inputs(root));
    extraction.extend(crate_inputs(root, "extraction"));
    extraction.extend(crate_inputs(root, "extraction-processor"));
    let mut recognition = paths(&[
        "legal-pdf-core/Cargo.toml",
        "legal-pdf-core/src/ocr_contract.rs",
        "legal-pdf-core/src/asset.rs",
        "legal-pdf-core/src/ort_backend.rs",
        "legal-pdf-core/src/ort_runtime.rs",
    ]);
    recognition.extend(crate_inputs(root, "ocr"));
    if kraken {
        recognition.push(root.join("rust/native/tesseract_layout.c"));
    }
    let mut document = paths(&[
        "rust/src/contract.rs",
        "rust/src/engine.rs",
        "rust/src/structure_engine.rs",
        "rust/src/supplied_ocr.rs",
        "legal-pdf-core/Cargo.toml",
        "legal-pdf-core/src/lib.rs",
        "legal-pdf-core/src/model.rs",
        "legal-pdf-core/src/analysis.rs",
        "legal-pdf-core/src/ocr_contract.rs",
        "legal-pdf-support/Cargo.toml",
        "legal-pdf-support/src/lib.rs",
        "legal-pdf-support/src/projection.rs",
        "legal-pdf-support/src/printed_paragraphs.rs",
        "legal-pdf-support/src/pairing_support.rs",
    ]);
    document.extend(crate_inputs(root, "structure"));
    document.extend(crate_inputs(root, "pairing"));
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
    // Fingerprint library build inputs, not CLI tools, documentation or generated outputs.
    let mut inputs = vec![
        root.join("Cargo.toml"),
        root.join("build.rs"),
        root.join("rust/src/lib.rs"),
        root.join("rust/src/structure_engine.rs"),
    ];
    if pdf {
        inputs.extend(inspector_inputs(&root));
        for module in ["contract", "engine", "supplied_ocr"] {
            inputs.push(root.join(format!("rust/src/{module}.rs")));
        }
    }
    if enabled("KRAKEN") {
        inputs.push(root.join("rust/native/tesseract_layout.c"));
    }
    for (active, name) in [
        (pdf || enabled("LANGUAGE"), "core"),
        (pdf, "extraction"),
        (pdf, "extraction-processor"),
        (enabled("LANGUAGE"), "language"),
        (enabled("OCR"), "ocr"),
        (pdf, "pairing"),
        (pdf, "structure"),
        (pdf, "support"),
    ] {
        if !active {
            continue;
        }
        inputs.extend(crate_inputs(&root, name));
    }
    let mut features: Vec<_> = env::vars_os()
        .filter_map(|(name, _)| {
            name.to_str()?
                .strip_prefix("CARGO_FEATURE_")
                .map(str::to_owned)
        })
        .collect();
    features.sort();
    println!(
        "cargo:rustc-env=LEGAL_PDF_ENGINE_SHA256={}",
        fingerprint(&root, &inputs, &features)
    );
    if pdf {
        for ((name, active, stage_features), inputs) in [
            ("EXTRACTION", true, vec![]),
            (
                "OCR",
                enabled("OCR"),
                features
                    .iter()
                    .filter(|name| ["OCR", "KRAKEN"].contains(&name.as_str()))
                    .cloned()
                    .collect(),
            ),
            ("DOCUMENT", true, vec![]),
            (
                "LAYOUT",
                enabled("PPDOC_FULL") || enabled("PPDOC_OPENVINO"),
                features
                    .iter()
                    .filter(|name| ["PPDOC_FULL", "PPDOC_OPENVINO"].contains(&name.as_str()))
                    .cloned()
                    .collect(),
            ),
        ]
        .into_iter()
        .zip(stage_inputs(&root, enabled("KRAKEN")))
        {
            let inputs = if active { inputs } else { vec![] };
            println!(
                "cargo:rustc-env=LEGAL_PDF_{name}_SHA256={}",
                fingerprint(&root, &inputs, &stage_features)
            );
        }
    }
}
