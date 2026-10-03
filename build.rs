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

fn main() {
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    // Fingerprint library build inputs, not CLI tools, documentation or generated outputs.
    let mut inputs = vec![
        root.join("Cargo.toml"),
        root.join("Cargo.lock"),
        root.join("build.rs"),
        root.join("rust/native/tesseract_layout.c"),
    ];
    for module in [
        "lib",
        "contract",
        "engine",
        "structure_engine",
        "supplied_ocr",
    ] {
        inputs.push(root.join(format!("rust/src/{module}.rs")));
    }
    for name in [
        "core",
        "extraction",
        "extraction-processor",
        "language",
        "ocr",
        "pairing",
        "structure",
        "support",
    ] {
        let crate_root = root.join(format!("legal-pdf-{name}"));
        inputs.push(crate_root.join("Cargo.toml"));
        inputs.push(crate_root.join("src"));
        let build_script = crate_root.join("build.rs");
        if build_script.is_file() {
            inputs.push(build_script);
        }
    }
    let mut paths = Vec::new();
    for input in &inputs {
        println!("cargo:rerun-if-changed={}", input.display());
        files(input, &mut paths);
    }
    paths.sort();

    let mut digest = Sha256::new();
    for path in paths {
        let relative = path
            .strip_prefix(&root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        digest.update(relative.as_bytes());
        digest.update([0]);
        digest.update(fs::read(path).unwrap());
        digest.update([0]);
    }
    let mut features: Vec<_> = env::vars_os()
        .filter_map(|(name, _)| {
            name.to_str()?
                .strip_prefix("CARGO_FEATURE_")
                .map(str::to_owned)
        })
        .collect();
    features.sort();
    for feature in features {
        digest.update(feature.as_bytes());
        digest.update([0]);
    }
    let digest = digest.finalize();
    println!("cargo:rustc-env=LEGAL_PDF_ENGINE_SHA256={digest:x}");
}
