#[cfg(feature = "kraken")]
use std::path::PathBuf;

fn main() {
    #[cfg(feature = "kraken")]
    build_layout_shim();
}

#[cfg(feature = "kraken")]
fn build_layout_shim() {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let root = if manifest.join("rust/native").is_dir() {
        manifest
    } else {
        manifest.join("../../..")
    };
    let source = root.join("rust/native/tesseract_layout.c");
    println!("cargo:rerun-if-changed={}", source.display());
    cc::Build::new()
        .file(source)
        .warnings(true)
        .compile("legalpdf_tesseract_layout");
}
