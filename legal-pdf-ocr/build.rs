#[cfg(feature = "ocr")]
use std::path::PathBuf;

fn main() {
    #[cfg(feature = "ocr")]
    build_layout_shim();
}

#[cfg(feature = "ocr")]
fn build_layout_shim() {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest
        .parent()
        .expect("core crate has a workspace parent");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let source = root.join("rust/native/gpu_adapter.cpp");
        println!("cargo:rerun-if-changed={}", source.display());
        cc::Build::new()
            .cpp(true)
            .file(source)
            .warnings(true)
            .compile("legalpdf_gpu_adapter");
        println!("cargo:rustc-link-lib=dxgi");
        println!("cargo:rustc-link-lib=d3d12");
    }
}
