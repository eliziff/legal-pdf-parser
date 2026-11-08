mod analysis;
mod asset;
mod error;
pub mod authority;
pub mod model;
pub mod profile;
mod ocr_contract;
mod ocr_markers;
mod ort_backend;
#[cfg(feature = "ort-runtime")]
mod ort_runtime;
mod storage;
pub use analysis::{install_layout_analysis, layout_analysis};
#[cfg(feature = "structure-analysis")]
pub use analysis::CHECKOUT_ENGINE;
pub use asset::provider_asset_sha256;
pub use error::{Error, FailureKind, Result};
pub use authority::{AuthoritySource, AuthorityTextUnit, SourceMapping};
pub use model::*;
pub use ocr_markers::{associate_ocr_markers, merge_marker_lines};
pub use ocr_contract::{OcrLine, OcrPageRequest, OcrPageResult, OcrWord, PdfOcrProvider, RasterReading};
pub use ort_backend::OrtBackend;
#[cfg(feature = "ort-runtime")]
#[doc(hidden)]
pub use ort_runtime::init as init_ort_runtime;
pub use storage::{read_gzip_json, write_gzip_json};


/// The regular expression `pattern`, compiled once, at its first use (the caller depends on `regex`).
#[macro_export]
macro_rules! regex {
    ($pattern:expr) => {{
        static RE: ::std::sync::OnceLock<::regex::Regex> = ::std::sync::OnceLock::new();
        RE.get_or_init(|| ::regex::Regex::new($pattern).expect("regex"))
    }};
}

/// The shape of a line's text: its length, its share of capitals, figures, punctuation and spaces,
/// how it opens and how it ends.
pub fn text_shape(text: &str) -> [f64; 15] {
    let chars: Vec<char> = text.trim().chars().collect();
    let n = chars.len().max(1) as f64;
    let letters = chars.iter().filter(|c| c.is_alphabetic()).count().max(1) as f64;
    let first = chars.first().copied().unwrap_or(' ');
    let last = chars.last().copied().unwrap_or(' ');
    let flag = |b: bool| f64::from(u8::from(b));
    [
        (1.0 + chars.len() as f64).ln() / 5.0,
        chars.iter().filter(|c| c.is_uppercase()).count() as f64 / letters,
        chars.iter().filter(|c| c.is_ascii_digit()).count() as f64 / n,
        chars.iter().filter(|c| c.is_ascii_punctuation()).count() as f64 / n,
        chars.iter().filter(|c| c.is_whitespace()).count() as f64 / n,
        flag(first.is_ascii_digit()),
        flag(first.is_uppercase()),
        flag(first.is_lowercase()),
        flag("([{".contains(first)),
        flag("•·▪–—-*†‡§¶".contains(first)),
        flag(last == '.'),
        flag(last == ':'),
        flag(last == ',' || last == ';'),
        flag(last == '-' || last == '\u{00AD}'),
        flag(last.is_ascii_digit()),
    ]
}
