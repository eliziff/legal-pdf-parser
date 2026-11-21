mod asset;
mod error;
pub mod model;
pub mod parallel;
pub use legal_layout::profile;
mod ocr_contract;
mod ocr_markers;
mod ort_backend;
#[cfg(feature = "ort-runtime")]
mod ort_runtime;
#[cfg(feature = "render")]
pub mod render;
mod storage;
pub use asset::{canonical_file, pack_file, provider_asset_sha256, required_path};
pub use error::{Error, FailureKind, Result};
pub use legal_structure_model::{lever, AuthoritySource, AuthorityTextUnit, SourceMapping};
pub use model::*;
pub use ocr_markers::associate_ocr_markers;
pub use ocr_contract::{OcrLine, OcrPageRequest, OcrPageResult, OcrWord, PdfOcrProvider, RasterReading};
pub use ort_backend::OrtBackend;
#[cfg(feature = "ort-runtime")]
#[doc(hidden)]
pub use ort_runtime::init as init_ort_runtime;
pub use storage::{read_gzip_json, read_gzip_packed, unpack, write_gzip_json, write_gzip_packed};


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
    let text = text.trim();
    // (One pass over the characters, counting each kind.)
    let (mut count, mut letters, mut upper, mut digits, mut punctuation, mut spaces) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    for c in text.chars() {
        count += 1;
        letters += usize::from(c.is_alphabetic());
        upper += usize::from(c.is_uppercase());
        digits += usize::from(c.is_ascii_digit());
        punctuation += usize::from(c.is_ascii_punctuation());
        spaces += usize::from(c.is_whitespace());
    }
    let (n, letters) = (count.max(1) as f64, letters.max(1) as f64);
    let first = text.chars().next().unwrap_or(' ');
    let last = text.chars().next_back().unwrap_or(' ');
    let flag = |b: bool| f64::from(u8::from(b));
    [
        (1.0 + count as f64).ln() / 5.0,
        upper as f64 / letters,
        digits as f64 / n,
        punctuation as f64 / n,
        spaces as f64 / n,
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
