mod asset;
mod error;
pub mod model;
pub mod profile;
mod ocr_contract;
mod ort_backend;
#[cfg(feature = "ort-runtime")]
mod ort_runtime;
mod storage;
pub use asset::provider_asset_sha256;
pub use error::{Error, Result};
pub use model::*;
pub use ocr_contract::{OcrLine, OcrPageRequest, OcrPageResult, OcrWord, PdfOcrProvider};
pub use ort_backend::OrtBackend;
#[cfg(feature = "ort-runtime")]
#[doc(hidden)]
pub use ort_runtime::init as init_ort_runtime;
pub use storage::{
    atomic_write_with, python_json, read_gzip_json, write_gzip_bytes, write_gzip_json, write_json,
};

pub fn line_font_size(line: &model::Line) -> f64 {
    const INLINE_SPANS: usize = 32;
    let exclude_superscripts = line
        .spans
        .iter()
        .any(|span| span.size > 0.0 && !span.superscript);
    let eligible = || {
        line.spans
            .iter()
            .filter(|span| span.size > 0.0 && (!exclude_superscripts || !span.superscript))
    };
    let count = eligible().count();
    let mut inline = [(0.0, 0_usize); INLINE_SPANS];
    let mut heap;
    let weighted = if count <= INLINE_SPANS {
        for (slot, span) in inline.iter_mut().zip(eligible()) {
            *slot = (span.size, span.text.chars().count().clamp(1, 100));
        }
        &mut inline[..count]
    } else {
        heap = eligible()
            .map(|span| (span.size, span.text.chars().count().clamp(1, 100)))
            .collect::<Vec<_>>();
        &mut heap
    };
    weighted.sort_by(|left, right| left.0.total_cmp(&right.0));
    let target = weighted.iter().map(|(_, count)| count).sum::<usize>() / 2;
    let mut seen = 0;
    weighted
        .iter()
        .find_map(|(size, count)| {
            seen += *count;
            (seen > target).then_some(*size)
        })
        .unwrap_or(0.0)
}

pub fn union_bbox(boxes: impl IntoIterator<Item = [f64; 4]>) -> [f64; 4] {
    let mut values = boxes.into_iter();
    let Some(first) = values.next() else {
        return [0.0; 4];
    };
    values.fold(first, |mut result, value| {
        result[0] = result[0].min(value[0]);
        result[1] = result[1].min(value[1]);
        result[2] = result[2].max(value[2]);
        result[3] = result[3].max(value[3]);
        result
    })
}
