//! Which way up a page's print stands, read before the page is recognized: PaddlePaddle's
//! PP-LCNet_x1_0_doc_ori document-orientation classifier (Apache-2.0; tools/model_manifests).
//! A page photocopied or printed sideways or upside down is turned upright, recognized, and its
//! lines are placed back on the page as it is drawn, so highlights and pinpoints land on the print.

use image::{imageops, RgbaImage};
use legal_pdf_core::{Error, OcrLine, Result};
use ort::{session::Session, value::TensorRef};
use std::path::Path;

/// How sure the classifier must be that a page is turned. On 90 upright scanned court-record and
/// court-practice pages it called three turned, at 0.26 to 0.41; it called the 12 sideways pages of
/// two scanned books turned at 0.74 and more, and 252 of the 270 upright pages turned a quarter, a
/// half and three quarters at 0.6 and more. A blank page reads 0.26 for every class.
const TURNED_CONFIDENCE: f32 = 0.6;

pub(crate) struct Orientation {
    session: Session,
}

impl Orientation {
    pub(crate) fn open(model: &Path) -> Result<Self> {
        let session = Session::builder()
            .and_then(|builder| builder.with_intra_threads(1))
            .and_then(|builder| builder.commit_from_file(model))
            .map_err(|error| Error::Message(format!("orientation model: {error}")))?;
        Ok(Self { session })
    }

    /// The quarter turns counter-clockwise that stand the page upright: 0 for an upright page, or
    /// one the classifier is not sure of.
    pub(crate) fn upright_turns(&mut self, page: &RgbaImage) -> Result<u8> {
        // As the classifier was trained: the short side scaled to 256, the middle 224 square,
        // ImageNet's mean and deviation per channel.
        let scale = 256.0 / page.width().min(page.height()) as f64;
        let (width, height) = (
            ((page.width() as f64 * scale).round() as u32).max(224),
            ((page.height() as f64 * scale).round() as u32).max(224),
        );
        let scaled = imageops::resize(page, width, height, imageops::FilterType::Triangle);
        let (left, top) = ((width - 224) / 2, (height - 224) / 2);
        const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
        const DEVIATION: [f32; 3] = [0.229, 0.224, 0.225];
        let mut input = vec![0f32; 3 * 224 * 224];
        for y in 0..224 {
            for x in 0..224 {
                let pixel = scaled.get_pixel(left + x, top + y).0;
                for channel in 0..3 {
                    input[(channel * 224 + y as usize) * 224 + x as usize] =
                        (f32::from(pixel[channel]) / 255.0 - MEAN[channel]) / DEVIATION[channel];
                }
            }
        }
        let tensor = TensorRef::from_array_view(([1usize, 3, 224, 224], input.as_slice()))
            .map_err(|error| Error::Message(format!("orientation input: {error}")))?;
        let outputs = self
            .session
            .run(ort::inputs![tensor])
            .map_err(|error| Error::Message(format!("orientation: {error}")))?;
        let (_, scores) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|error| Error::Message(format!("orientation output: {error}")))?;
        // Its classes: the print turned 0, 90, 180 and 270 degrees counter-clockwise.
        let (class, confidence) =
            scores
                .iter()
                .copied()
                .enumerate()
                .fold((0, f32::MIN), |best, (class, score)| {
                    if score > best.1 {
                        (class, score)
                    } else {
                        best
                    }
                });
        Ok(if class != 0 && confidence >= TURNED_CONFIDENCE {
            class as u8
        } else {
            0
        })
    }
}

/// The page turned `turns` quarter turns counter-clockwise.
pub(crate) fn turn(page: &RgbaImage, turns: u8) -> RgbaImage {
    match turns % 4 {
        1 => imageops::rotate270(page),
        2 => imageops::rotate180(page),
        3 => imageops::rotate90(page),
        _ => page.clone(),
    }
}

/// Places lines read from a page turned `turns` quarter turns counter-clockwise back on the page as
/// drawn, `width` by `height` pixels.
pub(crate) fn place_back(lines: &mut [OcrLine], turns: u8, width: f64, height: f64) {
    if turns % 4 == 0 {
        return;
    }
    // A point of the turned page, on the page as drawn.
    let point = |[x, y]: [f64; 2]| match turns % 4 {
        1 => [width - y, x],
        2 => [width - x, height - y],
        3 => [y, height - x],
        _ => [x, y],
    };
    let rectangle = |[x0, y0, x1, y1]: [f64; 4]| {
        let ([a, b], [c, d]) = (point([x0, y0]), point([x1, y1]));
        [a.min(c), b.min(d), a.max(c), b.max(d)]
    };
    for line in lines {
        line.bbox = rectangle(line.bbox);
        for word in &mut line.words {
            word.bbox = rectangle(word.bbox);
        }
        for value in line.baseline.iter_mut().chain(&mut line.boundary) {
            *value = point(*value);
        }
    }
}
