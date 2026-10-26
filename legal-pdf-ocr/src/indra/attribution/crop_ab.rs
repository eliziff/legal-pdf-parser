//! Crop-only experiment on retained native pixels and detector proposals.
use super::super::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct CachedPage {
    width: u32,
    height: u32,
    turns: u8,
    raster_sha256: String,
    core_boxes: Vec<[usize; 4]>,
    lines: Vec<CachedLine>,
}

#[derive(Deserialize)]
struct CachedLine {
    bbox: [usize; 4],
    input_bbox: Option<[usize; 4]>,
    recovered: bool,
    text: String,
    kept: bool,
    reading: Option<CachedReading>,
}

#[derive(Deserialize)]
struct CachedReading {
    normalized_sha256: String,
}

fn pixel_hash(pixels: &[u8]) -> String {
    format!("{:x}", Sha256::digest(pixels))
}

fn padded(core: [usize; 4], width: usize, height: usize) -> [usize; 4] {
    [
        core[0].saturating_sub(10),
        core[1].saturating_sub(6),
        (core[2] + 11).min(width),
        (core[3] + 7).min(height),
    ]
}

fn bounded_padding(mut bounds: [usize; 4], core: [usize; 4], others: &[[usize; 4]]) -> [usize; 4] {
    for other in others {
        let horizontal = core[2].min(other[2]).saturating_sub(core[0].max(other[0]));
        // Short raised glyphs and neighbouring columns must not become line boundaries.
        if horizontal * 2 < (core[2] - core[0]).max(other[2] - other[0]) {
            continue;
        }
        if other[3] <= core[1] {
            bounds[1] = bounds[1].max((other[3] + core[1]).div_ceil(2));
        }
        if other[1] >= core[3] {
            bounds[3] = bounds[3].min((core[3] + other[1]) / 2);
        }
    }
    bounds
}

fn policy_checks() {
    let core = [20, 50, 120, 60];
    let original = [10, 44, 131, 67];
    debug_assert_eq!(
        bounded_padding(original, core, &[[20, 30, 120, 45], [20, 65, 120, 75]]),
        [10, 48, 131, 62]
    );
    debug_assert_eq!(
        bounded_padding(original, core, &[[200, 65, 300, 75]]),
        original
    );
    debug_assert_eq!(
        bounded_padding(original, core, &[[100, 40, 110, 45]]),
        original
    );
    debug_assert_eq!(
        bounded_padding(original, core, &[[20, 52, 120, 56], [20, 90, 120, 100]]),
        original
    );
}

impl IndraOcr {
    /// Reuse a frozen page and proposals. Only vertical padding varies; ordinary OCR is unchanged.
    pub fn attribution_crop_ab(
        &mut self,
        image_path: &Path,
        trace: &Value,
        review: &Path,
        white_margin_ab: bool,
    ) -> Result<Value> {
        policy_checks();
        let saved: CachedPage = serde_json::from_value(trace.clone())?;
        let image = image::open(image_path)
            .map_err(|e| Error::Message(e.to_string()))?
            .into_rgba8();
        if saved.turns != 0
            || image.dimensions() != (saved.width, saved.height)
            || pixel_hash(image.as_raw()) != saved.raster_sha256
        {
            return Err(Error::Message(
                "Cached page pixels/dimensions/orientation differ".into(),
            ));
        }
        let valid = |b: &[usize; 4]| {
            b[0] < b[2]
                && b[1] < b[3]
                && b[2] <= saved.width as usize
                && b[3] <= saved.height as usize
        };
        if saved.core_boxes.iter().any(|b| !valid(b))
            || saved
                .lines
                .iter()
                .any(|l| !valid(&l.bbox) || l.input_bbox.is_some_and(|b| !valid(&b)))
        {
            return Err(Error::Message("Invalid cached crop rectangle".into()));
        }
        let page = PagePixels::Rgba {
            pixels: image.as_raw(),
            width: image.width(),
            height: image.height(),
        };
        let mut baselines = Vec::new();
        let mut candidates = Vec::new();
        let mut candidate_slots = Vec::new();
        let mut rows = Vec::new();
        fs::create_dir_all(review).map_err(|e| Error::io(review, e))?;
        for (index, line) in saved.lines.iter().enumerate() {
            let make = |b: [usize; 4]| LineBox {
                left: b[0],
                top: b[1],
                right: b[2],
                bottom: b[3],
                recovered: line.recovered,
            };
            let input_bbox = line.input_bbox.unwrap_or(line.bbox);
            let baseline =
                prepare_page_line(page, make(input_bbox), self.width_scale, self.input_height)
                    .ok_or_else(|| Error::Message("Cached prepared line is now empty".into()))?;
            let hash = pixel_hash(&baseline.pixels);
            if line
                .reading
                .as_ref()
                .is_some_and(|r| r.normalized_sha256 != hash)
            {
                return Err(Error::Message(format!(
                    "Normalized baseline pixels differ at line {index}"
                )));
            }
            let core = if line.recovered {
                None
            } else {
                saved
                    .core_boxes
                    .iter()
                    .find(|&&b| padded(b, saved.width as usize, saved.height as usize) == line.bbox)
            };
            let bounds = if white_margin_ab {
                input_bbox
            } else {
                core.map_or(line.bbox, |&b| {
                    bounded_padding(line.bbox, b, &saved.core_boxes)
                })
            };
            let candidate = if white_margin_ab {
                prepare_rgba_line_with_vertical_margin(
                    image.as_raw(),
                    image.width() as usize,
                    make(bounds),
                    self.width_scale,
                    self.input_height,
                    0.1,
                )
            } else {
                prepare_page_line(page, make(bounds), self.width_scale, self.input_height)
            }
            .ok_or_else(|| Error::Message("Candidate core-preserving crop is empty".into()))?;
            let candidate_hash = pixel_hash(&candidate.pixels);
            let changed = candidate_hash != hash;
            if changed {
                for (name, prepared) in [("baseline", &baseline), ("candidate", &candidate)] {
                    GrayImage::from_raw(
                        prepared.width as u32,
                        self.input_height as u32,
                        prepared.pixels.clone(),
                    )
                    .unwrap()
                    .save(review.join(format!("line-{index}-{name}.png")))
                    .map_err(|e| Error::Message(e.to_string()))?;
                }
                candidate_slots.push(index);
                candidates.push(candidate);
            }
            rows.push(json!({"index":index,"bbox":line.bbox,"baseline_bbox":input_bbox,"candidate_bbox":bounds,
                "core_matched":core.is_some(),"bounds_changed":bounds!=input_bbox,"pixels_changed":changed,
                "baseline_sha256":hash,"candidate_sha256":candidate_hash,"baseline_hash_checked":line.reading.is_some()}));
            baselines.push(baseline);
        }
        let codec = self.codec.as_ref().unwrap();
        let (mut read, _) = Self::recognize_prepared_pages(
            &mut self.sessions,
            codec,
            self.batch_size,
            self.width_bucket,
            self.input_height,
            std::slice::from_ref(&baselines),
        )?;
        for (index, result) in read.remove(0).into_iter().enumerate() {
            let kept = !result.text.trim().is_empty()
                && (!saved.lines[index].recovered || result.confidence >= RECOVERED_CONFIDENCE);
            if result.text != saved.lines[index].text || kept != saved.lines[index].kept {
                return Err(Error::Message(format!(
                    "Frozen baseline reading differs at line {index}"
                )));
            }
            rows[index]["baseline"] =
                json!({"text":result.text,"confidence":result.confidence,"kept":kept});
            rows[index]["candidate"] = rows[index]["baseline"].clone();
        }
        if !candidates.is_empty() {
            let (mut read, _) = Self::recognize_prepared_pages(
                &mut self.sessions,
                codec,
                self.batch_size,
                self.width_bucket,
                self.input_height,
                std::slice::from_ref(&candidates),
            )?;
            for (index, result) in candidate_slots.into_iter().zip(read.remove(0)) {
                let kept = !result.text.trim().is_empty()
                    && (!saved.lines[index].recovered || result.confidence >= RECOVERED_CONFIDENCE);
                rows[index]["candidate"] =
                    json!({"text":result.text,"confidence":result.confidence,"kept":kept});
            }
        }
        Ok(
            json!({"schema":"ocr-crop-ab.page.v1","lines":rows,"policy":if white_margin_ab {
                "vertical white margin 0.2 versus 0.1; input rectangles frozen"
            } else { "vertical midpoint; at least half of both line widths overlap" }}),
        )
    }
}
