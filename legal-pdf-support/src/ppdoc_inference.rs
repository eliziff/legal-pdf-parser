//! Shared image preparation and decoding for native and WASM layout inference.
use crate::ppdoc_postprocess::PPDocDetection;
use image::RgbImage;

const INTER_RESIZE_COEF_SCALE: f32 = 2048.0;
#[derive(Clone, Copy)]
struct CubicSample {
    source: [u32; 4],
    coefficients: [i16; 4],
}

pub fn resize_opencv_cubic_nchw(
    image: &RgbImage,
    width: u32,
    height: u32,
    normalization_scale: f32,
    mean: [f32; 3],
    std: [f32; 3],
) -> Vec<f32> {
    let x_samples = cubic_samples(image.width(), width);
    let y_samples = cubic_samples(image.height(), height);
    let source = image.as_raw();
    let source_width = image.width() as usize;
    let width = width as usize;
    let height = height as usize;
    let plane = width * height;
    let mut output = vec![0.0_f32; 3 * plane];
    let coefficient_scale = 1.0 / (INTER_RESIZE_COEF_SCALE * INTER_RESIZE_COEF_SCALE);
    for (y, y_sample) in y_samples.iter().copied().enumerate() {
        for (x, x_sample) in x_samples.iter().copied().enumerate() {
            let destination = y * width + x;
            for channel in 0..3 {
                let mut horizontal = [0_i32; 4];
                for (row, source_y) in y_sample.source.iter().enumerate() {
                    horizontal[row] = x_sample
                        .source
                        .iter()
                        .zip(x_sample.coefficients)
                        .map(|(source_x, coefficient)| {
                            let offset = ((*source_y as usize * source_width + *source_x as usize)
                                * 3)
                                + channel;
                            i32::from(source[offset]) * i32::from(coefficient)
                        })
                        .sum();
                }
                let weighted = (horizontal[0] as f32).mul_add(
                    f32::from(y_sample.coefficients[0]) * coefficient_scale,
                    (horizontal[1] as f32).mul_add(
                        f32::from(y_sample.coefficients[1]) * coefficient_scale,
                        (horizontal[2] as f32).mul_add(
                            f32::from(y_sample.coefficients[2]) * coefficient_scale,
                            horizontal[3] as f32
                                * f32::from(y_sample.coefficients[3])
                                * coefficient_scale,
                        ),
                    ),
                );
                let value = weighted.round_ties_even().clamp(0.0, 255.0);
                output[channel * plane + destination] =
                    (value * normalization_scale - mean[channel]) / std[channel];
            }
        }
    }
    output
}

fn cubic_samples(source_size: u32, target_size: u32) -> Vec<CubicSample> {
    let scale = f64::from(source_size) / f64::from(target_size);
    (0..target_size)
        .map(|destination| {
            let mut fraction = ((f64::from(destination) + 0.5) * scale - 0.5) as f32;
            let base = fraction.floor() as i32;
            fraction -= base as f32;
            let coefficients = cubic_coefficients(fraction).map(|value| {
                (value * INTER_RESIZE_COEF_SCALE)
                    .round_ties_even()
                    .clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16
            });
            let maximum = source_size.saturating_sub(1) as i32;
            CubicSample {
                source: std::array::from_fn(|index| {
                    (base - 1 + index as i32).clamp(0, maximum) as u32
                }),
                coefficients,
            }
        })
        .collect()
}

fn cubic_coefficients(x: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let x1 = x + 1.0;
    let inverse = 1.0 - x;
    let first = ((A * x1 - 5.0 * A) * x1 + 8.0 * A) * x1 - 4.0 * A;
    let second = ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let third = ((A + 2.0) * inverse - (A + 3.0)) * inverse * inverse + 1.0;
    [first, second, third, 1.0 - first - second - third]
}

pub fn postprocess(
    values: &[f32],
    row_width: usize,
    count: usize,
    labels: &[String],
    image_width: u32,
    image_height: u32,
    threshold: f32,
) -> Vec<PPDocDetection> {
    let mut detections: Vec<(f32, PPDocDetection)> = values
        .chunks_exact(row_width)
        .take(count)
        .enumerate()
        .filter_map(|(index, row)| {
            let label_id = row[0] as isize;
            if label_id < 0 || row[1] <= threshold || label_id as usize >= labels.len() {
                return None;
            }
            let bbox = [
                row[2].round_ties_even().clamp(0.0, image_width as f32),
                row[3].round_ties_even().clamp(0.0, image_height as f32),
                row[4].round_ties_even().clamp(0.0, image_width as f32),
                row[5].round_ties_even().clamp(0.0, image_height as f32),
            ];
            (bbox[2] > bbox[0] && bbox[3] > bbox[1]).then(|| {
                (
                    if row_width >= 7 { row[6] } else { index as f32 },
                    PPDocDetection {
                        label_id: label_id as usize,
                        label: labels[label_id as usize].clone(),
                        score: row[1],
                        bbox,
                        order: None,
                    },
                )
            })
        })
        .collect();
    if row_width >= 7 {
        detections.sort_by(|left, right| left.0.total_cmp(&right.0));
    }
    if detections.len() > 1 {
        let area_threshold = if image_width > image_height {
            0.82
        } else {
            0.93
        };
        let filtered: Vec<_> = detections
            .iter()
            .filter(|(_, detection)| {
                detection.label != "image"
                    || (detection.bbox[2] - detection.bbox[0])
                        * (detection.bbox[3] - detection.bbox[1])
                        <= area_threshold * image_width as f32 * image_height as f32
            })
            .cloned()
            .collect();
        if !filtered.is_empty() {
            detections = filtered;
        }
    }
    let mut next_order = 1;
    detections
        .into_iter()
        .map(|(_, mut detection)| {
            if !skips_reading_order(&detection.label) {
                detection.order = Some(next_order);
                next_order += 1;
            }
            detection
        })
        .collect()
}

fn skips_reading_order(label: &str) -> bool {
    matches!(
        label,
        "figure_title"
            | "vision_footnote"
            | "image"
            | "chart"
            | "table"
            | "header"
            | "header_image"
            | "footer"
            | "footer_image"
            | "footnote"
            | "aside_text"
    )
}
