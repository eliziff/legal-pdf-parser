use crate::detector_native::{DetectorNative, DetectorScratch};
use crate::indra::{open_session, ort_error, LineBox};
use image::GrayImage;
use legal_pdf_core::profile::measure;
use legal_pdf_core::{Error, OrtBackend, Result};
use ort::{
    session::{builder::PrepackedWeights, Session},
    value::TensorRef,
};
use std::path::Path;
use std::sync::Arc;

enum DetectorEngine {
    Native {
        model: Arc<DetectorNative>,
        scratch: DetectorScratch,
    },
    Ort {
        session: Session,
        input_name: String,
    },
}

/// Indra's line detector: a page's lines, in reading order.
pub(crate) struct LineDetector {
    engine: DetectorEngine,
    side: usize,
    input: Vec<f32>,
    scratch: crate::simd_kernel::DownsampleScratch,
    mask: Vec<u8>,
    queue: Vec<usize>,
}

impl LineDetector {
    pub fn open_group(
        path: &Path,
        count: usize,
        side: usize,
        backend: OrtBackend,
        device: &str,
        independent_pool: bool,
    ) -> Result<Vec<Self>> {
        let count = count.max(1);
        if backend == OrtBackend::Cpu {
            let model = Arc::new(DetectorNative::from_onnx_path(path).map_err(Error::Message)?);
            return Ok((0..count)
                .map(|_| Self {
                    engine: DetectorEngine::Native {
                        model: Arc::clone(&model),
                        scratch: DetectorScratch::default(),
                    },
                    side,
                    input: Vec::new(),
                    scratch: crate::simd_kernel::DownsampleScratch::default(),
                    mask: Vec::new(),
                    queue: Vec::new(),
                })
                .collect());
        }
        (0..count)
            .map(|_| {
                let session = open_session(path, 1, backend, device, true, &PrepackedWeights::new(), independent_pool)?;
                let input_name = session.inputs[0].name.clone();
                Ok(Self {
                    engine: DetectorEngine::Ort { session, input_name },
                    side,
                    input: Vec::new(),
                    scratch: crate::simd_kernel::DownsampleScratch::default(),
                    mask: Vec::new(),
                    queue: Vec::new(),
                })
            })
            .collect()
    }

    pub fn lines(&mut self, image: &GrayImage) -> Result<Vec<LineBox>> {
        let (width, height) = image.dimensions();
        let scale = self.side as f64 / f64::from(width.max(height));
        let resized_width = (f64::from(width) * scale).round().max(1.0) as usize;
        let resized_height = (f64::from(height) * scale).round().max(1.0) as usize;
        let input_width = resized_width.div_ceil(32) * 32;
        let input_height = resized_height.div_ceil(32) * 32;
        measure("ocr.layout.downsample", || {
            crate::simd_kernel::downsample_to_norm_f32(
                image,
                resized_width,
                resized_height,
                input_width,
                input_height,
                &mut self.input,
                &mut self.scratch,
            );
        });
        let raw_boxes = match &mut self.engine {
            DetectorEngine::Native { model, scratch } => {
                let out = measure("ocr.layout.nn", || model.run(&self.input, input_height, input_width, scratch));
                measure("ocr.layout.decode", || {
                    decode(
                        out,
                        input_width / 2,
                        input_height / 2,
                        scale,
                        width as usize,
                        height as usize,
                        &mut self.mask,
                        &mut self.queue,
                    )
                })
            }
            DetectorEngine::Ort { session, input_name } => {
                let tensor = TensorRef::from_array_view(([1, 1, input_height, input_width], self.input.as_slice()))
                    .map_err(ort_error)?;
                let outputs = measure("ocr.layout.nn", || {
                    session.run(ort::inputs![input_name.as_str() => tensor]).map_err(ort_error)
                })?;
                let (shape, values) = outputs[0].try_extract_tensor::<f32>().map_err(ort_error)?;
                if shape.len() != 4 || shape[0] != 1 || shape[1] != 3 || shape[2] <= 0 || shape[3] <= 0 {
                    return Err(Error::Message(format!(
                        "Indra detector returned unexpected dimensions {shape:?}"
                    )));
                }
                measure("ocr.layout.decode", || {
                    decode(
                        values,
                        shape[3] as usize,
                        shape[2] as usize,
                        scale,
                        width as usize,
                        height as usize,
                        &mut self.mask,
                        &mut self.queue,
                    )
                })
            }
        };
        let mut boxes = measure("ocr.layout.gutter", || {
            crate::gutter::split_across_gutters(image.as_raw(), width as usize, raw_boxes)
        });
        xycut(&mut boxes);
        let boxes = measure("ocr.layout.uncovered", || {
            crate::uncovered::recover_uncovered_lines(image.as_raw(), width as usize, height as usize, boxes)
        });
        Ok(
            order_footnotes(boxes, width as usize, height as usize)
                .into_iter()
                .map(|bbox| LineBox {
                    left: bbox.left.saturating_sub(10),
                    top: bbox.top.saturating_sub(6),
                    right: (bbox.right + 11).min(width as usize),
                    bottom: (bbox.bottom + 7).min(height as usize),
                    ..bbox
                })
                .collect(),
        )
    }
}

// Core-band decoding and XY-cut follow datalab-to/indra (Apache-2.0).
fn decode(
    data: &[f32],
    width: usize,
    height: usize,
    scale: f64,
    page_width: usize,
    page_height: usize,
    mask: &mut Vec<u8>,
    queue: &mut Vec<usize>,
) -> Vec<LineBox> {
    let plane = width * height;
    mask.clear();
    mask.extend(data[..plane].iter().map(|value| u8::from(*value > 0.0)));
    let mut boxes = Vec::new();
    for seed in 0..plane {
        if mask[seed] == 0 {
            continue;
        }
        queue.clear();
        queue.push(seed);
        mask[seed] = 0;
        let (mut head, mut left, mut right, mut top, mut bottom) = (0, width, 0, 0.0, 0.0);
        while head < queue.len() {
            let index = queue[head];
            head += 1;
            let (x, y) = (index % width, index / width);
            left = left.min(x);
            right = right.max(x + 1);
            top += (y as f64 + 0.5) * 2.0 - f64::from(data[plane + index]);
            bottom += (y as f64 + 0.5) * 2.0 + f64::from(data[plane * 2 + index]);
            for next in [
                x.checked_sub(1).map(|_| index - 1),
                (x + 1 < width).then_some(index + 1),
                y.checked_sub(1).map(|_| index - width),
                (y + 1 < height).then_some(index + width),
            ]
            .into_iter()
            .flatten()
            {
                if mask[next] > 0 {
                    mask[next] = 0;
                    queue.push(next);
                }
            }
        }
        if queue.len() < 2 {
            continue;
        }
        top /= queue.len() as f64;
        bottom /= queue.len() as f64;
        let extension = 0.15 * (bottom - top).max(1.0);
        let bbox = LineBox {
            left: (((left as f64 * 2.0 - extension) / scale).floor().max(0.0) as usize)
                .min(page_width),
            right: (((right as f64 * 2.0 + extension) / scale).ceil().max(0.0) as usize)
                .min(page_width),
            top: ((top / scale).floor().max(0.0) as usize).min(page_height),
            bottom: ((bottom / scale).ceil().max(0.0) as usize).min(page_height),
            recovered: false,
        };
        if bbox.width() > 0 && bbox.height() > 0 {
            boxes.push(bbox);
        }
    }
    boxes
}

fn xycut(lines: &mut [LineBox]) {
    if lines.len() < 2 {
        return;
    }
    let mut best = (0, 0, 0);
    for axis in 0..2 {
        lines.sort_by_key(|line| if axis == 0 { line.left } else { line.top });
        let mut end = if axis == 0 {
            lines[0].right
        } else {
            lines[0].bottom
        };
        for (index, line) in lines.iter().enumerate().skip(1) {
            let start = if axis == 0 { line.left } else { line.top };
            let gap = start.saturating_sub(end);
            if gap > 0 && (gap > best.0 || (gap == best.0 && axis == 1 && best.2 == 0)) {
                best = (gap, index, axis);
            }
            end = end.max(if axis == 0 { line.right } else { line.bottom });
        }
    }
    if best.0 == 0 {
        lines.sort_by_key(|line| (line.top, line.left));
        return;
    }
    lines.sort_by_key(|line| if best.2 == 0 { line.left } else { line.top });
    let (first, second) = lines.split_at_mut(best.1);
    xycut(first);
    xycut(second);
}

fn median(mut values: Vec<usize>) -> usize {
    values.sort_unstable();
    values[values.len() / 2]
}

fn footer_start(mut lines: Vec<LineBox>, page_height: usize) -> Option<usize> {
    lines.sort_by_key(|line| line.top);
    if lines.len() < 8 {
        return None;
    }
    let body_end = (lines.len() * 65).div_ceil(100);
    let body_height = median(lines[..body_end].iter().map(|line| line.height()).collect());
    for index in (lines.len() * 45).div_ceil(100)..lines.len() - 2 {
        let gap = lines[index].top.saturating_sub(lines[index - 1].bottom);
        let tail_height = median(lines[index..].iter().map(|line| line.height()).collect());
        if lines[index].top > page_height * 55 / 100
            && gap >= 12_usize.max(body_height * 3 / 4)
            && tail_height * 100 <= body_height * 92
        {
            return Some(lines[index].top);
        }
    }
    None
}

// Reading order is kept except for the two-column footnote band, which reads column by column.
pub(crate) fn order_footnotes(boxes: Vec<LineBox>, page_width: usize, page_height: usize) -> Vec<LineBox> {
    if boxes.len() < 16 {
        return boxes;
    }
    let center = |line: &LineBox| (line.left + line.right) as f64 / 2.0;
    let left = boxes
        .iter()
        .copied()
        .filter(|line| center(line) < page_width as f64 * 0.48)
        .collect::<Vec<_>>();
    let right = boxes
        .iter()
        .copied()
        .filter(|line| center(line) > page_width as f64 * 0.52)
        .collect::<Vec<_>>();
    let (Some(left_start), Some(right_start)) = (
        footer_start(left, page_height),
        footer_start(right, page_height),
    ) else {
        return boxes;
    };
    if left_start.abs_diff(right_start) > page_height * 8 / 100 {
        return boxes;
    }
    let is_footer = |line: &LineBox| {
        let middle = center(line);
        (middle < page_width as f64 * 0.48 && line.top >= left_start)
            || (middle > page_width as f64 * 0.52 && line.top >= right_start)
    };
    let mut output = boxes
        .iter()
        .copied()
        .filter(|line| !is_footer(line))
        .collect::<Vec<_>>();
    let mut footnotes = boxes.into_iter().filter(is_footer).collect::<Vec<_>>();
    footnotes.sort_by_key(|line| (center(line) > page_width as f64 / 2.0, line.top, line.left));
    output.extend(footnotes);
    output
}
