use crate::indra::{open_session, ort_error, IndraBackend, LineBox};
use image::{imageops, GrayImage};
use legal_pdf_core::{Error, Result};
use ort::{
    session::{builder::PrepackedWeights, Session},
    value::TensorRef,
};
use std::path::Path;

pub(crate) enum LineDetector {
    Indra(IndraDetector),
}

impl LineDetector {
    pub fn indra(
        path: &Path,
        side: usize,
        backend: IndraBackend,
        device: &str,
        independent_pool: bool,
    ) -> Result<Self> {
        Ok(Self::Indra(IndraDetector {
            session: open_session(
                path,
                1,
                backend,
                device,
                true,
                &PrepackedWeights::new(),
                independent_pool,
            )?,
            side,
        }))
    }

    pub fn lines(&mut self, image: &GrayImage) -> Result<Vec<LineBox>> {
        match self {
            Self::Indra(detector) => detector.lines(image),
        }
    }

    pub fn lines_rgba(&mut self, pixels: &[u8], width: u32, height: u32) -> Result<Vec<LineBox>> {
        match self {
            Self::Indra(detector) => {
                let gray = pixels
                    .chunks_exact(4)
                    .map(|p| {
                        ((299 * u32::from(p[0])
                            + 587 * u32::from(p[1])
                            + 114 * u32::from(p[2])
                            + 500)
                            / 1000) as u8
                    })
                    .collect();
                detector.lines(
                    &GrayImage::from_raw(width, height, gray)
                        .expect("page dimensions match RGBA pixels"),
                )
            }
        }
    }
}

pub(crate) struct IndraDetector {
    session: Session,
    side: usize,
}

impl IndraDetector {
    fn lines(&mut self, image: &GrayImage) -> Result<Vec<LineBox>> {
        let (width, height) = image.dimensions();
        let scale = self.side as f64 / f64::from(width.max(height));
        let resized_width = (f64::from(width) * scale).round().max(1.0) as usize;
        let resized_height = (f64::from(height) * scale).round().max(1.0) as usize;
        let input_width = resized_width.div_ceil(32) * 32;
        let input_height = resized_height.div_ceil(32) * 32;
        let resized = imageops::resize(
            image,
            resized_width as u32,
            resized_height as u32,
            imageops::FilterType::Triangle,
        );
        let mut input = vec![0.0_f32; input_width * input_height];
        for y in 0..resized_height {
            for x in 0..resized_width {
                input[y * input_width + x] =
                    1.0 - f32::from(resized[(x as u32, y as u32)][0]) / 255.0;
            }
        }
        let tensor =
            TensorRef::from_array_view(([1, 1, input_height, input_width], input.as_slice()))
                .map_err(ort_error)?;
        let name = self.session.inputs[0].name.clone();
        let outputs = self
            .session
            .run(ort::inputs![name => tensor])
            .map_err(ort_error)?;
        let (shape, values) = outputs[0].try_extract_tensor::<f32>().map_err(ort_error)?;
        if shape.len() != 4 || shape[0] != 1 || shape[1] != 3 || shape[2] <= 0 || shape[3] <= 0 {
            return Err(Error::Message(format!(
                "Indra detector returned unexpected dimensions {shape:?}"
            )));
        }
        let mut boxes = decode(
            values,
            shape[3] as usize,
            shape[2] as usize,
            scale,
            width as usize,
            height as usize,
        );
        xycut(&mut boxes);
        let boxes = crate::uncovered::recover_uncovered_lines(
            image.as_raw(),
            width as usize,
            height as usize,
            1,
            boxes,
        );
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
) -> Vec<LineBox> {
    let plane = width * height;
    let mut mask: Vec<u8> = data[..plane]
        .iter()
        .map(|value| u8::from(*value > 0.0))
        .collect();
    let mut queue = Vec::new();
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn core_bands_keep_diagonal_components_separate() {
        let mut data = vec![0.0; 3 * 5 * 4];
        for index in [1, 2, 8, 9] {
            data[index] = 1.0;
            data[20 + index] = 1.0;
            data[40 + index] = 2.0;
        }
        let boxes = decode(&data, 5, 4, 1.0, 20, 20);
        assert_eq!(boxes.len(), 2);
        assert_eq!((boxes[0].top, boxes[0].bottom), (0, 3));
        assert_eq!((boxes[1].top, boxes[1].bottom), (2, 5));
    }
    #[test]
    fn spanning_heading_precedes_each_column() {
        let rect = |left, top, right, bottom| LineBox {
            left,
            top,
            right,
            bottom,
            recovered: false,
        };
        let mut lines = vec![
            rect(60, 30, 90, 40),
            rect(0, 30, 40, 40),
            rect(0, 0, 100, 10),
            rect(60, 15, 90, 25),
            rect(0, 15, 40, 25),
        ];
        xycut(&mut lines);
        assert_eq!(
            lines
                .iter()
                .map(|line| (line.left, line.top))
                .collect::<Vec<_>>(),
            [(0, 0), (0, 15), (0, 30), (60, 15), (60, 30)]
        );
    }
}
