use crate::{Error, Result};
use legal_pdf_core::model::{Diagnostic, Line, Page, PdfExtractionMetadata, Span, Word};
use legal_pdf_core::{profile, union_bbox, OcrLine, OcrPageRequest, PdfOcrProvider};
use lopdf::{Document, Object, ObjectId};
use pdf_inspector::types::{FidelityGlyph, ItemType, PdfLine, TextItem, TextLine};
use pdf_inspector::PdfTypeResult;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::OnceLock;

fn prune_extraction_object(object: &mut Object) {
    if let Object::Array(values) = object {
        let nulls = (values.len() >= 4096).then(|| {
            values.iter().try_fold(0, |nulls, value| match value {
                Object::Null => Some(nulls + 1),
                Object::Reference(_) => Some(nulls),
                _ => None,
            })
        });
        if nulls
            .flatten()
            .is_some_and(|nulls| nulls * 20 >= values.len() * 19)
        {
            *object = Object::Null;
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PageGeometry {
    x0: f64,
    y0: f64,
    raw_width: f64,
    raw_height: f64,
    rotation: i32,
    width: f64,
    height: f64,
}

pub type PageGeometryMap = BTreeMap<u32, (ObjectId, PageGeometry)>;

fn number(value: &Object) -> Option<f64> {
    match value {
        Object::Integer(value) => Some(*value as f64),
        Object::Real(value) => Some(f64::from(*value)),
        _ => None,
    }
}

fn inherited_object<'a>(doc: &'a Document, mut id: ObjectId, key: &[u8]) -> Option<&'a Object> {
    for _ in 0..32 {
        let dictionary = doc.get_dictionary(id).ok()?;
        if let Ok(value) = dictionary.get(key) {
            return Some(value);
        }
        match dictionary.get(b"Parent") {
            Ok(Object::Reference(parent)) => id = *parent,
            _ => return None,
        }
    }
    None
}

fn resolve_array<'a>(doc: &'a Document, value: &'a Object) -> Option<&'a [Object]> {
    match value {
        Object::Array(values) => Some(values),
        Object::Reference(id) => match doc.get_object(*id).ok()? {
            Object::Array(values) => Some(values),
            _ => None,
        },
        _ => None,
    }
}

fn page_geometry(doc: &Document, id: ObjectId) -> PageGeometry {
    let bounds = inherited_object(doc, id, b"CropBox")
        .or_else(|| inherited_object(doc, id, b"MediaBox"))
        .and_then(|value| resolve_array(doc, value))
        .and_then(|values| {
            let mut numbers = values.iter().filter_map(number);
            let (x0, y0, x1, y1) = (
                numbers.next()?,
                numbers.next()?,
                numbers.next()?,
                numbers.next()?,
            );
            Some([x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1)])
        })
        .unwrap_or([0.0, 0.0, 612.0, 792.0]);
    let rotation = inherited_object(doc, id, b"Rotate")
        .and_then(number)
        .map_or(0, |value| value.round() as i32)
        .rem_euclid(360);
    let raw_width = (bounds[2] - bounds[0]).max(0.0);
    let raw_height = (bounds[3] - bounds[1]).max(0.0);
    let (width, height) = if matches!(rotation, 90 | 270) {
        (raw_height, raw_width)
    } else {
        (raw_width, raw_height)
    };
    PageGeometry {
        x0: bounds[0],
        y0: bounds[1],
        raw_width,
        raw_height,
        rotation,
        width,
        height,
    }
}

fn transform_point(geometry: PageGeometry, x: f64, y: f64) -> (f64, f64) {
    let x = x - geometry.x0;
    let y = y - geometry.y0;
    match geometry.rotation {
        90 => (y, geometry.raw_width - x),
        180 => (geometry.raw_width - x, geometry.raw_height - y),
        270 => (geometry.raw_height - y, x),
        _ => (x, y),
    }
}

fn transform_vector(geometry: PageGeometry, x: f32, y: f32) -> [f32; 2] {
    match geometry.rotation {
        90 => [y, -x],
        180 => [-x, -y],
        270 => [-y, x],
        _ => [x, y],
    }
}

fn transform_bbox(geometry: PageGeometry, bbox: [f32; 4]) -> ([f32; 4], bool) {
    let points = [
        transform_point(geometry, f64::from(bbox[0]), f64::from(bbox[1])),
        transform_point(geometry, f64::from(bbox[0]), f64::from(bbox[3])),
        transform_point(geometry, f64::from(bbox[2]), f64::from(bbox[1])),
        transform_point(geometry, f64::from(bbox[2]), f64::from(bbox[3])),
    ];
    let (mut x0, mut y0) = (f64::INFINITY, f64::INFINITY);
    let (mut x1, mut y1) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for (x, y) in points {
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    let visible = x1 >= 0.0 && x0 <= geometry.width && y1 >= 0.0 && y0 <= geometry.height;
    let x0 = x0.clamp(0.0, geometry.width);
    let y0 = y0.clamp(0.0, geometry.height);
    let x1 = x1.min(geometry.width).max(x0);
    let y1 = y1.min(geometry.height).max(y0);
    ([x0 as f32, y0 as f32, x1 as f32, y1 as f32], visible)
}

fn transform_fidelity_item(item: &mut TextItem, geometry: PageGeometry) -> bool {
    let Some(fidelity) = item.fidelity.as_mut() else {
        return false;
    };
    let baseline = transform_point(
        geometry,
        f64::from(fidelity.baseline[0]),
        f64::from(fidelity.baseline[1]),
    );
    fidelity.baseline = [baseline.0 as f32, baseline.1 as f32];
    fidelity.advance = transform_vector(geometry, fidelity.advance[0], fidelity.advance[1]);
    fidelity.em = transform_vector(geometry, fidelity.em[0], fidelity.em[1]);
    let source_glyph_count = fidelity.glyphs.len();
    fidelity.glyphs.retain_mut(|glyph| {
        let (bbox, visible) = transform_bbox(geometry, glyph.bbox);
        glyph.bbox = bbox;
        visible
    });
    let clipped_glyph = fidelity.glyphs.len() != source_glyph_count;
    if clipped_glyph {
        fidelity.text.clear();
        for glyph in &fidelity.glyphs {
            fidelity.text.push_str(&glyph.text);
        }
    }

    let points = [
        [
            fidelity.baseline[0] + fidelity.descender * fidelity.em[0],
            fidelity.baseline[1] + fidelity.descender * fidelity.em[1],
        ],
        [
            fidelity.baseline[0] + fidelity.ascender * fidelity.em[0],
            fidelity.baseline[1] + fidelity.ascender * fidelity.em[1],
        ],
        [
            fidelity.baseline[0] + fidelity.advance[0] + fidelity.descender * fidelity.em[0],
            fidelity.baseline[1] + fidelity.advance[1] + fidelity.descender * fidelity.em[1],
        ],
        [
            fidelity.baseline[0] + fidelity.advance[0] + fidelity.ascender * fidelity.em[0],
            fidelity.baseline[1] + fidelity.advance[1] + fidelity.ascender * fidelity.em[1],
        ],
    ];
    let (mut x0, mut y0) = (f32::INFINITY, f32::INFINITY);
    let (mut x1, mut y1) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for [x, y] in points {
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    let x0 = x0.clamp(0.0, geometry.width as f32);
    let y0 = y0.clamp(0.0, geometry.height as f32);
    let x1 = x1.min(geometry.width as f32).max(x0);
    let y1 = y1.min(geometry.height as f32).max(y0);
    let visible_glyph_bbox = clipped_glyph.then(|| {
        let mut boxes = fidelity.glyphs.iter().filter(|glyph| {
            !glyph.text.chars().all(char::is_whitespace)
                && glyph.bbox[2] - glyph.bbox[0] > 0.001
                && glyph.bbox[3] - glyph.bbox[1] > 0.001
        });
        let first = boxes.next()?;
        Some(boxes.fold(first.bbox, |mut bbox, glyph| {
            bbox[0] = bbox[0].min(glyph.bbox[0]);
            bbox[1] = bbox[1].min(glyph.bbox[1]);
            bbox[2] = bbox[2].max(glyph.bbox[2]);
            bbox[3] = bbox[3].max(glyph.bbox[3]);
            bbox
        }))
    });
    let bbox = visible_glyph_bbox.flatten().unwrap_or([x0, y0, x1, y1]);
    item.x = bbox[0];
    item.y = bbox[1];
    item.width = bbox[2] - bbox[0];
    item.height = bbox[3] - bbox[1];
    item.font_size = fidelity.em[0].hypot(fidelity.em[1]);
    item.text = std::mem::take(&mut fidelity.text);
    item.font = std::mem::take(&mut fidelity.font);
    item.is_italic = fidelity.flags & 2 != 0;
    item.is_bold = fidelity.flags & 16 != 0;
    true
}

fn transform_item(item: &mut TextItem, geometry: PageGeometry) {
    if transform_fidelity_item(item, geometry) {
        return;
    }
    let corners = [
        transform_point(geometry, f64::from(item.x), f64::from(item.y)),
        transform_point(geometry, f64::from(item.x + item.width), f64::from(item.y)),
        transform_point(geometry, f64::from(item.x), f64::from(item.y + item.height)),
        transform_point(
            geometry,
            f64::from(item.x + item.width),
            f64::from(item.y + item.height),
        ),
    ];
    let (mut x0, mut y0) = (f64::INFINITY, f64::INFINITY);
    let (mut x1, mut y1) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for (x, y) in corners {
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    item.x = x0.max(0.0) as f32;
    item.y = y0.max(0.0) as f32;
    item.width = (x1.min(geometry.width) - f64::from(item.x)).max(0.0) as f32;
    item.height = (y1.min(geometry.height) - f64::from(item.y)).max(0.0) as f32;
}

fn round3(value: f64) -> f64 {
    (value * 1_000.0).round() / 1_000.0
}

fn item_bbox(item: &TextItem, page_height: f64) -> [f64; 4] {
    let x = f64::from(item.x);
    let y = f64::from(item.y);
    let width = f64::from(item.width);
    let height = f64::from(item.height);
    [
        round3(x),
        round3(page_height - y - height),
        round3(x + width),
        round3(page_height - y),
    ]
}

fn same_painted_text(left: &TextItem, right: &TextItem) -> bool {
    let (Some(left_fidelity), Some(right_fidelity)) =
        (left.fidelity.as_ref(), right.fidelity.as_ref())
    else {
        return false;
    };
    if left.page != right.page
        || left.text != right.text
        || left.text.is_empty()
        || left_fidelity.resource != right_fidelity.resource
        || left_fidelity.flags != right_fidelity.flags
        || left_fidelity.glyphs.len() != right_fidelity.glyphs.len()
    {
        return false;
    }
    let tolerance = (left.font_size.max(right.font_size) * 0.04).clamp(0.01, 0.5);
    let close = |left: f32, right: f32| (left - right).abs() <= tolerance;
    left_fidelity
        .baseline
        .iter()
        .zip(right_fidelity.baseline)
        .all(|(left, right)| close(*left, right))
        && left_fidelity
            .advance
            .iter()
            .zip(right_fidelity.advance)
            .all(|(left, right)| close(*left, right))
        && left_fidelity
            .em
            .iter()
            .zip(right_fidelity.em)
            .all(|(left, right)| close(*left, right))
        && left_fidelity
            .glyphs
            .iter()
            .zip(&right_fidelity.glyphs)
            .all(|(left, right)| {
                left.text == right.text
                    && left
                        .bbox
                        .iter()
                        .zip(right.bbox)
                        .all(|(left, right)| close(*left, right))
            })
}

fn painted_glyph_exists_in(previous: &TextItem, current: &TextItem, glyph: &FidelityGlyph) -> bool {
    let (Some(previous_fidelity), Some(current_fidelity)) =
        (previous.fidelity.as_ref(), current.fidelity.as_ref())
    else {
        return false;
    };
    if previous.page != current.page
        || previous_fidelity.resource != current_fidelity.resource
        || previous_fidelity.flags != current_fidelity.flags
    {
        return false;
    }
    let tolerance = previous
        .font_size
        .max(current.font_size)
        .mul_add(0.02, 0.0)
        .clamp(0.01, 0.25);
    previous_fidelity.glyphs.iter().any(|candidate| {
        candidate.text == glyph.text
            && (candidate.bbox[0] - glyph.bbox[0]).abs() <= tolerance
            && (candidate.bbox[1] - glyph.bbox[1]).abs() <= tolerance
    })
}

fn deduplicate_painted_text(items: &mut Vec<TextItem>) {
    items.dedup_by(|current, previous| {
        same_painted_text(previous, current)
            || current.fidelity.as_ref().is_some_and(|fidelity| {
                let mut visible = fidelity
                    .glyphs
                    .iter()
                    .filter(|glyph| !glyph.text.chars().all(char::is_whitespace));
                let first = visible.next();
                first.is_some_and(|glyph| painted_glyph_exists_in(previous, current, glyph))
                    && visible.all(|glyph| painted_glyph_exists_in(previous, current, glyph))
            })
    });
}

fn assign_renderer_layout(items: &mut [TextItem]) {
    const BASE_MAX_DIST: f64 = 0.8;
    const PARAGRAPH_DIST: f64 = 1.5;
    const SPACE_DIST: f64 = 0.15;
    const SPACE_MAX_DIST: f64 = 0.8;

    let mut page = None;
    let mut block = 0u32;
    let mut line = 0u32;
    let mut has_text_block = false;
    let mut pen = None::<[f64; 2]>;
    let mut prior_direction = None::<[f64; 2]>;
    let mut line_start = None::<[f64; 2]>;
    let mut line_leading_size = None::<f64>;
    let mut line_has_small_leading_run = false;

    for item in items {
        if page != Some(item.page) {
            page = Some(item.page);
            block = 0;
            line = 0;
            has_text_block = false;
            pen = None;
            prior_direction = None;
            line_start = None;
            line_leading_size = None;
            line_has_small_leading_run = false;
        }
        if matches!(&item.item_type, ItemType::Image) {
            block = block.saturating_add(1);
            has_text_block = false;
            pen = None;
            prior_direction = None;
            line_start = None;
            line_leading_size = None;
            line_has_small_leading_run = false;
            continue;
        }
        if !matches!(&item.item_type, ItemType::Text) {
            continue;
        }
        let Some(fidelity) = item.fidelity.as_ref() else {
            continue;
        };
        let point = [
            f64::from(fidelity.baseline[0]),
            f64::from(fidelity.baseline[1]),
        ];
        let advance = [
            f64::from(fidelity.advance[0]),
            f64::from(fidelity.advance[1]),
        ];
        let size = f64::from(fidelity.em[0].hypot(fidelity.em[1])).max(0.001);
        let advance_length = advance[0].hypot(advance[1]);
        let direction = if advance_length > 0.001 {
            [advance[0] / advance_length, advance[1] / advance_length]
        } else if let Some(direction) = prior_direction {
            direction
        } else {
            [
                f64::from(fidelity.em[1]) / size,
                -f64::from(fidelity.em[0]) / size,
            ]
        };
        let mut new_paragraph = !has_text_block;
        let mut new_line = pen.is_none();
        if let (Some(prior_pen), Some(prior_direction)) = (pen, prior_direction) {
            if direction[0] * prior_direction[0] + direction[1] * prior_direction[1] < 0.999 {
                new_paragraph = true;
                new_line = true;
            } else {
                let delta = [point[0] - prior_pen[0], point[1] - prior_pen[1]];
                let spacing = (direction[0] * delta[0] + direction[1] * delta[1]) / size;
                let base_offset = (-direction[1] * delta[0] + direction[0] * delta[1]) / size;
                if base_offset.abs() < BASE_MAX_DIST {
                    new_line = !(spacing.abs() < SPACE_DIST
                        || (-SPACE_MAX_DIST..0.0).contains(&spacing)
                        || (0.0..SPACE_MAX_DIST).contains(&spacing));
                } else if base_offset.abs() <= PARAGRAPH_DIST {
                    new_line = true;
                    if let Some(start) = line_start {
                        let indent = direction[0] * (point[0] - start[0])
                            + direction[1] * (point[1] - start[1]);
                        // A raised, smaller line-initial run is a label, not
                        // the paragraph's indentation anchor. Keep its
                        // hanging continuation in the same text block.
                        new_paragraph |= indent > 0.5 && !line_has_small_leading_run;
                    }
                } else {
                    new_paragraph = true;
                    new_line = true;
                }
            }
        }
        if new_paragraph {
            block = block.saturating_add(1);
            has_text_block = true;
        }
        if new_line {
            line = line.saturating_add(1);
            line_start = Some(point);
            line_leading_size = Some(size);
            line_has_small_leading_run = false;
        } else if line_leading_size.is_some_and(|leading| size >= leading * 1.2) {
            line_has_small_leading_run = true;
        }
        if let Some(fidelity) = item.fidelity.as_mut() {
            fidelity.renderer_line = line;
            fidelity.renderer_block = block;
        }
        pen = Some([point[0] + advance[0], point[1] + advance[1]]);
        prior_direction = Some(direction);
    }
}

fn item_baseline(item: &TextItem) -> f64 {
    item.fidelity
        .as_ref()
        .map_or(f64::from(item.y), |fidelity| {
            f64::from(item.y - fidelity.descender.min(0.0) * item.font_size)
        })
}

fn normalize_text(value: &str) -> Cow<'_, str> {
    if !value.contains('\u{feff}') && !value.contains('\u{200b}') {
        return Cow::Borrowed(value);
    }
    let mut result = String::with_capacity(value.len());
    let mut characters = value
        .chars()
        .filter(|character| *character != '\u{feff}')
        .peekable();
    while let Some(character) = characters.next() {
        if character != '\u{200b}' {
            result.push(character);
            continue;
        }
        let mut count = 1;
        while characters.peek().is_some_and(|next| *next == '\u{200b}') {
            characters.next();
            count += 1;
        }
        if count >= 2 {
            result.push(' ');
        }
    }
    Cow::Owned(result)
}

#[derive(Debug)]
struct AssembledSpan {
    item_index: usize,
    bbox_start_item: usize,
    bbox_end_item: usize,
    start: usize,
    end: usize,
    byte_start: usize,
    byte_end: usize,
}

#[derive(Debug)]
struct AssembledLine {
    text: String,
    spans: Vec<AssembledSpan>,
}

fn same_renderer_span(left: &TextItem, right: &TextItem) -> bool {
    match (left.fidelity.as_ref(), right.fidelity.as_ref()) {
        (Some(left), Some(right)) => {
            left.resource == right.resource
                && left.flags == right.flags
                && left.text_rise == right.text_rise
                && left.em == right.em
        }
        _ => {
            left.font == right.font
                && left.font_size == right.font_size
                && left.is_bold == right.is_bold
                && left.is_italic == right.is_italic
        }
    }
}

fn same_source_text_object(left: &TextItem, right: &TextItem) -> bool {
    match (left.fidelity.as_ref(), right.fidelity.as_ref()) {
        (Some(left), Some(right)) => left.text_object == right.text_object,
        _ => true,
    }
}

/// Assemble the legal line contract from source spans. This deliberately does
/// not use `TextLine::text()`: its generic display heuristic suppresses spaces
/// around raised/small text, while the previous engine preserved an explicit
/// boundary whenever the PDF supplied one or the geometric gap reached 0.15em.
fn assemble_line(text_line: &TextLine) -> AssembledLine {
    let mut raw_text = String::new();
    let mut spans = Vec::<AssembledSpan>::new();
    let mut offset = 0;
    let mut previous_x1 = None;
    let mut previous_trailing_boundary = false;
    let mut pending_leading_whitespace = None::<(usize, usize, usize)>;
    let mut next_content = text_line
        .items
        .iter()
        .enumerate()
        .filter(|(_, item)| !item.text.trim().is_empty())
        .peekable();
    let mut previous_content = None;

    for (item_index, item) in text_line.items.iter().enumerate() {
        if item.text.trim().is_empty() {
            let previous = previous_content.map(|index| &text_line.items[index]);
            let next = next_content.peek().map(|(_, item)| *item);
            let extends_previous =
                previous.is_some_and(|candidate| same_renderer_span(candidate, item));
            if extends_previous || next.is_some_and(|candidate| same_renderer_span(candidate, item))
            {
                let start = offset;
                let byte_start = raw_text.len();
                let whitespace = normalize_text(&item.text);
                raw_text.push_str(&whitespace);
                offset += whitespace.chars().count();
                if extends_previous {
                    if let Some(previous) = spans.last_mut() {
                        previous.end = offset;
                        previous.byte_end = raw_text.len();
                        previous.bbox_end_item = item_index;
                    }
                } else {
                    pending_leading_whitespace.get_or_insert((start, byte_start, item_index));
                }
            }
            continue;
        }
        let _ = next_content.next();
        previous_content = Some(item_index);
        let leading_boundary =
            item.text.chars().find(|&character| character != '\u{feff}') == Some('\u{200b}');
        let trailing_boundary = item
            .text
            .chars()
            .rev()
            .find(|&character| character != '\u{feff}')
            == Some('\u{200b}');
        let span_text = normalize_text(&item.text);
        if span_text.is_empty() {
            previous_trailing_boundary |= trailing_boundary;
            continue;
        }

        let x0 = round3(f64::from(item.x));
        let x1 = round3(f64::from(item.x + item.width));
        let threshold = f64::from(item.font_size).max(10.0) * 0.15;
        if previous_x1.is_some_and(|prior| {
            !raw_text
                .chars()
                .next_back()
                .is_some_and(char::is_whitespace)
                && !span_text.chars().next().is_some_and(char::is_whitespace)
                && ((previous_trailing_boundary && leading_boundary) || x0 - prior >= threshold)
        }) {
            raw_text.push(' ');
            offset += 1;
        }

        previous_x1 = Some(x1);
        previous_trailing_boundary = trailing_boundary;
        let (start, byte_start, bbox_start_item) =
            pending_leading_whitespace
                .take()
                .unwrap_or((offset, raw_text.len(), item_index));
        raw_text.push_str(&span_text);
        offset += span_text.chars().count();
        spans.push(AssembledSpan {
            item_index,
            bbox_start_item,
            bbox_end_item: item_index,
            start,
            end: offset,
            byte_start,
            byte_end: raw_text.len(),
        });
    }

    let leading_bytes = raw_text.len() - raw_text.trim_start().len();
    let leading = raw_text[..leading_bytes].chars().count();
    let trimmed_end = raw_text.trim_end().len().max(leading_bytes);
    raw_text.truncate(trimmed_end);
    raw_text.replace_range(..leading_bytes, "");
    let text_len = raw_text.chars().count();
    for span in &mut spans {
        span.start = span.start.saturating_sub(leading).min(text_len);
        span.end = span
            .end
            .saturating_sub(leading)
            .max(span.start)
            .min(text_len);
        span.byte_start = span
            .byte_start
            .saturating_sub(leading_bytes)
            .min(raw_text.len());
        span.byte_end = span
            .byte_end
            .saturating_sub(leading_bytes)
            .max(span.byte_start)
            .min(raw_text.len());
    }
    spans.retain(|span| span.start < span.end);
    AssembledLine {
        text: raw_text,
        spans,
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let middle = values.len() / 2;
    let (_, right, _) = values.select_nth_unstable_by(middle, f64::total_cmp);
    let right = *right;
    if values.len().is_multiple_of(2) {
        let left = values[..middle]
            .iter()
            .copied()
            .max_by(f64::total_cmp)
            .unwrap();
        (left + right) / 2.0
    } else {
        right
    }
}

fn is_superscript(item: &TextItem, body_size: f64, baseline: f64, inline: bool) -> bool {
    let literal = item.text.trim().chars().all(|character| {
        matches!(
            character,
            '⁰' | '¹' | '²' | '³' | '⁴' | '⁵' | '⁶' | '⁷' | '⁸' | '⁹'
        )
    });
    literal
        || (inline
            && body_size > 0.0
            && item.font_size > 0.0
            && f64::from(item.font_size) <= body_size * 0.82
            && item_baseline(item) >= baseline + (body_size * 0.08).max(0.5))
}

fn line_spans(
    text_line: &TextLine,
    assembled: &AssembledLine,
    page_height: f64,
    id: &str,
) -> Vec<Span> {
    let body_size = text_line
        .items
        .iter()
        .filter(|item| item.font_size > 0.0 && !item.text.trim().is_empty())
        .map(|item| f64::from(item.font_size))
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    let baseline = median(
        text_line
            .items
            .iter()
            .filter(|item| {
                !item.text.trim().is_empty()
                    && (body_size <= 0.0 || f64::from(item.font_size) >= body_size * 0.85)
            })
            .map(item_baseline)
            .collect(),
    );
    let mut spans = Vec::<Span>::new();
    let mut previous_item_index = None::<usize>;
    let mut previous_byte_end = 0;
    for source_span in &assembled.spans {
        let item = &text_line.items[source_span.item_index];
        let start_byte = source_span.byte_start;
        let end_byte = source_span.byte_end;
        let bbox = union_bbox(
            text_line.items[source_span.bbox_start_item..=source_span.bbox_end_item]
                .iter()
                .map(|item| item_bbox(item, page_height)),
        );
        if let (Some(previous), Some(prior_item_index)) = (spans.last_mut(), previous_item_index) {
            let gap_start = previous_byte_end;
            let gap_end = start_byte;
            if same_renderer_span(&text_line.items[prior_item_index], item)
                && same_source_text_object(&text_line.items[prior_item_index], item)
                && assembled.text[gap_start..gap_end]
                    .chars()
                    .all(char::is_whitespace)
            {
                previous.text.push_str(&assembled.text[gap_start..end_byte]);
                previous.end = source_span.end;
                previous.bbox = union_bbox([previous.bbox, bbox]);
                previous_item_index = Some(source_span.item_index);
                previous_byte_end = end_byte;
                continue;
            }
        }
        let superscript = is_superscript(item, body_size, baseline, source_span.start > 0);
        let span = Span {
            id: format!("{id}-s{:03}", spans.len() + 1),
            text: assembled.text[start_byte..end_byte].to_owned(),
            bbox,
            font: item.font.clone(),
            size: round3(f64::from(item.font_size)),
            flags: item.fidelity.as_ref().map_or_else(
                || if item.is_italic { 2 } else { 0 } | if item.is_bold { 16 } else { 0 },
                |fidelity| fidelity.flags,
            ) | u32::from(superscript),
            superscript,
            start: source_span.start,
            end: source_span.end,
        };
        spans.push(span);
        previous_item_index = Some(source_span.item_index);
        previous_byte_end = end_byte;
    }
    spans
}

#[derive(Debug)]
struct PositionedGlyph {
    start: usize,
    end: usize,
    bbox: [f64; 4],
}

fn line_glyphs(
    text_line: &TextLine,
    assembled: &AssembledLine,
    page_height: f64,
) -> Vec<PositionedGlyph> {
    let mut result = Vec::new();
    for source_span in &assembled.spans {
        let item = &text_line.items[source_span.item_index];
        let Some(fidelity) = item.fidelity.as_ref() else {
            continue;
        };
        let start_byte = source_span.byte_start;
        let end_byte = source_span.byte_end;
        let target = &assembled.text[start_byte..end_byte];
        let item_text = normalize_text(&item.text);
        let Some(target_byte) = item_text.find(target) else {
            continue;
        };
        let skipped = item_text[..target_byte].chars().count();
        let mut local_offset = 0usize;
        for glyph in &fidelity.glyphs {
            let glyph_text = normalize_text(&glyph.text);
            let glyph_len = glyph_text.chars().count();
            let local_start = local_offset;
            local_offset += glyph_len;
            if glyph_len == 0 || glyph_text.chars().all(char::is_whitespace) {
                continue;
            }
            let start = source_span
                .start
                .saturating_add(local_start.saturating_sub(skipped));
            let end = source_span
                .start
                .saturating_add(local_offset.saturating_sub(skipped))
                .min(source_span.end);
            if start >= end || local_offset <= skipped {
                continue;
            }
            result.push(PositionedGlyph {
                start,
                end,
                bbox: [
                    round3(f64::from(glyph.bbox[0])),
                    round3(page_height - f64::from(glyph.bbox[3])),
                    round3(f64::from(glyph.bbox[2])),
                    round3(page_height - f64::from(glyph.bbox[1])),
                ],
            });
        }
    }
    result
}

fn line_words(
    text: &str,
    text_line: &TextLine,
    assembled: &AssembledLine,
    page_height: f64,
    id: &str,
) -> Vec<Word> {
    let glyphs = line_glyphs(text_line, assembled, page_height);
    if glyphs.is_empty() {
        return Vec::new();
    }
    debug_assert!(glyphs
        .windows(2)
        .all(|pair| pair[0].start <= pair[1].start && pair[0].end <= pair[1].end));
    let mut words = Vec::new();
    let mut start = None;
    let mut glyph_index = 0;
    for (char_offset, (byte, character)) in text
        .char_indices()
        .chain(std::iter::once((text.len(), ' ')))
        .enumerate()
    {
        if character.is_whitespace() {
            if let Some((begin, start)) = start.take() {
                let end = char_offset;
                while glyphs
                    .get(glyph_index)
                    .is_some_and(|glyph| glyph.end <= start)
                {
                    glyph_index += 1;
                }
                let mut boxes = glyphs[glyph_index..]
                    .iter()
                    .take_while(|glyph| glyph.start < end)
                    .map(|glyph| glyph.bbox);
                let Some(first) = boxes.next() else {
                    return Vec::new();
                };
                words.push(Word {
                    id: format!("{id}-w{:03}", words.len() + 1),
                    text: text[begin..byte].to_owned(),
                    bbox: union_bbox(std::iter::once(first).chain(boxes)),
                    start,
                    end,
                });
            }
        } else if start.is_none() {
            start = Some((byte, char_offset));
        }
    }
    words
}

fn model_line_font_size(line: &Line) -> f64 {
    let has_body = line
        .spans
        .iter()
        .any(|span| span.size > 0.0 && !span.superscript);
    let mut sizes: Vec<_> = line
        .spans
        .iter()
        .filter(|span| span.size > 0.0 && (!has_body || !span.superscript))
        .map(|span| (span.size, span.text.chars().count().clamp(1, 100)))
        .collect();
    if sizes.is_empty() {
        return 0.0;
    }
    sizes.sort_by(|left, right| left.0.total_cmp(&right.0));
    let count = sizes.iter().map(|(_, count)| count).sum::<usize>();
    let (left_index, right_index) = ((count - 1) / 2, count / 2);
    let mut seen = 0;
    let mut left_value = None;
    for (size, count) in sizes {
        seen += count;
        if left_value.is_none() && seen > left_index {
            left_value = Some(size);
        }
        if seen > right_index {
            return (left_value.unwrap_or(size) + size) / 2.0;
        }
    }
    unreachable!("weighted median has at least one value")
}

fn begins_with_note_label(text: &str) -> bool {
    let trimmed = text.trim_start();
    let count = trimmed
        .chars()
        .take_while(|character| character.is_ascii_digit())
        .count();
    if count == 0 || count > 3 {
        return matches!(
            trimmed.chars().next(),
            Some('*' | '\u{2217}' | '\u{f02a}' | '\u{2020}' | '\u{2021}')
        );
    }
    trimmed
        .chars()
        .nth(count)
        .is_none_or(|character| character.is_whitespace() || ".)]},:;-".contains(character))
}

fn separator_y(geometry: PageGeometry, lines: &[Line], rules: &[PdfLine]) -> Option<f64> {
    let candidates: Vec<(f64, f64)> = rules
        .iter()
        .map(|rule| {
            let first = transform_point(geometry, f64::from(rule.x1), f64::from(rule.y1));
            let second = transform_point(geometry, f64::from(rule.x2), f64::from(rule.y2));
            (
                (second.0 - first.0).abs(),
                geometry.height - (first.1 + second.1) / 2.0,
            )
        })
        .filter(|(length, y)| {
            *length >= geometry.width * 0.20
                && geometry.height * 0.30 <= *y
                && *y <= geometry.height * 0.98
        })
        .collect();
    if candidates.is_empty() {
        return None;
    }
    let body_size = median(
        lines
            .iter()
            .filter(|line| line.bbox[1] < geometry.height * 0.70)
            .map(model_line_font_size)
            .filter(|size| *size > 0.0)
            .collect(),
    );
    let first_label = lines
        .iter()
        .filter(|line| {
            line.bbox[1] >= geometry.height * 0.48
                && begins_with_note_label(&line.text)
                && body_size > 0.0
                && (model_line_font_size(line) <= body_size * 0.90
                    || line
                        .spans
                        .first()
                        .is_some_and(|span| span.size <= body_size * 0.78))
        })
        .map(|line| line.bbox[1])
        .min_by(f64::total_cmp);
    if let Some(first_label) = first_label {
        if let Some((_, y)) = candidates
            .iter()
            .filter(|(_, y)| *y <= first_label + (geometry.height * 0.004).max(1.0))
            .max_by(|left, right| left.1.total_cmp(&right.1))
        {
            return Some(*y);
        }
    }
    candidates
        .into_iter()
        .filter(|(_, y)| *y <= geometry.height * 0.92)
        .min_by(|left, right| left.0.total_cmp(&right.0).then(left.1.total_cmp(&right.1)))
        .map(|(_, y)| y)
}

fn make_line(
    text_line: TextLine,
    page_index: usize,
    local_index: usize,
    source_index: usize,
    page_height: f64,
) -> Option<Line> {
    let block_index = text_line
        .items
        .iter()
        .find_map(|item| {
            item.fidelity
                .as_ref()
                .map(|value| value.renderer_block as usize)
        })
        .filter(|value| *value > 0)
        .unwrap_or(local_index);
    let assembled = assemble_line(&text_line);
    if assembled.text.is_empty() {
        return None;
    }
    let id = format!("p{:04}-l{:04}", page_index + 1, local_index);
    let spans = line_spans(&text_line, &assembled, page_height, &id);
    // MuPDF's line bbox includes source whitespace and other retained runs
    // even when they do not survive as product spans.
    let bbox = union_bbox(
        text_line
            .items
            .iter()
            .map(|item| item_bbox(item, page_height)),
    );
    let words = line_words(&assembled.text, &text_line, &assembled, page_height, &id);
    let text = assembled.text;
    Some(Line {
        id,
        page_index,
        page_number: u32::try_from(page_index + 1).unwrap_or(u32::MAX),
        source_index,
        reading_order: source_index,
        block_index,
        text,
        bbox,
        spans,
        words,
        detached_references: vec![],
        exclude_from_body: false,
        suppress_footnote_label: false,
        note_region_mode: String::new(),
        region_id: String::new(),
        region_type: "unknown".to_owned(),
        source: "native".to_owned(),
    })
}

fn shares_source_line(line: &TextLine, item: &TextItem) -> bool {
    if line.page != item.page {
        return false;
    }
    let Some(prior) = line.items.last() else {
        return false;
    };
    if let (Some(prior), Some(current)) = (prior.fidelity.as_ref(), item.fidelity.as_ref()) {
        if prior.renderer_line > 0 && current.renderer_line > 0 {
            return prior.renderer_line == current.renderer_line;
        }
    }
    // MuPDF's text-device line model uses an 0.8em maximum baseline and
    // inter-run distance. Port those renderer-level invariants here instead
    // of allowing the former five-em geometric merge. Raised note labels
    // remain on their host line; independent same-row entries and columns do
    // not become one line merely because their vertical boxes overlap.
    const BASE_MAX_DIST: f64 = 0.8;
    const SPACE_MAX_DIST: f64 = 0.8;
    let line_em = f64::from(item.font_size).max(1.0);
    if (item_baseline(prior) - item_baseline(item)).abs() > line_em * BASE_MAX_DIST {
        return false;
    }
    let rtl = pdf_inspector::text_utils::is_rtl_text(
        line.items
            .iter()
            .map(|value| value.text.as_str())
            .chain(std::iter::once(item.text.as_str())),
    );
    let gap = if rtl {
        f64::from(prior.x - (item.x + item.width))
    } else {
        f64::from(item.x - (prior.x + prior.width))
    };
    (-line_em..=line_em * SPACE_MAX_DIST).contains(&gap)
}

/// Group adjacent content-stream items without imposing a second reading
/// order. The legal engine's own layout stage, like the Python extractor it
/// replaces, must receive the PDF's source order intact.
fn group_source_order_lines(items: Vec<TextItem>) -> Vec<TextLine> {
    let mut lines = Vec::<TextLine>::new();
    for item in items {
        if lines
            .last()
            .is_some_and(|line| shares_source_line(line, &item))
        {
            let line = lines.last_mut().expect("line exists");
            line.items.push(item);
        } else {
            lines.push(TextLine {
                y: item.y,
                page: item.page,
                items: vec![item],
                adaptive_threshold: 0.1,
            });
        }
    }
    lines
}

fn make_ocr_line(result: OcrLine, page: &Page, local_index: usize) -> Option<Line> {
    let mut text = result.text;
    let trimmed_end = text.trim_end().len();
    text.truncate(trimmed_end);
    let leading = text.len() - text.trim_start().len();
    text.replace_range(..leading, "");
    if text.is_empty() {
        return None;
    }
    let words = result
        .words
        .into_iter()
        .enumerate()
        .map(|(index, word)| Word {
            id: format!("p{:04}-l{:04}-w{:03}", page.number, local_index, index + 1),
            text: word.text,
            bbox: word.bbox,
            start: word.start,
            end: word.end,
        })
        .collect();
    Some(Line {
        id: format!("p{:04}-l{local_index:04}", page.number),
        page_index: page.index,
        page_number: page.number,
        source_index: 0,
        reading_order: 0,
        block_index: local_index,
        text,
        bbox: result.bbox,
        spans: vec![],
        words,
        detached_references: vec![],
        exclude_from_body: false,
        suppress_footnote_label: false,
        note_region_mode: String::new(),
        region_id: result.region_id,
        region_type: result.region_type,
        source: "ocr".to_owned(),
    })
}

fn reindex_source_lines(pages: &mut [Page]) {
    let mut source_index = 0;
    for page in pages {
        for line in &mut page.lines {
            source_index += 1;
            line.page_index = page.index;
            line.page_number = page.number;
            line.source_index = source_index;
            line.reading_order = source_index;
        }
    }
}

fn text_quality(lines: &[Line]) -> f64 {
    // Python's str.isprintable accepts Unicode L/M/N/P/S plus ASCII space,
    // and rejects every other separator/control/format/private/unassigned
    // scalar. The already-required regex crate exposes those categories.
    static PRINTABLE: OnceLock<Regex> = OnceLock::new();
    let printable =
        PRINTABLE.get_or_init(|| Regex::new(r"[\p{L}\p{M}\p{N}\p{P}\p{S} ]").expect("valid regex"));
    let mut count = lines.len().saturating_sub(1);
    let mut printable_count = 0;
    let mut replacements = 0;
    let mut trimmed = 0;
    let mut trailing_whitespace = 0;
    let mut has_text = false;
    for (index, line) in lines.iter().enumerate() {
        printable_count += printable.find_iter(&line.text).count();
        if index > 0 && has_text {
            trailing_whitespace += 1;
        }
        for character in line.text.chars() {
            count += 1;
            replacements += usize::from(character == '\u{fffd}');
            if character.is_whitespace() {
                trailing_whitespace += usize::from(has_text);
            } else {
                trimmed += trailing_whitespace + 1;
                trailing_whitespace = 0;
                has_text = true;
            }
        }
    }
    let count = count.max(1) as f64;
    let replacement_share = replacements as f64 / count;
    let printable_share = printable_count as f64 / count;
    let quantity = (trimmed as f64 / 100.0).min(1.0);
    ((quantity * printable_share * (1.0 - (replacement_share * 20.0).min(1.0))).max(0.0) * 10_000.0)
        .round_ties_even()
        / 10_000.0
}

#[derive(Serialize, Deserialize)]
pub struct ExtractedPdf {
    pub pages: Vec<Page>,
    pub separators: Vec<Option<f64>>,
    pub diagnostics: Vec<Diagnostic>,
    pub metadata: PdfExtractionMetadata,
}

pub fn load_extraction_document(bytes: &[u8]) -> Result<Document> {
    if let Some(mut document) = Document::load_mem(bytes)
        .ok()
        .filter(|document| !document.is_encrypted() && !document.get_pages().is_empty())
    {
        document
            .objects
            .values_mut()
            .for_each(prune_extraction_object);
        return Ok(document);
    }
    pdf_inspector::load_document_from_mem(bytes)
        .map(|value| value.0)
        .map_err(Into::into)
}

pub fn page_geometries(document: &Document) -> PageGeometryMap {
    document
        .get_pages()
        .into_iter()
        .map(|(page, id)| (page, (id, page_geometry(document, id))))
        .collect()
}

pub fn assemble_pdf(
    pdf: &[u8],
    geometries: &PageGeometryMap,
    mut items: Vec<TextItem>,
    painted_rules: Vec<PdfLine>,
    detection: PdfTypeResult,
    ocr: Option<&mut dyn PdfOcrProvider>,
    ocr_pages: Option<&[usize]>,
) -> Result<ExtractedPdf> {
    if geometries.is_empty() {
        return Err(Error::Message("PDF has no pages".to_owned()));
    }
    let mut by_page: HashMap<u32, Vec<TextLine>> = HashMap::new();
    items.retain_mut(|item| {
        let is_text = matches!(&item.item_type, ItemType::Text);
        if is_text {
            if let Some((_, geometry)) = geometries.get(&item.page) {
                transform_item(item, *geometry);
            }
        }
        !is_text || !item.text.is_empty()
    });
    deduplicate_painted_text(&mut items);
    assign_renderer_layout(&mut items);
    items.retain(|item| matches!(&item.item_type, ItemType::Text));
    for line in group_source_order_lines(items) {
        by_page.entry(line.page).or_default().push(line);
    }

    let mut pages = Vec::with_capacity(geometries.len());
    let mut diagnostics = Vec::new();
    let mut source_offset = 0;
    let mut weak_pages = BTreeSet::<usize>::new();
    for (&number, &(_, geometry)) in geometries {
        let page_index = usize::try_from(number.saturating_sub(1)).unwrap_or(usize::MAX);
        let mut lines = Vec::new();
        for text_line in by_page.remove(&number).unwrap_or_default() {
            let local_index = lines.len() + 1;
            if let Some(line) = make_line(
                text_line,
                page_index,
                local_index,
                source_offset + local_index,
                geometry.height,
            ) {
                lines.push(line);
            }
        }
        let quality = text_quality(&lines);
        if lines.is_empty() || quality < 0.15 {
            weak_pages.insert(page_index);
        }
        source_offset += lines.len();
        pages.push(Page {
            id: format!("p{number:04}"),
            index: page_index,
            number,
            width: round3(geometry.width),
            height: round3(geometry.height),
            lines,
            regions: vec![],
            source: "native".to_owned(),
            text_quality: quality,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        });
    }
    let mut rules_by_page = HashMap::<u32, Vec<PdfLine>>::new();
    for rule in painted_rules {
        rules_by_page.entry(rule.page).or_default().push(rule);
    }
    let mut separators = profile::measure("extract.separators", || {
        geometries
            .iter()
            .zip(&pages)
            .map(|((&number, &(_, geometry)), page)| {
                separator_y(
                    geometry,
                    &page.lines,
                    rules_by_page.get(&number).map_or(&[], Vec::as_slice),
                )
            })
            .collect::<Vec<_>>()
    });

    for page in &detection.pages_needing_ocr {
        if let Ok(index) = usize::try_from(page.saturating_sub(1)) {
            weak_pages.insert(index);
        }
    }

    let routed_pages: Vec<_> = weak_pages
        .iter()
        .copied()
        .filter(|page| ocr_pages.is_none_or(|selected| selected.contains(page)))
        .collect();
    let mut reindex = false;
    if let Some(provider) = ocr {
        let requests: Vec<_> = routed_pages
            .iter()
            .filter_map(|&page_index| {
                pages.get(page_index).map(|page| OcrPageRequest {
                    page_index,
                    width: page.width,
                    height: page.height,
                })
            })
            .collect();
        let results = legal_pdf_core::profile::measure("extract.ocr", || {
            provider.extract_pages(pdf, &requests)
        })?;
        for result in results {
            let Some(page) = pages.get_mut(result.page_index) else {
                return Err(Error::Message(format!(
                    "OCR returned an unknown page index: {}",
                    result.page_index
                )));
            };
            let lines: Vec<_> = result
                .lines
                .into_iter()
                .enumerate()
                .filter_map(|(index, line)| make_ocr_line(line, page, index + 1))
                .collect();
            if !lines.is_empty() {
                reindex = true;
                page.lines = lines;
                page.source = "ocr".to_owned();
                page.text_quality = 0.5;
                weak_pages.remove(&result.page_index);
                if let Some(separator) = result.separator_y {
                    separators[result.page_index] = Some(separator);
                }
            }
        }
    }
    if reindex {
        reindex_source_lines(&mut pages);
    }
    let unresolved_pages: Vec<_> = weak_pages.into_iter().collect();
    for &page_index in &unresolved_pages {
        diagnostics.push(Diagnostic::warning(
            "OCR_REQUIRED",
            "Page has no reliable embedded text and no usable OCR result.",
            Some(page_index),
        ));
    }

    Ok(ExtractedPdf {
        pages,
        separators,
        diagnostics,
        metadata: PdfExtractionMetadata {
            pages_needing_ocr: unresolved_pages,
            ocr_routed_pages: routed_pages,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extraction_filter_prunes_only_large_sparse_tag_arrays() {
        let mut sparse = Object::Array(
            (0..4096)
                .map(|index| {
                    (index % 100 == 0)
                        .then_some(Object::Reference((index + 1, 0)))
                        .unwrap_or(Object::Null)
                })
                .collect(),
        );
        prune_extraction_object(&mut sparse);
        assert!(matches!(sparse, Object::Null));

        let mut page_tree = Object::Array(
            (1..=4096)
                .map(|index| Object::Reference((index, 0)))
                .collect(),
        );
        prune_extraction_object(&mut page_tree);
        assert!(matches!(page_tree, Object::Array(values) if values.len() == 4096));
    }

    fn text_item(text: &str, x: f32, y: f32, width: f32, size: f32) -> TextItem {
        TextItem {
            text: text.to_owned(),
            x,
            y,
            width,
            height: size,
            font_size: size,
            font: String::new(),
            font_tag: String::new(),
            page: 1,
            is_bold: false,
            is_italic: false,
            is_underline: false,
            is_strikeout: false,
            fidelity: None,
            item_type: ItemType::Text,
            mcid: None,
        }
    }

    fn text_line(item: TextItem) -> TextLine {
        TextLine {
            y: item.y,
            page: item.page,
            items: vec![item],
            adaptive_threshold: 0.1,
        }
    }

    fn fidelity_item(text: &str, glyphs: &[(&str, f32, f32)]) -> TextItem {
        let mut item = text_item(text, 10.0, 80.0, 75.0, 10.0);
        item.fidelity = Some(Box::new(pdf_inspector::types::FidelityTextInfo {
            text: text.to_owned(),
            font: "Test".to_owned(),
            resource: "F1".to_owned(),
            flags: 4,
            ascender: 0.8,
            descender: -0.2,
            text_object: 1,
            source_line: 1,
            renderer_line: 0,
            renderer_block: 0,
            text_rise: 0.0,
            baseline: [10.0, 80.0],
            advance: [75.0, 0.0],
            em: [0.0, 10.0],
            glyphs: glyphs
                .iter()
                .map(|(text, x0, x1)| FidelityGlyph {
                    text: (*text).to_owned(),
                    x0: *x0,
                    x1: *x1,
                    bbox: [10.0 + *x0, 78.0, 10.0 + *x1, 88.0],
                })
                .collect(),
        }));
        item
    }

    fn positioned_fidelity_item(text: &str, x: f32, y: f32) -> TextItem {
        let mut item = fidelity_item(text, &[(text, 0.0, 10.0)]);
        let fidelity = item.fidelity.as_mut().unwrap();
        let dx = x - fidelity.baseline[0];
        let dy = y - fidelity.baseline[1];
        for glyph in &mut fidelity.glyphs {
            glyph.bbox[0] += dx;
            glyph.bbox[1] += dy;
            glyph.bbox[2] += dx;
            glyph.bbox[3] += dy;
        }
        fidelity.baseline = [x, y];
        fidelity.advance = [10.0, 0.0];
        item.x = x;
        item.y = y;
        item
    }

    #[test]
    fn renderer_layout_uses_paragraph_indents_and_counts_images() {
        let mut image = text_item("", 0.0, 0.0, 1.0, 1.0);
        image.item_type = ItemType::Image;
        let mut items = vec![
            positioned_fidelity_item("first", 10.0, 80.0),
            positioned_fidelity_item("continuation", 10.0, 92.0),
            positioned_fidelity_item("indented", 20.0, 104.0),
            image,
            positioned_fidelity_item("after image", 10.0, 116.0),
        ];

        assign_renderer_layout(&mut items);
        let identities: Vec<_> = items
            .iter()
            .filter_map(|item| {
                item.fidelity
                    .as_ref()
                    .map(|value| (value.renderer_line, value.renderer_block))
            })
            .collect();
        assert_eq!(identities, [(1, 1), (2, 1), (3, 2), (4, 4)]);
    }

    #[test]
    fn renderer_layout_keeps_hanging_label_continuations_in_one_block() {
        let mut label = positioned_fidelity_item("100", 10.0, 83.15);
        let label_fidelity = label.fidelity.as_mut().unwrap();
        label_fidelity.advance = [3.0, 0.0];
        label_fidelity.em = [0.0, 4.6];
        let mut body = positioned_fidelity_item(" body", 13.0, 80.0);
        body.fidelity.as_mut().unwrap().em = [0.0, 8.0];
        let mut continuation = positioned_fidelity_item("continuation", 20.0, 70.8);
        continuation.fidelity.as_mut().unwrap().em = [0.0, 8.0];
        let mut items = vec![label, body, continuation];

        assign_renderer_layout(&mut items);
        let identities: Vec<_> = items
            .iter()
            .map(|item| {
                let fidelity = item.fidelity.as_ref().unwrap();
                (fidelity.renderer_line, fidelity.renderer_block)
            })
            .collect();

        assert_eq!(identities, [(1, 1), (1, 1), (2, 1)]);
    }

    #[test]
    fn zero_advance_text_keeps_the_existing_line_direction() {
        let mut charter = positioned_fidelity_item("Charter", 10.0, 80.0);
        charter.fidelity.as_mut().unwrap().advance = [35.0, 0.0];
        let mut discretionary = positioned_fidelity_item("\u{00ad}", 45.0, 80.0);
        discretionary.fidelity.as_mut().unwrap().advance = [0.0, 0.0];
        let hyphen = positioned_fidelity_item("-based", 45.0, 80.0);
        let mut items = vec![charter, discretionary, hyphen];

        assign_renderer_layout(&mut items);

        assert!(items.iter().all(|item| {
            item.fidelity
                .as_ref()
                .is_some_and(|fidelity| fidelity.renderer_line == 1)
        }));
    }

    #[test]
    fn duplicate_paints_do_not_duplicate_text() {
        let first = positioned_fidelity_item("n", 10.0, 80.0);
        let mut duplicate = first.clone();
        duplicate.fidelity.as_mut().unwrap().source_line = 2;
        let repeated_letter = positioned_fidelity_item("n", 20.0, 80.0);
        let mut items = vec![first, duplicate, repeated_letter];

        deduplicate_painted_text(&mut items);

        assert_eq!(items.len(), 2);
        assert_eq!(
            items
                .iter()
                .map(|item| item.fidelity.as_ref().unwrap().baseline[0])
                .collect::<Vec<_>>(),
            [10.0, 20.0]
        );
    }

    #[test]
    fn duplicate_paint_inside_a_larger_prior_run_is_removed() {
        let first = fidelity_item(" n", &[(" ", 0.0, 5.0), ("n", 5.0, 10.0)]);
        let mut duplicate = fidelity_item("n", &[("n", 0.0, 5.0)]);
        let fidelity = duplicate.fidelity.as_mut().unwrap();
        fidelity.baseline = [15.0, 80.0];
        fidelity.glyphs[0].bbox = [15.0, 78.0, 17.0, 88.0];
        let mut items = vec![first, duplicate];

        deduplicate_painted_text(&mut items);

        assert_eq!(items.len(), 1);
    }

    #[test]
    fn identical_paints_on_different_pages_are_preserved() {
        let first = positioned_fidelity_item("n", 10.0, 80.0);
        let mut next_page = first.clone();
        next_page.page += 1;
        let mut items = vec![first, next_page];

        deduplicate_painted_text(&mut items);

        assert_eq!(items.len(), 2);
    }

    #[test]
    fn fully_clipped_glyphs_do_not_survive_in_text() {
        let mut item = fidelity_item("fo", &[("f", 0.0, 10.0), ("o", 16.0, 26.0)]);
        let fidelity = item.fidelity.as_mut().unwrap();
        fidelity.baseline = [-12.0, 80.0];
        fidelity.advance = [38.0, 0.0];
        fidelity.glyphs[0].bbox = [-12.0, 78.0, -2.0, 88.0];
        fidelity.glyphs[1].bbox = [4.0, 78.0, 14.0, 88.0];
        let geometry = PageGeometry {
            x0: 0.0,
            y0: 0.0,
            raw_width: 100.0,
            raw_height: 100.0,
            rotation: 0,
            width: 100.0,
            height: 100.0,
        };

        assert!(transform_fidelity_item(&mut item, geometry));

        assert_eq!(item.text, "o");
        assert_eq!((item.x, item.width), (4.0, 10.0));
    }

    #[test]
    fn visible_zero_height_text_remains_extraction_evidence() {
        let visible = text_item("■", 10.0, 80.0, 10.0, 0.0);
        let empty = text_item("", 10.0, 80.0, 0.0, 0.0);

        assert!(has_extraction_evidence(&visible));
        assert!(!has_extraction_evidence(&empty));
    }

    #[test]
    fn line_bbox_retains_non_product_whitespace_geometry() {
        let mut visible = text_item("A", 10.0, 80.0, 10.0, 10.0);
        visible.font = "Test".to_owned();
        let mut whitespace = text_item(" ", 20.0, 75.0, 5.0, 15.0);
        whitespace.font = "Test".to_owned();
        let line = make_line(
            TextLine {
                y: 80.0,
                page: 1,
                items: vec![visible, whitespace],
                adaptive_threshold: 0.1,
            },
            0,
            1,
            1,
            100.0,
        )
        .unwrap();

        assert_eq!(line.text, "A");
        assert_eq!(line.bbox, [10.0, 10.0, 25.0, 25.0]);
        assert_eq!(line.spans.len(), 1);
    }

    #[test]
    fn geometric_superscript_requires_an_inline_position() {
        let item = text_item("100", 10.0, 81.0, 6.0, 5.0);

        assert!(!is_superscript(&item, 10.0, 80.0, false));
        assert!(is_superscript(&item, 10.0, 80.0, true));
    }

    #[test]
    fn crop_and_rotation_normalize_into_display_space() {
        let geometry = PageGeometry {
            x0: 10.0,
            y0: 20.0,
            raw_width: 200.0,
            raw_height: 100.0,
            rotation: 90,
            width: 100.0,
            height: 200.0,
        };
        assert_eq!(transform_point(geometry, 10.0, 20.0), (0.0, 200.0));
        assert_eq!(transform_point(geometry, 210.0, 120.0), (100.0, 0.0));
    }

    #[test]
    fn unicode_word_offsets_are_character_offsets() {
        let line = make_line(
            text_line(fidelity_item(
                "État law",
                &[
                    ("É", 0.0, 10.0),
                    ("t", 10.0, 20.0),
                    ("a", 20.0, 30.0),
                    ("t", 30.0, 40.0),
                    (" ", 40.0, 45.0),
                    ("l", 45.0, 55.0),
                    ("a", 55.0, 65.0),
                    ("w", 65.0, 75.0),
                ],
            )),
            0,
            1,
            1,
            100.0,
        )
        .unwrap();
        assert_eq!((line.words[1].start, line.words[1].end), (5, 8));
        assert_eq!(line.words[1].bbox[0], 55.0);
        assert_eq!(line.words[1].bbox[2], 85.0);
    }

    #[test]
    fn words_are_not_fabricated_without_source_glyph_positions() {
        let line = make_line(
            text_line(text_item("word", 10.0, 80.0, 40.0, 10.0)),
            0,
            1,
            1,
            100.0,
        )
        .unwrap();
        assert!(line.words.is_empty());
    }

    #[test]
    fn text_quality_matches_python_printability_categories() {
        let first = make_line(
            text_line(text_item("A", 10.0, 80.0, 10.0, 10.0)),
            0,
            1,
            1,
            100.0,
        )
        .unwrap();
        let second = make_line(
            text_line(text_item("B", 10.0, 60.0, 10.0, 10.0)),
            0,
            2,
            2,
            100.0,
        )
        .unwrap();
        assert_eq!(text_quality(&[first, second]), 0.02);

        let private_use = make_line(
            text_line(text_item("A\u{e000}", 10.0, 80.0, 20.0, 10.0)),
            0,
            1,
            1,
            100.0,
        )
        .unwrap();
        assert_eq!(text_quality(&[private_use]), 0.01);
    }

    #[test]
    fn legal_line_preserves_a_geometric_boundary_after_a_small_note_label() {
        let line = TextLine {
            y: 376.0,
            page: 1,
            items: vec![
                text_item("33", 54.1, 376.0, 6.6, 4.6),
                text_item("Ibid, quoting Heather Jenkins.", 68.5, 375.5, 96.5, 8.0),
            ],
            adaptive_threshold: 0.1,
        };
        let line = make_line(line, 0, 1, 1, 792.0).unwrap();

        assert_eq!(line.text, "33 Ibid, quoting Heather Jenkins.");
        assert_eq!(line.spans[0].text, "33");
        assert_eq!((line.spans[0].start, line.spans[0].end), (0, 2));
        assert_eq!((line.spans[1].start, line.spans[1].end), (3, 33));
    }

    #[test]
    fn legal_line_does_not_separate_a_touching_inline_reference() {
        let line = TextLine {
            y: 456.0,
            page: 1,
            items: vec![
                text_item("word", 72.0, 456.0, 24.0, 11.5),
                text_item("33", 96.0, 459.8, 6.0, 6.7),
                text_item(", next", 102.0, 456.0, 30.0, 11.5),
            ],
            adaptive_threshold: 0.1,
        };

        assert_eq!(assemble_line(&line).text, "word33, next");
    }

    #[test]
    fn pdf_text_normalization_preserves_the_old_boundary_contract() {
        assert_eq!(normalize_text("a\u{200b}b"), "ab");
        assert_eq!(normalize_text("a\u{200b}\u{200b}b"), "a b");
        assert_eq!(normalize_text("a\u{feff}b"), "ab");
    }

    #[test]
    fn source_line_grouping_keeps_raised_markers_inline() {
        let grouped = group_source_order_lines(vec![
            text_item("Aboriginal context.", 72.0, 456.0, 87.85, 11.5),
            text_item("1", 159.85, 459.83, 3.22, 6.7),
            text_item(" In the most recent case", 163.07, 456.0, 130.0, 11.5),
            text_item("Next line", 72.0, 443.0, 50.0, 11.5),
        ]);
        assert_eq!(grouped.len(), 2);
        assert_eq!(
            grouped[0].text(),
            "Aboriginal context.1 In the most recent case"
        );
        assert_eq!(grouped[1].text(), "Next line");
    }

    #[test]
    fn source_line_grouping_does_not_join_distant_same_row_runs() {
        let grouped = group_source_order_lines(vec![
            text_item("left first", 72.0, 700.0, 60.0, 10.0),
            text_item("left second", 72.0, 688.0, 70.0, 10.0),
            text_item("14", 330.0, 700.0, 9.0, 10.0),
            text_item("Ibid.", 353.0, 700.0, 24.0, 10.0),
        ]);
        assert_eq!(
            grouped.iter().map(TextLine::text).collect::<Vec<_>>(),
            ["left first", "left second", "14", "Ibid."]
        );
    }

    #[test]
    fn source_line_grouping_merges_a_note_label_with_its_body() {
        let mut grouped = group_source_order_lines(vec![
            text_item("14", 54.0, 533.0, 6.0, 6.0),
            text_item("Ibid.", 65.0, 531.0, 24.0, 8.0),
        ]);
        assert_eq!(grouped.len(), 1);
        let line = make_line(grouped.remove(0), 0, 1, 1, 792.0).unwrap();
        assert_eq!(line.text, "14 Ibid.");
    }

    #[test]
    fn source_line_grouping_uses_whitespace_as_non_output_evidence() {
        let mut grouped = group_source_order_lines(vec![
            text_item("14", 54.0, 533.0, 6.0, 6.0),
            text_item(" ", 60.0, 531.0, 5.0, 8.0),
            text_item("Ibid.", 65.0, 531.0, 24.0, 8.0),
        ]);
        assert_eq!(grouped.len(), 1);
        grouped[0].items.retain(|item| !item.text.trim().is_empty());
        let line = make_line(grouped.remove(0), 0, 1, 1, 792.0).unwrap();
        assert_eq!(line.text, "14 Ibid.");
        assert_eq!(line.spans.len(), 2);
    }

    #[test]
    fn source_whitespace_survives_inside_one_renderer_span() {
        let mut grouped = group_source_order_lines(vec![
            text_item("before", 54.0, 531.0, 24.0, 8.0),
            text_item("\u{00a0}", 78.0, 531.0, 2.0, 8.0),
            text_item("after", 80.0, 531.0, 20.0, 8.0),
        ]);
        let line = make_line(grouped.remove(0), 0, 1, 1, 792.0).unwrap();

        assert_eq!(line.text, "before\u{00a0}after");
        assert_eq!(line.spans.len(), 1);
        assert_eq!(line.spans[0].text, line.text);
    }

    #[test]
    fn source_line_grouping_keeps_a_toc_label_separate_from_the_heading() {
        let grouped = group_source_order_lines(vec![
            text_item("A.", 72.1, 381.25, 9.7, 10.0),
            text_item("THE MANDATE", 94.6, 381.25, 60.0, 10.0),
        ]);
        assert_eq!(
            grouped.iter().map(TextLine::text).collect::<Vec<_>>(),
            ["A.", "THE MANDATE"]
        );
    }

    #[test]
    fn source_line_grouping_keeps_reverse_order_furniture_separate() {
        let grouped = group_source_order_lines(vec![
            text_item("Volume 16, Number 3, 2007", 431.0, 20.0, 111.0, 10.0),
            text_item("128", 59.0, 19.3, 15.0, 12.0),
        ]);
        assert_eq!(
            grouped.iter().map(TextLine::text).collect::<Vec<_>>(),
            ["Volume 16, Number 3, 2007", "128"]
        );
    }

    #[test]
    fn source_line_grouping_does_not_let_a_drop_cap_bridge_rows() {
        let grouped = group_source_order_lines(vec![
            text_item("Introduction", 144.0, 520.0, 58.0, 12.0),
            text_item("T", 143.0, 470.0, 35.0, 61.0),
            text_item("he first row", 178.0, 503.0, 80.0, 9.0),
            text_item("second row", 178.0, 486.5, 70.0, 9.0),
            text_item("third row", 178.0, 470.0, 65.0, 9.0),
        ]);
        assert_eq!(
            grouped.iter().map(TextLine::text).collect::<Vec<_>>(),
            [
                "Introduction",
                "T",
                "he first row",
                "second row",
                "third row"
            ]
        );
    }

    #[test]
    fn raised_eighty_percent_digit_is_a_superscript_span() {
        let text_line = group_source_order_lines(vec![
            text_item("approaches.", 144.0, 180.0, 50.0, 10.0),
            text_item("9", 194.0, 182.6, 4.0, 8.0),
        ])
        .remove(0);
        let line = make_line(text_line, 0, 1, 1, 792.0).unwrap();
        assert!(line
            .spans
            .iter()
            .any(|span| span.text == "9" && span.superscript));
    }

    #[test]
    fn raised_two_digit_reference_after_punctuation_is_a_superscript_span() {
        let text_line = group_source_order_lines(vec![
            text_item(".", 144.0, 180.0, 3.0, 12.0),
            text_item("27", 148.0, 183.0, 8.0, 7.92),
        ])
        .remove(0);
        let line = make_line(text_line, 0, 1, 1, 792.0).unwrap();
        assert!(line
            .spans
            .iter()
            .any(|span| span.text == "27" && span.superscript));
    }
}
