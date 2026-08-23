use crate::{Error, Result};
use legal_pdf_core::model::{Diagnostic, ImageBlock, Line, Page, Span, TableBlock, Word};
use legal_pdf_core::{profile, union_bbox, OcrLine, OcrPageRequest, PdfOcrProvider};
use lopdf::{Document, Object, ObjectId};
use pdf_inspector_core::types::{FidelityGlyph, ItemType, PdfLine, TextItem, TextLine};
use pdf_inspector_detector::detector::{
    detect_from_page_evidence, get_document_title, PageDetectionEvidence,
};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::sync::OnceLock;

fn prune_extraction_object(object: &mut Object) {
    if let Object::Array(values) = object {
        let nulls = values
            .iter()
            .filter(|value| matches!(value, Object::Null))
            .count();
        if values.len() >= 4096
            && nulls * 20 >= values.len() * 19
            && values
                .iter()
                .all(|value| matches!(value, Object::Null | Object::Reference(_)))
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

fn inherited_object(doc: &Document, mut id: ObjectId, key: &[u8]) -> Option<Object> {
    for _ in 0..32 {
        let dictionary = doc.get_dictionary(id).ok()?;
        if let Ok(value) = dictionary.get(key) {
            return Some(value.clone());
        }
        match dictionary.get(b"Parent") {
            Ok(Object::Reference(parent)) => id = *parent,
            _ => return None,
        }
    }
    None
}

fn resolve_array(doc: &Document, value: Object) -> Option<Vec<Object>> {
    match value {
        Object::Array(values) => Some(values),
        Object::Reference(id) => match doc.get_object(id).ok()? {
            Object::Array(values) => Some(values.clone()),
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
            let numbers: Vec<f64> = values.iter().filter_map(number).collect();
            (numbers.len() >= 4).then(|| {
                [
                    numbers[0].min(numbers[2]),
                    numbers[1].min(numbers[3]),
                    numbers[0].max(numbers[2]),
                    numbers[1].max(numbers[3]),
                ]
            })
        })
        .unwrap_or([0.0, 0.0, 612.0, 792.0]);
    let rotation = inherited_object(doc, id, b"Rotate")
        .as_ref()
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

fn transform_bbox(geometry: PageGeometry, bbox: [f32; 4]) -> [f32; 4] {
    let points = [
        transform_point(geometry, f64::from(bbox[0]), f64::from(bbox[1])),
        transform_point(geometry, f64::from(bbox[0]), f64::from(bbox[3])),
        transform_point(geometry, f64::from(bbox[2]), f64::from(bbox[1])),
        transform_point(geometry, f64::from(bbox[2]), f64::from(bbox[3])),
    ];
    let x0 = points
        .iter()
        .map(|point| point.0)
        .fold(f64::INFINITY, f64::min)
        .clamp(0.0, geometry.width);
    let y0 = points
        .iter()
        .map(|point| point.1)
        .fold(f64::INFINITY, f64::min)
        .clamp(0.0, geometry.height);
    let x1 = points
        .iter()
        .map(|point| point.0)
        .fold(f64::NEG_INFINITY, f64::max)
        .min(geometry.width)
        .max(x0);
    let y1 = points
        .iter()
        .map(|point| point.1)
        .fold(f64::NEG_INFINITY, f64::max)
        .min(geometry.height)
        .max(y0);
    [x0 as f32, y0 as f32, x1 as f32, y1 as f32]
}

fn bbox_intersects_page(geometry: PageGeometry, bbox: [f32; 4]) -> bool {
    let points = [
        transform_point(geometry, f64::from(bbox[0]), f64::from(bbox[1])),
        transform_point(geometry, f64::from(bbox[2]), f64::from(bbox[1])),
        transform_point(geometry, f64::from(bbox[0]), f64::from(bbox[3])),
        transform_point(geometry, f64::from(bbox[2]), f64::from(bbox[3])),
    ];
    let x0 = points
        .iter()
        .map(|point| point.0)
        .fold(f64::INFINITY, f64::min);
    let y0 = points
        .iter()
        .map(|point| point.1)
        .fold(f64::INFINITY, f64::min);
    let x1 = points
        .iter()
        .map(|point| point.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let y1 = points
        .iter()
        .map(|point| point.1)
        .fold(f64::NEG_INFINITY, f64::max);
    x1 >= 0.0 && x0 <= geometry.width && y1 >= 0.0 && y0 <= geometry.height
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
    let visible_glyphs = fidelity
        .glyphs
        .iter()
        .map(|glyph| bbox_intersects_page(geometry, glyph.bbox))
        .collect::<Vec<_>>();
    for glyph in &mut fidelity.glyphs {
        glyph.bbox = transform_bbox(geometry, glyph.bbox);
    }
    let mut index = 0;
    fidelity.glyphs.retain(|_| {
        let visible = visible_glyphs[index];
        index += 1;
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
    let x0 = points
        .iter()
        .map(|point| point[0])
        .fold(f32::INFINITY, f32::min)
        .clamp(0.0, geometry.width as f32);
    let y0 = points
        .iter()
        .map(|point| point[1])
        .fold(f32::INFINITY, f32::min)
        .clamp(0.0, geometry.height as f32);
    let x1 = points
        .iter()
        .map(|point| point[0])
        .fold(f32::NEG_INFINITY, f32::max)
        .min(geometry.width as f32)
        .max(x0);
    let y1 = points
        .iter()
        .map(|point| point[1])
        .fold(f32::NEG_INFINITY, f32::max)
        .min(geometry.height as f32)
        .max(y0);
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
    let x0 = corners
        .iter()
        .map(|point| point.0)
        .fold(f64::INFINITY, f64::min);
    let y0 = corners
        .iter()
        .map(|point| point.1)
        .fold(f64::INFINITY, f64::min);
    let x1 = corners
        .iter()
        .map(|point| point.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let y1 = corners
        .iter()
        .map(|point| point.1)
        .fold(f64::NEG_INFINITY, f64::max);
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

fn source_bbox(geometry: PageGeometry, bbox: [f32; 4]) -> [f64; 4] {
    let [x0, y0, x1, y1] = transform_bbox(geometry, bbox);
    [
        round3(f64::from(x0)),
        round3(geometry.height - f64::from(y1)),
        round3(f64::from(x1)),
        round3(geometry.height - f64::from(y0)),
    ]
}

fn detected_images(items: &[TextItem], geometries: &PageGeometryMap) -> Vec<ImageBlock> {
    let mut page_counts = HashMap::<u32, usize>::new();
    items
        .iter()
        .filter(|item| matches!(item.item_type, ItemType::Image))
        .filter_map(|item| Some((item, &geometries.get(&item.page)?.1)))
        .map(|(item, geometry)| {
            let index = page_counts.entry(item.page).or_default();
            *index += 1;
            let mut bbox = item_bbox(item, geometry.height);
            bbox[0] = bbox[0].clamp(0.0, geometry.width);
            bbox[1] = bbox[1].clamp(0.0, geometry.height);
            bbox[2] = bbox[2].clamp(bbox[0], geometry.width);
            bbox[3] = bbox[3].clamp(bbox[1], geometry.height);
            let area_ratio = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                / (geometry.width * geometry.height).max(1.0))
            .clamp(0.0, 1.0);
            let (route, route_reason) = if area_ratio >= 0.75 {
                ("ocr", "page_dominant_raster")
            } else if area_ratio < 0.01 {
                ("ignore", "tiny_raster")
            } else if bbox[3] <= geometry.height * 0.12 || bbox[1] >= geometry.height * 0.88 {
                ("ignore", "page_edge_decoration")
            } else {
                ("vision", "meaningful_embedded_raster")
            };
            ImageBlock {
                id: format!("p{:04}-image-{index:03}", item.page),
                page_index: usize::try_from(item.page.saturating_sub(1)).unwrap_or(usize::MAX),
                page_number: item.page,
                bbox,
                source_name: item
                    .text
                    .strip_prefix("[Image: ")
                    .and_then(|value| value.strip_suffix(']'))
                    .unwrap_or(&item.text)
                    .to_owned(),
                area_ratio: round3(area_ratio),
                route: route.to_owned(),
                route_reason: route_reason.to_owned(),
            }
        })
        .collect()
}

fn apply_fidelity(item: &mut TextItem) {
    let Some(fidelity) = item.fidelity.as_ref() else {
        return;
    };
    item.text.clone_from(&fidelity.text);
    item.font.clone_from(&fidelity.font);
    item.is_italic = fidelity.flags & 2 != 0;
    item.is_bold = fidelity.flags & 16 != 0;
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
    let mut unique = Vec::with_capacity(items.len());
    for item in items.drain(..) {
        let duplicate = unique.last().is_some_and(|previous| {
            same_painted_text(previous, &item)
                || item.fidelity.as_ref().is_some_and(|fidelity| {
                    let mut visible = fidelity
                        .glyphs
                        .iter()
                        .filter(|glyph| !glyph.text.chars().all(char::is_whitespace));
                    let first = visible.next();
                    first.is_some_and(|glyph| painted_glyph_exists_in(previous, &item, glyph))
                        && visible.all(|glyph| painted_glyph_exists_in(previous, &item, glyph))
                })
        });
        if duplicate {
            continue;
        }
        unique.push(item);
    }
    *items = unique;
}

fn has_extraction_evidence(item: &TextItem) -> bool {
    !matches!(&item.item_type, ItemType::Text) || !item.text.is_empty()
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

fn normalize_text(value: &str) -> String {
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
    result
}

#[derive(Debug)]
struct AssembledSpan {
    item_index: usize,
    bbox_start_item: usize,
    bbox_end_item: usize,
    span_index: usize,
    start: usize,
    end: usize,
}

#[derive(Debug)]
struct AssembledLine {
    text: String,
    spans: Vec<AssembledSpan>,
    scalar_bytes: Vec<usize>,
}

impl AssembledLine {
    fn byte(&self, scalar: usize) -> usize {
        self.scalar_bytes
            .get(scalar)
            .copied()
            .unwrap_or(self.text.len())
    }
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
    let mut span_index = 0;
    let mut previous_x1 = None;
    let mut previous_trailing_boundary = false;
    let mut pending_leading_whitespace = None::<(usize, usize)>;
    let mut next_content = vec![None; text_line.items.len()];
    let mut next = None;
    for (index, item) in text_line.items.iter().enumerate().rev() {
        next_content[index] = next;
        if !item.text.trim().is_empty() {
            next = Some(index);
        }
    }
    let mut previous_content = None;

    for (item_index, item) in text_line.items.iter().enumerate() {
        if item.text.trim().is_empty() {
            let previous = previous_content.map(|index| &text_line.items[index]);
            let next = next_content[item_index].map(|index| &text_line.items[index]);
            let extends_previous =
                previous.is_some_and(|candidate| same_renderer_span(candidate, item));
            if extends_previous || next.is_some_and(|candidate| same_renderer_span(candidate, item))
            {
                let start = offset;
                let whitespace = normalize_text(&item.text);
                raw_text.push_str(&whitespace);
                offset += whitespace.chars().count();
                if extends_previous {
                    if let Some(previous) = spans.last_mut() {
                        previous.end = offset;
                        previous.bbox_end_item = item_index;
                    }
                } else {
                    pending_leading_whitespace.get_or_insert((start, item_index));
                }
            }
            continue;
        }
        previous_content = Some(item_index);
        span_index += 1;
        let boundary_text = item.text.replace('\u{feff}', "");
        let leading_boundary = boundary_text.starts_with('\u{200b}');
        let trailing_boundary = boundary_text.ends_with('\u{200b}');
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
        let (start, bbox_start_item) = pending_leading_whitespace
            .take()
            .unwrap_or((offset, item_index));
        raw_text.push_str(&span_text);
        offset += span_text.chars().count();
        spans.push(AssembledSpan {
            item_index,
            bbox_start_item,
            bbox_end_item: item_index,
            span_index,
            start,
            end: offset,
        });
    }

    let leading = raw_text
        .chars()
        .take_while(|character| character.is_whitespace())
        .count();
    let text = raw_text.trim().to_owned();
    let text_len = text.chars().count();
    for span in &mut spans {
        span.start = span.start.saturating_sub(leading).min(text_len);
        span.end = span
            .end
            .saturating_sub(leading)
            .max(span.start)
            .min(text_len);
    }
    spans.retain(|span| span.start < span.end);
    let scalar_bytes = text
        .char_indices()
        .map(|(byte, _)| byte)
        .chain(std::iter::once(text.len()))
        .collect();
    AssembledLine {
        text,
        spans,
        scalar_bytes,
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn median_f32(mut values: Vec<f32>) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f32::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
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
    let mut spans = Vec::<(Span, usize)>::new();
    for source_span in &assembled.spans {
        let item = &text_line.items[source_span.item_index];
        let start_byte = assembled.byte(source_span.start);
        let end_byte = assembled.byte(source_span.end);
        let span_text = assembled.text[start_byte..end_byte].to_owned();
        let superscript = is_superscript(item, body_size, baseline, source_span.start > 0);
        let span = Span {
            id: format!("{id}-s{:03}", source_span.span_index),
            text: span_text,
            bbox: union_bbox(
                text_line.items[source_span.bbox_start_item..=source_span.bbox_end_item]
                    .iter()
                    .map(|item| item_bbox(item, page_height)),
            ),
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
        if let Some((previous, previous_item_index)) = spans.last_mut() {
            let gap_start = assembled.byte(previous.end);
            let gap_end = assembled.byte(span.start);
            if same_renderer_span(&text_line.items[*previous_item_index], item)
                && same_source_text_object(&text_line.items[*previous_item_index], item)
                && assembled.text[gap_start..gap_end]
                    .chars()
                    .all(char::is_whitespace)
            {
                previous.end = span.end;
                previous.text = assembled.text[assembled.byte(previous.start)..end_byte].to_owned();
                previous.bbox = union_bbox([previous.bbox, span.bbox]);
                *previous_item_index = source_span.item_index;
                continue;
            }
        }
        spans.push((span, source_span.item_index));
    }
    spans
        .into_iter()
        .enumerate()
        .map(|(index, (mut span, _))| {
            span.id = format!("{id}-s{:03}", index + 1);
            span
        })
        .collect()
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
        let start_byte = assembled.byte(source_span.start);
        let end_byte = assembled.byte(source_span.end);
        let target = &assembled.text[start_byte..end_byte];
        let item_text = normalize_text(&fidelity.text);
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
    let mut words = Vec::new();
    let mut start = None;
    for (char_offset, (byte, character)) in text
        .char_indices()
        .chain(std::iter::once((text.len(), ' ')))
        .enumerate()
    {
        if character.is_whitespace() {
            if let Some((begin, start)) = start.take() {
                let end = char_offset;
                let mut boxes = glyphs
                    .iter()
                    .filter(|glyph| glyph.start < end && start < glyph.end)
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
    let mut sizes = Vec::new();
    for span in line
        .spans
        .iter()
        .filter(|span| span.size > 0.0 && !span.superscript)
    {
        sizes.extend(std::iter::repeat_n(
            span.size,
            span.text.chars().count().clamp(1, 100),
        ));
    }
    if sizes.is_empty() {
        for span in line.spans.iter().filter(|span| span.size > 0.0) {
            sizes.extend(std::iter::repeat_n(
                span.size,
                span.text.chars().count().clamp(1, 100),
            ));
        }
    }
    median(sizes)
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

fn separator_y(
    geometry: PageGeometry,
    lines: &[Line],
    rules: &[PdfLine],
) -> Option<f64> {
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
    let rtl = pdf_inspector_core::text_utils::is_rtl_text(
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
    for line in &mut lines {
        line.y = median_f32(line.items.iter().map(|item| item.y).collect());
    }
    lines
}

fn make_ocr_line(result: OcrLine, page_index: usize, local_index: usize) -> Option<Line> {
    let text = result.text.trim().to_owned();
    if text.is_empty() {
        return None;
    }
    let words = result
        .words
        .into_iter()
        .enumerate()
        .map(|(index, word)| Word {
            id: format!(
                "p{:04}-l{:04}-w{:03}",
                page_index + 1,
                local_index,
                index + 1
            ),
            text: word.text,
            bbox: word.bbox,
            start: word.start,
            end: word.end,
        })
        .collect();
    Some(Line {
        id: format!("p{:04}-l{:04}", page_index + 1, local_index),
        page_index,
        page_number: u32::try_from(page_index + 1).unwrap_or(u32::MAX),
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
    let mut replacements = 0;
    let mut trimmed = 0;
    let mut trailing_whitespace = 0;
    let mut has_text = false;
    for (index, line) in lines.iter().enumerate() {
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
    let printable_share = lines
        .iter()
        .map(|line| printable.find_iter(&line.text).count())
        .sum::<usize>() as f64
        / count;
    let quantity = (trimmed as f64 / 100.0).min(1.0);
    ((quantity * printable_share * (1.0 - (replacement_share * 20.0).min(1.0))).max(0.0) * 10_000.0)
        .round_ties_even()
        / 10_000.0
}

fn pdf_type_name(value: pdf_inspector_detector::PdfType) -> &'static str {
    match value {
        pdf_inspector_detector::PdfType::TextBased => "TextBased",
        pdf_inspector_detector::PdfType::Scanned => "Scanned",
        pdf_inspector_detector::PdfType::ImageBased => "ImageBased",
        pdf_inspector_detector::PdfType::Mixed => "Mixed",
    }
}

fn metadata_text(document: &Document, value: &Object, depth: u8) -> Option<String> {
    if depth > 8 {
        return None;
    }
    match value {
        Object::String(_, _) => lopdf::decode_text_string(value).ok(),
        Object::Name(value) => Some(String::from_utf8_lossy(value).into_owned()),
        Object::Reference(id) => document
            .get_object(*id)
            .ok()
            .and_then(|value| metadata_text(document, value, depth + 1)),
        _ => None,
    }
    .filter(|value| !value.is_empty())
}

fn source_metadata(document: &Document) -> Map<String, Value> {
    let mut metadata = Map::new();
    for key in [
        "author",
        "creationDate",
        "creator",
        "keywords",
        "modDate",
        "producer",
        "subject",
        "title",
        "trapped",
    ] {
        metadata.insert(key.to_owned(), Value::String(String::new()));
    }
    metadata.insert("encryption".to_owned(), Value::Null);
    let info = document
        .trailer
        .get(b"Info")
        .ok()
        .and_then(|value| match value {
            Object::Reference(id) => document.get_dictionary(*id).ok(),
            Object::Dictionary(value) => Some(value),
            _ => None,
        });
    for (json_key, pdf_key) in [
        ("author", b"Author".as_slice()),
        ("creationDate", b"CreationDate".as_slice()),
        ("creator", b"Creator".as_slice()),
        ("keywords", b"Keywords".as_slice()),
        ("modDate", b"ModDate".as_slice()),
        ("producer", b"Producer".as_slice()),
        ("subject", b"Subject".as_slice()),
        ("title", b"Title".as_slice()),
        ("trapped", b"Trapped".as_slice()),
    ] {
        if let Some(value) = info
            .and_then(|dictionary| dictionary.get(pdf_key).ok())
            .and_then(|value| metadata_text(document, value, 0))
        {
            metadata.insert(json_key.to_owned(), Value::String(value));
        }
    }
    let catalog_version = document
        .trailer
        .get(b"Root")
        .ok()
        .and_then(|value| value.as_reference().ok())
        .and_then(|id| document.get_dictionary(id).ok())
        .and_then(|catalog| catalog.get(b"Version").ok())
        .and_then(|value| metadata_text(document, value, 0));
    metadata.insert(
        "format".to_owned(),
        Value::String(format!(
            "PDF {}",
            catalog_version.as_deref().unwrap_or(&document.version)
        )),
    );
    metadata
}

#[derive(Serialize, Deserialize)]
pub struct ExtractedPdf {
    pub pages: Vec<Page>,
    pub tables: Vec<TableBlock>,
    pub images: Vec<ImageBlock>,
    pub separators: Vec<Option<f64>>,
    pub diagnostics: Vec<Diagnostic>,
    pub metadata: Map<String, Value>,
}

pub struct PdfInspection {
    pub page_count: u32,
    pub pdf_type: &'static str,
    pub confidence: f32,
    pub pages_needing_ocr: Vec<u32>,
}

pub fn inspect_pdf(path: &Path) -> Result<PdfInspection> {
    let inspection = pdf_inspector_detector::detector::detect_pdf_type(path)?;
    Ok(PdfInspection {
        page_count: inspection.page_count,
        pdf_type: pdf_type_name(inspection.pdf_type),
        confidence: inspection.confidence,
        pages_needing_ocr: inspection.pages_needing_ocr,
    })
}

pub fn load_extraction_document(path: &Path) -> Result<Document> {
    if let Some(mut document) = Document::load(path)
        .ok()
        .filter(|document| !document.is_encrypted())
    {
        document
            .objects
            .values_mut()
            .for_each(prune_extraction_object);
        return Ok(document);
    }
    pdf_inspector_loader::load_document_from_path(path)
        .map(|value| value.0)
        .map_err(Into::into)
}

pub fn pdf_page_count(bytes: &[u8]) -> Result<u32> {
    let count = Document::load_metadata_mem(bytes)?.page_count;
    if count == 0 {
        return Err(Error::Message("PDF contains no pages".to_owned()));
    }
    Ok(count)
}

pub fn page_dimensions(document: &Document) -> (Vec<(u32, f64, f64)>, PageGeometryMap) {
    let geometries: PageGeometryMap = document
        .get_pages()
        .into_iter()
        .map(|(page, id)| (page, (id, page_geometry(document, id))))
        .collect();
    let dimensions = geometries
        .iter()
        .map(|(&page, (_, geometry))| (page, geometry.raw_width, geometry.raw_height))
        .collect();
    (dimensions, geometries)
}

pub fn project_table(
    geometries: &PageGeometryMap,
    page: u32,
    index: usize,
    bbox: [f32; 4],
    cells: Vec<Vec<String>>,
    method: &str,
    confidence: f64,
) -> Option<TableBlock> {
    let geometry = geometries.get(&page)?.1;
    Some(TableBlock {
        id: format!("p{page:04}-table-{:03}", index + 1),
        page_index: usize::try_from(page.saturating_sub(1)).unwrap_or(usize::MAX),
        page_number: page,
        bbox: source_bbox(geometry, bbox),
        cells,
        provenance: format!("pdf-inspector:{method}"),
        confidence,
    })
}

pub fn assemble_pdf(
    path: &Path,
    document: Document,
    geometries: &PageGeometryMap,
    mut items: Vec<TextItem>,
    painted_rules: Vec<PdfLine>,
    detection_evidence: Vec<PageDetectionEvidence>,
    tables: Vec<TableBlock>,
    ocr: Option<&mut dyn PdfOcrProvider>,
    ocr_pages: Option<&[usize]>,
) -> Result<ExtractedPdf> {
    if geometries.is_empty() {
        return Err(Error::Message("PDF has no pages".to_owned()));
    }
    let title = get_document_title(&document);
    let mut metadata = source_metadata(&document);
    drop(document);

    let mut by_page: HashMap<u32, Vec<TextLine>> = HashMap::new();
    for item in &mut items {
        if let Some((_, geometry)) = geometries.get(&item.page) {
            transform_item(item, *geometry);
            apply_fidelity(item);
        }
    }
    items.retain(has_extraction_evidence);
    deduplicate_painted_text(&mut items);
    assign_renderer_layout(&mut items);
    let images = detected_images(&items, &geometries);
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

    let detection = legal_pdf_core::profile::measure("extract.classify_pdf", || {
        detect_from_page_evidence(
            &detection_evidence,
            title,
            &pdf_inspector_detector::DetectionConfig::default(),
        )
    });
    for page in &detection.pages_needing_ocr {
        if let Ok(index) = usize::try_from(page.saturating_sub(1)) {
            weak_pages.insert(index);
        }
    }

    let detected_weak_pages = weak_pages.iter().copied().collect::<Vec<_>>();
    let routed_pages = match ocr_pages {
        Some(selected) => detected_weak_pages
            .into_iter()
            .filter(|page| selected.contains(page))
            .collect(),
        None => detected_weak_pages,
    };
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
            provider.extract_pages(path, &requests)
        })?;
        for result in results {
            let Some(page) = pages.get_mut(result.page_index) else {
                return Err(Error::Message(format!(
                    "OCR returned an unknown page index: {}",
                    result.page_index
                )));
            };
            let mut lines: Vec<_> = result
                .lines
                .into_iter()
                .enumerate()
                .filter_map(|(index, line)| make_ocr_line(line, result.page_index, index + 1))
                .collect();
            for (index, line) in lines.iter_mut().enumerate() {
                line.id = format!("p{:04}-l{:04}", page.number, index + 1);
                line.block_index = index + 1;
                for (word_index, word) in line.words.iter_mut().enumerate() {
                    word.id = format!("{}-w{:03}", line.id, word_index + 1);
                }
            }
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

    metadata.insert(
        "extractor".to_owned(),
        Value::String("pdf-inspector".to_owned()),
    );
    metadata.insert(
        "extractor_version".to_owned(),
        Value::String("1.14.0".to_owned()),
    );
    metadata.insert("page_count".to_owned(), json!(pages.len()));
    metadata.insert(
        "pdf_type".to_owned(),
        Value::String(pdf_type_name(detection.pdf_type).to_owned()),
    );
    metadata.insert("confidence".to_owned(), json!(detection.confidence));
    metadata.insert("pages_needing_ocr".to_owned(), json!(unresolved_pages));
    metadata.insert("ocr_routed_pages".to_owned(), json!(routed_pages));
    let rotated_pages: Vec<_> = geometries
        .iter()
        .filter_map(|(&page, (_, geometry))| {
            (geometry.rotation != 0).then_some(json!({
                "page": page,
                "rotation": geometry.rotation,
            }))
        })
        .collect();
    if !rotated_pages.is_empty() {
        metadata.insert(
            "normalized_rotated_pages".to_owned(),
            Value::Array(rotated_pages),
        );
    }
    Ok(ExtractedPdf {
        pages,
        tables,
        images,
        separators,
        diagnostics,
        metadata,
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
        let (_, filtered) = extraction_object_filter((1, 0), &mut sparse).unwrap();
        assert!(matches!(sparse, Object::Null));
        assert!(matches!(filtered, Object::Null));

        let mut page_tree = Object::Array(
            (1..=4096)
                .map(|index| Object::Reference((index, 0)))
                .collect(),
        );
        extraction_object_filter((2, 0), &mut page_tree);
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
        item.fidelity = Some(Box::new(pdf_inspector_core::types::FidelityTextInfo {
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
        apply_fidelity(&mut item);

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
    fn source_metadata_preserves_info_strings_and_catalog_version() {
        use lopdf::{dictionary, text_string};

        let mut document = Document::with_version("1.4");
        let producer = document.add_object(text_string("Indirect Producer"));
        let info = document.add_object(dictionary! {
            "Author" => text_string("Renée Author"),
            "CreationDate" => text_string("D:20260811170703-06'00'"),
            "Title" => text_string(""),
            "Trapped" => Object::Name(b"False".to_vec()),
            "Producer" => Object::Reference(producer),
        });
        let catalog = document.add_object(dictionary! {
            "Type" => "Catalog",
            "Version" => Object::Name(b"1.7".to_vec()),
        });
        document.trailer.set("Info", Object::Reference(info));
        document.trailer.set("Root", Object::Reference(catalog));

        let metadata = source_metadata(&document);

        assert_eq!(metadata["author"], "Renée Author");
        assert_eq!(metadata["creationDate"], "D:20260811170703-06'00'");
        assert_eq!(metadata["trapped"], "False");
        assert_eq!(metadata["producer"], "Indirect Producer");
        assert_eq!(metadata["format"], "PDF 1.7");
        assert_eq!(metadata["title"], "");
        assert!(metadata["encryption"].is_null());
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
