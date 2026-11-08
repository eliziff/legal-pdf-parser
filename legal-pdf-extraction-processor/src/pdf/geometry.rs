//! A page's geometry (its box, rotation and user unit) and the transforms that bring pdf-inspector's
//! items onto it; a symbol font's private-use text mapped back.

use super::*;

pub(crate) fn prune_extraction_object(object: &mut Object) {
    if let Object::Array(values) = object {
        let nulls = (values.len() >= 4096).then(|| {
            values.iter().try_fold(0, |nulls, value| match value {
                Object::Null => Some(nulls + 1),
                Object::Reference(_) => Some(nulls),
                _ => None,
            })
        });
        if nulls.flatten().is_some_and(|nulls| nulls * 20 >= values.len() * 19) {
            *object = Object::Null;
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PageGeometry {
    pub(crate) x0: f64,
    pub(crate) y0: f64,
    pub(crate) raw_width: f64,
    pub(crate) raw_height: f64,
    pub(crate) rotation: i32,
    pub(crate) width: f64,
    pub(crate) height: f64,
}

pub type PageGeometryMap = BTreeMap<u32, (ObjectId, PageGeometry)>;

/// The page's visible box and turn as pdf-inspector reads them (the CropBox clipped to the
/// MediaBox, indirect values resolved, the turn snapped to a right angle), Letter where the page
/// names none.
pub(crate) fn page_geometry(doc: &Document, id: ObjectId) -> PageGeometry {
    let bounds = pdf_inspector::extractor::visible_page_box(doc, id).map_or([0.0, 0.0, 612.0, 792.0], |page| [page.x0, page.y0, page.x1, page.y1].map(f64::from));
    let rotation = pdf_inspector::extractor::page_rotate(doc, id).degrees() as i32;
    let (raw_width, raw_height) = ((bounds[2] - bounds[0]).max(0.0), (bounds[3] - bounds[1]).max(0.0));
    let (width, height) = if matches!(rotation, 90 | 270) { (raw_height, raw_width) } else { (raw_width, raw_height) };
    PageGeometry { x0: bounds[0], y0: bounds[1], raw_width, raw_height, rotation, width, height }
}

pub(crate) fn transform_point(geometry: PageGeometry, x: f64, y: f64) -> (f64, f64) {
    let (x, y) = (x - geometry.x0, y - geometry.y0);
    match geometry.rotation {
        90 => (y, geometry.raw_width - x),
        180 => (geometry.raw_width - x, geometry.raw_height - y),
        270 => (geometry.raw_height - y, x),
        _ => (x, y),
    }
}

pub(crate) fn transform_vector(geometry: PageGeometry, x: f32, y: f32) -> [f32; 2] {
    match geometry.rotation {
        90 => [y, -x],
        180 => [-x, -y],
        270 => [-y, x],
        _ => [x, y],
    }
}

/// The box the points span: its least and greatest x and y.
fn extent(points: impl IntoIterator<Item = (f64, f64)>) -> (f64, f64, f64, f64) {
    let empty = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    points.into_iter().fold(empty, |(x0, y0, x1, y1), (x, y)| (x0.min(x), y0.min(y), x1.max(x), y1.max(y)))
}

/// The four corners of the box (x0, y0, x1, y1) brought onto the page.
fn corners(geometry: PageGeometry, [x0, y0, x1, y1]: [f64; 4]) -> [(f64, f64); 4] {
    [transform_point(geometry, x0, y0), transform_point(geometry, x0, y1), transform_point(geometry, x1, y0), transform_point(geometry, x1, y1)]
}

pub(crate) fn transform_bbox(geometry: PageGeometry, bbox: [f32; 4]) -> ([f32; 4], bool) {
    let (x0, y0, x1, y1) = extent(corners(geometry, bbox.map(f64::from)));
    let visible = x1 >= 0.0 && x0 <= geometry.width && y1 >= 0.0 && y0 <= geometry.height;
    let (x0, y0) = (x0.clamp(0.0, geometry.width), y0.clamp(0.0, geometry.height));
    let (x1, y1) = (x1.min(geometry.width).max(x0), y1.min(geometry.height).max(y0));
    ([x0 as f32, y0 as f32, x1 as f32, y1 as f32], visible)
}

pub(crate) fn transform_fidelity_item(item: &mut TextItem, geometry: PageGeometry) -> bool {
    let Some(fidelity) = item.fidelity.as_mut() else {
        return false;
    };
    fidelity.flags |= if item.is_bold { 16 } else { 0 } | if item.is_italic { 2 } else { 0 };
    let baseline = transform_point(geometry, f64::from(fidelity.baseline[0]), f64::from(fidelity.baseline[1]));
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
        fidelity.text = fidelity.glyphs.iter().map(|glyph| glyph.text.as_str()).collect();
    }
    let (base, advance, em) = (fidelity.baseline, fidelity.advance, fidelity.em);
    // The run's corners: its start and its end, each at its descender and its ascender.
    let corner = |[x, y]: [f32; 2], height: f32| [x + height * em[0], y + height * em[1]];
    let end = [base[0] + advance[0], base[1] + advance[1]];
    let points = [corner(base, fidelity.descender), corner(base, fidelity.ascender), corner(end, fidelity.descender), corner(end, fidelity.ascender)];
    let empty = (f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    let (x0, y0, x1, y1) = points.into_iter().fold(empty, |(x0, y0, x1, y1), [x, y]| (x0.min(x), y0.min(y), x1.max(x), y1.max(y)));
    let (width, height) = (geometry.width as f32, geometry.height as f32);
    let (x0, y0) = (x0.clamp(0.0, width), y0.clamp(0.0, height));
    let (x1, y1) = (x1.min(width).max(x0), y1.min(height).max(y0));
    let visible_glyph_bbox = clipped_glyph.then(|| {
        let mut boxes = fidelity.glyphs.iter().filter(|glyph| {
            !glyph.text.chars().all(char::is_whitespace) && glyph.bbox[2] - glyph.bbox[0] > 0.001 && glyph.bbox[3] - glyph.bbox[1] > 0.001
        });
        let first = boxes.next()?;
        Some(boxes.fold(first.bbox, |bbox, glyph| [bbox[0].min(glyph.bbox[0]), bbox[1].min(glyph.bbox[1]), bbox[2].max(glyph.bbox[2]), bbox[3].max(glyph.bbox[3])]))
    });
    let bbox = visible_glyph_bbox.flatten().unwrap_or([x0, y0, x1, y1]);
    (item.x, item.y, item.width, item.height) = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
    item.font_size = fidelity.em[0].hypot(fidelity.em[1]);
    item.text = std::mem::take(&mut fidelity.text);
    item.font = std::mem::take(&mut fidelity.font);
    item.is_italic = fidelity.flags & 2 != 0;
    item.is_bold = fidelity.flags & 16 != 0;
    true
}

pub(crate) fn transform_item(item: &mut TextItem, geometry: PageGeometry) {
    if transform_fidelity_item(item, geometry) {
        return;
    }
    let (x, y) = (f64::from(item.x), f64::from(item.y));
    let (right, top) = (f64::from(item.x + item.width), f64::from(item.y + item.height));
    let (x0, y0, x1, y1) = extent([
        transform_point(geometry, x, y),
        transform_point(geometry, right, y),
        transform_point(geometry, x, top),
        transform_point(geometry, right, top),
    ]);
    item.x = x0.max(0.0) as f32;
    item.y = y0.max(0.0) as f32;
    item.width = (x1.min(geometry.width) - f64::from(item.x)).max(0.0) as f32;
    item.height = (y1.min(geometry.height) - f64::from(item.y)).max(0.0) as f32;
}

/// The fidelity text is the font's own decoding: read its symbol-font private-use codes and
/// misnamed math glyphs as pdf-inspector's presentation text does, glyph by glyph. One character
/// for one, so the item's glyphs keep their places.
pub(crate) fn unmap_symbol_text(item: &mut TextItem) {
    let font = Some(item.font.as_str());
    let clean = |text: &mut String| *text = pdf_inspector::extractor::clean_symbol_pua(std::mem::take(text), font).0;
    clean(&mut item.text);
    if let Some(fidelity) = item.fidelity.as_mut() {
        clean(&mut fidelity.text);
        for glyph in &mut fidelity.glyphs {
            clean(&mut glyph.text);
        }
    }
}

pub(crate) fn round3(value: f64) -> f64 {
    (value * 1_000.0).round() / 1_000.0
}

pub(crate) fn item_bbox(item: &TextItem, page_height: f64) -> [f64; 4] {
    let [x, y, width, height] = [item.x, item.y, item.width, item.height].map(f64::from);
    [round3(x), round3(page_height - y - height), round3(x + width), round3(page_height - y)]
}
