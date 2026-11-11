//! Editable highlight and margin annotations written into a PDF as an incremental update (ISO 32000-1, 7.5.6):
//! the original bytes stay as they are, and the new annotations, the pages that list them and, when asked, the
//! disarmed dictionaries follow them. The same annotations as shared/pdf-annotation-writer.mjs writes: standard
//! /Highlight (or /Square for a margin mark) with printable appearance streams, never a change to page contents.
use legal_pdf_core::{Error, Result};
use lopdf::{dictionary, Dictionary, Document, IncrementalDocument, Object, ObjectId, Stream, StringFormat};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// A mark as `decodeAnnotationSet` (shared/pdf-annotations.mjs) returns it: rects are page fractions
/// [left, top, right, bottom] with the origin at the top left of the page as it is displayed.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
pub struct AnnotationMark {
    pub id: String,
    pub kind: String,
    #[serde(default)]
    pub origin: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub excerpt: String,
    pub rgb: [f64; 3],
    pub opacity: f64,
    pub fragments: Vec<AnnotationFragment>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
#[serde(rename_all = "camelCase")]
pub struct AnnotationFragment {
    pub page_number: u32,
    pub rects: Vec<[f64; 4]>,
}

fn fail(message: impl Into<String>) -> Error { Error::Message(message.into()) }
fn lo(error: lopdf::Error) -> Error { fail(format!("PDF could not be annotated: {error}")) }

/// The PDF with `marks` added as annotations, appended to `bytes`; with `disarm`, also what `removeActiveContent`
/// (shared/runtime/pdfAssembly.mjs) takes out: everything the PDF would run, launch, submit or fetch.
pub fn write_annotations(bytes: &[u8], marks: &[AnnotationMark], namespace: &str, disarm: bool) -> Result<Vec<u8>> {
    let previous = Document::load_mem(bytes).map_err(lo)?;
    if previous.is_encrypted() { return Err(fail("An encrypted PDF cannot be annotated.")); }
    let pages = previous.get_pages();
    let mut update = IncrementalDocument::create_from(bytes.to_vec(), previous);
    for mark in marks {
        let highlight = match mark.kind.as_str() { "highlight" => true, "margin" => false, _ => return Err(fail("Invalid PDF annotations.")) };
        for fragment in &mark.fragments {
            let page_id = *pages.get(&fragment.page_number).ok_or_else(|| fail("An annotation refers to a missing PDF page."))?;
            let (crop, rotation) = page_frame(update.get_prev_documents(), page_id)?;
            let quads = fragment.rects.iter().map(|rect| quad(*rect, crop, rotation)).collect::<Result<Vec<_>>>()?;
            // Squares cannot express disconnected geometry; one per margin segment.
            let groups: Vec<Vec<[f64; 8]>> = if highlight { vec![quads] } else { quads.into_iter().map(|q| vec![q]).collect() };
            for (part, group) in groups.iter().enumerate() {
                let annotation = annotation(&mut update, mark, highlight, group, &format!("{namespace}:{}:{}:{part}", mark.id, fragment.page_number));
                add_annotation(&mut update, page_id, annotation)?;
            }
        }
    }
    if disarm { disarm_document(&mut update)?; }
    let mut out = Vec::with_capacity(bytes.len() + 4096);
    update.save_to(&mut out).map_err(|e| fail(format!("PDF could not be saved: {e}")))?;
    Ok(out)
}

fn number(object: &Object) -> Option<f64> { object.as_float().ok().map(f64::from).or_else(|| object.as_i64().ok().map(|v| v as f64)) }

/// The page's crop box (falling back to its media box), inherited through /Parent, and its /Rotate.
pub(crate) fn page_frame(document: &Document, page_id: ObjectId) -> Result<([f64; 4], i64)> {
    let (mut crop, mut media, mut rotate) = (None, None, None);
    let mut node = document.get_dictionary(page_id).map_err(lo)?;
    for _ in 0..64 {
        let rect = |key: &[u8]| node.get(key).ok().and_then(|o| document.dereference(o).ok()).and_then(|(_, o)| o.as_array().ok())
            .and_then(|a| (a.len() == 4).then(|| a.iter().map(|v| document.dereference(v).ok().and_then(|(_, v)| number(v))).collect::<Option<Vec<_>>>()).flatten());
        crop = crop.or_else(|| rect(b"CropBox"));
        media = media.or_else(|| rect(b"MediaBox"));
        rotate = rotate.or_else(|| node.get(b"Rotate").ok().and_then(|o| document.dereference(o).ok()).and_then(|(_, o)| o.as_i64().ok()));
        match node.get(b"Parent").and_then(|p| p.as_reference()) { Ok(parent) => node = document.get_dictionary(parent).map_err(lo)?, Err(_) => break }
    }
    let r = crop.or(media).ok_or_else(|| fail("Invalid PDF geometry."))?;
    let (x0, y0, x1, y1) = (r[0].min(r[2]), r[1].min(r[3]), r[0].max(r[2]), r[1].max(r[3]));
    Ok(([x0, y0, x1 - x0, y1 - y0], rotate.unwrap_or(0)))
}

/// shared/pdf-annotations.mjs `rectToPdfQuad`: page fractions to a QuadPoints quad in user space.
pub(crate) fn quad(rect: [f64; 4], crop: [f64; 4], rotation: i64) -> Result<[f64; 8]> {
    let [cx, cy, cw, ch] = crop;
    if !rect.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)) || rect[0] >= rect[2] || rect[1] >= rect[3] || cw <= 0.0 || ch <= 0.0 {
        return Err(fail("Invalid PDF geometry."));
    }
    let angle = rotation.rem_euclid(360);
    let point = |u: f64, v: f64| -> Result<[f64; 2]> {
        let (x, y) = match angle { 0 => (u, 1.0 - v), 90 => (v, u), 180 => (1.0 - u, v), 270 => (1.0 - v, 1.0 - u), _ => return Err(fail("Unsupported PDF rotation.")) };
        Ok([cx + x * cw, cy + y * ch])
    };
    let (a, b, c, d) = (point(rect[0], rect[1])?, point(rect[2], rect[1])?, point(rect[0], rect[3])?, point(rect[2], rect[3])?);
    Ok([a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]])
}

fn bounds(quads: &[[f64; 8]]) -> [f64; 4] {
    let xs = quads.iter().flat_map(|q| [q[0], q[2], q[4], q[6]]);
    let ys = quads.iter().flat_map(|q| [q[1], q[3], q[5], q[7]]);
    let min = |i: &mut dyn Iterator<Item = f64>| i.fold(f64::INFINITY, f64::min);
    let max = |i: &mut dyn Iterator<Item = f64>| i.fold(f64::NEG_INFINITY, f64::max);
    [min(&mut xs.clone()), min(&mut ys.clone()), max(&mut xs.clone()), max(&mut ys.clone())]
}

/// PDF numbers have no exponent: round away float noise such as 1.4e-14 (as the JS writer's `number`).
fn fmt(value: f64) -> String {
    let rounded = (value * 10_000.0).round() / 10_000.0;
    let text = format!("{rounded:.4}");
    let text = text.trim_end_matches('0').trim_end_matches('.');
    if text == "-0" || text.is_empty() { "0".into() } else { text.into() }
}
fn real(value: f64) -> Object { Object::Real(value as f32) }

/// The annotation dictionary and its appearance stream, added to the update; returns the annotation's id.
fn annotation(update: &mut IncrementalDocument, mark: &AnnotationMark, highlight: bool, group: &[[f64; 8]], name: &str) -> ObjectId {
    let [x0, y0, x1, y1] = bounds(group);
    let shapes = group.iter().map(|q| {
        let point = |i: usize| format!("{} {}", fmt(q[i] - x0), fmt(q[i + 1] - y0));
        format!("{} m {} l {} l {} l h f", point(0), point(2), point(6), point(4))
    }).collect::<Vec<_>>().join("\n");
    let rgb = mark.rgb.iter().map(|v| fmt(*v)).collect::<Vec<_>>().join(" ");
    let content = format!("q /GS0 gs {rgb} rg\n{shapes}\nQ").into_bytes();
    let rgb_array = || Object::Array(mark.rgb.iter().map(|v| real(*v)).collect());
    let appearance = Stream::new(dictionary! {
        "Type" => "XObject", "Subtype" => "Form",
        "BBox" => vec![0.into(), 0.into(), real(x1 - x0), real(y1 - y0)],
        "Resources" => dictionary! { "ExtGState" => dictionary! { "GS0" => dictionary! {
            "Type" => "ExtGState", "ca" => real(mark.opacity), "CA" => real(mark.opacity), "BM" => "Multiply" } } },
    }, content);
    let document = &mut update.new_document;
    let appearance_id = document.add_object(appearance);
    // UTF-16BE with a byte order mark, as pdf-lib's PDFHexString.fromText writes it.
    let mut nm = vec![0xFE, 0xFF];
    nm.extend(name.encode_utf16().flat_map(u16::to_be_bytes));
    let mut annotation = dictionary! {
        "Type" => "Annot", "Subtype" => if highlight { "Highlight" } else { "Square" },
        "Rect" => vec![real(x0), real(y0), real(x1), real(y1)],
        "C" => rgb_array(), "CA" => real(mark.opacity), "Border" => vec![0.into(), 0.into(), 0.into()], "F" => 4,
        "NM" => Object::String(nm, StringFormat::Hexadecimal),
        "AP" => dictionary! { "N" => appearance_id },
    };
    if highlight { annotation.set("QuadPoints", Object::Array(group.iter().flatten().map(|v| real(*v)).collect())); }
    else { annotation.set("IC", rgb_array()); }
    document.add_object(annotation)
}

/// Appends the annotation to the page's /Annots: the page (and an /Annots array kept as its own object) are
/// copied into the update and changed there.
fn add_annotation(update: &mut IncrementalDocument, page_id: ObjectId, annotation: ObjectId) -> Result<()> {
    update.opt_clone_object_to_new_document(page_id).map_err(lo)?;
    let annots = update.new_document.get_dictionary(page_id).map_err(lo)?.get(b"Annots").ok().cloned();
    match annots {
        Some(Object::Reference(array_id)) => {
            update.opt_clone_object_to_new_document(array_id).map_err(lo)?;
            match update.new_document.get_object_mut(array_id).map_err(lo)? {
                Object::Array(items) => items.push(annotation.into()),
                _ => { update.new_document.get_dictionary_mut(page_id).map_err(lo)?.set("Annots", vec![annotation.into()]); }
            }
        }
        Some(Object::Array(mut items)) => {
            items.push(annotation.into());
            update.new_document.get_dictionary_mut(page_id).map_err(lo)?.set("Annots", items);
        }
        _ => { update.new_document.get_dictionary_mut(page_id).map_err(lo)?.set("Annots", vec![annotation.into()]); }
    }
    Ok(())
}

// ---- removeActiveContent, as an incremental update: changed dictionaries are rewritten in the update. ----

fn text_of(document: &Document, dict: &Dictionary, key: &[u8]) -> Option<String> {
    let (_, object) = document.dereference(dict.get(key).ok()?).ok()?;
    lopdf::decode_text_string(object).ok().map(|s| s.trim().to_string())
}

/// Whether an action only moves the reader: to a place in the document, to a web or mail address, or to a PDF
/// named relatively beside it (pdfAssembly.mjs `inertAction`).
pub(crate) fn inert(document: &Document, action: &Dictionary) -> bool {
    let kind = action.get(b"S").ok().and_then(|o| document.dereference(o).ok()).and_then(|(_, o)| o.as_name().ok().map(<[u8]>::to_vec));
    match kind.as_deref() {
        Some(b"GoTo") => true,
        Some(b"URI") => text_of(document, action, b"URI").is_some_and(|u| { let l = u.to_ascii_lowercase(); l.starts_with("http:") || l.starts_with("https:") || l.starts_with("mailto:") }),
        Some(b"GoToR") => text_of(document, action, b"F").is_some_and(|f| !f.is_empty() && !f.starts_with(['\\', '/']) && !f.contains(':')),
        _ => false,
    }
}

/// The dictionary at `id` with `edit` applied, written into the update only when `edit` changed it.
fn edit_object(update: &mut IncrementalDocument, id: ObjectId, edit: impl FnOnce(&Document, &mut Dictionary) -> bool) -> Result<()> {
    let current = match update.new_document.get_dictionary(id) { Ok(d) => d.clone(), Err(_) => match update.get_prev_documents().get_dictionary(id) { Ok(d) => d.clone(), Err(_) => return Ok(()) } };
    let mut changed = current.clone();
    if edit(update.get_prev_documents(), &mut changed) {
        update.new_document.objects.insert(id, Object::Dictionary(changed));
        update.new_document.max_id = update.new_document.max_id.max(id.0);
    }
    Ok(())
}

/// pdfAssembly.mjs `disarm`: drops /AA and, unless the action under `key` is inert (whose /Next chain is dropped)
/// or an array, the action itself.
pub(crate) fn disarm_dict(document: &Document, holder: &mut Dictionary, key: &[u8]) -> bool {
    let mut changed = holder.remove(b"AA").is_some();
    let action = holder.get(key).ok().cloned();
    match action {
        None => {}
        Some(object) => match document.dereference(&object) {
            Ok((_, Object::Array(_))) => {}
            Ok((_, Object::Dictionary(action))) if inert(document, action) => {
                if action.has(b"Next") {
                    let mut action = action.clone();
                    action.remove(b"Next");
                    holder.set(key.to_vec(), Object::Dictionary(action));
                    changed = true;
                }
            }
            _ => { holder.remove(key); changed = true; }
        },
    }
    changed
}

fn reference(dict: &Dictionary, key: &[u8]) -> Option<ObjectId> { dict.get(key).ok()?.as_reference().ok() }

fn disarm_document(update: &mut IncrementalDocument) -> Result<()> {
    let prev = update.get_prev_documents();
    let catalog_id = prev.trailer.get(b"Root").and_then(|o| o.as_reference()).map_err(lo)?;
    let catalog = prev.get_dictionary(catalog_id).map_err(lo)?.clone();
    let pages: Vec<ObjectId> = prev.get_pages().into_values().collect();
    let names = reference(&catalog, b"Names");
    let form = reference(&catalog, b"AcroForm");
    let outlines = reference(&catalog, b"Outlines");
    edit_object(update, catalog_id, |d, c| {
        let mut changed = disarm_dict(d, c, b"OpenAction");
        // Inline /Names or /AcroForm dictionaries are edited here; referenced ones below.
        if let Ok(Object::Dictionary(n)) = c.get_mut(b"Names") { changed |= n.remove(b"JavaScript").is_some(); }
        if let Ok(Object::Dictionary(f)) = c.get_mut(b"AcroForm") { changed |= f.remove(b"XFA").is_some(); }
        changed
    })?;
    if let Some(id) = names { edit_object(update, id, |_, n| n.remove(b"JavaScript").is_some())?; }
    let mut seen = HashSet::new();
    // Form fields and their kids.
    let mut stack: Vec<ObjectId> = Vec::new();
    if let Some(id) = form {
        edit_object(update, id, |_, f| f.remove(b"XFA").is_some())?;
        if let Ok(fields) = update.get_prev_documents().get_dictionary(id).and_then(|f| f.get(b"Fields")).and_then(|o| o.as_array()) {
            stack.extend(fields.iter().filter_map(|o| o.as_reference().ok()));
        }
    }
    while let Some(id) = stack.pop() {
        if !seen.insert(id) { continue; }
        edit_object(update, id, |d, f| disarm_dict(d, f, b"A"))?;
        if let Ok(kids) = update.get_prev_documents().get_dictionary(id).and_then(|f| f.get(b"Kids")).and_then(|o| o.as_array()) {
            stack.extend(kids.iter().filter_map(|o| o.as_reference().ok()));
        }
    }
    // Pages, their annotations, and the fields above each widget.
    for page_id in pages {
        edit_object(update, page_id, |_, p| p.remove(b"AA").is_some())?;
        let annots = update.new_document.get_dictionary(page_id).ok().or_else(|| update.get_prev_documents().get_dictionary(page_id).ok())
            .and_then(|p| p.get(b"Annots").ok().cloned());
        let list = match annots {
            Some(Object::Reference(id)) => update.get_prev_documents().get_object(id).ok().and_then(|o| o.as_array().ok().cloned()),
            Some(Object::Array(items)) => Some(items),
            _ => None,
        };
        for item in list.unwrap_or_default() {
            let mut node = item.as_reference().ok();
            while let Some(id) = node {
                if !seen.insert(id) { break; }
                edit_object(update, id, |d, a| disarm_dict(d, a, b"A"))?;
                node = update.get_prev_documents().get_dictionary(id).ok().and_then(|a| reference(a, b"Parent"));
            }
        }
    }
    // Bookmarks.
    let mut stack: Vec<ObjectId> = outlines.and_then(|id| update.get_prev_documents().get_dictionary(id).ok().and_then(|o| reference(o, b"First"))).into_iter().collect();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) { continue; }
        edit_object(update, id, |d, item| disarm_dict(d, item, b"A"))?;
        if let Ok(item) = update.get_prev_documents().get_dictionary(id) {
            stack.extend(reference(item, b"Next"));
            stack.extend(reference(item, b"First"));
        }
    }
    Ok(())
}
