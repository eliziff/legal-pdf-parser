//! PDF content-stream operator state machine.
//!
//! Walks the page's content stream, tracking the graphics state and text
//! matrix, and emits `TextItem`s and `PdfRect`s.

use crate::text_utils::{
    decode_text_string, effective_font_size, expand_ligatures, is_bold_font, is_italic_font,
};
use crate::tounicode::FontCMaps;
use crate::types::{
    FidelityGlyph, FidelityTextInfo, ItemType, PageExtraction, PdfLine, PdfRect, TextItem,
};
use crate::PdfError;
use log::trace;
use lopdf::{Document, Encoding, Object, ObjectId};
use std::borrow::Cow;
use std::collections::HashMap;

use super::fonts::{
    build_font_encodings, build_font_fidelity, build_font_widths, build_type3_scales,
    compute_string_width_ts, descriptor_style_flags, extract_fidelity_glyphs,
    extract_text_from_operand, extract_text_from_operand_fidelity, get_font_file2_obj_num,
    get_operand_bytes, CMapDecisionCache, FontFidelity, FontStyleCache,
};
use super::underline::UnderlineLine;
use super::xobjects::{expand_page_content, MAX_EXPANDED_OPERATIONS};
use super::{get_number, image_bbox_from_ctm, multiply_matrices};

fn fidelity_info(
    fonts: Option<&HashMap<String, FontFidelity>>,
    resource: &str,
    text: String,
    text_object: u32,
    source_line: u32,
    text_rise: f32,
    mut glyphs: Vec<FidelityGlyph>,
    combined: [f32; 6],
    font_size: f32,
    type3_scale: f32,
    advance_ts: f32,
) -> Option<Box<FidelityTextInfo>> {
    let fonts = fonts?;
    let font = fonts.get(resource);
    let ascender = font.map_or(1.0, |value| value.ascender);
    let descender = font.map_or(-0.2, |value| value.descender);
    let axis = [combined[0], combined[1]];
    let axis_length = axis[0].hypot(axis[1]);
    let em = [
        combined[2] * font_size * type3_scale,
        combined[3] * font_size * type3_scale,
    ];
    for glyph in &mut glyphs {
        let start = glyph.x0;
        let end = glyph.x1;
        let points = [
            [
                combined[4] + start * axis[0] + descender * em[0],
                combined[5] + start * axis[1] + descender * em[1],
            ],
            [
                combined[4] + start * axis[0] + ascender * em[0],
                combined[5] + start * axis[1] + ascender * em[1],
            ],
            [
                combined[4] + end * axis[0] + descender * em[0],
                combined[5] + end * axis[1] + descender * em[1],
            ],
            [
                combined[4] + end * axis[0] + ascender * em[0],
                combined[5] + end * axis[1] + ascender * em[1],
            ],
        ];
        glyph.x0 = start * axis_length;
        glyph.x1 = end * axis_length;
        glyph.bbox = [
            points
                .iter()
                .map(|point| point[0])
                .fold(f32::INFINITY, f32::min),
            points
                .iter()
                .map(|point| point[1])
                .fold(f32::INFINITY, f32::min),
            points
                .iter()
                .map(|point| point[0])
                .fold(f32::NEG_INFINITY, f32::max),
            points
                .iter()
                .map(|point| point[1])
                .fold(f32::NEG_INFINITY, f32::max),
        ];
    }
    Some(Box::new(FidelityTextInfo {
        text,
        font: font
            .map(|value| value.name.clone())
            .unwrap_or_else(|| resource.to_owned()),
        resource: resource.to_owned(),
        flags: font.map_or(0, |value| value.flags),
        ascender,
        descender,
        text_object,
        source_line,
        renderer_line: 0,
        renderer_block: 0,
        text_rise,
        baseline: [combined[4], combined[5]],
        advance: [advance_ts * axis[0], advance_ts * axis[1]],
        em,
        glyphs,
    }))
}

pub(super) fn append_tj_text(current: &mut String, pending_space: &mut bool, text: &str) {
    let Some(first) = text.chars().next() else {
        return;
    };
    let current_has_space = current.chars().next_back().is_some_and(char::is_whitespace);
    let text_has_space = first.is_whitespace();
    if *pending_space && !current_has_space && !text_has_space {
        current.push(' ');
    }
    *pending_space = false;
    if current_has_space && text_has_space {
        current.push_str(&text[first.len_utf8()..]);
    } else {
        current.push_str(text);
    }
}

/// Strip PDF comments (% to end of line) from content stream bytes.
///
/// Some PDF generators (e.g. PD4ML) embed comments in content streams that
/// confuse lopdf's `Content::decode` parser.  Comments inside string literals
/// (parentheses) are NOT stripped — only top-level comments.
pub(super) fn strip_pdf_comments(data: &[u8]) -> Cow<'_, [u8]> {
    // Quick check: if no '%' present, return as-is (common case)
    if !data.contains(&b'%') {
        return Cow::Borrowed(data);
    }

    let mut result = Vec::with_capacity(data.len());
    let mut i = 0;
    let mut in_string = 0i32; // parenthesis nesting depth
    let mut in_hex_string = false;

    while i < data.len() {
        let b = data[i];
        match b {
            // Inside a string literal, a backslash escapes the next byte —
            // `\(`, `\)`, and `\\` must not touch the nesting depth, or a
            // later `%` glyph inside a string gets stripped as a comment,
            // corrupting the stream.
            b'\\' if in_string > 0 => {
                result.push(b);
                if let Some(&next) = data.get(i + 1) {
                    result.push(next);
                    i += 1;
                }
            }
            b'(' if !in_hex_string => {
                in_string += 1;
                result.push(b);
            }
            b')' if !in_hex_string && in_string > 0 => {
                in_string -= 1;
                result.push(b);
            }
            b'<' if in_string == 0 && !in_hex_string => {
                in_hex_string = true;
                result.push(b);
            }
            b'>' if in_hex_string => {
                in_hex_string = false;
                result.push(b);
            }
            b'%' if in_string == 0 && !in_hex_string => {
                // Skip until end of line
                while i < data.len() && data[i] != b'\n' && data[i] != b'\r' {
                    i += 1;
                }
                // Replace comment with a space to preserve token separation
                result.push(b' ');
                continue; // Don't increment i again
            }
            _ => {
                result.push(b);
            }
        }
        i += 1;
    }

    Cow::Owned(result)
}

fn transform_path_point(x: f32, y: f32, ctm: &[f32; 6]) -> (f32, f32) {
    (
        x * ctm[0] + y * ctm[2] + ctm[4],
        x * ctm[1] + y * ctm[3] + ctm[5],
    )
}

fn transformed_stroke_width(
    line_width: f32,
    ctm: &[f32; 6],
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
) -> f32 {
    let user_width = line_width.abs();
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len = (dx * dx + dy * dy).sqrt();
    if len <= f32::EPSILON {
        return user_width;
    }

    // PDF stroke width scales perpendicular to the path direction.
    let nx = -dy / len;
    let ny = dx / len;
    let ndx = nx * ctm[0] + ny * ctm[2];
    let ndy = nx * ctm[1] + ny * ctm[3];
    user_width * (ndx * ndx + ndy * ndy).sqrt()
}

/// Text rise (Ts) displaces the glyph origin by (0, rise) in unscaled text
/// space — per the rendering-matrix definition it sits left of Tm, so the
/// offset maps through the text matrix's y column. Rise never contributes
/// to the advance, so callers apply it only to the rendering position and
/// keep advancing the unshifted text matrix.
fn rise_adjusted(tm: &[f32; 6], rise: f32) -> [f32; 6] {
    if rise == 0.0 {
        return *tm;
    }
    [
        tm[0],
        tm[1],
        tm[2],
        tm[3],
        tm[4] + rise * tm[2],
        tm[5] + rise * tm[3],
    ]
}

/// Returns `(page_extraction, has_gid_fonts)` where `has_gid_fonts` indicates
/// the page uses fonts with unresolvable gid-encoded glyphs.
pub(crate) fn extract_page_text_items(
    doc: &Document,
    page_id: ObjectId,
    page_num: u32,
    font_cmaps: &FontCMaps,
    include_invisible: bool,
    style_cache: &mut FontStyleCache,
) -> Result<(PageExtraction, bool, bool), PdfError> {
    extract_page_text_items_impl(
        doc,
        page_id,
        page_num,
        font_cmaps,
        include_invisible,
        style_cache,
        true,
        true,
        false,
    )
}

pub(crate) fn extract_page_text_items_preserving_scripts(
    doc: &Document,
    page_id: ObjectId,
    page_num: u32,
    font_cmaps: &FontCMaps,
    include_invisible: bool,
    style_cache: &mut FontStyleCache,
) -> Result<(PageExtraction, bool, bool), PdfError> {
    extract_page_text_items_impl(
        doc,
        page_id,
        page_num,
        font_cmaps,
        include_invisible,
        style_cache,
        true,
        false,
        false,
    )
}

pub(crate) fn extract_page_text_items_raw(
    doc: &Document,
    page_id: ObjectId,
    page_num: u32,
    font_cmaps: &FontCMaps,
    include_invisible: bool,
    style_cache: &mut FontStyleCache,
) -> Result<(PageExtraction, bool, bool), PdfError> {
    extract_page_text_items_impl(
        doc,
        page_id,
        page_num,
        font_cmaps,
        include_invisible,
        style_cache,
        false,
        false,
        false,
    )
}

pub(crate) fn extract_page_text_items_fidelity(
    doc: &Document,
    page_id: ObjectId,
    page_num: u32,
    font_cmaps: &FontCMaps,
    include_invisible: bool,
    style_cache: &mut FontStyleCache,
) -> Result<(PageExtraction, bool, bool), PdfError> {
    extract_page_text_items_impl(
        doc,
        page_id,
        page_num,
        font_cmaps,
        include_invisible,
        style_cache,
        false,
        false,
        true,
    )
}

fn extract_page_text_items_impl(
    doc: &Document,
    page_id: ObjectId,
    page_num: u32,
    font_cmaps: &FontCMaps,
    include_invisible: bool,
    style_cache: &mut FontStyleCache,
    merge_text: bool,
    merge_scripts: bool,
    fidelity: bool,
) -> Result<(PageExtraction, bool, bool), PdfError> {
    use lopdf::content::Content;

    let mut items = Vec::new();
    let mut rects: Vec<PdfRect> = Vec::new();
    let mut clip_rects: Vec<PdfRect> = Vec::new();
    let mut lines: Vec<PdfLine> = Vec::new();
    let mut underline_lines: Vec<UnderlineLine> = Vec::new();

    // Path construction state for m/l/h → S/s line extraction
    let mut path_subpath_start: Option<(f32, f32)> = None;
    let mut path_current: Option<(f32, f32)> = None;
    let mut pending_lines: Vec<(f32, f32, f32, f32)> = Vec::new();
    // Completed subpaths (each a vec of line segments) for f/f* rect extraction
    let mut pending_subpaths: Vec<Vec<(f32, f32, f32, f32)>> = Vec::new();
    let mut fill_rects: Vec<PdfRect> = Vec::new();
    // `re` rects awaiting a paint operator. Underline detection must only
    // see painted rects: a `re W n` clip path or `re n` no-op draws nothing
    // on the page, so treating every `re` as ink would underline text that
    // merely sits near an invisible clip boundary.
    let mut pending_re_rects: Vec<PdfRect> = Vec::new();
    let mut painted_rects: Vec<PdfRect> = Vec::new();

    // Decode page and nested Form content before building font state. Form
    // resources are qualified while their operations are inlined, so every
    // text-show operation below travels through this one interpreter.
    let page_fonts = doc.get_page_fonts(page_id).unwrap_or_default();
    let raw_content = doc
        .get_page_content(page_id)
        .map_err(|e| PdfError::Parse(e.to_string()))?;
    let content_data = strip_pdf_comments(&raw_content);
    let content = Content::decode(&content_data).map_err(|e| PdfError::Parse(e.to_string()))?;
    drop(content_data);
    drop(raw_content);
    let expanded = expand_page_content(doc, page_id, content.operations, page_fonts);
    if expanded.exceeded_limit {
        log::warn!(
            "page {}: skipping extraction — expanded content exceeds {} operations",
            page_num,
            MAX_EXPANDED_OPERATIONS
        );
        return Ok(((Vec::new(), Vec::new(), Vec::new()), false, false));
    }
    let fonts = expanded.fonts;
    let font_names = expanded.font_names;
    let xobjects = expanded.images;
    let operations = expanded.operations;

    // Build font encoding maps from Differences arrays
    let (font_encodings, has_gid_fonts) = build_font_encodings(doc, &fonts, font_cmaps);

    // Build font width info for accurate text positioning
    let font_widths = build_font_widths(doc, &fonts);
    let type3_scales = build_type3_scales(doc, &fonts);

    // Build maps of font resource names to their base font names and ToUnicode object refs
    let mut font_base_names: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    let mut font_tounicode_refs: std::collections::HashMap<String, u32> =
        std::collections::HashMap::new();
    let mut inline_cmaps: std::collections::HashMap<String, crate::tounicode::CMapEntry> =
        std::collections::HashMap::new();
    let mut font_style_flags: std::collections::HashMap<String, (bool, bool)> =
        std::collections::HashMap::new();
    for (font_name, font_dict) in &fonts {
        let resource_name = String::from_utf8_lossy(font_name).to_string();
        if let Ok(base_font) = font_dict.get(b"BaseFont") {
            if let Ok(name) = base_font.as_name() {
                let base_name = String::from_utf8_lossy(name).to_string();
                font_base_names.insert(resource_name.clone(), base_name);
            }
        }
        // Descriptor style flags rescue subset fonts whose BaseFont names
        // are opaque tags the name heuristics can't read.
        let style = descriptor_style_flags(doc, font_dict, style_cache);
        if style != (false, false) {
            font_style_flags.insert(resource_name.clone(), style);
        }
        // Track ToUnicode object reference, with FontFile2 fallback for Identity-H/V.
        // Also handle inline ToUnicode streams.
        match font_dict.get(b"ToUnicode") {
            Ok(tounicode) => {
                if let Ok(obj_ref) = tounicode.as_reference() {
                    font_tounicode_refs.insert(resource_name, obj_ref.0);
                } else if let Object::Stream(s) = tounicode {
                    let data = s
                        .decompressed_content()
                        .unwrap_or_else(|_| s.content.clone());
                    if let Some(entry) =
                        crate::tounicode::build_cmap_entry_from_stream(&data, font_dict, doc, 0)
                    {
                        inline_cmaps.insert(resource_name, entry);
                    }
                }
            }
            Err(_) => {
                if let Some(ff2_obj_num) = get_font_file2_obj_num(doc, font_dict) {
                    font_tounicode_refs.insert(resource_name, ff2_obj_num);
                }
            }
        }
    }
    let font_fidelity = fidelity.then(|| build_font_fidelity(doc, &fonts, style_cache));

    // Cache font encodings from lopdf (once per font, not per text operand).
    // This avoids re-parsing ToUnicode CMap streams for every Tj/TJ operator.
    let mut encoding_cache: HashMap<String, Encoding<'_>> = HashMap::new();
    for (font_name, font_dict) in &fonts {
        let name = String::from_utf8_lossy(font_name).to_string();
        if let Ok(enc) = font_dict.get_font_encoding(doc) {
            encoding_cache.insert(name, enc);
        }
    }

    let mut cmap_decisions = CMapDecisionCache::new();

    const MAX_OPERATIONS: usize = 1_000_000;
    if operations.len() > MAX_OPERATIONS {
        log::warn!(
            "page {}: skipping extraction — {} operations exceeds limit ({})",
            page_num,
            operations.len(),
            MAX_OPERATIONS
        );
        return Ok(((Vec::new(), Vec::new(), Vec::new()), false, false));
    }

    // Graphics state tracking
    let mut ctm = [1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0]; // Current Transformation Matrix
    let mut text_rendering_mode: i32 = 0; // 0=fill, 1=stroke, 2=fill+stroke, 3=invisible
    let mut line_width: f32 = 1.0;
    #[derive(Clone)]
    struct SavedGraphicsState {
        ctm: [f32; 6],
        text_rendering_mode: i32,
        line_width: f32,
        char_spacing: f32,
        word_spacing: f32,
        horizontal_scaling: f32,
        text_rise: f32,
        text_leading: f32,
        current_font: String,
        current_font_name: String,
        current_font_size: f32,
    }
    let mut gstate_stack: Vec<SavedGraphicsState> = Vec::new();

    // Text state tracking
    let mut current_font = String::new();
    let mut current_font_name = String::new();
    let mut current_font_size: f32 = 12.0;
    let mut text_leading: f32 = 0.0; // TL parameter (in text-space units)
    let mut char_spacing: f32 = 0.0; // Tc parameter (extra spacing per character, unscaled)
    let mut word_spacing: f32 = 0.0; // Tw parameter (extra spacing per space char, unscaled)
    let mut horizontal_scaling: f32 = 1.0; // Tz percentage as a multiplier
    let mut text_rise: f32 = 0.0; // Ts parameter (baseline shift for super/subscripts, unscaled)
    let mut text_matrix = [1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
    let mut line_matrix = [1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
    let mut in_text_block = false;
    let mut text_object_index = 0u32;
    let mut source_line_index = 0u32;
    let mut source_line_has_text = false;

    // Track text direction votes: (horizontal_count, rotated_count).
    // For each text item, if |combined[0]| > |combined[1]| the text runs
    // horizontally (normal); otherwise it's rotated ~90°.
    let mut rotation_votes = RotationVotes {
        horizontal: 0,
        rotated: 0,
    };

    // Marked content tracking: (ActualText, MCID) per nesting level
    struct MarkedContentEntry {
        actual_text: Option<String>,
        mcid: Option<i64>,
    }
    let mut marked_content_stack: Vec<MarkedContentEntry> = Vec::new();
    let mut suppress_glyph_extraction = false;
    let mut actual_text_start_tm: Option<[f32; 6]> = None; // text matrix at BDC entry
    let mut actual_text_glyph_tm: Option<[f32; 6]> = None; // text matrix at first glyph inside BDC
                                                           // Text rise in effect at each captured matrix — the item must render at
                                                           // the rise of its GLYPHS, not whatever rise is set by EMC time.
    let mut actual_text_start_rise: f32 = 0.0;
    let mut actual_text_glyph_rise: Option<f32> = None;
    /// Get the innermost MCID from the marked content stack.
    fn current_mcid(stack: &[MarkedContentEntry]) -> Option<i64> {
        stack.iter().rev().find_map(|e| e.mcid)
    }

    for op in &operations {
        trace!("{} {:?}", op.operator, op.operands);
        let mut operator = op.operator.as_str();
        let mut quote_operand = None;
        if operator == "\"" && op.operands.len() >= 3 {
            word_spacing = get_number(&op.operands[0]).unwrap_or(word_spacing);
            char_spacing = get_number(&op.operands[1]).unwrap_or(char_spacing);
            quote_operand = op.operands.get(2);
            operator = "'";
        }
        match operator {
            "q" => {
                // Save graphics state
                gstate_stack.push(SavedGraphicsState {
                    ctm,
                    text_rendering_mode,
                    line_width,
                    char_spacing,
                    word_spacing,
                    horizontal_scaling,
                    text_rise,
                    text_leading,
                    current_font: current_font.clone(),
                    current_font_name: current_font_name.clone(),
                    current_font_size,
                });
            }
            "Q" => {
                // Restore graphics state
                if let Some(saved) = gstate_stack.pop() {
                    ctm = saved.ctm;
                    text_rendering_mode = saved.text_rendering_mode;
                    line_width = saved.line_width;
                    char_spacing = saved.char_spacing;
                    word_spacing = saved.word_spacing;
                    horizontal_scaling = saved.horizontal_scaling;
                    text_rise = saved.text_rise;
                    text_leading = saved.text_leading;
                    current_font = saved.current_font;
                    current_font_name = saved.current_font_name;
                    current_font_size = saved.current_font_size;
                }
            }
            "cm" => {
                // Concatenate matrix to CTM
                if op.operands.len() >= 6 {
                    let new_matrix = [
                        get_number(&op.operands[0]).unwrap_or(1.0),
                        get_number(&op.operands[1]).unwrap_or(0.0),
                        get_number(&op.operands[2]).unwrap_or(0.0),
                        get_number(&op.operands[3]).unwrap_or(1.0),
                        get_number(&op.operands[4]).unwrap_or(0.0),
                        get_number(&op.operands[5]).unwrap_or(0.0),
                    ];
                    ctm = multiply_matrices(&new_matrix, &ctm);
                }
            }
            "w" => {
                if let Some(width) = op.operands.first().and_then(get_number) {
                    line_width = width;
                }
            }
            "BT" => {
                // Begin text block
                in_text_block = true;
                text_object_index = text_object_index.saturating_add(1);
                source_line_index = source_line_index.saturating_add(1);
                source_line_has_text = false;
                text_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
                line_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
            }
            "ET" => {
                // End text block
                in_text_block = false;
            }
            "Tf" => {
                // Set font and size
                if op.operands.len() >= 2 {
                    if let Ok(name) = op.operands[0].as_name() {
                        current_font = String::from_utf8_lossy(name).to_string();
                        current_font_name = font_names
                            .get(&current_font)
                            .cloned()
                            .unwrap_or_else(|| current_font.clone());
                    }
                    if let Ok(size) = op.operands[1].as_f32() {
                        current_font_size = size;
                    } else if let Ok(size) = op.operands[1].as_i64() {
                        current_font_size = size as f32;
                    }
                }
            }
            "TL" => {
                // Set text leading (used by T*, ', and " operators)
                if let Some(tl) = op.operands.first().and_then(get_number) {
                    text_leading = tl;
                }
            }
            "Tr" => {
                // Set text rendering mode (3 = invisible / OCR overlay)
                if let Some(mode) = op.operands.first().and_then(get_number) {
                    text_rendering_mode = mode as i32;
                }
            }
            "Tc" => {
                // Set character spacing (extra space added after each character)
                if let Some(tc) = op.operands.first().and_then(get_number) {
                    char_spacing = tc;
                }
            }
            "Tw" => {
                // Set word spacing (extra space added for each space character)
                if let Some(tw) = op.operands.first().and_then(get_number) {
                    word_spacing = tw;
                }
            }
            "Tz" => {
                if let Some(tz) = op.operands.first().and_then(get_number) {
                    horizontal_scaling = tz / 100.0;
                }
            }
            "Ts" => {
                // Set text rise (baseline shift for superscripts/subscripts)
                if let Some(ts) = op.operands.first().and_then(get_number) {
                    text_rise = ts;
                }
            }
            "Td" | "TD" => {
                // Move text position: TLM = T(tx,ty) × TLM; Tm = TLM
                // tx,ty are in text space — must be scaled by the text line matrix
                if op.operands.len() >= 2 {
                    if source_line_has_text {
                        source_line_index = source_line_index.saturating_add(1);
                        source_line_has_text = false;
                    }
                    let tx = get_number(&op.operands[0]).unwrap_or(0.0);
                    let ty = get_number(&op.operands[1]).unwrap_or(0.0);
                    line_matrix[4] += tx * line_matrix[0] + ty * line_matrix[2];
                    line_matrix[5] += tx * line_matrix[1] + ty * line_matrix[3];
                    text_matrix = line_matrix;
                    if op.operator == "TD" {
                        text_leading = -ty;
                    }
                }
            }
            "Tm" => {
                // Set text matrix
                if op.operands.len() >= 6 {
                    if source_line_has_text {
                        source_line_index = source_line_index.saturating_add(1);
                        source_line_has_text = false;
                    }
                    for (i, operand) in op.operands.iter().take(6).enumerate() {
                        text_matrix[i] =
                            get_number(operand).unwrap_or(if i == 0 || i == 3 { 1.0 } else { 0.0 });
                    }
                    line_matrix = text_matrix;
                }
            }
            "T*" => {
                // Move to start of next line: equivalent to 0 -TL Td
                source_line_index = source_line_index.saturating_add(1);
                source_line_has_text = false;
                let tl = text_leading;
                line_matrix[4] += (-tl) * line_matrix[2]; // Usually 0 for non-rotated text
                line_matrix[5] += (-tl) * line_matrix[3];
                text_matrix = line_matrix;
            }
            "Tj" => {
                // Show text string
                if in_text_block && !op.operands.is_empty() {
                    source_line_has_text = true;
                    // Advance text matrix regardless of visibility
                    let w_ts_opt = font_widths.get(&current_font).and_then(|fi| {
                        get_operand_bytes(&op.operands[0]).map(|raw| {
                            compute_string_width_ts(
                                raw,
                                fi,
                                current_font_size,
                                char_spacing,
                                word_spacing,
                            ) * horizontal_scaling
                        })
                    });
                    // ActualText: suppress glyph extraction, just advance text matrix.
                    // Capture the FIRST glyph's text matrix as the rendering position
                    // for the ActualText item. Td ops between BDC and the first Tj
                    // may have moved the position to the correct line — the BDC-entry
                    // position (actual_text_start_tm) can be on the previous line.
                    if suppress_glyph_extraction {
                        if actual_text_glyph_tm.is_none() {
                            actual_text_glyph_tm = Some(text_matrix);
                            actual_text_glyph_rise = Some(text_rise);
                        }
                        if let Some(w_ts) = w_ts_opt {
                            text_matrix[4] += w_ts * text_matrix[0];
                            text_matrix[5] += w_ts * text_matrix[1];
                        }
                        continue;
                    }
                    // Skip invisible (Tr=3) text but still advance text matrix.
                    // For Mixed/template PDFs, include_invisible=true extracts
                    // the OCR text layer that sits behind scanned images.
                    if text_rendering_mode == 3 && !include_invisible {
                        if let Some(w_ts) = w_ts_opt {
                            text_matrix[4] += w_ts * text_matrix[0];
                            text_matrix[5] += w_ts * text_matrix[1];
                        }
                        continue;
                    }
                    let mut glyph_cursor = 0.0f32;
                    let fidelity_glyphs: Vec<_> = if fidelity {
                        extract_fidelity_glyphs(
                            &op.operands[0],
                            &current_font,
                            font_base_names.get(&current_font).map(String::as_str),
                            font_cmaps,
                            &font_tounicode_refs,
                            &inline_cmaps,
                            &font_encodings,
                            &encoding_cache,
                            &mut cmap_decisions,
                            &font_widths,
                            current_font_size,
                            char_spacing,
                            word_spacing,
                        )
                        .into_iter()
                        .map(|(text, advance)| {
                            let advance = advance * horizontal_scaling;
                            let glyph = FidelityGlyph {
                                text,
                                x0: glyph_cursor,
                                x1: glyph_cursor + advance,
                                bbox: [0.0; 4],
                            };
                            glyph_cursor += advance;
                            glyph
                        })
                        .collect()
                    } else {
                        Vec::new()
                    };
                    let fidelity_text = (!fidelity_glyphs.is_empty()).then(|| {
                        fidelity_glyphs
                            .iter()
                            .map(|glyph| glyph.text.as_str())
                            .collect()
                    });
                    if let Some(text) = extract_text_from_operand(
                        &op.operands[0],
                        &current_font,
                        font_base_names.get(&current_font).map(|s| s.as_str()),
                        font_cmaps,
                        &font_tounicode_refs,
                        &inline_cmaps,
                        &font_encodings,
                        &encoding_cache,
                        &mut cmap_decisions,
                        &font_widths,
                    ) {
                        let combined =
                            multiply_matrices(&rise_adjusted(&text_matrix, text_rise), &ctm);
                        let rendered_size = effective_font_size(current_font_size, &combined)
                            * type3_scales.get(&current_font).copied().unwrap_or(1.0);
                        let (x, y) = (combined[4], combined[5]);
                        if combined[0].abs() >= combined[1].abs() {
                            rotation_votes.horizontal += 1;
                        } else {
                            rotation_votes.rotated += 1;
                        }
                        let width = if let Some(w_ts) = w_ts_opt {
                            text_matrix[4] += w_ts * text_matrix[0];
                            text_matrix[5] += w_ts * text_matrix[1];
                            let axis_x = text_matrix[0] * ctm[0] + text_matrix[1] * ctm[2];
                            let axis_y = text_matrix[0] * ctm[1] + text_matrix[1] * ctm[3];
                            (w_ts * axis_x.hypot(axis_y)).abs()
                        } else {
                            0.0
                        };
                        // Fidelity consumers need whitespace geometry as grouping
                        // evidence. The legal layer removes whitespace-only spans
                        // after it has reconstructed renderer lines.
                        if fidelity || !text.trim().is_empty() {
                            let base_font = font_base_names
                                .get(&current_font)
                                .map(|s| s.as_str())
                                .unwrap_or(&current_font);
                            let (desc_italic, desc_bold) = font_style_flags
                                .get(&current_font)
                                .copied()
                                .unwrap_or((false, false));
                            items.push(TextItem {
                                text: expand_ligatures(&text),
                                x,
                                y,
                                width,
                                height: rendered_size,
                                font: current_font_name.clone(),
                                font_size: rendered_size,
                                page: page_num,
                                is_bold: is_bold_font(base_font) || desc_bold,
                                is_italic: is_italic_font(base_font) || desc_italic,
                                is_underline: false,
                                is_strikeout: false,
                                fidelity: fidelity_info(
                                    font_fidelity.as_ref(),
                                    &current_font,
                                    fidelity_text.unwrap_or_else(|| text.clone()),
                                    text_object_index,
                                    source_line_index,
                                    text_rise,
                                    fidelity_glyphs,
                                    combined,
                                    current_font_size,
                                    type3_scales.get(&current_font).copied().unwrap_or(1.0),
                                    w_ts_opt.unwrap_or(0.0),
                                ),
                                item_type: ItemType::Text,
                                mcid: current_mcid(&marked_content_stack),
                            });
                        }
                    }
                }
            }
            "TJ" => {
                // Show text with positioning — split at column-sized gaps
                if in_text_block && !op.operands.is_empty() {
                    source_line_has_text = true;
                    if let Ok(array) = op.operands[0].as_array() {
                        let font_info = font_widths.get(&current_font);
                        let is_invisible = (text_rendering_mode == 3 && !include_invisible)
                            || suppress_glyph_extraction;
                        // Capture first-glyph position for ActualText
                        if suppress_glyph_extraction && actual_text_glyph_tm.is_none() {
                            actual_text_glyph_tm = Some(text_matrix);
                            actual_text_glyph_rise = Some(text_rise);
                        }

                        // Compute space threshold based on font metrics when available
                        let space_threshold = if let Some(font_info) = font_info {
                            let space_em = font_info.space_width as f32 * font_info.units_scale;
                            let threshold = space_em * 1000.0 * 0.4;
                            threshold.max(80.0)
                        } else {
                            120.0
                        };
                        let column_gap_threshold = space_threshold * 4.0;

                        // Track sub-items for column-gap splitting:
                        // (text, start_width_ts, end_width_ts)
                        let mut sub_items: Vec<(String, String, Vec<FidelityGlyph>, f32, f32)> =
                            Vec::new();
                        let mut current_text = String::new();
                        let mut current_fidelity_text = String::new();
                        let mut current_fidelity_glyphs = Vec::new();
                        let mut pending_space = false;
                        let mut sub_start_width_ts: f32 = 0.0;
                        let mut total_width_ts: f32 = 0.0;
                        for element in array {
                            match element {
                                Object::Integer(n) => {
                                    let n_val = *n as f32;
                                    let displacement =
                                        -n_val / 1000.0 * current_font_size * horizontal_scaling;
                                    if !is_invisible
                                        && n_val < -column_gap_threshold
                                        && !current_text.is_empty()
                                    {
                                        // Column gap: flush current segment
                                        pending_space = false;
                                        sub_items.push((
                                            std::mem::take(&mut current_text),
                                            std::mem::take(&mut current_fidelity_text),
                                            std::mem::take(&mut current_fidelity_glyphs),
                                            sub_start_width_ts,
                                            total_width_ts,
                                        ));
                                        total_width_ts += displacement;
                                        sub_start_width_ts = total_width_ts;
                                    } else {
                                        total_width_ts += displacement;
                                        if !is_invisible
                                            && n_val < -space_threshold
                                            && !current_text.is_empty()
                                            && !current_text.ends_with(' ')
                                        {
                                            pending_space = true;
                                        }
                                    }
                                    continue;
                                }
                                Object::Real(n) => {
                                    let n_val = *n;
                                    let displacement =
                                        -n_val / 1000.0 * current_font_size * horizontal_scaling;
                                    if !is_invisible
                                        && n_val < -column_gap_threshold
                                        && !current_text.is_empty()
                                    {
                                        pending_space = false;
                                        sub_items.push((
                                            std::mem::take(&mut current_text),
                                            std::mem::take(&mut current_fidelity_text),
                                            std::mem::take(&mut current_fidelity_glyphs),
                                            sub_start_width_ts,
                                            total_width_ts,
                                        ));
                                        total_width_ts += displacement;
                                        sub_start_width_ts = total_width_ts;
                                    } else {
                                        total_width_ts += displacement;
                                        if !is_invisible
                                            && n_val < -space_threshold
                                            && !current_text.is_empty()
                                            && !current_text.ends_with(' ')
                                        {
                                            pending_space = true;
                                        }
                                    }
                                    continue;
                                }
                                _ => {}
                            }
                            let element_start_width_ts = total_width_ts;
                            if let Some(fi) = font_info {
                                if let Some(raw_bytes) = get_operand_bytes(element) {
                                    total_width_ts += compute_string_width_ts(
                                        raw_bytes,
                                        fi,
                                        current_font_size,
                                        char_spacing,
                                        word_spacing,
                                    ) * horizontal_scaling;
                                }
                            }
                            if !is_invisible {
                                if fidelity {
                                    let glyphs = extract_fidelity_glyphs(
                                        element,
                                        &current_font,
                                        font_base_names.get(&current_font).map(String::as_str),
                                        font_cmaps,
                                        &font_tounicode_refs,
                                        &inline_cmaps,
                                        &font_encodings,
                                        &encoding_cache,
                                        &mut cmap_decisions,
                                        &font_widths,
                                        current_font_size,
                                        char_spacing,
                                        word_spacing,
                                    );
                                    if !glyphs.is_empty() {
                                        if pending_space
                                            && !current_fidelity_text.ends_with(char::is_whitespace)
                                            && !glyphs[0].0.starts_with(char::is_whitespace)
                                        {
                                            current_fidelity_text.push(' ');
                                            let x = element_start_width_ts - sub_start_width_ts;
                                            current_fidelity_glyphs.push(FidelityGlyph {
                                                text: " ".to_owned(),
                                                x0: x,
                                                x1: x,
                                                bbox: [0.0; 4],
                                            });
                                        }
                                        let mut glyph_cursor = element_start_width_ts;
                                        for (text, advance) in glyphs {
                                            let advance = advance * horizontal_scaling;
                                            current_fidelity_text.push_str(&text);
                                            current_fidelity_glyphs.push(FidelityGlyph {
                                                text,
                                                x0: glyph_cursor - sub_start_width_ts,
                                                x1: glyph_cursor + advance - sub_start_width_ts,
                                                bbox: [0.0; 4],
                                            });
                                            glyph_cursor += advance;
                                        }
                                    } else if let Some(text) = extract_text_from_operand_fidelity(
                                        element,
                                        &current_font,
                                        font_base_names.get(&current_font).map(String::as_str),
                                        font_cmaps,
                                        &font_tounicode_refs,
                                        &inline_cmaps,
                                        &font_encodings,
                                        &encoding_cache,
                                        &mut cmap_decisions,
                                        &font_widths,
                                    ) {
                                        current_fidelity_text.push_str(&text);
                                    }
                                }
                                if let Some(text) = extract_text_from_operand(
                                    element,
                                    &current_font,
                                    font_base_names.get(&current_font).map(|s| s.as_str()),
                                    font_cmaps,
                                    &font_tounicode_refs,
                                    &inline_cmaps,
                                    &font_encodings,
                                    &encoding_cache,
                                    &mut cmap_decisions,
                                    &font_widths,
                                ) {
                                    append_tj_text(&mut current_text, &mut pending_space, &text);
                                }
                            }
                        }
                        // Flush remaining text
                        if !is_invisible
                            && (fidelity || !current_text.trim().is_empty())
                            && !current_text.is_empty()
                        {
                            sub_items.push((
                                current_text,
                                current_fidelity_text,
                                current_fidelity_glyphs,
                                sub_start_width_ts,
                                total_width_ts,
                            ));
                        }
                        // Emit one TextItem per sub-item
                        if !sub_items.is_empty() {
                            let combined = multiply_matrices(&text_matrix, &ctm);
                            if combined[0].abs() >= combined[1].abs() {
                                rotation_votes.horizontal += 1;
                            } else {
                                rotation_votes.rotated += 1;
                            }
                            let rendered_size = effective_font_size(current_font_size, &combined)
                                * type3_scales.get(&current_font).copied().unwrap_or(1.0);
                            let base_font = font_base_names
                                .get(&current_font)
                                .map(|s| s.as_str())
                                .unwrap_or(&current_font);
                            let (desc_italic, desc_bold) = font_style_flags
                                .get(&current_font)
                                .copied()
                                .unwrap_or((false, false));
                            let axis_x = text_matrix[0] * ctm[0] + text_matrix[1] * ctm[2];
                            let axis_y = text_matrix[0] * ctm[1] + text_matrix[1] * ctm[3];
                            let advance_scale = axis_x.hypot(axis_y);
                            for (text, fidelity_text, fidelity_glyphs, start_w, end_w) in &sub_items
                            {
                                let offset_tm = [
                                    text_matrix[0],
                                    text_matrix[1],
                                    text_matrix[2],
                                    text_matrix[3],
                                    text_matrix[4] + start_w * text_matrix[0],
                                    text_matrix[5] + start_w * text_matrix[1],
                                ];
                                let combined =
                                    multiply_matrices(&rise_adjusted(&offset_tm, text_rise), &ctm);
                                let (x, y) = (combined[4], combined[5]);
                                let width = if font_info.is_some() {
                                    ((end_w - start_w) * advance_scale).abs()
                                } else {
                                    0.0
                                };
                                items.push(TextItem {
                                    text: expand_ligatures(text),
                                    x,
                                    y,
                                    width,
                                    height: rendered_size,
                                    font: current_font_name.clone(),
                                    font_size: rendered_size,
                                    page: page_num,
                                    is_bold: is_bold_font(base_font) || desc_bold,
                                    is_italic: is_italic_font(base_font) || desc_italic,
                                    is_underline: false,
                                    is_strikeout: false,
                                    fidelity: fidelity_info(
                                        font_fidelity.as_ref(),
                                        &current_font,
                                        if fidelity_text.is_empty() {
                                            text.clone()
                                        } else {
                                            fidelity_text.clone()
                                        },
                                        text_object_index,
                                        source_line_index,
                                        text_rise,
                                        fidelity_glyphs.clone(),
                                        combined,
                                        current_font_size,
                                        type3_scales.get(&current_font).copied().unwrap_or(1.0),
                                        end_w - start_w,
                                    ),
                                    item_type: ItemType::Text,
                                    mcid: current_mcid(&marked_content_stack),
                                });
                            }
                        }
                        // Always advance text matrix by total width
                        if font_info.is_some() {
                            text_matrix[4] += total_width_ts * text_matrix[0];
                            text_matrix[5] += total_width_ts * text_matrix[1];
                        }
                    }
                }
            }
            "'" => {
                // Move to next line and show text (equivalent to T* then Tj)
                source_line_index = source_line_index.saturating_add(1);
                source_line_has_text = true;
                let tl = text_leading;
                line_matrix[4] += (-tl) * line_matrix[2];
                line_matrix[5] += (-tl) * line_matrix[3];
                text_matrix = line_matrix;
                // Capture first-glyph position for ActualText AFTER the
                // line move — the BDC-entry matrix is on the previous line.
                if suppress_glyph_extraction && actual_text_glyph_tm.is_none() {
                    actual_text_glyph_tm = Some(text_matrix);
                    actual_text_glyph_rise = Some(text_rise);
                }
                // Advance width, as for Tj — without it the item stays
                // zero-width and geometric underline/strikeout detection
                // rejects it (`is_underline_candidate` needs width > 0).
                let shown = quote_operand.or_else(|| op.operands.first());
                let w_ts_opt = font_widths.get(&current_font).and_then(|fi| {
                    shown.and_then(get_operand_bytes).map(|raw| {
                        compute_string_width_ts(
                            raw,
                            fi,
                            current_font_size,
                            char_spacing,
                            word_spacing,
                        ) * horizontal_scaling
                    })
                });
                if !((text_rendering_mode == 3 && !include_invisible)
                    || suppress_glyph_extraction
                    || shown.is_none())
                {
                    let mut glyph_cursor = 0.0f32;
                    let fidelity_glyphs: Vec<_> = if fidelity {
                        extract_fidelity_glyphs(
                            shown.expect("show operand exists"),
                            &current_font,
                            font_base_names.get(&current_font).map(String::as_str),
                            font_cmaps,
                            &font_tounicode_refs,
                            &inline_cmaps,
                            &font_encodings,
                            &encoding_cache,
                            &mut cmap_decisions,
                            &font_widths,
                            current_font_size,
                            char_spacing,
                            word_spacing,
                        )
                        .into_iter()
                        .map(|(text, advance)| {
                            let advance = advance * horizontal_scaling;
                            let glyph = FidelityGlyph {
                                text,
                                x0: glyph_cursor,
                                x1: glyph_cursor + advance,
                                bbox: [0.0; 4],
                            };
                            glyph_cursor += advance;
                            glyph
                        })
                        .collect()
                    } else {
                        Vec::new()
                    };
                    let fidelity_text = fidelity.then(|| {
                        if fidelity_glyphs.is_empty() {
                            extract_text_from_operand_fidelity(
                                shown.expect("show operand exists"),
                                &current_font,
                                font_base_names.get(&current_font).map(String::as_str),
                                font_cmaps,
                                &font_tounicode_refs,
                                &inline_cmaps,
                                &font_encodings,
                                &encoding_cache,
                                &mut cmap_decisions,
                                &font_widths,
                            )
                        } else {
                            Some(
                                fidelity_glyphs
                                    .iter()
                                    .map(|glyph| glyph.text.as_str())
                                    .collect(),
                            )
                        }
                    });
                    if let Some(text) = extract_text_from_operand(
                        shown.expect("show operand exists"),
                        &current_font,
                        font_base_names.get(&current_font).map(|s| s.as_str()),
                        font_cmaps,
                        &font_tounicode_refs,
                        &inline_cmaps,
                        &font_encodings,
                        &encoding_cache,
                        &mut cmap_decisions,
                        &font_widths,
                    ) {
                        if fidelity || !text.trim().is_empty() {
                            let combined =
                                multiply_matrices(&rise_adjusted(&text_matrix, text_rise), &ctm);
                            if combined[0].abs() >= combined[1].abs() {
                                rotation_votes.horizontal += 1;
                            } else {
                                rotation_votes.rotated += 1;
                            }
                            let rendered_size = effective_font_size(current_font_size, &combined)
                                * type3_scales.get(&current_font).copied().unwrap_or(1.0);
                            let (x, y) = (combined[4], combined[5]);
                            let width = w_ts_opt
                                .map(|w_ts| {
                                    let axis_x = text_matrix[0] * ctm[0] + text_matrix[1] * ctm[2];
                                    let axis_y = text_matrix[0] * ctm[1] + text_matrix[1] * ctm[3];
                                    (w_ts * axis_x.hypot(axis_y)).abs()
                                })
                                .unwrap_or(0.0);
                            let base_font = font_base_names
                                .get(&current_font)
                                .map(|s| s.as_str())
                                .unwrap_or(&current_font);
                            let (desc_italic, desc_bold) = font_style_flags
                                .get(&current_font)
                                .copied()
                                .unwrap_or((false, false));
                            items.push(TextItem {
                                text: expand_ligatures(&text),
                                x,
                                y,
                                width,
                                height: rendered_size,
                                font: current_font_name.clone(),
                                font_size: rendered_size,
                                page: page_num,
                                is_bold: is_bold_font(base_font) || desc_bold,
                                is_italic: is_italic_font(base_font) || desc_italic,
                                is_underline: false,
                                is_strikeout: false,
                                fidelity: fidelity_info(
                                    font_fidelity.as_ref(),
                                    &current_font,
                                    fidelity_text.flatten().unwrap_or_else(|| text.clone()),
                                    text_object_index,
                                    source_line_index,
                                    text_rise,
                                    fidelity_glyphs,
                                    combined,
                                    current_font_size,
                                    type3_scales.get(&current_font).copied().unwrap_or(1.0),
                                    w_ts_opt.unwrap_or(0.0),
                                ),
                                item_type: ItemType::Text,
                                mcid: current_mcid(&marked_content_stack),
                            });
                        }
                    }
                }
                // Advance regardless of visibility so later show-text
                // operators on the same line stay positioned (as for Tj).
                if let Some(w_ts) = w_ts_opt {
                    text_matrix[4] += w_ts * text_matrix[0];
                    text_matrix[5] += w_ts * text_matrix[1];
                }
            }
            "Do" => {
                // Forms were expanded before interpretation, so any remaining
                // XObject invocation is an image in the resolved resource scope.
                if !suppress_glyph_extraction {
                    if let Some(name) = op.operands.first().and_then(|value| value.as_name().ok()) {
                        let resource_name = String::from_utf8_lossy(name);
                        if let Some(display_name) = xobjects.get(resource_name.as_ref()) {
                            let (x, y, width, height) = image_bbox_from_ctm(&ctm);
                            items.push(TextItem {
                                text: format!("[Image: {}]", display_name),
                                x,
                                y,
                                width,
                                height,
                                font: String::new(),
                                font_size: 0.0,
                                page: page_num,
                                is_bold: false,
                                is_italic: false,
                                is_underline: false,
                                is_strikeout: false,
                                fidelity: None,
                                item_type: ItemType::Image,
                                mcid: current_mcid(&marked_content_stack),
                            });
                        }
                    }
                }
            }
            "BMC" => {
                // Begin Marked Content (no properties)
                marked_content_stack.push(MarkedContentEntry {
                    actual_text: None,
                    mcid: None,
                });
            }
            "BDC" => {
                // Begin Marked Content with properties — extract ActualText and MCID
                let mut actual_text: Option<String> = None;
                let mut mcid: Option<i64> = None;
                if op.operands.len() >= 2 {
                    let dict = match &op.operands[1] {
                        Object::Dictionary(d) => Some(d.clone()),
                        Object::Reference(id) => doc.get_dictionary(*id).ok().cloned(),
                        _ => None,
                    };
                    if let Some(d) = dict {
                        if let Ok(val) = d.get(b"ActualText") {
                            actual_text = match val {
                                Object::String(bytes, _) => Some(decode_text_string(bytes)),
                                _ => None,
                            };
                        }
                        if let Ok(Object::Integer(id)) = d.get(b"MCID") {
                            mcid = Some(*id);
                        }
                    }
                }
                if actual_text.is_some() {
                    suppress_glyph_extraction = true;
                    actual_text_start_tm = Some(text_matrix);
                    actual_text_start_rise = text_rise;
                    actual_text_glyph_tm = None; // reset — will be captured at first Tj/TJ
                    actual_text_glyph_rise = None;
                }
                marked_content_stack.push(MarkedContentEntry { actual_text, mcid });
            }
            "EMC" => {
                // End Marked Content — emit ActualText item with correct width
                if let Some(entry) = marked_content_stack.pop() {
                    if let Some(at) = entry.actual_text {
                        // Use the first-glyph position (if available) instead of the
                        // BDC-entry position. Td operators between BDC and the first
                        // Tj may have moved the text position to the correct line —
                        // the BDC-entry position can be on the previous line.
                        let glyph_tm = actual_text_glyph_tm.take();
                        let glyph_rise = actual_text_glyph_rise.take();
                        let entry_tm = actual_text_start_tm.take();
                        if let Some(start_tm) = glyph_tm.or(entry_tm) {
                            let rise = glyph_rise.unwrap_or(actual_text_start_rise);
                            let combined = multiply_matrices(&rise_adjusted(&start_tm, rise), &ctm);
                            if combined[0].abs() >= combined[1].abs() {
                                rotation_votes.horizontal += 1;
                            } else {
                                rotation_votes.rotated += 1;
                            }
                            let rendered_size = effective_font_size(current_font_size, &combined)
                                * type3_scales.get(&current_font).copied().unwrap_or(1.0);
                            let (x, y) = (combined[4], combined[5]);
                            // The translation slots already include the text
                            // matrix's scale. Project their delta back onto the
                            // text axis before fidelity_info applies the CTM.
                            // Using the raw translation delta multiplied scaled
                            // ActualText advances twice (for example, a 10x Tm
                            // turned a 72pt heading into a 720pt run).
                            let delta =
                                [text_matrix[4] - start_tm[4], text_matrix[5] - start_tm[5]];
                            let axis = [start_tm[0], start_tm[1]];
                            let axis_squared = axis[0] * axis[0] + axis[1] * axis[1];
                            let advance_ts = if axis_squared > f32::EPSILON {
                                (delta[0] * axis[0] + delta[1] * axis[1]) / axis_squared
                            } else {
                                0.0
                            };
                            let axis_x = start_tm[0] * ctm[0] + start_tm[1] * ctm[2];
                            let axis_y = start_tm[0] * ctm[1] + start_tm[1] * ctm[3];
                            let width = (advance_ts * axis_x.hypot(axis_y)).abs();
                            if fidelity || !at.trim().is_empty() {
                                let base_font = font_base_names
                                    .get(&current_font)
                                    .map(|s| s.as_str())
                                    .unwrap_or(&current_font);
                                let (desc_italic, desc_bold) = font_style_flags
                                    .get(&current_font)
                                    .copied()
                                    .unwrap_or((false, false));
                                items.push(TextItem {
                                    text: expand_ligatures(&at),
                                    x,
                                    y,
                                    width,
                                    height: rendered_size,
                                    font: current_font_name.clone(),
                                    font_size: rendered_size,
                                    page: page_num,
                                    is_bold: is_bold_font(base_font) || desc_bold,
                                    is_italic: is_italic_font(base_font) || desc_italic,
                                    is_underline: false,
                                    is_strikeout: false,
                                    fidelity: fidelity_info(
                                        font_fidelity.as_ref(),
                                        &current_font,
                                        at.clone(),
                                        text_object_index,
                                        source_line_index,
                                        rise,
                                        Vec::new(),
                                        combined,
                                        current_font_size,
                                        type3_scales.get(&current_font).copied().unwrap_or(1.0),
                                        advance_ts,
                                    ),
                                    item_type: ItemType::Text,
                                    mcid: entry
                                        .mcid
                                        .or_else(|| current_mcid(&marked_content_stack)),
                                });
                            }
                        }
                        suppress_glyph_extraction =
                            marked_content_stack.iter().any(|e| e.actual_text.is_some());
                    }
                }
            }
            "re" => {
                // Rectangle operator: collect for table-grid detection
                if op.operands.len() >= 4 {
                    let rx = get_number(&op.operands[0]).unwrap_or(0.0);
                    let ry = get_number(&op.operands[1]).unwrap_or(0.0);
                    let rw = get_number(&op.operands[2]).unwrap_or(0.0);
                    let rh = get_number(&op.operands[3]).unwrap_or(0.0);
                    // Transform origin to device space
                    let x_dev = rx * ctm[0] + ry * ctm[2] + ctm[4];
                    let y_dev = rx * ctm[1] + ry * ctm[3] + ctm[5];
                    let w_dev = rw * ctm[0];
                    let h_dev = rh * ctm[3];
                    let rect = PdfRect {
                        x: x_dev,
                        y: y_dev,
                        width: w_dev,
                        height: h_dev,
                        page: page_num,
                    };
                    // Underline detection must only see rects that are
                    // actually painted — a `re` used purely as a clip path
                    // (`re W n`) or discarded (`re n`) draws nothing. Hold
                    // the rect as pending until a paint operator confirms it.
                    pending_re_rects.push(rect.clone());
                    rects.push(rect);
                }
            }
            // ── Path construction operators ──────────────────────
            "m" => {
                // moveto: start a new subpath
                if op.operands.len() >= 2 {
                    let px = get_number(&op.operands[0]).unwrap_or(0.0);
                    let py = get_number(&op.operands[1]).unwrap_or(0.0);
                    path_subpath_start = Some((px, py));
                    path_current = Some((px, py));
                }
            }
            "l" => {
                // lineto: add segment from current point
                if op.operands.len() >= 2 {
                    if let Some((cx, cy)) = path_current {
                        let px = get_number(&op.operands[0]).unwrap_or(0.0);
                        let py = get_number(&op.operands[1]).unwrap_or(0.0);
                        pending_lines.push((cx, cy, px, py));
                        path_current = Some((px, py));
                    }
                }
            }
            "h" => {
                // closepath: segment back to subpath start
                if let (Some((cx, cy)), Some((sx, sy))) = (path_current, path_subpath_start) {
                    if (cx - sx).abs() > 0.01 || (cy - sy).abs() > 0.01 {
                        pending_lines.push((cx, cy, sx, sy));
                    }
                    path_current = path_subpath_start;
                }
                // Save completed subpath for f/f* rect extraction and clear pending_lines.
                // The W/W* handler reads from pending_subpaths (last entry) instead.
                if !pending_lines.is_empty() {
                    pending_subpaths.push(std::mem::take(&mut pending_lines));
                }
            }
            // ── Path painting operators ──────────────────────────
            "S" | "s" => {
                // stroke / close-and-stroke: emit pending lines
                if op.operator == "s" {
                    // close first
                    if let (Some((cx, cy)), Some((sx, sy))) = (path_current, path_subpath_start) {
                        if (cx - sx).abs() > 0.01 || (cy - sy).abs() > 0.01 {
                            pending_lines.push((cx, cy, sx, sy));
                        }
                    }
                }
                for (x1, y1, x2, y2) in pending_lines.drain(..) {
                    let (x1d, y1d) = transform_path_point(x1, y1, &ctm);
                    let (x2d, y2d) = transform_path_point(x2, y2, &ctm);
                    lines.push(PdfLine {
                        x1: x1d,
                        y1: y1d,
                        x2: x2d,
                        y2: y2d,
                        page: page_num,
                    });
                    underline_lines.push(UnderlineLine {
                        x1: x1d,
                        y1: y1d,
                        x2: x2d,
                        y2: y2d,
                        stroke_width: transformed_stroke_width(line_width, &ctm, x1, y1, x2, y2),
                        page: page_num,
                    });
                }
                painted_rects.append(&mut pending_re_rects);
                pending_subpaths.clear();
                path_subpath_start = None;
                path_current = None;
            }
            "B" | "B*" | "b" | "b*" => {
                // fill+stroke: emit lines AND clear state
                if op.operator == "b" || op.operator == "b*" {
                    // close first
                    if let (Some((cx, cy)), Some((sx, sy))) = (path_current, path_subpath_start) {
                        if (cx - sx).abs() > 0.01 || (cy - sy).abs() > 0.01 {
                            pending_lines.push((cx, cy, sx, sy));
                        }
                    }
                }
                for (x1, y1, x2, y2) in pending_lines.drain(..) {
                    let (x1d, y1d) = transform_path_point(x1, y1, &ctm);
                    let (x2d, y2d) = transform_path_point(x2, y2, &ctm);
                    lines.push(PdfLine {
                        x1: x1d,
                        y1: y1d,
                        x2: x2d,
                        y2: y2d,
                        page: page_num,
                    });
                    underline_lines.push(UnderlineLine {
                        x1: x1d,
                        y1: y1d,
                        x2: x2d,
                        y2: y2d,
                        stroke_width: transformed_stroke_width(line_width, &ctm, x1, y1, x2, y2),
                        page: page_num,
                    });
                }
                painted_rects.append(&mut pending_re_rects);
                pending_subpaths.clear();
                path_subpath_start = None;
                path_current = None;
            }
            "f" | "F" | "f*" => {
                // fill-only: extract axis-aligned rects from completed subpaths
                // Also check any un-closed segments still in pending_lines
                if !pending_lines.is_empty() {
                    pending_subpaths.push(std::mem::take(&mut pending_lines));
                }
                for subpath in pending_subpaths.drain(..) {
                    // Synthesize closing segment if only 3 segments
                    let mut segs = subpath;
                    if segs.len() == 3 {
                        let (x0, y0, _, _) = segs[0];
                        let (_, _, ex, ey) = segs[2];
                        if (ex - x0).abs() > 0.01 || (ey - y0).abs() > 0.01 {
                            segs.push((ex, ey, x0, y0));
                        }
                    }
                    if segs.len() == 4 {
                        let mut xs = Vec::with_capacity(8);
                        let mut ys = Vec::with_capacity(8);
                        for &(x1, y1, x2, y2) in &segs {
                            xs.push(x1);
                            xs.push(x2);
                            ys.push(y1);
                            ys.push(y2);
                        }
                        let min_x = xs.iter().copied().fold(f32::INFINITY, f32::min);
                        let max_x = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let min_y = ys.iter().copied().fold(f32::INFINITY, f32::min);
                        let max_y = ys.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let w = max_x - min_x;
                        let h = max_y - min_y;
                        let eps: f32 = 0.5;
                        let axis_aligned = xs
                            .iter()
                            .all(|&x| (x - min_x).abs() < eps || (x - max_x).abs() < eps)
                            && ys
                                .iter()
                                .all(|&y| (y - min_y).abs() < eps || (y - max_y).abs() < eps);
                        if axis_aligned && w > 1.0 && h > 1.0 {
                            let x_dev = min_x * ctm[0] + min_y * ctm[2] + ctm[4];
                            let y_dev = min_x * ctm[1] + min_y * ctm[3] + ctm[5];
                            let w_dev = w * ctm[0];
                            let h_dev = h * ctm[3];
                            fill_rects.push(PdfRect {
                                x: x_dev,
                                y: y_dev,
                                width: w_dev,
                                height: h_dev,
                                page: page_num,
                            });
                        }
                    }
                }
                painted_rects.append(&mut pending_re_rects);
                pending_lines.clear();
                path_subpath_start = None;
                path_current = None;
            }
            "W" | "W*" => {
                // Clip operator: check if pending path forms an axis-aligned rectangle.
                // Many PDFs define table cells as clipping paths instead of stroked rects.
                // After `h` closes a subpath, pending_lines is cleared and the subpath
                // is saved to pending_subpaths. Read from the last subpath entry.
                let mut segs: Vec<(f32, f32, f32, f32)> = if pending_lines.is_empty() {
                    pending_subpaths.last().cloned().unwrap_or_default()
                } else {
                    pending_lines.clone()
                };
                // If only 3 segments, synthesize closing segment back to subpath start
                if segs.len() == 3 {
                    if let Some((sx, sy)) = path_subpath_start {
                        let (_, _, ex, ey) = segs[2];
                        if (ex - sx).abs() > 0.01 || (ey - sy).abs() > 0.01 {
                            segs.push((ex, ey, sx, sy));
                        }
                    }
                }
                if segs.len() == 4 {
                    // Collect all endpoints and compute bounding box
                    let mut xs = Vec::with_capacity(8);
                    let mut ys = Vec::with_capacity(8);
                    for &(x1, y1, x2, y2) in &segs {
                        xs.push(x1);
                        xs.push(x2);
                        ys.push(y1);
                        ys.push(y2);
                    }
                    let min_x = xs.iter().copied().fold(f32::INFINITY, f32::min);
                    let max_x = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let min_y = ys.iter().copied().fold(f32::INFINITY, f32::min);
                    let max_y = ys.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let w = max_x - min_x;
                    let h = max_y - min_y;
                    // Verify all points lie on bounding box edges (axis-aligned rectangle)
                    let eps: f32 = 0.5;
                    let axis_aligned = xs
                        .iter()
                        .all(|&x| (x - min_x).abs() < eps || (x - max_x).abs() < eps)
                        && ys
                            .iter()
                            .all(|&y| (y - min_y).abs() < eps || (y - max_y).abs() < eps);
                    if axis_aligned && w > 1.0 && h > 1.0 {
                        // Transform to device space using CTM (same as `re` handler)
                        let x_dev = min_x * ctm[0] + min_y * ctm[2] + ctm[4];
                        let y_dev = min_x * ctm[1] + min_y * ctm[3] + ctm[5];
                        let w_dev = w * ctm[0];
                        let h_dev = h * ctm[3];
                        clip_rects.push(PdfRect {
                            x: x_dev,
                            y: y_dev,
                            width: w_dev,
                            height: h_dev,
                            page: page_num,
                        });
                    }
                }
                // Do NOT clear pending_lines — the following `n` does that
            }
            "n" => {
                // end path (no-op): discard — including any `re` rects that
                // were only ever part of a clip path (`re W n`), which draw
                // no ink and must not feed underline detection.
                pending_re_rects.clear();
                pending_lines.clear();
                pending_subpaths.clear();
                path_subpath_start = None;
                path_current = None;
            }
            _ => {}
        }
    }

    // Underline detection reads only painted ink: `re` rects confirmed by
    // a paint operator plus filled-subpath rects — never clip-only rects,
    // which draw nothing.
    let mut underline_rects = painted_rects;
    underline_rects.extend(fill_rects.iter().cloned());

    // Only use clip/fill rects when no `re` rects exist on this page.
    // Clip rects take priority over fill rects, but first we deduplicate
    // them: some PDFs wrap every text block in a full-page W* clip path,
    // producing thousands of identical rects that yield a degenerate grid.
    // After dedup, if too few unique clip rects remain we fall through to
    // fill rects (explicitly drawn visible rectangles).
    //
    // When fill rects substantially outnumber clip rects, the clips are
    // typically section-level wrappers and the fills are the actual table
    // cell backgrounds (e.g. shaded-header tables drawn with `m`/`l`/`h`/`f*`
    // sequences). In that case, prefer fills.
    if rects.is_empty() {
        dedup_rects(&mut clip_rects);
        let prefer_fills = !fill_rects.is_empty() && fill_rects.len() >= clip_rects.len() * 3;
        if prefer_fills {
            rects = fill_rects;
        } else if clip_rects.len() >= 4 {
            rects = clip_rects;
        } else if !fill_rects.is_empty() {
            rects = fill_rects;
        } else if !clip_rects.is_empty() {
            rects = clip_rects;
        }
    }

    // Detect dominant text rotation and transform coordinates if needed.
    // Some PDFs embed landscape content in portrait pages using a rotated text
    // matrix (e.g. [0, b, -b, 0, tx, ty] for 90° CCW).  The layout engine
    // assumes x=horizontal, y=vertical — so we swap coordinates to match.
    let (mut items, rects, lines, coords_rotated) =
        correct_rotated_page(items, rects, lines, &rotation_votes);
    if coords_rotated {
        rotate_underline_graphics(&mut underline_rects, &mut underline_lines);
    }
    super::underline::mark_underlined_items(
        &mut items,
        &underline_rects,
        &underline_lines,
        page_num,
    );

    let items = if merge_text {
        super::merge_text_items(items)
    } else {
        items
    };
    let items = if merge_scripts {
        super::merge_subscript_items(items)
    } else {
        items
    };
    Ok(((items, rects, lines), has_gid_fonts, coords_rotated))
}

/// Counts of text operators with horizontal vs rotated combined matrices.
struct RotationVotes {
    horizontal: u32,
    rotated: u32,
}

/// Detect if most text items on a page are rotated 90° or 270°, and if so,
/// swap x↔y coordinates (plus widths/heights) so the layout engine sees
/// them as horizontal text on a landscape page.
fn correct_rotated_page(
    mut items: Vec<TextItem>,
    mut rects: Vec<PdfRect>,
    mut lines: Vec<PdfLine>,
    votes: &RotationVotes,
) -> (Vec<TextItem>, Vec<PdfRect>, Vec<PdfLine>, bool) {
    if items.len() < 2 {
        return (items, rects, lines, false);
    }

    // Use the combined-matrix direction votes collected during extraction.
    // For normal text, combined[0] (the x-component of the text x-axis) is
    // large; for 90° rotated text, combined[1] dominates instead.
    let total_votes = votes.horizontal + votes.rotated;
    if total_votes == 0 || votes.rotated * 3 < total_votes * 2 {
        // Less than ~67% of text operators are rotated → not a rotated page
        return (items, rects, lines, false);
    }

    log::debug!(
        "detected rotated page text: {}/{} text ops are rotated — swapping coordinates",
        votes.rotated,
        total_votes
    );

    // For 90° CCW rotation (the common case: Tm = [0, b, -b, 0, tx, ty]):
    //   device x increases = visual "down"   → negate when mapping to y
    //   device y increases = visual "right"   → use directly as x
    // The layout engine sorts by y descending (highest = top of page), so
    // we negate old_x so that visual-top (low device x) gets high new_y.
    for item in &mut items {
        let old_x = item.x;
        item.x = item.y;
        item.y = -old_x;
        if let Some(fidelity) = item.fidelity.as_mut() {
            let old_baseline = fidelity.baseline;
            fidelity.baseline = [old_baseline[1], -old_baseline[0]];
            fidelity.advance = [fidelity.advance[1], -fidelity.advance[0]];
            fidelity.em = [fidelity.em[1], -fidelity.em[0]];
            for glyph in &mut fidelity.glyphs {
                let bbox = glyph.bbox;
                glyph.bbox = [bbox[1], -bbox[2], bbox[3], -bbox[0]];
            }
        }
    }

    // Transform rectangles
    for rect in &mut rects {
        let new_x = rect.y;
        let new_y = -(rect.x + rect.width.abs());
        rect.x = new_x;
        rect.y = new_y;
        std::mem::swap(&mut rect.width, &mut rect.height);
    }

    // Transform lines
    for line in &mut lines {
        let new_x1 = line.y1;
        let new_y1 = -line.x1;
        let new_x2 = line.y2;
        let new_y2 = -line.x2;
        line.x1 = new_x1;
        line.y1 = new_y1;
        line.x2 = new_x2;
        line.y2 = new_y2;
    }

    (items, rects, lines, true)
}

fn rotate_underline_graphics(rects: &mut [PdfRect], lines: &mut [UnderlineLine]) {
    for rect in rects {
        let new_x = rect.y;
        let new_y = -(rect.x + rect.width.abs());
        rect.x = new_x;
        rect.y = new_y;
        std::mem::swap(&mut rect.width, &mut rect.height);
    }

    for line in lines {
        let new_x1 = line.y1;
        let new_y1 = -line.x1;
        let new_x2 = line.y2;
        let new_y2 = -line.x2;
        line.x1 = new_x1;
        line.y1 = new_y1;
        line.x2 = new_x2;
        line.y2 = new_y2;
    }
}

/// Remove near-duplicate rects (same coordinates within 0.5 pt tolerance).
/// Some PDFs emit a full-page clip path for every text block, producing
/// thousands of identical rects. After dedup these collapse to one rect,
/// which is too few for table detection and gets naturally skipped.
fn dedup_rects(rects: &mut Vec<PdfRect>) {
    if rects.len() <= 1 {
        return;
    }
    // Round to 0.5-pt grid for tolerance, then sort and dedup.
    rects.sort_by(|a, b| {
        let ak = (
            a.page,
            (a.x * 2.0) as i32,
            (a.y * 2.0) as i32,
            (a.width * 2.0) as i32,
            (a.height * 2.0) as i32,
        );
        let bk = (
            b.page,
            (b.x * 2.0) as i32,
            (b.y * 2.0) as i32,
            (b.width * 2.0) as i32,
            (b.height * 2.0) as i32,
        );
        ak.cmp(&bk)
    });
    rects.dedup_by(|a, b| {
        a.page == b.page
            && ((a.x - b.x).abs() < 0.5)
            && ((a.y - b.y).abs() < 0.5)
            && ((a.width - b.width).abs() < 0.5)
            && ((a.height - b.height).abs() < 0.5)
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: f32, y: f32, w: f32, h: f32, page: u32) -> PdfRect {
        PdfRect {
            x,
            y,
            width: w,
            height: h,
            page,
        }
    }

    fn simple_doc_with_content(content: &[u8]) -> (lopdf::Document, lopdf::ObjectId) {
        use lopdf::{dictionary, Object, Stream};

        let mut doc = lopdf::Document::new();
        let widths: Vec<Object> = (0..=255).map(|_| 600.into()).collect();
        let font_id = doc.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "Helvetica",
            "FirstChar" => 0,
            "LastChar" => 255,
            "Widths" => Object::Array(widths),
        });
        let content_id = doc.add_object(Object::Stream(Stream::new(
            dictionary! {},
            content.to_vec(),
        )));
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Contents" => Object::Reference(content_id),
            "Resources" => dictionary! {
                "Font" => dictionary! {
                    "F1" => Object::Reference(font_id),
                },
            },
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        });
        let pages_id = doc.add_object(dictionary! {
            "Type" => "Pages",
            "Count" => Object::Integer(1),
            "Kids" => vec![Object::Reference(page_id)],
        });
        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => Object::Reference(pages_id),
        });
        doc.trailer.set("Root", Object::Reference(catalog_id));

        (doc, page_id)
    }

    fn simple_doc_with_form(
        form_content: &[u8],
        page_tail: &[u8],
        form_width: Option<i64>,
        form_matrix: Option<[i64; 6]>,
        properties: Option<lopdf::Dictionary>,
    ) -> (lopdf::Document, lopdf::ObjectId) {
        use lopdf::{dictionary, Dictionary, Object, Stream};

        fn font(doc: &mut lopdf::Document, base: &str, width: i64) -> lopdf::ObjectId {
            let widths: Vec<Object> = (0..=255).map(|_| width.into()).collect();
            doc.add_object(dictionary! {
                "Type" => "Font",
                "Subtype" => "Type1",
                "BaseFont" => base,
                "FirstChar" => 0,
                "LastChar" => 255,
                "Widths" => Object::Array(widths),
            })
        }

        let mut doc = lopdf::Document::new();
        let page_font = font(&mut doc, "Helvetica", 600);
        let mut form_resources = Dictionary::new();
        if let Some(width) = form_width {
            let form_font = font(&mut doc, "Courier", width);
            form_resources.set("Font", dictionary! { "F1" => Object::Reference(form_font) });
        }
        if let Some(properties) = properties {
            form_resources.set("Properties", properties);
        }

        let mut form_dictionary = dictionary! {
            "Type" => "XObject",
            "Subtype" => "Form",
            "BBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        };
        if !form_resources.is_empty() {
            form_dictionary.set("Resources", form_resources);
        }
        if let Some(matrix) = form_matrix {
            form_dictionary.set(
                "Matrix",
                Object::Array(matrix.into_iter().map(Object::Integer).collect()),
            );
        }
        let form = doc.add_object(Stream::new(form_dictionary, form_content.to_vec()));

        let mut page_content = b"/Fm Do\n".to_vec();
        page_content.extend_from_slice(page_tail);
        let content = doc.add_object(Stream::new(dictionary! {}, page_content));
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Contents" => Object::Reference(content),
            "Resources" => dictionary! {
                "Font" => dictionary! { "F1" => Object::Reference(page_font) },
                "XObject" => dictionary! { "Fm" => Object::Reference(form) },
            },
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        });
        let pages = doc.add_object(dictionary! {
            "Type" => "Pages",
            "Count" => Object::Integer(1),
            "Kids" => vec![Object::Reference(page_id)],
        });
        let catalog = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => Object::Reference(pages),
        });
        doc.trailer.set("Root", Object::Reference(catalog));
        (doc, page_id)
    }

    fn extract_doc_fidelity_items(
        doc: &lopdf::Document,
        page_id: lopdf::ObjectId,
    ) -> Vec<TextItem> {
        let font_cmaps = FontCMaps::from_doc(doc);
        let ((items, _, _), _, _) = extract_page_text_items_fidelity(
            doc,
            page_id,
            1,
            &font_cmaps,
            false,
            &mut FontStyleCache::new(),
        )
        .unwrap();
        items
    }

    fn extract_simple_items(content: &[u8]) -> Vec<TextItem> {
        use crate::tounicode::FontCMaps;

        let (doc, page_id) = simple_doc_with_content(content);
        let font_cmaps = FontCMaps::from_doc(&doc);
        let ((items, _, _), _, _) = extract_page_text_items(
            &doc,
            page_id,
            1,
            &font_cmaps,
            false,
            &mut FontStyleCache::new(),
        )
        .unwrap();
        items
    }

    fn extract_simple_raw_items(content: &[u8]) -> Vec<TextItem> {
        use crate::tounicode::FontCMaps;

        let (doc, page_id) = simple_doc_with_content(content);
        let font_cmaps = FontCMaps::from_doc(&doc);
        let ((items, _, _), _, _) = extract_page_text_items_raw(
            &doc,
            page_id,
            1,
            &font_cmaps,
            false,
            &mut FontStyleCache::new(),
        )
        .unwrap();
        items
    }

    fn extract_simple_fidelity_items(content: &[u8]) -> Vec<TextItem> {
        use crate::tounicode::FontCMaps;

        let (doc, page_id) = simple_doc_with_content(content);
        let font_cmaps = FontCMaps::from_doc(&doc);
        let ((items, _, _), _, _) = extract_page_text_items_fidelity(
            &doc,
            page_id,
            1,
            &font_cmaps,
            false,
            &mut FontStyleCache::new(),
        )
        .unwrap();
        items
    }

    #[test]
    fn zero_leading_t_star_does_not_invent_a_line_advance() {
        let items =
            extract_simple_raw_items(b"BT /F1 12 Tf 1 0 0 1 100 500 Tm (A) Tj T* (B) Tj ET");
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].y, items[1].y);
    }

    #[test]
    fn horizontal_scaling_applies_to_text_and_glyph_advances() {
        let items =
            extract_simple_fidelity_items(b"BT /F1 12 Tf 50 Tz 1 0 0 1 100 500 Tm (AB) Tj ET");
        assert_eq!(items.len(), 1);
        assert!((items[0].width - 7.2).abs() < 0.01);
        let glyphs = &items[0].fidelity.as_ref().unwrap().glyphs;
        assert_eq!(glyphs.len(), 2);
        assert!((glyphs[0].x1 - 3.6).abs() < 0.01);
        assert!((glyphs[1].x1 - 7.2).abs() < 0.01);
    }

    #[test]
    fn fidelity_keeps_whitespace_geometry_for_line_grouping() {
        let content = b"BT /F1 12 Tf 1 0 0 1 100 500 Tm ( ) Tj ET";
        assert!(extract_simple_items(content).is_empty());

        let items = extract_simple_fidelity_items(content);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].text, " ");
        assert!(items[0].width > 0.0);
        assert_eq!(items[0].fidelity.as_ref().unwrap().glyphs.len(), 1);
    }

    #[test]
    fn fidelity_keeps_whitespace_actual_text() {
        let content = b"BT /F1 12 Tf 1 0 0 1 100 500 Tm \
/Span <</ActualText <feff00a0>>> BDC ( ) Tj EMC ET";
        assert!(extract_simple_items(content).is_empty());

        let items = extract_simple_fidelity_items(content);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].text, "\u{00a0}");
        assert!(items[0].width > 0.0);
    }

    #[test]
    fn scaled_actual_text_advance_is_not_multiplied_twice() {
        let items = extract_simple_fidelity_items(
            b"BT /F1 1 Tf 10 0 0 10 100 500 Tm \
/Span <</ActualText (replacement) >> BDC (AB) Tj EMC ET",
        );
        let item = &items[0];
        let fidelity = item.fidelity.as_ref().unwrap();

        assert!((item.width - 12.0).abs() < 0.01);
        assert!((fidelity.advance[0] - 12.0).abs() < 0.01);
    }

    #[test]
    fn rotated_fidelity_uses_source_advances_and_glyph_bounds() {
        let items = extract_simple_fidelity_items(
            b"BT /F1 12 Tf 0 1 -1 0 200 100 Tm (AB) Tj ET \
              BT /F1 12 Tf 0 1 -1 0 240 100 Tm (CD) Tj ET",
        );
        let first = items.iter().find(|item| item.text == "AB").unwrap();
        let fidelity = first.fidelity.as_ref().unwrap();

        assert!((first.width - 14.4).abs() < 0.01);
        assert!((fidelity.advance[0] - 14.4).abs() < 0.01);
        assert!(fidelity.advance[1].abs() < 0.01);
        assert_eq!(fidelity.glyphs.len(), 2);
        assert!((fidelity.glyphs[0].bbox[2] - fidelity.glyphs[0].bbox[0] - 7.2).abs() < 0.01);
        assert!(fidelity.glyphs[1].bbox[0] >= fidelity.glyphs[0].bbox[2] - 0.01);
    }

    #[test]
    fn form_content_uses_the_full_text_state_machine() {
        let (doc, page_id) = simple_doc_with_form(
            b"BT /F1 12 Tf 50 Tz 14 TL 1 0 0 1 100 500 Tm (A) Tj 3 Ts 1 2 (B) \" ET",
            b"",
            None,
            Some([1, 0, 0, 1, 20, 30]),
            None,
        );
        let items = extract_doc_fidelity_items(&doc, page_id);

        assert_eq!(items.len(), 2);
        assert!((items[0].x - 120.0).abs() < 0.01);
        assert!((items[0].y - 530.0).abs() < 0.01);
        assert!((items[0].width - 3.6).abs() < 0.01);
        assert!((items[1].x - 120.0).abs() < 0.01);
        assert!((items[1].y - 519.0).abs() < 0.01);
        assert!((items[1].width - 4.6).abs() < 0.01);
        assert!((items[1].fidelity.as_ref().unwrap().glyphs[0].x1 - 4.6).abs() < 0.01);
    }

    #[test]
    fn form_resources_do_not_collide_with_page_resources() {
        let (doc, page_id) = simple_doc_with_form(
            b"BT /F1 12 Tf 1 0 0 1 100 500 Tm (FORM) Tj ET",
            b"BT /F1 12 Tf 1 0 0 1 100 450 Tm (PAGE) Tj ET",
            Some(500),
            None,
            None,
        );
        let items = extract_doc_fidelity_items(&doc, page_id);
        let form = items.iter().find(|item| item.text == "FORM").unwrap();
        let page = items.iter().find(|item| item.text == "PAGE").unwrap();

        assert!((form.width - 24.0).abs() < 0.01);
        assert!((page.width - 28.8).abs() < 0.01);
        assert_eq!(form.font, "F1");
        assert_eq!(page.font, "F1");
        assert_eq!(form.fidelity.as_ref().unwrap().font, "Courier");
        assert_eq!(page.fidelity.as_ref().unwrap().font, "Helvetica");
    }

    #[test]
    fn form_named_actual_text_is_resolved_in_its_resource_scope() {
        use lopdf::{dictionary, Object};

        let properties = dictionary! {
            "P1" => dictionary! {
                "ActualText" => Object::string_literal("replacement"),
                "MCID" => Object::Integer(7),
            },
        };
        let (doc, page_id) = simple_doc_with_form(
            b"BT /F1 12 Tf 1 0 0 1 100 500 Tm /Span /P1 BDC (raw) Tj EMC ET",
            b"",
            Some(600),
            None,
            Some(properties),
        );
        let items = extract_doc_fidelity_items(&doc, page_id);

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].text, "replacement");
        assert_eq!(items[0].mcid, Some(7));
    }

    #[test]
    fn form_graphics_state_is_restored_before_page_text() {
        let (doc, page_id) = simple_doc_with_form(
            b"3 Tr BT /F1 12 Tf 1 0 0 1 100 500 Tm (HIDDEN) Tj ET",
            b"BT /F1 12 Tf 1 0 0 1 100 450 Tm (PAGE) Tj ET",
            None,
            None,
            None,
        );
        let items = extract_doc_fidelity_items(&doc, page_id);

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].text, "PAGE");
    }

    #[test]
    fn double_quote_sets_spacing_moves_line_and_shows_text() {
        let items = extract_simple_raw_items(
            b"BT /F1 12 Tf 12 TL 1 0 0 1 100 512 Tm 20 2 (AB) \" (CD) Tj ET",
        );
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].text, "AB");
        assert!((items[0].y - 500.0).abs() < 0.01);
        assert!((items[0].width - 18.4).abs() < 0.01);
        assert_eq!(items[1].text, "CD");
        assert!((items[1].x - 118.4).abs() < 0.01);
    }

    #[test]
    fn text_rendering_mode_persists_between_text_objects() {
        let items = extract_simple_raw_items(
            b"BT /F1 12 Tf 3 Tr 1 0 0 1 100 500 Tm (A) Tj ET BT 1 0 0 1 100 480 Tm (B) Tj ET",
        );
        assert!(items.is_empty());
    }

    #[test]
    fn tj_inferred_space_does_not_duplicate_a_literal_space() {
        let items = extract_simple_raw_items(
            b"BT /F1 12 Tf 1 0 0 1 100 500 Tm [(General) -300 ( statutory)] TJ ET",
        );

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].text, "General statutory");
    }

    #[test]
    fn tj_adjacent_literals_do_not_duplicate_their_boundary_space() {
        let items = extract_simple_raw_items(
            b"BT /F1 12 Tf 1 0 0 1 100 500 Tm [(General ) -300 ( statutory)] TJ ET",
        );

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].text, "General statutory");
    }

    #[test]
    fn test_dedup_rects_identical() {
        let mut rects = vec![rect(0.0, 0.0, 612.0, 792.0, 1); 3759];
        dedup_rects(&mut rects);
        assert_eq!(rects.len(), 1);
    }

    #[test]
    fn test_dedup_rects_within_tolerance() {
        let mut rects = vec![
            rect(10.0, 20.0, 100.0, 50.0, 1),
            rect(10.2, 20.1, 100.3, 50.4, 1),
        ];
        dedup_rects(&mut rects);
        assert_eq!(rects.len(), 1);
    }

    #[test]
    fn test_dedup_rects_distinct_kept() {
        let mut rects = vec![
            rect(10.0, 20.0, 100.0, 50.0, 1),
            rect(120.0, 20.0, 100.0, 50.0, 1),
            rect(10.0, 80.0, 100.0, 50.0, 1),
        ];
        dedup_rects(&mut rects);
        assert_eq!(rects.len(), 3);
    }

    #[test]
    fn test_dedup_rects_different_pages_kept() {
        let mut rects = vec![
            rect(0.0, 0.0, 612.0, 792.0, 1),
            rect(0.0, 0.0, 612.0, 792.0, 2),
        ];
        dedup_rects(&mut rects);
        assert_eq!(rects.len(), 2);
    }

    #[test]
    fn test_dedup_rects_empty_and_single() {
        let mut empty: Vec<PdfRect> = vec![];
        dedup_rects(&mut empty);
        assert!(empty.is_empty());

        let mut single = vec![rect(1.0, 2.0, 3.0, 4.0, 1)];
        dedup_rects(&mut single);
        assert_eq!(single.len(), 1);
    }

    #[test]
    fn thick_stroked_rule_does_not_mark_underline() {
        let content = b"BT /F1 12 Tf 1 0 0 1 100 500 Tm (THICK) Tj ET
4 w
100 498 m 170 498 l S
BT /F1 12 Tf 1 0 0 1 100 480 Tm (THIN) Tj ET
1 w
100 478 m 160 478 l S";

        let items = extract_simple_items(content);
        let thick = items.iter().find(|item| item.text == "THICK").unwrap();
        let thin = items.iter().find(|item| item.text == "THIN").unwrap();

        assert!(!thick.is_underline);
        assert!(thin.is_underline);
    }

    #[test]
    fn rotated_page_underline_is_detected_after_coordinate_correction() {
        let content = b"BT /F1 12 Tf 0 1 -1 0 200 100 Tm (HELLO) Tj ET
BT /F1 12 Tf 0 1 -1 0 240 100 Tm (WORLD) Tj ET
1 w
202 100 m 202 170 l S";

        let items = extract_simple_items(content);
        let hello = items.iter().find(|item| item.text == "HELLO").unwrap();
        let world = items.iter().find(|item| item.text == "WORLD").unwrap();

        assert!(hello.is_underline);
        assert!(!world.is_underline);
    }

    #[test]
    fn quote_operator_text_carries_advance_width() {
        // `'` (move-to-next-line-and-show-text) must retain the string's
        // advance width like Tj — zero-width items are invisible to
        // geometric underline/strikeout detection.
        let content = b"BT /F1 12 Tf 12 TL 1 0 0 1 100 512 Tm (first) Tj (struck) ' ET
1 w
99 503 m 145 503 l S";

        let items = extract_simple_items(content);
        let struck = items.iter().find(|item| item.text == "struck").unwrap();

        // 6 glyphs x 600/1000 x 12pt = 43.2pt, drawn one leading below Tm.
        assert!((struck.width - 43.2).abs() < 0.1);
        assert!((struck.y - 500.0).abs() < 0.1);
        assert!(struck.is_strikeout);
        assert!(!struck.is_underline);
    }

    #[test]
    fn quote_operator_advances_text_matrix() {
        // Text shown after `'` on the same line must start past the shown
        // string: "CD" lands at x=114.4 (2 glyphs x 600/1000 x 12pt after
        // x=100), flush against "AB", so the merge pass joins them. Without
        // the advance "CD" overlaps "AB" at x=100 and the items stay apart.
        let content = b"BT /F1 12 Tf 12 TL 1 0 0 1 100 512 Tm (AB) ' (CD) Tj ET";

        let items = extract_simple_items(content);
        let merged = items.iter().find(|item| item.text == "ABCD").unwrap();

        assert!((merged.x - 100.0).abs() < 0.1);
        assert!((merged.width - 28.8).abs() < 0.1);
        assert!((merged.y - 500.0).abs() < 0.1);
    }

    #[test]
    fn text_rise_shifts_item_baseline() {
        // Ts displaces the glyph origin vertically without touching the
        // advance; the next run at rise 0 must return to the original
        // baseline and follow the raised run horizontally.
        let content =
            b"BT /F1 12 Tf 1 0 0 1 100 500 Tm (base) Tj 5 Ts (super) Tj 0 Ts (after) Tj ET";

        let items = extract_simple_items(content);
        let base = items.iter().find(|item| item.text == "base").unwrap();
        let raised = items.iter().find(|item| item.text == "super").unwrap();
        let after = items.iter().find(|item| item.text == "after").unwrap();

        assert!((base.y - 500.0).abs() < 0.1);
        assert!((raised.y - 505.0).abs() < 0.1);
        assert!((after.y - 500.0).abs() < 0.1);
        assert!(after.x > raised.x);
    }

    #[test]
    fn actual_text_item_uses_glyph_rise() {
        // The ActualText replacement item must render at the rise in
        // effect when its glyphs were drawn — not the unshifted BDC
        // baseline, and not whatever rise is set by EMC time.
        let content = b"BT /F1 12 Tf 1 0 0 1 100 500 Tm \
/Span <</ActualText (super) >> BDC 5 Ts (sup) Tj 0 Ts EMC (after) Tj ET";

        let items = extract_simple_items(content);
        let sup = items.iter().find(|item| item.text == "super").unwrap();
        let after = items.iter().find(|item| item.text == "after").unwrap();

        assert!((sup.y - 505.0).abs() < 0.1);
        assert!((after.y - 500.0).abs() < 0.1);
    }

    #[test]
    fn actual_text_shown_with_quote_op_uses_moved_risen_baseline() {
        // When the tagged span's show op is `'`, the glyph position is
        // only known AFTER its line move — falling back to the BDC-entry
        // matrix would place the item on the previous line, unrisen.
        let content = b"BT /F1 12 Tf 14 TL 1 0 0 1 100 500 Tm \
/Span <</ActualText (replaced) >> BDC 3 Ts (raw) ' 0 Ts EMC ET";

        let items = extract_simple_items(content);
        let item = items.iter().find(|item| item.text == "replaced").unwrap();

        // Line move: 500 - 14 = 486; rise: +3 -> 489.
        assert!((item.y - 489.0).abs() < 0.1);
        assert!(item.width > 0.0);
    }

    #[test]
    fn strikeout_detected_on_risen_text() {
        // The rule crosses the glyphs at their risen position; without the
        // rise in item.y the strike window sits 4pt too low and misses.
        let content = b"BT /F1 12 Tf 1 0 0 1 100 500 Tm 4 Ts (struck) Tj ET
1 w
99 507 m 145 507 l S";

        let items = extract_simple_items(content);
        let struck = items.iter().find(|item| item.text == "struck").unwrap();

        assert!((struck.y - 504.0).abs() < 0.1);
        assert!(struck.is_strikeout);
        assert!(!struck.is_underline);
    }

    #[test]
    fn test_skip_excessive_operations() {
        use crate::tounicode::FontCMaps;
        use lopdf::{dictionary, Object, Stream};

        let mut doc = lopdf::Document::new();

        // "0 0 m\n" = 6 bytes per op, 1_100_000 ops → ~6.6 MB content stream
        let ops_bytes = "0 0 m\n".repeat(1_100_000).into_bytes();
        let stream = Stream::new(dictionary! {}, ops_bytes);
        let content_id = doc.add_object(Object::Stream(stream));

        let page_dict = dictionary! {
            "Type" => "Page",
            "Contents" => Object::Reference(content_id),
            "Resources" => dictionary! {},
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        };
        let page_id = doc.add_object(page_dict);

        // Register the page so get_page_content can find it
        let pages_dict = dictionary! {
            "Type" => "Pages",
            "Count" => Object::Integer(1),
            "Kids" => vec![Object::Reference(page_id)],
        };
        let pages_id = doc.add_object(pages_dict);
        let catalog = dictionary! {
            "Type" => "Catalog",
            "Pages" => Object::Reference(pages_id),
        };
        doc.add_object(catalog);

        let font_cmaps = FontCMaps::from_doc(&doc);
        let result = extract_page_text_items(
            &doc,
            page_id,
            1,
            &font_cmaps,
            false,
            &mut FontStyleCache::new(),
        )
        .unwrap();
        let ((items, rects, lines), _has_gid, _coords_rotated) = result;
        assert!(items.is_empty());
        assert!(rects.is_empty());
        assert!(lines.is_empty());
    }

    #[test]
    fn test_q_restores_current_font_for_text_decoding() {
        use crate::tounicode::FontCMaps;
        use lopdf::{dictionary, Object, Stream};

        fn cmap_stream(dst_hex: &str) -> Stream {
            let cmap = format!(
                r#"/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def
/CMapName /Test-UCS def
/CMapType 2 def
1 begincodespacerange
<00> <FF>
endcodespacerange
1 beginbfchar
<41> <{dst_hex}>
endbfchar
endcmap
CMapName currentdict /CMap defineresource pop
end
end"#
            );
            Stream::new(dictionary! {}, cmap.into_bytes())
        }

        let mut doc = lopdf::Document::new();
        let f1_cmap = doc.add_object(Object::Stream(cmap_stream("0058"))); // X
        let f2_cmap = doc.add_object(Object::Stream(cmap_stream("0059"))); // Y
        let f1 = doc.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "Helvetica",
            "ToUnicode" => Object::Reference(f1_cmap),
        });
        let f2 = doc.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "Helvetica",
            "ToUnicode" => Object::Reference(f2_cmap),
        });

        let content = b"BT /F1 12 Tf 10 700 Tm <41> Tj ET
q
BT /F2 12 Tf 20 700 Tm <41> Tj ET
Q
BT 30 700 Tm <41> Tj ET";
        let content_id = doc.add_object(Object::Stream(Stream::new(
            dictionary! {},
            content.to_vec(),
        )));
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Contents" => Object::Reference(content_id),
            "Resources" => dictionary! {
                "Font" => dictionary! {
                    "F1" => Object::Reference(f1),
                    "F2" => Object::Reference(f2),
                },
            },
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        });
        let pages_id = doc.add_object(dictionary! {
            "Type" => "Pages",
            "Count" => Object::Integer(1),
            "Kids" => vec![Object::Reference(page_id)],
        });
        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => Object::Reference(pages_id),
        });
        doc.trailer.set("Root", Object::Reference(catalog_id));

        let font_cmaps = FontCMaps::from_doc(&doc);
        let ((items, _, _), _, _) = extract_page_text_items(
            &doc,
            page_id,
            1,
            &font_cmaps,
            false,
            &mut FontStyleCache::new(),
        )
        .unwrap();
        let text = items
            .iter()
            .map(|item| item.text.as_str())
            .collect::<String>();

        assert_eq!(text, "XYX");
    }

    #[test]
    fn test_strip_pdf_comments() {
        // Basic comment stripping
        let input = b"BT\n% comment\nTj\nET\n";
        let output = strip_pdf_comments(input);
        assert_eq!(output.as_ref(), b"BT\n \nTj\nET\n");

        // No comments = unchanged
        let input = b"BT\nTj\nET\n";
        let output = strip_pdf_comments(input);
        assert_eq!(output, input.to_vec());

        // Don't strip inside string literals
        let input = b"(text with % not a comment)\n% real comment\n";
        let output = strip_pdf_comments(input);
        assert_eq!(output.as_ref(), b"(text with % not a comment)\n \n");

        // Don't strip inside hex strings
        let input = b"<0033% not a comment>\n% real comment\n";
        let output = strip_pdf_comments(input);
        assert_eq!(output.as_ref(), b"<0033% not a comment>\n \n");

        // PD4ML style: comment between Tj and ET
        let input = b"<0033> Tj\n\t% Mission Statement\n\tET\n";
        let output = strip_pdf_comments(input);
        let output_str = String::from_utf8_lossy(&output);
        assert!(
            output_str.contains("ET"),
            "ET should be preserved after comment stripping"
        );
    }

    #[test]
    fn test_strip_pdf_comments_escaped_parens() {
        // An escaped `\)` must not close the string: the `%` after it is
        // still string content, not a comment (subset fonts routinely map
        // glyphs to `%` and to escaped parens in the same TJ array).
        let input = b"[ (a\\)b) 1 (%) 1 (c) ] TJ\n";
        let output = strip_pdf_comments(input);
        assert_eq!(output, input.to_vec());

        // Same for an escaped `\(` — must not open a phantom string that
        // shields a real comment.
        let input = b"(x\\(y) Tj % real comment\nET\n";
        let output = strip_pdf_comments(input);
        assert_eq!(output.as_ref(), b"(x\\(y) Tj  \nET\n");

        // Escaped backslash before a real close-paren: `\\` ends the escape,
        // the `)` does close the string, and the comment is stripped.
        let input = b"(x\\\\) Tj % comment\nET\n";
        let output = strip_pdf_comments(input);
        assert_eq!(output.as_ref(), b"(x\\\\) Tj  \nET\n");
    }
}
