//! A page's words as pdf-inspector's text items, and the rules and filled rectangles its table readers read.

use legal_pdf_core::model::{Line, Page, Span, ITALIC, MONOSPACED};
use legal_pdf_core::{flag_share, BOLD};
use legal_pdf_structure_model::{PageItems, WHOLE};
use pdf_inspector::types::{ItemType, PdfLine, PdfRect, TextItem};

/// A page's words as text items on the page's own number (its index + 1), the number its rules
/// and filled rectangles are read on: the one reading of the page its table readers share.
pub fn page_items(page: &Page) -> PageItems {
    line_items(&page.lines, page.index as u32 + 1, page.height)
}

/// Lines' words as pdf-inspector's text items (points, y up, on page `number`), with the (line,
/// word) each was read from, so its readers read our pages, recognized ones too. Lines the parser
/// holds out of the body (a transcript's line numbers) are not offered.
pub fn line_items(lines: &[Line], number: u32, height: f64) -> PageItems {
    let (mut items, mut origin) = (Vec::new(), Vec::new());
    // Each word as its span sets it: face, weight, slant and pitch; a recognized line's by its
    // strokes alone.
    let item = |line: &Line,
                text: &str,
                bbox: [f64; 4],
                span: Option<&Span>,
                underlined: bool,
                struck: bool| TextItem {
        text: text.to_owned(),
        x: bbox[0] as f32,
        y: (height - bbox[3]) as f32,
        width: (bbox[2] - bbox[0]) as f32,
        height: (bbox[3] - bbox[1]) as f32,
        rotation: 0.0,
        advance_known: true,
        font: span.map_or(String::new(), |span| span.font.clone()),
        font_tag: String::new(),
        legacy_symbol_rewrite: false,
        font_size: line.size.max(bbox[3] - bbox[1]) as f32,
        page: number,
        is_bold: span.map_or_else(|| bold(line), |span| span.flags & BOLD != 0),
        is_italic: span.is_some_and(|span| span.flags & ITALIC != 0),
        font_weight: None,
        bold_source: None,
        fixed_pitch: span.map(|span| span.flags & MONOSPACED != 0),
        fill_color: None,
        stroke_color: None,
        render_mode: None,
        is_underline: underlined,
        is_strikeout: struck,
        fidelity: None,
        item_type: ItemType::Text,
        mcid: None,
        baseline_shift: 0.0,
    };
    for (slot, line) in lines.iter().enumerate() {
        if line.exclude_from_body
            || line.text.trim().is_empty()
            || !line.bbox.iter().all(|value| value.is_finite())
        {
            continue;
        }
        let span = |at: usize| {
            line.spans
                .iter()
                .find(|span| span.start <= at && at < span.end.max(span.start + 1))
                .or(line.spans.first())
        };
        let words: Vec<(usize, &legal_pdf_core::model::Word)> = line
            .words
            .iter()
            .enumerate()
            .filter(|(_, word)| !word.text.trim().is_empty())
            .collect();
        // A recognizer pads the box it reads a line in (a scan's two columns' boxes all but meet across their
        // gutter): a recognized line's words are offered within its ink, a quarter of its height in from each end.
        let pad = if line.ocr.is_some() {
            0.25 * (line.bbox[3] - line.bbox[1]).max(0.0)
        } else {
            0.0
        };
        let inked = |bbox: [f64; 4]| {
            let (left, right) = (
                bbox[0].max(line.bbox[0] + pad),
                bbox[2].min(line.bbox[2] - pad),
            );
            let middle = (bbox[0] + bbox[2]) / 2.0;
            if right > left {
                [left, bbox[1], right, bbox[3]]
            } else {
                [middle - 0.5, bbox[1], middle + 0.5, bbox[3]]
            }
        };
        if words.is_empty() {
            items.push(item(
                line,
                line.text.trim(),
                inked(line.bbox),
                span(0),
                false,
                false,
            ));
            origin.push((slot, WHOLE));
        }
        for (at, word) in words {
            items.push(item(
                line,
                &word.text,
                inked(word.bbox),
                span(word.start),
                word.underlined,
                word.struck,
            ));
            origin.push((slot, at));
        }
    }
    PageItems { items, origin }
}

/// Clears the underline and strike read under the words of a table drawn with rules or filled
/// cells, as pdf-inspector clears them (suppress_table_underlines): the table's rulings, not the
/// words'. The page's items (`page_items`) are cleared with its words.
pub fn clear_table_underlines(page: &mut Page, items: &mut PageItems) {
    if !items
        .items
        .iter()
        .any(|item| item.is_underline || item.is_strikeout)
    {
        return;
    }
    let marked: Vec<bool> = items
        .items
        .iter()
        .map(|item| item.is_underline || item.is_strikeout)
        .collect();
    pdf_inspector::extractor::suppress_table_underlines(
        &mut items.items,
        &rects(page),
        &rules(page),
        page.index as u32 + 1,
    );
    for ((item, marked), &(line, word)) in items.items.iter().zip(marked).zip(&items.origin) {
        if marked && !item.is_underline && !item.is_strikeout && word != WHOLE {
            let word = &mut page.lines[line].words[word];
            word.underlined = false;
            word.struck = false;
        }
    }
}

/// The page's painted rules as pdf-inspector's lines (points, y up).
pub fn rules(page: &Page) -> Vec<PdfLine> {
    let (number, flip) = (page.index as u32 + 1, |y: f64| (page.height - y) as f32);
    let rule = |[x0, y0, x1, y1]: [f64; 4]| {
        if (y1 - y0).abs() <= (x1 - x0).abs() {
            let at = flip((y0 + y1) / 2.0);
            PdfLine {
                x1: x0.min(x1) as f32,
                y1: at,
                x2: x0.max(x1) as f32,
                y2: at,
                page: number,
            }
        } else {
            let at = ((x0 + x1) / 2.0) as f32;
            PdfLine {
                x1: at,
                y1: flip(y0.max(y1)),
                x2: at,
                y2: flip(y0.min(y1)),
                page: number,
            }
        }
    };
    page.regions
        .iter()
        .filter(|region| region.kind == "rule")
        .map(|region| rule(region.bbox))
        .collect()
}

/// The page's filled rectangles as pdf-inspector's rects (points, y up, from the bottom-left
/// corner).
pub fn rects(page: &Page) -> Vec<PdfRect> {
    let number = page.index as u32 + 1;
    let rect = |[x0, y0, x1, y1]: [f64; 4]| PdfRect {
        x: x0.min(x1) as f32,
        y: (page.height - y0.max(y1)) as f32,
        width: (x1 - x0).abs() as f32,
        height: (y1 - y0).abs() as f32,
        page: number,
    };
    page.regions
        .iter()
        .filter(|region| region.kind == "rect")
        .map(|region| rect(region.bbox))
        .collect()
}

fn bold(line: &Line) -> bool {
    flag_share(line, BOLD).is_some_and(|share| share >= 0.8)
}
