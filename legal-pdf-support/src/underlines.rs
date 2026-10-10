//! Underlines on recognized pages, read from the page image: a recognized page has no painted
//! rules to read, but an underline is ink like any other. Under each word, in the band where an
//! underline is set, a thin dark run across most of the word is its underline; a run as tall as
//! the type is the next line's text or a filled box, not one.

use hayro::hayro_interpret::InterpreterSettings;
use hayro::hayro_syntax::Pdf;
use hayro::vello_cpu::color::palette::css::WHITE;
use hayro::{render, RenderCache, RenderSettings};
use legal_pdf_core::model::Page;

const DPI: f32 = 144.0;
const DARK: u8 = 110;

/// Mark the words a drawn line underlines on every recognized page.
pub fn mark_recognized_underlines(bytes: &[u8], pages: &mut [Page]) {
    if !pages.iter().any(|page| page.source != "native" && page.lines.iter().any(|line| !line.words.is_empty())) {
        return;
    }
    let Ok(pdf) = Pdf::new(bytes.to_vec()) else {
        return;
    };
    let cache = RenderCache::new();
    let interpreter = InterpreterSettings::default();
    let scale = DPI / 72.0;
    let settings = RenderSettings { x_scale: scale, y_scale: scale, bg_color: WHITE, ..Default::default() };
    for page in pages.iter_mut().filter(|page| page.source != "native") {
        if page.lines.iter().all(|line| line.words.is_empty()) {
            continue;
        }
        let Some(pdf_page) = pdf.pages().iter().nth(page.index) else {
            continue;
        };
        let pixmap = render(pdf_page, &cache, &interpreter, &settings);
        let (width, height) = (usize::from(pixmap.width()), usize::from(pixmap.height()));
        if width == 0 || height == 0 || page.width <= 0.0 || page.height <= 0.0 {
            continue;
        }
        let dark = pixmap
            .data_as_u8_slice()
            .chunks_exact(4)
            .map(|pixel| ((u16::from(pixel[0]) * 3 + u16::from(pixel[1]) * 6 + u16::from(pixel[2])) / 10) < u16::from(DARK))
            .collect::<Vec<bool>>();
        let (sx, sy) = (width as f64 / page.width, height as f64 / page.height);
        for line in &mut page.lines {
            let line_height = (line.bbox[3] - line.bbox[1]).max(1.0);
            for word in &mut line.words {
                word.underlined = underlined(&dark, width, height, word.bbox, line_height, sx, sy);
            }
        }
    }
}

fn underlined(dark: &[bool], width: usize, height: usize, word: [f64; 4], line_height: f64, sx: f64, sy: f64) -> bool {
    let x0 = ((word[0] * sx).floor().max(0.0) as usize).min(width.saturating_sub(1));
    let x1 = ((word[2] * sx).ceil() as usize).min(width.saturating_sub(1));
    if x1 <= x0 + 2 {
        return false;
    }
    let top = (((word[3] - 0.3 * line_height) * sy).floor().max(0.0) as usize).min(height.saturating_sub(1));
    let bottom = (((word[3] + 0.5 * line_height) * sy).ceil() as usize).min(height.saturating_sub(1));
    let span = (x1 - x0) as f64;
    // Each band row's longest dark run, bridging gaps of two pixels.
    let covered = |row: usize| {
        let (mut best, mut run, mut gap) = (0usize, 0usize, 0usize);
        for x in x0..=x1 {
            if dark[row * width + x] {
                run += gap + 1;
                gap = 0;
                best = best.max(run);
            } else if run > 0 && gap < 2 {
                gap += 1;
            } else {
                run = 0;
                gap = 0;
            }
        }
        best as f64 >= 0.6 * span
    };
    let rows = (top..=bottom).map(covered).collect::<Vec<_>>();
    let thickest = (0.15 * line_height * sy).ceil().max(2.0) as usize;
    let mut index = 0;
    while index < rows.len() {
        if rows[index] {
            let start = index;
            while index < rows.len() && rows[index] {
                index += 1;
            }
            if index - start <= thickest {
                return true;
            }
        } else {
            index += 1;
        }
    }
    false
}
