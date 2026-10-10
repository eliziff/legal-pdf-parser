//! The heading decision composed over each line's witnesses (legal-pdf-witnesses), applied over
//! the whole document: a line the composition reads as a heading is one, a reader's heading it
//! reads otherwise is not, and each new heading takes its place in the document's own heading
//! scheme.

use legal_pdf_core::model::{Line, Page};
use std::collections::HashMap;

fn counted(line: &Line) -> bool {
    !line.exclude_from_body
        && line.note_region_mode.is_empty()
        && matches!(line.region_type.as_str(), "body" | "heading")
        && !line.text.trim().is_empty()
}

fn bold(line: &Line) -> bool {
    let spans = line.spans.iter().filter(|span| !span.text.trim().is_empty()).collect::<Vec<_>>();
    !spans.is_empty() && spans.iter().all(|span| span.flags & 16 != 0 || span.font.to_lowercase().contains("bold"))
}

fn size(line: &Line) -> f64 {
    line.spans.iter().map(|span| span.size).fold(0.0, f64::max)
}

/// Apply `decide` (Some(heading?) per line id; None keeps the reader's reading) to every counted
/// line. `levels` keeps the reader's levels for the headings it kept and loses the demoted ones; a
/// new heading in a style the reader's headings use takes that style's shallowest level, and one in
/// a style of its own goes one below the nearest earlier heading set more prominently.
pub fn compose_headings(pages: &mut [Page], levels: &mut HashMap<String, usize>, decide: &dyn Fn(&str) -> Option<bool>) {
    let mut sizes = pages
        .iter()
        .flat_map(|page| page.lines.iter().filter(|line| counted(line)))
        .map(size)
        .filter(|s| *s > 0.0)
        .collect::<Vec<_>>();
    sizes.sort_by(f64::total_cmp);
    let body = sizes.get(sizes.len() / 2).copied().unwrap_or(0.0);
    let style = |line: &Line| {
        let step = if body > 0.0 { (size(line) / body * 10.0).round() as i64 } else { 0 };
        let letters = line.text.chars().filter(|c| c.is_alphabetic()).count();
        let caps = letters > 0 && line.text.chars().filter(|c| c.is_uppercase()).count() * 10 >= letters * 9;
        (bold(line), step, caps)
    };
    let prominence = |(bold, step, caps): (bool, i64, bool)| step * 4 + i64::from(bold) * 2 + i64::from(caps);
    let mut scheme = HashMap::<(bool, i64, bool), usize>::new();
    for line in pages.iter().flat_map(|page| &page.lines).filter(|line| counted(line) && line.region_type == "heading") {
        if let Some(&level) = levels.get(&line.id) {
            let entry = scheme.entry(style(line)).or_insert(level);
            *entry = (*entry).min(level);
        }
    }
    let mut earlier = Vec::<((bool, i64, bool), usize)>::new();
    for page in pages.iter_mut() {
        for line in page.lines.iter_mut().filter(|line| counted(line)) {
            let own = style(line);
            match decide(&line.id) {
                Some(true) => {
                    let level = match levels.get(&line.id) {
                        Some(&level) if line.region_type == "heading" => level,
                        _ => scheme.get(&own).copied().unwrap_or_else(|| {
                            earlier
                                .iter()
                                .rev()
                                .find(|(style, _)| prominence(*style) > prominence(own))
                                .map_or(1, |(_, level)| level + 1)
                        }),
                    };
                    line.region_type = "heading".to_owned();
                    levels.insert(line.id.clone(), level);
                    earlier.push((own, level));
                }
                Some(false) if line.region_type == "heading" => {
                    line.region_type = "body".to_owned();
                    levels.remove(&line.id);
                }
                _ => {
                    if line.region_type == "heading" {
                        if let Some(&level) = levels.get(&line.id) {
                            earlier.push((own, level));
                        }
                    }
                }
            }
        }
    }
}
