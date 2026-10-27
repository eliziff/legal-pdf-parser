//! The page's lines in the order its source writes them, made into the model's lines.

use super::*;

pub(crate) fn make_line(
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
        ocr: None,
        exclude_from_body: false,
        ink: 0.0,
        weight: 0.0,
        cap_height: 0.0,
        suppress_footnote_label: false,
        paragraph_label_candidate: false,
        note_region_mode: String::new(),
        region_id: String::new(),
        region_type: "text".to_owned(),
        source: "native".to_owned(),
    })
}

pub(crate) fn shares_source_line(line: &TextLine, item: &TextItem) -> bool {
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
pub(crate) fn group_source_order_lines(items: Vec<TextItem>) -> Vec<TextLine> {
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
