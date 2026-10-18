//! Physical associations retain both recognized source lines; semantic roles belong downstream.
use crate::{layout_analysis, OcrMarkerEvidence, Page};
use std::collections::HashMap;

pub fn associate_ocr_markers(pages: &mut [Page]) {
    if !pages.iter().flat_map(|page| &page.lines).any(|line| line.ocr.is_some()) {
        return;
    }
    let analysis = layout_analysis();
    let mut markers = Vec::new();
    let mut previous = HashMap::new();
    for (page_index, page) in pages.iter().enumerate() {
        for (line_index, line) in page.lines.iter().enumerate() {
            let Some(ocr) = &line.ocr else { continue };
            let text = line.text.trim();
            let column = (line.bbox[0] / page.width.max(1.0) * 50.0).round() as i32;
            let Some(marker) = analysis.numbering_marker(text, previous.get(&column)) else { continue };
            previous.insert(column, marker.clone());
            let height = line.bbox[3] - line.bbox[1];
            let mut bodies = page.lines.iter().enumerate().filter_map(|(index, body)| {
                let body_height = body.bbox[3] - body.bbox[1];
                let gap = body.bbox[0] - line.bbox[2];
                let overlap = line.bbox[3].min(body.bbox[3]) - line.bbox[1].max(body.bbox[1]);
                let order_gap = body.reading_order.checked_sub(line.reading_order)?;
                if index == line_index || height <= 0.0 || body_height <= 0.0
                    || order_gap == 0 || order_gap > 3 || gap < 0.0 || gap > body_height * 3.0
                    || overlap < height.min(body_height) * 0.5
                    || (line.bbox[1] + line.bbox[3] - body.bbox[1] - body.bbox[3]).abs() > body_height * 0.8
                    || body.text.chars().filter(|c| c.is_alphabetic()).count() < 4
                    || analysis.numbering_marker(body.text.trim(), None).is_some()
                {
                    return None;
                }
                Some((gap, body_height, body.id.clone()))
            }).collect::<Vec<_>>();
            bodies.sort_by(|a,b| a.0.total_cmp(&b.0));
            let body = bodies.first().filter(|first| bodies.get(1)
                .is_none_or(|second| second.0 - first.0 > first.1 * 0.5)).map(|body| body.2.clone());
            markers.push((page_index, line_index, marker, body,
                line.bbox[0] / page.width.max(1.0), ocr.confidence));
        }
    }
    let mut ordinals = HashMap::<_, Vec<usize>>::new();
    for (index, (_, _, marker, _, _, _)) in markers.iter().enumerate() {
        ordinals.entry((marker.family.clone(), marker.prefix.clone(), marker.ordinal))
            .or_default().push(index);
    }
    let mut assignments = Vec::new();
    for (index, (page_index, line_index, marker, body, column, confidence)) in markers.iter().enumerate() {
        let mut sequence_line_ids = Vec::new();
        for (ordinal, before) in [(marker.ordinal.checked_sub(1), true), (marker.ordinal.checked_add(1), false)] {
            let Some(ordinal) = ordinal else { continue };
            let Some(others) = ordinals.get(&(marker.family.clone(), marker.prefix.clone(), ordinal)) else { continue };
            if let Some(other) = others.iter().filter(|&&other| {
                let candidate = &markers[other];
                (other < index) == before && (candidate.4 - column).abs() <= 0.025
                    && candidate.3.is_some() && candidate.5 >= 0.85
            }).min_by_key(|&&other| index.abs_diff(other)) {
                let candidate = &markers[*other];
                sequence_line_ids.push(pages[candidate.0].lines[candidate.1].id.clone());
            }
        }
        let line = &pages[*page_index].lines[*line_index];
        let delimited = line.text.trim().chars().any(|c| !c.is_alphanumeric());
        let supported = *confidence >= 0.85 && (delimited || !sequence_line_ids.is_empty());
        assignments.push((*page_index, *line_index, OcrMarkerEvidence {
            body_line_id: body.clone().filter(|_| supported), sequence_line_ids,
        }));
    }
    for (page, line, marker) in assignments {
        if let Some(ocr) = &mut pages[page].lines[line].ocr {
            ocr.marker = Some(marker);
        }
    }
}
