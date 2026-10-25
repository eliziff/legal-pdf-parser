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
            let width = line.bbox[2] - line.bbox[0];
            let mut bodies = page.lines.iter().enumerate().filter_map(|(index, body)| {
                let body_height = body.bbox[3] - body.bbox[1];
                let gap = body.bbox[0] - line.bbox[2];
                let overlap = line.bbox[3].min(body.bbox[3]) - line.bbox[1].max(body.bbox[1]);
                // The body may be read just before the number as well as after it, and a recognizer's
                // body box may start inside the number's box (its padding, or the number's trailing
                // bracket): the body starts right of the number's middle.
                let order_gap = body.reading_order as i64 - line.reading_order as i64;
                if index == line_index || height <= 0.0 || body_height <= 0.0
                    || order_gap == 0 || order_gap > 3 || order_gap < -2 || gap < -width * 0.5 || gap > body_height * 3.0
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

/// A number standing on a line of its own joins its paragraph's first line, as a typeset page sets a
/// hanging number and as the structure stage reads it: "[35]" and "All of the decisions ..." become the
/// one line "[35] All of the decisions ...", the number its first word, the box the union. A recognized
/// number joins the body `associate_ocr_markers` read it beside; a native number (an extractor splits
/// "1." from its paragraph at the tab stop) joins the one line set beside it on its baseline, within three
/// line heights, when no other line competes. Only a line that is nothing but its number joins; a number
/// with words of its own stays. Returns how many joined.
pub fn merge_marker_lines(pages: &mut [Page]) -> usize {
    let analysis = layout_analysis();
    let mut merged = 0;
    for page in pages {
        let mut joins: Vec<(usize, usize)> = Vec::new();
        for (index, line) in page.lines.iter().enumerate() {
            let text = line.text.trim();
            // The whole line is the number: one token of a few characters, nothing after it.
            if text.chars().count() > 7 || text.chars().any(char::is_whitespace) || analysis.numbering_marker(text, None).is_none() {
                continue;
            }
            let body = match &line.ocr {
                Some(ocr) => ocr.marker.as_ref().and_then(|m| m.body_line_id.as_deref())
                    .and_then(|body_id| page.lines.iter().position(|other| other.id == body_id)),
                None => native_body(&page.lines, index),
            };
            if let Some(body) = body {
                if body != index && !joins.iter().any(|(_, b)| *b == body) {
                    joins.push((index, body));
                }
            }
        }
        if joins.is_empty() {
            continue;
        }
        let mut remove = Vec::new();
        for (marker_index, body_index) in joins {
            let marker = page.lines[marker_index].clone();
            let body = &mut page.lines[body_index];
            let label = marker.text.trim().to_owned();
            let shift = label.chars().count() + 1;
            for word in &mut body.words {
                word.start += shift;
                word.end += shift;
            }
            body.words.insert(0, crate::Word {
                id: format!("{}-w000", body.id),
                text: label.clone(),
                bbox: marker.bbox,
                start: 0,
                end: label.chars().count(),
                underlined: false,
                weight: 0.0,
            });
            for (k, word) in body.words.iter_mut().enumerate() {
                word.id = format!("{}-w{:03}", body.id, k + 1);
            }
            body.text = format!("{label} {}", body.text);
            body.bbox = [
                body.bbox[0].min(marker.bbox[0]), body.bbox[1].min(marker.bbox[1]),
                body.bbox[2].max(marker.bbox[2]), body.bbox[3].max(marker.bbox[3]),
            ];
            if let (Some(own), Some(theirs)) = (body.ocr.as_mut(), marker.ocr.as_ref()) {
                own.confidence = own.confidence.min(theirs.confidence);
                own.marker = None;
            }
            remove.push(marker_index);
            merged += 1;
        }
        remove.sort_unstable();
        for index in remove.into_iter().rev() {
            page.lines.remove(index);
        }
    }
    merged
}

/// The line a native number joins: set beside it on its baseline, starting right of the number's middle
/// and within three line heights of its end, with words of its own and not a number itself; the nearest
/// such line, and none when a second one competes for it.
fn native_body(lines: &[crate::Line], marker: usize) -> Option<usize> {
    let analysis = layout_analysis();
    let line = &lines[marker];
    let height = line.bbox[3] - line.bbox[1];
    let width = line.bbox[2] - line.bbox[0];
    if height <= 0.0 {
        return None;
    }
    let mut bodies: Vec<(f64, f64, usize)> = lines.iter().enumerate().filter_map(|(index, body)| {
        let body_height = body.bbox[3] - body.bbox[1];
        let gap = body.bbox[0] - line.bbox[2];
        let overlap = line.bbox[3].min(body.bbox[3]) - line.bbox[1].max(body.bbox[1]);
        if index == marker || body.ocr.is_some() || body_height <= 0.0
            || gap < -width * 0.5 || gap > body_height * 3.0
            || overlap < height.min(body_height) * 0.5
            || (line.bbox[1] + line.bbox[3] - body.bbox[1] - body.bbox[3]).abs() > body_height * 0.8
            || body.text.chars().filter(|c| c.is_alphabetic()).count() < 4
            || analysis.numbering_marker(body.text.trim(), None).is_some()
        {
            return None;
        }
        Some((gap, body_height, index))
    }).collect();
    bodies.sort_by(|a, b| a.0.total_cmp(&b.0));
    let first = bodies.first()?;
    if bodies.get(1).is_some_and(|second| second.0 - first.0 <= first.1 * 0.5) {
        return None;
    }
    Some(first.2)
}
