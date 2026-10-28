//! A numbering marker printed on a line of its own, joined to the line that carries its text.

use super::*;

pub(crate) fn marker_alone() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^\s*(?:\(?(?:\d{1,3}(?:\.\d{1,3}){0,3}|[A-Za-z]{1,2}|[ivxlc]{1,6}|[IVXLC]{1,6})[.)]|\(\s*(?:\d{1,3}|[A-Za-z]{1,2}|[ivxlc]{1,6}|[IVXLC]{1,6})\s*\)|\[\s*\d{1,4}\s*\]|[•·▪◦●○■□➢✓\-–—*\u{f0b7}\u{f0a7}\u{f0d8}])\s*$")
            .expect("detached marker regex")
    })
}

/// A clause's decimal number standing alone, as contracts print it in a column of its own
/// ("3.05", "*1.04" for an amended clause, "10.01 (a)" with its first item).
pub(crate) fn decimal_alone() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^\s*\*?\d{1,3}(?:\.\d{1,3}){1,3}(?:\s*\(\s*(?:\d{1,3}|[A-Za-z]{1,2}|[ivxlc]{1,6})\s*\))?\s*$")
            .expect("decimal marker regex")
    })
}

/// A numbering marker printed on a line of its own ("11.", "(a)", "[12]", "•", "3.05") belongs to the
/// line that carries its text: the nearest line to its right on the same row, within six line
/// heights. A bare decimal joins only words (a letter, bracket or quote first), never a table's
/// next figure. The marker joins the front of that line, which keeps its identity; the marker's
/// line goes. Returns the number of joins.
pub(crate) fn join_detached_markers(page: &mut Page) -> usize {
    let height = |line: &Line| (line.bbox[3] - line.bbox[1]).max(0.1);
    let alone = |line: &Line| marker_alone().is_match(&line.text) || decimal_alone().is_match(&line.text);
    let mut joins: Vec<(usize, usize)> = Vec::new(); // (marker, body)
    let mut taken: HashSet<usize> = HashSet::new();
    for (m, marker) in page.lines.iter().enumerate() {
        let decimal = decimal_alone().is_match(&marker.text);
        if !has_valid_bbox(marker) || marker.text.chars().count() > 12 || !(decimal || marker_alone().is_match(&marker.text)) {
            continue;
        }
        let h = height(marker);
        let mut best: Option<(f64, usize)> = None;
        for (b, body) in page.lines.iter().enumerate() {
            if b == m || taken.contains(&b) || !has_valid_bbox(body) || alone(body) {
                continue;
            }
            if decimal && !body.text.trim_start().starts_with(|c: char| c.is_alphabetic() || matches!(c, '(' | '"' | '“' | '\'' | '‘')) {
                continue;
            }
            let overlap = body.bbox[3].min(marker.bbox[3]) - body.bbox[1].max(marker.bbox[1]);
            let gap = body.bbox[0] - marker.bbox[2];
            if overlap >= 0.5 * h.min(height(body)) && gap >= -0.25 * h && gap <= 6.0 * h {
                if best.is_none_or(|(g, _)| gap < g) {
                    best = Some((gap, b));
                }
            }
        }
        if let Some((_, b)) = best {
            taken.insert(b);
            joins.push((m, b));
        }
    }
    if joins.is_empty() {
        return 0;
    }
    let mut removed: HashSet<usize> = HashSet::new();
    let mut renamed: HashMap<String, String> = HashMap::new();
    for &(m, b) in &joins {
        let marker = page.lines[m].clone();
        let body = &mut page.lines[b];
        let prefix = marker.text.trim().to_owned();
        let shift = prefix.chars().count() + 1;
        let mut text = prefix.clone();
        text.push(' ');
        text.push_str(&body.text);
        body.text = text;
        for span in &mut body.spans {
            span.start += shift;
            span.end += shift;
        }
        for word in &mut body.words {
            word.start += shift;
            word.end += shift;
        }
        let lead = marker.text.chars().count() - marker.text.trim_start().chars().count();
        let mut marker_spans = marker.spans;
        for span in &mut marker_spans {
            span.start = span.start.saturating_sub(lead).min(prefix.chars().count());
            span.end = span.end.saturating_sub(lead).min(prefix.chars().count());
        }
        let mut marker_words = marker.words;
        for word in &mut marker_words {
            word.start = word.start.saturating_sub(lead).min(prefix.chars().count());
            word.end = word.end.saturating_sub(lead).min(prefix.chars().count());
        }
        marker_spans.append(&mut body.spans);
        body.spans = marker_spans;
        marker_words.append(&mut body.words);
        body.words = marker_words;
        body.bbox = union_bbox([marker.bbox, body.bbox]);
        removed.insert(m);
        renamed.insert(marker.id, body.id.clone());
    }
    for region in &mut page.regions {
        region.line_ids.retain(|id| !renamed.contains_key(id));
    }
    if let Some(id) = page.printed_label_line_id.as_ref().and_then(|id| renamed.get(id)) {
        page.printed_label_line_id = Some(id.clone());
    }
    let mut index = 0usize;
    page.lines.retain(|_| {
        let keep = !removed.contains(&index);
        index += 1;
        keep
    });
    removed.len()
}
