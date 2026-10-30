//! What a line's text and its place among its page's lines say of it as a heading reads it: a line
//! carrying a sentence on from the line above it, a full line whose sentence the next line
//! finishes, a word hyphenated or a sentence going on across two lines, a date alone, a page whose
//! text was recognized from its image, and the heading tags a source puts on so many lines that
//! they mark its body.

use super::*;

/// A line set at its paragraph's leading under a full line that ends no sentence carries that
/// sentence on ("... Hogg and" / "A. A. Bushell, ..."): its initial is no enumerator. A title above
/// it stops short of its column's edge. The page's body lines are those `body` names (slots of the
/// page's lines).
pub(crate) fn continues_prose_among(page: &Page, line: &Line, body: impl Fn(usize) -> bool) -> bool {
    let height = line.bbox[3] - line.bbox[1];
    let body_lines = || page.lines.iter().enumerate().filter(|(slot, _)| body(*slot)).map(|(_, line)| line);
    body_lines()
        .filter(|prior| {
            !prior.exclude_from_body
                && prior.bbox[3] <= line.bbox[1] + height * 0.25
                && prior.bbox[0] < line.bbox[2]
                && line.bbox[0] < prior.bbox[2]
        })
        .max_by(|left, right| left.bbox[3].total_cmp(&right.bbox[3]))
        .is_some_and(|prior| {
            let text = prior.text.trim_end();
            let (left, right) = body_lines()
                .filter(|other| {
                    other.bbox[0] < prior.bbox[2]
                        && prior.bbox[0] < other.bbox[2]
                })
                .fold((prior.bbox[0], prior.bbox[2]), |(left, right), other| {
                    (left.min(other.bbox[0]), right.max(other.bbox[2]))
                });
            height > 0.0
                && prior.bbox[2] >= right - (right - left) * 0.1
                && line.bbox[1] - prior.bbox[3] < height * 0.5
                && !sentence_ended(text)
                && !text.ends_with([':', '\u{201d}', '"'])
                && !enumerated(text)
        })
}

/// A long numbered line running the full column whose sentence the next line, at its leading,
/// finishes ("1. IAEA, Safety Standards Series, Geological Disposal Facilities for" /
/// "Radioactive Waste, Vienna, 2011.") opens a paragraph or an entry, not a heading: the line at
/// `slot` onto the line `next` (slots of the page's lines), the page's body lines those `body` names.
pub(crate) fn carries_a_sentence_onto(page: &Page, slot: usize, next: usize, body: impl Fn(usize) -> bool) -> bool {
    let line = &page.lines[slot];
    let height = line.bbox[3] - line.bbox[1];
    let (left, right) = page
        .lines
        .iter()
        .enumerate()
        .filter(|(other_slot, other)| {
            body(*other_slot)
                && other.bbox[0] < line.bbox[2]
                && line.bbox[0] < other.bbox[2]
        })
        .fold((line.bbox[0], line.bbox[2]), |(left, right), (_, other)| {
            (left.min(other.bbox[0]), right.max(other.bbox[2]))
        });
    let followed_by_body = body(next);
    let next = &page.lines[next];
    let text = line.text.trim_end();
    let following = next.text.trim_end();
    height > 0.0
        && text.split_whitespace().count() >= 8
        && line.bbox[2] >= right - (right - left) * 0.1
        && !sentence_ended(text)
        && !text.ends_with(':')
        && followed_by_body
        && next.bbox[1] > line.bbox[1]
        && next.bbox[1] - line.bbox[3] < height * 0.5
        && sentence_ended(following)
        && !following.ends_with('?')
        && !layout_analysis().has_citation_signal(&format!("{text} {following}"))
}

/// The share of a line's characters its face sets bold.
pub(crate) fn bold_char_share(line: &Line) -> f64 {
    flag_share(line, BOLD).unwrap_or(0.0)
}

pub(crate) fn body_flow_edge(previous: &Line, current: &Line) -> bool {
    let previous_text = previous.text.trim();
    let current_text = current.text.trim();
    if previous_text.is_empty() || current_text.is_empty() {
        return false;
    }
    if hyphen_fragment_tail(previous_text) && hyphen_continuation(current_text) {
        return true;
    }
    if starts_note_or_list(current_text) {
        return false;
    }
    if !sentence_ended(previous_text) && current_text.chars().next().is_some_and(char::is_lowercase)
    {
        return true;
    }
    bracket_excess(previous_text) > 0 && closes_before_opening(current_text)
}

/// A line that is only a date: "10 JUNE 2026", "DATED 11 JUNE 2026", "June 29, 2020", "1/4/2025".
pub(crate) fn date_line(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        let month = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|janvier|f[ée]vrier|mars|avril|mai|juin|juillet|ao[uû]t|septembre|octobre|novembre|d[ée]cembre)";
        Regex::new(&format!(
            r"(?i)^(?:dated\s+(?:this\s+)?|le\s+)?(?:\d{{1,2}}(?:st|nd|rd|th|er)?\s+(?:day\s+of\s+)?{month}\.?,?\s+\d{{4}}|{month}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}|\d{{1,2}}[/.-]\d{{1,2}}[/.-]\d{{2,4}})$"
        ))
        .unwrap()
    })
    .is_match(text.trim())
}

/// A page whose text was recognized from its image: lines with recognition evidence or no spans,
/// or a text layer set in one face throughout with no bold or italic (a recognizer writes one
/// generic face and sizes its boxes; a typeset page changes face or weight somewhere).
pub(crate) fn recognized_page(page: &Page) -> bool {
    let body: Vec<&Line> = page.lines.iter().filter(|line| !line.exclude_from_body).collect();
    if body.is_empty() {
        return false;
    }
    // Only a scan's lines are measured on its image.
    if body.iter().any(|line| line.ocr.is_some() || line.spans.is_empty() || line.cap_height > 0.0) {
        return true;
    }
    let mut fonts: HashSet<&str> = HashSet::new();
    let mut styled = false;
    for span in body.iter().flat_map(|line| &line.spans) {
        fonts.insert(span.font.as_str());
        styled |= span.flags & (16 | 2) != 0;
    }
    // A tab divider ("Tab" / "1") or a cover of two lines in one face says nothing about the layer.
    (body.len() >= 8 && fonts.len() == 1 && !styled) || off_grid_sizes(body.iter().copied()) >= 5
}

/// The heading tags a document puts on so many of its tagged lines that they mark its body (every
/// numbered paragraph tagged H2), as pdf-inspector reads them
/// (`detect_overused_struct_heading_levels`: a level on over 15% of 20 tagged lines or more): each
/// line's tag (`role`, by page slot and line slot) read on the pages' items (`page_items`).
pub(crate) fn overused_tags<'a>(
    pages: &[Page],
    items: &[legal_pdf_tables::PageItems],
    role: impl Fn(usize, usize) -> Option<&'a str>,
) -> HashSet<String> {
    use pdf_inspector::structure_tree::StructRole;
    use pdf_inspector::types::TextLine;
    let mut roles: HashMap<u32, HashMap<i64, StructRole>> = HashMap::new();
    let mut lines: Vec<TextLine> = Vec::new();
    for (page_slot, (page, items)) in pages.iter().zip(items).enumerate() {
        let number = page.index as u32 + 1;
        for (item, &(slot, _)) in items.items.iter().zip(&items.origin) {
            let Some(role) = role(page_slot, slot) else {
                continue;
            };
            let mut item = item.clone();
            item.mcid = Some(slot as i64);
            roles.entry(number).or_default().insert(slot as i64, StructRole::from_name(role));
            match lines.last_mut() {
                Some(line) if line.page == number && line.items[0].mcid == item.mcid => line.items.push(item),
                _ => lines.push(TextLine { items: vec![item], y: 0.0, page: number, adaptive_threshold: 0.1 }),
            }
        }
    }
    pdf_inspector::markdown::detect_overused_struct_heading_levels(&lines, Some(&roles))
        .into_iter()
        .flat_map(|level| [Some(format!("H{level}")), (level == 1).then(|| "H".to_owned())])
        .flatten()
        .collect()
}
