//! A line assembled from its items: its spans (with raised note labels), glyphs, words and type size.

use super::*;

#[derive(Debug)]
pub(crate) struct AssembledSpan {
    pub(crate) item_index: usize,
    pub(crate) bbox_start_item: usize,
    pub(crate) bbox_end_item: usize,
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) byte_start: usize,
    pub(crate) byte_end: usize,
}

#[derive(Debug)]
pub(crate) struct AssembledLine {
    pub(crate) text: String,
    pub(crate) spans: Vec<AssembledSpan>,
}

pub(crate) fn same_renderer_span(left: &TextItem, right: &TextItem) -> bool {
    match (left.fidelity.as_ref(), right.fidelity.as_ref()) {
        (Some(left), Some(right)) => left.resource == right.resource && left.flags == right.flags && left.text_rise == right.text_rise && left.em == right.em,
        _ => left.font == right.font && left.font_size == right.font_size && left.is_bold == right.is_bold && left.is_italic == right.is_italic,
    }
}

pub(crate) fn same_source_text_object(left: &TextItem, right: &TextItem) -> bool {
    match (left.fidelity.as_ref(), right.fidelity.as_ref()) {
        (Some(left), Some(right)) => left.text_object == right.text_object,
        _ => true,
    }
}

/// Assemble the legal line contract from source spans. This deliberately does
/// not use `TextLine::text()`: its generic display heuristic suppresses spaces
/// around raised/small text, while the previous engine preserved an explicit
/// boundary whenever the PDF supplied one or the geometric gap reached 0.15em.
pub(crate) fn assemble_line(text_line: &TextLine) -> AssembledLine {
    let (mut raw_text, mut spans, mut offset) = (String::new(), Vec::<AssembledSpan>::new(), 0);
    let (mut previous_x1, mut previous_trailing_boundary) = (None, false);
    let mut pending_leading_whitespace = None::<(usize, usize, usize)>;
    let mut next_content = text_line.items.iter().enumerate().filter(|(_, item)| !item.text.trim().is_empty()).peekable();
    let mut previous_content = None;
    // A zero-width space at the item's start or end, a byte-order mark aside: a word boundary the PDF supplied.
    fn boundary(mut characters: impl Iterator<Item = char>) -> bool {
        characters.find(|&character| character != '\u{feff}') == Some('\u{200b}')
    }
    for (item_index, item) in text_line.items.iter().enumerate() {
        if item.text.trim().is_empty() {
            let previous = previous_content.map(|index| &text_line.items[index]);
            let next = next_content.peek().map(|(_, item)| *item);
            let extends_previous = previous.is_some_and(|candidate| same_renderer_span(candidate, item));
            if extends_previous || next.is_some_and(|candidate| same_renderer_span(candidate, item)) {
                let (start, byte_start) = (offset, raw_text.len());
                let whitespace = normalize_text(&item.text);
                raw_text.push_str(&whitespace);
                offset += whitespace.chars().count();
                if !extends_previous {
                    pending_leading_whitespace.get_or_insert((start, byte_start, item_index));
                } else if let Some(previous) = spans.last_mut() {
                    (previous.end, previous.byte_end, previous.bbox_end_item) = (offset, raw_text.len(), item_index);
                }
            }
            continue;
        }
        let _ = next_content.next();
        previous_content = Some(item_index);
        let (leading_boundary, trailing_boundary) = (boundary(item.text.chars()), boundary(item.text.chars().rev()));
        let span_text = normalize_text(&item.text);
        if span_text.is_empty() {
            previous_trailing_boundary |= trailing_boundary;
            continue;
        }
        let (x0, x1) = (round3(f64::from(item.x)), round3(f64::from(item.x + item.width)));
        let threshold = f64::from(item.font_size).max(10.0) * 0.15;
        if previous_x1.is_some_and(|prior| {
            !raw_text.chars().next_back().is_some_and(char::is_whitespace)
                && !span_text.chars().next().is_some_and(char::is_whitespace)
                && ((previous_trailing_boundary && leading_boundary) || x0 - prior >= threshold)
        }) {
            raw_text.push(' ');
            offset += 1;
        }
        previous_x1 = Some(x1);
        previous_trailing_boundary = trailing_boundary;
        let (start, byte_start, bbox_start_item) = pending_leading_whitespace.take().unwrap_or((offset, raw_text.len(), item_index));
        raw_text.push_str(&span_text);
        offset += span_text.chars().count();
        spans.push(AssembledSpan { item_index, bbox_start_item, bbox_end_item: item_index, start, end: offset, byte_start, byte_end: raw_text.len() });
    }
    let leading_bytes = raw_text.len() - raw_text.trim_start().len();
    let leading = raw_text[..leading_bytes].chars().count();
    let trimmed_end = raw_text.trim_end().len().max(leading_bytes);
    raw_text.truncate(trimmed_end);
    raw_text.replace_range(..leading_bytes, "");
    let (text_len, byte_len) = (raw_text.chars().count(), raw_text.len());
    for span in &mut spans {
        span.start = span.start.saturating_sub(leading).min(text_len);
        span.end = span.end.saturating_sub(leading).max(span.start).min(text_len);
        span.byte_start = span.byte_start.saturating_sub(leading_bytes).min(byte_len);
        span.byte_end = span.byte_end.saturating_sub(leading_bytes).max(span.byte_start).min(byte_len);
    }
    spans.retain(|span| span.start < span.end);
    AssembledLine { text: raw_text, spans }
}

pub(crate) fn is_superscript(item: &TextItem, body_size: f64, baseline: f64, inline: bool) -> bool {
    let literal = item.text.trim().chars().all(|character| matches!(character, '⁰' | '¹' | '²' | '³' | '⁴' | '⁵' | '⁶' | '⁷' | '⁸' | '⁹'));
    literal
        || (inline
            && body_size > 0.0
            && item.font_size > 0.0
            // A producer may raise a script without appreciably shrinking its type.
            && item_baseline(item) >= baseline + (f64::from(item.font_size) * 0.1).max(0.5))
}

pub(crate) fn line_spans(text_line: &TextLine, assembled: &AssembledLine, page_height: f64, id: &str) -> Vec<Span> {
    let written = |item: &&TextItem| !item.text.trim().is_empty();
    let body_size = text_line.items.iter().filter(|item| item.font_size > 0.0 && written(item)).map(|item| f64::from(item.font_size)).max_by(f64::total_cmp).unwrap_or(0.0);
    let baseline = median(
        text_line.items.iter().filter(|item| written(item) && (body_size <= 0.0 || f64::from(item.font_size) >= body_size * 0.85)).map(item_baseline).collect(),
    );
    let (mut spans, mut previous_item_index, mut previous_byte_end) = (Vec::<Span>::new(), None::<usize>, 0);
    for source_span in &assembled.spans {
        let item = &text_line.items[source_span.item_index];
        let (start_byte, end_byte) = (source_span.byte_start, source_span.byte_end);
        let bbox = union_bbox(text_line.items[source_span.bbox_start_item..=source_span.bbox_end_item].iter().map(|item| item_bbox(item, page_height)));
        let superscript = is_superscript(item, body_size, baseline, assembled.spans.len() > 1);
        if let (Some(previous), Some(prior_item_index)) = (spans.last_mut(), previous_item_index) {
            let prior = &text_line.items[prior_item_index];
            if same_renderer_span(prior, item)
                && previous.superscript == superscript
                && prior.mcid == item.mcid
                && same_source_text_object(prior, item)
                && assembled.text[previous_byte_end..start_byte].chars().all(char::is_whitespace)
            {
                previous.text.push_str(&assembled.text[previous_byte_end..end_byte]);
                previous.end = source_span.end;
                previous.bbox = union_bbox([previous.bbox, bbox]);
                (previous_item_index, previous_byte_end) = (Some(source_span.item_index), end_byte);
                continue;
            }
        }
        let styled = item.fidelity.as_ref().map_or_else(|| if item.is_italic { 2 } else { 0 } | if item.is_bold { 16 } else { 0 }, |fidelity| fidelity.flags);
        spans.push(Span {
            mcid: item.mcid,
            structure_role: None,
            id: format!("{id}-s{:03}", spans.len() + 1),
            text: assembled.text[start_byte..end_byte].to_owned(),
            bbox,
            font: item.font.clone(),
            size: round3(f64::from(item.font_size)),
            // Invisible text is a recognition layer behind a scan, not glyphs as printed.
            flags: styled | u32::from(superscript) | if item.render_mode == Some(3) { legal_pdf_core::model::INVISIBLE } else { 0 },
            superscript,
            start: source_span.start,
            end: source_span.end,
        });
        (previous_item_index, previous_byte_end) = (Some(source_span.item_index), end_byte);
    }
    spans
}

#[derive(Debug)]
pub(crate) struct PositionedGlyph {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) bbox: [f64; 4],
    /// Its run is underlined, as pdf-inspector reads the page's rules.
    pub(crate) underlined: bool,
    /// Its run is struck through, as pdf-inspector reads the page's rules.
    pub(crate) struck: bool,
}

pub(crate) fn line_glyphs(text_line: &TextLine, assembled: &AssembledLine, page_height: f64) -> Vec<PositionedGlyph> {
    let mut result = Vec::new();
    for source_span in &assembled.spans {
        let item = &text_line.items[source_span.item_index];
        let Some(fidelity) = item.fidelity.as_ref() else {
            continue;
        };
        // A span also owns the separate whitespace items beside its text, which
        // Word writes as text objects of their own: find only the item's text.
        let target = &assembled.text[source_span.byte_start..source_span.byte_end];
        let content = target.trim_start();
        let lead = target[..target.len() - content.len()].chars().count();
        let item_text = normalize_text(&item.text);
        let Some(target_byte) = item_text.find(content.trim_end()) else {
            continue;
        };
        let (skipped, base) = (item_text[..target_byte].chars().count(), source_span.start + lead);
        let mut local_offset = 0usize;
        for glyph in &fidelity.glyphs {
            let glyph_text = normalize_text(&glyph.text);
            let glyph_len = glyph_text.chars().count();
            let local_start = local_offset;
            local_offset += glyph_len;
            if glyph_len == 0 || glyph_text.chars().all(char::is_whitespace) {
                continue;
            }
            let start = base.saturating_add(local_start.saturating_sub(skipped));
            let end = base.saturating_add(local_offset.saturating_sub(skipped)).min(source_span.end);
            if start >= end || local_offset <= skipped {
                continue;
            }
            let [x0, y0, x1, y1] = glyph.bbox.map(f64::from);
            let bbox = [round3(x0), round3(page_height - y1), round3(x1), round3(page_height - y0)];
            result.push(PositionedGlyph { start, end, bbox, underlined: item.is_underline, struck: item.is_strikeout });
        }
    }
    result
}

pub(crate) fn line_words(text: &str, text_line: &TextLine, assembled: &AssembledLine, page_height: f64, id: &str) -> Vec<Word> {
    let glyphs = line_glyphs(text_line, assembled, page_height);
    if glyphs.is_empty() {
        return Vec::new();
    }
    debug_assert!(glyphs.windows(2).all(|pair| pair[0].start <= pair[1].start && pair[0].end <= pair[1].end));
    let (mut words, mut start, mut glyph_index) = (Vec::new(), None, 0);
    for (char_offset, (byte, character)) in text.char_indices().chain(std::iter::once((text.len(), ' '))).enumerate() {
        if !character.is_whitespace() {
            start = start.or(Some((byte, char_offset)));
            continue;
        }
        let Some((begin, start)) = start.take() else {
            continue;
        };
        let end = char_offset;
        while glyphs.get(glyph_index).is_some_and(|glyph| glyph.end <= start) {
            glyph_index += 1;
        }
        let covered: Vec<&PositionedGlyph> = glyphs[glyph_index..].iter().take_while(|glyph| glyph.start < end).collect();
        let Some(first) = covered.first().map(|glyph| glyph.bbox) else {
            return Vec::new();
        };
        let most = |marked: &dyn Fn(&PositionedGlyph) -> bool| covered.iter().filter(|glyph| marked(glyph)).count() * 2 > covered.len();
        words.push(Word {
            id: format!("{id}-w{:03}", words.len() + 1),
            text: text[begin..byte].to_owned(),
            bbox: union_bbox(std::iter::once(first).chain(covered[1..].iter().map(|glyph| glyph.bbox))),
            start,
            end,
            underlined: most(&|glyph| glyph.underlined),
            struck: most(&|glyph| glyph.struck),
            weight: 0.0,
        });
    }
    words
}

pub(crate) fn model_line_font_size(line: &Line) -> f64 {
    let has_body = line.spans.iter().any(|span| span.size > 0.0 && !span.superscript);
    let mut sizes: Vec<_> = line
        .spans
        .iter()
        .filter(|span| span.size > 0.0 && (!has_body || !span.superscript))
        .map(|span| (span.size, span.text.chars().count().clamp(1, 100)))
        .collect();
    if sizes.is_empty() {
        return 0.0;
    }
    sizes.sort_by(|left, right| left.0.total_cmp(&right.0));
    let count = sizes.iter().map(|(_, count)| count).sum::<usize>();
    let (left_index, right_index) = ((count - 1) / 2, count / 2);
    let (mut seen, mut left_value) = (0, None);
    for (size, count) in sizes {
        seen += count;
        if left_value.is_none() && seen > left_index {
            left_value = Some(size);
        }
        if seen > right_index {
            return (left_value.unwrap_or(size) + size) / 2.0;
        }
    }
    unreachable!("weighted median has at least one value")
}

pub(crate) fn begins_with_note_label(text: &str) -> bool {
    let trimmed = text.trim_start();
    let count = trimmed.chars().take_while(|character| character.is_ascii_digit()).count();
    if count == 0 || count > 3 {
        return matches!(trimmed.chars().next(), Some('*' | '\u{2217}' | '\u{f02a}' | '\u{2020}' | '\u{2021}'));
    }
    trimmed.chars().nth(count).is_none_or(|character| character.is_whitespace() || ".)]},:;-".contains(character))
}
