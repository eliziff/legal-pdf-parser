//! Each item's line and order as the renderer lays them out, and its text normalized.

use super::*;

pub(crate) fn assign_renderer_layout(items: &mut [TextItem]) {
    const BASE_MAX_DIST: f64 = 0.8;
    const PARAGRAPH_DIST: f64 = 1.5;
    const SPACE_DIST: f64 = 0.15;
    const SPACE_MAX_DIST: f64 = 0.8;

    let mut page = None;
    let mut block = 0u32;
    let mut line = 0u32;
    let mut has_text_block = false;
    let mut pen = None::<[f64; 2]>;
    let mut prior_direction = None::<[f64; 2]>;
    let mut line_start = None::<[f64; 2]>;
    let mut line_leading_size = None::<f64>;
    let mut line_has_small_leading_run = false;

    for item in items {
        if page != Some(item.page) {
            page = Some(item.page);
            block = 0;
            line = 0;
            has_text_block = false;
            pen = None;
            prior_direction = None;
            line_start = None;
            line_leading_size = None;
            line_has_small_leading_run = false;
        }
        if matches!(&item.item_type, ItemType::Image) {
            block = block.saturating_add(1);
            has_text_block = false;
            pen = None;
            prior_direction = None;
            line_start = None;
            line_leading_size = None;
            line_has_small_leading_run = false;
            continue;
        }
        if !matches!(&item.item_type, ItemType::Text) {
            continue;
        }
        let Some(fidelity) = item.fidelity.as_ref() else {
            continue;
        };
        let point = [
            f64::from(fidelity.baseline[0]),
            f64::from(fidelity.baseline[1]),
        ];
        let advance = [
            f64::from(fidelity.advance[0]),
            f64::from(fidelity.advance[1]),
        ];
        let size = f64::from(fidelity.em[0].hypot(fidelity.em[1])).max(0.001);
        let advance_length = advance[0].hypot(advance[1]);
        let direction = if advance_length > 0.001 {
            [advance[0] / advance_length, advance[1] / advance_length]
        } else if let Some(direction) = prior_direction {
            direction
        } else {
            [
                f64::from(fidelity.em[1]) / size,
                -f64::from(fidelity.em[0]) / size,
            ]
        };
        let mut new_paragraph = !has_text_block;
        let mut new_line = pen.is_none();
        if let (Some(prior_pen), Some(prior_direction)) = (pen, prior_direction) {
            if direction[0] * prior_direction[0] + direction[1] * prior_direction[1] < 0.999 {
                new_paragraph = true;
                new_line = true;
            } else {
                let delta = [point[0] - prior_pen[0], point[1] - prior_pen[1]];
                let spacing = (direction[0] * delta[0] + direction[1] * delta[1]) / size;
                let base_offset = (-direction[1] * delta[0] + direction[0] * delta[1]) / size;
                if base_offset.abs() < BASE_MAX_DIST {
                    new_line = !(spacing.abs() < SPACE_DIST
                        || (-SPACE_MAX_DIST..0.0).contains(&spacing)
                        || (0.0..SPACE_MAX_DIST).contains(&spacing));
                } else if base_offset.abs() <= PARAGRAPH_DIST {
                    new_line = true;
                    if let Some(start) = line_start {
                        let indent = direction[0] * (point[0] - start[0])
                            + direction[1] * (point[1] - start[1]);
                        // A raised, smaller line-initial run is a label, not
                        // the paragraph's indentation anchor. Keep its
                        // hanging continuation in the same text block.
                        new_paragraph |= indent > 0.5 && !line_has_small_leading_run;
                    }
                } else {
                    new_paragraph = true;
                    new_line = true;
                }
            }
        }
        if new_paragraph {
            block = block.saturating_add(1);
            has_text_block = true;
        }
        if new_line {
            line = line.saturating_add(1);
            line_start = Some(point);
            line_leading_size = Some(size);
            line_has_small_leading_run = false;
        } else if line_leading_size.is_some_and(|leading| size >= leading * 1.2) {
            line_has_small_leading_run = true;
        }
        if let Some(fidelity) = item.fidelity.as_mut() {
            fidelity.renderer_line = line;
            fidelity.renderer_block = block;
        }
        pen = Some([point[0] + advance[0], point[1] + advance[1]]);
        prior_direction = Some(direction);
    }
}

pub(crate) fn item_baseline(item: &TextItem) -> f64 {
    item.fidelity
        .as_ref()
        .map_or(f64::from(item.y), |fidelity| {
            f64::from(item.y - fidelity.descender.min(0.0) * item.font_size)
        })
}

pub(crate) fn normalize_text(value: &str) -> Cow<'_, str> {
    if !value.contains('\u{feff}') && !value.contains('\u{200b}') {
        return Cow::Borrowed(value);
    }
    let mut result = String::with_capacity(value.len());
    let mut characters = value
        .chars()
        .filter(|character| *character != '\u{feff}')
        .peekable();
    while let Some(character) = characters.next() {
        if character != '\u{200b}' {
            result.push(character);
            continue;
        }
        let mut count = 1;
        while characters.peek().is_some_and(|next| *next == '\u{200b}') {
            characters.next();
            count += 1;
        }
        if count >= 2 {
            result.push(' ');
        }
    }
    Cow::Owned(result)
}
