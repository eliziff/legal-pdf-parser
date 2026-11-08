//! Text painted twice at one place (a bold faked by overprinting, a shadow) read once.

use super::*;

pub(crate) fn same_painted_text(left: &TextItem, right: &TextItem) -> bool {
    let (Some(left_fidelity), Some(right_fidelity)) = (left.fidelity.as_ref(), right.fidelity.as_ref()) else {
        return false;
    };
    if left.page != right.page
        || left.text != right.text
        || left.text.is_empty()
        || left_fidelity.resource != right_fidelity.resource
        || left_fidelity.flags != right_fidelity.flags
        || left_fidelity.glyphs.len() != right_fidelity.glyphs.len()
    {
        return false;
    }
    let tolerance = (left.font_size.max(right.font_size) * 0.04).clamp(0.01, 0.5);
    let close = |left: &[f32], right: &[f32]| left.iter().zip(right).all(|(left, right)| (left - right).abs() <= tolerance);
    close(&left_fidelity.baseline, &right_fidelity.baseline)
        && close(&left_fidelity.advance, &right_fidelity.advance)
        && close(&left_fidelity.em, &right_fidelity.em)
        && left_fidelity.glyphs.iter().zip(&right_fidelity.glyphs).all(|(left, right)| left.text == right.text && close(&left.bbox, &right.bbox))
}

pub(crate) fn painted_glyph_exists_in(previous: &TextItem, current: &TextItem, glyph: &FidelityGlyph) -> bool {
    let (Some(previous_fidelity), Some(current_fidelity)) = (previous.fidelity.as_ref(), current.fidelity.as_ref()) else {
        return false;
    };
    if previous.page != current.page || previous_fidelity.resource != current_fidelity.resource || previous_fidelity.flags != current_fidelity.flags {
        return false;
    }
    let tolerance = previous.font_size.max(current.font_size).mul_add(0.02, 0.0).clamp(0.01, 0.25);
    previous_fidelity.glyphs.iter().any(|candidate| {
        candidate.text == glyph.text && (candidate.bbox[0] - glyph.bbox[0]).abs() <= tolerance && (candidate.bbox[1] - glyph.bbox[1]).abs() <= tolerance
    })
}

pub(crate) fn deduplicate_painted_text(items: &mut Vec<TextItem>) {
    items.dedup_by(|current, previous| {
        same_painted_text(previous, current)
            || current.fidelity.as_ref().is_some_and(|fidelity| {
                let mut visible = fidelity.glyphs.iter().filter(|glyph| !glyph.text.chars().all(char::is_whitespace));
                let first = visible.next();
                first.is_some_and(|glyph| painted_glyph_exists_in(previous, current, glyph)) && visible.all(|glyph| painted_glyph_exists_in(previous, current, glyph))
            })
    });
}
