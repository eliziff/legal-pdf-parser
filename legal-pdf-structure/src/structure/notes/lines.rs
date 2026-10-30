//! What the note band reads of single lines: a notes heading, a label's typography, a reference
//! wrapped onto the next line, a reference above a label.

use super::*;

/// "Notes", "Footnotes", "Endnotes": a heading, capitalized; a lowercase "note" alone on its line
/// is a sentence's word wrapped ("... on the information or indictment, / note").
pub(crate) fn endnote_heading(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    let text = text.trim();
    text.starts_with(char::is_uppercase)
        && RE.get_or_init(|| Regex::new(r"(?i)^(?:end|foot)?notes?$").unwrap()).is_match(text)
}

pub(crate) fn continuing_note_heading(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)^(?:endnotes?|footnotes?|notes?)(?:\s*\(?(?:continued|cont'd)\)?)?$")
            .unwrap()
    })
    .is_match(text.trim())
}

pub(crate) fn structural_reset_heading(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)^(?:appendix|annex|schedule|part|chapter|bibliography|references|works\s+cited|table\s+of\s+authorities|index|acknowledg(?:e)?ments|certificate\s+of\s+service|about\s+the\s+authors?)(?:\s+[\w.-]+)?$",
        )
        .unwrap()
    })
    .is_match(text.trim())
}

pub(crate) fn citation_shaped_candidate(page: &Page, candidate: &(usize, LabelPrefix, bool)) -> bool {
    let line = &page.lines[candidate.0];
    let tail = scalar_suffix(&line.text, candidate.1.end);
    layout_analysis().is_citation_shaped_tail(tail)
}

pub(crate) fn longest_label_run(values: &[u32]) -> usize {
    let mut runs = vec![1; values.len()];
    for index in 0..values.len() {
        for prior in 0..index {
            if (1..=3).contains(&values[index].saturating_sub(values[prior])) {
                runs[index] = runs[index].max(runs[prior] + 1);
            }
        }
    }
    runs.into_iter().max().unwrap_or(0)
}

/// Whether one of `lines` above `label_y` refers to `label`: a raised number, a reference read apart
/// from its line, or the symbol ending a word.
pub(crate) fn has_prior_reference<'a>(lines: impl IntoIterator<Item = &'a Line>, label: &str, label_y: f64) -> bool {
    let matches_label = |value: &str| {
        let value = value.trim();
        value == label || normalize_label(value) == label
    };
    lines.into_iter().any(|line| {
        line.bbox[1] < label_y
            && (line
                .spans
                .iter()
                .any(|span| span.superscript && matches_label(&span.text))
                || line
                    .detached_references
                    .iter()
                    .any(|reference| matches_label(&reference.note_id))
                || (label.chars().all(is_note_symbol)
                    && line
                        .text
                        .trim_end()
                        .strip_suffix(label)
                        .and_then(|before| before.chars().next_back())
                        .is_some_and(char::is_alphanumeric)))
    })
}

pub(crate) fn label_spans<'a>(line: &'a Line, prefix: &'a LabelPrefix) -> impl Iterator<Item = &'a Span> {
    line.spans
        .iter()
        .filter(|span| span.start < prefix.end && span.end > prefix.start)
}

/// A label set above or smaller than the rest of its own line, as a note label is.
pub(crate) fn label_is_raised(line: &Line, prefix: &LabelPrefix, line_size: f64) -> bool {
    let height = line.bbox[3] - line.bbox[1];
    label_spans(line, prefix).any(|span| {
        span.superscript
            || (line_size > 0.0 && span.size > 0.0 && span.size <= line_size * 0.75)
            || (line_size > 0.0
                && span.size > 0.0
                && span.size * 1.25 <= line_size
                && height > 0.0
                && span.bbox[3] <= line.bbox[3] - 0.25 * height)
    })
}

/// Whether the line just above `index`, in its type and at its leading, ends on a counted
/// enactment's noun ("s", "c", "art") still waiting for its number.
pub(crate) fn reference_wrapped_onto(lines: &[Line], sizes: &[f64], index: usize) -> bool {
    let line = &lines[index];
    let height = line.bbox[3] - line.bbox[1];
    lines
        .iter()
        .zip(sizes)
        .filter(|(above, size)| {
            (**size - sizes[index]).abs() <= 0.5
                && above.bbox[3] <= line.bbox[1] + height * 0.5
                && line.bbox[1] - above.bbox[3] < height * 0.6
                && above.bbox[0] < line.bbox[2]
                && line.bbox[0] < above.bbox[2]
        })
        .max_by(|(left, _), (right, _)| left.bbox[3].total_cmp(&right.bbox[3]))
        .and_then(|(above, _)| above.text.split_whitespace().last())
        .is_some_and(|word| {
            matches!(
                word.trim_end_matches('.'),
                "c" | "ch" | "s" | "ss" | "art" | "arts"
            )
        })
}
