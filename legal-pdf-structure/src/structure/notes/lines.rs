//! What the note band reads of single lines (a notes heading, a label's typography, a reference
//! wrapped onto the next line), and the endnotes a document's body refers to.

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

pub(crate) fn has_prior_reference(page: &Page, label: &str, label_y: f64) -> bool {
    let matches_label = |value: &str| {
        let value = value.trim();
        value == label || normalize_label(value) == label
    };
    page.lines.iter().any(|line| {
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

/// Endnotes are notes: the body before them refers to them. Each run of consecutive pages read as
/// endnotes keeps them only where a notes heading opens it, or where references earlier in the
/// document (raised numbers, detached references) name at least half of its note numbers; else
/// its lines are the body's, a numbered sequence the page-by-page reading took for notes. On the
/// structure gold (batches 5-7) endnote runs held 126 note lines and 5,472 lines of body; the
/// references keep 49 of the notes and give back 5,429 of the body.
pub(crate) fn verify_endnotes(pages: &mut [Page]) {
    let endnote = |page: &Page| page.lines.iter().any(|line| line.note_region_mode == "endnote");
    let mut slot = 0;
    while slot < pages.len() {
        if !endnote(&pages[slot]) {
            slot += 1;
            continue;
        }
        let mut end = slot + 1;
        while end < pages.len() && endnote(&pages[end]) && pages[end].index == pages[end - 1].index + 1 {
            end += 1;
        }
        let run = &pages[slot..end];
        let headed = run.iter().flat_map(|page| &page.lines).any(|line| endnote_heading(&line.text));
        let labels: BTreeSet<String> = run
            .iter()
            .flat_map(|page| &page.lines)
            .filter(|line| line.note_region_mode == "endnote")
            .filter_map(|line| label_prefix(&line.text).map(|prefix| prefix.label))
            .collect();
        let first_note = pages[slot]
            .lines
            .iter()
            .filter(|line| line.note_region_mode == "endnote")
            .map(|line| line.bbox[1])
            .min_by(f64::total_cmp)
            .unwrap_or(0.0);
        let referenced = labels
            .iter()
            .filter(|label| {
                pages[..slot].iter().any(|page| has_prior_reference(page, label, f64::INFINITY))
                    || has_prior_reference(&pages[slot], label, first_note)
            })
            .count();
        if !headed && referenced * 2 < labels.len().max(1) {
            for line in pages[slot..end].iter_mut().flat_map(|page| &mut page.lines) {
                if line.note_region_mode == "endnote" {
                    line.region_type = "body".to_owned();
                    line.note_region_mode.clear();
                }
            }
        }
        slot = end;
    }
}

pub(crate) fn infer_note_region_modes(pages: &mut [Page]) {
    let mut prior_note_page = false;
    let mut active_endnotes = false;
    let mut expected_endnote = None::<u32>;
    let mut previous_page = None;
    for page in pages {
        if previous_page.is_some_and(|prior: usize| prior.checked_add(1) != Some(page.index)) {
            prior_note_page = false;
            active_endnotes = false;
            expected_endnote = None;
        }
        previous_page = Some(page.index);
        let footnote_indexes: Vec<usize> = page
            .lines
            .iter()
            .enumerate()
            .filter_map(|(index, line)| (line.region_type == "footnote").then_some(index))
            .collect();
        let heading = page.lines.iter().any(|line| endnote_heading(&line.text));
        let explicit_endnote = footnote_indexes
            .iter()
            .any(|&index| page.lines[index].note_region_mode == "endnote");
        let numbers: Vec<u32> = footnote_indexes
            .iter()
            .filter_map(|&index| line_start_label_prefix(&page.lines[index].text))
            .filter_map(|prefix| normalize_label(&prefix.label).parse().ok())
            .collect();
        let continues_endnotes = active_endnotes
            && (numbers.is_empty()
                || expected_endnote.is_some_and(|expected| numbers[0] == expected));
        if !footnote_indexes.is_empty() && (heading || explicit_endnote || continues_endnotes) {
            for &index in &footnote_indexes {
                page.lines[index].note_region_mode = "endnote".to_owned();
            }
            active_endnotes = true;
            if let Some(number) = numbers.last() {
                expected_endnote = number.checked_add(1);
            }
        } else if !footnote_indexes.is_empty() {
            let explicit_footnote = footnote_indexes
                .iter()
                .any(|&index| page.lines[index].note_region_mode == "footnote");
            let mode = if explicit_footnote {
                "footnote"
            } else if prior_note_page {
                "footnote_continuation"
            } else {
                ""
            };
            if !mode.is_empty() {
                for &index in &footnote_indexes {
                    if page.lines[index].note_region_mode.is_empty() {
                        page.lines[index].note_region_mode = mode.to_owned();
                    }
                }
            }
            active_endnotes = false;
            expected_endnote = None;
        } else {
            active_endnotes = false;
            expected_endnote = None;
        }
        prior_note_page = !footnote_indexes.is_empty();
    }
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

pub(crate) fn label_is_typographic(line: &Line, prefix: &LabelPrefix, line_size: f64, body_size: f64) -> bool {
    let label_size = label_spans(line, prefix)
        .filter_map(|span| (span.size > 0.0).then_some(span.size))
        .min_by(f64::total_cmp)
        .unwrap_or(line_size);
    label_is_raised(line, prefix, line_size) || (label_size > 0.0 && label_size <= body_size * 0.75)
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

/// Text-Fidelity's native-PDF body-font estimator: the largest rounded font
/// bucket carrying at least ten percent of the supplied character mass. Smaller
/// quote/note fonts can be common without redefining the body face, while
/// headings normally lack enough mass to become a substantial bucket.
pub(crate) fn body_font_size<'a>(lines: impl IntoIterator<Item = &'a Line>) -> f64 {
    let mut mass = HashMap::<i64, usize>::new();
    for line in lines {
        let size = line_font_size(line);
        if size > 0.0 {
            *mass.entry((size * 100.0).round() as i64).or_default() += line.text.chars().count();
        }
    }
    let total: usize = mass.values().sum();
    mass.into_iter()
        .filter(|(_, chars)| total > 0 && chars.saturating_mul(10) >= total)
        .map(|(size, _)| size as f64 / 100.0)
        .max_by(f64::total_cmp)
        .unwrap_or(0.0)
}
