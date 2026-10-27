//! The headings each page's own readers read: a source's heading tags (demoted where they tag
//! prose), the numbered headings the ladder levels and the lines their titles wrap onto, and on a
//! recognized page a title-case line set apart.

use super::*;

pub(crate) fn heading_source_eligible(primitives: &PdfPrimitiveEvidence, line: &Line) -> bool {
    !primitives.claims.claimed(&line.id)
        // A blank to fill in ("CACR ____", "DATED ____") makes the line a form's field.
        && !line.text.contains("___")
        // Source labels constrain inference when available; they are not required to read a heading.
        && primitives
            .source_regions
            .as_ref()
            .and_then(|regions| regions.get(&line.id))
            .is_none_or(|region| {
                matches!(region.as_str(), "text" | "body")
                    || legal_pdf_core::is_heading_region(region)
            })
}

pub(crate) struct HeadingCandidate {
    page_slot: usize,
    line_slot: usize,
    joined_line_slot: Option<usize>,
    interpretations: Vec<EnumeratorInterpretation>,
}

/// A title set apart in bold may name a case by its parties; one in body type may not.
pub(crate) fn heading_title_plausible(line: &Line, text: &str) -> bool {
    // A title has a word; "F. R." is a name's initials.
    let worded = text.split_whitespace().any(|word| {
        word.chars()
            .filter(|character| character.is_alphabetic())
            .count()
            >= 2
    });
    if !worded || has_dot_leader(text) {
        false
    } else if bold_char_share(line) >= 0.60 {
        layout_analysis().styled_heading_text_plausible(text)
    } else {
        layout_analysis().heading_text_plausible(text)
    }
}

pub(crate) struct HeadingDecision {
    page_slot: usize,
    line_slot: usize,
    joined_line_slot: Option<usize>,
    level: Option<usize>,
}

pub(crate) fn heading_candidates(pages: &[Page], primitives: &PdfPrimitiveEvidence) -> Vec<HeadingCandidate> {
    let inline = heading_inline_re();
    let standalone = heading_standalone_re();
    let mut candidates = Vec::new();
    for (page_slot, page) in pages.iter().enumerate() {
        if primitives.claims.contents_pages.contains(&page.index) {
            continue;
        }
        for (line_slot, line) in page.lines.iter().enumerate().filter(|(_, line)| {
            !line.exclude_from_body
                && matches!(line.region_type.as_str(), "body" | "heading")
                && heading_source_eligible(primitives, line)
        }) {
            if continues_prose(page, line) {
                continue;
            }
            let text = line.text.trim();
            if (1..=7).contains(&text.len())
                && text.bytes().all(|byte| b"IVXLCDM".contains(&byte))
                && bold_char_share(line) >= 0.60
            {
                let interpretations = enumerator_interpretations(text, "")
                    .into_iter()
                    .filter(|(family, _, _)| family == "roman_")
                    .collect::<Vec<_>>();
                if !interpretations.is_empty() {
                    candidates.push(HeadingCandidate {
                        page_slot,
                        line_slot,
                        joined_line_slot: None,
                        interpretations,
                    });
                }
                continue;
            }
            if let Some(capture) = inline.captures(line.text.trim()) {
                let (value, punct, text) = heading_enumerator(&capture);
                let interpretations = enumerator_interpretations(value, punct);
                // A line opening "v. Party, 2016 ONCA 409" carries a style of cause on; a
                // fifth title ("v. The Rule of Lenity") cites nothing.
                let versus = value == "v" && punct == "." && layout_analysis().has_citation_signal(text);
                if !versus
                    && !carries_a_sentence_on(page, line_slot)
                    && heading_title_plausible(line, text)
                    && !interpretations.is_empty()
                {
                    candidates.push(HeadingCandidate {
                        page_slot,
                        line_slot,
                        joined_line_slot: None,
                        interpretations,
                    });
                }
                continue;
            }
            let Some(capture) = standalone.captures(line.text.trim()) else {
                continue;
            };
            let follower = ((line_slot + 1)..(line_slot + 3).min(page.lines.len())).find(|index| {
                let line = &page.lines[*index];
                !line.text.trim().is_empty()
                    && !line.exclude_from_body
                    && primitives.contexts.translation_line_ids.contains(&line.id)
                        == primitives
                            .contexts.translation_line_ids
                            .contains(&page.lines[line_slot].id)
                    && matches!(line.region_type.as_str(), "body" | "heading")
                    && heading_source_eligible(primitives, line)
            });
            let Some(follower_slot) = follower else {
                continue;
            };
            let text = page.lines[follower_slot]
                .text
                .trim()
                .trim_start_matches(['-', '\u{2013}', '\u{2014}'])
                .trim_start();
            let (value, punct, _) = heading_enumerator(&capture);
            // "v." alone between two parties' names is the style of cause's versus.
            let versus = value == "v" && punct == ".";
            let interpretations = enumerator_interpretations(value, punct);
            if !(!versus
                && heading_title_plausible(&page.lines[follower_slot], text)
                && !interpretations.is_empty())
            {
                heading_source_trace(&page.lines[follower_slot], if interpretations.is_empty() { "standalone-noint" } else { "standalone-implausible" });
            }
            if !versus
                && heading_title_plausible(&page.lines[follower_slot], text)
                && !interpretations.is_empty()
            {
                candidates.push(HeadingCandidate {
                    page_slot,
                    line_slot,
                    joined_line_slot: Some(follower_slot),
                    interpretations,
                });
            }
        }
    }
    candidates
}

/// A line set at its paragraph's leading under a full line that ends no sentence carries
/// that sentence on ("... Hogg and" / "A. A. Bushell, ..."): its initial is no enumerator.
/// A title above it stops short of its column's edge.
pub(crate) fn continues_prose(page: &Page, line: &Line) -> bool {
    let height = line.bbox[3] - line.bbox[1];
    page.lines
        .iter()
        .filter(|prior| {
            prior.region_type == "body"
                && !prior.exclude_from_body
                && prior.bbox[3] <= line.bbox[1] + height * 0.25
                && prior.bbox[0] < line.bbox[2]
                && line.bbox[0] < prior.bbox[2]
        })
        .max_by(|left, right| left.bbox[3].total_cmp(&right.bbox[3]))
        .is_some_and(|prior| {
            let text = prior.text.trim_end();
            let (left, right) = page
                .lines
                .iter()
                .filter(|other| {
                    other.region_type == "body"
                        && other.bbox[0] < prior.bbox[2]
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
/// "Radioactive Waste, Vienna, 2011.") opens a paragraph or an entry, not a heading.
pub(crate) fn carries_a_sentence_on(page: &Page, slot: usize) -> bool {
    let line = &page.lines[slot];
    let Some(next) = page.lines.get(slot + 1) else {
        return false;
    };
    let height = line.bbox[3] - line.bbox[1];
    let (left, right) = page
        .lines
        .iter()
        .filter(|other| {
            other.region_type == "body"
                && other.bbox[0] < line.bbox[2]
                && line.bbox[0] < other.bbox[2]
        })
        .fold((line.bbox[0], line.bbox[2]), |(left, right), other| {
            (left.min(other.bbox[0]), right.max(other.bbox[2]))
        });
    let text = line.text.trim_end();
    let following = next.text.trim_end();
    height > 0.0
        && text.split_whitespace().count() >= 8
        && line.bbox[2] >= right - (right - left) * 0.1
        && !sentence_ended(text)
        && !text.ends_with(':')
        && next.region_type == "body"
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

/// The lines an enumerated heading's title wraps onto: set in its type, flush under it or centred
/// beneath it, at its leading, with the body resuming further down. A title can run to
/// several lines, end in a question, and name a decision ("... 2021 ABCA 273, 30" /
/// "Alta. L.R. (7th) 1 (...)") when the whole title still reads as one.
///
/// A line in capitals that no enumerator announces (`capitals`) is a title only in words, at
/// least eight letters of them, and continues only in capitals and only on evidence that the
/// next line is its continuation (`continues_capitals_title`): a stamp ("CMD 25-M35#"), a date
/// under a title, or names and addresses stacked line by line are no wrapped title.
pub(crate) fn wrapped_heading_continuations(
    page: &Page,
    heading_slot: usize,
    structural: &HashSet<(usize, usize)>,
    page_slot: usize,
    primitives: &PdfPrimitiveEvidence,
    capitals: bool,
) -> Vec<usize> {
    const MAX_WRAPPED_LINES: usize = 3;
    let heading = &page.lines[heading_slot];
    let heading_text = heading.text.trim();
    let letters = heading_text
        .chars()
        .filter(|character| character.is_alphabetic())
        .count();
    let caps_head = (capitals || heading_text.split_whitespace().count() >= 2)
        && letters >= 8
        && !heading_text.chars().any(char::is_lowercase)
        && !heading_text.ends_with(['.', '?', '!', ';', ':']);
    if !has_valid_bbox(heading) || capitals && !caps_head {
        return Vec::new();
    }
    let mut title = title_after_enumerator(heading_text).to_owned();
    let heading_size = line_font_size(heading);
    let heading_height = (heading.bbox[3] - heading.bbox[1]).max(1.0);
    let centre_tolerance = page.width * FURNITURE_X_TOLERANCE_FRAC;
    let mut wrapped = Vec::new();
    let mut previous_slot = heading_slot;
    let mut previous = heading;
    let native_role = |line: &Line| {
        primitives
            .source_regions
            .as_ref()
            .and_then(|regions| regions.get(&line.id))
            .filter(|role| legal_pdf_core::is_heading_role(role))
    };
    for slot in heading_slot + 1..page.lines.len().min(heading_slot + 1 + MAX_WRAPPED_LINES) {
        let continuation = &page.lines[slot];
        let continuation_text = continuation.text.trim();
        let continuation_size = line_font_size(continuation);
        let continuation_height = (continuation.bbox[3] - continuation.bbox[1]).max(1.0);
        let x0_delta = continuation.bbox[0] - heading.bbox[0];
        let internal_gap = continuation.bbox[1] - previous.bbox[3];
        let joined = format!("{title} {continuation_text}");
        let cited = layout_analysis().has_citation_signal(&joined);
        let same_size = if heading_size > 0.0 && continuation_size > 0.0 {
            (continuation_size - heading_size).abs() <= (heading_size * 0.02).max(0.1)
        } else {
            // Without font spans, the measured line heights below are the size witness.
            heading.spans.is_empty() && continuation.spans.is_empty()
        };
        let continues = !structural.contains(&(page_slot, slot))
            && primitives.contexts.translation_line_ids.contains(&heading.id)
                == primitives.contexts.translation_line_ids.contains(&continuation.id)
            && native_role(heading) == native_role(continuation)
            && !continuation.exclude_from_body
            && !primitives.claims.claimed(&continuation.id)
            && primitives
                .source_regions
                .as_ref()
                .and_then(|regions| regions.get(&continuation.id))
                .is_none_or(|region| {
                    matches!(region.as_str(), "text" | "body" | "block_quote")
                        || legal_pdf_core::is_heading_region(region)
                })
            && matches!(continuation.region_type.as_str(), "body" | "heading")
            && (1..=12).contains(&continuation_text.split_whitespace().count())
            // A title's line opens on a word; a caption's bracket column (")", "§") is no title.
            && continuation_text.starts_with(|character: char| {
                character.is_alphanumeric() || "(['\"\u{2018}\u{201c}".contains(character)
            })
            && !(caps_head && continuation_text.chars().any(char::is_lowercase))
            && (!sentence_ended(continuation_text) || continuation_text.ends_with('?') || cited)
            && !starts_note_or_list(continuation_text)
            && (!cited || heading_title_plausible(heading, &joined))
            && has_valid_bbox(continuation)
            && same_size
            && (bold_char_share(continuation) - bold_char_share(heading)).abs() <= 0.1
            && (continuation_height - heading_height).abs() <= heading_height * 0.05
            && ((-3.0..=48.0_f64.max(heading_height * 1.75)).contains(&x0_delta)
                || (line_center_x(continuation) - line_center_x(heading)).abs()
                    <= centre_tolerance)
            && (-heading_height * 0.2..=(heading_height * 0.8).max(6.0)).contains(&internal_gap)
            && (!capitals
                || continues_capitals_title(page, heading_slot, previous_slot, slot, &title));
        if !continues {
            break;
        }
        wrapped.push(slot);
        title = joined;
        previous_slot = slot;
        previous = continuation;
        if continuation_text.ends_with('?') {
            break;
        }
    }
    // The body resumes below the title's last line, further than the title's own leading, or
    // in lower case under a title in capitals.
    while let Some(&last) = wrapped.last() {
        let line = &page.lines[last];
        let above = if last == heading_slot + 1 {
            heading
        } else {
            &page.lines[last - 1]
        };
        let resumed = page.lines.get(last + 1).is_some_and(|following| {
            following.region_type == "body"
                && !following.exclude_from_body
                && !primitives.claims.claimed(&following.id)
                && primitives
                    .source_regions
                    .as_ref()
                    .and_then(|regions| regions.get(&following.id))
                    .is_none_or(|region| matches!(region.as_str(), "text" | "body"))
                && has_valid_bbox(following)
                && (following.bbox[1] - line.bbox[1]
                    >= line.bbox[1] - above.bbox[1]
                        + ((line.bbox[3] - line.bbox[1]) * 0.5).max(8.0)
                    || caps_head && following.text.chars().any(char::is_lowercase))
        });
        if resumed {
            break;
        }
        wrapped.pop();
    }
    wrapped
}

/// Whether the line at `slot` carries on the title in capitals that ends at `above_slot`. Lines
/// stacked at one leading are each complete (counsel's names, an address, a date under a title)
/// unless the words run on across the break ("... COURT OF APPEAL FOR" / "ALBERTA"), the line
/// above was full (its next word would not fit within the title's or the page's widest line),
/// the lines are centred on one axis as a caption is ("IN THE UNITED STATES DISTRICT COURT" /
/// "FOR THE SOUTHERN DISTRICT OF TEXAS" / "HOUSTON DIVISION"), or the title closes at the line
/// and stands apart: what precedes it and what follows the line are set further off than its
/// leading or on another alignment. A date carries on a title only when the words run on into
/// it ("ORDERS FOR JUNE 18 THROUGH" / "JUNE 29, 2020").
pub(crate) fn continues_capitals_title(
    page: &Page,
    heading_slot: usize,
    above_slot: usize,
    slot: usize,
    title: &str,
) -> bool {
    let above = &page.lines[above_slot];
    let line = &page.lines[slot];
    let text = line.text.trim();
    let runs_on = title_runs_on(title);
    if date_line(text) {
        return runs_on;
    }
    let first_word = text.split_whitespace().next().unwrap_or_default();
    if runs_on || text.starts_with('(') || run_on_word(first_word) {
        return true;
    }
    let tolerance = page.width * FURNITURE_X_TOLERANCE_FRAC;
    let flush_left = (above.bbox[0] - line.bbox[0]).abs() <= 3.0;
    let flush_right = (above.bbox[2] - line.bbox[2]).abs() <= 3.0;
    let centred = (line_center_x(above) - line_center_x(line)).abs() <= tolerance;
    if centred && !flush_left && !flush_right {
        return true;
    }
    let width = |candidate: &Line| candidate.bbox[2] - candidate.bbox[0];
    let size = line_font_size(line);
    let first_word_width = line.words.first().map_or_else(
        || width(line) * first_word.chars().count() as f64 / text.chars().count().max(1) as f64,
        |word| word.bbox[2] - word.bbox[0],
    );
    let needed = width(above) + size * 0.25 + first_word_width;
    let same_type = |candidate: &Line| {
        has_valid_bbox(candidate) && (line_font_size(candidate) - size).abs() <= 0.3
    };
    let shares_axis = |candidate: &Line| {
        if flush_left {
            (candidate.bbox[0] - above.bbox[0]).abs() <= 3.0
        } else if flush_right {
            (candidate.bbox[2] - above.bbox[2]).abs() <= 3.0
        } else {
            (line_center_x(candidate) - line_center_x(above)).abs() <= tolerance
        }
    };
    let full_within = |measure: Option<f64>| {
        measure.is_some_and(|measure| measure >= width(above) - 1.0 && needed > measure)
    };
    // The title's measure is set by its other lines, down to one leading below this one; the
    // page's by any line but the one above.
    let leading = line.bbox[1] - above.bbox[1];
    let title_measure = page
        .lines
        .iter()
        .enumerate()
        .filter(|(other, candidate)| {
            (heading_slot..=heading_slot + 4).contains(other)
                && *other != above_slot
                && *other != slot
                && candidate.bbox[1] <= line.bbox[1] + leading + 2.0
                && same_type(candidate)
                && shares_axis(candidate)
        })
        .map(|(_, candidate)| width(candidate))
        .max_by(f64::total_cmp);
    let page_measure = page
        .lines
        .iter()
        .enumerate()
        .filter(|(other, candidate)| *other != above_slot && same_type(candidate))
        .map(|(_, candidate)| width(candidate))
        .max_by(f64::total_cmp);
    if full_within(title_measure) || full_within(page_measure) {
        return true;
    }
    // The title closes at this line when it stands apart: what precedes its first line and what
    // follows this one are set further off than its leading or on another alignment.
    let apart = |near: &Line, edge: &Line| {
        (near.bbox[1] - edge.bbox[1]).abs()
            >= leading + ((edge.bbox[3] - edge.bbox[1]) * 0.5).max(8.0)
            || (near.bbox[0] - edge.bbox[0]).abs() > 3.0
                && (near.bbox[2] - edge.bbox[2]).abs() > 3.0
                && (line_center_x(near) - line_center_x(edge)).abs() > 3.0
    };
    let first = &page.lines[heading_slot];
    page.lines[..heading_slot]
        .iter()
        .rev()
        .find(|candidate| has_valid_bbox(candidate) && candidate.bbox[3] <= first.bbox[1] + 1.0)
        .is_none_or(|preceding| apart(preceding, first))
        && page.lines[slot + 1..]
            .iter()
            .find(|candidate| has_valid_bbox(candidate) && candidate.bbox[1] >= line.bbox[3] - 1.0)
            .is_none_or(|below| apart(below, line))
}

/// A word a title's line can break after or open on: an article, a conjunction or a preposition.
pub(crate) fn run_on_word(word: &str) -> bool {
    const WORDS: &[&str] = &[
        "A",
        "AN",
        "THE",
        "AND",
        "OR",
        "NOR",
        "BUT",
        "OF",
        "FOR",
        "TO",
        "IN",
        "ON",
        "AT",
        "BY",
        "WITH",
        "FROM",
        "INTO",
        "UPON",
        "UNDER",
        "OVER",
        "THROUGH",
        "BETWEEN",
        "AGAINST",
        "ABOUT",
        "AS",
        "V",
        "VS",
        "VERSUS",
        "INCLUDING",
        "REGARDING",
        "CONCERNING",
        "WITHIN",
        "WITHOUT",
        "&",
        "DE",
        "DU",
        "DES",
        "LA",
        "LE",
        "LES",
        "ET",
        "EN",
        "AU",
        "AUX",
        "POUR",
        "SUR",
        "PAR",
    ];
    let word = word.trim_end_matches('.').to_uppercase();
    WORDS.contains(&word.as_str())
}

/// Whether a title's words run on past its line: it breaks after a run-on word, a comma, a dash,
/// a slash or an ampersand, or inside a bracket.
pub(crate) fn title_runs_on(title: &str) -> bool {
    let title = title.trim_end();
    title.ends_with([',', '-', '\u{2013}', '\u{2014}', '/', '&'])
        || title.split_whitespace().last().is_some_and(run_on_word)
        || bracket_excess(title) > 0
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

/// On a recognized page, a short title-case line set apart above a full body line: a heading the
/// page's lost bold and size would have shown. Never one in a run of short lines (a table, a
/// caption, an address).
pub(crate) fn title_case_headings(page: &Page, primitives: &PdfPrimitiveEvidence) -> Vec<usize> {
    let measure = page
        .lines
        .iter()
        .filter(|line| !line.exclude_from_body && line.region_type == "body")
        .map(|line| line.bbox[2] - line.bbox[0])
        .fold(0.0, f64::max);
    if measure <= 0.0 {
        return Vec::new();
    }
    let short = |line: &Line| line.bbox[2] - line.bbox[0] < 0.6 * measure;
    let cased = |line: &Line| {
        let text = line.text.trim();
        text.chars().filter(|c| c.is_alphabetic()).count() >= 4
            && (4..=70).contains(&text.chars().count())
            && titlecase_ratio(text) >= 0.6
    };
    let mut found = Vec::new();
    for (slot, line) in page.lines.iter().enumerate() {
        let text = line.text.trim();
        // A layout model's reading of a line stands: the typography stands in only where none read it.
        if line.region_type != "body"
            || layout_labelled(line)
            || line.exclude_from_body
            || !line.note_region_mode.is_empty()
            || !heading_source_eligible(primitives, line)
            || !cased(line)
            || !short(line)
            || text.chars().next().is_some_and(char::is_lowercase)
            || sentence_ended(text)
            || text.ends_with([':', ',', ';'])
            || layout_analysis().has_citation_signal(text)
            || !layout_analysis().heading_text_plausible(title_after_enumerator(text))
        {
            continue;
        }
        let height = (line.bbox[3] - line.bbox[1]).max(1.0);
        let previous = slot.checked_sub(1).map(|p| &page.lines[p]);
        let apart = previous.is_none_or(|p| {
            line.bbox[1] - p.bbox[3] >= 0.5 * height || p.region_type != "body"
        });
        let in_run = previous.is_some_and(|p| p.region_type == "body" && short(p) && cased(p));
        let Some(next) = page.lines.get(slot + 1) else {
            continue;
        };
        let followed = next.region_type == "body"
            && !next.exclude_from_body
            && !short(next)
            && next.bbox[1] >= line.bbox[3] - 0.25 * height;
        if apart && !in_run && followed {
            found.push(slot);
        }
    }
    if found.len() >= 20 {
        return Vec::new();
    }
    found
}

/// With LEGALPDF_HEADING_TRACE set, a heading taken by a rule other than the enumerated ladder.
pub(crate) fn heading_source_trace(line: &Line, why: &str) {
    static ON: OnceLock<bool> = OnceLock::new();
    if *ON.get_or_init(|| std::env::var_os("LEGALPDF_HEADING_TRACE").is_some()) {
        let text: String = line.text.trim().chars().take(48).collect();
        eprintln!("hsrc	{}	{}	{why}	{text}", line.page_index, line.id);
    }
}

pub(crate) fn apply_text_fidelity_headings(
    pages: &mut [Page],
    primitives: &PdfPrimitiveEvidence,
) -> HashMap<String, usize> {
    // Type (size and weight) is the typography reader's; these are the text's own signs.
    for page in pages.iter_mut() {
        let listed_top = page
            .lines
            .iter()
            .filter(|line| primitives.claims.contents_line_ids.contains(&line.id))
            .map(|line| line.bbox[1])
            .min_by(f64::total_cmp);
        for line in &mut page.lines {
            if line.region_type == "heading" {
                heading_source_trace(line, "upstream");
            }
            // A line a container claimed is no heading, whatever its source tagged it.
            if matches!(line.region_type.as_str(), "body" | "heading") && primitives.claims.claimed(&line.id) {
                line.region_type = "body".to_owned();
                continue;
            }
            if line.region_type == "body"
                && !line.exclude_from_body
                && contents_heading(&line.text)
                && listed_top.is_some_and(|top| line.bbox[3] <= top)
                && heading_source_eligible(primitives, line)
            {
                line.region_type = "heading".to_owned();
            }
        }
    }
    let mut capitalised = Vec::new();

    for (page_slot, page) in pages.iter_mut().enumerate() {
        if primitives.claims.contents_pages.contains(&page.index) || !recognized_page(page) {
            continue;
        }
        for line_slot in title_case_headings(page, primitives) {
            heading_source_trace(&page.lines[line_slot], "title");
            page.lines[line_slot].region_type = "heading".to_owned();
            capitalised.push((page_slot, line_slot));
        }
    }

    let decisions = {
        let candidates = heading_candidates(pages, primitives);
        let mut decisions = Vec::new();
        for translation in [false, true] {
            let own = candidates
                .iter()
                .filter(|candidate| {
                    primitives
                        .contexts.translation_line_ids
                        .contains(&pages[candidate.page_slot].lines[candidate.line_slot].id)
                        == translation
                })
                .collect::<Vec<_>>();
            let interpretations = own
                .iter()
                .map(|candidate| candidate.interpretations.as_slice())
                .collect::<Vec<_>>();
            // Reuse the same restart-aware ladder independently for each parallel sequence.
            for (start, parsed) in parse_heading_runs(&interpretations) {
                decisions.extend(own[start..].iter().zip(&parsed.assignments).map(|(candidate, assignment)| HeadingDecision {
                    page_slot: candidate.page_slot,
                    line_slot: candidate.line_slot,
                    joined_line_slot: candidate.joined_line_slot,
                    level: assignment.level,
                }));
            }
        }
        decisions.sort_by_key(|decision| (decision.page_slot, decision.line_slot));
        decisions
    };
    let mut heading_levels = decisions
        .iter()
        .flat_map(|decision| {
            [Some(decision.line_slot), decision.joined_line_slot]
                .into_iter()
                .flatten()
                .filter_map(move |line| {
                    decision
                        .level
                        .map(|level| (decision.page_slot, line, level))
                })
        })
        .filter_map(|(page, line, level)| {
            pages
                .get(page)
                .and_then(|page| page.lines.get(line))
                .map(|line| (line.id.clone(), level))
        })
        .collect::<HashMap<_, _>>();
    let structural: HashSet<(usize, usize)> = decisions
        .iter()
        .flat_map(|decision| {
            [
                Some((decision.page_slot, decision.line_slot)),
                decision
                    .joined_line_slot
                    .map(|line| (decision.page_slot, line)),
            ]
            .into_iter()
            .flatten()
        })
        .collect();

    // A title in capitals can wrap ("ON APPEAL FROM THE COURT OF APPEAL FOR" / "ALBERTA").
    for (page_slot, line_slot) in capitalised {
        if structural.contains(&(page_slot, line_slot))
            || pages[page_slot].lines[line_slot].region_type != "heading"
        {
            continue;
        }
        let block = pages[page_slot].lines[line_slot].block_index;
        for continuation_slot in wrapped_heading_continuations(
            &pages[page_slot],
            line_slot,
            &structural,
            page_slot,
            primitives,
            true,
        ) {
            pages[page_slot].lines[continuation_slot].region_type = "heading".to_owned();
            pages[page_slot].lines[continuation_slot].block_index = block;
        }
    }
    for decision in &decisions {
        let page_slot = decision.page_slot;
        let marker_slot = decision.line_slot;
        let target_slot = decision.joined_line_slot.unwrap_or(marker_slot);
        if page_slot >= pages.len() || target_slot >= pages[page_slot].lines.len() {
            continue;
        }
        if !heading_source_eligible(primitives, &pages[page_slot].lines[target_slot]) {
            continue;
        }
        if pages[page_slot].lines[target_slot].region_type != "heading" {
            continue;
        }
        if target_slot != marker_slot && marker_slot < pages[page_slot].lines.len() {
            let block = pages[page_slot].lines[target_slot].block_index;
            pages[page_slot].lines[marker_slot].region_type = "heading".to_owned();
            pages[page_slot].lines[marker_slot].block_index = block;
        }
        for continuation_slot in wrapped_heading_continuations(
            &pages[page_slot],
            target_slot,
            &structural,
            page_slot,
            primitives,
            false,
        ) {
            let block = pages[page_slot].lines[target_slot].block_index;
            pages[page_slot].lines[continuation_slot].region_type = "heading".to_owned();
            pages[page_slot].lines[continuation_slot].block_index = block;
            if let Some(level) = decision.level {
                heading_levels.insert(pages[page_slot].lines[continuation_slot].id.clone(), level);
            }
        }
    }
    let accepted = pages
        .iter()
        .flat_map(|page| &page.lines)
        .filter(|line| line.region_type == "heading")
        .map(|line| line.id.as_str())
        .collect::<HashSet<_>>();
    heading_levels.retain(|line_id, _| accepted.contains(line_id.as_str()));
    heading_levels
}

/// The heading tags a document puts on so many of its tagged lines that they mark its body (every
/// numbered paragraph tagged H2), as pdf-inspector reads them
/// (`detect_overused_struct_heading_levels`: a level on over 15% of 20 tagged lines or more).
pub(crate) fn overused_heading_tags(pages: &[Page], regions: &HashMap<String, String>) -> HashSet<String> {
    use pdf_inspector::structure_tree::StructRole;
    use pdf_inspector::types::TextLine;
    let mut roles: HashMap<u32, HashMap<i64, StructRole>> = HashMap::new();
    let mut lines: Vec<TextLine> = Vec::new();
    for page in pages {
        let number = page.index as u32 + 1;
        let (items, origin) = legal_pdf_tables::line_items(&page.lines, number, page.height);
        for (mut item, (slot, _)) in items.into_iter().zip(origin) {
            let Some(role) = regions.get(&page.lines[slot].id) else {
                continue;
            };
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

/// A mislabelled PDF paragraph style can tag every line of ordinary prose as Hn: a heading tag on
/// so many lines that it marks the body (`overused_heading_tags`), or on long paragraphs that end
/// their sentences, in the prose's own type.
pub(crate) fn demote_prose_heading_tags(pages: &mut [Page], evidence: &mut PdfPrimitiveEvidence) {
    let Some(regions) = evidence.source_regions.as_ref() else {
        return;
    };
    let overused = overused_heading_tags(pages, regions);
    let style = |span: &legal_pdf_core::Span| {
        (
            span.font.clone(),
            (span.size * 10.0).round() as i64,
            span.flags & 18,
        )
    };
    let mut rejected = HashSet::new();
    let mut prose_styles = HashMap::<String, HashSet<_>>::new();
    let heading_line = |line: &Line| {
        !line.exclude_from_body
            && !evidence.claims.claimed(&line.id)
            && regions
                .get(&line.id)
                .is_some_and(|role| legal_pdf_core::is_heading_role(role))
    };
    let measure = crate::paragraphs::document_measure(pages, heading_line);
    for page in pages.iter() {
        let assigned = crate::paragraphs::paragraph_indices(page, heading_line, measure);
        let mut paragraphs = HashMap::<(usize, &str), Vec<&Line>>::new();
        for (line, paragraph) in page.lines.iter().zip(assigned) {
            if let Some(paragraph) = paragraph {
                paragraphs
                    .entry((paragraph, regions[&line.id].as_str()))
                    .or_default()
                    .push(line);
            }
        }
        for lines in paragraphs.values() {
            let text = lines
                .iter()
                .map(|line| line.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            if text.chars().count() > MAX_FONT_HEADING_CHARS
                && sentence_ended(&text)
                && !text.ends_with('?')
                && titlecase_ratio(&text) < 0.45
            {
                rejected.extend(lines.iter().map(|line| line.id.clone()));
                let mut fonts = HashMap::<_, usize>::new();
                for span in lines.iter().flat_map(|line| &line.spans) {
                    *fonts.entry(style(span)).or_default() += span.text.chars().count();
                }
                let total: usize = fonts.values().sum();
                if let Some((font, mass)) = fonts.into_iter().max_by_key(|(_, mass)| *mass) {
                    if mass * 5 >= total * 3 {
                        prose_styles
                            .entry(regions[&lines[0].id].clone())
                            .or_default()
                            .insert(font);
                    }
                }
            }
        }
    }
    if let Some(regions) = evidence.source_regions.as_mut() {
        for line in pages.iter_mut().flat_map(|page| &mut page.lines) {
            if line.exclude_from_body || line.region_type != "heading" {
                continue;
            }
            let overused_tag = regions.get(&line.id).is_some_and(|role| overused.contains(role));
            let styles = regions
                .get(&line.id)
                .and_then(|role| prose_styles.get(role));
            if styles.is_none() && !rejected.contains(&line.id) && !overused_tag {
                continue;
            }
            let total: usize = line
                .spans
                .iter()
                .map(|span| span.text.chars().count())
                .sum();
            let prose: usize = line
                .spans
                .iter()
                .filter(|span| styles.is_some_and(|styles| styles.contains(&style(span))))
                .map(|span| span.text.chars().count())
                .sum();
            if overused_tag || rejected.contains(&line.id) || total > 0 && prose * 5 >= total * 3 {
                regions.insert(line.id.clone(), "body".to_owned());
                line.region_type = "body".to_owned();
            }
        }
    }
}
