use legal_pdf_core::layout_analysis;

// The citation grammar's layout cues, as the structure engine supplies them.
pub fn crossref_short_form(text: &str, byte_start: usize) -> String {
    layout_analysis().crossref_short_form(text, byte_start)
}

pub fn is_citation_shaped_tail(text: &str) -> bool {
    layout_analysis().is_citation_shaped_tail(text)
}

pub fn is_counter_noun(word: &str) -> bool {
    layout_analysis().is_counter_noun(word)
}

pub fn reporter_abbreviation_regex(abbreviation: &str) -> String {
    layout_analysis().reporter_abbreviation_regex(abbreviation)
}

pub fn has_legal_citation_cue(text: &str) -> bool {
    layout_analysis().has_citation_cue(text)
}

pub fn has_citation_signal(text: &str) -> bool {
    layout_analysis().has_citation_signal(text)
}

pub fn is_legal_citation_continuation(text: &str) -> bool {
    layout_analysis().is_citation_continuation(text)
}

/// Whether a short line is plausibly a heading's title, by the structure engine's citation grammar.
pub fn heading_text_plausible(value: &str) -> bool {
    layout_analysis().heading_text_plausible(value)
}

/// [`heading_text_plausible`], or a title set apart in bold naming a case by its parties.
pub fn styled_heading_text_plausible(value: &str) -> bool {
    layout_analysis().styled_heading_text_plausible(value)
}

pub fn protected_citation_spans(text: &str) -> Vec<(usize, usize)> {
    layout_analysis().protected_citation_spans(text)
}
