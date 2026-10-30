//! What a case caption's lines say of themselves: a labelled field ("Court File No.:", "BETWEEN:"),
//! where the counsel block or reasons begin, the judgment's first numbered paragraph, and a line
//! naming the case by its citation. The decision reads them as front matter, never headings.

use super::party_label;
use legal_pdf_core::{layout_analysis, CitationSpan, Line};

/// A caption's labelled field.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Field {
    File,
    Court,
    Place,
    Parties,
    Matter,
    Document,
    Contact,
    Date,
    Docket,
    Registry,
    Citation,
    Bench,
}

pub(super) fn field(text: &str) -> Option<Field> {
    let labelled = text.contains(':');
    let key = text.split_once(':').map_or(text, |(key, _)| key);
    let key = key.trim().trim_end_matches('.').to_ascii_lowercase();
    Some(match key.as_str() {
        "court file no" | "court file number" | "court file numbers" | "file no" => Field::File,
        "court" => Field::Court,
        "judicial centre" | "judicial center" => Field::Place,
        "matters" | "re" => Field::Matter,
        "between" | "not parties to the appeal" | "not parties to appeal" => Field::Parties,
        "document" | "title of proceeding" | "title of proceedings" | "style of cause" => {
            Field::Document
        }
        "address for service" => Field::Contact,
        "date" | "heard" | "judgment" if labelled => Field::Date,
        "docket" | "no. on list" | "no on list" if labelled => Field::Docket,
        "registry" if labelled => Field::Registry,
        "citation" if labelled => Field::Citation,
        "before" | "before justice" | "coram" | "the court" if labelled => Field::Bench,
        _ if (key.starts_with("dated ") || key.starts_with("filed "))
            && text.chars().any(|c| c.is_ascii_digit()) =>
        {
            Field::Date
        }
        _ if party_label().is_match(text.trim().trim_end_matches('/')) => Field::Parties,
        _ => return None,
    })
}

pub(super) fn closing(text: &str) -> bool {
    let text = text.trim().to_ascii_lowercase();
    let key = text.split_once(':').map_or(text.as_str(), |(key, _)| key);
    matches!(
        key.trim(),
        "counsel" | "appearances" | "solicitors" | "participant information"
    ) || (!text.contains(':')
        && matches!(
            text.as_str(),
            "memorandum of judgment" | "judgment" | "endorsement"
        ))
        || text.starts_with("reasons for judgment")
        || text.starts_with("reasons for decision")
}

fn numbered_start(marker: &str) -> bool {
    layout_analysis()
        .numbering_marker(marker, None)
        .is_some_and(|marker| marker.ordinal == 1)
}

pub(super) fn reasons(line: &Line) -> bool {
    let value = line.text.trim();
    if value.starts_with('[') && value.ends_with(']') && numbered_start(value) {
        return true;
    }
    let Some((marker, text)) = value.split_once(char::is_whitespace) else {
        return false;
    };
    text.trim().chars().count() >= 24 && numbered_start(marker)
}

/// A dedicated citation line can name the case; a reference inside another field cannot.
/// `citations` reads the citations in the line's text after its field's label, given that text and
/// where it starts in the line's (in bytes).
pub(super) fn citation_title(line: &Line, citations: impl FnOnce(&str, usize) -> Vec<CitationSpan>) -> Option<bool> {
    let key = field(&line.text);
    if key.is_some() && key != Some(Field::Citation) {
        return None;
    }
    let text = if key == Some(Field::Citation) {
        line.text.split_once(':')?.1.trim()
    } else {
        line.text.trim()
    };
    if !text.chars().any(|c| c.is_ascii_digit()) {
        return None;
    }
    let citations = citations(text, text.as_ptr() as usize - line.text.as_ptr() as usize);
    if citations.is_empty()
        || text
            .char_indices()
            .filter(|(at, c)| {
                c.is_alphabetic() && !citations.iter().any(|span| span.range.contains(at))
            })
            .count()
            >= 3
    {
        return None;
    }
    Some(
        citations
            .iter()
            .any(|span| span.range.start < span.core_range.start),
    )
}
