//! The words of headings and of the lines no heading is: a transcript's own hierarchy (a witness,
//! each examination of them, a hearing's parts, an index), its turns and events, the labels of
//! counsel, parties and contact details, and the numbering a heading opens with as the heading
//! ladder reads it.

use regex::Regex;
use std::sync::OnceLock;

fn pattern(cell: &'static OnceLock<Regex>, source: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(source).expect("heading rule regex"))
}

/// A witness opening their evidence: "WENDY CUKIER, affirmed, questioned by Mr. Phillips:",
/// "SCOTT REID, sworn".
pub(crate) fn witness(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    pattern(&RE, r"^[A-Z][A-Za-z.'\- ]{1,60},\s*(?i:(?:previously\s+|solemnly\s+)?(?:sworn|affirmed|recalled))\b").is_match(text)
}

/// An examination of a witness: "CROSS-EXAMINATION BY MR. SWAN:", "EXAMINATION-IN-CHIEF BY MS. LEE",
/// "CONTINUED CROSS-EXAMINATION BY MR. PHILLIPS", "MR. BOUCHELEV QUESTIONS THE WITNESS".
pub(crate) fn examination(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    text.split_whitespace().count() <= 16
        && pattern(&RE, concat!(
            r"(?i)^(?:continued\s+|continuation\s+of\s+)?(?:(?:re-?)?direct\s+|re-?)?(?:cross[-\s]*)?examination(?:[-\s]+in[-\s]+chief)?(?:\s+for\s+discovery)?\b.{0,80}\bby\b",
            r"|^(?:mr|ms|mrs|miss|dr|me)\.?\s+[a-z'\-]+(?:\s+[a-z'\-]+)?\s+(?:questions|examines|cross-examines|re-examines)\s+the\s+witness\b"
        )).is_match(text)
}

/// A transcript's index or certificate: "- I N D E X -", "INDEX OF UNDERTAKINGS", "***EXHIBITS***",
/// "WORD INDEX", "REPORTER'S CERTIFICATE".
pub(crate) fn transcript_part(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    // Letters printed apart ("I N D E X") read as their word.
    let squeezed: String = if text.split_whitespace().all(|word| word.chars().filter(|c| c.is_alphabetic()).count() <= 1) {
        text.chars().filter(|c| !c.is_whitespace()).collect()
    } else {
        text.to_owned()
    };
    pattern(&RE, concat!(
        r"^[-*\s]*(?:(?:WORD\s+)?INDEX(?:\s+OF\s+[A-Z' ,&]{3,60})?|TABLE\s+OF\s+CONTENTS|EXHIBITS|OBJECTIONS|REFUSALS|ADVISEMENTS",
        r"|UNDERTAKINGS(?:\s+(?:REQUESTED|GIVEN))?|(?:REPORTER'?S?|COURT\s+REPORTER'?S?)\s+CERTIFICATE|CERTIFICATE(?:\s+OF\s+[A-Z' ]{3,40})?)[-*\s:]*$"
    )).is_match(&squeezed)
}

/// An index's entry ends with the page it points to: "CROSS-EXAMINATION BY MR. BLINICK    3".
pub(crate) fn index_entry(text: &str) -> bool {
    text.rsplit(char::is_whitespace).next().is_some_and(|last| !last.is_empty() && last.chars().all(|c| c.is_ascii_digit()))
}

/// A hearing's part, in capitals: "SUBMISSIONS BY MR. X", "RULING", "REASONS FOR JUDGMENT", "SENTENCE".
pub(crate) fn hearing_part(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    crate::structure::all_caps(text) && text.split_whitespace().count() <= 10
        && pattern(&RE, r"^(?:(?:REPLY|CLOSING|OPENING)\s+)?(?:SUBMISSIONS|STATEMENT|ARGUMENT|RULING|DECISION|JUDGMENT|SENTENCE|SENTENCING|(?:ORAL\s+)?REASONS\s+FOR\s+(?:JUDGMENT|RULING|SENTENCE|DECISION))\b").is_match(text)
}

/// A transcript's line that is a turn or a procedural event, never a heading: an examiner resuming
/// ("BY MR. PHILLIPS:"), a speaker's turn ("Q.", "THE WITNESS:"), a recess, an exhibit marked, a
/// discussion off the record.
pub(crate) fn turn_or_event(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    legal_structure_model::transcript::speaker_label(text).is_some()
        || pattern(&RE, r"(?i)^(?:-{2,}|\(|\[)|^exhibit\s+no\b|off\s+the\s+record|^by\s+(?:mr|ms|mrs|miss|dr|me|madam)\.?\s").is_match(text)
}

/// A label never a heading, with or without its colon: counsel and appearances, a litigant's role
/// (the caption reader's `party_label`) and a contract's parties and signatories, a recital's
/// "WHEREAS", a caption's connectives, contact details, a placeholder.
pub(crate) fn label(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    crate::structure::party_label().is_match(text) || pattern(&RE, concat!(
        r"(?i)^\W*(?:appearances?|comparutions?|a\s*p\s*p\s*e\s*a\s*r\s*a\s*n\s*c\s*e\s*s|solicitors?\s+of\s+record|counsel(?:\s+of\s+record)?",
        r"|procureurs?(?:\s+inscrits?(?:\s+aux?\s+dossiers?)?)?|also\s+present|parties\s+present\b.*|written\s+representations\s+by",
        r"|(?:the\s+)?(?:additional\s+)?(?:borrowers?|lenders?|guarantors?|monitor|covenantors?",
        r"|vendors?|purchasers?|assignors?|assignees?)(?:\s*\(s\))?",
        r"|whereas|between|and(?:\s+to)?|et\s+entre|entre|of\s+the\s+(?:first|second|third)\s+part|copy\s+to|attention|attn",
        r"|address|adresse|tel(?:ephone)?|phone|fax|t[ée]l[ée]copieur|e-?mail|courriel|(?:court\s+)?file\s+no\.?|our\s+file|your\s+file",
        r"|for\s+(?:further\s+)?(?:information|inquiries)\b.*|intentionally\s+(?:deleted|left\s+blank|omitted)|\[?\s*reserved\s*\]?|not\s+used)",
        r"\s*:?\W*$|^re\s*:\s+\S"
    )).is_match(text)
}

/// A line opening with its division's number: the parser's own heading enumerators ("4.", "4.1",
/// "iv.", "B)", alone or before a title), a bracketed one ("(a)", "(1)"), a division word ("PART",
/// "Section", "ARTICLE"), or a numeral before capitals ("V RECOMMENDATIONS", OCR having dropped the
/// stop).
pub(crate) fn numbered(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    let text = text.trim();
    crate::structure::enumerated(text)
        || pattern(&RE, concat!(
            r"^\s*(?:(?i:part|section|article|chapter|schedule|division|title|book|appendix|annex|exhibit|partie|chapitre|annexe)\b",
            r"|\(\s*(?:\d{1,3}|[a-zA-Z]{1,2}|[ivxlc]{1,6})\s*\)(?:\s|$)|[IVXLC]{1,4}\s+[A-Z])"
        )).is_match(text)
}

/// A heading's enumerator as the ladder reads it, (value, punctuation): from the parser's own heading
/// patterns, a bracketed one ("(a)" read as "a" in brackets), or a division word's numeral ("PART IV",
/// "ARTICLE 5"), the word standing for the punctuation so that each word counts its own family.
pub(crate) fn enumerator(text: &str) -> Option<(String, String)> {
    static DIVISION: OnceLock<Regex> = OnceLock::new();
    static BRACKETED: OnceLock<Regex> = OnceLock::new();
    let text = text.trim();
    if let Some(found) = pattern(&DIVISION, r"(?i)^(part|section|article|chapter|schedule|division|title|book|appendix|annex|exhibit|partie|chapitre|annexe)\s+[\x22\u{201c}\u{2018}'(]?([IVXLCDM]{1,7}|\d{1,3}|[A-Z])\b").captures(text) {
        return Some((found[2].to_owned(), found[1].to_lowercase()));
    }
    if let Some(capture) = crate::structure::heading_inline_re().captures(text).or_else(|| crate::structure::heading_standalone_re().captures(text)) {
        let (value, punct, _) = crate::structure::heading_enumerator(&capture);
        return Some((value.to_owned(), punct.to_owned()));
    }
    pattern(&BRACKETED, r"^\(\s*(\d{1,3}|[A-Za-z]|[ivxlc]{1,6})\s*\)(?:\s|$)").captures(text).map(|found| (found[1].to_owned(), "()".to_owned()))
}

/// A number and nothing else ("III.", "4", "(b)"), as the paragraph reader knows one: a heading's
/// numeral on its own line, not a sentence.
pub(crate) fn bare_number(text: &str) -> bool {
    let numeral = text.trim().trim_end_matches(['.', ')']);
    crate::paragraphs::marker_only_re().is_match(text)
        || numeral.chars().all(|c| c.is_ascii_uppercase()) && legal_pdf_core::roman_value(numeral).is_some()
}
