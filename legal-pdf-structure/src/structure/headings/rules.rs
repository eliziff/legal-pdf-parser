//! Heading readings that take a whole document, after every other heading reader: a transcript's
//! own hierarchy (a witness, each examination of them, a hearing's parts), the headings the
//! readers took that a careful reader would not, the ones they missed, and then every heading's
//! level from the document's numbering. A heading is a short line set off from the body that
//! names what follows it; each rule is one way a line shows it is or is not one, read from its
//! words, its type against the document's body, and its place against its neighbours. Measured on
//! the structure gold (batches 5-7, image-reviewed): of 6,736 lines the readers headed, 2,712 are
//! headings; the demotions raise headings F1 from 0.507 to 0.597, and the promotions, chosen on
//! batches 5 and 6, add 0.017 on batch 7.

use crate::structure::{bold_char_share, PdfPrimitiveEvidence};
use legal_pdf_core::model::{Line, Page};
use legal_pdf_core::{dominant_face, flag_share, PageMeasure, ITALIC};
use legal_structure_model::{enumerator_interpretations, parse_heading_runs, EnumeratorInterpretation};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

fn pattern(cell: &'static OnceLock<Regex>, source: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(source).expect("heading rule regex"))
}

/// A witness opening their evidence: "WENDY CUKIER, affirmed, questioned by Mr. Phillips:",
/// "SCOTT REID, sworn".
fn witness(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    pattern(&RE, r"^[A-Z][A-Za-z.'\- ]{1,60},\s*(?i:(?:previously\s+|solemnly\s+)?(?:sworn|affirmed|recalled))\b").is_match(text)
}

/// An examination of a witness: "CROSS-EXAMINATION BY MR. SWAN:", "EXAMINATION-IN-CHIEF BY MS. LEE",
/// "CONTINUED CROSS-EXAMINATION BY MR. PHILLIPS", "MR. BOUCHELEV QUESTIONS THE WITNESS".
fn examination(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    text.split_whitespace().count() <= 16
        && pattern(&RE, concat!(
            r"(?i)^(?:continued\s+|continuation\s+of\s+)?(?:(?:re-?)?direct\s+|re-?)?(?:cross[-\s]*)?examination(?:[-\s]+in[-\s]+chief)?(?:\s+for\s+discovery)?\b.{0,80}\bby\b",
            r"|^(?:mr|ms|mrs|miss|dr|me)\.?\s+[a-z'\-]+(?:\s+[a-z'\-]+)?\s+(?:questions|examines|cross-examines|re-examines)\s+the\s+witness\b"
        )).is_match(text)
}

/// A transcript's index or certificate: "- I N D E X -", "INDEX OF UNDERTAKINGS", "***EXHIBITS***",
/// "WORD INDEX", "REPORTER'S CERTIFICATE".
fn transcript_part(text: &str) -> bool {
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
fn index_entry(text: &str) -> bool {
    text.rsplit(char::is_whitespace).next().is_some_and(|last| !last.is_empty() && last.chars().all(|c| c.is_ascii_digit()))
}

/// A hearing's part, in capitals: "SUBMISSIONS BY MR. X", "RULING", "REASONS FOR JUDGMENT", "SENTENCE".
fn hearing_part(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    crate::structure::all_caps(text) && text.split_whitespace().count() <= 10
        && pattern(&RE, r"^(?:(?:REPLY|CLOSING|OPENING)\s+)?(?:SUBMISSIONS|STATEMENT|ARGUMENT|RULING|DECISION|JUDGMENT|SENTENCE|SENTENCING|(?:ORAL\s+)?REASONS\s+FOR\s+(?:JUDGMENT|RULING|SENTENCE|DECISION))\b").is_match(text)
}

/// A transcript's line that is a turn or a procedural event, never a heading: an examiner resuming
/// ("BY MR. PHILLIPS:"), a speaker's turn ("Q.", "THE WITNESS:"), a recess, an exhibit marked, a
/// discussion off the record.
pub(super) fn turn_or_event(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    legal_structure_model::transcript::speaker_label(text).is_some()
        || pattern(&RE, r"(?i)^(?:-{2,}|\(|\[)|^exhibit\s+no\b|off\s+the\s+record|^by\s+(?:mr|ms|mrs|miss|dr|me|madam)\.?\s").is_match(text)
}

/// A label never a heading, with or without its colon: counsel and appearances, a litigant's role
/// (the caption reader's `party_label`) and a contract's parties and signatories, a recital's
/// "WHEREAS", a caption's connectives, contact details, a placeholder.
pub(super) fn label(text: &str) -> bool {
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

/// Division words whose numbering starts the document's ladder over at its top.
const ATTACHMENTS: [&str; 5] = ["schedule", "appendix", "annex", "exhibit", "annexe"];

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

/// The levels a run of headings' (or a contents list's entries') enumerators take on the engine's
/// heading ladder (`parse_heading_ladder`, the Text-Fidelity grammar the readers level numbered
/// headings with): each enumerator in every family it can be read in ("C." a letter or a numeral), a
/// skipped value read by its shortest jump, a restart at the top opening a new run, as the readers
/// split them. An attachment (a schedule, an appendix, an exhibit) stands at the top and starts the
/// ladder over, as does every text `breaks` marks (a part of the document an unnumbered top heading
/// opens: a judgment's reasons after its headnote). Each numbered text's depth on its run, and
/// whether it opens a run; None where unnumbered.
pub(crate) fn ladder_levels(texts: &[&str], breaks: &[bool]) -> Vec<Option<(usize, bool)>> {
    let mut levels = vec![None; texts.len()];
    let numbered: Vec<(usize, Vec<EnumeratorInterpretation>, bool)> = texts.iter().enumerate().filter_map(|(k, text)| {
        let (value, punct) = enumerator(text)?;
        let readings = enumerator_interpretations(&value, &punct);
        (!readings.is_empty()).then(|| (k, readings, ATTACHMENTS.contains(&punct.as_str())))
    }).collect();
    // Where a run starts over: at an attachment, or past a break since the numbered text before it.
    let restarts = |n: usize| numbered[n].2 || n > 0 && (numbered[n - 1].0 + 1..=numbered[n].0).any(|k| breaks.get(k).copied().unwrap_or(false));
    let mut start = 0;
    while start < numbered.len() {
        let end = (start + 1..numbered.len()).find(|&n| restarts(n)).unwrap_or(numbered.len());
        let run = &numbered[start..end];
        let readings = run.iter().map(|(_, readings, _)| readings.as_slice()).collect::<Vec<_>>();
        for (first, parsed) in parse_heading_runs(&readings) {
            for (n, ((k, _, attachment), assignment)) in run[first..].iter().zip(&parsed.assignments).enumerate() {
                let level = if *attachment { Some(1) } else { assignment.level };
                levels[*k] = level.map(|level| (level, n == 0));
            }
        }
        start = end;
    }
    levels
}

/// Heading levels once the headings are settled, walking them in reading order. An unnumbered heading
/// at the top of the readers' levels starts the ladder over (a judgment's reasons after its
/// headnote). A numbered heading takes its depth on its run of the ladder (`ladder_levels`), set on
/// the scale the run's placed headings show (the levels the document's own outline or contents gave
/// them, by the offset most of them agree on), or else under the unnumbered heading that opens the
/// run ("I. Introduction" under "Reasons for Judgment"). An unnumbered heading takes the readers'
/// step from the heading before it; a placed heading keeps its level.
fn numbering_levels(document: &[Page], levels: &HashMap<String, usize>, placed: &HashMap<String, Option<String>>) -> Vec<(String, usize)> {
    let mut headings: Vec<Vec<&Line>> = Vec::new();
    for page in document {
        let mut order: Vec<&Line> = page.lines.iter().filter(|line| !line.exclude_from_body).collect();
        order.sort_by_key(|line| line.reading_order);
        let mut previous: Option<&Line> = None;
        for line in order {
            if line.region_type != "heading" {
                previous = None;
                continue;
            }
            // A line opening with a number of its own is the next heading ("II. Background" over
            // "A. Sexual Activities"), whatever block holds them both.
            let joins = previous.is_some_and(|prior| prior.block_index == line.block_index && levels.get(&prior.id) == levels.get(&line.id))
                && !numbered(line.text.trim());
            match headings.last_mut() {
                Some(current) if joins => current.push(line),
                _ => headings.push(vec![line]),
            }
            previous = Some(line);
        }
    }
    let texts: Vec<String> = headings.iter()
        .map(|heading| heading.iter().map(|line| line.text.trim()).collect::<Vec<_>>().join(" "))
        .collect();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let enumerators: Vec<Option<(String, String)>> = refs.iter().map(|text| enumerator(text)).collect();
    let numbered_heading: Vec<bool> = enumerators.iter().map(Option::is_some).collect();
    let breaks: Vec<bool> = headings.iter().zip(&numbered_heading)
        .map(|(heading, numbered)| !numbered && levels.get(&heading[0].id).is_some_and(|&level| level <= 1))
        .collect();
    let ladder = ladder_levels(&refs, &breaks);
    // Each run's offset from its depths to the document's levels, by the placed headings in it.
    let mut offsets: HashMap<usize, usize> = HashMap::new();
    let mut run = 0;
    let mut votes: HashMap<usize, HashMap<isize, usize>> = HashMap::new();
    for (k, heading) in headings.iter().enumerate() {
        if let Some((depth, opens)) = ladder[k] {
            run += usize::from(opens);
            if placed.contains_key(&heading[0].id) {
                if let Some(&read) = levels.get(&heading[0].id) {
                    *votes.entry(run).or_default().entry(read as isize - depth as isize).or_default() += 1;
                }
            }
        }
    }
    for (run, counts) in votes {
        let (offset, _) = counts.into_iter().max_by_key(|&(offset, count)| (count, std::cmp::Reverse(offset.abs()))).unwrap_or((0, 0));
        offsets.insert(run, offset.max(0) as usize);
    }
    let mut out = Vec::new();
    // The level the current run of the ladder hangs from, where nothing in it was placed.
    let (mut base, mut run) = (0, 0);
    let (mut prior_level, mut prior_read, mut prior_numbered): (Option<usize>, Option<usize>, bool) = (None, None, false);
    for ((heading, rung), enumerator) in headings.iter().zip(ladder).zip(&enumerators) {
        // A line no reader levelled (one these rules raised) sits level with the heading before it.
        let read = levels.get(&heading[0].id).copied().or(prior_read).unwrap_or(1);
        if let Some((_, true)) = rung {
            run += 1;
            // An attachment stands at the top whatever heads the part before it.
            let attachment = enumerator.as_ref().is_some_and(|(_, word)| ATTACHMENTS.contains(&word.as_str()));
            base = if prior_numbered || attachment { 0 } else { prior_level.unwrap_or(0) };
        }
        let level = if placed.contains_key(&heading[0].id) {
            read
        } else if let Some((depth, _)) = rung {
            offsets.get(&run).copied().unwrap_or(base) + depth
        } else {
            match (prior_level, prior_read) {
                (Some(level), Some(before)) => (level + read).saturating_sub(before).max(1),
                _ => read,
            }
        };
        if std::env::var_os("LEGALPDF_LEVEL_TRACE").is_some() {
            eprintln!("level\t{}\t{:?}\tread {read}\tplaced {}\tbase {base}\t-> {level}\t{}", heading[0].id, rung,
                placed.contains_key(&heading[0].id), heading.iter().map(|line| line.text.trim()).collect::<Vec<_>>().join(" ").chars().take(60).collect::<String>());
        }
        out.extend(heading.iter().map(|line| (line.id.clone(), level)));
        (prior_level, prior_read, prior_numbered) = (Some(level), Some(read), rung.is_some());
    }
    out
}

/// A line opening a numbered paragraph or clause: "[45]", "131. Pour les motifs", "16 (1) La", "(a) the".
pub(super) fn numbered_paragraph(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    pattern(&RE, r"^\s*(?:\[\d{1,4}\]|\d{1,4}(?:\.\d{1,3})*[.)]?\s+\S|\d{1,4}\s*\(\d{1,3}\)|\([a-z0-9]{1,4}\)\s)").is_match(text)
}

/// A line opening a provision's subsection or a clause's term: "(3) For the purposes", "4 (1)
/// Every", "(a) the".
fn subsection(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    pattern(&RE, r"^\s*(?:\d{1,4}(?:\.\d{1,3})*\s*)?\((?:\d{1,3}|[a-z]{1,4})\)\s+\S").is_match(text)
}

/// Whether a line opens a subsection or a clause's terms in the body, not a heading of its own ("A."
/// over "(1) Overview" is a heading over its first subheading).
fn over_terms(next: &Line) -> bool {
    next.region_type != "heading" && subsection(next.text.trim())
}

/// A provision's marginal note: a short line in words set over its subsection ("Company controlled"
/// over "(3) For the purposes of this Act"), not a part's heading in capitals or a numbered caption.
/// Of the structure gold's headings standing over a subsection, four in five so set are marginal
/// notes.
fn marginal_note(text: &str, next: Option<&Line>) -> bool {
    next.is_some_and(over_terms)
        && !numbered(text) && !crate::structure::all_caps(text) && text.split_whitespace().count() <= 12
}

/// A number and nothing else ("III.", "4", "(b)"), as the paragraph reader knows one: a heading's
/// numeral on its own line, not a sentence.
pub(super) fn bare_number(text: &str) -> bool {
    let numeral = text.trim().trim_end_matches(['.', ')']);
    crate::paragraphs::marker_only_re().is_match(text)
        || numeral.chars().all(|c| c.is_ascii_uppercase()) && legal_pdf_core::roman_value(numeral).is_some()
}

/// A line's type: its commonest face and half-point size, by characters.
fn style(line: &Line) -> Option<(&str, i64)> {
    dominant_face(line).map(|(face, size)| (face, (size * 2.0) as i64))
}

/// One heading as the readers left it and the page's regions will hold it: consecutive heading
/// lines of one page, one layout block and one level.
struct Heading<'a> {
    lines: Vec<&'a Line>,
    text: String,
    /// Blank space over its first line and under its last, in the page's line heights; under is
    /// unknown at the foot of a page, where what it heads goes on overleaf.
    above: f64,
    below: Option<f64>,
    larger: bool,
    /// Set in the body's own type: its face and size, not bold, italic or underlined.
    body_type: bool,
}

/// Whether the heading is none, by the five ways the gold's false headings fail to be one. Each
/// rule's share of its catches that are not headings, of all it catches, is noted beside it.
fn not_a_heading(heading: &Heading) -> Option<&'static str> {
    let text = heading.text.as_str();
    let numbered = numbered(text);
    let caps = crate::structure::all_caps(text);
    [
        // It ends a sentence or a clause, unless it is a heading's number alone (97%: "A.   Yes.",
        // "MCCARTHY TÉTRAULT S.E.N.C.R.L., s.r.l.", a term of an order).
        ("demote_sentence", text.ends_with(['.', ',', ';']) && !bare_number(text)),
        ("demote_label", label(text)),
        // Capitals alone in the body's own type set off a name or a caption, not a division
        // (93%: "SPROTT PRIVATE RESOURCE LENDING", "ET ENTRE :", a postal code).
        ("demote_caps_body", !numbered && heading.body_type && caps),
        // It sits nearer the line above than the text below, which a heading's text never does
        // (89%: a signature block's names, "INTENTIONALLY DELETED", a form's field).
        ("demote_nearer_above", !numbered && !heading.larger && heading.below.is_some_and(|below| below > heading.above + 0.5)),
        // An unnumbered heading wrapping over lines is a party block, an address or a title page's
        // caption (91%; the long titles of a patent or a ruling's events are the misses).
        ("demote_wrapped", !numbered && heading.lines.len() > 1),
    ]
    .into_iter()
    .find(|&(_, holds)| holds)
    .map(|(rule, _)| rule)
}

/// The headings' one decision, over what every reader read: the lines the whole document raises (a
/// transcript's witnesses, examinations and parts), the document's outline (`outline`: the short
/// lines it takes for headings and the ones it lowers, decided by how the document sets them), the
/// document's title and the marginal notes it sets apart as their own regions, then every
/// heading's level. A heading opening on a
/// number, letter or numeral ("1.1 Definitions", "2. Purchase and Sale") over its clause's terms
/// ("(a) ...") is the clause's caption, the opening of its body unit, whatever placed it, as is
/// every such heading where the document numbers its provisions (`instrument`, the caller's
/// profile); a division word's ("ARTICLE 1", "SCHEDULE A") stays a heading.
pub(crate) fn read(pages: &mut [Page], primitives: &mut PdfPrimitiveEvidence, instrument: bool) {
    let reading = primitives.contexts.reading_lines().clone();
    let translation = primitives.contexts.translation_line_ids.clone();
    // Lines set apart from the headings as regions of their own: a title, a marginal note.
    let (mut demoted, mut promoted, mut lifted_lines, mut set_apart) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut outline_levels: Vec<(String, usize)> = Vec::new();
    crate::structure::constituents::within_document_contexts(pages, &reading, &translation, |document| {
        let lines = || document.iter().flat_map(|page| &page.lines).filter(|line| !line.exclude_from_body);
        // A transcript (its lines numbered) reads by witnesses and their examinations.
        let transcript = document.iter().any(|page| page.lines.iter().any(|line| line.region_type == "line_label"));
        let witnesses = transcript && lines().any(|line| witness(line.text.trim()));
        let mut raised = HashSet::new();
        for line in lines() {
            let text = line.text.trim();
            if transcript && !index_entry(text) && (witness(text) || hearing_part(text) || transcript_part(text)) {
                promoted.push((line.id.clone(), 1));
                raised.insert(line.id.as_str());
            } else if transcript && !index_entry(text) && examination(text) {
                promoted.push((line.id.clone(), if witnesses { 2 } else { 1 }));
                raised.insert(line.id.as_str());
            } else if transcript && crate::structure::contents_lists::index_letter(text) {
                // A word index's letters head its entries, under the index.
                promoted.push((line.id.clone(), 2));
                raised.insert(line.id.as_str());
            }
        }
        // The body's lines in reading order, page by page, for what each line follows and precedes.
        let measures: HashMap<usize, PageMeasure> = document.iter().map(|page| (page.index, PageMeasure::of(page))).collect();
        let sequence: Vec<&Line> = document.iter().flat_map(|page| {
            let order = &measures[&page.index].order;
            order.iter().map(move |&n| &page.lines[n]).filter(|line| !line.exclude_from_body)
        }).collect();
        // The document's outline: which short lines open its divisions. A line it weighs is a
        // heading as it decides; a heading it never weighed (a layout model's, a quotation's)
        // stands as its reader left it, for the gates below.
        let outline = super::outline::decide(document, &measures, &sequence, primitives, transcript, &raised);
        let heads = |line: &Line| outline.headings.contains(&line.id) || line.region_type == "heading" && !outline.weighed.contains(&line.id);
        for line in &sequence {
            if outline.headings.contains(&line.id) {
                if line.region_type != "heading" {
                    lifted_lines.push(line.id.clone());
                }
                outline_levels.extend(outline.levels.get(&line.id).map(|&level| (line.id.clone(), level)));
            } else if outline.weighed.contains(&line.id) && line.region_type == "heading" {
                demoted.push((line.id.clone(), "demote_outline"));
            }
        }
        // The document's body type is its commonest, by characters.
        let mut type_chars: HashMap<(&str, i64), usize> = HashMap::new();
        for line in document.iter().flat_map(|page| &page.lines) {
            if let Some(style) = style(line) {
                *type_chars.entry(style).or_default() += line.text.trim().chars().count();
            }
        }
        let body = type_chars.into_iter().max_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0))).map(|(style, _)| style);
        // Where the document's body opens: its first full line of prose. The first heading before it,
        // centred in capitals or larger type, or set larger in mixed case, is the document's title
        // (on the structure gold, three in five so set are; centred in the body's type and size, or
        // larger in capitals at the side, it is more often a heading).
        let place: HashMap<&str, usize> = sequence.iter().enumerate().map(|(k, line)| (line.id.as_str(), k)).collect();
        let opens = sequence.iter().position(|line| {
            let measure = &measures[&line.page_index];
            matches!(line.region_type.as_str(), "body" | "text") && line.text.trim().chars().count() >= 60
                && line.bbox[2] - line.bbox[0] >= 0.85 * (measure.right - measure.left)
        });
        let level_of = |id: &str| primitives.headings.heading_levels.get(id).copied().or_else(|| outline.levels.get(id).copied());
        let mut titled = false;
        for page in document.iter() {
            let measure = &measures[&page.index];
            let gap = |upper: &Line, lower: &Line| ((lower.bbox[1] - upper.bbox[3]) / measure.height).clamp(-3.0, 3.0);
            let mut slot = 0;
            while slot < measure.order.len() {
                let first = &page.lines[measure.order[slot]];
                let level = level_of(&first.id);
                if !heads(first) || first.exclude_from_body {
                    slot += 1;
                    continue;
                }
                let mut end = slot + 1;
                while end < measure.order.len() {
                    let next = &page.lines[measure.order[end]];
                    if !heads(next) || next.exclude_from_body || next.block_index != first.block_index
                        || level_of(&next.id) != level || numbered(next.text.trim())
                    {
                        break;
                    }
                    end += 1;
                }
                let members: Vec<&Line> = measure.order[slot..end].iter().map(|&n| &page.lines[n]).collect();
                let last = members[members.len() - 1];
                let bold = bold_char_share(first) >= 0.5;
                let underlined = !first.words.is_empty() && first.words.iter().filter(|w| w.underlined).count() * 2 > first.words.len();
                let own = style(first);
                let larger = measure.size > 0.0 && dominant_face(first).is_some_and(|(_, size)| size >= measure.size * 1.1);
                let heading = Heading {
                    text: members.iter().map(|line| line.text.trim()).collect::<Vec<_>>().join(" "),
                    above: slot.checked_sub(1).map_or(3.0, |prior| gap(&page.lines[measure.order[prior]], first)),
                    below: measure.order.get(end).map(|&n| gap(last, &page.lines[n])),
                    larger,
                    body_type: own.is_some() && own == body && !bold && !underlined && !larger && flag_share(first, ITALIC).unwrap_or(0.0) <= 0.5,
                    lines: members,
                };
                let text = heading.text.as_str();
                if !titled && opens.is_none_or(|opens| place.get(first.id.as_str()).is_some_and(|&at| at < opens)) {
                    titled = true;
                    let width = page.width.max(1.0);
                    let centred = (((first.bbox[0] + first.bbox[2]) / 2.0 - width / 2.0) / width).abs() < 0.03
                        && first.bbox[2] - first.bbox[0] < 0.8 * (measure.right - measure.left);
                    let caps = crate::structure::all_caps(text);
                    if centred && (caps || larger) || larger && !caps {
                        set_apart.extend(heading.lines.iter().map(|line| (line.id.clone(), "title")));
                        slot = end;
                        continue;
                    }
                }
                let next = measure.order.get(end).map(|&n| &page.lines[n]);
                // In a quotation, a statute's notes head the passage quoted (a judgment setting out its law).
                if marginal_note(text, next) && !heading.lines.iter().any(|line| primitives.quotations.scopes.contains_key(&line.id)) {
                    set_apart.extend(heading.lines.iter().map(|line| (line.id.clone(), "marginal_note")));
                    slot = end;
                    continue;
                }
                let terms = next.is_some_and(over_terms);
                let caption = (instrument || terms) && enumerator(text).is_some_and(|(_, punctuation)| !punctuation.chars().any(char::is_alphabetic));
                // A heading the document's own outline or contents placed is the document's word.
                let placed = heading.lines.iter().any(|line| raised.contains(line.id.as_str()) || primitives.headings.navigation_parents.contains_key(&line.id));
                // LEGALPDF_ABLATE=demotions leaves the five demotion gates out, for their measure
                // under the tagger.
                let why = if caption {
                    Some("demote_caption")
                } else if placed || legal_structure_model::ablated("demotions") || heading.lines.iter().any(|line| outline.weighed.contains(&line.id)) {
                    // The outline weighed it, its own vetoes among its evidence.
                    None
                } else {
                    not_a_heading(&heading)
                        .or_else(|| (transcript && turn_or_event(text)).then_some("demote_turn"))
                };
                if let Some(why) = why {
                    demoted.extend(heading.lines.iter().map(|line| (line.id.clone(), why)));
                }
                slot = end;
            }
        }
    });
    // A line a container claimed (contents, a table) stays the container's.
    let promoted: HashMap<String, usize> = promoted.into_iter()
        .filter(|(id, _)| !primitives.claims.claimed(id)).collect();
    let lifted_lines: HashSet<String> = lifted_lines.into_iter()
        .filter(|id| !primitives.claims.claimed(id)).collect();
    let demoted: HashMap<String, &str> = demoted.into_iter().collect();
    let outline_levels: HashMap<String, usize> = outline_levels.into_iter().collect();
    let set_apart: HashMap<String, &str> = set_apart.into_iter().filter(|(id, _)| !primitives.claims.claimed(id)).collect();
    let trail = &mut primitives.headings.trail;
    for line in pages.iter_mut().flat_map(|page| &mut page.lines) {
        if let Some(&level) = promoted.get(&line.id) {
            line.region_type = "heading".to_owned();
            primitives.headings.heading_levels.insert(line.id.clone(), level);
            super::mark(trail, &line.id, true, "rules_promote");
        } else if lifted_lines.contains(&line.id) {
            // Levelled below, with the rest, from its group's rank where no reader levelled it.
            line.region_type = "heading".to_owned();
            if let Some(&level) = outline_levels.get(&line.id) {
                primitives.headings.heading_levels.entry(line.id.clone()).or_insert(level);
            }
            super::mark(trail, &line.id, true, "outline");
        } else if let Some((region, why)) = set_apart.get(&line.id).map(|&region| (region, region)).or_else(|| demoted.get(&line.id).map(|&why| ("body", why))) {
            line.region_type = region.to_owned();
            primitives.headings.heading_levels.remove(&line.id);
            primitives.headings.navigation_parents.remove(&line.id);
            super::mark(trail, &line.id, false, why);
        }
    }
    // Only once the headings are settled can their levels be read: each step is from the heading
    // before, and a numbering family's level is set where it first appears.
    let mut levels = Vec::new();
    let current = primitives.headings.heading_levels.clone();
    let placed = primitives.headings.navigation_parents.clone();
    crate::structure::constituents::within_document_contexts(pages, &reading, &translation, |document| {
        levels.extend(numbering_levels(document, &current, &placed));
    });
    primitives.headings.heading_levels.extend(levels);
}
