//! A PDF's page map: each PDF page's printed number and the line printing it, from what the pages print
//! (`PageEvidence`). It is the parser's one reader of page numbers: the structure stage sets each page's line apart as
//! its label and gives the page its number from it.
//!
//! 1. What a page prints is its number, read on the page alone (`printed`: its outermost head or foot row in a page
//!    number's shape).
//! 2. The folio sequence decides (Text-Fidelity's sequence fit, which made the journals' page maps): the numbers each
//!    page's head and foot lines print in a folio's shape (alone, set apart from a running head's title, a page counter)
//!    vote for a shift (printed number less PDF page); the best supported shift numbers the document from its first
//!    anchor to the end, where it passes the fit's gates (enough anchors over enough of the document, at consistent
//!    heights, no close competitor), a misread folio taking the sequence's number.
//! 3. The PDF's own labels stand on a page with no number where they agree with the numbers pages print somewhere and
//!    differ from them nowhere; labels that only count the PDF's pages (1, 2, 3 ...) say nothing.
//! 4. Where a caller holds citations to the report the PDF prints (`starts`, their first pages): an opening page with no
//!    number is read at its edge for one (`margin_labels`), each page printing one is an anchor, and from each anchor to
//!    the next every page takes the anchor's number at its distance where most of the run's numbers agree.
//! 5. Each numbered page's line is the one it was read from, else the outermost line of its head or foot printing the
//!    number the map gives it (`printing_lines`).
//!
//! The structure stage, the engine's `pdfPageMap`, the backend's page map and the tools all read this.
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;

/// What one PDF page tells of its number.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PageEvidence {
    /// The PDF's own label for the page (its /PageLabels).
    #[serde(default)]
    pub embedded: Option<String>,
    /// The page's size, and its lines with their boxes ([x0, top, x1, bottom], display space), every one of them: a
    /// page's number may stand anywhere its layout puts it (a margin, a turned page). A line that numbers no page (a
    /// transcript's line number) comes blank.
    #[serde(default)]
    pub width: f64,
    #[serde(default)]
    pub height: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lines: Vec<EdgeLine>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EdgeLine {
    pub text: String,
    pub rect: [f64; 4],
}

/// One PDF page of the map.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
#[serde(rename_all = "camelCase")]
pub struct PdfPageBinding {
    /// The PDF page, from 1.
    pub pdf_page: u32,
    pub label: Option<String>,
    /// The line printing `label` (its index among the page's evidence lines), where the page prints it.
    #[serde(skip)]
    pub line: Option<usize>,
    /// "detected" (what the page prints), "sequence" (its place in the folio sequence), "embedded" (the PDF's own
    /// label), "reporter" (a citation's first page at its distance); none where the page has no number.
    pub source: Option<String>,
    /// "resolved", "ambiguous" (the page prints one number and the PDF labels it another), or "unknown".
    pub status: String,
    /// Every other numbering the page prints, as printed: a record's Bates number ("000208"), an exhibit's or
    /// appendix's page ("E-162"), a chapter's page ("5-1"), a second count of pages ("other"), the PDF's own count ("pdf").
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[cfg_attr(feature = "binding-types", ts(as = "Option<Vec<PrintedNumber>>", optional))]
    pub numbers: Vec<PrintedNumber>,
}

/// One numbering a page prints besides its page number.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
pub struct PrintedNumber {
    /// "bates", "exhibit", "chapter", "other" or "pdf".
    pub kind: String,
    pub label: String,
}

/// The share of a page's height at its top and at its bottom whose lines are its head and its foot.
pub const EDGE_BAND: f64 = 0.15;

/// A page's head and foot (its top and bottom EDGE_BAND) as a recognizer reads them from its image, and the page's size.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Bands {
    pub width: f64,
    pub height: f64,
    pub head: Vec<EdgeLine>,
    pub foot: Vec<EdgeLine>,
}

/// `pages`, each page `read_again` names read again in its head and foot (`read` gives the bands of the pages it is asked
/// for): a scan's text layer can leave out the folio its image prints, and what the bands read stands for the text
/// layer's lines there. Every other such page is read first, and the rest only where the page map still leaves them
/// without a number (the folio sequence numbers the pages between its anchors).
pub fn reread(mut pages: Vec<PageEvidence>, read: &mut dyn FnMut(&[usize]) -> Result<Vec<(usize, Bands)>, String>) -> Result<Vec<PageEvidence>, String> {
    let again: Vec<usize> = read_again(&pages).into_iter().enumerate().filter_map(|(index, again)| again.then_some(index)).collect();
    let first: Vec<usize> = again.iter().copied().step_by(2).collect();
    for (index, bands) in read(&first)? {
        merge_bands(&mut pages[index], bands);
    }
    let numbered = page_map(&pages, &[], &|_, _| false);
    let rest: Vec<usize> = again.iter().copied().filter(|index| !first.contains(index) && numbered[*index].label.is_none()).collect();
    for (index, bands) in read(&rest)? {
        merge_bands(&mut pages[index], bands);
    }
    Ok(pages)
}

/// A page's lines with its bands as read standing for its text layer's in each band they read anything.
fn merge_bands(page: &mut PageEvidence, bands: Bands) {
    let height = bands.height;
    let in_head = |line: &EdgeLine| line.rect[1] < height * EDGE_BAND;
    let in_foot = |line: &EdgeLine| line.rect[3] > height * (1.0 - EDGE_BAND);
    let head = if bands.head.is_empty() { page.lines.iter().filter(|line| in_head(line)).cloned().collect() } else { bands.head };
    let foot = if bands.foot.is_empty() { page.lines.iter().filter(|line| in_foot(line) && !in_head(line)).cloned().collect() } else { bands.foot };
    let middle: Vec<EdgeLine> = page.lines.iter().filter(|line| !in_head(line) && !in_foot(line)).cloned().collect();
    page.lines = head.into_iter().chain(middle).chain(foot).collect();
    if page.height <= 0.0 {
        (page.width, page.height) = (bands.width, bands.height);
    }
}

/// The page map (see the crate).
pub fn page_map(pages: &[PageEvidence], starts: &[u32], is_citation: &dyn Fn(&str, &str) -> bool) -> Vec<PdfPageBinding> {
    let mut read: Vec<Option<(String, usize)>> = pages.iter().map(printed).collect();
    if !starts.is_empty() && !read.iter().flatten().any(|(label, _)| label.parse::<u32>().is_ok_and(|n| starts.contains(&n))) {
        margin_labels(&mut read, pages, starts, is_citation);
    }
    let observed: Vec<Option<String>> = read.iter().map(|read| read.as_ref().map(|(label, _)| label.clone())).collect();
    let mut bindings = resolve(&observed, &pages.iter().map(|page| page.embedded.clone()).collect::<Vec<_>>());
    sequence(&mut bindings, &observed, pages, starts);
    counters(&mut bindings, pages);
    anchor(&mut bindings, &observed, starts);
    numberings(&mut bindings, pages);
    printing_lines(&mut bindings, pages, &read);
    bindings
}

/// Each page's number read on the page alone (`printed`), for the fit's helpers.
fn observed(pages: &[PageEvidence]) -> Vec<Option<String>> {
    pages.iter().map(|page| printed(page).map(|(label, _)| label)).collect()
}

/// What a page prints as its number, read on the page alone (the page-number rule the structure stage's furniture
/// read): a line of its outermost row at its foot or head (the page's bottom or top 14%) in a page number's shape
/// ("12", "- 12 -", "Page 12", "Page 2 of 30", "xiv"), the foot's before the head's; not a bare number beside a line of
/// the page's prose (twenty letters or more, no running head's words: a law report's paragraph number in its margin),
/// nor one opening a foot row that goes on in text (a note's). With the line printing it.
fn printed(page: &PageEvidence) -> Option<(String, usize)> {
    static SHAPE: OnceLock<Regex> = OnceLock::new();
    let shape = SHAPE.get_or_init(|| {
        Regex::new(r"(?i)^-?\s*(?:page\s+)?(\d{1,6}|[ivxlc]{1,12})(?:\s*(?:of|/|of\s*/\s*de|de|sur)\s*\d{1,6})?\s*-?$").unwrap()
    });
    let height = page.height;
    if height <= 0.0 {
        return None;
    }
    let p = patterns();
    let rows = rows(page);
    let foot = rows.last().filter(|row| row.whole.rect[1] > height * 0.86);
    let head = rows.first().filter(|row| row.whole.rect[3] < height * 0.14);
    for (row, at_foot) in foot.map(|row| (row, true)).into_iter().chain(head.map(|row| (row, false))) {
        for (k, cell) in row.cells.iter().enumerate() {
            let beside: String = row.cells.iter().enumerate().filter(|(j, _)| *j != k).map(|(_, other)| other.text.as_str()).collect::<Vec<_>>().join(" ");
            let running = p.header_context.is_match(&beside) || p.report_head.is_match(&beside);
            let prose = beside.chars().filter(|c| c.is_alphabetic()).count() >= 20;
            let note = at_foot && row.cells[k + 1..].iter().any(|cell| cell.text.chars().any(char::is_alphabetic));
            if isolated_number(&cell.text).is_some() && !running && (prose || note) {
                continue;
            }
            for (line, piece) in &row.pieces[k] {
                let Some(found) = shape.captures(piece.text.trim()) else { continue };
                let number = &found[1];
                if number.bytes().all(|b| b.is_ascii_digit()) || roman_value(number).is_some() {
                    return Some((number.to_owned(), *line));
                }
            }
        }
    }
    None
}

/// Each numbered page's line printing its number: the line it was read from where the map gives the page the number it
/// reads there (`printed`, `margin_labels`), else the outermost line of its head or foot printing that number (a
/// sequence's folio beside a running head's title, a counter, a typed numbering); none where it prints it nowhere there.
fn printing_lines(bindings: &mut [PdfPageBinding], pages: &[PageEvidence], read: &[Option<(String, usize)>]) {
    for ((binding, page), read) in bindings.iter_mut().zip(pages).zip(read) {
        let Some(label) = binding.label.as_deref() else { continue };
        binding.line = match read {
            Some((own, line)) if own.eq_ignore_ascii_case(label) => Some(*line),
            _ => printing_line(page, label),
        };
    }
}

/// The outermost line of a page's head or foot (its top or bottom 28%) printing `label`: a number among the line's
/// folio candidates, a roman numeral alone, a typed numbering ("E-162") as printed.
fn printing_line(page: &PageEvidence, label: &str) -> Option<usize> {
    let height = page.height.max(1.0);
    let number = label.parse::<i64>().ok();
    let compact = |text: &str| text.chars().filter(|c| !c.is_whitespace()).flat_map(char::to_lowercase).collect::<String>();
    let wanted = compact(label);
    rows(page)
        .iter()
        .filter_map(|row| {
            let y = (row.whole.rect[1] + row.whole.rect[3]) / 2.0 / height;
            let zone = if y <= 0.28 { Zone::Top } else if y >= 0.72 { Zone::Bottom } else { return None };
            let prints = |text: &str| match number {
                Some(number) => line_candidates(text, zone, Some(y), 0).iter().any(|c| c.number == number),
                None => {
                    let bare: String = text.trim().chars().filter(|c| !EDGE_PUNCT.contains(*c)).collect();
                    bare.eq_ignore_ascii_case(label) || compact(text).contains(&wanted)
                }
            };
            let (line, _) = row.pieces.iter().flatten().find(|(_, piece)| prints(&piece.text))?;
            Some((y.min(1.0 - y), *line))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, line)| line)
}

/// A page the sequence leaves without a number that names its place itself ("Page 1 of 1", "Page: 2", "1/4") is that
/// page: a form or a decision bound with others counts its own pages, and no sequence runs through them.
fn counters(bindings: &mut [PdfPageBinding], pages: &[PageEvidence]) {
    for (binding, page) in bindings.iter_mut().zip(pages) {
        if binding.label.is_some() {
            continue;
        }
        let numbers: std::collections::BTreeSet<i64> = page_candidates(None, Some(page)).iter()
            .filter(|c| c.kind == Kind::Counter && sequence_anchor(c)).map(|c| c.number).collect();
        if let [number] = numbers.into_iter().collect::<Vec<_>>()[..] {
            binding.label = Some(number.to_string());
            binding.source = Some("detected".to_owned());
            binding.status = "resolved".to_owned();
        }
    }
}

/// Each page's binding from what it prints (`observed`) and the PDF's own labels (`embedded`), as rule 3 takes them.
fn resolve(observed: &[Option<String>], embedded: &[Option<String>]) -> Vec<PdfPageBinding> {
    let count = observed.len().max(embedded.len());
    let at = |list: &[Option<String>], index: usize| list.get(index).cloned().flatten().map(|label| label.trim().to_owned()).filter(|label| !label.is_empty());
    let generic = embedded.iter().enumerate().all(|(index, label)| label.as_deref().is_none_or(|label| label.trim() == (index + 1).to_string()));
    let corroborated = (0..count).any(|index| at(embedded, index).is_some() && at(embedded, index) == at(observed, index))
        && !(0..count).any(|index| matches!((at(embedded, index), at(observed, index)), (Some(a), Some(b)) if a != b));
    (0..count)
        .map(|index| {
            let detected = at(observed, index);
            let metadata = if generic { None } else { at(embedded, index) };
            let conflict = matches!((&detected, &metadata), (Some(a), Some(b)) if a != b);
            let (label, source) = match (conflict, detected.clone(), metadata.filter(|_| corroborated)) {
                (true, _, _) => (None, None),
                (false, Some(label), _) => (Some(label), Some("detected")),
                (false, None, Some(label)) => (Some(label), Some("embedded")),
                (false, None, None) => (None, None),
            };
            let status = if conflict { "ambiguous" } else if label.is_some() { "resolved" } else { "unknown" };
            PdfPageBinding { pdf_page: index as u32 + 1, label, line: None, source: source.map(str::to_owned), status: status.to_owned(), numbers: Vec::new() }
        })
        .collect()
}

/// How a head or foot line prints a number (Text-Fidelity's candidate kinds).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    /// The line is the number alone.
    Isolated,
    /// "12 / 30".
    Counter,
    /// A running head's title, then a separator, then the number.
    HeaderTrailing,
    /// "S12".
    Supplement,
    /// The number opens or closes a longer line; `small` where it is 1 to 9.
    Leading { small: bool },
    Trailing { small: bool },
    /// A number of any shape in a chain of pages each printing the next one at the same height (`folio_runs`).
    Run,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Zone {
    Top,
    Bottom,
    NearTop,
    NearBottom,
    OrderTop,
    OrderBottom,
    Body,
}

impl Zone {
    fn edge(self) -> bool {
        self != Zone::Body
    }
    fn top(self) -> bool {
        matches!(self, Zone::Top | Zone::NearTop | Zone::OrderTop)
    }
    fn bottom(self) -> bool {
        matches!(self, Zone::Bottom | Zone::NearBottom | Zone::OrderBottom)
    }
}

/// A number a page's head or foot may print as its folio, with how folio-like it is.
#[derive(Clone, Debug)]
struct Candidate {
    number: i64,
    kind: Kind,
    zone: Zone,
    score: f64,
    year_like: bool,
    citation_like: bool,
    /// The line's middle, as a share of the page's height; none for the number the page prints as read (`printed`).
    y: Option<f64>,
    line: usize,
    text: String,
    /// The whole row the line stands in (a running head's folio, title and citation): what it reads beside.
    context: String,
    /// The number beside a running head's own words (`running_heads`).
    running: bool,
}

// Text-Fidelity's EDGE_PUNCT, and the marks a folio is set between that a reading may give ("| 2 |", "• 2 •", "� 2 �")
const EDGE_PUNCT: &str = " \t\r\n[](){}.,;:'\"`~_-=\u{2010}\u{2011}\u{2012}\u{2013}\u{2014}\u{2015}|\u{2022}\u{00b7}\u{fffd}";

struct Patterns {
    leading: Regex,
    trailing: Regex,
    supplement: Regex,
    counter: Regex,
    counter_end: Regex,
    header_trailing: Regex,
    month_or_volume: Regex,
    citation: Regex,
    bracket_year: Regex,
    year: Regex,
    edge_space: Regex,
    dot_leader: Regex,
    header_separator: Regex,
    header_context: Regex,
    report_head: Regex,
    note: Regex,
    small_leading_note: Regex,
    small_leading_gap: Regex,
    small_trailing_cited: Regex,
    small_trailing_gap: Regex,
    title_letters: Regex,
    title_opening: Regex,
    text_before: Regex,
}

fn patterns() -> &'static Patterns {
    static PATTERNS: OnceLock<Patterns> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        let punct = regex::escape(EDGE_PUNCT);
        Patterns {
            leading: Regex::new(&format!("^[{punct}]*(\\d+)")).unwrap(),
            trailing: Regex::new(&format!("(\\d+)[{punct}]*$")).unwrap(),
            // "S12" (`SUPPLEMENT_EDGE_FOLIO_RE`), and an exhibit's or appendix's lettered page: "E-162", "A - 2"
        supplement: Regex::new(&format!("^[{punct}]*(?:S|[A-Z]{{1,3}}\\s?-\\s?)(\\d{{1,6}})[{punct}]*$")).unwrap(),
            // "12 / 30" (`PAGE_COUNTER_RE`), and a page named as one: "Page 12", "Page: 12 of 30", "p. 12"
        // a line closing on its page's place in the document: "T-15.01 / 2 sur 46", "... Page 2 of 46"
        counter_end: Regex::new(r"(?i)(?:^|\s)(\d{1,4})\s+(?:of|de|sur)\s+(\d{1,4})\s*$").unwrap(),
        // the page word as a reading may give it ("P a g e", "Pa g e") and set off by a bar or colon, on either side:
        // "Page | 2", "- Page 2 -", "1 | P a g e"
        counter: Regex::new(r"(?i)^[\s\-–—|•·]*(?:(\d{1,4})\s*[|/]\s*(\d{1,4})|(?:p\s?a\s?g\s?e|pg\.?|p\.)\s*[:|]?\s*(\d{1,4})(?:\s*(?:of|/|de|sur)\s*(\d{1,4}))?|(\d{1,4})\s*\|?\s*p\s?a\s?g\s?e)[\s\-–—|•·]*$").unwrap(),
            header_trailing: Regex::new(r"(?i)\b(?:vol(?:ume)?|journal|review|law|legal|rights|quarterly|yearbook|annuaire|bulletin|appeal|forum)\b.{0,80}?(?:[%|•·–—]\s*|\s{2,})(\d{1,5})\s*$").unwrap(),
            month_or_volume: Regex::new(r"(?i)\b(vol(?:ume)?|issue|iss\.?|no\.?|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t)?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|page)\b").unwrap(),
            citation: Regex::new(r"(?i)\b(v\.|vs\.|s\.c\.r\.|d\.l\.r\.|w\.w\.r\.|c\.c\.c\.|para\.|art\.|rev\.|l\.j\.)\b").unwrap(),
            bracket_year: Regex::new(r"\[[12][0-9]{3}\]").unwrap(),
            year: Regex::new(r"\b[12][0-9]{3}\b").unwrap(),
            edge_space: Regex::new(r"\s{2,}|\t").unwrap(),
            dot_leader: Regex::new(r"(?:\.\s*){2,}").unwrap(),
            header_separator: Regex::new(r"^[\.\)]?\s+\S").unwrap(),
            // a law report's running head names its report without a page ("[1983] 2 S.C.R." then the style of cause or
            // nothing; a reading may take the bracket for a 1): a citation in the text gives the page after it
            report_head: Regex::new(r"(?:\[|\b1)?[12]\d{3}\]\s*(?:\d{1,3}\s+)?[A-Z](?:\.?\s?[A-Z]){1,4}\.(?:\s*\((?:2d|3d|\d+th)\))?(?:\s*$|\s+[^\d\s])").unwrap(),
            header_context: Regex::new(r"(?i)\b(journal|review|law|legal|rights|quarterly|yearbook|annuaire|bulletin|vol(?:ume)?\.?|issue|no\.?)\b").unwrap(),
            note: Regex::new(r"^\s*\d{1,4}[\.\)](?:\s|\u{00a0}|\u{202f})+").unwrap(),
            small_leading_note: Regex::new(r"^\s*[\.\)]\s+\S").unwrap(),
            small_leading_gap: Regex::new(r"^\s{2,}\S").unwrap(),
            small_trailing_cited: Regex::new(r"(?i)\b(?:at|ch|para|paras|scc|scr|d\.l\.r|c\.c\.c)\.?\s*$").unwrap(),
            small_trailing_gap: Regex::new(r"\S\s{2,}$").unwrap(),
            title_letters: Regex::new(r"[A-Za-z]{3,}").unwrap(),
            title_opening: Regex::new("^[\"'“‘(\\[]?[A-Z]|^[\"'“‘]").unwrap(),
            text_before: Regex::new(r"\S\s+$").unwrap(),
        }
    })
}

fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The last `count` characters of `text`.
fn tail(text: &str, count: usize) -> &str {
    let skip = text.chars().count().saturating_sub(count);
    text.char_indices().nth(skip).map_or("", |(at, _)| &text[at..])
}

/// The number at a line's start (Text-Fidelity's `LEADING_NUMBER_RE`): its digits' span, up to five digits.
fn leading_number(text: &str) -> Option<(usize, usize)> {
    let digits = patterns().leading.captures(text)?.get(1)?;
    (digits.as_str().len() <= 5).then(|| (digits.start(), digits.end()))
}

/// The number at a line's end (`TRAILING_NUMBER_RE`).
fn trailing_number(text: &str) -> Option<(usize, usize)> {
    let digits = patterns().trailing.captures(text)?.get(1)?;
    (digits.as_str().len() <= 5).then(|| (digits.start(), digits.end()))
}

fn isolated_number(text: &str) -> Option<i64> {
    // a hyphen, dash, point or tilde between two numbers joins two labels ("5-1" a chapter's page, "12-13" a spread,
    // "12.6" a section, "4.16.3 ~ 1" a section's page), not one number's digits a reading spaced ("74 1")
    static COMPOUND: OnceLock<Regex> = OnceLock::new();
    if COMPOUND.get_or_init(|| Regex::new(r"\d\s*[-.~\u{2010}-\u{2015}]\s*\d").unwrap()).is_match(text) {
        return None;
    }
    // any run of digits whose value is a page's (a Bates number "000208" is page 208)
    let compact: String = text.trim().chars().filter(|c| !EDGE_PUNCT.contains(*c)).collect();
    if compact.is_empty() || compact.len() > 12 || !compact.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    compact.parse::<i64>().ok().filter(|n| (1..=99_999).contains(n))
}

fn supplement_number(text: &str) -> Option<i64> {
    patterns().supplement.captures(text.trim())?[1].parse().ok().filter(|n| (1..=99_999).contains(n))
}

/// The line has a folio's shape: the number set apart from the rest by a gap, dot leaders, or a running head's title.
fn folio_shape(text: &str, kind: Kind) -> bool {
    let p = patterns();
    let value = text.trim();
    match kind {
        Kind::Isolated | Kind::Run => true,
        Kind::Counter => p.counter.is_match(&collapse(value)) || p.counter_end.is_match(&collapse(value)),
        Kind::HeaderTrailing => p.header_trailing.is_match(&collapse(value)),
        Kind::Supplement => supplement_number(value).is_some(),
        Kind::Leading { .. } => leading_number(value).is_some_and(|(_, end)| {
            let after = &value[end..];
            p.edge_space.find(after).is_some_and(|found| found.start() == 0)
                || (p.header_separator.is_match(after) && (p.header_context.is_match(after) || p.report_head.is_match(after)))
        }),
        Kind::Trailing { .. } => trailing_number(value).is_some_and(|(start, _)| {
            let before = &value[..start];
            p.edge_space.is_match(tail(before, 18))
                || p.dot_leader.is_match(tail(before, 32))
                || (p.text_before.is_match(before) && (p.header_context.is_match(before) || p.report_head.is_match(before)))
        }),
    }
}

/// The line reads as text or a note that happens to open or close on a number.
fn note_shape(text: &str, kind: Kind) -> bool {
    let p = patterns();
    let value = text.trim();
    match kind {
        Kind::Leading { small: false } => {
            if let Some((_, end)) = leading_number(value) {
                let after = &value[end..];
                if p.header_separator.is_match(after) && p.header_context.is_match(after) {
                    return false;
                }
            }
            p.note.is_match(value)
        }
        Kind::Trailing { small: false } => !folio_shape(value, kind),
        _ => false,
    }
}

/// A head line naming its running title, then the folio.
fn title_folio_shape(text: &str, kind: Kind, zone: Zone) -> bool {
    let p = patterns();
    if !matches!(kind, Kind::Trailing { .. }) || !zone.top() {
        return false;
    }
    let value = collapse(text);
    if value.is_empty() || value.chars().count() > 90 || p.citation.is_match(&value) {
        return false;
    }
    let Some((start, _)) = trailing_number(&value) else { return false };
    let before = value[..start].trim();
    p.title_letters.is_match(before) && p.title_opening.is_match(before)
}

fn small_edge_folio(text: &str, at_start: bool, (start, end): (usize, usize)) -> bool {
    let p = patterns();
    if at_start {
        let after = &text[end..];
        !p.small_leading_note.is_match(after) && p.small_leading_gap.is_match(after)
    } else {
        let before = &text[..start];
        !p.small_trailing_cited.is_match(before) && p.small_trailing_gap.is_match(before)
    }
}

/// The numbers a line in a head or foot may print as its folio.
fn embedded_numbers(text: &str) -> Vec<(i64, Kind)> {
    let p = patterns();
    let raw = text.trim();
    let value = collapse(raw);
    if value.chars().count() > 140 {
        return Vec::new();
    }
    if let Some(found) = p.counter.captures(&value).or_else(|| p.counter_end.captures(&value)) {
        let number = |k: usize| found.get(k).and_then(|m| m.as_str().parse::<i64>().ok());
        let (current, total) = (number(1).or(number(3)).or(number(5)).unwrap_or(0), number(2).or(number(4)));
        if current > 0 && total.is_none_or(|total| current <= total + 10 && total <= 99_999) {
            // "Page | 4" a reading gives as "Page 14": the bar its 1; the page's run says which it is
            static BAR: OnceLock<Regex> = OnceLock::new();
            let bar = BAR.get_or_init(|| Regex::new(r"(?i)p\s?a\s?g\s?e\s*1\s?(\d{1,4})\b").unwrap());
            let barred = bar.captures(&value).and_then(|c| c[1].parse::<i64>().ok()).filter(|&n| n >= 1 && total.is_none());
            return std::iter::once((current, Kind::Counter)).chain(barred.map(|n| (n, Kind::Counter))).collect();
        }
    }
    let mut found: Vec<(i64, Kind, bool)> = Vec::new();
    if let Some(number) = p.header_trailing.captures(&value).and_then(|c| c[1].parse::<i64>().ok()).filter(|n| (1..=99_999).contains(n)) {
        found.push((number, Kind::HeaderTrailing, false));
    }
    if let Some(number) = supplement_number(raw) {
        found.push((number, Kind::Supplement, false));
    }
    for (at_start, span) in [(true, leading_number(raw)), (false, trailing_number(raw))] {
        let Some(span) = span else { continue };
        let number: i64 = raw[span.0..span.1].parse().unwrap_or(0);
        if !(1..=99_999).contains(&number) {
            continue;
        }
        let small = number <= 9 && small_edge_folio(raw, at_start, span);
        if number <= 9 && !small {
            continue;
        }
        if !at_start && p.month_or_volume.is_match(tail(&raw[..span.0], 24)) {
            continue;
        }
        let year_like = (1800..=2099).contains(&number) && p.year.is_match(&value);
        found.push((number, if at_start { Kind::Leading { small } } else { Kind::Trailing { small } }, year_like));
    }
    found.sort_by_key(|&(_, _, year_like)| year_like);
    let mut out: Vec<(i64, Kind)> = Vec::new();
    for (number, kind, _) in found {
        if !out.iter().any(|&(n, _)| n == number) {
            out.push((number, kind));
        }
    }
    out
}

fn zone_score(zone: Zone, edge: f64, near: f64, order: f64, body: f64) -> f64 {
    match zone {
        Zone::Top | Zone::Bottom => edge,
        Zone::NearTop | Zone::NearBottom => near,
        Zone::OrderTop | Zone::OrderBottom => order,
        Zone::Body => body,
    }
}

/// How folio-like a number is, where it stands and how its line reads (Text-Fidelity's `candidate_score`).
fn candidate(number: i64, kind: Kind, zone: Zone, text: &str, y: Option<f64>, line: usize) -> Candidate {
    let p = patterns();
    let year_like = (1800..=2099).contains(&number) && (text.trim().chars().count() <= 16 || p.year.is_match(text));
    let small = matches!(kind, Kind::Leading { small: true } | Kind::Trailing { small: true });
    let folio = folio_shape(text, kind);
    let note = note_shape(text, kind);
    let title = title_folio_shape(text, kind, zone);
    let head = zone.top() && p.report_head.is_match(text);
    let citation_like = !small && !head && ((note && !title) || p.citation.is_match(text) || p.bracket_year.is_match(text));
    let mut base = match kind {
        Kind::Isolated => zone_score(zone, 1.0, 0.88, 0.82, 0.2),
        Kind::Supplement => zone_score(zone, 0.96, 0.86, 0.80, 0.12),
        Kind::HeaderTrailing => match zone {
            Zone::Top => 0.98,
            Zone::Bottom => 0.92,
            Zone::NearTop => 0.88,
            Zone::NearBottom => 0.82,
            Zone::OrderTop => 0.82,
            Zone::OrderBottom => 0.76,
            Zone::Body => 0.12,
        },
        _ if small => zone_score(zone, 0.92, 0.80, 0.76, 0.12),
        _ if folio => zone_score(zone, 0.86, 0.74, 0.70, 0.12),
        _ => match zone {
            Zone::Top => 0.34,
            Zone::Bottom | Zone::NearTop => 0.28,
            Zone::NearBottom | Zone::OrderTop => 0.24,
            Zone::OrderBottom => 0.22,
            Zone::Body => 0.08,
        },
    };
    if year_like {
        base *= 0.18;
    }
    if note && !title && kind != Kind::Isolated {
        base *= 0.25;
        if zone.edge() && matches!(kind, Kind::Trailing { .. }) {
            base = base.max(0.24);
        }
    } else if citation_like && kind != Kind::Isolated {
        base *= 0.55;
    }
    Candidate { number, kind, zone, score: (base * 10_000.0).round() / 10_000.0, year_like, citation_like, y, line, text: text.to_owned(), context: text.to_owned(), running: false }
}

/// The candidates one line offers.
fn line_candidates(text: &str, zone: Zone, y: Option<f64>, line: usize) -> Vec<Candidate> {
    let raw: Vec<(i64, Kind)> = match isolated_number(text) {
        Some(number) => vec![(number, Kind::Isolated)],
        None if zone.edge() => embedded_numbers(text),
        None => Vec::new(),
    };
    raw.into_iter().map(|(number, kind)| candidate(number, kind, zone, text, y, line)).filter(|c| c.score >= 0.18).collect()
}

/// A candidate strong enough to anchor a sequence (`credible_sequence_anchor`, at the 0.55 floor).
fn sequence_anchor(c: &Candidate) -> bool {
    const FLOOR: f64 = 0.55;
    if c.year_like {
        return false;
    }
    match c.kind {
        Kind::Run => return true,
        Kind::Isolated => return c.score >= FLOOR,
        Kind::Supplement => return folio_shape(&c.text, c.kind) && c.score >= FLOOR,
        _ if c.running => return c.score >= FLOOR,
        _ => {}
    }
    if c.citation_like {
        return false;
    }
    if matches!(c.kind, Kind::Leading { .. }) && c.zone.bottom() {
        let after = leading_number(c.text.trim()).map_or("", |(_, end)| &c.text.trim()[end..]);
        if after.chars().any(|ch| !ch.is_whitespace()) && !patterns().header_context.is_match(after) {
            return false;
        }
    }
    if note_shape(&c.text, c.kind) {
        return false;
    }
    if title_folio_shape(&c.text, c.kind, c.zone) {
        return c.zone.edge() && c.score >= 0.30;
    }
    folio_shape(&c.text, c.kind) && c.zone.edge() && c.score >= FLOOR
}

/// A candidate a short document's single numbered page may stand on (`credible`).
fn credible(c: &Candidate) -> bool {
    const FLOOR: f64 = 0.55;
    if c.year_like {
        return false;
    }
    if c.kind == Kind::Isolated {
        return c.score >= FLOOR;
    }
    if c.citation_like || note_shape(&c.text, c.kind) {
        return false;
    }
    if c.zone.top() { c.score >= FLOOR } else { c.score >= 0.70 }
}

fn structural(c: &Candidate) -> bool {
    c.kind == Kind::Run || c.zone.edge()
        && (matches!(c.kind, Kind::Counter | Kind::HeaderTrailing)
            || title_folio_shape(&c.text, c.kind, c.zone)
            || (matches!(c.kind, Kind::Trailing { .. }) && c.score >= 0.24)
            || (c.zone.top() && folio_shape(&c.text, c.kind)))
}

/// A page's lines as rows: the pieces a reader splits one printed line into, left to right. A row's cells are its pieces
/// set apart by a gap wider than a line is high (a running head's folio, title and citation); pieces closer than that read
/// as one. Two pieces share a row where each overlaps half the other's
/// height.
struct Row {
    cells: Vec<EdgeLine>,
    /// Each cell's pieces as the reader split them ("PAGE 6" beside "Mis à jour : 10 févr. 2012"), each with its index
    /// among the page's lines.
    pieces: Vec<Vec<(usize, EdgeLine)>>,
    /// The row whole, its cells three spaces apart.
    whole: EdgeLine,
}

fn rows(page: &PageEvidence) -> Vec<Row> {
    let mut lines: Vec<(usize, &EdgeLine)> = page.lines.iter().enumerate().filter(|(_, l)| !l.text.trim().is_empty()).collect();
    lines.sort_by(|(_, a), (_, b)| a.rect[1].total_cmp(&b.rect[1]).then(a.rect[0].total_cmp(&b.rect[0])));
    let mut rows: Vec<Vec<(usize, &EdgeLine)>> = Vec::new();
    for (at, line) in lines {
        let same_row = |(_, other): &(usize, &EdgeLine)| {
            let overlap = line.rect[3].min(other.rect[3]) - line.rect[1].max(other.rect[1]);
            let height = (line.rect[3] - line.rect[1]).min(other.rect[3] - other.rect[1]).max(0.1);
            overlap >= height * 0.5
        };
        match rows.iter_mut().find(|row| row.iter().any(same_row)) {
            Some(row) => row.push((at, line)),
            None => rows.push(vec![(at, line)]),
        }
    }
    let bounds = |pieces: &[(usize, &EdgeLine)]| pieces.iter().fold([f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY], |r, (_, l)| {
        [r[0].min(l.rect[0]), r[1].min(l.rect[1]), r[2].max(l.rect[2]), r[3].max(l.rect[3])]
    });
    let mut out: Vec<Row> = rows.into_iter().map(|mut row| {
        row.sort_by(|(_, a), (_, b)| a.rect[0].total_cmp(&b.rect[0]));
        let mut groups: Vec<Vec<(usize, &EdgeLine)>> = Vec::new();
        for (at, piece) in row.iter().copied() {
            let wide = groups.last().and_then(|group| group.last()).is_none_or(|(_, before)| {
                let height = (piece.rect[3] - piece.rect[1]).min(before.rect[3] - before.rect[1]).max(0.1);
                piece.rect[0] - before.rect[2] > height
            });
            if wide { groups.push(vec![(at, piece)]) } else { groups.last_mut().unwrap().push((at, piece)) }
        }
        let cells: Vec<EdgeLine> = groups.iter().map(|group| EdgeLine {
            text: group.iter().map(|(_, l)| l.text.trim()).collect::<Vec<_>>().join(" "),
            rect: bounds(group),
        }).collect();
        let whole = EdgeLine { text: cells.iter().map(|cell| cell.text.as_str()).collect::<Vec<_>>().join("   "), rect: bounds(&row) };
        let pieces = groups.iter().map(|group| group.iter().map(|&(at, piece)| (at, piece.clone())).collect()).collect();
        Row { cells, pieces, whole }
    }).collect();
    out.sort_by(|a, b| ((a.whole.rect[1] * 10.0).round()).total_cmp(&((b.whole.rect[1] * 10.0).round())).then(a.whole.rect[0].total_cmp(&b.whole.rect[0])));
    out
}

/// One page's candidates, best first, its structural head and foot numbers kept and the rest up to eight.
fn page_candidates(observed: Option<&str>, page: Option<&PageEvidence>) -> Vec<Candidate> {
    let mut candidates: Vec<Candidate> = Vec::new();
    if let Some(number) = observed.and_then(|l| l.parse::<i64>().ok()).filter(|n| (1..=99_999).contains(n)) {
        candidates.push(Candidate { number, kind: Kind::Isolated, zone: Zone::Top, score: 1.0, year_like: false, citation_like: false, y: None, line: 1_000_000, text: number.to_string(), context: number.to_string(), running: false });
    }
    if let Some(page) = page {
        let rows = rows(page);
        let (count, height) = (rows.len(), page.height.max(1.0));
        for (index, row) in rows.iter().enumerate() {
            let line = &row.whole;
            let y = (line.rect[1] + line.rect[3]) / 2.0 / height;
            let zone = if y <= 0.28 { Zone::Top }
                else if y >= 0.72 { Zone::Bottom }
                else if y <= 0.38 { Zone::NearTop }
                else if y >= 0.62 { Zone::NearBottom }
                else if index < 12 { Zone::OrderTop }
                else if index + 12 >= count { Zone::OrderBottom }
                else { Zone::Body };
            // each cell as its own line, its row the context it reads beside
            // a bare number opening a foot row that goes on in text, not a running title, is a note's number, read with its
            // text (the row), as `credible_sequence_anchor` reads a foot line opening on a number
            let note = |k: usize| zone.bottom() && k + 1 < row.cells.len() && isolated_number(&row.cells[k].text).is_some()
                && row.cells[k + 1..].iter().any(|cell| cell.text.chars().any(char::is_alphabetic))
                && !row.cells[k + 1..].iter().any(|cell| patterns().header_context.is_match(&cell.text));
            for (k, cell) in row.cells.iter().enumerate().filter(|(k, _)| !note(*k)) {
                candidates.extend(line_candidates(&cell.text, zone, Some(y), index + 1).into_iter().map(|mut c| {
                    c.context = line.text.clone();
                    c
                }));
                // where a cell joins several of the reader's pieces, a page counter among them reads alone ("PAGE 6"
                // beside "Mis à jour : 10 févr. 2012")
                let group = &row.pieces[k];
                if group.len() > 1 {
                    for (_, piece) in group {
                        candidates.extend(line_candidates(&piece.text, zone, Some(y), index + 1).into_iter().filter(|c| c.kind == Kind::Counter).map(|mut c| {
                            c.context = line.text.clone();
                            c
                        }));
                    }
                }
            }
        }
    }
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.year_like.cmp(&b.year_like)).then(a.citation_like.cmp(&b.citation_like))
        .then(a.line.cmp(&b.line)).then(a.number.cmp(&b.number)));
    if candidates.len() <= 8 {
        return candidates;
    }
    let (mut kept, rest): (Vec<Candidate>, Vec<Candidate>) = candidates.into_iter().partition(structural);
    let target = kept.len().max(8);
    kept.extend(rest.into_iter().take(target - kept.len()));
    kept
}

/// One shift's support (a row of `ranked_delta_sequences`).
struct Sequence {
    shift: i64,
    /// Its anchors by PDF page index, in page order.
    anchors: Vec<(usize, Candidate)>,
    span: usize,
    score: f64,
    support: f64,
    coverage: f64,
    span_ratio: f64,
    header_anchors: usize,
    y_span: Option<f64>,
}

impl Sequence {
    fn new(shift: i64, mut anchors: Vec<(usize, Candidate)>, total: usize) -> Sequence {
        anchors.sort_by_key(|(at, _)| *at);
        let span = anchors.last().unwrap().0 - anchors[0].0 + 1;
        let score = (anchors.iter().map(|(_, c)| c.score).sum::<f64>() * 10_000.0).round() / 10_000.0;
        let total = total.max(1) as f64;
        let (coverage, span_ratio) = (anchors.len() as f64 / total, span as f64 / total);
        let edge = anchors.iter().filter(|(_, c)| c.zone.edge()).count() as f64;
        let isolated = anchors.iter().filter(|(_, c)| c.kind == Kind::Isolated).count() as f64;
        // a number beside a running head's title (`candidate_has_running_header_context`), a cell of its row included
        let header_anchors = anchors.iter().filter(|(_, c)| (c.kind != Kind::Isolated || c.context != c.text) && c.zone.edge() && patterns().header_context.is_match(&c.context)).count();
        let ys: Vec<f64> = anchors.iter().filter_map(|(_, c)| c.y).collect();
        let y_span = (ys.len() >= 2).then(|| ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - ys.iter().cloned().fold(f64::INFINITY, f64::min));
        let mut support = score + anchors.len() as f64 * 1.75 + span_ratio * 2.5 + coverage * 1.5 + edge * 0.15 + isolated * 0.10;
        if let Some(y_span) = y_span {
            support += (0.10 - y_span).max(0.0) * 4.0;
        }
        Sequence { shift, anchors, span, score, support, coverage, span_ratio, header_anchors, y_span }
    }

    /// Its anchors sit at different heights on their pages, and too thinly for that not to matter.
    fn inconsistent(&self, total: usize, required: usize, required_span: usize) -> bool {
        if self.y_span.is_none_or(|y_span| y_span <= 0.10) {
            return false;
        }
        let count = self.anchors.len();
        if (self.coverage >= 0.95 || total <= 4) && count >= required && self.span >= required_span {
            return false;
        }
        let isolated = self.anchors.iter().all(|(_, c)| matches!(c.kind, Kind::Isolated | Kind::Supplement));
        if total <= 20 && count >= 3 && self.span >= (total as f64 * 0.65).ceil() as usize && isolated {
            return false;
        }
        let dense = (total as f64 * 0.40).ceil() as usize;
        !(count >= (required + 2).max(dense) && self.coverage >= 0.40 && self.span >= required_span.max(dense))
    }

    /// Why it cannot number the document (`sequence_fails_hard_requirements`).
    fn fails(&self, total: usize, required: usize, required_span: usize) -> bool {
        let count = self.anchors.len();
        count < required
            || (self.header_anchors >= count && count < 4 && self.anchors.iter().take(12).all(|(_, c)| c.score < 0.55))
            || (self.span < required_span && count < (required + 1).max((total as f64 * 0.10).ceil() as usize))
            || self.inconsistent(total, required, required_span)
            || (total > 20 && count < 4 && self.coverage < 0.10)
    }

    /// The pages it numbers, first anchor to the end; the pages before its first anchor only as
    /// `sequence_allows_leading_projection` lets them, or where they would number below 1.
    fn projection(&self, total: usize) -> std::ops::Range<usize> {
        let first = self.anchors[0].0;
        let prefix: Vec<i64> = (0..first).map(|at| at as i64 + 1 + self.shift).collect();
        if prefix.is_empty() || (prefix.iter().all(|&label| label > 0) && self.allows_leading(&prefix)) {
            return 0..total;
        }
        // a volume's first page, its 1 unprinted, after front matter of no number in this sequence: the single missing
        // first page the sequence infers (`sequence_allows_leading_projection`)
        if first >= 1 && prefix[first - 1] == 1 {
            return first - 1..total;
        }
        first..total
    }

    fn allows_leading(&self, labels: &[i64]) -> bool {
        if labels.len() == 1 && labels[0] == 1 {
            return true;
        }
        let sample = &self.anchors[0].1;
        let count = self.anchors.len();
        let estimated = if self.coverage > 0.0 { (count as f64 / self.coverage).round() as usize } else { 0 };
        if labels.len() == 1 && sample.kind != Kind::Supplement && count >= 2 && estimated <= 4 && self.coverage >= 0.50 {
            return true;
        }
        if count < 3 || self.span < 3 {
            return false;
        }
        if count >= 4.max(8.min(labels.len())) && self.span >= count && self.coverage >= 0.35 {
            return true;
        }
        labels.len() <= 4 && sample.kind != Kind::Supplement && sample.zone.top() && sample.score >= 0.86
    }

    /// It numbers the document from 1 to 3 on and stops well short of `start`, a citation's first page
    /// (`local_article_folio_conflicts_with_scraped_first_page`).
    fn local_count_below(&self, total: usize, start: i64) -> bool {
        let (first, last) = self.printed(self.projection(total));
        first <= 3 && start >= 10 && start - first >= 8 && last < start
    }

    /// It is the document's own count of its pages (`sequence_is_local_article_folio_run`): from 1 or 2, no further than
    /// the document runs, its anchors (half its pages or more) nearly all bare numbers at the foot.
    fn local_article_count(&self, total: usize) -> bool {
        let (first, last) = self.printed(self.projection(total));
        let count = self.anchors.len();
        let threshold = (count as f64 * 0.80).ceil() as usize;
        first <= 2 && last <= total as i64 + 2 && count >= 6.max((total as f64 * 0.50).ceil() as usize)
            && self.anchors.iter().filter(|(_, c)| c.kind == Kind::Isolated).count() >= threshold
            && self.anchors.iter().filter(|(_, c)| c.zone.bottom()).count() >= threshold
    }

    /// The printed numbers it gives `range`.
    fn printed(&self, range: std::ops::Range<usize>) -> (i64, i64) {
        (range.start as i64 + 1 + self.shift, range.end as i64 + self.shift)
    }
}

/// Rule 2 (Text-Fidelity's sequence fit, as the journals' page maps were made): each page's candidates that can anchor a
/// sequence vote for a shift (printed number less PDF page); the shifts are ranked by their anchors, span and support;
/// the best numbers the document from its first anchor to the end (the pages before it as the leading rule allows), unless
/// it fails the hard requirements (too few anchors, too short a span, anchors at inconsistent heights, too sparse for a
/// long document), a close competitor leaves the numbering in doubt, or a running head's sequence outranks a local
/// footer count. A short document with one numbered page takes that page's shift.
fn sequence(bindings: &mut [PdfPageBinding], observed: &[Option<String>], pages: &[PageEvidence], starts: &[u32]) {
    let total = bindings.len();
    if total == 0 {
        return;
    }
    let (candidates, ranked) = ranked_sequences(observed, pages);
    let required = if total == 1 { 1 } else { 2 };
    // a PDF binding documents of their own numberings (a book of authorities, a volume's plates): each part its own run
    let parts = parts(&ranked, total);
    if parts.len() >= 2 {
        let roman = roman_runs(bindings, pages);
        let mut numbered = vec![false; total];
        for part in &parts {
            let (first, last) = (part.anchors[0].0, part.anchors.last().unwrap().0);
            for at in first..=last {
                numbered[at] = true;
                let number = at as i64 + 1 + part.shift;
                let binding = &mut bindings[at];
                if number < 1 || roman[at] {
                    continue;
                }
                if binding.label.as_deref() != Some(number.to_string().as_str()) {
                    binding.label = Some(number.to_string());
                    binding.source = Some("sequence".to_owned());
                    binding.status = "resolved".to_owned();
                }
            }
        }
        for (at, binding) in bindings.iter_mut().enumerate() {
            if !numbered[at] && binding.source.as_deref() == Some("detected") && binding.label.as_deref().is_some_and(|label| label.parse::<i64>().is_ok()) {
                binding.label = None;
                binding.source = None;
                binding.status = "unknown".to_owned();
            }
        }
        return;
    }
    let required_span = if total <= 4 { 1 } else { 2.max((total as f64 * 0.15).ceil() as usize) };
    let chosen = choose(&ranked, total, required, required_span);
    // the PDF's own count of its pages (its 1 on the PDF's first page) says nothing the PDF's page does not: where another
    // numbering is printed through the document, that numbering is the page's
    let chosen = match chosen {
        Some(best) if best.shift == 0 => ranked.iter().find(|row| row.shift != 0 && row.anchors.len() >= 3 && !row.fails(total, required, required_span)
            // a numbering of its own: printed on half the pages the count is, at one place on them
            && row.anchors.len() * 2 >= best.anchors.len() && row.y_span.is_none_or(|y| y <= 0.10)).or(Some(best)),
        other => other,
    };
    let roman = roman_runs(bindings, pages);
    let (shift, range) = match chosen {
        Some(best) => {
            let competitors = ranked.iter().filter(|row| !std::ptr::eq(*row, best)).take(7)
                .any(|row| row.anchors.len() >= required && row.anchors.len() + 1 >= best.anchors.len() && row.support >= best.support * 0.85);
            // the citations' first pages arbitrate (`metadata_arbitrated_fit`): a sequence that is plainly the document's
            // own count of its pages where the report's numbering is cited gives way to one that is not
            let hypotheses: Vec<&Sequence> = std::iter::once(best).chain(ranked.iter().filter(|row| !std::ptr::eq(*row, best))).take(20).collect();
            let rejected = |row: &Sequence| !starts.is_empty() && starts.iter().all(|&start| row.local_count_below(total, i64::from(start)));
            let compatible: Vec<&Sequence> = hypotheses.iter().copied()
                .filter(|row| !rejected(row) && (std::ptr::eq(*row, best) || !row.fails(total, required, required_span))).collect();
            // a sequence that opens the document at a cited first page (after a cover or two) is the citation's
            // (`scraped_range_matches_page_map_endpoints`)
            let opens_at_start = |row: &Sequence| {
                let range = row.projection(total);
                (range.start..range.end.min(range.start + 3)).any(|at| starts.iter().any(|&start| at as i64 + 1 + row.shift == i64::from(start)))
            };
            let opening = (!opens_at_start(best)).then(|| compatible.iter().copied().find(|row| opens_at_start(row))).flatten();
            let arbitrated = !starts.is_empty() && !compatible.is_empty()
                && (opening.is_some() || rejected(best) || compatible.len() < hypotheses.len() || competitors);
            let best = if arbitrated { opening.unwrap_or(compatible[0]) } else { best };
            // two numberings printed through the document (a journal's pages over a repository's count, Bates numbers
            // over a filing's pages): each is printed, and the page takes one: not the PDF's own count of its pages, nor
            // the document's own count at the foot where there is another; else the one its heads print, else the better
            // supported
            let best = if competitors && !arbitrated {
                let rivals: Vec<&Sequence> = std::iter::once(best).chain(ranked.iter().filter(|row| !std::ptr::eq(*row, best)).take(7)
                    .filter(|row| row.anchors.len() >= required && row.anchors.len() + 1 >= best.anchors.len() && row.support >= best.support * 0.85
                        && !row.fails(total, required, required_span))).collect();
                let heads = |row: &Sequence| row.anchors.iter().filter(|(_, c)| c.zone.top()).count();
                let chosen = rivals.iter().copied().max_by(|a, b| (a.shift != 0).cmp(&(b.shift != 0)).then((!a.local_article_count(total)).cmp(&!b.local_article_count(total)))
                    .then(heads(a).cmp(&heads(b))).then(a.anchors.len().cmp(&b.anchors.len())).then(a.support.total_cmp(&b.support)));
                match chosen {
                    Some(chosen) if rivals.len() >= 2 => chosen,
                    _ => return unconfirmed(bindings, 0..0),
                }
            } else {
                best
            };
            if rejected(best) {
                return unconfirmed(bindings, 0..0);
            }
            (best.shift, best.projection(total))
        }
        // no sequence forms (too few numbered pages): each folio as read stands
        None => match short_single_anchor(&candidates, total) {
            Some(shift) => (shift, 0..total),
            None => match cited_anchor(&candidates, total, starts) {
                Some(found) => found,
                None => return,
            },
        },
    };
    unconfirmed(bindings, range.clone());
    // a page the sequence's own folio stands on is the sequence's, a roman numeral there notwithstanding (a clause's "(i)")
    let anchored: std::collections::HashSet<usize> = ranked.iter().find(|row| row.shift == shift).map(|row| row.anchors.iter().map(|(at, _)| *at).collect()).unwrap_or_default();
    for (at, ((binding, number), in_roman)) in bindings.iter_mut().zip(1i64..).zip(roman).enumerate().skip(range.start).take(range.len()) {
        let number = number + shift;
        // the sequence decides a page whose folio as read and the PDF's own label disagree
        if number < 1 || (in_roman && !anchored.contains(&at)) {
            continue;
        }
        if binding.label.as_deref() != Some(number.to_string().as_str()) {
            binding.label = Some(number.to_string());
            binding.source = Some("sequence".to_owned());
            binding.status = "resolved".to_owned();
        }
    }
}

/// What a folio may print beside its number: the page word, its count of pages ("of 30", "/ 30", "sur 30").
fn folio_dress() -> &'static Regex {
    static DRESS: OnceLock<Regex> = OnceLock::new();
    DRESS.get_or_init(|| Regex::new(r"(?i)\b(?:p\s?a\s?g\s?e|pg|pp?)\b\.?|\b(?:of|de|sur)\s+\d{1,5}\b|/\s*\d{1,5}").unwrap())
}

/// A run of numbers a document prints page after page, one higher each page, at one place on the page (`chains`).
struct Chain {
    /// "E" of "E-162", "5" of a chapter's "5-1", "ABC" of "ABC0001234"; empty for a bare number.
    prefix: String,
    /// The printed form's kind: "letter" (a prefix and a dash), "serial" (letters and digits run on), "chapter", "bare".
    form: &'static str,
    /// The digits' printed width where it is zero-padded ("000208": 6), else 0.
    pad: usize,
    /// Each page of the run: its index, its number, and the number's height and line.
    pages: Vec<(usize, i64, f64, String)>,
}

/// The runs of numbers a document prints, whatever their shape or place: every number on a page ("12", "Page | 12",
/// "- 12 -", "E-162", "000208", "5-1", a folio misread around), chained to the number one higher on the next page (or two
/// higher two pages on, past a misread or a blank) with the same prefix at the same place (its height, and its side or
/// the facing page's, or both centred). A chain of three pages on the outermost rows, or of five within the page where a
/// text's own numbers stand, is a run. A year repeats on a running head rather than climbs, a transcript's line numbers
/// start again on each page, and a note's numbers seldom climb one a page at one place.
fn chains(pages: &[PageEvidence]) -> Vec<Chain> {
    static TOKEN: OnceLock<Regex> = OnceLock::new();
    static NOT_A_PREFIX: OnceLock<Regex> = OnceLock::new();
    let token = TOKEN.get_or_init(|| Regex::new(r"(?:\b(?P<letters>[A-Za-z]{1,6})(?P<dash>\s?-\s?)?|\b(?P<chapter>\d{1,3})\s?-\s?)?\b(?P<digits>\d{1,8})\b").unwrap());
    // words a number follows that name no numbering of pages: the page word itself, a statute's or a list's
    let not_a_prefix = NOT_A_PREFIX.get_or_init(|| Regex::new(r"(?i)^(?:page|pages|pg|p|pp|no|nos|vol|art|arts|s|ss|ch|c|para|paras|at|n|fn|r|rr|tab|id|v)$").unwrap());
    struct Token {
        value: i64,
        prefix: String,
        form: &'static str,
        pad: usize,
        y: f64,
        x: f64,
        outer: bool,
        text: String,
    }
    let tokens: Vec<Vec<Token>> = pages.iter().map(|page| {
        let (height, width) = (page.height.max(1.0), page.width.max(1.0));
        let mut found = Vec::new();
        let rows = rows(page);
        let count = rows.len();
        for (index, row) in rows.iter().enumerate() {
            let y = (row.whole.rect[1] + row.whole.rect[3]) / 2.0 / height;
            let outer = index < 2 || index + 2 >= count;
            for (_, piece) in row.pieces.iter().flatten() {
                let x = (piece.rect[0] + piece.rect[2]) / 2.0 / width;
                let text = piece.text.trim();
                for capture in token.captures_iter(text) {
                    // a number standing as a folio stands: its line alone, in a folio's dress ("- 12 -", "Page 12", "12 of
                    // 30", "E-162"), or opening or closing a head's or foot's outermost row (a running head's "48 UBC LAW
                    // REVIEW"); a number within a sentence or a note's text is the text's
                    let whole = capture.get(0).unwrap();
                    let rest = format!("{} {}", &text[..whole.start()], &text[whole.end()..]);
                    let dress = folio_dress().replace_all(&rest, "");
                    let alone = dress.chars().all(|c| !c.is_alphanumeric());
                    let edge = outer && (whole.start() == 0 || whole.end() == text.len() || text[..whole.start()].chars().all(|c| !c.is_alphanumeric())
                        || text[whole.end()..].chars().all(|c| !c.is_alphanumeric()));
                    if !alone && !edge {
                        continue;
                    }
                    let digits = &capture["digits"];
                    let Ok(value) = digits.parse::<i64>() else { continue };
                    let letters = capture.name("letters").map(|m| m.as_str()).filter(|l| !not_a_prefix.is_match(l));
                    let (prefix, form) = match (letters, capture.name("chapter")) {
                        (Some(letters), _) if capture.name("dash").is_some() => (letters.to_uppercase(), "letter"),
                        (Some(letters), _) if capture.name("dash").is_none() && digits.len() >= 4 => (letters.to_uppercase(), "serial"),
                        (_, Some(chapter)) => (chapter.as_str().to_owned(), "chapter"),
                        _ => (String::new(), "bare"),
                    };
                    let pad = if digits.len() >= 4 && digits.starts_with('0') { digits.len() } else { 0 };
                    // a year is no page: it repeats on a running head, and a text's citations may climb one a page
                    let year = form == "bare" && pad == 0 && digits.len() == 4 && (1800..=2099).contains(&value);
                    if value >= 1 && !year {
                        found.push(Token { value, prefix, form, pad, y, x, outer, text: piece.text.clone() });
                    }
                }
            }
        }
        found
    }).collect();
    let total = pages.len();
    let mut next: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    let mut linked: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    for p in 0..total {
        for (i, t) in tokens[p].iter().enumerate() {
            for gap in 1..=2usize {
                let q = p + gap;
                if q >= total {
                    break;
                }
                let best = tokens[q].iter().enumerate()
                    .filter(|(j, u)| u.value == t.value + gap as i64 && u.prefix == t.prefix && u.form == t.form && (u.y - t.y).abs() <= 0.03
                        && ((u.x - t.x).abs() <= 0.08 || (u.x - (1.0 - t.x)).abs() <= 0.08) && !linked.contains(&(q, *j)))
                    .min_by(|(_, a), (_, b)| (a.y - t.y).abs().total_cmp(&(b.y - t.y).abs()));
                if let Some((j, _)) = best {
                    next.insert((p, i), (q, j));
                    linked.insert((q, j));
                    break;
                }
            }
        }
    }
    let mut out = Vec::new();
    for p in 0..total {
        for i in 0..tokens[p].len() {
            if linked.contains(&(p, i)) {
                continue;
            }
            let mut chain = vec![(p, i)];
            while let Some(&after) = next.get(chain.last().unwrap()) {
                chain.push(after);
            }
            let outer = chain.iter().all(|&(q, j)| tokens[q][j].outer);
            if chain.len() < if outer { 3 } else { 5 } {
                continue;
            }
            let first = &tokens[p][i];
            out.push(Chain {
                prefix: first.prefix.clone(),
                form: first.form,
                pad: chain.iter().map(|&(q, j)| tokens[q][j].pad).max().unwrap_or(0),
                pages: chain.iter().map(|&(q, j)| (q, tokens[q][j].value, tokens[q][j].y, tokens[q][j].text.clone())).collect(),
            });
        }
    }
    out
}

/// The fit's candidates from the runs (`chains`) a page number may be: bare numbers and lettered pages ("E-162"), not a
/// chapter's ("5-1") or a record's serial ("ABC0001234"), which are numberings of their own (`numberings`).
fn folio_runs(pages: &[PageEvidence]) -> Vec<Vec<Candidate>> {
    let mut out: Vec<Vec<Candidate>> = (0..pages.len()).map(|_| Vec::new()).collect();
    for chain in chains(pages).into_iter().filter(|chain| matches!(chain.form, "bare" | "letter") && chain.pad == 0) {
        for (q, value, y, text) in chain.pages {
            let zone = if y <= 0.28 { Zone::Top } else if y >= 0.72 { Zone::Bottom } else if y <= 0.5 { Zone::NearTop } else { Zone::NearBottom };
            if (1..=99_999).contains(&value) {
                out[q].push(Candidate { number: value, kind: Kind::Run, zone, score: 1.0, year_like: false, citation_like: false, y: Some(y),
                    line: 1_000_000, context: text.clone(), text, running: false });
            }
        }
    }
    out
}

/// Every numbering the pages print besides the page number, as printed (`PdfPageBinding::numbers`): each run of numbers
/// (`chains`), its runs at one prefix and shift taken together, over the pages from its first to its last; a number the
/// page's own label already gives is not repeated. The kind is the printed form's: a lettered page "exhibit", a serial or
/// zero-padded number "bates", a chapter's page "chapter", the PDF's own count "pdf", another count of pages "other". A page
/// the fit leaves without a number takes the strongest typed one (exhibit, bates, chapter) as its label.
fn numberings(bindings: &mut [PdfPageBinding], pages: &[PageEvidence]) {
    let mut runs: HashMap<(String, &'static str, usize, i64), (usize, usize, usize)> = HashMap::new();
    for chain in chains(pages) {
        let shift = chain.pages[0].1 - chain.pages[0].0 as i64 - 1;
        let (first, last) = (chain.pages[0].0, chain.pages.last().unwrap().0);
        let entry = runs.entry((chain.prefix.clone(), chain.form, chain.pad, shift)).or_insert((first, last, 0));
        *entry = (entry.0.min(first), entry.1.max(last), entry.2 + chain.pages.len());
    }
    let mut runs: Vec<_> = runs.into_iter().filter(|(_, (_, _, count))| *count >= 3).collect();
    runs.sort_by(|a, b| b.1 .2.cmp(&a.1 .2));
    for ((prefix, form, pad, shift), (first, last, _)) in runs {
        let kind = match form {
            "letter" => "exhibit",
            "serial" => "bates",
            "chapter" => "chapter",
            _ if pad > 0 => "bates",
            _ if shift == 0 => "pdf",
            _ => "other",
        };
        for (at, binding) in bindings.iter_mut().enumerate().take(last + 1).skip(first) {
            let value = at as i64 + 1 + shift;
            if value < 1 {
                continue;
            }
            let digits = if pad > 0 { format!("{value:0pad$}") } else { value.to_string() };
            let label = match form {
                "letter" | "chapter" => format!("{prefix}-{digits}"),
                "serial" => format!("{prefix}{digits}"),
                _ => digits,
            };
            let own = binding.label.as_deref().is_some_and(|own| own == label || (form == "bare" && pad == 0 && own == value.to_string()));
            // a page the fit leaves without a number takes the strongest typed numbering it prints; a bare count is one the
            // fit already weighed against its own
            if binding.label.is_none() && !matches!(kind, "pdf" | "other") {
                binding.label = Some(label);
                binding.source = Some("sequence".to_owned());
                binding.status = "resolved".to_owned();
                continue;
            }
            if !own && !binding.numbers.iter().any(|number| number.label == label) {
                binding.numbers.push(PrintedNumber { kind: kind.to_owned(), label });
            }
        }
    }
}

/// Each page's candidates (`observed`: what each prints as read), and the shifts their anchors support, best first
/// (`ranked_delta_sequences`).
fn ranked_sequences(observed: &[Option<String>], pages: &[PageEvidence]) -> (Vec<Vec<Candidate>>, Vec<Sequence>) {
    let total = observed.len();
    let mut candidates: Vec<Vec<Candidate>> = observed.iter().enumerate().map(|(at, observed)| page_candidates(observed.as_deref(), pages.get(at))).collect();
    for (page, chained) in candidates.iter_mut().zip(folio_runs(pages)) {
        page.extend(chained);
    }
    running_heads(&mut candidates);
    let valid = |shift: i64| {
        let skipped = (1..=total as i64).filter(|page| page + shift <= 0).count();
        // pages before the sequence's 1 (an issue's front matter, a repository's cover) are left unnumbered
        total as i64 + shift <= 99_999 && (skipped == 0 || total - skipped >= 2)
    };
    let mut by_shift: HashMap<i64, Vec<(usize, Candidate)>> = HashMap::new();
    for (at, page) in candidates.iter().enumerate() {
        let mut best: HashMap<i64, &Candidate> = HashMap::new();
        for c in page.iter().filter(|c| sequence_anchor(c)) {
            let shift = c.number - (at as i64 + 1);
            let priority = |c: &Candidate| (u8::from(c.kind == Kind::Supplement), c.score, -(c.line as i64));
            if best.get(&shift).is_none_or(|prior| priority(c).partial_cmp(&priority(prior)) == Some(std::cmp::Ordering::Greater)) {
                best.insert(shift, c);
            }
        }
        for (shift, c) in best {
            if valid(shift) {
                by_shift.entry(shift).or_default().push((at, c.clone()));
            }
        }
    }
    let mut ranked: Vec<Sequence> = by_shift.into_iter().map(|(shift, anchors)| Sequence::new(shift, anchors, total)).collect();
    ranked.sort_by(|a, b| b.anchors.len().cmp(&a.anchors.len()).then(b.span_ratio.total_cmp(&a.span_ratio))
        .then(b.support.total_cmp(&a.support)).then(b.score.total_cmp(&a.score)).then(a.shift.abs().cmp(&b.shift.abs())));
    (candidates, ranked)
}

/// A running head's folio: a number opening or closing a head's or foot's line whose words the same edge of three pages
/// or more prints alike (a fifth of the pages at least: a journal's title on its even pages, "40 Underneath the Golden
/// Boy"), the numbers beside them three or more and not one repeated. Its words are the head's, not a sentence's, so the
/// number is a folio's whatever the words name.
fn running_heads(candidates: &mut [Vec<Candidate>]) {
    static WORD: OnceLock<Regex> = OnceLock::new();
    let word = WORD.get_or_init(|| Regex::new(r"[^\W\d_]{3,}").unwrap());
    let key = |c: &Candidate| -> Option<(String, bool)> {
        if !matches!(c.kind, Kind::Leading { small: false } | Kind::Trailing { small: false }) || !c.zone.edge() || c.year_like {
            return None;
        }
        let words: Vec<String> = word.find_iter(&c.text).map(|w| w.as_str().to_lowercase()).collect();
        (words.len() >= 2).then(|| (words.join(" "), c.zone.top()))
    };
    let mut seen: HashMap<(String, bool), (std::collections::HashSet<usize>, std::collections::HashSet<i64>)> = HashMap::new();
    for (at, page) in candidates.iter().enumerate() {
        for c in page {
            if let Some(k) = key(c) {
                let entry = seen.entry(k).or_default();
                entry.0.insert(at);
                entry.1.insert(c.number);
            }
        }
    }
    let total = candidates.len();
    for page in candidates.iter_mut() {
        for c in page.iter_mut() {
            let Some(k) = key(c) else { continue };
            let (on, numbers) = &seen[&k];
            if on.len() >= 3 && on.len() * 5 >= total && numbers.len() * 10 >= on.len() * 9 {
                c.running = true;
                c.citation_like = false;
                c.score = c.score.max(zone_score(c.zone, 0.86, 0.74, 0.70, 0.12));
            }
        }
    }
}

/// A roman numeral's value ("xiv" is 14), up to 100 (`roman_page_value`, `ROMAN_PAGE_MAX`).
fn roman_value(label: &str) -> Option<i64> {
    static ROMAN: OnceLock<Regex> = OnceLock::new();
    let roman = ROMAN.get_or_init(|| Regex::new(r"(?i)^(?:c|xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})$").unwrap());
    let label = label.trim();
    if label.is_empty() || !roman.is_match(label) {
        return None;
    }
    let digit = |c: char| match c.to_ascii_lowercase() { 'i' => 1, 'v' => 5, 'x' => 10, 'l' => 50, 'c' => 100, _ => 0 };
    let values: Vec<i64> = label.chars().map(digit).collect();
    let total: i64 = values.iter().enumerate().map(|(at, &v)| if values.get(at + 1).is_some_and(|&next| next > v) { -v } else { v }).sum();
    (1..=100).contains(&total).then_some(total)
}

/// The roman runs (Text-Fidelity's roman fit, safe mode): roman folios read on pages whose numbers agree with their
/// places (two or more at one shift) number the pages from the first to the last of them; a roman folio read alone, or
/// at odds with its neighbours, is no page number. Returns which pages a roman run holds.
fn roman_runs(bindings: &mut [PdfPageBinding], pages: &[PageEvidence]) -> Vec<bool> {
    // each page's roman folios: the one it prints as read (`printed`), and a cell of its head or foot that is a numeral alone, between
    // marks perhaps ("(iv)", "- xii -"; `roman_candidate_text`)
    let mut by_shift: HashMap<i64, Vec<usize>> = HashMap::new();
    let mut heights: HashMap<(i64, usize), f64> = HashMap::new();
    for (at, binding) in bindings.iter().enumerate() {
        let mut values: Vec<i64> = binding.label.as_deref().filter(|_| binding.source.as_deref() == Some("detected")).and_then(roman_value).into_iter().collect();
        if let Some(page) = pages.get(at) {
            let height = page.height.max(1.0);
            for row in rows(page) {
                let y = (row.whole.rect[1] + row.whole.rect[3]) / 2.0 / height;
                if y > 0.28 && y < 0.72 {
                    continue;
                }
                for cell in &row.cells {
                    let bare: String = cell.text.trim().chars().filter(|c| !EDGE_PUNCT.contains(*c)).collect();
                    if let Some(value) = roman_value(&bare) {
                        values.push(value);
                        heights.insert((value - at as i64 - 1, at), y);
                    }
                }
            }
        }
        values.sort_unstable();
        values.dedup();
        for value in values {
            by_shift.entry(value - at as i64 - 1).or_default().push(at);
        }
    }
    let mut held = vec![false; bindings.len()];
    // a run's numerals stand close (`safe_roman_anchor_segment`): two more than two pages apart begin another run
    let mut segments: Vec<(i64, Vec<usize>)> = Vec::new();
    for (shift, pages) in by_shift {
        let mut segment: Vec<usize> = Vec::new();
        for at in pages {
            if segment.last().is_some_and(|&before| at - before > 3) {
                segments.push((shift, std::mem::take(&mut segment)));
            }
            segment.push(at);
        }
        segments.push((shift, segment));
    }
    // a run's numerals stand where a folio stands, at one height (a clause's "(i)", "(ii)" wanders down its pages)
    let steady = |shift: i64, held: &[usize]| {
        let ys: Vec<f64> = held.iter().filter_map(|&at| heights.get(&(shift, at)).copied()).collect();
        ys.len() < 2 || ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - ys.iter().cloned().fold(f64::INFINITY, f64::min) <= 0.10
    };
    let mut runs: Vec<(usize, usize, i64, bool)> = segments.into_iter().filter(|(shift, held)| held.len() >= 2 && steady(*shift, held))
        .map(|(shift, held)| (held[0], *held.last().unwrap(), shift, held.iter().all(|&at| !bindings[at].label.as_deref().is_some_and(|l| l.chars().any(|c| c.is_ascii_uppercase())))))
        .collect();
    runs.sort_by_key(|&(first, last, _, _)| std::cmp::Reverse(last - first));
    for (first, last, shift, lower) in runs {
        if held[first..=last].iter().any(|&h| h) {
            continue;
        }
        for at in first..=last {
            held[at] = true;
            let value = at as i64 + 1 + shift;
            if value < 1 {
                continue;
            }
            let numeral = roman_numeral(value);
            let numeral = if lower { numeral } else { numeral.to_uppercase() };
            if bindings[at].label.as_deref() != Some(numeral.as_str()) {
                bindings[at].label = Some(numeral);
                bindings[at].source = Some("sequence".to_owned());
                bindings[at].status = "resolved".to_owned();
            }
        }
    }
    for (at, binding) in bindings.iter_mut().enumerate() {
        if !held[at] && binding.source.as_deref() == Some("detected") && binding.label.as_deref().and_then(roman_value).is_some() {
            binding.label = None;
            binding.source = None;
            binding.status = "unknown".to_owned();
        }
    }
    held
}

fn roman_numeral(mut value: i64) -> String {
    let mut out = String::new();
    for (arabic, roman) in [(100, "c"), (90, "xc"), (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")] {
        while value >= arabic {
            out.push_str(roman);
            value -= arabic;
        }
    }
    out
}

/// Where a sequence was weighed and ruled on, a folio read on a page it does not number (`numbered`) is one the fit did not
/// bear out (a local count, a rival in doubt): no page number.
fn unconfirmed(bindings: &mut [PdfPageBinding], numbered: std::ops::Range<usize>) {
    for (at, binding) in bindings.iter_mut().enumerate() {
        if !numbered.contains(&at) && binding.source.as_deref() == Some("detected") && binding.label.as_deref().is_some_and(|label| label.parse::<i64>().is_ok()) {
            binding.label = None;
            binding.source = None;
            binding.status = "unknown".to_owned();
        }
    }
}

/// The shifts a document's pages support, best first, each with its anchors (PDF page, number, the line it was read
/// from): what the fit weighed, for a person checking a page map.
#[doc(hidden)]
pub fn weighed(pages: &[PageEvidence]) -> Vec<(i64, Vec<(usize, i64, String)>)> {
    let (_, ranked) = ranked_sequences(&observed(pages), pages);
    ranked.into_iter().map(|row| (row.shift, row.anchors.into_iter().map(|(at, c)| (at + 1, c.number, c.text)).collect())).collect()
}

/// Whether a page's lines print a number in a folio's shape.
pub fn prints_a_folio(page: &PageEvidence) -> bool {
    page_candidates(printed(page).map(|(label, _)| label).as_deref(), Some(page)).iter().any(sequence_anchor)
}

/// Each page's candidates (number, how it is printed, its zone, its score, whether it anchors a sequence, its line), for a
/// person checking a page map.
#[doc(hidden)]
pub fn candidates_of(pages: &[PageEvidence]) -> Vec<Vec<String>> {
    observed(pages).iter().zip(pages).map(|(observed, page)| page_candidates(observed.as_deref(), Some(page)).iter()
        .map(|c| format!("{} {:?} {:?} {:.2} {} | {}", c.number, c.kind, c.zone, c.score, if sequence_anchor(c) { "ANCHOR" } else { "-" }, c.text))
        .collect()).collect()
}

/// The parts the fit finds (shift, first and last PDF page, anchors), for a person checking a page map.
#[doc(hidden)]
pub fn parts_of(pages: &[PageEvidence]) -> Vec<(i64, usize, usize, usize)> {
    let (_, ranked) = ranked_sequences(&observed(pages), pages);
    parts(&ranked, pages.len()).iter().map(|part| (part.shift, part.anchors[0].0 + 1, part.anchors.last().unwrap().0 + 1, part.anchors.len())).collect()
}

/// The pages whose head and foot a caller may read again: each page whose head and foot (its top and bottom 15%) show no
/// number in a folio's shape, and every page where the best sequence is the document's
/// own count of its pages at the foot (`local_article_count`), as a repository's copy prints over the journal's page whose
/// folio is in the image of its head. A page the body's numbers seem to number is no page whose folio was read.
pub fn read_again(pages: &[PageEvidence]) -> Vec<bool> {
    let (_, ranked) = ranked_sequences(&observed(pages), pages);
    let local = ranked.first().is_some_and(|best| best.local_article_count(pages.len()));
    pages.iter().map(|page| {
        let band = PageEvidence {
            lines: page.lines.iter().filter(|line| line.rect[3] <= page.height * EDGE_BAND || line.rect[1] >= page.height * (1.0 - EDGE_BAND)).cloned().collect(),
            ..page.clone()
        };
        local || !prints_a_folio(&band)
    }).collect()
}

/// The parts of a PDF that binds documents of their own numberings: each shift's anchors split where another shift has
/// anchors on three or more of the pages between them; a stretch of three anchors or more, at consistent heights or dense
/// over its pages, is a part, the
/// strongest first, each over pages no stronger part holds. Fewer than two parts: the PDF is one numbering.
fn parts(ranked: &[Sequence], total: usize) -> Vec<Sequence> {
    let mut at_page: Vec<Vec<i64>> = vec![Vec::new(); total];
    for row in ranked {
        for (at, _) in &row.anchors {
            at_page[*at].push(row.shift);
        }
    }
    let mut stretches: Vec<Sequence> = Vec::new();
    // a run of its own: three folios, or two clean ones on facing pages (a short document whose PDF skips a page)
    let holds = |anchors: &[(usize, Candidate)]| anchors.len() >= 3
        || anchors.len() == 2 && anchors[1].0 == anchors[0].0 + 1 && anchors.iter().all(|(_, c)| c.score >= 0.88);
    for row in ranked.iter().filter(|row| row.anchors.len() >= 2) {
        let mut stretch: Vec<(usize, Candidate)> = Vec::new();
        for (at, c) in &row.anchors {
            if let Some((before, _)) = stretch.last() {
                // another numbering printed on three or more of the pages between (not misread numbers agreeing by chance)
                let mut others: HashMap<i64, usize> = HashMap::new();
                for p in *before + 1..*at {
                    for &shift in at_page[p].iter().filter(|&&shift| shift != row.shift) {
                        *others.entry(shift).or_default() += 1;
                    }
                }
                if others.values().any(|&n| n >= 3) {
                    stretches.push(Sequence::new(row.shift, std::mem::take(&mut stretch), total));
                }
            }
            stretch.push((*at, c.clone()));
        }
        if !stretch.is_empty() {
            stretches.push(Sequence::new(row.shift, stretch, total));
        }
    }
    // anchors at one height, or dense enough over their stretch that their heights do not matter (`sequence_position_
    // inconsistency_is_disqualifying`: 40% of the pages)
    stretches.retain(|stretch| holds(&stretch.anchors)
        && (stretch.y_span.is_none_or(|y| y <= 0.10) || stretch.anchors.len() as f64 >= stretch.span as f64 * 0.40));
    stretches.sort_by(|a, b| b.anchors.len().cmp(&a.anchors.len()).then(b.support.total_cmp(&a.support)));
    let mut held = vec![false; total];
    let mut parts: Vec<Sequence> = Vec::new();
    for stretch in stretches {
        // a stretch reaching into a stronger part's pages (a misread folio there agreeing with it) keeps its own pages
        let own: Vec<(usize, Candidate)> = stretch.anchors.iter().filter(|(at, _)| !held[*at]).cloned().collect();
        if !holds(&own) {
            continue;
        }
        let (first, last) = (own[0].0, own.last().unwrap().0);
        if held[first..=last].iter().any(|&h| h) {
            continue;
        }
        held[first..=last].iter_mut().for_each(|h| *h = true);
        parts.push(Sequence::new(stretch.shift, own, total));
    }
    // a PDF of parts is the parts: together they cover most of it (a few short runs of list or note numbers do not)
    let covered: usize = parts.iter().map(|part| part.anchors.last().unwrap().0 - part.anchors[0].0 + 1).sum();
    if covered * 2 < total {
        return Vec::new();
    }
    parts
}

/// The ranked sequence that numbers the document, if one may (`fit_sequence`'s gates).
fn choose(ranked: &[Sequence], total: usize, required: usize, required_span: usize) -> Option<&Sequence> {
    let mut best = ranked.first()?;
    let count = best.anchors.len();
    if count < required || (best.span < required_span && count < (required + 1).max((total as f64 * 0.10).ceil() as usize)) {
        return None;
    }
    if best.header_anchors >= count && count < 4 && best.anchors.iter().take(12).all(|(_, c)| c.score < 0.55) {
        return None;
    }
    if best.inconsistent(total, required, required_span) {
        best = ranked[1..].iter().find(|row| !row.fails(total, required, required_span))?;
    }
    // a footer count of the article's own pages (1, 2, 3 ... at the foot) gives way to a running head's sequence
    let first = best.printed(best.projection(total)).0;
    if best.local_article_count(total) {
        if let Some(header) = ranked[1..].iter().find(|row| {
            !row.fails(total, required, required_span)
                && row.header_anchors >= 2.max((total as f64 * 0.20).ceil() as usize)
                && row.span >= required_span.max((total as f64 * 0.40).ceil() as usize)
                && row.printed(row.projection(total)).0 > first + 2
        }) {
            best = header;
        }
    }
    if total > 20 && best.anchors.len() < 4 && best.coverage < 0.10 {
        return None;
    }
    Some(best)
}

/// A document of four pages or fewer with one page that prints a credible number (`fit_short_single_anchor_sequence`).
fn short_single_anchor(candidates: &[Vec<Candidate>], total: usize) -> Option<i64> {
    if total == 0 || total > 4 {
        return None;
    }
    let numbered: Vec<(usize, Vec<&Candidate>)> = candidates.iter().enumerate()
        .map(|(at, page)| (at, page.iter().filter(|c| credible(c)).collect::<Vec<_>>()))
        .filter(|(_, page)| !page.is_empty()).collect();
    let [(at, page)] = numbered.as_slice() else { return None };
    let mut page: Vec<&Candidate> = page.iter().copied().filter(|c| c.score >= 0.55 && c.kind != Kind::Counter && c.kind != Kind::HeaderTrailing).collect();
    page.sort_by(|a, b| b.score.total_cmp(&a.score).then(b.zone.top().cmp(&a.zone.top())).then((b.kind == Kind::Isolated).cmp(&(a.kind == Kind::Isolated))).then(a.number.cmp(&b.number)));
    let top = page.first()?;
    if page.iter().filter(|c| (c.score - top.score).abs() < 0.05).count() > 1 || (*at + 1 == total && total > 2) {
        return None;
    }
    let shift = top.number - (*at as i64 + 1);
    (1 + shift >= 1 && total as i64 + shift <= 99_999).then_some(shift)
}

/// Where too few pages print a number for a sequence to form, a citation's first page and one folio printed at its
/// distance agree (the cited first page falls on one of the document's first three pages, at or before the folio): the
/// document from that first page takes their numbering. Every folio that agrees with a start must give one numbering.
fn cited_anchor(candidates: &[Vec<Candidate>], total: usize, starts: &[u32]) -> Option<(i64, std::ops::Range<usize>)> {
    let mut found: Option<(i64, usize)> = None;
    for (at, page) in candidates.iter().enumerate() {
        for c in page.iter().filter(|c| c.kind != Kind::Run && sequence_anchor(c)) {
            let shift = c.number - (at as i64 + 1);
            for &start in starts {
                let first = i64::from(start) - shift - 1;
                if (0..3.min(total as i64)).contains(&first) && first <= at as i64 {
                    if found.is_some_and(|(prior, _)| prior != shift) {
                        return None;
                    }
                    found = Some((shift, first as usize));
                }
            }
        }
    }
    found.map(|(shift, first)| (shift, first..total))
}

/// Rule 4's anchors: each page printing a citation's first page numbers its run, up to the next anchor, where most of
/// the run's numbers agree with that numbering (`observed`: what each page prints as read).
fn anchor(bindings: &mut [PdfPageBinding], observed: &[Option<String>], starts: &[u32]) {
    let anchors: Vec<(usize, u32)> = bindings.iter().zip(observed).enumerate()
        .filter(|(_, (binding, _))| binding.status == "resolved")
        .filter_map(|(index, (_, observed))| observed.as_deref()?.parse::<u32>().ok().filter(|n| *n >= 1 && starts.contains(n)).map(|n| (index, n)))
        .collect();
    for (k, &(start, first)) in anchors.iter().enumerate() {
        let end = anchors.get(k + 1).map_or(bindings.len(), |&(next, _)| next);
        let expected = |index: usize| (first as usize + index - start).to_string();
        let numbered: Vec<usize> = (start..end).filter(|&index| bindings[index].label.is_some()).collect();
        let agreeing = numbered.iter().filter(|&&index| bindings[index].label.as_deref() == Some(expected(index).as_str())).count();
        if agreeing * 2 <= numbered.len() {
            continue;
        }
        for (index, binding) in bindings.iter_mut().enumerate().take(end).skip(start) {
            if binding.label.as_deref() != Some(expected(index).as_str()) {
                binding.label = Some(expected(index));
                binding.source = Some("reporter".to_owned());
                binding.status = "resolved".to_owned();
            }
        }
    }
}

/// The folio an opening page (the first three) prints at its edge where none was read (Text-Fidelity's edge-folio rule):
/// a citation's first page (`starts`) alone at the start or end of a line in the page's outer bands or outermost rows, at
/// the page's side, the citation parser's veto (`is_citation`) taking out a line that cites it. Recognition can space a
/// folio ("74 1" for 741); only a known start, alone at a page's side, repairs it. Each with the line printing it.
fn margin_labels(labels: &mut [Option<(String, usize)>], pages: &[PageEvidence], starts: &[u32], is_citation: &dyn Fn(&str, &str) -> bool) {
    static SPACED: OnceLock<Regex> = OnceLock::new();
    static LEADING: OnceLock<Regex> = OnceLock::new();
    static TRAILING: OnceLock<Regex> = OnceLock::new();
    static YEAR: OnceLock<Regex> = OnceLock::new();
    static NAMED: OnceLock<Regex> = OnceLock::new();
    let spaced = SPACED.get_or_init(|| Regex::new(r"^\s*\d{1,4}(?:\s+\d){1,4}\s*$").unwrap());
    let leading = LEADING.get_or_init(|| Regex::new(r#"^[\s\[({.,;:'"_\-=–—]*(\d+)"#).unwrap());
    let trailing = TRAILING.get_or_init(|| Regex::new(r#"(\d+)[\s\])}.,;:'"_\-=–—]*$"#).unwrap());
    let year = YEAR.get_or_init(|| Regex::new(r"^\s*\[\s*(?:18|19|20)\d{2}\s*\]").unwrap());
    let named = NAMED.get_or_init(|| Regex::new(r"(?i)\b(?:vol(?:ume)?|issue|no|para|paragraph)\.?\s*$").unwrap());
    let is_start = |digits: &str| digits.parse::<u32>().is_ok_and(|n| starts.contains(&n));
    for (index, page) in pages.iter().enumerate().take(3) {
        if labels[index].is_some() || page.lines.is_empty() {
            continue;
        }
        let top = page.lines.iter().map(|line| line.rect[3]).fold(f64::INFINITY, f64::min);
        let bottom = page.lines.iter().map(|line| line.rect[1]).fold(f64::NEG_INFINITY, f64::max);
        let mut candidates = std::collections::BTreeMap::new();
        for (at, line) in page.lines.iter().enumerate() {
            let (rect, width, height) = (line.rect, page.width, page.height);
            if !(rect[3] <= height * 0.14 || rect[1] >= height * 0.86 || rect[1] <= top || rect[3] >= bottom) {
                continue;
            }
            let value = line.text.trim();
            if value.chars().count() > 140 {
                continue;
            }
            if spaced.is_match(value) {
                let joined: String = value.split_whitespace().collect();
                if is_start(&joined) && (rect[0] <= width * 0.2 || rect[2] >= width * 0.8) && !is_citation(value, &joined) {
                    candidates.entry(joined).or_insert(at);
                }
            }
            let at_start = leading.captures(value).map(|found| found.get(1).unwrap()).filter(|digits| digits.as_str().len() <= 5);
            let at_end = trailing.captures(value).map(|found| found.get(1).unwrap())
                .filter(|digits| digits.as_str().len() <= 5 && !value[..digits.start()].ends_with(|c: char| c.is_ascii_digit()));
            for (found, at_line_start) in [(at_start, true), (at_end, false)] {
                let Some(digits) = found else { continue };
                let number = digits.as_str();
                if !is_start(number)
                    || (at_line_start && year.is_match(value))
                    || (if at_line_start { rect[0] > width * 0.2 } else { rect[2] < width * 0.8 })
                    || is_citation(value, number)
                    || (!at_line_start && named.is_match(&value[..digits.start()]))
                {
                    continue;
                }
                candidates.entry(number.to_owned()).or_insert(at);
            }
        }
        if candidates.len() == 1 {
            labels[index] = candidates.into_iter().next();
        }
    }
}
