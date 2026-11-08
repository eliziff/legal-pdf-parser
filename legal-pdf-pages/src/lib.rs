//! A PDF's page map: each PDF page's printed number, from what the pages print (`PageEvidence`):
//!
//! 1. What a page prints is its number (the structure stage's folio, as read).
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
//!
//! The engine's `pdfPageMap`, the backend's page map and the tools all read this.
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;

/// What one PDF page tells of its number.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PageEvidence {
    /// The number the page prints, as the structure stage read its folio.
    #[serde(default)]
    pub observed: Option<String>,
    /// The PDF's own label for the page (its /PageLabels).
    #[serde(default)]
    pub embedded: Option<String>,
    /// The page's size, and its lines with their boxes ([x0, top, x1, bottom], display space): an opening page's
    /// alone are read (`margin_labels`).
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
#[serde(rename_all = "camelCase")]
pub struct PdfPageBinding {
    /// The PDF page, from 1.
    pub pdf_page: u32,
    /// The number the page prints as the structure read it, where it read one.
    pub observed: Option<String>,
    pub label: Option<String>,
    /// "detected" (what the page prints), "sequence" (its place in the folio sequence), "embedded" (the PDF's own
    /// label), "reporter" (a citation's first page at its distance); none where the page has no number.
    pub source: Option<String>,
    /// "resolved", "ambiguous" (the page prints one number and the PDF labels it another), or "unknown".
    pub status: String,
}

/// The page map (see the crate).
pub fn page_map(pages: &[PageEvidence], starts: &[u32], is_citation: &dyn Fn(&str, &str) -> bool) -> Vec<PdfPageBinding> {
    let mut observed: Vec<Option<String>> = pages.iter().map(|page| page.observed.as_deref().map(str::trim).filter(|l| !l.is_empty()).map(str::to_owned)).collect();
    if !starts.is_empty() && !observed.iter().flatten().any(|label| label.parse::<u32>().is_ok_and(|n| starts.contains(&n))) {
        margin_labels(&mut observed, pages, starts, is_citation);
    }
    let mut bindings = resolve(&observed, &pages.iter().map(|page| page.embedded.clone()).collect::<Vec<_>>());
    sequence(&mut bindings, pages, starts);
    anchor(&mut bindings, starts);
    bindings
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
            PdfPageBinding { pdf_page: index as u32 + 1, observed: detected, label, source: source.map(str::to_owned), status: status.to_owned() }
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
    /// The line's middle, as a share of the page's height; none for the folio the structure read.
    y: Option<f64>,
    line: usize,
    text: String,
}

const EDGE_PUNCT: &str = " \t\r\n[](){}.,;:'\"`~_-=\u{2010}\u{2011}\u{2012}\u{2013}\u{2014}\u{2015}";

struct Patterns {
    leading: Regex,
    trailing: Regex,
    supplement: Regex,
    counter: Regex,
    header_trailing: Regex,
    month_or_volume: Regex,
    citation: Regex,
    bracket_year: Regex,
    year: Regex,
    edge_space: Regex,
    dot_leader: Regex,
    header_separator: Regex,
    header_context: Regex,
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
            supplement: Regex::new(&format!("^[{punct}]*S(\\d{{1,4}})[{punct}]*$")).unwrap(),
            counter: Regex::new(r"^\s*(\d{1,4})\s*[|/]\s*(\d{1,4})\s*$").unwrap(),
            header_trailing: Regex::new(r"(?i)\b(?:vol(?:ume)?|journal|review|law|legal|rights|quarterly|yearbook|annuaire|bulletin|appeal|forum)\b.{0,80}?(?:[%|•·–—]\s*|\s{2,})(\d{1,5})\s*$").unwrap(),
            month_or_volume: Regex::new(r"(?i)\b(vol(?:ume)?|issue|iss\.?|no\.?|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t)?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|page)\b").unwrap(),
            citation: Regex::new(r"(?i)\b(v\.|vs\.|s\.c\.r\.|d\.l\.r\.|w\.w\.r\.|c\.c\.c\.|para\.|art\.|rev\.|l\.j\.)\b").unwrap(),
            bracket_year: Regex::new(r"\[[12][0-9]{3}\]").unwrap(),
            year: Regex::new(r"\b[12][0-9]{3}\b").unwrap(),
            edge_space: Regex::new(r"\s{2,}|\t").unwrap(),
            dot_leader: Regex::new(r"(?:\.\s*){2,}").unwrap(),
            header_separator: Regex::new(r"^[\.\)]?\s+\S").unwrap(),
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
    let compact: String = text.trim().chars().filter(|c| !EDGE_PUNCT.contains(*c)).collect();
    if compact.is_empty() || compact.len() > 5 || !compact.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    compact.parse().ok().filter(|n| (1..=5000).contains(n))
}

fn supplement_number(text: &str) -> Option<i64> {
    patterns().supplement.captures(text.trim())?[1].parse().ok().filter(|n| (1..=5000).contains(n))
}

/// The line has a folio's shape: the number set apart from the rest by a gap, dot leaders, or a running head's title.
fn folio_shape(text: &str, kind: Kind) -> bool {
    let p = patterns();
    let value = text.trim();
    match kind {
        Kind::Isolated => true,
        Kind::Counter => p.counter.is_match(&collapse(value)),
        Kind::HeaderTrailing => p.header_trailing.is_match(&collapse(value)),
        Kind::Supplement => supplement_number(value).is_some(),
        Kind::Leading { .. } => leading_number(value).is_some_and(|(_, end)| {
            let after = &value[end..];
            p.edge_space.find(after).is_some_and(|found| found.start() == 0)
                || (p.header_separator.is_match(after) && p.header_context.is_match(after))
        }),
        Kind::Trailing { .. } => trailing_number(value).is_some_and(|(start, _)| {
            let before = &value[..start];
            p.edge_space.is_match(tail(before, 18))
                || p.dot_leader.is_match(tail(before, 32))
                || (p.text_before.is_match(before) && p.header_context.is_match(before))
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
    if let Some(found) = p.counter.captures(&value) {
        let (current, total): (i64, i64) = (found[1].parse().unwrap_or(0), found[2].parse().unwrap_or(0));
        if current > 0 && current <= total + 10 && total <= 5000 {
            return vec![(current, Kind::Counter)];
        }
    }
    let mut found: Vec<(i64, Kind, bool)> = Vec::new();
    if let Some(number) = p.header_trailing.captures(&value).and_then(|c| c[1].parse::<i64>().ok()).filter(|n| (1..=5000).contains(n)) {
        found.push((number, Kind::HeaderTrailing, false));
    }
    if let Some(number) = supplement_number(raw) {
        found.push((number, Kind::Supplement, false));
    }
    for (at_start, span) in [(true, leading_number(raw)), (false, trailing_number(raw))] {
        let Some(span) = span else { continue };
        let number: i64 = raw[span.0..span.1].parse().unwrap_or(0);
        if !(1..=5000).contains(&number) {
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
    let citation_like = !small && ((note && !title) || p.citation.is_match(text) || p.bracket_year.is_match(text));
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
    Candidate { number, kind, zone, score: (base * 10_000.0).round() / 10_000.0, year_like, citation_like, y, line, text: text.to_owned() }
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
        Kind::Isolated => return c.score >= FLOOR,
        Kind::Supplement => return folio_shape(&c.text, c.kind) && c.score >= FLOOR,
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
    c.zone.edge()
        && (matches!(c.kind, Kind::Counter | Kind::HeaderTrailing)
            || title_folio_shape(&c.text, c.kind, c.zone)
            || (matches!(c.kind, Kind::Trailing { .. }) && c.score >= 0.24)
            || (c.zone.top() && folio_shape(&c.text, c.kind)))
}

/// A page's lines as rows: the pieces a reader splits one printed line into (a running head's folio, title and volume;
/// a note's number and its text) joined left to right with a gap between them, as a line of the page reads whole. Two
/// pieces share a row where each overlaps half the other's height.
fn rows(page: &PageEvidence) -> Vec<EdgeLine> {
    let mut lines: Vec<&EdgeLine> = page.lines.iter().filter(|l| !l.text.trim().is_empty()).collect();
    lines.sort_by(|a, b| a.rect[1].total_cmp(&b.rect[1]).then(a.rect[0].total_cmp(&b.rect[0])));
    let mut rows: Vec<Vec<&EdgeLine>> = Vec::new();
    for line in lines {
        let same_row = |other: &&EdgeLine| {
            let overlap = line.rect[3].min(other.rect[3]) - line.rect[1].max(other.rect[1]);
            let height = (line.rect[3] - line.rect[1]).min(other.rect[3] - other.rect[1]).max(0.1);
            overlap >= height * 0.5
        };
        match rows.iter_mut().find(|row| row.iter().any(same_row)) {
            Some(row) => row.push(line),
            None => rows.push(vec![line]),
        }
    }
    let mut out: Vec<EdgeLine> = rows.into_iter().map(|mut row| {
        row.sort_by(|a, b| a.rect[0].total_cmp(&b.rect[0]));
        let rect = row.iter().fold([f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY], |r, l| {
            [r[0].min(l.rect[0]), r[1].min(l.rect[1]), r[2].max(l.rect[2]), r[3].max(l.rect[3])]
        });
        EdgeLine { text: row.iter().map(|l| l.text.trim()).collect::<Vec<_>>().join("   "), rect }
    }).collect();
    out.sort_by(|a, b| ((a.rect[1] * 10.0).round()).total_cmp(&((b.rect[1] * 10.0).round())).then(a.rect[0].total_cmp(&b.rect[0])));
    out
}

/// One page's candidates, best first, its structural head and foot numbers kept and the rest up to eight.
fn page_candidates(binding: &PdfPageBinding, page: Option<&PageEvidence>) -> Vec<Candidate> {
    let mut candidates: Vec<Candidate> = Vec::new();
    if let Some(number) = binding.observed.as_deref().and_then(|l| l.parse::<i64>().ok()).filter(|n| (1..=5000).contains(n)) {
        candidates.push(Candidate { number, kind: Kind::Isolated, zone: Zone::Top, score: 1.0, year_like: false, citation_like: false, y: None, line: 1_000_000, text: number.to_string() });
    }
    if let Some(page) = page {
        let lines = rows(page);
        let lines: Vec<&EdgeLine> = lines.iter().collect();
        let (count, height) = (lines.len(), page.height.max(1.0));
        for (index, line) in lines.iter().enumerate() {
            let y = (line.rect[1] + line.rect[3]) / 2.0 / height;
            let zone = if y <= 0.28 { Zone::Top }
                else if y >= 0.72 { Zone::Bottom }
                else if y <= 0.38 { Zone::NearTop }
                else if y >= 0.62 { Zone::NearBottom }
                else if index < 12 { Zone::OrderTop }
                else if index + 12 >= count { Zone::OrderBottom }
                else { Zone::Body };
            candidates.extend(line_candidates(&line.text, zone, Some(y), index + 1));
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
        let header_anchors = anchors.iter().filter(|(_, c)| c.kind != Kind::Isolated && c.zone.edge() && patterns().header_context.is_match(&c.text)).count();
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
fn sequence(bindings: &mut [PdfPageBinding], pages: &[PageEvidence], starts: &[u32]) {
    let total = bindings.len();
    if total == 0 {
        return;
    }
    let (candidates, ranked) = ranked_sequences(bindings, pages);
    let required = if total == 1 { 1 } else { 2 };
    let required_span = if total <= 4 { 1 } else { 2.max((total as f64 * 0.15).ceil() as usize) };
    let chosen = choose(&ranked, total, required, required_span);
    let roman = roman_runs(bindings);
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
            if (competitors && !arbitrated) || rejected(best) {
                return unconfirmed(bindings, 0..0);
            }
            (best.shift, best.projection(total))
        }
        None => match short_single_anchor(&candidates, total) {
            Some(shift) => (shift, 0..total),
            None => return unconfirmed(bindings, 0..0),
        },
    };
    unconfirmed(bindings, range.clone());
    for ((binding, number), in_roman) in bindings.iter_mut().zip(1i64..).zip(roman).skip(range.start).take(range.len()) {
        let number = number + shift;
        if number < 1 || binding.status == "ambiguous" || in_roman {
            continue;
        }
        if binding.label.as_deref() != Some(number.to_string().as_str()) {
            binding.label = Some(number.to_string());
            binding.source = Some("sequence".to_owned());
            binding.status = "resolved".to_owned();
        }
    }
}

/// Each page's candidates, and the shifts their anchors support, best first (`ranked_delta_sequences`).
fn ranked_sequences(bindings: &[PdfPageBinding], pages: &[PageEvidence]) -> (Vec<Vec<Candidate>>, Vec<Sequence>) {
    let total = bindings.len();
    let candidates: Vec<Vec<Candidate>> = bindings.iter().enumerate().map(|(at, binding)| page_candidates(binding, pages.get(at))).collect();
    let valid = |shift: i64| {
        let skipped = (1..=total as i64).filter(|page| page + shift <= 0).count();
        // pages before the sequence's 1 (an issue's front matter, a repository's cover) are left unnumbered
        total as i64 + shift <= 5000 && (skipped == 0 || total - skipped >= 2)
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
fn roman_runs(bindings: &mut [PdfPageBinding]) -> Vec<bool> {
    let mut by_shift: HashMap<i64, Vec<usize>> = HashMap::new();
    for (at, binding) in bindings.iter().enumerate() {
        if let Some(value) = binding.label.as_deref().filter(|_| binding.source.as_deref() == Some("detected")).and_then(roman_value) {
            by_shift.entry(value - at as i64 - 1).or_default().push(at);
        }
    }
    let mut held = vec![false; bindings.len()];
    let mut runs: Vec<(usize, usize, i64, bool)> = by_shift.into_iter().filter(|(_, pages)| pages.len() >= 2)
        .map(|(shift, pages)| (pages[0], *pages.last().unwrap(), shift, pages.iter().all(|&at| bindings[at].label.as_deref().is_some_and(|l| l.chars().all(|c| c.is_ascii_lowercase())))))
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

/// A folio read on a page the sequence does not number (`numbered`) stands on that one reading alone: the sequence fit
/// numbers what it numbers, and a reading it does not bear out is no page number (the page keeps it as `observed`).
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
    let observed: Vec<Option<String>> = pages.iter().map(|page| page.observed.clone()).collect();
    let (_, ranked) = ranked_sequences(&resolve(&observed, &vec![None; pages.len()]), pages);
    ranked.into_iter().map(|row| (row.shift, row.anchors.into_iter().map(|(at, c)| (at + 1, c.number, c.text)).collect())).collect()
}

/// The pages whose head and foot a caller may read again: a page printing no number in a folio's shape there (nor
/// a folio the structure read), and every page where the best sequence is the document's own count of its pages at
/// the foot (`local_article_count`), as a repository's copy prints over the journal's page whose folio is in the
/// image of its head.
pub fn read_again(pages: &[PageEvidence]) -> Vec<bool> {
    let observed: Vec<Option<String>> = pages.iter().map(|page| page.observed.clone()).collect();
    let bindings = resolve(&observed, &vec![None; pages.len()]);
    let (candidates, ranked) = ranked_sequences(&bindings, pages);
    let local = ranked.first().is_some_and(|best| best.local_article_count(pages.len()));
    candidates.iter().map(|page| local || !page.iter().any(sequence_anchor)).collect()
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
    (1 + shift >= 1 && total as i64 + shift <= 5000).then_some(shift)
}

/// Rule 4's anchors: each page printing a citation's first page numbers its run, up to the next anchor, where most of
/// the run's numbers agree with that numbering.
fn anchor(bindings: &mut [PdfPageBinding], starts: &[u32]) {
    let anchors: Vec<(usize, u32)> = bindings.iter().enumerate()
        .filter(|(_, binding)| binding.status == "resolved")
        .filter_map(|(index, binding)| binding.observed.as_deref()?.parse::<u32>().ok().filter(|n| *n >= 1 && starts.contains(n)).map(|n| (index, n)))
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
/// folio ("74 1" for 741); only a known start, alone at a page's side, repairs it.
fn margin_labels(labels: &mut [Option<String>], pages: &[PageEvidence], starts: &[u32], is_citation: &dyn Fn(&str, &str) -> bool) {
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
        let mut candidates = std::collections::BTreeSet::new();
        for line in &page.lines {
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
                    candidates.insert(joined);
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
                candidates.insert(number.to_owned());
            }
        }
        if candidates.len() == 1 {
            labels[index] = candidates.into_iter().next();
        }
    }
}
