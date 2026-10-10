//! The note lanes of region postcorrection, read over the whole document: which lines are notes,
//! from the evidence a page carries and the evidence the document's other pages carry with it.
//! Ported from the Open Access Journals Database's `region_postcorrection.py` (the post-footnote
//! reroute and the note band closure) and `note_band_witness.py` (the printed separator's band and
//! the font-role strands).
//!
//! Every lane proposes; the claims are applied together once all have spoken, an earlier lane's
//! claim reserving its line from the later ones, so a weaker witness never overrides a stronger.
//! Each lane needs two independent witnesses before it retypes a line, and a declined line is a
//! miss, never a misplacement. The document's own state carries across pages: the note label
//! ladder, the layout prior over every page whose notes end their band, and the bridge from one
//! page's unfinished note to the next page's band.

use legal_pdf_core::{line_font_size, model::Page};
use legal_pdf_support::protected_citation_spans;
use regex::Regex;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::OnceLock;

const SEPARATOR_Y_TOLERANCE: f64 = 0.006;
const MAX_NUMERIC_LABEL_STEP: i64 = 5;
const SYMBOL_MARKER_CHARS: &str = "*†‡§¶";
const CLOSURE_MIN_PRIOR_PAGES: usize = 2;
const CLOSURE_MIN_PRIOR_RATIO: f64 = 0.5;
const CLOSURE_MAX_TAIL_LINES: usize = 12;

/// A line, by its page and its place among the page's lines.
type LineRef = (usize, usize);

fn marker_start_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^\s*(?:\(?\d{1,3}\)?[.)]?|[*†‡§¶]+)\s+\S").expect("marker regex")
    })
}

fn numeric_head_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\(?(\d{1,3})\)?[.)]?").expect("numeric head regex"))
}

fn bare_numeric_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\(?(\d{1,3})\)?[.)]?$").expect("bare numeric regex"))
}

fn leading_number_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^\s*\(?(\d{1,3})\)?[.)]?(?:\s|$)").expect("leading number regex")
    })
}

fn page_label_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^\s*(?:[-\u{2013}\u{2014}]\s*)?(?:\d{1,4}|[ivxlcdmIVXLCDM]{1,12})(?:\s*[-\u{2013}\u{2014}])?\s*$")
            .expect("page label regex")
    })
}

fn toc_leader_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?:\.\s){3,}\.?|\.{4,}").expect("leader regex"))
}

fn trailing_page_number_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\s\d{1,4}$").expect("trailing page number regex"))
}

/// The line as marker evidence judges it: superscript labels ("²² See") read as plain digits.
fn folded(text: &str) -> String {
    text.chars()
        .map(|character| match character {
            '⁰' => '0',
            '¹' => '1',
            '²' => '2',
            '³' => '3',
            '⁴' => '4',
            '⁵' => '5',
            '⁶' => '6',
            '⁷' => '7',
            '⁸' => '8',
            '⁹' => '9',
            '⁽' => '(',
            '⁾' => ')',
            other => other,
        })
        .collect::<String>()
        .trim()
        .to_owned()
}

fn marker_match(text: &str) -> bool {
    marker_start_re().is_match(text)
}

enum Marker {
    Symbol(String),
    Numeric(i64),
    Other,
}

/// A marker-matched line split into its marker and the rest.
fn marker_label(text: &str) -> Marker {
    let head = text.trim_start();
    if head.starts_with(|c: char| SYMBOL_MARKER_CHARS.contains(c)) {
        let rest = head.trim_start_matches(|c: char| SYMBOL_MARKER_CHARS.contains(c));
        return Marker::Symbol(rest.trim().to_owned());
    }
    if let Some(found) = numeric_head_re().captures(head) {
        let value = found[1].parse().unwrap_or(0);
        return Marker::Numeric(value);
    }
    Marker::Other
}

/// A line that is nothing but a note's label: its number, or a symbol run.
fn bare_label(text: &str) -> Option<Option<i64>> {
    let text = folded(text);
    if text.is_empty() {
        return None;
    }
    if text.chars().all(|c| SYMBOL_MARKER_CHARS.contains(c)) {
        return Some(None);
    }
    bare_numeric_re()
        .captures(&text)
        .map(|found| Some(found[1].parse().unwrap_or(0)))
}

fn leading_note_number(text: &str) -> Option<i64> {
    leading_number_re()
        .captures(text)
        .and_then(|found| found[1].parse().ok())
}

fn terminated(text: &str) -> bool {
    let stripped = text
        .trim_end()
        .trim_end_matches(|c: char| "\"'\u{2019}\u{201d}\u{bb})]}".contains(c));
    stripped.ends_with(['.', '?', '!', '\u{2026}'])
}

fn strict_citation(text: &str) -> bool {
    !protected_citation_spans(text).is_empty()
}

fn citation_shaped(text: &str) -> bool {
    static PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();
    static ABBREVIATION: OnceLock<Regex> = OnceLock::new();
    static TAIL: OnceLock<Regex> = OnceLock::new();
    let text = folded(text);
    let patterns = PATTERNS.get_or_init(|| {
        [
            r"(?i)\bsupra\s+note\b",
            r"(?i)\bibid\b",
            r"^\(\d{4}\)",
            r"\beds?\.(\s|$|,)",
            r"\(.+\b\d{4}\)\.?$",
        ]
        .iter()
        .map(|pattern| Regex::new(pattern).expect("citation shape regex"))
        .collect()
    });
    if patterns.iter().any(|pattern| pattern.is_match(&text)) {
        return true;
    }
    let abbreviation = ABBREVIATION.get_or_init(|| {
        Regex::new(r"^(?:[A-Z][A-Za-z]{0,3}\.|(?:[A-Z]\.){1,3})$").expect("abbreviation regex")
    });
    let tokens = text.split_whitespace().collect::<Vec<_>>();
    if tokens
        .iter()
        .filter(|token| abbreviation.is_match(token.trim_end_matches([',', ';'])))
        .count()
        >= 2
    {
        return true;
    }
    tokens.len() <= 4
        && TAIL
            .get_or_init(|| Regex::new(r"\d+\)?[.,]?$").expect("citation tail regex"))
            .is_match(&text)
}

/// The order-tolerant note ladder: the longest chain of labels each within a few of the one
/// before it (a two-column band shuffles reading order).
fn ladder_consensus(labels: &[i64]) -> HashSet<usize> {
    if labels.is_empty() {
        return HashSet::new();
    }
    let mut best = vec![1usize; labels.len()];
    let mut predecessor = vec![None; labels.len()];
    for index in 0..labels.len() {
        for earlier in 0..index {
            if (labels[index] - labels[earlier]).abs() <= MAX_NUMERIC_LABEL_STEP
                && best[earlier] + 1 > best[index]
            {
                best[index] = best[earlier] + 1;
                predecessor[index] = Some(earlier);
            }
        }
    }
    let mut end = (0..labels.len())
        .max_by(|left, right| best[*left].cmp(&best[*right]).then(right.cmp(left)))
        .unwrap();
    let mut consensus = HashSet::from([end]);
    while let Some(previous) = predecessor[end] {
        consensus.insert(previous);
        end = previous;
    }
    consensus
}

fn dense(labels: &[i64]) -> bool {
    labels.len() >= 2
        && labels
            .windows(2)
            .all(|pair| (1..=MAX_NUMERIC_LABEL_STEP).contains(&(pair[1] - pair[0])))
}

struct DocumentLines<'a> {
    pages: &'a [Page],
    /// Each page's lines in reading order: (page slot, line slots).
    order: Vec<(usize, Vec<usize>)>,
    note_size: Option<f64>,
}

impl<'a> DocumentLines<'a> {
    fn new(pages: &'a [Page]) -> Self {
        let mut slots = (0..pages.len()).collect::<Vec<_>>();
        slots.sort_by_key(|slot| pages[*slot].index);
        let order = slots
            .into_iter()
            .map(|slot| {
                let mut lines = (0..pages[slot].lines.len()).collect::<Vec<_>>();
                lines.sort_by_key(|line| pages[slot].lines[*line].reading_order);
                (slot, lines)
            })
            .collect();
        Self {
            pages,
            order,
            note_size: note_font_size(pages),
        }
    }

    fn text(&self, (page, line): LineRef) -> &str {
        self.pages[page].lines[line].text.trim()
    }

    fn region(&self, (page, line): LineRef) -> &str {
        &self.pages[page].lines[line].region_type
    }

    fn is_note(&self, line: LineRef) -> bool {
        self.region(line).contains("note")
    }

    /// A line whose label may still change: a body line, not a heading, a running head, a folio,
    /// a table's cell or a line the page leaves out.
    fn mutable(&self, (page, line): LineRef) -> bool {
        let line = &self.pages[page].lines[line];
        line.region_type == "body" && !line.exclude_from_body
    }

    fn y_ratio(&self, (page, line): LineRef) -> f64 {
        let page = &self.pages[page];
        let line = &page.lines[line];
        if page.height > 0.0 {
            (line.bbox[1] + line.bbox[3]) / 2.0 / page.height
        } else {
            0.0
        }
    }

    /// The line set in the document's note type.
    fn note_font(&self, (page, line): LineRef) -> bool {
        let size = line_font_size(&self.pages[page].lines[line]);
        self.note_size
            .is_some_and(|note| size > 0.0 && (size - note).abs() <= 0.6)
    }
}

/// The document's note type: the size most of its note lines are set in, where it is smaller than
/// its body's. Fitted over the whole document, never a page.
fn note_font_size(pages: &[Page]) -> Option<f64> {
    let mut notes = Vec::new();
    let mut body = Vec::new();
    for line in pages.iter().flat_map(|page| &page.lines) {
        let size = line_font_size(line);
        if size <= 0.0 {
            continue;
        }
        if line.region_type.contains("note") {
            notes.push(size);
        } else if line.region_type == "body" {
            body.push(size);
        }
    }
    if notes.len() < 3 || body.is_empty() {
        return None;
    }
    let median = |values: &mut Vec<f64>| {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    };
    let (note, body) = (median(&mut notes), median(&mut body));
    (note < body - 0.5).then_some(note)
}

/// Settle which lines are notes over the whole document. Returns how many lines changed.
pub(super) fn correct_note_regions(pages: &mut [Page], separators: &[Option<f64>]) -> usize {
    let claims = {
        let lines = DocumentLines::new(pages);
        let mut claims = BTreeMap::<LineRef, &'static str>::new();
        let rerouted = post_footnote_reroute(&lines);
        let closed = band_closure(&lines, &rerouted);
        for line in rerouted.iter().chain(&closed) {
            claims.entry(*line).or_insert("footnote");
        }
        let reserved = claims.keys().copied().collect::<HashSet<_>>();
        let band = note_band(&lines, separators, &reserved);
        for line in band.promoted.iter().map(|promotion| promotion.line) {
            claims.entry(line).or_insert("footnote");
        }
        let reserved = claims.keys().copied().collect::<HashSet<_>>();
        let strands = font_role_strands(&lines, separators, &reserved, &band);
        for line in strands {
            claims.entry(line).or_insert("footnote");
        }
        claims
    };
    let mut changed = 0;
    for ((page, line), region) in claims {
        let line = &mut pages[page].lines[line];
        if line.region_type != region {
            line.region_type = region.to_owned();
            changed += 1;
        }
    }
    changed
}

/// Text lines after a page's first note line that carry note evidence of their own: a note's
/// marker, a printed label continuing the page's note numbers, or the unfinished sentence of the
/// note before them where a later note or a citation's shape anchors it. Declining a line breaks
/// the chain, so the body below a page's notes inherits nothing.
fn post_footnote_reroute(lines: &DocumentLines<'_>) -> Vec<LineRef> {
    let mut claims = Vec::new();
    for (page, slots) in &lines.order {
        let refs = slots.iter().map(|slot| (*page, *slot)).collect::<Vec<_>>();
        let Some(first_note) = refs.iter().position(|line| lines.is_note(*line)) else {
            continue;
        };
        let last_note = refs
            .iter()
            .rposition(|line| lines.is_note(*line))
            .unwrap_or(first_note);
        let mut last_text = String::new();
        let mut last_anchor = false;
        let mut last_number = None::<i64>;
        for (position, &line) in refs.iter().enumerate() {
            if lines.is_note(line) {
                last_text = lines.text(line).to_owned();
                last_anchor = true;
                if let Some(number) = leading_note_number(&folded(&last_text)) {
                    last_number = Some(number);
                }
                continue;
            }
            let text = lines.text(line);
            if position < first_note || text.is_empty() || !lines.mutable(line) {
                continue;
            }
            let fold = folded(text);
            let bare = fold.len() <= 3
                && !fold.is_empty()
                && fold.bytes().all(|byte| byte.is_ascii_digit());
            if page_label_re().is_match(text) && !bare {
                continue;
            }
            let marker = marker_match(&fold);
            let sandwiched = position < last_note;
            let continuation = !last_text.is_empty() && last_anchor && !terminated(&last_text);
            let citation = strict_citation(text);
            let number = (bare || marker)
                .then(|| leading_note_number(&fold))
                .flatten();
            let continues =
                number.is_some() && last_number.is_some_and(|last| number == Some(last + 1));
            let bare_claim = bare && continues;
            if bare && !bare_claim {
                last_text.clear();
                last_anchor = false;
                continue;
            }
            let anchored_continuation = continuation && (sandwiched || citation || last_anchor);
            let anchored_citation = citation && sandwiched && !last_text.is_empty();
            if !(marker || anchored_continuation || anchored_citation || bare_claim) {
                last_text.clear();
                last_anchor = false;
                continue;
            }
            claims.push(line);
            last_text = text.to_owned();
            last_anchor = marker || bare_claim;
            if number.is_some() && (marker || bare_claim) {
                last_number = number;
            }
        }
    }
    claims
}

/// The bottom note band closed over a page-final run of text when the band's last note breaks
/// off mid-sentence, where the document's layout says notes end their pages: among the pages whose
/// notes end finished, most must have nothing after them. A tail must absorb whole.
fn band_closure(lines: &DocumentLines<'_>, rerouted: &[LineRef]) -> Vec<LineRef> {
    let rerouted = rerouted.iter().copied().collect::<HashSet<_>>();
    struct State {
        last_terminated: bool,
        tail: Vec<LineRef>,
        broken: bool,
    }
    let mut states = Vec::new();
    for (page, slots) in &lines.order {
        let refs = slots.iter().map(|slot| (*page, *slot)).collect::<Vec<_>>();
        let note = |line: &LineRef| lines.is_note(*line) || rerouted.contains(line);
        let Some(last) = refs.iter().rposition(note) else {
            continue;
        };
        let mut tail = Vec::new();
        let mut broken = false;
        for &line in &refs[last + 1..] {
            let text = lines.text(line);
            if note(&line) || text.is_empty() {
                continue;
            }
            let region = lines.region(line);
            if matches!(region, "header" | "footer") || page_label_re().is_match(text) {
                continue;
            }
            if !lines.mutable(line) {
                broken = true;
                break;
            }
            tail.push(line);
        }
        states.push(State {
            last_terminated: terminated(lines.text(refs[last])),
            tail,
            broken,
        });
    }
    let prior = states
        .iter()
        .filter(|state| state.last_terminated)
        .collect::<Vec<_>>();
    let terminal = prior
        .iter()
        .filter(|state| state.tail.is_empty() && !state.broken)
        .count();
    if prior.len() < CLOSURE_MIN_PRIOR_PAGES
        || (terminal as f64) / (prior.len() as f64) < CLOSURE_MIN_PRIOR_RATIO
    {
        return Vec::new();
    }
    let mut claims = Vec::new();
    for state in states {
        if state.last_terminated
            || state.broken
            || state.tail.is_empty()
            || state.tail.len() > CLOSURE_MAX_TAIL_LINES
        {
            continue;
        }
        let mut taken = Vec::new();
        let mut ended = false;
        for &line in &state.tail {
            taken.push(line);
            if terminated(lines.text(line)) {
                ended = true;
                break;
            }
        }
        if ended && taken.len() < state.tail.len() {
            continue;
        }
        claims.extend(taken);
    }
    claims
}

struct Promotion {
    line: LineRef,
    label: Option<i64>,
}

struct BandFindings {
    promoted: Vec<Promotion>,
}

/// The printed separator's band. Below the rule, a text line opening as a note does is a note
/// where the page's note ladder (its own note lines' labels with the candidates') takes its number;
/// a markerless line below the rule is the wrapped text of the note above it, the topmost labelled
/// line its ceiling. A page whose band has no label of its own continues the previous page's note
/// when that page's band broke off mid-sentence.
fn note_band(
    lines: &DocumentLines<'_>,
    separators: &[Option<f64>],
    reserved: &HashSet<LineRef>,
) -> BandFindings {
    struct Candidate {
        line: LineRef,
        y: f64,
    }
    let mut findings = BandFindings {
        promoted: Vec::new(),
    };
    let mut states = HashMap::<usize, (bool, f64, String)>::new();
    let mut deferred = Vec::<(usize, Vec<Candidate>)>::new();
    for (page, slots) in &lines.order {
        let Some(separator) = separators.get(*page).copied().flatten() else {
            continue;
        };
        let height = lines.pages[*page].height;
        if height <= 0.0 {
            continue;
        }
        let rule = separator / height;
        let refs = slots.iter().map(|slot| (*page, *slot)).collect::<Vec<_>>();
        let mut band_top_below = None::<f64>;
        let mut ladder = Vec::<(usize, i64)>::new();
        for (position, &line) in refs.iter().enumerate() {
            let text = folded(lines.text(line));
            if !lines.is_note(line) || text.is_empty() || !marker_match(&text) {
                continue;
            }
            let y = lines.y_ratio(line);
            if y > rule + SEPARATOR_Y_TOLERANCE {
                band_top_below = Some(band_top_below.map_or(y, |top| top.min(y)));
                if let Marker::Numeric(label) = marker_label(&text) {
                    ladder.push((position, label));
                }
            }
        }
        let mut band_lines = Vec::<(f64, String)>::new();
        let mut continuations = Vec::<Candidate>::new();
        let mut candidates = Vec::<(usize, LineRef, f64, Option<i64>, bool)>::new();
        for (position, &line) in refs.iter().enumerate() {
            let y = lines.y_ratio(line);
            if y <= rule + SEPARATOR_Y_TOLERANCE {
                continue;
            }
            if lines.is_note(line) {
                band_lines.push((y, lines.text(line).to_owned()));
                continue;
            }
            if !lines.mutable(line) {
                continue;
            }
            let text = lines.text(line);
            let fold = folded(text);
            if text.is_empty() || !marker_match(&fold) {
                if !text.is_empty() {
                    continuations.push(Candidate { line, y });
                }
                continue;
            }
            if reserved.contains(&line) {
                continue;
            }
            match marker_label(&fold) {
                Marker::Symbol(rest) if !rest.chars().any(char::is_alphabetic) => continue,
                Marker::Numeric(label) => candidates.push((position, line, y, Some(label), false)),
                _ => candidates.push((position, line, y, None, false)),
            }
        }
        let numeric = candidates
            .iter()
            .enumerate()
            .filter(|(_, candidate)| candidate.3.is_some())
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if !numeric.is_empty() {
            let mut entries = ladder
                .iter()
                .map(|(position, label)| (*position, *label, None))
                .chain(numeric.iter().map(|index| {
                    (
                        candidates[*index].0,
                        candidates[*index].3.unwrap(),
                        Some(*index),
                    )
                }))
                .collect::<Vec<_>>();
            entries.sort_by_key(|entry| (entry.0, entry.1));
            if entries.len() > 1 {
                let consensus =
                    ladder_consensus(&entries.iter().map(|entry| entry.1).collect::<Vec<_>>());
                if consensus.len() * 2 <= entries.len() {
                    for index in &numeric {
                        candidates[*index].4 = true;
                    }
                } else {
                    for (position, entry) in entries.iter().enumerate() {
                        if let Some(index) = entry.2 {
                            if !consensus.contains(&position) {
                                candidates[index].4 = true;
                            }
                        }
                    }
                }
            }
        }
        for &(_, line, y, label, dropped) in &candidates {
            if dropped {
                continue;
            }
            findings.promoted.push(Promotion { line, label });
            band_lines.push((y, lines.text(line).to_owned()));
        }
        if !continuations.is_empty() {
            let tops = band_top_below
                .into_iter()
                .chain(
                    candidates
                        .iter()
                        .filter(|candidate| !candidate.4)
                        .map(|candidate| candidate.2),
                )
                .collect::<Vec<_>>();
            if tops.is_empty() {
                deferred.push((*page, continuations));
            } else {
                let top = tops.iter().copied().fold(f64::INFINITY, f64::min);
                let nearest_above = continuations
                    .iter()
                    .filter(|candidate| candidate.y < top - SEPARATOR_Y_TOLERANCE)
                    .max_by(|left, right| left.y.total_cmp(&right.y))
                    .map(|candidate| candidate.line);
                let mut above = Vec::new();
                for candidate in continuations {
                    if reserved.contains(&candidate.line) {
                        continue;
                    }
                    let above_top = candidate.y < top - SEPARATOR_Y_TOLERANCE;
                    if above_top
                        && (nearest_above != Some(candidate.line)
                            || !citation_shaped(lines.text(candidate.line)))
                    {
                        above.push(candidate);
                        continue;
                    }
                    band_lines.push((candidate.y, lines.text(candidate.line).to_owned()));
                    findings.promoted.push(Promotion {
                        line: candidate.line,
                        label: None,
                    });
                }
                if !above.is_empty() {
                    deferred.push((*page, above));
                }
            }
        }
        let tail = band_lines
            .iter()
            .max_by(|left, right| left.0.total_cmp(&right.0))
            .cloned();
        states.insert(
            lines.pages[*page].index,
            (
                !band_lines.is_empty(),
                tail.as_ref().map_or(0.0, |tail| tail.0),
                tail.map_or(String::new(), |tail| tail.1),
            ),
        );
    }
    deferred.sort_by_key(|(page, _)| lines.pages[*page].index);
    for (page, candidates) in deferred {
        let index = lines.pages[page].index;
        let Some(previous) = index
            .checked_sub(1)
            .and_then(|previous| states.get(&previous))
            .cloned()
        else {
            continue;
        };
        if !previous.0 || terminated(&previous.2) {
            continue;
        }
        let mut tail = None::<(f64, String)>;
        let mut emitted = false;
        for candidate in candidates {
            if reserved.contains(&candidate.line) {
                continue;
            }
            findings.promoted.push(Promotion {
                line: candidate.line,
                label: None,
            });
            emitted = true;
            if tail.as_ref().is_none_or(|tail| candidate.y > tail.0) {
                tail = Some((candidate.y, lines.text(candidate.line).to_owned()));
            }
        }
        if emitted {
            let state = states.entry(index).or_insert((false, 0.0, String::new()));
            if let Some(tail) = tail.filter(|tail| !state.0 || tail.0 > state.1) {
                state.1 = tail.0;
                state.2 = tail.1;
            }
            state.0 = true;
        }
    }
    findings
}

/// Runs of lines in the document's note type the separator lanes could not anchor: on a page with
/// no rule, a run that ends the page; below a rule, a run of labels printed on their own lines.
/// Type alone never makes a note (a block quotation can share it): a run needs a label that takes
/// the document's next note number, a dense chain of labels of its own, a symbol label on a worded
/// note, or a note beside it. A contents page is no note's.
fn font_role_strands(
    lines: &DocumentLines<'_>,
    separators: &[Option<f64>],
    reserved: &HashSet<LineRef>,
    band: &BandFindings,
) -> Vec<LineRef> {
    if lines.note_size.is_none() {
        return Vec::new();
    }
    let promoted = band
        .promoted
        .iter()
        .map(|promotion| promotion.line)
        .collect::<HashSet<_>>();
    let mut promoted_labels = HashMap::<usize, Vec<i64>>::new();
    for promotion in &band.promoted {
        if let Some(label) = promotion.label {
            promoted_labels
                .entry(promotion.line.0)
                .or_default()
                .push(label);
        }
    }
    let mut ladder_last = None::<i64>;
    let mut claims = Vec::new();
    let mut claimed = HashSet::new();
    for (page, slots) in &lines.order {
        let refs = slots.iter().map(|slot| (*page, *slot)).collect::<Vec<_>>();
        let advance = |label: i64, last: &mut Option<i64>| {
            if last.is_none_or(|last| label > last) {
                *last = Some(label);
            }
        };
        for &line in &refs {
            let text = folded(lines.text(line));
            if lines.is_note(line) && marker_match(&text) {
                if let Marker::Numeric(label) = marker_label(&text) {
                    advance(label, &mut ladder_last);
                }
            }
        }
        for label in promoted_labels.get(page).into_iter().flatten() {
            advance(*label, &mut ladder_last);
        }
        let page_ladder_start = ladder_last;
        let rule = separators
            .get(*page)
            .copied()
            .flatten()
            .filter(|_| lines.pages[*page].height > 0.0)
            .map(|separator| separator / lines.pages[*page].height);
        let texts = refs
            .iter()
            .map(|line| lines.text(*line))
            .collect::<Vec<_>>();
        if texts
            .iter()
            .filter(|text| toc_leader_re().is_match(text))
            .count()
            >= 3
            || texts
                .iter()
                .filter(|text| trailing_page_number_re().is_match(text))
                .count()
                >= 3
        {
            continue;
        }
        let below = |line: LineRef| {
            rule.is_none_or(|rule| lines.y_ratio(line) > rule + SEPARATOR_Y_TOLERANCE)
        };
        let base = refs
            .iter()
            .map(|&line| {
                !lines.text(line).is_empty()
                    && lines.mutable(line)
                    && below(line)
                    && (lines.note_font(line) || bare_label(lines.text(line)).is_some())
            })
            .collect::<Vec<_>>();
        let mut expanded = refs
            .iter()
            .zip(&base)
            .map(|(&line, &member)| {
                member
                    || (!lines.text(line).is_empty()
                        && lines.mutable(line)
                        && below(line)
                        && marker_match(&folded(lines.text(line))))
            })
            .collect::<Vec<_>>();
        for index in 1..refs.len().saturating_sub(1) {
            if !expanded[index]
                && expanded[index - 1]
                && expanded[index + 1]
                && !lines.text(refs[index]).is_empty()
                && lines.mutable(refs[index])
                && below(refs[index])
            {
                expanded[index] = true;
            }
        }
        let page_promoted = refs.iter().any(|line| promoted.contains(line));
        let mut passes = vec![("base", base.clone())];
        if expanded != base || page_promoted {
            passes.push(("expanded", expanded));
        }
        for (pass, members) in passes {
            let mut position = 0;
            while position < refs.len() {
                if !members[position] {
                    position += 1;
                    continue;
                }
                let start = position;
                while position < refs.len() && members[position] {
                    position += 1;
                }
                let run = &refs[start..position];
                if !run.iter().any(|line| lines.note_font(*line)) {
                    continue;
                }
                let bottom_terminal = refs[position..].iter().all(|line| {
                    matches!(lines.region(*line), "header" | "footer")
                        || lines.text(*line).is_empty()
                });
                let position_ok = rule.is_some() || bottom_terminal;
                let adjacent_note = (start > 0 && lines.is_note(refs[start - 1]))
                    || (position < refs.len() && lines.is_note(refs[position]));
                let separator_anchor = pass == "expanded"
                    && rule.is_some()
                    && run.iter().any(|line| promoted.contains(line));
                let mut labels = Vec::new();
                for &line in run {
                    let text = folded(lines.text(line));
                    if let Some(Some(label)) = bare_label(&text) {
                        labels.push(label);
                    } else if marker_match(&text) {
                        if let Marker::Numeric(label) = marker_label(&text) {
                            labels.push(label);
                        }
                    }
                }
                let head_text = folded(lines.text(run[0]));
                let head = match bare_label(&head_text) {
                    Some(label) => Some(label),
                    None if marker_match(&head_text) => match marker_label(&head_text) {
                        Marker::Symbol(rest) if rest.chars().any(char::is_alphabetic) => Some(None),
                        Marker::Numeric(label) => Some(Some(label)),
                        _ => None,
                    },
                    None => None,
                };
                if let Some(Some(label)) = head {
                    if !labels.contains(&label) {
                        labels.insert(0, label);
                    }
                }
                let confirmed = if adjacent_note || separator_anchor {
                    true
                } else if !position_ok {
                    false
                } else {
                    match head {
                        Some(None) => run.iter().any(|line| {
                            lines.note_font(*line)
                                && lines.text(*line).chars().any(char::is_alphabetic)
                        }),
                        Some(Some(label)) => {
                            let anchor = if pass == "expanded" {
                                page_ladder_start
                            } else {
                                ladder_last
                            };
                            let expected = anchor.map_or(1, |anchor| anchor + 1);
                            (expected..expected + MAX_NUMERIC_LABEL_STEP).contains(&label)
                                || dense(&labels)
                        }
                        None => dense(&labels),
                    }
                };
                if !confirmed {
                    continue;
                }
                if pass == "base" {
                    for label in &labels {
                        advance(*label, &mut ladder_last);
                    }
                }
                for &line in run {
                    if claimed.insert(line) && !reserved.contains(&line) {
                        claims.push(line);
                    }
                }
            }
        }
    }
    claims
}
