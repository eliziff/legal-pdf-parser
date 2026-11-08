//! The note lanes of region postcorrection, read over the whole document: which lines are notes,
//! from the evidence a page carries and the evidence the document's other pages carry with it.
//! Ported from the Open Access Journals Database's `region_postcorrection.py` (the post-footnote
//! reroute and the note band closure) and `note_band_witness.py` (the printed separator's band and
//! the font-role strands).
//!
//! The lanes read a document's lines as the note band left them (`LaneLines`) and write nothing
//! (`note_promotions`). Every lane proposes; the claims are taken together once all have spoken, an
//! earlier lane's claim reserving its line from the later ones, so a weaker witness never
//! overrides a stronger.
//! Each lane needs two independent witnesses before it retypes a line, and a declined line is a
//! miss, never a misplacement. The document's own state carries across pages: the note label
//! ladder, the layout prior over every page whose notes end their band, and the bridge from one
//! page's unfinished note to the next page's band.

use super::sentence_ended;
use legal_pdf_core::{
    line_font_size,
    model::{Line, Page},
    regex,
};
use std::collections::{BTreeSet, HashMap, HashSet};

const SEPARATOR_Y_TOLERANCE: f64 = 0.006;
const MAX_NUMERIC_LABEL_STEP: i64 = 5;
const SYMBOL_MARKER_CHARS: &str = "*†‡§¶";
const CLOSURE_MIN_PRIOR_PAGES: usize = 2;
const CLOSURE_MIN_PRIOR_RATIO: f64 = 0.5;
const CLOSURE_MAX_TAIL_LINES: usize = 12;

/// A line, by its page and its place among the page's lines.
pub type LineRef = (usize, usize);

/// The line as marker evidence judges it: superscript labels ("²² See") read as plain digits.
fn folded(text: &str) -> String {
    let fold = |c| match c {
        '⁽' => '(',
        '⁾' => ')',
        _ => "⁰¹²³⁴⁵⁶⁷⁸⁹".chars().position(|digit| digit == c).map_or(c, |digit| char::from(b'0' + digit as u8)),
    };
    text.chars().map(fold).collect::<String>().trim().to_owned()
}

fn marker_match(text: &str) -> bool {
    regex!(r"^\s*(?:\(?\d{1,3}\)?[.)]?|[*†‡§¶]+)\s+\S").is_match(text)
}

fn page_label(text: &str) -> bool {
    regex!(r"^\s*(?:[-\u{2013}\u{2014}]\s*)?(?:\d{1,4}|[ivxlcdmIVXLCDM]{1,12})(?:\s*[-\u{2013}\u{2014}])?\s*$").is_match(text)
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
        return Marker::Symbol(head.trim_start_matches(|c: char| SYMBOL_MARKER_CHARS.contains(c)).trim().to_owned());
    }
    regex!(r"^\(?(\d{1,3})\)?[.)]?").captures(head).map_or(Marker::Other, |found| Marker::Numeric(found[1].parse().unwrap_or(0)))
}

/// The number a line opening as a note does prints, where it prints one.
fn marker_number(text: &str) -> Option<i64> {
    match marker_label(text) {
        Marker::Numeric(label) if marker_match(text) => Some(label),
        _ => None,
    }
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
    regex!(r"^\(?(\d{1,3})\)?[.)]?$").captures(&text).map(|found| Some(found[1].parse().unwrap_or(0)))
}

fn leading_note_number(text: &str) -> Option<i64> {
    regex!(r"^\s*\(?(\d{1,3})\)?[.)]?(?:\s|$)").captures(text).and_then(|found| found[1].parse().ok())
}

fn citation_shaped(text: &str) -> bool {
    let text = folded(text);
    if regex!(r"(?i:\bsupra\s+note\b|\bibid\b)|^\(\d{4}\)|\beds?\.(\s|$|,)|\(.+\b\d{4}\)\.?$").is_match(&text) {
        return true;
    }
    let abbreviation = regex!(r"^(?:[A-Z][A-Za-z]{0,3}\.|(?:[A-Z]\.){1,3})$");
    let tokens = text.split_whitespace().collect::<Vec<_>>();
    tokens.iter().filter(|token| abbreviation.is_match(token.trim_end_matches([',', ';']))).count() >= 2
        || (tokens.len() <= 4 && regex!(r"\d+\)?[.,]?$").is_match(&text))
}

/// The order-tolerant note ladder: the longest chain of labels each within a few of the one
/// before it (a two-column band shuffles reading order).
fn ladder_consensus(labels: &[i64]) -> HashSet<usize> {
    let mut best = vec![1usize; labels.len()];
    let mut predecessor = vec![None; labels.len()];
    for index in 0..labels.len() {
        for earlier in 0..index {
            if (labels[index] - labels[earlier]).abs() <= MAX_NUMERIC_LABEL_STEP && best[earlier] + 1 > best[index] {
                best[index] = best[earlier] + 1;
                predecessor[index] = Some(earlier);
            }
        }
    }
    let Some(mut end) = (0..labels.len()).max_by(|left, right| best[*left].cmp(&best[*right]).then(right.cmp(left))) else {
        return HashSet::new();
    };
    let mut consensus = HashSet::from([end]);
    while let Some(previous) = predecessor[end] {
        consensus.insert(previous);
        end = previous;
    }
    consensus
}

fn dense(labels: &[i64]) -> bool {
    labels.len() >= 2 && labels.windows(2).all(|pair| (1..=MAX_NUMERIC_LABEL_STEP).contains(&(pair[1] - pair[0])))
}

/// A note's line: the note band's.
pub const LANE_NOTE: u8 = 1;
/// A body line whose role may still change: no heading, running head, folio, table's cell, or line
/// the reading holds out of the body.
pub const LANE_BODY: u8 = 1 << 1;
/// A running head or foot, or a folio.
pub const LANE_FURNITURE: u8 = 1 << 2;
/// A numbered paragraph's label, as the note band read it.
pub const LANE_PARAGRAPH: u8 = 1 << 3;

/// A document's lines as the lanes read them: its pages, each page's lines of the document in
/// reading order (page slot, the lines' places on the page), each page's first slot, and each
/// line's `LANE_*` flags by slot.
pub struct LaneLines<'a> {
    pub pages: &'a [Page],
    pub order: &'a [(usize, Vec<usize>)],
    pub starts: &'a [usize],
    pub flags: &'a [u8],
}

struct DocumentLines<'a> {
    lines: LaneLines<'a>,
    /// Native font sizes and recognized glyph heights are distinct measurements.
    note_sizes: [Option<f64>; 2],
}

impl<'a> DocumentLines<'a> {
    fn new(lines: LaneLines<'a>) -> Self {
        let mut document = Self { lines, note_sizes: [None; 2] };
        document.note_sizes = note_type_sizes(&document);
        document
    }

    /// Each page's slot and its lines in reading order.
    fn pages(&self) -> impl Iterator<Item = (usize, Vec<LineRef>)> + 'a {
        self.lines.order.iter().map(|(page, slots)| (*page, slots.iter().map(|slot| (*page, *slot)).collect()))
    }

    fn line(&self, (page, line): LineRef) -> &'a Line {
        &self.lines.pages[page].lines[line]
    }

    fn flagged(&self, (page, line): LineRef, flag: u8) -> bool {
        self.lines.flags[self.lines.starts[page] + line] & flag != 0
    }

    fn text(&self, line: LineRef) -> &'a str {
        self.line(line).text.trim()
    }

    fn is_note(&self, line: LineRef) -> bool {
        self.flagged(line, LANE_NOTE)
    }

    /// A line whose label may still change: a body line, not a heading, a running head, a folio,
    /// a table's cell or a line the reading holds out.
    fn mutable(&self, line: LineRef) -> bool {
        self.flagged(line, LANE_BODY)
    }

    fn paragraph_label(&self, line: LineRef) -> bool {
        self.flagged(line, LANE_PARAGRAPH) && bare_label(&self.line(line).text).is_some()
    }

    fn y_ratio(&self, at: LineRef) -> f64 {
        let (height, bbox) = (self.lines.pages[at.0].height, self.line(at).bbox);
        if height > 0.0 { (bbox[1] + bbox[3]) / 2.0 / height } else { 0.0 }
    }

    /// The line's size beside the document's note type for its measurement, where that type is known.
    fn note_type(&self, at: LineRef) -> Option<(f64, f64)> {
        let line = self.line(at);
        self.note_sizes[usize::from(line.spans.is_empty())].map(|note| (note_text_size(line), note))
    }

    /// A line set larger than the document's note type, where that type is known: body text a
    /// band below a rule must not take (a statute's sections under their heading's rule, with the
    /// small source notes between them).
    fn body_type(&self, at: LineRef) -> bool {
        self.note_type(at).is_some_and(|(size, note)| size > note + 0.6)
    }

    /// The line set in the document's note type.
    fn matches_note_type(&self, at: LineRef) -> bool {
        self.note_type(at).is_some_and(|(size, note)| size > 0.0 && (size - note).abs() <= 0.6)
    }
}

fn note_text_size(line: &Line) -> f64 {
    if line.spans.is_empty() { line.bbox[3] - line.bbox[1] } else { line_font_size(line) }
}

/// The document's note type, smaller than its body's. Learn font sizes and recognized glyph
/// heights separately over the whole document, never borrowing one measurement for the other.
fn note_type_sizes(lines: &DocumentLines<'_>) -> [Option<f64>; 2] {
    let (mut notes, mut body) = ([Vec::new(), Vec::new()], [Vec::new(), Vec::new()]);
    for at in lines.pages().flat_map(|(_, refs)| refs) {
        let line = lines.line(at);
        let (size, source) = (note_text_size(line), usize::from(line.spans.is_empty()));
        if size <= 0.0 {
            continue;
        }
        if lines.is_note(at) {
            notes[source].push(size);
        } else if lines.flagged(at, LANE_BODY | LANE_PARAGRAPH) {
            body[source].push(size);
        }
    }
    std::array::from_fn(|source| {
        if notes[source].len() < 3 || body[source].is_empty() {
            return None;
        }
        let (note, body) = (legal_pdf_core::upper_median(notes[source].clone()), legal_pdf_core::upper_median(body[source].clone()));
        (note < body - 0.5).then_some(note)
    })
}

/// The lines one logical document's lanes take for notes, beside the note band's (`lines`), with
/// separators aligned to its page slots: each by its page slot and its place on the page, in order.
pub fn note_promotions(lines: LaneLines<'_>, separators: &[Option<f64>]) -> Vec<LineRef> {
    let lines = DocumentLines::new(lines);
    let rerouted = post_footnote_reroute(&lines);
    let mut claims = band_closure(&lines, &rerouted).into_iter().chain(rerouted).collect::<BTreeSet<_>>();
    let band = note_band(&lines, separators, &claims);
    claims.extend(band.iter().map(|promotion| promotion.line));
    let strands = font_role_strands(&lines, separators, &claims, &band);
    claims.extend(strands);
    claims.into_iter().collect()
}

/// Text lines after a page's first note line that carry note evidence of their own: a note's
/// marker, a printed label continuing the page's note numbers, or the unfinished sentence of the
/// note before them where a later note or a citation's shape anchors it. Declining a line breaks
/// the chain, so the body below a page's notes inherits nothing.
fn post_footnote_reroute(lines: &DocumentLines<'_>) -> Vec<LineRef> {
    let mut claims = Vec::new();
    for (_, refs) in lines.pages() {
        let Some(first_note) = refs.iter().position(|line| lines.is_note(*line)) else {
            continue;
        };
        let last_note = refs.iter().rposition(|line| lines.is_note(*line)).unwrap_or(first_note);
        let mut trail = Trail::Broken;
        for (position, &line) in refs.iter().enumerate() {
            let text = lines.text(line);
            if lines.is_note(line) {
                trail = trail.past_note(text);
                continue;
            }
            if position < first_note || text.is_empty() {
                continue;
            }
            if !lines.mutable(line) {
                trail = Trail::Broken;
                continue;
            }
            let Trail::Note { text: last_text, anchored, number: last_number } = &trail else {
                continue;
            };
            if last_text.is_empty() {
                continue;
            }
            let fold = folded(text);
            let bare = fold.len() <= 3 && !fold.is_empty() && fold.bytes().all(|byte| byte.is_ascii_digit());
            if page_label(text) && !bare {
                continue;
            }
            let marker = marker_match(&fold);
            let number = if bare || marker { leading_note_number(&fold) } else { None };
            let bare_claim = bare && number.is_some() && last_number.is_some_and(|last| number == Some(last + 1));
            let continuation = *anchored && !sentence_ended(last_text);
            let anchored_citation = position < last_note && !legal_structure_model::structure_analysis().protected_citation_spans(text).is_empty();
            if (bare && !bare_claim) || !(marker || continuation || anchored_citation || bare_claim) {
                trail = Trail::Broken;
                continue;
            }
            claims.push(line);
            let numbered = marker || bare_claim;
            trail = Trail::Note { text: text.to_owned(), anchored: numbered, number: number.filter(|_| numbered).or(*last_number) };
        }
    }
    claims
}

/// What a walk down a page carries from the notes above the line it is at.
enum Trail {
    /// A line no note goes on over broke the trail.
    Broken,
    /// The last line a note holds: its text, whether a note label or a note line anchors it (a
    /// continuation claimed without its own label does not), and the last note number read.
    Note { text: String, anchored: bool, number: Option<i64> },
}

impl Trail {
    /// The trail past a line of a note, whose own number, where it prints one, replaces the last.
    fn past_note(self, text: &str) -> Self {
        let last = if let Self::Note { number, .. } = self { number } else { None };
        Self::Note { text: text.to_owned(), anchored: true, number: leading_note_number(&folded(text)).or(last) }
    }
}

/// The bottom note band closed over a page-final run of text when the band's last note breaks
/// off mid-sentence, where the document's layout says notes end their pages: among the pages whose
/// notes end finished, most must have nothing after them. A tail must absorb whole.
fn band_closure(lines: &DocumentLines<'_>, rerouted: &[LineRef]) -> Vec<LineRef> {
    let rerouted = rerouted.iter().copied().collect::<HashSet<_>>();
    // Each page with a note: whether its last note's sentence ends, the text after it, and whether a
    // line that keeps its role breaks that text.
    let mut states = Vec::new();
    for (_, refs) in lines.pages() {
        let note = |line: &LineRef| lines.is_note(*line) || rerouted.contains(line);
        let Some(last) = refs.iter().rposition(note) else {
            continue;
        };
        let (mut tail, mut broken) = (Vec::new(), false);
        for &line in &refs[last + 1..] {
            let text = lines.text(line);
            if note(&line) || text.is_empty() || lines.flagged(line, LANE_FURNITURE) || page_label(text) {
                continue;
            }
            if !lines.mutable(line) {
                broken = true;
                break;
            }
            tail.push(line);
        }
        states.push((sentence_ended(lines.text(refs[last])), tail, broken));
    }
    let prior = states.iter().filter(|state| state.0).collect::<Vec<_>>();
    let terminal = prior.iter().filter(|state| state.1.is_empty() && !state.2).count();
    if prior.len() < CLOSURE_MIN_PRIOR_PAGES || (terminal as f64) / (prior.len() as f64) < CLOSURE_MIN_PRIOR_RATIO {
        return Vec::new();
    }
    let mut claims = Vec::new();
    for (terminated, tail, broken) in states {
        let whole = !tail.iter().position(|line| sentence_ended(lines.text(*line))).is_some_and(|end| end + 1 < tail.len());
        if !terminated && !broken && !tail.is_empty() && tail.len() <= CLOSURE_MAX_TAIL_LINES && whole {
            claims.extend(tail);
        }
    }
    claims
}

struct Promotion {
    line: LineRef,
    label: Option<i64>,
}

/// The printed separator's band. Below the rule, a text line opening as a note does is a note
/// where the page's note ladder (its own note lines' labels with the candidates') takes its number;
/// a markerless line below the rule is the wrapped text of the note above it, the topmost labelled
/// line its ceiling. A page whose band has no label of its own continues the previous page's note
/// when that page's band broke off mid-sentence.
fn note_band(lines: &DocumentLines<'_>, separators: &[Option<f64>], reserved: &BTreeSet<LineRef>) -> Vec<Promotion> {
    let mut promoted = Vec::new();
    // Each page's band, by page index: whether it holds a line, and its lowest line's place and text.
    let mut states = HashMap::<usize, (bool, f64, String)>::new();
    let mut deferred = Vec::<(usize, Vec<(LineRef, f64)>)>::new();
    for (page, refs) in lines.pages() {
        let Some(separator) = separators.get(page).copied().flatten() else {
            continue;
        };
        let height = lines.lines.pages[page].height;
        if height <= 0.0 {
            continue;
        }
        let floor = separator / height + SEPARATOR_Y_TOLERANCE;
        let mut band_top_below = None::<f64>;
        // The page's note ladder: each entry's place, label, and the candidate it is, if it is one.
        let mut ladder = Vec::<(usize, i64, Option<usize>)>::new();
        for (position, &line) in refs.iter().enumerate() {
            let (text, y) = (folded(lines.text(line)), lines.y_ratio(line));
            if lines.is_note(line) && marker_match(&text) && y > floor {
                band_top_below = Some(band_top_below.map_or(y, |top| top.min(y)));
                ladder.extend(marker_number(&text).map(|label| (position, label, None)));
            }
        }
        let mut band_lines = Vec::<(f64, String)>::new();
        let mut continuations = Vec::<(LineRef, f64)>::new();
        // Each candidate's place, line, height on the page, label and whether the ladder drops it.
        let mut candidates = Vec::<(usize, LineRef, f64, Option<i64>, bool)>::new();
        for (position, &line) in refs.iter().enumerate() {
            let (text, y) = (lines.text(line), lines.y_ratio(line));
            if y <= floor {
                continue;
            }
            if lines.is_note(line) {
                band_lines.push((y, text.to_owned()));
                continue;
            }
            if !lines.mutable(line) || lines.body_type(line) || text.is_empty() {
                continue;
            }
            let fold = folded(text);
            if !marker_match(&fold) {
                continuations.push((line, y));
                continue;
            }
            match marker_label(&fold) {
                _ if reserved.contains(&line) => {}
                Marker::Symbol(rest) if !rest.chars().any(char::is_alphabetic) => {}
                Marker::Numeric(label) => candidates.push((position, line, y, Some(label), false)),
                _ => candidates.push((position, line, y, None, false)),
            }
        }
        let numbered = candidates.iter().enumerate().filter_map(|(index, candidate)| Some((candidate.0, candidate.3?, Some(index))));
        let mut entries = ladder.into_iter().chain(numbered).collect::<Vec<_>>();
        if entries.iter().any(|entry| entry.2.is_some()) && entries.len() > 1 {
            entries.sort_by_key(|entry| (entry.0, entry.1));
            let consensus = ladder_consensus(&entries.iter().map(|entry| entry.1).collect::<Vec<_>>());
            let drop_all = consensus.len() * 2 <= entries.len();
            for (position, entry) in entries.iter().enumerate() {
                if let Some(index) = entry.2.filter(|_| drop_all || !consensus.contains(&position)) {
                    candidates[index].4 = true;
                }
            }
        }
        for &(_, line, y, label, _) in candidates.iter().filter(|candidate| !candidate.4) {
            promoted.push(Promotion { line, label });
            band_lines.push((y, lines.text(line).to_owned()));
        }
        if !continuations.is_empty() {
            let tops = band_top_below.into_iter().chain(candidates.iter().filter(|candidate| !candidate.4).map(|candidate| candidate.2)).collect::<Vec<_>>();
            if tops.is_empty() {
                deferred.push((page, continuations));
            } else {
                let top = tops.into_iter().fold(f64::INFINITY, f64::min) - SEPARATOR_Y_TOLERANCE;
                let nearest_above = continuations.iter().filter(|(_, y)| *y < top).max_by(|left, right| left.1.total_cmp(&right.1)).map(|(line, _)| *line);
                let mut above = Vec::new();
                for (line, y) in continuations.into_iter().filter(|(line, _)| !reserved.contains(line)) {
                    if y < top && (nearest_above != Some(line) || !citation_shaped(lines.text(line))) {
                        above.push((line, y));
                    } else {
                        band_lines.push((y, lines.text(line).to_owned()));
                        promoted.push(Promotion { line, label: None });
                    }
                }
                if !above.is_empty() {
                    deferred.push((page, above));
                }
            }
        }
        let (y, text) = band_lines.iter().max_by(|left, right| left.0.total_cmp(&right.0)).cloned().unwrap_or_default();
        states.insert(lines.lines.pages[page].index, (!band_lines.is_empty(), y, text));
    }
    deferred.sort_by_key(|(page, _)| lines.lines.pages[*page].index);
    for (page, candidates) in deferred {
        let index = lines.lines.pages[page].index;
        let Some(previous) = index.checked_sub(1).and_then(|previous| states.get(&previous)) else {
            continue;
        };
        if !previous.0 || sentence_ended(&previous.2) {
            continue;
        }
        let mut tail = None::<(f64, String)>;
        for (line, y) in candidates.into_iter().filter(|(line, _)| !reserved.contains(line)) {
            promoted.push(Promotion { line, label: None });
            if tail.as_ref().is_none_or(|tail| y > tail.0) {
                tail = Some((y, lines.text(line).to_owned()));
            }
        }
        if let Some(tail) = tail {
            let state = states.entry(index).or_insert((false, 0.0, String::new()));
            if !state.0 || tail.0 > state.1 {
                (state.1, state.2) = tail;
            }
            state.0 = true;
        }
    }
    promoted
}

/// Runs of lines in the document's note type the separator lanes could not anchor: on a page with
/// no rule, a run that ends the page; below a rule, a run of labels printed on their own lines.
/// Type alone never makes a note (a block quotation can share it): a run needs a label that takes
/// the document's next note number, a dense chain of labels of its own, a symbol label on a worded
/// note, or a note beside it. A contents page is no note's.
fn font_role_strands(lines: &DocumentLines<'_>, separators: &[Option<f64>], reserved: &BTreeSet<LineRef>, band: &[Promotion]) -> Vec<LineRef> {
    if lines.note_sizes.iter().all(Option::is_none) {
        return Vec::new();
    }
    let promoted = band.iter().map(|promotion| promotion.line).collect::<HashSet<_>>();
    let mut promoted_labels = HashMap::<usize, Vec<i64>>::new();
    for promotion in band {
        promoted_labels.entry(promotion.line.0).or_default().extend(promotion.label);
    }
    let (mut ladder_last, mut claims, mut claimed) = (None::<i64>, Vec::new(), HashSet::new());
    for (page, refs) in lines.pages() {
        for &line in refs.iter().filter(|line| lines.is_note(**line)) {
            ladder_last = ladder_last.max(marker_number(&folded(lines.text(line))));
        }
        for label in promoted_labels.get(&page).into_iter().flatten() {
            ladder_last = ladder_last.max(Some(*label));
        }
        let page_ladder_start = ladder_last;
        let height = lines.lines.pages[page].height;
        let rule = separators.get(page).copied().flatten().filter(|_| height > 0.0).map(|separator| separator / height);
        let texts = refs.iter().map(|line| lines.text(*line)).collect::<Vec<_>>();
        if texts.iter().filter(|text| pdf_inspector::markdown::has_dot_leaders(text)).count() >= 3
            || texts.iter().filter(|text| regex!(r"\s\d{1,4}$").is_match(text)).count() >= 3
        {
            continue;
        }
        let open = |line: LineRef| {
            !lines.text(line).is_empty() && lines.mutable(line) && rule.is_none_or(|rule| lines.y_ratio(line) > rule + SEPARATOR_Y_TOLERANCE)
        };
        let base = refs
            .iter()
            .map(|&line| open(line) && !lines.paragraph_label(line) && (lines.matches_note_type(line) || bare_label(lines.text(line)).is_some()))
            .collect::<Vec<_>>();
        let mut expanded = refs
            .iter()
            .zip(&base)
            .map(|(&line, &member)| member || (open(line) && !lines.paragraph_label(line) && marker_match(&folded(lines.text(line)))))
            .collect::<Vec<_>>();
        for index in 1..refs.len().saturating_sub(1) {
            if !expanded[index] && expanded[index - 1] && expanded[index + 1] && open(refs[index]) {
                expanded[index] = true;
            }
        }
        let expand = expanded != base || refs.iter().any(|line| promoted.contains(line));
        for (expanded_pass, members) in [(false, base), (true, expanded)] {
            if expanded_pass && !expand {
                continue;
            }
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
                if !run.iter().any(|line| lines.matches_note_type(*line)) {
                    continue;
                }
                let bottom_terminal = refs[position..].iter().all(|line| lines.flagged(*line, LANE_FURNITURE) || lines.text(*line).is_empty());
                let adjacent_note = (start > 0 && lines.is_note(refs[start - 1])) || refs.get(position).is_some_and(|line| lines.is_note(*line));
                let separator_anchor = expanded_pass && rule.is_some() && run.iter().any(|line| promoted.contains(line));
                let mut labels = run
                    .iter()
                    .filter_map(|line| {
                        let text = folded(lines.text(*line));
                        bare_label(&text).flatten().or_else(|| marker_number(&text))
                    })
                    .collect::<Vec<_>>();
                let head_text = folded(lines.text(run[0]));
                let head = bare_label(&head_text).or_else(|| match marker_label(&head_text) {
                    Marker::Symbol(rest) if marker_match(&head_text) && rest.chars().any(char::is_alphabetic) => Some(None),
                    Marker::Numeric(label) if marker_match(&head_text) => Some(Some(label)),
                    _ => None,
                });
                if let Some(Some(label)) = head {
                    if !labels.contains(&label) {
                        labels.insert(0, label);
                    }
                }
                let confirmed = adjacent_note
                    || separator_anchor
                    || ((rule.is_some() || bottom_terminal)
                        && match head {
                            Some(None) => run.iter().any(|line| lines.matches_note_type(*line) && lines.text(*line).chars().any(char::is_alphabetic)),
                            Some(Some(label)) => {
                                let expected = if expanded_pass { page_ladder_start } else { ladder_last }.map_or(1, |anchor| anchor + 1);
                                (expected..expected + MAX_NUMERIC_LABEL_STEP).contains(&label) || dense(&labels)
                            }
                            None => dense(&labels),
                        });
                if !confirmed {
                    continue;
                }
                if !expanded_pass {
                    ladder_last = labels.iter().copied().fold(ladder_last, |last, label| last.max(Some(label)));
                }
                claims.extend(run.iter().copied().filter(|line| claimed.insert(*line) && !reserved.contains(line)));
            }
        }
    }
    claims
}
