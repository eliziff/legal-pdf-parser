//! Paragraph numbers a scan lost, recovered from the run they stand in. A judgment numbers its
//! paragraphs "[1]" to "[N]" (a factum "1." to "N."), each at the body's left edge, opening a
//! paragraph; recognition drops or garbles the small bracketed glyphs ("[11 The", "l2.", nothing at
//! all). Between two numbers the run supports, the paragraph-start lines that lie at the numbers'
//! column carry the values between, in order: a garbled token takes the value its characters fit,
//! a bare line is read again from the page image at three times the scale, in the margin where
//! its number must be, and takes the value that reading fits, or the only value left for it. A
//! value the run does not bound on both sides is written only when the re-reading shows it.
//! Printed text is exact and never touched.

use legal_pdf_core::model::{Line, Page, Word, INVISIBLE};
use legal_pdf_core::{OcrLine, PdfOcrProvider};
use std::collections::BTreeMap;

/// How many candidates a run's neighbour may be away and still support a value.
const SUPPORT_WINDOW: usize = 12;
/// The most values one gap may name.
const GAP_VALUES: u32 = 40;
/// The scale the margin is read again at.
const REREAD_SCALE: f64 = 3.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Family {
    /// "[12]"
    Bracket,
    /// "12."
    Dot,
}

impl Family {
    fn write(self, value: u32) -> String {
        match self {
            Family::Bracket => format!("[{value}]"),
            Family::Dot => format!("{value}."),
        }
    }
}

/// A reading of a line's opening token as a paragraph number: the family, the value, and whether
/// the characters were exactly the number (an exact reading is evidence; a repaired one a fit).
#[derive(Clone, Debug)]
pub(crate) struct Reading {
    family: Family,
    /// The value as read, then the values a misread bracket could hide ("[131": 131, or 13 with
    /// its closing bracket read as a 1); an exact reading has one.
    values: Vec<u32>,
    exact: bool,
    /// Characters of the line the token took, including the space after it.
    taken: usize,
}

impl Reading {
    fn value(&self) -> u32 {
        self.values[0]
    }
}

/// A misread digit as recognition writes it.
fn digit_of(c: char) -> Option<u32> {
    Some(match c {
        '0'..='9' => c as u32 - '0' as u32,
        'O' | 'o' | 'Q' | 'D' => 0,
        'I' | 'l' | 'i' | '|' | '!' | 'j' | ']' | 'J' | 't' => 1,
        'Z' | 'z' => 2,
        'E' => 3,
        'A' => 4,
        'S' | 's' => 5,
        'G' | 'b' => 6,
        'T' => 7,
        'B' | '&' => 8,
        'g' | 'q' => 9,
        _ => return None,
    })
}

/// The opening token of `text` read as a paragraph number, exact or repaired: first with the
/// brackets as printed, then with a letter recognition may have made of an opening bracket.
pub(crate) fn read_marker(text: &str) -> Option<Reading> {
    read_marker_with(text, false).or_else(|| read_marker_with(text, true))
}

fn read_marker_with(text: &str, misread_open: bool) -> Option<Reading> {
    let chars: Vec<char> = text.chars().collect();
    let mut k = 0;
    while k < chars.len() && chars[k] == ' ' {
        k += 1;
    }
    let start = k;
    let opened = k < chars.len()
        && (matches!(chars[k], '[' | '(' | '{') || (misread_open && matches!(chars[k], 'I' | 'l' | '|' | '!' | 'f' | 'r' | 't')));
    let open_char = if opened { Some(chars[k]) } else { None };
    if opened {
        k += 1;
        while k < chars.len() && chars[k] == ' ' {
            k += 1;
        }
    }
    let mut body = Vec::new();
    while k < chars.len() && body.len() < 4 && digit_of(chars[k]).is_some() && !matches!(chars[k], ']' | 'j' | 'J') {
        body.push(chars[k]);
        k += 1;
    }
    if body.is_empty() || body.len() > 3 {
        return None;
    }
    // The closing character: a bracket, a misread bracket, or a stop.
    let close = chars.get(k).copied();
    let (family, closed) = match close {
        Some(']' | ')' | '}' | 'j' | 'J' | '1' | 'l' | 'I' | '|' | '!') if open_char.is_some() => (Family::Bracket, true),
        Some('.' | ',' | '_') if open_char.is_none() => (Family::Dot, true),
        Some(' ') | None if open_char == Some('[') => (Family::Bracket, false),
        _ => return None,
    };
    if closed {
        k += 1;
    }
    // The token ends the word: a space or the line's end follows, else it is a word's start ("l2abc").
    if k < chars.len() && chars[k] != ' ' {
        return None;
    }
    while k < chars.len() && chars[k] == ' ' {
        k += 1;
    }
    let exact_digits = body.iter().all(|c| c.is_ascii_digit());
    let exact = exact_digits
        && closed
        && match family {
            Family::Bracket => open_char == Some('[') && close == Some(']'),
            Family::Dot => close == Some('.'),
        };
    let value = body.iter().fold(0u32, |v, &c| v * 10 + digit_of(c).unwrap_or(0));
    if value == 0 || value > 999 {
        return None;
    }
    let mut values = vec![value];
    // A bracket read as a digit: the body's last character may be the closing bracket.
    if !closed && family == Family::Bracket && body.len() >= 2 && matches!(body[body.len() - 1], '1' | 'l' | 'I' | '|' | '!' | 'J' | 'j') {
        let shorter = body[..body.len() - 1].iter().fold(0u32, |v, &c| v * 10 + digit_of(c).unwrap_or(0));
        if shorter > 0 {
            values.push(shorter);
        }
    }
    let _ = start;
    Some(Reading { family, values, exact, taken: k })
}

/// Whether recognition read the line (our own, or a scan's invisible layer): the only readings
/// that lose glyphs.
fn recognized(line: &Line) -> bool {
    line.source == "ocr" || line.spans.iter().any(|span| span.flags & INVISIBLE != 0)
}

struct Place {
    page: usize,
    line: usize,
    /// Rank in the document's reading order.
    rank: usize,
    reading: Option<Reading>,
    /// Whether the line opens a paragraph by its place: at the left edge, under a gap or a
    /// sentence's end or a short last line.
    opens: bool,
}

fn height(line: &Line) -> f64 {
    (line.bbox[3] - line.bbox[1]).max(0.1)
}

/// Every line of the scanned pages in reading order, with its number reading and whether its place
/// opens a paragraph.
fn places(pages: &[Page], scanned: &[usize]) -> Vec<Place> {
    let mut out = Vec::new();
    for &p in scanned {
        let Some(page) = pages.get(p) else { continue };
        let mut order: Vec<usize> = (0..page.lines.len()).filter(|&n| !page.lines[n].text.trim().is_empty()).collect();
        order.sort_by(|&a, &b| {
            let (la, lb) = (&page.lines[a], &page.lines[b]);
            (la.reading_order, la.bbox[1].to_bits()).cmp(&(lb.reading_order, lb.bbox[1].to_bits()))
        });
        let body: Vec<&Line> = order.iter().map(|&n| &page.lines[n]).filter(|l| l.text.split_whitespace().count() >= 3).collect();
        let right = {
            let mut xs: Vec<f64> = body.iter().map(|l| l.bbox[2]).collect();
            xs.sort_by(f64::total_cmp);
            xs.get(xs.len() * 9 / 10).copied().unwrap_or(page.width)
        };
        let left = {
            let mut xs: Vec<f64> = body.iter().map(|l| l.bbox[0]).collect();
            xs.sort_by(f64::total_cmp);
            xs.get(xs.len() / 10).copied().unwrap_or(0.0)
        };
        let width = (right - left).max(1.0);
        for (k, &n) in order.iter().enumerate() {
            let line = &page.lines[n];
            let h = height(line);
            let previous = k.checked_sub(1).map(|j| &page.lines[order[j]]);
            let opens = match previous {
                None => true,
                Some(prev) => {
                    let gap = line.bbox[1] - prev.bbox[3];
                    let ends = prev.text.trim_end().ends_with(['.', ';', ':', '?', '!', '”', '"', ')']);
                    let short = prev.bbox[2] < right - 0.25 * width;
                    gap > 0.35 * h || ends || short
                }
            };
            out.push(Place { page: p, line: n, rank: out.len(), reading: read_marker(&line.text), opens });
        }
    }
    out
}

/// Supported readings of one family: exact readings a neighbour at one more or one less supports,
/// the first of a run from its count's start.
fn supported(places: &[Place], family: Family) -> Vec<usize> {
    let exact: Vec<usize> = places.iter().enumerate()
        .filter(|(_, p)| p.reading.as_ref().is_some_and(|r| r.family == family && r.exact))
        .map(|(k, _)| k)
        .collect();
    let value = |k: usize| places[k].reading.as_ref().map(Reading::value).unwrap_or(0);
    exact.iter().enumerate().filter(|(i, &k)| {
        let v = value(k);
        let low = i.saturating_sub(SUPPORT_WINDOW);
        let high = (i + SUPPORT_WINDOW + 1).min(exact.len());
        exact[low..high].iter().any(|&other| other != k && (
            (other < k && value(other) + 1 == v) || (other > k && value(other) == v + 1)
        )) || (v == 1 && exact.get(i + 1).is_some_and(|&next| value(next) == 2))
    }).map(|(_, &k)| k).collect()
}

/// The column the family's numbers stand in: the median left edge of its supported lines, in
/// points, per page where the page has one, else the document's.
fn column(places: &[Place], pages: &[Page], supported: &[usize]) -> (f64, BTreeMap<usize, f64>) {
    let mut all = Vec::new();
    let mut by_page: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    for &k in supported {
        let x = pages[places[k].page].lines[places[k].line].bbox[0];
        all.push(x);
        by_page.entry(places[k].page).or_default().push(x);
    }
    let document = legal_pdf_core::upper_median(all);
    (document, by_page.into_iter().map(|(p, xs)| (p, legal_pdf_core::upper_median(xs))).collect())
}

/// What a candidate line fits: the values its own garbled token or a re-reading could be, or any value.
#[derive(Clone)]
enum Fit {
    Some(Vec<u32>),
    Any,
}

struct Candidate {
    place: usize,
    fit: Fit,
    reread: Option<OcrLine>,
}

/// The run's values assigned to candidates in order, each candidate taking a value it fits; the
/// longest such assignment, candidates with a value of their own first.
fn assign(candidates: &[Candidate], values: &[u32]) -> Vec<(usize, u32)> {
    let (n, m) = (candidates.len(), values.len());
    let fit = |c: &Candidate, v: u32| match &c.fit {
        Fit::Some(own) => if own.contains(&v) { 3 } else { 0 },
        Fit::Any => 1,
    };
    let mut score = vec![vec![0u32; m + 1]; n + 1];
    for x in 1..=n {
        for y in 1..=m {
            let f = fit(&candidates[x - 1], values[y - 1]);
            let take = if f > 0 { score[x - 1][y - 1] + f } else { 0 };
            score[x][y] = score[x - 1][y].max(score[x][y - 1]).max(take);
        }
    }
    let mut out = Vec::new();
    let (mut x, mut y) = (n, m);
    while x > 0 && y > 0 {
        let f = fit(&candidates[x - 1], values[y - 1]);
        if f > 0 && score[x][y] == score[x - 1][y - 1] + f {
            out.push((x - 1, values[y - 1]));
            x -= 1;
            y -= 1;
        } else if score[x][y] == score[x - 1][y] {
            x -= 1;
        } else {
            y -= 1;
        }
    }
    out.reverse();
    out
}

/// The margin to read again for a line at `column`: from a line height and a half left of the
/// column to two heights into the line, the line's band.
fn margin_region(line: &Line, column: f64) -> [f64; 4] {
    let h = height(line);
    [(column - 1.5 * h).max(0.0), line.bbox[1] - 0.25 * h, (column + 2.5 * h).min(line.bbox[2]), line.bbox[3] + 0.25 * h]
}

/// The number a re-reading of the margin shows at the line's band, if any: its possible values.
fn reread_value(lines: &[OcrLine], family: Family) -> Option<(Vec<u32>, bool, OcrLine)> {
    lines.iter().filter_map(|l| {
        let reading = read_marker(l.text.trim())?;
        (reading.family == family).then(|| (reading.values, reading.exact, l.clone()))
    }).max_by_key(|(_, exact, l)| (u8::from(*exact), (l.confidence * 1000.0) as u32))
}

/// The marker written into the line: the garbled token (if any) replaced, else the marker prepended;
/// words shift, and the marker's word takes the re-read box or an estimate left of the text.
fn splice(line: &mut Line, family: Family, value: u32, taken: usize, reread: Option<&OcrLine>) {
    let marker = family.write(value);
    let rest: String = line.text.chars().skip(taken).collect();
    let new_text = if rest.is_empty() { marker.clone() } else { format!("{marker} {rest}") };
    let delta = marker.chars().count() + usize::from(!rest.is_empty());
    let h = height(line);
    let text_start = line
        .words
        .iter()
        .filter(|w| w.start >= taken)
        .map(|w| w.bbox[0])
        .fold(f64::INFINITY, f64::min);
    let text_start = if text_start.is_finite() { text_start } else { line.bbox[0] };
    let marker_box = reread.map(|l| l.bbox).unwrap_or([
        (text_start - 0.6 * h * marker.chars().count() as f64 - 0.3 * h).max(0.0),
        line.bbox[1],
        text_start - 0.3 * h,
        line.bbox[3],
    ]);
    let mut words: Vec<Word> = line
        .words
        .drain(..)
        .filter(|w| w.start >= taken)
        .map(|mut w| {
            w.start = w.start - taken + delta;
            w.end = w.end - taken + delta;
            w
        })
        .collect();
    words.insert(0, Word {
        id: format!("{}-w000", line.id),
        text: marker.clone(),
        bbox: marker_box,
        start: 0,
        end: marker.chars().count(),
        underlined: false,
        struck: false,
        weight: 0.0,
    });
    for (k, word) in words.iter_mut().enumerate() {
        word.id = format!("{}-w{:03}", line.id, k + 1);
    }
    line.words = words;
    line.text = new_text;
    line.bbox[0] = line.bbox[0].min(marker_box[0]);
    if let Some(ocr) = &mut line.ocr {
        ocr.recovered = true;
    }
    if line.source != "ocr" {
        // A kept layer's line now carries our reading of its margin: its spans no longer match its text.
        line.spans.clear();
    }
}

/// Recover the lost paragraph numbers of the scanned pages; returns how many lines took a number.
pub(crate) fn recover(pdf: &[u8], pages: &mut [Page], scanned: &[usize], provider: &mut dyn PdfOcrProvider) -> usize {
    // LEGALPDF_PARAGRAPH_RECOVERY=off: the yardstick's baseline without this pass.
    if std::env::var("LEGALPDF_PARAGRAPH_RECOVERY").is_ok_and(|v| v == "off") {
        return 0;
    }
    let mut repaired = 0;
    for family in [Family::Bracket, Family::Dot] {
        let places = places(pages, scanned);
        let supported = supported(&places, family);
        if supported.len() < 2 {
            continue;
        }
        let (document_column, page_columns) = column(&places, pages, &supported);
        let column_at = |page: usize| page_columns.get(&page).copied().unwrap_or(document_column);
        let at_column = |place: &Place| {
            let line = &pages[place.page].lines[place.line];
            let h = height(line);
            (line.bbox[0] - column_at(place.page)).abs() <= h
        };
        let mut writes: Vec<(usize, usize, u32, usize, Option<OcrLine>)> = Vec::new();
        for pair in supported.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let (va, vb) = (places[a].reading.as_ref().unwrap().value(), places[b].reading.as_ref().unwrap().value());
            if vb <= va + 1 || vb - va - 1 > GAP_VALUES || places[b].rank <= places[a].rank + 1 {
                continue;
            }
            let values: Vec<u32> = (va + 1..vb).collect();
            // Candidates between: recognized lines at the column that open a paragraph, with a
            // garbled number of the family or none.
            let mut candidates: Vec<Candidate> = (places[a].rank + 1..places[b].rank).filter_map(|k| {
                let place = &places[k];
                let line = &pages[place.page].lines[place.line];
                if !recognized(line) || !at_column(place) {
                    return None;
                }
                match &place.reading {
                    Some(r) if r.family == family && !r.exact => Some(Candidate { place: k, fit: Fit::Some(r.values.clone()), reread: None }),
                    Some(_) => None,
                    None if place.opens => Some(Candidate { place: k, fit: Fit::Any, reread: None }),
                    None => None,
                }
            }).collect();
            if candidates.is_empty() {
                continue;
            }
            // Bare candidates read again in the margin where the count is not settled by position.
            let needs_reread = candidates.len() != values.len() || candidates.iter().any(|c| matches!(c.fit, Fit::Any));
            if needs_reread {
                for candidate in candidates.iter_mut().filter(|c| matches!(c.fit, Fit::Any)) {
                    let place = &places[candidate.place];
                    let page = &pages[place.page];
                    let line = &page.lines[place.line];
                    let region = margin_region(line, column_at(place.page));
                    let read = provider.read_region(pdf, place.page, region, REREAD_SCALE).unwrap_or_default();
                    if let Some((values, _exact, shown)) = reread_value(&read, family) {
                        candidate.fit = Fit::Some(values);
                        candidate.reread = Some(shown);
                    }
                }
            }
            // Position settles a gap the candidates fill exactly; otherwise only candidates with a
            // value of their own (garbled or re-read) are written.
            let exact_fill = candidates.len() == values.len();
            for (c, value) in assign(&candidates, &values) {
                let candidate = &candidates[c];
                if !exact_fill && matches!(candidate.fit, Fit::Any) {
                    continue;
                }
                let place = &places[candidate.place];
                let taken = place.reading.as_ref().map_or(0, |r| r.taken);
                writes.push((place.page, place.line, value, taken, candidate.reread.clone()));
            }
        }
        // The run's open ends: before its first and after its last supported value, a line at the
        // column takes the next value only where the margin re-read shows that very value.
        for (edge, forward) in [(supported[0], false), (*supported.last().unwrap(), true)] {
            let va = places[edge].reading.as_ref().unwrap().value();
            let mut expected = if forward { va + 1 } else { va.saturating_sub(1) };
            let range: Box<dyn Iterator<Item = usize>> = if forward {
                Box::new(places[edge].rank + 1..places.len())
            } else {
                Box::new((0..places[edge].rank).rev())
            };
            let mut misses = 0;
            for k in range {
                if expected == 0 || misses > 3 {
                    break;
                }
                let place = &places[k];
                let line = &pages[place.page].lines[place.line];
                if !recognized(line) || !at_column(place) || !place.opens {
                    continue;
                }
                if place.reading.as_ref().is_some_and(|r| r.exact) {
                    break;
                }
                let own: Vec<u32> = place.reading.as_ref().filter(|r| r.family == family).map(|r| r.values.clone()).unwrap_or_default();
                let region = margin_region(line, column_at(place.page));
                let read = provider.read_region(pdf, place.page, region, REREAD_SCALE).unwrap_or_default();
                let shown = reread_value(&read, family);
                let confirmed = shown.as_ref().is_some_and(|(v, _, _)| v.contains(&expected)) || own.contains(&expected);
                if confirmed {
                    let taken = place.reading.as_ref().map_or(0, |r| r.taken);
                    writes.push((place.page, place.line, expected, taken, shown.map(|(_, _, l)| l)));
                    expected = if forward { expected + 1 } else { expected - 1 };
                } else {
                    misses += 1;
                }
            }
        }
        for (page, line, value, taken, reread) in writes {
            splice(&mut pages[page].lines[line], family, value, taken, reread.as_ref());
            repaired += 1;
        }
    }
    repaired
}
