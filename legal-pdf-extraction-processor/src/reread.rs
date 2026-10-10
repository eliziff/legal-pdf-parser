//! A scan's kept text layer read again by our recognizer where its small raised numbers may have
//! been lost: the layer's words stay, and the recognizer's digits take the places where the layer
//! has a stray mark, a letter a digit is misread as, or nothing, at a line's start (a note's label)
//! or glued to the word or stop before it (a reference). Ported from the Open Access Journals
//! Database's chain-gated re-read, our recognizer standing where its transcriber stood.

use legal_pdf_core::model::Line;

/// What a layer's recognizer leaves of a small raised number: quotes, primes, a degree sign, an
/// asterisk, a bar, or a letter in its place.
const LOST: &str = "'\u{2019}\u{2018}\"\u{201d}\u{201c}`\u{b0}*^|!sSIlti";

#[derive(Clone, Copy, PartialEq)]
enum Step {
    Keep,
    Change,
    Drop,
    Add,
}

/// The cheapest way from `a` to `b`, step by step.
fn alignment(a: &[char], b: &[char]) -> Vec<Step> {
    let (n, m) = (a.len(), b.len());
    let mut cost = vec![vec![0u32; m + 1]; n + 1];
    for i in 0..=n {
        cost[i][0] = i as u32;
    }
    for j in 0..=m {
        cost[0][j] = j as u32;
    }
    for i in 1..=n {
        for j in 1..=m {
            let change = cost[i - 1][j - 1] + u32::from(a[i - 1] != b[j - 1]);
            cost[i][j] = change.min(cost[i - 1][j] + 1).min(cost[i][j - 1] + 1);
        }
    }
    let (mut i, mut j) = (n, m);
    let mut steps = Vec::with_capacity(n.max(m));
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && cost[i][j] == cost[i - 1][j - 1] + u32::from(a[i - 1] != b[j - 1]) {
            steps.push(if a[i - 1] == b[j - 1] {
                Step::Keep
            } else {
                Step::Change
            });
            i -= 1;
            j -= 1;
        } else if i > 0 && cost[i][j] == cost[i - 1][j] + 1 {
            steps.push(Step::Drop);
            i -= 1;
        } else {
            steps.push(Step::Add);
            j -= 1;
        }
    }
    steps.reverse();
    steps
}

/// The edits the recognizer's reading makes to the layer's: (start, end, digits) over the layer's
/// scalars, each a number put where the layer lost it. `raised` marks the layer's scalars set as
/// superscripts: there a number the layer kept only part of ("3" for 30, "r" for 32) is a
/// reference, though the layer spaces it from the word before.
fn number_edits(layer: &[char], read: &[char], raised: &[bool]) -> Vec<(usize, usize, String)> {
    let steps = alignment(layer, read);
    let mut edits = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    let mut index = 0;
    // Where the run of kept scalars before the current edit starts.
    let mut kept_from = 0;
    while index < steps.len() {
        if steps[index] == Step::Keep {
            i += 1;
            j += 1;
            index += 1;
            continue;
        }
        let (mut start_i, mut start_j) = (i, j);
        while index < steps.len() && steps[index] != Step::Keep {
            match steps[index] {
                Step::Change => {
                    i += 1;
                    j += 1;
                }
                Step::Drop => i += 1,
                Step::Add => j += 1,
                Step::Keep => unreachable!(),
            }
            index += 1;
        }
        // The digits the layer kept beside the edit are the same number's.
        while start_i > kept_from && layer[start_i - 1].is_ascii_digit() {
            start_i -= 1;
            start_j -= 1;
        }
        while index < steps.len() && steps[index] == Step::Keep && layer[i].is_ascii_digit() {
            i += 1;
            j += 1;
            index += 1;
        }
        kept_from = i;
        let lost = &layer[start_i..i];
        let digits = read[start_j..j]
            .iter()
            .filter(|c| !c.is_whitespace())
            .collect::<String>();
        if !(1..=3).contains(&digits.len()) || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
            continue;
        }
        let kept = lost
            .iter()
            .filter(|c| !c.is_whitespace())
            .collect::<Vec<_>>();
        if kept.iter().copied().collect::<String>() == digits {
            continue;
        }
        // The recognizer adds the digits the layer lost; a digit the layer kept stays what it is.
        let mut read_digits = digits.chars();
        if !kept
            .iter()
            .filter(|c| c.is_ascii_digit())
            .all(|c| read_digits.any(|d| d == **c))
        {
            continue;
        }
        let superscript = (start_i..i)
            .chain(start_i.checked_sub(1))
            .chain((i < layer.len()).then_some(i))
            .any(|at| raised.get(at).copied().unwrap_or(false) && !layer[at].is_whitespace());
        let marks = kept
            .iter()
            .all(|c| LOST.contains(**c) || c.is_ascii_digit());
        if kept.len() > if superscript { 3 } else { 2 } || !(marks || superscript) {
            continue;
        }
        // A letter stands for a digit only alone, never inside a word it belongs to.
        let before = layer[..start_i]
            .iter()
            .rev()
            .find(|c| !c.is_whitespace())
            .copied();
        let after = layer.get(i).copied();
        if kept.iter().any(|c| c.is_alphabetic()) && after.is_some_and(char::is_alphabetic) {
            continue;
        }
        let label = before.is_none();
        let glued = before
            .is_some_and(|c| c.is_alphabetic() || ".,;:)\u{201d}\"\u{2019}'".contains(c))
            && start_i > 0
            && !layer[start_i - 1].is_whitespace()
            || kept.is_empty() && start_i > 0 && !layer[start_i - 1].is_whitespace();
        let ends = after.is_none_or(|c| c.is_whitespace() || ".,;:)".contains(c));
        if (label || glued || superscript) && ends {
            edits.push((start_i, i, digits));
        }
    }
    edits
}

/// Put the recognizer's numbers into a layer line where the layer lost them; returns how many.
pub(crate) fn splice_numbers(line: &mut Line, read: &str) -> usize {
    let layer = line.text.chars().collect::<Vec<_>>();
    let read = read.chars().collect::<Vec<_>>();
    if layer.is_empty() || read.is_empty() || layer.len() * read.len() > 400_000 {
        return 0;
    }
    let mut raised = vec![false; layer.len()];
    for span in line.spans.iter().filter(|span| span.superscript) {
        for at in span.start.min(layer.len())..span.end.min(layer.len()) {
            raised[at] = true;
        }
    }
    let edits = number_edits(&layer, &read, &raised);
    for (start, end, digits) in edits.iter().rev() {
        let mut text = layer_slice(&line.text, 0, *start);
        text.push_str(digits);
        text.push_str(&layer_slice(&line.text, *end, line.text.chars().count()));
        line.text = text;
        // Offsets before the edit stay, after it move with it, and inside it go to its edges.
        let (start, end, length) = (*start, *end, digits.chars().count());
        let moved = |at: usize| (at + start + length).saturating_sub(end);
        let opening = |at: usize| {
            if at <= start {
                at
            } else if at >= end {
                moved(at)
            } else {
                start
            }
        };
        let closing = |at: usize| {
            if at <= start {
                at
            } else if at >= end {
                moved(at)
            } else {
                start + length
            }
        };
        for word in &mut line.words {
            (word.start, word.end) = (opening(word.start), closing(word.end));
        }
        for span in &mut line.spans {
            (span.start, span.end) = (opening(span.start), closing(span.end));
        }
    }
    // Words stay whole, in order and apart, each reading as its stretch of the line.
    let count = line.text.chars().count();
    line.words.retain(|word| word.start < word.end.min(count));
    line.words.sort_by_key(|word| word.start);
    let mut words: Vec<legal_pdf_core::model::Word> = Vec::with_capacity(line.words.len());
    for mut word in line.words.drain(..) {
        word.end = word.end.min(count);
        match words.last_mut() {
            Some(prior) if word.start < prior.end => {
                prior.end = prior.end.max(word.end);
                prior.bbox = legal_pdf_core::union_bbox([prior.bbox, word.bbox]);
            }
            _ => words.push(word),
        }
    }
    for word in &mut words {
        word.text = layer_slice(&line.text, word.start, word.end);
    }
    line.words = words;
    for span in &mut line.spans {
        span.end = span.end.min(count);
        span.start = span.start.min(span.end);
        span.text = layer_slice(&line.text, span.start, span.end);
    }
    edits.len()
}

fn layer_slice(text: &str, start: usize, end: usize) -> String {
    text.chars()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect()
}

/// The recognizer's reading of a layer line: its words on the same row within the line's
/// stretch, left to right. The stretch reaches two line heights before the line, where a lost
/// note label stands, and a narrow mark's width after it, where a lost reference stands; a folio
/// or a running head's other half further along the row is no part of it.
pub(crate) fn reading_of(line: &Line, recognized: &[Line]) -> String {
    let height = line.bbox[3] - line.bbox[1];
    let (from, to) = (line.bbox[0] - 2.0 * height, line.bbox[2] + 0.6 * height);
    let mut matched = recognized
        .iter()
        .filter(|other| {
            let overlap = line.bbox[3].min(other.bbox[3]) - line.bbox[1].max(other.bbox[1]);
            let across = line.bbox[2].min(other.bbox[2]) - line.bbox[0].max(other.bbox[0]);
            let other_height = other.bbox[3] - other.bbox[1];
            height > 0.0
                && other_height > 0.0
                && overlap >= 0.5 * height.min(other_height)
                && across > 0.0
        })
        .flat_map(|other| {
            if other.words.is_empty() {
                return vec![(other.bbox[0], other.text.trim().to_owned())];
            }
            other
                .words
                .iter()
                .filter(|word| (from..=to).contains(&((word.bbox[0] + word.bbox[2]) / 2.0)))
                .map(|word| (word.bbox[0], word.text.clone()))
                .collect()
        })
        .collect::<Vec<_>>();
    matched.sort_by(|left, right| left.0.total_cmp(&right.0));
    matched
        .into_iter()
        .map(|(_, text)| text)
        .collect::<Vec<_>>()
        .join(" ")
}
