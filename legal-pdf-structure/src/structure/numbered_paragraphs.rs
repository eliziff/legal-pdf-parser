//! A document's printed paragraph numbers ("[12]") against those of another text it quotes: the
//! document's own run on from one another, and a quoted number's passage is a quotation.

use std::collections::HashSet;

/// "[12]" opening a line, before its words or alone: a printed paragraph number in the body.
fn bracket_label(text: &str) -> Option<usize> {
    let (number, rest) = text.trim_start().strip_prefix('[')?.split_once(']')?;
    if (!rest.is_empty() && !rest.starts_with(char::is_whitespace))
        || number.is_empty()
        || !number.chars().all(|c| c.is_ascii_digit())
    {
        return None;
    }
    number.parse().ok()
}

/// A body line's bracketed paragraph number ("[12] ..."); a bracketed year ("[2008] 2 S.C.R. 3")
/// numbers no paragraph.
pub(super) fn paragraph_number(text: &str) -> Option<usize> {
    bracket_label(text).filter(|number| *number < 1000)
}

/// Where a quoted number's passage ends among its reading's body lines (`text` of each, by place):
/// the number's line (at `start`) through to the line before the next numbered one (at
/// `next_label`), the next line that opens an enumerated item, heading or set-apart passage of its
/// own, or the first line that does not go on with the paragraph of the line before it
/// (`goes_on`); and the editorial lines set directly under it (`editorial_line`).
pub(super) fn passage_end<'a>(
    start: usize,
    next_label: usize,
    text: impl Fn(usize) -> &'a str,
    goes_on: impl Fn(usize) -> bool,
) -> usize {
    let mut end = start + 1;
    while end < next_label && goes_on(end) && !opens_item(text(end)) {
        end += 1;
    }
    while end < next_label && editorial_line(text(end)) {
        end += 1;
    }
    end
}

/// A quotation's editorial line, wholly in square brackets or an ellipsis alone: "[Emphasis
/// added]", "[Citations have been omitted.]", "[…]", "…", ". . .". A bracketed paragraph number is
/// none.
pub(super) fn editorial_line(text: &str) -> bool {
    let text = text.trim();
    let bracketed = text.starts_with('[')
        && text.trim_end_matches(['.', ',', ';']).ends_with(']')
        && text[1..].find(']').is_some_and(|close| close + 2 >= text.trim_end_matches(['.', ',', ';']).len())
        && bracket_label(text).is_none();
    let ellipsis = !text.is_empty()
        && text.chars().all(|c| matches!(c, '.' | '\u{2026}' | '[' | ']' | '(' | ')') || c.is_whitespace())
        && (text.contains('\u{2026}') || text.matches('.').count() >= 3);
    bracketed || ellipsis
}

/// Which of a reading's bracketed numbers, in order, are the document's own: the members of the
/// longest chain of consecutive numbers, and of further chains of three or more set wholly before
/// or after every chain found (numbering that restarts, or resumes past a number the reading lost);
/// a number set among an own chain's members and on no chain is quoted. Fewer than four numbers
/// are all the document's own.
pub(super) fn own_sequence(values: &[usize]) -> Vec<bool> {
    let n = values.len();
    let mut own = vec![true; n];
    if n < 4 {
        return own;
    }
    let mut decided = vec![false; n];
    let mut spans: Vec<(usize, usize)> = Vec::new();
    loop {
        let free: Vec<usize> = (0..n)
            .filter(|&i| !decided[i] && !spans.iter().any(|&(a, b)| a < i && i < b))
            .collect();
        if free.len() < 3 {
            break;
        }
        let mut best = vec![1usize; free.len()];
        let mut prev = vec![usize::MAX; free.len()];
        for j in 0..free.len() {
            for i in 0..j {
                if values[free[i]] + 1 == values[free[j]] && best[i] + 1 > best[j] {
                    best[j] = best[i] + 1;
                    prev[j] = i;
                }
            }
        }
        let (mut at, len) = best.iter().enumerate().map(|(k, &l)| (k, l)).max_by_key(|&(_, l)| l).unwrap_or((0, 0));
        if len < 3 {
            break;
        }
        let mut members = Vec::new();
        loop {
            members.push(free[at]);
            if prev[at] == usize::MAX {
                break;
            }
            at = prev[at];
        }
        for &m in &members {
            decided[m] = true;
        }
        spans.push((*members.iter().min().unwrap_or(&0), *members.iter().max().unwrap_or(&0)));
    }
    // A parallel text prints the whole sequence twice, the second copy interleaved with the first
    // page by page: a chain among the leftovers inside a span that is at least half the longest
    // chain's length is a parallel reading, the document's own; quotations are short fragments.
    let longest = spans.iter().map(|&(a, b)| (a..=b).filter(|&i| decided[i]).count()).max().unwrap_or(0);
    loop {
        let free: Vec<usize> = (0..n).filter(|&i| !decided[i] && spans.iter().any(|&(a, b)| a < i && i < b)).collect();
        if free.len() < 3 {
            break;
        }
        let mut best = vec![1usize; free.len()];
        let mut prev = vec![usize::MAX; free.len()];
        for j in 0..free.len() {
            for i in 0..j {
                if values[free[i]] + 1 == values[free[j]] && best[i] + 1 > best[j] {
                    best[j] = best[i] + 1;
                    prev[j] = i;
                }
            }
        }
        let (mut at, len) = best.iter().enumerate().map(|(k, &l)| (k, l)).max_by_key(|&(_, l)| l).unwrap_or((0, 0));
        if len < 3 || len * 2 < longest {
            break;
        }
        loop {
            decided[free[at]] = true;
            if prev[at] == usize::MAX {
                break;
            }
            at = prev[at];
        }
    }
    // A scanned parallel text's second copy may read too broken for a chain, but it still runs
    // upward: when the leftovers inside the spans mostly keep increasing (a non-decreasing
    // subsequence of at least four fifths of them, five or more), they are the document's own.
    let leftovers: Vec<usize> = (0..n).filter(|&i| !decided[i] && spans.iter().any(|&(a, b)| a < i && i < b)).collect();
    if leftovers.len() >= 5 {
        let mut tails: Vec<usize> = Vec::new();
        for &i in &leftovers {
            let at = tails.partition_point(|&t| t <= values[i]);
            if at == tails.len() { tails.push(values[i]); } else { tails[at] = values[i]; }
        }
        if tails.len() * 5 >= leftovers.len() * 4 {
            return own;
        }
    }
    // A quoted number duplicates one the document itself reaches; a number nothing else prints
    // is the reading's own, however it broke the chain (a misread digit).
    let printed: HashSet<usize> = (0..n).filter(|&i| decided[i]).map(|i| values[i]).collect();
    for &i in &leftovers {
        if printed.contains(&values[i]) {
            own[i] = false;
        }
    }
    own
}

/// A line that opens an enumerated item ("(a)", "1.", "iv)", "•"), a heading-like short capitalised
/// line, or another numbered paragraph: where a quoted paragraph's extent ends.
fn opens_item(text: &str) -> bool {
    let text = text.trim_start();
    if bracket_label(text).is_some() || text.starts_with(['•', '-', '–', '●', '▪']) {
        return true;
    }
    let head: String = text.chars().take_while(|c| !c.is_whitespace()).collect();
    let token = head.trim_start_matches('(');
    let item = token.ends_with([')', '.']) && {
        let core = token.trim_end_matches([')', '.']);
        !core.is_empty() && core.chars().count() <= 4
            && (core.chars().all(|c| c.is_ascii_digit() || c == '.') || core.chars().all(|c| matches!(c, 'a'..='z')) || core.chars().all(|c| matches!(c, 'A'..='Z')))
    };
    item || legal_pdf_core::layout_analysis().heading_text_plausible(text)
}
