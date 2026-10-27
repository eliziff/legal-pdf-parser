//! Reading a PDF's print: the paragraphs a printed paragraph number addresses, in the body or
//! in the margin.

use std::collections::{BTreeMap, HashMap, HashSet};

// Printed paragraph numbers are addresses, not structural ordinal positions.
// Split on the actual native lines: a prose node can contain several numbered
// paragraphs, and one printed paragraph can contain several prose nodes.
// Parallel-language columns print one paragraph number per column. A line
// belongs to the column holding its centre, which mirrored margins preserve.
// Detached paragraph labels sit outside the body, unlike reporter page
// numbers. Their aligned body rows bound both columns in parallel text.
// Classification supplies the eligible body lines; note and furniture lines never enter here.
// A heading set out to the margin ("Circumstances of the offence") is no part of the body the
// numbers stand beside, and a paragraph ends where the next heading begins.
/// Each page's body bounds, and the paragraph numbers in the margin: number, page index, row top.
type MarginLabels = (Vec<Option<[f64; 4]>>, Vec<(usize, usize, f64)>);

/// Each page's body bounds, and the paragraph numbers standing in the margin or set apart at the
/// start of a body line, in reading order: each number, its page index and the top of its row.
fn margin_labels(pages: &[legal_pdf_support::PdfTextPage], headings: &HashSet<String>) -> MarginLabels {
    let body_bounds = pages
        .iter()
        .map(|page| {
            page.lines
                .iter()
                .filter(|line| {
                    line.rect[2] - line.rect[0] > page.width * 0.20
                        && !headings.contains(&line.id)
                        && line.text.chars().any(char::is_alphabetic)
                })
                .map(|line| line.rect)
                .reduce(|a, b| {
                    [
                        a[0].min(b[0]),
                        a[1].min(b[1]),
                        a[2].max(b[2]),
                        a[3].max(b[3]),
                    ]
                })
        })
        .collect::<Vec<_>>();
    let fixed = fixed_number_gaps(pages);
    let mut labels = Vec::new();
    for (page_index, page) in pages.iter().enumerate() {
        let Some(bounds) = body_bounds[page_index] else {
            continue;
        };
        for line in &page.lines {
            if let Some(number) = set_apart_number(page, line, &fixed) {
                labels.push((number, page_index, line.rect[1]));
                continue;
            }
            let text = line.text.trim();
            let text = text
                .strip_prefix('[')
                .and_then(|s| s.strip_suffix(']'))
                .unwrap_or(text);
            if text.is_empty() || text.len() > 5 || !text.bytes().all(|b| b.is_ascii_digit()) {
                continue;
            }
            let height = line.rect[3] - line.rect[1];
            let gap = (bounds[0] - line.rect[2]).max(line.rect[0] - bounds[2]);
            if gap < height * 0.5 || gap > height * 8.0 {
                continue;
            }
            let row = page
                .lines
                .iter()
                .filter(|peer| {
                    peer.rect[2] - peer.rect[0] > page.width * 0.20
                        && peer.text.chars().any(char::is_alphabetic)
                        && (peer.rect[1] - line.rect[1]).abs() <= height * 0.6
                })
                .map(|peer| peer.rect[1])
                .min_by(f64::total_cmp);
            if let (Some(y), Ok(number)) = (row, text.parse::<usize>()) {
                labels.push((number, page_index, y));
            }
        }
    }
    labels.sort_by(|a, b| a.1.cmp(&b.1).then(a.2.total_cmp(&b.2)));
    (body_bounds, labels)
}

/// Three numbers in a row: a numbering run, not an isolated number that could be a note's.
fn numbering_run(labels: &[(usize, usize, f64)]) -> bool {
    labels
        .windows(3)
        .any(|w| w[0].0 + 1 == w[1].0 && w[1].0 + 1 == w[2].0)
}

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

/// A number that opens a line of the body set apart from the text after it by more than a space
/// ("1      At the conclusion of the hearing", as Westlaw prints a judgment's paragraphs), by the
/// fixed separation a numbering run keeps (`fixed`), or by more than that and more than the line's
/// own spaces (Lexis's first paragraph, "1   When Robert King"): the paragraph's printed number,
/// standing bare. A number in running text ("11 of the Act") is followed by the line's ordinary
/// space; a page's running head and foot are no body.
fn set_apart_number(
    page: &legal_pdf_support::PdfTextPage,
    line: &legal_pdf_support::PdfTextLine,
    fixed: &[f64],
) -> Option<usize> {
    let (number, gap) = opening_number(page, line)?;
    let height = line.words[0].rect[3] - line.words[0].rect[1];
    let spaces = line.words[1..]
        .windows(2)
        .map(|pair| pair[1].rect[0] - pair[0].rect[2])
        .filter(|space| *space > 0.0)
        .collect::<Vec<_>>();
    let wider = spaces.len() >= 3 && gap > 1.2 * super::quantile(spaces, 0.75);
    (gap >= 0.75 * height
        || fixed
            .iter()
            .any(|width| (gap - width).abs() <= 0.05 || wider && gap > *width))
    .then_some(number)
}

/// A number of up to four digits that opens a body line, and the space after it. A note's number
/// ("6 Some words that", set smaller than the body) opens no paragraph.
fn opening_number(
    page: &legal_pdf_support::PdfTextPage,
    line: &legal_pdf_support::PdfTextLine,
) -> Option<(usize, f64)> {
    let [number, next, ..] = line.words.as_slice() else {
        return None;
    };
    let height = number.rect[3] - number.rect[1];
    let gap = next.rect[0] - number.rect[2];
    (number.text.len() <= 4
        && number.text.bytes().all(|byte| byte.is_ascii_digit())
        && height > 0.0
        && height >= 0.9 * body_height(page)
        && gap >= 0.3 * height
        && line.rect[1] > page.height * 0.05
        && line.rect[3] < page.height * 0.95)
        .then(|| number.text.parse().ok().map(|value| (value, gap)))
        .flatten()
}

/// The height most of a page's lines are set at: its body text's.
fn body_height(page: &legal_pdf_support::PdfTextPage) -> f64 {
    let mut heights = page
        .lines
        .iter()
        .map(|line| line.rect[3] - line.rect[1])
        .filter(|height| *height > 0.0)
        .collect::<Vec<_>>();
    heights.sort_by(f64::total_cmp);
    heights.get(heights.len() / 2).copied().unwrap_or_default()
}

/// The separations after a paragraph number that a numbering run keeps unchanged: Lexis sets each
/// number off by two unstretched spaces ("19  The standard of review") while justification
/// stretches the line's own. Three numbers in a row, each set off at least that far and two of
/// them by just that, to a twentieth of a point, make it the run's; running text's numbers take
/// each line's own stretched space.
fn fixed_number_gaps<'a>(
    pages: impl IntoIterator<Item = &'a legal_pdf_support::PdfTextPage>,
) -> Vec<f64> {
    let openings = pages
        .into_iter()
        .flat_map(|page| {
            page.lines
                .iter()
                .filter_map(move |line| opening_number(page, line))
        })
        .collect::<Vec<_>>();
    let mut fixed = Vec::<f64>::new();
    for &(_, gap) in &openings {
        if fixed.iter().any(|width| (gap - width).abs() <= 0.05) {
            continue;
        }
        let run = openings
            .iter()
            .filter(|(_, other)| *other >= gap - 0.05)
            .collect::<Vec<_>>();
        if run.windows(3).any(|w| {
            w[0].0 + 1 == w[1].0
                && w[1].0 + 1 == w[2].0
                && w.iter()
                    .filter(|(_, other)| (other - gap).abs() <= 0.05)
                    .count()
                    >= 2
        }) {
            fixed.push(gap);
        }
    }
    fixed
}

/// Detached paragraph labels share a body row and a numbering sequence, not a note band.
pub(super) fn margin_label_ids(pages: &[legal_pdf_core::model::Page]) -> HashSet<String> {
    use legal_pdf_support::{PdfTextLine, PdfTextPage};
    let witnesses = pages
        .iter()
        .map(|page| PdfTextPage {
            page_number: page.number,
            source: page.source.clone(),
            regions: page.regions.clone(),
            width: page.width,
            height: page.height,
            lines: page
                .lines
                .iter()
                .filter(|line| {
                    !line.exclude_from_body
                        && !matches!(line.region_type.as_str(), "header" | "footer")
                })
                .map(|line| PdfTextLine {
                    id: line.id.clone(),
                    text: line.text.clone(),
                    rect: line.bbox,
                    spans:line.spans.clone(),ocr:line.ocr.clone(),
                    region_id:line.region_id.clone(),region_type:line.region_type.clone(),source:line.source.clone(),
                    words: Vec::new(),
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    let (_, labels) = margin_labels(&witnesses, &HashSet::new());
    if !numbering_run(&labels) {
        return HashSet::new();
    }
    labels
        .iter()
        .flat_map(|(number, page_index, y)| {
            let page = &pages[*page_index];
            let body_size = super::median(
                page.lines
                    .iter()
                    .filter(|line| {
                        line.bbox[1] < page.height * 0.60
                            && line.text.chars().filter(|c| c.is_alphabetic()).count() >= 8
                    })
                    .map(legal_pdf_core::line_font_size)
                    .filter(|size| *size > 0.0)
                    .collect(),
            );
            page.lines
                .iter()
                .filter(move |line| {
                    let text = line
                        .text
                        .trim()
                        .trim_start_matches('[')
                        .trim_end_matches(']');
                    text.parse::<usize>().ok() == Some(*number)
                        && (line.bbox[1] - y).abs() <= (line.bbox[3] - line.bbox[1]) * 0.6
                        && page.lines.iter().any(|body| {
                            let size = legal_pdf_core::line_font_size(body);
                            body.bbox[2] - body.bbox[0] > page.width * 0.20
                                && body.text.chars().any(char::is_alphabetic)
                                && (body.bbox[1] - y).abs() < 1.0
                                && size >= body_size * 0.98
                                && size > legal_pdf_core::line_font_size(line) * 1.1
                        })
                })
                .map(|line| line.id.clone())
        })
        .collect()
}

/// A document's own paragraph numbers run on from one another; a number that breaks the run where
/// the run goes on around it is another text's, quoted (R. v. White, 2020 NSCA 33 sets R. v. L.M.'s
/// "[14]" and "[15]" between its own "[20]" and "[21]"). Per document context, the body's bracketed
/// numbers in reading order are sorted by `own_sequence`; a quoted number's paragraph (the composed
/// paragraph holding its line) is returned, paragraph by paragraph, as the lines of a quotation.
pub(super) fn foreign_numbered_paragraphs(
    pages: &[legal_pdf_core::model::Page],
    primitives: &super::PdfPrimitiveEvidence,
) -> Vec<Vec<String>> {
    let paragraphs = super::build_paragraphs_with_context(pages, &HashMap::new(), primitives);
    let mut paragraph_of = HashMap::<&str, usize>::new();
    for (k, paragraph) in paragraphs.iter().enumerate() {
        for id in &paragraph.line_ids {
            paragraph_of.insert(id.as_str(), k);
        }
    }
    let mut ordered = pages.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|page| page.index);
    let mut labels = BTreeMap::<super::DocumentContext, Vec<(usize, &str)>>::new();
    // The body's lines of each context in reading order, and each label's place among them.
    let mut body = BTreeMap::<super::DocumentContext, Vec<&str>>::new();
    let mut place = HashMap::<&str, usize>::new();
    let mut text_of = HashMap::<&str, &str>::new();
    for page in ordered {
        let mut lines = page
            .lines
            .iter()
            .filter(|line| {
                !line.exclude_from_body
                    && line.note_region_mode.is_empty()
                    && line.region_type == "body"
                    && !primitives.claims.table_cell_line_ids.contains(&line.id)
                    && !primitives.claims.contents_line_ids.contains(&line.id)
            })
            .collect::<Vec<_>>();
        lines.sort_by(|a, b| a.reading_order.cmp(&b.reading_order).then_with(|| a.id.cmp(&b.id)));
        for line in lines {
            let context = primitives.contexts.document_context(&line.id);
            let lines = body.entry(context).or_default();
            place.insert(line.id.as_str(), lines.len());
            text_of.insert(line.id.as_str(), line.text.as_str());
            lines.push(line.id.as_str());
            // A bracketed year ("[2008] 2 S.C.R. 3") numbers no paragraph.
            if let Some(number) = bracket_label(&line.text).filter(|number| *number < 1000) {
                labels.entry(context).or_default().push((number, line.id.as_str()));
            }
        }
    }
    let mut foreign = Vec::new();
    let mut taken = HashSet::new();
    if std::env::var_os("LEGALPDF_SEQUENCE_TRACE").is_some() {
        eprintln!("sequence	contexts {}	labels {:?}	paragraphs {}", labels.len(),
            labels.values().map(|l| l.iter().map(|(n, _)| *n).collect::<Vec<_>>()).collect::<Vec<_>>(), paragraphs.len());
    }
    for (context, labels) in labels {
        let own = own_sequence(&labels.iter().map(|(number, _)| *number).collect::<Vec<_>>());
        if std::env::var_os("LEGALPDF_SEQUENCE_TRACE").is_some() {
            eprintln!("sequence	foreign {} of {}	unmapped {}	paragraph ids like {:?}	label ids like {:?}", own.iter().filter(|o| !**o).count(), own.len(),
                labels.iter().zip(&own).filter(|((_, id), o)| !**o && !paragraph_of.contains_key(*id)).count(),
                paragraphs.first().and_then(|p| p.line_ids.first()), labels.first().map(|(_, id)| id));
        }
        let lines = &body[&context];
        for (k, ((_, id), own)) in labels.iter().zip(&own).enumerate() {
            if *own {
                continue;
            }
            // The composed paragraph holding the number; failing that, the number's line through to
            // the line before the next numbered one.
            if let Some(&k) = paragraph_of.get(id) {
                if paragraphs[k].region_type == "heading" || !taken.insert(k) {
                    continue;
                }
                foreign.push(paragraphs[k].line_ids.clone());
                continue;
            }
            // The number's line through to the line before the next numbered one or the next line
            // that opens an enumerated item, heading or set-apart passage of its own.
            let start = place[id];
            let next_label = labels.get(k + 1).and_then(|(_, next)| place.get(next).copied()).unwrap_or(lines.len());
            let mut end = start + 1;
            while end < next_label && !opens_item(text_of[lines[end]]) {
                end += 1;
            }
            foreign.push(lines[start..end].iter().map(|id| (*id).to_owned()).collect());
        }
    }
    foreign
}

/// Which of a reading's bracketed numbers, in order, are the document's own: the members of the
/// longest chain of consecutive numbers, and of further chains of three or more set wholly before
/// or after every chain found (numbering that restarts, or resumes past a number the reading lost);
/// a number set among an own chain's members and on no chain is quoted. Fewer than four numbers
/// are all the document's own.
fn own_sequence(values: &[usize]) -> Vec<bool> {
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
