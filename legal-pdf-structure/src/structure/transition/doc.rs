//! Detect-Order-Construct (Wang et al., Pattern Recognition 2024) in the engine: the stages `docparse.py`
//! fits on the gold and exports (LEGALPDF_TRANSITION_DOC, a folder of `detect.onnx`, `order.onnx`,
//! `headings.onnx`, `bodies.onnx`, `docparse.json`, and the text model's `tokenizer.json` and
//! `embeddings.f32`), and the paper's decoding of what they score into the parser's moves:
//! - Detect, a page at a time, its lines' states then given their templates' across the document (the parser's
//!   stylesheet, `stylesheet`: each template a node of its measured row and its lines' states) and each line its
//!   outline anchor (the level the bookmarks name it at): each line's role, its successor
//!   and predecessor among its page's lines, and
//!   two flags (a unit opening on its printed marker, a line opening a constituent document); the page's
//!   lines grouped into regions by union-find over the links both heads choose (or the successor's sure
//!   ones, over a half), none into a line whose role opens a unit; each region's role its lines' plurality;
//! - Order, over the document's regions (rotary positions over the parser's order; each pair of regions within a
//!   page of each other scored on their places too: their boxes, pages, distance in order, roles, and whether a
//!   note's number is a mark the other sets): the region a unit goes on in (a paragraph past a page or a
//!   quotation), a note's region the region holding its reference mark, and each region's successor in the
//!   reading, the regions read by following it (the next unread region in the parser's order where it is unsure)
//!   and each page's lines written in that order;
//! - Construct, over the headings that open a unit and over the body's: each node's parent and left sibling
//!   among the nodes before it, the tree built by the paper's Algorithm 1 (each node the right-most child of
//!   the node on the tree's right-most path whose parent score times its child's sibling score is greatest),
//!   a constituent document's start emptying it; a heading's level, or a body unit's depth, its parent's and
//!   its gap (how many levels under its parent it stands: the print may skip one).
//!
//! The decoding after the heads is the oracle's too (LEGALPDF_TRANSITION_DOC_ORACLE, the gold given): the
//! heads' answers read off the gold, so that what it grades below the gold's own is what the decoding and
//! the moves cannot hold, not what the heads miss.
//!
//! A line's text is the text model's reading of it as Model2Vec reads one: its tokens (no special ones, the
//! unknown token left out) averaged and scaled to unit length.

use super::*;

/// The regions whose units take a depth in the body's tree (`docparse.BODY`): all that the gold's depth steps chain
/// over (`RANKED_BODY`).
const BODY: [u8; 7] = RANKED_BODY;
/// A heading's or body unit's gap under its parent is one of 1..=GAPS levels.
pub(crate) const GAPS: usize = 4;

/// What the line heads say, as the decoding reads it: each line's role (`region * 2 + opens`), whether it
/// opens a unit on its printed marker, whether it opens a constituent document, and the line it links to as
/// its region's next (itself: none).
struct Lines {
    roles: Vec<usize>,
    marked: Vec<bool>,
    starts: Vec<bool>,
    links: Vec<usize>,
}

/// What the region heads say over the regions read (those of a role): each one's earlier part where it goes
/// on in an earlier region's unit, its level or depth where it opens a heading or a body unit, and a note's
/// region the region holding its mark.
struct Units {
    before: Vec<Option<usize>>,
    level: Vec<usize>,
    note: Vec<Option<usize>>,
}

fn find(group: &mut [usize], k: usize) -> usize {
    let mut root = k;
    while group[root] != root {
        root = group[root];
    }
    let mut at = k;
    while group[at] != root {
        let next = group[at];
        group[at] = root;
        at = next;
    }
    root
}

/// The regions: the lines' groups under their links, each in reading order and the groups by their first
/// lines; each one's role its lines' plurality (the first line's on a tie).
fn regions_of(lines: &Lines) -> (Vec<Vec<usize>>, Vec<u8>) {
    let n = lines.roles.len();
    let mut group: Vec<usize> = (0..n).collect();
    for (k, &next) in lines.links.iter().enumerate() {
        if next != k {
            let (a, b) = (find(&mut group, k), find(&mut group, next));
            group[a.max(b)] = a.min(b);
        }
    }
    let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
    for k in 0..n {
        let root = find(&mut group, k);
        members.entry(root).or_default().push(k);
    }
    let mut regions: Vec<Vec<usize>> = members.into_values().collect();
    regions.sort_by_key(|lines| lines[0]);
    let roles = regions.iter().map(|members| {
        let mut counts = [0usize; 17];
        for &k in members {
            counts[lines.roles[k] / 2] += 1;
        }
        let first = lines.roles[members[0]] / 2;
        (0..17).fold(first, |best, r| if counts[r] > counts[best] { r } else { best }) as u8
    }).collect();
    (regions, roles)
}

/// The paragraphs t0 decided in the regions' links: a line the judgment reader reads opening a numbered paragraph
/// (`LineView::count`) takes no link from any other line. The margin's paragraph numbers (`LineTable::margin_rows`):
/// a number and the row's line it numbers are one region, the number its first line (it is read just before it);
/// the row's line and its twin's in the other column take no link from any other line, each opening its column's
/// paragraph, nor does the number.
fn margin_links(links: &mut [usize], margin_rows: &HashMap<usize, (usize, Option<usize>)>, views: &[LineView]) {
    let mut opened: std::collections::HashSet<usize> = margin_rows.iter().flat_map(|(&margin, &(row, twin))| [margin, row].into_iter().chain(twin)).collect();
    opened.extend((0..views.len()).filter(|&k| views[k].count.is_some()));
    for (k, next) in links.iter_mut().enumerate() {
        if opened.contains(next) && !margin_rows.contains_key(&k) {
            *next = k;
        }
    }
    for (&margin, &(row, _)) in margin_rows {
        links[margin] = row;
    }
}

/// A body region whose first line Detect reads going on (its role's opening bit unset) and that Order puts in no
/// earlier unit goes on in the last region before it of its role in its column's language (a paragraph past the
/// page's foot, or read past the other column's lines): the region read last of that role and language, on its page
/// or the page before (as Order pairs regions), its unit the one open in that stream. A region the margin numbers
/// opens.
fn goes_on_in_column(before: &mut [Option<usize>], regions: &[Vec<usize>], roles: &[u8], read: &[usize], line_roles: &[usize], views: &[LineView]) {
    let mut last: HashMap<(u8, bool), usize> = HashMap::new();
    for i in 0..read.len() {
        let first = regions[read[i]][0];
        let key = (roles[read[i]], views[first].translation);
        if before[i].is_none() && BODY.contains(&key.0) && line_roles[first] % 2 == 0 && views[first].margin.is_none() && views[first].count.is_none() {
            let page = views[first].page;
            before[i] = last.get(&key).copied().filter(|&earlier| views[regions[read[earlier]][0]].page + 1 >= page);
        }
        last.insert(key, i);
    }
}

/// The paper's Algorithm 1 over `n` nodes: each node in turn the right-most child of the node on the tree's
/// right-most path (the root first) whose parent probability times the node's sibling probability for that
/// node's last child (itself where it has none) is greatest; `resets[i]` empties the tree before node `i`.
/// Each node's level: its parent's (0 for the root) and its gap.
fn insert(parent: &[f32], sibling: &[f32], gaps: &[usize], n: usize, resets: &[bool]) -> Vec<usize> {
    let mut path: Vec<usize> = Vec::new();
    let mut levels = vec![0usize; n];
    for i in 0..n {
        if resets[i] {
            path.clear();
        }
        let p = |j: usize| parent[i * n + j];
        let s = |j: usize| sibling[i * n + j];
        // Under the root: no parent (itself), its sibling the root's last child, the path's first node.
        let mut best = (0usize, p(i) * s(path.first().copied().unwrap_or(i)));
        for (at, &node) in path.iter().enumerate() {
            let score = p(node) * s(path.get(at + 1).copied().unwrap_or(i));
            if score > best.1 {
                best = (at + 1, score);
            }
        }
        path.truncate(best.0);
        let under = path.last().map_or(0, |&node| levels[node]);
        levels[i] = under + gaps[i].max(1);
        path.push(i);
    }
    levels
}

/// A note's number: its label's where the label reader found one, else an arabic reading's, else the figures
/// its text opens on ("27" of "27 Ibid", as the gold's notes are numbered), else 0 for a note symbol, as a reference
/// mark's value is (`reference_spans`).
pub(super) fn note_number(line: &LineView, text: &str) -> Option<u32> {
    line.label
        .or_else(|| line.readings.iter().find(|r| r.family.chars().all(|c| !c.is_alphabetic())).map(|r| r.value))
        .or_else(|| super::emit::leading_figures(text).map(|(figures, _)| figures))
        .or_else(|| text.trim_start().starts_with(['*', '\u{2020}', '\u{2021}', '\u{a7}']).then_some(0))
}

/// The document's moves from the heads' answers: each region's lines its role's; a unit's first line opens it
/// (at its level or depth, counting on its marker where the line is marked, a note on its number); a line of a
/// unit another unit of its stream came before goes on with it by name (its first line); a note answers the
/// marks of its number in the region its link names.
fn moves_of(lines: &Lines, regions: &[Vec<usize>], roles: &[u8], read: &[usize], units: &Units, views: &[LineView], texts: &[&str], only: Option<&[bool]>) -> Vec<Move> {
    let n = lines.roles.len();
    let mut moves = vec![Move { region: 0, opens: false, reading: None, note: None, level: 0, depth: 0, resume: false, resumes: None, rank: None, starts: false, answers: [None; 4] }; n];
    for (k, mv) in moves.iter_mut().enumerate() {
        mv.starts = lines.starts[k];
    }
    let mut of_line = vec![usize::MAX; n];
    for (i, &r) in read.iter().enumerate() {
        for &k in &regions[r] {
            of_line[k] = i;
        }
    }
    // Each region's unit, its first line: a region going on in an earlier one's unit is in that unit.
    let mut unit = vec![usize::MAX; read.len()];
    for i in 0..read.len() {
        unit[i] = match units.before[i] {
            Some(earlier) if earlier < i => unit[earlier],
            _ => regions[read[i]][0],
        };
    }
    // The unit each stream last read a line of, and the open marker frames per context for disambiguating
    // Roman vs. alphabetic readings (e.g. `(i)`, `(v)`, `(x)`).
    let mut last_in_stream: HashMap<usize, usize> = HashMap::new();
    let mut doc_id = 0u16;
    let mut stacks: HashMap<u16, Open> = HashMap::new();
    for k in 0..n {
        if lines.starts[k] {
            doc_id = doc_id.saturating_add(1);
        }
        let i = of_line[k];
        if i == usize::MAX {
            continue;
        }
        let region = roles[read[i]];
        let stream = stream_of(region);
        let ctx = doc_id.saturating_mul(2) + u16::from(views[k].translation);
        let stands = match (unit[i] == k, last_in_stream.insert(stream, unit[i])) {
            (true, _) => Stands::Opens,
            (false, Some(last)) if last == unit[i] => Stands::GoesOn(last),
            _ => Stands::Resumes(unit[i]),
        };
        if matches!(stands, Stands::Opens) {
            let open = stacks.entry(ctx).or_default();
            if region == QUOTE {
                if !open.quoting {
                    open.quoted.clear();
                }
                open.quoting = true;
            } else if region == TEXT {
                open.quoting = false;
            }
            let chosen_reading = if region != NOTE && region != NOTE_QUOTE && lines.marked[k] {
                let t0_counted = views[k].count.and_then(|n| views[k].readings.iter().position(|r| r.value == n));
                let stack_matched = if views[k].readings.len() > 1 {
                    views[k].readings.iter().position(|r| {
                        open.frames().iter().rev().any(|f| f.family == r.family && r.value == f.value + 1)
                    })
                } else {
                    None
                };
                Some(match (views[k].margin, t0_counted, stack_matched, views[k].readings.is_empty()) {
                    (Some(_), ..) => MARGIN,
                    (_, Some(r), ..) => r as u8,
                    (_, None, Some(r), false) => r as u8,
                    (_, None, None, false) => 0,
                    _ => OPENING,
                })
            } else {
                None
            };
            if let Some(reading) = views[k].reading(chosen_reading) {
                let next_frames = machine::counted(open.frames(), reading, region);
                if open.quoting {
                    open.quoted = next_frames;
                } else {
                    open.stack = next_frames;
                }
            }
            let mv = &mut moves[k];
            mv.region = region;
            mv.opens = true;
            if region == HEADING {
                mv.level = units.level[i].clamp(1, 9) as i8;
            } else if BODY.contains(&region) {
                mv.depth = units.level[i].clamp(1, 9) as u8;
            }
            if region == NOTE || region == NOTE_QUOTE {
                mv.note = note_number(&views[k], texts[k]);
            } else {
                mv.reading = chosen_reading;
            }
        } else {
            let mv = &mut moves[k];
            mv.region = region;
            if let Stands::Resumes(first) = stands {
                mv.resume = true;
                mv.resumes = Some(first as u32);
            }
        }
    }
    for (i, target) in units.note.iter().enumerate() {
        let first = regions[read[i]][0];
        let Some(value) = moves[first].opens.then_some(moves[first].note).flatten() else { continue };
        // A note the heads pair with no mark answers none, not the last of its number the text set.
        let Some(target) = target else {
            moves[first].answers[0] = Some((u32::MAX, 0, 0));
            continue;
        };
        // Of the region's marks of its number, those on the lines `only` names, where it names any (the gold's).
        let marks: Vec<(u32, u32, u32)> = regions[read[*target]].iter().filter(|&&k| only.is_none_or(|lines| lines[k]))
            .flat_map(|&k| views[k].references.iter().filter(move |&&(v, ..)| v == value).map(move |&(_, s, e)| (views[k].place, s, e)))
            .take(4).collect();
        if marks.is_empty() {
            moves[first].answers[0] = Some((u32::MAX, 0, 0));
            continue;
        }
        for (slot, mark) in moves[first].answers.iter_mut().zip(marks) {
            *slot = Some(mark);
        }
    }
    moves
}

/// A constituent document's start between each node and the one before (the first: from the document's top)
/// empties the tree before it.
fn resets(firsts: &[usize], starts: &[bool]) -> Vec<bool> {
    (0..firsts.len()).map(|i| {
        let from = if i == 0 { 0 } else { firsts[i - 1] + 1 };
        (from..=firsts[i]).any(|k| starts[k])
    }).collect()
}

/// The parser's stylesheet as the DOC reader reads it: each line's template and outline anchor, and each template's
/// measured row (`width` a row), the first reading's own measures left out (the reader reads before any).
pub(crate) struct Style {
    pub(crate) templates: Vec<u32>,
    pub(crate) anchors: Vec<u16>,
    pub(crate) rows: Vec<f32>,
    pub(crate) width: usize,
}

impl Style {
    pub(crate) fn of(sheet: &Stylesheet, table: &LineTable) -> Self {
        let rows = joint::template_rows(sheet, table);
        let width = rows.first().map_or(1, |r| r.len());
        let mut flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        for row in flat.chunks_mut(width) {
            // How the walker's first reading read the template (`read_heading`, `read_level`).
            row[9] = 0.0;
            row[10] = 0.0;
        }
        if flat.is_empty() {
            flat = vec![0.0; width];
        }
        let flat = flat.into_iter().map(|v| if v.is_finite() { v.clamp(-8.0, 8.0) } else { 0.0 }).collect();
        Self { templates: sheet.template.clone(), anchors: sheet.anchor.clone(), rows: flat, width }
    }
}

// ------------------------------------------------------------------------------------------------- the oracle
/// The gold's region of a row (an unmarked body line: text).
fn gold_region(g: &[i64; 8]) -> i64 {
    if g[0] == UNMARKED_BODY { 1 } else { g[0] }
}

/// The document's moves as the decoding makes them from the heads' answers read off the gold (`gold`, each
/// line's row): each line's role, its flags and its next line in its unit on its page; each region's earlier
/// part, its tree (each node under the nearest open node of a smaller level, the left sibling that parent's
/// last child), its gap, and a note's region the region before it (two pages back at most) holding a mark of
/// its number the gold pairs; read through the same `insert` and `moves_of` as the model's.
pub(crate) fn oracle(gold: &[[i64; 8]], ranks: &[i64], views: &[LineView], texts: &[&str], pages: &[usize]) -> Vec<Move> {
    let n = gold.len();
    // Each line's unit, as the gold's decoder reads its rows (`docparse.gold_units`).
    let mut unit = vec![usize::MAX; n];
    let mut open: HashMap<usize, usize> = HashMap::new();
    // A line the gold reads first in a unit the parser reads later (the two orders differ) names a line after it: that
    // line goes on in this one's unit.
    let named_later: HashMap<usize, usize> = gold.iter().enumerate().filter(|(k, g)| g[1] == 2 && usize::try_from(g[6]).is_ok_and(|p| p > *k)).map(|(k, g)| (g[6] as usize, k)).collect();
    for (k, g) in gold.iter().enumerate() {
        let region = gold_region(g);
        if region <= 0 {
            continue;
        }
        let stream = stream_of(region as u8);
        let pointer = usize::try_from(g[6]).ok().filter(|&p| p < k && unit[p] != usize::MAX);
        unit[k] = match (g[1], pointer, open.get(&stream), named_later.get(&k)) {
            (_, _, _, Some(&earlier)) if unit[earlier] != usize::MAX => unit[earlier],
            (2, Some(p), ..) => unit[p],
            (0, _, Some(&u), _) if gold_region(&gold[u]) == region => u,
            _ => k,
        };
        open.insert(stream, unit[k]);
    }
    // Each line's next in its unit on its page.
    let mut links: Vec<usize> = (0..n).collect();
    let mut last: HashMap<(usize, usize), usize> = HashMap::new();
    for k in 0..n {
        if unit[k] != usize::MAX {
            if let Some(&before) = last.get(&(unit[k], pages[k])) {
                links[before] = k;
            }
            last.insert((unit[k], pages[k]), k);
        }
    }
    let lines = Lines {
        roles: gold.iter().map(|g| gold_region(g).max(0) as usize * 2 + usize::from(g[1] == 1)).collect(),
        marked: gold.iter().map(|g| g[1] == 1 && g[5] == 1).collect(),
        starts: gold.iter().map(|g| g[2] == 1).collect(),
        links,
    };
    let (regions, roles) = regions_of(&lines);
    let read: Vec<usize> = (0..regions.len()).filter(|&r| roles[r] != 0).collect();
    let count = read.len();
    let first = |i: usize| regions[read[i]][0];
    // A unit's later regions go on in its earlier ones.
    let mut before = vec![None; count];
    let mut unit_region: HashMap<usize, usize> = HashMap::new();
    for i in 0..count {
        let u = unit[first(i)];
        if u == usize::MAX {
            continue;
        }
        before[i] = unit_region.insert(u, i);
    }
    // The trees, from the gold's levels (`docparse.absolute`, `docparse.tree`), read through `insert`.
    let mut level = vec![0usize; count];
    for (column, wanted) in [(4usize, &[HEADING][..]), (7, &BODY[..])] {
        let mut absolute = vec![0usize; n];
        let mut current = 0i64;
        for (k, g) in gold.iter().enumerate() {
            if g[2] == 1 {
                current = 0;
            }
            if wanted.contains(&(gold_region(g).max(0) as u8)) && g[1] == 1 {
                if g[column] < 0 {
                    current = 0;
                    continue;
                }
                current = (current.max(1) + g[column] - 10).clamp(1, 9);
                absolute[k] = current as usize;
            }
        }
        let nodes: Vec<usize> = (0..count).filter(|&i| before[i].is_none() && wanted.contains(&roles[read[i]])).collect();
        let h = nodes.len();
        if h == 0 {
            continue;
        }
        let (mut parent, mut sibling, mut gaps) = (vec![0f32; h * h], vec![0f32; h * h], vec![1usize; h]);
        let starts: Vec<bool> = lines.starts.clone();
        let firsts: Vec<usize> = nodes.iter().map(|&i| first(i)).collect();
        let reset = resets(&firsts, &starts);
        let mut stack: Vec<usize> = Vec::new();
        let mut last_child: HashMap<Option<usize>, usize> = HashMap::new();
        for (a, &i) in nodes.iter().enumerate() {
            if reset[a] {
                stack.clear();
                last_child.clear();
            }
            // A node of unknown level stands one under the open node.
            let mine = if absolute[first(i)] > 0 { absolute[first(i)] } else { stack.last().map_or(1, |&b| absolute[firsts[b]].max(1) + 1) };
            while stack.last().is_some_and(|&b| absolute[firsts[b]].max(1) >= mine) {
                stack.pop();
            }
            let p = stack.last().copied();
            parent[a * h + p.unwrap_or(a)] = 1.0;
            sibling[a * h + last_child.get(&p).copied().unwrap_or(a)] = 1.0;
            gaps[a] = mine - p.map_or(0, |b| absolute[firsts[b]].max(1));
            last_child.insert(p, a);
            stack.push(a);
        }
        for (&i, l) in nodes.iter().zip(insert(&parent, &sibling, &gaps, h, &reset)) {
            level[i] = l;
        }
    }
    // A note's region: the gold's paired mark of its number nearest before it, else after it, wherever it stands.
    let mut region_of = vec![usize::MAX; n];
    for (i, &r) in read.iter().enumerate() {
        for &k in &regions[r] {
            region_of[k] = i;
        }
    }
    // Each mark is one note's; a parallel report's two notes of a number take the marks of their own languages.
    let mut taken: HashSet<(usize, u32)> = HashSet::new();
    let note = (0..count).map(|i| {
        let k = first(i);
        if (roles[read[i]] != NOTE && roles[read[i]] != NOTE_QUOTE) || before[i].is_some() {
            return None;
        }
        let value = note_number(&views[k], texts[k])?;
        let paired = |j: &usize| gold[*j][3] == 1 && !taken.contains(&(*j, value)) && region_of[*j] != usize::MAX && views[*j].references.iter().any(|&(v, ..)| v == value);
        let same = |j: &usize| views[*j].translation == views[k].translation;
        let found = (0..k).rev().filter(paired).find(same).or_else(|| (k + 1..n).filter(paired).find(same))
            .or_else(|| (0..k).rev().find(paired)).or_else(|| (k + 1..n).find(paired));
        if let Some(j) = found {
            taken.insert((j, value));
        }
        found.map(|j| region_of[j])
    }).collect();
    let gold_marks: Vec<bool> = gold.iter().map(|g| g[3] == 1).collect();
    let mut moves = moves_of(&lines, &regions, &roles, &read, &Units { before, level, note }, views, texts, Some(&gold_marks));
    read_in(&mut moves, ranks, pages);
    moves
}

/// The reading's order of the lines: each page's lines by their ranks (a line without one just after the line
/// before it), as places in the document.
fn read_in(moves: &mut [Move], ranks: &[i64], pages: &[usize]) {
    let n = moves.len();
    if ranks.iter().all(|&r| r < 0) {
        return;
    }
    // A line without a rank takes the last rank before it on its page, read after that line.
    let mut keyed: Vec<(usize, i64, usize)> = Vec::with_capacity(n);
    let mut last = (usize::MAX, -1i64);
    for k in 0..n {
        if pages[k] != last.0 {
            last = (pages[k], -1);
        }
        if ranks[k] >= 0 {
            last.1 = ranks[k];
        }
        keyed.push((pages[k], last.1, k));
    }
    keyed.sort();
    for (place, &(.., k)) in keyed.iter().enumerate() {
        moves[k].rank = Some(place as u32);
    }
}

// ------------------------------------------------------------------------------------------------- the model
#[cfg(feature = "encoder")]
pub(crate) use model::Reader;

#[cfg(feature = "encoder")]
mod model {
    use super::*;
    use ort::session::{builder::GraphOptimizationLevel, Session};
    use ort::value::TensorRef;
    use std::collections::HashSet;
    use std::sync::Mutex;

    #[derive(serde::Deserialize)]
    struct Meta {
        width: usize,
        text_dim: usize,
        d: usize,
        classes: usize,
        mean: Vec<f32>,
        std: Vec<f32>,
        /// A template row's width (the stylesheet's measures, `joint::template_rows`).
        row_width: usize,
        vocabulary: usize,
        unknown: Option<u32>,
        normalize: bool,
    }

    #[derive(serde::Deserialize)]
    struct Parity {
        texts: Vec<String>,
        vectors: Vec<Vec<f32>>,
    }

    /// One set of the stages' sessions: a parse holds one for its document, parses at once each their own (made
    /// as they are wanted, kept for the next), none waiting on another's.
    struct Sessions {
        detect: Session,
        stylesheet: Session,
        lines: Session,
        order: Session,
        headings: Session,
        bodies: Session,
    }

    impl Sessions {
        fn load(dir: &Path) -> std::result::Result<Self, String> {
            Ok(Self {
                detect: session(&dir.join("detect.onnx"))?,
                stylesheet: session(&dir.join("stylesheet.onnx"))?,
                lines: session(&dir.join("lines.onnx"))?,
                order: session(&dir.join("order.onnx"))?,
                headings: session(&dir.join("headings.onnx"))?,
                bodies: session(&dir.join("bodies.onnx"))?,
            })
        }
    }

    pub(crate) struct Reader {
        dir: PathBuf,
        sessions: Mutex<Vec<Sessions>>,
        /// Whether each tree's graph gives its nodes' gaps (`gap`).
        gaps: [bool; 2],
        /// Whether each tree's graph takes the pairwise grammar and geometry inputs (`family`).
        tree_pairs: [bool; 2],
        /// Whether the reading's order is read off the successor head (by default the parser's order is kept:
        /// ordered region by region, a transcript's line numbers would leave their lines).
        reorder: bool,
        /// Whether each document's stages' times are written, for the reader's speed.
        times: bool,
        tokenizer: tokenizers::Tokenizer,
        embeddings: Vec<f32>,
        meta: Meta,
    }

    /// A successor link the predecessor head does not choose back is kept where the successor head holds it
    /// likelier than this.
    const SURE: f32 = 0.5;
    /// The reading leaves the parser's order where the successor head holds another region at least this likely
    /// (and the parser's next at most its complement).
    const SURE_ORDER: f32 = 0.9;
    /// The Order graph's attention block (`docparse.BLOCK`): its regions come padded to whole blocks.
    const BLOCK: usize = 64;

    fn session(path: &Path) -> std::result::Result<Session, String> {
        Session::builder()
            .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
            .and_then(|b| b.with_intra_threads(1))
            .and_then(|b| b.with_inter_threads(1))
            .and_then(|b| b.commit_from_file(path))
            .map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Each row of a `width`-wide matrix made a probability distribution.
    fn softmax_rows(values: &[f32], width: usize) -> Vec<f32> {
        values.chunks_exact(width).flat_map(|row| {
            let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let total: f32 = row.iter().map(|x| (x - top).exp()).sum();
            row.iter().map(move |x| (x - top).exp() / total)
        }).collect()
    }

    fn argmax(row: &[f32]) -> usize {
        row.iter().enumerate().fold((0, f32::NEG_INFINITY), |best, (i, &x)| if x > best.1 { (i, x) } else { best }).0
    }

    /// A tree head's scores over the nodes before each node and itself (none) as probabilities: each row's
    /// later nodes left out.
    fn earlier(values: &[f32], n: usize) -> Vec<f32> {
        let masked: Vec<f32> = values.iter().enumerate().map(|(at, &x)| if at % n > at / n { f32::NEG_INFINITY } else { x }).collect();
        softmax_rows(&masked, n)
    }

    impl Reader {
        /// The stages under `dir`, their ONNX Runtime the one LEGALPDF_ONNX_RUNTIME or LEGALPDF_ONNXRUNTIME
        /// names; the text model's reading checked against Python's on `parity.json`'s texts.
        pub(crate) fn load(dir: &Path, reorder: bool, times: bool) -> std::result::Result<Self, String> {
            let read = |name: &str| std::fs::read_to_string(dir.join(name)).map_err(|e| format!("{name}: {e}"));
            let meta: Meta = serde_json::from_str(&read("docparse.json")?).map_err(|e| format!("docparse.json: {e}"))?;
            if meta.width != ALL || meta.classes != CLASSES || meta.mean.len() != ALL || meta.std.len() != ALL {
                return Err(format!("docparse.json reads {} features and {} classes; the parser has {ALL} and {CLASSES}", meta.width, meta.classes));
            }
            let runtime = ["LEGALPDF_ONNX_RUNTIME", "LEGALPDF_ONNXRUNTIME"].iter().find_map(|name| std::env::var_os(name).filter(|v| !v.is_empty()))
                .ok_or("the DOC reader needs LEGALPDF_ONNX_RUNTIME")?;
            legal_pdf_core::init_ort_runtime(Path::new(&runtime), None).map_err(|e| e.to_string())?;
            let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| format!("tokenizer.json: {e}"))?;
            let bytes = std::fs::read(dir.join("embeddings.f32")).map_err(|e| format!("embeddings.f32: {e}"))?;
            let embeddings: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            if embeddings.len() != meta.vocabulary * meta.text_dim {
                return Err(format!("embeddings.f32 holds {} values; docparse.json says {} by {}", embeddings.len(), meta.vocabulary, meta.text_dim));
            }
            let first = Sessions::load(dir)?;
            let gaps = [&first.headings, &first.bodies].map(|s| s.outputs.iter().any(|o| o.name == "gap"));
            let tree_pairs = [&first.headings, &first.bodies].map(|s| s.inputs.iter().any(|i| i.name == "family"));
            let reader = Self {
                dir: dir.to_path_buf(),
                sessions: Mutex::new(vec![first]),
                gaps,
                tree_pairs,
                reorder,
                times,
                tokenizer,
                embeddings,
                meta,
            };
            let parity: Parity = serde_json::from_str(&read("parity.json")?).map_err(|e| format!("parity.json: {e}"))?;
            let texts: Vec<&str> = parity.texts.iter().map(String::as_str).collect();
            let ours = reader.embed(&texts)?;
            let worst = parity.vectors.iter().flatten().zip(&ours).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            if worst > 1e-4 {
                return Err(format!("the text model reads parity.json's texts {worst} off Python's"));
            }
            Ok(reader)
        }

        /// Each text's vector (`text_dim` a text).
        fn embed(&self, texts: &[&str]) -> std::result::Result<Vec<f32>, String> {
            let dim = self.meta.text_dim;
            let mut out = vec![0f32; texts.len() * dim];
            for (k, text) in texts.iter().enumerate() {
                let text = if text.trim().is_empty() { " " } else { text };
                let encoding = self.tokenizer.encode(text, false).map_err(|e| format!("tokenizer: {e}"))?;
                let ids: Vec<usize> = encoding.get_ids().iter().filter(|&&id| Some(id) != self.meta.unknown).map(|&id| id as usize).filter(|&id| id < self.meta.vocabulary).collect();
                if ids.is_empty() {
                    continue;
                }
                let row = &mut out[k * dim..(k + 1) * dim];
                for &id in &ids {
                    for (v, e) in row.iter_mut().zip(&self.embeddings[id * dim..(id + 1) * dim]) {
                        *v += e;
                    }
                }
                let count = ids.len() as f32;
                row.iter_mut().for_each(|v| *v /= count);
                if self.meta.normalize {
                    let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-32;
                    row.iter_mut().for_each(|v| *v /= norm);
                }
            }
            Ok(out)
        }

        /// The document's moves, a line each in reading order: `texts` and `pages` each line's; read with a set of
        /// sessions of the reader's own, given back after.
        pub(crate) fn read(&self, table: &LineTable, views: &[LineView], texts: &[&str], pages: &[usize], style: &Style) -> std::result::Result<Vec<Move>, String> {
            let held = self.sessions.lock().ok().and_then(|mut pool| pool.pop());
            let mut sessions = match held {
                Some(sessions) => sessions,
                None => Sessions::load(&self.dir)?,
            };
            let moves = self.read_with(&mut sessions, table, views, texts, pages, style);
            if let Ok(mut pool) = self.sessions.lock() {
                pool.push(sessions);
            }
            moves
        }

        fn read_with(&self, sessions: &mut Sessions, table: &LineTable, views: &[LineView], texts: &[&str], pages: &[usize], style: &Style) -> std::result::Result<Vec<Move>, String> {
            let n = table.order.len();
            let (d, m) = (self.meta.d, &self.meta);
            if n == 0 {
                return Ok(Vec::new());
            }
            let features: Vec<f32> = table.features.iter().enumerate().map(|(at, &v)| ((v - m.mean[at % ALL]) / m.std[at % ALL]).clamp(-8.0, 8.0)).collect();
            let at = |name: &str| LINE_FEATURES.iter().position(|f| *f == name).unwrap_or(0);
            let corners = [at("x0"), at("y0"), at("x1"), at("y1")];
            let boxes: Vec<f32> = (0..n).flat_map(|k| corners.map(|c| table.features[k * ALL + c])).collect();
            let clock = std::time::Instant::now();
            let mut stages = [0f64; 5];
            let mut lap = |stage: usize, since: &mut std::time::Instant| {
                stages[stage] += since.elapsed().as_secs_f64();
                *since = std::time::Instant::now();
            };
            let mut since = clock;
            let text = self.embed(texts)?;
            lap(0, &mut since);
            // Detect's encoder, a page at a time; the stylesheet over the document; Detect's heads, a page at a time.
            let anchors: Vec<i64> = style.anchors.iter().map(|&a| i64::from(a.min(9))).collect();
            let templates: Vec<i64> = style.templates.iter().map(|&t| i64::from(t)).collect();
            let t0: Vec<f32> = table.t0.iter().map(|&v| f32::from(v)).collect();
            let runs: Vec<(usize, usize)> = {
                let mut runs = Vec::new();
                let mut start = 0;
                while start < n {
                    let end = (start..n).find(|&k| pages[k] != pages[start]).unwrap_or(n);
                    runs.push((start, end));
                    start = end;
                }
                runs
            };
            let mut states = vec![0f32; n * d];
            for &(start, end) in &runs {
                let l = end - start;
                let f = TensorRef::from_array_view(([l, ALL], &features[start * ALL..end * ALL])).map_err(|e| e.to_string())?;
                let t = TensorRef::from_array_view(([l, m.text_dim], &text[start * m.text_dim..end * m.text_dim])).map_err(|e| e.to_string())?;
                let b = TensorRef::from_array_view(([l, 4], &boxes[start * 4..end * 4])).map_err(|e| e.to_string())?;
                let a = TensorRef::from_array_view(([l], &anchors[start..end])).map_err(|e| e.to_string())?;
                let z = TensorRef::from_array_view(([l, T0], &t0[start * T0..end * T0])).map_err(|e| e.to_string())?;
                let outputs = sessions.detect.run(ort::inputs!["features" => f, "text" => t, "boxes" => b, "anchors" => a, "t0" => z]).map_err(|e| format!("detect: {e}"))?;
                let (_, s) = outputs["states"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
                states[start * d..end * d].copy_from_slice(s);
            }
            if style.width != m.row_width {
                return Err(format!("the stylesheet's rows are {} wide; docparse.json says {}", style.width, m.row_width));
            }
            let count = style.rows.len() / style.width.max(1);
            let states = {
                let s = TensorRef::from_array_view(([n, d], &states[..])).map_err(|e| e.to_string())?;
                let t = TensorRef::from_array_view(([n], &templates[..])).map_err(|e| e.to_string())?;
                let r = TensorRef::from_array_view(([count, style.width], &style.rows[..])).map_err(|e| e.to_string())?;
                let outputs = sessions.stylesheet.run(ort::inputs!["states" => s, "templates" => t, "rows" => r]).map_err(|e| format!("stylesheet: {e}"))?;
                outputs["styled"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?.1.to_vec()
            };
            let mut lines = Lines { roles: vec![0; n], marked: vec![false; n], starts: vec![false; n], links: (0..n).collect() };
            let mut chosen = vec![(0usize, 0usize, 0f32); n];
            for &(start, end) in &runs {
                let l = end - start;
                let s = TensorRef::from_array_view(([l, d], &states[start * d..end * d])).map_err(|e| e.to_string())?;
                let b = TensorRef::from_array_view(([l, 4], &boxes[start * 4..end * 4])).map_err(|e| e.to_string())?;
                let outputs = sessions.lines.run(ort::inputs!["states" => s, "boxes" => b]).map_err(|e| format!("lines: {e}"))?;
                let (_, role) = outputs["role"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
                let (_, flag) = outputs["flags"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
                let (_, successor) = outputs["successor"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
                let (_, predecessor) = outputs["predecessor"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
                let successor = softmax_rows(successor, l);
                for i in 0..l {
                    lines.roles[start + i] = argmax(&role[i * CLASSES..(i + 1) * CLASSES]);
                    lines.marked[start + i] = flag[i * 2] > 0.0;
                    lines.starts[start + i] = flag[i * 2 + 1] > 0.0;
                    let next = argmax(&successor[i * l..(i + 1) * l]);
                    chosen[start + i] = (start + next, start + argmax(&predecessor[next * l..(next + 1) * l]), successor[i * l + next]);
                }
            }
            lap(1, &mut since);
            // A link both heads choose, or the successor's sure one; none into a line whose role opens a unit.
            for k in 0..n {
                let (next, back, sure) = chosen[k];
                if next != k && (back == k || sure > SURE) && lines.roles[next] % 2 == 0 {
                    lines.links[k] = next;
                }
            }
            margin_links(&mut lines.links, &table.margin_rows, views);
            // A line t0 reads opening a numbered paragraph on a number it prints opens on that marker.
            for (k, marked) in lines.marked.iter_mut().enumerate() {
                *marked |= views[k].count.is_some_and(|number| views[k].readings.iter().any(|r| r.value == number));
            }
            let (regions, roles) = regions_of(&lines);
            let read: Vec<usize> = (0..regions.len()).filter(|&r| roles[r] != 0).collect();
            let count = read.len();
            if count == 0 {
                return Ok(moves_of(&lines, &regions, &roles, &read, &Units { before: Vec::new(), level: Vec::new(), note: Vec::new() }, views, texts, None));
            }
            // Order over the regions read.
            let mut of_line = vec![usize::MAX; n];
            for (i, &r) in read.iter().enumerate() {
                for &k in &regions[r] {
                    of_line[k] = i;
                }
            }
            let mut line_states = Vec::new();
            let mut line_region = Vec::new();
            for k in 0..n {
                if of_line[k] != usize::MAX {
                    line_states.extend_from_slice(&states[k * d..(k + 1) * d]);
                    line_region.push(of_line[k] as i64);
                }
            }
            let kinds: Vec<i64> = read.iter().map(|&r| i64::from(roles[r])).collect();
            // The regions' places for the pair terms (`docparse.pairs_of`): each one's lines' union box and its first
            // line's page; the regions within a page of each other; whether a note's number is a mark the other
            // region sets.
            let mut region_boxes = vec![f32::MAX, f32::MAX, f32::MIN, f32::MIN].repeat(count);
            let mut marks: Vec<HashSet<u32>> = vec![HashSet::new(); count];
            for k in 0..n {
                let i = of_line[k];
                if i == usize::MAX {
                    continue;
                }
                for c in 0..2 {
                    region_boxes[i * 4 + c] = region_boxes[i * 4 + c].min(boxes[k * 4 + c]);
                    region_boxes[i * 4 + c + 2] = region_boxes[i * 4 + c + 2].max(boxes[k * 4 + c + 2]);
                }
                if REFERRING.contains(&roles[read[i]]) {
                    marks[i].extend(views[k].references.iter().map(|&(v, ..)| v));
                }
            }
            let region_pages: Vec<i64> = read.iter().map(|&r| pages[regions[r][0]] as i64).collect();
            let number: Vec<Option<u32>> = read.iter().map(|&r| {
                (roles[r] == NOTE || roles[r] == NOTE_QUOTE).then(|| note_number(&views[regions[r][0]], texts[regions[r][0]])).flatten()
            }).collect();
            let mut by_page: HashMap<i64, Vec<usize>> = HashMap::new();
            for (i, &page) in region_pages.iter().enumerate() {
                by_page.entry(page).or_default().push(i);
            }
            let mut setting: HashMap<u32, Vec<usize>> = HashMap::new();
            for (j, set) in marks.iter().enumerate() {
                for &value in set {
                    setting.entry(value).or_default().push(j);
                }
            }
            let (mut pair_i, mut pair_j, mut pair_match) = (Vec::new(), Vec::new(), Vec::new());
            for (i, &page) in region_pages.iter().enumerate() {
                for near in [page - 1, page, page + 1] {
                    for &j in by_page.get(&near).into_iter().flatten() {
                        pair_i.push(i as i64);
                        pair_j.push(j as i64);
                        pair_match.push(j != i && number[i].is_some_and(|v| marks[j].contains(&v)));
                    }
                }
                // A note's candidates are every region setting its number, however far (endnotes).
                for &j in number[i].and_then(|v| setting.get(&v)).into_iter().flatten() {
                    if j != i && (region_pages[j] - page).abs() > 1 {
                        pair_i.push(i as i64);
                        pair_j.push(j as i64);
                        pair_match.push(true);
                    }
                }
            }
            // The regions padded to whole blocks of the graph's local attention (`docparse.BLOCK`), the padding marked.
            let padded = count.div_ceil(BLOCK) * BLOCK;
            let mut kinds_padded = kinds.clone();
            kinds_padded.resize(padded, 0);
            let region_valid: Vec<bool> = (0..padded).map(|i| i < count).collect();
            let mut boxes_padded = region_boxes.clone();
            boxes_padded.resize(padded * 4, 0.0);
            let mut pages_padded = region_pages.clone();
            pages_padded.resize(padded, -1);
            let (region_states, goes_on, note_of, successor) = {
                let ls = TensorRef::from_array_view(([line_region.len(), d], &line_states[..])).map_err(|e| e.to_string())?;
                let lr = TensorRef::from_array_view(([line_region.len()], &line_region[..])).map_err(|e| e.to_string())?;
                let rr = TensorRef::from_array_view(([padded], &kinds_padded[..])).map_err(|e| e.to_string())?;
                let rv = TensorRef::from_array_view(([padded], &region_valid[..])).map_err(|e| e.to_string())?;
                let rb = TensorRef::from_array_view(([padded, 4], &boxes_padded[..])).map_err(|e| e.to_string())?;
                let rp = TensorRef::from_array_view(([padded], &pages_padded[..])).map_err(|e| e.to_string())?;
                let pi = TensorRef::from_array_view(([pair_i.len()], &pair_i[..])).map_err(|e| e.to_string())?;
                let pj = TensorRef::from_array_view(([pair_j.len()], &pair_j[..])).map_err(|e| e.to_string())?;
                let pm = TensorRef::from_array_view(([pair_match.len()], &pair_match[..])).map_err(|e| e.to_string())?;
                let outputs = sessions.order.run(ort::inputs!["line_states" => ls, "line_region" => lr, "region_roles" => rr, "region_valid" => rv, "region_boxes" => rb,
                    "region_pages" => rp, "pair_i" => pi, "pair_j" => pj, "pair_match" => pm]).map_err(|e| format!("order: {e}"))?;
                let take = |name: &str| outputs[name].try_extract_tensor::<f32>().map(|(_, v)| v.to_vec()).map_err(|e| e.to_string());
                // A head's scores a pair: its scaled query of the pair's region against its key of the other, and the
                // pair's term.
                let head = |name: &str| -> std::result::Result<Vec<f32>, String> {
                    let (q, k, r) = (take(&format!("{name}_q"))?, take(&format!("{name}_k"))?, take(&format!("{name}_r"))?);
                    Ok((0..pair_i.len()).map(|p| {
                        let (i, j) = (pair_i[p] as usize, pair_j[p] as usize);
                        q[i * d..(i + 1) * d].iter().zip(&k[j * d..(j + 1) * d]).map(|(a, b)| a * b).sum::<f32>() + r[p]
                    }).collect())
                };
                (take("regions")?, head("goes_on")?, head("note_of")?, head("successor")?)
            };
            // Each region's pairs (contiguous: built region by region), and each head's softmax over them.
            let mut rows: Vec<(usize, usize)> = vec![(0, 0); count];
            for (p, &i) in pair_i.iter().enumerate() {
                let i = i as usize;
                if rows[i].1 == 0 {
                    rows[i].0 = p;
                }
                rows[i].1 = p + 1;
            }
            let row_softmax = |logits: &[f32]| -> Vec<f32> {
                let mut probs = vec![0f32; logits.len()];
                for &(a, z) in &rows {
                    if a < z {
                        probs[a..z].copy_from_slice(&softmax_rows(&logits[a..z], z - a));
                    }
                }
                probs
            };
            let (goes_on, note_of, successor) = (row_softmax(&goes_on), row_softmax(&note_of), row_softmax(&successor));
            // A region's likeliest other region by a head, with its probability (itself, where it has no pairs).
            let best = |probs: &[f32], i: usize, open: &dyn Fn(usize) -> bool| -> Option<(usize, f32)> {
                let (a, z) = rows[i];
                (a..z).filter(|&p| open(pair_j[p] as usize)).max_by(|&x, &y| probs[x].total_cmp(&probs[y])).map(|p| (pair_j[p] as usize, probs[p]))
            };
            // The reading: from the parser's first region, each region's likeliest unread successor where the head
            // holds it likelier than none (the region itself), else the next unread region in the parser's order.
            let mut place = vec![usize::MAX; count];
            let (mut at, mut next_unread) = (0usize, 0usize);
            for position in 0..count {
                place[at] = position;
                while next_unread < count && place[next_unread] != usize::MAX {
                    next_unread += 1;
                }
                if position + 1 == count {
                    break;
                }
                // The parser's next unread region, unless the head is sure of another (and holds the parser's next
                // all but impossible); a line's number or a page's label is never read out of the parser's order.
                let labels = |i: usize| matches!(roles[read[i]], LINE_LABEL | PAGE_LABEL);
                let unread = |j: usize| place[j] == usize::MAX && !labels(j);
                let parsers = {
                    let (a, z) = rows[at];
                    (a..z).find(|&p| pair_j[p] as usize == next_unread).map_or(0.0, |p| successor[p])
                };
                at = match best(&successor, at, &unread) {
                    Some((j, p)) if !labels(at) && j != next_unread && p >= SURE_ORDER && parsers <= 1.0 - SURE_ORDER => j,
                    _ => next_unread,
                };
            }
            // Units: a region goes on in the later region its link holds likeliest, each region taken once, the
            // surest links first.
            let mut links: Vec<(usize, usize, f32)> = (0..count).filter_map(|i| {
                let (j, p) = best(&goes_on, i, &|_| true)?;
                (j > i).then_some((i, j, p))
            }).collect();
            links.sort_by(|a, b| b.2.total_cmp(&a.2));
            let mut before = vec![None; count];
            let mut after = vec![false; count];
            for (i, j, _) in links {
                // A paragraph the margin numbers, or t0 reads opening, or in a different stream, opens: it goes on in no earlier one.
                let opens = { let first = &views[regions[read[j]][0]]; first.margin.is_some() || first.count.is_some() };
                let same_stream = stream_of(roles[read[i]]) == stream_of(roles[read[j]])
                    && (roles[read[i]] == HEADING) == (roles[read[j]] == HEADING)
                    && !(regions[read[i]][0] + 1..=regions[read[j]][0]).any(|k| lines.starts[k]);
                if before[j].is_none() && !after[i] && !opens && same_stream {
                    before[j] = Some(i);
                    after[i] = true;
                }
            }
            goes_on_in_column(&mut before, &regions, &roles, &read, &lines.roles, views);
            lap(2, &mut since);
            // The trees over the regions that open a unit, headings and the body's apart.
            let mut sections = vec![0i64; count];
            let mut enclosing_heading = vec![None; count];
            let mut section = 0i64;
            let mut last_heading = None;
            for i in 0..count {
                let from = if i == 0 { 0 } else { regions[read[i - 1]][0] + 1 };
                if (from..=regions[read[i]][0]).any(|k| lines.starts[k]) {
                    section += 1;
                    last_heading = None;
                }
                enclosing_heading[i] = last_heading;
                if before[i].is_none() && roles[read[i]] == HEADING {
                    section += 1;
                    last_heading = Some(i);
                }
                sections[i] = section;
            }
            let feat_at = |name: &str| LINE_FEATURES.iter().position(|f| *f == name).unwrap_or(0);
            let (idx_marker, idx_val, idx_pref, idx_c0, idx_w0, idx_x0, idx_size) = (
                feat_at("marker"), feat_at("marker_value"), feat_at("marker_prefix"), feat_at("count_1"), feat_at("wrap_()"), feat_at("x0"), feat_at("size")
            );
            let line_family = |k: usize| -> (i64, i64, i64) {
                let row = &table.features[k * ALL..(k + 1) * ALL];
                if row[idx_marker] <= 0.5 {
                    return (0, -1, 0);
                }
                let val = (row[idx_val].max(0.0).exp_m1()).round() as i64;
                let pref = row[idx_pref].max(0.0).round() as i64;
                let count_idx = argmax(&row[idx_c0..idx_c0 + 9]) as i64;
                let wrap_idx = argmax(&row[idx_w0..idx_w0 + 5]) as i64;
                (count_idx * 5 + wrap_idx + 1, val, pref)
            };
            let tree = |session: &mut Session, has_gap: bool, has_pairs: bool, nodes: &[usize]| -> std::result::Result<Vec<usize>, String> {
                let h = nodes.len();
                if h == 0 {
                    return Ok(Vec::new());
                }
                let x: Vec<f32> = nodes.iter().flat_map(|&i| region_states[i * d..(i + 1) * d].iter().copied()).collect();
                let s = TensorRef::from_array_view(([h, d], &x[..])).map_err(|e| e.to_string())?;
                let mut h_states = vec![0f32; h * d];
                for (at, &i) in nodes.iter().enumerate() {
                    if let Some(hd) = enclosing_heading[i] {
                        h_states[at * d..(at + 1) * d].copy_from_slice(&region_states[hd * d..(hd + 1) * d]);
                    }
                }
                let node_templates: Vec<i64> = nodes.iter().map(|&i| templates[regions[read[i]][0]]).collect();
                let node_anchors: Vec<i64> = nodes.iter().map(|&i| anchors[regions[read[i]][0]]).collect();
                let t = TensorRef::from_array_view(([h], &node_templates[..])).map_err(|e| e.to_string())?;
                let a = TensorRef::from_array_view(([h], &node_anchors[..])).map_err(|e| e.to_string())?;
                let (mut node_fam, mut node_val, mut node_pref) = (Vec::with_capacity(h), Vec::with_capacity(h), Vec::with_capacity(h));
                let (mut node_sec, mut node_x0, mut node_size) = (Vec::with_capacity(h), Vec::with_capacity(h), Vec::with_capacity(h));
                let (mut node_looks, mut node_pages, mut node_roles) = (Vec::with_capacity(h), Vec::with_capacity(h), Vec::with_capacity(h));
                for &i in nodes {
                    let k = regions[read[i]][0];
                    let (fam, val, pref) = line_family(k);
                    node_fam.push(fam);
                    node_val.push(val);
                    node_pref.push(pref);
                    node_sec.push(sections[i]);
                    node_x0.push(table.features[k * ALL + idx_x0]);
                    node_size.push(table.features[k * ALL + idx_size]);
                    node_looks.push(i64::from(table.looks[k]));
                    node_pages.push(pages[k] as i64);
                    node_roles.push(i64::from(roles[read[i]]));
                }
                let outputs = if has_pairs {
                    let ths = TensorRef::from_array_view(([h, d], &h_states[..])).map_err(|e| e.to_string())?;
                    let tf = TensorRef::from_array_view(([h], &node_fam[..])).map_err(|e| e.to_string())?;
                    let tv = TensorRef::from_array_view(([h], &node_val[..])).map_err(|e| e.to_string())?;
                    let tp = TensorRef::from_array_view(([h], &node_pref[..])).map_err(|e| e.to_string())?;
                    let ts = TensorRef::from_array_view(([h], &node_sec[..])).map_err(|e| e.to_string())?;
                    let tx = TensorRef::from_array_view(([h], &node_x0[..])).map_err(|e| e.to_string())?;
                    let tsz = TensorRef::from_array_view(([h], &node_size[..])).map_err(|e| e.to_string())?;
                    let tl = TensorRef::from_array_view(([h], &node_looks[..])).map_err(|e| e.to_string())?;
                    let tpg = TensorRef::from_array_view(([h], &node_pages[..])).map_err(|e| e.to_string())?;
                    let tr = TensorRef::from_array_view(([h], &node_roles[..])).map_err(|e| e.to_string())?;
                    session.run(ort::inputs!["states" => s, "heading_states" => ths, "templates" => t, "anchors" => a, "family" => tf, "value" => tv, "prefix" => tp,
                        "section" => ts, "x0" => tx, "size" => tsz, "looks" => tl, "pages" => tpg, "roles" => tr]).map_err(|e| format!("construct: {e}"))?
                } else {
                    session.run(ort::inputs!["states" => s, "templates" => t, "anchors" => a]).map_err(|e| format!("construct: {e}"))?
                };
                let (_, parent) = outputs["parent"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
                let (_, sibling) = outputs["sibling"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
                // Each node's gap where the graph gives one (`gap`, GAPS a node), else one level.
                let gaps: Vec<usize> = if has_gap {
                    outputs["gap"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?.1.chunks_exact(GAPS).map(|row| argmax(row) + 1).collect()
                } else {
                    vec![1; h]
                };
                let firsts: Vec<usize> = nodes.iter().map(|&i| regions[read[i]][0]).collect();
                Ok(insert(&earlier(parent, h), &earlier(sibling, h), &gaps, h, &resets(&firsts, &lines.starts)))
            };
            let opening = |i: usize| before[i].is_none();
            let headings: Vec<usize> = (0..count).filter(|&i| opening(i) && roles[read[i]] == HEADING).collect();
            let bodies: Vec<usize> = (0..count).filter(|&i| opening(i) && BODY.contains(&roles[read[i]])).collect();
            let mut level = vec![0usize; count];
            for (&i, l) in headings.iter().zip(tree(&mut sessions.headings, self.gaps[0], self.tree_pairs[0], &headings)?) {
                level[i] = l;
            }
            for (&i, l) in bodies.iter().zip(tree(&mut sessions.bodies, self.gaps[1], self.tree_pairs[1], &bodies)?) {
                level[i] = l;
            }
            // A note's region: the one holding a mark of its number its link names, where not itself, surest notes first
            // so two notes of a number in parallel or multi-part reports do not take the same region's mark.
            let mut note = vec![None; count];
            let mut note_candidates: Vec<(usize, f32)> = (0..count).filter(|&i| before[i].is_none() && (roles[read[i]] == NOTE || roles[read[i]] == NOTE_QUOTE) && number[i].is_some()).map(|i| {
                let valid = |j: usize| j == i || number[i].is_some_and(|v| marks[j].contains(&v));
                let p = best(&note_of, i, &valid).map_or(0.0, |(j, p)| if j != i { p } else { 0.0 });
                (i, p)
            }).collect();
            note_candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let mut claimed: HashSet<(usize, u32)> = HashSet::new();
            for (i, _) in note_candidates {
                let Some(val) = number[i] else { continue };
                let valid = |j: usize| j == i || (marks[j].contains(&val) && !claimed.contains(&(j, val)));
                let target = best(&note_of, i, &valid)
                    .or_else(|| best(&note_of, i, &|j| j == i || marks[j].contains(&val)))
                    .map_or(i, |(j, _)| j);
                if target != i {
                    claimed.insert((target, val));
                    note[i] = Some(target);
                }
            }
            lap(3, &mut since);
            let mut moves = moves_of(&lines, &regions, &roles, &read, &Units { before, level, note }, views, texts, None);
            // Each line's rank: its region's place in the reading, then its place in the region.
            let mut ranks = vec![-1i64; n];
            for (i, &r) in read.iter().enumerate() {
                for (at, &k) in regions[r].iter().enumerate() {
                    ranks[k] = place[i] as i64 * 100_000 + at as i64;
                }
            }
            if self.reorder {
                read_in(&mut moves, &ranks, pages);
            }
            lap(4, &mut since);
            if self.times {
                eprintln!("doc times: {n} lines {count} regions {} pairs; embed {:.3}s detect {:.3}s order {:.3}s trees {:.3}s decode {:.3}s",
                    pair_i.len(), stages[0], stages[1], stages[2], stages[3], stages[4]);
            }
            Ok(moves)
        }
    }
}
