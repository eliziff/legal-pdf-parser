//! Detect-Order-Construct (Wang et al., Pattern Recognition 2024) in the engine: the stages `docparse.py`
//! fits on the gold and exports (LEGALPDF_DOC, a folder of `detect.onnx`, `order.onnx`,
//! `headings.onnx`, `bodies.onnx` and `docparse.json`, which names the text model's folder: its `tokenizer.json`
//! and `embeddings.f32`), and the paper's decoding of what they score into the parser's moves:
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
//!   quotation), and a note's region the region holding its reference mark; the lines are read in the parser's
//!   order (the successor head's reading of another was never written: it is not read);
//! - Construct, over the headings that open a unit and over the body's: each node's parent and left sibling
//!   among the nodes before it, the tree built by the paper's Algorithm 1 (each node the right-most child of
//!   the node on the tree's right-most path whose parent score times its child's sibling score is greatest),
//!   a constituent document's start emptying it; a heading's level, or a body unit's depth, its parent's and
//!   its gap (how many levels under its parent it stands: the print may skip one).
//!
//! The decoding after the heads is the oracle's too (LEGALPDF_DOC_ORACLE, the gold given): the
//! heads' answers read off the gold, so that what it grades below the gold's own is what the decoding and
//! the moves cannot hold, not what the heads miss.
//!
//! A line's text is the text model's reading of it as Model2Vec reads one: its tokens (no special ones, the
//! unknown token left out) averaged and scaled to unit length.

use crate::model::machine::lines::SHAPE;
use crate::model::machine::lines::T0;
use crate::model::machine::lines::marker_family;
use crate::model::machine::lines::WIDTH;
use crate::model::machine::lines::Measures;
use crate::model::machine::lines::LineTable;
use super::*;

/// The regions whose units take a depth in the body's tree (`docparse.BODY`): all that the gold's depth steps chain
/// over (`RANKED_BODY`).
const BODY: [u8; 7] = RANKED_BODY;
/// A heading's or body unit's gap under its parent is one of 1..=GAPS levels.
#[cfg(feature = "encoder")]
pub const GAPS: usize = 4;

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

/// The regions: the lines' groups under their links, each in reading order and the groups by their first
/// lines; each one's role its lines' plurality (the first line's on a tie); the regions read (those of a role),
/// and each read region's first line.
struct Regions {
    lines: Vec<Vec<usize>>,
    roles: Vec<u8>,
    read: Vec<usize>,
    firsts: Vec<usize>,
}

impl Regions {
    fn of(lines: &Lines) -> Self {
        let n = lines.roles.len();
        let strict = |r: u8| !(BODY.contains(&r) && r != TABLE);
        let mut group: Vec<usize> = (0..n).collect();
        for (k, &next) in lines
            .links
            .iter()
            .enumerate()
            .filter(|&(k, &next)| next != k)
        {
            let (rk, rn) = ((lines.roles[k] / 2) as u8, (lines.roles[next] / 2) as u8);
            if rk == 0 || rn == 0 || ((strict(rk) || strict(rn)) && rk != rn) {
                continue;
            }
            let (a, b) = (find(&mut group, k), find(&mut group, next));
            group[a.max(b)] = a.min(b);
        }
        let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
        (0..n).for_each(|k| members.entry(find(&mut group, k)).or_default().push(k));
        let mut regions: Vec<Vec<usize>> = members.into_values().collect();
        regions.sort_by_key(|lines| lines[0]);
        let roles: Vec<u8> = regions
            .iter()
            .map(|members| {
                let counts = members.iter().fold([0usize; 17], |mut counts, &k| {
                    counts[lines.roles[k] / 2] += 1;
                    counts
                });
                (0..17).fold(lines.roles[members[0]] / 2, |best, r| {
                    if counts[r] > counts[best] {
                        r
                    } else {
                        best
                    }
                }) as u8
            })
            .collect();
        let read: Vec<usize> = (0..regions.len()).filter(|&r| roles[r] != 0).collect();
        let firsts = read.iter().map(|&r| regions[r][0]).collect();
        Self {
            lines: regions,
            roles,
            read,
            firsts,
        }
    }

    /// Each region's role its lines' summed type probabilities' likeliest (`type_probs`, 17 a line), not their argmaxes'
    /// plurality (LEGALPDF_DOC_DECODE `vote`); the regions read again (those of a role).
    fn vote(&mut self, type_probs: &[f32]) {
        for (members, role) in self.lines.iter().zip(self.roles.iter_mut()) {
            let mut sums = [0f32; 17];
            members
                .iter()
                .for_each(|&k| (0..17).for_each(|r| sums[r] += type_probs[k * 17 + r]));
            let start = usize::from(*role);
            *role =
                (0..17).fold(start, |best, r| if sums[r] > sums[best] { r } else { best }) as u8;
        }
        self.read = (0..self.lines.len())
            .filter(|&r| self.roles[r] != 0)
            .collect();
        self.firsts = self.read.iter().map(|&r| self.lines[r][0]).collect();
    }

    /// A read region's role.
    fn role(&self, i: usize) -> u8 {
        self.roles[self.read[i]]
    }

    /// Each of `n` lines' place among the regions read (`usize::MAX` for a line in none).
    fn of_lines(&self, n: usize) -> Vec<usize> {
        let mut of_line = vec![usize::MAX; n];
        self.read
            .iter()
            .enumerate()
            .for_each(|(i, &r)| self.lines[r].iter().for_each(|&k| of_line[k] = i));
        of_line
    }
}

fn find(group: &mut [usize], k: usize) -> usize {
    let mut root = k;
    while group[root] != root {
        root = group[root];
    }
    let mut at = k;
    while group[at] != root {
        at = std::mem::replace(&mut group[at], root);
    }
    root
}

/// Each line's marker from the line table: its numbering's family (its count and wrap, 0 for none), value (-1 for none)
/// and prefix length. Where a line prints an ambiguous marker (Roman vs. alphabetic, such as `(i)`, `(v)` or `(x)`), the
/// reading that continues a family seen earlier in the document is chosen over the first reading, so Construct's tree
/// codes and the dump's targets see the active sequence.
pub fn line_markers(table: &LineTable) -> Vec<(i64, i64, i64)> {
    let encode = |reading: &crate::model::MarkerReading| -> (i64, i64, i64) {
        let (_, count, wrap) = marker_family(reading);
        (
            count.unwrap_or(0) as i64 * 5 + wrap.unwrap_or(0) as i64 + 1,
            i64::from(reading.value),
            reading.prefix.len() as i64,
        )
    };
    let mut last_value: HashMap<&str, u32> = HashMap::new();
    table
        .readings
        .iter()
        .map(|readings| {
            let Some(first) = readings.first() else {
                return (0, -1, 0);
            };
            let chosen = if readings.len() > 1 {
                readings
                    .iter()
                    .find(|r| {
                        last_value
                            .get(r.family.as_str())
                            .is_some_and(|&v| r.value == v + 1)
                    })
                    .or_else(|| {
                        readings
                            .iter()
                            .find(|r| r.value == 1 && !last_value.contains_key(r.family.as_str()))
                    })
                    .unwrap_or(first)
            } else {
                first
            };
            last_value.insert(chosen.family.as_str(), chosen.value);
            encode(chosen)
        })
        .collect()
}

/// Whether a region is a note's.
fn is_note(region: u8) -> bool {
    region == NOTE || region == NOTE_QUOTE
}

/// The paragraphs t0 decided in the regions' links: a line the judgment reader reads opening a numbered paragraph
/// (`LineView::count`) takes no link from any other line. The margin's paragraph numbers (`LineTable::margin_rows`):
/// a number and the row's line it numbers are one region, the number its first line (it is read just before it);
/// the row's line and its twin's in the other column take no link from any other line, each opening its column's
/// paragraph, nor does the number.
#[cfg(feature = "encoder")]
fn margin_links(
    links: &mut [usize],
    margin_rows: &HashMap<usize, (usize, Option<usize>)>,
    views: &[LineView],
) {
    let mut opened: HashSet<usize> = margin_rows
        .iter()
        .flat_map(|(&margin, &(row, twin))| [margin, row].into_iter().chain(twin))
        .collect();
    opened.extend((0..views.len()).filter(|&k| views[k].count().is_some()));
    links
        .iter_mut()
        .enumerate()
        .filter(|(k, next)| opened.contains(*next) && !margin_rows.contains_key(k))
        .for_each(|(k, next)| *next = k);
    margin_rows
        .iter()
        .for_each(|(&margin, &(row, _))| links[margin] = row);
}

/// A body region whose first line Detect reads going on (its role's opening bit unset, or t0 proved a cross-page/column
/// continuation onto it with no marker reading of its own) and that Order puts in no earlier unit goes on in the last
/// region before it of its role in its column's language (a paragraph past the page's foot, or read past the other
/// column's lines): the region read last of that role and language, on its page or the page before (as Order pairs
/// regions), its unit the one open in that stream. A region the margin numbers opens.
#[cfg(feature = "encoder")]
fn goes_on_in_column(
    before: &mut [Option<usize>],
    regions: &Regions,
    line_roles: &[usize],
    marked: &[bool],
    t0: &[u8],
    views: &[LineView],
    pages: &[usize],
) {
    let mut last: HashMap<(u8, bool), usize> = HashMap::new();
    for (i, &first) in regions.firsts.iter().enumerate() {
        let key = (regions.role(i), views[first].translation());
        let t0_continues = t0.get(first * T0 + 20).copied().unwrap_or(0) != 0
            && !marked[first]
            && views[first].readings().is_empty()
            && first > 0
            && pages[first] > pages[first - 1];
        if before[i].is_none()
            && BODY.contains(&key.0)
            && (line_roles[first] % 2 == 0 || t0_continues)
            && views[first].margin().is_none()
            && views[first].count().is_none()
        {
            before[i] = last
                .get(&key)
                .copied()
                .filter(|&earlier| pages[regions.firsts[earlier]] + 1 >= pages[first]);
        }
        last.insert(key, i);
    }
}

/// The paper's Algorithm 1 over `n` nodes: each node in turn the right-most child of the node on the tree's
/// right-most path (the root first) whose parent probability times the node's sibling probability for that
/// node's last child (itself where it has none) is greatest; `resets[i]` empties the tree before node `i`.
/// Each node's level: its parent's (0 for the root) and its gap.
#[cfg(feature = "encoder")]
fn insert_scored(
    scores: &super::trees::Scores,
    gaps: &[usize],
    n: usize,
    resets: &[bool],
) -> Vec<usize> {
    let mut path: Vec<usize> = Vec::new();
    let mut levels = vec![0usize; n];
    for i in 0..n {
        if resets[i] {
            path.clear();
        }
        // A parent's and a sibling's probabilities are each a softmax over the node's row, the same for every candidate:
        // the candidate with the greatest product of the two has the greatest sum of their scores.
        let score =
            |parent: usize, sibling: usize| scores.score(i, parent).0 + scores.score(i, sibling).1;
        let mut best = (0usize, score(i, path.first().copied().unwrap_or(i)));
        for (at, &node) in path.iter().enumerate() {
            let candidate = score(node, path.get(at + 1).copied().unwrap_or(i));
            if candidate > best.1 {
                best = (at + 1, candidate);
            }
        }
        path.truncate(best.0);
        levels[i] = path.last().map_or(0, |&node| levels[node]) + gaps[i].max(1);
        path.push(i);
    }
    levels
}

/// The same decoding over every pair's probabilities (`parent`, `sibling`, n a row), as the oracle reads them.
fn insert(
    parent: &[f32],
    sibling: &[f32],
    gaps: &[usize],
    n: usize,
    resets: &[bool],
) -> Vec<usize> {
    let mut path: Vec<usize> = Vec::new();
    let mut levels = vec![0usize; n];
    for i in 0..n {
        if resets[i] {
            path.clear();
        }
        let (p, s) = (|j: usize| parent[i * n + j], |j: usize| sibling[i * n + j]);
        // Under the root: no parent (itself), its sibling the root's last child, the path's first node.
        let mut best = (0usize, p(i) * s(path.first().copied().unwrap_or(i)));
        for (at, &node) in path.iter().enumerate() {
            let score = p(node) * s(path.get(at + 1).copied().unwrap_or(i));
            if score > best.1 {
                best = (at + 1, score);
            }
        }
        path.truncate(best.0);
        levels[i] = path.last().map_or(0, |&node| levels[node]) + gaps[i].max(1);
        path.push(i);
    }
    levels
}

/// A note's number: its label's where the label reader found one, else an arabic reading's, else the figures
/// its text opens on ("27" of "27 Ibid", as the gold's notes are numbered), else 0 for a note symbol, as a reference
/// mark's value is (`evidence::references`).
pub fn note_number(line: &LineView, text: &str) -> Option<u32> {
    line.label()
        .or_else(|| {
            line.readings()
                .iter()
                .find(|r| r.family.chars().all(|c| !c.is_alphabetic()))
                .map(|r| r.value)
        })
        .or_else(|| crate::rules::text::leading_figures(text).map(|(figures, _)| figures))
        .or_else(|| {
            text.trim_start()
                .starts_with(['*', '\u{2020}', '\u{2021}', '\u{a7}'])
                .then_some(0)
        })
}

/// The document's moves from the heads' answers: each region's lines its role's; a unit's first line opens it
/// (at its level or depth, counting on its marker where the line is marked, a note on its number); a line of a
/// unit another unit of its stream came before goes on with it by name (its first line); a note answers the
/// marks of its number in the region its link names.
fn moves_of(
    lines: &Lines,
    regions: &Regions,
    units: &Units,
    views: &[LineView],
    texts: &[&str],
    only: Option<&[bool]>,
) -> Vec<Move> {
    let mut moves: Vec<Move> = lines
        .starts
        .iter()
        .map(|&starts| Move {
            starts,
            ..Move::plain(0, false)
        })
        .collect();
    // Each region's unit, its first line: a region going on in an earlier one's unit is in that unit.
    let mut unit: Vec<usize> = Vec::with_capacity(regions.read.len());
    regions.firsts.iter().enumerate().for_each(|(i, &first)| {
        unit.push(
            units.before[i]
                .filter(|&earlier| earlier < i)
                .map_or(first, |earlier| unit[earlier]),
        )
    });
    // The open marker frames per context for disambiguating Roman vs. alphabetic readings (e.g. `(i)`, `(v)`, `(x)`).
    let mut doc_id = 0u16;
    let mut stacks: HashMap<u16, Open> = HashMap::new();
    for (k, &i) in regions.of_lines(lines.roles.len()).iter().enumerate() {
        doc_id = doc_id.saturating_add(u16::from(lines.starts[k]));
        if i == usize::MAX {
            continue;
        }
        let region = regions.role(i);
        let mv = &mut moves[k];
        mv.region = region;
        // A line of a unit opened before it is in that unit, by its first line.
        if unit[i] != k {
            mv.resumes = Some(unit[i] as u32);
            continue;
        }
        let open = stacks
            .entry(doc_id.saturating_mul(2) + u16::from(views[k].translation()))
            .or_default();
        open.opens(region);
        // DOC's own pick among several readings: the one that goes on from an open frame (the next number of its family).
        let chosen_reading = (!is_note(region) && lines.marked[k]).then(|| {
            views[k].opening(|| {
                (views[k].readings().len() > 1)
                    .then(|| {
                        views[k].readings().iter().position(|r| {
                            open.frames()
                                .iter()
                                .rev()
                                .any(|f| f.family == r.family && r.value == f.value + 1)
                        })
                    })
                    .flatten()
            })
        });
        if let Some(reading) = views[k].reading(chosen_reading) {
            open.count(reading, region);
        }
        mv.opens = true;
        if region == HEADING {
            mv.level = units.level[i].clamp(1, 9) as u8;
        } else if BODY.contains(&region) {
            mv.depth = units.level[i].clamp(1, 9) as u8;
        }
        if is_note(region) {
            mv.note = note_number(&views[k], texts[k]);
        } else {
            mv.reading = chosen_reading;
        }
    }
    for (i, target) in units.note.iter().enumerate() {
        let first = regions.firsts[i];
        if !(moves[first].opens && is_note(regions.role(i))) {
            continue;
        }
        // Of the region's marks of its number, those on the lines `only` names, where it names any (the gold's). A note
        // the heads pair with no mark, or one of no number, answers none: never the last of its number the text set.
        let marks: Vec<(u32, u32, u32)> = moves[first]
            .note
            .into_iter()
            .flat_map(|value| {
                target
                    .iter()
                    .flat_map(|&target| &regions.lines[regions.read[target]])
                    .filter(|&&k| only.is_none_or(|lines| lines[k]))
                    .flat_map(move |&k| {
                        views[k]
                            .references()
                            .iter()
                            .filter(move |&&(v, ..)| v == value)
                            .map(move |&(_, s, e)| (views[k].place, s, e))
                    })
            })
            .collect();
        moves[first].answers = Some(marks);
    }
    moves
}

/// A constituent document's start between each node and the one before (the first: from the document's top)
/// empties the tree before it.
pub fn resets(firsts: &[usize], starts: &[bool]) -> Vec<bool> {
    (0..firsts.len())
        .map(|i| (if i == 0 { 0 } else { firsts[i - 1] + 1 }..=firsts[i]).any(|k| starts[k]))
        .collect()
}

/// Each line's box (its x0, y0, x1, y1), four a line: what the reader and the fit read of where a line stands.
pub fn boxes(measures: &Measures) -> Vec<f32> {
    let corners = [feature("x0"), feature("y0"), feature("x1"), feature("y1")];
    measures
        .own
        .chunks_exact(WIDTH)
        .flat_map(|row| corners.map(|c| row[c]))
        .collect()
}

/// Each region's box, the union of its lines' (`boxes`, four a line; `lines` each line with its region), four a region.
pub fn region_boxes(
    boxes: &[f32],
    lines: impl Iterator<Item = (usize, usize)>,
    count: usize,
) -> Vec<f32> {
    let mut out = [f32::MAX, f32::MAX, f32::MIN, f32::MIN].repeat(count);
    for (k, i) in lines {
        for c in 0..2 {
            out[i * 4 + c] = out[i * 4 + c].min(boxes[k * 4 + c]);
            out[i * 4 + c + 2] = out[i * 4 + c + 2].max(boxes[k * 4 + c + 2]);
        }
    }
    out
}

/// What the trees read of where each region stands: its section (the constituent documents begun, `reset`, and the
/// opening headings, `opening_heading`, up to it) and the opening heading it stands under (none past a document's start).
pub fn sections(
    reset: &[bool],
    opening_heading: impl Fn(usize) -> bool,
) -> (Vec<i64>, Vec<Option<usize>>) {
    let (mut sections, mut enclosing) = (vec![0i64; reset.len()], vec![None; reset.len()]);
    let (mut section, mut last) = (0i64, None);
    for i in 0..reset.len() {
        if reset[i] {
            (section, last) = (section + 1, None);
        }
        enclosing[i] = last;
        if opening_heading(i) {
            (section, last) = (section + 1, Some(i));
        }
        sections[i] = section;
    }
    (sections, enclosing)
}

// ------------------------------------------------------------------------------------------------- the oracle
/// The gold's region of a row (an unmarked body line: text).
fn gold_region(g: &[i64; 8]) -> i64 {
    if g[0] == UNMARKED_BODY {
        1
    } else {
        g[0]
    }
}

/// The document's moves as the decoding makes them from the heads' answers read off the gold (`gold`, each
/// line's row): each line's role, its flags and its next line in its unit on its page; each region's earlier
/// part, its tree (each node under the nearest open node of a smaller level, the left sibling that parent's
/// last child), its gap, and a note's region the region before it (two pages back at most) holding a mark of
/// its number the gold pairs; read through the same `insert` and `moves_of` as the model's.
pub fn oracle(
    gold: &[[i64; 8]],
    views: &[LineView],
    texts: &[&str],
    pages: &[usize],
    ranks: &[i64],
    placed: &crate::Placed,
) -> Vec<Move> {
    let n = gold.len();
    let (lines, unit) = gold_lines(gold, pages);
    let regions = Regions::of(&lines);
    let count = regions.read.len();
    let mut moves = oracle_units(gold, views, texts, n, lines, unit, regions, count);
    // What the gold prints is given as it prints it, wherever tier 0 read it or did not: each unit's markers (a note's
    // its label), and each note's reference marks.
    for opening in moves.iter_mut().filter(|chosen| chosen.opens) {
        opening.markers = Some(Vec::new());
        if is_note(opening.region) {
            opening.answers = Some(Vec::new());
        }
    }
    let mark = |quad: &[i64; 4]| (quad[1] as u32, quad[2] as u32, quad[3] as u32);
    for quad in &placed.markers {
        if let Some(markers) = moves
            .get_mut(quad[0] as usize)
            .and_then(|opening| opening.markers.as_mut())
        {
            markers.push(mark(quad));
        }
    }
    for quad in &placed.references {
        if let Some(answers) = moves
            .get_mut(quad[0] as usize)
            .and_then(|opening| opening.answers.as_mut())
        {
            answers.push(mark(quad));
        }
    }
    // A table row's cells, as the gold prints them on each of its lines.
    for chosen in moves.iter_mut().filter(|chosen| chosen.region == TABLE) {
        chosen.cells = Some(Vec::new());
    }
    for quad in &placed.cells {
        if let Some(cells) = moves
            .get_mut(quad[0] as usize)
            .and_then(|chosen| chosen.cells.as_mut())
        {
            cells.push(mark(quad));
        }
    }
    // The labels that open speakers' turns (every line names its own, most none), and the documents' types: the
    // source's own on its first line, each document's on the line that starts it.
    for chosen in &mut moves {
        chosen.speakers = Some(Vec::new());
    }
    for quad in &placed.speakers {
        if let Some(speakers) = moves
            .get_mut(quad[0] as usize)
            .and_then(|chosen| chosen.speakers.as_mut())
        {
            speakers.push((quad[1] as u32, quad[2] as u32));
        }
    }
    for (place, kind) in &placed.documents {
        let own = *place < 0;
        if let Some(chosen) = moves
            .get_mut(if own { 0 } else { *place as usize })
            .filter(|chosen| chosen.starts != own)
        {
            chosen.document = Some(kind.clone());
        }
    }
    // The lines in the gold's order: a line the gold ranks by its rank, one it does not after the ranked line before
    // it in the parser's order.
    let mut before = -1;
    let follows: Vec<i64> = ranks
        .iter()
        .map(|&rank| {
            before = if rank >= 0 { rank } else { before };
            before
        })
        .collect();
    let mut order: Vec<usize> = (0..n.min(ranks.len())).collect();
    order.sort_by_key(|&k| (follows[k], ranks[k] < 0, k));
    for (position, k) in order.into_iter().enumerate() {
        moves[k].rank = Some(position as u32);
    }
    moves
}

/// The line heads' answers read off the gold: each line's role, its flags and its next line in its unit on its page; and
/// each line's unit (its first line), as the gold's decoder reads the rows (`docparse.gold_units`).
fn gold_lines(gold: &[[i64; 8]], pages: &[usize]) -> (Lines, Vec<usize>) {
    let n = gold.len();
    // Each line's unit, as the gold's decoder reads its rows (`docparse.gold_units`).
    let mut unit = vec![usize::MAX; n];
    let mut open: HashMap<usize, usize> = HashMap::new();
    // A line the gold reads first in a unit the parser reads later (the two orders differ) names a line after it: that
    // line goes on in this one's unit.
    let named_later: HashMap<usize, usize> = gold
        .iter()
        .enumerate()
        .filter(|(k, g)| g[1] == 2 && usize::try_from(g[6]).is_ok_and(|p| p > *k))
        .map(|(k, g)| (g[6] as usize, k))
        .collect();
    for (k, g) in gold.iter().enumerate().filter(|(_, g)| gold_region(g) > 0) {
        let region = gold_region(g);
        let stream = stream_of(region as u8);
        let pointer = usize::try_from(g[6])
            .ok()
            .filter(|&p| p < k && unit[p] != usize::MAX);
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
    (0..n).filter(|&k| unit[k] != usize::MAX).for_each(|k| {
        if let Some(before) = last.insert((unit[k], pages[k]), k) {
            links[before] = k;
        }
    });
    let lines = Lines {
        roles: gold
            .iter()
            .map(|g| gold_region(g).max(0) as usize * 2 + usize::from(g[1] == 1))
            .collect(),
        marked: gold.iter().map(|g| g[1] == 1 && g[5] == 1).collect(),
        starts: gold.iter().map(|g| g[2] == 1).collect(),
        links,
    };
    (lines, unit)
}

/// The region heads' answers read off the gold over the gold's own regions, and the moves they make.
#[allow(clippy::too_many_arguments)]
fn oracle_units(
    gold: &[[i64; 8]],
    views: &[LineView],
    texts: &[&str],
    n: usize,
    lines: Lines,
    unit: Vec<usize>,
    regions: Regions,
    count: usize,
) -> Vec<Move> {
    // A unit's later regions go on in its earlier ones.
    let mut unit_region: HashMap<usize, usize> = HashMap::new();
    let before: Vec<Option<usize>> = regions
        .firsts
        .iter()
        .enumerate()
        .map(|(i, &k)| {
            Some(unit[k])
                .filter(|&u| u != usize::MAX)
                .and_then(|u| unit_region.insert(u, i))
        })
        .collect();
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
        let nodes: Vec<usize> = (0..count)
            .filter(|&i| before[i].is_none() && wanted.contains(&regions.role(i)))
            .collect();
        let h = nodes.len();
        if h == 0 {
            continue;
        }
        let (mut parent, mut sibling, mut gaps) =
            (vec![0f32; h * h], vec![0f32; h * h], vec![1usize; h]);
        let firsts: Vec<usize> = nodes.iter().map(|&i| regions.firsts[i]).collect();
        let reset = resets(&firsts, &lines.starts);
        let (mut stack, mut last_child): (Vec<usize>, HashMap<Option<usize>, usize>) =
            (Vec::new(), HashMap::new());
        // A node's level: its first line's (one at least).
        let at = |b: usize| absolute[firsts[b]].max(1);
        for (a, &k) in firsts.iter().enumerate() {
            if reset[a] {
                stack.clear();
                last_child.clear();
            }
            // A node of unknown level stands one under the open node.
            let mine = if absolute[k] > 0 {
                absolute[k]
            } else {
                stack.last().map_or(1, |&b| at(b) + 1)
            };
            while stack.last().is_some_and(|&b| at(b) >= mine) {
                stack.pop();
            }
            let p = stack.last().copied();
            parent[a * h + p.unwrap_or(a)] = 1.0;
            sibling[a * h + last_child.get(&p).copied().unwrap_or(a)] = 1.0;
            gaps[a] = mine - p.map_or(0, at);
            last_child.insert(p, a);
            stack.push(a);
        }
        nodes
            .iter()
            .zip(insert(&parent, &sibling, &gaps, h, &reset))
            .for_each(|(&i, l)| level[i] = l);
    }
    // A note's region: the gold's paired mark of its number nearest before it, else after it, wherever it stands.
    let region_of = regions.of_lines(n);
    // Each mark is one note's; a parallel report's two notes of a number take the marks of their own languages.
    let mut taken: HashSet<(usize, u32)> = HashSet::new();
    let note = regions
        .firsts
        .iter()
        .enumerate()
        .map(|(i, &k)| {
            if !is_note(regions.role(i)) || before[i].is_some() {
                return None;
            }
            let value = note_number(&views[k], texts[k])?;
            let paired = |j: &usize| {
                gold[*j][3] == 1
                    && !taken.contains(&(*j, value))
                    && region_of[*j] != usize::MAX
                    && views[*j].references().iter().any(|&(v, ..)| v == value)
            };
            let same = |j: &usize| views[*j].translation() == views[k].translation();
            let found = (0..k)
                .rev()
                .filter(paired)
                .find(same)
                .or_else(|| (k + 1..n).filter(paired).find(same))
                .or_else(|| (0..k).rev().find(paired))
                .or_else(|| (k + 1..n).find(paired));
            if let Some(j) = found {
                taken.insert((j, value));
            }
            found.map(|j| region_of[j])
        })
        .collect();
    let gold_marks: Vec<bool> = gold.iter().map(|g| g[3] == 1).collect();
    moves_of(
        &lines,
        &regions,
        &Units {
            before,
            level,
            note,
        },
        views,
        texts,
        Some(&gold_marks),
    )
}

pub const DOC_KINDS: [&str; 15] = [
    "none",
    "judgment",
    "order",
    "affidavit",
    "pleading",
    "submission",
    "notice",
    "contract",
    "correspondence",
    "transcript",
    "legislation",
    "form",
    "report",
    "article",
    "other",
];
pub const TABLE_COLS: usize = 16;

/// The decoding's options (LEGALPDF_DOC_DECODE, comma-separated): `marginal`, each line's opening from the
/// opening's probability summed over the region types and its type from the type's summed over opening and going on,
/// not the 34 classes' argmax; `vote`, each region's role from its lines' summed type probabilities; `joint` (default
/// true, disabled by `no_joint`), unified log-odds line-chain linking over successor, predecessor, and opening probs;
/// `order` (default true, disabled by `no_order`), reading-order decoding via `Order.successor`; `cells` (default true,
/// disabled by `no_cells`), table-column decoding; `speakers` (default true, disabled by `no_speakers`), transcript
/// speaker-label decoding; `order_w=X`, weight on `Order`'s document-level region role refinement head (default 0.5);
/// `sure=X`, the successor's probability past which its link is kept unmatched by the predecessor's in non-joint mode
/// (else 0.5); `start=X`, the probability past which a line starts a constituent document (else 0.5), as a logit.
pub struct Decode {
    pub marginal: bool,
    pub vote: bool,
    pub joint: bool,
    pub order: bool,
    pub cells: bool,
    pub speakers: bool,
    pub order_w: f32,
    pub sure: f32,
    pub start: f32,
}

pub fn decode_options() -> &'static Decode {
    static DECODE: std::sync::OnceLock<Decode> = std::sync::OnceLock::new();
    DECODE.get_or_init(|| {
        let named = std::env::var("LEGALPDF_DOC_DECODE").unwrap_or_default();
        let parts: Vec<&str> = named.split(',').map(str::trim).collect();
        let sure = parts
            .iter()
            .find_map(|p| p.strip_prefix("sure="))
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.5);
        let order_w = parts
            .iter()
            .find_map(|p| p.strip_prefix("order_w="))
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.5);
        let start: f32 = parts
            .iter()
            .find_map(|p| p.strip_prefix("start="))
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.5);
        Decode {
            marginal: !parts.contains(&"no_marginal"),
            vote: !parts.contains(&"no_vote"),
            joint: !parts.contains(&"no_joint"),
            order: !parts.contains(&"no_order"),
            cells: !parts.contains(&"no_cells"),
            speakers: !parts.contains(&"no_speakers"),
            order_w,
            sure,
            start: (start / (1.0 - start)).ln(),
        }
    })
}

/// The line heads' answers LEGALPDF_DOC_ORACLE_PARTS names (any of `region`, `opens`, `links`, `starts`,
/// `marked`, comma-separated), read off the gold in place of the model's, on the lines the gold knows: what each head's
/// misses cost the units it decodes. A machine-aligned gold's unmarked body (text, a quotation or a list item) keeps the
/// model's region, which that gold does not decide.
pub fn oracle_parts() -> &'static HashSet<String> {
    static PARTS: std::sync::OnceLock<HashSet<String>> = std::sync::OnceLock::new();
    PARTS.get_or_init(|| {
        std::env::var("LEGALPDF_DOC_ORACLE_PARTS")
            .unwrap_or_default()
            .split(',')
            .map(|part| part.trim().to_owned())
            .filter(|part| !part.is_empty())
            .collect()
    })
}

#[cfg_attr(not(feature = "encoder"), allow(dead_code))]
fn partial_oracle(lines: &mut Lines, gold: &[[i64; 8]], pages: &[usize]) {
    let parts = oracle_parts();
    if parts.is_empty() || gold.len() != lines.roles.len() {
        return;
    }
    let (truth, _) = gold_lines(gold, pages);
    for (k, g) in gold.iter().enumerate().filter(|(_, g)| g[0] >= 0) {
        if parts.contains("region") && g[0] != UNMARKED_BODY {
            lines.roles[k] = truth.roles[k] / 2 * 2 + lines.roles[k] % 2;
        }
        if parts.contains("opens") {
            lines.roles[k] = lines.roles[k] / 2 * 2 + truth.roles[k] % 2;
        }
        if parts.contains("starts") {
            lines.starts[k] = truth.starts[k];
        }
        if parts.contains("marked") {
            lines.marked[k] = truth.marked[k];
        }
        if parts.contains("links") {
            lines.links[k] = truth.links[k];
        }
    }
    if (parts.contains("region") || parts.contains("opens")) && !parts.contains("links") {
        for k in 0..lines.links.len() {
            let next = lines.links[k];
            if next != k {
                let (rk, rn) = ((lines.roles[k] / 2) as u8, (lines.roles[next] / 2) as u8);
                if (parts.contains("region") && rk != rn)
                    || (parts.contains("opens") && lines.roles[next] % 2 == 1)
                {
                    lines.links[k] = k;
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------- the model
#[cfg(feature = "encoder")]
pub use model::Reading;
#[cfg(feature = "encoder")]
pub use model::{Reader, TextModel};

#[cfg(feature = "encoder")]
mod model {
    use super::*;
    use crate::doc::onnx::{init_runtime, session, softmax_rows};
    use ort::session::Session;
    use std::sync::Mutex;

    fn default_line_block() -> usize {
        32
    }

    fn default_link_dim() -> usize {
        64
    }

    fn default_text_parts() -> usize {
        1
    }

    #[derive(serde::Deserialize)]
    struct Meta {
        width: usize,
        text_dim: usize,
        #[serde(default = "default_text_parts")]
        text_parts: usize,
        d: usize,
        classes: usize,
        /// A template row's width (`Stylesheet::rows`).
        row_width: usize,
        /// The text model's folder, which the fit read the lines' texts with.
        text: PathBuf,
        /// The lines within this many places of a line on its page take the line heads' spatial term
        /// (`docparse.LINE_WINDOW`); the successor's and predecessor's bias for the pairs beyond (`line_far`).
        line_window: usize,
        line_far: [f32; 2],
        /// Detect's document-stream local attention block (`docparse.LINE_BLOCK`).
        #[serde(default = "default_line_block")]
        line_block: usize,
        /// Line-link query/key projection width (`docparse.RelationHead.link_dim`).
        #[serde(default = "default_link_dim")]
        link_dim: usize,
        /// Order's pairs within this many regions of each other in reading order, or whose note's number is a mark the
        /// other sets, take each head's pair term (`docparse.ORDER_WINDOW`); the pairs beyond, its bias (`order_far`).
        /// An export without them reads every pair's term.
        #[serde(default = "every_region")]
        order_window: usize,
        #[serde(default)]
        order_far: HashMap<String, f32>,
        /// Each tree's pair tables (`headings`, `bodies`), which its candidates' scores add (`trees::Scores`).
        trees: HashMap<String, super::trees::Tables>,
    }

    /// The text model as its folder holds it (`tokenizer.json`, `embeddings.f32`, `docparse.json`'s text fields): what
    /// the dump writes each line's tokens with (LEGALPDF_DOC_TEXT), and the reader each line's text vector, the fit's
    /// `Text` read the same: one tokenizing, one table, read once (where a reading first wants it), never copied.
    pub struct TextModel {
        dir: PathBuf,
        tokenizer: tokenizers::Tokenizer,
        vocabulary: usize,
        unknown: Option<u32>,
        normalize: bool,
        table: std::sync::OnceLock<std::result::Result<Vec<f32>, String>>,
    }

    impl TextModel {
        pub fn load(dir: &Path) -> std::result::Result<Self, String> {
            #[derive(serde::Deserialize)]
            struct Fields {
                vocabulary: usize,
                unknown: Option<u32>,
                normalize: bool,
            }
            let fields: Fields = serde_json::from_str(
                &std::fs::read_to_string(dir.join("docparse.json"))
                    .map_err(|e| format!("docparse.json: {e}"))?,
            )
            .map_err(|e| format!("docparse.json: {e}"))?;
            let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
                .map_err(|e| format!("tokenizer.json: {e}"))?;
            Ok(Self {
                dir: dir.to_path_buf(),
                tokenizer,
                vocabulary: fields.vocabulary,
                unknown: fields.unknown,
                normalize: fields.normalize,
                table: std::sync::OnceLock::new(),
            })
        }

        /// Each text's vector as Model2Vec reads one (`parts = 1`: mean; `parts = 3`: [mean, first, last]),
        /// L2-normalized independently per `dim` slice when `normalize` is set; a line of no tokens nought.
        pub fn vectors(
            &self,
            texts: &[&str],
            dim: usize,
            parts: usize,
        ) -> std::result::Result<Vec<f32>, String> {
            let (tokens, counts) = self.tokens(texts)?;
            self.vectors_of(&tokens, &counts, dim, parts)
        }

        /// Each line's vector as `vectors` reads one, from its tokens (`counts` of them a line, end to end).
        pub fn vectors_of(
            &self,
            tokens: &[u32],
            counts: &[usize],
            dim: usize,
            parts: usize,
        ) -> std::result::Result<Vec<f32>, String> {
            let table = self
                .table
                .get_or_init(|| {
                    let bytes = std::fs::read(self.dir.join("embeddings.f32"))
                        .map_err(|e| format!("embeddings.f32: {e}"))?;
                    Ok(bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect())
                })
                .as_ref()
                .map_err(Clone::clone)?;
            if table.len() != self.vocabulary * dim {
                return Err(format!(
                    "embeddings.f32 holds {} values, not {} tokens of {dim}",
                    table.len(),
                    self.vocabulary
                ));
            }
            let parts = parts.max(1);
            let stride = parts * dim;
            let mut out = vec![0f32; counts.len() * stride];
            let norm_slice = |s: &mut [f32]| {
                if self.normalize {
                    let norm = s.iter().map(|x| x * x).sum::<f32>().sqrt() + 1e-32;
                    s.iter_mut().for_each(|x| *x /= norm);
                }
            };
            let mut at = 0;
            for (line, &count) in counts.iter().enumerate() {
                let row = &mut out[line * stride..(line + 1) * stride];
                for &token in &tokens[at..at + count] {
                    row[..dim]
                        .iter_mut()
                        .zip(&table[token as usize * dim..(token as usize + 1) * dim])
                        .for_each(|(x, &v)| *x += v);
                }
                let mean = count.max(1) as f32;
                row[..dim].iter_mut().for_each(|x| *x /= mean);
                norm_slice(&mut row[..dim]);
                if parts >= 3 && count > 0 {
                    let first = tokens[at] as usize * dim;
                    if let Some(src) = table.get(first..first + dim) {
                        row[dim..2 * dim].copy_from_slice(src);
                        norm_slice(&mut row[dim..2 * dim]);
                    }
                    let last = tokens[at + count - 1] as usize * dim;
                    if let Some(src) = table.get(last..last + dim) {
                        row[2 * dim..3 * dim].copy_from_slice(src);
                        norm_slice(&mut row[2 * dim..3 * dim]);
                    }
                }
                at += count;
            }
            Ok(out)
        }

        /// Each text's tokens as Model2Vec reads a text (no special tokens; the unknown one and any past the vocabulary
        /// left out; an empty text read as a space), end to end, and each text's count of them.
        pub fn tokens(
            &self,
            texts: &[&str],
        ) -> std::result::Result<(Vec<u32>, Vec<usize>), String> {
            let (mut tokens, mut counts) = (Vec::new(), Vec::with_capacity(texts.len()));
            for text in texts {
                let text = if text.trim().is_empty() { " " } else { text };
                let encoding = self
                    .tokenizer
                    .encode_fast(text, false)
                    .map_err(|e| format!("tokenizer: {e}"))?;
                let before = tokens.len();
                tokens.extend(
                    encoding
                        .get_ids()
                        .iter()
                        .copied()
                        .filter(|&id| Some(id) != self.unknown && (id as usize) < self.vocabulary),
                );
                counts.push(tokens.len() - before);
            }
            Ok((tokens, counts))
        }
    }

    /// One set of the stages' sessions: a parse holds one for its document, parses at once each their own (made
    /// as they are wanted, kept for the next), none waiting on another's.
    struct Sessions {
        detect: Session,
        spatial: Session,
        order: Session,
        headings: Session,
        bodies: Session,
    }

    impl Sessions {
        fn load(dir: &Path, threads: usize) -> std::result::Result<Self, String> {
            let load = |name: &str| {
                let path = dir.join(format!("{name}.onnx"));
                session(&path, path.display(), threads)
            };
            Ok(Self {
                detect: load("detect")?,
                spatial: load("spatial")?,
                order: load("order")?,
                headings: load("headings")?,
                bodies: load("bodies")?,
            })
        }
    }

    const CONTEXT: usize = 4 * 40 + 2 + SHAPE;
    const BYTE: f32 = 8.0 / 127.0;

    /// What Detect reads of each line from the parser's bytes (`docparse.Context`): its own features, those of the two
    /// lines before and after it in reading order within its document (`0..20` and `72..92` of each; -8 past the
    /// document's edges), whether the lines before and after are on its page, and its document's shape, padded to `l_pad`.
    fn context_features(doc: &Reading, n: usize, l_pad: usize) -> Vec<f32> {
        let width = WIDTH + CONTEXT;
        let mut out = vec![0f32; l_pad * width];
        let (own, pages, shape) = (&doc.inputs.own, &doc.inputs.pages, &doc.inputs.shape);
        let shape_scaled: [f32; SHAPE] = std::array::from_fn(|s| f32::from(shape[s]) * BYTE);
        for i in 0..n {
            let row = &mut out[i * width..(i + 1) * width];
            for (dst, &b) in row[..WIDTH]
                .iter_mut()
                .zip(&own[i * WIDTH..(i + 1) * WIDTH])
            {
                *dst = f32::from(b) * BYTE;
            }
            let mut col = WIDTH;
            for offset in [-1isize, 1, -2, 2] {
                let other = i as isize + offset;
                if other >= 0 && (other as usize) < n {
                    let src = &own[other as usize * WIDTH..(other as usize + 1) * WIDTH];
                    for &b in src[0..20].iter().chain(&src[72..92]) {
                        row[col] = f32::from(b) * BYTE;
                        col += 1;
                    }
                } else {
                    row[col..col + 40].fill(-8.0);
                    col += 40;
                }
            }
            row[col] = if i > 0 && pages[i - 1] == pages[i] {
                1.0
            } else {
                0.0
            };
            row[col + 1] = if i + 1 < n && pages[i + 1] == pages[i] {
                1.0
            } else {
                0.0
            };
            row[col + 2..col + 2 + SHAPE].copy_from_slice(&shape_scaled);
        }
        out
    }

    /// DOC's stages, loaded: tier 1, which a parse reads with (`derive`) where it is given one.
    pub struct Reader {
        dir: PathBuf,
        /// Each session's threads: one where documents are read at once each on its own (a grade), every core where a
        /// document is read alone (an app's parse).
        threads: usize,
        sessions: Mutex<Vec<Sessions>>,
        /// Whether each document's stages' times are written, for the reader's speed.
        times: bool,
        text: TextModel,
        meta: Meta,
    }

    /// An export's Order window where it names none: every pair within it.
    fn every_region() -> usize {
        usize::MAX
    }

    /// The Order graph's attention block (`docparse.BLOCK`): its regions come padded to whole blocks.
    const BLOCK: usize = 64;

    fn argmax(row: &[f32]) -> usize {
        row.iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |best, (i, &x)| {
                if x > best.1 {
                    (i, x)
                } else {
                    best
                }
            })
            .0
    }

    /// A document as DOC reads it: its inputs (`inputs::Inputs`), its line table and line views, its lines' texts, and
    /// the gold's rows where the partial oracle reads them.
    pub struct Reading<'a> {
        pub inputs: &'a super::inputs::Inputs,
        pub table: &'a LineTable,
        pub views: &'a [LineView<'a>],
        pub texts: &'a [&'a str],
        pub gold: Option<&'a [[i64; 8]]>,
    }

    /// A document's lines read in, before Detect (`Reader::begin`).
    struct Begun {
        n: usize,
        text: Vec<f32>,
        anchors: Vec<i64>,
        t0: Vec<f32>,
        /// Each page's lines, in reading order.
        runs: Vec<(usize, usize)>,
    }

    /// Outputs of fused `detect.onnx` + `spatial.onnx` for one document (`n` real lines).
    struct Detected {
        states: Vec<f32>,
        role: Vec<f32>,
        flags: Vec<f32>,
        succ_q: Vec<f32>,
        succ_k: Vec<f32>,
        pred_q: Vec<f32>,
        pred_k: Vec<f32>,
        table_col: Vec<f32>,
        speaker: Vec<f32>,
        near_succ: Vec<f32>,
        near_pred: Vec<f32>,
        window: usize,
    }

    fn dot_slice(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    impl Reader {
        /// The stages under `dir`, their ONNX Runtime at `runtime` (else the one LEGALPDF_ONNX_RUNTIME or
        /// LEGALPDF_ONNXRUNTIME names), and its text model; each session run on `threads` threads.
        pub fn load(
            dir: &Path,
            runtime: Option<&Path>,
            times: bool,
            threads: usize,
        ) -> std::result::Result<Self, String> {
            let read = |name: &str| {
                std::fs::read_to_string(dir.join(name)).map_err(|e| format!("{name}: {e}"))
            };
            let meta: Meta = serde_json::from_str(&read("docparse.json")?)
                .map_err(|e| format!("docparse.json: {e}"))?;
            if meta.width != WIDTH || meta.classes != CLASSES {
                return Err(format!("docparse.json reads {} features and {} classes; the parser has {WIDTH} and {CLASSES}", meta.width, meta.classes));
            }
            init_runtime(runtime, "the DOC reader needs LEGALPDF_ONNX_RUNTIME")?;
            let text = TextModel::load(&meta.text)?;
            let first = Sessions::load(dir, threads)?;
            Ok(Self {
                dir: dir.to_path_buf(),
                threads,
                sessions: Mutex::new(vec![first]),
                times,
                text,
                meta,
            })
        }

        /// The text model the reader reads the lines' texts with.
        pub fn text(&self) -> &TextModel {
            &self.text
        }

        /// The document's moves, a line each in reading order, read off `inputs` (`texts` each line's): `read_many` of it
        /// alone. `gold`, where given with LEGALPDF_DOC_ORACLE_PARTS, takes the line heads' answers it names from the gold
        /// (`partial_oracle`): what each head's misses cost the units, the rest read as the model reads it.
        pub fn read(
            &self,
            inputs: &super::inputs::Inputs,
            table: &LineTable,
            views: &[LineView],
            texts: &[&str],
            gold: Option<&[[i64; 8]]>,
        ) -> std::result::Result<Vec<Move>, String> {
            self.read_many(&[Reading {
                inputs,
                table,
                views,
                texts,
                gold,
            }])
            .pop()
            .unwrap_or_else(|| Ok(Vec::new()))
        }

        /// Each document's moves: its lines read in a document at a time (the context, the texts, the stylesheet), Detect
        /// and its heads over every page of all of them at once (`paged`: a page reads only its own lines, so beside
        /// another document's it reads as alone), Order and the trees a document at a time; read with a set of sessions
        /// of the reader's own, given back after.
        pub fn read_many(
            &self,
            docs: &[Reading],
        ) -> Vec<std::result::Result<Vec<Move>, String>> {
            let mut sessions = match self
                .sessions
                .lock()
                .ok()
                .and_then(|mut pool| pool.pop())
                .map_or_else(|| Sessions::load(&self.dir, self.threads), Ok)
            {
                Ok(sessions) => sessions,
                Err(error) => return docs.iter().map(|_| Err(error.clone())).collect(),
            };
            let moves = self.read_many_with(&mut sessions, docs);
            if let Ok(mut pool) = self.sessions.lock() {
                pool.push(sessions);
            }
            moves
        }

        fn read_many_with(
            &self,
            sessions: &mut Sessions,
            docs: &[Reading],
        ) -> Vec<std::result::Result<Vec<Move>, String>> {
            let (d, m) = (self.meta.d, &self.meta);
            if ROW != m.row_width {
                return docs
                    .iter()
                    .map(|_| {
                        Err(format!(
                            "the stylesheet's rows are {ROW} wide; docparse.json says {}",
                            m.row_width
                        ))
                    })
                    .collect();
            }
            let block = m.line_block.max(1);
            let link_dim = m.link_dim.max(1);
            let window = m.line_window.clamp(1, 16);
            let mut laps = [0.0f64; 4];
            let mut total_lines = 0usize;
            let mut results = Vec::with_capacity(docs.len());
            for doc in docs {
                let doc_start = std::time::Instant::now();
                let mut at = doc_start;
                let begun = match self.begin(doc) {
                    Ok(b) => b,
                    Err(e) => {
                        results.push(Err(e));
                        continue;
                    }
                };
                laps[0] += at.elapsed().as_secs_f64();
                at = std::time::Instant::now();
                if begun.n == 0 {
                    results.push(Ok(Vec::new()));
                    continue;
                }
                let n = begun.n;
                total_lines += n;
                let l_pad = n.div_ceil(block) * block;
                let features = context_features(doc, n, l_pad);
                let feat_width = features.len() / l_pad;
                laps[1] += at.elapsed().as_secs_f64();
                at = std::time::Instant::now();

                let text_width = m.text_dim * m.text_parts.max(1);
                let mut text = vec![0f32; l_pad * text_width];
                text[..n * text_width].copy_from_slice(&begun.text);
                let mut boxes = vec![0f32; l_pad * 4];
                boxes[..n * 4].copy_from_slice(&doc.inputs.boxes[..n * 4]);
                let doc_pages: Vec<i64> = (0..l_pad)
                    .map(|i| if i < n { doc.inputs.pages[i] as i64 } else { -1 })
                    .collect();
                let mut anchors = vec![0i64; l_pad];
                anchors[..n].copy_from_slice(&begun.anchors);
                let mut t0 = vec![0f32; l_pad * T0];
                t0[..n * T0].copy_from_slice(&begun.t0);
                let valid: Vec<bool> = (0..l_pad).map(|i| i < n).collect();
                let mut templates = vec![0i64; l_pad];
                templates[..n].copy_from_slice(&doc.inputs.templates[..n]);
                let rows = if doc.inputs.rows.is_empty() {
                    std::borrow::Cow::Owned(vec![0.0f32; ROW])
                } else {
                    std::borrow::Cow::Borrowed(&doc.inputs.rows[..])
                };
                let num_templates = rows.len() / ROW;

                let detected = (|| -> std::result::Result<Detected, String> {
                    let shrink_detect = (l_pad > 2048)
                        .then(super::onnx::shrink_arena_options)
                        .flatten();
                    let det_inputs = ort::inputs![
                        "features" => tensor!([l_pad, feat_width], &features[..]),
                        "text" => tensor!([l_pad, text_width], &text[..]),
                        "boxes" => tensor!([l_pad, 4], &boxes[..]),
                        "pages" => tensor!([l_pad], &doc_pages[..]),
                        "anchors" => tensor!([l_pad], &anchors[..]),
                        "t0" => tensor!([l_pad, T0], &t0[..]),
                        "valid" => tensor!([l_pad], &valid[..]),
                        "templates" => tensor!([l_pad], &templates[..]),
                        "rows" => tensor!([num_templates, ROW], &rows[..])
                    ];
                    let outputs = if let Some(ref opts) = shrink_detect {
                        sessions
                            .detect
                            .run_with_options(det_inputs, opts)
                            .map_err(|e| format!("detect: {e}"))?
                    } else {
                        sessions
                            .detect
                            .run(det_inputs)
                            .map_err(|e| format!("detect: {e}"))?
                    };
                    let slice = |name: &str, w: usize| -> std::result::Result<Vec<f32>, String> {
                        Ok(output!(outputs, name)[..n * w].to_vec())
                    };
                    let (states, role, flags, succ_q, succ_k, pred_q, pred_k, table_col, speaker) = (
                        slice("states", d)?,
                        slice("role", CLASSES)?,
                        slice("flags", 2)?,
                        slice("succ_q", link_dim)?,
                        slice("succ_k", link_dim)?,
                        slice("pred_q", link_dim)?,
                        slice("pred_k", link_dim)?,
                        slice("table_col", TABLE_COLS)?,
                        slice("speaker", 1)?,
                    );
                    laps[2] += at.elapsed().as_secs_f64();
                    at = std::time::Instant::now();

                    let (mut pair_i, mut pair_j) = (Vec::new(), Vec::new());
                    for &(start, end) in &begun.runs {
                        for i in start..end {
                            for j in i..=(i + window).min(end - 1) {
                                pair_i.push(i as i64);
                                pair_j.push(j as i64);
                            }
                        }
                    }
                    let spatial_out = sessions
                        .spatial
                        .run(ort::inputs![
                            "boxes" => tensor!([n, 4], &doc.inputs.boxes[..n * 4]),
                            "pair_i" => tensor!([pair_i.len()], &pair_i[..]),
                            "pair_j" => tensor!([pair_j.len()], &pair_j[..])
                        ])
                        .map_err(|e| format!("spatial: {e}"))?;
                    let (near_succ, near_pred) = (
                        output!(spatial_out, "successor").to_vec(),
                        output!(spatial_out, "predecessor").to_vec(),
                    );
                    laps[3] += at.elapsed().as_secs_f64();
                    Ok(Detected {
                        states,
                        role,
                        flags,
                        succ_q,
                        succ_k,
                        pred_q,
                        pred_k,
                        table_col,
                        speaker,
                        near_succ,
                        near_pred,
                        window,
                    })
                })();
                let embedded = doc_start.elapsed().as_secs_f64();
                results.push(match detected {
                    Ok(det) => self.rest(sessions, doc, &begun, det, embedded),
                    Err(e) => Err(e),
                });
            }
            if self.times {
                eprintln!(
                    "group times: {} documents {total_lines} lines; texts {:.3}s context {:.3}s detect {:.3}s spatial {:.3}s",
                    docs.len(), laps[0], laps[1], laps[2], laps[3]
                );
            }
            results
        }

        /// A document's lines read in, as Detect reads them: its texts' vectors, its pages' lines, its anchors and t0.
        fn begin(&self, doc: &Reading) -> std::result::Result<Begun, String> {
            let (inputs, n) = (doc.inputs, doc.views.len());
            let runs: Vec<(usize, usize)> = inputs.pages[..n]
                .chunk_by(|a, b| a == b)
                .scan(0, |end, run| {
                    *end += run.len();
                    Some((*end - run.len(), *end))
                })
                .collect();
            let text = if n == 0 {
                Vec::new()
            } else {
                match &inputs.tokens {
                    Some((tokens, counts)) => {
                        self.text.vectors_of(tokens, counts, self.meta.text_dim, self.meta.text_parts)?
                    }
                    None => self.text.vectors(&doc.texts[..n], self.meta.text_dim, self.meta.text_parts)?,
                }
            };
            Ok(Begun {
                n,
                text,
                anchors: inputs.anchors.iter().map(|&a| i64::from(a)).collect(),
                t0: inputs.t0.iter().map(|&v| f32::from(v)).collect(),
                runs,
            })
        }

        /// A document's reading from Detect's outputs on: its lines' roles, flags and links, its regions, Order over
        /// them and the trees.
        fn rest(
            &self,
            sessions: &mut Sessions,
            doc: &Reading,
            begun: &Begun,
            det: Detected,
            embedded: f64,
        ) -> std::result::Result<Vec<Move>, String> {
            let (inputs, table, views, texts, gold) =
                (doc.inputs, doc.table, doc.views, doc.texts, doc.gold);
            let (n, runs) = (begun.n, &begun.runs);
            let d = self.meta.d;
            let link_dim = self.meta.link_dim.max(1);
            let pages = &inputs.pages;
            let boxes = &inputs.boxes;
            let (anchors, templates) = (&begun.anchors, &inputs.templates);
            let states = &det.states;
            let mut stages = [embedded, 0.0, 0.0, 0.0, 0.0];
            let mut lap = |stage: usize, since: &mut std::time::Instant| {
                stages[stage] += since.elapsed().as_secs_f64();
                *since = std::time::Instant::now();
            };
            let mut since = std::time::Instant::now();
            let mut lines = Lines {
                roles: vec![0; n],
                marked: vec![false; n],
                starts: vec![false; n],
                links: (0..n).collect(),
            };
            let decode = decode_options();
            let mut type_probs = vec![0f32; if decode.vote || decode.joint { n * 17 } else { 0 }];
            let mut p_opens = vec![0f32; if decode.joint { n } else { 0 }];
            let mut at_pair = 0usize;
            for &(start, end) in runs {
                let l = end - start;
                for i in 0..l {
                    let scores = &det.role[(start + i) * CLASSES..(start + i + 1) * CLASSES];
                    lines.roles[start + i] = if decode.marginal || decode.vote || decode.joint {
                        // Each region type's probability (opening and going on summed) and the opening's (over the types).
                        let probs = softmax_rows(scores, CLASSES);
                        let types: Vec<f32> =
                            (0..17).map(|r| probs[r * 2] + probs[r * 2 + 1]).collect();
                        let opens: f32 = (1..17).map(|r| probs[r * 2 + 1]).sum();
                        if decode.vote || decode.joint {
                            type_probs[(start + i) * 17..(start + i + 1) * 17]
                                .copy_from_slice(&types);
                        }
                        if decode.joint {
                            p_opens[start + i] = opens;
                        }
                        if decode.marginal {
                            argmax(&types) * 2 + usize::from(opens > 0.5)
                        } else {
                            argmax(scores)
                        }
                    } else {
                        argmax(scores)
                    };
                    (lines.marked[start + i], lines.starts[start + i]) = (
                        det.flags[(start + i) * 2] > 0.0,
                        det.flags[(start + i) * 2 + 1] > decode.start,
                    );
                }
                let mut succ_terms = vec![self.meta.line_far[0]; l * l];
                let mut pred_terms = vec![self.meta.line_far[1]; l * l];
                for i in 0..l {
                    for j in i..=(i + det.window).min(l - 1) {
                        succ_terms[i * l + j] = det.near_succ[at_pair];
                        pred_terms[j * l + i] = det.near_pred[at_pair];
                        at_pair += 1;
                    }
                }
                let mut succ_probs = vec![0f32; l * l];
                let mut row_buf = Vec::with_capacity(l);
                for i in 0..l {
                    row_buf.clear();
                    let qi = &det.succ_q[(start + i) * link_dim..(start + i + 1) * link_dim];
                    for j in i..l {
                        let kj = &det.succ_k[(start + j) * link_dim..(start + j + 1) * link_dim];
                        row_buf.push(dot_slice(qi, kj) + succ_terms[i * l + j]);
                    }
                    let sm = softmax_rows(&row_buf, l - i);
                    succ_probs[i * l + i..i * l + l].copy_from_slice(&sm);
                }
                let mut pred_probs = vec![0f32; l * l];
                for j in 0..l {
                    row_buf.clear();
                    let qj = &det.pred_q[(start + j) * link_dim..(start + j + 1) * link_dim];
                    for i in 0..=j {
                        let ki = &det.pred_k[(start + i) * link_dim..(start + i + 1) * link_dim];
                        row_buf.push(dot_slice(qj, ki) + pred_terms[j * l + i]);
                    }
                    let sm = softmax_rows(&row_buf, j + 1);
                    pred_probs[j * l..j * l + j + 1].copy_from_slice(&sm);
                }
                if decode.joint {
                    let mut candidates: Vec<(usize, usize)> = Vec::new();
                    for i in 0..l {
                        if let Some(best_j) = (i + 1..l).max_by(|&a, &b| {
                            succ_probs[i * l + a].total_cmp(&succ_probs[i * l + b])
                        }) {
                            candidates.push((i, best_j));
                        }
                    }
                    for j in 1..l {
                        if let Some(best_i) = (0..j).max_by(|&a, &b| {
                            pred_probs[j * l + a].total_cmp(&pred_probs[j * l + b])
                        }) {
                            candidates.push((best_i, j));
                        }
                    }
                    candidates.sort_unstable();
                    candidates.dedup();
                    let clamp_p = |x: f32| x.clamp(1e-6, 1.0 - 1e-6);
                    let strict = |r: u8| !(BODY.contains(&r) && r != TABLE);
                    let mut edges: Vec<(usize, usize, f32)> = candidates
                        .into_iter()
                        .filter_map(|(i, j)| {
                            let (a, b) = (start + i, start + j);
                            let (ra, rb) = ((lines.roles[a] / 2) as u8, (lines.roles[b] / 2) as u8);
                            if ra == 0 || rb == 0 || ((strict(ra) || strict(rb)) && ra != rb) {
                                return None;
                            }
                            let link_lo = (clamp_p(succ_probs[i * l + j])
                                / clamp_p(succ_probs[i * l + i]))
                            .ln()
                                + (clamp_p(pred_probs[j * l + i])
                                    / clamp_p(pred_probs[j * l + j]))
                                .ln();
                            if link_lo < -0.5 {
                                return None;
                            }
                            let same_role = (1..17)
                                .map(|r| type_probs[a * 17 + r] * type_probs[b * 17 + r])
                                .sum::<f32>()
                                .clamp(1e-6, 1.0);
                            let open_lo = ((1.0 - clamp_p(p_opens[b])) / clamp_p(p_opens[b]))
                                .ln()
                                .clamp(-2.5, 2.0);
                            let score = link_lo + 0.5 * open_lo + 0.5 * same_role.ln();
                            (score > 0.0).then_some((i, j, score))
                        })
                        .collect();
                    edges.sort_by(|a, b| {
                        b.2.total_cmp(&a.2)
                            .then_with(|| a.0.cmp(&b.0))
                            .then_with(|| a.1.cmp(&b.1))
                    });
                    let (mut out_used, mut in_used) = (vec![false; l], vec![false; l]);
                    for (i, j, _) in edges {
                        if !out_used[i] && !in_used[j] {
                            out_used[i] = true;
                            in_used[j] = true;
                            lines.links[start + i] = start + j;
                        }
                    }
                } else {
                    // Legacy mode (`no_joint`): link both heads choose, or successor's sure one; none into opening role.
                    for i in 0..l {
                        let next = i + argmax(&succ_probs[i * l + i..i * l + l]);
                        let back = argmax(&pred_probs[next * l..next * l + next + 1]);
                        if next != i
                            && (back == i || succ_probs[i * l + next] > decode.sure)
                            && lines.roles[start + next] % 2 == 0
                        {
                            lines.links[start + i] = start + next;
                        }
                    }
                }
            }
            lap(1, &mut since);
            margin_links(&mut lines.links, &table.margin_rows, views);
            // A line t0 reads opening a numbered paragraph on a number it prints opens on that marker.
            lines.marked.iter_mut().enumerate().for_each(|(k, marked)| {
                *marked |= views[k]
                    .count()
                    .is_some_and(|number| views[k].readings().iter().any(|r| r.value == number))
            });
            if let Some(gold) = gold {
                partial_oracle(&mut lines, gold, pages);
            }
            let oracle_region = gold.is_some() && oracle_parts().contains("region");
            let mut regions = Regions::of(&lines);
            if decode.vote && !oracle_region {
                regions.vote(&type_probs);
            }
            let count = regions.read.len();
            if count == 0 {
                return Ok(moves_of(
                    &lines,
                    &regions,
                    &Units {
                        before: Vec::new(),
                        level: Vec::new(),
                        note: Vec::new(),
                    },
                    views,
                    texts,
                    None,
                ));
            }
            // Order over the regions read.
            let of_line = regions.of_lines(n);
            let in_read = || (0..n).filter(|&k| of_line[k] != usize::MAX);
            let line_states: Vec<f32> = in_read()
                .flat_map(|k| states[k * d..(k + 1) * d].iter().copied())
                .collect();
            let line_region: Vec<i64> = in_read().map(|k| of_line[k] as i64).collect();
            let mut region_boxes = region_boxes(&boxes, in_read().map(|k| (k, of_line[k])), count);
            let mut marks: Vec<HashSet<u32>> = vec![HashSet::new(); count];
            for k in in_read() {
                let i = of_line[k];
                if REFERRING.contains(&regions.role(i)) {
                    marks[i].extend(views[k].references().iter().map(|&(v, ..)| v));
                }
            }
            let mut region_pages: Vec<i64> =
                regions.firsts.iter().map(|&k| pages[k] as i64).collect();
            let number: Vec<Option<u32>> = regions
                .firsts
                .iter()
                .enumerate()
                .map(|(i, &k)| {
                    is_note(regions.role(i))
                        .then(|| note_number(&views[k], texts[k]))
                        .flatten()
                })
                .collect();
            let mut by_page: HashMap<i64, Vec<usize>> = HashMap::new();
            region_pages
                .iter()
                .enumerate()
                .for_each(|(i, &page)| by_page.entry(page).or_default().push(i));
            let mut setting: HashMap<u32, Vec<usize>> = HashMap::new();
            marks.iter().enumerate().for_each(|(j, set)| {
                set.iter()
                    .for_each(|&value| setting.entry(value).or_default().push(j))
            });
            // Three filtered pair lists: `goes_on`, `note_of`, and `order` ([page - 1, page + 1] pairs).
            let (mut goes_i, mut goes_j) = (Vec::new(), Vec::new());
            let mut goes_rows = vec![(0usize, 0usize); count];
            let (mut note_i, mut note_j, mut note_match) = (Vec::new(), Vec::new(), Vec::new());
            let mut note_rows = vec![(0usize, 0usize); count];
            let (mut ord_i, mut ord_j) = (Vec::new(), Vec::new());
            let mut ord_rows = vec![(0usize, 0usize); count];
            for (i, &page) in region_pages.iter().enumerate() {
                let ri = regions.role(i);
                let g0 = goes_i.len();
                goes_i.push(i as i64);
                goes_j.push(i as i64);
                if ri != LINE_LABEL {
                    for near_page in [page, page + 1] {
                        for &j in by_page.get(&near_page).into_iter().flatten() {
                            if j > i {
                                let rj = regions.role(j);
                                if rj != LINE_LABEL && is_note(ri) == is_note(rj) {
                                    goes_i.push(i as i64);
                                    goes_j.push(j as i64);
                                }
                            }
                        }
                    }
                }
                goes_rows[i] = (g0, goes_i.len());
                let o0 = ord_i.len();
                ord_i.push(i as i64);
                ord_j.push(i as i64);
                for near_page in [page - 1, page, page + 1] {
                    for &j in by_page.get(&near_page).into_iter().flatten() {
                        if j != i {
                            ord_i.push(i as i64);
                            ord_j.push(j as i64);
                        }
                    }
                }
                ord_rows[i] = (o0, ord_i.len());
                if is_note(ri) {
                    if let Some(val) = number[i] {
                        let n0 = note_i.len();
                        note_i.push(i as i64);
                        note_j.push(i as i64);
                        note_match.push(false);
                        for &j in setting.get(&val).into_iter().flatten() {
                            if j != i {
                                note_i.push(i as i64);
                                note_j.push(j as i64);
                                note_match.push(true);
                            }
                        }
                        note_rows[i] = (n0, note_i.len());
                    }
                }
            }
            let has_notes = !note_i.is_empty();
            if note_i.is_empty() {
                note_i.push(0);
                note_j.push(0);
                note_match.push(false);
            }
            // The regions padded to whole blocks of the graph's local attention (`docparse.BLOCK`), the padding marked.
            let padded = count.div_ceil(BLOCK) * BLOCK;
            let mut opener = vec![0f32; padded * d];
            let mut closer = vec![0f32; padded * d];
            for (i, &r_idx) in regions.read.iter().enumerate() {
                let members = &regions.lines[r_idx];
                let first = members[0];
                let last = *members.last().expect("read region has a line");
                opener[i * d..(i + 1) * d].copy_from_slice(&states[first * d..(first + 1) * d]);
                closer[i * d..(i + 1) * d].copy_from_slice(&states[last * d..(last + 1) * d]);
            }
            let kinds: Vec<i64> = (0..padded)
                .map(|i| {
                    if i < count {
                        i64::from(regions.role(i))
                    } else {
                        0
                    }
                })
                .collect();
            let region_valid: Vec<bool> = (0..padded).map(|i| i < count).collect();
            region_boxes.resize(padded * 4, 0.0);
            region_pages.resize(padded, -1);
            let near_goes: Vec<usize> = (0..goes_i.len())
                .filter(|&p| goes_i[p].abs_diff(goes_j[p]) <= self.meta.order_window as u64)
                .collect();
            let (near_goes_i, near_goes_j): (Vec<i64>, Vec<i64>) = (
                near_goes.iter().map(|&p| goes_i[p]).collect(),
                near_goes.iter().map(|&p| goes_j[p]).collect(),
            );
            let near_note: Vec<usize> = (0..note_i.len())
                .filter(|&p| {
                    note_match[p] || note_i[p].abs_diff(note_j[p]) <= self.meta.order_window as u64
                })
                .collect();
            let (near_note_i, near_note_j, near_note_match): (Vec<i64>, Vec<i64>, Vec<bool>) = (
                near_note.iter().map(|&p| note_i[p]).collect(),
                near_note.iter().map(|&p| note_j[p]).collect(),
                near_note.iter().map(|&p| note_match[p]).collect(),
            );
            let near_ord: Vec<usize> = (0..ord_i.len())
                .filter(|&p| ord_i[p].abs_diff(ord_j[p]) <= self.meta.order_window as u64)
                .collect();
            let (near_ord_i, near_ord_j): (Vec<i64>, Vec<i64>) = (
                near_ord.iter().map(|&p| ord_i[p]).collect(),
                near_ord.iter().map(|&p| ord_j[p]).collect(),
            );
            let total_pairs = near_goes.len() + near_note.len() + near_ord.len();
            let (region_states, goes_on, note_of, order_scores, region_logits, region_start, region_doc_kind) = {
                let shrink_opts = (padded > 512)
                    .then(super::onnx::shrink_arena_options)
                    .flatten();
                let ord_inputs = ort::inputs![
                    "line_states" => tensor!([line_region.len(), d], &line_states[..]),
                    "line_region" => tensor!([line_region.len()], &line_region[..]),
                    "opener" => tensor!([padded, d], &opener[..]),
                    "closer" => tensor!([padded, d], &closer[..]),
                    "region_roles" => tensor!([padded], &kinds[..]),
                    "region_valid" => tensor!([padded], &region_valid[..]),
                    "region_boxes" => tensor!([padded, 4], &region_boxes[..]),
                    "region_pages" => tensor!([padded], &region_pages[..]),
                    "goes_i" => tensor!([near_goes_i.len()], &near_goes_i[..]),
                    "goes_j" => tensor!([near_goes_j.len()], &near_goes_j[..]),
                    "note_i" => tensor!([near_note_i.len()], &near_note_i[..]),
                    "note_j" => tensor!([near_note_j.len()], &near_note_j[..]),
                    "note_match" => tensor!([near_note_match.len()], &near_note_match[..]),
                    "ord_i" => tensor!([near_ord_i.len()], &near_ord_i[..]),
                    "ord_j" => tensor!([near_ord_j.len()], &near_ord_j[..])
                ];
                let outputs = if let Some(ref opts) = shrink_opts {
                    sessions
                        .order
                        .run_with_options(ord_inputs, opts)
                        .map_err(|e| format!("order: {e}"))?
                } else {
                    sessions
                        .order
                        .run(ord_inputs)
                        .map_err(|e| format!("order: {e}"))?
                };
                let take = |name: &str| {
                    outputs[name]
                        .try_extract_tensor::<f32>()
                        .map(|(_, v)| v.to_vec())
                        .map_err(|e| e.to_string())
                };
                let head = |name: &str,
                            out_name: &str,
                            pi: &[i64],
                            pj: &[i64],
                            near: &[usize]|
                 -> std::result::Result<Vec<f32>, String> {
                    let (q, k, near_r) = (
                        take(&format!("{out_name}_q"))?,
                        take(&format!("{out_name}_k"))?,
                        take(&format!("{out_name}_r"))?,
                    );
                    let mut r =
                        vec![self.meta.order_far.get(name).copied().unwrap_or(0.0); pi.len()];
                    near.iter().zip(&near_r).for_each(|(&p, &term)| r[p] = term);
                    Ok((0..pi.len())
                        .map(|p| {
                            let (i, j) = (pi[p] as usize, pj[p] as usize);
                            q[i * d..(i + 1) * d]
                                .iter()
                                .zip(&k[j * d..(j + 1) * d])
                                .map(|(a, b)| a * b)
                                .sum::<f32>()
                                + r[p]
                        })
                        .collect())
                };
                let opt = |name: &str| {
                    outputs
                        .get(name)
                        .and_then(|v| v.try_extract_tensor::<f32>().ok().map(|(_, s)| s.to_vec()))
                };
                let ord_scores = if decode.order && outputs.contains_key("order_q") {
                    head("successor", "order", &ord_i, &ord_j, &near_ord)?
                } else {
                    Vec::new()
                };
                (
                    take("regions")?,
                    head("goes_on", "goes_on", &goes_i, &goes_j, &near_goes)?,
                    if has_notes {
                        head("note_of", "note_of", &note_i, &note_j, &near_note)?
                    } else {
                        Vec::new()
                    },
                    ord_scores,
                    opt("region_logits"),
                    opt("region_start"),
                    opt("region_doc_kind"),
                )
            };
            let oracle_starts = gold.is_some() && oracle_parts().contains("starts");
            if let Some(ref logits) = region_logits {
                if decode.vote && !oracle_region && decode.order_w > 0.0 && logits.len() >= count * 17
                {
                    let refinable = |r: u8| {
                        (BODY.contains(&r) && r != TABLE)
                            || matches!(r, HEADING | FURNITURE | PAGE_LABEL)
                    };
                    for i in 0..count {
                        let r_idx = regions.read[i];
                        if !refinable(regions.roles[r_idx]) {
                            continue;
                        }
                        let members = &regions.lines[r_idx];
                        let order_probs = softmax_rows(&logits[i * 17..(i + 1) * 17], 17);
                        if order_probs
                            .iter()
                            .take(17)
                            .any(|&p| (p - 1.0 / 17.0).abs() > 1e-3)
                        {
                            let mut sums = [0f32; 17];
                            for &k in members {
                                for r in 1..17 {
                                    sums[r] += type_probs[k * 17 + r];
                                }
                            }
                            let weight = (members.len() as f32).max(1.0) * decode.order_w;
                            for r in 1..17 {
                                sums[r] += weight * order_probs[r];
                            }
                            let best = (1..17)
                                .filter(|&r| refinable(r as u8))
                                .fold(usize::from(regions.roles[r_idx]), |best, r| {
                                    if sums[r] > sums[best] {
                                        r
                                    } else {
                                        best
                                    }
                                }) as u8;
                            regions.roles[r_idx] = best;
                        }
                    }
                }
            }
            if !oracle_starts {
                if let Some(ref reg_start) = region_start {
                    lines.starts.fill(false);
                    for (i, &first) in regions.firsts.iter().enumerate() {
                        let r = regions.role(i);
                        if first > 0
                            && i < reg_start.len()
                            && !matches!(
                                r,
                                PROVISION
                                    | LIST_ITEM
                                    | QUOTE
                                    | NOTE
                                    | NOTE_QUOTE
                                    | TABLE
                                    | LINE_LABEL
                            )
                        {
                            let det_s = det.flags[first * 2 + 1];
                            let ord_s = reg_start[i];
                            let combined = 0.4 * det_s + 0.6 * ord_s;
                            lines.starts[first] =
                                combined > decode.start + 1.1 && det_s > -0.5 && ord_s > 0.0;
                        }
                    }
                }
            }
            let (firsts, role) = (&regions.firsts, |i: usize| regions.role(i));
            let row_softmax = |logits: &[f32], rows: &[(usize, usize)]| -> Vec<f32> {
                let mut probs = vec![0f32; logits.len()];
                rows.iter().filter(|&&(a, z)| a < z).for_each(|&(a, z)| {
                    probs[a..z].copy_from_slice(&softmax_rows(&logits[a..z], z - a))
                });
                probs
            };
            let goes_on = row_softmax(&goes_on, &goes_rows);
            let note_of = if has_notes {
                row_softmax(&note_of, &note_rows)
            } else {
                Vec::new()
            };
            // A region's likeliest other region by a head, with its probability.
            let best = |probs: &[f32],
                        rows: &[(usize, usize)],
                        pj: &[i64],
                        i: usize,
                        open: &dyn Fn(usize) -> bool|
             -> Option<(usize, f32)> {
                let (a, z) = rows[i];
                (a..z)
                    .filter(|&p| open(pj[p] as usize))
                    .max_by(|&x, &y| probs[x].total_cmp(&probs[y]))
                    .map(|p| (pj[p] as usize, probs[p]))
            };
            // Units: a region goes on in the later region its link holds likeliest, each region taken once, the
            // surest links first.
            let mut links: Vec<(usize, usize, f32)> = (0..count)
                .filter_map(|i| {
                    best(&goes_on, &goes_rows, &goes_j, i, &|_| true)
                        .filter(|&(j, _)| j > i)
                        .map(|(j, p)| (i, j, p))
                })
                .collect();
            links.sort_by(|a, b| b.2.total_cmp(&a.2));
            let (mut before, mut after) = (vec![None; count], vec![false; count]);
            for (i, j, _) in links {
                // A paragraph the margin numbers, or t0 reads opening, or in a different stream, opens: it goes on in no earlier one.
                let opens =
                    views[firsts[j]].margin().is_some() || views[firsts[j]].count().is_some();
                let same_stream = stream_of(role(i)) == stream_of(role(j))
                    && (role(i) == HEADING) == (role(j) == HEADING)
                    && !(firsts[i] + 1..=firsts[j]).any(|k| lines.starts[k]);
                if before[j].is_none() && !after[i] && !opens && same_stream {
                    (before[j], after[i]) = (Some(i), true);
                }
            }
            goes_on_in_column(
                &mut before,
                &regions,
                &lines.roles,
                &lines.marked,
                &inputs.t0,
                views,
                pages,
            );
            lap(2, &mut since);
            // The trees over the regions that open a unit, headings and the body's apart.
            let (sections, enclosing_heading) = sections(&resets(firsts, &lines.starts), |i| {
                before[i].is_none() && role(i) == HEADING
            });
            let line_family = |k: usize| inputs.markers[k];
            let tree = |session: &mut Session,
                        tables: &super::trees::Tables,
                        nodes: &[usize]|
             -> std::result::Result<Vec<usize>, String> {
                let h = nodes.len();
                if h == 0 {
                    return Ok(Vec::new());
                }
                // Each node's measure: of its region (`i`) and its first line (`k`).
                let node = |value: &dyn Fn(usize, usize) -> i64| -> Vec<i64> {
                    nodes.iter().map(|&i| value(i, firsts[i])).collect()
                };
                let node_f32 = |of: &dyn Fn(usize) -> f32| -> Vec<f32> {
                    nodes.iter().map(|&i| of(firsts[i])).collect()
                };
                let x: Vec<f32> = nodes
                    .iter()
                    .flat_map(|&i| region_states[i * d..(i + 1) * d].iter().copied())
                    .collect();
                let node_openers: Vec<f32> = nodes
                    .iter()
                    .flat_map(|&i| opener[i * d..(i + 1) * d].iter().copied())
                    .collect();
                let mut h_states = vec![0f32; h * d];
                nodes
                    .iter()
                    .enumerate()
                    .filter_map(|(at, &i)| Some((at, enclosing_heading[i]?)))
                    .for_each(|(at, hd)| {
                        h_states[at * d..(at + 1) * d]
                            .copy_from_slice(&region_states[hd * d..(hd + 1) * d])
                    });
                let (node_templates, node_anchors) =
                    (node(&|_, k| templates[k]), node(&|_, k| anchors[k]));
                let (node_fam, node_val, node_pref) = (
                    node(&|_, k| line_family(k).0),
                    node(&|_, k| line_family(k).1),
                    node(&|_, k| line_family(k).2),
                );
                let (node_sec, node_x0, node_size) = (
                    node(&|i, _| sections[i]),
                    node_f32(&|k| boxes[k * 4]),
                    node_f32(&|k| inputs.size[k]),
                );
                let (node_looks, node_pages, node_roles) = (
                    node(&|_, k| inputs.looks[k]),
                    node(&|_, k| pages[k] as i64),
                    node(&|i, _| i64::from(role(i))),
                );
                let outputs = session.run(ort::inputs![
                    "states" => tensor!([h, d], &x[..]),
                    "opener_states" => tensor!([h, d], &node_openers[..]),
                    "heading_states" => tensor!([h, d], &h_states[..]),
                    "family" => tensor!([h], &node_fam[..]), "value" => tensor!([h], &node_val[..]), "prefix" => tensor!([h], &node_pref[..]),
                    "x0" => tensor!([h], &node_x0[..]), "size" => tensor!([h], &node_size[..]), "roles" => tensor!([h], &node_roles[..])])
                    .map_err(|e| format!("construct: {e}"))?;
                let vectors = [
                    output!(outputs, "parent_q"),
                    output!(outputs, "parent_k"),
                    output!(outputs, "sibling_q"),
                    output!(outputs, "sibling_k"),
                ];
                let gaps: Vec<usize> = output!(outputs, "gap")
                    .chunks_exact(GAPS)
                    .map(|row| argmax(row) + 1)
                    .collect();
                let scores = super::trees::Scores::new(
                    super::trees::Nodes {
                        templates: &node_templates,
                        anchors: &node_anchors,
                        family: &node_fam,
                        value: &node_val,
                        prefix: &node_pref,
                        section: &node_sec,
                        x0: &node_x0,
                        size: &node_size,
                        looks: &node_looks,
                        pages: &node_pages,
                        roles: &node_roles,
                    },
                    tables,
                    vectors,
                    d,
                );
                let node_firsts: Vec<usize> = nodes.iter().map(|&i| firsts[i]).collect();
                Ok(insert_scored(
                    &scores,
                    &gaps,
                    h,
                    &resets(&node_firsts, &lines.starts),
                ))
            };
            let opening = |i: usize| before[i].is_none();
            let headings: Vec<usize> = (0..count)
                .filter(|&i| opening(i) && role(i) == HEADING)
                .collect();
            let bodies: Vec<usize> = (0..count)
                .filter(|&i| opening(i) && BODY.contains(&role(i)))
                .collect();
            let mut level = vec![0usize; count];
            for (nodes, session, name) in [
                (&headings, &mut sessions.headings, "headings"),
                (&bodies, &mut sessions.bodies, "bodies"),
            ] {
                let tables = self
                    .meta
                    .trees
                    .get(name)
                    .ok_or_else(|| format!("docparse.json has no {name} tables"))?;
                nodes
                    .iter()
                    .zip(tree(session, tables, nodes)?)
                    .for_each(|(&i, l)| level[i] = l);
            }
            // A note's region: the one holding a mark of its number its link names, where not itself, surest notes first
            // so two notes of a number in parallel or multi-part reports do not take the same region's mark.
            let mut note = vec![None; count];
            let mut note_candidates: Vec<(usize, f32)> = (0..count)
                .filter(|&i| before[i].is_none() && is_note(role(i)) && number[i].is_some())
                .map(|i| {
                    let valid =
                        |j: usize| j == i || number[i].is_some_and(|v| marks[j].contains(&v));
                    (
                        i,
                        best(&note_of, &note_rows, &note_j, i, &valid)
                            .map_or(0.0, |(j, p)| if j != i { p } else { 0.0 }),
                    )
                })
                .collect();
            note_candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let mut claimed: HashSet<(usize, u32)> = HashSet::new();
            for (i, _) in note_candidates {
                let Some(val) = number[i] else { continue };
                let valid =
                    |j: usize| j == i || (marks[j].contains(&val) && !claimed.contains(&(j, val)));
                let target = best(&note_of, &note_rows, &note_j, i, &valid)
                    .or_else(|| {
                        best(&note_of, &note_rows, &note_j, i, &|j| {
                            j == i || marks[j].contains(&val)
                        })
                    })
                    .map_or(i, |(j, _)| j);
                if target != i {
                    claimed.insert((target, val));
                    note[i] = Some(target);
                }
            }
            lap(3, &mut since);
            let mut moves = moves_of(
                &lines,
                &regions,
                &Units {
                    before,
                    level,
                    note,
                },
                views,
                texts,
                None,
            );
            if decode.order && !order_scores.is_empty() {
                for &(p_start, p_end) in runs {
                    if p_end - p_start < 2 {
                        continue;
                    }
                    let page = pages[p_start] as i64;
                    let Some(page_regs) = by_page.get(&page) else {
                        continue;
                    };
                    if page_regs.len() < 2
                        || page_regs.iter().any(|&i| regions.role(i) == LINE_LABEL)
                    {
                        continue;
                    }
                    let mut side_by_side = false;
                    for (idx, &a) in page_regs.iter().enumerate() {
                        if matches!(regions.role(a), FURNITURE | PAGE_LABEL | LINE_LABEL) {
                            continue;
                        }
                        let (ax0, ay0, ax1, ay1) = (
                            region_boxes[a * 4],
                            region_boxes[a * 4 + 1],
                            region_boxes[a * 4 + 2],
                            region_boxes[a * 4 + 3],
                        );
                        let ah = (ay1 - ay0).max(1e-3);
                        for &b in &page_regs[idx + 1..] {
                            if matches!(regions.role(b), FURNITURE | PAGE_LABEL | LINE_LABEL) {
                                continue;
                            }
                            let (bx0, by0, bx1, by1) = (
                                region_boxes[b * 4],
                                region_boxes[b * 4 + 1],
                                region_boxes[b * 4 + 2],
                                region_boxes[b * 4 + 3],
                            );
                            let bh = (by1 - by0).max(1e-3);
                            let y_ov = (ay1.min(by1) - ay0.max(by0)).max(0.0);
                            if y_ov > 0.20 * ah.min(bh)
                                && (ax1 <= bx0 + 0.02 || bx1 <= ax0 + 0.02)
                            {
                                side_by_side = true;
                                break;
                            }
                        }
                        if side_by_side {
                            break;
                        }
                    }
                    if !side_by_side {
                        continue;
                    }
                    let mut edges: Vec<(usize, usize, f32)> = Vec::new();
                    for &i in page_regs {
                        let (o0, o1) = ord_rows[i];
                        if o1 <= o0 {
                            continue;
                        }
                        let probs = softmax_rows(&order_scores[o0..o1], o1 - o0);
                        let (best_p, &best_prob) = probs
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))
                            .expect("non-empty ord row");
                        let j = ord_j[o0 + best_p] as usize;
                        if j != i && region_pages[j] == page && best_prob > 0.55 {
                            edges.push((i, j, best_prob));
                        }
                    }
                    if edges.is_empty() {
                        continue;
                    }
                    edges.sort_by(|a, b| {
                        b.2.total_cmp(&a.2)
                            .then_with(|| a.0.cmp(&b.0))
                            .then_with(|| a.1.cmp(&b.1))
                    });
                    let mut next_reg: HashMap<usize, usize> = HashMap::new();
                    let mut has_prev: HashSet<usize> = HashSet::new();
                    for (i, j, _) in edges {
                        if next_reg.contains_key(&i) || has_prev.contains(&j) {
                            continue;
                        }
                        let mut cur = j;
                        let mut cycle = false;
                        while let Some(&nxt) = next_reg.get(&cur) {
                            if nxt == i {
                                cycle = true;
                                break;
                            }
                            cur = nxt;
                        }
                        if !cycle {
                            next_reg.insert(i, j);
                            has_prev.insert(j);
                        }
                    }
                    let mut ordered_regs = Vec::with_capacity(page_regs.len());
                    for &head in page_regs {
                        if has_prev.contains(&head) {
                            continue;
                        }
                        let mut cur = head;
                        loop {
                            ordered_regs.push(cur);
                            let Some(&nxt) = next_reg.get(&cur) else {
                                break;
                            };
                            cur = nxt;
                        }
                    }
                    if ordered_regs.len() == page_regs.len() && &ordered_regs != page_regs {
                        let mut after_line: HashMap<isize, Vec<usize>> = HashMap::new();
                        let mut prev_placed: isize = -1;
                        for k in p_start..p_end {
                            if of_line[k] != usize::MAX {
                                prev_placed = k as isize;
                            } else {
                                after_line.entry(prev_placed).or_default().push(k);
                            }
                        }
                        let mut reordered = Vec::with_capacity(p_end - p_start);
                        if let Some(prefix_lines) = after_line.get(&-1) {
                            reordered.extend_from_slice(prefix_lines);
                        }
                        for r in ordered_regs {
                            for &k in &regions.lines[regions.read[r]] {
                                reordered.push(k);
                                if let Some(trailing) = after_line.get(&(k as isize)) {
                                    reordered.extend_from_slice(trailing);
                                }
                            }
                        }
                        if reordered.len() == p_end - p_start {
                            for (pos, line) in reordered.into_iter().enumerate() {
                                moves[line].rank = Some(pos as u32);
                            }
                        }
                    }
                }
            }
            if let Some(ref kind_logits) = region_doc_kind {
                let n_kinds = DOC_KINDS.len();
                if kind_logits.len() >= count * n_kinds {
                    let mut seg_start_line: Option<usize> = Some(0);
                    let mut sums = [0f32; 15];
                    let flush_seg =
                        |start_line: Option<usize>, sums: &[f32; 15], moves: &mut [Move]| {
                            let Some(sk) = start_line else { return };
                            if sk == 0 && moves.first().is_some_and(|m| m.starts) {
                                return;
                            }
                            if let Some(best) =
                                (1..n_kinds).max_by(|&a, &b| sums[a].total_cmp(&sums[b]))
                            {
                                if sums[best] > 0.0 {
                                    moves[sk].document = Some(DOC_KINDS[best].to_owned());
                                }
                            }
                        };
                    for i in 0..count {
                        let fk = regions.firsts[i];
                        if fk > 0 && moves[fk].starts {
                            flush_seg(seg_start_line, &sums, &mut moves);
                            seg_start_line = Some(fk);
                            sums = [0f32; 15];
                        }
                        let probs = softmax_rows(&kind_logits[i * n_kinds..(i + 1) * n_kinds], n_kinds);
                        for (c, &p) in probs.iter().enumerate() {
                            sums[c] += p;
                        }
                    }
                    flush_seg(seg_start_line, &sums, &mut moves);
                }
            }
            if decode.cells {
                let mut tables: Vec<Vec<Vec<usize>>> = Vec::new();
                let mut running = false;
                let mut unit_idx: HashMap<usize, (usize, usize)> = HashMap::new();
                for k in 0..n {
                    match moves[k].region {
                        TABLE => {
                            if !running {
                                tables.push(Vec::new());
                                running = true;
                            }
                            let t_at = tables.len() - 1;
                            if let Some(&(_, u_at)) = moves[k]
                                .resumes
                                .and_then(|first| unit_idx.get(&(first as usize)))
                                .filter(|&&(t_prev, _)| t_prev == t_at)
                            {
                                tables[t_at][u_at].push(k);
                            } else {
                                let u_at = tables[t_at].len();
                                tables[t_at].push(vec![k]);
                                unit_idx.insert(k, (t_at, u_at));
                            }
                        }
                        0 | FURNITURE | PAGE_LABEL | LINE_LABEL | NOTE => {}
                        _ => running = false,
                    }
                }
                for t_units in tables {
                    if !t_units.iter().any(|u| u.len() >= 2) {
                        continue;
                    }
                    let mut all_rows: Vec<Vec<(f32, f32, Vec<usize>)>> =
                        Vec::with_capacity(t_units.len());
                    for u_lines in &t_units {
                        let mut row_cols: Vec<(f32, f32, Vec<usize>)> = Vec::new();
                        for &k in u_lines {
                            if texts[k].trim().is_empty() {
                                continue;
                            }
                            let (x0, y0, x1, y1) = (
                                boxes[k * 4],
                                boxes[k * 4 + 1],
                                boxes[k * 4 + 2],
                                boxes[k * 4 + 3],
                            );
                            let yc = 0.5 * (y0 + y1);
                            let hk = (y1 - y0).max(1e-3);
                            let wk = (x1 - x0).max(1e-3);
                            let mut best_c: Option<(usize, f32)> = None;
                            for (c_idx, (cx0, cx1, members)) in row_cols.iter().enumerate() {
                                let same_y = members.iter().any(|&m| {
                                    if pages[m] != pages[k] {
                                        return false;
                                    }
                                    let (my0, my1) = (boxes[m * 4 + 1], boxes[m * 4 + 3]);
                                    let myc = 0.5 * (my0 + my1);
                                    let hm = (my1 - my0).max(1e-3);
                                    (yc - myc).abs() < 0.45 * hk.max(hm)
                                });
                                if same_y {
                                    continue;
                                }
                                let cw = (cx1 - cx0).max(1e-3);
                                let x_ov = (x1.min(*cx1) - x0.max(*cx0)).max(0.0);
                                let dx0 = (x0 - *cx0).abs();
                                if x_ov >= 0.20 * wk.min(cw) || dx0 < 0.03 {
                                    if best_c.is_none_or(|(_, b_dx)| dx0 < b_dx) {
                                        best_c = Some((c_idx, dx0));
                                    }
                                }
                            }
                            if let Some((c_idx, _)) = best_c {
                                let entry = &mut row_cols[c_idx];
                                entry.0 = entry.0.min(x0);
                                entry.1 = entry.1.max(x1);
                                entry.2.push(k);
                            } else {
                                row_cols.push((x0, x1, vec![k]));
                            }
                        }
                        row_cols.sort_by(|a, b| a.0.total_cmp(&b.0));
                        all_rows.push(row_cols);
                    }
                    let max_cols = all_rows
                        .iter()
                        .map(Vec::len)
                        .max()
                        .unwrap_or(0)
                        .min(TABLE_COLS);
                    if max_cols < 2 {
                        continue;
                    }
                    let mut canon: Vec<(f32, f32)> = all_rows
                        .iter()
                        .find(|r| r.len() >= max_cols)
                        .map(|r| r.iter().take(TABLE_COLS).map(|&(x0, x1, _)| (x0, x1)).collect())
                        .unwrap_or_default();
                    for row in &all_rows {
                        if row.len() < 2 {
                            continue;
                        }
                        for &(cx0, cx1, _) in row {
                            let cw = (cx1 - cx0).max(1e-3);
                            let mut matched = None;
                            for (b_idx, &(bx0, bx1)) in canon.iter().enumerate() {
                                let bw = (bx1 - bx0).max(1e-3);
                                let x_ov = (cx1.min(bx1) - cx0.max(bx0)).max(0.0);
                                let dx0 = (cx0 - bx0).abs();
                                if x_ov >= 0.25 * cw.min(bw) || dx0 < 0.035 {
                                    if matched.is_none_or(|(_, best_dx)| dx0 < best_dx) {
                                        matched = Some((b_idx, dx0));
                                    }
                                }
                            }
                            if let Some((b_idx, _)) = matched {
                                canon[b_idx].0 = 0.5 * (canon[b_idx].0 + cx0);
                                canon[b_idx].1 = canon[b_idx].1.max(cx1);
                            } else if canon.len() < TABLE_COLS {
                                canon.push((cx0, cx1));
                                canon.sort_by(|a, b| a.0.total_cmp(&b.0));
                            }
                        }
                    }
                    let w_table = canon.len().max(2).min(TABLE_COLS);
                    for row in &all_rows {
                        let m_row = row.len();
                        if m_row < 2 {
                            continue;
                        }
                        let mut prev_col: Option<usize> = None;
                        for (i, (cx0, cx1, members)) in row.iter().enumerate() {
                            let col = if m_row == w_table {
                                i
                            } else {
                                let min_c = prev_col.map_or(0, |p| (p + 1).min(w_table - 1));
                                let max_c = if m_row <= w_table {
                                    w_table.saturating_sub(m_row - i).max(min_c)
                                } else {
                                    w_table - 1
                                };
                                let cost = |c: usize| -> f32 {
                                    let x_ov = (cx1.min(canon[c].1) - cx0.max(canon[c].0)).max(0.0);
                                    let prior = members
                                        .iter()
                                        .filter_map(|&k| det.table_col.get(k * TABLE_COLS + c))
                                        .copied()
                                        .sum::<f32>()
                                        / (members.len().max(1) as f32);
                                    (cx0 - canon[c].0).abs() - x_ov - 0.002 * prior.clamp(-10.0, 10.0)
                                };
                                (min_c..=max_c)
                                    .min_by(|&a, &b| cost(a).total_cmp(&cost(b)))
                                    .unwrap_or(i.min(w_table - 1))
                            };
                            prev_col = Some(col);
                            for &k in members {
                                let len = texts[k].encode_utf16().count() as u32;
                                moves[k].cells = Some(if len > 0 {
                                    vec![(col.min(TABLE_COLS - 1) as u32, 0, len)]
                                } else {
                                    Vec::new()
                                });
                            }
                        }
                    }
                }
            }
            if decode.speakers {
                let to_u16 = |t: &str, b: usize| t[..b.min(t.len())].encode_utf16().count() as u32;
                for k in 0..n {
                    let mut spans = Vec::new();
                    if det.speaker.get(k).copied().unwrap_or(0.0) > 0.0 {
                        let t = texts[k];
                        for (s, e) in crate::rules::layout::speaker_labels(t) {
                            spans.push((to_u16(t, s), to_u16(t, e)));
                        }
                        if spans.is_empty() {
                            if let Some(colon) = t.find(':') {
                                let raw = &t[..=colon];
                                let lead = raw.len() - raw.trim_start().len();
                                if lead < raw.len() && raw[lead..].encode_utf16().count() <= 45 {
                                    spans.push((to_u16(t, lead), to_u16(t, colon + 1)));
                                }
                            }
                        }
                    }
                    moves[k].speakers = Some(spans);
                }
            }
            lap(4, &mut since);
            if self.times {
                eprintln!("doc times: {n} lines {count} regions {total_pairs} pairs; embed {:.3}s detect {:.3}s order {:.3}s trees {:.3}s decode {:.3}s",
                    stages[0], stages[1], stages[2], stages[3], stages[4]);
            }
            Ok(moves)
        }
    }
}
