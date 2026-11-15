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

use super::*;

/// The regions whose units take a depth in the body's tree (`docparse.BODY`): all that the gold's depth steps chain
/// over (`RANKED_BODY`).
const BODY: [u8; 7] = RANKED_BODY;
/// A heading's or body unit's gap under its parent is one of 1..=GAPS levels.
#[cfg(feature = "encoder")]
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
        let mut group: Vec<usize> = (0..n).collect();
        for (k, &next) in lines.links.iter().enumerate().filter(|&(k, &next)| next != k) {
            let (a, b) = (find(&mut group, k), find(&mut group, next));
            group[a.max(b)] = a.min(b);
        }
        let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
        (0..n).for_each(|k| members.entry(find(&mut group, k)).or_default().push(k));
        let mut regions: Vec<Vec<usize>> = members.into_values().collect();
        regions.sort_by_key(|lines| lines[0]);
        let roles: Vec<u8> = regions.iter().map(|members| {
            let counts = members.iter().fold([0usize; 17], |mut counts, &k| { counts[lines.roles[k] / 2] += 1; counts });
            (0..17).fold(lines.roles[members[0]] / 2, |best, r| if counts[r] > counts[best] { r } else { best }) as u8
        }).collect();
        let read: Vec<usize> = (0..regions.len()).filter(|&r| roles[r] != 0).collect();
        let firsts = read.iter().map(|&r| regions[r][0]).collect();
        Self { lines: regions, roles, read, firsts }
    }

    /// Each region's role its lines' summed type probabilities' likeliest (`type_probs`, 17 a line), not their argmaxes'
    /// plurality (LEGALPDF_DOC_DECODE `vote`); the regions read again (those of a role).
    fn vote(&mut self, type_probs: &[f32]) {
        for (members, role) in self.lines.iter().zip(self.roles.iter_mut()) {
            let mut sums = [0f32; 17];
            members.iter().for_each(|&k| (0..17).for_each(|r| sums[r] += type_probs[k * 17 + r]));
            *role = (0..17).fold(0, |best, r| if sums[r] > sums[best] { r } else { best }) as u8;
        }
        self.read = (0..self.lines.len()).filter(|&r| self.roles[r] != 0).collect();
        self.firsts = self.read.iter().map(|&r| self.lines[r][0]).collect();
    }

    /// A read region's role.
    fn role(&self, i: usize) -> u8 {
        self.roles[self.read[i]]
    }

    /// Each of `n` lines' place among the regions read (`usize::MAX` for a line in none).
    fn of_lines(&self, n: usize) -> Vec<usize> {
        let mut of_line = vec![usize::MAX; n];
        self.read.iter().enumerate().for_each(|(i, &r)| self.lines[r].iter().for_each(|&k| of_line[k] = i));
        of_line
    }
}

fn find(group: &mut [usize], k: usize) -> usize {
    let mut root = k;
    while group[root] != root { root = group[root]; }
    let mut at = k;
    while group[at] != root { at = std::mem::replace(&mut group[at], root); }
    root
}

/// A line's marker, its first reading in the line table: its numbering's family (its count and wrap, 0 for none), value
/// (-1 for none) and prefix length. The trees read it so, and the dump writes it so for the fit.
pub(crate) fn line_marker(table: &LineTable, k: usize) -> (i64, i64, i64) {
    let Some(first) = table.readings[k].first() else { return (0, -1, 0) };
    let (_, count, wrap) = marker_family(first);
    (count.unwrap_or(0) as i64 * 5 + wrap.unwrap_or(0) as i64 + 1, i64::from(first.value), first.prefix.len() as i64)
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
fn margin_links(links: &mut [usize], margin_rows: &HashMap<usize, (usize, Option<usize>)>, views: &[LineView]) {
    let mut opened: HashSet<usize> = margin_rows.iter().flat_map(|(&margin, &(row, twin))| [margin, row].into_iter().chain(twin)).collect();
    opened.extend((0..views.len()).filter(|&k| views[k].count().is_some()));
    links.iter_mut().enumerate().filter(|(k, next)| opened.contains(*next) && !margin_rows.contains_key(k)).for_each(|(k, next)| *next = k);
    margin_rows.iter().for_each(|(&margin, &(row, _))| links[margin] = row);
}

/// A body region whose first line Detect reads going on (its role's opening bit unset) and that Order puts in no
/// earlier unit goes on in the last region before it of its role in its column's language (a paragraph past the
/// page's foot, or read past the other column's lines): the region read last of that role and language, on its page
/// or the page before (as Order pairs regions), its unit the one open in that stream. A region the margin numbers
/// opens.
#[cfg(feature = "encoder")]
fn goes_on_in_column(before: &mut [Option<usize>], regions: &Regions, line_roles: &[usize], views: &[LineView], pages: &[usize]) {
    let mut last: HashMap<(u8, bool), usize> = HashMap::new();
    for (i, &first) in regions.firsts.iter().enumerate() {
        let key = (regions.role(i), views[first].translation());
        if before[i].is_none() && BODY.contains(&key.0) && line_roles[first] % 2 == 0 && views[first].margin().is_none() && views[first].count().is_none() {
            before[i] = last.get(&key).copied().filter(|&earlier| pages[regions.firsts[earlier]] + 1 >= pages[first]);
        }
        last.insert(key, i);
    }
}

/// The paper's Algorithm 1 over `n` nodes: each node in turn the right-most child of the node on the tree's
/// right-most path (the root first) whose parent probability times the node's sibling probability for that
/// node's last child (itself where it has none) is greatest; `resets[i]` empties the tree before node `i`.
/// Each node's level: its parent's (0 for the root) and its gap.
#[cfg(feature = "encoder")]
fn insert_scored(scores: &super::trees::Scores, gaps: &[usize], n: usize, resets: &[bool]) -> Vec<usize> {
    let mut path: Vec<usize> = Vec::new();
    let mut levels = vec![0usize; n];
    for i in 0..n {
        if resets[i] { path.clear(); }
        // A parent's and a sibling's probabilities are each a softmax over the node's row, the same for every candidate:
        // the candidate with the greatest product of the two has the greatest sum of their scores.
        let score = |parent: usize, sibling: usize| scores.score(i, parent).0 + scores.score(i, sibling).1;
        let mut best = (0usize, score(i, path.first().copied().unwrap_or(i)));
        for (at, &node) in path.iter().enumerate() {
            let candidate = score(node, path.get(at + 1).copied().unwrap_or(i));
            if candidate > best.1 { best = (at + 1, candidate); }
        }
        path.truncate(best.0);
        levels[i] = path.last().map_or(0, |&node| levels[node]) + gaps[i].max(1);
        path.push(i);
    }
    levels
}

/// The same decoding over every pair's probabilities (`parent`, `sibling`, n a row), as the oracle reads them.
fn insert(parent: &[f32], sibling: &[f32], gaps: &[usize], n: usize, resets: &[bool]) -> Vec<usize> {
    let mut path: Vec<usize> = Vec::new();
    let mut levels = vec![0usize; n];
    for i in 0..n {
        if resets[i] { path.clear(); }
        let (p, s) = (|j: usize| parent[i * n + j], |j: usize| sibling[i * n + j]);
        // Under the root: no parent (itself), its sibling the root's last child, the path's first node.
        let mut best = (0usize, p(i) * s(path.first().copied().unwrap_or(i)));
        for (at, &node) in path.iter().enumerate() {
            let score = p(node) * s(path.get(at + 1).copied().unwrap_or(i));
            if score > best.1 { best = (at + 1, score); }
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
pub(crate) fn note_number(line: &LineView, text: &str) -> Option<u32> {
    line.label()
        .or_else(|| line.readings().iter().find(|r| r.family.chars().all(|c| !c.is_alphabetic())).map(|r| r.value))
        .or_else(|| crate::leading_figures(text).map(|(figures, _)| figures))
        .or_else(|| text.trim_start().starts_with(['*', '\u{2020}', '\u{2021}', '\u{a7}']).then_some(0))
}

/// The document's moves from the heads' answers: each region's lines its role's; a unit's first line opens it
/// (at its level or depth, counting on its marker where the line is marked, a note on its number); a line of a
/// unit another unit of its stream came before goes on with it by name (its first line); a note answers the
/// marks of its number in the region its link names.
fn moves_of(lines: &Lines, regions: &Regions, units: &Units, views: &[LineView], texts: &[&str], only: Option<&[bool]>) -> Vec<Move> {
    let mut moves: Vec<Move> = lines.starts.iter().map(|&starts| Move { starts, ..Move::plain(0, false) }).collect();
    // Each region's unit, its first line: a region going on in an earlier one's unit is in that unit.
    let mut unit: Vec<usize> = Vec::with_capacity(regions.read.len());
    regions.firsts.iter().enumerate().for_each(|(i, &first)| unit.push(units.before[i].filter(|&earlier| earlier < i).map_or(first, |earlier| unit[earlier])));
    // The open marker frames per context for disambiguating Roman vs. alphabetic readings (e.g. `(i)`, `(v)`, `(x)`).
    let mut doc_id = 0u16;
    let mut stacks: HashMap<u16, Open> = HashMap::new();
    for (k, &i) in regions.of_lines(lines.roles.len()).iter().enumerate() {
        doc_id = doc_id.saturating_add(u16::from(lines.starts[k]));
        if i == usize::MAX { continue; }
        let region = regions.role(i);
        let mv = &mut moves[k];
        mv.region = region;
        // A line of a unit opened before it is in that unit, by its first line.
        if unit[i] != k {
            mv.resumes = Some(unit[i] as u32);
            continue;
        }
        let open = stacks.entry(doc_id.saturating_mul(2) + u16::from(views[k].translation())).or_default();
        open.opens(region);
        // DOC's own pick among several readings: the one that goes on from an open frame (the next number of its family).
        let chosen_reading = (!is_note(region) && lines.marked[k]).then(|| views[k].opening(|| (views[k].readings().len() > 1).then(|| views[k].readings().iter().position(|r| {
            open.frames().iter().rev().any(|f| f.family == r.family && r.value == f.value + 1)
        })).flatten()));
        if let Some(reading) = views[k].reading(chosen_reading) { open.count(reading, region); }
        mv.opens = true;
        if region == HEADING {
            mv.level = units.level[i].clamp(1, 9) as u8;
        } else if BODY.contains(&region) {
            mv.depth = units.level[i].clamp(1, 9) as u8;
        }
        if is_note(region) { mv.note = note_number(&views[k], texts[k]); } else { mv.reading = chosen_reading; }
    }
    for (i, target) in units.note.iter().enumerate() {
        let first = regions.firsts[i];
        let Some(value) = moves[first].opens.then_some(moves[first].note).flatten() else { continue };
        // Of the region's marks of its number, those on the lines `only` names, where it names any (the gold's).
        let marks: Vec<(u32, u32, u32)> = target.iter().flat_map(|&target| &regions.lines[regions.read[target]]).filter(|&&k| only.is_none_or(|lines| lines[k]))
            .flat_map(|&k| views[k].references().iter().filter(move |&&(v, ..)| v == value).map(move |&(_, s, e)| (views[k].place, s, e)))
            .take(4).collect();
        // A note the heads pair with no mark answers none, not the last of its number the text set.
        if marks.is_empty() { moves[first].answers[0] = Some((u32::MAX, 0, 0)); }
        moves[first].answers.iter_mut().zip(marks).for_each(|(slot, mark)| *slot = Some(mark));
    }
    moves
}

/// A constituent document's start between each node and the one before (the first: from the document's top)
/// empties the tree before it.
pub(crate) fn resets(firsts: &[usize], starts: &[bool]) -> Vec<bool> {
    (0..firsts.len()).map(|i| (if i == 0 { 0 } else { firsts[i - 1] + 1 }..=firsts[i]).any(|k| starts[k])).collect()
}

/// Each line's box (its x0, y0, x1, y1), four a line: what the reader and the fit read of where a line stands.
pub(crate) fn boxes(measures: &Measures) -> Vec<f32> {
    let corners = [feature("x0"), feature("y0"), feature("x1"), feature("y1")];
    measures.own.chunks_exact(WIDTH).flat_map(|row| corners.map(|c| row[c])).collect()
}

/// Each region's box, the union of its lines' (`boxes`, four a line; `lines` each line with its region), four a region.
pub(crate) fn region_boxes(boxes: &[f32], lines: impl Iterator<Item = (usize, usize)>, count: usize) -> Vec<f32> {
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
pub(crate) fn sections(reset: &[bool], opening_heading: impl Fn(usize) -> bool) -> (Vec<i64>, Vec<Option<usize>>) {
    let (mut sections, mut enclosing) = (vec![0i64; reset.len()], vec![None; reset.len()]);
    let (mut section, mut last) = (0i64, None);
    for i in 0..reset.len() {
        if reset[i] { (section, last) = (section + 1, None); }
        enclosing[i] = last;
        if opening_heading(i) { (section, last) = (section + 1, Some(i)); }
        sections[i] = section;
    }
    (sections, enclosing)
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
pub(crate) fn oracle(gold: &[[i64; 8]], views: &[LineView], texts: &[&str], pages: &[usize]) -> Vec<Move> {
    let n = gold.len();
    let (lines, unit) = gold_lines(gold, pages);
    let regions = Regions::of(&lines);
    let count = regions.read.len();
    oracle_units(gold, views, texts, n, lines, unit, regions, count)
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
    let named_later: HashMap<usize, usize> = gold.iter().enumerate().filter(|(k, g)| g[1] == 2 && usize::try_from(g[6]).is_ok_and(|p| p > *k)).map(|(k, g)| (g[6] as usize, k)).collect();
    for (k, g) in gold.iter().enumerate().filter(|(_, g)| gold_region(g) > 0) {
        let region = gold_region(g);
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
    (0..n).filter(|&k| unit[k] != usize::MAX).for_each(|k| if let Some(before) = last.insert((unit[k], pages[k]), k) { links[before] = k; });
    let lines = Lines {
        roles: gold.iter().map(|g| gold_region(g).max(0) as usize * 2 + usize::from(g[1] == 1)).collect(),
        marked: gold.iter().map(|g| g[1] == 1 && g[5] == 1).collect(),
        starts: gold.iter().map(|g| g[2] == 1).collect(),
        links,
    };
    (lines, unit)
}

/// The region heads' answers read off the gold over the gold's own regions, and the moves they make.
#[allow(clippy::too_many_arguments)]
fn oracle_units(gold: &[[i64; 8]], views: &[LineView], texts: &[&str], n: usize, lines: Lines, unit: Vec<usize>, regions: Regions, count: usize) -> Vec<Move> {
    // A unit's later regions go on in its earlier ones.
    let mut unit_region: HashMap<usize, usize> = HashMap::new();
    let before: Vec<Option<usize>> = regions.firsts.iter().enumerate().map(|(i, &k)| Some(unit[k]).filter(|&u| u != usize::MAX).and_then(|u| unit_region.insert(u, i))).collect();
    // The trees, from the gold's levels (`docparse.absolute`, `docparse.tree`), read through `insert`.
    let mut level = vec![0usize; count];
    for (column, wanted) in [(4usize, &[HEADING][..]), (7, &BODY[..])] {
        let mut absolute = vec![0usize; n];
        let mut current = 0i64;
        for (k, g) in gold.iter().enumerate() {
            if g[2] == 1 { current = 0; }
            if wanted.contains(&(gold_region(g).max(0) as u8)) && g[1] == 1 {
                if g[column] < 0 { current = 0; continue; }
                current = (current.max(1) + g[column] - 10).clamp(1, 9);
                absolute[k] = current as usize;
            }
        }
        let nodes: Vec<usize> = (0..count).filter(|&i| before[i].is_none() && wanted.contains(&regions.role(i))).collect();
        let h = nodes.len();
        if h == 0 { continue; }
        let (mut parent, mut sibling, mut gaps) = (vec![0f32; h * h], vec![0f32; h * h], vec![1usize; h]);
        let firsts: Vec<usize> = nodes.iter().map(|&i| regions.firsts[i]).collect();
        let reset = resets(&firsts, &lines.starts);
        let (mut stack, mut last_child): (Vec<usize>, HashMap<Option<usize>, usize>) = (Vec::new(), HashMap::new());
        // A node's level: its first line's (one at least).
        let at = |b: usize| absolute[firsts[b]].max(1);
        for (a, &k) in firsts.iter().enumerate() {
            if reset[a] { stack.clear(); last_child.clear(); }
            // A node of unknown level stands one under the open node.
            let mine = if absolute[k] > 0 { absolute[k] } else { stack.last().map_or(1, |&b| at(b) + 1) };
            while stack.last().is_some_and(|&b| at(b) >= mine) { stack.pop(); }
            let p = stack.last().copied();
            parent[a * h + p.unwrap_or(a)] = 1.0;
            sibling[a * h + last_child.get(&p).copied().unwrap_or(a)] = 1.0;
            gaps[a] = mine - p.map_or(0, at);
            last_child.insert(p, a);
            stack.push(a);
        }
        nodes.iter().zip(insert(&parent, &sibling, &gaps, h, &reset)).for_each(|(&i, l)| level[i] = l);
    }
    // A note's region: the gold's paired mark of its number nearest before it, else after it, wherever it stands.
    let region_of = regions.of_lines(n);
    // Each mark is one note's; a parallel report's two notes of a number take the marks of their own languages.
    let mut taken: HashSet<(usize, u32)> = HashSet::new();
    let note = regions.firsts.iter().enumerate().map(|(i, &k)| {
        if !is_note(regions.role(i)) || before[i].is_some() { return None; }
        let value = note_number(&views[k], texts[k])?;
        let paired = |j: &usize| gold[*j][3] == 1 && !taken.contains(&(*j, value)) && region_of[*j] != usize::MAX && views[*j].references().iter().any(|&(v, ..)| v == value);
        let same = |j: &usize| views[*j].translation() == views[k].translation();
        let found = (0..k).rev().filter(paired).find(same).or_else(|| (k + 1..n).filter(paired).find(same))
            .or_else(|| (0..k).rev().find(paired)).or_else(|| (k + 1..n).find(paired));
        if let Some(j) = found { taken.insert((j, value)); }
        found.map(|j| region_of[j])
    }).collect();
    let gold_marks: Vec<bool> = gold.iter().map(|g| g[3] == 1).collect();
    moves_of(&lines, &regions, &Units { before, level, note }, views, texts, Some(&gold_marks))
}

/// The decoding's options (LEGALPDF_DOC_DECODE, comma-separated): `marginal`, each line's opening from the
/// opening's probability summed over the region types and its type from the type's summed over opening and going on,
/// not the 34 classes' argmax; `vote`, each region's role from its lines' summed type probabilities; `sure=X`, the
/// successor's probability past which its link is kept unmatched by the predecessor's (else 0.5); `start=X`, the
/// probability past which a line starts a constituent document (else 0.5), as a logit.
pub(crate) struct Decode { pub(crate) marginal: bool, pub(crate) vote: bool, pub(crate) sure: f32, pub(crate) start: f32 }

pub(crate) fn decode_options() -> &'static Decode {
    static DECODE: std::sync::OnceLock<Decode> = std::sync::OnceLock::new();
    DECODE.get_or_init(|| {
        let named = std::env::var("LEGALPDF_DOC_DECODE").unwrap_or_default();
        let parts: Vec<&str> = named.split(',').map(str::trim).collect();
        let sure = parts.iter().find_map(|p| p.strip_prefix("sure=")).and_then(|v| v.parse().ok()).unwrap_or(0.5);
        let start: f32 = parts.iter().find_map(|p| p.strip_prefix("start=")).and_then(|v| v.parse().ok()).unwrap_or(0.5);
        Decode { marginal: parts.contains(&"marginal"), vote: parts.contains(&"vote"), sure, start: (start / (1.0 - start)).ln() }
    })
}

/// The line heads' answers LEGALPDF_DOC_ORACLE_PARTS names (any of `region`, `opens`, `links`, `starts`,
/// `marked`, comma-separated), read off the gold in place of the model's, on the lines the gold knows: what each head's
/// misses cost the units it decodes. A machine-aligned gold's unmarked body (text, a quotation or a list item) keeps the
/// model's region, which that gold does not decide.
pub(crate) fn oracle_parts() -> &'static HashSet<String> {
    static PARTS: std::sync::OnceLock<HashSet<String>> = std::sync::OnceLock::new();
    PARTS.get_or_init(|| std::env::var("LEGALPDF_DOC_ORACLE_PARTS").unwrap_or_default().split(',')
        .map(|part| part.trim().to_owned()).filter(|part| !part.is_empty()).collect())
}

#[cfg_attr(not(feature = "encoder"), allow(dead_code))]
fn partial_oracle(lines: &mut Lines, gold: &[[i64; 8]], pages: &[usize]) {
    let parts = oracle_parts();
    if parts.is_empty() || gold.len() != lines.roles.len() { return; }
    let (truth, _) = gold_lines(gold, pages);
    for (k, g) in gold.iter().enumerate().filter(|(_, g)| g[0] >= 0) {
        if parts.contains("region") && g[0] != UNMARKED_BODY { lines.roles[k] = truth.roles[k] / 2 * 2 + lines.roles[k] % 2; }
        if parts.contains("opens") { lines.roles[k] = lines.roles[k] / 2 * 2 + truth.roles[k] % 2; }
        if parts.contains("starts") { lines.starts[k] = truth.starts[k]; }
        if parts.contains("marked") { lines.marked[k] = truth.marked[k]; }
        if parts.contains("links") { lines.links[k] = truth.links[k]; }
    }
}

// ------------------------------------------------------------------------------------------------- the model
#[cfg(feature = "encoder")]
pub use model::{Reader, TextModel};
#[cfg(feature = "encoder")]
pub(crate) use model::Reading;

#[cfg(feature = "encoder")]
mod model {
    use super::*;
    use crate::doc::onnx::{init_runtime, session, softmax_rows};
    use ort::session::Session;
    use std::sync::Mutex;

    #[derive(serde::Deserialize)]
    struct Meta {
        width: usize,
        text_dim: usize,
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
            struct Fields { vocabulary: usize, unknown: Option<u32>, normalize: bool }
            let fields: Fields = serde_json::from_str(&std::fs::read_to_string(dir.join("docparse.json")).map_err(|e| format!("docparse.json: {e}"))?)
                .map_err(|e| format!("docparse.json: {e}"))?;
            let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| format!("tokenizer.json: {e}"))?;
            Ok(Self { dir: dir.to_path_buf(), tokenizer, vocabulary: fields.vocabulary, unknown: fields.unknown, normalize: fields.normalize,
                table: std::sync::OnceLock::new() })
        }

        /// Each text's vector as Model2Vec reads one: its tokens' rows of the table averaged (none, nought), scaled to unit
        /// length where the model normalizes; `dim` wide, end to end.
        pub(crate) fn vectors(&self, texts: &[&str], dim: usize) -> std::result::Result<Vec<f32>, String> {
            let (tokens, counts) = self.tokens(texts)?;
            self.vectors_of(&tokens, &counts, dim)
        }

        /// Each line's vector as `vectors` reads one, from its tokens (`counts` of them a line, end to end).
        pub(crate) fn vectors_of(&self, tokens: &[u32], counts: &[usize], dim: usize) -> std::result::Result<Vec<f32>, String> {
            let table = self.table.get_or_init(|| {
                let bytes = std::fs::read(self.dir.join("embeddings.f32")).map_err(|e| format!("embeddings.f32: {e}"))?;
                Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
            }).as_ref().map_err(Clone::clone)?;
            if table.len() != self.vocabulary * dim {
                return Err(format!("embeddings.f32 holds {} values, not {} tokens of {dim}", table.len(), self.vocabulary));
            }
            let mut out = vec![0f32; counts.len() * dim];
            let mut at = 0;
            for (line, &count) in counts.iter().enumerate() {
                let row = &mut out[line * dim..(line + 1) * dim];
                for &token in &tokens[at..at + count] {
                    row.iter_mut().zip(&table[token as usize * dim..(token as usize + 1) * dim]).for_each(|(x, &v)| *x += v);
                }
                at += count;
                let mean = count.max(1) as f32;
                row.iter_mut().for_each(|x| *x /= mean);
                if self.normalize {
                    let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt() + 1e-32;
                    row.iter_mut().for_each(|x| *x /= norm);
                }
            }
            Ok(out)
        }

        /// Each text's tokens as Model2Vec reads a text (no special tokens; the unknown one and any past the vocabulary
        /// left out; an empty text read as a space), end to end, and each text's count of them.
        pub fn tokens(&self, texts: &[&str]) -> std::result::Result<(Vec<u32>, Vec<usize>), String> {
            let (mut tokens, mut counts) = (Vec::new(), Vec::with_capacity(texts.len()));
            for text in texts {
                let text = if text.trim().is_empty() { " " } else { text };
                let encoding = self.tokenizer.encode_fast(text, false).map_err(|e| format!("tokenizer: {e}"))?;
                let before = tokens.len();
                tokens.extend(encoding.get_ids().iter().copied().filter(|&id| Some(id) != self.unknown && (id as usize) < self.vocabulary));
                counts.push(tokens.len() - before);
            }
            Ok((tokens, counts))
        }
    }

    /// One set of the stages' sessions: a parse holds one for its document, parses at once each their own (made
    /// as they are wanted, kept for the next), none waiting on another's.
    struct Sessions {
        context: Session,
        detect: Session,
        stylesheet: Session,
        spatial: Session,
        lines: Session,
        order: Session,
        headings: Session,
        bodies: Session,
    }

    impl Sessions {
        fn load(dir: &Path, threads: usize) -> std::result::Result<Self, String> {
            let load = |name: &str| { let path = dir.join(format!("{name}.onnx")); session(&path, path.display(), threads) };
            Ok(Self { context: load("context")?, detect: load("detect")?, stylesheet: load("stylesheet")?, spatial: load("spatial")?, lines: load("lines")?, order: load("order")?, headings: load("headings")?, bodies: load("bodies")? })
        }
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
        row.iter().enumerate().fold((0, f32::NEG_INFINITY), |best, (i, &x)| if x > best.1 { (i, x) } else { best }).0
    }

    /// A tree head's scores over the nodes before each node and itself (none) as probabilities: each row's
    /// later nodes left out.
    fn earlier(values: &[f32], n: usize) -> Vec<f32> {
        let masked: Vec<f32> = values.iter().enumerate().map(|(at, &x)| if at % n > at / n { f32::NEG_INFINITY } else { x }).collect();
        softmax_rows(&masked, n)
    }

    /// A document as DOC reads it: its inputs (`inputs::Inputs`), its line table and line views, its lines' texts, and
    /// the gold's rows where the partial oracle reads them.
    pub(crate) struct Reading<'a> {
        pub(crate) inputs: &'a super::inputs::Inputs,
        pub(crate) table: &'a LineTable,
        pub(crate) views: &'a [LineView<'a>],
        pub(crate) texts: &'a [&'a str],
        pub(crate) gold: Option<&'a [[i64; 8]]>,
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

    /// A document's pages as a paged graph reads them (`paged`): each page's lines, each line's float inputs (each
    /// `width` a line, in the graph's order) and its outline anchor where the graph takes one.
    struct Pages<'a> {
        runs: &'a [(usize, usize)],
        floats: Vec<(&'a [f32], usize)>,
        anchors: Option<&'a [i64]>,
    }

    /// At most this many places (pages times their longest) in one run of a paged graph.
    const PLACES: usize = 1 << 18;

    /// A graph that reads each page's lines alone (Detect, its heads) run over every page of `docs`: the pages sorted by
    /// length and run in groups (each page padded to its group's longest, `valid` its real lines, at most PLACES places
    /// a group), `names` its float inputs, then `anchors` where the documents give them, then `valid`. Each document's
    /// pages' outputs, in its pages' order: each output (`outputs`: its name, its width a line, and whether it is
    /// pairwise, a line against its page's lines) of its real lines only.
    #[allow(clippy::type_complexity)]
    fn paged(session: &mut Session, docs: &[Pages], names: &[&str], outputs: &[(&str, usize, bool)])
        -> std::result::Result<Vec<Vec<Vec<Vec<f32>>>>, String> {
        let mut pages: Vec<(usize, usize, usize, usize)> = docs.iter().enumerate()
            .flat_map(|(at, doc)| doc.runs.iter().enumerate().map(move |(p, &(start, end))| (at, p, start, end))).collect();
        pages.sort_by_key(|&(_, _, start, end)| end - start);
        let mut out: Vec<Vec<Vec<Vec<f32>>>> = docs.iter().map(|doc| vec![Vec::new(); doc.runs.len()]).collect();
        let widths: Vec<usize> = docs.first().map(|doc| doc.floats.iter().map(|&(_, w)| w).collect()).unwrap_or_default();
        let anchored = docs.iter().any(|doc| doc.anchors.is_some());
        let mut first = 0;
        while first < pages.len() {
            // The group: as many pages of like length as fit PLACES, the longest last.
            let mut last = first + 1;
            while last < pages.len() && (last + 1 - first) * (pages[last].3 - pages[last].2) <= PLACES {
                last += 1;
            }
            let group = &pages[first..last];
            let count = group.len();
            let longest = group.iter().map(|&(_, _, start, end)| end - start).max().unwrap_or(1).max(1);
            let mut floats: Vec<Vec<f32>> = widths.iter().map(|&w| vec![0f32; count * longest * w]).collect();
            let mut anchors = vec![0i64; count * longest];
            let mut valid = vec![false; count * longest];
            for (g, &(at, _, start, end)) in group.iter().enumerate() {
                let l = end - start;
                for (c, &(values, w)) in docs[at].floats.iter().enumerate() {
                    floats[c][g * longest * w..(g * longest + l) * w].copy_from_slice(&values[start * w..end * w]);
                }
                if let Some(values) = docs[at].anchors {
                    anchors[g * longest..g * longest + l].copy_from_slice(&values[start..end]);
                }
                valid[g * longest..g * longest + l].iter_mut().for_each(|v| *v = true);
            }
            let mut feeds: Vec<(String, ort::session::SessionInputValue<'_>)> = Vec::new();
            for ((name, values), &w) in names.iter().zip(&floats).zip(&widths) {
                feeds.push(((*name).to_owned(), tensor!([count, longest, w], &values[..]).into()));
            }
            if anchored {
                feeds.push(("anchors".to_owned(), tensor!([count, longest], &anchors[..]).into()));
            }
            feeds.push(("valid".to_owned(), tensor!([count, longest], &valid[..]).into()));
            let results = session.run(feeds).map_err(|e| format!("{}: {e}", names.join(", ")))?;
            let mut read: Vec<Vec<f32>> = Vec::new();
            for &(name, _, _) in outputs {
                read.push(output!(results, name).to_vec());
            }
            for (g, &(at, p, start, end)) in group.iter().enumerate() {
                let l = end - start;
                out[at][p] = outputs.iter().zip(&read).map(|(&(_, w, pairwise), all)| if pairwise {
                    (0..l).flat_map(|i| all[(g * longest + i) * longest..(g * longest + i) * longest + l].iter().copied()).collect()
                } else {
                    all[g * longest * w..(g * longest + l) * w].to_vec()
                }).collect();
            }
            first = last;
        }
        Ok(out)
    }

    impl Reader {
        /// The stages under `dir`, their ONNX Runtime at `runtime` (else the one LEGALPDF_ONNX_RUNTIME or
        /// LEGALPDF_ONNXRUNTIME names), and its text model; each session run on `threads` threads.
        pub fn load(dir: &Path, runtime: Option<&Path>, times: bool, threads: usize) -> std::result::Result<Self, String> {
            let read = |name: &str| std::fs::read_to_string(dir.join(name)).map_err(|e| format!("{name}: {e}"));
            let meta: Meta = serde_json::from_str(&read("docparse.json")?).map_err(|e| format!("docparse.json: {e}"))?;
            if meta.width != WIDTH || meta.classes != CLASSES {
                return Err(format!("docparse.json reads {} features and {} classes; the parser has {WIDTH} and {CLASSES}", meta.width, meta.classes));
            }
            init_runtime(runtime, "the DOC reader needs LEGALPDF_ONNX_RUNTIME")?;
            let text = TextModel::load(&meta.text)?;
            let first = Sessions::load(dir, threads)?;
            Ok(Self { dir: dir.to_path_buf(), threads, sessions: Mutex::new(vec![first]), times, text, meta })
        }

        /// The text model the reader reads the lines' texts with.
        pub(crate) fn text(&self) -> &TextModel {
            &self.text
        }

        /// The document's moves, a line each in reading order, read off `inputs` (`texts` each line's): `read_many` of it
        /// alone. `gold`, where given with LEGALPDF_DOC_ORACLE_PARTS, takes the line heads' answers it names from the gold
        /// (`partial_oracle`): what each head's misses cost the units, the rest read as the model reads it.
        pub(crate) fn read(&self, inputs: &super::inputs::Inputs, table: &LineTable, views: &[LineView], texts: &[&str], gold: Option<&[[i64; 8]]>) -> std::result::Result<Vec<Move>, String> {
            self.read_many(&[Reading { inputs, table, views, texts, gold }]).pop().unwrap_or_else(|| Ok(Vec::new()))
        }

        /// Each document's moves: its lines read in a document at a time (the context, the texts, the stylesheet), Detect
        /// and its heads over every page of all of them at once (`paged`: a page reads only its own lines, so beside
        /// another document's it reads as alone), Order and the trees a document at a time; read with a set of sessions
        /// of the reader's own, given back after.
        pub(crate) fn read_many(&self, docs: &[Reading]) -> Vec<std::result::Result<Vec<Move>, String>> {
            let mut sessions = match self.sessions.lock().ok().and_then(|mut pool| pool.pop()).map_or_else(|| Sessions::load(&self.dir, self.threads), Ok) {
                Ok(sessions) => sessions,
                Err(error) => return docs.iter().map(|_| Err(error.clone())).collect(),
            };
            let moves = self.read_many_with(&mut sessions, docs);
            if let Ok(mut pool) = self.sessions.lock() { pool.push(sessions); }
            moves
        }

        fn read_many_with(&self, sessions: &mut Sessions, docs: &[Reading]) -> Vec<std::result::Result<Vec<Move>, String>> {
            let (d, m) = (self.meta.d, &self.meta);
            if ROW != m.row_width {
                return docs.iter().map(|_| Err(format!("the stylesheet's rows are {ROW} wide; docparse.json says {}", m.row_width))).collect();
            }
            let since = std::time::Instant::now();
            let begun: Vec<std::result::Result<Begun, String>> = docs.iter().map(|doc| self.begin(doc)).collect();
            // Each stage's time over the group, for the reader's speed (`times`): the texts, the context, Detect, the
            // stylesheet and the heads.
            let mut laps = [since.elapsed().as_secs_f64(), 0.0, 0.0, 0.0, 0.0];
            let mut at = std::time::Instant::now();
            let mut lap = |k: usize| { laps[k] = at.elapsed().as_secs_f64(); at = std::time::Instant::now(); };
            // The documents read: those begun with lines, laid end to end as a pack's are.
            let ready: Vec<usize> = (0..docs.len()).filter(|&at| begun[at].as_ref().is_ok_and(|b| b.n > 0)).collect();
            let stages = (|| -> std::result::Result<(Vec<Vec<f32>>, Vec<Vec<Vec<Vec<f32>>>>), String> {
                let b = |at: usize| begun[at].as_ref().expect("ready");
                let total: usize = ready.iter().map(|&at| b(at).n).sum();
                // The context graph over every line of them, each line its document's.
                let own: Vec<i8> = ready.iter().flat_map(|&at| docs[at].inputs.own.iter().copied()).collect();
                let line_doc: Vec<i64> = ready.iter().enumerate().flat_map(|(k, &at)| std::iter::repeat(k as i64).take(b(at).n)).collect();
                let page_of: Vec<i64> = ready.iter().flat_map(|&at| docs[at].inputs.pages[..b(at).n].iter().map(|&p| p as i64)).collect();
                let shape: Vec<i8> = ready.iter().flat_map(|&at| docs[at].inputs.shape.iter().copied()).collect();
                let outputs = sessions.context.run(ort::inputs!["own" => tensor!([total, WIDTH], &own[..]), "docs" => tensor!([total], &line_doc[..]),
                    "pages" => tensor!([total], &page_of[..]), "shape" => tensor!([ready.len(), SHAPE], &shape[..])]).map_err(|e| format!("context: {e}"))?;
                let features = output!(outputs, "features").to_vec();
                lap(1);
                let width = features.len() / total.max(1);
                let mut first = 0;
                let features: Vec<&[f32]> = ready.iter().map(|&at| { let n = b(at).n; first += n; &features[(first - n) * width..first * width] }).collect();
                // Detect over every page of them.
                let detected = paged(&mut sessions.detect, &ready.iter().zip(&features).map(|(&at, f)| Pages {
                    runs: &b(at).runs, floats: vec![(f, width), (&b(at).text[..], m.text_dim), (&docs[at].inputs.boxes[..], 4), (&b(at).t0[..], T0)],
                    anchors: Some(&b(at).anchors[..]),
                }).collect::<Vec<_>>(), &["features", "text", "boxes", "t0"], &[("states", d, false)])?;
                lap(2);
                // The stylesheet over every line of them: each document's templates numbered on past the last's, read together.
                let mut states = vec![0f32; total * d];
                let mut templates: Vec<i64> = Vec::with_capacity(total);
                let mut rows: Vec<f32> = Vec::new();
                let mut layout: Vec<(usize, usize)> = Vec::new();
                let (mut line0, mut template0) = (0usize, 0usize);
                for (k, &at) in ready.iter().enumerate() {
                    for (&(start, end), outs) in b(at).runs.iter().zip(&detected[k]) {
                        states[(line0 + start) * d..(line0 + end) * d].copy_from_slice(&outs[0]);
                    }
                    let count = docs[at].inputs.rows.len() / ROW;
                    templates.extend(docs[at].inputs.templates.iter().map(|&t| t + template0 as i64));
                    rows.extend_from_slice(&docs[at].inputs.rows);
                    layout.push((template0, count));
                    line0 += b(at).n;
                    template0 += count;
                }
                let widest = layout.iter().map(|&(_, count)| count).max().unwrap_or(1).max(1);
                let mut place = vec![0i64; ready.len() * widest];
                let mut valid = vec![false; ready.len() * widest];
                let mut flat: Vec<i64> = Vec::with_capacity(template0);
                for (k, &(first, count)) in layout.iter().enumerate() {
                    for t in 0..count {
                        place[k * widest + t] = (first + t) as i64;
                        valid[k * widest + t] = true;
                        flat.push((k * widest + t) as i64);
                    }
                }
                let outputs = sessions.stylesheet.run(ort::inputs!["states" => tensor!([total, d], &states[..]), "templates" => tensor!([total], &templates[..]),
                    "rows" => tensor!([template0, ROW], &rows[..]), "at" => tensor!([ready.len(), widest], &place[..]),
                    "valid" => tensor!([ready.len(), widest], &valid[..]), "flat" => tensor!([template0], &flat[..])]).map_err(|e| format!("stylesheet: {e}"))?;
                let all = output!(outputs, "styled");
                lap(3);
                let mut first = 0;
                let styled: Vec<Vec<f32>> = ready.iter().map(|&at| { let n = b(at).n; first += n; all[(first - n) * d..first * d].to_vec() }).collect();
                // Detect's heads over every line of them, each line against its candidates: the lines within the window on its
                // page, in reading order.
                let window = m.line_window.max(1);
                let (mut pair_i, mut pair_j) = (Vec::new(), Vec::new());
                let mut line0 = 0;
                for &at in &ready {
                    let (n, pages) = (b(at).n, &docs[at].inputs.pages);
                    for i in 0..n {
                        for j in i.saturating_sub(window)..=(i + window).min(n - 1) {
                            if pages[j] == pages[i] {
                                pair_i.push((line0 + i) as i64);
                                pair_j.push((line0 + j) as i64);
                            }
                        }
                    }
                    line0 += n;
                }
                let boxes: Vec<f32> = ready.iter().flat_map(|&at| docs[at].inputs.boxes[..b(at).n * 4].iter().copied()).collect();
                // Every two lines of each page: their query-key scores, one batched product a head over the pages.
                let mut heads = paged(&mut sessions.lines, &ready.iter().zip(&styled).map(|(&at, states)| Pages {
                    runs: &b(at).runs, floats: vec![(&states[..], d)], anchors: None,
                }).collect::<Vec<_>>(), &["states"], &[("role", CLASSES, false), ("flags", 2, false), ("successor", 0, true), ("predecessor", 0, true)])?;
                // The far bias on every pair, and in its place, within the window, the pair's spatial term.
                let outputs = sessions.spatial.run(ort::inputs!["boxes" => tensor!([total, 4], &boxes[..]),
                    "pair_i" => tensor!([pair_i.len()], &pair_i[..]), "pair_j" => tensor!([pair_j.len()], &pair_j[..])]).map_err(|e| format!("spatial: {e}"))?;
                let (near_successor, near_predecessor) = (output!(outputs, "successor"), output!(outputs, "predecessor"));
                let mut at_pair = 0;
                let mut line0 = 0;
                for (k, &at) in ready.iter().enumerate() {
                    for (page, &(start, end)) in heads[k].iter_mut().zip(&b(at).runs) {
                        let l = end - start;
                        page[2].iter_mut().for_each(|x| *x += m.line_far[0]);
                        page[3].iter_mut().for_each(|x| *x += m.line_far[1]);
                        while at_pair < pair_i.len() && (pair_i[at_pair] as usize) < line0 + end {
                            let (i, j) = (pair_i[at_pair] as usize - line0 - start, pair_j[at_pair] as usize - line0 - start);
                            page[2][i * l + j] += near_successor[at_pair] - m.line_far[0];
                            page[3][i * l + j] += near_predecessor[at_pair] - m.line_far[1];
                            at_pair += 1;
                        }
                    }
                    line0 += b(at).n;
                }
                lap(4);
                Ok((styled, heads))
            })();
            let embedded = since.elapsed().as_secs_f64();
            if self.times {
                let lines: usize = begun.iter().filter_map(|b| b.as_ref().ok()).map(|b| b.n).sum();
                eprintln!("group times: {} documents {lines} lines; texts {:.3}s context {:.3}s detect {:.3}s stylesheet {:.3}s heads {:.3}s",
                    docs.len(), laps[0], laps[1], laps[2], laps[3], laps[4]);
            }
            // Each document's answer: its failure where a stage failed it, else the rest of its reading.
            let mut read = match stages {
                Ok((styled, heads)) => styled.into_iter().zip(heads).map(Ok).collect::<Vec<_>>().into_iter(),
                Err(error) => ready.iter().map(|_| Err(error.clone())).collect::<Vec<_>>().into_iter(),
            };
            (0..docs.len()).map(|at| {
                let b = begun[at].as_ref().map_err(Clone::clone)?;
                if b.n == 0 { return Ok(Vec::new()); }
                let (states, heads) = read.next().ok_or("a document was not read")??;
                self.rest(sessions, &docs[at], b, states, &heads, embedded)
            }).collect()
        }

        /// A document's lines read in, as Detect reads them: its texts' vectors, its pages' lines, its anchors and t0.
        fn begin(&self, doc: &Reading) -> std::result::Result<Begun, String> {
            let (inputs, n) = (doc.inputs, doc.views.len());
            let runs: Vec<(usize, usize)> = inputs.pages[..n].chunk_by(|a, b| a == b).scan(0, |end, run| { *end += run.len(); Some((*end - run.len(), *end)) }).collect();
            let text = if n == 0 { Vec::new() } else {
                match &inputs.tokens {
                    Some((tokens, counts)) => self.text.vectors_of(tokens, counts, self.meta.text_dim)?,
                    None => self.text.vectors(&doc.texts[..n], self.meta.text_dim)?,
                }
            };
            Ok(Begun { n, text, anchors: inputs.anchors.iter().map(|&a| i64::from(a)).collect(), t0: inputs.t0.iter().map(|&v| f32::from(v)).collect(), runs })
        }

        /// A document's reading from Detect's heads on (`heads`, each page's role scores, flags, and successor and
        /// predecessor scores, its lines against its lines): its lines' roles, flags and links, its regions, Order over
        /// them and the trees.
        fn rest(&self, sessions: &mut Sessions, doc: &Reading, begun: &Begun, states: Vec<f32>, heads: &[Vec<Vec<f32>>], embedded: f64) -> std::result::Result<Vec<Move>, String> {
            let (inputs, table, views, texts, gold) = (doc.inputs, doc.table, doc.views, doc.texts, doc.gold);
            let (n, runs) = (begun.n, &begun.runs);
            let d = self.meta.d;
            let pages = &inputs.pages;
            let boxes = &inputs.boxes;
            let (anchors, templates) = (&begun.anchors, &inputs.templates);
            let mut stages = [embedded, 0.0, 0.0, 0.0, 0.0];
            let mut lap = |stage: usize, since: &mut std::time::Instant| { stages[stage] += since.elapsed().as_secs_f64(); *since = std::time::Instant::now(); };
            let mut since = std::time::Instant::now();
            let mut lines = Lines { roles: vec![0; n], marked: vec![false; n], starts: vec![false; n], links: (0..n).collect() };
            let decode = decode_options();
            let mut type_probs = vec![0f32; if decode.vote { n * 17 } else { 0 }];
            for (&(start, end), page) in runs.iter().zip(heads) {
                let l = end - start;
                let (role, flag) = (&page[0], &page[1]);
                let (successor, predecessor) = (softmax_rows(&page[2], l), &page[3]);
                for i in 0..l {
                    let scores = &role[i * CLASSES..(i + 1) * CLASSES];
                    lines.roles[start + i] = if decode.marginal || decode.vote {
                        // Each region type's probability (opening and going on summed) and the opening's (over the types).
                        let probs = softmax_rows(scores, CLASSES);
                        let types: Vec<f32> = (0..17).map(|r| probs[r * 2] + probs[r * 2 + 1]).collect();
                        if decode.vote { type_probs[(start + i) * 17..(start + i + 1) * 17].copy_from_slice(&types); }
                        if decode.marginal {
                            let opens: f32 = (0..17).map(|r| probs[r * 2 + 1]).sum();
                            argmax(&types) * 2 + usize::from(opens > 0.5)
                        } else { argmax(scores) }
                    } else { argmax(scores) };
                    (lines.marked[start + i], lines.starts[start + i]) = (flag[i * 2] > 0.0, flag[i * 2 + 1] > decode.start);
                }
                // A link both heads choose, or the successor's sure one; none into a line whose role opens a unit.
                for i in 0..l {
                    let next = argmax(&successor[i * l..(i + 1) * l]);
                    let back = argmax(&predecessor[next * l..(next + 1) * l]);
                    if next != i && (back == i || successor[i * l + next] > decode.sure) && lines.roles[start + next] % 2 == 0 { lines.links[start + i] = start + next; }
                }
            }
            lap(1, &mut since);
            margin_links(&mut lines.links, &table.margin_rows, views);
            // A line t0 reads opening a numbered paragraph on a number it prints opens on that marker.
            lines.marked.iter_mut().enumerate().for_each(|(k, marked)| *marked |= views[k].count().is_some_and(|number| views[k].readings().iter().any(|r| r.value == number)));
            if let Some(gold) = gold { partial_oracle(&mut lines, gold, pages); }
            let mut regions = Regions::of(&lines);
            if decode.vote { regions.vote(&type_probs); }
            let (count, firsts, role) = (regions.read.len(), &regions.firsts, |i: usize| regions.role(i));
            if count == 0 {
                return Ok(moves_of(&lines, &regions, &Units { before: Vec::new(), level: Vec::new(), note: Vec::new() }, views, texts, None));
            }
            // Order over the regions read.
            let of_line = regions.of_lines(n);
            let in_read = || (0..n).filter(|&k| of_line[k] != usize::MAX);
            let line_states: Vec<f32> = in_read().flat_map(|k| states[k * d..(k + 1) * d].iter().copied()).collect();
            let line_region: Vec<i64> = in_read().map(|k| of_line[k] as i64).collect();
            // The regions' places for the pair terms (`docparse.pairs_of`): each one's lines' union box and its first
            // line's page; the regions within a page of each other; whether a note's number is a mark the other
            // region sets.
            let mut region_boxes = region_boxes(&boxes, in_read().map(|k| (k, of_line[k])), count);
            let mut marks: Vec<HashSet<u32>> = vec![HashSet::new(); count];
            for k in in_read() {
                let i = of_line[k];
                if REFERRING.contains(&role(i)) { marks[i].extend(views[k].references().iter().map(|&(v, ..)| v)); }
            }
            let mut region_pages: Vec<i64> = firsts.iter().map(|&k| pages[k] as i64).collect();
            let number: Vec<Option<u32>> = firsts.iter().enumerate().map(|(i, &k)| is_note(role(i)).then(|| note_number(&views[k], texts[k])).flatten()).collect();
            let mut by_page: HashMap<i64, Vec<usize>> = HashMap::new();
            region_pages.iter().enumerate().for_each(|(i, &page)| by_page.entry(page).or_default().push(i));
            let mut setting: HashMap<u32, Vec<usize>> = HashMap::new();
            marks.iter().enumerate().for_each(|(j, set)| set.iter().for_each(|&value| setting.entry(value).or_default().push(j)));
            let (mut pair_i, mut pair_j, mut pair_match) = (Vec::new(), Vec::new(), Vec::new());
            let mut pair = |i: usize, j: usize, matched: bool| {
                pair_i.push(i as i64);
                pair_j.push(j as i64);
                pair_match.push(matched);
            };
            for (i, &page) in region_pages.iter().enumerate() {
                [page - 1, page, page + 1].iter().flat_map(|near| by_page.get(near).into_iter().flatten()).for_each(|&j| pair(i, j, j != i && number[i].is_some_and(|v| marks[j].contains(&v))));
                // A note's candidates are every region setting its number, however far (endnotes).
                number[i].and_then(|v| setting.get(&v)).into_iter().flatten().filter(|&&j| j != i && (region_pages[j] - page).abs() > 1).for_each(|&j| pair(i, j, true));
            }
            // The regions padded to whole blocks of the graph's local attention (`docparse.BLOCK`), the padding marked.
            let padded = count.div_ceil(BLOCK) * BLOCK;
            let kinds: Vec<i64> = (0..padded).map(|i| if i < count { i64::from(role(i)) } else { 0 }).collect();
            let region_valid: Vec<bool> = (0..padded).map(|i| i < count).collect();
            region_boxes.resize(padded * 4, 0.0);
            region_pages.resize(padded, -1);
            // The pairs whose term the graph reads: those within the window, or a note's against a mark of its number.
            let near: Vec<usize> = (0..pair_i.len()).filter(|&p| pair_match[p] || pair_i[p].abs_diff(pair_j[p]) <= self.meta.order_window as u64).collect();
            let (near_i, near_j, near_match): (Vec<i64>, Vec<i64>, Vec<bool>) =
                (near.iter().map(|&p| pair_i[p]).collect(), near.iter().map(|&p| pair_j[p]).collect(), near.iter().map(|&p| pair_match[p]).collect());
            let (region_states, goes_on, note_of) = {
                let outputs = sessions.order.run(ort::inputs!["line_states" => tensor!([line_region.len(), d], &line_states[..]),
                    "line_region" => tensor!([line_region.len()], &line_region[..]), "region_roles" => tensor!([padded], &kinds[..]),
                    "region_valid" => tensor!([padded], &region_valid[..]), "region_boxes" => tensor!([padded, 4], &region_boxes[..]),
                    "region_pages" => tensor!([padded], &region_pages[..]), "pair_i" => tensor!([near_i.len()], &near_i[..]),
                    "pair_j" => tensor!([near_j.len()], &near_j[..]), "pair_match" => tensor!([near_match.len()], &near_match[..])]).map_err(|e| format!("order: {e}"))?;
                let take = |name: &str| outputs[name].try_extract_tensor::<f32>().map(|(_, v)| v.to_vec()).map_err(|e| e.to_string());
                // A head's scores a pair: its scaled query of the pair's region against its key of the other, and the
                // pair's term (the graph's within the window, the head's far bias beyond).
                let head = |name: &str| -> std::result::Result<Vec<f32>, String> {
                    let (q, k, near_r) = (take(&format!("{name}_q"))?, take(&format!("{name}_k"))?, take(&format!("{name}_r"))?);
                    let mut r = vec![self.meta.order_far.get(name).copied().unwrap_or(0.0); pair_i.len()];
                    near.iter().zip(&near_r).for_each(|(&p, &term)| r[p] = term);
                    Ok((0..pair_i.len()).map(|p| {
                        let (i, j) = (pair_i[p] as usize, pair_j[p] as usize);
                        q[i * d..(i + 1) * d].iter().zip(&k[j * d..(j + 1) * d]).map(|(a, b)| a * b).sum::<f32>() + r[p]
                    }).collect())
                };
                (take("regions")?, head("goes_on")?, head("note_of")?)
            };
            // Each region's pairs (contiguous: built region by region), and each head's softmax over them.
            let mut rows: Vec<(usize, usize)> = vec![(0, 0); count];
            for (p, &i) in pair_i.iter().enumerate() {
                let row = &mut rows[i as usize];
                if row.1 == 0 { row.0 = p; }
                row.1 = p + 1;
            }
            let row_softmax = |logits: &[f32]| -> Vec<f32> {
                let mut probs = vec![0f32; logits.len()];
                rows.iter().filter(|&&(a, z)| a < z).for_each(|&(a, z)| probs[a..z].copy_from_slice(&softmax_rows(&logits[a..z], z - a)));
                probs
            };
            let (goes_on, note_of) = (row_softmax(&goes_on), row_softmax(&note_of));
            // A region's likeliest other region by a head, with its probability (itself, where it has no pairs).
            let best = |probs: &[f32], i: usize, open: &dyn Fn(usize) -> bool| -> Option<(usize, f32)> {
                let (a, z) = rows[i];
                (a..z).filter(|&p| open(pair_j[p] as usize)).max_by(|&x, &y| probs[x].total_cmp(&probs[y])).map(|p| (pair_j[p] as usize, probs[p]))
            };
            // Units: a region goes on in the later region its link holds likeliest, each region taken once, the
            // surest links first.
            let mut links: Vec<(usize, usize, f32)> = (0..count).filter_map(|i| best(&goes_on, i, &|_| true).filter(|&(j, _)| j > i).map(|(j, p)| (i, j, p))).collect();
            links.sort_by(|a, b| b.2.total_cmp(&a.2));
            let (mut before, mut after) = (vec![None; count], vec![false; count]);
            for (i, j, _) in links {
                // A paragraph the margin numbers, or t0 reads opening, or in a different stream, opens: it goes on in no earlier one.
                let opens = views[firsts[j]].margin().is_some() || views[firsts[j]].count().is_some();
                let same_stream = stream_of(role(i)) == stream_of(role(j)) && (role(i) == HEADING) == (role(j) == HEADING) && !(firsts[i] + 1..=firsts[j]).any(|k| lines.starts[k]);
                if before[j].is_none() && !after[i] && !opens && same_stream { (before[j], after[i]) = (Some(i), true); }
            }
            goes_on_in_column(&mut before, &regions, &lines.roles, views, pages);
            lap(2, &mut since);
            // The trees over the regions that open a unit, headings and the body's apart.
            let (sections, enclosing_heading) = sections(&resets(firsts, &lines.starts), |i| before[i].is_none() && role(i) == HEADING);
            let line_family = |k: usize| inputs.markers[k];
            let tree = |session: &mut Session, tables: &super::trees::Tables, nodes: &[usize]| -> std::result::Result<Vec<usize>, String> {
                let h = nodes.len();
                if h == 0 { return Ok(Vec::new()); }
                // Each node's measure: of its region (`i`) and its first line (`k`).
                let node = |value: &dyn Fn(usize, usize) -> i64| -> Vec<i64> { nodes.iter().map(|&i| value(i, firsts[i])).collect() };
                let node_f32 = |of: &dyn Fn(usize) -> f32| -> Vec<f32> { nodes.iter().map(|&i| of(firsts[i])).collect() };
                let x: Vec<f32> = nodes.iter().flat_map(|&i| region_states[i * d..(i + 1) * d].iter().copied()).collect();
                let mut h_states = vec![0f32; h * d];
                nodes.iter().enumerate().filter_map(|(at, &i)| Some((at, enclosing_heading[i]?)))
                    .for_each(|(at, hd)| h_states[at * d..(at + 1) * d].copy_from_slice(&region_states[hd * d..(hd + 1) * d]));
                let (node_templates, node_anchors) = (node(&|_, k| templates[k]), node(&|_, k| anchors[k]));
                let (node_fam, node_val, node_pref) = (node(&|_, k| line_family(k).0), node(&|_, k| line_family(k).1), node(&|_, k| line_family(k).2));
                let (node_sec, node_x0, node_size) = (node(&|i, _| sections[i]), node_f32(&|k| boxes[k * 4]), node_f32(&|k| inputs.size[k]));
                let (node_looks, node_pages, node_roles) = (node(&|_, k| inputs.looks[k]), node(&|_, k| pages[k] as i64), node(&|i, _| i64::from(role(i))));
                let outputs = session.run(ort::inputs!["states" => tensor!([h, d], &x[..]), "heading_states" => tensor!([h, d], &h_states[..]),
                    "family" => tensor!([h], &node_fam[..]), "value" => tensor!([h], &node_val[..]), "prefix" => tensor!([h], &node_pref[..]),
                    "x0" => tensor!([h], &node_x0[..]), "size" => tensor!([h], &node_size[..]), "roles" => tensor!([h], &node_roles[..])])
                    .map_err(|e| format!("construct: {e}"))?;
                let vectors = [output!(outputs, "parent_q"), output!(outputs, "parent_k"), output!(outputs, "sibling_q"), output!(outputs, "sibling_k")];
                let gaps: Vec<usize> = output!(outputs, "gap").chunks_exact(GAPS).map(|row| argmax(row) + 1).collect();
                let scores = super::trees::Scores::new(super::trees::Nodes {
                    templates: &node_templates, anchors: &node_anchors, family: &node_fam, value: &node_val, prefix: &node_pref, section: &node_sec,
                    x0: &node_x0, size: &node_size, looks: &node_looks, pages: &node_pages, roles: &node_roles,
                }, tables, vectors, d);
                let node_firsts: Vec<usize> = nodes.iter().map(|&i| firsts[i]).collect();
                Ok(insert_scored(&scores, &gaps, h, &resets(&node_firsts, &lines.starts)))
            };
            let opening = |i: usize| before[i].is_none();
            let headings: Vec<usize> = (0..count).filter(|&i| opening(i) && role(i) == HEADING).collect();
            let bodies: Vec<usize> = (0..count).filter(|&i| opening(i) && BODY.contains(&role(i))).collect();
            let mut level = vec![0usize; count];
            for (nodes, session, name) in [(&headings, &mut sessions.headings, "headings"), (&bodies, &mut sessions.bodies, "bodies")] {
                let tables = self.meta.trees.get(name).ok_or_else(|| format!("docparse.json has no {name} tables"))?;
                nodes.iter().zip(tree(session, tables, nodes)?).for_each(|(&i, l)| level[i] = l);
            }
            // A note's region: the one holding a mark of its number its link names, where not itself, surest notes first
            // so two notes of a number in parallel or multi-part reports do not take the same region's mark.
            let mut note = vec![None; count];
            let mut note_candidates: Vec<(usize, f32)> = (0..count).filter(|&i| before[i].is_none() && is_note(role(i)) && number[i].is_some()).map(|i| {
                let valid = |j: usize| j == i || number[i].is_some_and(|v| marks[j].contains(&v));
                (i, best(&note_of, i, &valid).map_or(0.0, |(j, p)| if j != i { p } else { 0.0 }))
            }).collect();
            note_candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let mut claimed: HashSet<(usize, u32)> = HashSet::new();
            for (i, _) in note_candidates {
                let Some(val) = number[i] else { continue };
                let valid = |j: usize| j == i || (marks[j].contains(&val) && !claimed.contains(&(j, val)));
                let target = best(&note_of, i, &valid).or_else(|| best(&note_of, i, &|j| j == i || marks[j].contains(&val))).map_or(i, |(j, _)| j);
                if target != i { claimed.insert((target, val)); note[i] = Some(target); }
            }
            lap(3, &mut since);
            let moves = moves_of(&lines, &regions, &Units { before, level, note }, views, texts, None);
            lap(4, &mut since);
            if self.times {
                eprintln!("doc times: {n} lines {count} regions {} pairs; embed {:.3}s detect {:.3}s order {:.3}s trees {:.3}s decode {:.3}s",
                    pair_i.len(), stages[0], stages[1], stages[2], stages[3], stages[4]);
            }
            Ok(moves)
        }
    }
}
