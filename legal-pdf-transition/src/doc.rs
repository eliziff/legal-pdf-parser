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

/// Whether a region is a note's.
fn is_note(region: u8) -> bool {
    region == NOTE || region == NOTE_QUOTE
}

/// The paragraphs t0 decided in the regions' links: a line the judgment reader reads opening a numbered paragraph
/// (`LineView::count`) takes no link from any other line. The margin's paragraph numbers (`LineTable::margin_rows`):
/// a number and the row's line it numbers are one region, the number its first line (it is read just before it);
/// the row's line and its twin's in the other column take no link from any other line, each opening its column's
/// paragraph, nor does the number.
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
fn goes_on_in_column(before: &mut [Option<usize>], regions: &Regions, line_roles: &[usize], views: &[LineView]) {
    let mut last: HashMap<(u8, bool), usize> = HashMap::new();
    for (i, &first) in regions.firsts.iter().enumerate() {
        let key = (regions.role(i), views[first].translation());
        if before[i].is_none() && BODY.contains(&key.0) && line_roles[first] % 2 == 0 && views[first].margin().is_none() && views[first].count().is_none() {
            before[i] = last.get(&key).copied().filter(|&earlier| views[regions.firsts[earlier]].page() + 1 >= views[first].page());
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
/// mark's value is (`reference_spans`).
pub fn note_number(line: &LineView, text: &str) -> Option<u32> {
    line.label()
        .or_else(|| line.readings().iter().find(|r| r.family.chars().all(|c| !c.is_alphabetic())).map(|r| r.value))
        .or_else(|| super::emit::leading_figures(text).map(|(figures, _)| figures))
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
    // The unit each stream last read a line of, and the open marker frames per context for disambiguating
    // Roman vs. alphabetic readings (e.g. `(i)`, `(v)`, `(x)`).
    let mut last_in_stream: HashMap<usize, usize> = HashMap::new();
    let mut doc_id = 0u16;
    let mut stacks: HashMap<u16, Open> = HashMap::new();
    for (k, &i) in regions.of_lines(lines.roles.len()).iter().enumerate() {
        doc_id = doc_id.saturating_add(u16::from(lines.starts[k]));
        if i == usize::MAX { continue; }
        let region = regions.role(i);
        let stands = match (unit[i] == k, last_in_stream.insert(stream_of(region), unit[i])) {
            (true, _) => Stands::Opens,
            (false, Some(last)) if last == unit[i] => Stands::GoesOn(last),
            _ => Stands::Resumes(unit[i]),
        };
        let mv = &mut moves[k];
        mv.region = region;
        if let Stands::Resumes(first) = stands { (mv.resume, mv.resumes) = (true, Some(first as u32)); }
        if !matches!(stands, Stands::Opens) { continue; }
        let open = stacks.entry(doc_id.saturating_mul(2) + u16::from(views[k].translation())).or_default();
        open.opens(region);
        // DOC's own pick among several readings: the one that goes on from an open frame (the next number of its family).
        let chosen_reading = (!is_note(region) && lines.marked[k]).then(|| views[k].opening(|| (views[k].readings().len() > 1).then(|| views[k].readings().iter().position(|r| {
            open.frames().iter().rev().any(|f| f.family == r.family && r.value == f.value + 1)
        })).flatten()));
        if let Some(reading) = views[k].reading(chosen_reading) { open.count(reading, region); }
        mv.opens = true;
        if region == HEADING {
            mv.level = units.level[i].clamp(1, 9) as i8;
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
fn resets(firsts: &[usize], starts: &[bool]) -> Vec<bool> {
    (0..firsts.len()).map(|i| (if i == 0 { 0 } else { firsts[i - 1] + 1 }..=firsts[i]).any(|k| starts[k])).collect()
}

/// The parser's stylesheet as the DOC reader reads it: each line's template and outline anchor, and each template's
/// measured row (`ROW` a row, `Stylesheet::rows`).
#[cfg(feature = "encoder")]
pub struct Style<'a> {
    pub templates: &'a [u32],
    pub anchors: &'a [u16],
    pub rows: Vec<f32>,
}

#[cfg(feature = "encoder")]
impl<'a> Style<'a> {
    pub fn of(sheet: &'a Stylesheet, table: &LineTable, measures: &Measures) -> Self {
        let rows = sheet.rows(table, measures);
        let rows = if rows.is_empty() { vec![0.0; ROW] } else { rows.iter().flatten().map(|&v| if v.is_finite() { v.clamp(-8.0, 8.0) } else { 0.0 }).collect() };
        Self { templates: &sheet.template, anchors: &sheet.anchor, rows }
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
pub fn oracle(gold: &[[i64; 8]], ranks: &[i64], views: &[LineView], texts: &[&str], pages: &[usize]) -> Vec<Move> {
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
    let regions = Regions::of(&lines);
    let count = regions.read.len();
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
    let mut moves = moves_of(&lines, &regions, &Units { before, level, note }, views, texts, Some(&gold_marks));
    read_in(&mut moves, ranks, pages);
    moves
}

/// The reading's order of the lines: each page's lines by their ranks (a line without one just after the line
/// before it), as places in the document.
fn read_in(moves: &mut [Move], ranks: &[i64], pages: &[usize]) {
    if ranks.iter().all(|&r| r < 0) { return; }
    // A line without a rank takes the last rank before it on its page, read after that line.
    let mut last = (usize::MAX, -1i64);
    let mut keyed: Vec<(usize, i64, usize)> = (0..moves.len()).map(|k| {
        if pages[k] != last.0 { last = (pages[k], -1); }
        if ranks[k] >= 0 { last.1 = ranks[k]; }
        (pages[k], last.1, k)
    }).collect();
    keyed.sort();
    keyed.iter().enumerate().for_each(|(place, &(.., k))| moves[k].rank = Some(place as u32));
}

// ------------------------------------------------------------------------------------------------- the model
#[cfg(feature = "encoder")]
pub use model::Reader;

#[cfg(feature = "encoder")]
mod model {
    use super::*;
    use crate::onnx::{init_runtime, session, softmax_rows};
    use ort::session::Session;
    use std::sync::Mutex;

    #[derive(serde::Deserialize)]
    struct Meta {
        width: usize,
        text_dim: usize,
        d: usize,
        classes: usize,
        mean: Vec<f32>,
        std: Vec<f32>,
        /// A template row's width (`Stylesheet::rows`).
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
            let load = |name: &str| { let path = dir.join(format!("{name}.onnx")); session(&path, path.display()) };
            Ok(Self { detect: load("detect")?, stylesheet: load("stylesheet")?, lines: load("lines")?, order: load("order")?, headings: load("headings")?, bodies: load("bodies")? })
        }
    }

    pub struct Reader {
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
        pub fn load(dir: &Path, reorder: bool, times: bool) -> std::result::Result<Self, String> {
            let read = |name: &str| std::fs::read_to_string(dir.join(name)).map_err(|e| format!("{name}: {e}"));
            let meta: Meta = serde_json::from_str(&read("docparse.json")?).map_err(|e| format!("docparse.json: {e}"))?;
            if meta.width != ALL || meta.classes != CLASSES || meta.mean.len() != ALL || meta.std.len() != ALL {
                return Err(format!("docparse.json reads {} features and {} classes; the parser has {ALL} and {CLASSES}", meta.width, meta.classes));
            }
            init_runtime("the DOC reader needs LEGALPDF_ONNX_RUNTIME")?;
            let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| format!("tokenizer.json: {e}"))?;
            let embeddings = f32s(&std::fs::read(dir.join("embeddings.f32")).map_err(|e| format!("embeddings.f32: {e}"))?);
            if embeddings.len() != meta.vocabulary * meta.text_dim {
                return Err(format!("embeddings.f32 holds {} values; docparse.json says {} by {}", embeddings.len(), meta.vocabulary, meta.text_dim));
            }
            let first = Sessions::load(dir)?;
            let gaps = [&first.headings, &first.bodies].map(|s| s.outputs.iter().any(|o| o.name == "gap"));
            let tree_pairs = [&first.headings, &first.bodies].map(|s| s.inputs.iter().any(|i| i.name == "family"));
            let reader = Self { dir: dir.to_path_buf(), sessions: Mutex::new(vec![first]), gaps, tree_pairs, reorder, times, tokenizer, embeddings, meta };
            let parity: Parity = serde_json::from_str(&read("parity.json")?).map_err(|e| format!("parity.json: {e}"))?;
            let ours = reader.embed(&parity.texts.iter().map(String::as_str).collect::<Vec<&str>>())?;
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
                if ids.is_empty() { continue; }
                let row = &mut out[k * dim..(k + 1) * dim];
                ids.iter().for_each(|&id| row.iter_mut().zip(&self.embeddings[id * dim..(id + 1) * dim]).for_each(|(v, e)| *v += e));
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
        pub fn read(&self, table: &LineTable, measures: &Measures, views: &[LineView], texts: &[&str], pages: &[usize], style: &Style) -> std::result::Result<Vec<Move>, String> {
            let mut sessions = self.sessions.lock().ok().and_then(|mut pool| pool.pop()).map_or_else(|| Sessions::load(&self.dir), Ok)?;
            let moves = self.read_with(&mut sessions, table, measures, views, texts, pages, style);
            if let Ok(mut pool) = self.sessions.lock() { pool.push(sessions); }
            moves
        }

        fn read_with(&self, sessions: &mut Sessions, table: &LineTable, measures: &Measures, views: &[LineView], texts: &[&str], pages: &[usize], style: &Style) -> std::result::Result<Vec<Move>, String> {
            let n = views.len();
            let (d, m) = (self.meta.d, &self.meta);
            if n == 0 { return Ok(Vec::new()); }
            let features: Vec<f32> = measures.features.iter().enumerate().map(|(at, &v)| ((v - m.mean[at % ALL]) / m.std[at % ALL]).clamp(-8.0, 8.0)).collect();
            let corners = [feature("x0"), feature("y0"), feature("x1"), feature("y1")];
            let boxes: Vec<f32> = (0..n).flat_map(|k| corners.map(|c| measures.features[k * ALL + c])).collect();
            let mut stages = [0f64; 5];
            let mut lap = |stage: usize, since: &mut std::time::Instant| { stages[stage] += since.elapsed().as_secs_f64(); *since = std::time::Instant::now(); };
            let mut since = std::time::Instant::now();
            let text = self.embed(texts)?;
            lap(0, &mut since);
            // Detect's encoder, a page at a time; the stylesheet over the document; Detect's heads, a page at a time.
            let anchors: Vec<i64> = style.anchors.iter().map(|&a| i64::from(a.min(9))).collect();
            let templates: Vec<i64> = style.templates.iter().map(|&t| i64::from(t)).collect();
            let t0: Vec<f32> = measures.t0.iter().map(|&v| f32::from(v)).collect();
            let runs: Vec<(usize, usize)> = pages[..n].chunk_by(|a, b| a == b).scan(0, |end, run| { *end += run.len(); Some((*end - run.len(), *end)) }).collect();
            let mut states = vec![0f32; n * d];
            for &(start, end) in &runs {
                let l = end - start;
                let outputs = sessions.detect.run(ort::inputs!["features" => tensor!([l, ALL], &features[start * ALL..end * ALL]),
                    "text" => tensor!([l, m.text_dim], &text[start * m.text_dim..end * m.text_dim]), "boxes" => tensor!([l, 4], &boxes[start * 4..end * 4]),
                    "anchors" => tensor!([l], &anchors[start..end]), "t0" => tensor!([l, T0], &t0[start * T0..end * T0])]).map_err(|e| format!("detect: {e}"))?;
                states[start * d..end * d].copy_from_slice(output!(outputs, "states"));
            }
            if ROW != m.row_width {
                return Err(format!("the stylesheet's rows are {ROW} wide; docparse.json says {}", m.row_width));
            }
            let states = {
                let outputs = sessions.stylesheet.run(ort::inputs!["states" => tensor!([n, d], &states[..]), "templates" => tensor!([n], &templates[..]),
                    "rows" => tensor!([style.rows.len() / ROW, ROW], &style.rows[..])]).map_err(|e| format!("stylesheet: {e}"))?;
                output!(outputs, "styled").to_vec()
            };
            let mut lines = Lines { roles: vec![0; n], marked: vec![false; n], starts: vec![false; n], links: (0..n).collect() };
            for &(start, end) in &runs {
                let l = end - start;
                let outputs = sessions.lines.run(ort::inputs!["states" => tensor!([l, d], &states[start * d..end * d]), "boxes" => tensor!([l, 4], &boxes[start * 4..end * 4])])
                    .map_err(|e| format!("lines: {e}"))?;
                let (role, flag, successor, predecessor) = (output!(outputs, "role"), output!(outputs, "flags"), output!(outputs, "successor"), output!(outputs, "predecessor"));
                let successor = softmax_rows(successor, l);
                for i in 0..l {
                    lines.roles[start + i] = argmax(&role[i * CLASSES..(i + 1) * CLASSES]);
                    (lines.marked[start + i], lines.starts[start + i]) = (flag[i * 2] > 0.0, flag[i * 2 + 1] > 0.0);
                }
                // A link both heads choose, or the successor's sure one; none into a line whose role opens a unit.
                for i in 0..l {
                    let next = argmax(&successor[i * l..(i + 1) * l]);
                    let back = argmax(&predecessor[next * l..(next + 1) * l]);
                    if next != i && (back == i || successor[i * l + next] > SURE) && lines.roles[start + next] % 2 == 0 { lines.links[start + i] = start + next; }
                }
            }
            lap(1, &mut since);
            margin_links(&mut lines.links, &table.margin_rows, views);
            // A line t0 reads opening a numbered paragraph on a number it prints opens on that marker.
            lines.marked.iter_mut().enumerate().for_each(|(k, marked)| *marked |= views[k].count().is_some_and(|number| views[k].readings().iter().any(|r| r.value == number)));
            let regions = Regions::of(&lines);
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
            let mut region_boxes = vec![f32::MAX, f32::MAX, f32::MIN, f32::MIN].repeat(count);
            let mut marks: Vec<HashSet<u32>> = vec![HashSet::new(); count];
            for k in in_read() {
                let i = of_line[k];
                for c in 0..2 {
                    region_boxes[i * 4 + c] = region_boxes[i * 4 + c].min(boxes[k * 4 + c]);
                    region_boxes[i * 4 + c + 2] = region_boxes[i * 4 + c + 2].max(boxes[k * 4 + c + 2]);
                }
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
            let (region_states, goes_on, note_of, successor) = {
                let outputs = sessions.order.run(ort::inputs!["line_states" => tensor!([line_region.len(), d], &line_states[..]),
                    "line_region" => tensor!([line_region.len()], &line_region[..]), "region_roles" => tensor!([padded], &kinds[..]),
                    "region_valid" => tensor!([padded], &region_valid[..]), "region_boxes" => tensor!([padded, 4], &region_boxes[..]),
                    "region_pages" => tensor!([padded], &region_pages[..]), "pair_i" => tensor!([pair_i.len()], &pair_i[..]),
                    "pair_j" => tensor!([pair_j.len()], &pair_j[..]), "pair_match" => tensor!([pair_match.len()], &pair_match[..])]).map_err(|e| format!("order: {e}"))?;
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
                let row = &mut rows[i as usize];
                if row.1 == 0 { row.0 = p; }
                row.1 = p + 1;
            }
            let row_softmax = |logits: &[f32]| -> Vec<f32> {
                let mut probs = vec![0f32; logits.len()];
                rows.iter().filter(|&&(a, z)| a < z).for_each(|&(a, z)| probs[a..z].copy_from_slice(&softmax_rows(&logits[a..z], z - a)));
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
                while next_unread < count && place[next_unread] != usize::MAX { next_unread += 1; }
                if position + 1 == count { break; }
                // The parser's next unread region, unless the head is sure of another (and holds the parser's next
                // all but impossible); a line's number or a page's label is never read out of the parser's order.
                let labels = |i: usize| matches!(role(i), LINE_LABEL | PAGE_LABEL);
                let unread = |j: usize| place[j] == usize::MAX && !labels(j);
                let parsers = (rows[at].0..rows[at].1).find(|&p| pair_j[p] as usize == next_unread).map_or(0.0, |p| successor[p]);
                at = match best(&successor, at, &unread) {
                    Some((j, p)) if !labels(at) && j != next_unread && p >= SURE_ORDER && parsers <= 1.0 - SURE_ORDER => j,
                    _ => next_unread,
                };
            }
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
            goes_on_in_column(&mut before, &regions, &lines.roles, views);
            lap(2, &mut since);
            // The trees over the regions that open a unit, headings and the body's apart.
            let reset = resets(firsts, &lines.starts);
            let (mut sections, mut enclosing_heading) = (vec![0i64; count], vec![None; count]);
            let (mut section, mut last_heading) = (0i64, None);
            for i in 0..count {
                if reset[i] { (section, last_heading) = (section + 1, None); }
                enclosing_heading[i] = last_heading;
                if before[i].is_none() && role(i) == HEADING { (section, last_heading) = (section + 1, Some(i)); }
                sections[i] = section;
            }
            let (idx_marker, idx_val, idx_pref, idx_c0, idx_w0, idx_x0, idx_size) =
                (feature("marker"), feature("marker_value"), feature("marker_prefix"), feature("count_1"), feature("wrap_()"), feature("x0"), feature("size"));
            // A line's marker: its numbering's family (its count and wrap), value and prefix.
            let line_family = |k: usize| -> (i64, i64, i64) {
                let row = measures.row(k);
                if row[idx_marker] <= 0.5 { return (0, -1, 0); }
                let (count_idx, wrap_idx) = (argmax(&row[idx_c0..idx_c0 + 9]) as i64, argmax(&row[idx_w0..idx_w0 + 5]) as i64);
                (count_idx * 5 + wrap_idx + 1, (row[idx_val].max(0.0).exp_m1()).round() as i64, row[idx_pref].max(0.0).round() as i64)
            };
            let tree = |session: &mut Session, has_gap: bool, has_pairs: bool, nodes: &[usize]| -> std::result::Result<Vec<usize>, String> {
                let h = nodes.len();
                if h == 0 { return Ok(Vec::new()); }
                // Each node's measure: of its region (`i`) and its first line (`k`).
                let node = |value: &dyn Fn(usize, usize) -> i64| -> Vec<i64> { nodes.iter().map(|&i| value(i, firsts[i])).collect() };
                let node_f32 = |at: usize| -> Vec<f32> { nodes.iter().map(|&i| measures.features[firsts[i] * ALL + at]).collect() };
                let x: Vec<f32> = nodes.iter().flat_map(|&i| region_states[i * d..(i + 1) * d].iter().copied()).collect();
                let mut h_states = vec![0f32; h * d];
                nodes.iter().enumerate().filter_map(|(at, &i)| Some((at, enclosing_heading[i]?)))
                    .for_each(|(at, hd)| h_states[at * d..(at + 1) * d].copy_from_slice(&region_states[hd * d..(hd + 1) * d]));
                let (node_templates, node_anchors) = (node(&|_, k| templates[k]), node(&|_, k| anchors[k]));
                let (s, t, a) = (tensor!([h, d], &x[..]), tensor!([h], &node_templates[..]), tensor!([h], &node_anchors[..]));
                let (node_fam, node_val, node_pref) = (node(&|_, k| line_family(k).0), node(&|_, k| line_family(k).1), node(&|_, k| line_family(k).2));
                let (node_sec, node_x0, node_size) = (node(&|i, _| sections[i]), node_f32(idx_x0), node_f32(idx_size));
                let (node_looks, node_pages, node_roles) = (node(&|_, k| i64::from(table.looks[k])), node(&|_, k| pages[k] as i64), node(&|i, _| i64::from(role(i))));
                let outputs = if has_pairs {
                    session.run(ort::inputs!["states" => s, "heading_states" => tensor!([h, d], &h_states[..]), "templates" => t, "anchors" => a,
                        "family" => tensor!([h], &node_fam[..]), "value" => tensor!([h], &node_val[..]), "prefix" => tensor!([h], &node_pref[..]),
                        "section" => tensor!([h], &node_sec[..]), "x0" => tensor!([h], &node_x0[..]), "size" => tensor!([h], &node_size[..]),
                        "looks" => tensor!([h], &node_looks[..]), "pages" => tensor!([h], &node_pages[..]), "roles" => tensor!([h], &node_roles[..])])
                } else {
                    session.run(ort::inputs!["states" => s, "templates" => t, "anchors" => a])
                }.map_err(|e| format!("construct: {e}"))?;
                let (parent, sibling) = (output!(outputs, "parent"), output!(outputs, "sibling"));
                // Each node's gap where the graph gives one (`gap`, GAPS a node), else one level.
                let gaps: Vec<usize> = if has_gap { output!(outputs, "gap").chunks_exact(GAPS).map(|row| argmax(row) + 1).collect() } else { vec![1; h] };
                let node_firsts: Vec<usize> = nodes.iter().map(|&i| firsts[i]).collect();
                Ok(insert(&earlier(parent, h), &earlier(sibling, h), &gaps, h, &resets(&node_firsts, &lines.starts)))
            };
            let opening = |i: usize| before[i].is_none();
            let headings: Vec<usize> = (0..count).filter(|&i| opening(i) && role(i) == HEADING).collect();
            let bodies: Vec<usize> = (0..count).filter(|&i| opening(i) && BODY.contains(&role(i))).collect();
            let mut level = vec![0usize; count];
            for (nodes, session, k) in [(&headings, &mut sessions.headings, 0), (&bodies, &mut sessions.bodies, 1)] {
                nodes.iter().zip(tree(session, self.gaps[k], self.tree_pairs[k], nodes)?).for_each(|(&i, l)| level[i] = l);
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
            let mut moves = moves_of(&lines, &regions, &Units { before, level, note }, views, texts, None);
            // Each line's rank: its region's place in the reading, then its place in the region.
            let mut ranks = vec![-1i64; n];
            regions.read.iter().enumerate().for_each(|(i, &r)| regions.lines[r].iter().enumerate().for_each(|(at, &k)| ranks[k] = place[i] as i64 * 100_000 + at as i64));
            if self.reorder { read_in(&mut moves, &ranks, pages); }
            lap(4, &mut since);
            if self.times {
                eprintln!("doc times: {n} lines {count} regions {} pairs; embed {:.3}s detect {:.3}s order {:.3}s trees {:.3}s decode {:.3}s",
                    pair_i.len(), stages[0], stages[1], stages[2], stages[3], stages[4]);
            }
            Ok(moves)
        }
    }
}
