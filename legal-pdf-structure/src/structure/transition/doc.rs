//! Detect-Order-Construct (Wang et al., Pattern Recognition 2024) in the engine: the stages `docparse.py`
//! fits on the gold and exports (LEGALPDF_TRANSITION_DOC, a folder of `detect.onnx`, `order.onnx`,
//! `headings.onnx`, `bodies.onnx`, `docparse.json`, and the text model's `tokenizer.json` and
//! `embeddings.f32`), and the paper's decoding of what they score into the parser's moves:
//! - Detect, a page at a time: each line's role, its successor and predecessor among its page's lines, and
//!   two flags (a unit opening on its printed marker, a line opening a constituent document); the page's
//!   lines grouped into regions by union-find over the links both heads choose (or the successor's sure
//!   ones, over a half), none into a line whose role opens a unit; each region's role its lines' plurality;
//! - Order, over the document's regions: the region a unit goes on in (a paragraph past a page or a
//!   quotation), and a note's region the region holding its reference mark;
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

/// The regions whose units take a depth in the body's tree (`docparse.BODY`): text, list items, quotations,
/// provisions.
const BODY: [u8; 4] = [1, 2, 4, 16];
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
/// its text opens on ("27" of "27 Ibid", as the gold's notes are numbered).
fn note_number(line: &LineView, text: &str) -> Option<u32> {
    line.label
        .or_else(|| line.readings.iter().find(|r| r.family.chars().all(|c| !c.is_alphabetic())).map(|r| r.value))
        .or_else(|| text.trim_start().chars().take_while(char::is_ascii_digit).collect::<String>().parse().ok())
}

/// The document's moves from the heads' answers: each region's lines its role's; a unit's first line opens it
/// (at its level or depth, counting on its marker where the line is marked, a note on its number); a line of a
/// unit another unit of its stream came before goes on with it by name (its first line); a note answers the
/// marks of its number in the region its link names.
fn moves_of(lines: &Lines, regions: &[Vec<usize>], roles: &[u8], read: &[usize], units: &Units, views: &[LineView], texts: &[&str]) -> Vec<Move> {
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
    // The unit each stream last read a line of.
    let mut last_in_stream: HashMap<usize, usize> = HashMap::new();
    for k in 0..n {
        let i = of_line[k];
        if i == usize::MAX {
            continue;
        }
        let region = roles[read[i]];
        let stream = stream_of(region);
        let mv = &mut moves[k];
        mv.region = region;
        if unit[i] == k {
            mv.opens = true;
            if region == HEADING {
                mv.level = units.level[i].clamp(1, 9) as i8;
            } else if BODY.contains(&region) {
                mv.depth = units.level[i].clamp(1, 8) as u8;
            }
            if region == NOTE {
                mv.note = note_number(&views[k], texts[k]);
            } else if lines.marked[k] && !views[k].readings.is_empty() {
                mv.reading = Some(0);
            }
        } else if last_in_stream.get(&stream) != Some(&unit[i]) {
            mv.resume = true;
            mv.resumes = Some(unit[i] as u32);
        }
        last_in_stream.insert(stream, unit[i]);
    }
    for (i, target) in units.note.iter().enumerate() {
        let first = regions[read[i]][0];
        let (Some(target), Some(value)) = (target, moves[first].opens.then_some(moves[first].note).flatten()) else { continue };
        let marks: Vec<(u32, u32)> = regions[read[*target]].iter()
            .flat_map(|&k| views[k].references.iter().filter(move |&&(v, ..)| v == value).map(move |&(_, s, _)| (views[k].place, s)))
            .take(4).collect();
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
    for (k, g) in gold.iter().enumerate() {
        let region = gold_region(g);
        if region <= 0 {
            continue;
        }
        let stream = stream_of(region as u8);
        let pointer = usize::try_from(g[6]).ok().filter(|&p| p < k && unit[p] != usize::MAX);
        unit[k] = match (g[1], pointer, open.get(&stream)) {
            (2, Some(p), _) => unit[p],
            (0, _, Some(&u)) if gold_region(&gold[u]) == region => u,
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
    // A note's region: the gold's paired mark of its number before it, two pages back at most.
    let mut region_of = vec![usize::MAX; n];
    for (i, &r) in read.iter().enumerate() {
        for &k in &regions[r] {
            region_of[k] = i;
        }
    }
    let note = (0..count).map(|i| {
        let k = first(i);
        if roles[read[i]] != NOTE || before[i].is_some() {
            return None;
        }
        let value = note_number(&views[k], texts[k])?;
        (0..k).rev().take_while(|&j| pages[k].saturating_sub(pages[j]) <= 2)
            .find(|&j| gold[j][3] == 1 && region_of[j] != usize::MAX && views[j].references.iter().any(|&(v, ..)| v == value))
            .map(|j| region_of[j])
    }).collect();
    let mut moves = moves_of(&lines, &regions, &roles, &read, &Units { before, level, note }, views, texts);
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
    use std::sync::Mutex;

    #[derive(serde::Deserialize)]
    struct Meta {
        width: usize,
        text_dim: usize,
        d: usize,
        classes: usize,
        mean: Vec<f32>,
        std: Vec<f32>,
        vocabulary: usize,
        unknown: Option<u32>,
        normalize: bool,
    }

    #[derive(serde::Deserialize)]
    struct Parity {
        texts: Vec<String>,
        vectors: Vec<Vec<f32>>,
    }

    pub(crate) struct Reader {
        detect: Mutex<Session>,
        order: Mutex<Session>,
        headings: Mutex<Session>,
        bodies: Mutex<Session>,
        /// Whether each tree's graph gives its nodes' gaps (`gap`).
        gaps: [bool; 2],
        tokenizer: tokenizers::Tokenizer,
        embeddings: Vec<f32>,
        meta: Meta,
    }

    /// A successor link the predecessor head does not choose back is kept where the successor head holds it
    /// likelier than this.
    const SURE: f32 = 0.5;

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
        pub(crate) fn load(dir: &Path) -> std::result::Result<Self, String> {
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
            let (headings, bodies) = (session(&dir.join("headings.onnx"))?, session(&dir.join("bodies.onnx"))?);
            let gaps = [&headings, &bodies].map(|s| s.outputs.iter().any(|o| o.name == "gap"));
            let reader = Self {
                detect: Mutex::new(session(&dir.join("detect.onnx"))?),
                order: Mutex::new(session(&dir.join("order.onnx"))?),
                headings: Mutex::new(headings),
                bodies: Mutex::new(bodies),
                gaps,
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

        /// The document's moves, a line each in reading order: `texts` and `pages` each line's.
        pub(crate) fn read(&self, table: &LineTable, views: &[LineView], texts: &[&str], pages: &[usize]) -> std::result::Result<Vec<Move>, String> {
            let n = table.order.len();
            let (d, m) = (self.meta.d, &self.meta);
            if n == 0 {
                return Ok(Vec::new());
            }
            let features: Vec<f32> = table.features.iter().enumerate().map(|(at, &v)| ((v - m.mean[at % ALL]) / m.std[at % ALL]).clamp(-8.0, 8.0)).collect();
            let at = |name: &str| LINE_FEATURES.iter().position(|f| *f == name).unwrap_or(0);
            let corners = [at("x0"), at("y0"), at("x1"), at("y1")];
            let boxes: Vec<f32> = (0..n).flat_map(|k| corners.map(|c| table.features[k * ALL + c])).collect();
            let text = self.embed(texts)?;
            // Detect, a page at a time.
            let mut states = vec![0f32; n * d];
            let mut lines = Lines { roles: vec![0; n], marked: vec![false; n], starts: vec![false; n], links: (0..n).collect() };
            let mut chosen = vec![(0usize, 0usize, 0f32); n];
            let mut start = 0;
            while start < n {
                let end = (start..n).find(|&k| pages[k] != pages[start]).unwrap_or(n);
                let l = end - start;
                let f = TensorRef::from_array_view(([l, ALL], &features[start * ALL..end * ALL])).map_err(|e| e.to_string())?;
                let t = TensorRef::from_array_view(([l, m.text_dim], &text[start * m.text_dim..end * m.text_dim])).map_err(|e| e.to_string())?;
                let b = TensorRef::from_array_view(([l, 4], &boxes[start * 4..end * 4])).map_err(|e| e.to_string())?;
                let mut session = self.detect.lock().map_err(|_| "detect session poisoned")?;
                let outputs = session.run(ort::inputs!["features" => f, "text" => t, "boxes" => b]).map_err(|e| format!("detect: {e}"))?;
                let (_, s) = outputs["states"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
                states[start * d..end * d].copy_from_slice(s);
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
                start = end;
            }
            // A link both heads choose, or the successor's sure one; none into a line whose role opens a unit.
            for k in 0..n {
                let (next, back, sure) = chosen[k];
                if next != k && (back == k || sure > SURE) && lines.roles[next] % 2 == 0 {
                    lines.links[k] = next;
                }
            }
            let (regions, roles) = regions_of(&lines);
            let read: Vec<usize> = (0..regions.len()).filter(|&r| roles[r] != 0).collect();
            let count = read.len();
            if count == 0 {
                return Ok(moves_of(&lines, &regions, &roles, &read, &Units { before: Vec::new(), level: Vec::new(), note: Vec::new() }, views, texts));
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
            let (region_states, goes_on, note_of) = {
                let ls = TensorRef::from_array_view(([line_region.len(), d], &line_states[..])).map_err(|e| e.to_string())?;
                let lr = TensorRef::from_array_view(([line_region.len()], &line_region[..])).map_err(|e| e.to_string())?;
                let rr = TensorRef::from_array_view(([count], &kinds[..])).map_err(|e| e.to_string())?;
                let mut session = self.order.lock().map_err(|_| "order session poisoned")?;
                let outputs = session.run(ort::inputs!["line_states" => ls, "line_region" => lr, "region_roles" => rr]).map_err(|e| format!("order: {e}"))?;
                let take = |name: &str| outputs[name].try_extract_tensor::<f32>().map(|(_, v)| v.to_vec()).map_err(|e| e.to_string());
                (take("regions")?, softmax_rows(&take("goes_on")?, count), take("note_of")?)
            };
            // Units: a region goes on in the later region its link holds likeliest, each region taken once, the
            // surest links first.
            let mut links: Vec<(usize, usize, f32)> = (0..count).filter_map(|i| {
                let row = &goes_on[i * count..(i + 1) * count];
                let j = argmax(row);
                (j > i).then_some((i, j, row[j]))
            }).collect();
            links.sort_by(|a, b| b.2.total_cmp(&a.2));
            let mut before = vec![None; count];
            let mut after = vec![false; count];
            for (i, j, _) in links {
                if before[j].is_none() && !after[i] {
                    before[j] = Some(i);
                    after[i] = true;
                }
            }
            // The trees over the regions that open a unit, headings and the body's apart.
            let tree = |session: &Mutex<Session>, has_gap: bool, nodes: &[usize]| -> std::result::Result<Vec<usize>, String> {
                let h = nodes.len();
                if h == 0 {
                    return Ok(Vec::new());
                }
                let x: Vec<f32> = nodes.iter().flat_map(|&i| region_states[i * d..(i + 1) * d].iter().copied()).collect();
                let s = TensorRef::from_array_view(([h, d], &x[..])).map_err(|e| e.to_string())?;
                let mut session = session.lock().map_err(|_| "construct session poisoned")?;
                let outputs = session.run(ort::inputs!["states" => s]).map_err(|e| format!("construct: {e}"))?;
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
            for (&i, l) in headings.iter().zip(tree(&self.headings, self.gaps[0], &headings)?) {
                level[i] = l;
            }
            for (&i, l) in bodies.iter().zip(tree(&self.bodies, self.gaps[1], &bodies)?) {
                level[i] = l;
            }
            // A note's region: the one its link names, where not itself.
            let note = (0..count).map(|i| {
                let target = argmax(&note_of[i * count..(i + 1) * count]);
                (roles[read[i]] == NOTE && target != i).then_some(target)
            }).collect();
            Ok(moves_of(&lines, &regions, &roles, &read, &Units { before, level, note }, views, texts))
        }
    }
}
