//! Detect-Order-Construct (Wang et al., Pattern Recognition 2024) in the engine: the stages `docparse.py`
//! fits on the gold and exports (LEGALPDF_TRANSITION_DOC, a folder of `detect.onnx`, `order.onnx`,
//! `headings.onnx`, `bodies.onnx`, `docparse.json`, and the text model's `tokenizer.json` and
//! `embeddings.f32`), and the paper's decoding of what they score into the parser's moves:
//! - Detect, a page at a time: each line's role, its successor and predecessor among its page's lines, and
//!   two flags (a unit opening on its printed marker, a line opening a constituent document); the page's
//!   lines grouped into regions by union-find over the links both heads choose (or the successor's sure
//!   ones, over a half), each region's role its lines' plurality;
//! - Order, over the document's regions: the region a unit goes on in (a paragraph past a page or a
//!   quotation), and a note's region the region holding its reference mark;
//! - Construct, over the headings that open a unit and over the body's: each node's parent and left sibling
//!   among the nodes before it, the tree built by the paper's Algorithm 1 (each node the right-most child of
//!   the node on the tree's right-most path whose parent score times its child's sibling score is greatest),
//!   a constituent document's start emptying it; a heading's level and a body unit's depth its depth there.
//!
//! A line's text is the text model's reading of it as Model2Vec reads one: its tokens (no special ones, the
//! unknown token left out) averaged and scaled to unit length.

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
    tokenizer: tokenizers::Tokenizer,
    embeddings: Vec<f32>,
    meta: Meta,
}

/// The regions whose units take a depth in the body's tree (`docparse.BODY`): text, list items, quotations,
/// provisions.
const BODY: [u8; 4] = [1, 2, 4, 16];
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

/// The paper's Algorithm 1: each node in turn the right-most child of the node on the tree's right-most path
/// (the root first) whose parent probability times the node's sibling probability for that node's last child
/// (itself where it has none) is greatest; `resets[i]` empties the tree before node `i`. Each node's depth
/// (1 under the root).
fn insert(parent: &[f32], sibling: &[f32], n: usize, resets: &[bool]) -> Vec<usize> {
    let mut path: Vec<usize> = Vec::new();
    let mut depths = Vec::with_capacity(n);
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
        path.push(i);
        depths.push(path.len());
    }
    depths
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
        let reader = Self {
            detect: Mutex::new(session(&dir.join("detect.onnx"))?),
            order: Mutex::new(session(&dir.join("order.onnx"))?),
            headings: Mutex::new(session(&dir.join("headings.onnx"))?),
            bodies: Mutex::new(session(&dir.join("bodies.onnx"))?),
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
        let mut roles = vec![0usize; n];
        let mut flags = vec![[0f32; 2]; n];
        let mut group: Vec<usize> = (0..n).collect();
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
                roles[start + i] = argmax(&role[i * CLASSES..(i + 1) * CLASSES]);
                flags[start + i] = [flag[i * 2], flag[i * 2 + 1]];
                let next = argmax(&successor[i * l..(i + 1) * l]);
                let back = argmax(&predecessor[next * l..(next + 1) * l]);
                if next != i && (back == i || successor[i * l + next] > SURE) {
                    let (a, b) = (find(&mut group, start + i), find(&mut group, start + next));
                    group[a.max(b)] = a.min(b);
                }
            }
            start = end;
        }
        // Regions: each group's lines in reading order, its role its lines' plurality (the first line's on a
        // tie); a region of none's role holds lines of no region, which Order does not read.
        let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
        for k in 0..n {
            let root = find(&mut group, k);
            members.entry(root).or_default().push(k);
        }
        let mut regions: Vec<Vec<usize>> = members.into_values().collect();
        regions.sort_by_key(|lines| lines[0]);
        let role_of = |lines: &[usize]| -> u8 {
            let mut counts = [0usize; 17];
            for &k in lines {
                counts[roles[k] / 2] += 1;
            }
            let first = roles[lines[0]] / 2;
            (0..17).fold(first, |best, r| if counts[r] > counts[best] { r } else { best }) as u8
        };
        let region_roles: Vec<u8> = regions.iter().map(|lines| role_of(lines)).collect();
        let read: Vec<usize> = (0..regions.len()).filter(|&r| region_roles[r] != 0).collect();
        let mut moves = vec![Move { region: 0, opens: false, reading: None, note: None, level: 0, depth: 0, resume: false, starts: false, answers: [None; 4] }; n];
        for (k, flag) in flags.iter().enumerate() {
            moves[k].starts = flag[1] > 0.0;
        }
        if read.is_empty() {
            return Ok(moves);
        }
        // Order over the regions read.
        let mut line_states = Vec::new();
        let mut line_region = Vec::new();
        let mut of_line = vec![usize::MAX; n];
        for (i, &r) in read.iter().enumerate() {
            for &k in &regions[r] {
                of_line[k] = i;
            }
        }
        for k in 0..n {
            if of_line[k] != usize::MAX {
                line_states.extend_from_slice(&states[k * d..(k + 1) * d]);
                line_region.push(of_line[k] as i64);
            }
        }
        let count = read.len();
        let kinds: Vec<i64> = read.iter().map(|&r| i64::from(region_roles[r])).collect();
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
        // The trees: the regions that open a unit, headings and the body's apart.
        let start_lines: Vec<usize> = (0..n).filter(|&k| moves[k].starts).collect();
        let tree = |session: &Mutex<Session>, nodes: &[usize]| -> std::result::Result<Vec<usize>, String> {
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
            // A constituent document's start between a node and the one before empties the tree.
            let first = |i: usize| regions[read[nodes[i]]][0];
            let resets: Vec<bool> = (0..h).map(|i| {
                let from = if i == 0 { 0 } else { first(i - 1) + 1 };
                start_lines.iter().any(|&k| k >= from && k <= first(i))
            }).collect();
            Ok(insert(&earlier(parent, h), &earlier(sibling, h), h, &resets))
        };
        let opening = |i: usize| before[i].is_none();
        let headings: Vec<usize> = (0..count).filter(|&i| opening(i) && region_roles[read[i]] == HEADING).collect();
        let bodies: Vec<usize> = (0..count).filter(|&i| opening(i) && BODY.contains(&region_roles[read[i]])).collect();
        let mut level = vec![0usize; count];
        for (&i, depth) in headings.iter().zip(tree(&self.headings, &headings)?) {
            level[i] = depth;
        }
        for (&i, depth) in bodies.iter().zip(tree(&self.bodies, &bodies)?) {
            level[i] = depth;
        }
        // The moves: each region's lines its role's; a unit's first line opens it (at its level or depth,
        // counting on its marker where the flag holds it marked); a region going on in an earlier unit's
        // resumes it where another region of its stream came between.
        let mut last_in_stream: HashMap<usize, usize> = HashMap::new();
        for k in 0..n {
            let i = of_line[k];
            if i == usize::MAX {
                continue;
            }
            let region = region_roles[read[i]];
            let stream = stream_of(region);
            let first = regions[read[i]][0] == k;
            let mv = &mut moves[k];
            mv.region = region;
            if first {
                match before[i] {
                    None => {
                        mv.opens = true;
                        if region == HEADING {
                            mv.level = level[i].clamp(1, 9) as i8;
                        } else if BODY.contains(&region) {
                            mv.depth = level[i].clamp(1, 8) as u8;
                        }
                        if region == NOTE {
                            let line = &views[k];
                            mv.note = line.label.or_else(|| line.readings.iter().find(|r| r.family.chars().all(|c| !c.is_alphabetic())).map(|r| r.value));
                        } else if flags[k][0] > 0.0 && !views[k].readings.is_empty() {
                            mv.reading = Some(0);
                        }
                    }
                    Some(earlier_part) => mv.resume = last_in_stream.get(&stream) != Some(&earlier_part),
                }
            }
            last_in_stream.insert(stream, i);
        }
        // A note's reference: the mark of its number in the region its link names.
        for i in 0..count {
            let first = regions[read[i]][0];
            let Some(value) = (moves[first].opens && region_roles[read[i]] == NOTE).then_some(moves[first].note).flatten() else { continue };
            let target = argmax(&note_of[i * count..(i + 1) * count]);
            if target == i {
                continue;
            }
            let found = regions[read[target]].iter().find_map(|&k| views[k].references.iter().find(|&&(v, ..)| v == value).map(|&(_, s, _)| (views[k].place, s)));
            moves[first].answers[0] = found;
        }
        Ok(moves)
    }
}
