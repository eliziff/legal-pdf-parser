//! The transition parser: one pass over a document's lines in reading order, each line making one
//! move (`machine::Move`) against what is open (`machine::State`), so that every competing reading of
//! a line meets in one decision, scored on the line's evidence and the state together. Nothing is
//! claimed first and given back later.
//!
//! Its scores come from tables fitted on the gold (`trees::Forest`): the local scores rank the line's
//! region-and-opening classes on its evidence alone (`lines::LineTable`); the ranker scores each
//! candidate move on its local score and its fit to the state. Tier 1 and tier 2 give the same moves
//! other scores; the parser stays the same.
//!
//! It runs beside the stage parser while it is fitted and measured; its environment:
//! - LEGALPDF_TRANSITION_LOCAL, a local forest (multiclass over `Move::class`), and
//!   LEGALPDF_TRANSITION_LOCAL_FOLDS, two forests a document's half reads in turn (out of fold);
//! - LEGALPDF_TRANSITION_RANKER, a ranker forest (binary over a candidate's `ranked` features);
//! - LEGALPDF_TRANSITION_BEAM, how many whole readings a beam keeps (1, greedy, by default);
//! - LEGALPDF_TRANSITION_GOLD, the gold's labels (the grader's predictions format, rows' gold),
//!   for the oracle;
//! - LEGALPDF_TRANSITION_DUMP, a folder: each line's features (`<sha>.lines.f32`), each candidate's
//!   (`<sha>.moves.f32`) and which the oracle makes (`<sha>.json`);
//! - LEGALPDF_TRANSITION_LABELS, a folder: each document's labels as the parser reads it
//!   (`<sha>.labels.json`), for the grader's `--labels`.

use super::*;

mod lines;
pub(crate) use lines::*;
mod machine;
pub(crate) use machine::*;
mod trees;
use trees::Forest;

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// The region-and-opening classes the local scores keep for each line.
const KEEP: usize = 4;
/// A ranked candidate's features: its class's local log-probability and rank, its `CANDIDATE`
/// features, and the line's place, type and marker (`lines::LINE_FEATURES`'s first twenty and its marker's).
pub(crate) const RANKED: usize = 2 + CANDIDATES + 40;

struct Settings {
    local: Option<Forest>,
    folds: Option<(Forest, Forest)>,
    ranker: Option<Forest>,
    gold: Option<HashMap<String, HashMap<String, [i64; 8]>>>,
    dump: Option<PathBuf>,
    labels: Option<PathBuf>,
    beam: usize,
}

fn settings() -> &'static Settings {
    static SETTINGS: OnceLock<Settings> = OnceLock::new();
    SETTINGS.get_or_init(|| {
        let var = |name: &str| std::env::var_os(name).map(PathBuf::from);
        let forest = |path: PathBuf| Forest::read(&path).map_err(|e| eprintln!("transition parser: {e}")).ok();
        let folds = var("LEGALPDF_TRANSITION_LOCAL_FOLDS").and_then(|folder| Some((forest(folder.join("fold0.txt"))?, forest(folder.join("fold1.txt"))?)));
        Settings {
            local: var("LEGALPDF_TRANSITION_LOCAL").and_then(forest),
            folds,
            ranker: var("LEGALPDF_TRANSITION_RANKER").and_then(forest),
            gold: var("LEGALPDF_TRANSITION_GOLD").map(|path| read_gold(&path)),
            dump: var("LEGALPDF_TRANSITION_DUMP"),
            labels: var("LEGALPDF_TRANSITION_LABELS"),
            beam: std::env::var("LEGALPDF_TRANSITION_BEAM").ok().and_then(|w| w.parse().ok()).unwrap_or(1),
        }
    })
}

/// Whether the transition parser runs on this parse.
pub(crate) fn enabled() -> bool {
    let s = settings();
    s.dump.is_some() || s.labels.is_some()
}

/// The gold's labels by document and piece (`line@offset`), from rows of the grader's predictions
/// format ({sha256, ids, labels}).
fn read_gold(path: &Path) -> HashMap<String, HashMap<String, [i64; 8]>> {
    let mut found = HashMap::new();
    let Ok(text) = std::fs::read_to_string(path) else {
        eprintln!("transition parser: no gold at {}", path.display());
        return found;
    };
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(row) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        let (Some(sha), Some(ids), Some(labels)) = (row["sha256"].as_str(), row["ids"].as_array(), row["labels"].as_array()) else { continue };
        let mut per_piece = HashMap::new();
        for (id, label) in ids.iter().zip(labels) {
            let mut l = [-1i64; 8];
            for (k, cell) in l.iter_mut().enumerate() {
                *cell = label.get(k).and_then(serde_json::Value::as_i64).unwrap_or(-1);
            }
            per_piece.insert(id.as_str().unwrap_or_default().to_owned(), l);
        }
        found.insert(sha.to_owned(), per_piece);
    }
    found
}

/// The move the gold makes at a line, among `candidates`: its region and opening, counting on the
/// reading that goes on with what is open (the next of an open family, else a new family's first,
/// else any) where the gold marks the unit's marker, else on none; a note on the next number or an
/// unanswered reference's; a heading at the gold's step from the heading before it (the nearest the
/// candidates allow).
fn oracle(gold: &[i64; 8], state: &State, line: &LineView, candidates: &[Move]) -> Option<usize> {
    let (region, opens, marked) = (gold[0], gold[1] == 1, gold[5] == 1);
    if region < 0 {
        return None;
    }
    let empty = Open::default();
    let stream = state.streams.get(&line.ctx).unwrap_or(&empty);
    let target = if gold[4] > 0 { (gold[4] - 10) as i8 } else { 0 };
    let preference = |m: &Move| -> (u8, i8) {
        let step = (m.step - target).abs();
        if let Some(value) = m.note {
            let fit = if stream.note.is_some_and(|last| value == last + 1) { 0 } else if stream.unanswered.iter().any(|&(r, _)| r == value) { 1 } else { 2 };
            return (fit, step);
        }
        // A note counts on its number where it has one.
        if m.opens && m.region == NOTE {
            return (3, step);
        }
        let fit = match (m.reading.and_then(|r| line.readings.get(usize::from(r))), marked) {
            (Some(reading), true) => match stream.stack.iter().rposition(|f| f.family == reading.family) {
                Some(depth) if reading.value == stream.stack[depth].value + 1 => 0,
                None if reading.value == 1 => 1,
                _ => 2,
            },
            (None, false) => 0,
            (None, true) => 3,
            (Some(_), false) => 4,
        };
        (fit, step)
    };
    (0..candidates.len())
        .filter(|&c| i64::from(candidates[c].region) == region && candidates[c].opens == opens)
        .min_by_key(|&c| {
            let (fit, step) = preference(&candidates[c]);
            (step, fit, c)
        })
}

/// A line's local scores: each region-and-opening class's probability, the classes best first, and
/// the classes its candidates are drawn from.
struct Local {
    probabilities: Vec<f64>,
    ranked: Vec<usize>,
    classes: Vec<usize>,
}

/// A candidate move's ranked features: its class's local log-probability and rank, its fit to the
/// state, and the line's place, type and marker.
fn ranked_features(state: &State, line: &LineView, row: &[f32], local: &Local, candidate: &Move) -> [f32; RANKED] {
    let mut f = [0f32; RANKED];
    f[0] = local.probabilities[candidate.class()].max(1e-6).ln() as f32;
    f[1] = local.ranked.iter().position(|&c| c == candidate.class()).unwrap_or(34) as f32;
    f[2..2 + CANDIDATES].copy_from_slice(&state.features(line, candidate));
    f[2 + CANDIDATES..22 + CANDIDATES].copy_from_slice(&row[..20]);
    f[22 + CANDIDATES..RANKED].copy_from_slice(&row[72..92]);
    f
}

/// One document parsed: its lines' moves in reading order (greedily, or by a beam of whole readings
/// where LEGALPDF_TRANSITION_BEAM says how wide), and (dumping) what the scorer is fitted on.
pub(crate) fn run(pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, sha256: &str) {
    let s = settings();
    let table = LineTable::read(pages, slots, evidence, fences);
    let n = table.order.len();
    let gold = s.gold.as_ref().and_then(|g| g.get(sha256));
    let forest = match &s.folds {
        // Out of fold: a document's half is read by the forest fitted on the other half.
        Some((fold0, fold1)) => Some(if half(sha256) == 0 { fold1 } else { fold0 }),
        None => s.local.as_ref(),
    };
    let ids: Vec<String> = table.order.iter().map(|&slot| slots.line(pages, slot).id.clone()).collect();
    let views: Vec<LineView> = (0..n).map(|k| {
        let slot = table.order[k];
        let at = slot.index();
        LineView {
            ctx: fences.ctx[at],
            readings: &table.readings[k],
            label: evidence.label[at].map(|l| l.value),
            line_number: slots.line(pages, slot).text.trim().parse::<u32>().ok().filter(|_| evidence.line_number[at]),
            look: table.looks[k],
            references: &table.references[k],
            place: k as u32,
        }
    }).collect();
    let gold_rows: Vec<[i64; 8]> = ids.iter().map(|id| gold.and_then(|g| g.get(&format!("{id}@0"))).copied().unwrap_or([-1; 8])).collect();
    let locals: Vec<Local> = (0..n).map(|k| {
        let probabilities = forest.map(|f| f.probabilities(table.row(k))).unwrap_or_else(|| vec![1.0 / 34.0; 34]);
        let mut ranked: Vec<usize> = (0..probabilities.len()).collect();
        ranked.sort_by(|&a, &b| probabilities[b].total_cmp(&probabilities[a]).then(a.cmp(&b)));
        let mut classes: Vec<usize> = ranked.iter().copied().take(KEEP).collect();
        let gold_class = (gold_rows[k][0] >= 0).then(|| (gold_rows[k][0] * 2 + i64::from(gold_rows[k][1] == 1)) as usize);
        if let Some(c) = gold_class.filter(|c| !classes.contains(c) && s.dump.is_some()) {
            classes.push(c);
        }
        Local { probabilities, ranked, classes }
    }).collect();
    let moves = match (&s.ranker, s.beam) {
        (Some(ranker), width) if width > 1 && s.dump.is_none() => beam(ranker, width, &table, &views, &locals),
        _ => {
            let mut state = State::default();
            let mut moves = Vec::with_capacity(n);
            let (mut move_rows, mut move_lines, mut move_classes, mut move_oracle) = (Vec::<f32>::new(), Vec::<u32>::new(), Vec::<u16>::new(), Vec::<u8>::new());
            for k in 0..n {
                let (line, local) = (&views[k], &locals[k]);
                let candidates = state.candidates(line, &local.classes);
                let features: Vec<[f32; RANKED]> = candidates.iter().map(|m| ranked_features(&state, line, table.row(k), local, m)).collect();
                let best_oracle = gold.and_then(|_| oracle(&gold_rows[k], &state, line, &candidates));
                let choice = match &s.ranker {
                    Some(ranker) => {
                        let scores: Vec<f64> = features.iter().map(|f| ranker.scores(f)[0]).collect();
                        (0..candidates.len()).max_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(b.cmp(&a))).unwrap_or(0)
                    }
                    // No ranker: the oracle where the gold is read (the first round's rollout), else the
                    // local scores' best class, counting on nothing.
                    None => best_oracle.unwrap_or_else(|| {
                        candidates.iter().position(|m| m.class() == local.ranked[0] && m.reading.is_none() && m.note.is_none() && m.step == 0).unwrap_or(0)
                    }),
                };
                if s.dump.is_some() {
                    for (c, f) in features.iter().enumerate() {
                        move_rows.extend_from_slice(f);
                        move_lines.push(k as u32);
                        move_classes.push(candidates[c].class() as u16);
                        move_oracle.push(u8::from(Some(c) == best_oracle));
                    }
                }
                state.apply(line, &candidates[choice]);
                moves.push(candidates[choice]);
            }
            if let Some(folder) = &s.dump {
                let write = || -> std::io::Result<()> {
                    std::fs::create_dir_all(folder)?;
                    let bytes = |values: &[f32]| values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
                    std::fs::write(folder.join(format!("{sha256}.lines.f32")), bytes(&table.features))?;
                    std::fs::write(folder.join(format!("{sha256}.moves.f32")), bytes(&move_rows))?;
                    let meta = serde_json::json!({
                        "sha256": sha256, "ids": ids, "lines": n, "line_width": ALL, "move_width": RANKED,
                        "regions": REGIONS, "line_features": LINE_FEATURES, "candidate_features": CANDIDATE,
                        "gold": gold_rows, "move_lines": move_lines, "move_classes": move_classes, "move_oracle": move_oracle,
                        "chosen": moves.iter().map(Move::class).collect::<Vec<_>>(),
                    });
                    std::fs::write(folder.join(format!("{sha256}.json")), serde_json::to_vec(&meta)?)
                };
                if let Err(error) = write() {
                    eprintln!("transition parser dump {sha256}: {error}");
                }
            }
            moves
        }
    };
    if let Some(folder) = &s.labels {
        let labels = labelled(&table, fences, &views, &moves);
        let body = serde_json::json!({"sha256": sha256, "ids": ids.iter().map(|id| format!("{id}@0")).collect::<Vec<_>>(), "labels": labels});
        if let Err(error) = std::fs::create_dir_all(folder).and_then(|()| std::fs::write(folder.join(format!("{sha256}.labels.json")), body.to_string())) {
            eprintln!("transition parser labels {sha256}: {error}");
        }
    }
}

/// The best whole reading a beam of `width` keeps: every reading kept extended by each of the line's
/// candidates, scored by the ranker's log-probability of the candidate among the line's (its scores
/// softmaxed per line) summed over the reading; the best `width` go on.
fn beam(ranker: &Forest, width: usize, table: &LineTable, views: &[LineView], locals: &[Local]) -> Vec<Move> {
    // Each move made, with the move before it on its reading.
    let mut made: Vec<(Option<usize>, Move)> = Vec::new();
    let mut readings: Vec<(State, f64, Option<usize>)> = vec![(State::default(), 0.0, None)];
    for k in 0..views.len() {
        let (line, local) = (&views[k], &locals[k]);
        let mut extended: Vec<(usize, Move, f64)> = Vec::new();
        for (r, (state, score, _)) in readings.iter().enumerate() {
            let candidates = state.candidates(line, &local.classes);
            let scores: Vec<f64> = candidates.iter().map(|m| ranker.scores(&ranked_features(state, line, table.row(k), local, m))[0]).collect();
            let top = scores.iter().copied().fold(f64::MIN, f64::max);
            let total: f64 = scores.iter().map(|x| (x - top).exp()).sum();
            for (m, x) in candidates.into_iter().zip(scores) {
                extended.push((r, m, score + (x - top) - total.ln()));
            }
        }
        extended.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
        extended.truncate(width);
        readings = extended.into_iter().map(|(r, m, score)| {
            let mut state = readings[r].0.clone();
            state.apply(line, &m);
            made.push((readings[r].2, m));
            (state, score, Some(made.len() - 1))
        }).collect();
    }
    let mut moves = Vec::with_capacity(views.len());
    let mut at = readings.first().and_then(|r| r.2);
    while let Some(k) = at {
        moves.push(made[k].1);
        at = made[k].0;
    }
    moves.reverse();
    moves
}

/// The moves as the grader's labels, the state replayed over them: region, opening, document start,
/// paired reference (a body line a note answered), a heading's level step, a marker, no resumption, a
/// body unit's depth step (its depth among the body units open on the stack, stepped from the one
/// before).
fn labelled(table: &LineTable, fences: &Fences, views: &[LineView], moves: &[Move]) -> Vec<[i64; 8]> {
    let mut state = State::default();
    let mut labels: Vec<[i64; 8]> = Vec::with_capacity(moves.len());
    let mut body_depth: Option<usize> = None;
    for (k, (line, chosen)) in views.iter().zip(moves).enumerate() {
        let at = table.order[k].index();
        let starts_document = fences.document[at] != 0 && (k == 0 || fences.document[table.order[k - 1].index()] != fences.document[at]);
        let mut label = [i64::from(chosen.region), i64::from(chosen.opens), i64::from(starts_document), 0, 0, i64::from(chosen.reading.is_some()), -1, 0];
        if chosen.opens && chosen.region == HEADING {
            label[4] = 10 + i64::from(chosen.step);
        }
        if let Some(paired) = state.apply(line, chosen) {
            labels[paired as usize][3] = 1;
        }
        if chosen.opens && RANKED_BODY.contains(&chosen.region) {
            let depth = state.streams.get(&line.ctx).map_or(0, |open| open.stack.iter().filter(|f| RANKED_BODY.contains(&f.region)).count()).max(1);
            label[7] = (10 + depth as i64 - body_depth.unwrap_or(1) as i64).clamp(2, 18);
            body_depth = Some(depth);
        }
        labels.push(label);
    }
    labels
}

/// The body regions that take a depth (the tagger's ranked regions, headings aside).
const RANKED_BODY: [u8; 7] = [1, 16, 2, 4, 6, 8, 9];

/// Which half of the documents a source falls in: the CRC-32 of its sha256 (zlib's), modulo two.
fn half(sha256: &str) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for byte in sha256.bytes() {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xedb8_8320 } else { crc >> 1 };
        }
    }
    !crc % 2
}
