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
pub(crate) const RANKED: usize = 2 + 23 + 40;

struct Settings {
    local: Option<Forest>,
    folds: Option<(Forest, Forest)>,
    ranker: Option<Forest>,
    gold: Option<HashMap<String, HashMap<String, [i64; 8]>>>,
    dump: Option<PathBuf>,
    labels: Option<PathBuf>,
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
/// else any) where the gold marks the unit's marker, else on none; a note on its label's value.
fn oracle(gold: &[i64; 8], state: &State, ctx: u16, readings: &[MarkerReading], candidates: &[Move]) -> Option<usize> {
    let (region, opens, marked) = (gold[0], gold[1] == 1, gold[5] == 1);
    if region < 0 {
        return None;
    }
    let matching: Vec<usize> = (0..candidates.len()).filter(|&c| i64::from(candidates[c].region) == region && candidates[c].opens == opens).collect();
    let empty = Open::default();
    let stream = state.streams.get(&ctx).unwrap_or(&empty);
    let preference = |m: &Move| -> u8 {
        if let Some(value) = m.note {
            return if stream.note.is_some_and(|last| value == last + 1) { 0 } else if stream.unanswered.contains(&value) { 1 } else { 2 };
        }
        match (m.reading.and_then(|r| readings.get(usize::from(r))), marked) {
            (Some(reading), true) => match stream.stack.iter().rposition(|f| f.family == reading.family) {
                Some(depth) if reading.value == stream.stack[depth].value + 1 => 0,
                None if reading.value == 1 => 1,
                _ => 2,
            },
            (None, false) => 0,
            (None, true) => 3,
            (Some(_), false) => 4,
        }
    };
    matching.into_iter().min_by_key(|&c| (preference(&candidates[c]), c))
}

/// One document parsed: its lines' moves in reading order, and (dumping) what the scorer is fitted on.
pub(crate) fn run(pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, sha256: &str) {
    let s = settings();
    let table = LineTable::read(pages, slots, evidence, fences);
    let n = table.order.len();
    let gold = s.gold.as_ref().and_then(|g| g.get(sha256));
    let local = match &s.folds {
        // Out of fold: a document's half is read by the forest fitted on the other half.
        Some((fold0, fold1)) => Some(if half(sha256) == 0 { fold1 } else { fold0 }),
        None => s.local.as_ref(),
    };
    let ids: Vec<String> = table.order.iter().map(|&slot| slots.line(pages, slot).id.clone()).collect();
    let mut state = State::default();
    let mut chosen_moves = Vec::with_capacity(n);
    let mut move_rows: Vec<f32> = Vec::new();
    let mut move_lines: Vec<u32> = Vec::new();
    let mut move_classes: Vec<u16> = Vec::new();
    let mut move_oracle: Vec<u8> = Vec::new();
    let mut gold_rows: Vec<[i64; 8]> = Vec::with_capacity(n);
    for k in 0..n {
        let slot = table.order[k];
        let at = slot.index();
        let ctx = fences.ctx[at];
        let row = table.row(k);
        let readings = &table.readings[k];
        let label = evidence.label[at].map(|l| l.value);
        let line_value = slots.line(pages, slot).text.trim().parse::<u32>().ok().filter(|_| evidence.line_number[at]);
        let gold_here = gold.and_then(|g| g.get(&format!("{}@0", ids[k]))).copied().unwrap_or([-1; 8]);
        gold_rows.push(gold_here);
        let probabilities = local.map(|f| f.probabilities(row)).unwrap_or_else(|| vec![1.0 / 34.0; 34]);
        let mut ranked: Vec<usize> = (0..probabilities.len()).collect();
        ranked.sort_by(|&a, &b| probabilities[b].total_cmp(&probabilities[a]).then(a.cmp(&b)));
        let mut classes: Vec<usize> = ranked.iter().copied().take(KEEP).collect();
        let gold_class = (gold_here[0] >= 0).then(|| (gold_here[0] * 2 + i64::from(gold_here[1] == 1)) as usize);
        if let Some(c) = gold_class.filter(|c| !classes.contains(c) && s.dump.is_some()) {
            classes.push(c);
        }
        let candidates = state.candidates(readings, label, &classes);
        let features: Vec<[f32; RANKED]> = candidates.iter().map(|m| {
            let mut f = [0f32; RANKED];
            f[0] = probabilities[m.class()].max(1e-6).ln() as f32;
            f[1] = ranked.iter().position(|&c| c == m.class()).unwrap_or(34) as f32;
            f[2..25].copy_from_slice(&state.features(ctx, readings, line_value, m));
            f[25..45].copy_from_slice(&row[..20]);
            f[45..65].copy_from_slice(&row[72..92]);
            f
        }).collect();
        let best_oracle = gold.and_then(|_| oracle(&gold_here, &state, ctx, readings, &candidates));
        let choice = match &s.ranker {
            Some(ranker) => {
                let scores: Vec<f64> = features.iter().map(|f| ranker.scores(f)[0]).collect();
                (0..candidates.len()).max_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(b.cmp(&a))).unwrap_or(0)
            }
            // No ranker: the oracle where the gold is read (the first round's rollout), else the local
            // scores' best class, counting on nothing.
            None => best_oracle.unwrap_or_else(|| candidates.iter().position(|m| m.class() == ranked[0] && m.reading.is_none() && m.note.is_none()).unwrap_or(0)),
        };
        if s.dump.is_some() {
            for (c, f) in features.iter().enumerate() {
                move_rows.extend_from_slice(f);
                move_lines.push(k as u32);
                move_classes.push(candidates[c].class() as u16);
                move_oracle.push(u8::from(Some(c) == best_oracle));
            }
        }
        let chosen = candidates[choice];
        state.apply(ctx, readings, line_value, &table.references[k], &chosen);
        chosen_moves.push(chosen);
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
                "chosen": chosen_moves.iter().map(|m| m.class()).collect::<Vec<_>>(),
            });
            std::fs::write(folder.join(format!("{sha256}.json")), serde_json::to_vec(&meta)?)
        };
        if let Err(error) = write() {
            eprintln!("transition parser dump {sha256}: {error}");
        }
    }
    if let Some(folder) = &s.labels {
        let labels: Vec<[i64; 8]> = levelled(&chosen_moves, &table.readings);
        let body = serde_json::json!({"sha256": sha256, "ids": ids.iter().map(|id| format!("{id}@0")).collect::<Vec<_>>(), "labels": labels});
        if let Err(error) = std::fs::create_dir_all(folder).and_then(|()| std::fs::write(folder.join(format!("{sha256}.labels.json")), body.to_string())) {
            eprintln!("transition parser labels {sha256}: {error}");
        }
    }
}

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

/// The moves as the grader's labels: region, opening, document start, reference, a heading's level
/// step, a marker, no resumption, a body unit's depth step. A heading's level and a body unit's depth
/// are its depth among the open units of its kind after the move, stepped from the one before.
fn levelled(moves: &[Move], readings: &[Vec<MarkerReading>]) -> Vec<[i64; 8]> {
    const HEADING: u8 = 3;
    const RANKED_BODY: [u8; 7] = [1, 16, 2, 4, 6, 8, 9];
    let mut out = Vec::with_capacity(moves.len());
    let (mut heading_level, mut body_depth) = (None::<usize>, None::<usize>);
    let mut stack: Vec<(String, u8)> = Vec::new();
    for (m, found) in moves.iter().zip(readings) {
        if let Some(reading) = m.reading.and_then(|r| found.get(usize::from(r))) {
            match stack.iter().rposition(|(family, _)| *family == reading.family) {
                Some(depth) => {
                    stack.truncate(depth);
                    stack.push((reading.family.clone(), m.region));
                }
                None => stack.push((reading.family.clone(), m.region)),
            }
        }
        let mut label = [i64::from(m.region), i64::from(m.opens), 0, 0, 0, i64::from(m.reading.is_some()), -1, 0];
        if m.opens && m.region == HEADING {
            let level = stack.iter().filter(|(_, r)| *r == HEADING).count().max(1);
            label[4] = (10 + level as i64 - heading_level.unwrap_or(1) as i64).clamp(2, 18);
            heading_level = Some(level);
        } else if m.opens && RANKED_BODY.contains(&m.region) {
            let depth = stack.iter().filter(|(_, r)| RANKED_BODY.contains(r)).count().max(1);
            label[7] = (10 + depth as i64 - body_depth.unwrap_or(1) as i64).clamp(2, 18);
            body_depth = Some(depth);
        }
        out.push(label);
    }
    out
}
