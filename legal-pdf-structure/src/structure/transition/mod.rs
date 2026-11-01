//! The transition parser: one pass over a document's lines in reading order, each line making one
//! move (`machine::Move`) against what is open (`machine::State`), so that every competing reading of
//! a line meets in one decision, scored on the line's evidence and the state together. Nothing is
//! claimed first and given back later.
//!
//! Its scores come from tables fitted on the gold (`trees::Forest`), in two parts: the local scores, each
//! region-and-opening class's log-probability on the line's evidence alone (`lines::LineTable`), read once
//! per line; and the ranker's correction, how the move fits the state, fitted from the local score as its
//! base. A move's score is the two summed. Every move competes but one the local scores hold all but
//! impossible and the state does not propose (`State::proposed`): no learned cut decides what may be
//! weighed. Tier 1 and tier 2 give the same moves other local scores; the parser stays the same.
//!
//! It runs beside the stage parser while it is fitted and measured; its environment:
//! - LEGALPDF_TRANSITION_LOCAL, a local forest (multiclass over `Move::class`; the product's, fitted
//!   on all of train), or LEGALPDF_TRANSITION_LOCAL_FOLDS, a folder of k (fold0.txt ...), each fitted on
//!   every document but those of its fold, which read it (out of fold, train's rollouts);
//! - LEGALPDF_TRANSITION_RANKER, a ranker forest: its score, added to a candidate's local log-probability,
//!   from the candidate's `RANKED` features;
//! - LEGALPDF_TRANSITION_BEAM, how many whole readings a beam keeps (1, greedy, by default);
//! - LEGALPDF_TRANSITION_GRAMMAR, the grammar tables counted over legal text (`grammar`);
//! - LEGALPDF_TRANSITION_GOLD, the gold's labels (the grader's predictions format, rows' gold),
//!   for the oracle;
//! - LEGALPDF_TRANSITION_DUMP, a folder: each line's features (`<sha>.lines.f32`), each candidate's
//!   (`<sha>.moves.f32`) and which the oracle makes (`<sha>.json`);
//! - LEGALPDF_TRANSITION_LABELS, a folder: each document's labels as the parser reads it
//!   (`<sha>.labels.json`), for the grader's `--labels`;
//! - LEGALPDF_TRANSITION_EMIT: the parse's output is the transition parser's structure (`emit`).

use super::*;

mod lines;
pub(crate) use lines::*;
mod machine;
pub(crate) use machine::*;
mod trees;
use trees::Forest;
mod emit;
mod grammar;
pub(crate) use grammar::{Grammar, Tables};

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// A class the local scores hold below this probability competes only where the state proposes it: a
/// guard on the parse's cost, not a judgement.
const FLOOR: f64 = 1e-4;
/// The region-and-opening classes.
const CLASSES: usize = 34;
/// The origin the transition parser's nodes name.
pub(crate) const ORIGIN: &str = "legal-pdf-transition.v1";
/// A ranked candidate's features: its class's local log-probability and rank, its `CANDIDATE`
/// features, and the line's place, type and marker (`lines::LINE_FEATURES`'s first twenty and its marker's).
pub(crate) const RANKED: usize = 2 + CANDIDATES + 40;

enum LocalScores {
    One(Forest),
    /// Out of fold: a document read by the forest fitted on every fold but its own (`fold_of`).
    Folds(Vec<Forest>),
}

struct Settings {
    local: Option<LocalScores>,
    ranker: Option<Forest>,
    gold: Option<HashMap<String, HashMap<String, [i64; 8]>>>,
    dump: Option<PathBuf>,
    labels: Option<PathBuf>,
    beam: usize,
    emit: bool,
    grammar: Option<grammar::Grammar>,
}

fn settings() -> &'static Settings {
    static SETTINGS: OnceLock<Settings> = OnceLock::new();
    SETTINGS.get_or_init(|| {
        let var = |name: &str| std::env::var_os(name).map(PathBuf::from);
        let forest = |path: PathBuf, width: usize, classes: usize| Forest::read(&path, width, classes).map_err(|e| eprintln!("transition parser: {e}")).ok();
        let local = match (var("LEGALPDF_TRANSITION_LOCAL"), var("LEGALPDF_TRANSITION_LOCAL_FOLDS")) {
            (Some(path), _) => forest(path, ALL, CLASSES).map(LocalScores::One),
            (None, Some(folder)) => {
                let folds: Vec<Forest> = (0..).map(|k| folder.join(format!("fold{k}.txt"))).take_while(|path| path.exists())
                    .filter_map(|path| forest(path, ALL, CLASSES)).collect();
                (!folds.is_empty()).then_some(LocalScores::Folds(folds))
            }
            (None, None) => None,
        };
        Settings {
            local,
            ranker: var("LEGALPDF_TRANSITION_RANKER").and_then(|path| forest(path, RANKED, 1)),
            gold: var("LEGALPDF_TRANSITION_GOLD").map(|path| read_gold(&path)),
            dump: var("LEGALPDF_TRANSITION_DUMP"),
            labels: var("LEGALPDF_TRANSITION_LABELS"),
            beam: std::env::var("LEGALPDF_TRANSITION_BEAM").ok().and_then(|w| w.parse().ok()).unwrap_or(1),
            emit: std::env::var_os("LEGALPDF_TRANSITION_EMIT").is_some(),
            grammar: var("LEGALPDF_TRANSITION_GRAMMAR").and_then(|path| grammar::Grammar::read(&path).map_err(|e| eprintln!("transition parser: {e}")).ok()),
        }
    })
}

/// Whether the transition parser runs on this parse.
pub(crate) fn enabled() -> bool {
    let s = settings();
    s.dump.is_some() || s.labels.is_some() || s.emit
}

/// Whether the transition parser's reading is the parse's output (LEGALPDF_TRANSITION_EMIT).
pub(crate) fn emitting() -> bool {
    settings().emit
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

/// The move the gold makes at a line, among `candidates`: of those of the gold's region and opening,
/// the one whose result, the state replayed one move on, the gold agrees with most: a heading's level
/// step, a body unit's depth step, whether the unit opens on its marker, a note answering a reference
/// on a line the gold pairs. None where no candidate is of the gold's region and opening.
fn oracle(gold: &[[i64; 8]], k: usize, state: &State, line: &LineView, candidates: &[Move]) -> Option<usize> {
    let here = &gold[k];
    let (region, opens) = (here[0], here[1] == 1);
    if region < 0 {
        return None;
    }
    let disagreements = |m: &Move| -> u32 {
        let made = state.clone().apply(line, m);
        let mut wrong = 0;
        if m.opens && m.region == HEADING {
            if here[4] >= 0 {
                wrong += (made.level_step - here[4]).unsigned_abs() as u32 * 2;
            } else {
                // A gold without levels (A2AJ's): the level the line's look last took in its document,
                // else the open heading's.
                let outline = state.outlines.get(&(state.document + u16::from(m.starts)));
                let looked = outline.and_then(|o| o.look_levels.iter().find(|(look, _)| *look == line.look)).map(|&(_, l)| l);
                let preferred = looked.unwrap_or_else(|| outline.map_or(1, |o| o.heading.max(1)));
                wrong += u32::from((m.level - preferred).unsigned_abs());
            }
        }
        // A document the gold opens here, or none it does not.
        wrong += 4 * u32::from(m.starts != (here[2] == 1));
        if m.opens && RANKED_BODY.contains(&m.region) && here[7] > 0 {
            wrong += u32::from(made.depth_step != here[7]);
        }
        if m.opens && m.region != NOTE {
            wrong += u32::from(m.reading.is_some() != (here[5] == 1));
        }
        if !m.opens {
            wrong += 2 * u32::from(m.resume != (here[1] == 2));
        }
        if m.opens && m.region == NOTE {
            wrong += match (m.note, made.paired) {
                (Some(_), Some((line, ..))) if gold.get(line as usize).is_some_and(|g| g[3] == 1) => 0,
                (Some(_), _) => 1,
                (None, _) => 3,
            };
        }
        wrong
    };
    (0..candidates.len())
        .filter(|&c| i64::from(candidates[c].region) == region && candidates[c].opens == opens)
        .min_by_key(|&c| (disagreements(&candidates[c]), c))
}

/// A line's local scores: each region-and-opening class's probability, and the classes best first.
pub(crate) struct Local {
    probabilities: Vec<f64>,
    ranked: Vec<usize>,
}

/// A candidate move's ranked features: its class's local log-probability and rank, its fit to the
/// state, and the line's place, type and marker.
fn ranked_features(state: &State, line: &LineView, row: &[f32], local: &Local, candidate: &Move) -> [f32; RANKED] {
    let mut f = [0f32; RANKED];
    f[0] = local.probabilities[candidate.class()].max(1e-6).ln() as f32;
    f[1] = local.ranked.iter().position(|&c| c == candidate.class()).unwrap_or(CLASSES) as f32;
    f[2..2 + CANDIDATES].copy_from_slice(&state.features(line, candidate));
    f[2 + CANDIDATES..22 + CANDIDATES].copy_from_slice(&row[..20]);
    f[22 + CANDIDATES..RANKED].copy_from_slice(&row[72..92]);
    f
}

/// The classes a line's candidates come from: every class the local scores hold at `FLOOR` or above,
/// and those the state proposes.
fn classes_of(state: &State, line: &LineView, local: &Local) -> Vec<usize> {
    let mut classes: Vec<usize> = local.ranked.iter().copied().filter(|&c| local.probabilities[c] >= FLOOR).collect();
    for c in state.proposed(line) {
        if !classes.contains(&c) {
            classes.push(c);
        }
    }
    classes
}

/// Each line as the state reads it, in reading order.
fn views_of<'a>(pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, table: &'a LineTable, locals: &[Local]) -> Vec<LineView<'a>> {
    let grammar = settings().grammar.as_ref();
    // Each look's lines as the local scores read them, before any move.
    let heading = usize::from(HEADING) * 2 + 1;
    let mut looks: HashMap<u32, (f64, usize, usize)> = HashMap::new();
    for (k, local) in locals.iter().enumerate() {
        let entry = looks.entry(table.looks[k]).or_default();
        entry.0 += local.probabilities[heading];
        entry.1 += 1;
        entry.2 += usize::from(local.probabilities[heading] > 0.5);
    }
    // The openings the constituents reader names, and each line's place among its page's lines.
    let proposed: HashMap<&str, u8> = fences.starts.iter().map(|start| (start.line_id.as_str(), start.kind_code())).collect();
    let mut on_page: HashMap<usize, u32> = HashMap::new();
    (0..table.order.len()).map(|k| {
        let (sum, lines, likely) = looks[&table.looks[k]];
        let slot = table.order[k];
        let at = slot.index();
        let page_place = {
            let seen = on_page.entry(slots.page(slot)).or_default();
            *seen += 1;
            *seen - 1
        };
        let start_kind = proposed.get(slots.line(pages, slot).id.as_str()).copied().unwrap_or(0);
        LineView {
            translation: fences.translation[at],
            // The source's own document opens on its first line: a later line near its page's top, set as
            // a title, or one the reader names may open another.
            may_start: k > 0 && (page_place < 3 || start_kind != 0 || evidence.title_shaped[at]),
            start_kind,
            page_place,
            readings: &table.readings[k],
            label: evidence.label[at].map(|l| l.value),
            line_number: slots.line(pages, slot).text.trim().parse::<u32>().ok().filter(|_| evidence.line_number[at]),
            look: table.looks[k],
            references: &table.references[k],
            place: k as u32,
            grammar,
            heading_words: {
                let word = grammar::first_word(&slots.line(pages, slot).text, !table.readings[k].is_empty());
                let odds = |tables: &Tables| tables.heading_word(&word).unwrap_or(0.0);
                grammar.map_or([0.0; 2], |g| [odds(&g.instruments), odds(&g.others)])
            },
            operative: evidence.operative,
            look_heading: (sum / lines.max(1) as f64) as f32,
            look_heading_lines: (1.0 + likely as f32).ln(),
        }
    }).collect()
}

/// One document parsed: its lines read (`LineTable`) and the moves they make in reading order
/// (greedily, or by a beam of whole readings where LEGALPDF_TRANSITION_BEAM says how wide); and, as
/// its environment asks, what the scorer is fitted on and the labels the grader reads.
pub(crate) fn run(pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, runs: &Runs, sha256: &str) -> (LineTable, Vec<Local>, Vec<Move>) {
    let table = LineTable::read(pages, slots, evidence, fences, &runs.before);
    let locals = locals_of(&table, sha256);
    let moves = parse(pages, slots, evidence, fences, sha256, &table, &locals);
    (table, locals, moves)
}

/// Each line's local scores, read once.
fn locals_of(table: &LineTable, sha256: &str) -> Vec<Local> {
    let s = settings();
    let probabilities = |row: &[f32]| -> Vec<f64> {
        match &s.local {
            Some(LocalScores::One(forest)) => forest.probabilities(row),
            Some(LocalScores::Folds(folds)) => folds[fold_of(sha256, folds.len())].probabilities(row),
            None => vec![1.0 / CLASSES as f64; CLASSES],
        }
    };
    (0..table.order.len()).map(|k| {
        let probabilities = probabilities(table.row(k));
        let mut ranked: Vec<usize> = (0..probabilities.len()).collect();
        ranked.sort_by(|&a, &b| probabilities[b].total_cmp(&probabilities[a]).then(a.cmp(&b)));
        Local { probabilities, ranked }
    }).collect()
}

fn parse(pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, sha256: &str, table: &LineTable, locals: &[Local]) -> Vec<Move> {
    let s = settings();
    let n = table.order.len();
    let gold = s.gold.as_ref().and_then(|g| g.get(sha256));
    let ids: Vec<String> = table.order.iter().map(|&slot| slots.line(pages, slot).id.clone()).collect();
    let views = views_of(pages, slots, evidence, fences, table, locals);
    let gold_rows: Vec<[i64; 8]> = ids.iter().map(|id| gold.and_then(|g| g.get(&format!("{id}@0"))).copied().unwrap_or([-1; 8])).collect();
    let moves = match (&s.ranker, s.beam) {
        (Some(ranker), width) if width > 1 && s.dump.is_none() => beam(ranker, width, table, &views, locals),
        _ => {
            let mut state = State::default();
            let mut moves = Vec::with_capacity(n);
            let (mut move_rows, mut move_lines, mut move_classes, mut move_oracle) = (Vec::<f32>::new(), Vec::<u32>::new(), Vec::<u16>::new(), Vec::<u8>::new());
            for k in 0..n {
                let (line, local) = (&views[k], &locals[k]);
                let candidates = state.candidates(line, &classes_of(&state, line, local));
                let features: Vec<[f32; RANKED]> = candidates.iter().map(|m| ranked_features(&state, line, table.row(k), local, m)).collect();
                let best_oracle = gold.and_then(|_| oracle(&gold_rows, k, &state, line, &candidates));
                let choice = match &s.ranker {
                    Some(ranker) => {
                        let scores: Vec<f64> = features.iter().map(|f| move_score(ranker, f)).collect();
                        (0..candidates.len()).max_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(b.cmp(&a))).unwrap_or(0)
                    }
                    // No ranker: the oracle where the gold is read and reachable (the first round's
                    // rollout), else the local scores' best class, counting on nothing.
                    None => best_oracle.unwrap_or_else(|| {
                        candidates.iter().position(|m| m.class() == local.ranked[0] && m.reading.is_none() && m.note.is_none() && m.level <= 1 && !m.resume).unwrap_or(0)
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
        let labels = labelled(&views, &moves);
        let body = serde_json::json!({"sha256": sha256, "ids": ids.iter().map(|id| format!("{id}@0")).collect::<Vec<_>>(), "labels": labels});
        if let Err(error) = std::fs::create_dir_all(folder).and_then(|()| std::fs::write(folder.join(format!("{sha256}.labels.json")), body.to_string())) {
            eprintln!("transition parser labels {sha256}: {error}");
        }
    }
    moves
}

/// The document's structure as the transition parser reads it, its reading through `emit`: what
/// `derive` gives consumers in place of the stage parser's where LEGALPDF_TRANSITION_EMIT is set.
pub(crate) fn output(pages: &mut [Page], slots: &Slots, evidence: &Evidence, fences: &Fences, runs: &Runs, identity: &StructureIdentity) -> Result<StructureOutput> {
    let (table, locals, moves) = run(pages, slots, evidence, fences, runs, &identity.source_sha256);
    let views = views_of(pages, slots, evidence, fences, &table, &locals);
    let (structure_graph, paragraphs, footnotes) = emit::structure_graph(pages, slots, evidence, fences, &table, &views, &moves, identity)?;
    Ok(StructureOutput { paragraphs, footnotes, diagnostics: Vec::new(), structure_graph, heading_sources: HashMap::new() })
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
            let candidates = state.candidates(line, &classes_of(state, line, local));
            let scores: Vec<f64> = candidates.iter().map(|m| move_score(ranker, &ranked_features(state, line, table.row(k), local, m))).collect();
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
/// paired reference (a line a note answered), a heading's level step, a marker, no resumption, a body
/// unit's depth step.
fn labelled(views: &[LineView], moves: &[Move]) -> Vec<[i64; 8]> {
    let mut state = State::default();
    let mut labels: Vec<[i64; 8]> = Vec::with_capacity(moves.len());
    // The first line of the last unit of each region opened in each context's stream.
    let mut first_of: HashMap<(u16, usize, u8), usize> = HashMap::new();
    for (k, (line, chosen)) in views.iter().zip(moves).enumerate() {
        let stream = (state.ctx(line, chosen.starts), stream_of(chosen.region));
        let opened = chosen.opens || (!chosen.resume && state.open_regions.get(&stream) != Some(&chosen.region));
        let resumed = chosen.resume.then(|| first_of.get(&(stream.0, stream.1, chosen.region)).copied()).flatten();
        if opened {
            first_of.insert((stream.0, stream.1, chosen.region), k);
        }
        let starts_document = chosen.starts;
        let made = state.apply(line, chosen);
        if let Some((paired, ..)) = made.paired {
            labels[paired as usize][3] = 1;
        }
        let marked = chosen.opens && chosen.reading.is_some();
        let start = if chosen.opens { 1 } else if resumed.is_some() { 2 } else { 0 };
        let pointer = resumed.map_or(-1, |first| first as i64);
        labels.push([i64::from(chosen.region), start, i64::from(starts_document), 0, made.level_step, i64::from(marked), pointer, made.depth_step]);
    }
    labels
}

/// A candidate's score: its class's local log-probability, and the ranker's correction for its fit to the
/// state (fitted from the local score as its base).
fn move_score(ranker: &Forest, features: &[f32; RANKED]) -> f64 {
    f64::from(features[0]) + ranker.scores(features)[0]
}

/// Which of `folds` folds a source falls in: the CRC-32 of its sha256 (zlib's), modulo `folds`.
fn fold_of(sha256: &str, folds: usize) -> usize {
    let mut crc = 0xffff_ffffu32;
    for byte in sha256.bytes() {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xedb8_8320 } else { crc >> 1 };
        }
    }
    (!crc as usize) % folds.max(1)
}
