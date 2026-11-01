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
//!   every document but those of its fold, which read it (out of fold, train's rollouts); or
//!   LEGALPDF_TRANSITION_LOCAL_SCORES, a folder of each document's local probabilities from another
//!   tier's model (`<sha>.local.f32`, a line's `CLASSES` in reading order), or LEGALPDF_TRANSITION_ENCODER,
//!   a folder holding tier 1's or tier 2's document encoder (`encoder`), run in the engine;
//! - LEGALPDF_TRANSITION_RANKER, a ranker forest: its score, added to a candidate's local log-probability,
//!   from the candidate's `RANKED` features;
//! - LEGALPDF_TRANSITION_BEAM, how many whole readings a beam keeps (1, greedy, by default);
//! - LEGALPDF_TRANSITION_JOINT, `weight,penalty`: the reading the joint stylesheet solve makes (`joint`),
//!   the template prior's term weighted `weight` and a move against its template's role costing
//!   `penalty`; LEGALPDF_TRANSITION_PRIOR, the template prior's forest (none: the roles weighed on the
//!   reading alone);
//! - LEGALPDF_TRANSITION_GRAMMAR, the grammar tables counted over legal text (`grammar`);
//! - LEGALPDF_TRANSITION_CONFUSIONS, how recognition misreads the characters of a line's opening marker,
//!   counted over the gold's corrected lines ({read: {meant: share}}): a recognized line's marker is
//!   read too as each misreading would have it;
//! - LEGALPDF_TRANSITION_GOLD, the gold's labels (the grader's predictions format, rows' gold),
//!   for the oracle;
//! - LEGALPDF_TRANSITION_DUMP, a folder: each line's features (`<sha>.lines.f32`), each candidate's
//!   (`<sha>.moves.f32`) and which the oracle makes (`<sha>.json`), the candidates of the share of lines
//!   LEGALPDF_TRANSITION_DUMP_LINES draws (all by default; none for a dump only the forests and encoders
//!   read: the ranker's fit draws no more than a quarter of a round's lines);
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
mod stylesheet;
pub(crate) use stylesheet::*;
mod joint;
#[cfg(feature = "encoder")]
mod encoder;

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// A class the local scores hold below this probability competes only where the state proposes it: a
/// guard on the parse's cost, not a judgement.
const FLOOR: f64 = 1e-4;
/// The region-and-opening classes.
const CLASSES: usize = 34;
/// The gold region of a line a gold reads only as body prose, a quotation or list item set in it unmarked
/// (`gold_labels.py`, A2AJ's alignment).
const UNMARKED_BODY: i64 = 101;
/// The origin the transition parser's nodes name.
pub(crate) const ORIGIN: &str = "legal-pdf-transition.v1";
/// A ranked candidate's features: its class's local log-probability and rank, its `CANDIDATE`
/// features, and the line's place, type and marker (`lines::LINE_FEATURES`'s first twenty and its marker's).
pub(crate) const RANKED: usize = 2 + CANDIDATES + 40 + 1;
/// The heading levels a local model's level distribution covers (0 unused).
pub(crate) const LEVEL_CLASSES: usize = 10;

enum LocalScores {
    One(Forest),
    /// Out of fold: a document read by the forest fitted on every fold but its own (`fold_of`).
    Folds(Vec<Forest>),
    /// Each document's probabilities as another tier's model read them, from this folder.
    Precomputed(PathBuf),
    #[cfg(feature = "encoder")]
    Encoder(encoder::Encoder),
}

struct Settings {
    local: Option<LocalScores>,
    ranker: Option<Forest>,
    gold: Option<HashMap<String, HashMap<String, [i64; 8]>>>,
    dump: Option<PathBuf>,
    dump_lines: f64,
    labels: Option<PathBuf>,
    beam: usize,
    joint: Option<(f64, f64)>,
    prior: Option<Forest>,
    emit: bool,
    grammar: Option<grammar::Grammar>,
    pub(crate) confusions: HashMap<char, Vec<(char, f32)>>,
}

fn settings() -> &'static Settings {
    static SETTINGS: OnceLock<Settings> = OnceLock::new();
    SETTINGS.get_or_init(|| {
        let var = |name: &str| std::env::var_os(name).map(PathBuf::from);
        let forest = |path: PathBuf, width: usize, classes: usize| Forest::read(&path, width, classes).map_err(|e| eprintln!("transition parser: {e}")).ok();
        let local = match (var("LEGALPDF_TRANSITION_LOCAL"), var("LEGALPDF_TRANSITION_LOCAL_FOLDS")) {
            #[cfg(feature = "encoder")]
            _ if var("LEGALPDF_TRANSITION_ENCODER").is_some() => var("LEGALPDF_TRANSITION_ENCODER")
                .and_then(|dir| encoder::Encoder::load(&dir).map_err(|e| eprintln!("transition parser: {e}")).ok()).map(LocalScores::Encoder),
            _ if var("LEGALPDF_TRANSITION_LOCAL_SCORES").is_some() => var("LEGALPDF_TRANSITION_LOCAL_SCORES").map(LocalScores::Precomputed),
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
            dump_lines: std::env::var("LEGALPDF_TRANSITION_DUMP_LINES").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0),
            labels: var("LEGALPDF_TRANSITION_LABELS"),
            beam: std::env::var("LEGALPDF_TRANSITION_BEAM").ok().and_then(|w| w.parse().ok()).unwrap_or(1),
            joint: std::env::var("LEGALPDF_TRANSITION_JOINT").ok().and_then(|v| {
                let (weight, penalty) = v.split_once(',')?;
                Some((weight.trim().parse().ok()?, penalty.trim().parse().ok()?))
            }),
            prior: var("LEGALPDF_TRANSITION_PRIOR").and_then(|path| forest(path, joint::PRIOR, joint::ROLES)),
            emit: std::env::var_os("LEGALPDF_TRANSITION_EMIT").is_some(),
            grammar: var("LEGALPDF_TRANSITION_GRAMMAR").and_then(|path| grammar::Grammar::read(&path).map_err(|e| eprintln!("transition parser: {e}")).ok()),
            confusions: var("LEGALPDF_TRANSITION_CONFUSIONS").map(|path| read_confusions(&path)).unwrap_or_default(),
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

/// The confusion table at `path`: each misread character's meant ones, likeliest first.
fn read_confusions(path: &Path) -> HashMap<char, Vec<(char, f32)>> {
    let table: HashMap<String, HashMap<String, f32>> = std::fs::read_to_string(path).ok().and_then(|text| serde_json::from_str(&text).ok()).unwrap_or_else(|| {
        eprintln!("transition parser: no confusion table at {}", path.display());
        HashMap::new()
    });
    table.into_iter().filter_map(|(read, meant)| {
        let mut meant: Vec<(char, f32)> = meant.into_iter().filter_map(|(c, p)| Some((c.chars().next()?, p))).collect();
        meant.sort_by(|a, b| b.1.total_cmp(&a.1));
        Some((read.chars().next()?, meant))
    }).collect()
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
                let outline = state.outlines.get(&state.outline_key(line, m.starts));
                let looked = outline.and_then(|o| o.look_levels.iter().find(|(look, _)| *look == line.look)).map(|&(_, l)| l);
                let preferred = looked.unwrap_or_else(|| outline.map_or(1, |o| o.heading.max(1)));
                wrong += u32::from((m.level - preferred).unsigned_abs());
            }
        }
        // A document the gold opens here, or none it does not.
        wrong += 4 * u32::from(m.starts != (here[2] == 1));
        // A body unit's depth weighs above its marker: the gold's nesting is the print's.
        if m.opens && RANKED_BODY.contains(&m.region) && here[7] > 0 {
            wrong += (made.depth_step - here[7]).unsigned_abs() as u32 * 3;
        }
        if m.opens && m.region != NOTE {
            wrong += 2 * u32::from(m.reading.is_some() != (here[5] == 1));
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
    // A gold that does not tell its text from a quotation or a list item set in it (A2AJ's, its body
    // prose unmarked: `UNMARKED_BODY`) takes any of the three.
    let regions: &[i64] = if region == UNMARKED_BODY { &[1, 4, 2] } else { std::slice::from_ref(&region) };
    // Ties go to the depth the numbering gives (a gold without depths: A2AJ's), then to the plain move.
    let tie = |m: &Move| (m.depth > 0 && usize::from(m.depth) != state.stack_depth(line, m), m.reading.is_some());
    (0..candidates.len())
        .filter(|&c| regions.contains(&i64::from(candidates[c].region)) && candidates[c].opens == opens)
        .min_by_key(|&c| (disagreements(&candidates[c]), tie(&candidates[c]), c))
}

/// A line's local scores: each region-and-opening class's probability, and the classes best first.
pub(crate) struct Local {
    probabilities: Vec<f64>,
    ranked: Vec<usize>,
    /// Where the local model says it (tier 1's and tier 2's encoders): each heading level's
    /// probability were the line a heading's opening.
    levels: Option<Vec<f32>>,
}

/// A candidate move's ranked features: its class's local log-probability and rank, its fit to the
/// state, and the line's place, type and marker.
fn ranked_features(state: &State, line: &LineView, row: &[f32], local: &Local, candidate: &Move) -> [f32; RANKED] {
    let mut f = [0f32; RANKED];
    f[0] = local.probabilities[candidate.class()].max(1e-6).ln() as f32;
    f[1] = local.ranked.iter().position(|&c| c == candidate.class()).unwrap_or(CLASSES) as f32;
    f[2..2 + CANDIDATES].copy_from_slice(&state.features(line, candidate));
    f[2 + CANDIDATES..22 + CANDIDATES].copy_from_slice(&row[..20]);
    f[22 + CANDIDATES..42 + CANDIDATES].copy_from_slice(&row[72..92]);
    // The heading's level by the local model's own reading of it.
    f[42 + CANDIDATES] = match &local.levels {
        Some(levels) if candidate.opens && candidate.region == HEADING => levels[usize::from(candidate.level.max(0) as u8).min(LEVEL_CLASSES - 1)].max(1e-6).ln(),
        _ => -99.0,
    };
    f
}

/// The classes a line's candidates come from: every class the local scores hold at `FLOOR` or above,
/// and those the state proposes; where the gold is given (`gold`, the line's row), its class too, so
/// the oracle always has the gold's move and the ranker sees it among the candidates.
fn classes_of(state: &State, line: &LineView, local: &Local, gold: Option<&[i64; 8]>) -> Vec<usize> {
    let mut classes: Vec<usize> = local.ranked.iter().copied().filter(|&c| local.probabilities[c] >= FLOOR).collect();
    let golden = gold.filter(|g| g[0] >= 0).map(|g| {
        let regions: &[i64] = if g[0] == UNMARKED_BODY { &[1, 4, 2] } else { std::slice::from_ref(&g[0]) };
        regions.iter().map(|&r| r as usize * 2 + usize::from(g[1] == 1)).collect::<Vec<_>>()
    });
    for c in state.proposed(line).into_iter().chain(golden.into_iter().flatten()) {
        if c < CLASSES && !classes.contains(&c) {
            classes.push(c);
        }
    }
    classes
}

/// Each line as the state reads it, in reading order.
fn views_of<'a>(pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, table: &'a LineTable, locals: &[Local], sheet: &Stylesheet) -> Vec<LineView<'a>> {
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
            // The source's own document opens on its first line; any later line's opening may open another
            // (a letter mid-page, the next message of an email chain).
            may_start: k > 0,
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
            template: *sheet.of(k),
            anchor: sheet.anchor[k],
            listed: sheet.listed[k],
            regime: sheet.regime[k],
            title_key: {
                let text = &slots.line(pages, slot).text;
                if text.chars().count() <= 160 { crate::structure::headings::bookmarks::normalize_title(text) } else { String::new() }
            },
            contents_title: sheet.contents_titles[k].clone(),
            page: slots.page(slot) as u32,
            left: table.row(k)[LINE_FEATURES.iter().position(|n| *n == "x0").unwrap_or(0)],
        }
    }).collect()
}

/// One document parsed: its lines read (`LineTable`), its stylesheet (`Stylesheet`), and the moves
/// they make in reading order (greedily, or by a beam of whole readings where LEGALPDF_TRANSITION_BEAM
/// says how wide), read twice: a first greedy reading, whose reading of each template the second reads;
/// and, as its environment asks, what the scorer is fitted on and the labels the grader reads.
pub(crate) fn run(pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, runs: &Runs, outline: &[PdfOutlineEntry], sha256: &str) -> (LineTable, Vec<Local>, Stylesheet, Vec<Move>) {
    let table = LineTable::read(pages, slots, evidence, fences, &runs.before);
    let locals = locals_of(pages, slots, &table, sha256);
    let mut sheet = Stylesheet::read(pages, slots, evidence, fences, outline, &table, &locals);
    let moves = parse(pages, slots, evidence, fences, sha256, &table, &locals, &mut sheet);
    (table, locals, sheet, moves)
}

/// Each line's local scores, read once.
#[cfg_attr(not(feature = "encoder"), allow(unused_variables))]
fn locals_of(pages: &[Page], slots: &Slots, table: &LineTable, sha256: &str) -> Vec<Local> {
    let s = settings();
    let n = table.order.len();
    // Another tier's probabilities, where they cover the document's lines.
    let read = |path: PathBuf, width: usize| -> Option<Vec<f32>> {
        let values = std::fs::read(&path).ok().map(|bytes| bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect::<Vec<f32>>());
        let fits = values.filter(|values| values.len() == n * width);
        if fits.is_none() {
            eprintln!("transition parser: no local scores for {} lines at {}", n, path.display());
        }
        fits
    };
    let (precomputed, levels): (Option<Vec<f32>>, Option<Vec<f32>>) = match &s.local {
        #[cfg(feature = "encoder")]
        Some(LocalScores::Encoder(model)) => {
            let texts: Vec<&str> = table.order.iter().map(|&slot| slots.line(pages, slot).text.as_str()).collect();
            match model.probabilities(table, &texts) {
                Ok((classes, levels)) => (Some(classes), Some(levels)),
                Err(e) => {
                    eprintln!("transition parser {sha256}: {e}");
                    (None, None)
                }
            }
        }
        Some(LocalScores::Precomputed(folder)) => (read(folder.join(format!("{sha256}.local.f32")), CLASSES), read(folder.join(format!("{sha256}.levels.f32")), LEVEL_CLASSES)),
        _ => (None, None),
    };
    let probabilities = |k: usize, row: &[f32]| -> Vec<f64> {
        match &s.local {
            Some(LocalScores::One(forest)) => forest.probabilities(row),
            Some(LocalScores::Folds(folds)) => folds[fold_of(sha256, folds.len())].probabilities(row),
            _ if precomputed.is_some() => {
                precomputed.as_ref().unwrap()[k * CLASSES..(k + 1) * CLASSES].iter().map(|&p| f64::from(p)).collect()
            }
            _ => vec![1.0 / CLASSES as f64; CLASSES],
        }
    };
    (0..n).map(|k| {
        let probabilities = probabilities(k, table.row(k));
        let mut ranked: Vec<usize> = (0..probabilities.len()).collect();
        ranked.sort_by(|&a, &b| probabilities[b].total_cmp(&probabilities[a]).then(a.cmp(&b)));
        let levels = levels.as_ref().map(|levels| levels[k * LEVEL_CLASSES..(k + 1) * LEVEL_CLASSES].to_vec());
        Local { probabilities, ranked, levels }
    }).collect()
}

/// What a greedy walk recorded for the scorer's fit: each candidate's ranked features, its line, its
/// class and whether the oracle makes it.
#[derive(Default)]
struct Recorded {
    rows: Vec<f32>,
    lines: Vec<u32>,
    classes: Vec<u16>,
    oracle: Vec<u8>,
}

/// Whether a document's line `k` is among the share `share` of its lines a fixed draw takes (CRC-32 of the
/// sha256 and the line's place).
fn drawn(sha256: &str, k: usize, share: f64) -> bool {
    share >= 1.0 || (share > 0.0 && (fold_of(&format!("{sha256}:{k}"), 1 << 20) as f64) < share * f64::from(1u32 << 20))
}

/// One greedy reading of `views`: each line's move the ranker's best; without a ranker, the oracle's
/// where `gold` is given and reachable (the first round's rollout), else the local scores' best class,
/// counting on nothing. With `record` (the document's sha256 and the share of its lines
/// drawn), each drawn line's candidates as the scorer is fitted on them.
fn walk(table: &LineTable, views: &[LineView], locals: &[Local], gold: Option<&[[i64; 8]]>, record: Option<(&str, f64)>) -> (Vec<Move>, Recorded) {
    let s = settings();
    let mut state = State::default();
    let mut moves = Vec::with_capacity(views.len());
    let mut recorded = Recorded::default();
    for k in 0..views.len() {
        let (line, local) = (&views[k], &locals[k]);
        let candidates = state.candidates(line, &classes_of(&state, line, local, gold.map(|g| &g[k])));
        let features: Vec<[f32; RANKED]> = candidates.iter().map(|m| ranked_features(&state, line, table.row(k), local, m)).collect();
        let best_oracle = gold.and_then(|gold| oracle(gold, k, &state, line, &candidates));
        let choice = match &s.ranker {
            Some(ranker) => {
                let scores: Vec<f64> = features.iter().map(|f| move_score(ranker, f)).collect();
                (0..candidates.len()).max_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(b.cmp(&a))).unwrap_or(0)
            }
            None => best_oracle.unwrap_or_else(|| {
                candidates.iter().position(|m| m.class() == local.ranked[0] && m.reading.is_none() && m.note.is_none() && m.level <= 1 && !m.resume && !m.starts
                    && (m.depth == 0 || usize::from(m.depth) == state.stack_depth(line, m))).unwrap_or(0)
            }),
        };
        if record.is_some_and(|(sha256, share)| drawn(sha256, k, share)) {
            for (c, f) in features.iter().enumerate() {
                recorded.rows.extend_from_slice(f);
                recorded.lines.push(k as u32);
                recorded.classes.push(candidates[c].class() as u16);
                recorded.oracle.push(u8::from(Some(c) == best_oracle));
            }
        }
        state.apply(line, &candidates[choice]);
        moves.push(candidates[choice]);
    }
    (moves, recorded)
}

#[allow(clippy::too_many_arguments)]
fn parse(pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, sha256: &str, table: &LineTable, locals: &[Local], sheet: &mut Stylesheet) -> Vec<Move> {
    let s = settings();
    let n = table.order.len();
    let gold = s.gold.as_ref().and_then(|g| g.get(sha256));
    let ids: Vec<String> = table.order.iter().map(|&slot| slots.line(pages, slot).id.clone()).collect();
    let gold_rows: Vec<[i64; 8]> = ids.iter().map(|id| gold.and_then(|g| g.get(&format!("{id}@0"))).copied().unwrap_or([-1; 8])).collect();
    // The first reading, never the oracle's (the gold would read into the second's view of the
    // templates); each template then carries how it read.
    let first = walk(table, &views_of(pages, slots, evidence, fences, table, locals, sheet), locals, None, None).0;
    sheet.read_back(&first);
    let views = views_of(pages, slots, evidence, fences, table, locals, sheet);
    let moves = match (&s.ranker, s.beam) {
        (Some(ranker), _) if s.dump.is_none() && s.joint.is_some() => {
            let (weight, penalty) = s.joint.unwrap_or_default();
            joint::solve(ranker, s.prior.as_ref(), weight, penalty, table, &views, locals, sheet)
        }
        (Some(ranker), width) if width > 1 && s.dump.is_none() => beam(ranker, width, table, &views, locals),
        _ => {
            let (moves, recorded) = walk(table, &views, locals, gold.map(|_| &gold_rows[..]), s.dump.is_some().then_some((sha256, s.dump_lines)));
            let (move_rows, move_lines, move_classes, move_oracle) = (recorded.rows, recorded.lines, recorded.classes, recorded.oracle);
            if let Some(folder) = &s.dump {
                let write = || -> std::io::Result<()> {
                    std::fs::create_dir_all(folder)?;
                    let bytes = |values: &[f32]| values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
                    std::fs::write(folder.join(format!("{sha256}.lines.f32")), bytes(&table.features))?;
                    std::fs::write(folder.join(format!("{sha256}.moves.f32")), bytes(&move_rows))?;
                    let meta = serde_json::json!({
                        "sha256": sha256, "ids": ids, "lines": n, "line_width": ALL, "move_width": RANKED,
                        "texts": table.order.iter().map(|&slot| slots.line(pages, slot).text.as_str()).collect::<Vec<_>>(),
                        "looks": &table.looks,
                        "templates": &sheet.template,
                        "template_rows": joint::template_rows(sheet, table), "held": joint::held_templates(sheet),
                        "openers": (0..n).map(|k| opener(table, k)).collect::<Vec<_>>(),
                        "anchors": &sheet.anchor,
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
pub(crate) fn output(pages: &mut [Page], slots: &Slots, evidence: &Evidence, fences: &Fences, runs: &Runs, outline: &[PdfOutlineEntry], identity: &StructureIdentity) -> Result<StructureOutput> {
    let (table, locals, sheet, moves) = run(pages, slots, evidence, fences, runs, outline, &identity.source_sha256);
    let views = views_of(pages, slots, evidence, fences, &table, &locals, &sheet);
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
            let candidates = state.candidates(line, &classes_of(state, line, local, None));
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
        for (paired, ..) in made.paired.into_iter().chain(made.also.into_iter().flatten()) {
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
