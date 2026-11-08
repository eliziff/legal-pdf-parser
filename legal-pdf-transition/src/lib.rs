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
//! Tier 0 is its reader with nothing fitted (`tier0`): the rules' readings its candidates, its scorer compiled in, chosen by
//! `Reader::Machine` in the request and reading none of the names below.
//!
//! It runs beside the stage parser while it is fitted and measured. What it reads is one value, `Settings`, handed
//! down from its entry points; `Settings::from_environment` fills that value for the harnesses, and nothing else here
//! reads the environment's LEGALPDF_TRANSITION_ names:
//! - LEGALPDF_TRANSITION_LOCAL, a local forest (multiclass over `Move::class`; the product's, fitted
//!   on all of train), or LEGALPDF_TRANSITION_LOCAL_FOLDS, a folder of k (fold0.txt ...), each fitted on
//!   every document but those of its fold, which read it (out of fold, train's rollouts); or
//!   LEGALPDF_TRANSITION_LOCAL_SCORES, a folder of each document's local probabilities from another
//!   tier's model (`<sha>.local.f32`, a line's `CLASSES` in reading order), or LEGALPDF_TRANSITION_ENCODER,
//!   a folder holding tier 1's or tier 2's document encoder (`encoder`), run in the engine;
//! - LEGALPDF_TRANSITION_DOC, a folder holding Detect-Order-Construct's stages as `docparse.py` exports
//!   them (`doc`): the document read by them in place of the walk; LEGALPDF_TRANSITION_DOC_ORACLE, the gold
//!   given: the document read by the same decoding from the heads' answers read off the gold (`doc::oracle`),
//!   what the decoding and the moves can hold; LEGALPDF_TRANSITION_DOC_ORDER=on, the reading's order read off the
//!   successor head; LEGALPDF_TRANSITION_DOC_TIMES, each document's stages' times;
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
//! - The parse's output is tier 0's (`tier0::output`) whatever the tier: LEGALPDF_TRANSITION_DOC has DOC score its moves.

use legal_pdf_core::model::{Page, PdfOutlineEntry};
use legal_pdf_decide::{Decision, Fences, Notes};
use legal_pdf_structure_model::evidence::{MarkerReading, WORD_SPEAKER};
use legal_pdf_structure_model::{Evidence, Runs, Slot, Slots};
use std::{collections::{HashMap, HashSet}, path::{Path, PathBuf}, sync::OnceLock};

pub use legal_pdf_machine::lines::*;
pub use legal_pdf_machine::machine::*;
mod trees;
use trees::Forest;
use legal_pdf_emit as emit;
pub use legal_pdf_machine::grammar::{self, Tables};
mod stylesheet;
pub use stylesheet::*;
mod joint;
#[cfg(feature = "encoder")]
#[macro_use]
mod encoder;
mod doc;
pub use legal_pdf_tier0 as tier0;

/// A class the local scores hold below this probability competes only where the state proposes it: a
/// guard on the parse's cost, not a judgement.
const FLOOR: f64 = 1e-4;
/// The region-and-opening classes.
const CLASSES: usize = 34;
/// The gold region of a line a gold reads only as body prose, a quotation or list item set in it unmarked
/// (`gold_labels.py`, A2AJ's alignment).
const UNMARKED_BODY: i64 = 101;
/// A ranked candidate's features: its class's local log-probability and rank, its `CANDIDATE`
/// features, and the line's place, type and marker (`lines::LINE_FEATURES`'s first twenty and its marker's).
pub const RANKED: usize = 2 + CANDIDATES + 40 + 1;
/// The heading levels a local model's level distribution covers (0 unused).
pub const LEVEL_CLASSES: usize = 10;

/// The gold's labels by document: each piece's row, each piece's reading rank where the gold gives one, and the pieces
/// in the order of the rows, which a row's pointer counts.
type GoldLabels = (HashMap<String, HashMap<String, [i64; 8]>>, HashMap<String, HashMap<String, i64>>, HashMap<String, Vec<String>>);

enum LocalScores {
    One(Forest),
    /// Out of fold: a document read by the forest fitted on every fold but its own (`fold_of`).
    Folds(Vec<Forest>),
    /// Each document's probabilities as another tier's model read them, from this folder.
    Precomputed(PathBuf),
    #[cfg(feature = "encoder")]
    Encoder(encoder::Encoder),
}

/// What the transition parser is given: its models, the gold where it is fitted or measured, and what it writes.
pub struct Settings {
    local: Option<LocalScores>,
    gold: Option<GoldLabels>,
    dump: Option<PathBuf>,
    dump_lines: f64,
    labels: Option<PathBuf>,
    #[cfg(feature = "encoder")]
    doc: Option<doc::Reader>,
    doc_oracle: bool,
    grammar: Option<grammar::Grammar>,
    pub confusions: HashMap<char, Vec<(char, f32)>>,
}

/// The harnesses' settings, read once: the entry points' only source of them.
fn settings() -> &'static Settings {
    static SETTINGS: OnceLock<Settings> = OnceLock::new();
    SETTINGS.get_or_init(Settings::from_environment)
}

/// What a model read from the environment's settings holds, its error written where it could not be read.
fn loaded<T>(read: Result<T, String>) -> Option<T> {
    read.map_err(|e| eprintln!("transition parser: {e}")).ok()
}

impl Settings {
    /// The settings the LEGALPDF_TRANSITION_ names in the environment give.
    fn from_environment() -> Self {
        let var = |name: &str| std::env::var_os(name).map(PathBuf::from);
        let forest = |path: PathBuf| loaded(Forest::read(&path, ALL, CLASSES));
        let local = match (var("LEGALPDF_TRANSITION_LOCAL"), var("LEGALPDF_TRANSITION_LOCAL_FOLDS")) {
            #[cfg(feature = "encoder")]
            _ if var("LEGALPDF_TRANSITION_ENCODER").is_some() => var("LEGALPDF_TRANSITION_ENCODER").and_then(|dir| loaded(encoder::Encoder::load(&dir))).map(LocalScores::Encoder),
            _ if var("LEGALPDF_TRANSITION_LOCAL_SCORES").is_some() => var("LEGALPDF_TRANSITION_LOCAL_SCORES").map(LocalScores::Precomputed),
            (Some(path), _) => forest(path).map(LocalScores::One),
            (None, Some(folder)) => Some((0..).map(|k| folder.join(format!("fold{k}.txt"))).take_while(|path| path.exists()).filter_map(forest).collect::<Vec<Forest>>())
                .filter(|folds| !folds.is_empty()).map(LocalScores::Folds),
            (None, None) => None,
        };
        Settings {
            local,
            gold: var("LEGALPDF_TRANSITION_GOLD").map(|path| read_gold(&path)),
            dump: var("LEGALPDF_TRANSITION_DUMP"),
            dump_lines: std::env::var("LEGALPDF_TRANSITION_DUMP_LINES").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0),
            labels: var("LEGALPDF_TRANSITION_LABELS"),
            doc_oracle: std::env::var_os("LEGALPDF_TRANSITION_DOC_ORACLE").is_some(),
            #[cfg(feature = "encoder")]
            doc: var("LEGALPDF_TRANSITION_DOC").and_then(|dir| loaded(doc::Reader::load(&dir, std::env::var("LEGALPDF_TRANSITION_DOC_ORDER").is_ok_and(|v| v == "on"),
                std::env::var_os("LEGALPDF_TRANSITION_DOC_TIMES").is_some()))),
            grammar: var("LEGALPDF_TRANSITION_GRAMMAR").and_then(|path| loaded(grammar::Grammar::read(&path))),
            confusions: var("LEGALPDF_TRANSITION_CONFUSIONS").map(|path| read_confusions(&path)).unwrap_or_default(),
        }
    }
}

/// Whether the transition parser runs on this parse.
pub fn enabled() -> bool {
    settings().dump.is_some() || settings().labels.is_some()
}

/// Little-endian 32-bit floats, four bytes each.
fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
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

/// The gold's labels, from rows of the grader's predictions format ({sha256, ids, labels}): each document's by piece
/// (`line@offset`), each piece's reading rank, the labels' ninth column where a row has one, and its pieces in row order.
fn read_gold(path: &Path) -> GoldLabels {
    let mut found = GoldLabels::default();
    let Ok(text) = std::fs::read_to_string(path) else {
        eprintln!("transition parser: no gold at {}", path.display());
        return found;
    };
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(row) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        let (Some(sha), Some(ids), Some(labels)) = (row["sha256"].as_str(), row["ids"].as_array(), row["labels"].as_array()) else { continue };
        let (mut per_piece, mut ranks) = (HashMap::new(), HashMap::new());
        for (id, label) in ids.iter().zip(labels) {
            let piece = id.as_str().unwrap_or_default().to_owned();
            let cell = |k: usize| label.get(k).and_then(serde_json::Value::as_i64);
            if let Some(rank) = cell(8).filter(|&r| r >= 0) { ranks.insert(piece.clone(), rank); }
            per_piece.insert(piece, std::array::from_fn::<i64, 8, _>(|k| cell(k).unwrap_or(-1)));
        }
        found.0.insert(sha.to_owned(), per_piece);
        found.1.insert(sha.to_owned(), ranks);
        found.2.insert(sha.to_owned(), ids.iter().filter_map(|id| id.as_str().map(str::to_owned)).collect());
    }
    found
}

/// The regions a gold row's region allows: a gold that does not tell its text from a quotation or a list item set
/// in it (A2AJ's, its body prose unmarked: `UNMARKED_BODY`) takes any of the three.
fn gold_regions(region: &i64) -> &[i64] {
    if *region == UNMARKED_BODY { &[1, 4, 2] } else { std::slice::from_ref(region) }
}

/// The move the gold makes at a line, among `candidates`: of those of the gold's region and opening,
/// the one whose result, the state replayed one move on, the gold agrees with most: a heading's level
/// step, a body unit's depth step, whether the unit opens on its marker, a note answering a reference
/// on a line the gold pairs. None where no candidate is of the gold's region and opening.
fn oracle(gold: &[[i64; 8]], k: usize, state: &State, line: &LineView, candidates: &[Move]) -> Option<usize> {
    let here = &gold[k];
    let (region, opens) = (here[0], here[1] == 1);
    if region < 0 { return None; }
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
        if m.opens && RANKED_BODY.contains(&m.region) && here[7] > 0 { wrong += (made.depth_step - here[7]).unsigned_abs() as u32 * 3; }
        if m.opens && m.region != NOTE { wrong += 2 * u32::from(m.reading.is_some() != (here[5] == 1)); }
        if !m.opens { wrong += 2 * u32::from(m.resume != (here[1] == 2)); }
        // A note counts on its number where the gold's note prints one (its marker), and on none where it
        // prints none.
        if m.opens && m.region == NOTE {
            wrong += match (m.note, made.paired, here[5] == 1) {
                (None, _, false) => 0,
                (Some(_), _, false) => 2,
                (Some(_), Some((line, ..)), true) if gold.get(line as usize).is_some_and(|g| g[3] == 1) => 0,
                (Some(_), _, true) => 1,
                (None, _, true) => 3,
            };
        }
        wrong
    };
    let regions = gold_regions(&region);
    // Ties go to the depth the numbering gives (a gold without depths: A2AJ's), then to the plain move.
    let tie = |m: &Move| (m.depth > 0 && usize::from(m.depth) != state.stack_depth(line, m), m.reading.is_some());
    (0..candidates.len())
        .filter(|&c| regions.contains(&i64::from(candidates[c].region)) && candidates[c].opens == opens)
        .min_by_key(|&c| (disagreements(&candidates[c]), tie(&candidates[c]), c))
}

/// A line's local scores: each region-and-opening class's probability, and the classes best first.
pub struct Local {
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
    let golden = gold.filter(|g| g[0] >= 0).map(|g| gold_regions(&g[0]).iter().map(|&r| r as usize * 2 + usize::from(g[1] == 1)).collect::<Vec<_>>());
    state.proposed(line).into_iter().chain(golden.into_iter().flatten()).for_each(|c| if c < CLASSES && !classes.contains(&c) { classes.push(c); });
    classes
}

/// Each line as the state reads it, in reading order: tier 0's view (`tier0::line_views`) with what the walk measures.
fn views_of<'a>(s: &'a Settings, pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, table: &'a LineTable, locals: &[Local], sheet: &Stylesheet) -> Vec<LineView<'a>> {
    let grammar = s.grammar.as_ref();
    // Each look's lines as the local scores read them, before any move.
    let heading = usize::from(HEADING) * 2 + 1;
    let mut looks: HashMap<u32, (f64, usize, usize)> = HashMap::new();
    for (k, local) in locals.iter().enumerate() {
        let (p, entry) = (local.probabilities[heading], looks.entry(table.looks[k]).or_default());
        *entry = (entry.0 + p, entry.1 + 1, entry.2 + usize::from(p > 0.5));
    }
    // The openings the constituents reader names, and each line's place among its page's lines.
    let proposed: HashMap<&str, u8> = fences.starts.iter().map(|start| (start.line_id.as_str(), start.kind_code())).collect();
    let mut on_page: HashMap<usize, u32> = HashMap::new();
    tier0::line_views(pages, slots, evidence, fences, table).into_iter().enumerate().map(|(k, view)| {
        let (sum, lines, likely) = looks[&table.looks[k]];
        let slot = table.order[k];
        let seen = on_page.entry(slots.page(slot)).or_default();
        *seen += 1;
        let text = &slots.line(pages, slot).text;
        let word = grammar::first_word(text, !table.readings[k].is_empty());
        LineView {
            // The source's own document opens on its first line; any later line's opening may open another
            // (a letter mid-page, the next message of an email chain).
            may_start: k > 0,
            start_kind: proposed.get(slots.line(pages, slot).id.as_str()).copied().unwrap_or(0),
            page_place: *seen - 1,
            grammar,
            heading_words: grammar.map_or([0.0; 2], |g| [&g.instruments, &g.others].map(|tables: &Tables| tables.heading_word(&word).unwrap_or(0.0))),
            look_heading: (sum / lines.max(1) as f64) as f32,
            look_heading_lines: (1.0 + likely as f32).ln(),
            template: *sheet.of(k),
            anchor: sheet.anchor[k],
            listed: sheet.listed[k],
            regime: sheet.regime[k],
            title_key: if text.chars().count() <= 160 { legal_pdf_headings::headings::bookmarks::normalize_title(text) } else { String::new() },
            contents_title: sheet.contents_titles[k].clone(),
            left: table.row(k)[feature("x0")],
            ..view
        }
    }).collect()
}

/// The walk, where it is asked for: the moves the DOC tier reads of the document, where its reader is given
/// (LEGALPDF_TRANSITION_DOC), with each line's marker readings the moves' `reading` names; and what the scorer is fitted
/// on and the grader's labels, where those are asked (`enabled`). None where DOC is not given. The output is tier 0's
/// all the same (`tier0::output`).
///
/// The walk over tier 0's reading of a document (its notes and decision, read once by `tier0::output`): its lines read
/// (`LineTable`, the decision among their evidence), its stylesheet (`Stylesheet`), and the moves they make in reading
/// order (greedily, or by a beam of whole readings where LEGALPDF_TRANSITION_BEAM says how wide), read twice: a first
/// greedy reading, whose reading of each template the second reads; and, as its environment asks, what the scorer is
/// fitted on and the labels the grader reads.
#[allow(clippy::too_many_arguments)]
pub fn over_tier0(pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, runs: &Runs, outline: &[PdfOutlineEntry],
    sha256: &str, notes: &Notes, decision: &Decision) -> Option<(Vec<Move>, Vec<Vec<MarkerReading>>)> {
    let s = settings();
    #[cfg(feature = "encoder")]
    let doc = s.doc.is_some();
    #[cfg(not(feature = "encoder"))]
    let doc = false;
    if !doc && !enabled() { return None; }
    let table = LineTable::read(pages, slots, evidence, fences, notes, decision, &runs.before, &s.confusions);
    let locals = locals_of(s, pages, slots, &table, sha256);
    let mut sheet = Stylesheet::read(pages, slots, evidence, fences, outline, &table, &locals);
    let moves = parse(s, pages, slots, evidence, fences, sha256, &table, &locals, &mut sheet);
    doc.then(|| (moves, table.readings))
}

/// Each line's local scores, read once.
#[cfg_attr(not(feature = "encoder"), allow(unused_variables))]
fn locals_of(s: &Settings, pages: &[Page], slots: &Slots, table: &LineTable, sha256: &str) -> Vec<Local> {
    let n = table.order.len();
    // Another tier's probabilities, where they cover the document's lines.
    let read = |path: PathBuf, width: usize| -> Option<Vec<f32>> {
        let fits = std::fs::read(&path).ok().map(|bytes| f32s(&bytes)).filter(|values| values.len() == n * width);
        if fits.is_none() { eprintln!("transition parser: no local scores for {} lines at {}", n, path.display()); }
        fits
    };
    let (precomputed, levels): (Option<Vec<f32>>, Option<Vec<f32>>) = match &s.local {
        #[cfg(feature = "encoder")]
        Some(LocalScores::Encoder(model)) => {
            let texts: Vec<&str> = table.order.iter().map(|&slot| slots.line(pages, slot).text.as_str()).collect();
            model.probabilities(table, &texts).map_or_else(|e| {
                eprintln!("transition parser {sha256}: {e}");
                (None, None)
            }, |(classes, levels)| (Some(classes), Some(levels)))
        }
        Some(LocalScores::Precomputed(folder)) => (read(folder.join(format!("{sha256}.local.f32")), CLASSES), read(folder.join(format!("{sha256}.levels.f32")), LEVEL_CLASSES)),
        _ => (None, None),
    };
    (0..n).map(|k| {
        let probabilities = match (&s.local, &precomputed) {
            (Some(LocalScores::One(forest)), _) => forest.probabilities(table.row(k)),
            (Some(LocalScores::Folds(folds)), _) => folds[fold_of(sha256, folds.len())].probabilities(table.row(k)),
            (_, Some(precomputed)) => precomputed[k * CLASSES..(k + 1) * CLASSES].iter().map(|&p| f64::from(p)).collect(),
            _ => vec![1.0 / CLASSES as f64; CLASSES],
        };
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
    let mut state = State::default();
    let mut moves = Vec::with_capacity(views.len());
    let mut recorded = Recorded::default();
    for k in 0..views.len() {
        let (line, local) = (&views[k], &locals[k]);
        let candidates = state.candidates(line, &classes_of(&state, line, local, gold.map(|g| &g[k])));
        let features: Vec<[f32; RANKED]> = candidates.iter().map(|m| ranked_features(&state, line, table.row(k), local, m)).collect();
        let best_oracle = gold.and_then(|gold| oracle(gold, k, &state, line, &candidates));
        let choice = best_oracle.unwrap_or_else(|| candidates.iter().position(|m| m.class() == local.ranked[0] && m.reading.is_none() && m.note.is_none() && m.level <= 1
            && !m.resume && !m.starts && (m.depth == 0 || usize::from(m.depth) == state.stack_depth(line, m))).unwrap_or(0));
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
fn parse(s: &Settings, pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, sha256: &str, table: &LineTable, locals: &[Local], sheet: &mut Stylesheet) -> Vec<Move> {
    let n = table.order.len();
    let gold = s.gold.as_ref().and_then(|g| g.0.get(sha256));
    // Each line's reading rank in the gold, -1 for none.
    let ranks_of = || -> Vec<i64> {
        let ranks = s.gold.as_ref().and_then(|g| g.1.get(sha256));
        table.order.iter().map(|&slot| ranks.and_then(|r| r.get(&format!("{}@0", slots.line(pages, slot).id))).copied().unwrap_or(-1)).collect()
    };
    let ids: Vec<String> = table.order.iter().map(|&slot| slots.line(pages, slot).id.clone()).collect();
    // A row's pointer names a row of the gold's pieces (a line may hold several): here, the place of that piece's line.
    let place: HashMap<&str, i64> = ids.iter().enumerate().map(|(k, id)| (id.as_str(), k as i64)).collect();
    let pieces = s.gold.as_ref().and_then(|g| g.2.get(sha256));
    let gold_rows: Vec<[i64; 8]> = ids.iter().map(|id| {
        let mut row = gold.and_then(|g| g.get(&format!("{id}@0"))).copied().unwrap_or([-1; 8]);
        if row[6] >= 0 {
            row[6] = pieces.and_then(|pieces| pieces.get(row[6] as usize)).and_then(|piece| place.get(piece.split('@').next().unwrap_or(piece))).copied().unwrap_or(-1);
        }
        row
    }).collect();
    let views = views_of(s, pages, slots, evidence, fences, table, locals, sheet);
    let texts: Vec<&str> = table.order.iter().map(|&slot| slots.line(pages, slot).text.as_str()).collect();
    let line_pages: Vec<usize> = table.order.iter().map(|&slot| slots.page(slot)).collect();
    // Detect-Order-Construct's decoding from the gold's answers: what it can hold.
    let read = (s.doc_oracle && gold.is_some()).then(|| doc::oracle(&gold_rows, &ranks_of(), &views, &texts, &line_pages));
    // Detect-Order-Construct's reading, where its stages are given.
    #[cfg(feature = "encoder")]
    let read = read.or_else(|| s.doc.as_ref().and_then(|reader| {
        reader.read(table, &views, &texts, &line_pages, &doc::Style::of(sheet, table)).map_err(|error| eprintln!("transition parser DOC {sha256}: {error}")).ok()
    }));
    if let Some(moves) = read {
        write_labels(s, sha256, &ids, &views, &moves);
        return moves;
    }
    // The first reading, never the oracle's (the gold would read into the second's view of the
    // templates); each template then carries how it read.
    sheet.read_back(&walk(table, &views, locals, None, None).0);
    let views = views_of(s, pages, slots, evidence, fences, table, locals, sheet);
    let (moves, recorded) = walk(table, &views, locals, gold.map(|_| &gold_rows[..]), s.dump.is_some().then_some((sha256, s.dump_lines)));
    if let Some(folder) = &s.dump {
        let write = || -> std::io::Result<()> {
            std::fs::create_dir_all(folder)?;
            let bytes = |values: &[f32]| values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
            std::fs::write(folder.join(format!("{sha256}.lines.f32")), bytes(&table.features))?;
            std::fs::write(folder.join(format!("{sha256}.moves.f32")), bytes(&recorded.rows))?;
            let meta = serde_json::json!({
                "sha256": sha256, "ids": ids, "lines": n, "line_width": ALL, "move_width": RANKED,
                "texts": texts, "looks": &table.looks, "pages": line_pages, "references": &table.references, "ranks": ranks_of(), "templates": &sheet.template,
                "template_rows": joint::template_rows(sheet, table), "held": joint::held_templates(sheet),
                "openers": (0..n).map(|k| opener(table, k)).collect::<Vec<_>>(), "anchors": &sheet.anchor, "t0": &table.t0,
                "note_numbers": (0..n).map(|k| doc::note_number(&views[k], texts[k]).map_or(-1, i64::from)).collect::<Vec<_>>(),
                "regions": REGIONS, "line_features": LINE_FEATURES, "candidate_features": CANDIDATE,
                "gold": gold_rows, "move_lines": recorded.lines, "move_classes": recorded.classes, "move_oracle": recorded.oracle,
                "chosen": moves.iter().map(Move::class).collect::<Vec<_>>(),
            });
            std::fs::write(folder.join(format!("{sha256}.json")), serde_json::to_vec(&meta)?)
        };
        write().unwrap_or_else(|error| eprintln!("transition parser dump {sha256}: {error}"));
    }
    write_labels(s, sha256, &ids, &views, &moves);
    moves
}

/// The moves as the grader's labels, written to LEGALPDF_TRANSITION_LABELS (`<sha>.labels.json`) where set.
fn write_labels(s: &Settings, sha256: &str, ids: &[String], views: &[LineView], moves: &[Move]) {
    let Some(folder) = &s.labels else { return };
    let body = serde_json::json!({"sha256": sha256, "ids": ids.iter().map(|id| format!("{id}@0")).collect::<Vec<_>>(), "labels": labelled(views, moves)});
    std::fs::create_dir_all(folder).and_then(|()| std::fs::write(folder.join(format!("{sha256}.labels.json")), body.to_string()))
        .unwrap_or_else(|error| eprintln!("transition parser labels {sha256}: {error}"));
}

/// The moves as the grader's labels, the state replayed over them: region, opening, document start,
/// paired reference (a line a note answered), a heading's level step, a marker, no resumption, a body
/// unit's depth step.
fn labelled(views: &[LineView], moves: &[Move]) -> Vec<[i64; 8]> {
    let mut state = State::default();
    let mut labels: Vec<[i64; 8]> = Vec::with_capacity(moves.len());
    let mut paired_lines: Vec<usize> = Vec::new();
    let mut streams = Streams::default();
    for (k, (line, chosen)) in views.iter().zip(moves).enumerate() {
        let key = (state.ctx(line, chosen.starts), stream_of(chosen.region));
        let stands = streams.stands(key, chosen);
        if chosen.region != 0 { streams.step(key, chosen.region, stands, k); }
        let made = state.apply(line, chosen);
        // A mark may stand after its note: its line is labelled once every line is.
        paired_lines.extend(made.paired.into_iter().chain(made.also.into_iter().flatten()).map(|(line, ..)| line as usize));
        let (start, pointer) = match stands {
            Stands::Resumes(first) => (2, first as i64),
            _ => (i64::from(chosen.opens), -1),
        };
        let marked = chosen.opens && chosen.reading.is_some();
        labels.push([i64::from(chosen.region), start, i64::from(chosen.starts), 0, made.level_step, i64::from(marked), pointer, made.depth_step]);
    }
    paired_lines.into_iter().for_each(|line| labels[line][3] = 1);
    labels
}

/// Which of `folds` folds a source falls in: the CRC-32 of its sha256 (zlib's), modulo `folds`.
fn fold_of(sha256: &str, folds: usize) -> usize {
    let mut crc = 0xffff_ffffu32;
    for byte in sha256.bytes() {
        crc ^= u32::from(byte);
        (0..8).for_each(|_| crc = if crc & 1 != 0 { (crc >> 1) ^ 0xedb8_8320 } else { crc >> 1 });
    }
    (!crc as usize) % folds.max(1)
}
