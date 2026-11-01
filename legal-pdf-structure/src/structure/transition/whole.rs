//! The document read whole: one reading of every line chosen at once, nothing claimed by a first pass.
//! Five parts read the document, each over all of it and solved exactly, and are made to agree on what
//! they share by dual decomposition:
//! - the lines' roles: every line's region and whether it opens a unit, by Viterbi over the whole
//!   document, the state the region open in the body's stream and the one before it (a paragraph may
//!   go on past a quotation) and whether a note is open in the notes' stream;
//! - the numbers: every line-opening marker reading is a candidate; a candidate continues another of
//!   its family whose value is one less, earlier in the document, each with one successor at most and
//!   one predecessor at most, each family's value pair (v, v + 1) its own exact assignment; the lines
//!   that open a constituent document, each worth its evidence less the links it would cut; each chain
//!   one role for all its members (a paragraph's, a provision's, a list item's, a quotation's, a
//!   heading's, another unit's number), the role worth the most over its members and links;
//! - the notes: every reference mark is answered by at most one note label of its number, on its page
//!   or the two after; a note may answer several marks;
//! - the outline: which lines open headings and at what level, by dynamic programming over the lines
//!   that may, on the open heading's level, a document's start opening none;
//! - the body: which lines open body units and at what depth, by dynamic programming over the lines that
//!   may, on the open depth, as the outline's levels are read.
//!
//! What they share: a line opens a unit of a region on its marker exactly where its number counts in
//! that region's role; a line opens a note exactly where its label is a note's; a line opens a heading
//! exactly where the outline holds one; a line opens a body unit exactly where the body part gives it a
//! depth. The prices on those facts move until the parts agree (or the rounds run out: the roles' regions
//! then stand, the numbers deciding the markers, the notes the notes' numbers, the outline and the body the
//! roles' headings' levels and body units' depths). The reading is the moves the emit turns into the
//! consumer structure.
//!
//! Its scores come from six learned heads (`Heads`, LEGALPDF_TRANSITION_HEADS, forests `fit_whole.py`
//! fits on the gold): a number's role, a link between two numbers, a mark's pair with a note label, a
//! heading's level given the open one's, a body unit's depth given the open one's, a line's opening a
//! constituent document; each over features the
//! whole document's candidates give (how far a number's family runs before and after it, how many of it
//! the document holds, what stands between two numbers or a mark and a label, a heading's template over
//! the document, the printed contents entry it answers and that entry's tier), never over a reading. How much each part's scores weigh against the others' is learned
//! on whole documents (`train`, LEGALPDF_TRANSITION_WHOLE_WEIGHTS): each training document's whole reading
//! solved with a cost on every line read otherwise than the gold reads it, and the weights moved toward
//! the gold reading's totals and away from the solved reading's (a structured perceptron, its steps
//! scaled per weight by the steps before, averaged over every step). Before training, each problem is
//! solved pulled hard toward the gold's reading, and how much of the gold the solve can make is reported:
//! the architecture's ceiling on the training documents.
//!
//! Where LEGALPDF_TRANSITION_WHOLE_DUMP names a folder and the gold is given, each document's problem
//! (`<sha>.problem.json`: every candidate and its heads' scores, the gold reading's totals) and the heads'
//! training rows (`<sha>.whole.json`: every candidate's features and the gold's answer) are written there.

use super::*;
use serde::{Deserialize, Serialize};

/// The local scores' classes a line is read among: those at this probability or more, and the ones the
/// numbers and notes need, at most this many by the local scores.
const KEEP: f64 = 1e-3;
const MOST: usize = 10;
/// The parts' scores without heads (log-odds).
const LINK: f32 = 3.0;
const PAIR: f32 = 3.0;
const ROUNDS: usize = 40;
/// The body regions a body continuation goes on in (the state's region slots), 0 for none.
const SLOTS: usize = 17;

/// A number's roles, the heads' classes: none, and the openings it may number.
pub(crate) const ROLES: usize = 7;
pub(crate) const NUMBER_WIDTH: usize = 68;
pub(crate) const LINK_WIDTH: usize = 48;
pub(crate) const PAIR_WIDTH: usize = 22;
pub(crate) const OUTLINE_WIDTH: usize = 55;
pub(crate) const START_WIDTH: usize = 32;
pub(crate) const DEPTH_WIDTH: usize = 53;

/// The weights of the parts' scores against each other, in this order: the local scores' log-probability,
/// a paragraph resumed past another unit, a numbered opening on its marker, a counted number's role, a
/// link, a pair, a note opened, a heading's level, a start's log-odds, a start, a heading opened, a body
/// unit's depth.
pub(crate) const WEIGHTS: usize = 12;
const W_LOCAL: usize = 0;
const W_RESUME: usize = 1;
const W_MARKED: usize = 2;
const W_NUMBER: usize = 3;
const W_LINK: usize = 4;
const W_PAIR: usize = 5;
const W_NOTE: usize = 6;
const W_LEVEL: usize = 7;
const W_START: usize = 8;
const W_START_BIAS: usize = 9;
const W_HEADING: usize = 10;
const W_DEPTH: usize = 11;
/// The weights before any training.
pub(crate) const PRIOR_WEIGHTS: [f64; WEIGHTS] = [1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, -2.0, 0.0, 1.0];

/// The role a region's opening gives its number.
fn role_of(region: u8) -> usize {
    match region {
        1 => 1,
        16 => 2,
        2 => 3,
        4 => 4,
        3 => 5,
        r if numbered(r) => 6,
        _ => 0,
    }
}

fn numbered(region: u8) -> bool {
    [1u8, 2, 3, 4, 7, 8, 10, 14, 16].contains(&region)
}

/// The learned heads (`fit_whole.py`).
pub(crate) struct Heads {
    number: Forest,
    link: Forest,
    pair: Forest,
    outline: Forest,
    start: Forest,
    depth: Forest,
}

impl Heads {
    pub(crate) fn read(folder: &Path) -> Result<Self, String> {
        Ok(Self {
            number: Forest::read(&folder.join("number.txt"), NUMBER_WIDTH, ROLES)?,
            link: Forest::read(&folder.join("link.txt"), LINK_WIDTH, 1)?,
            pair: Forest::read(&folder.join("pair.txt"), PAIR_WIDTH, 1)?,
            outline: Forest::read(&folder.join("outline.txt"), OUTLINE_WIDTH, LEVEL_CLASSES)?,
            start: Forest::read(&folder.join("start.txt"), START_WIDTH, 1)?,
            depth: Forest::read(&folder.join("depth.txt"), DEPTH_WIDTH, LEVEL_CLASSES)?,
        })
    }
}

/// The trained weights at `path` (a JSON array, `train`'s), else the prior ones.
pub(crate) fn read_weights(path: &Path) -> Result<[f64; WEIGHTS], String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let values: Vec<f64> = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    values.try_into().map_err(|_| format!("{}: not {WEIGHTS} weights", path.display()))
}

/// A line-opening number: its line, its reading's index there, family and value.
#[derive(Clone, Debug)]
struct Number {
    line: usize,
    reading: usize,
    family: String,
    value: u32,
}

/// A document as the solve reads it: every line's choices (a class, opening on a marker or not, the local
/// scores' log-probability), every number (line, reading, family, value) with its role head's log-odds
/// of each role against none, every link and pair with its head's log-odds, the note labels (line,
/// number, page) and reference marks (line, place on it, number, page), each line's levels' log-odds
/// given the open level where it may open a heading, each line's start log-odds where it may open a
/// constituent document; and, for training, each line's gold class and the gold reading's totals.
#[derive(Clone, Default, Serialize, Deserialize)]
pub(crate) struct Problem {
    choices: Vec<Vec<(u16, bool, f32)>>,
    numbers: Vec<(u32, u8, u32, u32)>,
    unary: Vec<[f32; ROLES]>,
    links: Vec<(u32, u32, f32)>,
    labels: Vec<(u32, u32, u32)>,
    marks: Vec<(u32, u16, u32, u32)>,
    pairs: Vec<(u32, u16, u32, f32)>,
    levels: Vec<Vec<[f32; LEVEL_CLASSES]>>,
    depths: Vec<Vec<[f32; LEVEL_CLASSES]>>,
    starts: Vec<Option<f32>>,
    gold: Vec<i16>,
    gold_totals: Option<[f64; WEIGHTS]>,
    gold_reading: Option<GoldReading>,
}

/// The gold reading in the solve's terms, for the reachability check: each line's counted role (0 none),
/// heading level and body depth (0 none), start and note.
#[derive(Clone, Default, Serialize, Deserialize)]
pub(crate) struct GoldReading {
    roles: Vec<u8>,
    levels: Vec<u8>,
    depths: Vec<u8>,
    starts: Vec<bool>,
    notes: Vec<bool>,
}

/// A whole reading: each line's choice (class, on its marker), each line's counted number (reading,
/// role), the links kept, the lines that open a constituent document, the pairs chosen, the labels that
/// open notes, each line's heading level.
#[derive(Clone, Debug, Default)]
pub(crate) struct Solution {
    path: Vec<(u16, bool)>,
    line_choice: Vec<Option<(u8, u8)>>,
    links: Vec<usize>,
    starts: Vec<bool>,
    pairs: Vec<usize>,
    notes: Vec<bool>,
    levels: Vec<Option<i8>>,
    depths: Vec<Option<u8>>,
    chain_of: Vec<Option<usize>>,
}

/// The body stream's state through a reading: the region open in it and the one before it, a note open.
fn step_state(region: u8, opens: bool, body: usize, before: usize, note: bool) -> Option<(usize, usize, bool, bool)> {
    let stream = stream_of(region);
    if stream == 0 && region != 0 {
        let r = usize::from(region);
        if opens || r == body {
            Some(if r == body { (r, before, note, false) } else { (r, body, note, false) })
        } else if r == before {
            Some((r, body, note, true))
        } else {
            None
        }
    } else if stream == 1 {
        (opens || note).then_some((body, before, true, false))
    } else {
        Some((body, before, note, false))
    }
}

/// The lines' roles under the prices: the best reading by Viterbi, each line's choice; with `cost`, a
/// unit more for every line read otherwise than the gold reads it (training's loss-augmented reading).
fn roles(problem: &Problem, w: &[f64; WEIGHTS], role_price: &[[f64; ROLES]], note_price: &[f64], heading_price: &[f64], body_price: &[f64], cost: bool, toward: f64) -> Vec<(u16, bool)> {
    let n = problem.choices.len();
    let state_of = |body: usize, before: usize, note: bool| (body * SLOTS + before) * 2 + usize::from(note);
    let states = SLOTS * SLOTS * 2;
    let mut score = vec![f64::NEG_INFINITY; states];
    score[state_of(0, 0, false)] = 0.0;
    let mut back: Vec<Vec<(u32, u16)>> = Vec::with_capacity(n);
    for (k, options) in problem.choices.iter().enumerate() {
        let mut next = vec![f64::NEG_INFINITY; states];
        let mut from = vec![(u32::MAX, 0u16); states];
        let gains: Vec<f64> = options.iter().map(|&(class, marked, local)| {
            let region = (class / 2) as u8;
            let opens = class % 2 == 1;
            let mut gain = w[W_LOCAL] * f64::from(local);
            if marked {
                gain += w[W_MARKED] + role_price[k][role_of(region)];
            }
            if opens && region == NOTE {
                gain += note_price[k];
            }
            if opens && region == HEADING {
                gain += w[W_HEADING] + heading_price[k];
            }
            if opens && RANKED_BODY.contains(&region) {
                gain += body_price[k];
            }
            if cost && problem.gold.get(k).is_some_and(|&g| g >= 0 && g != class as i16) {
                gain += 1.0;
            }
            // The reachability check's pull toward the gold's reading.
            if toward > 0.0 && problem.gold.get(k).is_some_and(|&g| g == class as i16) {
                let marked_gold = problem.gold_reading.as_ref().is_some_and(|g| g.roles[k] > 0);
                gain += if marked == marked_gold { toward } else { toward / 2.0 };
            }
            gain
        }).collect();
        for s in 0..states {
            if score[s] == f64::NEG_INFINITY {
                continue;
            }
            let (body, before, note) = (s / 2 / SLOTS, s / 2 % SLOTS, s % 2 == 1);
            for (c, &(class, _, _)) in options.iter().enumerate() {
                let Some((b, f, o, resumed)) = step_state((class / 2) as u8, class % 2 == 1, body, before, note) else { continue };
                let total = score[s] + gains[c] + if resumed { w[W_RESUME] } else { 0.0 };
                let target = state_of(b, f, o);
                if total > next[target] {
                    next[target] = total;
                    from[target] = (s as u32, c as u16);
                }
            }
        }
        back.push(from);
        score = next;
    }
    let mut s = (0..states).max_by(|&a, &b| score[a].total_cmp(&score[b])).unwrap_or(0);
    let mut path = vec![(0u16, false); n];
    for k in (0..n).rev() {
        let (previous, c) = back[k][s];
        let (class, marked, _) = problem.choices[k][if previous == u32::MAX { 0 } else { usize::from(c) }];
        path[k] = (class, marked);
        if previous != u32::MAX {
            s = previous as usize;
        }
    }
    path
}

/// The numbers under the prices, exactly in each piece: every family's value pair (v, v + 1) an assignment
/// of links by their worth; the document's starts, each worth its evidence less the links it would cut
/// (a start cuts every link across it); then each chain one role for all its members, the role whose
/// members' worth (the number head's log-odds of the role against none, and the price on it) and links'
/// worth sum the most, the chain counting only where that sum is positive; a number on no chain counts
/// in its own best role where that is worth something; a line counts once.
fn numbers_part(problem: &Problem, w: &[f64; WEIGHTS], price: &[[f64; ROLES]], toward: f64) -> (Vec<Option<(u8, u8)>>, Vec<usize>, Vec<bool>, Vec<Option<usize>>) {
    let numbers = &problem.numbers;
    let gold = problem.gold_reading.as_ref().filter(|_| toward > 0.0);
    let price: Vec<[f64; ROLES]> = price.iter().enumerate().map(|(k, p)| {
        let mut p = *p;
        if let Some(role) = gold.map(|g| usize::from(g.roles[k])).filter(|&r| r > 0) {
            p[role] += toward;
        }
        p
    }).collect();
    let n_lines = problem.choices.len();
    let mut by: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (i, &(_, _, family, value)) in numbers.iter().enumerate() {
        by.entry((family, value)).or_default().push(i);
    }
    let link_at: HashMap<(usize, usize), usize> = problem.links.iter().enumerate().map(|(j, &(p, s, _))| ((p as usize, s as usize), j)).collect();
    let worth = |p: usize, s: usize| link_at.get(&(p, s)).map_or(f64::NEG_INFINITY, |&j| w[W_LINK] * f64::from(problem.links[j].2));
    let mut kept: Vec<(usize, usize)> = Vec::new();
    for (&(family, value), succs) in &by {
        let Some(preds) = by.get(&(family, value.wrapping_sub(1))).filter(|_| value > 1) else { continue };
        kept.extend(assignment(preds, succs, |p, s| (numbers[p].0 < numbers[s].0).then(|| worth(p, s)).filter(|v| *v > 0.0)));
    }
    let mut starts = vec![false; n_lines];
    for k in 1..n_lines {
        let Some(odds) = problem.starts[k] else { continue };
        let evidence = w[W_START] * f64::from(odds) + w[W_START_BIAS] + gold.map_or(0.0, |g| if g.starts[k] { toward } else { -toward });
        let cut: f64 = kept.iter().filter(|&&(p, s)| (numbers[p].0 as usize) < k && k <= numbers[s].0 as usize).map(|&(p, s)| worth(p, s)).sum();
        starts[k] = evidence - cut > 0.0;
    }
    kept.retain(|&(p, s)| !(numbers[p].0 as usize + 1..=numbers[s].0 as usize).any(|k| starts[k]));
    // Chains: the links' components, each one role.
    let mut first: Vec<usize> = (0..numbers.len()).collect();
    fn root(first: &mut [usize], i: usize) -> usize {
        let mut r = i;
        while first[r] != r {
            r = first[r];
        }
        let mut at = i;
        while first[at] != r {
            let next = first[at];
            first[at] = r;
            at = next;
        }
        r
    }
    for &(p, s) in &kept {
        let (a, b) = (root(&mut first, p), root(&mut first, s));
        if a != b {
            first[a.max(b)] = a.min(b);
        }
    }
    let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..numbers.len() {
        let r = root(&mut first, i);
        members.entry(r).or_default().push(i);
    }
    let mut link_total: HashMap<usize, f64> = HashMap::new();
    for &(p, s) in &kept {
        *link_total.entry(root(&mut first, p)).or_default() += worth(p, s);
    }
    let gain = |i: usize, role: usize| w[W_NUMBER] * f64::from(problem.unary[i][role]) + price[numbers[i].0 as usize][role];
    let mut role_of_number: Vec<Option<usize>> = vec![None; numbers.len()];
    let mut chain_of: Vec<Option<usize>> = vec![None; numbers.len()];
    for (&r, group) in &members {
        let links_worth = link_total.get(&r).copied().unwrap_or(0.0);
        let (best, total) = (1..ROLES).map(|role| (role, group.iter().map(|&i| gain(i, role)).sum::<f64>() + links_worth))
            .max_by(|a, b| a.1.total_cmp(&b.1)).unwrap_or((0, 0.0));
        if total > 0.0 {
            let head = *group.iter().min_by_key(|&&i| numbers[i].0).unwrap_or(&r);
            for &i in group {
                role_of_number[i] = Some(best);
                chain_of[i] = Some(head);
            }
        }
    }
    let mut line_choice: Vec<Option<(u8, u8)>> = vec![None; n_lines];
    let mut line_best = vec![f64::NEG_INFINITY; n_lines];
    let mut chosen_number: Vec<Option<usize>> = vec![None; n_lines];
    for (i, &(line, reading, _, _)) in numbers.iter().enumerate() {
        if let Some(role) = role_of_number[i] {
            let value = gain(i, role);
            if value > line_best[line as usize] {
                line_best[line as usize] = value;
                line_choice[line as usize] = Some((reading, role as u8));
                chosen_number[line as usize] = Some(i);
            }
        }
    }
    let counted: HashSet<usize> = chosen_number.iter().flatten().copied().collect();
    let links: Vec<usize> = kept.iter().filter(|(p, s)| counted.contains(p) && counted.contains(s)).filter_map(|key| link_at.get(key).copied()).collect();
    for i in 0..numbers.len() {
        if !counted.contains(&i) {
            chain_of[i] = None;
        }
    }
    (line_choice, links, starts, chain_of)
}

/// The notes under the prices: each reference mark to its best note label (its pair's worth, positive),
/// a label opening a note where its worth (a note's weight, its price, its pairs') is positive.
fn notes_part(problem: &Problem, w: &[f64; WEIGHTS], price: &[f64], toward: f64) -> (Vec<usize>, Vec<bool>) {
    let gold = problem.gold_reading.as_ref().filter(|_| toward > 0.0);
    let n = problem.choices.len();
    let mut best: HashMap<(u32, u16), (usize, f64)> = HashMap::new();
    for (j, &(line, index, _, worth)) in problem.pairs.iter().enumerate() {
        let worth = w[W_PAIR] * f64::from(worth);
        if worth > 0.0 && best.get(&(line, index)).is_none_or(|&(_, b)| worth > b) {
            best.insert((line, index), (j, worth));
        }
    }
    let mut gained = vec![0.0; n];
    for &(j, worth) in best.values() {
        gained[problem.pairs[j].2 as usize] += worth;
    }
    let mut notes = vec![false; n];
    for &(line, _, _) in &problem.labels {
        let k = line as usize;
        notes[k] = w[W_NOTE] + price[k] + gained[k] + gold.map_or(0.0, |g| if g.notes[k] { toward } else { -toward }) > 0.0;
    }
    let mut pairs: Vec<usize> = best.values().map(|&(j, _)| j).filter(|&j| notes[problem.pairs[j].2 as usize]).collect();
    pairs.sort_unstable();
    (pairs, notes)
}

/// The outline under the prices, exactly: over the lines that may open a heading, in order, which do and
/// at what level, by dynamic programming on the open heading's level (none after a document's start); a
/// heading's worth its price and its level's log-odds given the open one's. Each line's level, where it
/// opens a heading.
fn outline_part(problem: &Problem, w: &[f64; WEIGHTS], starts: &[bool], price: &[f64], toward: f64) -> Vec<Option<i8>> {
    let gold = problem.gold_reading.as_ref().filter(|_| toward > 0.0).map(|g| &g.levels[..]);
    ladder(&problem.levels, w[W_LEVEL], starts, price, gold, toward).into_iter().map(|l| l.map(|l| l as i8)).collect()
}

/// The body units' depths under the prices, exactly, as the outline's levels are read: over the lines that
/// may open a body unit, which do and at what depth, on the open depth (none after a document's start).
fn depth_part(problem: &Problem, w: &[f64; WEIGHTS], starts: &[bool], price: &[f64], toward: f64) -> Vec<Option<u8>> {
    let gold = problem.gold_reading.as_ref().filter(|_| toward > 0.0).map(|g| &g.depths[..]);
    ladder(&problem.depths, w[W_DEPTH], starts, price, gold, toward).into_iter().map(|d| d.map(|d| d as u8)).collect()
}

/// A ladder read whole: over the lines `table` holds scores for (each step's log-odds given the open step,
/// by open step), which take a step and which, by dynamic programming on the open step, a start opening
/// none; a step's worth its line's price and its weighted log-odds (and, for the reachability check, a pull
/// toward `gold`'s steps).
fn ladder(table: &[Vec<[f32; LEVEL_CLASSES]>], weight: f64, starts: &[bool], price: &[f64], gold: Option<&[u8]>, toward: f64) -> Vec<Option<usize>> {
    let n = price.len();
    let mut score = vec![f64::NEG_INFINITY; LEVEL_CLASSES];
    score[0] = 0.0;
    let mut back: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
    for k in 0..n {
        if starts[k] {
            let best = score.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let from = (0..LEVEL_CLASSES).max_by(|&a, &b| score[a].total_cmp(&score[b])).unwrap_or(0);
            score = vec![f64::NEG_INFINITY; LEVEL_CLASSES];
            score[0] = best;
            back.push((k, (0..LEVEL_CLASSES).map(|_| (from, usize::MAX)).collect()));
        }
        let Some(levels) = table.get(k).filter(|levels| !levels.is_empty()) else { continue };
        let pull = |level: usize| gold.map_or(0.0, |g| if usize::from(g[k]) == level { toward } else if g[k] > 0 { toward / 2.0 } else { -toward });
        let mut next = score.clone();
        let mut from: Vec<(usize, usize)> = (0..LEVEL_CLASSES).map(|s| (s, 0)).collect();
        for open in 0..LEVEL_CLASSES {
            if score[open] == f64::NEG_INFINITY {
                continue;
            }
            for level in 1..LEVEL_CLASSES {
                let total = score[open] + price[k] + weight * f64::from(levels[open][level]) + pull(level);
                if total > next[level] {
                    next[level] = total;
                    from[level] = (open, level);
                }
            }
        }
        back.push((k, from));
        score = next;
    }
    let mut out = vec![None; n];
    let mut s = (0..LEVEL_CLASSES).max_by(|&a, &b| score[a].total_cmp(&score[b])).unwrap_or(0);
    for (k, from) in back.iter().rev() {
        let (previous, level) = from[s];
        if level != usize::MAX && level > 0 {
            out[*k] = Some(level);
        }
        s = previous;
    }
    out
}

/// The best one-to-one pairing of `left` with `right` by `worth` (None: no edge): exactly by search over
/// the right's choices where both sets are small, else each edge best first.
fn assignment(left: &[usize], right: &[usize], worth: impl Fn(usize, usize) -> Option<f64>) -> Vec<(usize, usize)> {
    let edges: Vec<Vec<Option<f64>>> = left.iter().map(|&l| right.iter().map(|&r| worth(l, r)).collect()).collect();
    if left.len() <= 8 && right.len() <= 8 {
        fn search(at: usize, edges: &[Vec<Option<f64>>], used: &mut Vec<bool>, current: &mut Vec<Option<usize>>, best: &mut (f64, Vec<Option<usize>>), total: f64) {
            if at == edges.len() {
                if total > best.0 {
                    *best = (total, current.clone());
                }
                return;
            }
            current.push(None);
            search(at + 1, edges, used, current, best, total);
            current.pop();
            for (r, edge) in edges[at].iter().enumerate() {
                if let (Some(w), false) = (edge, used[r]) {
                    used[r] = true;
                    current.push(Some(r));
                    search(at + 1, edges, used, current, best, total + w);
                    current.pop();
                    used[r] = false;
                }
            }
        }
        let mut best = (0.0, vec![None; left.len()]);
        search(0, &edges, &mut vec![false; right.len()], &mut Vec::new(), &mut best, 0.0);
        return best.1.iter().enumerate().filter_map(|(l, r)| r.map(|r| (left[l], right[r]))).collect();
    }
    let mut all: Vec<(f64, usize, usize)> = edges.iter().enumerate()
        .flat_map(|(l, row)| row.iter().enumerate().filter_map(move |(r, w)| w.map(|w| (w, l, r)))).collect();
    all.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (mut left_used, mut right_used) = (vec![false; left.len()], vec![false; right.len()]);
    let mut out = Vec::new();
    for (_, l, r) in all {
        if !left_used[l] && !right_used[r] {
            left_used[l] = true;
            right_used[r] = true;
            out.push((left[l], right[r]));
        }
    }
    out
}

/// The document's whole reading under weights `w`: the five parts made to agree by dual decomposition;
/// with `cost`, the loss-augmented reading training compares the gold's with; with `toward`, every part
/// pulled that much toward the gold's reading (the reachability check: what of the gold the solve can make).
pub(crate) fn solve(problem: &Problem, w: &[f64; WEIGHTS], cost: bool, toward: f64) -> Solution {
    let n = problem.choices.len();
    let mut role_price = vec![[0f64; ROLES]; n];
    let mut note_price = vec![0.0; n];
    let mut heading_price = vec![0.0; n];
    let mut body_price = vec![0.0; n];
    let negated = |prices: &[[f64; ROLES]]| prices.iter().map(|p| p.map(|x| -x)).collect::<Vec<_>>();
    let flip = |prices: &[f64]| prices.iter().map(|p| -p).collect::<Vec<_>>();
    let may_head: Vec<bool> = (0..n).map(|k| problem.levels.get(k).is_some_and(|l| !l.is_empty())).collect();
    let may_body: Vec<bool> = (0..n).map(|k| problem.depths.get(k).is_some_and(|l| !l.is_empty())).collect();
    let is_label: Vec<bool> = {
        let mut out = vec![false; n];
        for &(line, _, _) in &problem.labels {
            out[line as usize] = true;
        }
        out
    };
    let mut solution = Solution::default();
    for round in 0..ROUNDS {
        let path = roles(problem, w, &role_price, &note_price, &heading_price, &body_price, cost, toward);
        let (line_choice, links, starts, chain_of) = numbers_part(problem, w, &negated(&role_price), toward);
        let (pairs, notes) = notes_part(problem, w, &flip(&note_price), toward);
        let levels = outline_part(problem, w, &starts, &flip(&heading_price), toward);
        let depths = depth_part(problem, w, &starts, &flip(&body_price), toward);
        let step = 1.0 / (1.0 + round as f64);
        let mut agree = true;
        for k in 0..n {
            let (class, marked) = path[k];
            let (region, opens) = ((class / 2) as u8, class % 2 == 1);
            let said = if marked { role_of(region) } else { 0 };
            let counted = line_choice[k].map_or(0, |(_, role)| usize::from(role));
            if said != counted {
                agree = false;
                if counted > 0 {
                    role_price[k][counted] += step;
                }
                if said > 0 {
                    role_price[k][said] -= step;
                }
            }
            if is_label[k] && (opens && region == NOTE) != notes[k] {
                agree = false;
                note_price[k] += step * if notes[k] { 1.0 } else { -1.0 };
            }
            if may_head[k] && (opens && region == HEADING) != levels[k].is_some() {
                agree = false;
                heading_price[k] += step * if levels[k].is_some() { 1.0 } else { -1.0 };
            }
            if may_body[k] && (opens && RANKED_BODY.contains(&region)) != depths[k].is_some() {
                agree = false;
                body_price[k] += step * if depths[k].is_some() { 1.0 } else { -1.0 };
            }
        }
        solution = Solution { path, line_choice, links, starts, pairs, notes, levels, depths, chain_of };
        if agree {
            break;
        }
    }
    // Where the parts still differ, the roles' regions stand: the outline levels the roles' headings, the
    // depths their body units.
    let forced = |wanted: &dyn Fn(u8, bool) -> bool| -> Vec<f64> {
        (0..n).map(|k| { let (c, _) = solution.path[k]; if wanted((c / 2) as u8, c % 2 == 1) { 1e6 } else { -1e6 } }).collect()
    };
    let heads = forced(&|region, opens| opens && region == HEADING);
    let bodies = forced(&|region, opens| opens && RANKED_BODY.contains(&region));
    solution.levels = outline_part(problem, w, &solution.starts, &heads, toward);
    solution.depths = depth_part(problem, w, &solution.starts, &bodies, toward);
    solution
}

/// A reading's totals, one per weight: what each part's scores sum to over the reading.
pub(crate) fn totals(problem: &Problem, solution: &Solution) -> [f64; WEIGHTS] {
    let mut t = [0f64; WEIGHTS];
    let (mut body, mut before, mut note) = (0usize, 0usize, false);
    for (k, &(class, marked)) in solution.path.iter().enumerate() {
        let local = problem.choices[k].iter().find(|&&(c, m, _)| c == class && m == marked).map_or(-20.0, |&(_, _, l)| f64::from(l));
        t[W_LOCAL] += local;
        t[W_MARKED] += f64::from(u8::from(marked));
        let (region, opens) = ((class / 2) as u8, class % 2 == 1);
        if let Some((b, f, o, resumed)) = step_state(region, opens, body, before, note) {
            (body, before, note) = (b, f, o);
            t[W_RESUME] += f64::from(u8::from(resumed));
        }
        if opens && region == HEADING {
            t[W_HEADING] += 1.0;
        }
    }
    let number_at: HashMap<(u32, u8), usize> = problem.numbers.iter().enumerate().map(|(i, &(line, reading, _, _))| ((line, reading), i)).collect();
    for (k, choice) in solution.line_choice.iter().enumerate() {
        if let Some((reading, role)) = choice {
            if let Some(&i) = number_at.get(&(k as u32, *reading)) {
                t[W_NUMBER] += f64::from(problem.unary[i][usize::from(*role)]);
            }
        }
    }
    t[W_LINK] = solution.links.iter().map(|&j| f64::from(problem.links[j].2)).sum();
    t[W_PAIR] = solution.pairs.iter().map(|&j| f64::from(problem.pairs[j].3)).sum();
    t[W_NOTE] = problem.labels.iter().filter(|&&(line, _, _)| solution.notes.get(line as usize).copied().unwrap_or(false)).count() as f64;
    let mut open = 0usize;
    for k in 0..solution.levels.len() {
        if solution.starts.get(k).copied().unwrap_or(false) {
            open = 0;
        }
        if let (Some(level), Some(levels)) = (solution.levels[k], problem.levels.get(k).filter(|l| !l.is_empty())) {
            t[W_LEVEL] += f64::from(levels[open][level as usize]);
            open = level as usize;
        }
    }
    for (k, &started) in solution.starts.iter().enumerate() {
        if started {
            t[W_START] += f64::from(problem.starts[k].unwrap_or(0.0));
            t[W_START_BIAS] += 1.0;
        }
    }
    let mut open = 0usize;
    for k in 0..solution.depths.len() {
        if solution.starts.get(k).copied().unwrap_or(false) {
            open = 0;
        }
        if let (Some(depth), Some(depths)) = (solution.depths[k], problem.depths.get(k).filter(|d| !d.is_empty())) {
            t[W_DEPTH] += f64::from(depths[open][usize::from(depth)]);
            open = usize::from(depth);
        }
    }
    t
}

/// The weights trained on whole documents: every problem in `folder` (`<sha>.problem.json`, the gold's
/// totals in it) read `epochs` times, each read loss-augmented under the weights so far and the weights
/// moved toward the gold reading's totals and away from the read one's, each weight's step scaled by
/// its steps before (AdaGrad), the local scores' weight kept positive; the average over every step is
/// written to `out` (a JSON array) and each epoch's agreement with the gold reported.
pub(crate) fn train(folder: &Path, out: &Path, epochs: usize) -> Result<String, String> {
    let mut problems: Vec<(String, Problem)> = Vec::new();
    for entry in std::fs::read_dir(folder).map_err(|e| format!("{}: {e}", folder.display()))? {
        let path = entry.map_err(|e| e.to_string())?.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()).filter(|n| n.ends_with(".problem.json")) else { continue };
        let problem: Problem = serde_json::from_slice(&std::fs::read(&path).map_err(|e| e.to_string())?).map_err(|e| format!("{name}: {e}"))?;
        if problem.gold_totals.is_some() {
            problems.push((name.to_owned(), problem));
        }
    }
    problems.sort_by(|a, b| a.0.cmp(&b.0));
    // The reachability check: each problem solved pulled hard toward the gold's reading; how much of the
    // gold the solve can make at all (the architecture's ceiling on these documents).
    let mut report = String::new();
    {
        let mut reach = [(0usize, 0usize); 5];
        for (_, problem) in &problems {
            let Some(gold) = &problem.gold_reading else { continue };
            let solved = solve(problem, &PRIOR_WEIGHTS, false, 50.0);
            for k in 0..problem.choices.len() {
                if problem.gold[k] >= 0 {
                    reach[0].1 += 1;
                    reach[0].0 += usize::from(solved.path[k].0 as i16 == problem.gold[k]);
                }
                let role = solved.line_choice[k].map_or(0, |(_, r)| r);
                if gold.roles[k] > 0 || role > 0 {
                    reach[1].1 += 1;
                    reach[1].0 += usize::from(role == gold.roles[k]);
                }
                let level = solved.levels[k].map_or(0, |l| l as u8);
                if gold.levels[k] > 0 || level > 0 {
                    reach[2].1 += 1;
                    reach[2].0 += usize::from(level == gold.levels[k]);
                }
                let depth = solved.depths[k].unwrap_or(0);
                if gold.depths[k] > 0 || depth > 0 {
                    reach[3].1 += 1;
                    reach[3].0 += usize::from(depth == gold.depths[k]);
                }
                if gold.starts[k] || solved.starts[k] {
                    reach[4].1 += 1;
                    reach[4].0 += usize::from(gold.starts[k] == solved.starts[k]);
                }
            }
        }
        let share = |(right, all): (usize, usize)| right as f64 / all.max(1) as f64;
        report.push_str(&format!("reachable: lines' classes {:.4} of {}, numbers' roles {:.4} of {}, heading levels {:.4} of {}, body depths {:.4} of {}, starts {:.4} of {}\n",
            share(reach[0]), reach[0].1, share(reach[1]), reach[1].1, share(reach[2]), reach[2].1, share(reach[3]), reach[3].1, share(reach[4]), reach[4].1));
    }
    let mut w = PRIOR_WEIGHTS;
    let mut squares = [1e-6f64; WEIGHTS];
    let (mut sum, mut steps) = ([0f64; WEIGHTS], 0usize);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(8);
    for epoch in 0..epochs {
        let (mut right, mut lines) = (0usize, 0usize);
        // Batches read in parallel under one set of weights, each batch one step.
        let order: Vec<usize> = { let mut o: Vec<usize> = (0..problems.len()).collect(); o.rotate_left(epoch * 7 % problems.len().max(1)); o };
        for batch in order.chunks(threads * 4) {
            let read: Vec<([f64; WEIGHTS], usize, usize)> = std::thread::scope(|scope| {
                let handles: Vec<_> = batch.chunks(4).map(|part| {
                    let (problems, w) = (&problems, &w);
                    scope.spawn(move || part.iter().map(|&i| {
                        let problem = &problems[i].1;
                        let solved = solve(problem, w, true, 0.0);
                        let read = totals(problem, &solved);
                        let gold = problem.gold_totals.unwrap_or_default();
                        let mut delta = [0f64; WEIGHTS];
                        for d in 0..WEIGHTS {
                            delta[d] = gold[d] - read[d];
                        }
                        let hits = solved.path.iter().zip(&problem.gold).filter(|(p, &g)| g >= 0 && p.0 as i16 == g).count();
                        let known = problem.gold.iter().filter(|&&g| g >= 0).count();
                        (delta, hits, known)
                    }).collect::<Vec<_>>())
                }).collect();
                handles.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect()
            });
            let mut delta = [0f64; WEIGHTS];
            for (d, hits, known) in &read {
                for i in 0..WEIGHTS {
                    delta[i] += d[i];
                }
                right += hits;
                lines += known;
            }
            for i in 0..WEIGHTS {
                squares[i] += delta[i] * delta[i];
                w[i] += 0.1 * delta[i] / squares[i].sqrt();
            }
            w[W_LOCAL] = w[W_LOCAL].max(0.05);
            for i in 0..WEIGHTS {
                sum[i] += w[i];
            }
            steps += 1;
        }
        report.push_str(&format!("epoch {}: the loss-augmented reading's lines as the gold reads them {:.4} of {lines}; weights {:?}\n", epoch + 1, right as f64 / lines.max(1) as f64, w.map(|x| (x * 1000.0).round() / 1000.0)));
    }
    let average: Vec<f64> = sum.iter().map(|s| s / steps.max(1) as f64).collect();
    std::fs::write(out, serde_json::to_string(&average).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", out.display()))?;
    report.push_str(&format!("{} documents, averaged weights {:?} -> {}", problems.len(), average, out.display()));
    Ok(report)
}

/// A line feature's value on line `k`.
fn feature(table: &LineTable, k: usize, name: &str) -> f32 {
    LINE_FEATURES.iter().position(|n| *n == name).map_or(0.0, |at| table.row(k)[at])
}

/// What the whole document's numbers say of each: its nearest earlier number of its family a value less
/// and nearest later a value more, how far the family runs back and on through them, how many numbers
/// each family holds, and the families by size.
struct Context {
    pred: Vec<Option<usize>>,
    succ: Vec<Option<usize>>,
    back: Vec<u32>,
    forward: Vec<u32>,
    family_size: HashMap<String, usize>,
    rank: HashMap<String, usize>,
}

fn context(numbers: &[Number]) -> Context {
    let mut by: HashMap<(&str, u32), Vec<usize>> = HashMap::new();
    for (i, number) in numbers.iter().enumerate() {
        by.entry((number.family.as_str(), number.value)).or_default().push(i);
    }
    let n = numbers.len();
    let (mut pred, mut succ) = (vec![None; n], vec![None; n]);
    for (i, number) in numbers.iter().enumerate() {
        if number.value > 1 {
            pred[i] = by.get(&(number.family.as_str(), number.value - 1)).and_then(|found| found.iter().rev().copied().find(|&p| numbers[p].line < number.line));
        }
        succ[i] = by.get(&(number.family.as_str(), number.value + 1)).and_then(|found| found.iter().copied().find(|&s| numbers[s].line > number.line));
    }
    let mut back = vec![1u32; n];
    for i in 0..n {
        if let Some(p) = pred[i] {
            back[i] = back[p] + 1;
        }
    }
    let mut forward = vec![1u32; n];
    for i in (0..n).rev() {
        if let Some(s) = succ[i] {
            forward[i] = forward[s] + 1;
        }
    }
    let mut family_size: HashMap<String, usize> = HashMap::new();
    for number in numbers {
        *family_size.entry(number.family.clone()).or_default() += 1;
    }
    let mut sizes: Vec<(&String, &usize)> = family_size.iter().collect();
    sizes.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    let rank = sizes.iter().enumerate().map(|(r, (f, _))| ((*f).clone(), r)).collect();
    Context { pred, succ, back, forward, family_size, rank }
}

/// A number's features: its reading, what the document's numbers say of it, where its line stands, how
/// the local scores read its line, and its line's place and marker features.
fn number_features(i: usize, numbers: &[Number], context: &Context, table: &LineTable, views: &[LineView], locals: &[Local]) -> [f32; NUMBER_WIDTH] {
    let number = &numbers[i];
    let k = number.line;
    let ln = |x: f32| (1.0 + x.max(0.0)).ln();
    let flag = |b: bool| f32::from(u8::from(b));
    let gap = |other: Option<usize>| other.map_or(-1.0, |o| ln((numbers[o].line as f32 - k as f32).abs()));
    let mut f = [0f32; NUMBER_WIDTH];
    f[0] = ln(number.value as f32);
    f[1] = flag(number.value == 1);
    f[2] = count_class(&number.family);
    f[3] = flag(number.family.contains("[]"));
    f[4] = flag(number.family.contains("()"));
    f[5] = flag(number.family.ends_with(')') || number.family.contains("):"));
    f[6] = flag(number.family.contains('.'));
    f[7] = flag(number.family.matches(':').count() > 1);
    f[8] = number.reading as f32;
    f[9] = views[k].readings.len() as f32;
    f[10] = flag(context.pred[i].is_some());
    f[11] = gap(context.pred[i]);
    f[12] = flag(context.succ[i].is_some());
    f[13] = gap(context.succ[i]);
    f[14] = ln(context.back[i] as f32);
    f[15] = ln(context.forward[i] as f32);
    let size = context.family_size.get(&number.family).copied().unwrap_or(1);
    f[16] = ln(size as f32);
    f[17] = size as f32 / numbers.len().max(1) as f32;
    f[18] = context.rank.get(&number.family).copied().unwrap_or(9).min(5) as f32;
    f[19] = k as f32 / views.len().max(1) as f32;
    f[20] = views[k].page_place.min(60) as f32;
    let p = &locals[k].probabilities;
    for (at, class) in [3usize, 33, 5, 9, 7, 11].into_iter().enumerate() {
        f[21 + at] = p[class] as f32;
    }
    f[27] = (p[0] + p[1]) as f32;
    let row = table.row(k);
    f[28..48].copy_from_slice(&row[..20]);
    f[48..68].copy_from_slice(&row[72..92]);
    f
}

/// A link's features: both numbers' own, and what stands between them.
fn link_features(p: usize, s: usize, numbers: &[Number], context: &Context, own: &[[f32; NUMBER_WIDTH]], table: &LineTable, views: &[LineView]) -> [f32; LINK_WIDTH] {
    let (a, b) = (&numbers[p], &numbers[s]);
    let mut f = [0f32; LINK_WIDTH];
    f[..20].copy_from_slice(&own[p][..20]);
    f[20..40].copy_from_slice(&own[s][..20]);
    f[40] = (1.0 + (b.line - a.line) as f32).ln();
    f[41] = views[b.line].page as f32 - views[a.line].page as f32;
    let between = numbers.iter().filter(|x| x.line > a.line && x.line < b.line && x.family == a.family);
    f[42] = (1.0 + between.clone().count() as f32).ln();
    f[43] = (1.0 + between.filter(|x| x.value == b.value).count() as f32).ln();
    f[44] = feature(table, b.line, "indent") - feature(table, a.line, "indent");
    f[45] = feature(table, b.line, "size") - feature(table, a.line, "size");
    f[46] = f32::from(u8::from(context.succ[p] == Some(s)));
    f[47] = f32::from(u8::from(context.pred[s] == Some(p)));
    f
}

/// A mark's candidate labels: its number's labels after it on its page or the two after, nearest three.
fn pair_candidates(views: &[LineView]) -> Vec<(usize, usize, usize)> {
    let labels: Vec<usize> = (0..views.len()).filter(|&k| views[k].label.is_some_and(|v| v > 0)).collect();
    let mut out = Vec::new();
    for (k, view) in views.iter().enumerate() {
        for (m, &(value, ..)) in view.references.iter().enumerate() {
            out.extend(labels.iter().copied().filter(|&l| l > k && views[l].label == Some(value) && views[l].page <= view.page + 2).take(3).map(|l| (k, m, l)));
        }
    }
    out
}

/// A pair's features: the mark's line and how its marks are set, the label's line and how its label is
/// set, and what stands between them.
fn pair_features(k: usize, m: usize, l: usize, table: &LineTable, views: &[LineView], locals: &[Local]) -> [f32; PAIR_WIDTH] {
    let value = views[k].references[m].0;
    let mut f = [0f32; PAIR_WIDTH];
    f[0] = (1.0 + value as f32).ln();
    f[1] = m as f32;
    for (at, name) in ["reference_superscript", "reference_detached", "reference_raised", "reference_glued"].into_iter().enumerate() {
        f[2 + at] = feature(table, k, name);
    }
    for (at, name) in ["label_raised", "label_size", "size", "page_position", "below_separator", "length"].into_iter().enumerate() {
        f[6 + at] = feature(table, l, name);
    }
    f[12] = locals[l].probabilities[usize::from(NOTE) * 2 + 1] as f32;
    f[13] = locals[k].probabilities[usize::from(NOTE) * 2 + 1] as f32;
    f[14] = views[l].page as f32 - views[k].page as f32;
    f[15] = (1.0 + (l - k) as f32).ln();
    f[16] = ((k + 1)..l).filter(|&j| views[j].label == Some(value)).count() as f32;
    f[17] = ((k + 1)..l).filter(|&j| views[j].references.iter().any(|r| r.0 == value)).count() as f32;
    f[18] = f32::from(u8::from(f[16] == 0.0));
    f[19] = (1.0 + views.iter().filter(|v| v.label.is_some()).count() as f32).ln();
    f[20] = views[l].page_place.min(60) as f32;
    f[21] = feature(table, l, "first_on_page");
    f
}

/// A heading's features given the open heading's level: the level, its line's place and marker features,
/// its template's measures over the whole document, and what the document's outline names it.
fn outline_features(k: usize, open: usize, entry: Option<u8>, table: &LineTable, views: &[LineView]) -> [f32; OUTLINE_WIDTH] {
    let mut f = [0f32; OUTLINE_WIDTH];
    f[0] = open as f32;
    let row = table.row(k);
    f[1..21].copy_from_slice(&row[..20]);
    f[21..41].copy_from_slice(&row[72..92]);
    let t = &views[k].template;
    f[41..50].copy_from_slice(&[t.lines, t.pages, t.heading, t.monotone, t.anchored, t.anchor_level, t.level, t.edge, t.beside]);
    f[50] = f32::from(views[k].anchor);
    f[51] = f32::from(u8::from(views[k].listed));
    // The printed contents entry it answers: whether one, the entry's tier, the step that tier makes.
    f[52] = f32::from(u8::from(entry.is_some()));
    f[53] = entry.map_or(0.0, f32::from);
    f[54] = entry.map_or(-9.0, |tier| f32::from(tier) - open.max(1) as f32);
    f
}

/// Each line's printed contents entry, where its text answers one: the document's rows read as a contents
/// list's (a title and the locator it prints), each entry's tier its indent's rank among the entries'
/// indents; a line answers the first entry on an earlier page whose title is its text, or which its text
/// opens or closes holding half of it or more (a heading wrapped, a numbering label only one prints).
fn contents_entries(views: &[LineView]) -> Vec<Option<u8>> {
    let mut lefts: Vec<f32> = Vec::new();
    for view in views.iter().filter(|v| v.contents_title.is_some()) {
        if !lefts.iter().any(|&l| (l - view.left).abs() < 0.012) {
            lefts.push(view.left);
        }
    }
    lefts.sort_by(f32::total_cmp);
    let tier = |left: f32| lefts.iter().position(|&l| (l - left).abs() < 0.012).map_or(1, |at| (at + 1).min(9)) as u8;
    let entries: Vec<(&str, u8, u32)> = views.iter().filter_map(|v| v.contents_title.as_deref().map(|title| (title, tier(v.left), v.page))).collect();
    let half = |of: &str| of.chars().count().div_ceil(2);
    views.iter().map(|view| {
        let key = view.title_key.as_str();
        let length = key.chars().count();
        if view.contents_title.is_some() || length < 3 {
            return None;
        }
        entries.iter().find(|(entry, _, page)| *page < view.page && (*entry == key
            || (length >= 8 && length >= half(entry) && (entry.starts_with(key) || entry.ends_with(key)))
            || (entry.chars().count() >= 8 && entry.chars().count() >= half(key) && key.ends_with(*entry))))
            .map(|&(_, tier, _)| tier)
    }).collect()
}

/// A body opening's features given the open depth: the depth, its line's place and marker features, what
/// the whole document's numbers say of its first number (`number_features`), how the local scores read it.
fn depth_features(k: usize, open: usize, number: Option<&[f32; NUMBER_WIDTH]>, table: &LineTable, locals: &[Local]) -> [f32; DEPTH_WIDTH] {
    let mut f = [0f32; DEPTH_WIDTH];
    f[0] = open as f32;
    let row = table.row(k);
    f[1..21].copy_from_slice(&row[..20]);
    f[21..41].copy_from_slice(&row[72..92]);
    if let Some(x) = number {
        f[41] = 1.0;
        f[42] = x[1];
        f[43] = x[2];
        f[44] = x[10];
        f[45] = x[14];
        f[46] = x[15];
        f[47] = x[16];
        f[48] = x[18];
    } else {
        f[43] = 9.0;
        f[48] = 9.0;
    }
    let p = &locals[k].probabilities;
    for (at, class) in [3usize, 33, 5, 9].into_iter().enumerate() {
        f[49 + at] = p[class] as f32;
    }
    f
}

/// The gold's body openings with their depths and the open depth before each.
fn gold_depths(gold: &[[i64; 8]]) -> Vec<(usize, usize, usize)> {
    let (mut out, mut current) = (Vec::new(), 0usize);
    for (k, g) in gold.iter().enumerate() {
        if g[2] == 1 {
            current = 0;
        }
        let region = if g[0] == UNMARKED_BODY { 1 } else { g[0] };
        if g[1] == 1 && region >= 0 && RANKED_BODY.contains(&(region as u8)) && g[7] > 0 {
            let depth = (current.max(1) as i64 + g[7] - 10).clamp(1, LEVEL_CLASSES as i64 - 1) as usize;
            out.push((k, current, depth));
            current = depth;
        }
    }
    out
}

/// The gold's headings with their levels and the open level before each, for the outline head's fit.
fn gold_outline(gold: &[[i64; 8]]) -> Vec<(usize, usize, usize)> {
    let (mut out, mut current) = (Vec::new(), 0usize);
    for (k, g) in gold.iter().enumerate() {
        if g[2] == 1 {
            current = 0;
        }
        if g[0] == i64::from(HEADING) && g[1] == 1 {
            if g[4] < 0 {
                current = 0;
                continue;
            }
            let level = (current.max(1) as i64 + g[4] - 10).clamp(1, LEVEL_CLASSES as i64 - 1) as usize;
            out.push((k, current, level));
            current = level;
        }
    }
    out
}

/// The gold's answers for the dump: each number's role, each link and each pair, from the gold's labels.
fn gold_answers(numbers: &[Number], links: &[(usize, usize)], pairs: &[(usize, usize, usize)], views: &[LineView], gold: &[[i64; 8]]) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let role = |k: usize| -> i64 {
        let g = &gold[k];
        if g[0] < 0 {
            return -1;
        }
        if g[1] == 1 && g[5] == 1 {
            let region = if g[0] == UNMARKED_BODY { 1 } else { g[0] as u8 };
            role_of(region) as i64
        } else {
            0
        }
    };
    let roles: Vec<i64> = numbers.iter().map(|x| role(x.line)).collect();
    let link_gold = links.iter().map(|&(p, s)| {
        let (rp, rs) = (roles[p], roles[s]);
        if rp < 0 || rs < 0 {
            return -1;
        }
        // The gold links them where both count, in one role, and the later is the first of the family's
        // next value the gold counts after the earlier.
        let first = numbers.iter().enumerate().find(|(i, x)| x.line > numbers[p].line && x.family == numbers[p].family && x.value == numbers[p].value + 1 && roles[*i] > 0).map(|(i, _)| i);
        i64::from(rp > 0 && rp == rs && first == Some(s))
    }).collect();
    let pair_gold = pairs.iter().map(|&(k, m, l)| {
        if gold[k][0] < 0 || gold[l][0] < 0 {
            return -1;
        }
        let value = views[k].references[m].0;
        let first = ((k + 1)..views.len()).find(|&j| gold[j][0] == i64::from(NOTE) && gold[j][1] == 1 && views[j].label == Some(value));
        i64::from(gold[k][3] == 1 && first == Some(l))
    }).collect();
    (roles, link_gold, pair_gold)
}


/// Whether a line may open a constituent document: the constituents reader names it, a title's or
/// heading's line near its page's top, or the document's main numbering starting over there.
fn may_start(k: usize, views: &[LineView], locals: &[Local], numbers: &[Number], main: Option<&String>) -> (bool, bool, f32) {
    let p = &locals[k].probabilities;
    let titled = p[usize::from(TITLE) * 2 + 1] + p[usize::from(HEADING) * 2 + 1] >= 0.5 && views[k].page_place < 5;
    let restart = numbers.iter().any(|x| x.line >= k && x.line <= k + 8 && x.value == 1 && Some(&x.family) == main);
    let before = numbers.iter().filter(|x| x.line < k && Some(&x.family) == main).count() as f32;
    (views[k].start_kind != 0 || titled || (restart && before >= 2.0), restart, before)
}

/// A line's features as a start: what names it one, how the local scores read it, where it stands.
fn start_features(k: usize, restart: bool, before: f32, table: &LineTable, views: &[LineView], locals: &[Local]) -> [f32; START_WIDTH] {
    let p = &locals[k].probabilities;
    let mut f = [0f32; START_WIDTH];
    f[0] = f32::from(views[k].start_kind);
    f[1] = views[k].page_place.min(60) as f32;
    f[2] = p[usize::from(TITLE) * 2 + 1] as f32;
    f[3] = p[usize::from(HEADING) * 2 + 1] as f32;
    f[4] = p[usize::from(TEXT) * 2 + 1] as f32;
    f[5] = f32::from(u8::from(restart));
    f[6] = (1.0 + before).ln();
    f[7] = feature(table, k, "first_on_page");
    f[8..28].copy_from_slice(&table.row(k)[..20]);
    f[28] = f32::from(views[k].anchor);
    f[29] = f32::from(u8::from(views[k].listed));
    f[30] = views[k].template.heading;
    f[31] = k as f32 / views.len().max(1) as f32;
    f
}

/// The document's reading: its lines' moves, chosen whole. With `dump` (a folder, the document's sha256)
/// and the gold, its problem with the gold reading's totals, and the heads' training rows, written there.
pub(crate) fn read(table: &LineTable, views: &[LineView], locals: &[Local], heads: Option<&Heads>, weights: &[f64; WEIGHTS], gold: Option<&[[i64; 8]]>, dump: Option<(&Path, &str)>) -> Vec<Move> {
    let n = views.len();
    // Every line-opening number, and what the whole document's numbers say of each.
    let numbers: Vec<Number> = (0..n).flat_map(|k| views[k].readings.iter().enumerate().take(3)
        .map(move |(j, r)| Number { line: k, reading: j, family: r.family.clone(), value: r.value })).collect();
    let numbered_lines: HashSet<usize> = numbers.iter().map(|x| x.line).collect();
    let context = context(&numbers);
    let own: Vec<[f32; NUMBER_WIDTH]> = (0..numbers.len()).map(|i| number_features(i, &numbers, &context, table, views, locals)).collect();
    let mut links: Vec<(usize, usize)> = Vec::new();
    {
        let mut by: HashMap<(&str, u32), Vec<usize>> = HashMap::new();
        for (i, number) in numbers.iter().enumerate() {
            by.entry((number.family.as_str(), number.value)).or_default().push(i);
        }
        for (s, number) in numbers.iter().enumerate() {
            if number.value > 1 {
                for &p in by.get(&(number.family.as_str(), number.value - 1)).into_iter().flatten() {
                    if numbers[p].line < number.line {
                        links.push((p, s));
                    }
                }
            }
        }
    }
    let link_rows: Vec<[f32; LINK_WIDTH]> = links.iter().map(|&(p, s)| link_features(p, s, &numbers, &context, &own, table, views)).collect();
    let pairs = pair_candidates(views);
    let pair_rows: Vec<[f32; PAIR_WIDTH]> = pairs.iter().map(|&(k, m, l)| pair_features(k, m, l, table, views, locals)).collect();
    let main = context.rank.iter().find(|(_, r)| **r == 0).map(|(f, _)| f.clone());
    let start_rows: Vec<Option<([f32; START_WIDTH], bool)>> = (0..n).map(|k| {
        let (may, restart, before) = may_start(k, views, locals, &numbers, main.as_ref());
        (k > 0 && may).then(|| (start_features(k, restart, before, table, views, locals), restart))
    }).collect();
    // Each line's choices.
    let gold_class = |k: usize| -> Option<usize> {
        let g = gold?.get(k)?;
        (g[0] >= 0).then(|| (if g[0] == UNMARKED_BODY { 1 } else { g[0] as usize }) * 2 + usize::from(g[1] == 1))
    };
    let choices: Vec<Vec<(u16, bool, f32)>> = (0..n).map(|k| {
        let local = &locals[k];
        let mut classes: Vec<usize> = local.ranked.iter().copied().filter(|&c| local.probabilities[c] >= KEEP).take(MOST).collect();
        let mut need = vec![usize::from(TEXT) * 2, usize::from(TEXT) * 2 + 1];
        if numbered_lines.contains(&k) {
            need.extend([2 * 2 + 1, 16 * 2 + 1, usize::from(QUOTE) * 2 + 1]);
        }
        if views[k].label.is_some() {
            need.push(usize::from(NOTE) * 2 + 1);
        }
        // A dump's problem holds the gold's own class, so the gold reading is one the solve can make.
        if dump.is_some() {
            need.extend(gold_class(k).filter(|&c| c < CLASSES));
        }
        for c in need {
            if !classes.contains(&c) {
                classes.push(c);
            }
        }
        let mut out = Vec::new();
        for c in classes {
            let score = local.probabilities[c].max(1e-9).ln() as f32;
            out.push((c as u16, false, score));
            if c % 2 == 1 && numbered((c / 2) as u8) && numbered_lines.contains(&k) {
                out.push((c as u16, true, score));
            }
        }
        out
    }).collect();
    // The heads' scores, and the problem.
    let mut families: HashMap<&str, u32> = HashMap::new();
    for number in &numbers {
        let next = families.len() as u32;
        families.entry(number.family.as_str()).or_insert(next);
    }
    let unary: Vec<[f32; ROLES]> = own.iter().map(|row| {
        let mut u = [0f32; ROLES];
        if let Some(h) = heads {
            let p = h.number.probabilities(row);
            for role in 0..ROLES {
                u[role] = (p[role].max(1e-9).ln() - p[0].max(1e-9).ln()) as f32;
            }
        }
        u
    }).collect();
    let even = (1.0 / (LEVEL_CLASSES - 1) as f64).ln();
    let entries = contents_entries(views);
    let levels: Vec<Vec<[f32; LEVEL_CLASSES]>> = (0..n).map(|k| {
        if !choices[k].iter().any(|&(c, _, _)| c == u16::from(HEADING) * 2 + 1) {
            return Vec::new();
        }
        (0..LEVEL_CLASSES).map(|open| {
            let mut out = [0f32; LEVEL_CLASSES];
            let scores: Option<Vec<f64>> = match (heads, &locals[k].levels) {
                (Some(h), _) => Some(h.outline.probabilities(&outline_features(k, open, entries[k], table, views))),
                (None, Some(levels)) => Some(levels.iter().map(|&p| f64::from(p)).collect()),
                (None, None) => None,
            };
            for level in 1..LEVEL_CLASSES {
                out[level] = match (&scores, entries[k]) {
                    (Some(p), _) => (p[level].max(1e-9).ln() - even) as f32,
                    (None, Some(tier)) => if usize::from(tier) == level { 2.0 } else { -2.0 },
                    (None, None) => -1.5 * level.saturating_sub(open.max(1) + 1) as f32,
                };
            }
            out
        }).collect()
    }).collect();
    // Each line's depths given the open depth where it may open a body unit.
    let first_number: HashMap<usize, usize> = numbers.iter().enumerate().rev().map(|(i, x)| (x.line, i)).collect();
    let depths: Vec<Vec<[f32; LEVEL_CLASSES]>> = (0..n).map(|k| {
        if !choices[k].iter().any(|&(c, _, _)| c % 2 == 1 && RANKED_BODY.contains(&((c / 2) as u8))) {
            return Vec::new();
        }
        let number = first_number.get(&k).map(|&i| &own[i]);
        (0..LEVEL_CLASSES).map(|open| {
            let mut out = [0f32; LEVEL_CLASSES];
            let scores = heads.map(|h| h.depth.probabilities(&depth_features(k, open, number, table, locals)));
            for depth in 1..LEVEL_CLASSES {
                out[depth] = match &scores {
                    Some(p) => (p[depth].max(1e-9).ln() - even) as f32,
                    None => -1.5 * depth.saturating_sub(open.max(1) + 1) as f32 - 0.5 * open.saturating_sub(depth) as f32,
                };
            }
            out
        }).collect()
    }).collect();
    let mut problem = Problem {
        choices,
        numbers: numbers.iter().map(|x| (x.line as u32, x.reading as u8, families[x.family.as_str()], x.value)).collect(),
        unary,
        links: links.iter().zip(&link_rows).map(|(&(p, s), row)| (p as u32, s as u32, heads.map_or(LINK, |h| h.link.scores(row)[0] as f32))).collect(),
        labels: (0..n).filter_map(|k| views[k].label.filter(|&v| v > 0).map(|v| (k as u32, v, views[k].page))).collect(),
        marks: views.iter().enumerate().flat_map(|(k, v)| v.references.iter().enumerate().map(move |(m, r)| (k as u32, m as u16, r.0, v.page))).collect(),
        pairs: pairs.iter().zip(&pair_rows).map(|(&(k, m, l), row)| (k as u32, m as u16, l as u32, heads.map_or(PAIR, |h| h.pair.scores(row)[0] as f32))).collect(),
        levels,
        depths,
        starts: start_rows.iter().map(|found| found.as_ref().map(|(row, restart)| match heads {
            Some(h) => h.start.scores(row)[0] as f32,
            None => if row[0] > 0.0 { 2.5 } else { 0.0 } + if row[2] + row[3] >= 0.5 { 0.5 } else { 0.0 } + if *restart { 2.0 } else { 0.0 },
        })).collect(),
        gold: (0..n).map(|k| gold_class(k).map_or(-1, |c| c as i16)).collect(),
        gold_totals: None,
        gold_reading: None,
    };
    if let (Some((folder, sha256)), Some(gold)) = (dump, gold) {
        let (roles, link_gold, pair_gold) = gold_answers(&numbers, &links, &pairs, views, gold);
        let outline = gold_outline(gold);
        let body = gold_depths(gold);
        // The gold reading in the solve's terms, and its totals.
        let mut gold_levels = vec![None; n];
        for &(k, _, level) in &outline {
            gold_levels[k] = Some(level as i8);
        }
        let mut gold_body = vec![None; n];
        for &(k, _, depth) in &body {
            gold_body[k] = Some(depth as u8);
        }
        let first_reading: HashMap<usize, usize> = numbers.iter().rev().map(|x| (x.line, x.reading)).collect();
        let path: Vec<(u16, bool)> = (0..n).map(|k| match gold_class(k) {
            Some(c) => {
                let marked = c % 2 == 1 && gold[k][5] == 1 && numbered((c / 2) as u8) && numbered_lines.contains(&k);
                (c as u16, marked)
            }
            None => (problem.choices[k][0].0, problem.choices[k][0].1),
        }).collect();
        let line_choice: Vec<Option<(u8, u8)>> = (0..n).map(|k| path[k].1.then(|| (first_reading.get(&k).copied().unwrap_or(0) as u8, role_of((path[k].0 / 2) as u8) as u8))).collect();
        let solution = Solution {
            line_choice,
            links: link_gold.iter().enumerate().filter(|(_, &g)| g == 1).map(|(j, _)| j).collect(),
            starts: (0..n).map(|k| gold[k][2] == 1).collect(),
            pairs: pair_gold.iter().enumerate().filter(|(_, &g)| g == 1).map(|(j, _)| j).collect(),
            notes: (0..n).map(|k| path[k].0 == u16::from(NOTE) * 2 + 1).collect(),
            levels: gold_levels,
            depths: gold_body,
            chain_of: Vec::new(),
            path,
        };
        problem.gold_totals = Some(totals(&problem, &solution));
        problem.gold_reading = Some(GoldReading {
            roles: solution.line_choice.iter().map(|c| c.map_or(0, |(_, r)| r)).collect(),
            levels: solution.levels.iter().map(|l| l.map_or(0, |l| l as u8)).collect(),
            depths: solution.depths.iter().map(|d| d.unwrap_or(0)).collect(),
            starts: solution.starts.clone(),
            notes: solution.notes.clone(),
        });
        let rows = serde_json::json!({
            "sha256": sha256, "numbers": own.iter().map(|r| r.to_vec()).collect::<Vec<_>>(), "number_gold": roles,
            "links": link_rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>(), "link_gold": link_gold,
            "pairs": pair_rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>(), "pair_gold": pair_gold,
            "outline": outline.iter().map(|&(k, open, _)| outline_features(k, open, entries[k], table, views).to_vec()).collect::<Vec<_>>(),
            "outline_gold": outline.iter().map(|&(_, _, level)| level).collect::<Vec<_>>(),
            "starts": start_rows.iter().flatten().map(|(row, _)| row.to_vec()).collect::<Vec<_>>(),
            "start_gold": start_rows.iter().enumerate().filter(|(_, r)| r.is_some()).map(|(k, _)| i64::from(gold[k][2] == 1)).collect::<Vec<_>>(),
            "depths": body.iter().map(|&(k, open, _)| depth_features(k, open, first_number.get(&k).map(|&i| &own[i]), table, locals).to_vec()).collect::<Vec<_>>(),
            "depth_gold": body.iter().map(|&(_, _, depth)| depth).collect::<Vec<_>>(),
        });
        let write = || -> std::io::Result<()> {
            std::fs::create_dir_all(folder)?;
            std::fs::write(folder.join(format!("{sha256}.whole.json")), rows.to_string())?;
            std::fs::write(folder.join(format!("{sha256}.problem.json")), serde_json::to_vec(&problem)?)
        };
        if let Err(error) = write() {
            eprintln!("whole reading dump {sha256}: {error}");
        }
    }
    let solution = solve(&problem, weights, false, 0.0);
    moves_of(&solution, &numbers, views, locals)
}

/// The reading's moves: the roles' regions; the markers the numbers count; the notes' numbers; the
/// outline's levels and depths; the starts; resumptions as the state replays them.
fn moves_of(solution: &Solution, numbers: &[Number], views: &[LineView], locals: &[Local]) -> Vec<Move> {
    let n = views.len();
    let _ = (locals, numbers);
    let mut state = State::default();
    let mut moves = Vec::with_capacity(n);
    let mut depth_before = 1u8;
    for k in 0..n {
        let line = &views[k];
        let (class, marked) = solution.path[k];
        let (region, opens) = ((class / 2) as u8, class % 2 == 1);
        let mut chosen = Move { region, opens, reading: None, note: None, level: 0, depth: 0, resume: false, starts: solution.starts[k] && opens };
        if solution.starts[k] {
            depth_before = 1;
        }
        if opens && numbered(region) && marked {
            chosen.reading = solution.line_choice[k].map(|(j, _)| j);
        }
        if opens && region == NOTE {
            chosen.note = line.label.filter(|_| solution.notes[k]);
        }
        if opens && region == HEADING {
            chosen.level = solution.levels[k].unwrap_or(1).max(1);
        }
        if opens && RANKED_BODY.contains(&region) {
            chosen.depth = solution.depths[k].unwrap_or(depth_before).max(1);
            depth_before = chosen.depth;
        }
        let key = (state.ctx(line, false), stream_of(region));
        if !opens && region != 0 && state.open_regions.get(&key) != Some(&region) && state.interrupted.get(&key) == Some(&region) {
            chosen.resume = true;
        }
        state.apply(line, &chosen);
        moves.push(chosen);
    }
    moves
}
