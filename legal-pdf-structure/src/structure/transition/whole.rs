//! The document read whole: one reading of every line chosen at once, nothing claimed by a first pass.
//! Three parts read the document, each over all of it and solved exactly, and are made to agree on what
//! they share by dual decomposition:
//! - the lines' roles: every line's region and whether it opens a unit, by Viterbi over the whole
//!   document, the state the region open in the body's stream and the one before it (a paragraph may
//!   go on past a quotation) and whether a note is open in the notes' stream;
//! - the numbers: every line-opening marker reading is a candidate; a candidate continues another of
//!   its family whose value is one less, earlier in the document, each with one successor at most and
//!   one predecessor at most; each family's value pair (v, v + 1) is its own assignment, solved exactly;
//!   a number counts where it continues one, or where it starts a chain that goes on;
//! - the notes: every reference mark is answered by at most one note label of its number, on its page
//!   or the two after; a note may answer several marks.
//!
//! What they share: a line opens a numbered unit on its marker exactly where its number counts, and a
//! line opens a note exactly where its label is a note's. The prices on those facts move until the parts
//! agree (or the rounds run out, the numbers' and the notes' reading then deciding the markers and notes,
//! the roles' the regions). The reading is the moves the emit turns into the consumer structure.

use super::*;

/// The local scores' classes a line is read among: those at this probability or more, and the ones the
/// numbers and notes need, at most this many by the local scores.
const KEEP: f64 = 1e-3;
const MOST: usize = 10;
/// The parts' weights (log-probability units), set by hand until they are learned on whole documents.
const LINK: f64 = 3.0;
const NOTE_BASE: f64 = 0.5;
const PAIR: f64 = 3.0;
const RESUME: f64 = -1.0;
const ROUNDS: usize = 40;
/// The body regions a body continuation goes on in (the state's region slots), 0 for none.
const SLOTS: usize = 17;

/// A line-opening number: its line, its reading's index there, family and value.
#[derive(Clone, Debug)]
struct Number {
    line: usize,
    reading: usize,
    family: String,
    value: u32,
}

/// One line's choices: a class, and for a numbered opening whether it opens on its marker.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Choice {
    class: usize,
    marked: bool,
}

fn numbered(region: u8) -> bool {
    [1u8, 2, 3, 4, 7, 8, 10, 14, 16].contains(&region)
}

/// The lines' roles under the prices: the best reading by Viterbi, each line's choice.
fn roles(choices: &[Vec<(Choice, f64)>], marked_price: &[f64], note_price: &[f64]) -> Vec<Choice> {
    let n = choices.len();
    // A state: the body stream's open region, the one before it, a note open.
    let state_of = |body: usize, before: usize, note: bool| (body * SLOTS + before) * 2 + usize::from(note);
    let states = SLOTS * SLOTS * 2;
    let mut score = vec![f64::NEG_INFINITY; states];
    score[state_of(0, 0, false)] = 0.0;
    let mut back: Vec<Vec<(u32, u16)>> = Vec::with_capacity(n);
    for (k, options) in choices.iter().enumerate() {
        let mut next = vec![f64::NEG_INFINITY; states];
        let mut from = vec![(u32::MAX, 0u16); states];
        for s in 0..states {
            if score[s] == f64::NEG_INFINITY {
                continue;
            }
            let (body, before, note) = (s / 2 / SLOTS, s / 2 % SLOTS, s % 2 == 1);
            for (c, &(choice, local)) in options.iter().enumerate() {
                let region = (choice.class / 2) as u8;
                let opens = choice.class % 2 == 1;
                let stream = stream_of(region);
                let mut gain = local;
                if choice.marked {
                    gain += marked_price[k];
                }
                if opens && region == NOTE {
                    gain += note_price[k];
                }
                let target = if stream == 0 && region != 0 {
                    let r = usize::from(region);
                    if opens || r == body {
                        if r == body { state_of(r, before, note) } else { state_of(r, body, note) }
                    } else if r == before {
                        gain += RESUME;
                        state_of(r, body, note)
                    } else {
                        continue;
                    }
                } else if stream == 1 {
                    if !opens && !note {
                        continue;
                    }
                    state_of(body, before, true)
                } else {
                    state_of(body, before, note)
                };
                let total = score[s] + gain;
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
    let mut path = vec![Choice { class: 0, marked: false }; n];
    for k in (0..n).rev() {
        let (previous, c) = back[k][s];
        if previous == u32::MAX {
            path[k] = choices[k][0].0;
            continue;
        }
        path[k] = choices[k][usize::from(c)].0;
        s = previous as usize;
    }
    path
}

/// The numbers' chains under the prices: each candidate's predecessor (an index), exactly, value pair by
/// value pair; and which numbers count.
fn chains(numbers: &[Number], price: &[f64]) -> (Vec<Option<usize>>, Vec<bool>) {
    let mut by: HashMap<(&str, u32), Vec<usize>> = HashMap::new();
    for (i, number) in numbers.iter().enumerate() {
        by.entry((number.family.as_str(), number.value)).or_default().push(i);
    }
    let mut predecessor = vec![None; numbers.len()];
    let mut successor = vec![None; numbers.len()];
    for (&(family, value), succs) in &by {
        let Some(preds) = by.get(&(family, value.wrapping_sub(1))).filter(|_| value > 1) else { continue };
        // A link's worth: the link, and the price of counting what it makes count (the successor; a
        // first value's predecessor counts with it as the chain's start).
        let worth = |p: usize, s: usize| {
            let start = if numbers[p].value == 1 { price[numbers[p].line] } else { 0.0 };
            LINK + price[numbers[s].line] + start
        };
        let pairs = assignment(preds, succs, |p, s| (numbers[p].line < numbers[s].line).then(|| worth(p, s)).filter(|w| *w > 0.0));
        for (p, s) in pairs {
            predecessor[s] = Some(p);
            successor[p] = Some(s);
        }
    }
    // A number counts where it continues one, or starts a chain that goes on; a line counts once.
    let counts: Vec<bool> = (0..numbers.len()).map(|i| predecessor[i].is_some() || successor[i].is_some()).collect();
    (predecessor, counts)
}

/// The best one-to-one pairing of `left` with `right` by `worth` (None: no edge), exactly: small sets by
/// search over the right's choices in order, larger by the Hungarian method's greedy bound (each edge
/// best first) where the sets run past what search can take.
fn assignment(left: &[usize], right: &[usize], worth: impl Fn(usize, usize) -> Option<f64>) -> Vec<(usize, usize)> {
    let edges: Vec<Vec<Option<f64>>> = left.iter().map(|&l| right.iter().map(|&r| worth(l, r)).collect()).collect();
    if left.len() <= 8 && right.len() <= 8 {
        // Exact: each left takes a right or none, the rights each once.
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

/// The notes under the prices: each reference mark's note label (a line), and which labels open notes.
fn notes(views: &[LineView], price: &[f64]) -> (Vec<(usize, usize)>, Vec<bool>) {
    let labels: Vec<usize> = (0..views.len()).filter(|&k| views[k].label.is_some_and(|v| v > 0)).collect();
    let mut opens = vec![false; views.len()];
    for &k in &labels {
        opens[k] = NOTE_BASE + price[k] > 0.0;
    }
    // Each mark to the nearest label of its number after it, on its page or the two after, where the
    // label's worth with the mark is positive.
    let mut pairs = Vec::new();
    let mut gained: HashMap<usize, f64> = HashMap::new();
    let mut wanted: Vec<(usize, usize)> = Vec::new();
    for (k, view) in views.iter().enumerate() {
        for (m, &(value, ..)) in view.references.iter().enumerate() {
            if let Some(&label) = labels.iter().find(|&&l| l > k && views[l].label == Some(value) && views[l].page <= view.page + 2) {
                wanted.push((k * 64 + m.min(63), label));
                *gained.entry(label).or_default() += PAIR;
            }
        }
    }
    for &label in &labels {
        if NOTE_BASE + price[label] + gained.get(&label).copied().unwrap_or(0.0) > 0.0 {
            opens[label] = true;
        }
    }
    for (mark, label) in wanted {
        if opens[label] {
            pairs.push((mark, label));
        }
    }
    (pairs, opens)
}

/// The document's reading: its lines' moves, chosen whole.
pub(crate) fn read(table: &LineTable, views: &[LineView], locals: &[Local]) -> Vec<Move> {
    let n = views.len();
    // Every line-opening number.
    let numbers: Vec<Number> = (0..n).flat_map(|k| views[k].readings.iter().enumerate().take(3)
        .map(move |(j, r)| Number { line: k, reading: j, family: r.family.clone(), value: r.value })).collect();
    let numbered_lines: HashSet<usize> = numbers.iter().map(|x| x.line).collect();
    // Each line's choices: its likeliest classes, those its numbers and notes need, numbered openings
    // on and off their marker.
    let choices: Vec<Vec<(Choice, f64)>> = (0..n).map(|k| {
        let local = &locals[k];
        let mut classes: Vec<usize> = local.ranked.iter().copied().filter(|&c| local.probabilities[c] >= KEEP).take(MOST).collect();
        let mut need = vec![usize::from(TEXT) * 2, usize::from(TEXT) * 2 + 1];
        if numbered_lines.contains(&k) {
            need.extend([usize::from(TEXT) * 2 + 1, 2 * 2 + 1, 16 * 2 + 1, usize::from(QUOTE) * 2 + 1]);
        }
        if views[k].label.is_some() {
            need.push(usize::from(NOTE) * 2 + 1);
        }
        for c in need {
            if !classes.contains(&c) {
                classes.push(c);
            }
        }
        let mut out = Vec::new();
        for c in classes {
            let local = local.probabilities[c].max(1e-9).ln();
            out.push((Choice { class: c, marked: false }, local));
            if c % 2 == 1 && numbered((c / 2) as u8) && numbered_lines.contains(&k) {
                out.push((Choice { class: c, marked: true }, local));
            }
        }
        out
    }).collect();
    // Dual decomposition: the roles' marked openings against the numbers that count, the roles' note
    // openings against the labels that are notes.
    let mut marked_price = vec![0.0; n];
    let mut note_price = vec![0.0; n];
    let mut path = roles(&choices, &marked_price, &note_price);
    let (mut predecessor, mut counts) = chains(&numbers, &vec![0.0; n]);
    let (mut pairs, mut is_note) = notes(views, &vec![0.0; n]);
    for round in 0..ROUNDS {
        // Each part sees the other's price with the opposite sign.
        let chain_price: Vec<f64> = marked_price.iter().map(|p| -p).collect();
        let label_price: Vec<f64> = note_price.iter().map(|p| -p).collect();
        path = roles(&choices, &marked_price, &note_price);
        (predecessor, counts) = chains(&numbers, &chain_price);
        (pairs, is_note) = notes(views, &label_price);
        let mut counted_line = vec![false; n];
        for (i, number) in numbers.iter().enumerate() {
            counted_line[number.line] |= counts[i];
        }
        let step = 1.0 / (1.0 + round as f64);
        let mut agree = true;
        for k in 0..n {
            let marked = path[k].marked;
            if marked != counted_line[k] {
                agree = false;
                // The roles pay for a marker the numbers do not count, and are paid for one they do.
                marked_price[k] += step * if counted_line[k] { 1.0 } else { -1.0 };
            }
            let note = path[k].class == usize::from(NOTE) * 2 + 1;
            if views[k].label.is_some() && note != is_note[k] {
                agree = false;
                note_price[k] += step * if is_note[k] { 1.0 } else { -1.0 };
            }
        }
        if agree {
            break;
        }
    }
    // The moves: the roles' regions; the markers the numbers count; the notes' numbers; and levels,
    // depths and resumptions read off as the state replays them.
    let mut reading_of: HashMap<usize, usize> = HashMap::new();
    for (i, number) in numbers.iter().enumerate() {
        if counts[i] {
            reading_of.entry(number.line).or_insert(number.reading);
        }
    }
    let _ = (&predecessor, &pairs, table);
    let mut state = State::default();
    let mut moves = Vec::with_capacity(n);
    for k in 0..n {
        let line = &views[k];
        let choice = path[k];
        let (region, opens) = ((choice.class / 2) as u8, choice.class % 2 == 1);
        let mut chosen = Move { region, opens, reading: None, note: None, level: 0, depth: 0, resume: false, starts: false };
        if opens && numbered(region) {
            chosen.reading = reading_of.get(&k).map(|&j| j as u8);
        }
        if opens && region == NOTE {
            chosen.note = line.label.filter(|_| is_note[k]);
        }
        if opens && region == HEADING {
            let level = locals[k].levels.as_ref().and_then(|levels| (1..LEVEL_CLASSES).max_by(|&a, &b| levels[a].total_cmp(&levels[b])));
            chosen.level = level.map_or(1, |l| l as i8).max(1);
        }
        if opens && RANKED_BODY.contains(&region) {
            chosen.depth = state.stack_depth(line, &chosen).min(9) as u8;
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
