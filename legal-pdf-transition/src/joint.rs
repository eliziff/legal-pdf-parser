//! The joint stylesheet solve: the document's templates given their roles together, the reading the parse
//! makes under them. A template that may be headings is held to one role, its lines no heading or headings
//! at one level; each line's move is still the ranker's, a move against its template's role costing
//! `penalty`. An assignment's worth is its reading's log-probability (each move's among its line's
//! candidates, softmaxed), less its penalties, plus `weight` times each template's lines times the log of
//! the template prior's probability of its role (a forest over the template's measures, `template_rows`,
//! fitted on the gold's majority role and level of each template). The assignment starts from the free
//! reading's (each template's commonest reading) and climbs one template at a time, every role of each
//! tried by walking the document again from the template's first line, until a sweep changes none.

use super::*;

/// The template prior's measures of a template.
pub const PRIOR: usize = 20;
/// The template prior's classes: no heading, then headings at levels 1 to 9.
pub const ROLES: usize = 10;
/// The templates the solve gives roles: the likeliest headings, by their lines.
const HELD: usize = 16;
/// The deepest level a template is tried at.
const DEEPEST: i8 = 6;

/// What a template's lines are held to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Body,
    Heading(i8),
}

impl Role {
    fn class(self) -> usize {
        match self {
            Role::Body => 0,
            Role::Heading(level) => usize::from(level.clamp(1, 9) as u8),
        }
    }

    /// Whether a move goes against the role: a heading opened in a template held to none, or at another
    /// level than its template's.
    fn broken(self, m: &Move) -> bool {
        m.opens && m.region == HEADING && match self {
            Role::Body => true,
            Role::Heading(level) => m.level != level,
        }
    }
}

/// Each template's measures as the prior reads them: its stylesheet view, the mean of its lines' look and
/// place, the share of them opening on a marker, and where in the document it first stands.
pub fn template_rows(sheet: &Stylesheet, table: &LineTable) -> Vec<[f32; PRIOR]> {
    let count = sheet.templates.len();
    let names = ["size", "bold", "upper", "centre", "indent", "length", "width"];
    let at: Vec<usize> = names.iter().map(|name| LINE_FEATURES.iter().position(|n| n == name).expect("a line feature")).collect();
    let n = table.order.len();
    let mut sums = vec![[0f32; 8]; count];
    let mut lines = vec![0f32; count];
    let mut first = vec![None; count];
    for k in 0..n {
        let t = sheet.template[k] as usize;
        let row = table.row(k);
        for (sum, &f) in sums[t].iter_mut().zip(&at) {
            *sum += row[f];
        }
        sums[t][7] += f32::from(u8::from(!table.readings[k].is_empty()));
        lines[t] += 1.0;
        first[t].get_or_insert(k);
    }
    (0..count).map(|t| {
        let v = &sheet.templates[t];
        let mut r = [0f32; PRIOR];
        r[..11].copy_from_slice(&[v.lines, v.pages, v.heading, v.monotone, v.anchored, v.anchor_level, v.level, v.edge, v.beside, v.read_heading, v.read_level]);
        for (out, sum) in r[11..19].iter_mut().zip(sums[t]) {
            *out = sum / lines[t].max(1.0);
        }
        r[19] = first[t].map_or(1.0, |k| k as f32 / n.max(1) as f32);
        r
    }).collect()
}

/// The templates the solve gives roles: those the local scores or the first reading hold headings, or the
/// outline names, the `HELD` with the most lines.
pub fn held_templates(sheet: &Stylesheet) -> Vec<u32> {
    let mut held: Vec<u32> = (0..sheet.templates.len() as u32).filter(|&t| {
        let v = &sheet.templates[t as usize];
        v.heading >= 0.25 || v.read_heading > 0.0 || v.anchored > 0.0
    }).collect();
    held.sort_by(|&a, &b| sheet.templates[b as usize].lines.total_cmp(&sheet.templates[a as usize].lines).then(a.cmp(&b)));
    held.truncate(HELD);
    held
}

/// A walk's state before a line, kept to walk again from there.
#[derive(Clone)]
struct Checkpoint {
    at: usize,
    state: State,
    score: f64,
}

struct Solve<'a> {
    ranker: &'a Forest,
    table: &'a LineTable,
    views: &'a [LineView<'a>],
    locals: &'a [Local],
    template: &'a [u32],
    penalty: f64,
}

impl Solve<'_> {
    /// The reading from `from` on under `roles` (by template): its moves, its score with `from`'s, and its
    /// state before each line `keep` marks.
    fn walk(&self, roles: &[Option<Role>], from: &Checkpoint, keep: &[bool]) -> (Vec<Move>, f64, Vec<Checkpoint>) {
        let mut state = from.state.clone();
        let mut score = from.score;
        let mut moves = Vec::with_capacity(self.views.len() - from.at);
        let mut kept = Vec::new();
        for k in from.at..self.views.len() {
            if keep[k] {
                kept.push(Checkpoint { at: k, state: state.clone(), score });
            }
            let (line, local) = (&self.views[k], &self.locals[k]);
            let candidates = state.candidates(line, &classes_of(&state, line, local, None));
            let scores: Vec<f64> = candidates.iter().map(|m| move_score(self.ranker, &ranked_features(&state, line, self.table.row(k), local, m))).collect();
            let top = scores.iter().copied().fold(f64::MIN, f64::max);
            let total = scores.iter().map(|x| (x - top).exp()).sum::<f64>().ln() + top;
            let role = roles[self.template[k] as usize];
            let cost = |c: usize| if role.is_some_and(|r| r.broken(&candidates[c])) { self.penalty } else { 0.0 };
            let choice = (0..candidates.len()).max_by(|&a, &b| (scores[a] - cost(a)).total_cmp(&(scores[b] - cost(b))).then(b.cmp(&a))).unwrap_or(0);
            score += scores[choice] - total - cost(choice);
            state.apply(line, &candidates[choice]);
            moves.push(candidates[choice]);
        }
        (moves, score, kept)
    }
}

/// The document's reading under the best assignment of roles to its likeliest heading templates the climb
/// finds (the module's account), the prior's term weighted `weight` (none without a prior).
#[allow(clippy::too_many_arguments)]
pub fn solve(ranker: &Forest, prior: Option<&Forest>, weight: f64, penalty: f64, table: &LineTable, views: &[LineView], locals: &[Local], sheet: &Stylesheet) -> Vec<Move> {
    let n = views.len();
    let held = held_templates(sheet);
    if n == 0 || held.is_empty() {
        let solve = Solve { ranker, table, views, locals, template: &sheet.template, penalty };
        let start = Checkpoint { at: 0, state: State::default(), score: 0.0 };
        return solve.walk(&vec![None; sheet.templates.len()], &start, &vec![false; n]).0;
    }
    // Each held template's prior, by role, and its first line.
    let rows = template_rows(sheet, table);
    let log_prior: HashMap<u32, Vec<f64>> = held.iter().map(|&t| {
        let p = prior.map_or_else(|| vec![1.0 / ROLES as f64; ROLES], |forest| forest.probabilities(&rows[t as usize]));
        (t, p.iter().map(|p| p.max(1e-6).ln()).collect())
    }).collect();
    let mut lines = vec![0f64; sheet.templates.len()];
    let mut first = vec![n; sheet.templates.len()];
    for (k, &t) in sheet.template.iter().enumerate() {
        lines[t as usize] += 1.0;
        first[t as usize] = first[t as usize].min(k);
    }
    let mut keep = vec![false; n];
    for &t in &held {
        keep[first[t as usize]] = true;
    }
    let solve = Solve { ranker, table, views, locals, template: &sheet.template, penalty };
    let start = Checkpoint { at: 0, state: State::default(), score: 0.0 };
    let worth = |roles: &[Option<Role>], score: f64| -> f64 {
        score + weight * held.iter().map(|&t| roles[t as usize].map_or(0.0, |r| lines[t as usize] * log_prior[&t][r.class()])).sum::<f64>()
    };
    // The free reading, and each held template's role as it reads it: headings at their commonest level
    // where half its lines or more open headings, else none.
    let mut roles: Vec<Option<Role>> = vec![None; sheet.templates.len()];
    let (free, _, _) = solve.walk(&roles, &start, &keep);
    for &t in &held {
        let mut levels: HashMap<i8, u32> = HashMap::new();
        let mut headings = 0;
        for (k, m) in free.iter().enumerate() {
            if sheet.template[k] == t && m.opens && m.region == HEADING {
                headings += 1;
                *levels.entry(m.level).or_default() += 1;
            }
        }
        roles[t as usize] = Some(if f64::from(headings) * 2.0 >= lines[t as usize] {
            Role::Heading(levels.iter().max_by_key(|(level, c)| (**c, std::cmp::Reverse(**level))).map_or(1, |(level, _)| *level).clamp(1, DEEPEST))
        } else {
            Role::Body
        });
    }
    let (mut moves, score, mut checkpoints) = solve.walk(&roles, &start, &keep);
    let mut best = worth(&roles, score);
    for _sweep in 0..2 {
        let mut changed = false;
        for &t in &held {
            let at = first[t as usize];
            for role in std::iter::once(Role::Body).chain((1..=DEEPEST).map(Role::Heading)) {
                if roles[t as usize] == Some(role) {
                    continue;
                }
                let from = checkpoints.iter().find(|c| c.at == at).expect("a checkpoint at each held template's first line");
                let mut trial = roles.clone();
                trial[t as usize] = Some(role);
                let (suffix, score, kept) = solve.walk(&trial, from, &keep);
                let value = worth(&trial, score);
                if value > best + 1e-9 {
                    best = value;
                    roles = trial;
                    moves.truncate(at);
                    moves.extend(suffix);
                    checkpoints.retain(|c| c.at < at);
                    checkpoints.extend(kept);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    moves
}
