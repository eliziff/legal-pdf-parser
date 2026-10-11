//! The outline composed over the whole document: each unit's parent (a heading, a list item, a
//! numbered paragraph or provision) chosen among the earlier units by the witnesses of the pair,
//! read with the line witnesses of both: their kinds, their numbering (one family, the next value,
//! the previous sibling and its parent), their type and indent against each other, a heading or a
//! sub-document break between them, a lead-in, a quotation, the parent the readers gave, the
//! readers' heading levels of the two, how often the corpus nests a unit like this one under one
//! like the candidate (the outline prior, harvested from the gold's training half), whether this
//! document has already nested them so, and whether the candidate is still open (the unit before,
//! or one of its ancestors). The learned parent model (`model::parent_model`) scores each candidate
//! over these and both lines' witnesses; the best scored, nearest first on a tie, is the parent.
//! With the models off the readers' parents stand. Levels follow from the parents.

use crate::witness::{index, Row, Table};
use legal_structure_model::{DocumentStructure, NodeKind};
use std::collections::HashMap;
use std::sync::OnceLock;

const PRIOR: &str = include_str!("../data/outline_prior.json");

/// The pair witnesses, in the order of each pair row's values.
pub const PAIR_NAMES: &[&str] = &[
    "u_heading", "u_list", "u_paragraph", "u_section", "c_heading", "c_list", "c_paragraph", "c_section",
    "u_marked", "c_marked", "same_family", "c_prev_sibling", "c_parent_of_prev_sibling",
    "c_more_prominent", "same_style", "c_less_prominent", "indent_deeper", "indent_same", "indent_shallower",
    "heading_between", "c_lead_in", "u_lead_in", "same_quote", "u_quoted", "subdoc_break",
    "distance", "pages_apart", "c_reader_parent", "c_reader_ancestor", "u_backbone", "c_backbone",
    "c_level_shallower", "c_level_same", "c_level_deeper", "prior", "prior_best", "prior_root",
    "doc_nested", "doc_inverse", "c_open", "open_steps",
];

/// The outline prior: per unit signature, the share of its parents with each signature ("root" for
/// none). `LEGALPDF_OUTLINE_PRIOR` names another file.
pub fn prior() -> &'static (String, HashMap<String, HashMap<String, f32>>) {
    static LOADED: OnceLock<(String, HashMap<String, HashMap<String, f32>>)> = OnceLock::new();
    LOADED.get_or_init(|| {
        let text = std::env::var("LEGALPDF_OUTLINE_PRIOR")
            .ok()
            .and_then(|path| std::fs::read_to_string(path).ok())
            .unwrap_or_else(|| PRIOR.to_owned());
        let parsed = serde_json::from_str(&text).unwrap_or_default();
        (text, parsed)
    })
}

const WINDOW: usize = 40;

struct Unit {
    node: usize,
    line: String,
    kind: NodeKind,
    family: Option<usize>,
    start: usize,
    signature: String,
}

fn signature(kind: NodeKind, family: Option<usize>) -> String {
    let kind = match kind {
        NodeKind::Heading => "heading",
        NodeKind::ListItem => "list_item",
        _ => "paragraph",
    };
    format!("{kind}/{}", family.map_or("none", |f| &FAMILIES[f][2..]))
}

const FAMILIES: &[&str] = &["m_bullet", "m_paren_num", "m_paren_letter", "m_paren_roman", "m_decimal", "m_num", "m_num_bare", "m_letter", "m_roman", "m_word"];

fn value(row: Option<&&Row>, name: &str) -> f32 {
    row.and_then(|row| index(name).map(|at| row.values[at])).unwrap_or(0.0)
}

pub struct Pair {
    pub unit: String,
    pub candidate: String,
    pub values: Vec<f32>,
}

/// Rechoose every unit's parent by the parent model, where one is in force. Returns the pairs read,
/// in order, for the dump the model is trained from.
pub fn compose_outline(structure: &mut DocumentStructure, table: &Table, levels: &HashMap<String, usize>) -> Vec<Pair> {
    crate::dump(table, "outline_lines");
    let model = crate::model::parent_model();
    let prior = &prior().1;
    let rows = table.by_id();
    let nodes = &structure.nodes;
    let ids = nodes.iter().enumerate().map(|(n, node)| (node.id.as_str(), n)).collect::<HashMap<_, _>>();
    let mut units = nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| {
            matches!(node.kind, NodeKind::Heading | NodeKind::ListItem | NodeKind::Paragraph)
                || (node.kind == NodeKind::Section && node.marker_range.is_some() && node.grammar.as_deref() != Some("heading_section"))
        })
        .filter_map(|(n, node)| {
            let line = node.line_ids.first()?.clone();
            let row = rows.get(line.as_str());
            let family = FAMILIES.iter().position(|name| value(row, name) >= 0.5);
            Some(Unit { node: n, signature: signature(node.kind, family), line, kind: node.kind, family, start: node.range.start })
        })
        .collect::<Vec<_>>();
    units.sort_by_key(|unit| unit.start);
    let by_line = units.iter().enumerate().map(|(k, unit)| (unit.line.clone(), k)).collect::<HashMap<_, _>>();
    // The parent the readers gave each unit, in the scorer's terms: its nearest ancestor that is a
    // heading, paragraph, section or list item, by that ancestor's first line.
    let projected = |n: usize| {
        let mut parent = nodes[n].parent_id.as_deref().and_then(|id| ids.get(id).copied());
        let mut steps = 0;
        while let Some(p) = parent {
            if matches!(nodes[p].kind, NodeKind::Heading | NodeKind::Paragraph | NodeKind::Section | NodeKind::ListItem) {
                return nodes[p].line_ids.first().and_then(|line| by_line.get(line).copied());
            }
            parent = nodes[p].parent_id.as_deref().and_then(|id| ids.get(id).copied());
            steps += 1;
            if steps > 256 { break; }
        }
        None
    };
    let reader = units.iter().map(|unit| projected(unit.node)).collect::<Vec<_>>();
    let reader_ancestors = |k: usize, c: usize| {
        let mut at = reader[k];
        let mut steps = 0;
        while let Some(p) = at {
            if p == c { return true; }
            at = reader[p];
            steps += 1;
            if steps > 256 { break; }
        }
        false
    };
    let flag = |b: bool| if b { 1.0 } else { 0.0 };
    let mut chosen: Vec<Option<usize>> = vec![None; units.len()];
    let mut pairs = Vec::new();
    // The signatures this document has nested so far, (child, parent), by its chosen parents.
    let mut nested = HashMap::<(&str, &str), usize>::new();
    let zeros = vec![0.0; table.rows.first().map_or(0, |row| row.values.len())];
    for k in 0..units.len() {
        let u = &units[k];
        let urow = rows.get(u.line.as_str());
        let shares = prior.get(&u.signature);
        let share = |sig: &str| shares.and_then(|s| s.get(sig)).copied().unwrap_or(0.0);
        let best = shares.and_then(|s| s.iter().filter(|(sig, _)| *sig != "root").max_by(|a, b| a.1.total_cmp(b.1)).map(|(sig, _)| sig.as_str()));
        // The open units: the one before and its chosen ancestors, by steps up.
        let mut open = HashMap::<usize, usize>::new();
        let (mut at, mut steps) = (k.checked_sub(1), 0);
        while let Some(p) = at {
            open.insert(p, steps);
            at = chosen[p];
            steps += 1;
            if steps > 256 { break; }
        }
        let ulevel = levels.get(&u.line).copied();
        let prev_sibling = (0..k).rev().take(WINDOW).find(|&j| units[j].family.is_some() && units[j].family == u.family
            && value(rows.get(units[j].line.as_str()), "quoted") == value(urow, "quoted"));
        // The nearest units, and every ancestor the readers gave this unit and the one before it:
        // a heading opened many units back is still open.
        let mut candidates = (0..k).rev().take(WINDOW).collect::<Vec<_>>();
        for start in [Some(k), k.checked_sub(1)].into_iter().flatten() {
            let (mut at, mut steps) = (reader[start], 0);
            while let Some(p) = at {
                if p < k && !candidates.contains(&p) {
                    candidates.push(p);
                }
                at = reader[p];
                steps += 1;
                if steps > 256 { break; }
            }
        }
        candidates.sort_unstable_by(|a, b| b.cmp(a));
        let mut scores = Vec::<(usize, f64)>::new();
        for c in candidates {
            let distance = k - c - 1;
            let cu = &units[c];
            let crow = rows.get(cu.line.as_str());
            let heading_between = (c + 1..k).any(|j| units[j].kind == NodeKind::Heading);
            let subdoc = (c + 1..=k).any(|j| {
                let row = rows.get(units[j].line.as_str());
                value(row, "odd_width") != value(crow, "odd_width") || value(row, "source_switch") >= 0.5
            });
            let indent = value(urow, "indent") - value(crow, "indent");
            let clevel = levels.get(&cu.line).copied();
            let values = vec![
                flag(u.kind == NodeKind::Heading), flag(u.kind == NodeKind::ListItem), flag(u.kind == NodeKind::Paragraph), flag(u.kind == NodeKind::Section),
                flag(cu.kind == NodeKind::Heading), flag(cu.kind == NodeKind::ListItem), flag(cu.kind == NodeKind::Paragraph), flag(cu.kind == NodeKind::Section),
                flag(u.family.is_some()), flag(cu.family.is_some()), flag(u.family.is_some() && u.family == cu.family),
                flag(prev_sibling == Some(c)), flag(prev_sibling.is_some_and(|s| chosen[s] == Some(c))),
                flag(value(crow, "style_rank") < value(urow, "style_rank")), flag(value(crow, "style_rank") == value(urow, "style_rank")),
                flag(value(crow, "style_rank") > value(urow, "style_rank")),
                flag(indent > 0.02), flag(indent.abs() <= 0.02), flag(indent < -0.02),
                flag(heading_between), flag(value(crow, "end_colon") >= 0.5 || value(crow, "end_dash") >= 0.5), value(urow, "lead_in"),
                flag(value(urow, "quoted") == value(crow, "quoted")), value(urow, "quoted"), flag(subdoc),
                distance as f32, (value(urow, "page_frac") - value(crow, "page_frac")).abs(),
                flag(reader[k] == Some(c)), flag(reader_ancestors(k, c)), value(urow, "backbone"), value(crow, "backbone"),
                flag(clevel.zip(ulevel).is_some_and(|(c, u)| c < u)), flag(clevel.zip(ulevel).is_some_and(|(c, u)| c == u)),
                flag(clevel.zip(ulevel).is_some_and(|(c, u)| c > u)),
                share(&cu.signature), flag(best == Some(cu.signature.as_str())), share("root"),
                nested.get(&(u.signature.as_str(), cu.signature.as_str())).copied().unwrap_or(0) as f32,
                nested.get(&(cu.signature.as_str(), u.signature.as_str())).copied().unwrap_or(0) as f32,
                flag(open.contains_key(&c)), open.get(&c).map_or(-1.0, |s| *s as f32),
            ];
            debug_assert_eq!(values.len(), PAIR_NAMES.len());
            if let Some(model) = model {
                // The pair's witnesses, then the unit's line's, then the candidate's.
                let mut features = values.clone();
                features.extend(urow.map_or(&zeros, |row| &row.values));
                features.extend(crow.map_or(&zeros, |row| &row.values));
                scores.push((c, model.scores(&features)[0]));
            }
            pairs.push(Pair { unit: u.line.clone(), candidate: cu.line.clone(), values });
        }
        // The best score, nearest first on a tie (the candidates come nearest first).
        let decided = scores.iter().fold(None::<(usize, f64)>, |best, &(c, n)| match best {
            Some((_, m)) if m >= n => best,
            _ => Some((c, n)),
        });
        // On a recognized (scanned) line, a best score below the model's floor leaves the unit at the top.
        let recognized = urow.is_some_and(|row| row.recognized);
        let decided = decided.filter(|(_, score)| !recognized || model.and_then(|m| m.none_below).is_none_or(|floor| *score >= floor));
        chosen[k] = if model.is_some() { decided.map(|(c, _)| c) } else { reader[k] };
        if let Some(p) = chosen[k] {
            *nested.entry((units[k].signature.as_str(), units[p].signature.as_str())).or_default() += 1;
        }
    }
    if model.is_some() {
        // A unit given no parent keeps its nearest ancestor that is no unit (a page, the document).
        let outside = |n: usize| {
            let mut parent = nodes[n].parent_id.as_deref().and_then(|id| ids.get(id).copied());
            let mut steps = 0;
            while let Some(p) = parent {
                if !matches!(nodes[p].kind, NodeKind::Heading | NodeKind::Paragraph | NodeKind::Section | NodeKind::ListItem) {
                    return Some(nodes[p].id.clone());
                }
                parent = nodes[p].parent_id.as_deref().and_then(|id| ids.get(id).copied());
                steps += 1;
                if steps > 256 { break; }
            }
            None
        };
        let new_parents = units
            .iter()
            .zip(&chosen)
            .map(|(u, c)| (u.node, c.map_or_else(|| outside(u.node), |c| Some(nodes[units[c].node].id.clone()))))
            .collect::<Vec<_>>();
        for (node, parent) in new_parents {
            structure.nodes[node].parent_id = parent;
        }
    }
    pairs
}
