//! Decisions as data: per decision, for typeset and for recognized lines, a union of conjunctions,
//! each literal a witness compared with a threshold.

use crate::witness::{index, Row};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::OnceLock;

const SHIPPED: &str = include_str!("../data/compositions.json");

#[derive(Deserialize)]
struct Literal(String, String, f32);

#[derive(Deserialize, Default)]
struct Sets {
    #[serde(default)]
    typeset: Vec<Vec<Literal>>,
    #[serde(default)]
    recognized: Vec<Vec<Literal>>,
}

/// One decision's compiled conjunctions: (witness index, at least?, threshold).
pub struct Decision {
    typeset: Vec<Vec<(usize, bool, f32)>>,
    recognized: Vec<Vec<(usize, bool, f32)>>,
}

fn satisfied(sets: &[Vec<(usize, bool, f32)>], values: &[f32]) -> bool {
    sets.iter().any(|conjunction| {
        conjunction.iter().all(|&(at, least, threshold)| if least { values[at] >= threshold } else { values[at] < threshold })
    })
}

impl Decision {
    /// Whether the decision holds on a row: some conjunction of its line's kind holds. None where
    /// the decision has no conjunctions for that kind of line, so the parser keeps its own reading.
    pub fn holds(&self, row: &Row) -> Option<bool> {
        let sets = if row.recognized { &self.recognized } else { &self.typeset };
        (!sets.is_empty()).then(|| satisfied(sets, &row.values))
    }
}

pub struct Compositions {
    decisions: HashMap<String, Decision>,
    sha256: String,
}

impl Compositions {
    pub fn decision(&self, name: &str) -> Option<&Decision> {
        self.decisions.get(name)
    }
}

fn compile(sets: Vec<Vec<Literal>>, index: &dyn Fn(&str) -> Option<usize>) -> Vec<Vec<(usize, bool, f32)>> {
    sets.into_iter()
        .filter_map(|conjunction| {
            conjunction
                .into_iter()
                .map(|Literal(name, op, threshold)| index(&name).map(|at| (at, op == ">=", threshold)))
                .collect::<Option<Vec<_>>>()
        })
        .collect()
}

/// The compositions in force: `LEGALPDF_COMPOSITIONS` names a file, else the shipped ones.
pub fn compositions() -> &'static Compositions {
    static LOADED: OnceLock<Compositions> = OnceLock::new();
    LOADED.get_or_init(|| {
        let text = std::env::var("LEGALPDF_COMPOSITIONS")
            .ok()
            .and_then(|path| std::fs::read_to_string(path).ok())
            .unwrap_or_else(|| SHIPPED.to_owned());
        // What the compositions decide rests on the lexicon's words and the outline prior too.
        let sha256 = format!(
            "{:x}",
            Sha256::digest(format!("{text}\n{}\n{}\n{}", crate::lexicon::lexicon().text, crate::outline::prior().0, crate::model::models_text()).as_bytes())
        );
        let value = serde_json::from_str::<serde_json::Value>(&text).unwrap_or_default();
        let parsed: HashMap<String, Sets> = value
            .get("decisions")
            .cloned()
            .and_then(|value| serde_json::from_value(value).ok())
            .unwrap_or_default();
        let decisions = parsed
            .into_iter()
            .map(|(name, sets)| {
                (name, Decision { typeset: compile(sets.typeset, &index), recognized: compile(sets.recognized, &index) })
            })
            .collect();
        Compositions { decisions, sha256 }
    })
}

/// The digest of the compositions in force, for the caches of what they decide.
pub fn compositions_sha256() -> &'static str {
    &compositions().sha256
}
