//! How numbering runs in real Canadian legal text, as counted over A2AJ's cases and laws and the court
//! sites' pages (the tables of `fable-tables/build_tables.py`, LEGALPDF_TRANSITION_GRAMMAR): for the
//! family open on top of the stack, the log-probability of each relation the next marker takes to it
//! (the next value, another value of the same family, a pop to an open family's next value or another,
//! a push of a new family at 1 or another, no marker); for a family open, the log-probability of each
//! family pushed under it; and how far each first word of a line is a heading's. A move reads both
//! the federal laws' tables and every source's pooled, with the document's operative words beside
//! them to weigh which it goes by; headings' words come from the pages and the laws, whose headings
//! their markup marks.

use super::*;
use std::path::Path;

/// The relations a marker takes to the family open on top of the stack, in the tables' names.
pub const NEXT: &str = "next";
pub const SAME_OTHER: &str = "same_other";
pub const PUSH_1: &str = "push_1";
pub const PUSH_OTHER: &str = "push_other";
pub const POP_NEXT: &str = "pop_next";
pub const POP_OTHER: &str = "pop_other";
pub const NONE: &str = "none";

/// One source group's tables.
#[derive(Default)]
pub struct Tables {
    transitions: HashMap<String, HashMap<String, f32>>,
    children: HashMap<String, (HashMap<String, f32>, f32)>,
    heading_words: HashMap<String, f32>,
}

impl Tables {
    /// The log-probability of `relation` after `top` (`<other>`'s row for a family the tables keep no
    /// row of).
    pub fn relation(&self, top: &str, relation: &str) -> Option<f32> {
        let row = self.transitions.get(top).or_else(|| self.transitions.get("<other>"))?;
        row.get(relation).copied()
    }

    /// The log-probability of `child` pushed under `parent`.
    pub fn child(&self, parent: &str, child: &str) -> Option<f32> {
        let (row, unseen) = self.children.get(parent).or_else(|| self.children.get("<other>"))?;
        Some(row.get(child).copied().unwrap_or(*unseen))
    }

    /// The log-odds of `word` (lower-cased) opening a heading.
    pub fn heading_word(&self, word: &str) -> Option<f32> {
        self.heading_words.get(word).copied()
    }
}

pub struct Grammar {
    pub instruments: Tables,
    pub others: Tables,
}

fn number(value: &serde_json::Value) -> Option<f32> {
    value.as_f64().map(|v| v as f32)
}

fn tables(root: &serde_json::Value, group: &str, heading_groups: &[&str]) -> Tables {
    let mut out = Tables::default();
    if let Some(rows) = root["transitions"][group].as_object() {
        for (top, row) in rows {
            let lp = row["lp"].as_object().map(|lp| lp.iter().filter_map(|(k, v)| Some((k.clone(), number(v)?))).collect()).unwrap_or_default();
            out.transitions.insert(top.clone(), lp);
        }
    }
    if let Some(rows) = root["children"][group].as_object() {
        for (parent, row) in rows {
            let lp = row["lp"].as_object().map(|lp| lp.iter().filter_map(|(k, v)| Some((k.clone(), number(v)?))).collect()).unwrap_or_default();
            out.children.insert(parent.clone(), (lp, number(&row["lp_unseen"]).unwrap_or(-12.0)));
        }
    }
    for heading_group in heading_groups {
        for row in root["headings"][*heading_group]["w1"].as_array().into_iter().flatten() {
            if let (Some(word), Some(odds)) = (row[0].as_str(), number(&row[3])) {
                out.heading_words.entry(word.to_owned()).or_insert(odds);
            }
        }
    }
    out
}

impl Grammar {
    pub fn read(path: &Path) -> std::result::Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let root: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Self {
            instruments: tables(&root, "legislation:FED", &["legislation:FED", "legislation:ON"]),
            others: tables(&root, "all", &["decisions.scc-csc.ca", "decisions.fca-caf.gc.ca", "decisions.fct-cf.gc.ca", "coadecisions.ontariocourts.ca"]),
        })
    }

}

/// A line's first word, lower-cased, its opening marker skipped where it has one.
pub fn first_word(text: &str, marked: bool) -> String {
    let mut words = text.split_whitespace();
    if marked {
        words.next();
    }
    words.next().map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase()).unwrap_or_default()
}
