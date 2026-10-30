//! The court-forms lexicon (tools/build_witness_lexicon.py): forms' titles, courts' names and the
//! lines printed across many forms. `LEGALPDF_WITNESS_LEXICON` names another file.

use aho_corasick::AhoCorasick;
use serde::Deserialize;
use std::collections::HashSet;
use std::sync::OnceLock;

const SHIPPED: &str = include_str!("../data/lexicon.json");

#[derive(Deserialize, Default)]
struct Words {
    #[serde(default)]
    form_titles: Vec<String>,
    #[serde(default)]
    courts: Vec<String>,
    #[serde(default)]
    form_boilerplate: Vec<String>,
}

pub struct Lexicon {
    pub text: String,
    titles: HashSet<String>,
    /// Each court's name between spaces, as a line's words are read for it: one automaton finding
    /// any of them in a line.
    courts: AhoCorasick,
    boilerplate: HashSet<String>,
}

/// What the lexicon says of one line: a form's title, a court's name, a line printed across forms.
pub struct Reading {
    pub form_title: bool,
    pub court: bool,
    pub boilerplate: bool,
}

pub fn normal(text: &str) -> String {
    text.to_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

impl Lexicon {
    /// The line read once against the lexicon: a form's title, a court the forms come from (named
    /// anywhere in it, word for word), a line printed across many forms.
    pub fn read(&self, text: &str) -> Reading {
        let key = normal(text);
        let spaced = format!(" {key} ");
        Reading {
            form_title: !key.is_empty() && self.titles.contains(&key),
            court: self.courts.is_match(&spaced),
            boilerplate: self.boilerplate.contains(&key),
        }
    }
}

pub fn lexicon() -> &'static Lexicon {
    static LOADED: OnceLock<Lexicon> = OnceLock::new();
    LOADED.get_or_init(|| {
        let text = std::env::var("LEGALPDF_WITNESS_LEXICON")
            .ok()
            .and_then(|path| std::fs::read_to_string(path).ok())
            .unwrap_or_else(|| SHIPPED.to_owned());
        let words: Words = serde_json::from_str(&text).unwrap_or_default();
        Lexicon {
            titles: words.form_titles.into_iter().collect(),
            courts: AhoCorasick::new(words.courts.iter().map(|court| format!(" {court} "))).expect("courts' names"),
            boilerplate: words.form_boilerplate.into_iter().collect(),
            text,
        }
    })
}
