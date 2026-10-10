//! The court-forms lexicon (tools/build_witness_lexicon.py): forms' titles, courts' names and the
//! lines printed across many forms. `LEGALPDF_WITNESS_LEXICON` names another file.

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
    courts: Vec<String>,
    boilerplate: HashSet<String>,
}

pub fn normal(text: &str) -> String {
    text.to_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

impl Lexicon {
    /// The line is a form's title.
    pub fn form_title(&self, text: &str) -> bool {
        let key = normal(text);
        !key.is_empty() && self.titles.contains(&key)
    }

    /// The line names a court the forms come from.
    pub fn court(&self, text: &str) -> bool {
        let key = format!(" {} ", normal(text));
        self.courts.iter().any(|court| key.contains(&format!(" {court} ")))
    }

    /// The line is printed across many forms.
    pub fn boilerplate(&self, text: &str) -> bool {
        self.boilerplate.contains(&normal(text))
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
            courts: words.courts,
            boilerplate: words.form_boilerplate.into_iter().collect(),
            text,
        }
    })
}
