use crate::{Error, Result};
use fancy_regex::{Match as FancyMatch, Regex as FancyRegex};
use std::collections::BTreeMap;

pub use legal_grammar_tables::TableEntry;

pub fn compile_table_entry(entry_id: &str) -> Result<FancyRegex> {
    legal_grammar_tables::compile_table_entry(entry_id)
        .map_err(|error| Error::Message(error.to_string()))
}
pub fn load_tables() -> Result<&'static BTreeMap<String, TableEntry>> {
    legal_grammar_tables::load_tables().map_err(|error| Error::Message(error.to_string()))
}

pub fn find_table_matches<'a>(
    entry_id: &str,
    regex: &FancyRegex,
    text: &'a str,
) -> Result<Vec<FancyMatch<'a>>> {
    legal_grammar_tables::find_table_matches(entry_id, regex, text)
        .map_err(|error| Error::Message(error.to_string()))
}

pub fn run_vectors() -> Result<Vec<String>> {
    legal_grammar_tables::run_vectors().map_err(|error| Error::Message(error.to_string()))
}

#[cfg(test)]
mod tests {
    #[test]
    fn frozen_tables_compile_and_pass_every_oracle_vector() {
        let failures = super::run_vectors().unwrap();
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
