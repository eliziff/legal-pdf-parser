//! The structure decisions composed over every line's witnesses.
//!
//! Each counted line (body or heading, outside notes and furniture) gets one witness vector, read
//! against its own page and document: its type (bold, italic, size, a font of its own, an
//! underline), its geometry (height, width, indent, centring, the space around it), its words (its
//! ending, its marker's family, a lead-in before it, a court's name, an address, a party's label,
//! a citation), and its document (the style ladder, the backbone counts, the contents list, the
//! pages set at another width or switching between scan and type, the text repeated page after
//! page), with the readers' verdicts so far (a heading, a quotation, a table cell, a layout title).
//!
//! A decision is a union of conjunctions over these witnesses, one set for typeset lines and one for
//! recognized lines, loaded as data (`LEGALPDF_COMPOSITIONS`, or the compositions shipped in
//! `data/compositions.json`): the structure gold's composition search writes them, and changing them
//! needs no build. `LEGALPDF_WITNESS_DUMP` writes every line's witnesses for that search.

mod compositions;
mod lexicon;
mod model;
mod outline;
mod witness;

pub use compositions::{compositions, compositions_sha256, Compositions, Decision};
pub use model::{parent_model, unit_kinds, UNIT_KINDS};
pub use outline::{compose_outline, Pair, PAIR_NAMES};
pub use witness::{witnesses, Context, Row, Table, NAMES};

/// Write the outline's pairs to the dump file, one JSON line per pair.
pub fn dump_pairs(pairs: &[Pair]) {
    let Ok(path) = std::env::var("LEGALPDF_WITNESS_DUMP") else {
        return;
    };
    let mut out = String::new();
    for pair in pairs {
        let values = PAIR_NAMES
            .iter()
            .zip(&pair.values)
            .map(|(name, value)| (name.to_string(), serde_json::json!(value)))
            .collect::<serde_json::Map<_, _>>();
        out.push_str(&serde_json::json!({"stage": "outline", "id": pair.unit, "candidate": pair.candidate, "w": values}).to_string());
        out.push('\n');
    }
    use std::io::Write;
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = file.write_all(out.as_bytes());
    }
}

/// Write the table to the dump file `LEGALPDF_WITNESS_DUMP` names, if any, one JSON line per row.
/// `stage` tells the dumps apart where the witnesses are read more than once.
pub fn dump(table: &Table, stage: &str) {
    let Ok(path) = std::env::var("LEGALPDF_WITNESS_DUMP") else {
        return;
    };
    let mut out = String::new();
    for row in &table.rows {
        let values = NAMES
            .iter()
            .zip(&row.values)
            .map(|(name, value)| (name.to_string(), serde_json::json!(value)))
            .collect::<serde_json::Map<_, _>>();
        out.push_str(&serde_json::json!({"stage": stage, "id": row.id, "page": row.page, "text": row.text, "w": values}).to_string());
        out.push('\n');
    }
    use std::io::Write;
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = file.write_all(out.as_bytes());
    }
}
