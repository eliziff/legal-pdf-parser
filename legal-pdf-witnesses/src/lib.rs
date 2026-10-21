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
mod outline;
mod witness;

pub use compositions::{compositions, compositions_sha256, Compositions, Decision};
pub use outline::{assemble_outline, Context as OutlineContext};
pub use witness::{witnesses, Context, Row, Table, NAMES};

/// What the structure stage read while a `capture` ran: every witness table, by stage, and the
/// outline's own evidence: each bookmarked heading line's parent heading line (`None`: at the top)
/// and each heading line's level on the document's ladder.
#[derive(Default)]
pub struct Captured {
    pub witnesses: Vec<(String, Row)>,
    pub navigation: std::collections::HashMap<String, Option<String>>,
    pub heading_levels: std::collections::HashMap<String, usize>,
}

thread_local! {
    static CAPTURED: std::cell::RefCell<Option<Captured>> = const { std::cell::RefCell::new(None) };
}

/// Run `read`, keeping what the structure stage reads on this thread meanwhile: each line's
/// evidence exactly as the decisions saw it, for a training export.
pub fn capture<R>(read: impl FnOnce() -> R) -> (R, Captured) {
    CAPTURED.with(|captured| *captured.borrow_mut() = Some(Captured::default()));
    let result = read();
    (result, CAPTURED.with(|captured| captured.borrow_mut().take()).unwrap_or_default())
}

fn keep(add: impl FnOnce(&mut Captured)) {
    CAPTURED.with(|captured| {
        if let Some(captured) = captured.borrow_mut().as_mut() {
            add(captured);
        }
    });
}

/// Write the table to the dump file `LEGALPDF_WITNESS_DUMP` names, if any, one JSON line per row,
/// and keep it for a `capture` running on this thread. `stage` tells the dumps apart where the
/// witnesses are read more than once.
pub fn dump(table: &Table, stage: &str) {
    keep(|captured| captured.witnesses.extend(table.rows.iter().map(|row| (stage.to_owned(), row.clone()))));
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
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = file.write_all(out.as_bytes());
    }
}


