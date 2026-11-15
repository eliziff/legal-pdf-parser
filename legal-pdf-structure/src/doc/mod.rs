//! Tier 1 over tier 0's reading: Detect-Order-Construct (`reader`, with `encoder`) reads the document's lines, measured over
//! tier 0's line table and decision, and gives each line a move (`machine::Move`); the writer takes them in place of tier
//! 0's, the notes aside. A parse given a `Doc` (the engine's, where the request names DOC and its mode is promoted) reads
//! with it. What else it reads is one value, `Settings`, filled once from the environment for the harnesses:
//! - LEGALPDF_DOC, a folder holding Detect-Order-Construct's stages as `docparse.py` exports them; LEGALPDF_DOC_TIMES,
//!   each document's stages' times;
//! - LEGALPDF_DOC_ORACLE, the gold given: the document read by the same decoding from the heads' answers read off the
//!   gold (`reader::oracle`), what the decoding and the moves can hold;
//! - LEGALPDF_DOC_GOLD, the gold's labels (the grader's predictions format, rows' gold), for the oracle and the dump;
//! - LEGALPDF_DOC_TEXT, the text model the dump tokenizes each line's text with. The dump itself is the caller's to
//!   name (`derive`'s `dump`): each document's entry (`Entry`, `<sha>.dump`), which a grade resumes (`resume_many`).

use legal_pdf_core::model::Page;
use crate::decide::{Decision, Notes};
use crate::fences::Fences;
use crate::model::{Evidence, Slots};
use std::{collections::{HashMap, HashSet}, path::{Path, PathBuf}, sync::OnceLock};

use crate::machine::{lines::*, machine::*};
use crate::model::regions::*;
mod stylesheet;
pub(crate) use stylesheet::*;
#[cfg(feature = "encoder")]
#[macro_use]
mod onnx;
mod entry;
pub use entry::Entry;
mod inputs;
pub(crate) use inputs::Inputs;
mod reader;
mod scale;
mod targets;
#[cfg(feature = "encoder")]
mod trees;
pub use targets::targets;
pub mod tensors;

/// Tier 1, DOC's stages loaded (`Doc::load`), which a parse reads with where it is given one; none without `encoder`.
#[cfg(feature = "encoder")]
pub use reader::Reader as Doc;
#[cfg(not(feature = "encoder"))]
pub enum Doc {}
#[cfg(not(feature = "encoder"))]
impl Doc {
    pub fn load(_: &Path, _: Option<&Path>, _: bool) -> Result<Self, String> {
        Err("this parser was built without DOC (the encoder feature)".to_owned())
    }
}

/// The region-and-opening classes (`region * 2 + opens`) DOC's line head scores.
const CLASSES: usize = 34;
/// The gold region of a line a gold reads only as body prose, a quotation or list item set in it unmarked
/// (the grader's `labels`, A2AJ's alignment).
const UNMARKED_BODY: i64 = 101;

/// The gold's labels by document: each piece's row, each piece's reading rank where the gold gives one, and the pieces
/// in the order of the rows, which a row's pointer counts; and each note's first piece's paired reference (a row).
type GoldLabels = (HashMap<String, HashMap<String, [i64; 8]>>, HashMap<String, HashMap<String, i64>>, HashMap<String, Vec<String>>, HashMap<String, HashMap<String, i64>>);

/// What tier 1 is given: DOC's stages, the gold where it is measured, and what it writes.
pub(crate) struct Settings {
    gold: Option<GoldLabels>,
    #[cfg(feature = "encoder")]
    doc: Option<reader::Reader>,
    /// The text model the dump writes each line's text vector with (LEGALPDF_DOC_TEXT, an export's folder).
    #[cfg(feature = "encoder")]
    text: Option<reader::TextModel>,
    doc_oracle: bool,
}

/// The harnesses' settings, read once: the entry points' only source of them.
fn settings() -> &'static Settings {
    static SETTINGS: OnceLock<Settings> = OnceLock::new();
    SETTINGS.get_or_init(Settings::from_environment)
}

impl Settings {
    /// The settings the LEGALPDF_DOC_ names in the environment give.
    fn from_environment() -> Self {
        let var = |name: &str| std::env::var_os(name).map(PathBuf::from);
        Settings {
            gold: var("LEGALPDF_DOC_GOLD").map(|path| read_gold(&path)),
            doc_oracle: std::env::var_os("LEGALPDF_DOC_ORACLE").is_some(),
            #[cfg(feature = "encoder")]
            text: var("LEGALPDF_DOC_TEXT").and_then(|dir| reader::TextModel::load(&dir).map_err(|e| eprintln!("DOC text model: {e}")).ok()),
            #[cfg(feature = "encoder")]
            doc: var("LEGALPDF_DOC").and_then(|dir| {
                reader::Reader::load(&dir, None, std::env::var_os("LEGALPDF_DOC_TIMES").is_some())
                    .map_err(|e| eprintln!("DOC: {e}")).ok()
            }),
        }
    }
}

/// The gold's labels, from rows of the grader's predictions format ({sha256, ids, labels}): each document's by piece
/// (`line@offset`), each piece's reading rank (the ninth column), its pieces in row order, and a note's paired reference
/// (the tenth).
fn read_gold(path: &Path) -> GoldLabels {
    let mut found = GoldLabels::default();
    let Ok(text) = std::fs::read_to_string(path) else {
        eprintln!("DOC: no gold at {}", path.display());
        return found;
    };
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(row) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        let (Some(sha), Some(ids), Some(labels)) = (row["sha256"].as_str(), row["ids"].as_array(), row["labels"].as_array()) else { continue };
        let (mut per_piece, mut ranks, mut refs) = (HashMap::new(), HashMap::new(), HashMap::new());
        for (id, label) in ids.iter().zip(labels) {
            let piece = id.as_str().unwrap_or_default().to_owned();
            let cell = |k: usize| label.get(k).and_then(serde_json::Value::as_i64);
            if let Some(rank) = cell(8).filter(|&r| r >= 0) { ranks.insert(piece.clone(), rank); }
            if let Some(at) = cell(9).filter(|&r| r >= 0) { refs.insert(piece.clone(), at); }
            per_piece.insert(piece, std::array::from_fn::<i64, 8, _>(|k| cell(k).unwrap_or(-1)));
        }
        found.0.insert(sha.to_owned(), per_piece);
        found.1.insert(sha.to_owned(), ranks);
        found.2.insert(sha.to_owned(), ids.iter().filter_map(|id| id.as_str().map(str::to_owned)).collect());
        found.3.insert(sha.to_owned(), refs);
    }
    found
}

/// Kept parses read by DOC together (`Reader::read_many`), each from its inputs as its entry keeps them: each one's
/// moves, none where DOC is not given, its inputs could not be read, or DOC failed it. Each document: its pages, slots,
/// fences, line table, line views, inputs and source.
#[allow(clippy::type_complexity)]
pub(crate) fn read_kept(docs: &[(&[Page], &Slots, &Fences, &LineTable, &[LineView], Option<&Inputs>, &str)], doc: Option<&Doc>) -> Vec<Option<Vec<Move>>> {
    let mut out = vec![None; docs.len()];
    #[cfg(feature = "encoder")]
    if let Some(reader) = doc.or(settings().doc.as_ref()) {
        let texts: Vec<Vec<&str>> = docs.iter().map(|&(pages, slots, fences, ..)| fences.order.iter().map(|&slot| slots.line(pages, slot).text.as_str()).collect()).collect();
        let readable: Vec<usize> = (0..docs.len()).filter(|&at| docs[at].5.is_some()).collect();
        let readings: Vec<reader::Reading> = readable.iter().map(|&at| {
            let (_, _, _, table, views, inputs, _) = docs[at];
            reader::Reading { inputs: inputs.expect("readable"), table, views, texts: &texts[at], gold: None }
        }).collect();
        for (&at, moves) in readable.iter().zip(reader.read_many(&readings)) {
            out[at] = moves.map_err(|error| eprintln!("DOC {}: {error}", docs[at].6)).ok();
        }
    }
    #[cfg(not(feature = "encoder"))]
    let _ = doc;
    out
}

/// What DOC reads of a document and what its dump entry records of it, made once from the parse: its lines' texts and
/// ids, its inputs (`Inputs`), and the gold's rows, reading ranks and paired references where the gold is given here.
struct Measured<'a> {
    texts: Vec<&'a str>,
    ids: Vec<&'a str>,
    inputs: Inputs,
    gold: Option<&'a HashMap<String, [i64; 8]>>,
    gold_rows: Vec<[i64; 8]>,
    ranks: Vec<i64>,
    note_refs: Vec<i64>,
}

/// DOC's moves over tier 0's reading of a document (its notes and decision, `table` the line table both tiers read),
/// where its reader (`doc`, else the harnesses') or its oracle is given, each move's `reading` naming one of the line's
/// readings in `table`; and its dump entry, where `dump` names the dump (`parse` the parse as tier 0 left it, as the
/// entry keeps it). None where neither reads: the output is tier 0's.
#[allow(clippy::too_many_arguments)]
pub(crate) fn over_tier0(pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, geometry: &crate::layout::Geometry, sha256: &str,
    notes: &Notes, decision: &Decision, table: &LineTable, views: &[LineView], doc: Option<&Doc>, dump: Option<(crate::Keep<'_>, &[u8])>) -> Option<Vec<Move>> {
    let s = settings();
    let oracle = s.doc_oracle && s.gold.as_ref().is_some_and(|g| g.0.contains_key(sha256));
    #[cfg(feature = "encoder")]
    let reader = doc.or(s.doc.as_ref());
    #[cfg(not(feature = "encoder"))]
    let reader = doc;
    if reader.is_none() && !oracle && dump.is_none() { return None; }
    let measured = measure(pages, slots, evidence, fences, geometry, sha256, notes, decision, table, reader)?;
    if let Some((kept, parse)) = dump {
        keep(kept, sha256, &measured, table, views, parse).unwrap_or_else(|error| eprintln!("DOC dump {sha256}: {error}"));
    }
    // Detect-Order-Construct's decoding from the gold's answers, else its reading where its stages are given.
    if oracle {
        return Some(reader::oracle(&measured.gold_rows, views, &measured.texts, &measured.inputs.pages));
    }
    #[cfg(feature = "encoder")]
    return reader.and_then(|reader| reader.read(&measured.inputs, table, views, &measured.texts,
        measured.gold.is_some().then_some(&measured.gold_rows[..]).filter(|_| !reader::oracle_parts().is_empty()))
        .map_err(|error| eprintln!("DOC {sha256}: {error}")).ok());
    #[cfg(not(feature = "encoder"))]
    None
}

/// A document measured as DOC reads it (`Measured`): its lines' tokens by the reader's text model, else the one the dump
/// is written with.
#[allow(clippy::too_many_arguments)]
fn measure<'a>(pages: &'a [Page], slots: &Slots, evidence: &Evidence, fences: &Fences, geometry: &crate::layout::Geometry, sha256: &str,
    notes: &Notes, decision: &Decision, table: &LineTable, reader: Option<&Doc>) -> Option<Measured<'a>> {
    let s = settings();
    let texts: Vec<&str> = fences.order.iter().map(|&slot| slots.line(pages, slot).text.as_str()).collect();
    let ids: Vec<&str> = fences.order.iter().map(|&slot| slots.line(pages, slot).id.as_str()).collect();
    #[cfg(feature = "encoder")]
    let model = reader.map(|reader| reader.text()).or(s.text.as_ref());
    #[cfg(feature = "encoder")]
    let tokenize = model.map(|model| move |texts: &[&str]| model.tokens(texts));
    #[cfg(feature = "encoder")]
    let tokenize = tokenize.as_ref().map(|f| f as &dyn Fn(&[&str]) -> Result<(Vec<u32>, Vec<usize>), String>);
    #[cfg(not(feature = "encoder"))]
    let tokenize = { let _ = reader; None };
    let inputs = Inputs::of(pages, slots, evidence, fences, geometry, notes, decision, table, &texts, tokenize)
        .map_err(|error| eprintln!("DOC {sha256}: {error}")).ok()?;
    let gold = s.gold.as_ref().and_then(|g| g.0.get(sha256));
    // Each line's reading rank in the gold, -1 for none.
    let ranks: Vec<i64> = {
        let ranks = s.gold.as_ref().and_then(|g| g.1.get(sha256));
        ids.iter().map(|id| ranks.and_then(|r| r.get(&format!("{id}@0"))).copied().unwrap_or(-1)).collect()
    };
    // A row's pointer names a row of the gold's pieces (a line may hold several): here, the place of that piece's line.
    let place: HashMap<&str, i64> = ids.iter().enumerate().map(|(k, &id)| (id, k as i64)).collect();
    let pieces = s.gold.as_ref().and_then(|g| g.2.get(sha256));
    // A note's paired reference, a row of the gold's pieces too: the place of that piece's line, -1 for none.
    let line_of = |row: i64| pieces.and_then(|pieces| pieces.get(row as usize)).and_then(|piece| place.get(piece.split('@').next().unwrap_or(piece))).copied().unwrap_or(-1);
    let note_refs: Vec<i64> = {
        let refs = s.gold.as_ref().and_then(|g| g.3.get(sha256));
        ids.iter().map(|id| refs.and_then(|r| r.get(&format!("{id}@0"))).map_or(-1, |&row| line_of(row))).collect()
    };
    let gold_rows: Vec<[i64; 8]> = ids.iter().map(|id| {
        let mut row = gold.and_then(|g| g.get(&format!("{id}@0"))).copied().unwrap_or([-1; 8]);
        if row[6] >= 0 {
            row[6] = line_of(row[6]);
        }
        row
    }).collect();
    Some(Measured { texts, ids, inputs, gold, gold_rows, ranks, note_refs })
}

/// A document's dump entry written (`Entry`): its inputs, its record (the lines and their places, reference marks and
/// note numbers, what the grader reads back), its targets where the gold is given here, and the parse as tier 0 left it.
/// The entry is keyed by what the parse read, not by the build that read it: a document this build reads as the last one
/// did (the same arrays, lines and texts) keeps its targets and everything the fit made of them. The parse is kept under
/// `kept.key` (`read_key`), what its reader reads it back by.
fn keep(kept: crate::Keep<'_>, sha256: &str, measured: &Measured, table: &LineTable, views: &[LineView], parse: &[u8]) -> std::io::Result<()> {
    let folder = kept.folder;
    use std::hash::{Hash, Hasher};
    let Measured { texts, ids, inputs, gold, gold_rows, ranks, note_refs } = measured;
    let note_numbers: Vec<i64> = views.iter().zip(texts).map(|(view, text)| reader::note_number(view, text).map_or(-1, i64::from)).collect();
    // What the fit reads, as arrays, each once and in the form the reader reads them. What DOC reads of the lines around
    // a line, of a region's lines, of every two lines or regions and of a tree's nodes it makes from these (its context
    // graph, the fit's step), so nothing here repeats a line.
    let arrays = inputs.tensors();
    let mut record = serde_json::json!({
        "sha256": sha256, "ids": ids, "lines": ids.len(), "pages": inputs.pages, "references": &table.references, "ranks": ranks,
        "note_numbers": note_numbers,
    });
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    arrays.iter().for_each(|t| { t.name.hash(&mut hasher); t.bytes.hash(&mut hasher); });
    serde_json::to_vec(&record)?.hash(&mut hasher);
    texts.hash(&mut hasher);
    let read = format!("{:016x}", hasher.finish());
    let mut entry = Entry::read(folder, sha256).unwrap_or_default();
    if entry.record["parse_hash"] == read.as_str() {
        // The same reading: its record, labels and targets stand; the parse is kept anew (the code that keeps it may have
        // changed what it holds, never what it read).
        entry.set_bytes("parse", parse);
        entry.record["read_key"] = serde_json::Value::from(kept.key);
        return entry.write(folder, sha256);
    }
    // A parse read anew keeps no targets of the last one: the grader writes them again, where the gold is not here.
    entry.set_part("targets", Vec::new());
    if gold.is_some() {
        let (made, summary) = targets::targets(gold_rows, &inputs.pages, note_refs, ranks, &table.references, &note_numbers, &arrays);
        entry.set_part("targets", made);
        record["summary"] = summary;
    }
    entry.set_part("inputs", arrays);
    entry.set_bytes("parse", parse);
    record["parse_hash"] = serde_json::Value::from(read);
    record["read_key"] = serde_json::Value::from(kept.key);
    entry.record = record;
    entry.write(folder, sha256)
}
