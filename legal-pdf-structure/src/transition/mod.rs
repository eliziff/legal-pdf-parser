//! Tier 1 over tier 0's reading: Detect-Order-Construct (`doc`, with `encoder`) reads the document's lines, measured over
//! tier 0's line table and decision, and gives each line a move (`machine::Move`); the writer takes them in place of tier
//! 0's, the notes aside (`crate::emit`). What it reads is one value, `Settings`, filled once from the environment's
//! LEGALPDF_TRANSITION_ names for the harnesses:
//! - LEGALPDF_TRANSITION_DOC, a folder holding Detect-Order-Construct's stages as `docparse.py` exports them (`doc`);
//!   LEGALPDF_TRANSITION_DOC_ORDER=on, the reading's order read off the successor head; LEGALPDF_TRANSITION_DOC_TIMES,
//!   each document's stages' times;
//! - LEGALPDF_TRANSITION_DOC_ORACLE, the gold given: the document read by the same decoding from the heads' answers read
//!   off the gold (`doc::oracle`), what the decoding and the moves can hold;
//! - LEGALPDF_TRANSITION_GOLD, the gold's labels (the grader's predictions format, rows' gold), for the oracle and the dump;
//! - LEGALPDF_TRANSITION_CONFUSIONS, how recognition misreads the characters of a line's opening marker, counted over
//!   the gold's corrected lines ({read: {meant: share}}): a recognized line's marker is read too as each misreading would
//!   have it, in the line table both tiers read;
//! - LEGALPDF_TRANSITION_DUMP, a folder: each document's line features (`<sha>.lines.f32`) and what DOC is fitted on
//!   beside them (`<sha>.json`).

use legal_pdf_core::model::{Page, PdfOutlineEntry};
use crate::decide::{Decision, Notes};
use crate::fences::Fences;
use crate::model::{Evidence, Slots};
use std::{collections::{HashMap, HashSet}, path::{Path, PathBuf}, sync::OnceLock};

pub use crate::machine::lines::*;
pub use crate::machine::machine::*;
use crate::emit as emit;
mod stylesheet;
pub use stylesheet::*;
#[cfg(feature = "encoder")]
#[macro_use]
mod onnx;
mod doc;

/// The region-and-opening classes (`region * 2 + opens`) DOC's line head scores.
const CLASSES: usize = 34;
/// The gold region of a line a gold reads only as body prose, a quotation or list item set in it unmarked
/// (`gold_labels.py`, A2AJ's alignment).
const UNMARKED_BODY: i64 = 101;

/// The gold's labels by document: each piece's row, each piece's reading rank where the gold gives one, and the pieces
/// in the order of the rows, which a row's pointer counts.
type GoldLabels = (HashMap<String, HashMap<String, [i64; 8]>>, HashMap<String, HashMap<String, i64>>, HashMap<String, Vec<String>>);

/// What tier 1 is given: DOC's stages, the gold where it is measured, and what it writes.
pub struct Settings {
    gold: Option<GoldLabels>,
    dump: Option<PathBuf>,
    #[cfg(feature = "encoder")]
    doc: Option<doc::Reader>,
    doc_oracle: bool,
    pub confusions: HashMap<char, Vec<(char, f32)>>,
}

/// The harnesses' settings, read once: the entry points' only source of them.
fn settings() -> &'static Settings {
    static SETTINGS: OnceLock<Settings> = OnceLock::new();
    SETTINGS.get_or_init(Settings::from_environment)
}

impl Settings {
    /// The settings the LEGALPDF_TRANSITION_ names in the environment give.
    fn from_environment() -> Self {
        let var = |name: &str| std::env::var_os(name).map(PathBuf::from);
        Settings {
            gold: var("LEGALPDF_TRANSITION_GOLD").map(|path| read_gold(&path)),
            dump: var("LEGALPDF_TRANSITION_DUMP"),
            doc_oracle: std::env::var_os("LEGALPDF_TRANSITION_DOC_ORACLE").is_some(),
            #[cfg(feature = "encoder")]
            doc: var("LEGALPDF_TRANSITION_DOC").and_then(|dir| {
                doc::Reader::load(&dir, std::env::var("LEGALPDF_TRANSITION_DOC_ORDER").is_ok_and(|v| v == "on"), std::env::var_os("LEGALPDF_TRANSITION_DOC_TIMES").is_some())
                    .map_err(|e| eprintln!("transition parser: {e}")).ok()
            }),
            confusions: var("LEGALPDF_TRANSITION_CONFUSIONS").map(|path| read_confusions(&path)).unwrap_or_default(),
        }
    }
}

/// How recognition misreads the characters of a line's opening marker (LEGALPDF_TRANSITION_CONFUSIONS), for the line
/// table both tiers read; none where it is not given.
pub fn confusions() -> &'static HashMap<char, Vec<(char, f32)>> {
    &settings().confusions
}

/// Little-endian 32-bit floats, four bytes each.
#[cfg_attr(not(feature = "encoder"), allow(dead_code))]
fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

/// The confusion table at `path`: each misread character's meant ones, likeliest first.
fn read_confusions(path: &Path) -> HashMap<char, Vec<(char, f32)>> {
    let table: HashMap<String, HashMap<String, f32>> = std::fs::read_to_string(path).ok().and_then(|text| serde_json::from_str(&text).ok()).unwrap_or_else(|| {
        eprintln!("transition parser: no confusion table at {}", path.display());
        HashMap::new()
    });
    table.into_iter().filter_map(|(read, meant)| {
        let mut meant: Vec<(char, f32)> = meant.into_iter().filter_map(|(c, p)| Some((c.chars().next()?, p))).collect();
        meant.sort_by(|a, b| b.1.total_cmp(&a.1));
        Some((read.chars().next()?, meant))
    }).collect()
}

/// The gold's labels, from rows of the grader's predictions format ({sha256, ids, labels}): each document's by piece
/// (`line@offset`), each piece's reading rank, the labels' ninth column where a row has one, and its pieces in row order.
fn read_gold(path: &Path) -> GoldLabels {
    let mut found = GoldLabels::default();
    let Ok(text) = std::fs::read_to_string(path) else {
        eprintln!("transition parser: no gold at {}", path.display());
        return found;
    };
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(row) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        let (Some(sha), Some(ids), Some(labels)) = (row["sha256"].as_str(), row["ids"].as_array(), row["labels"].as_array()) else { continue };
        let (mut per_piece, mut ranks) = (HashMap::new(), HashMap::new());
        for (id, label) in ids.iter().zip(labels) {
            let piece = id.as_str().unwrap_or_default().to_owned();
            let cell = |k: usize| label.get(k).and_then(serde_json::Value::as_i64);
            if let Some(rank) = cell(8).filter(|&r| r >= 0) { ranks.insert(piece.clone(), rank); }
            per_piece.insert(piece, std::array::from_fn::<i64, 8, _>(|k| cell(k).unwrap_or(-1)));
        }
        found.0.insert(sha.to_owned(), per_piece);
        found.1.insert(sha.to_owned(), ranks);
        found.2.insert(sha.to_owned(), ids.iter().filter_map(|id| id.as_str().map(str::to_owned)).collect());
    }
    found
}

/// DOC's moves over tier 0's reading of a document (its notes and decision, `table` the line table both tiers read),
/// where its reader or its oracle is given, each move's `reading` naming one of the line's readings in `table`; and the
/// dump DOC is fitted on, where it is asked. None where neither reads: the output is tier 0's.
#[allow(clippy::too_many_arguments)]
pub fn over_tier0(pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, geometry: &crate::decide::Geometry, outline: &[PdfOutlineEntry], sha256: &str,
    notes: &Notes, decision: &Decision, table: &LineTable) -> Option<Vec<Move>> {
    let s = settings();
    let gold = s.gold.as_ref().and_then(|g| g.0.get(sha256));
    let oracle = s.doc_oracle && gold.is_some();
    #[cfg(feature = "encoder")]
    let reader = s.doc.as_ref();
    #[cfg(not(feature = "encoder"))]
    let reader: Option<()> = None;
    if reader.is_none() && !oracle && s.dump.is_none() { return None; }
    let measures = table.measure(pages, slots, evidence, fences, geometry, notes, decision);
    let sheet = Stylesheet::read(pages, slots, evidence, fences, geometry, outline, table, &measures);
    let views = LineView::all(evidence, fences, table);
    let texts: Vec<&str> = fences.order.iter().map(|&slot| slots.line(pages, slot).text.as_str()).collect();
    let ids: Vec<&str> = fences.order.iter().map(|&slot| slots.line(pages, slot).id.as_str()).collect();
    let line_pages: Vec<usize> = fences.order.iter().map(|&slot| slots.page(slot)).collect();
    // Each line's reading rank in the gold, -1 for none.
    let ranks: Vec<i64> = {
        let ranks = s.gold.as_ref().and_then(|g| g.1.get(sha256));
        ids.iter().map(|id| ranks.and_then(|r| r.get(&format!("{id}@0"))).copied().unwrap_or(-1)).collect()
    };
    // A row's pointer names a row of the gold's pieces (a line may hold several): here, the place of that piece's line.
    let place: HashMap<&str, i64> = ids.iter().enumerate().map(|(k, &id)| (id, k as i64)).collect();
    let pieces = s.gold.as_ref().and_then(|g| g.2.get(sha256));
    let gold_rows: Vec<[i64; 8]> = ids.iter().map(|id| {
        let mut row = gold.and_then(|g| g.get(&format!("{id}@0"))).copied().unwrap_or([-1; 8]);
        if row[6] >= 0 {
            row[6] = pieces.and_then(|pieces| pieces.get(row[6] as usize)).and_then(|piece| place.get(piece.split('@').next().unwrap_or(piece))).copied().unwrap_or(-1);
        }
        row
    }).collect();
    if let Some(folder) = &s.dump {
        let write = || -> std::io::Result<()> {
            std::fs::create_dir_all(folder)?;
            let bytes: Vec<u8> = measures.features.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(folder.join(format!("{sha256}.lines.f32")), bytes)?;
            let meta = serde_json::json!({
                "sha256": sha256, "ids": ids, "lines": ids.len(), "line_width": ALL,
                "texts": texts, "looks": &table.looks, "pages": line_pages, "references": &table.references, "ranks": ranks,
                "templates": &sheet.template, "template_rows": sheet.rows(table, &measures), "anchors": &sheet.anchor, "t0": &measures.t0,
                "note_numbers": views.iter().zip(&texts).map(|(view, text)| doc::note_number(view, text).map_or(-1, i64::from)).collect::<Vec<_>>(),
                "regions": REGIONS, "line_features": LINE_FEATURES, "gold": gold_rows,
            });
            std::fs::write(folder.join(format!("{sha256}.json")), serde_json::to_vec(&meta)?)
        };
        write().unwrap_or_else(|error| eprintln!("transition parser dump {sha256}: {error}"));
    }
    // Detect-Order-Construct's decoding from the gold's answers, else its reading where its stages are given.
    if oracle {
        return Some(doc::oracle(&gold_rows, &ranks, &views, &texts, &line_pages));
    }
    #[cfg(feature = "encoder")]
    return reader.and_then(|reader| reader.read(table, &measures, &views, &texts, &line_pages, &doc::Style::of(&sheet, table, &measures))
        .map_err(|error| eprintln!("transition parser DOC {sha256}: {error}")).ok());
    #[cfg(not(feature = "encoder"))]
    None
}
