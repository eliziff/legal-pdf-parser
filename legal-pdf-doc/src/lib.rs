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

// The crates this stage reads, by the names its paths use.
use legal_pdf_structure_model as model;
use legal_pdf_rules as rules;
use crate as doc;

use crate::model::machine::lines::LineTable;
use crate::model::decide::Decision;
use crate::model::fences::Fences;
use crate::model::Slots;
use legal_pdf_core::model::Page;
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::OnceLock,
};

use crate::model::machine::{machine::*};
use crate::model::regions::*;
mod stylesheet;
pub use stylesheet::*;
pub use stylesheet::ROW;
#[cfg(feature = "encoder")]
#[macro_use]
mod onnx;
mod entry;
pub use entry::Entry;
mod inputs;
pub use inputs::Inputs;
mod reader;
// The reader's geometry the grader's fit targets are made in (lib.rs doc_vocabulary).
pub use reader::{region_boxes, resets, sections, DOC_KINDS, TABLE_COLS};
mod scale;
#[cfg(feature = "encoder")]
mod trees;
pub mod tensors;

/// Tier 1, DOC's stages loaded (`Doc::load`), which a parse reads with where it is given one; none without `encoder`.
#[cfg(feature = "encoder")]
pub use reader::Reader as Doc;
#[cfg(not(feature = "encoder"))]
pub enum Doc {}
#[cfg(not(feature = "encoder"))]
impl Doc {
    pub fn load(_: &Path, _: Option<&Path>, _: bool, _: usize) -> Result<Self, String> {
        Err("this parser was built without DOC (the encoder feature)".to_owned())
    }
}

/// The region-and-opening classes (`region * 2 + opens`) DOC's line head scores.
const CLASSES: usize = 34;
/// The gold region of a line a gold reads only as body prose, a quotation or list item set in it unmarked
/// (the grader's `labels`, A2AJ's alignment).
pub const UNMARKED_BODY: i64 = 101;

/// The gold's labels by document: each piece's row, each piece's reading rank where the gold gives one, the pieces
/// in the order of the rows, which a row's pointer counts, and what the gold prints that a row cannot hold, placed on
/// the rows' lines (`Placed`).
type GoldLabels = (
    HashMap<String, HashMap<String, [i64; 8]>>,
    HashMap<String, HashMap<String, i64>>,
    HashMap<String, Vec<String>>,
    HashMap<String, Placed>,
);

/// The gold's printed markers and paired reference marks as its labels place them: each one the row of its unit's (its
/// note's) first line, the row of the line it stands on, and its first and past-last UTF-16 code unit there; in a
/// document's reading (`Measured`), the rows are the lines' places.
#[derive(Default, Clone)]
pub struct Placed {
    pub markers: Vec<[i64; 4]>,
    pub references: Vec<[i64; 4]>,
    /// Each table row's cells: the row of the line a cell's text stands on, its column, and its span there.
    pub cells: Vec<[i64; 4]>,
    /// Each label opening a speaker's turn: the row of its line, its span there, and 0.
    pub speakers: Vec<[i64; 4]>,
    /// The documents' types: the source's own (row -1), and each a unit opens under it (the row of its first line).
    pub documents: Vec<(i64, String)>,
}

/// What tier 1 is given: DOC's stages, the gold where it is measured, and what it writes.
pub struct Settings {
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
            text: var("LEGALPDF_DOC_TEXT").and_then(|dir| {
                reader::TextModel::load(&dir)
                    .map_err(|e| eprintln!("DOC text model: {e}"))
                    .ok()
            }),
            #[cfg(feature = "encoder")]
            doc: var("LEGALPDF_DOC").and_then(|dir| {
                reader::Reader::load(
                    &dir,
                    None,
                    std::env::var_os("LEGALPDF_DOC_TIMES").is_some(),
                    1,
                )
                .map_err(|e| eprintln!("DOC: {e}"))
                .ok()
            }),
        }
    }
}

/// The gold's labels, from rows of the grader's predictions format ({sha256, ids, labels}): each document's by piece
/// (`line@offset`), each piece's reading rank (the ninth column) and its pieces in row order.
fn read_gold(path: &Path) -> GoldLabels {
    let mut found = GoldLabels::default();
    let Ok(text) = std::fs::read_to_string(path) else {
        eprintln!("DOC: no gold at {}", path.display());
        return found;
    };
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let Ok(row) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let (Some(sha), Some(ids), Some(labels)) = (
            row["sha256"].as_str(),
            row["ids"].as_array(),
            row["labels"].as_array(),
        ) else {
            continue;
        };
        let (mut per_piece, mut ranks) = (HashMap::new(), HashMap::new());
        for (id, label) in ids.iter().zip(labels) {
            let piece = id.as_str().unwrap_or_default().to_owned();
            let cell = |k: usize| label.get(k).and_then(serde_json::Value::as_i64);
            if let Some(rank) = cell(8).filter(|&r| r >= 0) {
                ranks.insert(piece.clone(), rank);
            }
            per_piece.insert(
                piece,
                std::array::from_fn::<i64, 8, _>(|k| cell(k).unwrap_or(-1)),
            );
        }
        let quads = |name: &str| -> Vec<[i64; 4]> {
            row[name]
                .as_array()
                .into_iter()
                .flatten()
                .map(|quad| std::array::from_fn(|k| quad.get(k).and_then(serde_json::Value::as_i64).unwrap_or(-1)))
                .collect()
        };
        found.3.insert(
            sha.to_owned(),
            Placed {
                markers: quads("markers"),
                references: quads("references"),
                cells: quads("cells"),
                speakers: quads("speakers"),
                documents: row["documents"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|pair| Some((pair.get(0)?.as_i64()?, pair.get(1)?.as_str()?.to_owned())))
                    .collect(),
            },
        );
        found.0.insert(sha.to_owned(), per_piece);
        found.1.insert(sha.to_owned(), ranks);
        found.2.insert(
            sha.to_owned(),
            ids.iter()
                .filter_map(|id| id.as_str().map(str::to_owned))
                .collect(),
        );
    }
    found
}

/// Kept parses read by DOC together (`Reader::read_many`), each from its inputs as its entry keeps them: each one's
/// moves, none where DOC is not given, its inputs could not be read, or DOC failed it. Each document: its pages, slots,
/// fences, line table, line views, inputs and source.
#[allow(clippy::type_complexity)]
pub fn read_kept(
    docs: &[(
        &[Page],
        &Slots,
        &Fences,
        &LineTable,
        &[LineView],
        Option<&Inputs>,
        &str,
    )],
    doc: Option<&Doc>,
) -> Vec<Option<Vec<Move>>> {
    let mut out = vec![None; docs.len()];
    #[cfg(feature = "encoder")]
    if let Some(reader) = doc.or(settings().doc.as_ref()) {
        let s = settings();
        let texts: Vec<Vec<&str>> = docs
            .iter()
            .map(|&(pages, slots, fences, ..)| {
                fences
                    .order
                    .iter()
                    .map(|&slot| slots.line(pages, slot).text.as_str())
                    .collect()
            })
            .collect();
        let readable: Vec<usize> = (0..docs.len()).filter(|&at| docs[at].5.is_some()).collect();
        let wants_gold = !reader::oracle_parts().is_empty();
        let golds: Vec<Option<Vec<[i64; 8]>>> = readable
            .iter()
            .map(|&at| {
                let (pages, slots, fences, _, _, _, sha256) = docs[at];
                let gold_all = wants_gold.then_some(s.gold.as_ref()).flatten()?;
                gold_all.0.get(sha256)?;
                let ids: Vec<&str> = fences
                    .order
                    .iter()
                    .map(|&slot| slots.line(pages, slot).id.as_str())
                    .collect();
                Some(map_gold_rows(gold_all, sha256, &ids))
            })
            .collect();
        let readings: Vec<reader::Reading> = readable
            .iter()
            .zip(&golds)
            .map(|(&at, gold)| {
                let (_, _, _, table, views, inputs, _) = docs[at];
                reader::Reading {
                    inputs: inputs.expect("readable"),
                    table,
                    views,
                    texts: &texts[at],
                    gold: gold.as_deref(),
                }
            })
            .collect();
        for (&at, moves) in readable.iter().zip(reader.read_many(&readings)) {
            out[at] = moves
                .map_err(|error| eprintln!("DOC {}: {error}", docs[at].6))
                .ok();
        }
    }
    #[cfg(not(feature = "encoder"))]
    let _ = doc;
    out
}

fn gold_line_map<'a>(
    gold_all: &'a GoldLabels,
    sha256: &str,
    ids: &[&'a str],
) -> Vec<i64> {
    let place: HashMap<&str, i64> = ids
        .iter()
        .enumerate()
        .map(|(k, &id)| (id, k as i64))
        .collect();
    let pieces = gold_all.2.get(sha256);
    pieces
        .map(|pieces| {
            pieces
                .iter()
                .map(|piece| {
                    place
                        .get(piece.split('@').next().unwrap_or(piece))
                        .copied()
                        .unwrap_or(-1)
                })
                .collect()
        })
        .unwrap_or_default()
}

fn map_gold_rows(gold_all: &GoldLabels, sha256: &str, ids: &[&str]) -> Vec<[i64; 8]> {
    let gold = gold_all.0.get(sha256);
    let piece_line = gold_line_map(gold_all, sha256, ids);
    ids.iter()
        .map(|id| {
            let mut row = gold
                .and_then(|g| g.get(&format!("{id}@0")))
                .copied()
                .unwrap_or([-1; 8]);
            if row[6] >= 0 {
                row[6] = piece_line.get(row[6] as usize).copied().unwrap_or(-1);
            }
            row
        })
        .collect()
}

/// What DOC reads of a document and what its dump entry records of it, made once from the parse: its lines' texts and
/// ids, its inputs (`Inputs`), and the gold's rows and reading ranks where the gold is given here.
struct Measured<'a> {
    texts: Vec<&'a str>,
    ids: Vec<&'a str>,
    inputs: Inputs,
    gold: Option<&'a HashMap<String, [i64; 8]>>,
    gold_rows: Vec<[i64; 8]>,
    ranks: Vec<i64>,
    /// The gold's markers and reference marks, each on its line's place.
    placed: Placed,
}

/// DOC's moves over tier 0's reading of a document (its notes and decision, `table` the line table both tiers read),
/// where its reader (`doc`, else the harnesses') or its oracle is given, each move's `reading` naming one of the line's
/// readings in `table`; and its dump entry, where `dump` names the dump (`parse` the parse as tier 0 left it, as the
/// entry keeps it). None where neither reads: the output is tier 0's.
#[allow(clippy::too_many_arguments)]
pub fn over_tier0(
    pages: &[Page],
    slots: &Slots,
    fences: &Fences,
    sha256: &str,
    decision: &Decision,
    table: &LineTable,
    views: &[LineView],
    doc: Option<&Doc>,
    dump: Option<(crate::Keep<'_>, &[u8])>,
    measure_lines: &dyn Fn() -> crate::model::machine::lines::Measures,
) -> Option<Vec<Move>> {
    let s = settings();
    let oracle = s.doc_oracle && s.gold.as_ref().is_some_and(|g| g.0.contains_key(sha256));
    #[cfg(feature = "encoder")]
    let reader = doc.or(s.doc.as_ref());
    #[cfg(not(feature = "encoder"))]
    let reader = doc;
    if reader.is_none() && !oracle && dump.is_none_or(|(kept, _)| !kept.inputs) {
        // (A parse kept for itself alone: DOC's inputs, its measuring's whole cost, are not made.)
        if let Some((kept, parse)) = dump {
            keep_parse(kept, sha256, parse)
                .unwrap_or_else(|error| eprintln!("DOC dump {sha256}: {error}"));
        }
        return None;
    }
    let measured = measure(
        pages,
        slots,
        fences,
        sha256,
        decision,
        table,
        reader,
        dump.is_some(),
        measure_lines,
    )?;
    if let Some((kept, parse)) = dump {
        keep(kept, sha256, &measured, table, views, parse)
            .unwrap_or_else(|error| eprintln!("DOC dump {sha256}: {error}"));
    }
    // Detect-Order-Construct's decoding from the gold's answers, else its reading where its stages are given.
    if oracle {
        return Some(reader::oracle(
            &measured.gold_rows,
            views,
            &measured.texts,
            &measured.inputs.pages,
            &measured.ranks,
            &measured.placed,
        ));
    }
    #[cfg(feature = "encoder")]
    return reader.and_then(|reader| {
        reader
            .read(
                &measured.inputs,
                table,
                views,
                &measured.texts,
                measured
                    .gold
                    .is_some()
                    .then_some(&measured.gold_rows[..])
                    .filter(|_| !reader::oracle_parts().is_empty()),
            )
            .map_err(|error| eprintln!("DOC {sha256}: {error}"))
            .ok()
    });
    #[cfg(not(feature = "encoder"))]
    None
}

/// A document measured as DOC reads it (`Measured`): its lines' tokens by the reader's text model, else the one the dump
/// is written with.
#[allow(clippy::too_many_arguments)]
fn measure<'a>(
    pages: &'a [Page],
    slots: &Slots,
    fences: &Fences,
    sha256: &str,
    decision: &Decision,
    table: &LineTable,
    reader: Option<&Doc>,
    keep_dump: bool,
    measure_lines: &dyn Fn() -> crate::model::machine::lines::Measures,
) -> Option<Measured<'a>> {
    let s = settings();
    let texts: Vec<&str> = fences
        .order
        .iter()
        .map(|&slot| slots.line(pages, slot).text.as_str())
        .collect();
    #[cfg(feature = "encoder")]
    let model = reader.map(|reader| reader.text()).or(s.text.as_ref());
    #[cfg(feature = "encoder")]
    let tokenize = model.map(|model| move |texts: &[&str]| model.tokens(texts));
    #[cfg(feature = "encoder")]
    let tokenize = tokenize
        .as_ref()
        .map(|f| f as &dyn Fn(&[&str]) -> Result<(Vec<u32>, Vec<usize>), String>);
    #[cfg(not(feature = "encoder"))]
    let tokenize = {
        let _ = reader;
        None
    };
    let inputs = Inputs::of(slots, fences, decision, table, &texts, tokenize, measure_lines)
        .map_err(|error| eprintln!("DOC {sha256}: {error}"))
        .ok()?;
    let ids: Vec<&str> = if keep_dump || s.gold.is_some() {
        fences
            .order
            .iter()
            .map(|&slot| slots.line(pages, slot).id.as_str())
            .collect()
    } else {
        Vec::new()
    };
    let gold = s.gold.as_ref().and_then(|g| g.0.get(sha256));
    let (ranks, gold_rows, placed) = if s.gold.is_some() {
        // Each line's reading rank in the gold, -1 for none.
        let ranks: Vec<i64> = {
            let ranks = s.gold.as_ref().and_then(|g| g.1.get(sha256));
            ids.iter()
                .map(|id| {
                    ranks
                        .and_then(|r| r.get(&format!("{id}@0")))
                        .copied()
                        .unwrap_or(-1)
                })
                .collect()
        };
        let piece_line = gold_line_map(s.gold.as_ref().expect("gold checked"), sha256, &ids);
        let line_of = |row: i64| piece_line.get(row as usize).copied().unwrap_or(-1);
        let gold_rows = map_gold_rows(s.gold.as_ref().expect("gold checked"), sha256, &ids);
        // The gold's markers and reference marks: their rows the places of those rows' lines (one a line of this parse
        // does not hold is left out).
        let on_lines = |quads: &[[i64; 4]]| -> Vec<[i64; 4]> {
            quads
                .iter()
                .map(|quad| [line_of(quad[0]), line_of(quad[1]), quad[2], quad[3]])
                .filter(|quad| quad[0] >= 0 && quad[1] >= 0)
                .collect()
        };
        let placed = s
            .gold
            .as_ref()
            .and_then(|g| g.3.get(sha256))
            .map(|placed| Placed {
                markers: on_lines(&placed.markers),
                references: on_lines(&placed.references),
                // (A cell's second number is its column, not a row.)
                cells: placed
                    .cells
                    .iter()
                    .map(|quad| [line_of(quad[0]), quad[1], quad[2], quad[3]])
                    .filter(|quad| quad[0] >= 0)
                    .collect(),
                speakers: placed
                    .speakers
                    .iter()
                    .map(|quad| [line_of(quad[0]), quad[1], quad[2], 0])
                    .filter(|quad| quad[0] >= 0)
                    .collect(),
                // (The source's own type keeps its row, -1; a start on a line this parse does not hold is left out.)
                documents: placed
                    .documents
                    .iter()
                    .filter_map(|(row, kind)| {
                        let at = if *row < 0 {
                            -1
                        } else {
                            Some(line_of(*row)).filter(|at| *at >= 0)?
                        };
                        Some((at, kind.clone()))
                    })
                    .collect(),
            })
            .unwrap_or_default();
        (ranks, gold_rows, placed)
    } else {
        (Vec::new(), Vec::new(), Placed::default())
    };
    Some(Measured {
        texts,
        ids,
        inputs,
        gold,
        gold_rows,
        ranks,
        placed,
    })
}

/// A document's parse kept in its dump entry under its key, the entry's other parts as they were.
fn keep_parse(kept: crate::Keep<'_>, sha256: &str, parse: &[u8]) -> std::io::Result<()> {
    let mut entry = Entry::read(kept.folder, sha256).unwrap_or_default();
    entry.set_bytes("parse", parse);
    entry.record["read_key"] = serde_json::Value::from(kept.key);
    entry.write(kept.folder, sha256)
}

/// A document's dump entry written (`Entry`): its inputs, its record (the lines and their places, reference marks and
/// note numbers, what the grader reads back and makes the fit's targets of), and the parse as tier 0 left it.
/// The entry is keyed by what the parse read, not by the build that read it: a document this build reads as the last one
/// did (the same arrays, lines and texts) keeps its targets and everything the fit made of them. The parse is kept under
/// `kept.key` (`read_key`), what its reader reads it back by.
fn keep(
    kept: crate::Keep<'_>,
    sha256: &str,
    measured: &Measured,
    table: &LineTable,
    views: &[LineView],
    parse: &[u8],
) -> std::io::Result<()> {
    let folder = kept.folder;
    use std::hash::{Hash, Hasher};
    let Measured { texts, ids, inputs, ranks, .. } = measured;
    let note_numbers: Vec<i64> = views
        .iter()
        .zip(texts)
        .map(|(view, text)| reader::note_number(view, text).map_or(-1, i64::from))
        .collect();
    // What the fit reads, as arrays, each once and in the form the reader reads them. What DOC reads of the lines around
    // a line, of a region's lines, of every two lines or regions and of a tree's nodes it makes from these (its context
    // graph, the fit's step), so nothing here repeats a line.
    let arrays = inputs.tensors();
    let mut record = serde_json::json!({
        "sha256": sha256, "ids": ids, "lines": ids.len(), "pages": inputs.pages, "references": &table.references, "ranks": ranks,
        "note_numbers": note_numbers,
    });
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    arrays.iter().for_each(|t| {
        t.name.hash(&mut hasher);
        t.bytes.hash(&mut hasher);
    });
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
    // A parse read anew keeps no targets of the last one: the grader writes them again from the gold (its fill_dump).
    entry.set_part("targets", Vec::new());
    entry.set_part("inputs", arrays);
    entry.set_bytes("parse", parse);
    record["parse_hash"] = serde_json::Value::from(read);
    record["read_key"] = serde_json::Value::from(kept.key);
    entry.record = record;
    entry.write(folder, sha256)
}

/// Where a parse is kept (`Entry`): its folder, and the key it is kept under (what made it), which its reader reads it
/// back by; and whether the entry keeps DOC's inputs too, which DOC reading the parse later and its fit read (`inputs`).
#[derive(Clone, Copy)]
pub struct Keep<'a> {
    pub folder: &'a Path,
    pub key: &'a str,
    pub inputs: bool,
}
