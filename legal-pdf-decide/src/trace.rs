//! A document's decision traced, for fitting its weights on the gold: given a folder, each document's `<source sha256>.trace.json.gz` there holds each line's id and
//! page, what each stage read of it (the evidence, fences and runs, the notes, the decision), and
//! each weighed choice with the evidence it weighed (`ResidualRecord`). The parse is the same with a
//! trace or without one.

use crate::model::decide::ResidualRecord;
use crate::model::decide::Notes;
use crate::model::decide::Decision;
use crate::model::fences::Fences;
use crate::model::{Evidence, MarkerReading, Runs, Slot, Slots};
use legal_pdf_core::model::Page;


/// A testimony's line, or a transcript's on a page of speakers' turns or a turn or event itself.
pub const VETO_TESTIMONY: u16 = 1;
/// A marker alone whose text goes on into the next line, or ends a sentence.
pub const VETO_RUNS_ON: u16 = 1 << 1;
/// An editorial line (`numbered_paragraphs::editorial_line`).
pub const VETO_EDITORIAL: u16 = 1 << 2;
/// Ending on a period, comma or semicolon.
pub const VETO_ENDS: u16 = 1 << 3;
/// A label (`rules::label`) or a date alone.
pub const VETO_LABEL: u16 = 1 << 4;
/// Over twenty words.
pub const VETO_WORDS: u16 = 1 << 5;
/// Tagged a heading by its source, but no numbered one, of several lines, in the body's type in
/// capitals, or not larger with more space under it than over it.
pub const VETO_TAGGED: u16 = 1 << 6;
/// (TEMP lever "residual-fragment".) No heading's words: a lone enumerator ("4°", "(7.)") or a scan layer's glyphs
/// ("©", "eee", "otc").
pub const VETO_FRAGMENT: u16 = 1 << 7;

/// A line as a decoder fitted on the trace reads it: its text; its opening marker's readings as the parse reads them
/// (family, value and dotted prefix: `LineTable::readings`); and its reference marks' values (0 for a note symbol:
/// `LineTable::references`).
#[derive(serde::Serialize)]
struct LineTrace<'a> {
    text: &'a str,
    readings: Vec<(String, u32, Vec<u32>)>,
    references: Vec<u32>,
}

#[derive(serde::Serialize)]
struct Trace<'a> {
    source_sha256: &'a str,
    ids: Vec<&'a str>,
    pages: Vec<u32>,
    lines: Vec<LineTrace<'a>>,
    evidence: &'a Evidence,
    fences: &'a Fences,
    runs: &'a Runs,
    notes: &'a Notes,
    decision: &'a Decision,
    residual: &'a [ResidualRecord],
}

/// Writes the document's trace into `folder`, each line's marker readings and reference marks those the parse read
/// (`readings`, `references`: by place in reading order); a trace that cannot be written is reported on stderr and the
/// parse goes on.
#[allow(clippy::too_many_arguments)]
pub fn write_trace(
    folder: &std::path::Path,
    source_sha256: &str,
    pages: &[Page],
    slots: &Slots,
    evidence: &Evidence,
    fences: &Fences,
    runs: &Runs,
    notes: &Notes,
    decision: &Decision,
    readings: &[Vec<MarkerReading>],
    references: &[Vec<(u32, u32, u32)>],
) {
    let all = (0..slots.len()).map(|at| Slot(at as u32));
    let trace = Trace {
        source_sha256,
        ids: all
            .clone()
            .map(|slot| slots.line(pages, slot).id.as_str())
            .collect(),
        pages: all.clone().map(|slot| slots.page(slot) as u32).collect(),
        lines: all
            .map(|slot| {
                let place = fences.place[slot.index()] as usize;
                LineTrace {
                    text: &slots.line(pages, slot).text,
                    readings: readings[place]
                        .iter()
                        .map(|r| (r.family.clone(), r.value, r.prefix.clone()))
                        .collect(),
                    references: references[place].iter().map(|&(value, ..)| value).collect(),
                }
            })
            .collect(),
        evidence,
        fences,
        runs,
        notes,
        decision,
        residual: &decision.residual,
    };
    let path = folder.join(format!("{source_sha256}.trace.json.gz"));
    if let Err(error) = std::fs::create_dir_all(folder)
        .map_err(|e| e.to_string())
        .and_then(|()| legal_pdf_core::write_gzip_json(&path, &trace).map_err(|e| e.to_string()))
    {
        eprintln!("decision trace {}: {error}", path.display());
    }
}
