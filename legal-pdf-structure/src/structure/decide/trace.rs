//! A document's decision traced, for fitting its weights on the gold: with LEGALPDF_DECISION_TRACE
//! naming a folder, each document's `<source sha256>.trace.json.gz` there holds each line's id and
//! page, what each stage read of it (the evidence, fences and runs, the notes, the decision), and
//! each weighed choice with the evidence it weighed (`ResidualRecord`). The parse is the same with a
//! trace or without one.

use super::*;

/// The folder traces are written to, once read from the environment.
pub(crate) fn trace_folder() -> Option<&'static std::path::Path> {
    static FOLDER: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();
    FOLDER.get_or_init(|| std::env::var_os("LEGALPDF_DECISION_TRACE").map(std::path::PathBuf::from)).as_deref()
}

/// One residual candidate as the residual weighed it: its lines (slots), its reading context and look
/// (an index among the context's looks), the indicators its own log-odds sum (`OWN`, in order), the
/// measures they were read from, the vetoes that bar it (`VETO_*`), its look's signs (`LOOK`, in
/// order) and members, and the scores and outcome.
#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct ResidualRecord {
    pub(crate) lines: Vec<u32>,
    pub(crate) context: u32,
    pub(crate) look: u32,
    pub(crate) terms: [bool; OWN.len()],
    pub(crate) alone: bool,
    pub(crate) above: f64,
    pub(crate) below: Option<f64>,
    pub(crate) words: u32,
    pub(crate) caps: bool,
    pub(crate) body: bool,
    pub(crate) larger: bool,
    pub(crate) numbered: bool,
    pub(crate) vetoes: u16,
    pub(crate) signs: [bool; LOOK.len()],
    pub(crate) members: u32,
    pub(crate) own: f64,
    pub(crate) score: f64,
    pub(crate) heading: bool,
}

/// A testimony's line, or a transcript's on a page of speakers' turns or a turn or event itself.
pub(crate) const VETO_TESTIMONY: u16 = 1;
/// A marker alone whose text goes on into the next line, or ends a sentence.
pub(crate) const VETO_RUNS_ON: u16 = 1 << 1;
/// An editorial line (`numbered_paragraphs::editorial_line`).
pub(crate) const VETO_EDITORIAL: u16 = 1 << 2;
/// Ending on a period, comma or semicolon.
pub(crate) const VETO_ENDS: u16 = 1 << 3;
/// A label (`rules::label`) or a date alone.
pub(crate) const VETO_LABEL: u16 = 1 << 4;
/// Over twenty words.
pub(crate) const VETO_WORDS: u16 = 1 << 5;
/// Tagged a heading by its source, but no numbered one, of several lines, in the body's type in
/// capitals, or not larger with more space under it than over it.
pub(crate) const VETO_TAGGED: u16 = 1 << 6;

#[derive(serde::Serialize)]
struct NotesTrace<'a> {
    note: &'a [Option<NoteKind>],
    paragraph_label: &'a [bool],
    suppressed: &'a [bool],
    running_head: &'a [bool],
    heading: &'a [bool],
}

#[derive(serde::Serialize)]
struct DecisionTrace<'a> {
    role: &'a [Role],
    rule: &'a [Rule],
    unit: &'a [Option<Opening>],
    joined: &'a [bool],
    given: &'a [u8],
    level: &'a [u8],
    anchor: &'a [u32],
}

/// A line as a decoder fitted on the trace reads it: its text; its opening marker's readings by the
/// numbering grammar (family, value and dotted prefix), read alone and, where a letter is also a
/// roman numeral, as the letter after the one before it, and a division word's number ("PART II")
/// under the word; and its reference marks' values (0 for a note symbol), raised or detached.
#[derive(serde::Serialize)]
struct LineTrace<'a> {
    text: &'a str,
    readings: Vec<(String, u32, Vec<u32>)>,
    references: Vec<u32>,
}

/// The readings of a line's opening token, and of the number after a division word.
fn readings(text: &str) -> Vec<(String, u32, Vec<u32>)> {
    let grammar = legal_pdf_core::layout_analysis();
    let mut tokens = text.split_whitespace();
    let Some(first) = tokens.next().filter(|token| token.chars().count() <= 12) else { return Vec::new() };
    let read = |token: &str, word: Option<&str>| {
        let alone = grammar.numbering_marker(token, None);
        // A roman reading of one letter read again as the letter after the letter before it.
        let letter = alone.as_ref().filter(|m| m.family.ends_with(":i") || m.family.ends_with(":I")).and_then(|m| {
            let core = token.trim_matches(|c: char| !c.is_ascii_alphabetic());
            let c = core.chars().next().filter(|_| core.chars().count() == 1)?;
            let before = u32::from(c.to_ascii_lowercase()) - u32::from('a');
            let family = format!("{}{}", &m.family[..m.family.len() - 1], if c.is_ascii_uppercase() { 'A' } else { 'a' });
            let previous = legal_structure_model::NumberingMarker { family, ordinal: before, prefix: Vec::new() };
            grammar.numbering_marker(token, Some(&previous))
        });
        alone.into_iter().chain(letter).map(|m| {
            (word.map_or(m.family.clone(), |word| format!("{}:{}", word.to_lowercase(), m.family)), m.ordinal, m.prefix)
        }).collect::<Vec<_>>()
    };
    let mut found = read(first, None);
    if first.chars().count() > 1 && first.chars().all(char::is_alphabetic) {
        if let Some(number) = tokens.next() {
            found.extend(read(number.trim_end_matches(['.', ':', '\u{2014}', '-']), Some(first)));
        }
    }
    found.dedup();
    found
}

/// The values of a line's reference marks (`reference_marks`, and its detached references).
fn reference_values(line: &Line, size: f64) -> Vec<u32> {
    let value = |text: &str| text.trim().parse().unwrap_or(0);
    reference_marks(line, size).map(|(text, _)| value(text))
        .chain(line.detached_references.iter().map(|reference| value(&reference.selected_text)))
        .collect()
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
    notes: NotesTrace<'a>,
    decision: DecisionTrace<'a>,
    residual: &'a [ResidualRecord],
}

/// Writes the document's trace into `folder`; a trace that cannot be written is reported on stderr
/// and the parse goes on.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_trace(
    folder: &std::path::Path,
    source_sha256: &str,
    pages: &[Page],
    slots: &Slots,
    evidence: &Evidence,
    fences: &Fences,
    runs: &Runs,
    notes: &Notes,
    decision: &Decision,
) {
    let all = (0..slots.len()).map(|at| Slot(at as u32));
    let trace = Trace {
        source_sha256,
        ids: all.clone().map(|slot| slots.line(pages, slot).id.as_str()).collect(),
        pages: all.clone().map(|slot| slots.page(slot) as u32).collect(),
        lines: all
            .map(|slot| {
                let line = slots.line(pages, slot);
                LineTrace {
                    text: &line.text,
                    readings: readings(&line.text),
                    references: reference_values(line, evidence.size[slot.index()]),
                }
            })
            .collect(),
        evidence,
        fences,
        runs,
        notes: NotesTrace {
            note: &notes.note,
            paragraph_label: &notes.paragraph_label,
            suppressed: &notes.suppressed,
            running_head: &notes.running_head,
            heading: &notes.heading,
        },
        decision: DecisionTrace {
            role: &decision.role,
            rule: &decision.rule,
            unit: &decision.unit,
            joined: &decision.joined,
            given: &decision.given,
            level: &decision.level,
            anchor: &decision.anchor,
        },
        residual: &decision.residual,
    };
    let path = folder.join(format!("{source_sha256}.trace.json.gz"));
    if let Err(error) = std::fs::create_dir_all(folder).map_err(|e| e.to_string()).and_then(|()| {
        legal_pdf_core::write_gzip_json(&path, &trace).map_err(|e| e.to_string())
    }) {
        eprintln!("decision trace {}: {error}", path.display());
    }
}
