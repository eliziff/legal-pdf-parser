//! The structure stage: a PDF's extracted pages and lines in, its paragraphs, notes, headings and structure graph out.
//! Each stage is a module over the contract (`model`); `derive` calls them in order:
//!
//! 0. The pages normalized (`evidence::normalize`): a transcript's line numbers parted from the lines
//!    they were read into, a table's rulings cleared from under its words, each line given its slot.
//! 1. Measured (`Evidence`): each page's words, rows and note separator; each line's type, its source's tag, its marker,
//!    note label, reference tokens, words and citations; the document's bookmarks and whether it is an instrument.
//! 2. Fenced and ordered (`Fences`): a transcript's line numbers, the pages' numbers and the lines printing them (the
//!    page map's, `legal_pdf_pages::page_map`), furniture (`furniture`), each
//!    page's columns, contents lists (`headings`), tables, the note band under a separator, the body's type,
//!    the constituent documents (`constituents`), the reading order, quotations and reading contexts, and each
//!    line's continuation of the one before.
//! 3. Numbered (`Runs`): each reading context's numbering runs, read and resolved by the engine.
//! 4. Notes (`Notes`): each document's note band and the lanes that take more lines for notes (`notes`).
//! 5. Decided (`tier0`): every rule heard on every line (`Decision::hear`), then one walk in reading order deciding each
//!    line, its block and the unit it opens or goes on in (`settle`), then the title and the headings' levels
//!    (`Decision::finish`). Paragraph numbers are fixed input. Tier 1 (DOC, with `encoder`) reads over tier 0's
//!    reading (`doc`), its moves in tier 0's place.
//! 6. Written once (`write_lines`): each line's place in its page's reading order; and read by slot, each line's zone.
//! 7. Derived from the decision: notes paired (the engine's pairer, over `note_lines`), laid over the units the moves
//!    make (a line a note holds is the note's; a note band's line none holds is the body's), propositions and
//!    cross-references (`note_context`), the units emitted as the structure graph.
//!
//! Measure a change with the grader's digest (`structure-gold digest`): every parse hashed, two builds compared.

use crate::graph::write::{folio_field, write_structure};
use legal_pdf_decide as decide;
use legal_pdf_doc as doc;
use legal_pdf_evidence as evidence;
use legal_pdf_fences as fences;
use legal_pdf_furniture as furniture;
use legal_pdf_graph as graph;
use legal_pdf_machine as machine;
// The stage's contract, a crate of its own: an edit to a stage recompiles no other crate that reads it.
use legal_pdf_structure_model as model;
use legal_pdf_notes as notes;
mod rendered;
use legal_pdf_runs as runs;
// The line rules every stage reads, a crate of their own (legal-pdf-rules).
use legal_pdf_rules::text;

use decide::{note_lines, write_lines, write_trace, zones};
use evidence::normalize;
use crate::model::fences::Fences;
use model::Evidence;
use model::{Runs, Slots};
use notes::note_context::unmatched_reference_diagnostics;
use text::text_index::PdfTextIndex;

pub use doc::{Doc, Entry, Keep};
use crate::model::machine::lines::LineTable;
use crate::model::machine::machine::LineView;
use std::path::Path;

/// What a kept parse answers to: a change to what the structure stage reads of a document's pages (its output) bumps it,
/// and every parse kept under the last one is read again. Never a fingerprint of the source: a change that reads the same
/// (a speed fix, a timer, a rename) keeps every kept parse.
pub const PARSE_VERSION: &str = "6";

/// The vocabulary DOC's dump is written in (its regions, its readers' roles, its rows), which the grader's fit targets
/// are made in.
#[doc(hidden)]
pub mod doc_vocabulary {
    pub use crate::doc::{region_boxes, resets, sections, DOC_KINDS, ROW, TABLE_COLS, UNMARKED_BODY};
    pub use crate::model::machine::machine::{RANKED_BODY, REFERRING};
    pub use crate::model::regions::{stream_of, HEADING, NOTE, NOTE_QUOTE};
}
/// The dump's arrays as safetensors (the grader reads the parser's and writes the targets with it).
#[doc(hidden)]
pub use doc::tensors;
pub use furniture::page_evidence;
/// The regions a line may be in and the stream each runs in: the one vocabulary DOC's labels, its dump and the grader's
/// label export read.
pub use model::regions::{stream_of, REGIONS};

use crate::model::machine::machine::Move;
use crate::model::regions::{role_of_region, PAGE_LABEL};
use crate::model::Decided;
use legal_pdf_core::model::{Diagnostic, LineZone, Page, PdfOutlineEntry};
use legal_pdf_core::{Error, Result};
use legal_structure_model::DocumentStructure;
use legal_structure_model::StructureNode;
use legal_structure_model::ScalarText;


pub use crate::model::StructureIdentity;

pub struct StructureOutput {
    /// The authority units, read off the rendered text (`emit`).
    pub authority_text_units: Vec<legal_pdf_core::AuthorityTextUnit>,
    pub diagnostics: Vec<Diagnostic>,
    pub structure_graph: DocumentStructure,
}

/// A line of the note band that no note holds (the pairer made no note of its label, or bounded the note's body short of
/// it) is the body's, and joins its paragraph rather than standing alone: on the gold, 6,825 such lines were no notes,
/// 2,146 were.
fn give_back(zones: &mut [LineZone], slots: &Slots, notes: &[StructureNode]) {
    let mut held = vec![false; zones.len()];
    for slot in notes
        .iter()
        .flat_map(|note| &note.line_ids)
        .filter_map(|id| slots.of_id(id))
    {
        held[slot.index()] = true;
    }
    for (zone, held) in zones.iter_mut().zip(held) {
        if matches!(zone, LineZone::Note(_)) && !held {
            *zone = LineZone::Body;
        }
    }
}
/// The document's structure, its moves DOC's where `doc` is given (tier 1), else tier 0's. With `trace`, the decision is also
/// written to that folder (`decide::trace`); with `dump`, the document's dump entry there (`Entry`: what DOC reads of it
/// and the parse as tier 0 left it, which `resume_many` reads back); the parse is the same.
#[allow(clippy::too_many_arguments)]
pub fn derive(
    pages: &mut [Page],
    separators: &[Option<f64>],
    outline: &[PdfOutlineEntry],
    identity: StructureIdentity,
    trace: Option<&std::path::Path>,
    doc: Option<&Doc>,
    dump: Option<Keep<'_>>,
) -> Result<StructureOutput> {
    let parsed = parse(pages, separators, outline, &identity)?;
    legal_pdf_core::profile::measure("derive.finish", || finish(
        pages,
        parsed,
        &identity,
        trace,
        doc,
        Reading::Here { dump },
    ))
}

/// Where `finish` has DOC's moves from: DOC reads the document there (its entry written to `dump` where given), or they
/// were read already (`Given`, with other documents' by `resume_many`; none: tier 0's).
enum Reading<'a> {
    Here { dump: Option<Keep<'a>> },
    Given(Option<Vec<Move>>),
}

/// What a parse holds where DOC reads: all tier 0 derived of a document (its lines' slots, evidence, fences, numbering
/// runs, notes, decision, blocks and line table, and the warnings so far). Kept with the pages as `parse` left them, it
/// is finished with any DOC (`finish`), the document never read again.
#[derive(serde::Serialize, serde::Deserialize)]
struct Parsed {
    slots: Slots,
    evidence: Evidence,
    fences: Fences,
    runs: Runs,
    notes: crate::model::decide::Notes,
    decision: crate::model::decide::Decision,
    blocks: Vec<usize>,
    table: LineTable,
    diagnostics: Vec<Diagnostic>,
    /// Each line whose marker the repair rewrote (its place among the pages' lines), with its text and size: given back
    /// where the parse is resumed over the extraction's pages (`resume_many`).
    repaired: Vec<(usize, String, f64)>,
}

/// Tier 0's parse of the pages, to where DOC reads: the pages normalized (their line numbers parted, their markers
/// repaired), measured, fenced, their runs read, and tier 0's decision made.
fn parse(
    pages: &mut [Page],
    separators: &[Option<f64>],
    outline: &[PdfOutlineEntry],
    identity: &StructureIdentity,
) -> Result<Parsed> {
    use legal_pdf_core::profile::measure;
    if separators.len() != pages.len() {
        return Err(Error::Message(
            "common input must contain one separator value per page".to_owned(),
        ));
    }
    // The document's type, where known (the caller's profile): an instrument numbers its provisions.
    let instrument = identity.profile == Some(legal_structure_model::DetectionProfile::Instrument);
    legal_pdf_core::profile::document(pages.len(), pages.iter().map(|page| page.lines.len()).sum());
    let (slots, normalized) = measure("derive.normalize", || normalize(pages));
    // Each page's measures and columns, read once for every stage that reads the pages before they are written.
    let geometry = model::Geometry::of(pages);
    let mut evidence = measure("derive.measure", || {
        crate::evidence::measure(pages, &slots, normalized, separators, outline, instrument)
    });
    let mut fences = measure("derive.fences", || {
        fences::find::read(pages, &slots, &evidence, outline, &identity.page_labels)
    });
    let runs = measure("derive.runs", || {
        let repaired = measure("runs.repair", || crate::runs::repair_markers(pages, &fences));
        let pages: &[Page] = pages;
        legal_pdf_core::profile::tally("runs.reread", || {
            crate::evidence::reread(&mut evidence, pages, &slots, &repaired)
        });
        legal_pdf_core::profile::tally("runs.read_numbers", || {
            fences::find::read_numbers(&mut fences, pages, &slots, &evidence, !repaired.is_empty(), &legal_structure::case_outline::judgment_paragraphs)
        });
        let runs = crate::runs::read(pages, &slots, &mut evidence, &fences, &geometry, repaired)?;
        if let Some(spine) = crate::runs::spine(pages, &slots, &fences, &runs) {
            fences::find::confirm_section_heads(&mut fences, pages, &slots, &evidence, |page, number| spine[page].contains(&number));
        }
        Ok::<_, Error>(runs)
    })?;
    measure("derive.decide", || {
        let diagnostics = std::mem::take(&mut fences.diagnostics);
        // One line table for both tiers: tier 0 reads it, and the walk measures it for DOC.
        let table = machine::find::plain(pages, &slots, &evidence, &fences, &runs.before);
        let notes = notes::find::read(pages, &slots, &evidence, &fences, &runs);
        // The fences' roles are the decision's to fill from here.
        let fenced = (
            std::mem::take(&mut fences.role),
            std::mem::take(&mut fences.rule),
        );
        let views = LineView::all(&evidence, &fences, &table);
        let (decision, blocks) = decide::tier0::decide(
            pages, &slots, &evidence, &fences, &runs, &geometry, outline, &notes, &views, fenced,
            false,
        );
        drop(views);
        let repaired = runs
            .before
            .iter()
            .map(|&(slot, _)| {
                let line = slots.line(pages, slot);
                (slot.index(), line.text.clone(), line.size)
            })
            .collect();
        Ok(Parsed {
            slots,
            evidence,
            fences,
            runs,
            notes,
            decision,
            blocks,
            table,
            diagnostics,
            repaired,
        })
    })
}

/// Parses kept in dump entries (`Entry`'s `parse`: tier 0's derivations, each with its source and its extraction's
/// pages and outline) finished together: DOC read over all of them at once (`doc::read_kept`, each from its entry's
/// inputs), then each one's rest of the parse (`finish`). Each source's structure; a failure where its entry keeps no
/// parse it can read.
#[allow(clippy::type_complexity)]
pub fn resume_many(
    entries: Vec<(String, Entry, Vec<Page>, Vec<PdfOutlineEntry>)>,
    doc: Option<&Doc>,
) -> Vec<Result<StructureOutput>> {
    #[allow(clippy::type_complexity)]
    let kept: Vec<
        Result<(
            String,
            Vec<Page>,
            Vec<PdfOutlineEntry>,
            Parsed,
            Option<doc::Inputs>,
        )>,
    > = entries
        .into_iter()
        .map(|(sha256, entry, mut pages, outline)| {
            let bytes = entry
                .bytes("parse")
                .ok_or_else(|| Error::Message(format!("{sha256}: its entry keeps no parse")))?;
            let parsed: Parsed = rmp_serde::from_slice(bytes)
                .map_err(|error| Error::Message(format!("{sha256}: {error}")))?;
            // The extraction's pages as the parse left them: normalized again, their repaired markers given back.
            normalize(&mut pages);
            let mut lines: Vec<&mut legal_pdf_core::model::Line> = pages
                .iter_mut()
                .flat_map(|page| page.lines.iter_mut())
                .collect();
            for (at, text, size) in &parsed.repaired {
                let line = lines.get_mut(*at).ok_or_else(|| {
                    Error::Message(format!(
                        "{sha256}: its kept parse names a line its pages lack"
                    ))
                })?;
                line.text.clone_from(text);
                line.size = *size;
            }
            let inputs = doc::Inputs::read(&entry.part("inputs"))
                .map_err(|error| eprintln!("DOC {sha256}: {error}"))
                .ok();
            Ok((sha256, pages, outline, parsed, inputs))
        })
        .collect();
    let mut moves = {
        let present: Vec<&(
            String,
            Vec<Page>,
            Vec<PdfOutlineEntry>,
            Parsed,
            Option<doc::Inputs>,
        )> = kept.iter().filter_map(|kept| kept.as_ref().ok()).collect();
        let views: Vec<Vec<LineView>> = present
            .iter()
            .map(|(_, _, _, parsed, _)| {
                LineView::all(&parsed.evidence, &parsed.fences, &parsed.table)
            })
            .collect();
        let docs: Vec<(
            &[Page],
            &Slots,
            &Fences,
            &LineTable,
            &[LineView],
            Option<&doc::Inputs>,
            &str,
        )> = present
            .iter()
            .zip(&views)
            .map(|((sha256, pages, _, parsed, inputs), views)| {
                (
                    &pages[..],
                    &parsed.slots,
                    &parsed.fences,
                    &parsed.table,
                    &views[..],
                    inputs.as_ref(),
                    sha256.as_str(),
                )
            })
            .collect();
        doc::read_kept(&docs, doc).into_iter()
    };
    kept.into_iter()
        .map(|kept| {
            let (sha256, mut pages, _, parsed, _) = kept?;
            let identity = StructureIdentity {
                source_sha256: sha256,
                profile: None,
                page_labels: Vec::new(),
            };
            // DOC's moves where it read the document, else tier 0's: a kept parse is never measured again.
            let given = moves.next().flatten();
            finish(
                &mut pages,
                parsed,
                &identity,
                None,
                doc,
                Reading::Given(given),
            )
        })
        .collect()
}

/// The parse from where DOC reads: DOC's moves over tier 0's (`doc::over_tier0`), the lines written, the notes paired,
/// and the units the moves make, the pairer's notes laid over them, written as the structure graph the consumers read.
/// `pages` as `parse` left them; `reading`, where DOC's moves come from (`Reading`).
#[allow(clippy::too_many_arguments)]
fn finish(
    pages: &mut [Page],
    parsed: Parsed,
    identity: &StructureIdentity,
    trace: Option<&Path>,
    doc: Option<&Doc>,
    reading: Reading,
) -> Result<StructureOutput> {
    // The parse as it stands, kept in its dump entry where DOC reads it here and a dump is named.
    let dump = match &reading {
        Reading::Here { dump: Some(keep) } => {
            legal_pdf_core::profile::tally("finish.pack", || rmp_serde::to_vec_named(&parsed))
                .map_err(|error| eprintln!("DOC dump {}: {error}", identity.source_sha256))
                .ok()
                .map(|parse| (*keep, parse))
        }
        _ => None,
    };
    let Parsed {
        mut slots,
        evidence,
        fences,
        runs,
        mut notes,
        mut decision,
        blocks,
        table,
        mut diagnostics,
        repaired: _,
    } = parsed;
    let (slots, evidence, runs) = (&mut slots, &evidence, &runs);
    // Each page's measures and columns as the parse read them: a page's boxes and type, which no repair changes.
    let geometry =
        &legal_pdf_core::profile::tally("finish.geometry", || model::Geometry::of(pages));
    let views =
        legal_pdf_core::profile::tally("finish.views", || LineView::all(evidence, &fences, &table));
    // DOC's reading where it is given, read off the pages as they came, before this reading writes and orders them.
    let doc = match reading {
        Reading::Given(moves) => moves,
        Reading::Here { .. } => legal_pdf_core::profile::tally("finish.over_tier0", || {
            doc::over_tier0(
                pages,
                slots,
                &fences,
                &identity.source_sha256,
                &decision,
                &table,
                &views,
                doc,
                dump.as_ref().map(|(keep, parse)| (*keep, &parse[..])),
                &|| machine::find::measure(&table, pages, slots, evidence, &fences, geometry, &notes, &decision),
            )
        }),
    };
    // A reader's moves (tier 1's, or its oracle's) are the parse's record of every line, a note's too: given the right
    // moves the parse is right, so nothing tier 0 read stands over them (its notes, its judgment paragraphs, its
    // quotations).
    let given = doc.is_some();
    let mut folio_edits: Vec<(usize, Option<(String, Option<crate::model::Slot>)>)> = Vec::new();
    for (k, (record, read)) in decision
        .moves
        .iter_mut()
        .zip(doc.iter().flatten())
        .enumerate()
    {
        let slot = fences.order[k];
        let page_slot = slots.page(slot);
        if record.role == crate::model::Role::PageLabel && read.region != PAGE_LABEL {
            if fences.folios[page_slot]
                .as_ref()
                .is_some_and(|(_, s)| *s == Some(slot))
            {
                folio_edits.push((page_slot, None));
            }
        } else if read.region == PAGE_LABEL && fences.folios[page_slot].is_none() {
            let text = slots.line(pages, slot).text.trim();
            let label = folio_field(text)
                .and_then(|((s, e), _)| {
                    ScalarText::new(text)
                        .slice_utf16(s..e)
                        .map(|t| t.trim().to_owned())
                })
                .unwrap_or_else(|| text.trim_matches(|c: char| !c.is_alphanumeric()).to_owned());
            if !label.is_empty()
                && label.chars().count() <= 12
                && label.chars().any(|c| c.is_ascii_digit())
            {
                folio_edits.push((page_slot, Some((label, Some(slot)))));
            }
        }
        *record = Move {
            role: role_of_region(read.region, record.role),
            rule: record.rule,
            joined: !read.opens && read.resumes.is_some(),
            unit: record.unit,
            member: record.member.take(),
            within: std::mem::take(&mut record.within),
            ..read.clone()
        };
    }
    diagnostics.append(&mut notes.diagnostics);
    if let Some(folder) = trace {
        write_trace(
            folder,
            &identity.source_sha256,
            pages,
            slots,
            evidence,
            &fences,
            runs,
            &notes,
            &decision,
            &table.readings,
            &table.references,
        );
    }
    // The order the lines are written in, each written line's place in the parser's order: the parser's own, or each
    // page's lines in a reader's where it ranks them (`Move::rank`).
    let written: Vec<usize> = {
        let mut places: Vec<usize> = (0..fences.order.len()).collect();
        for page_slot in 0..pages.len() {
            let range = slots.on(page_slot);
            places[range.start as usize..range.end as usize].sort_by_key(|&k| {
                decision.moves[k]
                    .rank
                    .map_or((1, k as u32), |rank| (0, rank))
            });
        }
        places
    };
    let order: Vec<crate::model::Slot> = written.iter().map(|&k| fences.order[k]).collect();
    legal_pdf_core::profile::tally("finish.write_lines", || {
        write_lines(pages, slots, &fences, &decision, &order);
        for (page_slot, folio) in folio_edits {
            let range = slots.on(page_slot);
            pages[page_slot].printed_label = folio.as_ref().map(|(label, _)| label.clone());
            pages[page_slot].printed_label_line_id = folio.and_then(|(_, slot)| slot).map(|slot| {
                pages[page_slot].lines[(slot.0 - range.start) as usize]
                    .id
                    .clone()
            });
        }
    });
    let mut zones = legal_pdf_core::profile::tally("finish.zones", || {
        zones(slots, evidence, &fences, &notes, &decision)
    });
    // The lines are read in reading order from here on, the pairer's and the output's: each page's put in it once.
    legal_pdf_core::profile::tally("finish.read_in_order", || {
        slots.read_in_order(pages, &order)
    });
    let slots: &Slots = slots;
    let decided =
        legal_pdf_core::profile::tally("finish.decided", || Decided::of(pages, slots, &fences));
    let pages: &[Page] = pages;
    // A line joins the one before it where evidence reads that line's marker standing beside it (`Evidence::beside`).
    let index = legal_pdf_core::profile::tally("finish.index", || {
        PdfTextIndex::from_pages(pages, |before, line| {
            slots
                .of_id(&before.id)
                .and_then(|at| evidence.beside[at.index()])
                .is_some_and(|beside| slots.of_id(&line.id) == Some(beside))
        })
    });
    // The notes: a reader's are its moves' own (each the unit its move opens, answering the marks it names); tier 0's
    // are the pairer's over its note lines.
    let notes = if given {
        graph::write::moved_notes(pages, slots, evidence, &fences, &decision.moves, &index)
    } else {
        let citations = legal_pdf_core::profile::tally("finish.citations", || {
            graph::citation_spans(pages, slots, evidence)
        });
        let lines = legal_pdf_core::profile::tally("finish.note_lines", || {
            note_lines(
                pages, slots, &fences, &notes, &decision, &zones, &blocks, &citations,
            )
        });
        let pairing = legal_pdf_core::profile::tally("finish.pair_footnotes", || {
            legal_structure::note_pairing::pair_footnotes(&lines, false, &|at, start, end| {
                index.global_range(lines[at].id, start, end)
            })
        });
        diagnostics.extend(pairing.diagnostics);
        let mut notes = pairing.notes;
        // (TEMP lever "notes-not-sections", remove before landing.) An instrument's line the decision opens as a section
        // ("13. Every County or Municipality ...") is the body's, though the pairer read its number as a note's: the note
        // it opens is no note.
        // (TEMP lever "notes-counted", remove before landing: the same, only where the section goes on with the count in
        // reading order, the section the decision opens before it one less or the one after it one more: "14" between 13
        // and 15 is the body's; a judgment's footnote "3" set after paragraph 40 is a note.)
        let counted = legal_structure_model::lever("notes-counted");
        // Each section the decision opens, by its move's place, and its whole number (none for "15(C)").
        let opened: Vec<(usize, u32)> = decision
            .moves
            .iter()
            .enumerate()
            .filter(|(_, chosen)| chosen.opens && chosen.unit.is_some_and(|(_, role)| role == legal_structure_model::UnitRole::Section))
            .filter_map(|(k, chosen)| {
                let (run, at) = chosen.member?;
                let value = &runs.runs.get(run as usize)?.members.get(at as usize)?.grammar_value;
                Some((k, value.trim_end_matches('.').parse::<u32>().ok()?))
            })
            .collect();
        // (Counted through the body: the sections no pairer note holds stand either side, and the held run between them
        // fills that gap exactly, 12 then held 13, 14, 15 then 16. A judgment's footnotes 1, 2, 3 held between its
        // paragraphs 40 and 41 fill nothing.)
        let held: std::collections::HashSet<usize> = notes
            .iter()
            .filter_map(|note| note.line_ids.first())
            .filter_map(|id| slots.of_id(id))
            .filter_map(|slot| decision.place.get(slot.index()).map(|&at| at as usize))
            .collect();
        let continues = |place: usize| {
            let Some(here) = opened.iter().position(|&(k, _)| k == place) else { return false };
            let (mut first, mut last) = (here, here);
            while first > 0 && held.contains(&opened[first - 1].0) {
                first -= 1;
            }
            while last + 1 < opened.len() && held.contains(&opened[last + 1].0) {
                last += 1;
            }
            let (Some(before), Some(after)) = (first.checked_sub(1).map(|k| opened[k].1), opened.get(last + 1).map(|o| o.1)) else {
                return false;
            };
            let run: Vec<u32> = opened[first..=last].iter().map(|o| o.1).collect();
            after == before + run.len() as u32 + 1 && run.iter().enumerate().all(|(k, &value)| value == before + k as u32 + 1)
        };
        // TEMP diagnostic (remove before landing; prints only for the line ids in LEGALPDF_DEBUG_LINES).
        if let Ok(watched) = std::env::var("LEGALPDF_DEBUG_LINES") {
            for note in &notes {
                let Some(id) = note.line_ids.first().filter(|id| watched == "*" || watched.split(',').any(|w| w == id.as_str())) else { continue };
                let place = slots.of_id(id).and_then(|slot| decision.place.get(slot.index()).copied()).map(|at| at as usize);
                let here = place.and_then(|place| opened.iter().position(|&(k, _)| k == place));
                eprintln!("NOTE {id} kind {:?} label {:?} section {:?} before {:?} after {:?} continues {}", note.kind, note.label,
                    here.map(|h| opened[h].1), here.and_then(|h| h.checked_sub(1)).map(|h| opened[h].1),
                    here.and_then(|h| opened.get(h + 1)).map(|o| o.1), place.is_some_and(|place| continues(place)));
            }
        }
        if evidence.instrument
            && (counted || legal_structure_model::lever("notes-not-sections"))
        {
            // (Its line carries the section's words; a number alone, "5", may be a page's folio the decision misread.)
            // (Under notes-counted, an endnote only: a judgment's page footnotes that a scan's decision opens as sections
            // stay notes.)
            notes.retain(|note| {
                (counted && note.kind != legal_structure_model::NodeKind::Endnote) || !note.line_ids.first().and_then(|id| slots.of_id(id)).is_some_and(|slot| {
                    slots.line(pages, slot).text.chars().filter(|c| c.is_alphabetic()).count() >= 3
                        && decision.place.get(slot.index()).is_some_and(|&at| {
                            decision.moves.get(at as usize).is_some_and(|chosen| {
                                chosen.unit.is_some_and(|(_, role)| role == legal_structure_model::UnitRole::Section)
                                    && (!counted || continues(at as usize))
                            })
                        })
                })
            });
        }
        notes
    };
    give_back(&mut zones, slots, &notes);
    diagnostics.extend(legal_pdf_core::profile::tally("finish.unmatched", || {
        unmatched_reference_diagnostics(
            fences
                .order
                .iter()
                .map(|&slot| (slots.line(pages, slot), zones[slot.index()])),
            &notes,
            &index,
            &decided,
        )
    }));
    let mut structure = legal_pdf_core::profile::tally("finish.write_structure", || write_structure(
        pages, slots, evidence, &fences, runs, &decision, &views, &index, &decided, notes, identity,
        &written, given,
    ))?;
    let runs = legal_pdf_core::profile::tally("finish.render", || rendered::render(pages, &mut structure))?;
    let units = legal_pdf_core::profile::tally("finish.authority_units", || rendered::authority_units(&structure, &runs));
    legal_pdf_core::profile::tally("finish.propositions", || {
        crate::notes::note_context::attach_propositions(&mut structure, &units, &decided)
    });
    structure
        .refresh_revision()
        .map_err(|e| Error::Message(e.to_string()))?;
    Ok(StructureOutput {
        authority_text_units: units,
        diagnostics,
        structure_graph: structure,
    })
}

