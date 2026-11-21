//! Which lines are notes, read in each document and language once the runs stand: each page's note
//! band (`read_band`); every line under the separator, from the band's cut, of a page whose band
//! reads a note (the band read one there; or two band labels count on by one, or one is the note
//! the document expects next once the band has read one, and then only a line set in a note's
//! type); then the lanes that take more of the document's lines for notes against it
//! (`note_promotions`); then each page's notes settled as footnotes or endnotes, the lines taken
//! beside the band's in an endnote run the body never refers to given back (`verified_endnotes`).
//! A page's first note line with no label goes on with the page before's last note, unless it is a
//! line taken beside the band's on a page the band read no note on.
//!
//! Two of the band's inputs come from the fences and the runs. A line whose number continues a
//! counted run of the body, set in the body's type, is a numbered paragraph's, never a note's,
//! unless its page shows otherwise (under the separator, referred to above, a notes heading over
//! it, or the endnotes' count going on through it); the lanes leave it alone too. A table's cells
//! in a band holding a note label are the band's, not the table's (`Fences`). With no separator, a
//! small label numbered next after the document's notes, with its reference above it, is a note to
//! the page's foot. The body's type the band and the runs are read against is the largest size
//! holding a tenth of the document's running text, so that a note's type, however common, never
//! becomes the body's.

use crate::model::decide::NoteSegment;
use crate::model::decide::Notes;
use crate::model::fences::Fences;
use crate::model::{Evidence, PageView, Role, Runs, Slot, Slots};
use crate::band::{read_band, verified_endnotes};
use crate::rules::notes::band::separator_tolerance;
use crate::model::notes::{Document, NoteKind};
use crate::rules::notes::lines::endnote_heading;
use crate::lanes::note_promotions;
use legal_pdf_core::model::Page;
use legal_pdf_core::regex;
use legal_structure_model::text::{label_prefix, line_start_label_prefix, normalize_label};
use std::collections::{BTreeMap, HashMap};

/// A line saying a note runs on across the page: "(Footnote continued…)", "Notes continued".
fn continued_note(text: &str) -> bool {
    regex!(r"(?i)^\W*(?:foot)?notes?\s+continued\b").is_match(text)
}



/// The body's type size in a document and language (`lines`): the largest size (`line_font_size`,
/// which a scan's lines carry too) holding a tenth or more of the characters of its lines no fence
/// took outside the note band, so that a note's or a quotation's type, however common, never
/// becomes the body's; 0 where no line carries a size.
fn body_size(pages: &[Page], slots: &Slots, fences: &Fences, lines: &[Slot]) -> f64 {
    let mut mass: HashMap<i64, usize> = HashMap::new();
    for &slot in lines {
        let at = slot.index();
        if fences.role[at] == Role::Open
            && !fences.band[at]
            && slots.line(pages, Slot(at as u32)).size > 0.0
        {
            *mass
                .entry((slots.line(pages, Slot(at as u32)).size * 100.0).round() as i64)
                .or_default() += slots.line(pages, slot).text.chars().count();
        }
    }
    let total: usize = mass.values().sum();
    mass.into_iter()
        .filter(|(_, chars)| total > 0 && chars.saturating_mul(10) >= total)
        .map(|(size, _)| size as f64 / 100.0)
        .max_by(f64::total_cmp)
        .unwrap_or(0.0)
}

/// Where a walk over a document's pages stands among its endnotes.
#[derive(Clone, Copy)]
enum EndnoteRun {
    Outside,
    /// In a run of endnotes; the label the next page's should open with, where the run printed one.
    Within {
        expected: Option<u32>,
    },
}

impl EndnoteRun {
    /// Whether notes numbered `numbers` go on with the run: no label, or the one it expects.
    fn goes_on_at(self, numbers: &[u32]) -> bool {
        matches!(self, Self::Within { expected } if numbers.is_empty() || expected.is_some_and(|expected| numbers[0] == expected))
    }

    /// The run, entered or kept, past a page of endnotes numbered `numbers`.
    fn through(self, numbers: &[u32]) -> Self {
        let kept = match self {
            Self::Within { expected } => expected,
            Self::Outside => None,
        };
        Self::Within {
            expected: numbers.last().map_or(kept, |number| number.checked_add(1)),
        }
    }
}

/// The notes of the document's pages, measured (`evidence`), fenced and ordered (`fences`), its
/// numbering runs read (`runs`).
/// The notes of the document's pages, measured (`evidence`), fenced and ordered (`fences`), its
/// numbering runs read (`runs`).
pub fn read(
    pages: &[Page],
    slots: &Slots,
    evidence: &Evidence,
    fences: &Fences,
    runs: &Runs,
) -> Notes {
    let count = slots.len();
    let separators = &fences.separators;
    let starts: Vec<usize> = (0..pages.len())
        .map(|page| slots.on(page).start as usize)
        .collect();
    // Each document and language: its lines in page order, and each page's in reading order.
    let group = |slot: Slot| {
        (
            fences.at(slot.index()).document,
            fences.at(slot.index()).translation,
        )
    };
    let mut groups: BTreeMap<(u16, bool), (Vec<Slot>, Vec<(usize, Vec<usize>)>)> =
        BTreeMap::new();
    for slot in (0..count).map(|at| Slot(at as u32)) {
        groups.entry(group(slot)).or_default().0.push(slot);
    }
    for &slot in &fences.order {
        let (page, place) = (slots.page(slot), slot.index() - starts[slots.page(slot)]);
        let order = &mut groups.entry(group(slot)).or_default().1;
        match order.last_mut() {
            Some((last, places)) if *last == page => places.push(place),
            _ => order.push((page, vec![place])),
        }
    }
    let body_sizes: BTreeMap<(u16, bool), f64> = groups
        .iter()
        .map(|(key, (lines, _))| (*key, body_size(pages, slots, fences, lines)))
        .collect();
    // A counted run's member that opens a unit, its text set in its document's body type.
    let mut unit = vec![false; count];
    for member in runs
        .members()
        .filter(|member| member.counted && member.kind.is_some())
    {
        let body_size = body_sizes.get(&group(member.slot)).copied().unwrap_or(0.0);
        let text = member.beside.unwrap_or(member.slot);
        if body_size > 0.0 && slots.line(pages, text).size >= body_size * 0.90 {
            unit[member.slot.index()] = true;
        }
    }
    let mut notes = Notes {
        note: vec![None; count],
        paragraph_label: vec![false; count],
        suppressed: vec![false; count],
        running_head: vec![false; count],
        heading: vec![false; count],
        ..Notes::default()
    };
    // The lines taken beside the band's notes under the separator, and the pages the band read
    // a note on, by page slot.
    let mut added = vec![false; count];
    let mut read = vec![false; pages.len()];
    for (key, (lines_of, order)) in groups {
        let document = Document {
            slots,
            evidence,
            fences,
            unit: &unit,
            body_size: body_sizes[&key],
        };
        let views = PageView::of_slots(pages, slots, &lines_of);
        let reading =
            legal_pdf_core::profile::measure("notes.band", || read_band(&views, document));
        for &(slot, kind) in &reading.notes {
            notes.note[slot.index()] = Some(kind);
        }
        for (list, marks) in [
            (&reading.paragraph_labels, &mut notes.paragraph_label),
            (&reading.suppressed, &mut notes.suppressed),
            (&reading.running_heads, &mut notes.running_head),
            (&reading.headings, &mut notes.heading),
        ] {
            for slot in list {
                marks[slot.index()] = true;
            }
        }
        notes.diagnostics.extend(reading.diagnostics);
        // Each page whose band reads a note: the band read one there, or two of its labels count
        // on by one, or one is the note the document's notes expect next once the band has read
        // one; whether the band read one there, and where the band begins on it, the band's cut,
        // else the separator.
        let mut noted: HashMap<usize, (bool, Option<f64>)> = HashMap::new();
        let mut seen = false;
        for (view, page_reading) in views.iter().zip(&reading.pages) {
            let values: Vec<u32> = view
                .lines
                .iter()
                .filter(|slot| {
                    fences.band[slot.index()]
                        && matches!(fences.role[slot.index()], Role::Open | Role::Table)
                })
                .filter_map(|slot| {
                    evidence.label[slot.index()]
                        .filter(|label| label.suppress == 0 && label.value > 0)
                })
                .map(|label| label.value)
                .collect();
            let page = slots.page(view.slot(0));
            if page_reading.noted
                || seen && values.contains(&page_reading.next_label)
                || values.iter().any(|value| values.contains(&(value + 1)))
            {
                noted.insert(page, (page_reading.noted, page_reading.cut));
            }
            read[page] = page_reading.noted;
            seen |= page_reading.noted;
        }
        // Every line under the separator, and under the band's cut, of a page whose band reads
        // a note is a note's: its notes heading and an endnote page's running head aside. On a
        // page the band read no note on, only a line set in a note's type, smaller than the
        // body's (a map's labels and a column of the body under a stray rule are none).
        let body = document.body_size;
        for &slot in &lines_of {
            let at = slot.index();
            let Some(&(read_here, cut)) = noted.get(&slots.page(slot)) else {
                continue;
            };
            let typed = read_here
                || body > 0.0
                    && slots.line(pages, Slot(at as u32)).size > 0.0
                    && slots.line(pages, Slot(at as u32)).size < body * 0.90;
            if fences.band[at]
                && typed
                && !document.paragraph(at)
                && cut.is_none_or(|cut| {
                    slots.line(pages, slot).bbox[1]
                        >= cut - separator_tolerance(&pages[slots.page(slot)])
                })
                && fences.role[at] == Role::Open
                && !notes.heading[at]
                && !notes.running_head[at]
                && notes.note[at].is_none()
            {
                notes.note[at] = Some(NoteKind::Footnote);
                added[at] = true;
            }
        }
        // The lanes read the band's notes, and take only the body lines it left open: no
        // heading, furniture, line label, contents, table cell, contents page's line or
        // numbered paragraph's line (`DocumentLines::mutable`).
        let promoted = legal_pdf_core::profile::measure("notes.lanes", || {
            note_promotions(pages, &order, document, &notes, separators)
        });
        for (page, place) in promoted {
            notes.note[starts[page] + place].get_or_insert(NoteKind::Footnote);
        }
        crate::find::settle_kinds(&mut notes, pages, slots, &views);
        // The lines taken beside the band's own notes in an endnote run the body never refers
        // to are the body's; the band's own and the lanes' stand.
        let mut held: Vec<Vec<(usize, NoteKind)>> = views
            .iter()
            .map(|view| {
                view.iter()
                    .filter_map(|(slot, _)| Some((view.place(slot), notes.note[slot.index()]?)))
                    .collect()
            })
            .collect();
        verified_endnotes(&views, document, &mut held);
        for (view, held) in views.iter().zip(&held) {
            for &slot in &view.lines {
                let place = view.place(slot);
                if added[slot.index()] && !held.iter().any(|(kept, _)| *kept == place) {
                    notes.note[slot.index()] = None;
                }
            }
        }
        // A note's line has no body text under it across its span: one that has stands above
        // the page's foot (a small-type headnote, a cover's lines), and is the body's (on the
        // gold, 1,951 such lines were no notes, 170 were).
        let mut lifted = Vec::new();
        for view in &views {
            for &slot in view
                .lines
                .iter()
                .filter(|slot| notes.note[slot.index()].is_some())
            {
                let [x0, _, x1, y1] = slots.line(pages, slot).bbox;
                let under = view.lines.iter().any(|&other| {
                    let (o, below) = (other.index(), slots.line(pages, other).bbox);
                    notes.note[o].is_none()
                        && fences.role[o] == Role::Open
                        && !notes.heading[o]
                        && !notes.running_head[o]
                        && below[1] >= y1 - 0.5
                        && below[0] < x1
                        && x0 < below[2]
                });
                if under {
                    lifted.push(slot);
                }
            }
        }
        for slot in lifted {
            notes.note[slot.index()] = None;
        }
        crate::find::segment(&mut notes, pages, slots, &order, &added, &read);
    }
    notes
}

/// Each page's notes settled as footnotes or endnotes, page by page through a document and
/// language: all its notes are endnotes on a page under a notes heading, holding an endnote,
/// or going on with the endnotes of the page before (no label, or the label they expect).
/// Each page's notes settled as footnotes or endnotes, page by page through a document and
/// language: all its notes are endnotes on a page under a notes heading, holding an endnote,
/// or going on with the endnotes of the page before (no label, or the label they expect).
pub fn settle_kinds(notes: &mut Notes, pages: &[Page], slots: &Slots, views: &[PageView]) {
    let mut run = EndnoteRun::Outside;
    let mut previous_page: Option<usize> = None;
    for view in views {
        let index = view.page.index;
        if previous_page.is_some_and(|prior| prior.checked_add(1) != Some(index)) {
            run = EndnoteRun::Outside;
        }
        previous_page = Some(index);
        let noted: Vec<Slot> = view
            .lines
            .iter()
            .copied()
            .filter(|slot| notes.note[slot.index()].is_some())
            .collect();
        if noted.is_empty() {
            run = EndnoteRun::Outside;
            continue;
        }
        let heading = view
            .lines
            .iter()
            .any(|&slot| endnote_heading(&slots.line(pages, slot).text));
        let explicit = noted
            .iter()
            .any(|slot| notes.note[slot.index()] == Some(NoteKind::Endnote));
        let numbers: Vec<u32> = noted
            .iter()
            .filter_map(|&slot| line_start_label_prefix(&slots.line(pages, slot).text))
            .filter_map(|prefix| normalize_label(&prefix.label).parse().ok())
            .collect();
        if heading || explicit || run.goes_on_at(&numbers) {
            for slot in &noted {
                notes.note[slot.index()] = Some(NoteKind::Endnote);
            }
            run = run.through(&numbers);
        } else {
            run = EndnoteRun::Outside;
        }
    }
}

/// Each page's note lines of a document and language (`order`: each page's lines in reading
/// order, by place), and whether their first goes on with the last note of the page before: the
/// page before ended its notes saying they continue, or its own first note line says so
/// ("(Footnote continued…)"), or it opens with no label, unless it is a line taken beside the
/// band's under the separator (`added`, by slot) on a page the band read no note on (`read`, by
/// page slot).
/// Each page's note lines of a document and language (`order`: each page's lines in reading
/// order, by place), and whether their first goes on with the last note of the page before: the
/// page before ended its notes saying they continue, or its own first note line says so
/// ("(Footnote continued…)"), or it opens with no label, unless it is a line taken beside the
/// band's under the separator (`added`, by slot) on a page the band read no note on (`read`, by
/// page slot).
pub fn segment(notes: &mut Notes, pages: &[Page],
    slots: &Slots,
    order: &[(usize, Vec<usize>)],
    added: &[bool],
    read: &[bool],
) {
    // The page index of the last page with notes, and whether its last note line says it goes on.
    let mut previous: Option<(usize, bool)> = None;
    for (page, places) in order {
        let start = slots.on(*page).start;
        let mut lines: Vec<Slot> = places
            .iter()
            .map(|&place| Slot(start + place as u32))
            .filter(|slot| notes.note[slot.index()].is_some())
            .collect();
        let Some(&first) = lines.first() else {
            continue;
        };
        let index = pages[*page].index;
        let first_text = &slots.line(pages, first).text;
        let unlabelled =
            label_prefix(first_text).is_none() && (!added[first.index()] || read[*page]);
        let continued = previous.is_some_and(|(prior, said)| {
            prior + 1 == index && (said || unlabelled || continued_note(first_text))
        });
        // A page's notes open at a note's label, unless they go on with the page before's: the
        // lines before its first label are the body's (a cover's small type, a headnote over
        // the text; on the gold, 2,444 such lines were no notes, 449 were).
        if !continued {
            let opens = lines
                .iter()
                .position(|&slot| label_prefix(&slots.line(pages, slot).text).is_some())
                .unwrap_or(lines.len());
            for slot in lines.drain(..opens) {
                notes.note[slot.index()] = None;
            }
        }
        let Some(&last) = lines.last() else {
            continue;
        };
        previous = Some((index, continued_note(&slots.line(pages, last).text)));
        notes.segments.push(NoteSegment { lines, continued });
    }
}

