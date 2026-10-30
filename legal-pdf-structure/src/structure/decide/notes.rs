//! Which lines are notes, read in each document and language once the runs stand: each page's note
//! band (`read_band`), every line of a band holding a note label (`Fences::noted`), then the lanes
//! that take more of the document's lines for notes against it (`note_promotions`), then each
//! page's notes settled as footnotes or endnotes.
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

use super::*;
use crate::structure::notes::lanes::{note_promotions, LaneLines, LANE_BODY, LANE_FURNITURE, LANE_NOTE, LANE_PARAGRAPH};

/// A line saying a note runs on across the page: "(Footnote continued…)", "Notes continued".
fn continued_note(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)^\W*(?:foot)?notes?\s+continued\b").unwrap()).is_match(text)
}

/// A page's note lines in one document and language, in reading order, and whether the first opens
/// with no label of its own and so goes on with the last note of the page before.
#[derive(Debug)]
pub(crate) struct NoteSegment {
    pub(crate) lines: Vec<Slot>,
    pub(crate) continued: bool,
}

/// What the note band and its lanes read of a document, by slot: each note line's kind; the lines
/// whose number the band read as a numbered paragraph's; the labels it read as no note's; an
/// endnote page's running heads; the notes headings; each page's note segments; and the warnings
/// of a band drawn on weak geometry.
#[derive(Debug, Default)]
pub(crate) struct Notes {
    pub(crate) note: Vec<Option<NoteKind>>,
    pub(crate) paragraph_label: Vec<bool>,
    pub(crate) suppressed: Vec<bool>,
    pub(crate) running_head: Vec<bool>,
    pub(crate) heading: Vec<bool>,
    pub(crate) segments: Vec<NoteSegment>,
    pub(crate) diagnostics: Vec<Diagnostic>,
}

/// The body's type size in a document and language (`lines`): the largest size (`line_font_size`,
/// which a scan's lines carry too) holding a tenth or more of the characters of its lines no fence
/// took outside the note band, so that a note's or a quotation's type, however common, never
/// becomes the body's; 0 where no line carries a size.
fn body_size(pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, lines: &[Slot]) -> f64 {
    let mut mass: HashMap<i64, usize> = HashMap::new();
    for &slot in lines {
        let at = slot.index();
        if fences.role[at] == Role::Open && !fences.band[at] && evidence.size[at] > 0.0 {
            *mass.entry((evidence.size[at] * 100.0).round() as i64).or_default() += slots.line(pages, slot).text.chars().count();
        }
    }
    let total: usize = mass.values().sum();
    mass.into_iter()
        .filter(|(_, chars)| total > 0 && chars.saturating_mul(10) >= total)
        .map(|(size, _)| size as f64 / 100.0)
        .max_by(f64::total_cmp)
        .unwrap_or(0.0)
}

impl Notes {
    /// The notes of the document's pages, measured (`evidence`), fenced and ordered (`fences`), its
    /// numbering runs read (`runs`).
    pub(crate) fn read(pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, runs: &Runs) -> Self {
        let count = slots.len();
        let separators: Vec<Option<f64>> = evidence.pages.iter().map(|page| page.separator).collect();
        let starts: Vec<usize> = (0..pages.len()).map(|page| slots.on(page).start as usize).collect();
        // Each document and language: its lines in page order, and each page's in reading order.
        let group = |slot: Slot| (fences.document[slot.index()], fences.translation[slot.index()]);
        let mut groups: BTreeMap<(u16, bool), (Vec<Slot>, Vec<(usize, Vec<usize>)>)> = BTreeMap::new();
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
        let body_sizes: BTreeMap<(u16, bool), f64> =
            groups.iter().map(|(key, (lines, _))| (*key, body_size(pages, slots, evidence, fences, lines))).collect();
        // A counted run's member that opens a unit, its text set in its document's body type.
        let mut unit = vec![false; count];
        for member in runs.members().filter(|member| member.counted && member.kind.is_some()) {
            let body_size = body_sizes.get(&group(member.slot)).copied().unwrap_or(0.0);
            let text = member.beside.unwrap_or(member.slot);
            if body_size > 0.0 && evidence.size[text.index()] >= body_size * 0.90 {
                unit[member.slot.index()] = true;
            }
        }
        let headed = |at: usize| {
            fences.role[at] == Role::Heading || matches!(evidence.tag[at], SourceTag::Heading(_) | SourceTag::Title | SourceTag::ParagraphTitle)
        };
        let layout_notes = legal_structure_model::ablated("layout_notes");
        let flags: Vec<u8> = (0..count)
            .map(|at| {
                [
                    (BAND_EXCLUDED, matches!(fences.role[at], Role::LineLabel | Role::Contents)),
                    (BAND_FURNITURE, matches!(fences.role[at], Role::Furniture | Role::PageLabel)),
                    (BAND_CELL, fences.cell[at] != 0),
                    (BAND_PARAGRAPH, unit[at]),
                    (BAND_NOTED, evidence.tag[at] == SourceTag::Note),
                    (BAND_HEADED, headed(at)),
                    (BAND_LAYOUT, layout_notes && layout_labelled(slots.line(pages, Slot(at as u32)))),
                ]
                .into_iter()
                .filter(|(_, holds)| *holds)
                .fold(0, |bits, (bit, _)| bits | bit)
            })
            .collect();
        let mut notes = Self {
            note: vec![None; count],
            paragraph_label: vec![false; count],
            suppressed: vec![false; count],
            running_head: vec![false; count],
            heading: vec![false; count],
            ..Self::default()
        };
        let mut lane_flags = vec![0u8; count];
        for (key, (lines_of, order)) in groups {
            let lines = BandLines { sizes: &evidence.size, labels: &evidence.label, flags: &flags, body_size: body_sizes[&key] };
            let views = PageView::of_slots(pages, slots, &lines_of);
            let band_pages: Vec<BandPage> = views
                .into_iter()
                .map(|view| {
                    let page = slots.page(view.slot(0));
                    BandPage {
                        view,
                        separator: separators[page],
                        columns: fences.columns[page],
                        items: &evidence.items[page],
                        contents: fences.contents_pages[page],
                    }
                })
                .collect();
            let reading = legal_pdf_support::profile::measure("notes.band", || read_band(&band_pages, &lines));
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
            // Every line of a band holding a note label is a note's: its notes heading and an endnote
            // page's running head aside.
            for &slot in &lines_of {
                let at = slot.index();
                if fences.band[at]
                    && fences.noted[slots.page(slot)]
                    && fences.role[at] == Role::Open
                    && !notes.heading[at]
                    && !notes.running_head[at]
                {
                    notes.note[at].get_or_insert(NoteKind::Footnote);
                }
            }
            // The lanes read the band's notes, and take only the body lines it left open: no
            // heading, furniture, line label, contents, table cell, contents page's line or
            // numbered paragraph's line.
            for &slot in &lines_of {
                let at = slot.index();
                let furniture = flags[at] & BAND_FURNITURE != 0 || notes.running_head[at];
                let note = notes.note[at].is_some();
                let held = flags[at] & (BAND_EXCLUDED | BAND_CELL | BAND_HEADED) != 0
                    || notes.heading[at]
                    || notes.paragraph_label[at]
                    || fences.contents_pages[slots.page(slot)];
                lane_flags[at] = [
                    (LANE_NOTE, note),
                    (LANE_BODY, !note && !furniture && !held),
                    (LANE_FURNITURE, furniture),
                    (LANE_PARAGRAPH, notes.paragraph_label[at]),
                ]
                .into_iter()
                .filter(|(_, holds)| *holds)
                .fold(0, |bits, (bit, _)| bits | bit);
            }
            let promoted = legal_pdf_support::profile::measure("notes.lanes", || {
                note_promotions(LaneLines { pages, order: &order, starts: &starts, flags: &lane_flags }, &separators)
            });
            for (page, place) in promoted {
                notes.note[starts[page] + place].get_or_insert(NoteKind::Footnote);
            }
            notes.settle_kinds(pages, slots, &band_pages);
            notes.segment(pages, slots, &order);
        }
        notes
    }

    /// Each page's notes settled as footnotes or endnotes, page by page through a document and
    /// language: all its notes are endnotes on a page under a notes heading, holding an endnote,
    /// or going on with the endnotes of the page before (no label, or the label they expect).
    fn settle_kinds(&mut self, pages: &[Page], slots: &Slots, band_pages: &[BandPage]) {
        let mut active = false;
        let mut expected: Option<u32> = None;
        let mut previous_page: Option<usize> = None;
        for band in band_pages {
            let index = band.view.page.index;
            if previous_page.is_some_and(|prior| prior.checked_add(1) != Some(index)) {
                active = false;
                expected = None;
            }
            previous_page = Some(index);
            let noted: Vec<Slot> = band.view.lines.iter().copied().filter(|slot| self.note[slot.index()].is_some()).collect();
            if noted.is_empty() {
                active = false;
                expected = None;
                continue;
            }
            let heading = band.view.lines.iter().any(|&slot| endnote_heading(&slots.line(pages, slot).text));
            let explicit = noted.iter().any(|slot| self.note[slot.index()] == Some(NoteKind::Endnote));
            let numbers: Vec<u32> = noted
                .iter()
                .filter_map(|&slot| line_start_label_prefix(&slots.line(pages, slot).text))
                .filter_map(|prefix| normalize_label(&prefix.label).parse().ok())
                .collect();
            let continues = active && (numbers.is_empty() || expected.is_some_and(|expected| numbers[0] == expected));
            if heading || explicit || continues {
                for slot in &noted {
                    self.note[slot.index()] = Some(NoteKind::Endnote);
                }
                active = true;
                if let Some(number) = numbers.last() {
                    expected = number.checked_add(1);
                }
            } else {
                active = false;
                expected = None;
            }
        }
    }

    /// Each page's note lines of a document and language (`order`: each page's lines in reading
    /// order, by place), and whether their first goes on with the last note of the page before: it
    /// opens with no label, or the page before ended its notes saying they continue, or its own
    /// first note line says so ("(Footnote continued…)").
    fn segment(&mut self, pages: &[Page], slots: &Slots, order: &[(usize, Vec<usize>)]) {
        // The page index of the last page with notes, and whether its last note line says it goes on.
        let mut previous: Option<(usize, bool)> = None;
        for (page, places) in order {
            let start = slots.on(*page).start;
            let lines: Vec<Slot> = places.iter().map(|&place| Slot(start + place as u32)).filter(|slot| self.note[slot.index()].is_some()).collect();
            let (Some(&first), Some(&last)) = (lines.first(), lines.last()) else {
                continue;
            };
            let index = pages[*page].index;
            let first_text = &slots.line(pages, first).text;
            let continued = previous.is_some_and(|(prior, said)| {
                prior + 1 == index && (said || label_prefix(first_text).is_none() || continued_note(first_text))
            });
            previous = Some((index, continued_note(&slots.line(pages, last).text)));
            self.segments.push(NoteSegment { lines, continued });
        }
    }
}
