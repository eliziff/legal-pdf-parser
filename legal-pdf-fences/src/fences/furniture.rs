//! Page numbers, running heads and feet: the document's and, once its constituent documents are known, each one's own.
use crate::model::fences::Fences;
use crate::model::{Edge, Evidence, PageView, Role, Rows, Rule, Slot, Slots};
use legal_pdf_core::model::Page;

/// Each page's number and the line printing it, as the page map reads them (`legal_pdf_pages::page_map`) over the
/// pages' lines, a transcript's line numbers left out, and the labels the PDF declares (`page_labels`, by page
/// index): the line its page's label.
/// Each page's number and the line printing it, as the page map reads them (`legal_pdf_pages::page_map`) over the
/// pages' lines, a transcript's line numbers left out, and the labels the PDF declares (`page_labels`, by page
/// index): the line its page's label.
pub fn read_page_numbers(fences: &mut Fences, pages: &[Page],
    slots: &Slots,
    evidence: &Evidence,
    page_labels: &[Option<String>],
) {
    let line_label = |page: usize, at: usize| {
        evidence.line_number[slots.at(page, at).index()]
            || pages[page].lines[at].exclude_from_body
    };
    let mut read = legal_pdf_core::profile::tally("pages.evidence", || {
        crate::furniture::page_evidence(pages, page_labels, line_label)
    });
    let mut rowed = legal_pdf_pages::PageRows::of(&read);
    let map = legal_pdf_core::profile::tally("pages.map", || {
        legal_pdf_pages::page_map_rowed(&read, &rowed, &[], &|_, _| false)
    });
    fences.folios = pages
        .iter()
        .enumerate()
        .map(|(page_slot, page)| {
            let binding = map.get(page.index)?;
            let line = binding
                .line
                .map(|at| crate::find::folio(fences, pages, slots, page_slot, at));
            Some((binding.label.clone()?, line))
        })
        .collect();
    // A page printing a second numbering (a record's page over a transcript's own, a book's over an article's): the
    // same fit over the pages with the first numbering's lines left blank. Its lines that print the number alone (a
    // folio's few characters, not a running head holding one) in the page's head or foot are page labels too; the
    // page keeps the first's number.
    // Only a page whose line was blanked has rows of its own to read again.
    let mut blanked = Vec::new();
    for (at, (page, binding)) in read.iter_mut().zip(&map).enumerate() {
        if let Some(line) = binding.line.and_then(|line| page.lines.get_mut(line)) {
            line.text.clear();
            blanked.push(at);
        }
    }
    for at in blanked {
        rowed.reread(&read, at);
    }
    let second = legal_pdf_core::profile::tally("pages.second", || {
        legal_pdf_pages::page_map_rowed(&read, &rowed, &[], &|_, _| false)
    });
    // A numbering counts the pages: most of its lines print the PDF page's number at one shift, on three pages at
    // least; its lines at that shift are its own (a note's number in the foot is none).
    let shift = |page: &Page| {
        let binding = second.get(page.index)?;
        binding.line?;
        Some(binding.label.as_deref()?.parse::<i64>().ok()? - page.index as i64)
    };
    let mut shifts: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    for shift in pages.iter().filter_map(shift) {
        *shifts.entry(shift).or_default() += 1;
    }
    let total: usize = shifts.values().sum();
    let counted = shifts
        .into_iter()
        .max_by_key(|&(shift, count)| (count, -shift.abs()))
        .filter(|&(_, count)| count >= 3 && count * 2 > total)
        .map(|(shift, _)| shift);
    for (page_slot, page) in pages
        .iter()
        .enumerate()
        .filter(|(_, page)| counted.is_some() && shift(page) == counted)
    {
        // In the page's head or foot (a margin's paragraph numbers count too, in the body).
        let band = legal_pdf_pages::EDGE_BAND * page.height;
        let alone = |at: &usize| {
            let line = &page.lines[*at];
            line.text.trim().chars().count() <= 12
                && (line.bbox[3] < band || line.bbox[1] > page.height - band)
        };
        if let Some(at) = second
            .get(page.index)
            .and_then(|binding| binding.line)
            .filter(alone)
        {
            crate::find::folio(fences, pages, slots, page_slot, at);
        }
    }
}

/// Line `at` of page slot `page_slot` set apart as its page's label, at the edge it stands on.
/// Line `at` of page slot `page_slot` set apart as its page's label, at the edge it stands on.
pub fn folio(fences: &mut Fences, pages: &[Page], slots: &Slots, page_slot: usize, at: usize) -> Slot {
    let slot = slots.at(page_slot, at);
    let (bbox, height) = (pages[page_slot].lines[at].bbox, pages[page_slot].height);
    crate::find::set(fences, slot, Role::PageLabel, Rule::Folio);
    fences.edge[slot.index()] = Some(if bbox[1] + bbox[3] < height {
        Edge::Top
    } else {
        Edge::Bottom
    });
    slot
}

/// Running heads and feet, the folios' rows among them (`furniture::read`); and the edge lines naming their page's
/// sections.
/// Running heads and feet, the folios' rows among them (`furniture::read`); and the edge lines naming their page's
/// sections.
pub fn read_furniture(fences: &mut Fences, evidence: &Evidence,
    views: &[PageView],
    rows: &[&Rows],
) {
    let found = legal_pdf_core::profile::measure("prepare.furniture", || {
        crate::furniture::read(
            views,
            rows,
            |slot| evidence.line_number[slot.index()],
            |slot| fences.role[slot.index()] == Role::PageLabel,
        )
    });
    for (slot, edge, rule) in found {
        if crate::find::set(fences, slot, Role::Furniture, rule) {
            fences.edge[slot.index()] = Some(edge);
        }
    }
    // TEMP lever "glyph-columns" (remove before landing). A note set sideways in the margin (a provision's history, in a
    // consolidation that prints it up the page) is read a letter to a line: four or more single characters, one close
    // under the next at one left edge. They are no lines of the text. A list's own marks standing alone are none of
    // them: they count on ("1", "2", "3"; "a", "b", "c") or are one bullet repeated.
    if legal_structure_model::lever("glyph-columns") {
        for view in views {
            let mut singles: Vec<(Slot, [f64; 4], char)> = view
                .lines
                .iter()
                .filter_map(|&slot| {
                    let line = &view.page.lines[view.place(slot)];
                    let mut characters = line.text.trim().chars();
                    let only = characters.next().filter(|_| characters.next().is_none())?;
                    Some((slot, line.bbox, only))
                })
                .collect();
            singles.sort_by(|a, b| a.1[0].total_cmp(&b.1[0]).then(a.1[1].total_cmp(&b.1[1])));
            let mut start = 0;
            for at in 1..=singles.len() {
                let goes_on = at < singles.len() && {
                    let (above, below) = (singles[at - 1].1, singles[at].1);
                    (below[0] - above[0]).abs() <= 1.5 && below[1] - above[3] <= 0.6 * (above[3] - above[1])
                };
                if goes_on {
                    continue;
                }
                let column = &singles[start..at];
                let listed = column.windows(2).all(|pair| pair[0].2 == pair[1].2)
                    || column.windows(2).all(|pair| u32::from(pair[1].2) == u32::from(pair[0].2) + 1);
                if column.len() >= 4 && !listed {
                    for &(slot, ..) in column {
                        crate::find::set(fences, slot, Role::Furniture, Rule::PrintMargin);
                    }
                }
                start = at;
            }
        }
    }
    fences.section_heads = crate::furniture::section_heads(views, rows)
        .into_iter()
        .filter(|(slot, _, _)| fences.role[slot.index()] == Role::Open)
        .collect();
}

/// Each constituent document's own furniture, read over its own lines as a document of its own
/// (`furniture::read`): the running heads and feet it repeats on its own pages, a short
/// document's edges. A line no fence took is its furniture.
/// Each constituent document's own furniture, read over its own lines as a document of its own
/// (`furniture::read`): the running heads and feet it repeats on its own pages, a short
/// document's edges. A line no fence took is its furniture.
pub fn read_document_furniture(fences: &mut Fences, pages: &[Page],
    slots: &Slots,
    evidence: &Evidence,
    rows: &[&Rows],
) {
    let documents = fences.documents();
    let mut lines: Vec<Vec<Slot>> = vec![Vec::new(); documents];
    for slot in 0..slots.len() {
        lines[usize::from(fences.at(slot).document)].push(Slot(slot as u32));
    }
    for lines in lines.iter().filter(|lines| !lines.is_empty()) {
        let views = PageView::of_slots(pages, slots, lines);
        let rows: Vec<&Rows> = views
            .iter()
            .map(|view| rows[slots.page(view.lines[0])])
            .collect();
        let found = legal_pdf_core::profile::measure("prepare.furniture", || {
            crate::furniture::read(
                &views,
                &rows,
                |slot| evidence.line_number[slot.index()],
                |slot| fences.role[slot.index()] == Role::PageLabel,
            )
        });
        for (slot, edge, rule) in found {
            if crate::find::set(fences, slot, Role::Furniture, rule) {
                fences.edge[slot.index()] = Some(edge);
                fences.band[slot.index()] = false;
            }
        }
    }
}

