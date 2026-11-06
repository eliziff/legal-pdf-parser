//! The reading order page by page, and the paragraph numbers a report prints in its margin, set before their rows.
use super::*;

impl Fences {
    /// Every page's lines in reading order (`page_order`): its running head, its body by its columns
    /// with its note band woven in, its foot; a table's lines in its rows' order.
    pub(super) fn read_order(&mut self, pages: &[Page], slots: &Slots, evidence: &Evidence) {
        self.order = Vec::with_capacity(slots.len());
        // Each table line's table and its place among the table's lines, row by row, cell by cell.
        let mut in_table: Vec<Option<(u32, u32)>> = vec![None; slots.len()];
        for (table, value) in self.tables.iter().enumerate() {
            for (at, id) in value.line_ids().into_iter().enumerate() {
                if let Some(slot) = slots.of_id(id) {
                    in_table[slot.index()].get_or_insert((table as u32, at as u32));
                }
            }
        }
        legal_pdf_support::profile::measure("classify.order_pages", || {
            for (page_slot, page) in pages.iter().enumerate() {
                let start = slots.on(page_slot).start as usize;
                let stream = |line: usize| {
                    let slot = start + line;
                    match (self.role[slot], self.edge[slot]) {
                        (Role::Furniture | Role::PageLabel, Some(Edge::Top)) => Stream::Header,
                        (Role::Furniture | Role::PageLabel, Some(Edge::Bottom)) => Stream::Footer,
                        _ if self.band[slot] => Stream::Note,
                        _ => Stream::Body,
                    }
                };
                let table_page = (0..page.lines.len()).any(|line| self.cell[start + line] != 0);
                let (order, diagnostics) = page_order(
                    page,
                    stream,
                    |slot| self.role[start + slot] == Role::LineLabel,
                    &evidence.items[page_slot],
                    self.columns[page_slot],
                    table_page,
                    self.contents_pages[page_slot],
                );
                let order = if table_page { table_order(order, &in_table[start..start + page.lines.len()]) } else { order };
                self.diagnostics.extend(diagnostics);
                self.order.extend(order.into_iter().map(|line| Slot((start + line) as u32)));
            }
        });
        self.place = vec![0; slots.len()];
        for (place, slot) in self.order.iter().enumerate() {
            self.place[slot.index()] = place as u32;
        }
    }

    /// The paragraph numbers a report prints in its margin, beside the row where each paragraph's text opens (in
    /// both columns of a parallel report): a number alone on its line, standing clear of the page's body text by
    /// up to eight lines' height, level with a line of it, and counting on with the document's other such
    /// numbers, three or more in a row, a scan's count going on over a number it lost or a stray figure it read
    /// ("104", "1", "105"). Each is read just before the first line of its row in the main text (a
    /// parallel translation's after), so it opens that line's paragraph; a page's reading order otherwise sets the
    /// margin after both columns, where its numbers would open whatever follows (the next page's running head).
    pub(super) fn read_margin_numbers(&mut self, pages: &[Page], slots: &Slots) {
        self.margin_number = vec![None; slots.len()];
        self.margin_twin = vec![None; slots.len()];
        let mut found: Vec<(u32, Slot, Slot, Option<Slot>)> = Vec::new();
        for (page_slot, page) in pages.iter().enumerate() {
            // A page's note band may start high on it, over text the notes do not hold: its margin numbers are read too.
            let open = |slot: Slot| self.role[slot.index()] == Role::Open;
            // Where the page's lone numbers stand, top to bottom: a paragraph the other language opens rows below its
            // number's row opens before the next number's.
            let mut standing: Vec<f64> = slots.on(page_slot).map(Slot).filter(|&slot| open(slot)).filter_map(|slot| {
                let line = slots.line(pages, slot);
                lone_number(line).map(|_| line.bbox[1])
            }).collect();
            standing.sort_by(f64::total_cmp);
            if standing.is_empty() {
                continue;
            }
            // The page's text lines; its body's edges and height are read on those of some length, so a short line
            // ("The PSR stated at page 8:") opens its row's paragraph but moves no edge. (A column of numbers beside a column of
            // words reads as a table's grid: those words are text all the same. A lone letter is a report's margin letter, no row.)
            let words: Vec<Slot> = slots.on(page_slot).map(Slot)
                .filter(|&slot| (open(slot) || self.role[slot.index()] == Role::Table) && slots.line(pages, slot).text.chars().filter(|c| c.is_alphabetic()).count() >= 2).collect();
            let body: Vec<Slot> = words.iter().copied().filter(|&slot| {
                let line = slots.line(pages, slot);
                line.bbox[2] - line.bbox[0] > page.width * 0.20
            }).collect();
            // A line's ink across the page: a recognizer pads the box it reads a line in, a quarter of its height at
            // each end (a scan's margin number all but meets its column's padded edge).
            let ink = |line: &Line| {
                let pad = if line.ocr.is_some() { 0.25 * (line.bbox[3] - line.bbox[1]).max(0.0) } else { 0.0 };
                (line.bbox[0] + pad, line.bbox[2] - pad)
            };
            // The body's edges, where nearly all its lines stand: one line run out into the margin moves neither.
            let edge = |side: usize, share: f64| {
                let mut xs: Vec<f64> = body.iter().map(|&slot| { let (left, right) = ink(slots.line(pages, slot)); if side == 0 { left } else { right } }).collect();
                xs.sort_by(f64::total_cmp);
                xs.get((xs.len().saturating_sub(1) as f64 * share).round() as usize).copied()
            };
            let (Some(left), Some(right)) = (edge(0, 0.05), edge(2, 0.95)) else { continue };
            let bounds = [left, 0.0, right, 0.0];
            // The height most of the page's body lines stand.
            // (A PDF that gives its lines no height gives them their type's.)
            let mut heights: Vec<f64> = body.iter().map(|&slot| {
                let line = slots.line(pages, slot);
                Some(line.bbox[3] - line.bbox[1]).filter(|height| *height >= 1.0).unwrap_or_else(|| legal_pdf_core::line_font_size(line))
            }).filter(|h| *h >= 1.0).collect();
            heights.sort_by(f64::total_cmp);
            let text_height = heights.get(heights.len() / 2).copied().unwrap_or(10.0);
            for slot in slots.on(page_slot).map(Slot).filter(|&slot| open(slot)) {
                let line = slots.line(pages, slot);
                let Some(number) = lone_number(line).and_then(|text| text.parse::<u32>().ok()) else { continue };
                // A line a PDF gives no height (its box a baseline alone) stands as high as the page's text, above it.
                let height = if line.bbox[3] - line.bbox[1] >= 1.0 { line.bbox[3] - line.bbox[1] } else { text_height };
                let (left, right) = ink(line);
                let gap = (bounds[0] - right).max(left - bounds[2]);
                let middle_of = |bbox: [f64; 4]| if bbox[3] - bbox[1] >= 1.0 { (bbox[1] + bbox[3]) / 2.0 } else { bbox[3] - text_height / 2.0 };
                let middle = middle_of(line.bbox);
                let level = |other: Slot| (middle_of(slots.line(pages, other).bbox) - middle).abs() <= height * 0.6;
                // Wholly outside the body's edge, tight against the text it numbers (a recognizer's padded box nearly
                // touches a justified column's edge) or well out in the margin; or inside the body at its edge, the
                // paragraph's first line set on to its right ("[2]" at the left margin, its text indented on the row:
                // a word processor's numbered paragraph).
                let margin = gap > 0.0 && gap <= height * 8.0;
                let leading = !margin && line.bbox[0] <= bounds[0] + height
                    && words.iter().any(|&other| level(other) && slots.line(pages, other).bbox[0] >= line.bbox[2]);
                // Inside the column, under a note band, a number at the line's left is that note's.
                if !margin && (!leading || self.band[slot.index()]) {
                    continue;
                }
                // The row's line that opens a paragraph, set in from its column's edge as a first line is (the other
                // language's paragraph may open rows away, its line on this row running on mid-sentence), the main
                // text's before the translation's.
                let indented = |other: Slot| {
                    let x = slots.line(pages, other).bbox[0];
                    // (A justified line's row may be split into its words: they are no edge of one another.)
                    let edge = body.iter().filter(|&&line| !level(line)).map(|&line| slots.line(pages, line).bbox[0]).filter(|edge| *edge <= x && x - edge < height * 3.0).fold(x, f64::min);
                    let behind = words.iter().any(|&word| level(word) && self.translation[word.index()] == self.translation[other.index()] && slots.line(pages, word).bbox[2] <= x);
                    x - edge > height * 0.4 && !behind
                };
                // No paragraph opens on a lower-case word ("and dispassionate person in the community."): a stray
                // number beside a line running on is no paragraph's.
                let opens = |other: Slot| indented(other)
                    || slots.line(pages, other).text.trim_start().chars().next().is_some_and(|c| !c.is_lowercase());
                let row = words.iter().copied()
                    .filter(|&other| level(other) && opens(other) && (margin || slots.line(pages, other).bbox[0] >= line.bbox[2]))
                    .min_by_key(|&other| (!indented(other), self.translation[other.index()], self.place[other.index()]));
                if let Some(row) = row {
                    let other_language = |other: Slot| self.translation[other.index()] != self.translation[row.index()];
                    let next = standing.iter().copied().find(|&top| top > line.bbox[1] + height * 0.5).unwrap_or(f64::MAX);
                    let twin = words.iter().copied()
                        .filter(|&other| level(other) && opens(other) && other_language(other))
                        .min_by_key(|&other| (!indented(other), self.place[other.index()]))
                        .or_else(|| words.iter().copied()
                            .filter(|&other| {
                                let top = slots.line(pages, other).bbox[1];
                                other_language(other) && indented(other) && top >= line.bbox[1] - height * 0.6 && top < next
                            })
                            .min_by(|&a, &b| slots.line(pages, a).bbox[1].total_cmp(&slots.line(pages, b).bbox[1])));
                    found.push((number, slot, row, twin));
                }
            }
        }
        found.sort_by_key(|&(_, _, row, _)| self.place[row.index()]);
        // Each number's longest count ending on it, each going on from one of the few before it by a step of at most three.
        let mut length = vec![1usize; found.len()];
        let mut before = vec![None; found.len()];
        for at in 0..found.len() {
            for back in at.saturating_sub(4)..at {
                let step = found[at].0.checked_sub(found[back].0).filter(|step| (1..=3).contains(step));
                if step.is_some() && length[back] + 1 > length[at] {
                    length[at] = length[back] + 1;
                    before[at] = Some(back);
                }
            }
        }
        let mut kept = vec![false; found.len()];
        for at in (0..found.len()).filter(|&at| length[at] >= 3) {
            let mut link = Some(at);
            while let Some(at) = link {
                kept[at] = true;
                link = before[at];
            }
        }
        // A number belongs to the text it numbers: its language and document are its row line's (a number beside a
        // parallel report's French column numbers the English paragraph opening on its row as well).
        for (&(_, margin, row, twin), _) in found.iter().zip(&kept).filter(|(_, kept)| **kept) {
            self.margin_number[margin.index()] = Some(row);
            self.margin_twin[margin.index()] = twin;
            self.translation[margin.index()] = self.translation[row.index()];
            self.document[margin.index()] = self.document[row.index()];
        }
        if !kept.contains(&true) {
            return;
        }
        let mut before: HashMap<Slot, Vec<Slot>> = HashMap::new();
        for (slot, row) in self.margin_number.iter().enumerate().filter_map(|(slot, row)| row.map(|row| (Slot(slot as u32), row))) {
            before.entry(row).or_default().push(slot);
        }
        let order = std::mem::take(&mut self.order);
        for slot in order.into_iter().filter(|slot| self.margin_number[slot.index()].is_none()) {
            self.order.extend(before.remove(&slot).into_iter().flatten());
            self.order.push(slot);
        }
        for (place, slot) in self.order.iter().enumerate() {
            self.place[slot.index()] = place as u32;
        }
    }
}

/// A page's reading order (`order`, slots of its lines) with each table's lines in its rows' order,
/// cell by cell, where its first line stood; `in_table` holds each of the page's lines' table and
/// place among that table's lines.
fn table_order(order: Vec<usize>, in_table: &[Option<(u32, u32)>]) -> Vec<usize> {
    let mut held: Vec<(u32, Vec<(u32, usize)>)> = Vec::new();
    let mut sequence: Vec<std::result::Result<usize, u32>> = Vec::with_capacity(order.len());
    for line in order {
        match in_table[line] {
            Some((table, at)) => match held.iter_mut().find(|(seen, _)| *seen == table) {
                Some((_, lines)) => lines.push((at, line)),
                None => {
                    held.push((table, vec![(at, line)]));
                    sequence.push(Err(table));
                }
            },
            None => sequence.push(Ok(line)),
        }
    }
    let mut ordered = Vec::with_capacity(sequence.len());
    for entry in sequence {
        match entry {
            Ok(line) => ordered.push(line),
            Err(table) => {
                let at = held.iter().position(|(seen, _)| *seen == table).expect("a held table");
                let mut lines = std::mem::take(&mut held[at].1);
                lines.sort_by_key(|(place, _)| *place);
                ordered.extend(lines.into_iter().map(|(_, line)| line));
            }
        }
    }
    ordered
}

/// A number alone on its line, in brackets or not: its figures.
fn lone_number(line: &Line) -> Option<&str> {
    let text = line.text.trim();
    let text = text.strip_prefix('[').and_then(|rest| rest.strip_suffix(']')).map_or(text, str::trim);
    (!text.is_empty() && text.len() <= 4 && text.bytes().all(|byte| byte.is_ascii_digit())).then_some(text)
}
