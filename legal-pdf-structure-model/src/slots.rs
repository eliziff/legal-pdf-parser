//! Each line's slot among the document's lines, and the pages' lines by slot.

use legal_pdf_core::model::{Line, Page};
use std::collections::HashMap;
use std::ops::Range;

/// A line's place among the document's lines: pages in order, each page's lines in page order.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct Slot(pub u32);

impl Slot {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Where each slot stands in the pages: each page's first slot, each slot's page and its line's place among its page's
/// lines, and each line id's slot (the first line's, where two lines share an id).
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Slots {
    /// Each page's first slot, by page slot, and the number of lines after the last.
    starts: Vec<u32>,
    pages: Vec<u32>,
    places: Vec<u32>,
    ids: HashMap<String, Slot>,
}

impl Slots {
    pub fn of(pages: &[Page]) -> Self {
        let mut starts = Vec::with_capacity(pages.len() + 1);
        let mut of_page = Vec::new();
        let mut ids = HashMap::new();
        let mut next = 0;
        for (page_slot, page) in pages.iter().enumerate() {
            starts.push(next);
            for line in &page.lines {
                ids.entry(line.id.clone()).or_insert(Slot(next));
                of_page.push(page_slot as u32);
                next += 1;
            }
        }
        starts.push(next);
        let places = of_page
            .iter()
            .enumerate()
            .map(|(slot, &page)| slot as u32 - starts[page as usize])
            .collect();
        Self {
            starts,
            pages: of_page,
            places,
            ids,
        }
    }

    /// Each page's lines put in `order` (slots, a page's together), each slot keeping its line's new place: the pages are
    /// read in reading order from here on, and a slot still names its line.
    pub fn read_in_order(&mut self, pages: &mut [Page], order: &[Slot]) {
        for (page_slot, page) in pages.iter_mut().enumerate() {
            let range = self.starts[page_slot]..self.starts[page_slot + 1];
            let mut lines: Vec<Option<Line>> = std::mem::take(&mut page.lines)
                .into_iter()
                .map(Some)
                .collect();
            let ordered = &order[range.start as usize..range.end as usize];
            page.lines = ordered
                .iter()
                .filter_map(|slot| lines[self.places[slot.index()] as usize].take())
                .collect();
            for (place, slot) in ordered.iter().enumerate() {
                self.places[slot.index()] = place as u32;
            }
        }
    }

    /// The number of lines.
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// The page slot of the slot's line.
    pub fn page(&self, slot: Slot) -> usize {
        self.pages[slot.index()] as usize
    }

    /// The slots of a page's lines.
    pub fn on(&self, page: usize) -> Range<u32> {
        self.starts[page]..self.starts[page + 1]
    }

    /// The slot of a page's line.
    pub fn at(&self, page: usize, line: usize) -> Slot {
        Slot(self.starts[page] + line as u32)
    }

    /// The slot's line.
    pub fn line<'p>(&self, pages: &'p [Page], slot: Slot) -> &'p Line {
        &pages[self.page(slot)].lines[self.places[slot.index()] as usize]
    }

    /// The slot of the line holding the id.
    pub fn of_id(&self, id: &str) -> Option<Slot> {
        self.ids.get(id).copied()
    }
}

/// The lines of one page that read together: the page, the slot of its first line, and the slots
/// of the lines in the view, in page order.
pub struct PageView<'a> {
    pub page: &'a Page,
    first: u32,
    pub lines: Vec<Slot>,
}

impl<'a> PageView<'a> {
    /// Every page whole, each line's slot its place among all the pages' lines.
    pub fn whole(pages: &'a [Page]) -> Vec<Self> {
        let mut first = 0;
        pages
            .iter()
            .map(|page| {
                let count = page.lines.len() as u32;
                let view = Self {
                    page,
                    first,
                    lines: (first..first + count).map(Slot).collect(),
                };
                first += count;
                view
            })
            .collect()
    }

    /// The lines `lines` names (slots in page order), page by page.
    pub fn of_slots(pages: &'a [Page], slots: &Slots, lines: &[Slot]) -> Vec<Self> {
        let mut views: Vec<Self> = Vec::new();
        for &slot in lines {
            let page_slot = slots.page(slot);
            match views.last_mut() {
                Some(view) if std::ptr::eq(view.page, &pages[page_slot]) => view.lines.push(slot),
                _ => views.push(Self {
                    page: &pages[page_slot],
                    first: slots.on(page_slot).start,
                    lines: vec![slot],
                }),
            }
        }
        views
    }

    /// The slot's line.
    pub fn line(&self, slot: Slot) -> &'a Line {
        &self.page.lines[self.place(slot)]
    }

    /// The place of the slot's line among its page's lines.
    pub fn place(&self, slot: Slot) -> usize {
        (slot.0 - self.first) as usize
    }

    /// The slot of the page's line at `place`.
    pub fn slot(&self, place: usize) -> Slot {
        Slot(self.first + place as u32)
    }

    /// The slot of a line of the view's page, by where it stands in the page's lines.
    pub fn slot_of(&self, line: &Line) -> Slot {
        self.slot(
            (std::ptr::from_ref(line) as usize - self.page.lines.as_ptr() as usize)
                / std::mem::size_of::<Line>(),
        )
    }

    /// The view's lines with their slots, in page order.
    pub fn iter(&self) -> impl Iterator<Item = (Slot, &'a Line)> + '_ {
        self.lines.iter().map(|&slot| (slot, self.line(slot)))
    }
}
