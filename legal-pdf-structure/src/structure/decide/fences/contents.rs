//! The contents lists, a table of authorities' lines left to the body, and the pages wholly contents.
use super::*;

impl Fences {
    /// The contents lists (`contents_lists::read`), pdf-inspector's contents among them, each from
    /// its first entry to its last, a table of authorities' lines aside; the row naming a list, its
    /// heading; the pages wholly contents. Returns the lines the lists read as contents (a table
    /// more than half of whose lines they hold is that list).
    pub(super) fn read_contents(
        &mut self,
        analysis: &dyn StructureAnalysis,
        pages: &[Page],
        slots: &Slots,
        evidence: &Evidence,
        views: &[PageView],
        in_table: &[bool],
        listed: &HashSet<String>,
    ) -> Vec<bool> {
        // Each page's number: the label the PDF declares, else its folio's, else its place by the
        // folios' offset.
        let numbers: Vec<Option<u32>> = pages
            .iter()
            .zip(&self.folios)
            .map(|(page, folio)| {
                evidence
                    .page_labels
                    .get(page.index)
                    .and_then(Option::as_deref)
                    .and_then(contents_lists::page_numeral)
                    .or_else(|| folio.as_ref().and_then(|(_, label)| contents_lists::page_numeral(label)))
                    .or_else(|| {
                        self.folio_offset
                            .and_then(|offset| u32::try_from(i64::from(page.number) + offset).ok())
                            .filter(|number| *number > 0)
                    })
            })
            .collect();
        let lists = legal_pdf_support::profile::measure("classify.contents_lists", || {
            let role = &self.role;
            contents_lists::read(
                views,
                &contents_lists::ListInput {
                    analysis,
                    line_label: &|slot| role[slot.index()] == Role::LineLabel,
                    furniture: &|slot| matches!(role[slot.index()], Role::Furniture | Role::PageLabel),
                    heading: &|slot| matches!(evidence.tag[slot.index()], SourceTag::Heading(_) | SourceTag::Title | SourceTag::ParagraphTitle),
                    claimed: &|slot| in_table[slot.index()],
                    numbers: &numbers,
                },
            )
        });
        let mut listed_contents = vec![false; slots.len()];
        for &slot in &lists.contents {
            listed_contents[slot.index()] = true;
        }
        // A table of authorities is no contents list, whatever its leaders and locators: its lines
        // are the body's, as its pages are no contents pages.
        let mut authority_pages = vec![false; pages.len()];
        let mut authority = vec![false; slots.len()];
        for &slot in &lists.authorities {
            authority[slot.index()] = true;
            authority_pages[slots.page(slot)] = true;
        }
        for slot in listed.iter().filter_map(|id| slots.of_id(id)).chain(lists.contents.iter().copied()) {
            if !authority[slot.index()] {
                self.set(slot, Role::Contents, Rule::Contents);
            }
        }
        for &slot in &lists.titles {
            self.set(slot, Role::Heading, Rule::ContentsTitle);
        }
        self.contents_pages = pages
            .iter()
            .enumerate()
            .map(|(page_slot, page)| lists.pages.contains(&page.index) && !authority_pages[page_slot])
            .collect();
        self.contents_entries = lists.entries;
        listed_contents
    }
}
