//! A contents list's entry as the fences keep it, the document a list is read on, and what its reading finds.

use crate::fences::Fences;
use crate::{Evidence, Role, Rule, Slot, Slots, SourceTag};
use std::collections::HashSet;

/// One entry of a contents list as the heading outline reads it: its title (the rows it wraps over
/// joined), the locator it prints, where it starts, and the page it is on.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Entry {
    pub title: String,
    pub locator: Option<String>,
    pub left: f64,
    pub page_index: usize,
}

/// What the lists are read against: the fences so far (each line's role: a transcript's line number, the page's
/// furniture, folios), the evidence (the source's tag, a heading already; each line's row), whether a table holds
/// each line, by slot; and each view's page number (the label the PDF declares for it, else its folio's).
pub struct ListInput<'a> {
    pub fences: &'a Fences,
    pub slots: &'a Slots,
    pub evidence: &'a Evidence,
    pub in_table: &'a [bool],
    pub numbers: &'a [Option<u32>],
}

impl ListInput<'_> {
    pub fn line_label(&self, slot: Slot) -> bool {
        self.fences.role[slot.index()] == Role::LineLabel
    }

    /// A print margin's line (a note set sideways, a stamp down the page's edge): no part of a list's row.
    pub fn print_margin(&self, slot: Slot) -> bool {
        self.fences.rule[slot.index()] == Rule::PrintMargin
    }

    pub fn furniture(&self, slot: Slot) -> bool {
        matches!(
            self.fences.role[slot.index()],
            Role::Furniture | Role::PageLabel
        )
    }

    pub fn heading(&self, slot: Slot) -> bool {
        matches!(
            self.evidence.tag[slot.index()],
            SourceTag::Heading(_) | SourceTag::Title | SourceTag::ParagraphTitle
        )
    }

    /// A transcript's line number or the page's furniture: no entry of a list.
    pub fn set_aside(&self, slot: Slot) -> bool {
        self.line_label(slot) || self.furniture(slot)
    }
}

/// The lists a document prints: its contents (and word index), the pages wholly contents, the
/// entries of its contents lists, its tables of authorities, and the rows naming a list above it.
#[derive(Default)]
pub struct Lists {
    pub contents: Vec<Slot>,
    pub pages: HashSet<usize>,
    pub entries: Vec<Entry>,
    /// A list whose entries are mostly citations: the authorities a factum or a judgment relies on.
    pub authorities: Vec<Slot>,
    /// The row naming a list above its entries ("TABLE OF CONTENTS"): the list's title.
    pub titles: Vec<Slot>,
}
