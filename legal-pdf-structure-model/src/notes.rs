//! A note's kind, and the document as a note band is read on it.

use crate::fences::Fences;
use crate::{Evidence, Role, Slots, SourceTag};

/// A note's kind: at its page's foot, or among the notes collected at a part's or the document's end.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NoteKind {
    Footnote,
    Endnote,
}

/// What the band reads a document's lines by: the pages' slots, the evidence (each line's type size, note label and
/// source tag), the fences (each line's role, cell, held paragraph and margin number; each page's separator, columns and
/// whether it is a contents page), which counted runs' members open a unit in their document's body type (`unit`, by
/// slot), and the document's body type size, against which notes are set small (0 where unknown: each page then reads
/// its own).
#[derive(Clone, Copy)]
pub struct Document<'a> {
    pub slots: &'a Slots,
    pub evidence: &'a Evidence,
    pub fences: &'a Fences,
    pub unit: &'a [bool],
    pub body_size: f64,
}

impl Document<'_> {
    /// A line the band never reads: a transcript's line number, a contents list's line.
    pub fn excluded(&self, at: usize) -> bool {
        matches!(self.fences.role[at], Role::LineLabel | Role::Contents)
    }
    /// A running head or foot, or a folio.
    pub fn furniture(&self, at: usize) -> bool {
        matches!(self.fences.role[at], Role::Furniture | Role::PageLabel)
    }
    /// A table's cell: body, its number no note's label.
    pub fn cell(&self, at: usize) -> bool {
        self.fences.cell[at] != 0
    }
    /// A numbered paragraph's number, unless its page shows it a note's (`paragraph_labels`): a counted run's member
    /// opening a unit, a paragraph's held number, a margin's number.
    pub fn paragraph(&self, at: usize) -> bool {
        self.unit[at]
            || !self.fences.paragraphs.held[at].is_empty()
            || self.fences.margin_number[at].is_some()
    }
    /// Labelled a note by its source.
    pub fn noted(&self, at: usize) -> bool {
        self.evidence.tag[at] == SourceTag::Note
    }
    /// A heading by its fence or its source.
    pub fn headed(&self, at: usize) -> bool {
        self.fences.role[at] == Role::Heading
            || matches!(
                self.evidence.tag[at],
                SourceTag::Heading(_) | SourceTag::Title | SourceTag::ParagraphTitle
            )
    }
}
