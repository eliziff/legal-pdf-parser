//! The regions a line may be in: the gold's (`labels::REGIONS`), the region a decided role is, and the name a page gives
//! a line of a region. The one vocabulary the decision's writer and DOC read.

use crate::model::{Edge, Role};
use legal_pdf_core::model::{LineZone, NoteMode};

/// The regions a line may be in, the gold's (`labels::REGIONS`).
pub const REGIONS: [&str; 17] = [
    "none", "text", "list_item", "heading", "quote", "note", "table", "contents", "caption", "headnote", "marginal_note",
    "furniture", "page_label", "line_label", "title", "note_quote", "provision",
];
pub(crate) const TEXT: u8 = 1;
pub(crate) const LIST_ITEM: u8 = 2;
pub(crate) const HEADING: u8 = 3;
pub(crate) const QUOTE: u8 = 4;
pub(crate) const CONTENTS: u8 = 7;
pub(crate) const NOTE: u8 = 5;
pub(crate) const TABLE: u8 = 6;
pub(crate) const MARGINAL_NOTE: u8 = 10;
pub(crate) const FURNITURE: u8 = 11;
pub(crate) const PAGE_LABEL: u8 = 12;
pub(crate) const LINE_LABEL: u8 = 13;
pub(crate) const TITLE: u8 = 14;
pub(crate) const NOTE_QUOTE: u8 = 15;
pub(crate) const PROVISION: u8 = 16;

/// The region a role is, where the line opens the unit `kind`.
pub(crate) fn region_of(role: Role, kind: Option<legal_structure_model::UnitRole>) -> u8 {
    use legal_structure_model::UnitRole;
    match role {
        Role::LineLabel => LINE_LABEL,
        Role::PageLabel => PAGE_LABEL,
        Role::Furniture => FURNITURE,
        Role::Contents => CONTENTS,
        Role::Table => TABLE,
        Role::Note => NOTE,
        Role::Heading => HEADING,
        Role::Title => TITLE,
        Role::MarginalNote => MARGINAL_NOTE,
        Role::ListItem => LIST_ITEM,
        Role::Quote => QUOTE,
        Role::Open | Role::Body => match kind {
            Some(UnitRole::Section) => PROVISION,
            Some(UnitRole::Item) => LIST_ITEM,
            _ => TEXT,
        },
    }
}

/// Where a line of `role` stands on its page: a page's head or foot by its edge, a note in a band of `note`'s kind, a
/// heading or title, a marginal note or a line label as itself, the rest the body.
pub(crate) fn zone_of(role: Role, edge: Option<Edge>, note: NoteMode) -> LineZone {
    match region_of(role, None) {
        LINE_LABEL => LineZone::LineLabel,
        PAGE_LABEL | FURNITURE if edge == Some(Edge::Bottom) => LineZone::Footer,
        PAGE_LABEL | FURNITURE => LineZone::Header,
        NOTE => LineZone::Note(Some(note)),
        HEADING | TITLE => LineZone::Heading,
        MARGINAL_NOTE => LineZone::MarginalNote,
        _ => LineZone::Body,
    }
}

/// Regions whose units run on in one stream, as the gold's do (`labels::stream`): the body's, notes',
/// marginal notes', furniture's, page and line labels'.
pub fn stream_of(region: u8) -> usize {
    match REGIONS[usize::from(region)] {
        "note" | "note_quote" => 1,
        "marginal_note" => 2,
        "furniture" => 3,
        "page_label" => 4,
        "line_label" => 5,
        _ => 0,
    }
}
