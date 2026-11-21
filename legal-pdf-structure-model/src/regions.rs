//! The regions a line may be in: the gold's (`labels::REGIONS`), the region a decided role is, and the name a page gives
//! a line of a region. The one vocabulary the decision's writer and DOC read.

use crate::{Edge, Role};
use legal_pdf_core::model::{LineZone, NoteMode};

/// The regions a line may be in, the gold's (`labels::REGIONS`).
pub const REGIONS: [&str; 17] = [
    "none",
    "text",
    "list_item",
    "heading",
    "quote",
    "note",
    "table",
    "contents",
    "caption",
    "headnote",
    "marginal_note",
    "furniture",
    "page_label",
    "line_label",
    "title",
    "note_quote",
    "provision",
];
pub const TEXT: u8 = 1;
pub const LIST_ITEM: u8 = 2;
pub const HEADING: u8 = 3;
pub const QUOTE: u8 = 4;
pub const CONTENTS: u8 = 7;
pub const NOTE: u8 = 5;
pub const TABLE: u8 = 6;
pub const MARGINAL_NOTE: u8 = 10;
pub const FURNITURE: u8 = 11;
pub const PAGE_LABEL: u8 = 12;
pub const LINE_LABEL: u8 = 13;
pub const TITLE: u8 = 14;
pub const NOTE_QUOTE: u8 = 15;
pub const PROVISION: u8 = 16;

/// The region a role is, where the line opens the unit `kind`.
pub fn region_of(role: Role, kind: Option<legal_structure_model::UnitRole>) -> u8 {
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

/// The role a region corresponds to, keeping `prior` when the region is plain body text or unlabelled.
pub fn role_of_region(region: u8, prior: Role) -> Role {
    match region {
        LINE_LABEL => Role::LineLabel,
        PAGE_LABEL => Role::PageLabel,
        FURNITURE => Role::Furniture,
        CONTENTS => Role::Contents,
        TABLE => Role::Table,
        NOTE | NOTE_QUOTE => Role::Note,
        HEADING => Role::Heading,
        TITLE => Role::Title,
        MARGINAL_NOTE => Role::MarginalNote,
        LIST_ITEM => Role::ListItem,
        QUOTE => Role::Quote,
        PROVISION | TEXT => Role::Body,
        _ => match REGIONS.get(usize::from(region)).copied() {
            Some("caption" | "headnote") => Role::Body,
            _ => prior,
        },
    }
}

/// Where a line of `role` stands on its page: a page's head or foot by its edge, a note in a band of `note`'s kind, a
/// heading or title, a marginal note or a line label as itself, the rest the body.
pub fn zone_of(role: Role, edge: Option<Edge>, note: NoteMode) -> LineZone {
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
