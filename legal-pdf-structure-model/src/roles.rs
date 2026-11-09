//! What a line is, the rule that decided it, and the edge of its page a line of furniture stands at.

/// What a line is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub enum Role {
    /// Nothing has decided it yet.
    #[default]
    Open,
    LineLabel,
    PageLabel,
    Furniture,
    Contents,
    Table,
    Note,
    Heading,
    Title,
    MarginalNote,
    ListItem,
    Quote,
    Body,
}

/// The rule that gave a line its role.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub enum Rule {
    #[default]
    None,
    /// A transcript's line number (`split_line_numbers`).
    LineNumber,
    /// A folio (`edge_folios`, `alternating_folios`), or the line printing its page's label.
    Folio,
    /// A running head or foot repeated at the edge bands, or a short document's edge.
    RunningHead,
    /// A running head or foot repeated on frame rows outside the edge bands.
    FrameRow,
    /// A line on the row of a running head or folio.
    RowMate,
    /// A print driver's margin (`print_margins`).
    PrintMargin,
    /// A document management identifier (`document_id_line`).
    DocumentId,
    /// A page number at the edge (`footer_page_number`).
    PageNumber,
    /// A statute's running head naming its page's sections (`section_heads`), borne out by its spine.
    SectionHead,
    /// A contents list's or word index's line (`contents_lists::read`, pdf-inspector's contents).
    Contents,
    /// The row naming a contents list.
    ContentsTitle,
    /// A cell of a table (`document_tables`).
    Grid,
    /// A line of the note band (`Notes`).
    NoteBand,
    /// A notes heading over the band ("Notes", "Footnotes").
    NotesHeading,
    /// An endnote page's running head.
    EndnoteHead,
    /// A transcript's line beside its number column going on with the line before: no heading.
    TranscriptLine,
    /// A transcript's part, witness, examination or index letter, opening a turn or a sentence.
    TranscriptPart,
    /// A dash and an enactment's history, wholly: the heading over its provision.
    HistoryHeading,
    /// An enactment's history closing its provision: no heading.
    HistoryLine,
    /// A quotation's own number: none of the document's units.
    QuotedNumber,
    /// A counted marker carrying prose: the unit it opens.
    Opener,
    /// An instrument's numbered caption: the first line of the clause it opens.
    Caption,
    /// A division's word and number ("PART IV", "ARTICLE 8").
    DivisionHeading,
    /// A count of numbered titles between the units: a heading ladder.
    Ladder,
    /// A count of titles alone, one under the other: contents.
    TitleOnly,
    /// A statute's note beside or over the one provision it names.
    MarginalNote,
    /// A line the document's bookmarks or contents list name.
    Anchor,
    /// A line after the opening of bookmarks naming every heading, that they do not name.
    Unnamed,
    /// Front matter: a label, a party's role, a jurat, a date, a field, a signature, an author line.
    FrontMatter,
    /// Weighed against the document's short lines (`residual`).
    Residual,
    /// The document's title.
    Title,
    /// Body text: no rule took the line for anything else.
    Body,
}

impl Rule {
    /// The rule's name, as a line's trail and the diagnostics name it.
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::LineNumber => "line_number",
            Self::Folio => "folio",
            Self::RunningHead => "running_head",
            Self::FrameRow => "frame_row",
            Self::RowMate => "row_mate",
            Self::PrintMargin => "print_margin",
            Self::DocumentId => "document_id",
            Self::PageNumber => "page_number",
            Self::SectionHead => "section_head",
            Self::Contents => "contents",
            Self::ContentsTitle => "contents_title",
            Self::Grid => "grid",
            Self::NoteBand => "note_band",
            Self::NotesHeading => "notes_heading",
            Self::EndnoteHead => "endnote_head",
            Self::TranscriptLine => "transcript_line",
            Self::TranscriptPart => "transcript_part",
            Self::HistoryHeading => "history_heading",
            Self::HistoryLine => "history_line",
            Self::QuotedNumber => "quoted_number",
            Self::Opener => "opener",
            Self::Caption => "caption",
            Self::DivisionHeading => "division_heading",
            Self::Ladder => "ladder",
            Self::TitleOnly => "title_only",
            Self::MarginalNote => "marginal_note",
            Self::Anchor => "anchor",
            Self::Unnamed => "unnamed",
            Self::FrontMatter => "front_matter",
            Self::Residual => "residual",
            Self::Title => "title",
            Self::Body => "body",
        }
    }
}

/// The edge of its page a line of furniture stands at: the running head's or the foot's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum Edge {
    Top,
    Bottom,
}
