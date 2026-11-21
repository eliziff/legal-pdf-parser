//! What a line is, the rule that decided it, and the edge of its page a line of furniture stands at.

/// What a line is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Rule {
    #[default]
    None,
    /// A transcript's line number (`split_line_numbers`).
    LineNumber,
    /// The line printing its page's number (`legal_pdf_pages::page_map`).
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
    /// Front matter: a label, a party's role, a jurat, a date, a field, a signature, an author line.
    FrontMatter,
    /// Weighed against the document's short lines (`residual`).
    Residual,
    /// The document's title.
    Title,
    /// Body text: no rule took the line for anything else.
    Body,
}

/// The edge of its page a line of furniture stands at: the running head's or the foot's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Edge {
    Top,
    Bottom,
}
