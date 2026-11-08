//! The evidence: what measuring reads of each page, line and the document before anything decides what the lines are
//! (`Evidence`), and the flags it reads them with.

use crate::{Slot, Slots};
use legal_pdf_core::model::Line;
use legal_pdf_tables::PageItems;

/// A line opening with a note label (`label_prefix`): the label's value (0 for a symbol), where it
/// ends in the line's text (in characters), its type size, whether it is raised or set smaller
/// than its line (`label_is_raised`), and why a note band would read it as no label (`SUPPRESS_*`).
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct NoteLabel {
    pub value: u32,
    pub end: u16,
    pub size: f32,
    pub raised: bool,
    pub suppress: u8,
}

impl NoteLabel {
    /// Whether the label is set as a note's: raised, or set at three quarters of the body's type
    /// or smaller (`label_is_typographic`).
    pub fn typographic(&self, body_size: f64) -> bool {
        self.raised || self.size > 0.0 && f64::from(self.size) <= body_size * 0.75
    }
}

/// At the very foot of its page (the bottom 6%).
pub const SUPPRESS_FOOT: u8 = 1;

/// In the bottom right corner, not a compact note's line: a page number.
pub const SUPPRESS_CORNER: u8 = 2;

/// A number before a comma, not raised: the line's own text ("1991, c. 43").
pub const SUPPRESS_COMMA: u8 = 4;

/// Under a line ending on an enactment's noun, not raised: that reference's number, wrapped.
pub const SUPPRESS_WRAPPED: u8 = 8;

/// A page's rows, lines sharing a baseline band (`contents_lists::row_bands`): each line's row, top
/// row 0 (`u32::MAX` for a line with no text or box, or set on its side); the number of rows; and
/// how many of its first and of its last rows stand apart from the rest by at least one and a half
/// of the page's typical step from row to row (frame rows: where a running head or foot sits
/// outside the edge bands): its first or last row alone, or a stack of two in the page's fifth at
/// that edge ("RSA 2000" over "Section 1 | OIL AND GAS CONSERVATION ACT | Chapter O-6").
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct Rows {
    pub of: Vec<u32>,
    pub count: u32,
    pub frame: [u32; 2],
}

/// The source's own reading of a line, before anything else reads it: a tagged PDF's heading (its
/// level, 0 for H) or list item, or a page layout model's label.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SourceTag {
    #[default]
    None,
    Heading(u8),
    ListItem,
    Title,
    ParagraphTitle,
    Quote,
    Aside,
    Note,
    Number,
    Table,
    Visual,
    Header,
    Footer,
    Other,
}

impl SourceTag {
    /// The tag a line's source puts on it: its structure tree's heading or list item where its region
    /// is plain text, else its region's label.
    pub fn of(line: &Line) -> Self {
        let source = line.region_type.trim();
        if matches!(source, "text" | "body" | "" | "unknown" | "unknown_region") {
            return legal_pdf_core::tagged_heading(line, |span| span.structure_role.as_deref())
                .or_else(|| {
                    legal_pdf_core::tagged_line_role(line, |span| span.structure_role.as_deref().filter(|role| *role == "LI"))
                })
                .map_or(Self::None, Self::read);
        }
        Self::read(source)
    }

    pub fn read(region: &str) -> Self {
        if legal_pdf_core::is_heading_role(region) {
            return Self::Heading(region[1..].parse::<u8>().unwrap_or(0));
        }
        if region == "LI" {
            return Self::ListItem;
        }
        match region.to_ascii_lowercase().as_str() {
            "doc_title" => Self::Title,
            "paragraph_title" | "heading" => Self::ParagraphTitle,
            "block_quote" => Self::Quote,
            "aside_text" => Self::Aside,
            "footnote" => Self::Note,
            "number" => Self::Number,
            "table" | "table_cell" => Self::Table,
            "form" | "figure" | "image" | "chart" | "formula" | "separator" | "visual" => Self::Visual,
            "header" => Self::Header,
            "footer" => Self::Footer,
            _ => Self::Other,
        }
    }

    /// The tag's name, as a structure tree or a layout model names it.
    pub fn name(self) -> &'static str {
        const HEADINGS: [&str; 7] = ["H", "H1", "H2", "H3", "H4", "H5", "H6"];
        match self {
            Self::None => "",
            Self::Heading(level) => HEADINGS[usize::from(level).min(6)],
            Self::ListItem => "LI",
            Self::Title => "doc_title",
            Self::ParagraphTitle => "paragraph_title",
            Self::Quote => "block_quote",
            Self::Aside => "aside_text",
            Self::Note => "footnote",
            Self::Number => "number",
            Self::Table => "table",
            Self::Visual => "figure",
            Self::Header => "header",
            Self::Footer => "footer",
            Self::Other => "other",
        }
    }
}

/// A line's opening marker as the numbering grammar reads it (`numbering_marker`): its family (an
/// index into the families read), its value, the depth of its dotted prefix, where it ends in the
/// line's text (in bytes), and whether it is all the line holds.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Marker {
    pub family: u16,
    pub value: u32,
    pub depth: u8,
    pub end: u16,
    pub only: bool,
}

/// A citation on a line, as the citation grammar reads the window holding the line: its byte range
/// in the line's text, and that of the citation itself without its signal and style; a span the
/// grammar protects (`protected`) is its own citation.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Cited {
    pub slot: Slot,
    pub range: (u32, u32),
    pub core: (u32, u32),
    pub protected: bool,
}

/// A page's measures: its rows, and the note separator rule the extraction found on it.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct PageEvidence {
    pub rows: Rows,
    pub separator: Option<f64>,
}

/// Each line's measures by slot, each page's by page slot, and the document's.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Evidence {
    pub pages: Vec<PageEvidence>,
    /// Each page's words as text items (`page_items`), a transcript's line numbers left out.
    /// Its columns are read on them once its furniture is fenced (`Fences::columns`).
    #[serde(skip)]
    pub items: Vec<PageItems>,
    pub line_number: Vec<bool>,
    pub size: Vec<f64>,
    /// The line's commonest face (an index into `faces`; `u16::MAX` for none) and its size in half
    /// points (`dominant_face`).
    pub face: Vec<u16>,
    pub half_points: Vec<u16>,
    /// `LOOK_*`.
    pub look: Vec<u8>,
    pub tag: Vec<SourceTag>,
    pub marker: Vec<Option<Marker>>,
    /// The line on a marker's row carrying its text, where the marker stands alone on its line.
    pub beside: Vec<Option<Slot>>,
    pub label: Vec<Option<NoteLabel>>,
    /// `REFERENCE_*`.
    pub references: Vec<u8>,
    /// Twelve words or fewer, ending no sentence nor on a comma, semicolon or colon.
    pub title_shaped: Vec<bool>,
    /// `WORD_*`.
    pub words: Vec<u32>,
    /// Every line's citations, by slot (`cited` holds where each slot's begin).
    pub citations: Vec<Cited>,
    pub cited: Vec<u32>,
    pub faces: Vec<String>,
    pub families: Vec<String>,
    /// The document's bookmarks outline its divisions (`bookmarks::usable`) and every level of them
    /// (`bookmarks::complete`).
    pub outline_usable: bool,
    pub outline_complete: bool,
    /// An instrument: its operative words (`OPERATIVE`) six in a thousand of its words or more, over
    /// two hundred words; or its caller's profile says so.
    pub instrument: bool,
    /// The operative words in a thousand of the document's words (0 under two hundred words), what
    /// `instrument` thresholds; a thousand where the caller's profile says instrument.
    pub operative: f32,
}

/// A reading of a line's opening marker by the numbering grammar: its family, the value it counts and
/// its dotted prefix; a number after a division word ("PART II") reads under the word.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MarkerReading {
    pub family: String,
    pub value: u32,
    pub prefix: Vec<u32>,
}

/// A span set as a superscript.
pub const REFERENCE_SUPERSCRIPT: u8 = 1;

/// A reference the extraction read apart from its line.
pub const REFERENCE_DETACHED: u8 = 2;

/// Figures set smaller than their line, or raised a quarter of its height.
pub const REFERENCE_RAISED: u8 = 4;

/// Figures run on after a word's punctuation ("licensing.59").
pub const REFERENCE_GLUED: u8 = 8;

/// Set bold (half its characters or more).
pub const LOOK_BOLD: u8 = 1;

/// Set in italic (over half its characters).
pub const LOOK_ITALIC: u8 = 2;

/// In capitals.
pub const LOOK_CAPS: u8 = 4;

/// Most of its words underlined.
pub const LOOK_UNDERLINED: u8 = 8;

/// A label never a heading (`rules::label`): counsel, a party's or signatory's role, a recital's
/// "WHEREAS", contact details, a placeholder.
pub const WORD_LABEL: u32 = 1;

/// A party's role set as a label (`party_label`).
pub const WORD_PARTY: u32 = 1 << 1;

/// An affidavit's jurat ("Sworn before me").
pub const WORD_JURAT: u32 = 1 << 2;

/// A date alone (`date_line`).
pub const WORD_DATE: u32 = 1 << 3;

/// A caption's labelled field (`captions::field`).
pub const WORD_FIELD: u32 = 1 << 4;

/// Where a caption's counsel block or reasons begin (`captions::closing`).
pub const WORD_CLOSING: u32 = 1 << 5;

/// A judgment's first numbered paragraph (`captions::reasons`).
pub const WORD_REASONS: u32 = 1 << 6;

/// A line naming the case by its citation (`captions::citation_title`), on the first two pages.
pub const WORD_CITATION_TITLE: u32 = 1 << 7;

/// A judge's signature or a release line (`judgment::signature_line`).
pub const WORD_SIGNATURE: u32 = 1 << 8;

/// The line naming who gives the reasons (`judgment::author_line`).
pub const WORD_AUTHOR: u32 = 1 << 9;

/// A witness sworn or affirmed (`rules::witness`).
pub const WORD_WITNESS: u32 = 1 << 10;

/// A witness's examination (`rules::examination`).
pub const WORD_EXAMINATION: u32 = 1 << 11;

/// A transcript's index or certificate (`rules::transcript_part`).
pub const WORD_TRANSCRIPT_PART: u32 = 1 << 12;

/// A hearing's part in capitals (`rules::hearing_part`).
pub const WORD_HEARING_PART: u32 = 1 << 13;

/// A transcript's turn or procedural event (`rules::turn_or_event`).
pub const WORD_TURN: u32 = 1 << 14;

/// An index's entry, ending on its page (`rules::index_entry`).
pub const WORD_INDEX_ENTRY: u32 = 1 << 15;

/// A speaker's label opening the line (`LayoutAnalysis::speaker_label`).
pub const WORD_SPEAKER: u32 = 1 << 16;

/// A notes heading (`endnote_heading`).
pub const WORD_NOTES_HEADING: u32 = 1 << 17;

/// A heading starting a new part no endnote sequence runs across (`structural_reset_heading`).
pub const WORD_RESET_HEADING: u32 = 1 << 18;

/// A contents list's title (`contents_lists::contents_title`).
pub const WORD_CONTENTS_TITLE: u32 = 1 << 19;

/// A word index's letter (`contents_lists::index_letter`).
pub const WORD_INDEX_LETTER: u32 = 1 << 20;

/// A statute's history note (`StructureAnalysis::history_note`).
pub const WORD_HISTORY: u32 = 1 << 21;

/// A division's word and its numeral, letter or roman numeral opening the line ("PART IV",
/// "ARTICLE 5", "SCHEDULE A", "CHAPITRE 2": `division_word`). A section's number is no division's.
pub const WORD_DIVISION: u32 = 1 << 22;

/// A court's name and nothing else (`court_name`): "IN THE SUPREME COURT OF THE NORTHWEST
/// TERRITORIES", "Federal Court", "COUR SUPÉRIEURE".
pub const WORD_COURT: u32 = 1 << 23;

impl Evidence {
    /// The slot's citations.
    pub fn citations_of(&self, slot: Slot) -> &[Cited] {
        let (start, end) = (self.cited[slot.index()] as usize, self.cited[slot.index() + 1] as usize);
        &self.citations[start..end]
    }

    /// The slot's row on its page (`u32::MAX` for none).
    pub fn row(&self, slots: &Slots, slot: Slot) -> u32 {
        let page = slots.page(slot);
        self.pages[page].rows.of[(slot.0 - slots.on(page).start) as usize]
    }
}
