//! A document's lines measured before anything decides what they are. Normalizing parts a
//! transcript's line numbers from the lines the extraction read them into and clears the
//! underlines a table's rulings drew under its words; each line then has its slot. Measuring reads
//! each page's words as text items, its rows and note separator; each line's row, type,
//! the source's tag on it, its opening marker and the line beside it carrying the marker's text, its
//! note label, its reference tokens, the lexicon its words match, and its citations; and the
//! document's bookmarks, the page labels it declares and whether it is an instrument.

use super::*;
use legal_pdf_core::{dominant_face, CitationSpan, ITALIC};
use legal_pdf_tables::PageItems;
use std::ops::Range;

/// A line's place among the document's lines: pages in order, each page's lines in page order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub(crate) struct Slot(pub(crate) u32);

impl Slot {
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

/// Where each slot stands in the pages: each page's first slot, each slot's page, and each line
/// id's slot (the first line's, where two lines share an id).
#[derive(Debug, Default)]
pub(crate) struct Slots {
    /// Each page's first slot, by page slot, and the number of lines after the last.
    starts: Vec<u32>,
    pages: Vec<u32>,
    ids: HashMap<String, Slot>,
}

impl Slots {
    pub(crate) fn of(pages: &[Page]) -> Self {
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
        Self { starts, pages: of_page, ids }
    }

    /// The number of lines.
    pub(crate) fn len(&self) -> usize {
        self.pages.len()
    }

    /// The page slot of the slot's line.
    pub(crate) fn page(&self, slot: Slot) -> usize {
        self.pages[slot.index()] as usize
    }

    /// The slots of a page's lines.
    pub(crate) fn on(&self, page: usize) -> Range<u32> {
        self.starts[page]..self.starts[page + 1]
    }

    /// The slot of a page's line.
    pub(crate) fn at(&self, page: usize, line: usize) -> Slot {
        Slot(self.starts[page] + line as u32)
    }

    /// The slot's line.
    pub(crate) fn line<'p>(&self, pages: &'p [Page], slot: Slot) -> &'p Line {
        let page = self.page(slot);
        &pages[page].lines[(slot.0 - self.starts[page]) as usize]
    }

    /// The slot's line, to change.
    pub(crate) fn line_mut<'p>(&self, pages: &'p mut [Page], slot: Slot) -> &'p mut Line {
        let page = self.page(slot);
        &mut pages[page].lines[(slot.0 - self.starts[page]) as usize]
    }

    /// The slot of the line holding the id.
    pub(crate) fn of_id(&self, id: &str) -> Option<Slot> {
        self.ids.get(id).copied()
    }
}

/// The lines of one page that read together: the page, the slot of its first line, and the slots
/// of the lines in the view, in page order.
pub(crate) struct PageView<'a> {
    pub(crate) page: &'a Page,
    first: u32,
    pub(crate) lines: Vec<Slot>,
}

impl<'a> PageView<'a> {
    /// Every page whole, each line's slot its place among all the pages' lines.
    pub(crate) fn whole(pages: &'a [Page]) -> Vec<Self> {
        let mut first = 0;
        pages
            .iter()
            .map(|page| {
                let count = page.lines.len() as u32;
                let view = Self { page, first, lines: (first..first + count).map(Slot).collect() };
                first += count;
                view
            })
            .collect()
    }

    /// The lines `lines` names (slots in page order), page by page.
    pub(crate) fn of_slots(pages: &'a [Page], slots: &Slots, lines: &[Slot]) -> Vec<Self> {
        let mut views: Vec<Self> = Vec::new();
        for &slot in lines {
            let page_slot = slots.page(slot);
            match views.last_mut() {
                Some(view) if std::ptr::eq(view.page, &pages[page_slot]) => view.lines.push(slot),
                _ => views.push(Self { page: &pages[page_slot], first: slots.on(page_slot).start, lines: vec![slot] }),
            }
        }
        views
    }

    /// The slot's line.
    pub(crate) fn line(&self, slot: Slot) -> &'a Line {
        &self.page.lines[self.place(slot)]
    }

    /// The place of the slot's line among its page's lines.
    pub(crate) fn place(&self, slot: Slot) -> usize {
        (slot.0 - self.first) as usize
    }

    /// The slot of the page's line at `place`.
    pub(crate) fn slot(&self, place: usize) -> Slot {
        Slot(self.first + place as u32)
    }

    /// The slot of a line of the view's page, by where it stands in the page's lines.
    pub(crate) fn slot_of(&self, line: &Line) -> Slot {
        let place = (std::ptr::from_ref(line) as usize - self.page.lines.as_ptr() as usize) / std::mem::size_of::<Line>();
        debug_assert!(place < self.page.lines.len(), "a line of another page");
        self.slot(place)
    }

    /// The view's lines with their slots, in page order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (Slot, &'a Line)> + '_ {
        self.lines.iter().map(|&slot| (slot, self.line(slot)))
    }
}

/// A page's rows, lines sharing a baseline band (`contents_lists::row_bands`): each line's row, top
/// row 0 (`u32::MAX` for a line with no text or box, or set on its side); the number of rows; and
/// how many of its first and of its last rows stand apart from the rest by at least one and a half
/// of the page's typical step from row to row (frame rows: where a running head or foot sits
/// outside the edge bands): its first or last row alone, or a stack of two in the page's fifth at
/// that edge ("RSA 2000" over "Section 1 | OIL AND GAS CONSERVATION ACT | Chapter O-6").
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct Rows {
    pub(crate) of: Vec<u32>,
    pub(crate) count: u32,
    pub(crate) frame: [u32; 2],
}

pub(crate) fn page_rows(page: &Page) -> Rows {
    let lines = &page.lines;
    let mut order: Vec<usize> = (0..lines.len())
        .filter(|&slot| {
            let line = &lines[slot];
            !line.text.trim().is_empty()
                && line.bbox.iter().all(|value| value.is_finite())
                && line.bbox[3] - line.bbox[1] <= 3.0 * (line.bbox[2] - line.bbox[0])
        })
        .collect();
    order.sort_by(|&a, &b| lines[a].bbox[1].total_cmp(&lines[b].bbox[1]).then(lines[a].bbox[0].total_cmp(&lines[b].bbox[0])));
    let bands = contents_lists::row_bands(&order, |slot| lines[slot].bbox);
    let mut of = vec![u32::MAX; lines.len()];
    let mut tops = Vec::with_capacity(bands.len());
    for (row, band) in bands.iter().enumerate() {
        for &slot in band {
            of[slot] = row as u32;
        }
        tops.push(band.iter().map(|&slot| lines[slot].bbox[1]).fold(f64::INFINITY, f64::min));
    }
    let steps: Vec<f64> = tops.windows(2).map(|pair| pair[1] - pair[0]).filter(|step| *step > page.height * 0.002).collect();
    let step = if steps.is_empty() { page.height * 0.02 } else { median(steps) };
    let apart = |gap: f64| gap >= 1.5 * step;
    let count = tops.len();
    let top = if count > 1 && apart(tops[1] - tops[0]) {
        1
    } else if count > 3 && apart(tops[2] - tops[1]) && tops[1] < page.height * 0.2 {
        2
    } else {
        0
    };
    let bottom = if count > 1 && apart(tops[count - 1] - tops[count - 2]) {
        1
    } else if count > 3 && apart(tops[count - 2] - tops[count - 3]) && tops[count - 2] > page.height * 0.8 {
        2
    } else {
        0
    };
    Rows { of, count: count as u32, frame: [top, bottom] }
}

/// The source's own reading of a line, before anything else reads it: a tagged PDF's heading (its
/// level, 0 for H) or list item, or a page layout model's label.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum SourceTag {
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
    fn of(line: &Line) -> Self {
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

    fn read(region: &str) -> Self {
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
    pub(crate) fn name(self) -> &'static str {
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
pub(crate) struct Marker {
    pub(crate) family: u16,
    pub(crate) value: u32,
    pub(crate) depth: u8,
    pub(crate) end: u16,
    pub(crate) only: bool,
}

/// A line opening with a note label (`label_prefix`): the label's value (0 for a symbol), where it
/// ends in the line's text (in characters), its type size, whether it is raised or set smaller
/// than its line (`label_is_raised`), and why a note band would read it as no label (`SUPPRESS_*`).
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct NoteLabel {
    pub(crate) value: u32,
    pub(crate) end: u16,
    pub(crate) size: f32,
    pub(crate) raised: bool,
    pub(crate) suppress: u8,
}

impl NoteLabel {
    /// Whether the label is set as a note's: raised, or set at three quarters of the body's type
    /// or smaller (`label_is_typographic`).
    pub(crate) fn typographic(&self, body_size: f64) -> bool {
        self.raised || self.size > 0.0 && f64::from(self.size) <= body_size * 0.75
    }
}

/// At the very foot of its page (the bottom 6%).
pub(crate) const SUPPRESS_FOOT: u8 = 1;
/// In the bottom right corner, not a compact note's line: a page number.
pub(crate) const SUPPRESS_CORNER: u8 = 2;
/// A number before a comma, not raised: the line's own text ("1991, c. 43").
pub(crate) const SUPPRESS_COMMA: u8 = 4;
/// Under a line ending on an enactment's noun, not raised: that reference's number, wrapped.
pub(crate) const SUPPRESS_WRAPPED: u8 = 8;

/// A span set as a superscript.
pub(crate) const REFERENCE_SUPERSCRIPT: u8 = 1;
/// A reference the extraction read apart from its line.
pub(crate) const REFERENCE_DETACHED: u8 = 2;
/// Figures set smaller than their line, or raised a quarter of its height.
pub(crate) const REFERENCE_RAISED: u8 = 4;
/// Figures run on after a word's punctuation ("licensing.59").
pub(crate) const REFERENCE_GLUED: u8 = 8;

/// Set bold (half its characters or more).
pub(crate) const LOOK_BOLD: u8 = 1;
/// Set in italic (over half its characters).
pub(crate) const LOOK_ITALIC: u8 = 2;
/// In capitals.
pub(crate) const LOOK_CAPS: u8 = 4;
/// Most of its words underlined.
pub(crate) const LOOK_UNDERLINED: u8 = 8;

/// A label never a heading (`rules::label`): counsel, a party's or signatory's role, a recital's
/// "WHEREAS", contact details, a placeholder.
pub(crate) const WORD_LABEL: u32 = 1;
/// A party's role set as a label (`party_label`).
pub(crate) const WORD_PARTY: u32 = 1 << 1;
/// An affidavit's jurat ("Sworn before me").
pub(crate) const WORD_JURAT: u32 = 1 << 2;
/// A date alone (`date_line`).
pub(crate) const WORD_DATE: u32 = 1 << 3;
/// A caption's labelled field (`captions::field`).
pub(crate) const WORD_FIELD: u32 = 1 << 4;
/// Where a caption's counsel block or reasons begin (`captions::closing`).
pub(crate) const WORD_CLOSING: u32 = 1 << 5;
/// A judgment's first numbered paragraph (`captions::reasons`).
pub(crate) const WORD_REASONS: u32 = 1 << 6;
/// A line naming the case by its citation (`captions::citation_title`), on the first two pages.
pub(crate) const WORD_CITATION_TITLE: u32 = 1 << 7;
/// A judge's signature or a release line (`judgment::signature_line`).
pub(crate) const WORD_SIGNATURE: u32 = 1 << 8;
/// The line naming who gives the reasons (`judgment::author_line`).
pub(crate) const WORD_AUTHOR: u32 = 1 << 9;
/// A witness sworn or affirmed (`rules::witness`).
pub(crate) const WORD_WITNESS: u32 = 1 << 10;
/// A witness's examination (`rules::examination`).
pub(crate) const WORD_EXAMINATION: u32 = 1 << 11;
/// A transcript's index or certificate (`rules::transcript_part`).
pub(crate) const WORD_TRANSCRIPT_PART: u32 = 1 << 12;
/// A hearing's part in capitals (`rules::hearing_part`).
pub(crate) const WORD_HEARING_PART: u32 = 1 << 13;
/// A transcript's turn or procedural event (`rules::turn_or_event`).
pub(crate) const WORD_TURN: u32 = 1 << 14;
/// An index's entry, ending on its page (`rules::index_entry`).
pub(crate) const WORD_INDEX_ENTRY: u32 = 1 << 15;
/// A speaker's label opening the line (`transcript::speaker_label`).
pub(crate) const WORD_SPEAKER: u32 = 1 << 16;
/// A notes heading (`endnote_heading`).
pub(crate) const WORD_NOTES_HEADING: u32 = 1 << 17;
/// A heading starting a new part no endnote sequence runs across (`structural_reset_heading`).
pub(crate) const WORD_RESET_HEADING: u32 = 1 << 18;
/// A contents list's title (`contents_lists::contents_title`).
pub(crate) const WORD_CONTENTS_TITLE: u32 = 1 << 19;
/// A word index's letter (`contents_lists::index_letter`).
pub(crate) const WORD_INDEX_LETTER: u32 = 1 << 20;
/// A statute's history note (`statute::history_note`).
pub(crate) const WORD_HISTORY: u32 = 1 << 21;
/// A division's word and its numeral, letter or roman numeral opening the line ("PART IV",
/// "ARTICLE 5", "SCHEDULE A", "CHAPITRE 2": `division_word`). A section's number is no division's.
pub(crate) const WORD_DIVISION: u32 = 1 << 22;
/// A court's name and nothing else (`court_name`): "IN THE SUPREME COURT OF THE NORTHWEST
/// TERRITORIES", "Federal Court", "COUR SUPÉRIEURE".
pub(crate) const WORD_COURT: u32 = 1 << 23;

/// A citation on a line, as the citation grammar reads the window holding the line: its byte range
/// in the line's text, and that of the citation itself without its signal and style; a span the
/// grammar protects (`protected`) is its own citation.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct Cited {
    pub(crate) slot: Slot,
    pub(crate) range: (u32, u32),
    pub(crate) core: (u32, u32),
    pub(crate) protected: bool,
}

/// A page's measures: its rows, and the note separator rule the extraction found on it.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct PageEvidence {
    pub(crate) rows: Rows,
    pub(crate) separator: Option<f64>,
}

/// Each line's measures by slot, each page's by page slot, and the document's.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct Evidence {
    pub(crate) pages: Vec<PageEvidence>,
    /// Each page's words as text items (`page_items`), a transcript's line numbers left out.
    /// Its columns are read on them once its furniture is fenced (`Fences::columns`).
    #[serde(skip)]
    pub(crate) items: Vec<PageItems>,
    pub(crate) line_number: Vec<bool>,
    pub(crate) size: Vec<f64>,
    /// The line's commonest face (an index into `faces`; `u16::MAX` for none) and its size in half
    /// points (`dominant_face`).
    pub(crate) face: Vec<u16>,
    pub(crate) half_points: Vec<u16>,
    /// `LOOK_*`.
    pub(crate) look: Vec<u8>,
    /// The stroke weight measured on a recognized page's image (`Line::weight`).
    pub(crate) weight: Vec<f32>,
    pub(crate) tag: Vec<SourceTag>,
    pub(crate) marker: Vec<Option<Marker>>,
    /// The line on a marker's row carrying its text, where the marker stands alone on its line.
    pub(crate) beside: Vec<Option<Slot>>,
    pub(crate) label: Vec<Option<NoteLabel>>,
    /// `REFERENCE_*`.
    pub(crate) references: Vec<u8>,
    /// Twelve words or fewer, ending no sentence nor on a comma, semicolon or colon.
    pub(crate) title_shaped: Vec<bool>,
    /// `WORD_*`.
    pub(crate) words: Vec<u32>,
    /// Every line's citations, by slot (`cited` holds where each slot's begin).
    pub(crate) citations: Vec<Cited>,
    pub(crate) cited: Vec<u32>,
    pub(crate) faces: Vec<String>,
    pub(crate) families: Vec<String>,
    /// The document's bookmarks outline its divisions (`bookmarks::usable`) and every level of them
    /// (`bookmarks::complete`).
    pub(crate) outline_usable: bool,
    pub(crate) outline_complete: bool,
    /// The page labels the PDF declares (`/PageLabels`), by page index.
    pub(crate) page_labels: Vec<Option<String>>,
    /// An instrument: its operative words (`OPERATIVE`) six in a thousand of its words or more, over
    /// two hundred words; or its caller's profile says so.
    pub(crate) instrument: bool,
}

/// What normalizing leaves to be measured: which lines are a transcript's line numbers, by slot,
/// and each page's words as text items.
pub(crate) struct Normalized {
    pub(crate) line_numbers: Vec<bool>,
    pub(crate) items: Vec<PageItems>,
}

/// The pages normalized, the only change to their text before the runs are read: a transcript's
/// line numbers parted from the lines they were read into and held out of the body
/// (`split_line_numbers`, so no page's items offer them), a table's rulings cleared from under its
/// words (`clear_table_underlines`), and every line's slot.
pub(crate) fn normalize(pages: &mut [Page]) -> (Slots, Normalized) {
    let numbers = graph::split_line_numbers(pages);
    let mut items: Vec<PageItems> = pages.iter().map(legal_pdf_tables::page_items).collect();
    for (page, items) in pages.iter_mut().zip(&mut items) {
        legal_pdf_tables::clear_table_underlines(page, items);
    }
    let slots = Slots::of(pages);
    let mut line_numbers = vec![false; slots.len()];
    for (page, numbers) in numbers.into_iter().enumerate() {
        for line in numbers {
            line_numbers[slots.at(page, line).index()] = true;
        }
    }
    (slots, Normalized { line_numbers, items })
}

/// Indices for strings read again and again (faces, marker families).
#[derive(Default)]
struct Names {
    ids: HashMap<String, u16>,
    names: Vec<String>,
}

impl Names {
    fn id(&mut self, name: &str) -> u16 {
        if let Some(&id) = self.ids.get(name) {
            return id;
        }
        let id = self.names.len().min(usize::from(u16::MAX - 1)) as u16;
        self.ids.insert(name.to_owned(), id);
        self.names.push(name.to_owned());
        id
    }
}

/// An affidavit's jurat: "Sworn before me", "AFFIRMED BEFORE ME at Toronto".
fn jurat(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\b(?:sworn|affirmed|declared)\b.{0,40}\bbefore\s+me\b").unwrap())
        .is_match(text)
}

/// A line opening with a division's word and number, as the heading ladder reads one
/// (`rules::enumerator`): a part, an article, a schedule, a division or a chapter, in English or
/// French.
fn division_word(text: &str) -> bool {
    headings::rules::enumerator(text).is_some_and(|(_, word)| {
        matches!(word.as_str(), "part" | "partie" | "article" | "schedule" | "annexe" | "division" | "chapter" | "chapitre")
    })
}

/// A court's name and nothing else, as a caption's header prints it: "IN THE COURT OF APPEAL FOR
/// SASKATCHEWAN", "THE CONSTITUTIONAL COURT OF SOUTH AFRICA", "ONTARIO SUPERIOR COURT OF JUSTICE",
/// "SUPERIOR COURT OF JUSTICE - ONTARIO", "Cour d'appel du Québec". It opens in capitals, each
/// word of the name is capitalized but for its connectives, and the words before the court's are
/// its name's ("ON APPEAL FROM THE COURT OF APPEAL", "JUDGES OF THE COURT" name none); a line going
/// on after the name ("THE COURT ORDERS:", "the Court of Appeal's reasons", "Court of Appeal,"), a
/// statute's title ("THE COURT OF KING'S BENCH ACT") and the word alone name none.
fn court_name(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    const NAME: &str = r"[\p{Lu}(][\p{L}'’.&,()-]*";
    const JOIN: &str = r"(?i:of|for|the|and|de|du|des|la|le|d'|d’|l'|l’)";
    const FRENCH: &str = r"(?i:sup[ée]rieure|supr[êe]me|f[ée]d[ée]rale|provinciale|municipale|d['’]appel)";
    let text = text.trim();
    let lower = text.to_lowercase();
    let before = lower.split("cour").next().unwrap_or("");
    text.starts_with(char::is_uppercase)
        && text.split_whitespace().count() >= 2
        && !text.ends_with([',', ';', ':'])
        && !before.split_whitespace().any(|word| matches!(word, "from" | "on" | "to" | "by" | "of" | "at" | "and" | "et"))
        && !lower.split_whitespace().next_back().is_some_and(|word| matches!(word, "act" | "rules" | "regulations" | "loi"))
        && RE
            .get_or_init(|| {
                Regex::new(&format!(
                    r"^(?:(?:IN|In)\s+)?(?:(?:THE|The)\s+)?(?:{NAME}\s+){{0,5}}(?:COURT|Court)(?:\s+(?i:of|for)\s+(?:{JOIN}\s+|{NAME}\s*)+)?(?:\s*[-–—]\s*{NAME}(?:\s+{NAME})*)?\s*$|^(?:(?:LA|La)\s+)?(?:COUR|Cour)(?:\s+(?:{FRENCH}|{JOIN}|{NAME}))+\s*$"
                ))
                .unwrap()
            })
            .is_match(text)
}

/// The lexicon a line's words match (`WORD_*`), the citation title aside.
fn lexicon(line: &Line) -> u32 {
    let text = line.text.trim();
    [
        (WORD_LABEL, headings::rules::label(text)),
        (WORD_PARTY, party_label().is_match(text)),
        (WORD_JURAT, jurat(text)),
        (WORD_DATE, date_line(text)),
        (WORD_FIELD, captions::field(&line.text).is_some()),
        (WORD_CLOSING, captions::closing(&line.text)),
        (WORD_REASONS, captions::reasons(line)),
        (WORD_SIGNATURE, legal_structure_model::judgment::signature_line(text)),
        (WORD_AUTHOR, legal_structure_model::judgment::author_line(text)),
        (WORD_WITNESS, headings::rules::witness(text)),
        (WORD_EXAMINATION, headings::rules::examination(text)),
        (WORD_TRANSCRIPT_PART, headings::rules::transcript_part(text)),
        (WORD_HEARING_PART, headings::rules::hearing_part(text)),
        (WORD_TURN, headings::rules::turn_or_event(text)),
        (WORD_INDEX_ENTRY, headings::rules::index_entry(text)),
        (WORD_SPEAKER, legal_structure_model::transcript::speaker_label(text).is_some()),
        (WORD_NOTES_HEADING, endnote_heading(text)),
        (WORD_RESET_HEADING, structural_reset_heading(text)),
        (WORD_CONTENTS_TITLE, contents_lists::contents_title(text)),
        (WORD_INDEX_LETTER, contents_lists::index_letter(text)),
        (WORD_HISTORY, legal_structure_model::statute::history_note(text)),
        (WORD_DIVISION, division_word(text)),
        (WORD_COURT, court_name(text)),
    ]
    .into_iter()
    .filter(|(_, holds)| *holds)
    .fold(0, |bits, (bit, _)| bits | bit)
}

/// Figures run on after a word's punctuation: "licensing.59", "Act;12". A decimal ("59.5") and a
/// one-letter abbreviation's number ("p.12") are none.
fn glued_reference(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\p{L}{2}[.,;:)\]\u{201d}\u{2019}]\d{1,3}(?:\s|$)").unwrap())
        .is_match(text)
}

/// A line's reference tokens (`REFERENCE_*`), its type set at `size`.
fn reference_tokens(line: &Line, size: f64) -> u8 {
    let mut found = 0;
    if !line.detached_references.is_empty() {
        found |= REFERENCE_DETACHED;
    }
    let height = line.bbox[3] - line.bbox[1];
    for span in &line.spans {
        let text = span.text.trim();
        if text.is_empty() || !text.chars().all(|c| c.is_ascii_digit() || is_note_symbol(c)) {
            continue;
        }
        if span.superscript {
            found |= REFERENCE_SUPERSCRIPT;
        } else if (size > 0.0 && span.size > 0.0 && span.size * 1.25 <= size)
            || (height > 0.0 && span.bbox[3] <= line.bbox[3] - 0.25 * height)
        {
            found |= REFERENCE_RAISED;
        }
    }
    if glued_reference(&line.text) {
        found |= REFERENCE_GLUED;
    }
    found
}

/// The line's opening marker (`Marker`), its family named in `families`.
fn opening_marker(line: &Line, families: &mut Names) -> Option<Marker> {
    let text = line.text.trim_start();
    let token = text.split_whitespace().next()?;
    if token.chars().count() > 12 {
        return None;
    }
    let reading = layout_analysis().numbering_marker(token, None)?;
    let end = line.text.len() - text.len() + token.len();
    Some(Marker {
        family: families.id(&reading.family),
        value: reading.ordinal,
        depth: reading.prefix.len().min(usize::from(u8::MAX)) as u8,
        end: end.min(usize::from(u16::MAX)) as u16,
        only: text.trim_end() == token || crate::paragraphs::marker_only_re().is_match(&line.text),
    })
}

/// The page's line at `slot` opening with a note label (`NoteLabel`), the page's lines' type sizes
/// `sizes`.
pub(crate) fn note_label(page: &Page, sizes: &[f64], slot: usize) -> Option<NoteLabel> {
    let line = &page.lines[slot];
    let prefix = label_prefix(&line.text)?;
    let size = sizes[slot];
    let raised = label_is_raised(line, &prefix, size);
    let label_size = label_spans(line, &prefix)
        .filter_map(|span| (span.size > 0.0).then_some(span.size))
        .min_by(f64::total_cmp)
        .unwrap_or(size);
    let mut suppress = 0;
    if line.bbox[1] >= page.height * 0.94 {
        suppress |= SUPPRESS_FOOT;
    }
    if line.bbox[1] >= page.height * 0.91 && line.bbox[0] >= page.width * 0.50 && !compact_note_line(&line.text) {
        suppress |= SUPPRESS_CORNER;
    }
    if line.text.chars().nth(prefix.end) == Some(',') && !raised {
        suppress |= SUPPRESS_COMMA;
    }
    if !raised && reference_wrapped_onto(&page.lines, sizes, slot) {
        suppress |= SUPPRESS_WRAPPED;
    }
    Some(NoteLabel {
        value: normalize_label(&prefix.label).parse().unwrap_or(0),
        end: prefix.end.min(usize::from(u16::MAX)) as u16,
        size: label_size as f32,
        raised,
        suppress,
    })
}

/// An instrument's operative words.
const OPERATIVE: &str = r"(?i)\b(?:shall|hereby|herein|hereto|hereof|hereunder|hereinafter|thereof|whereas|aux présentes|doit|doivent)\b";

/// A text's words (two letters or more), and how many of them are operative (`OPERATIVE`).
fn operative_words(text: &str) -> (usize, usize) {
    static WORD: OnceLock<Regex> = OnceLock::new();
    static WORDS: OnceLock<Regex> = OnceLock::new();
    let word = WORD.get_or_init(|| Regex::new(r"[A-Za-zÀ-ÿ]{2,}").unwrap());
    let operative = WORDS.get_or_init(|| Regex::new(OPERATIVE).unwrap());
    (word.find_iter(text).count(), operative.find_iter(text).count())
}

impl Evidence {
    /// Every page, line and the document measured once: `separators` holds each page's note
    /// separator as the extraction found it, `outline` the document's bookmarks, `page_labels` the
    /// page labels it declares (by page index), and `instrument` whether its caller knows it for an
    /// instrument.
    pub(crate) fn measure(
        pages: &[Page],
        slots: &Slots,
        normalized: Normalized,
        separators: &[Option<f64>],
        outline: &[PdfOutlineEntry],
        page_labels: &[Option<String>],
        instrument: bool,
    ) -> Self {
        let count = slots.len();
        let Normalized { line_numbers, items } = normalized;
        let mut evidence = Self {
            size: Vec::with_capacity(count),
            face: Vec::with_capacity(count),
            half_points: Vec::with_capacity(count),
            look: Vec::with_capacity(count),
            weight: Vec::with_capacity(count),
            tag: Vec::with_capacity(count),
            marker: Vec::with_capacity(count),
            beside: vec![None; count],
            label: Vec::with_capacity(count),
            references: Vec::with_capacity(count),
            title_shaped: Vec::with_capacity(count),
            words: Vec::with_capacity(count),
            page_labels: page_labels.to_vec(),
            ..Self::default()
        };
        let (mut faces, mut families) = (Names::default(), Names::default());
        let (mut words, mut operative) = (0, 0);
        for (page_slot, page) in pages.iter().enumerate() {
            let first = slots.on(page_slot).start as usize;
            let rows = page_rows(page);
            let sizes: Vec<f64> = page.lines.iter().map(line_font_size).collect();
            for (at, line) in page.lines.iter().enumerate() {
                let text = line.text.trim();
                let face = dominant_face(line);
                evidence.size.push(sizes[at]);
                evidence.face.push(face.map_or(u16::MAX, |(face, _)| faces.id(face)));
                evidence.half_points.push(face.map_or(0, |(_, size)| (size * 2.0).round().clamp(0.0, f64::from(u16::MAX)) as u16));
                let underlined = !line.words.is_empty() && line.words.iter().filter(|word| word.underlined).count() * 2 > line.words.len();
                evidence.look.push(
                    [
                        (LOOK_BOLD, bold_char_share(line) >= 0.5),
                        (LOOK_ITALIC, flag_share(line, ITALIC).unwrap_or(0.0) > 0.5),
                        (LOOK_CAPS, all_caps(text)),
                        (LOOK_UNDERLINED, underlined),
                    ]
                    .into_iter()
                    .filter(|(_, holds)| *holds)
                    .fold(0, |bits, (bit, _)| bits | bit),
                );
                evidence.weight.push(line.weight as f32);
                evidence.tag.push(SourceTag::of(line));
                evidence.marker.push(opening_marker(line, &mut families));
                evidence.label.push(note_label(page, &sizes, at));
                evidence.references.push(reference_tokens(line, sizes[at]));
                evidence.title_shaped.push(
                    !text.is_empty()
                        && text.split_whitespace().count() <= 12
                        && !sentence_ended(text)
                        && !text.ends_with([',', ';', ':']),
                );
                evidence.words.push(lexicon(line));
                if !line_numbers[first + at] {
                    let (read, found) = operative_words(&line.text);
                    words += read;
                    operative += found;
                }
            }
            // A marker alone on its line, its text on the line beside it on its row.
            let mut by_row: Vec<Vec<usize>> = vec![Vec::new(); rows.count as usize];
            for (at, &row) in rows.of.iter().enumerate() {
                if row != u32::MAX {
                    by_row[row as usize].push(at);
                }
            }
            for (at, marker) in page.lines.iter().enumerate() {
                let only = evidence.marker[first + at].is_some_and(|reading| reading.only);
                if !only || line_numbers[first + at] || rows.of[at] == u32::MAX {
                    continue;
                }
                evidence.beside[first + at] = by_row[rows.of[at] as usize]
                    .iter()
                    .copied()
                    .filter(|&other| {
                        let line = &page.lines[other];
                        other != at
                            && !line_numbers[first + other]
                            && !line.text.trim().is_empty()
                            && crate::paragraphs::same_row(marker, line)
                            && line.bbox[0] >= marker.bbox[2] - 1.0
                    })
                    .min_by(|&a, &b| page.lines[a].bbox[0].total_cmp(&page.lines[b].bbox[0]))
                    .map(|other| slots.at(page_slot, other));
            }
            evidence.pages.push(PageEvidence { rows, separator: separators.get(page_slot).copied().flatten() });
        }
        // A heading tag on so many of the tagged lines that it marks the body is no heading's.
        let overused = headings::overused_tags(pages, &items, |page, line| {
            let slot = slots.at(page, line).index();
            (!line_numbers[slot] && evidence.tag[slot] != SourceTag::None).then(|| evidence.tag[slot].name())
        });
        if !overused.is_empty() {
            for tag in &mut evidence.tag {
                if matches!(tag, SourceTag::Heading(_)) && overused.contains(tag.name()) {
                    *tag = SourceTag::None;
                }
            }
        }
        evidence.read_citations(pages, slots, &line_numbers);
        evidence.outline_usable = headings::bookmarks::usable(outline, pages.len());
        evidence.outline_complete = headings::bookmarks::complete(outline, pages.len());
        evidence.instrument = instrument || words >= 200 && operative * 1000 >= words * 6;
        evidence.faces = faces.names;
        evidence.families = families.names;
        evidence.line_number = line_numbers;
        evidence.items = items;
        evidence
    }

    /// The citations on the lines worth reading for them, read in one window: each line carrying a
    /// marker, a note label, a reference token or a figure (a number the note pairer could read as a
    /// reference, which a citation holding it protects), or shaped as a title, and every line of the
    /// first two pages, each with the line either side of it; then which of the first two pages'
    /// lines name the case by its citation.
    fn read_citations(&mut self, pages: &[Page], slots: &Slots, line_numbers: &[bool]) {
        let count = slots.len();
        let mut wanted: Vec<bool> = (0..count)
            .map(|slot| {
                !line_numbers[slot]
                    && (self.marker[slot].is_some()
                        || self.label[slot].is_some()
                        || self.references[slot] != 0
                        || self.title_shaped[slot]
                        || slots.page(Slot(slot as u32)) < 2
                        || slots.line(pages, Slot(slot as u32)).text.chars().any(char::is_numeric))
            })
            .collect();
        let seeds = wanted.clone();
        for slot in (0..count).filter(|&slot| seeds[slot]) {
            if slot > 0 {
                wanted[slot - 1] = true;
            }
            if slot + 1 < count {
                wanted[slot + 1] = true;
            }
        }
        // The window: each run of wanted lines a paragraph of its own, line under line.
        let mut window = String::new();
        // Each windowed line's slot, and where it starts in the window, in bytes and in scalars.
        let mut placed: Vec<(Slot, usize, usize)> = Vec::new();
        let mut scalars = 0;
        let mut previous = None;
        for slot in (0..count).filter(|&slot| wanted[slot]) {
            if let Some(prior) = previous {
                let separator = if prior + 1 == slot { "\n" } else { "\n\n" };
                window.push_str(separator);
                scalars += separator.len();
            }
            let text = &slots.line(pages, Slot(slot as u32)).text;
            placed.push((Slot(slot as u32), window.len(), scalars));
            window.push_str(text);
            scalars += text.chars().count();
            previous = Some(slot);
        }
        let mut found: Vec<Cited> = Vec::new();
        if !window.is_empty() {
            let analysis = layout_analysis();
            let placed = &placed;
            let line_text = |at: usize| &slots.line(pages, placed[at].0).text;
            // Each windowed line a byte range overlaps, where the line starts in the window, and the
            // range's part on it.
            let on_lines = |start: usize, end: usize| {
                let first = placed.partition_point(|&(_, byte, _)| byte <= start).saturating_sub(1);
                (first..placed.len())
                    .take_while(move |&at| placed[at].1 < end)
                    .filter_map(move |at| {
                        let (slot, byte, _) = placed[at];
                        let length = line_text(at).len();
                        let (from, to) = (start.max(byte), end.min(byte + length));
                        (from < to).then(|| (slot, byte as u32, (from - byte) as u32, (to - byte) as u32))
                    })
            };
            for citation in analysis.citations(&window) {
                for (slot, byte, from, to) in on_lines(citation.range.start, citation.range.end) {
                    let core_from = (citation.core_range.start as u32).saturating_sub(byte).clamp(from, to);
                    let core_to = (citation.core_range.end as u32).saturating_sub(byte).clamp(core_from, to);
                    found.push(Cited { slot, range: (from, to), core: (core_from, core_to), protected: false });
                }
            }
            // The protected spans, in scalars: each read back to bytes on its lines.
            for (start, end) in analysis.protected_citation_spans(&window) {
                let first = placed.partition_point(|&(_, _, scalar)| scalar <= start).saturating_sub(1);
                for at in (first..placed.len()).take_while(|&at| placed[at].2 < end) {
                    let (slot, _, scalar) = placed[at];
                    let text = line_text(at);
                    let length = text.chars().count();
                    let (from, to) = (start.max(scalar) - scalar, end.min(scalar + length).saturating_sub(scalar));
                    if from >= to {
                        continue;
                    }
                    let byte = |offset: usize| text.char_indices().nth(offset).map_or(text.len(), |(byte, _)| byte) as u32;
                    let range = (byte(from), byte(to));
                    found.push(Cited { slot, range, core: range, protected: true });
                }
            }
        }
        found.sort_by_key(|cited| (cited.slot, cited.range.0, cited.range.1));
        let mut cited = Vec::with_capacity(count + 1);
        let mut at = 0;
        for slot in 0..count {
            cited.push(at as u32);
            while at < found.len() && found[at].slot.index() == slot {
                at += 1;
            }
        }
        cited.push(found.len() as u32);
        self.citations = found;
        self.cited = cited;
        // A citation naming the case on the first two pages.
        let opening = if pages.is_empty() { 0 } else { slots.on(pages.len().min(2) - 1).end as usize };
        for slot in (0..opening).filter(|&slot| !line_numbers[slot]) {
            let line = slots.line(pages, Slot(slot as u32));
            let cites = self.citations_of(Slot(slot as u32));
            let titled = captions::citation_title(line, |text, offset| {
                let (start, end) = (offset as u32, (offset + text.len()) as u32);
                cites
                    .iter()
                    .filter(|cited| !cited.protected && cited.range.0 >= start && cited.range.1 <= end)
                    .map(|cited| CitationSpan {
                        range: (cited.range.0 - start) as usize..(cited.range.1 - start) as usize,
                        core_range: (cited.core.0.max(start) - start) as usize..(cited.core.1.max(start) - start) as usize,
                        text: String::new(),
                    })
                    .collect()
            });
            if titled.is_some() {
                self.words[slot] |= WORD_CITATION_TITLE;
            }
        }
    }

    /// Reads again what a line's text says of it, for the lines whose opening marker the runs'
    /// repair rewrote (`repaired`: each slot with its text before): its marker, note label, shape and
    /// lexicon, and its citations moved with the words after the marker.
    pub(crate) fn reread(&mut self, pages: &[Page], slots: &Slots, repaired: &[(Slot, String)]) {
        if repaired.is_empty() {
            return;
        }
        let mut families = Names {
            ids: self.families.iter().enumerate().map(|(id, name)| (name.clone(), id as u16)).collect(),
            names: std::mem::take(&mut self.families),
        };
        for (slot, before) in repaired {
            let at = slot.index();
            let page_slot = slots.page(*slot);
            let range = slots.on(page_slot);
            let page = &pages[page_slot];
            let line = slots.line(pages, *slot);
            let text = line.text.trim();
            self.marker[at] = opening_marker(line, &mut families);
            self.label[at] = note_label(page, &self.size[range.start as usize..range.end as usize], (slot.0 - range.start) as usize);
            self.title_shaped[at] =
                !text.is_empty() && text.split_whitespace().count() <= 12 && !sentence_ended(text) && !text.ends_with([',', ';', ':']);
            self.words[at] = lexicon(line) | self.words[at] & WORD_CITATION_TITLE;
            // The words after the rewritten marker keep their citations, moved by what the marker
            // gained or lost.
            let after = before.len() - before.bytes().rev().zip(line.text.bytes().rev()).take_while(|(a, b)| a == b).count();
            let shift = line.text.len() as i64 - before.len() as i64;
            let length = line.text.len() as u32;
            let moved = |offset: u32| {
                let offset = if offset as usize >= after { (i64::from(offset) + shift).max(0) as u32 } else { offset };
                offset.min(length)
            };
            let (start, end) = (self.cited[at] as usize, self.cited[at + 1] as usize);
            for cited in &mut self.citations[start..end] {
                cited.range = (moved(cited.range.0), moved(cited.range.1));
                cited.core = (moved(cited.core.0), moved(cited.core.1));
            }
        }
        self.families = families.names;
    }

    /// The slot's citations.
    pub(crate) fn citations_of(&self, slot: Slot) -> &[Cited] {
        let (start, end) = (self.cited[slot.index()] as usize, self.cited[slot.index() + 1] as usize);
        &self.citations[start..end]
    }

    /// The slot's row on its page (`u32::MAX` for none).
    pub(crate) fn row(&self, slots: &Slots, slot: Slot) -> u32 {
        let page = slots.page(slot);
        self.pages[page].rows.of[(slot.0 - slots.on(page).start) as usize]
    }
}
