//! What a page's geometry and a document's own lists settle before any run of numbers is read, and
//! the order the lines read in. Each fence takes only the lines no fence before it took, in this
//! order: a transcript's line numbers; the furniture and folios at the pages' edges; then each
//! page's columns, read on the words the fences left; the parallel translation's lines; the
//! contents lists (their title a heading; a table of authorities' lines left to the body); the
//! tables' cells. Then the
//! note separator clear of the tables and the band under it, whose table cells are its notes' where
//! a note label stands in it; the body's type; each line's constituent document, and each
//! document's own furniture; whether each document is a transcript, and the lines beside its number
//! column, whose testimony no table holds; the reading order page by page; the quotations a
//! document's own count does not hold, each with the editorial lines set under it; the reading
//! contexts; and each line's continuation of the line before it.

use super::*;
use legal_pdf_tables::PageItems;

/// What a line is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum Role {
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
pub(crate) enum Rule {
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
    pub(crate) fn name(self) -> &'static str {
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

/// Why a line goes on with the one before it in reading order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum Continuation {
    /// A word hyphenated across, an unfinished sentence going on in lower case, or a bracket closed
    /// (`body_flow_edge`).
    Flow,
    /// The line above ends on an enactment's noun its number completes (`reference_wrapped_onto`).
    Reference,
    /// A full line ending no sentence whose sentence this line finishes (`carries_a_sentence_onto`).
    Sentence,
    /// Set at its paragraph's leading under the line before it, a full line ending no sentence
    /// (`continues_prose_from`).
    Prose,
}

/// A reading context: a constituent document of the packet (0 for its own), its parallel
/// translation or not, and a quotation in it (0 for none).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Context {
    pub(crate) document: u16,
    pub(crate) translation: bool,
    pub(crate) quotation: u16,
}

/// The fences, contexts and reading order, by slot.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct Fences {
    pub(crate) role: Vec<Role>,
    pub(crate) rule: Vec<Rule>,
    /// The edge a furniture or page label line stands at.
    pub(crate) edge: Vec<Option<Edge>>,
    /// Each page's columns, read on the words of its lines no line-number or furniture fence took
    /// (`columns_of`), by page slot.
    pub(crate) columns: Vec<ColumnModel>,
    /// Each constituent document is a transcript, by document: its lines are numbered, or a tenth
    /// of them or more (twenty or more) open on a speaker's label.
    pub(crate) transcripts: Vec<bool>,
    /// Each transcript whose lines are testimony throughout, by document: a tenth of them or more
    /// (twenty or more) open on a speaker's label, or its line numbers stand on half its pages.
    pub(crate) testimony: Vec<bool>,
    /// A transcript's line beside its number column: on a row holding a line number.
    pub(crate) numbered_row: Vec<bool>,
    /// Each page's printed label and the slot of the line printing it, by page slot.
    pub(crate) folios: Vec<Option<(Slot, String)>>,
    /// The printed page number less the page's own, where most folios agree on one.
    pub(crate) folio_offset: Option<i64>,
    /// Edge lines naming their page's sections ("Section 63"), with their edge and the first
    /// section named, for the statute's spine to bear out (`confirm_section_heads`).
    pub(crate) section_heads: Vec<(Slot, Edge, u32)>,
    pub(in crate::structure) contents_entries: Vec<contents_lists::Entry>,
    /// Pages wholly a contents list's, by page slot.
    pub(crate) contents_pages: Vec<bool>,
    /// The document's tables, a contents list read as one aside.
    pub(crate) tables: Vec<legal_pdf_tables::PdfTable>,
    /// The cell each line stands in (1 for the tables' first cell; 0 for none), a line parted
    /// across cells standing in the first.
    pub(crate) cell: Vec<u32>,
    /// Each page's note separator, unless it runs through a table.
    pub(crate) separators: Vec<Option<f64>>,
    /// Under its page's note separator, and no line label, furniture or folio.
    pub(crate) band: Vec<bool>,
    /// Each page's band holds a note label no reason suppresses, by page slot: its table cells are
    /// its notes', no table's.
    pub(crate) noted: Vec<bool>,
    /// The body's type: the commonest face (an index into the evidence's faces) and half-point size
    /// of the lines no fence took outside the note band, by characters.
    pub(crate) body: Option<(u16, u16)>,
    pub(crate) translation: Vec<bool>,
    pub(in crate::structure) starts: Vec<constituents::Start>,
    pub(crate) document: Vec<u16>,
    pub(crate) quotation: Vec<u16>,
    /// Each line's reading context, an index into `contexts`.
    pub(crate) ctx: Vec<u16>,
    pub(crate) contexts: Vec<Context>,
    /// A paragraph number printed in the margin (`read_margin_numbers`): the body line it numbers.
    pub(crate) margin_number: Vec<Option<Slot>>,
    /// The line opening the same paragraph in the other language, on a margin number's row.
    pub(crate) margin_twin: Vec<Option<Slot>>,
    /// The paragraph number a line opens, and the spans of each line each numbered paragraph holds (its language,
    /// number and UTF-16 span of the line's text): the document's paragraphs, read once before any other reading
    /// claims a line (`read_paragraphs`).
    pub(crate) count: Vec<Option<u32>>,
    pub(crate) held: Vec<Vec<(bool, u32, u32, u32)>>,
    /// Every line in reading order, page by page.
    pub(crate) order: Vec<Slot>,
    /// Each line's place in `order`.
    pub(crate) place: Vec<u32>,
    /// How each line no fence took goes on with the line before it in its reading context and
    /// stream (the band's or the body's), in reading order.
    pub(crate) continues: Vec<Option<Continuation>>,
    pub(crate) diagnostics: Vec<Diagnostic>,
}

impl Fences {
    /// Gives an open line a role, by a rule; a line some fence took already keeps its own.
    fn set(&mut self, slot: Slot, role: Role, rule: Rule) -> bool {
        let open = self.role[slot.index()] == Role::Open;
        if open {
            self.role[slot.index()] = role;
            self.rule[slot.index()] = rule;
        }
        open
    }

    /// Whether the slot's constituent document is a transcript.
    pub(crate) fn transcript(&self, slot: Slot) -> bool {
        self.transcripts.get(usize::from(self.document[slot.index()])).copied().unwrap_or(false)
    }

    /// Whether the slot's constituent document is a transcript's testimony throughout (`testimony`).
    pub(crate) fn testimony(&self, slot: Slot) -> bool {
        self.testimony.get(usize::from(self.document[slot.index()])).copied().unwrap_or(false)
    }

    /// The fences, contexts and reading order of the document's pages, measured (`evidence`), its
    /// bookmarks `outline`.
    pub(crate) fn read(pages: &[Page], slots: &Slots, evidence: &Evidence, outline: &[PdfOutlineEntry]) -> Self {
        let count = slots.len();
        let mut fences = Self {
            role: vec![Role::Open; count],
            rule: vec![Rule::None; count],
            edge: vec![None; count],
            cell: vec![0; count],
            band: vec![false; count],
            translation: vec![false; count],
            document: vec![0; count],
            quotation: vec![0; count],
            continues: vec![None; count],
            ..Self::default()
        };
        let views = PageView::whole(pages);
        let rows: Vec<&Rows> = evidence.pages.iter().map(|page| &page.rows).collect();
        // A transcript's line numbers.
        for slot in (0..count).filter(|&slot| evidence.line_number[slot]) {
            fences.set(Slot(slot as u32), Role::LineLabel, Rule::LineNumber);
        }
        fences.read_furniture(pages, slots, evidence, &views, &rows);
        // Each page's words outside its line numbers and furniture, and its columns read on them.
        let starts: Vec<usize> = (0..pages.len()).map(|page_slot| slots.on(page_slot).start as usize).collect();
        let items: Vec<PageItems> = evidence
            .items
            .iter()
            .zip(&starts)
            .map(|(items, &start)| items_of(items, |line| fences.role[start + line] == Role::Open))
            .collect();
        fences.columns = pages
            .iter()
            .zip(&items)
            .zip(&starts)
            .map(|((page, items), &start)| {
                let lines = (0..page.lines.len()).filter(|&line| fences.role[start + line] == Role::Open);
                columns_of(&page.lines, lines, &items.items, page.index as u32 + 1)
            })
            .collect();
        // The parallel translation, on the pages' columns.
        for slot in translation_lines(&views, &fences.columns, |slot| fences.role[slot.index()] != Role::Open) {
            fences.translation[slot.index()] = true;
        }
        // A justified line the PDF sets a word at a time crosses the page's middle, in one language: a row of three or more
        // pieces of a word or two is one line, its language its first piece's.
        for page_slot in 0..pages.len() {
            let mut rows = HashMap::<u32, Vec<Slot>>::new();
            for slot in slots.on(page_slot).map(Slot) {
                rows.entry(evidence.row(slots, slot)).or_default().push(slot);
            }
            for (row, mut pieces) in rows {
                let pieces_of_words = pieces.iter().all(|&slot| slots.line(pages, slot).text.split_whitespace().count() <= 2);
                if row != u32::MAX && pieces.len() >= 3 && pieces_of_words {
                    pieces.sort_by(|&a, &b| slots.line(pages, a).bbox[0].total_cmp(&slots.line(pages, b).bbox[0]));
                    // (Across the gutter a row is two: "I. Introduction" level in both columns.)
                    let word_apart = pieces.windows(2).all(|pair| {
                        let (left, right) = (slots.line(pages, pair[0]).bbox, slots.line(pages, pair[1]).bbox);
                        right[0] - left[2] <= (left[3] - left[1]) * 3.0
                    });
                    if !word_apart {
                        continue;
                    }
                    let language = fences.translation[pieces[0].index()];
                    for slot in pieces {
                        fences.translation[slot.index()] = language;
                    }
                }
            }
        }
        let (tables, listed) = legal_pdf_support::profile::measure("classify.tables", || {
            legal_pdf_tables::document_tables(
                pages,
                &items,
                &|id| slots.of_id(id).is_some_and(|slot| fences.translation[slot.index()]),
                &TableReadings,
            )
        });
        let mut in_table = vec![false; count];
        for slot in tables.iter().flat_map(|table| table.line_ids()).filter_map(|id| slots.of_id(id)) {
            in_table[slot.index()] = true;
        }
        let listed_contents = fences.read_contents(pages, slots, evidence, &views, &in_table, &listed);
        fences.read_tables(slots, tables, &listed_contents);
        // The note band: under the page's separator, clear of its tables.
        for (page_slot, page) in pages.iter().enumerate() {
            let range = slots.on(page_slot);
            let cells = range.clone().filter(|&slot| fences.cell[slot as usize] != 0).map(|slot| slots.line(pages, Slot(slot)));
            let separator = separator_beside_tables(page, evidence.pages[page_slot].separator, cells);
            fences.separators.push(separator);
            if let Some(cut) = separator {
                let tolerance = separator_tolerance(page);
                for slot in range.map(Slot) {
                    let line = slots.line(pages, slot);
                    fences.band[slot.index()] = line.bbox[1] >= cut - tolerance
                        && !matches!(fences.role[slot.index()], Role::LineLabel | Role::Furniture | Role::PageLabel);
                }
            }
        }
        // The body's type, of the lines no fence took, the note band aside.
        let mut mass: HashMap<(u16, u16), usize> = HashMap::new();
        for slot in (0..count).filter(|&slot| fences.role[slot] == Role::Open && !fences.band[slot] && evidence.face[slot] != u16::MAX) {
            *mass.entry((evidence.face[slot], evidence.half_points[slot])).or_default() +=
                slots.line(pages, Slot(slot as u32)).text.trim().chars().count();
        }
        fences.body = mass
            .into_iter()
            .max_by(|a, b| {
                a.1.cmp(&b.1).then_with(|| {
                    (evidence.faces[usize::from(a.0 .0)].as_str(), a.0 .1).cmp(&(evidence.faces[usize::from(b.0 .0)].as_str(), b.0 .1))
                })
            })
            .map(|(style, _)| style);
        // A rule drawn across a page's running text is no note separator ("Cette disposition est r\u{e9}dig\u{e9}e ainsi:" under it,
        // an enactment's small type below): a page's notes open on a label, or hold no words in the body's type. (A margin's
        // figure opens nothing.)
        for page_slot in 0..pages.len() {
            let worded = |slot: &usize| slots.line(pages, Slot(*slot as u32)).text.chars().any(char::is_alphabetic);
            let band: Vec<usize> = slots.on(page_slot).map(|slot| slot as usize).filter(|&slot| fences.band[slot]).collect();
            let top = |slot: usize| slots.line(pages, Slot(slot as u32)).bbox[1];
            let opens_on_label = band.iter().copied().filter(|slot| worded(slot)).min_by(|&a, &b| top(a).total_cmp(&top(b))).is_some_and(|first| evidence.label[first].is_some_and(|label| label.suppress == 0));
            // (Nor is the rule under an underlined line: "Issues", a heading's, over a list opening on its label.)
            let underlined = fences.separators[page_slot].is_some_and(|cut| slots.on(page_slot).any(|slot| {
                let line = slots.line(pages, Slot(slot));
                !fences.band[slot as usize] && (cut - line.bbox[3]).abs() <= 1.0 && !line.words.is_empty() && line.words.iter().all(|word| word.underlined)
            }));
            if underlined || !opens_on_label && band.iter().filter(|slot| worded(slot)).any(|&slot| fences.body.is_some_and(|(_, size)| evidence.half_points[slot] == size)) {
                band.iter().for_each(|&slot| fences.band[slot] = false);
                fences.separators[page_slot] = None;
            }
        }
        // A band holding a note label is the page's notes, its table cells among them.
        fences.noted = (0..pages.len())
            .map(|page_slot| {
                slots.on(page_slot).map(|slot| slot as usize).any(|slot| {
                    fences.band[slot]
                        && matches!(fences.role[slot], Role::Open | Role::Table)
                        && evidence.label[slot].is_some_and(|label| label.suppress == 0)
                })
            })
            .collect();
        let freed: Vec<usize> = (0..count)
            .filter(|&slot| fences.band[slot] && fences.role[slot] == Role::Table && fences.noted[slots.page(Slot(slot as u32))])
            .collect();
        let mut voided = !freed.is_empty();
        for slot in freed {
            fences.free_cell(slot);
        }
        fences.read_documents(&views, outline);
        if !fences.starts.is_empty() {
            fences.read_document_furniture(pages, slots, evidence, &rows);
        }
        fences.read_transcripts(slots, evidence);
        // A transcript's testimony beside its number column, read as a grid (a speaker's turn in
        // one of its cells), is no table's; a table the transcript sets among its numbered lines
        // stays one.
        let mut spoken = vec![false; count];
        for table in &fences.tables {
            let lines: Vec<Slot> = table.line_ids().into_iter().filter_map(|id| slots.of_id(id)).collect();
            if lines.iter().any(|slot| fences.numbered_row[slot.index()] && evidence.words[slot.index()] & WORD_SPEAKER != 0) {
                for slot in lines {
                    spoken[slot.index()] = true;
                }
            }
        }
        let freed: Vec<usize> =
            (0..count).filter(|&slot| fences.numbered_row[slot] && fences.role[slot] == Role::Table && spoken[slot]).collect();
        voided |= !freed.is_empty();
        for slot in freed {
            fences.free_cell(slot);
        }
        if voided {
            fences.prune_tables(slots);
        }
        fences.read_order(pages, slots, evidence);
        fences.read_margin_numbers(pages, slots);
        fences
    }

    /// The document's paragraph numbers, read once its misread markers are restored (`Runs::read`), the margin's again
    /// where any was. The judgment's paragraphs are counted on the lines' continuations, and the printed numbers the
    /// count does not take are another writing's, quoted: their passages are contexts of their own, read with the lines again.
    pub(crate) fn read_numbers(&mut self, pages: &[Page], slots: &Slots, evidence: &Evidence, repaired: bool) {
        if repaired {
            self.read_margin_numbers(pages, slots);
        }
        self.read_contexts();
        self.read_continuations(pages, slots, evidence);
        self.read_paragraphs(pages, slots, evidence);
        if self.read_quotations(pages, slots) {
            self.read_contexts();
            self.read_continuations(pages, slots, evidence);
        }
    }

    /// Each line's reading context: its document, its language and the quotation it stands in.
    fn read_contexts(&mut self) {
        let mut contexts: Vec<Context> = Vec::new();
        self.ctx = (0..self.document.len())
            .map(|slot| {
                let context = Context { document: self.document[slot], translation: self.translation[slot], quotation: self.quotation[slot] };
                contexts.iter().position(|known| *known == context).unwrap_or_else(|| {
                    contexts.push(context);
                    contexts.len() - 1
                }) as u16
            })
            .collect();
        self.contexts = contexts;
    }

    /// Folios, running heads and feet and the lines on their rows (`furniture::read`); the folios'
    /// offset, tallied over every label each page's furniture prints; each page's printed label
    /// from its furniture (`page_label`), its line the page's label; and the edge lines naming their
    /// page's sections.
    fn read_furniture(&mut self, pages: &[Page], slots: &Slots, evidence: &Evidence, views: &[PageView], rows: &[&Rows]) {
        let found = legal_pdf_support::profile::measure("prepare.furniture", || {
            furniture::read(views, rows, |slot| evidence.line_number[slot.index()])
        });
        for (slot, edge, rule) in found {
            if self.set(slot, Role::Furniture, rule) {
                self.edge[slot.index()] = Some(edge);
            }
        }
        let labels: Vec<Vec<(String, usize)>> =
            pages.iter().enumerate().map(|(page_slot, page)| self.printed_labels(page, slots, page_slot)).collect();
        // Each page counts each offset its labels print once: a record that prints its own page
        // beside the page of the transcript it files counts both.
        let mut offsets: HashMap<i64, usize> = HashMap::new();
        for (page, labels) in pages.iter().zip(&labels) {
            let mut printed: Vec<i64> =
                labels.iter().filter_map(|(label, _)| label.parse::<i64>().ok()).map(|number| number - i64::from(page.number)).collect();
            printed.sort_unstable();
            printed.dedup();
            for offset in printed {
                *offsets.entry(offset).or_default() += 1;
            }
        }
        let folios: usize = offsets.values().sum();
        self.folio_offset = offsets
            .into_iter()
            .max_by_key(|&(offset, pages)| (pages, std::cmp::Reverse(offset.abs())))
            .filter(|&(_, pages)| pages >= 2 && pages * 2 >= folios)
            .map(|(offset, _)| offset);
        self.folios = pages
            .iter()
            .zip(&labels)
            .enumerate()
            .map(|(page_slot, (page, labels))| self.page_label(page, slots, page_slot, labels))
            .collect();
        self.section_heads = furniture::section_heads(views, rows)
            .into_iter()
            .filter(|(slot, _, _)| self.role[slot.index()] == Role::Open)
            .collect();
    }

    /// The labels the page's furniture prints (`page_labels_among`), by the line printing each.
    fn printed_labels(&self, page: &Page, slots: &Slots, page_slot: usize) -> Vec<(String, usize)> {
        let range = slots.on(page_slot);
        let furniture = range
            .clone()
            .map(Slot)
            .filter(|slot| self.role[slot.index()] == Role::Furniture)
            .filter_map(|slot| Some(((slot.0 - range.start) as usize, self.edge[slot.index()]?)));
        page_labels_among(page, furniture)
    }

    /// The page's printed label among those its furniture prints (`labels`), the line printing it
    /// the page's label: the one label, or where they differ the one on the folios' offset; none
    /// where its furniture prints none, or prints labels that differ and none on the offset.
    fn page_label(&mut self, page: &Page, slots: &Slots, page_slot: usize, labels: &[(String, usize)]) -> Option<(Slot, String)> {
        let first = labels.first()?;
        let agree = labels.iter().all(|(label, _)| label.eq_ignore_ascii_case(&first.0));
        let on_offset = |(label, _): &&(String, usize)| {
            self.folio_offset.is_some_and(|offset| label.parse::<i64>().is_ok_and(|number| number - i64::from(page.number) == offset))
        };
        let Some((label, line)) = (if agree { Some(first) } else { labels.iter().find(on_offset) }) else {
            let mut diagnostic = Diagnostic::info(
                "PRINTED_PAGE_LABEL_AMBIGUOUS",
                "Conflicting header/footer page labels, none on the folios' offset, were left unresolved.",
                Some(page.index),
            );
            diagnostic.line_ids = labels.iter().map(|(_, line)| page.lines[*line].id.clone()).collect();
            diagnostic.details.insert("candidates".to_owned(), json!(labels.iter().map(|(label, _)| label).collect::<Vec<_>>()));
            self.diagnostics.push(diagnostic);
            return None;
        };
        let slot = slots.at(page_slot, *line);
        self.role[slot.index()] = Role::PageLabel;
        self.rule[slot.index()] = Rule::Folio;
        Some((slot, label.clone()))
    }

    /// Each constituent document's own furniture, read over its own lines as a document of its own
    /// (`furniture::read`): the running heads and feet it repeats on its own pages, a short
    /// document's edges. A line no fence took is its furniture; a page that gains furniture and had
    /// no printed label takes its label from it.
    fn read_document_furniture(&mut self, pages: &[Page], slots: &Slots, evidence: &Evidence, rows: &[&Rows]) {
        let documents = usize::from(self.document.iter().copied().max().unwrap_or(0)) + 1;
        let mut lines: Vec<Vec<Slot>> = vec![Vec::new(); documents];
        for slot in 0..slots.len() {
            lines[usize::from(self.document[slot])].push(Slot(slot as u32));
        }
        let mut gained = vec![false; pages.len()];
        for lines in lines.iter().filter(|lines| !lines.is_empty()) {
            let views = PageView::of_slots(pages, slots, lines);
            let rows: Vec<&Rows> = views.iter().map(|view| rows[slots.page(view.lines[0])]).collect();
            let found = legal_pdf_support::profile::measure("prepare.furniture", || {
                furniture::read(&views, &rows, |slot| evidence.line_number[slot.index()])
            });
            for (slot, edge, rule) in found {
                if self.set(slot, Role::Furniture, rule) {
                    self.edge[slot.index()] = Some(edge);
                    self.band[slot.index()] = false;
                    gained[slots.page(slot)] = true;
                }
            }
        }
        for (page_slot, page) in pages.iter().enumerate().filter(|(page_slot, _)| gained[*page_slot]) {
            if self.folios[page_slot].is_none() {
                let labels = self.printed_labels(page, slots, page_slot);
                self.folios[page_slot] = self.page_label(page, slots, page_slot, &labels);
            }
        }
    }

    /// Which constituent documents are transcripts (`transcripts`) and which of them are testimony
    /// throughout (`testimony`), and each transcript's lines beside its number column
    /// (`numbered_row`).
    fn read_transcripts(&mut self, slots: &Slots, evidence: &Evidence) {
        let count = slots.len();
        let documents = usize::from(self.document.iter().copied().max().unwrap_or(0)) + 1;
        let (mut numbered, mut speakers, mut lines) = (vec![0usize; documents], vec![0usize; documents], vec![0usize; documents]);
        // Each document's pages, and those printing its line numbers: (count, last page met).
        let (mut pages, mut numbered_pages) = (vec![(0usize, usize::MAX); documents], vec![(0usize, usize::MAX); documents]);
        let meet = |met: &mut (usize, usize), page: usize| {
            if met.1 != page {
                *met = (met.0 + 1, page);
            }
        };
        for slot in 0..count {
            let document = usize::from(self.document[slot]);
            let page = slots.page(Slot(slot as u32));
            meet(&mut pages[document], page);
            if evidence.line_number[slot] {
                numbered[document] += 1;
                meet(&mut numbered_pages[document], page);
            } else {
                lines[document] += 1;
                speakers[document] += usize::from(evidence.words[slot] & WORD_SPEAKER != 0);
            }
        }
        let spoken = |document: usize| speakers[document] >= 20 && speakers[document] * 10 >= lines[document];
        self.transcripts = (0..documents).map(|document| numbered[document] > 0 || spoken(document)).collect();
        // A contents page's locators or a table's row numbers read as line numbers make a document a
        // transcript, not its testimony.
        self.testimony = (0..documents)
            .map(|document| spoken(document) || numbered_pages[document].0 * 2 >= pages[document].0.max(1))
            .collect();
        self.numbered_row = vec![false; count];
        for page_slot in 0..evidence.pages.len() {
            let range = slots.on(page_slot);
            let rows = &evidence.pages[page_slot].rows;
            let numbered: Vec<u32> = range
                .clone()
                .filter(|&slot| self.role[slot as usize] == Role::LineLabel)
                .map(|slot| rows.of[(slot - range.start) as usize])
                .filter(|row| *row != u32::MAX)
                .collect();
            if numbered.is_empty() {
                continue;
            }
            for slot in range.clone() {
                let row = rows.of[(slot - range.start) as usize];
                self.numbered_row[slot as usize] = row != u32::MAX && numbered.contains(&row) && self.transcript(Slot(slot));
            }
        }
    }

    /// Gives a table's cell back to the lines no fence took.
    fn free_cell(&mut self, slot: usize) {
        self.role[slot] = Role::Open;
        self.rule[slot] = Rule::None;
        self.cell[slot] = 0;
    }

    /// The tables without the lines freed of their cells: a cell's parts on those lines dropped, and
    /// a row or table left with no part dropped; each line's cell numbered again.
    fn prune_tables(&mut self, slots: &Slots) {
        let cell = &self.cell;
        let kept = |part: &legal_pdf_tables::CellPart| slots.of_id(&part.line_id).is_some_and(|slot| cell[slot.index()] != 0);
        for table in &mut self.tables {
            for row in &mut table.rows {
                for value in &mut row.cells {
                    value.parts.retain(|part| kept(part));
                }
            }
            table.rows.retain(|row| row.cells.iter().any(|value| !value.parts.is_empty()));
        }
        self.tables.retain(|table| !table.rows.is_empty());
        self.number_cells(slots);
    }

    /// Each line's cell (`cell`), counted over the tables' rows and their cells, from 1; a line
    /// parted across cells stands in the first.
    fn number_cells(&mut self, slots: &Slots) {
        self.cell = vec![0; slots.len()];
        let cells = self.tables.iter().flat_map(|table| &table.rows).flat_map(|row| &row.cells);
        for (cell, value) in cells.enumerate() {
            for slot in value.parts.iter().filter_map(|part| slots.of_id(&part.line_id)) {
                if self.cell[slot.index()] == 0 {
                    self.cell[slot.index()] = cell as u32 + 1;
                }
            }
        }
    }

    /// The contents lists (`contents_lists::read`), pdf-inspector's contents among them, each from
    /// its first entry to its last, a table of authorities' lines aside; the row naming a list, its
    /// heading; the pages wholly contents. Returns the lines the lists read as contents (a table
    /// more than half of whose lines they hold is that list).
    fn read_contents(
        &mut self,
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

    /// The tables a contents list does not hold for the most part, and their cells (`cell`); a cell
    /// no fence took is the table's.
    fn read_tables(&mut self, slots: &Slots, tables: Vec<legal_pdf_tables::PdfTable>, listed: &[bool]) {
        let trace = std::env::var_os("LEGALPDF_CONTENTS_TRACE").is_some();
        self.tables = tables
            .into_iter()
            .filter(|table| {
                let ids = table.line_ids();
                let held = ids.iter().filter(|id| slots.of_id(id).is_some_and(|slot| listed[slot.index()])).count();
                if trace {
                    eprintln!("contents\ttable of {} lines, {held} listed: {}", ids.len(), if held * 2 <= ids.len() { "kept" } else { "dropped" });
                }
                held * 2 <= ids.len()
            })
            .collect();
        self.number_cells(slots);
        for slot in 0..slots.len() {
            if self.cell[slot] != 0 {
                self.set(Slot(slot as u32), Role::Table, Rule::Grid);
            }
        }
    }

    /// Each line's constituent document: the openings the document's bookmarks name, and those its
    /// own lines show, read on the lines no fence took and the note band left.
    fn read_documents(&mut self, views: &[PageView], outline: &[PdfOutlineEntry]) {
        let (role, band) = (&self.role, &self.band);
        let set_aside = |slot: Slot| role[slot.index()] != Role::Open || band[slot.index()];
        let mut starts = constituents::bookmarked_starts(views, outline, set_aside);
        constituents::reconcile_starts(views, &mut starts, set_aside);
        let documents = constituents::documents(views, &starts, |slot| {
            matches!(role[slot.index()], Role::LineLabel | Role::Furniture | Role::PageLabel)
        });
        for (slot, document) in documents {
            self.document[slot.index()] = document.min(usize::from(u16::MAX)) as u16;
        }
        self.starts = starts;
    }

    /// Every page's lines in reading order (`page_order`): its running head, its body by its columns
    /// with its note band woven in, its foot; a table's lines in its rows' order.
    fn read_order(&mut self, pages: &[Page], slots: &Slots, evidence: &Evidence) {
        self.order = Vec::with_capacity(slots.len());
        // Each table line's table and its place among the table's lines, row by row, cell by cell.
        let mut in_table: Vec<Option<(u32, u32)>> = vec![None; slots.len()];
        for (table, value) in self.tables.iter().enumerate() {
            for (at, id) in value.line_ids().into_iter().enumerate() {
                if let Some(slot) = slots.of_id(id) {
                    in_table[slot.index()].get_or_insert((table as u32, at as u32));
                }
            }
        }
        legal_pdf_support::profile::measure("classify.order_pages", || {
            for (page_slot, page) in pages.iter().enumerate() {
                let start = slots.on(page_slot).start as usize;
                let stream = |line: usize| {
                    let slot = start + line;
                    match (self.role[slot], self.edge[slot]) {
                        (Role::Furniture | Role::PageLabel, Some(Edge::Top)) => Stream::Header,
                        (Role::Furniture | Role::PageLabel, Some(Edge::Bottom)) => Stream::Footer,
                        _ if self.band[slot] => Stream::Note,
                        _ => Stream::Body,
                    }
                };
                let table_page = (0..page.lines.len()).any(|line| self.cell[start + line] != 0);
                let (order, diagnostics) = page_order(
                    page,
                    stream,
                    &evidence.items[page_slot],
                    self.columns[page_slot],
                    table_page,
                    self.contents_pages[page_slot],
                );
                let order = if table_page { table_order(order, &in_table[start..start + page.lines.len()]) } else { order };
                self.diagnostics.extend(diagnostics);
                self.order.extend(order.into_iter().map(|line| Slot((start + line) as u32)));
            }
        });
        self.place = vec![0; slots.len()];
        for (place, slot) in self.order.iter().enumerate() {
            self.place[slot.index()] = place as u32;
        }
    }

    /// The paragraph numbers a report prints in its margin, beside the row where each paragraph's text opens (in
    /// both columns of a parallel report): a number alone on its line, standing clear of the page's body text by
    /// up to eight lines' height, level with a line of it, and counting on with the document's other such
    /// numbers, three or more in a row. Each is read just before the first line of its row in the main text (a
    /// parallel translation's after), so it opens that line's paragraph; a page's reading order otherwise sets the
    /// margin after both columns, where its numbers would open whatever follows (the next page's running head).
    fn read_margin_numbers(&mut self, pages: &[Page], slots: &Slots) {
        self.margin_number = vec![None; slots.len()];
        self.margin_twin = vec![None; slots.len()];
        let mut found: Vec<(u32, Slot, Slot, Option<Slot>)> = Vec::new();
        for (page_slot, page) in pages.iter().enumerate() {
            // A page's note band may start high on it, over text the notes do not hold: its margin numbers are read too.
            let open = |slot: Slot| self.role[slot.index()] == Role::Open;
            // The page's text lines; its body's edges and height are read on those of some length, so a short line
            // ("The PSR stated at page 8:") opens its row's paragraph but moves no edge. (A column of numbers beside a column of
            // words reads as a table's grid: those words are text all the same.)
            let words: Vec<Slot> = slots.on(page_slot).map(Slot)
                .filter(|&slot| (open(slot) || self.role[slot.index()] == Role::Table) && slots.line(pages, slot).text.chars().any(char::is_alphabetic)).collect();
            let body: Vec<Slot> = words.iter().copied().filter(|&slot| {
                let line = slots.line(pages, slot);
                line.bbox[2] - line.bbox[0] > page.width * 0.20
            }).collect();
            // A line's ink across the page: a recognizer pads the box it reads a line in, a quarter of its height at
            // each end (a scan's margin number all but meets its column's padded edge).
            let ink = |line: &Line| {
                let pad = if line.ocr.is_some() { 0.25 * (line.bbox[3] - line.bbox[1]).max(0.0) } else { 0.0 };
                (line.bbox[0] + pad, line.bbox[2] - pad)
            };
            // The body's edges, where nearly all its lines stand: one line run out into the margin moves neither.
            let edge = |side: usize, share: f64| {
                let mut xs: Vec<f64> = body.iter().map(|&slot| { let (left, right) = ink(slots.line(pages, slot)); if side == 0 { left } else { right } }).collect();
                xs.sort_by(f64::total_cmp);
                xs.get((xs.len().saturating_sub(1) as f64 * share).round() as usize).copied()
            };
            let (Some(left), Some(right)) = (edge(0, 0.05), edge(2, 0.95)) else { continue };
            let bounds = [left, 0.0, right, 0.0];
            // The height most of the page's body lines stand.
            // (A PDF that gives its lines no height gives them their type's.)
            let mut heights: Vec<f64> = body.iter().map(|&slot| {
                let line = slots.line(pages, slot);
                Some(line.bbox[3] - line.bbox[1]).filter(|height| *height >= 1.0).unwrap_or_else(|| legal_pdf_core::line_font_size(line))
            }).filter(|h| *h >= 1.0).collect();
            heights.sort_by(f64::total_cmp);
            let text_height = heights.get(heights.len() / 2).copied().unwrap_or(10.0);
            // Where the page's lone numbers stand, top to bottom: a paragraph the other language opens rows below its
            // number's row opens before the next number's.
            let mut standing: Vec<f64> = slots.on(page_slot).map(Slot).filter(|&slot| open(slot)).filter_map(|slot| {
                let line = slots.line(pages, slot);
                let text = line.text.trim();
                let text = text.strip_prefix('[').and_then(|rest| rest.strip_suffix(']')).map_or(text, str::trim);
                (!text.is_empty() && text.len() <= 4 && text.bytes().all(|byte| byte.is_ascii_digit())).then_some(line.bbox[1])
            }).collect();
            standing.sort_by(f64::total_cmp);
            for slot in slots.on(page_slot).map(Slot).filter(|&slot| open(slot)) {
                let line = slots.line(pages, slot);
                let text = line.text.trim();
                let text = text.strip_prefix('[').and_then(|rest| rest.strip_suffix(']')).map_or(text, str::trim);
                let Some(number) = (!text.is_empty() && text.len() <= 4 && text.bytes().all(|byte| byte.is_ascii_digit()))
                    .then(|| text.parse::<u32>().ok()).flatten() else { continue };
                // A line a PDF gives no height (its box a baseline alone) stands as high as the page's text, above it.
                let height = if line.bbox[3] - line.bbox[1] >= 1.0 { line.bbox[3] - line.bbox[1] } else { text_height };
                let (left, right) = ink(line);
                let gap = (bounds[0] - right).max(left - bounds[2]);
                let middle_of = |bbox: [f64; 4]| if bbox[3] - bbox[1] >= 1.0 { (bbox[1] + bbox[3]) / 2.0 } else { bbox[3] - text_height / 2.0 };
                let middle = middle_of(line.bbox);
                let level = |other: Slot| (middle_of(slots.line(pages, other).bbox) - middle).abs() <= height * 0.6;
                // Wholly outside the body's edge, tight against the text it numbers (a recognizer's padded box nearly
                // touches a justified column's edge) or well out in the margin; or inside the body at its edge, the
                // paragraph's first line set on to its right ("[2]" at the left margin, its text indented on the row:
                // a word processor's numbered paragraph).
                let margin = gap > 0.0 && gap <= height * 8.0;
                let leading = !margin && line.bbox[0] <= bounds[0] + height
                    && words.iter().any(|&other| level(other) && slots.line(pages, other).bbox[0] >= line.bbox[2]);
                // Inside the column, under a note band, a number at the line's left is that note's.
                if !margin && (!leading || self.band[slot.index()]) {
                    continue;
                }
                // The row's line that opens a paragraph, set in from its column's edge as a first line is (the other
                // language's paragraph may open rows away, its line on this row running on mid-sentence), the main
                // text's before the translation's.
                let indented = |other: Slot| {
                    let x = slots.line(pages, other).bbox[0];
                    // (A justified line's row may be split into its words: they are no edge of one another.)
                    let edge = body.iter().filter(|&&line| !level(line)).map(|&line| slots.line(pages, line).bbox[0]).filter(|edge| *edge <= x && x - edge < height * 3.0).fold(x, f64::min);
                    x - edge > height * 0.4
                };
                // No paragraph opens on a lower-case word ("and dispassionate person in the community."): a stray
                // number beside a line running on is no paragraph's.
                let opens = |other: Slot| indented(other)
                    || slots.line(pages, other).text.trim_start().chars().next().is_some_and(|c| !c.is_lowercase());
                let row = words.iter().copied()
                    .filter(|&other| level(other) && opens(other) && (margin || slots.line(pages, other).bbox[0] >= line.bbox[2]))
                    .min_by_key(|&other| (!indented(other), self.translation[other.index()], self.place[other.index()]));
                if let Some(row) = row {
                    let other_language = |other: Slot| self.translation[other.index()] != self.translation[row.index()];
                    let next = standing.iter().copied().find(|&top| top > line.bbox[1] + height * 0.5).unwrap_or(f64::MAX);
                    let twin = words.iter().copied()
                        .filter(|&other| level(other) && opens(other) && other_language(other))
                        .min_by_key(|&other| (!indented(other), self.place[other.index()]))
                        .or_else(|| words.iter().copied()
                            .filter(|&other| {
                                let top = slots.line(pages, other).bbox[1];
                                other_language(other) && indented(other) && top >= line.bbox[1] - height * 0.6 && top < next
                            })
                            .min_by(|&a, &b| slots.line(pages, a).bbox[1].total_cmp(&slots.line(pages, b).bbox[1])));
                    found.push((number, slot, row, twin));
                }
            }
        }
        found.sort_by_key(|&(_, _, row, _)| self.place[row.index()]);
        let counts = |at: usize, step: usize| at + step < found.len() && found[at + step].0 == found[at].0 + step as u32;
        let mut kept = vec![false; found.len()];
        for at in 0..found.len() {
            if counts(at, 1) && counts(at, 2) {
                kept[at..at + 3].iter_mut().for_each(|kept| *kept = true);
            }
        }
        // A number belongs to the text it numbers: its language and document are its row line's (a number beside a
        // parallel report's French column numbers the English paragraph opening on its row as well).
        for (&(_, margin, row, twin), _) in found.iter().zip(&kept).filter(|(_, kept)| **kept) {
            self.margin_number[margin.index()] = Some(row);
            self.margin_twin[margin.index()] = twin;
            self.translation[margin.index()] = self.translation[row.index()];
            self.document[margin.index()] = self.document[row.index()];
        }
        if !kept.contains(&true) {
            return;
        }
        let mut before: HashMap<Slot, Vec<Slot>> = HashMap::new();
        for (slot, row) in self.margin_number.iter().enumerate().filter_map(|(slot, row)| row.map(|row| (Slot(slot as u32), row))) {
            before.entry(row).or_default().push(slot);
        }
        let order = std::mem::take(&mut self.order);
        for slot in order.into_iter().filter(|slot| self.margin_number[slot.index()].is_none()) {
            self.order.extend(before.remove(&slot).into_iter().flatten());
            self.order.push(slot);
        }
        for (place, slot) in self.order.iter().enumerate() {
            self.place[slot.index()] = place as u32;
        }
    }

    /// The document's numbered paragraphs, read once, before notes, lists or headings claim a line. Each language's
    /// body is written as a provider writes a decision, a block to a line: a block the lines `read_continuations`
    /// chains, a marker standing alone on its line set before the words of its row (`Evidence::beside`; a margin
    /// number before its row in both languages); the page's furniture, labels, contents and notes (smaller type under
    /// the band, or opening on a note's label) are no part of it. The shared reading of a
    /// decision's text (`judgment_paragraphs`) reads it, and its paragraphs come back to the lines as `count` and `held`.
    fn read_paragraphs(&mut self, pages: &[Page], slots: &Slots, evidence: &Evidence) {
        self.count = vec![None; slots.len()];
        self.held = vec![Vec::new(); slots.len()];
        // A figure in far smaller type than the body is a note's label, however it lines up: a page's labels count on as
        // margin numbers do, and a scan's layer boxes its lines alike. Beside a line in the body's own type it numbers that
        // line: a tribunal's word processor sets its paragraph numbers small.
        let label_size = self.body.map_or(0.0, |(_, size)| f64::from(size) * 0.75);
        let in_body_type = |row: Slot| self.body.is_some_and(|(_, size)| evidence.half_points[row.index()] == size);
        let note_label = |at: usize, row: Option<Slot>| evidence.size[at] * 2.0 < label_size && !row.is_some_and(in_body_type);
        for at in 0..slots.len() {
            if note_label(at, self.margin_number[at]) {
                self.margin_number[at] = None;
                self.margin_twin[at] = None;
            }
        }
        let analysis = legal_structure_model::structure_analysis();
        let layout = legal_pdf_core::layout_analysis();
        // A judgment in one language may quote a statute in two, its columns side by side: the other language numbers
        // nothing, and its lines are the paragraph's quoting them. A parallel report prints its paragraphs in both.
        let numbered_apart = |at: usize| self.margin_twin[at].is_some() || self.translation[at] && (self.margin_number[at].is_some()
            || layout.printed_paragraph_number(&slots.line(pages, Slot(at as u32)).text).is_some());
        let translated = (0..slots.len()).filter(|&at| self.translation[at] && self.role[at] == Role::Open).count();
        let open = (0..slots.len()).filter(|&at| self.role[at] == Role::Open).count();
        // (A quoted section's own number numbers no count: what counts is how few the other language prints.)
        let printed_number = |at: usize| self.margin_number[at].is_some() || layout.printed_paragraph_number(&slots.line(pages, Slot(at as u32)).text).is_some();
        let (apart, own) = ((0..slots.len()).filter(|&at| numbered_apart(at)).count(), (0..slots.len()).filter(|&at| !self.translation[at] && printed_number(at)).count());
        let one_text = translated * 4 < open && apart * 4 < own;
        let language_of: Vec<bool> = self.translation.iter().map(|&translation| translation && !one_text).collect();
        // How much further in than the judgment's own numbers a quotation's number stands: more than a few characters.
        const QUOTED_INDENT: f64 = 9.0;
        // The most letters and figures a recognizer's reading of a folio holds.
        const FOLIO_MARKS: usize = 4;
        // How far a note's label may pass the last, its lost labels between.
        const NOTE_LOST: u32 = 3;
        // The most lines a heading over a numbered paragraph runs to.
        const CLOSING_LINES: usize = 4;
        let set_apart =|at: usize| matches!(self.role[at], Role::LineLabel | Role::PageLabel | Role::Furniture | Role::Contents | Role::Note);
        // Each row's marker, standing alone on its line: the margin's numbers by their rows, other markers by theirs.
        let mut marker_of = HashMap::<(Slot, bool), (Slot, bool)>::new();
        for at in 0..slots.len() {
            let marker = Slot(at as u32);
            // (A marker takes one row: the margin's, where it has one.)
            let rows = [self.margin_number[at], self.margin_twin[at], evidence.beside[at].filter(|&row| !note_label(at, Some(row)) && self.margin_number[at].is_none())];
            for (k, row) in rows.into_iter().enumerate().filter_map(|(k, row)| Some((k, row?))) {
                // A marker's row is the line beside it in its own language (the nearest line on a parallel report's
                // row may be the other column's); only a margin number serves the other language's row.
                if set_apart(row.index()) || k != 1 && self.translation[row.index()] != self.translation[at] {
                    continue;
                }
                // The twin row's language takes the margin's number as text the stream adds, not as its line.
                marker_of.entry((row, language_of[row.index()])).or_insert((marker, k == 1));
            }
        }
        let attached: HashSet<Slot> = marker_of.values().filter(|(_, added)| !added).map(|(marker, _)| *marker).collect();
        // The body's type, and where on each page its last running line of that type ends (a folio or a running
        // citation, a few words, is none): a page's notes stand below it.
        let below_body = |at: usize, share: f64| evidence.size[at] > 0.0 && self.margin_number[at].is_none() && self.body.is_some_and(|(_, size)| evidence.size[at] * 2.0 < f64::from(size) * share);
        let small = |at: usize| below_body(at, 0.9);
        let smaller = |at: usize| below_body(at, 0.95);
        // (Where a page's note band opens: small type above it is a quotation, whatever it numbers.)
        let (mut body_end, mut band_top) = (vec![[f64::MIN; 2]; pages.len()], vec![[f64::MAX; 2]; pages.len()]);
        for &slot in &self.order {
            let (page, language) = (slots.page(slot), usize::from(self.translation[slot.index()]));
            if self.role[slot.index()] == Role::Open && !smaller(slot.index()) && slots.line(pages, slot).text.split_whitespace().count() > 3 {
                body_end[page][language] = body_end[page][language].max(slots.line(pages, slot).bbox[3]);
            }
            if self.band[slot.index()] {
                band_top[page][language] = band_top[page][language].min(slots.line(pages, slot).bbox[1]);
            }
        }
        // Development only: each line's fence and type, beside the streams.
        if let Some(path) = std::env::var_os("LEGALPDF_JUDGMENT_STREAM") {
            let lines = self.order.iter().map(|&slot| {
                let at = slot.index();
                format!("{:?}\t{}\t{}\t{:.1}\t{:?}\t{}\n", self.role[at], self.band[at], self.translation[at], evidence.size[at], self.body, slots.line(pages, slot).text)
            }).collect::<String>();
            let _ = std::fs::write(std::path::PathBuf::from(path).with_extension("lines.txt"), lines);
        }
        // Where the judgment's own numbers print, in each language: the place, from the left edge of its page's body,
        // where most of its lines that open on a printed paragraph number open. A number printed further in is a
        // quotation's: its line runs on in the paragraph quoting it rather than opening one.
        let body = |at: usize| self.role[at] == Role::Open && !self.band[at] && !set_apart(at);
        let mut edge = HashMap::<(usize, bool), f64>::new();
        for &slot in self.order.iter().filter(|slot| body(slot.index())) {
            let left = edge.entry((slots.page(slot), self.translation[slot.index()])).or_insert(f64::MAX);
            *left = left.min(slots.line(pages, slot).bbox[0]);
        }
        // Where the document's running text stands on a page: most of its lines of words open there, odd pages and even apart.
        let mut lefts = HashMap::<(usize, bool), BTreeMap<i64, usize>>::new();
        for &slot in self.order.iter().filter(|slot| body(slot.index())) {
            let line = slots.line(pages, slot);
            if line.text.split_whitespace().count() >= 5 {
                *lefts.entry((slots.page(slot) % 2, self.translation[slot.index()])).or_default().entry((line.bbox[0] / 3.0).round() as i64).or_default() += 1;
            }
        }
        let column = |page: usize, language: bool| lefts.get(&(page % 2, language)).and_then(|lefts| lefts.iter().max_by_key(|(_, count)| **count)).map_or(f64::MAX, |(left, _)| *left as f64 * 3.0);
        // Where a numbered line's words stand, from that edge: a quotation may hang its number in the judgment's own margin
        // (its words alone set in); a number alone on its line has them on its row.
        let indent = |slot: Slot| {
            let line = slots.line(pages, slot);
            let row = self.margin_number[slot.index()].or(evidence.beside[slot.index()]);
            let words = match row {
                Some(row) if evidence.marker[slot.index()].is_some_and(|reading| reading.only) => slots.line(pages, row).bbox[0],
                _ => line.words.get(1).map_or(line.bbox[0], |word| word.bbox[0]),
            };
            words - edge.get(&(slots.page(slot), self.translation[slot.index()])).copied().unwrap_or(0.0)
        };
        let numbered: Vec<Slot> = self.order.iter().copied()
            .filter(|slot| body(slot.index()) && layout.printed_paragraph_number(&slots.line(pages, *slot).text).is_some())
            .collect();
        let mut places = BTreeMap::<(bool, i64), usize>::new();
        for &slot in &numbered {
            *places.entry((self.translation[slot.index()], (indent(slot) / 3.0).round() as i64)).or_default() += 1;
        }
        let place = |translation: bool| places.iter().filter(|((own, _), _)| *own == translation).max_by_key(|(_, count)| **count).map(|((_, at), _)| *at as f64 * 3.0);
        // Words set further in than the judgment's are a quotation's. Words closer to their number say nothing: the judgment's
        // own paragraphs may print so among quotations that print as it does (`quoted_runs` reads those by their counts).
        let further = |slot: Slot| place(self.translation[slot.index()]).is_some_and(|own| indent(slot) - own > QUOTED_INDENT);
        let number_of = |slot: Slot| (self.translation[slot.index()], layout.printed_paragraph_number(&slots.line(pages, slot).text));
        // A number no line at the judgment's own place prints is the judgment's, however it stands.
        let standing: HashSet<(bool, Option<usize>)> = numbered.iter().copied().filter(|&slot| !further(slot)).map(number_of).collect();
        let indented: HashSet<Slot> = numbered.iter().copied().filter(|&slot| further(slot) && standing.contains(&number_of(slot))).collect();
        // Where each page's text starts and ends.
        let mut page_ends = vec![(f64::MAX, f64::MIN); pages.len()];
        for &slot in &self.order {
            let (ends, bbox) = (&mut page_ends[slots.page(slot)], slots.line(pages, slot).bbox);
            *ends = (ends.0.min(bbox[1]), ends.1.max(bbox[3]));
        }
        let running: HashSet<&str> = self.order.iter().filter(|slot| self.role[slot.index()] == Role::Furniture).map(|&slot| slots.line(pages, slot).text.trim()).collect();
        // Each number alone on its line and the line of words it opens.
        let mut led = Vec::<(Slot, Slot)>::new();
        // A new count opens at a "1" in the body's type and at the judgment's own place, five or more numbers on, that goes
        // on 2, 3, 4, 5: another decision in the file. A note list's "[1]" is small, a quotation's stands elsewhere, a
        // list's stops short.
        let (mut count_of, mut printed_numbers, mut counts) = (vec![[0u16; 2]; slots.len()], [0usize; 2], [0u16; 2]);
        let sequence: [Vec<(Slot, Option<usize>)>; 2] = [false, true].map(|language| {
            numbered.iter().copied().filter(|slot| language_of[slot.index()] == language)
                .map(|slot| (slot, layout.printed_paragraph_number(&slots.line(pages, slot).text))).collect()
        });
        let place_in: HashMap<Slot, usize> = sequence.iter().flat_map(|numbers| numbers.iter().enumerate().map(|(place, (slot, _))| (*slot, place))).collect();
        // Notes listed after the reasons in the body's own type open no decision: the reasons' lines call their numbers in
        // turn ("... of violence.[1]"), and the list is no one's paragraphs.
        let called = |upto: usize, language: bool| {
            let mut calls = [1, 2, 3].map(|number| format!("[{number}]")).into_iter().peekable();
            for slot in self.order[..upto].iter().filter(|slot| language_of[slot.index()] == language) {
                let line = slots.line(pages, *slot).text.trim();
                while calls.peek().is_some_and(|call| line.find(call.as_str()).is_some_and(|at| at > 0)) {
                    calls.next();
                }
            }
            calls.peek().is_none()
        };
        let mut note_lists = HashSet::<(u16, bool)>::new();
        for (reading_place, &slot) in self.order.iter().enumerate() {
            let (at, language) = (slot.index(), usize::from(language_of[slot.index()]));
            if let Some(&place) = place_in.get(&slot) {
                let numbers = &sequence[language];
                let goes_on = (1..5).all(|step| numbers.get(place + step).is_some_and(|(_, number)| *number == Some(step + 1)));
                if numbers[place].1 == Some(1) && goes_on && !smaller(at) && !further(slot) && printed_numbers[language] >= 5 {
                    counts[language] += 1;
                    printed_numbers[language] = 0;
                    if called(reading_place, language_of[at]) {
                        note_lists.insert((counts[language], language_of[at]));
                    }
                }
                printed_numbers[language] += 1;
            }
            count_of[at][language] = counts[language];
        }
        let streams = counts.iter().copied().max().unwrap_or(0) + 1;
        for (document, translation) in (0..streams).flat_map(|count| [(count, false), (count, true)]).filter(|stream| !note_lists.contains(stream)) {
            // The stream, and each of its UTF-16 units' line and place in that line's text (none for what the
            // stream adds: breaks, a margin number set before the other language's row).
            let (mut text, mut place) = (Vec::<u16>::new(), Vec::<Option<(Slot, u32)>>::new());
            let write = |text: &mut Vec<u16>, place: &mut Vec<Option<(Slot, u32)>>, words: &str, line: Option<Slot>| {
                for (offset, unit) in words.encode_utf16().enumerate() {
                    text.push(if unit == 10 { 32 } else { unit });
                    place.push(line.map(|line| (line, offset as u32)));
                }
            };
            let (mut kept, mut in_note, mut last_note, mut note_open) = (Vec::<Slot>::new(), false, 0u32, false);
            // The type and page of the last note's last line: a note carried over is set as it was.
            let (mut note_size, mut note_at) = (0.0f64, None::<usize>);
            // The page a quoted list a colon opened runs on: its next numbers are no notes.
            let mut quoted_list = None::<usize>;
            // The note a labelled line leaves the count at: a figure opening a line of a note going on is its words ("2012 ABQB 375").
            let noted = |last: u32, open: bool, value: u32| if !open || value == 1 || (last + 1..=last + NOTE_LOST).contains(&value) { value } else { last };
            let mut note_page = 0;
            for (reading_place, &slot) in self.order.iter().enumerate() {
                let at = slot.index();
                // The notes a fence took keep the count, and may go on onto the next page: a note left among the words goes on from them.
                if self.role[at] == Role::Note && language_of[at] == translation {
                    last_note = evidence.label[at].map_or(last_note, |label| noted(last_note, note_open, label.value));
                    note_open = !sentence_ended(&slots.line(pages, slot).text);
                    (note_size, note_at) = (evidence.size[at], Some(slots.page(slot)));
                }
                if count_of[at][usize::from(translation)] != document || language_of[at] != translation || set_apart(at) || attached.contains(&slot) {
                    continue;
                }
                let line = &slots.line(pages, slot).text;
                // A figure in the margin beside no words, and out of the count's turn, numbers nothing: a heading's call on a row of its own.
                let out_of_turn = place_in.get(&slot).is_some_and(|&place| {
                    let numbers = &sequence[usize::from(language_of[at])];
                    let here = numbers[place].1;
                    let last = place.checked_sub(1).and_then(|before| numbers[before].1);
                    let next = numbers.get(place + 1).and_then(|(_, number)| *number);
                    here != last.map(|number| number + 1) && next != here.map(|number| number + 1)
                });
                if out_of_turn && evidence.beside[at].is_none() && self.margin_number[at].is_none() && self.margin_twin[at].is_none()
                    && evidence.marker[at].is_some_and(|reading| reading.only) && slots.line(pages, slot).bbox[2] < column(slots.page(slot), self.translation[at]) {
                    continue;
                }
                // A folio that lost its figures ("-  -") is no one's words; nor is a recognizer's reading of one, a few marks
                // at the very top or foot of its page ("4-", "=f>~").
                if !line.chars().any(char::is_alphanumeric) && line.chars().all(|c| c.is_whitespace() || matches!(c, '-' | '\u{2013}' | '\u{2014}')) {
                    continue;
                }
                let (bbox, (top, foot)) = (slots.line(pages, slot).bbox, page_ends[slots.page(slot)]);
                // (Set apart from the page's lines: a sentence's last word may stand alone at its foot, "thew", "692,".)
                let neighbour = |step: isize| self.order.get(reading_place.wrapping_add_signed(step)).filter(|other| slots.page(**other) == slots.page(slot)).map(|other| slots.line(pages, *other).bbox);
                let height = bbox[3] - bbox[1];
                let apart = if bbox[3] >= foot { neighbour(-1).is_none_or(|above| bbox[1] - above[3] > height) } else { neighbour(1).is_none_or(|below| below[1] - bbox[3] > height) };
                if slots.line(pages, slot).recognized() && (bbox[1] <= top || bbox[3] >= foot) && apart && line.chars().filter(|c| c.is_alphanumeric()).count() <= FOLIO_MARKS {
                    continue;
                }
                // A folio in words ("Page: 3"), and what the page's furniture prints elsewhere, are the page's.
                let edge_line = bbox[1] <= top + (bbox[3] - bbox[1]) * 3.0 || bbox[3] >= foot - (bbox[3] - bbox[1]) * 3.0;
                let folio = line.trim().strip_prefix("Page").or_else(|| line.trim().strip_prefix("PAGE")).is_some_and(|rest| {
                    let rest = rest.trim_start_matches([':', ' ']);
                    !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit())
                });
                if edge_line && (folio || running.contains(line.trim())) {
                    continue;
                }
                // A note is known by its smaller type: small under the page's note band; or, wherever the band stands,
                // set smaller than the body and opening on a note's label below the page's last line in the body's type,
                // or carrying such a note on, its next line or the lines below the body after it (a label standing alone
                // on its line, its words on the next). A quotation is set small too, among the body's lines, and is the
                // paragraph's.
                let (top, language) = (slots.line(pages, slot).bbox[1], usize::from(translation));
                let band = band_top[slots.page(slot)][language];
                let below = top >= body_end[slots.page(slot)][language] - 1.0 && (band == f64::MAX || top >= band);
                // A note goes on over a page only as `carried`.
                in_note &= slots.page(slot) == note_page;
                note_page = slots.page(slot);
                // Notes count on from 1, each page's or the document's: a number that goes on with no count is a small
                // quotation's at the page's foot ("75 Along the same lines", "2 In this Act,").
                // A figure alone, far smaller than the body, is a label a scan's layer set on its own line.
                let lone_figure = || line.trim().parse::<u32>().ok().filter(|_| evidence.size[at] * 2.0 < label_size);
                let label = evidence.label[at].filter(|label| label.suppress == 0).map(|label| label.value).or_else(lone_figure);
                // (Set in a note's type, or raised: a quotation's own numbers run on at the foot of a page in its larger one.)
                // (Or a tab after its label: "1\tHighway 5 runs ...", some reports' notes set little under the body.)
                let tabbed = smaller(at) && line.trim_start().trim_start_matches(|c: char| c.is_ascii_digit()).starts_with('\t');
                let set_as_note = below_body(at, 0.85) || tabbed || evidence.label[at].is_some_and(|label| label.raised);
                // (A colon over it sets a quotation out: "... as follows:" / "1. The party ...".)
                // (A note may stand under one all the same: its label a lone small figure, or raised.)
                let lead_in = kept.last().is_some_and(|last| slots.line(pages, *last).text.trim_end().ends_with(':')) && lone_figure().is_none() && !evidence.label[at].is_some_and(|label| label.raised);
                if lead_in && label.is_some() {
                    quoted_list = Some(slots.page(slot));
                }
                let lead_in = lead_in || quoted_list == Some(slots.page(slot));
                // (A table's cell is no note's label.)
                let lead_in = lead_in || self.role[at] == Role::Table;
                let labelled = below && set_as_note && !lead_in && label.is_some_and(|value| value == 1 || (last_note + 1..=last_note + NOTE_LOST).contains(&value));
                // A scan's layer may read a label standing alone after its note's first line.
                let label_after = self.order.get(reading_place + 1).is_some_and(|next| {
                    evidence.size[next.index()] * 2.0 < label_size && evidence.label[next.index()].is_some() && slots.line(pages, *next).text.trim().chars().count() <= 4
                });
                let labelled = labelled || below && label_after && set_as_note && !lead_in;
                // Under the page's rule a label makes a note in any type: some set their notes in the body's.
                // A note a page's foot left unfinished goes on at the next page's, with no label, in its own type: a quotation
                // stands smaller than the body too, "Immunity waived".
                let carried = note_open && smaller(at) && below && (evidence.size[at] - note_size).abs() <= 0.5 && note_at.is_some_and(|page| page + 1 == slots.page(slot));
                in_note = self.band[at] && (small(at) || label.is_some() || in_note)
                    || smaller(at) && (labelled || in_note && (self.continues[at].is_some() || below)) || carried;
                if in_note {
                    last_note = label.map_or(last_note, |value| noted(last_note, note_open, value));
                    // (A note's line stopping on "s." goes on.)
                    note_open = !sentence_ended(line) || [" s.", " ss.", " para.", " paras.", " p.", " pp."].iter().any(|shortened| line.trim_end().ends_with(shortened));
                    (note_size, note_at) = (evidence.size[at], Some(slots.page(slot)));
                } else if evidence.size[at] * 2.0 >= label_size || evidence.size[at] == 0.0 || analysis.editorial_line(line)
                    || slots.line(pages, slot).recognized() && (!edge_line || kept.last().is_some_and(|last| !sentence_ended(&slots.line(pages, *last).text))) {
                    // (Words far smaller than the body and in no note are a page's aside, "OUR VISION...": but a size no one
                    // measured, an editor's "[Emphasis added]", and on a scan a line among the page's lines, whose layer boxes
                    // lines by their letters, are the text's.)
                    // (A letterhead, far larger than the body, is no one's: "Federal Court" / "of Appeal" at twice its type.)
                    let wordmark = self.body.is_some_and(|(_, size)| evidence.size[at] * 2.0 >= f64::from(size) * 1.6) && line.trim().chars().count() > 1 && line.split_whitespace().count() <= 6;
                    // (A layer may print one line twice, the second a suffix of the first in its box.)
                    let repeated = kept.last().is_some_and(|last| {
                        let above = slots.line(pages, *last);
                        slots.page(*last) == slots.page(slot) && (above.bbox[1] - bbox[1]).abs() <= 2.0 && line.trim().chars().count() >= 12 && above.text.trim_end().ends_with(line.trim())
                    });
                    if !wordmark && !repeated {
                        kept.push(slot);
                    }
                }
            }
            // A table's reading may take prose set about it out of its order ("[4] This appeal ..." under its own next lines):
            // what a table holds reads down its page.
            let mut from = 0;
            while from < kept.len() {
                let page = slots.page(kept[from]);
                let to = from + kept[from..].iter().take_while(|slot| self.role[slot.index()] == Role::Table && slots.page(**slot) == page).count();
                kept[from..to].sort_by(|a, b| {
                    let (a, b) = (slots.line(pages, *a).bbox, slots.line(pages, *b).bbox);
                    (a[1] / 2.0).round().total_cmp(&(b[1] / 2.0).round()).then(a[0].total_cmp(&b[0]))
                });
                from = to.max(from + 1);
            }
            // A layer may split a line and set its tail first: the tail's top stands a little above its line's.
            for at in 1..kept.len() {
                let (tail, opening) = (slots.line(pages, kept[at - 1]).bbox, slots.line(pages, kept[at]).bbox);
                let shared = tail[3].min(opening[3]) - tail[1].max(opening[1]);
                // (A scan's pieces may overlap by a few points.)
                let overlap = if slots.line(pages, kept[at]).recognized() { opening[3] - opening[1] } else { 0.0 };
                if slots.page(kept[at - 1]) == slots.page(kept[at]) && opening[2] <= tail[0] + overlap && shared * 2.0 > (tail[3] - tail[1]).min(opening[3] - opening[1]) {
                    kept.swap(at - 1, at);
                }
            }
            // The printed paragraph each line stands in, as the page reader reads a page (its own leading, first lines
            // set in, a short last line, a scan's blanks) and a paragraph over a page (`continues_onto`).
            let mut runs = Vec::<&[Slot]>::new();
            let mut from = 0;
            for at in 1..=kept.len() {
                if at == kept.len() || slots.page(kept[at]) != slots.page(kept[from]) {
                    runs.push(&kept[from..at]);
                    from = at;
                }
            }
            let read: Vec<Vec<&Line>> = runs.iter().map(|run| run.iter().map(|&slot| slots.line(pages, slot)).collect()).collect();
            let measure = crate::paragraphs::measure_of(&read);
            let mut printed = HashMap::<Slot, (usize, usize)>::new();
            let mut flows = HashSet::<Slot>::new();
            for (k, run) in runs.iter().enumerate() {
                let found = crate::paragraphs::paragraphs_of(&read[k], &vec![true; run.len()], measure);
                printed.extend(run.iter().zip(&found).filter_map(|(&slot, paragraph)| Some((slot, (k, (*paragraph)?)))));
                if k > 0 && crate::paragraphs::continues_onto(&read[k - 1], &read[k]) {
                    flows.insert(run[0]);
                }
            }
            // How many lines each line stands before the next that opens on a number (none after the last).
            let mut to_number = vec![usize::MAX; kept.len()];
            let mut number = None::<usize>;
            for at in (0..kept.len()).rev() {
                if evidence.marker[kept[at].index()].is_some() || marker_of.contains_key(&(kept[at], translation)) {
                    number = Some(at);
                }
                to_number[at] = number.map_or(usize::MAX, |number| number - at);
            }
            // A first paragraph's number fused to its judge's drop capital: "1I" / "ACOBUCCI J. (dissenting) -- ...".
            let drop_capital = |at: usize| {
                let words = slots.line(pages, kept[at]).text.trim();
                let mut letters = words.chars();
                letters.next() == Some('1') && letters.next().is_some_and(char::is_uppercase) && letters.next().is_none()
                    && kept.get(at + 1).is_some_and(|next| slots.line(pages, *next).text.chars().filter(|c| !c.is_whitespace()).take(3).filter(|c| c.is_uppercase()).count() == 3)
            };
            let (mut previous, mut last_advance) = (None::<Slot>, 0.0f64);
            for (place_in_kept, &slot) in kept.iter().enumerate() {
                let at = slot.index();
                let line = &slots.line(pages, slot).text;
                let marker = marker_of.get(&(slot, translation)).copied();
                if let Some(before) = previous {
                    let (above, here) = (slots.line(pages, before), slots.line(pages, slot));
                    // A marker standing alone on its line opens the line after it, whatever its row, a line of words set
                    // no further left: "2025." ending a wrapped sentence opens no heading at the margin.
                    // (Figures only: a heading's "I." alone in a gutter opens the heading, not the next line read.)
                    let figures = above.text.trim().chars().any(|c| c.is_ascii_digit());
                    // (Nor does a range's end wrapped over its line: "at paras. 11-" / "13.".)
                    let wrapped = place_in_kept >= 2 && slots.line(pages, kept[place_in_kept - 2]).text.trim_end().ends_with(['-', '\u{2013}']);
                    let alone = figures && !wrapped && evidence.marker[before.index()].is_some_and(|reading| reading.only) && evidence.marker[at].is_none()
                        && above.bbox[0] <= here.bbox[0] + (here.bbox[3] - here.bbox[1]);
                    // A line goes on with the printed paragraph it stands in, unless it opens on a marker of its own.
                    let same = evidence.marker[at].is_none() && (flows.contains(&slot) || printed.get(&slot).is_some_and(|paragraph| printed.get(&before) == Some(paragraph)));
                    // The last few lines before a number, under a line ending a sentence, stand on their own, whatever the
                    // page's geometry says: whether they head what follows is the reader's to say (`trailing_headings`).
                    // A line in the body's own type ending as a sentence or a lead-in does is the paragraph's words, and so are the
                    // lines of a sentence that wraps there ("Finally, would admission of the evidence bring the" / "administration of
                    // justice into disrepute?").
                    let in_body = |slot: &&Slot| self.body == Some((evidence.face[slot.index()], evidence.half_points[slot.index()]));
                    let stops = |slot: &Slot| slots.line(pages, *slot).text.trim_end().trim_end_matches([')', '"', '\u{201d}', '\u{bb}']).ends_with(['.', '?', '!', ':']);
                    let plain = kept[place_in_kept..].iter().take(CLOSING_LINES.min(to_number[place_in_kept])).take_while(in_body).any(stops);
                    let closes = sentence_ended(&above.text) || above.text.trim_end().ends_with([')', ']']);
                    let apart = closes && !plain && to_number[place_in_kept] <= CLOSING_LINES;
                    // After the last number a line at the same place and leading goes on with the paragraph, however short
                    // the line above it stopped: only a real break lets the last paragraph end before the signatures.
                    let advance = here.bbox[1] - above.bbox[1];
                    let level = (here.bbox[0] - above.bbox[0]).abs() <= here.bbox[3] - here.bbox[1];
                    let closing = analysis.signature_line(line) || analysis.end_matter(line) || analysis.author_line(line) || analysis.date_line(line) || analysis.disposition_line(line);
                    let last_goes_on = to_number[place_in_kept] == usize::MAX && level && !closing && last_advance > 0.0 && advance > 0.0 && advance <= last_advance * 1.15;
                    let same = same || last_goes_on;
                    if level && advance > 0.0 {
                        last_advance = advance;
                    }
                    // The lines closing the last paragraph stand on their own, for the reader to end it there.
                    // (Under a sentence's end they do wherever they stand: a schedule's numbers may follow.)
                    // (An author's line stands on its own wherever it does: a column's last word may have gone over the gutter.)
                    // (A signer's name over the title under it too: "... signée en son nom par Claude Roy" / "Vice-president".)
                    let signer = line.split_whitespace().count() <= 4 && line.split_whitespace().all(|word| word.starts_with(char::is_uppercase)) && !line.trim_end().ends_with('.') && kept.get(place_in_kept + 1).is_some_and(|next| analysis.signature_line(&slots.line(pages, *next).text));
                    let apart = apart || (closing || signer) && (to_number[place_in_kept] == usize::MAX || closes) || analysis.author_line(line);
                    let fused = place_in_kept > 0 && drop_capital(place_in_kept - 1);
                    let joined = marker.is_none() && (alone || fused || !apart && (same || self.continues[at].is_some())) || indented.contains(&slot);
                    if alone && marker.is_none() {
                        led.push((before, slot));
                    }
                    // A break between blocks, which a line's own text never holds (`write` makes its breaks spaces).
                    text.push(if joined { 32 } else { 10 });
                    place.push(None);
                }
                previous = Some(slot);
                if let Some((marker, added)) = marker {
                    write(&mut text, &mut place, slots.line(pages, marker).text.trim_end(), (!added).then_some(marker));
                    write(&mut text, &mut place, " ", None);
                }
                if drop_capital(place_in_kept) {
                    let (figure, capital) = line.trim().split_at(1);
                    write(&mut text, &mut place, figure, Some(slot));
                    write(&mut text, &mut place, " ", None);
                    write(&mut text, &mut place, capital, Some(slot));
                } else {
                    write(&mut text, &mut place, line, Some(slot));
                }
            }
            let text = String::from_utf16_lossy(&text);
            if text.trim().is_empty() {
                continue;
            }
            // Development only: the stream the reader takes.
            if let Some(path) = std::env::var_os("LEGALPDF_JUDGMENT_STREAM") {
                let _ = std::fs::write(std::path::PathBuf::from(path).with_extension(if translation { "fr.txt" } else { "txt" }), &text);
            }
            let kept_at: HashMap<Slot, usize> = kept.iter().enumerate().map(|(at, slot)| (*slot, at)).collect();
            for (number, ranges) in analysis.judgment_paragraphs(&text) {
                let number = number as u32;
                let mut spans = BTreeMap::<Slot, (u32, u32)>::new();
                let mut first = None;
                // A number alone on its line, before the words its range opens at, is the paragraph's.
                let opening = ranges.first().and_then(|range| place[..range.start.min(place.len())].iter().rev().flatten().next().copied());
                for range in ranges {
                    for &(slot, offset) in place.get(range.start..range.end.min(place.len())).unwrap_or_default().iter().flatten() {
                        first.get_or_insert(slot);
                        let span = spans.entry(slot).or_insert((offset, offset + 1));
                        *span = (span.0.min(offset), span.1.max(offset + 1));
                    }
                }
                if let Some((slot, _)) = opening {
                    let words = slots.line(pages, slot).text.trim();
                    let digits = words.trim_matches(['[', ']', '(', ')', '.', ' ']);
                    // (Unless the paragraph before holds it: "101." ending a citation just above the number's row.)
                    if digits.parse::<u32>().ok() == Some(number) && self.held[slot.index()].is_empty() {
                        spans.entry(slot).or_insert((0, words.encode_utf16().count() as u32));
                    }
                }
                // A bold or underlined line at the margin, under a sentence's end, heads what follows, its stop or not ("Impact on
                // the Community."): the text alone cannot say so, and the paragraph ends there. A heading set in a quotation
                // stands further in. (A quoted paragraph's bold number, alone on its line, heads nothing: a heading has words.)
                let marked = |slot: Slot, share: f64| {
                    let words = &slots.line(pages, slot).words;
                    legal_pdf_core::flag_share(slots.line(pages, slot), legal_pdf_core::BOLD).is_some_and(|bold| bold > share) || !words.is_empty() && words.iter().all(|word| word.underlined)
                };
                let held_lines: Vec<Slot> = spans.keys().copied().collect();
                // (A bold line under an elision is a quotation's emphasis.)
                let elided = |text: &str| matches!(text.trim(), "..." | "\u{2026}" | "[...]" | "[\u{2026}]" | ". . .");
                let heading = held_lines.windows(2).enumerate().position(|(at, pair)| {
                    let (before, line) = (slots.line(pages, pair[0]), slots.line(pages, pair[1]));
                    // (A page all quotation has its edge there: the margin is no further in than the document's running text.)
                    let (page, language) = (slots.page(pair[1]), self.translation[pair[1].index()]);
                    let margin = edge.get(&(page, language)).copied().unwrap_or(0.0).min(column(page, language));
                    // (What a heading in a quotation heads stands further in, or smaller: an enactment's "Prohibited grounds". A
                    // table's head stands over figures: "Township" / "2012".)
                    // (An enactment's marginal note stands over its section's number, its own lines bold or not: "Burden of proof" /
                    // "(3) The burden ...", "Publication, etc., of an" / "intimate image without" / "consent" / "162.1 (1) ...".)
                    let under_heading = held_lines[(at + 2).min(held_lines.len())..].iter().find(|&&under| !marked(under, 0.8)).copied();
                    let section = under_heading.is_some_and(|under| {
                        let words = slots.line(pages, under).text.trim_start().trim_start_matches('(');
                        words.starts_with(|c: char| c.is_ascii_digit()) && words.split_whitespace().next().is_some_and(|figures| figures.chars().filter(char::is_ascii_digit).count() >= figures.chars().filter(|c| c.is_alphabetic()).count())
                    });
                    let body_under = held_lines.get(at + 2).is_none_or(|&under| !smaller(under.index()) && slots.line(pages, under).bbox[0] - line.bbox[0] <= QUOTED_INDENT
                        && slots.line(pages, under).text.chars().any(char::is_alphabetic)
                        // (A run-in label's sentence goes on under it: "Video #3 – Published October 13, 2021 – This 12" / "second video ...".)
                        && (marked(under, 0.5) || !slots.line(pages, under).text.trim_start().starts_with(char::is_lowercase)));
                    // (A bracket left open goes on: "Les séances du conseil doivent être publiques (art. 322".)
                    marked(pair[1], 0.8) && !marked(pair[0], 0.2) && line.text.split_whitespace().count() <= 12 && line.text.chars().any(char::is_alphabetic)
                        && sentence_ended(&before.text) && !elided(&before.text) && line.bbox[0] - margin <= QUOTED_INDENT && body_under && !section
                        && line.text.matches('(').count() <= line.text.matches(')').count()
                });
                if let Some(heading) = heading {
                    for line in &held_lines[heading + 1..] {
                        spans.remove(line);
                    }
                }
                // All in bold from a line at the margin or on an enumerator to the paragraph's end, over a line that is not, is a
                // heading whatever its length or the stop above it. (An italic last line under a sentence's end too.)
                let tail = (1..held_lines.len()).find(|&from| {
                    let (run, line) = (&held_lines[from..], slots.line(pages, held_lines[from]));
                    let (page, language) = (slots.page(held_lines[from]), self.translation[held_lines[from].index()]);
                    let margin = edge.get(&(page, language)).copied().unwrap_or(0.0).min(column(page, language));
                    let enumerator = line.text.split_whitespace().next().is_some_and(|token| token.chars().count() <= 5 && token.ends_with(['.', ')']));
                    let above = slots.line(pages, held_lines[from - 1]);
                    // (A heading in italics ends on no stop: "Basi, supra." is the paragraph's source.)
                    let italic = run.len() == 1 && !line.text.trim_end().ends_with('.') && legal_pdf_core::flag_share(line, legal_pdf_core::ITALIC).is_some_and(|share| share > 0.8) && sentence_ended(&above.text);
                    let bold = |slot: Slot| legal_pdf_core::flag_share(slots.line(pages, slot), legal_pdf_core::BOLD).is_some_and(|bold| bold > 0.8);
                    // (Emphasis running over a line is the sentence's: "... (b) a deprivation or risk of" / "deprivation.".)
                    let wrapped = !sentence_ended(&above.text) && (line.text.trim_start().starts_with(char::is_lowercase) || above.words.last().is_some_and(|word| word.underlined)
                        || above.spans.iter().rev().find(|span| !span.text.trim().is_empty()).is_some_and(|span| span.flags & legal_pdf_core::BOLD != 0));
                    // (Nor is what a colon sets out a heading, or what follows an elision.)
                    let set_out = above.text.trim_end().ends_with(':') || elided(&above.text);
                    run.len() <= 24 && run.iter().any(|&held| slots.line(pages, held).text.chars().any(char::is_alphabetic)) && !marked(held_lines[from - 1], 0.2) && (line.bbox[0] - margin <= QUOTED_INDENT || enumerator)
                        && !wrapped && !set_out && (run.iter().all(|&held| bold(held)) || italic || run.iter().all(|&held| marked(held, 0.8)) && sentence_ended(&above.text))
                });
                if let Some(from) = tail {
                    for line in &held_lines[from..] {
                        spans.remove(line);
                    }
                }
                // A quotation's last lines, in its indent, are the paragraph's: the text alone cannot see an indent ("Kelty", a letter's "Per:").
                if let Some(&last) = spans.keys().next_back().filter(|_| heading.is_none() && tail.is_none()) {
                    let (bbox, page) = (slots.line(pages, last).bbox, slots.page(last));
                    // (Inside a quotation: the line above stands at its indent too, as a first line's does not.)
                    let held: Vec<f64> = spans.keys().map(|&slot| slots.line(pages, slot).bbox[0]).collect();
                    let left = held.iter().copied().fold(f64::MAX, f64::min);
                    if bbox[0] - left > QUOTED_INDENT && held.len() >= 2 && (held[held.len() - 2] - bbox[0]).abs() <= 3.0 {
                        let from = kept_at.get(&last).copied().unwrap_or(kept.len());
                        let mut row = bbox;
                        for at in from + 1..kept.len() {
                            let under = slots.line(pages, kept[at]);
                            let enumerator = under.text.split_whitespace().next().is_some_and(|token| token.chars().count() <= 5 && token.ends_with(['.', ')']));
                            let same_row = (under.bbox[1] - row[1]).abs() <= 2.0;
                            if to_number[at] == 0 || slots.page(kept[at]) != page || (under.bbox[0] - bbox[0]).abs() > 3.0 && !same_row || enumerator {
                                break;
                            }
                            row = under.bbox;
                            spans.entry(kept[at]).or_insert((0, under.text.encode_utf16().count() as u32));
                        }
                    }
                }
                if let Some(first) = first {
                    self.count[first.index()].get_or_insert(number);
                }
                for (slot, (start, end)) in spans {
                    self.held[slot.index()].push((translation, number, start, end));
                }
            }
        }
        // A number standing alone beside the row it opens, or over it, is that paragraph's, in every language the row prints it.
        let rows = (0..slots.len()).flat_map(|at| {
            [self.margin_twin[at], self.margin_number[at], evidence.beside[at]].into_iter().flatten().map(move |row| (Slot(at as u32), row))
        });
        for (number_line, row) in rows.chain(led).collect::<Vec<_>>() {
            let at = number_line.index();
            let opened: Vec<(bool, u32)> = self.held[row.index()].iter().filter(|held| self.count[row.index()] == Some(held.1)).map(|held| (held.0, held.1)).collect();
            let length = slots.line(pages, number_line).text.encode_utf16().count() as u32;
            // (A heading's enumerator on the row, "i.", is no number of the paragraph's.)
            if !slots.line(pages, number_line).text.chars().any(|c| c.is_ascii_digit()) {
                continue;
            }
            for (translation, number) in opened {
                if !self.held[at].iter().any(|held| held.0 == translation && held.1 == number) {
                    self.held[at].push((translation, number, 0, length));
                }
            }
        }
        // An enactment quoted in both languages inside reasons in one: on a page none of whose lines in a language any paragraph
        // holds, that language's lines beside a paragraph are that paragraph's.
        for page_slot in 0..pages.len() {
            let on: Vec<usize> = slots.on(page_slot).map(|slot| slot as usize).filter(|&at| self.role[at] == Role::Open && !self.band[at]).collect();
            for language in [false, true] {
                let own: Vec<usize> = on.iter().copied().filter(|&at| self.translation[at] == language).collect();
                if own.iter().any(|&at| !self.held[at].is_empty()) {
                    continue;
                }
                let mut spans = BTreeMap::<(bool, u32), (f64, f64)>::new();
                for &at in on.iter().filter(|&&at| self.translation[at] != language) {
                    let bbox = slots.line(pages, Slot(at as u32)).bbox;
                    for held in &self.held[at] {
                        let span = spans.entry((held.0, held.1)).or_insert((bbox[1], bbox[3]));
                        *span = (span.0.min(bbox[1]), span.1.max(bbox[3]));
                    }
                }
                for at in own {
                    let line = slots.line(pages, Slot(at as u32));
                    let middle = (line.bbox[1] + line.bbox[3]) / 2.0;
                    if let Some((&(translation, number), _)) = spans.iter().find(|(_, span)| span.0 <= middle && middle <= span.1) {
                        self.held[at].push((translation, number, 0, line.text.encode_utf16().count() as u32));
                    }
                }
            }
        }
    }

    /// The quotations of another writing's numbered paragraphs, read off the judgment's count (`read_paragraphs`): in
    /// each document and language, a line no fence took and the band left that opens on a printed paragraph number the
    /// count takes at another line (a number nothing else prints is the count's own, however it broke the count: a
    /// misread digit), with the lines its block goes on into (`continues`) and the editorial lines under it ("[Emphasis
    /// added]"), each a quotation of its own. Whether it found any.
    fn read_quotations(&mut self, pages: &[Page], slots: &Slots) -> bool {
        // LEGALPDF_ABLATE=sequence leaves the quotations out, for their measure under the tagger.
        if legal_structure_model::ablated("sequence") {
            return false;
        }
        let analysis = legal_pdf_core::layout_analysis();
        let mut own = HashMap::<(u16, bool), HashSet<u32>>::new();
        for at in 0..self.count.len() {
            if let Some(number) = self.count[at] {
                own.entry((self.document[at], self.translation[at])).or_default().insert(number);
            }
        }
        let (mut next, mut open) = (0u16, HashMap::<(u16, bool), u16>::new());
        for slot in self.order.clone() {
            let at = slot.index();
            if self.role[at] != Role::Open || self.band[at] {
                continue;
            }
            let stream = (self.document[at], self.translation[at]);
            let text = slots.line(pages, slot).text.as_str();
            if let Some(&quotation) = open.get(&stream) {
                if self.continues[at].is_some() || analysis.editorial_line(text) {
                    self.quotation[at] = quotation;
                    continue;
                }
                open.remove(&stream);
            }
            let quoted = self.count[at].is_none()
                && analysis.printed_paragraph_number(text).is_some_and(|number| own.get(&stream).is_some_and(|numbers| numbers.contains(&(number as u32))));
            if quoted {
                next += 1;
                self.quotation[at] = next;
                open.insert(stream, next);
            }
        }
        next > 0
    }

    /// How each line no fence took goes on with the line before it, in its reading context and
    /// stream, in reading order: on one page by any continuation, across a page by the text's flow.
    fn read_continuations(&mut self, pages: &[Page], slots: &Slots, evidence: &Evidence) {
        self.continues = vec![None; slots.len()];
        let mut last: Vec<Option<Slot>> = vec![None; self.contexts.len() * 2];
        for &slot in &self.order {
            if self.role[slot.index()] != Role::Open {
                continue;
            }
            let band = self.band[slot.index()];
            let stream = usize::from(self.ctx[slot.index()]) * 2 + usize::from(band);
            let Some(prior) = last[stream].replace(slot) else {
                continue;
            };
            let (line, before) = (slots.line(pages, slot), slots.line(pages, prior));
            let page_slot = slots.page(slot);
            self.continues[slot.index()] = if body_flow_edge(before, line) {
                Some(Continuation::Flow)
            } else if slots.page(prior) != page_slot {
                None
            } else {
                let page = &pages[page_slot];
                let range = slots.on(page_slot);
                let start = range.start as usize;
                let (at, prior_at) = (slot.index() - start, prior.index() - start);
                let body = |other: usize| self.role[start + other] == Role::Open && self.band[start + other] == band;
                if reference_wrapped_onto(&page.lines, &evidence.size[start..range.end as usize], at) {
                    Some(Continuation::Reference)
                } else if carries_a_sentence_onto(page, prior_at, at, body) {
                    Some(Continuation::Sentence)
                } else if continues_prose_from(page, prior_at, at, body) {
                    Some(Continuation::Prose)
                } else {
                    None
                }
            };
        }
    }

    /// The statute's running heads naming their page's sections, where its spine bears them out
    /// (`spine`: a section numbered so opens on that page slot), on their page or the one before;
    /// and the lines on their rows. The lines' continuations are read again without them.
    pub(crate) fn confirm_section_heads(&mut self, pages: &[Page], slots: &Slots, evidence: &Evidence, spine: impl Fn(usize, u32) -> bool) {
        let mut fenced = false;
        for (slot, edge, number) in std::mem::take(&mut self.section_heads) {
            let page = slots.page(slot);
            if !(spine(page, number) || page > 0 && spine(page - 1, number)) || !self.set(slot, Role::Furniture, Rule::SectionHead) {
                continue;
            }
            fenced = true;
            self.edge[slot.index()] = Some(edge);
            self.band[slot.index()] = false;
            let row = evidence.row(slots, slot);
            for mate in slots.on(page).map(Slot) {
                if row != u32::MAX && evidence.row(slots, mate) == row && self.set(mate, Role::Furniture, Rule::RowMate) {
                    self.edge[mate.index()] = Some(edge);
                    self.band[mate.index()] = false;
                }
            }
        }
        if fenced {
            self.read_continuations(pages, slots, evidence);
        }
    }

    /// The slot's reading context.
    pub(crate) fn context(&self, slot: Slot) -> Context {
        self.contexts[usize::from(self.ctx[slot.index()])]
    }
}

/// The page's items (`page_items`) of the lines `keep` holds (slots of the page's lines).
fn items_of(items: &PageItems, keep: impl Fn(usize) -> bool) -> PageItems {
    let (items, origin) = items
        .items
        .iter()
        .zip(&items.origin)
        .filter(|(_, (line, _))| keep(*line))
        .map(|(item, origin)| (item.clone(), *origin))
        .unzip();
    PageItems { items, origin }
}

/// A page's reading order (`order`, slots of its lines) with each table's lines in its rows' order,
/// cell by cell, where its first line stood; `in_table` holds each of the page's lines' table and
/// place among that table's lines.
fn table_order(order: Vec<usize>, in_table: &[Option<(u32, u32)>]) -> Vec<usize> {
    let mut held: Vec<(u32, Vec<(u32, usize)>)> = Vec::new();
    let mut sequence: Vec<std::result::Result<usize, u32>> = Vec::with_capacity(order.len());
    for line in order {
        match in_table[line] {
            Some((table, at)) => match held.iter_mut().find(|(seen, _)| *seen == table) {
                Some((_, lines)) => lines.push((at, line)),
                None => {
                    held.push((table, vec![(at, line)]));
                    sequence.push(Err(table));
                }
            },
            None => sequence.push(Ok(line)),
        }
    }
    let mut ordered = Vec::with_capacity(sequence.len());
    for entry in sequence {
        match entry {
            Ok(line) => ordered.push(line),
            Err(table) => {
                let at = held.iter().position(|(seen, _)| *seen == table).expect("a held table");
                let mut lines = std::mem::take(&mut held[at].1);
                lines.sort_by_key(|(place, _)| *place);
                ordered.extend(lines.into_iter().map(|(_, line)| line));
            }
        }
    }
    ordered
}

