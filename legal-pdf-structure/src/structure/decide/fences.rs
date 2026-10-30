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
        fences.read_quotations(pages, slots);
        let mut contexts: Vec<Context> = Vec::new();
        fences.ctx = (0..count)
            .map(|slot| {
                let context = Context {
                    document: fences.document[slot],
                    translation: fences.translation[slot],
                    quotation: fences.quotation[slot],
                };
                contexts.iter().position(|known| *known == context).unwrap_or_else(|| {
                    contexts.push(context);
                    contexts.len() - 1
                }) as u16
            })
            .collect();
        fences.contexts = contexts;
        fences.read_continuations(pages, slots, evidence);
        fences
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

    /// The quotations: in each document and language, the bracketed paragraph numbers of the lines
    /// no fence took and the band left, in reading order, that break the document's own count
    /// (`own_sequence`), each with its passage (`passage_end`) to the end of its printed paragraph:
    /// on a page, where the page's geometry parts two paragraphs (`paragraphs_of`); across a page,
    /// where the text does not flow on (`body_flow_edge`).
    fn read_quotations(&mut self, pages: &[Page], slots: &Slots) {
        // LEGALPDF_ABLATE=sequence leaves the quoted-paragraph sequence out, for its measure under the tagger.
        if legal_structure_model::ablated("sequence") {
            return;
        }
        let mut body: BTreeMap<(u16, bool), Vec<Slot>> = BTreeMap::new();
        for &slot in &self.order {
            if self.role[slot.index()] == Role::Open && !self.band[slot.index()] {
                body.entry((self.document[slot.index()], self.translation[slot.index()])).or_default().push(slot);
            }
        }
        let mut next = 0;
        for lines in body.values() {
            let text = |at: usize| slots.line(pages, lines[at]).text.as_str();
            let labels: Vec<(usize, usize)> = (0..lines.len())
                .filter_map(|at| numbered_paragraphs::paragraph_number(text(at)).map(|number| (number, at)))
                .collect();
            let own = numbered_paragraphs::own_sequence(&labels.iter().map(|(number, _)| *number).collect::<Vec<_>>());
            if own.iter().all(|own| *own) {
                continue;
            }
            let read: Vec<&Line> = lines.iter().map(|&slot| slots.line(pages, slot)).collect();
            let paragraph = printed_paragraphs(slots, lines, &read);
            let goes_on = |at: usize| {
                if slots.page(lines[at - 1]) == slots.page(lines[at]) {
                    paragraph[at] == paragraph[at - 1]
                } else {
                    body_flow_edge(read[at - 1], read[at])
                }
            };
            for (k, (&(_, start), own)) in labels.iter().zip(&own).enumerate() {
                if *own {
                    continue;
                }
                let next_label = labels.get(k + 1).map_or(lines.len(), |&(_, at)| at);
                let end = numbered_paragraphs::passage_end(start, next_label, text, goes_on);
                next += 1;
                for &slot in &lines[start..end] {
                    if self.quotation[slot.index()] == 0 {
                        self.quotation[slot.index()] = next;
                    }
                }
            }
        }
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

/// The printed paragraph each of a reading's lines (`lines`, read as `read`, in reading order)
/// stands in on its page (`paragraphs_of`, on the reading's measure).
fn printed_paragraphs(slots: &Slots, lines: &[Slot], read: &[&Line]) -> Vec<Option<usize>> {
    let mut spans: Vec<std::ops::Range<usize>> = Vec::new();
    for at in 0..lines.len() {
        match spans.last_mut() {
            Some(span) if slots.page(lines[span.start]) == slots.page(lines[at]) => span.end = at + 1,
            _ => spans.push(at..at + 1),
        }
    }
    let by_page: Vec<Vec<&Line>> = spans.iter().map(|span| read[span.clone()].to_vec()).collect();
    let measure = crate::paragraphs::measure_of(&by_page);
    let mut paragraph = Vec::with_capacity(lines.len());
    for page in &by_page {
        paragraph.extend(crate::paragraphs::paragraphs_of(page, &vec![true; page.len()], measure));
    }
    paragraph
}
