//! What each line is, decided once. The fences stand first; then, in this order of precedence:
//! the note band (a transcript's lines beside their number column never notes), the headings a
//! transcript's words and an enactment's history set; the runs' quotations, contents of titles
//! alone, counted markers opening their units, divisions, an instrument's captions and heading
//! ladders; a statute's marginal notes over its openers; the lines the document's bookmarks and
//! contents list name; the front matter and the lines that go on with the line before, which are
//! no headings; the residual weighing of the lines still open (`residual`); the title; and the
//! levels (`levels`). A rule fills only an open line: where two strong rules disagree on one, the
//! first keeps it and the document's diagnostics name both. Each line keeps the rule that decided
//! it, and each marker the unit it opens.

use super::*;
use legal_pdf_core::{Columns, PageMeasure};
use legal_structure_model::{CandidateGrammar, CandidateObservationV2, HeadingLadderStatus, UnitRole};
use std::cell::OnceCell;

/// A unit a line opens: its run (an index into the document's runs) and its kind.
pub(crate) type Opening = (u32, UnitRole);

/// Lines a confirmed or found bookmark or contents entry names: its lines, its level, and whether
/// its title is found whole.
#[derive(Debug)]
pub(crate) struct OutlineAnchor {
    pub(crate) lines: Vec<Slot>,
    pub(crate) level: u16,
    pub(crate) confirmed: bool,
}

/// Each reading context's lines in reading order, its line labels, folios, furniture and contents
/// left out; and each line's place among its context's lines (`u32::MAX` for a line left out).
#[derive(Debug, Default)]
pub(crate) struct Sequences {
    pub(crate) lines: Vec<Vec<Slot>>,
    pub(crate) at: Vec<u32>,
}

impl Sequences {
    /// The line `step` places after (or before, negative) the slot's line in its context.
    pub(crate) fn beside(&self, fences: &Fences, slot: Slot, step: isize) -> Option<Slot> {
        let at = self.at[slot.index()];
        if at == u32::MAX {
            return None;
        }
        let lines = &self.lines[usize::from(fences.ctx[slot.index()])];
        at.checked_add_signed(step as i32).and_then(|at| lines.get(at as usize).copied())
    }
}

/// Each page's measures and columns, read once a rule asks for them.
pub(crate) struct Geometry<'a> {
    pages: &'a [Page],
    read: Vec<OnceCell<(PageMeasure, Columns)>>,
}

impl<'a> Geometry<'a> {
    pub(crate) fn of(pages: &'a [Page]) -> Self {
        Self { pages, read: (0..pages.len()).map(|_| OnceCell::new()).collect() }
    }

    /// The page's measures (`PageMeasure`) and columns (`Columns`), by page slot.
    pub(crate) fn page(&self, page: usize) -> &(PageMeasure, Columns) {
        self.read[page].get_or_init(|| (PageMeasure::of(&self.pages[page]), Columns::of(&self.pages[page])))
    }
}

/// What each line is, by slot: its role, the rule that gave it, the reading of its opening marker
/// that may open its unit (`readings`) and the unit it opens, whether it goes on with the line above
/// it in one heading, caption or unit, the level a rule gives it (0 for none) and its heading level
/// once read (0 for none), the anchor naming it (an index among the anchors, plus one; 0 for none);
/// the disagreements of strong rules; and, where the decision is traced, each residual candidate as
/// the residual weighed it.
#[derive(Debug, Default)]
pub(crate) struct Decision {
    pub(crate) role: Vec<Role>,
    pub(crate) rule: Vec<Rule>,
    pub(crate) reading: Vec<Option<(u32, u32)>>,
    pub(crate) unit: Vec<Option<Opening>>,
    pub(crate) joined: Vec<bool>,
    pub(crate) given: Vec<u8>,
    pub(crate) level: Vec<u8>,
    pub(crate) anchor: Vec<u32>,
    pub(crate) anchors: Vec<OutlineAnchor>,
    pub(crate) diagnostics: Vec<Diagnostic>,
    pub(crate) residual: Vec<ResidualRecord>,
    conflicts: Vec<(Rule, Rule, Slot)>,
}

/// The one reading of each line's opening marker that may open its unit, as (run, member) indexes,
/// by slot: of the counted, unquoted markers on the line that resolve to a unit, the one standing
/// first on it; where runs read that one marker, a count of numbers's before the provision
/// ladder's and the ladder's before a list's, wherever each run starts (the engine has already
/// read an order's count inside a numbered paragraph as a list; a factum's "PART I" ladder,
/// starting above its first paragraph, takes no paragraph's number from the count).
fn readings(runs: &Runs, count: usize) -> Vec<Option<(u32, u32)>> {
    let mut chosen: Vec<Option<(u32, u32)>> = vec![None; count];
    let rank = |r: usize, k: usize| (runs.runs[r].members[k].marker[0].offset, crate::structure::graph::grammar_rank(runs.runs[r].grammar));
    for (r, run) in runs.runs.iter().enumerate() {
        for (k, member) in run.members.iter().enumerate() {
            if !member.counted || member.kind.is_none() || quoted(member) {
                continue;
            }
            let at = member.slot.index();
            let better = chosen[at].is_none_or(|(r0, k0)| rank(r, k) < rank(r0 as usize, k0 as usize));
            if better {
                chosen[at] = Some((r as u32, k as u32));
            }
        }
    }
    chosen
}

/// Body text, a list's item or a quotation: what the residual reads as prose.
pub(crate) fn body_like(role: Role) -> bool {
    matches!(role, Role::Open | Role::Body | Role::ListItem | Role::Quote)
}

/// The share of a line's text its citations cover.
fn cited_share(evidence: &Evidence, slot: Slot, length: usize) -> f64 {
    let mut ranges: Vec<(u32, u32)> = evidence.citations_of(slot).iter().map(|cited| cited.range).collect();
    ranges.sort_unstable();
    let (mut covered, mut reach) = (0, 0);
    for (start, end) in ranges {
        let start = start.max(reach);
        if end > start {
            covered += end - start;
            reach = end;
        }
    }
    if length == 0 { 0.0 } else { f64::from(covered) / length as f64 }
}

/// A line opening a provision's subsection or a clause's term: "(3) For the purposes", "4 (1)
/// Every", "(a) the".
fn subsection(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*(?:\d{1,4}(?:\.\d{1,3})*\s*)?\((?:\d{1,3}|[a-z]{1,4})\)\s+\S").unwrap()).is_match(text)
}

/// A colon closing a lead-in on a small joining word: "ORDERS THAT:", "PURPOSE OF:", "APPEARING on:".
fn joining_colon(text: &str) -> bool {
    text.strip_suffix(':').and_then(|before| before.split_whitespace().next_back()).is_some_and(|word| {
        matches!(
            word.to_lowercase().as_str(),
            "that" | "of" | "on" | "to" | "for" | "by" | "and" | "or" | "the" | "as" | "at" | "in" | "with" | "from" | "under"
        )
    })
}

impl Decision {
    /// Gives an open line a role by a strong rule; a line another rule decided keeps its own, and
    /// where the roles differ the two rules are noted.
    fn claim(&mut self, slot: Slot, role: Role, rule: Rule) -> bool {
        let at = slot.index();
        if self.role[at] == Role::Open {
            self.role[at] = role;
            self.rule[at] = rule;
            return true;
        }
        if self.role[at] != role {
            self.conflicts.push((self.rule[at], rule, slot));
        }
        false
    }

    /// Gives an open line a role; a decided line keeps its own.
    fn fill(&mut self, slot: Slot, role: Role, rule: Rule) -> bool {
        let at = slot.index();
        let open = self.role[at] == Role::Open;
        if open {
            self.role[at] = role;
            self.rule[at] = rule;
        }
        open
    }

    fn open(&self, slot: Slot) -> bool {
        self.role[slot.index()] == Role::Open
    }

    /// Every line of the document's pages decided, measured (`evidence`), fenced and ordered
    /// (`fences`), its runs read (`runs`) and its notes (`notes`), its bookmarks `outline`.
    pub(crate) fn decide(
        pages: &[Page],
        slots: &Slots,
        evidence: &Evidence,
        fences: &Fences,
        runs: &Runs,
        notes: &Notes,
        outline: &[PdfOutlineEntry],
    ) -> Self {
        let count = slots.len();
        let mut decision = Self {
            role: fences.role.clone(),
            rule: fences.rule.clone(),
            reading: readings(runs, count),
            unit: vec![None; count],
            joined: vec![false; count],
            given: vec![0; count],
            level: vec![0; count],
            anchor: vec![0; count],
            ..Self::default()
        };
        let geometry = Geometry::of(pages);
        let text = |slot: Slot| slots.line(pages, slot).text.trim();
        let mut sequences = Sequences { lines: vec![Vec::new(); fences.contexts.len()], at: vec![u32::MAX; count] };
        // Each document a transcript of whose witnesses are named.
        let mut witnesses = vec![false; fences.transcripts.len()];
        let mut examinations = Vec::new();
        // Rules that only keep a line from the headings, applied once the anchors stand.
        let mut deferred: Vec<(Slot, Rule)> = Vec::new();
        for &slot in &fences.order {
            let at = slot.index();
            let words = evidence.words[at];
            let beside_numbers = fences.numbered_row[at];
            let transcript = fences.transcript(slot);
            let previous = if matches!(decision.role[at], Role::LineLabel | Role::PageLabel | Role::Furniture | Role::Contents) {
                None
            } else {
                let lines = &mut sequences.lines[usize::from(fences.ctx[at])];
                let previous = lines.last().copied();
                sequences.at[at] = lines.len() as u32;
                lines.push(slot);
                previous
            };
            let line = text(slot);
            if transcript && words & WORD_WITNESS != 0 {
                witnesses[usize::from(fences.document[at])] = true;
            }
            if notes.running_head[at] {
                decision.fill(slot, Role::Furniture, Rule::EndnoteHead);
            }
            if notes.heading[at] {
                decision.claim(slot, Role::Heading, Rule::NotesHeading);
            } else if notes.note[at].is_some() {
                if beside_numbers {
                    decision.conflicts.push((Rule::TranscriptLine, Rule::NoteBand, slot));
                } else {
                    decision.claim(slot, Role::Note, Rule::NoteBand);
                }
            }
            if !decision.open(slot) {
                continue;
            }
            // An enactment's history after a dash heads its provision.
            let dashed = line.strip_prefix(['\u{2014}', '\u{2013}', '-']).map(str::trim_start);
            if dashed.is_some_and(|rest| {
                legal_structure_model::statute::history_note(rest) || cited_share(evidence, slot, slots.line(pages, slot).text.len()) >= 0.8
            }) {
                decision.claim(slot, Role::Heading, Rule::HistoryHeading);
                continue;
            }
            let opens = !line.starts_with(char::is_lowercase) && fences.continues[at].is_none();
            if transcript && words & WORD_INDEX_ENTRY == 0 && opens {
                if words & (WORD_WITNESS | WORD_HEARING_PART | WORD_TRANSCRIPT_PART) != 0 {
                    decision.fill(slot, Role::Heading, Rule::TranscriptPart);
                    decision.given[at] = 1;
                    continue;
                }
                if words & WORD_EXAMINATION != 0 {
                    examinations.push(slot);
                    continue;
                }
            }
            if transcript && words & WORD_INDEX_LETTER != 0 && opens {
                decision.fill(slot, Role::Heading, Rule::TranscriptPart);
                decision.given[at] = 2;
                continue;
            }
            // A transcript's line going on from the one before is no heading; a question its number
            // opens ("11 Q. Maybe jot in huge letters" after "BY MR. RUBY:") opens its unit.
            if words & WORD_HISTORY != 0 {
                deferred.push((slot, Rule::HistoryLine));
            } else if beside_numbers
                && (line.starts_with(char::is_lowercase) || previous.is_some_and(|previous| !sentence_ended(text(previous))))
            {
                if decision.reading[at].is_none() {
                    deferred.push((slot, Rule::TranscriptLine));
                }
            } else if front_matter(pages, slots, evidence, fences, slot) {
                deferred.push((slot, Rule::FrontMatter));
            }
        }
        for slot in examinations {
            if decision.fill(slot, Role::Heading, Rule::TranscriptPart) {
                decision.given[slot.index()] = if witnesses[usize::from(fences.document[slot.index()])] { 2 } else { 1 };
            }
        }
        decision.read_runs(pages, slots, evidence, fences, runs, &sequences);
        decision.marginal_notes(pages, slots, evidence, fences, runs, &sequences);
        let anchoring = anchors(pages, slots, evidence, fences, &geometry, &decision, outline);
        for (index, anchor) in anchoring.anchors.iter().enumerate() {
            for (k, &slot) in anchor.lines.iter().enumerate() {
                decision.claim(slot, Role::Heading, Rule::Anchor);
                decision.anchor[slot.index()] = index as u32 + 1;
                decision.given[slot.index()] = anchor.level.min(u16::from(u8::MAX)) as u8;
                decision.joined[slot.index()] |= k > 0;
            }
        }
        decision.anchors = anchoring.anchors;
        for (slot, rule) in deferred {
            decision.fill(slot, Role::Body, rule);
        }
        // Bookmarks naming every heading leave none of the lines after their opening they do not name.
        let unnamed = |slot: Slot| {
            evidence.outline_complete
                && anchoring.opened.iter().any(|&(document, translation, place)| {
                    document == fences.document[slot.index()]
                        && translation == fences.translation[slot.index()]
                        && place <= fences.place[slot.index()]
                })
        };
        weigh(pages, slots, evidence, fences, &sequences, &geometry, &unnamed, &mut decision);
        decision.title(pages, slots, evidence, fences, &sequences, &geometry);
        // A counted marker no rule settled opens the unit its reading reads, the line carrying its
        // text going on with it; every other line is body, or a quotation's.
        for at in 0..count {
            let Some((r, k)) = decision.reading[at] else { continue };
            let member = &runs.runs[r as usize].members[k as usize];
            let Some(kind) = member.kind else { continue };
            let role = if kind == UnitRole::Item { Role::ListItem } else { Role::Body };
            if decision.unit[at].is_none() && decision.fill(member.slot, role, Rule::Opener) {
                decision.unit[at] = Some((r, kind));
                decision.join(member.text, role, Rule::Opener);
            }
        }
        for at in 0..count {
            if decision.role[at] == Role::Open {
                decision.role[at] = if fences.quotation[at] != 0 { Role::Quote } else { Role::Body };
                if decision.rule[at] == Rule::None {
                    decision.rule[at] = Rule::Body;
                }
            }
        }
        heading_levels(pages, slots, evidence, fences, &sequences, &geometry, &mut decision);
        decision.report(pages, slots);
        decision
    }

    /// The rules each run settles: a quotation's numbers open nothing; three or more titles alone,
    /// one under the next, are contents; a counted marker carrying prose opens its unit, by its
    /// line's one reading (`readings`); one before a title alone heads its division where a
    /// division's word opens it, and opens its
    /// clause with the caption as its first line in an instrument, or over its clause's terms
    /// ("(a)", "(3)"); outside an instrument, a count of titles alone between the text's paragraphs
    /// is a heading ladder.
    fn read_runs(&mut self, pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, runs: &Runs, sequences: &Sequences) {
        let place = |slot: Slot| sequences.at[slot.index()];
        // The last place of a member's own lines: its line, the line carrying its text, its text's
        // place.
        let end = |member: &Member| {
            [Some(member.slot), member.text, Some(member.content.slot)]
                .into_iter()
                .flatten()
                .map(place)
                .filter(|at| *at != u32::MAX)
                .max()
                .unwrap_or(u32::MAX)
        };
        for run in &runs.runs {
            for member in run.members.iter().filter(|member| quoted(member)) {
                self.claim(member.slot, Role::Quote, Rule::QuotedNumber);
            }
        }
        // Titles alone, each directly over the next (a division's line between them aside), none of
        // three words or more (a statute's "118 [Repealed, 2014, c. 38, s. 17]" is its provision); a
        // transcript's line beside its number column is none.
        for run in &runs.runs {
            let lines = &sequences.lines[usize::from(run.ctx)];
            let alone = |member: &Member| {
                member.counted
                    && member.title_shaped
                    && !member.carries_prose
                    && !member.worded
                    && !quoted(member)
                    && self.open(member.slot)
                    && !fences.numbered_row[member.slot.index()]
            };
            let directly = |before: &Member, after: &Member| {
                let (from, to) = (end(before), place(after.slot));
                from != u32::MAX
                    && to != u32::MAX
                    && (from + 1..to).all(|at| evidence.words[lines[at as usize].index()] & WORD_DIVISION != 0)
            };
            let mut stretches: Vec<(usize, usize)> = Vec::new();
            let mut start = 0;
            for k in 0..=run.members.len() {
                let goes_on = k < run.members.len()
                    && alone(&run.members[k])
                    && (k == start || directly(&run.members[k - 1], &run.members[k]));
                if goes_on {
                    continue;
                }
                if k - start >= 3 {
                    stretches.push((start, k));
                }
                start = if k < run.members.len() && alone(&run.members[k]) { k } else { k + 1 };
            }
            for (first, last) in stretches {
                for k in first..last {
                    let member = &run.members[k];
                    for slot in [Some(member.slot), member.text].into_iter().flatten() {
                        self.claim(slot, Role::Contents, Rule::TitleOnly);
                    }
                    if k + 1 < last {
                        for at in end(member) + 1..place(run.members[k + 1].slot) {
                            self.claim(lines[at as usize], Role::Contents, Rule::TitleOnly);
                        }
                    }
                }
            }
        }
        for (r, run) in runs.runs.iter().enumerate() {
            let lines = &sequences.lines[usize::from(run.ctx)];
            for (k, member) in run.members.iter().enumerate() {
                let at = member.slot.index();
                if self.reading[at] != Some((r as u32, k as u32)) {
                    continue;
                }
                let Some(kind) = member.kind else { continue };
                if self.rule[at] == Rule::HistoryHeading || self.rule[at] == Rule::TitleOnly {
                    continue;
                }
                if member.carries_prose {
                    let role = if kind == UnitRole::Item { Role::ListItem } else { Role::Body };
                    if self.claim(member.slot, role, Rule::Opener) {
                        self.unit[at] = Some((r as u32, kind));
                        self.join(member.text, role, Rule::Opener);
                    }
                } else if member.title_shaped && self.open(member.slot) {
                    if evidence.words[at] & WORD_DIVISION != 0 {
                        self.claim(member.slot, Role::Heading, Rule::DivisionHeading);
                        self.join(member.text, Role::Heading, Rule::DivisionHeading);
                        continue;
                    }
                    // A caption opens its clause where text, not another title, follows it: in an
                    // instrument, or over its clause's terms.
                    let from = end(member);
                    let to = run.members.get(k + 1).map_or(from.saturating_add(2), |next| place(next.slot));
                    let followed = from != u32::MAX
                        && (from + 1..to.max(from + 1)).take_while(|at| (*at as usize) < lines.len()).any(|at| {
                            !evidence.title_shaped[lines[at as usize].index()]
                        });
                    let next = (from != u32::MAX).then(|| lines.get(from as usize + 1).copied()).flatten();
                    let over_terms = next.is_some_and(|next| {
                        self.role[next.index()] != Role::Heading
                            && !evidence.title_shaped[next.index()]
                            && subsection(slots.line(pages, next).text.trim())
                    });
                    // The clause's text after its caption is the clause's, set apart from it where the
                    // page parts them ("2.2 / Landlord's Work" over "The Landlord shall perform").
                    if (evidence.instrument && followed || over_terms) && self.claim(member.slot, Role::Body, Rule::Caption) {
                        self.unit[at] = Some((r as u32, kind));
                        self.join(member.text, Role::Body, Rule::Caption);
                    }
                }
            }
            if !evidence.instrument {
                self.ladder(pages, slots, fences, run, sequences);
            }
        }
    }

    /// Gives a marker's text line, where it has one of its own, the marker's role, going on with it.
    fn join(&mut self, text: Option<Slot>, role: Role, rule: Rule) {
        if let Some(text) = text {
            let filled = self.fill(text, role, rule);
            self.joined[text.index()] |= filled;
        }
    }

    /// A run whose counted markers are all titles alone between the text's paragraphs: none ending
    /// as a list's item does, none after a lead-in, each followed by a paragraph of its own, and
    /// their numbers a ladder the heading grammar parses (`parse_heading_runs`), not a count of
    /// notes' numbers.
    fn ladder(&mut self, pages: &[Page], slots: &Slots, fences: &Fences, run: &Run, sequences: &Sequences) {
        let counted: Vec<&Member> = run.members.iter().filter(|member| member.counted && !quoted(member)).collect();
        if counted.len() < 2 {
            return;
        }
        let readings: Vec<Vec<EnumeratorInterpretation>> = counted
            .iter()
            .filter_map(|member| headings::rules::enumerator(slots.line(pages, member.slot).text.trim()))
            .map(|(value, punctuation)| enumerator_interpretations(&value, &punctuation))
            .filter(|readings| !readings.is_empty())
            .collect();
        let readings: Vec<&[EnumeratorInterpretation]> = readings.iter().map(Vec::as_slice).collect();
        let parsed = parse_heading_runs(&readings).into_iter().all(|(_, ladder)| {
            ladder.status != HeadingLadderStatus::Unparseable && ladder.families.values().all(|family| !family.footnote_suspect)
        });
        if !parsed {
            return;
        }
        let text = |slot: Slot| slots.line(pages, slot).text.trim();
        let title_line = |member: &Member| member.text.unwrap_or(member.slot);
        let listed = |line: &str| {
            line.ends_with([',', ';']) || line.rsplit(char::is_whitespace).next().is_some_and(|word| matches!(word, "and" | "or"))
        };
        let shaped = counted.iter().all(|member| {
            let title = title_line(member);
            member.title_shaped && !member.carries_prose && self.open(member.slot) && !listed(text(title))
        });
        let lead_in = sequences.beside(fences, counted[0].slot, -1).is_some_and(|before| {
            body_like(self.role[before.index()]) && (!sentence_ended(text(before)) || text(before).ends_with(':'))
        });
        let parted = counted.iter().all(|member| {
            sequences.beside(fences, title_line(member), 1).is_none_or(|after| fences.continues[after.index()].is_none())
        });
        if !shaped || lead_in || !parted {
            return;
        }
        for member in counted {
            self.claim(member.slot, Role::Heading, Rule::Ladder);
            let title = title_line(member);
            self.join((title != member.slot).then_some(title), Role::Heading, Rule::Ladder);
        }
    }

    /// A statute's marginal notes, in each reading context holding its spine (three or more
    /// counted provisions, rooted and consecutive, opening prose): one or two short lines shaped
    /// as a note directly over an opener (a subsection's too; its second line may go on in lower
    /// case, "Endorsements" over "on writ"), or in the page's margin column level with it, are a
    /// caption. A caption level with its opener's first line beside it, or over exactly one
    /// opener of its depth before the next caption and set as most such captions of its look
    /// (bold, size, centring) are, is the opener's marginal note; any other is left to the lines
    /// still open. A caption's twin on its row in the other language's column reads as it does.
    fn marginal_notes(&mut self, pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, runs: &Runs, sequences: &Sequences) {
        let text = |slot: Slot| slots.line(pages, slot).text.trim();
        let mut decided: Vec<Slot> = Vec::new();
        for (ctx, lines) in sequences.lines.iter().enumerate() {
            if fences.contexts[ctx].quotation != 0 {
                continue;
            }
            // The context's openers in reading order: place, depth, slot. A consolidation's ladder
            // skips its repealed sections and inserts others ("19.02"), so it need not count on by one.
            let mut integer_roots = 0usize;
            let mut openers: Vec<(u32, u16, Slot)> = runs
                .runs
                .iter()
                .filter(|run| usize::from(run.ctx) == ctx && run.grammar == CandidateGrammar::Hierarchy && run.rooted)
                .flat_map(|run| &run.members)
                .filter(|member| member.counted && member.carries_prose && member.kind.is_some() && self.rule[member.slot.index()] == Rule::Opener)
                .map(|member| {
                    if member.parent.is_none() && !member.grammar_value.contains('.') {
                        integer_roots += 1;
                    }
                    (sequences.at[member.slot.index()], if member.parent.is_none() { 0 } else { member.level.max(1) }, member.slot)
                })
                .collect();
            if openers.iter().filter(|(_, depth, _)| *depth == 0).count() < 3 {
                continue;
            }
            openers.sort_unstable();
            openers.dedup_by_key(|(at, _, _)| *at);
            let shaped = |slot: Slot| {
                let line = text(slot);
                let letters = line.chars().filter(|c| c.is_alphabetic()).count();
                let upper = line.chars().filter(|c| c.is_uppercase()).count();
                self.open(slot)
                    && legal_structure_model::statute::marginal_note_shaped(line)
                    && upper * 5 < letters * 4
                    && evidence.words[slot.index()] & WORD_DIVISION == 0
                    && !fences.numbered_row[slot.index()]
            };
            // A note's line wrapped under its first: open, short, going on in lower case.
            let wrapped = |slot: Slot| {
                self.open(slot)
                    && fences.continues[slot.index()] == Some(Continuation::Flow)
                    && text(slot).starts_with(char::is_lowercase)
                    && (1..=16).contains(&text(slot).split_whitespace().count())
                    && !fences.numbered_row[slot.index()]
            };
            // Each caption: its lines, top first, and the opener under it.
            let mut captions: Vec<(Vec<Slot>, usize)> = Vec::new();
            for (k, &(at, _, opener)) in openers.iter().enumerate() {
                let page = slots.page(opener);
                let above = |step: u32| at.checked_sub(step).map(|place| lines[place as usize]).filter(|&slot| slots.page(slot) == page);
                let mut caption: Vec<Slot> = Vec::new();
                if integer_roots >= 2 {
                    if let Some(first) = above(1).filter(|&slot| shaped(slot)) {
                        caption.push(first);
                        if let Some(upper) = above(2).filter(|&slot| shaped(slot) && fences.continues[first.index()].is_some()) {
                            caption.insert(0, upper);
                        }
                    } else if let Some(lower) = above(1).filter(|&slot| wrapped(slot)) {
                        if let Some(upper) = above(2).filter(|&slot| shaped(slot)) {
                            caption = vec![upper, lower];
                        }
                    }
                    if caption.first().is_some_and(|first| fences.continues[first.index()].is_some()) {
                        caption.clear();
                    }
                }
                if caption.is_empty() {
                    caption = self.margin_caption(pages, slots, fences, opener, &shaped);
                }
                if !caption.is_empty() && captions.last().is_none_or(|(last, _)| last[0] != caption[0]) {
                    captions.push((caption, k));
                }
            }
            // Whether each caption governs one opener of its depth before the next caption, and
            // its look.
            let repealed = |slot: Slot| {
                let line = text(slot);
                line.contains("[Repealed") || line.contains("[Abrogé")
            };
            let governs: Vec<bool> = (0..captions.len())
                .map(|c| {
                    let (_, k) = captions[c];
                    let (at, depth, _) = openers[k];
                    let next = captions.get(c + 1).map_or(u32::MAX, |(lines, _)| sequences.at[lines[0].index()].min(openers[captions[c + 1].1].0));
                    openers
                        .iter()
                        .filter(|&&(place, other, slot)| place == at || (place > at && place < next && other <= depth && !repealed(slot)))
                        .count()
                        == 1
                })
                .collect();
            let look = |slot: Slot| {
                let page = &pages[slots.page(slot)];
                let line = slots.line(pages, slot);
                let width = page.width.max(1.0);
                let centred = (((line.bbox[0] + line.bbox[2]) / 2.0 - width / 2.0) / width).abs() < 0.03;
                (evidence.look[slot.index()] & LOOK_BOLD != 0, evidence.half_points[slot.index()], centred)
            };
            let mut groups: Vec<((bool, u16, bool), usize, usize)> = Vec::new();
            for ((caption, _), &one) in captions.iter().zip(&governs) {
                let key = look(caption[0]);
                match groups.iter_mut().find(|(seen, _, _)| *seen == key) {
                    Some((_, members, ones)) => {
                        *members += 1;
                        *ones += usize::from(one);
                    }
                    None => groups.push((key, 1, usize::from(one))),
                }
            }
            // A caption level with its opener's first line, beside it in a margin, is its note: a
            // heading stands over its provision, not beside it.
            let beside = |caption: Slot, opener: Slot| {
                let (note, line) = (slots.line(pages, caption), slots.line(pages, opener));
                let row = evidence.row(slots, caption);
                let level = row != u32::MAX && row == evidence.row(slots, opener)
                    || (note.bbox[1] - line.bbox[1]).abs() <= (line.bbox[3] - line.bbox[1]).max(1.0);
                level && (note.bbox[2] <= line.bbox[0] || note.bbox[0] >= line.bbox[2])
            };
            for ((caption, opener), &one) in captions.iter().zip(&governs) {
                let key = look(caption[0]);
                let (members, share) = groups
                    .iter()
                    .find(|(seen, _, _)| *seen == key)
                    .map_or((0, 0.0), |(_, members, ones)| (*members, *ones as f64 / *members as f64));
                if !(beside(caption[0], openers[*opener].2) || (one || members >= 4) && share >= 0.6) {
                    continue;
                }
                for (k, &slot) in caption.iter().enumerate() {
                    if self.claim(slot, Role::MarginalNote, Rule::MarginalNote) {
                        self.joined[slot.index()] = k > 0;
                        decided.push(slot);
                    }
                }
            }
        }
        // A note's twin in the other language's column, on its row.
        for slot in decided {
            let page = slots.page(slot);
            let row = evidence.row(slots, slot);
            if row == u32::MAX {
                continue;
            }
            for twin in slots.on(page).map(Slot) {
                if fences.translation[twin.index()] != fences.translation[slot.index()]
                    && evidence.row(slots, twin) == row
                    && self.open(twin)
                    && legal_structure_model::statute::marginal_note_shaped(text(twin))
                {
                    self.claim(twin, Role::MarginalNote, Rule::MarginalNote);
                }
            }
        }
    }

    /// The caption set in the page's margin column level with an opener: the open lines of its
    /// context on the column's side (the side holding fewer lines), the first shaped as a note and
    /// level with the opener's first line, with the lines wrapped under it (three at most).
    fn margin_caption(&self, pages: &[Page], slots: &Slots, fences: &Fences, opener: Slot, shaped: &impl Fn(Slot) -> bool) -> Vec<Slot> {
        let page_slot = slots.page(opener);
        let columns = fences.columns[page_slot];
        if columns.kind != "margin_column" {
            return Vec::new();
        }
        let left = columns.left_count < columns.right_count;
        let line = slots.line(pages, opener);
        let height = (line.bbox[3] - line.bbox[1]).max(1.0);
        let in_margin = |slot: Slot| {
            let other = slots.line(pages, slot);
            let middle = (other.bbox[0] + other.bbox[2]) / 2.0;
            self.open(slot) && fences.ctx[slot.index()] == fences.ctx[opener.index()] && (middle < columns.split_x) == left
        };
        let mut margin: Vec<Slot> = slots.on(page_slot).map(Slot).filter(|&slot| in_margin(slot)).collect();
        margin.sort_by(|a, b| slots.line(pages, *a).bbox[1].total_cmp(&slots.line(pages, *b).bbox[1]));
        let Some(first) = margin.iter().position(|&slot| (slots.line(pages, slot).bbox[1] - line.bbox[1]).abs() <= height && shaped(slot)) else {
            return Vec::new();
        };
        let mut caption = vec![margin[first]];
        for &slot in &margin[first + 1..] {
            let (above, below) = (slots.line(pages, caption[caption.len() - 1]), slots.line(pages, slot));
            if caption.len() == 3 || below.bbox[1] - above.bbox[3] > 0.5 * height {
                break;
            }
            caption.push(slot);
        }
        caption
    }

    /// The document's title, in each context outside a quotation: the first heading after its
    /// court's header and its parties and before its body opens (its first full line of prose), a
    /// layout model's title among them, where the model labels it so or it is set as titles are
    /// (centred in capitals or larger type, or larger in mixed case).
    fn title(&mut self, pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, sequences: &Sequences, geometry: &Geometry) {
        for (ctx, lines) in sequences.lines.iter().enumerate() {
            if fences.contexts[ctx].quotation != 0 {
                continue;
            }
            let measure = |slot: Slot| &geometry.page(slots.page(slot)).0;
            let opens = lines.iter().position(|&slot| {
                let line = slots.line(pages, slot);
                let measure = measure(slot);
                body_like(self.role[slot.index()])
                    && line.text.trim().chars().count() >= 60
                    && line.bbox[2] - line.bbox[0] >= 0.85 * (measure.right - measure.left)
            });
            let before = &lines[..opens.unwrap_or(lines.len())];
            let parties = before
                .iter()
                .rposition(|&slot| evidence.words[slot.index()] & (WORD_PARTY | WORD_FIELD) != 0)
                .map_or(0, |at| at + 1);
            for (k, &slot) in before.iter().enumerate().skip(parties) {
                let at = slot.index();
                let cue = evidence.tag[at] == SourceTag::Title;
                if self.role[at] != Role::Heading && !(cue && self.role[at] == Role::Open) {
                    continue;
                }
                let line = slots.line(pages, slot);
                let measure = measure(slot);
                let width = pages[slots.page(slot)].width.max(1.0);
                let centred = (((line.bbox[0] + line.bbox[2]) / 2.0 - width / 2.0) / width).abs() < 0.03
                    && line.bbox[2] - line.bbox[0] < 0.8 * (measure.right - measure.left);
                let caps = all_caps(line.text.trim());
                let larger = measure.size > 0.0 && f64::from(evidence.half_points[at]) / 2.0 >= measure.size * 1.1;
                let set = cue || evidence.title_shaped[at] && (centred && (caps || larger) || larger && !caps);
                if set && self.anchor[at] == 0 {
                    let wrapped = before[k + 1..]
                        .iter()
                        .take_while(|next| self.joined[next.index()] && self.role[next.index()] == Role::Heading)
                        .count();
                    for &title in &before[k..=k + wrapped] {
                        self.role[title.index()] = Role::Title;
                        self.rule[title.index()] = Rule::Title;
                    }
                }
                break;
            }
        }
    }

    /// The disagreements of strong rules, one diagnostic for each pair of rules: the rule that
    /// kept the lines, the rule refused, and the lines.
    fn report(&mut self, pages: &[Page], slots: &Slots) {
        if self.conflicts.is_empty() {
            return;
        }
        let mut conflicts = std::mem::take(&mut self.conflicts);
        conflicts.sort_unstable_by_key(|(kept, refused, slot)| (*kept as u8, *refused as u8, *slot));
        for pair in conflicts.chunk_by(|a, b| a.0 == b.0 && a.1 == b.1) {
            let (kept, refused, first) = pair[0];
            let mut diagnostic = Diagnostic::info(
                "LINE_RULES_DISAGREE",
                format!("Rule {} kept lines that rule {} read otherwise.", kept.name(), refused.name()),
                Some(pages[slots.page(first)].index),
            );
            diagnostic.line_ids = pair.iter().map(|(_, _, slot)| slots.line(pages, *slot).id.clone()).collect();
            diagnostic.details.insert("kept".to_owned(), json!(kept.name()));
            diagnostic.details.insert("refused".to_owned(), json!(refused.name()));
            self.diagnostics.push(diagnostic);
        }
    }
}

/// A quotation's own number: the quoted text's, none of the document's.
fn quoted(member: &Member) -> bool {
    member.observations.contains(&CandidateObservationV2::Quotation)
}

/// Front matter no heading is: a label (counsel, a party's role, contact details, "WHEREAS"), a
/// jurat, a date, a signature, the line naming who gives the reasons ("MARTLAND J.—",
/// "Fitzpatrick J.:"), a field's label with its blank, its value or another line on its row, a
/// lead-in's colon after a joining word, a court's name heading its document's first two pages or
/// standing over its parties (a party's role, a field or a label within the twelve lines after it).
fn front_matter(pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, slot: Slot) -> bool {
    let words = evidence.words[slot.index()];
    let line = slots.line(pages, slot);
    let text = line.text.trim();
    let court = || {
        let document = fences.document[slot.index()];
        let opening = fences.order.iter().find(|first| fences.document[first.index()] == document).map_or(0, |first| slots.page(*first));
        let place = fences.place[slot.index()] as usize;
        slots.page(slot) < opening + 2
            || fences.order[place + 1..]
                .iter()
                .take(12)
                .any(|next| evidence.words[next.index()] & (WORD_PARTY | WORD_FIELD | WORD_LABEL) != 0)
    };
    if words & (WORD_LABEL | WORD_PARTY | WORD_JURAT | WORD_DATE | WORD_SIGNATURE | WORD_AUTHOR) != 0
        || text.strip_suffix(':').is_some_and(legal_structure_model::judgment::signature_line)
        || joining_colon(text)
        || words & WORD_COURT != 0 && court()
    {
        return true;
    }
    if words & WORD_FIELD == 0 && !crate::paragraphs::field_label(line) {
        return false;
    }
    let row = evidence.row(slots, slot);
    text.contains("___")
        || text.split_once(':').is_some_and(|(_, value)| !value.trim().is_empty())
        || row != u32::MAX
            && slots.on(slots.page(slot)).map(Slot).any(|other| other != slot && evidence.row(slots, other) == row && !slots.line(pages, other).text.trim().is_empty())
}
