//! The structure the moves write: the units they make, the pairer's notes over them, the pages, the outline and
//! each provision's path, as nodes over the document's text.

use crate::model::fences::Fences;
use crate::model::{Evidence, Runs, Slots};
use crate::model::{LOOK_BOLD, LOOK_CAPS, LOOK_ITALIC, WORD_DIVISION, WORD_HISTORY};
use crate::text::text_index::PdfTextIndex;
use crate::model::machine::machine::{LineView, Move};
use crate::model::machine::machine::{MARGIN, OPENING};
use crate::model::regions::{
    FURNITURE, HEADING, LINE_LABEL, NOTE, PAGE_LABEL, PROVISION, REGIONS, TABLE, TEXT,
};
use crate::model::{Place, StructureIdentity, ORIGIN};
use crate::model::Decided;
use crate::text::joining::{seam, split_words};
use legal_pdf_core::model::Page;
use legal_pdf_core::{Error, Result};
use legal_structure_model::DocumentStructure;
use legal_structure_model::StructureNode;
use legal_structure_model::{
    utf16_len, ContinuationKind, NodeKind, ScalarRange, ScalarText, SourceContinuation, SourceJoin,
};
use pdf_inspector::markdown::Join;
use std::collections::{BTreeMap, HashMap, HashSet};
use legal_structure_model::{Derivation, Origin, PrimaryRole};

pub struct Unit {
    pub region: u8,
    /// The line whose move opened the unit, and the unit's lines as they are written.
    pub opener: usize,
    pub lines: Vec<usize>,
    pub marker: Option<(usize, usize)>,
    /// The line `marker` stands on: the unit's first, or the margin's line numbering it.
    pub marker_line: usize,
    pub label: Option<String>,
    /// A heading's level (0 for none).
    pub level: usize,
    /// The unit's last line with text, and the page it holds lines on now with where its first there starts.
    pub last: Option<usize>,
    pub page_start: Option<(usize, usize)>,
}

/// The opening token in a line's text, in UTF-16 code units: its first and past-last, leading spaces
/// skipped; with `word`, the token after it too ("PART II").
pub fn opening(text: &str, word: bool) -> Option<(usize, usize)> {
    let past = |from: usize, blank: bool| {
        from + text[from..]
            .find(|c: char| c.is_whitespace() != blank)
            .unwrap_or(text.len() - from)
    };
    let start = past(0, true);
    if start == text.len() {
        return None;
    }
    let mut end = past(start, false);
    if word {
        end = past(past(end, true), false);
    }
    Some((utf16_len(&text[..start]), utf16_len(&text[..end])))
}

/// A folio set apart as its own field of a line (a running head's title, then a tab, an em space or a wide gap, then the
/// page's number): the field at the line's end, else at its start, printing a number in a folio's few characters; and
/// the line's other fields. Each in UTF-16 code units of the line; none where the line is one field.
pub fn folio_field(text: &str) -> Option<((usize, usize), Vec<(usize, usize)>)> {
    let gap = |c: char| matches!(c, '\t' | '\u{8}' | '\u{2001}' | '\u{2002}' | '\u{2003}');
    let mut fields: Vec<(usize, usize)> = Vec::new();
    let (mut start, mut previous) = (None, ' ');
    for (at, c) in text.char_indices() {
        let breaks = gap(c) || (c == ' ' && previous == ' ');
        if breaks || c.is_whitespace() {
            if breaks {
                if let Some(from) = start.take() {
                    fields.push((from, text[..at].trim_end().len()));
                }
            }
        } else if start.is_none() {
            start = Some(at);
        }
        previous = if gap(c) { ' ' } else { c };
    }
    if let Some(from) = start {
        fields.push((from, text.trim_end().len()));
    }
    // Its letters and figures figures only, after at most a three-letter prefix ("719", "- 12 -", "E-018").
    let folio = |&(from, to): &(usize, usize)| {
        let printed: String = text[from..to]
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect();
        let figures = printed.trim_start_matches(|c: char| c.is_alphabetic());
        text[from..to].chars().count() <= 12
            && !figures.is_empty()
            && printed.len() - figures.len() <= 3
            && figures.chars().all(|c| c.is_ascii_digit())
    };
    // A field naming the page ("Page:  2", "2  of  5") is the folio's own words, not a running head.
    let naming = |&(from, to): &(usize, usize)| {
        let word = text[from..to].trim_end_matches([':', '.']).to_lowercase();
        matches!(
            word.as_str(),
            "page" | "pages" | "pg" | "p" | "pp" | "of" | "de" | "sur" | "/" | "-"
        )
    };
    if fields.len() < 2 || fields.iter().any(naming) {
        return None;
    }
    let at = if folio(&fields[fields.len() - 1]) {
        fields.len() - 1
    } else if folio(&fields[0]) {
        0
    } else {
        return None;
    };
    let utf16 = |(from, to): (usize, usize)| (utf16_len(&text[..from]), utf16_len(&text[..to]));
    let label = utf16(fields.remove(at));
    Some((label, fields.into_iter().map(utf16).collect()))
}

/// A reader's notes, written from its moves: each unit a note's line opens with its marks chosen (`Move::answers`) a
/// Footnote over the lines its moves put in it, its label where its first line prints the number it counts on, its body
/// from past the label, its calls the marks it answers. (Tier 0's notes are the pairer's.)
pub fn moved_notes(
    pages: &[Page],
    slots: &Slots,
    evidence: &Evidence,
    fences: &Fences,
    moves: &[Move],
    index: &PdfTextIndex,
) -> Vec<StructureNode> {
    let line = |k: usize| slots.line(pages, fences.order[k]);
    // Each note's lines, by its first line's place: a line resumes the unit its move names, which opens before it.
    let mut held: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (k, chosen) in moves.iter().enumerate() {
        if chosen.opens && chosen.region == NOTE && chosen.answers.is_some() {
            held.insert(k, vec![k]);
        } else if let Some(lines) = chosen
            .resumes
            .and_then(|first| held.get_mut(&(first as usize)))
        {
            lines.push(k);
        }
    }
    let mut notes = Vec::with_capacity(held.len());
    for (first, lines) in held {
        let chosen = &moves[first];
        // The label, on its line: the first marker the reading names, wherever in the note it stands; else, on the
        // note's first line, the number the move counts on where the label reader found it, else the figures the line
        // opens on; a note symbol's its opening token.
        let label: Option<(usize, usize, usize)> = match &chosen.markers {
            Some(named) => named
                .first()
                .map(|&(place, start, end)| (place as usize, start as usize, end as usize)),
            None => chosen.note.and_then(|value| {
                let text = line(first).text.as_str();
                let found = evidence.label[fences.order[first].index()]
                    .filter(|label| label.value == value);
                found
                    .and_then(|label| {
                        opening(text, false)
                            .map(|(start, _)| (start, start + usize::from(label.end).max(1)))
                    })
                    .or_else(|| {
                        crate::text::leading_figures(text)
                            .filter(|&(figures, _)| figures == value)
                            .map(|(_, range)| range)
                    })
                    .or_else(|| (value == 0).then(|| opening(text, false)).flatten())
                    .map(|(start, end)| (first, start, end))
            }),
        };
        let label_range = label
            .and_then(|(place, start, end)| index.global_range(&line(place).id, start, end))
            .filter(|range| range.start < range.end);
        // The body opens past the label and what sets it off on the label's line (".", ")", a space); past a label the
        // reading names, past its space alone (what the reading left out of the label is the note's text).
        let named = chosen.markers.is_some();
        let body_start = label.zip(label_range).map(|((place, _, end), placed)| {
            let text = line(place).text.as_str();
            let past = text
                .char_indices()
                .scan(0usize, |units, (byte, c)| {
                    let at = *units;
                    *units += c.len_utf16();
                    Some((at, byte))
                })
                .find(|&(units, _)| units >= end)
                .map_or(text.len(), |(_, byte)| byte);
            let tail = &text[past..];
            let set_off = |c: char| c.is_whitespace() || (!named && ".)],:;-".contains(c));
            placed.end + utf16_len(&tail[..tail.len() - tail.trim_start_matches(set_off).len()])
        });
        let mut ranges: Vec<ScalarRange> = lines
            .iter()
            .filter_map(|&k| {
                let whole = index.line(&line(k).id)?.range;
                let start = body_start
                    .filter(|_| label.is_some_and(|(place, ..)| place == k))
                    .map_or(whole.start, |start| start.clamp(whole.start, whole.end));
                (start < whole.end).then_some(ScalarRange {
                    start,
                    end: whole.end,
                })
            })
            .collect();
        ranges.sort_by_key(|range| range.start);
        // (A note of a label alone is its label; one with no indexed text is none.)
        let Some(range) = ranges
            .first()
            .zip(ranges.last())
            .map(|(first, last)| ScalarRange {
                start: first.start,
                end: last.end,
            })
            .or(label_range)
        else {
            continue;
        };
        let mut node = StructureNode::new(
            format!("note-{:06}", notes.len() + 1),
            NodeKind::Footnote,
            range,
            ORIGIN,
            Derivation::Heuristic,
            None,
        );
        node.ranges = ranges;
        node.line_ids = lines.iter().map(|&k| line(k).id.clone()).collect();
        node.page_indexes = index.page_indexes_for_line_ids(&node.line_ids);
        node.marker_range = label_range;
        node.calls = chosen
            .answers
            .iter()
            .flatten()
            .filter_map(|&(place, start, end)| {
                index.global_range(&line(place as usize).id, start as usize, end as usize)
            })
            .collect();
        node.label = chosen
            .note
            .filter(|&value| value > 0)
            .map(|value| value.to_string());
        notes.push(node);
    }
    notes
}

/// The structure the moves write: the units they make, the notes laid over them (the pairer's over tier 0's moves, a
/// reader's own, `moved_notes`), the pages, the outline and each provision's path, as nodes over the document's text
/// (`index`). `order`: the lines' places as they are written. `reader`: the moves are a reader's (tier 1's, its
/// oracle's), and what they say is written as they say it: a quotation a Quote over its unit, a heading's level and a
/// body unit's depth stated on its node, the markers it names the unit's, nothing of tier 0's reading (its quotations,
/// its judgment paragraphs) laid over them.
#[allow(clippy::too_many_arguments)]
pub fn write_structure(
    pages: &[Page],
    slots: &Slots,
    evidence: &Evidence,
    fences: &Fences,
    runs: &Runs,
    decision: &crate::model::decide::Decision,
    views: &[LineView],
    index: &PdfTextIndex,
    decided: &Decided,
    notes: Vec<StructureNode>,
    identity: &StructureIdentity,
    order: &[usize],
    reader: bool,
) -> Result<DocumentStructure> {
    // The units the moves make: each line opening one or in the one its move names; as a line joins its unit the seam
    // with the unit's line before it is written (`joining::seam`): a dropped hyphen a manuscript edit, a join with no
    // space a join, a page crossed a continuation; and each hyphen inside a line that splits a word
    // (`joining::split_words`) an edit. A line's move is DOC's where it read one and neither reads the line as a note's,
    // the run members tier 0 opens on kept: DOC reads the same table, so its reading of a marker names one of this
    // reading's. The pairer's notes lie over the units: a line a note holds is the note's and no unit's, and a line of
    // the note band no note holds is the body's, a paragraph of its own from each note label.
    let moves = &decision.moves;
    let mut notes = notes;
    // (TEMP diagnostic's watched lines, read once.)
    let watched_lines = std::env::var("LEGALPDF_DEBUG_LINES").ok();
    let texts: Vec<String> = fences
        .order
        .iter()
        .map(|&slot| slots.line(pages, slot).text.clone())
        .collect();
    let ids: Vec<String> = fences
        .order
        .iter()
        .map(|&slot| slots.line(pages, slot).id.clone())
        .collect();
    // Each line's source fragment (UTF-16) and page, what a seam is written in.
    let fragment = |id: &str| index.line(id).map(|f| (f.range, f.page_index));
    let vocabulary = legal_pdf_core::profile::tally("finish.vocabulary", || {
        crate::text::joining::vocabulary(pages)
    });
    let mut edits: HashMap<String, Vec<(usize, usize, String)>> = HashMap::new();
    let mut joins: Vec<SourceJoin> = Vec::new();
    let mut continuations: Vec<SourceContinuation> = Vec::new();
    let place_of: HashMap<&str, usize> = ids
        .iter()
        .enumerate()
        .map(|(k, id)| (id.as_str(), k))
        .collect();
    let mut noted = vec![false; ids.len()];
    for k in notes
        .iter()
        .flat_map(|note| &note.line_ids)
        .filter_map(|id| place_of.get(id.as_str()))
    {
        noted[*k] = true;
    }
    let mut units: Vec<Unit> = Vec::new();
    // Each unit by its first line, and each line's unit.
    let mut unit_at: HashMap<usize, usize> = HashMap::new();
    let mut unit_of: Vec<Option<usize>> = vec![None; ids.len()];
    // The body unit the note band's lines no note holds go on in.
    let mut given: Option<usize> = None;
    // The lines the parser opened a constituent document on; a start on a line of no region passes to the next unit.
    let mut started: Vec<usize> = Vec::new();
    for (k, line) in views.iter().enumerate() {
        let chosen = &moves[k];
        if chosen.starts {
            started.push(k);
        }
        let back = chosen.region == NOTE;
        // TEMP diagnostic (remove before landing; prints only for the line ids in LEGALPDF_DEBUG_LINES).
        if let Some(watched) = watched_lines.as_deref() {
            if watched.split(',').any(|w| w == ids[k]) {
                let holder = notes.iter().find(|note| note.line_ids.iter().any(|id| *id == ids[k]));
                eprintln!("LINE {} region {} noted {} note {:?}", ids[k], chosen.region, noted[k],
                    holder.map(|note| (note.id.clone(), note.kind, note.label.clone(), note.line_ids.len(), note.line_ids.first().cloned())));
            }
        }
        if chosen.region == 0 || noted[k] {
            given = None;
            continue;
        }
        let region = if back { TEXT } else { chosen.region };
        let single = matches!(region, LINE_LABEL | PAGE_LABEL);
        let held = if back {
            given.filter(|_| evidence.label[fences.order[k].index()].is_none())
        } else {
            chosen
                .resumes
                .and_then(|first| unit_at.get(&(first as usize)).copied())
        };
        let u = if let Some(u) = held {
            u
        } else {
            // The unit's marker: the reading's own where it names its markers (the first, wherever it stands), else the
            // marker of the line's reading; a unit counting on the margin's number prints its marker on the margin's line.
            let named = chosen.markers.as_ref().filter(|_| !back);
            let marker_line = match (named, chosen.reading, line.margin()) {
                (Some(named), ..) => named.first().map_or(k, |mark| mark.0 as usize),
                (None, Some(MARGIN), Some((margin, _))) if !back => margin as usize,
                _ => k,
            };
            let text = texts[marker_line].clone();
            let marker = if let Some(named) = named {
                named
                    .first()
                    .map(|&(_, start, end)| (start as usize, end as usize))
            } else if back {
                None
            } else if chosen.reading == Some(OPENING) {
                opening(&text, false)
            } else {
                line.reading(chosen.reading)
                    .and_then(|r| opening(&text, r.worded()))
            };
            // A unit's label is its marker as printed ("12", "(a)", "[4]"), a line number's or page label's its text.
            let printed = |(start, end): (usize, usize)| {
                ScalarText::new(&text)
                    .slice_utf16(start..end)
                    .unwrap_or_default()
                    .trim()
                    .trim_end_matches('.')
                    .to_owned()
            };
            let label = if single {
                Some(text.trim().to_owned())
            } else {
                marker.map(printed)
            };
            let level = if chosen.opens && region == HEADING {
                chosen.level.max(1) as usize
            } else {
                0
            };
            units.push(Unit {
                region,
                opener: k,
                lines: Vec::new(),
                marker,
                marker_line,
                label,
                level,
                last: None,
                page_start: None,
            });
            unit_at.insert(k, units.len() - 1);
            units.len() - 1
        };
        given = back.then_some(u);
        unit_of[k] = Some(u);
    }
    // Each unit's lines, in the order they are written (`order`, each written line's place: the parser's order, or a
    // reader's of each page's lines).
    for (k, u) in order.iter().filter_map(|&k| Some((k, unit_of[k]?))) {
        let region = units[u].region;
        units[u].lines.push(k);
        let written = texts[k].trim();
        if !matches!(
            REGIONS[usize::from(region)],
            "furniture" | "page_label" | "line_label"
        ) && !written.is_empty()
        {
            if let Some((range, page)) = fragment(&ids[k]) {
                let unit = &mut units[u];
                if let Some(j) = unit.last {
                    let before = texts[j].trim();
                    let separator = match seam(before, written, &vocabulary) {
                        Join::Plain => {
                            // The dropped hyphen's UTF-16 offset in its line, `before` being the line trimmed.
                            let at = utf16_len(
                                &texts[j][..texts[j].len() - texts[j].trim_start().len()],
                            ) + utf16_len(before)
                                - 1;
                            edits.entry(ids[j].clone()).or_default().push((
                                at,
                                at + 1,
                                String::new(),
                            ));
                            ""
                        }
                        Join::Hyphen => "",
                        Join::Keep => " ",
                    };
                    if separator.is_empty() {
                        joins.push(SourceJoin {
                            from: fragment(&ids[j])
                                .expect("a unit's last line is indexed")
                                .0
                                .start,
                            to: range.start,
                            separator: String::new(),
                        });
                    }
                    if let Some((_, from)) = unit.page_start.filter(|&(held, _)| held != page) {
                        let kind = if unit.region == HEADING {
                            ContinuationKind::Heading
                        } else {
                            ContinuationKind::Paragraph
                        };
                        continuations.push(SourceContinuation {
                            kind,
                            from,
                            to: range.start,
                            separator: Some(separator.to_owned()),
                        });
                    }
                }
                if unit.page_start.is_none_or(|(held, _)| held != page) {
                    unit.page_start = Some((page, range.start));
                }
                unit.last = Some(k);
            }
            for at in split_words(&texts[k], &vocabulary) {
                edits
                    .entry(ids[k].clone())
                    .or_default()
                    .push((at, at + 1, String::new()));
            }
        }
    }
    // Nodes, their ranges in UTF-16 code units of the text, as the index places the lines; their parents are the
    // outline's (`assemble_outline`).
    let line_range = |k: usize| index.line(&ids[k]).map(|l| l.range);
    // Where a run member's place stands in the text.
    let placed = |place: Place| {
        index
            .line(&ids[fences.place[place.slot.index()] as usize])
            .map(|line| line.range.start + place.offset as usize)
    };
    let mut nodes: Vec<StructureNode> = Vec::new();
    // A written page's line set apart keeps the id its line's kind gives it, as the projection reads such lines.
    let folios: HashSet<&str> = pages
        .iter()
        .filter_map(|page| page.printed_label_line_id.as_deref())
        .collect();
    let unit_ids: Vec<String> = units
        .iter()
        .enumerate()
        .map(|(u, unit)| {
            let id = ids[unit.opener].as_str();
            match REGIONS[usize::from(unit.region)] {
                "line_label" => format!("{id}:line"),
                "furniture" | "page_label" => format!(
                    "{id}:{}",
                    if folios.contains(id) {
                        "page"
                    } else {
                        "furniture"
                    }
                ),
                _ => format!("unit-{:06}", u + 1),
            }
        })
        .collect();
    let node_id = |u: usize| unit_ids[u].clone();
    // TEMP lever "ladder-depths" (remove before landing): each node's own depth, by its id, where its line opens more than
    // one rung of a provision ladder.
    let ladder_depths =
        legal_structure_model::lever("ladder-depths");
    let count_roots =
        legal_structure_model::lever("count-roots");
    let mut own_depths: HashMap<String, usize> = HashMap::new();
    // Each page a node: its text's range (an empty one where its text would open, where it prints none), its index, and
    // where its text was read from.
    let mut page_end = 0;
    for page in pages.iter() {
        let range = index.page_range(page.index).unwrap_or(ScalarRange {
            start: page_end,
            end: page_end,
        });
        page_end = range.end;
        let mut node = StructureNode::new(
            page.id.clone(),
            NodeKind::Page,
            range,
            ORIGIN,
            Derivation::Native,
            None,
        );
        node.label = Some(page.number.to_string());
        node.aliases = page.printed_label.clone().map(|label| vec![label]);
        node.anchor.clone_from(&page.printed_label_line_id);
        node.line_ids = page.lines.iter().map(|l| l.id.clone()).collect();
        node.page_indexes = vec![page.index];
        node.properties
            .insert("source".to_owned(), page.source.clone());
        nodes.push(node);
    }
    for (u, unit) in units.iter().enumerate() {
        // (A reader's table unit is a row of its table, written with the tables below.)
        if reader && unit.region == TABLE {
            continue;
        }
        let ranges: Vec<ScalarRange> = unit.lines.iter().filter_map(|&k| line_range(k)).collect();
        // TEMP diagnostic (remove before landing; prints only for the line ids in LEGALPDF_DEBUG_LINES).
        if let Some(watched) = watched_lines.as_deref() {
            if unit.lines.iter().any(|&k| watched.split(',').any(|w| w == ids[k])) {
                eprintln!("UNIT {u} region {} lines {:?} ranges {}", unit.region, unit.lines.iter().map(|&k| ids[k].clone()).collect::<Vec<_>>(), ranges.len());
            }
        }
        let Some(envelope) = ranges
            .iter()
            .map(|r| r.start)
            .min()
            .zip(ranges.iter().map(|r| r.end).max())
        else {
            continue;
        };
        let range = ScalarRange {
            start: envelope.0,
            end: envelope.1,
        };
        let region = if unit.region == TABLE {
            "text"
        } else {
            REGIONS[usize::from(unit.region)]
        };
        // The member a settled unit opens on (tier 0's move at its first line); a heading numbering a provision is that
        // provision, a heading still. A unit whose reading names its markers is written from them, not from the member
        // tier 0 read on its line.
        let opening = &moves[unit.opener];
        let named = opening.markers.as_ref();
        let member = opening
            .member
            .filter(|_| named.is_none())
            .map(|(run, at)| &runs.runs[run as usize].members[at as usize]);
        let (kind, role) = match region {
            "heading"
                if member.is_some_and(|member| {
                    member.kind == Some(legal_structure_model::UnitRole::Section)
                }) =>
            {
                (NodeKind::Section, Some(PrimaryRole::Heading))
            }
            "heading" => (NodeKind::Heading, None),
            "text" if unit.marker.is_some() => (NodeKind::Paragraph, None),
            "provision" => (NodeKind::Section, Some(PrimaryRole::Text)),
            "list_item" => (NodeKind::ListItem, None),
            "title" => (NodeKind::Prose, Some(PrimaryRole::Title)),
            "caption" => (NodeKind::Prose, Some(PrimaryRole::Caption)),
            "headnote" => (NodeKind::Prose, Some(PrimaryRole::Headnote)),
            "marginal_note" => (NodeKind::Prose, Some(PrimaryRole::MarginalNote)),
            "furniture" => (NodeKind::Prose, Some(PrimaryRole::Furniture)),
            "page_label" => (NodeKind::Prose, Some(PrimaryRole::PageLabel)),
            "line_label" => (NodeKind::Prose, Some(PrimaryRole::LineLabel)),
            _ => (NodeKind::Prose, None),
        };
        // A unit is the document's own, as tier 0's are; a heading is read from its look.
        let derivation = if kind != NodeKind::Heading {
            Derivation::Native
        } else {
            Derivation::Heuristic
        };
        let mut node = StructureNode::new(node_id(u), kind, range, ORIGIN, derivation, None);
        node.role = role;
        node.ranges = ranges.clone();
        node.line_ids = unit.lines.iter().map(|&k| ids[k].clone()).collect();
        let marker_range = unit
            .marker
            .and_then(|(start, end)| index.global_range(&ids[unit.marker_line], start, end));
        let lines: Vec<&str> = unit.lines.iter().map(|&k| texts[k].as_str()).collect();
        // A settled unit is addressed as the run it opens on reads it: a paragraph by its count, a provision by its
        // grammar's value (a clause, a part, an article), its printing an alias; its marker and text where they stand.
        // Its marker is the run member's where the unit opens on one, else a heading's number as its text reads it, else
        // the unit's own (a note's none); its label a heading's text, else the member's, else the unit's own, a numbered
        // paragraph's its number ("12" for "[12]" or "12."), as the document prints it.
        node.marker_range = member
            .and_then(|member| placed(member.marker[0]).zip(placed(member.marker[1])))
            .map(|(start, end)| ScalarRange {
                start,
                end: end.max(start),
            })
            .or_else(|| match kind {
                NodeKind::Heading if named.is_none() => ranges.first().and_then(|first| {
                    crate::graph::heading_marker(first.start, lines[0], lines.len())
                }),
                _ => marker_range,
            });
        // Every marker the reading names, where it names more than the one the node's marker range holds.
        if let Some(named) = named.filter(|named| named.len() > 1) {
            let printed: Vec<String> = named
                .iter()
                .filter_map(|&(place, start, end)| {
                    index.global_range(&ids[place as usize], start as usize, end as usize)
                })
                .map(|range| format!("{}-{}", range.start, range.end))
                .collect();
            node.markup_attributes
                .insert("markers".to_owned(), printed.join(" "));
        }
        node.label = match (kind, member) {
            (NodeKind::Heading, _) => Some(lines.join(" ")),
            (NodeKind::Paragraph, Some(member)) => Some(member.grammar_value.clone()),
            (NodeKind::ListItem, Some(member)) => Some(member.label.clone()),
            (NodeKind::Section, Some(member)) => Some(member.grammar_value.clone()),
            (NodeKind::Paragraph, None) => unit
                .label
                .as_deref()
                .map(|label| label.trim_matches(|c: char| !c.is_ascii_digit()))
                .filter(|digits| {
                    !digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
                })
                .map(str::to_owned),
            _ => unit.label.clone(),
        };
        // (A heading's heading range is its reading context's, `outline_contexts`.)
        if kind == NodeKind::Section {
            node.locator_kind = Some(
                legal_structure_model::provision_locator(
                    None,
                    usize::from(opening.depth.max(1)).saturating_sub(1),
                )
                .to_owned(),
            );
            if member.is_none() {
                if let Some((start, end)) = unit.marker {
                    let text = &texts[unit.marker_line];
                    let byte_end = text
                        .char_indices()
                        .scan(0usize, |u, (b, c)| {
                            let at = *u;
                            *u += c.len_utf16();
                            Some((at, b))
                        })
                        .find(|&(u, _)| u >= end)
                        .map_or(text.len(), |(_, b)| b);
                    let tail = &text[byte_end..];
                    let pad = utf16_len(&tail[..tail.len() - tail.trim_start().len()]);
                    if let Some(mr) = index.global_range(&ids[unit.marker_line], start, end) {
                        node.content_start = Some((mr.end + pad).min(range.end));
                    }
                }
                if let Some(printed) = unit.label.as_deref() {
                    let norm =
                        printed.trim_matches(|c: char| matches!(c, '(' | ')' | '[' | ']' | '.'));
                    if !norm.is_empty() && norm != printed {
                        node.label = Some(norm.to_owned());
                        node.aliases = Some(vec![printed.to_owned()]);
                    }
                }
            }
        }
        if let Some(member) = member {
            node.content_start = placed(member.content);
            match kind {
                NodeKind::Paragraph => node.aliases = Some(vec![member.label.clone()]),
                NodeKind::Section => {
                    node.locator_kind = Some(
                        legal_structure_model::provision_locator(
                            member.locator_kind.as_deref(),
                            usize::from(member.level),
                        )
                        .to_owned(),
                    );
                    let printed =
                        (member.label != member.grammar_value).then(|| member.label.clone());
                    node.aliases = printed.map(|printed| vec![printed]);
                    if role == Some(PrimaryRole::Heading) {
                        node.heading_range = Some(range);
                    }
                }
                _ => {}
            }
        }
        // A contents list's lines are under the `contents` grammar. A numbered paragraph wholly in the translation prints
        // that paragraph in the other language (`paragraph_printings`), as the stage's nodes say.
        node.grammar = if region == "contents" {
            Some("contents".to_owned())
        } else if matches!(
            kind,
            NodeKind::Paragraph | NodeKind::Prose | NodeKind::Section
        ) && unit.lines.iter().all(|&k| views[k].translation())
        {
            Some("translation".to_owned())
        } else {
            None
        };
        // The labels opening speakers' turns in the unit, where its lines' moves name them.
        let mut speakers: Vec<String> = Vec::new();
        for &k in &unit.lines {
            for &(start, end) in moves[k].speakers.iter().flatten() {
                if let Some(range) = index.global_range(&ids[k], start as usize, end as usize) {
                    speakers.push(format!("{}-{}", range.start, range.end));
                }
            }
        }
        // (Named with none, a line has none: the unit states as much.)
        if unit.lines.iter().any(|&k| moves[k].speakers.is_some()) {
            node.markup_attributes
                .insert("speakers".to_owned(), speakers.join(" "));
        }
        // A reader's level for a heading and depth for a body unit, stated as it gave them: a heading two levels under
        // the one before stays two under, though no heading stands between them in the tree.
        if reader && opening.opens {
            if unit.level > 0 {
                node.markup_attributes
                    .insert("heading_level".to_owned(), unit.level.to_string());
            } else if opening.depth > 0 {
                node.markup_attributes
                    .insert("depth".to_owned(), opening.depth.to_string());
            }
        }
        // A folio printed as its own field of a running head ("R. c. BEAVER⇥719"): that field is the page label, the rest
        // of the line the head.
        let field = (region == "page_label" && unit.lines.len() == 1)
            .then(|| folio_field(&texts[unit.lines[0]]))
            .flatten();
        if let Some(((start, end), rest)) = field {
            let k = unit.lines[0];
            if let Some(label) = index.global_range(&ids[k], start, end) {
                node.range = label;
                node.ranges = vec![label];
                node.label = ScalarText::new(&texts[k])
                    .slice_utf16(start..end)
                    .map(|text| text.trim().to_owned());
                let head: Vec<ScalarRange> = rest
                    .iter()
                    .filter_map(|&(start, end)| index.global_range(&ids[k], start, end))
                    .collect();
                if let Some(envelope) =
                    head.first()
                        .zip(head.last())
                        .map(|(first, last)| ScalarRange {
                            start: first.start,
                            end: last.end,
                        })
                {
                    let mut furniture = StructureNode::new(
                        format!("{}:furniture", ids[k]),
                        NodeKind::Prose,
                        envelope,
                        ORIGIN,
                        Derivation::Native,
                        None,
                    );
                    furniture.role = Some(PrimaryRole::Furniture);
                    furniture.ranges = head;
                    furniture.line_ids = vec![ids[k].clone()];
                    nodes.push(furniture);
                }
            }
        }
        // TEMP diagnostic (remove before landing; prints only for the line ids in LEGALPDF_DEBUG_LINES).
        if let Some(watched) = watched_lines.as_deref() {
            if node.line_ids.iter().any(|id| watched.split(',').any(|w| w == id)) {
                eprintln!("WRITE {} {:?} label {:?} locator {:?} grammar {:?} role {:?} range {}..{} region {region} lines {:?}",
                    node.id, node.kind, node.label, node.locator_kind, node.grammar, node.role, node.range.start, node.range.end, node.line_ids);
            }
        }
        nodes.push(node);
        // (TEMP lever "ladder-depths", remove before landing.) Where the line opens more than one rung of a provision
        // ladder ("39 (1) Except ...": the section and its first subsection), each node stands at its own rung's depth:
        // the walk's depth for the line is its deepest rung's.
        if ladder_depths && named.is_none() && !opening.within.is_empty() {
            if let Some(member) = member.filter(|_| {
                opening.member.is_some_and(|(run, _)| runs.runs[run as usize].grammar == legal_structure_model::CandidateGrammar::Hierarchy)
            }) {
                own_depths.insert(node_id(u), usize::from(member.level) + 1);
            }
        }
        // A further marker on the unit's first line opens a unit under it, to the unit's end, the unit before it left with what
        // stands before that marker: "2(1) The Court has ...", the section "2" and its subsection.
        // (TEMP lever "count-roots", remove before landing.) A Part's, a Division's, an Article's or a Schedule's own unit
        // stands above the sections printed under it, which a count reads at the top of their own ladders: it holds them
        // as its heading does, to the next of its kind.
        if count_roots
            && named.is_none()
            && kind == NodeKind::Section
            && member.is_some_and(|member| {
                member.level == 0
                    && matches!(
                        member.locator_kind.as_deref(),
                        Some("part" | "division" | "article" | "schedule" | "exhibit" | "annex" | "appendix")
                    )
            })
        {
            own_depths.insert(node_id(u), 0);
        }
        let mut holder = nodes.len() - 1;
        for &(run, at) in opening.within.iter().filter(|_| named.is_none()) {
            let member = &runs.runs[run as usize].members[at as usize];
            if ladder_depths && runs.runs[run as usize].grammar == legal_structure_model::CandidateGrammar::Hierarchy {
                if let Some((start, _)) = placed(member.marker[0]).zip(placed(member.marker[1])).filter(|_| member.kind.is_some()) {
                    own_depths.insert(format!("{}-{start}", node_id(u)), usize::from(member.level) + 1);
                }
            }
            let (Some((start, end)), Some(role)) = (
                placed(member.marker[0]).zip(placed(member.marker[1])),
                member.kind,
            ) else {
                continue;
            };
            let kind = match role {
                legal_structure_model::UnitRole::Section => NodeKind::Section,
                legal_structure_model::UnitRole::Item => NodeKind::ListItem,
                legal_structure_model::UnitRole::Paragraph => NodeKind::Paragraph,
            };
            let above = &mut nodes[holder];
            above.range.end = start;
            above.ranges = above
                .ranges
                .iter()
                .filter(|part| part.start < start)
                .map(|part| ScalarRange {
                    start: part.start,
                    end: part.end.min(start),
                })
                .collect();
            above.line_ids.truncate(1);
            let mut own = StructureNode::new(
                format!("{}-{start}", node_id(u)),
                kind,
                ScalarRange {
                    start,
                    end: range.end,
                },
                ORIGIN,
                Derivation::Native,
                Some(above.id.clone()),
            );
            own.role = Some(PrimaryRole::Text);
            own.ranges = ranges
                .iter()
                .filter(|part| part.end > start)
                .map(|part| ScalarRange {
                    start: part.start.max(start),
                    end: part.end,
                })
                .collect();
            own.line_ids = unit.lines.iter().map(|&k| ids[k].clone()).collect();
            own.marker_range = Some(ScalarRange {
                start,
                end: end.max(start),
            });
            own.grammar.clone_from(&nodes[holder].grammar);
            own.content_start = placed(member.content);
            own.label = Some(if kind == NodeKind::Section {
                member.grammar_value.clone()
            } else {
                member.label.clone()
            });
            if kind == NodeKind::Section {
                own.locator_kind = Some(
                    legal_structure_model::provision_locator(
                        member.locator_kind.as_deref(),
                        usize::from(member.level),
                    )
                    .to_owned(),
                );
            }
            nodes.push(own);
            holder = nodes.len() - 1;
        }
        // A reader's quotation is a Quote over its unit too: a display quotation (one over sixty lines is a document
        // reproduced, which types nothing), or one set in a note, a note's quotation.
        if reader && matches!(region, "quote" | "note_quote") && unit.lines.len() <= 60 {
            let mut quote = StructureNode::new(
                format!("{}:quote", node_id(u)),
                NodeKind::Quote,
                range,
                ORIGIN,
                Derivation::Heuristic,
                None,
            );
            quote.ranges = ranges;
            quote.line_ids = unit.lines.iter().map(|&k| ids[k].clone()).collect();
            let grammar = if region == "quote" {
                "display_quotation"
            } else {
                "note_quotation"
            };
            quote.grammar = Some(grammar.to_owned());
            nodes.push(quote);
        }
    }
    // The pairer's notes, over the lines they claim, whatever the moves made of them: a note over a page goes on with a
    // space, and a hyphen inside its line that splits a word is an edit.
    for note in &notes {
        let mut held: Option<(usize, usize)> = None;
        for id in &note.line_ids {
            // (A line of no text is no place a note goes on from or to.)
            let Some((range, page)) = fragment(id).filter(|(range, _)| range.start < range.end) else {
                continue;
            };
            if let Some((_, from)) = held.filter(|&(on, _)| on != page) {
                continuations.push(SourceContinuation {
                    kind: ContinuationKind::Note,
                    from,
                    to: range.start,
                    separator: Some(" ".to_owned()),
                });
            }
            if held.is_none_or(|(on, _)| on != page) {
                held = Some((page, range.start));
            }
            let Some(&k) = place_of.get(id.as_str()) else {
                continue;
            };
            for at in split_words(&texts[k], &vocabulary) {
                edits
                    .entry(id.clone())
                    .or_default()
                    .push((at, at + 1, String::new()));
            }
        }
    }
    nodes.append(&mut notes);
    if reader {
        // A reader's tables, from its moves: its table units consecutive as they are written, with only furniture, page
        // and line labels and notes between, are one table, one grid a page, linked page to page; each unit a row, its
        // cells by column those its lines' moves name, else the table reader's on its lines (a line of one cell under a
        // row of several goes on in the row's last column), else its lines' one.
        let mut reader_cells: HashMap<&str, Vec<(usize, Option<(usize, usize)>)>> = HashMap::new();
        for cell in fences
            .tables
            .iter()
            .flat_map(|found| &found.rows)
            .flat_map(|row| &row.cells)
        {
            for part in &cell.parts {
                reader_cells
                    .entry(part.line_id.as_str())
                    .or_default()
                    .push((cell.column, part.chars));
            }
        }
        let mut position = vec![usize::MAX; ids.len()];
        order
            .iter()
            .enumerate()
            .for_each(|(at, &k)| position[k] = at);
        let mut read: Vec<usize> = (0..units.len())
            .filter(|&u| !units[u].lines.is_empty())
            .collect();
        read.sort_by_key(|&u| position[units[u].lines[0]]);
        let (mut tables, mut running): (Vec<Vec<usize>>, bool) = (Vec::new(), false);
        for u in read {
            match REGIONS[usize::from(units[u].region)] {
                "table" => {
                    if !running {
                        tables.push(Vec::new());
                        running = true;
                    }
                    tables.last_mut().expect("a table running").push(u);
                }
                "furniture" | "page_label" | "line_label" | "note" => {}
                _ => running = false,
            }
        }
        for (t, rows) in tables.iter().enumerate() {
            // Each row's cells by column, and the first of its lines holding a cell's text: the cells its lines' moves name
            // (one named with no text is a cell still, of its row's width), else the table reader's on its lines, else its
            // lines' one.
            let cells_of = |u: usize| -> (BTreeMap<usize, Vec<ScalarRange>>, Option<usize>) {
                let unit = &units[u];
                let columns = reader_cells
                    .get(ids[unit.lines[0]].as_str())
                    .map_or(0, |cells| cells.iter().map(|(c, _)| c + 1).max().unwrap_or(0));
                let mut cells: BTreeMap<usize, Vec<ScalarRange>> = BTreeMap::new();
                let mut first: Option<usize> = None;
                for &k in &unit.lines {
                    let id = ids[k].as_str();
                    let mut put = |column: usize, range: Option<ScalarRange>| {
                        let held = cells.entry(column).or_default();
                        if let Some(range) = range.filter(|range| range.start < range.end) {
                            held.push(range);
                            first.get_or_insert(k);
                        }
                    };
                    match (&moves[k].cells, reader_cells.get(id)) {
                        (Some(named), _) => {
                            for &(column, start, end) in named {
                                put(
                                    column as usize,
                                    index.global_range(id, start as usize, end as usize),
                                );
                            }
                        }
                        (None, Some(found)) => {
                            let wrapped = columns >= 2
                                && found.iter().map(|(c, _)| *c).collect::<HashSet<_>>().len() == 1;
                            for &(column, chars) in found {
                                let range = match chars {
                                    Some((start, end)) => index.global_range(id, start, end),
                                    None => line_range(k),
                                };
                                put(if wrapped { columns - 1 } else { column }, range);
                            }
                        }
                        (None, None) => put(columns.saturating_sub(1), line_range(k)),
                    }
                }
                (cells, first)
            };
            let read: Vec<(usize, BTreeMap<usize, Vec<ScalarRange>>, Option<usize>)> = rows
                .iter()
                .map(|&u| {
                    let (cells, first) = cells_of(u);
                    (u, cells, first)
                })
                .collect();
            // The table's width: its widest row's, among the rows holding a cell's text (every grid of it is that wide).
            let width = read
                .iter()
                .filter(|(_, _, first)| first.is_some())
                .filter_map(|(_, cells, _)| cells.keys().next_back().map(|column| column + 1))
                .max()
                .unwrap_or(0);
            // Each page's rows: a row is on the page its first cell's text stands on; a row of no cell's text is in no
            // grid, a row still.
            let mut by_page: Vec<(usize, Vec<usize>)> = Vec::new();
            for (at, (u, _, first)) in read.iter().enumerate() {
                let page = first.and_then(|k| index.line(&ids[k]).map(|line| line.page_index));
                let Some(page) = page else {
                    let unit = &units[*u];
                    let row_ranges: Vec<ScalarRange> =
                        unit.lines.iter().filter_map(|&k| line_range(k)).collect();
                    if let Some(hull) = row_ranges
                        .iter()
                        .map(|range| range.start)
                        .min()
                        .zip(row_ranges.iter().map(|range| range.end).max())
                    {
                        let mut row = StructureNode::new(
                            node_id(*u),
                            NodeKind::Row,
                            ScalarRange {
                                start: hull.0,
                                end: hull.1,
                            },
                            ORIGIN,
                            Derivation::Heuristic,
                            None,
                        );
                        row.role = Some(PrimaryRole::Text);
                        row.ranges = row_ranges;
                        row.line_ids = unit.lines.iter().map(|&k| ids[k].clone()).collect();
                        if moves[unit.opener].depth > 0 {
                            row.markup_attributes
                                .insert("depth".to_owned(), moves[unit.opener].depth.to_string());
                        }
                        nodes.push(row);
                    }
                    continue;
                };
                match by_page.iter_mut().find(|(held, _)| *held == page) {
                    Some((_, rows)) => rows.push(at),
                    None => by_page.push((page, vec![at])),
                }
            }
            // Where each page's grid opens: its first cell's first text, which the next page's goes on from.
            let mut chain: Vec<usize> = Vec::new();
            for (g, (_, rows)) in by_page.iter().enumerate() {
                let table_id = format!("table-{:04}-{:03}", t + 1, g + 1);
                let mut table_ranges: Vec<ScalarRange> = Vec::new();
                let mut grid: Vec<StructureNode> = Vec::new();
                let mut opens: Option<usize> = None;
                let mut widest = 0;
                for (r, &at) in rows.iter().enumerate() {
                    let (u, cells, _) = &read[at];
                    let unit = &units[*u];
                    let row_ranges: Vec<ScalarRange> =
                        unit.lines.iter().filter_map(|&k| line_range(k)).collect();
                    let Some(hull) = row_ranges
                        .iter()
                        .map(|range| range.start)
                        .min()
                        .zip(row_ranges.iter().map(|range| range.end).max())
                    else {
                        continue;
                    };
                    let row_id = node_id(*u);
                    let mut row = StructureNode::new(
                        row_id.clone(),
                        NodeKind::Row,
                        ScalarRange {
                            start: hull.0,
                            end: hull.1,
                        },
                        ORIGIN,
                        Derivation::Heuristic,
                        Some(table_id.clone()),
                    );
                    row.role = Some(PrimaryRole::Text);
                    row.ranges = row_ranges.clone();
                    row.row = Some(r + 1);
                    row.line_ids = unit.lines.iter().map(|&k| ids[k].clone()).collect();
                    row.page_indexes = index.page_indexes_for_line_ids(&row.line_ids);
                    if moves[unit.opener].depth > 0 {
                        row.markup_attributes
                            .insert("depth".to_owned(), moves[unit.opener].depth.to_string());
                    }
                    // Its cells of text; and, on the grid's last row, a cell of none in the table's last column where no
                    // row of the grid reaches it, so the grid is as wide as its table.
                    let last = r + 1 == rows.len();
                    for column in 0..width {
                        let mut ranges = cells.get(&column).cloned().unwrap_or_default();
                        ranges.sort_by_key(|range| range.start);
                        let blank = ScalarRange {
                            start: hull.1,
                            end: hull.1,
                        };
                        let (range, pads) = match (ranges.first(), ranges.iter().map(|range| range.end).max()) {
                            (Some(first), Some(end)) => (
                                ScalarRange {
                                    start: first.start,
                                    end,
                                },
                                false,
                            ),
                            _ if last && column + 1 == width && widest < width => (blank, true),
                            _ => continue,
                        };
                        let mut cell = StructureNode::new(
                            format!("{row_id}/col:{}", column + 1),
                            NodeKind::Cell,
                            range,
                            ORIGIN,
                            Derivation::Heuristic,
                            Some(row_id.clone()),
                        );
                        cell.ranges = ranges;
                        cell.row = Some(r + 1);
                        cell.column = Some(column + 1);
                        cell.page_indexes.clone_from(&row.page_indexes);
                        if !pads {
                            opens.get_or_insert(range.start);
                        }
                        widest = widest.max(column + 1);
                        grid.push(cell);
                    }
                    table_ranges.extend(row_ranges);
                    grid.push(row);
                }
                let (Some(start), Some(end)) = (
                    table_ranges.iter().map(|range| range.start).min(),
                    table_ranges.iter().map(|range| range.end).max(),
                ) else {
                    continue;
                };
                let mut table = StructureNode::new(
                    table_id,
                    NodeKind::Table,
                    ScalarRange { start, end },
                    ORIGIN,
                    Derivation::Heuristic,
                    None,
                );
                table.ranges = table_ranges;
                table.table_number = Some(t + 1);
                table.line_ids = rows
                    .iter()
                    .flat_map(|&at| units[read[at].0].lines.iter().map(|&k| ids[k].clone()))
                    .collect();
                table.page_indexes = index.page_indexes_for_line_ids(&table.line_ids);
                chain.extend(opens);
                nodes.push(table);
                nodes.extend(grid);
            }
            for pair in chain.windows(2) {
                continuations.push(SourceContinuation {
                    kind: ContinuationKind::Table,
                    from: pair[0],
                    to: pair[1],
                    separator: None,
                });
            }
        }
    } else {
        nodes.extend(crate::graph::table_nodes(
            index,
            fences.tables.iter().enumerate(),
            ORIGIN,
        )?);
        nodes.extend(crate::graph::quotation_nodes(index, decided, ORIGIN)?);
    }
    // Text tier 0 read as no region's lies outside the body: furniture. (A reader's line of no region is a line it says
    // nothing of, and no node is made of it.)
    for (k, chosen) in moves.iter().enumerate() {
        if reader || chosen.region != 0 || noted[k] || texts[k].trim().is_empty() {
            continue;
        }
        let Some(range) = line_range(k) else { continue };
        let mut node = StructureNode::new(
            format!("none-{:06}", k + 1),
            NodeKind::Prose,
            range,
            ORIGIN,
            Derivation::Heuristic,
            None,
        );
        node.role = Some(PrimaryRole::Furniture);
        node.ranges = vec![range];
        node.line_ids = vec![ids[k].clone()];
        nodes.push(node);
    }
    // The constituent documents the parser opened, each of the kind its move names, else the kind the constituents
    // reader names its opening line, else a document.
    let starts: Vec<crate::model::constituents::Start> = started
        .iter()
        .map(|&k| {
            // (A reader's document opens on the first line written of the unit its line opens.)
            let opening = unit_of[k]
                .filter(|_| reader)
                .and_then(|u| units[u].lines.first().copied());
            let id = ids[opening.unwrap_or(k)].clone();
            let kind = moves[k].document.clone().unwrap_or_else(|| {
                fences
                    .starts
                    .iter()
                    .find(|start| start.line_id == id)
                    .map_or("document", |start| start.kind())
                    .to_owned()
            });
            crate::model::constituents::Start::new(id, &kind)
        })
        .collect();
    nodes.extend(legal_pdf_constituents::constituents(index, pages, &starts));
    let mut structure = DocumentStructure::new(
        identity.document_id(),
        "local-pdf",
        index.text().to_owned(),
        ORIGIN,
    );
    structure.origins.push(Origin {
        id: legal_structure_model::NOTE_PAIRING_ORIGIN.to_owned(),
    });
    structure.source_sha256 = Some(identity.source_sha256.clone());
    // The source's own kind of document, where its first line's move names it.
    structure.doc_type = moves
        .first()
        .filter(|first| !first.starts)
        .and_then(|first| first.document.clone());
    structure.nodes = nodes;
    // The outline is tier 0's: each constituent document's and language's own stacks, a quotation within what quotes it.
    let levels: HashMap<&str, usize> = units
        .iter()
        .filter(|unit| unit.level > 0)
        .flat_map(|unit| unit.lines.iter().map(|&k| (ids[k].as_str(), unit.level)))
        .collect();
    let contexts = crate::graph::outline_contexts(
        &mut structure.nodes,
        &structure.document_id.clone(),
        index,
        decided,
    );
    // Each body unit's depth as the walk opened it, by its first line.
    let depths: HashMap<&str, usize> = moves
        .iter()
        .enumerate()
        .filter(|(_, chosen)| chosen.opens && chosen.depth > 0)
        .map(|(k, chosen)| (ids[k].as_str(), usize::from(chosen.depth)))
        .collect();
    // TEMP levers (remove before landing): the lines that part a provision from what follows it (`assemble_outline`): one
    // in capitals or opening on a division's word ("parts-caps"), an enactment's history ("parts-history").
    let lever = |name: &str| legal_structure_model::lever(name);
    let (caps, history) = (lever("parts-caps"), lever("parts-history"));
    // (TEMP lever "parts-numbered-division": a division's own heading, its word, a roman numeral and a stop or dash, "Title
    // VI.—Government Deposits.", "TITLE IX.—FEES.", "PART II.—", parts the provision before it from what follows,
    // whatever its look. A citation running on, "Title III in the United States", has no such dash.)
    let numbered_division = lever("parts-numbered-division")
        .then(|| regex::Regex::new(r"^(?:TITLE|Title|TITRE|Titre|PART|Part|PARTIE|Partie|DIVISION|Division)\s+[IVXLC]{1,6}\.?\s?[\u{2014}\u{2013}-]").ok())
        .flatten();
    let mut parting: HashSet<&str> = (0..ids.len())
        .filter(|&k| {
            let at = fences.order[k].index();
            // (Set as a heading: title-shaped, or short and closed by a stop or a bracket, "Title VI.—Government Deposits.",
            // "TITLE IX.—FEES.—(All Companies.)". A wrapped cross-reference, "Part II—Insurance Companies, the Minister
            // may", or a lead-in, "Part II--Loss of Income provisions provide:", is neither.)
            let closed = texts[k].split_whitespace().count() <= 8 && texts[k].trim_end().ends_with(['.', ')']);
            if (evidence.title_shaped[at] || closed) && numbered_division.as_ref().is_some_and(|division| division.is_match(texts[k].trim_start())) {
                // TEMP diagnostic (remove before landing; prints only with LEGALPDF_DEBUG_GAPS set).
                if std::env::var_os("LEGALPDF_DEBUG_GAPS").is_some() {
                    eprintln!("DIVPART {} {:?}", ids[k], texts[k].chars().take(60).collect::<String>());
                }
                return true;
            }
            // (A caption in capitals has four letters or more: a scan's stray "A" or "AS" parts nothing.)
            caps && (evidence.look[at] & LOOK_CAPS != 0 && texts[k].chars().filter(|c| c.is_alphabetic()).count() >= 4
                || evidence.words[at] & WORD_DIVISION != 0)
                || history && evidence.words[at] & WORD_HISTORY != 0
        })
        .map(|k| ids[k].as_str())
        .collect();
    // ("parts-headnotes") A few lines in bold shaped as a title (`Evidence::title_shaped`: no sentence, no clause) set
    // directly over a provision's number are that provision's headnote, not the last words of the provision before: a
    // statute that prints its marginal notes over its sections. (Of such units in the 145 statutes of the gold, the bold
    // ones head the next provision 136 times and are the text before it 14 times; the plain ones 161 and 178; the italic
    // ones never.)
    // ("parts-plain") Set plain, such a unit is the next section's heading where it is words alone (four letters or more,
    // no figure, no mark, no word left hanging) after a sentence's end and directly over a section's own number. (Of such
    // units in the 145 statutes, 141 head the section and 5 are the text before it; after a colon 4 and 1, where what
    // follows is what the sentence brings in; after no sentence's end, 31 and 11.)
    let plain_parts = evidence.instrument && lever("parts-plain");
    // ("parts-stop", TEMP lever) A caption of a few words ending in a full stop ("General Provisions.", "Administration
    // of Justice.", in any look) set after a sentence's end directly over a section's own number is that section's
    // heading. (Of such units in the 145 statutes: 124 the gold's heading or marginal note, 7 a schedule's table or
    // quotation line.)
    let stop_parts = evidence.instrument && lever("parts-stop");
    let stop_figures = lever("stop-no-figures");
    if lever("parts-headnotes") || plain_parts || stop_parts {
        let region = |unit: &Unit| REGIONS[usize::from(unit.region)];
        let context = |unit: &Unit| unit.lines.first().map(|&k| fences.ctx[fences.order[k].index()]);
        let set_aside = |unit: &Unit| matches!(region(unit), "furniture" | "page_label" | "line_label" | "marginal_note" | "note" | "contents");
        for (u, unit) in units.iter().enumerate() {
            if region(unit) != "text" || unit.marker.is_some() || !(1..=3).contains(&unit.lines.len()) {
                continue;
            }
            let shaped = unit.lines.iter().all(|&k| evidence.title_shaped[fences.order[k].index()]);
            let look = evidence.look[fences.order[unit.lines[0]].index()] & (LOOK_BOLD | LOOK_ITALIC);
            // (A stray mark or two between them, a scan's noise, does not part a caption from the section under it.)
            let noise = |next: &Unit| stop_parts && next.lines.iter().map(|&k| texts[k].chars().filter(|c| c.is_alphanumeric()).count()).sum::<usize>() <= 2;
            let next = units[u + 1..].iter().find(|next| context(next) == context(unit) && !set_aside(next) && !noise(next));
            let Some(next) = next.filter(|next| region(next) == "provision" && next.marker.is_some()) else { continue };
            let words = unit.lines.iter().map(|&k| texts[k].trim()).collect::<Vec<_>>().join(" ");
            let before = units[..u].iter().rev().find(|before| context(before) == context(unit) && !set_aside(before));
            let ended = before.and_then(|before| before.lines.last()).is_some_and(|&k| {
                texts[k].trim_end().trim_end_matches(['"', '\u{201d}', ')', ']']).ends_with(['.', '!', '?'])
            });
            // The section's own number: its count's, with no marker of a ladder under it.
            let section = next.lines.first().and_then(|&k| moves[k].member).is_some_and(|(run, at)| {
                let member = &runs.runs[run as usize].members[at as usize];
                member.locator_kind.as_deref() == Some("section") && !member.grammar_value.contains('(')
            });
            let stopped = stop_parts
                && ended
                && section
                && words.trim_end_matches(['\u{2014}', '\u{2013}', '-']).ends_with('.')
                && words.starts_with(char::is_uppercase)
                && words.split_whitespace().count() <= 10
                // (TEMP lever "stop-no-figures": a caption carries no figures; "R. S. O. 1877, c. 161, s. 64." closing a
                // section is its enactment's history, the section's own last line.)
                && !(stop_figures && words.chars().any(|c| c.is_ascii_digit()))
                && words.chars().all(|c| c.is_alphanumeric() || c.is_whitespace() || matches!(c, '\'' | '\u{2019}' | '-' | ',' | '.' | ';' | '&' | '\u{a7}' | '\u{2014}' | '\u{2013}' | '(' | ')'));
            let headnote = stopped
                || shaped && if look == LOOK_BOLD {
                    lever("parts-headnotes")
                } else if look == 0 && plain_parts {
                    let alone = words.chars().filter(|c| c.is_alphabetic()).count() >= 4
                        && words.chars().all(|c| c.is_alphabetic() || c.is_whitespace() || matches!(c, '\'' | '\u{2019}' | '-'))
                        && !words.rsplit(char::is_whitespace).next().is_some_and(|last| {
                            matches!(last.to_lowercase().as_str(), "and" | "or" | "by" | "the" | "of" | "et" | "ou" | "de" | "la" | "le")
                        });
                    alone && ended && section
                } else {
                    false
                };
            if headnote {
                parting.insert(ids[unit.lines[0]].as_str());
            }
        }
    }
    // TEMP diagnostic (remove before landing; prints only for the line ids in LEGALPDF_DEBUG_LINES): what the parting
    // rules read of each watched line and whether it parts.
    if let Some(watched) = watched_lines.as_deref() {
        for k in (0..ids.len()).filter(|&k| watched.split(',').any(|w| w == ids[k])) {
            let at = fences.order[k].index();
            eprintln!("PARTING {} caps {} letters {} division {} history {} parts {} levers caps={caps} history={history}",
                ids[k], evidence.look[at] & LOOK_CAPS != 0, texts[k].chars().filter(|c| c.is_alphabetic()).count(),
                evidence.words[at] & WORD_DIVISION != 0, evidence.words[at] & WORD_HISTORY != 0, parting.contains(ids[k].as_str()));
        }
    }
    crate::graph::assemble_outline(
        &mut structure.nodes,
        &contexts,
        &|id| levels.get(id).copied(),
        &|id| depths.get(id).copied(),
        &|id| parting.contains(id),
        &|id| own_depths.get(id).copied(),
    );
    // A provision's path from the outermost provision it stands in ("12(1)(a)"), by the parents the outline kept, after
    // its printing's alias where it has one: each unit's printed label along the chain.
    let label_of: HashMap<&str, &str> = units
        .iter()
        .enumerate()
        .filter_map(|(u, unit)| Some((unit_ids[u].as_str(), unit.label.as_deref()?)))
        .collect();
    let provisions: HashSet<&str> = units
        .iter()
        .enumerate()
        .filter(|(_, unit)| REGIONS[usize::from(unit.region)] == "provision")
        .map(|(u, _)| unit_ids[u].as_str())
        .collect();
    let parent_of: HashMap<String, String> = structure
        .nodes
        .iter()
        .filter_map(|node| Some((node.id.clone(), node.parent_id.clone()?)))
        .collect();
    for node in structure
        .nodes
        .iter_mut()
        .filter(|node| node.kind == NodeKind::Section && label_of.contains_key(node.id.as_str()))
    {
        let mut chain = vec![node.id.as_str()];
        while let Some(parent) = parent_of
            .get(*chain.last().unwrap())
            .map(String::as_str)
            .filter(|parent| provisions.contains(parent) && !chain.contains(parent))
        {
            chain.push(parent);
        }
        let labels: Vec<&str> = chain
            .iter()
            .rev()
            .filter_map(|id| label_of.get(id).copied())
            .collect();
        if labels.len() > 1 {
            let path = labels
                .iter()
                .enumerate()
                .map(|(at, label)| {
                    if at == 0 || label.starts_with('(') {
                        (*label).to_owned()
                    } else {
                        format!("({label})")
                    }
                })
                .collect::<String>();
            node.aliases.get_or_insert_with(Vec::new).push(path);
        }
    }
    legal_structure_model::assemble_in_place(&mut structure).map_err(Error::Message)?;
    // A judgment's numbered paragraphs are t0's (`read_judgment_paragraphs`): each of its numbers the unit carrying it,
    // with the extent the reading holds; a number no unit carries a node of its own; a number off its count none. The
    // reading is the fences', all `Decided` reads for it. On a line a model reads as a provision the number is the
    // provision's: a contract's clauses count as paragraphs do, and no paragraph stands over them. Tier 0's rules read
    // the same numbers, and nothing here overrules them.
    structure.source_fragments = index.lines.clone();
    for (line, written) in edits {
        let held = structure.manuscript_edits.entry(line).or_default();
        for edit in written {
            if !held
                .iter()
                .any(|other| other.0 < edit.1 && edit.0 < other.1 || *other == edit)
            {
                held.push(edit);
            }
        }
        held.sort_by_key(|edit| (edit.0, edit.1));
    }
    structure.joins.extend(joins);
    structure.continuations = Some(continuations);
    if !reader {
        let apart: HashSet<&str> = moves
            .iter()
            .enumerate()
            .filter(|(_, m)| matches!(m.region, FURNITURE | PAGE_LABEL | LINE_LABEL | PROVISION))
            .map(|(k, _)| ids[k].as_str())
            .collect();
        crate::graph::read_judgment_paragraphs(&mut structure, pages, decided, &apart);
    }
    structure.profile = identity.profile;
    Ok(structure)
}
