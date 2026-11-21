//! The lines the document's own outline names, and every heading's level. The document's bookmarks
//! (where they outline its divisions) and its printed contents list, merged into one outline, are
//! found as lines of the pages they name: each found entry's lines are a heading at the entry's
//! level. Once the headings stand, each takes its level in this order: the level its anchor or a
//! transcript's rule gave it, then its depth on the numbering ladder of its context's headings, set
//! on the scale its anchored headings show, then its look's rank among the context's headings.

use super::residual::{apart, look_of, marker_alone, Families, Look};
use super::body_like;
use crate::model::decide::Decision;
use crate::model::decide::OutlineAnchor;
use crate::model::decide::Sequences;
use crate::pdf_rules::evidence::enumerated_in;
use crate::model::fences::Fences;
use legal_pdf_rules::headings::bookmarks::{
    classify, generic_title, normalize_title, restack, same_title, validate, Entry, Tier,
    BACKFILL_MAX_TITLE, MAX_TITLE_LINES,
};
use crate::pdf_rules::headings::rules::{ladder_levels, ATTACHMENTS};
use crate::model::Geometry;
use crate::model::{Evidence, MarkerReading, Role, Rule, Slot, Slots, SourceTag};
use legal_pdf_core::model::{Page, PdfOutlineEntry};
use pdf_inspector::markdown::has_dot_leaders;

/// The lines the document's outline names (`OutlineAnchor`), and where each frame entry of it opens,
/// as (document, parallel translation, place in the reading order).
pub struct Anchoring {
    pub anchors: Vec<OutlineAnchor>,
    pub opened: Vec<(u16, bool, u32)>,
}

/// The document's bookmarks `outline` (where they outline its divisions) and its contents lists'
/// entries confirmed on their pages, merged into one outline (`merge_outlines`), found as lines of
/// the pages they name (`validate`, `classify`, the title matching of PageIndex's bookmark frame).
/// A title is found on up to four lines of its page, or the pages either side of it, that no rule
/// gave another role than a heading's, outside a quotation and a contents list, in one document and
/// language; found whole before found as a fragment, and each document's titles in order. A title
/// found on lines no source or rule read as a heading stands only where the bookmarks are titles
/// (not passages), a skeleton's chapters hold it, and its lines neither go on with the line before
/// nor carry their sentence on. One found within a heading takes the whole heading: its lines one
/// under the other with no blank of most of a line between them (`apart`). Each found entry's level
/// is its depth in its document's frame (every bookmark of a full set, a skeleton's chapters), else
/// one under its nearest levelled ancestor.
pub fn anchors(
    pages: &[Page],
    slots: &Slots,
    evidence: &Evidence,
    fences: &Fences,
    geometry: &Geometry,
    role: &[Role],
    outline: &[PdfOutlineEntry],
    entries: &[crate::model::contents_lists::Entry],
) -> Anchoring {
    let listed = crate::contents::contents_outline_among(pages, entries, |page, at| {
        !matches!(
            role[slots.at(page, at).index()],
            Role::LineLabel | Role::Furniture | Role::PageLabel | Role::Contents
        ) && !fences.contents_pages[page]
    });
    let merged = legal_pdf_rules::headings::bookmarks::merge_outlines(
        if evidence.outline_usable {
            outline
        } else {
            &[]
        },
        &listed,
    );
    let mut entries = validate(&merged, pages.len());
    let tier = classify(&entries, pages.len());
    if tier == Tier::Ignore {
        return Anchoring {
            anchors: Vec::new(),
            opened: Vec::new(),
        };
    }
    let mut page_slot: Vec<Option<usize>> = Vec::new();
    for (slot, page) in pages.iter().enumerate() {
        if page.index >= page_slot.len() {
            page_slot.resize(page.index + 1, None);
        }
        page_slot[page.index] = Some(slot);
    }
    let slot_of_page = |page: usize| page_slot.get(page).copied().flatten();
    let contents: Vec<bool> = (0..pages.len())
        .map(|page| {
            fences.contents_pages[page]
                || slots
                    .on(page)
                    .any(|slot| role[slot as usize] == Role::Contents)
        })
        .collect();
    // A bookmark on a contents page holds no other: its subtree rises a level.
    let mut holder: Option<usize> = None;
    for entry in &mut entries {
        if holder.is_some_and(|level| entry.level <= level) {
            holder = None;
        }
        if holder.is_some() {
            entry.level -= 1;
        } else if slot_of_page(entry.page - 1).is_some_and(|page| contents[page]) {
            holder = Some(entry.level);
        }
    }
    restack(&mut entries);
    let promotes = 2 * entries
        .iter()
        .filter(|entry| entry.title.chars().count() > BACKFILL_MAX_TITLE)
        .count()
        < entries.len();
    let text = |slot: Slot| slots.line(pages, slot).text.trim();
    // Two lines read one under the other in one block.
    let together = |upper: Slot, lower: Slot| !apart(pages, slots, geometry, upper, lower);
    let eligible = |slot: Slot| {
        matches!(role[slot.index()], Role::Open | Role::Heading)
            && fences.at(slot.index()).quotation == 0
            && !has_dot_leaders(&slots.line(pages, slot).text)
    };
    let heading = |slot: Slot| {
        role[slot.index()] == Role::Heading
            || matches!(
                evidence.tag[slot.index()],
                SourceTag::Heading(_) | SourceTag::Title | SourceTag::ParagraphTitle
            )
    };
    let context = |slot: Slot| {
        (
            fences.at(slot.index()).document,
            fences.at(slot.index()).translation,
        )
    };
    // A page's lines in reading order.
    let order = |page: usize| {
        let range = slots.on(page);
        &fences.order[range.start as usize..range.end as usize]
    };
    let mut norms: Vec<Option<Vec<String>>> = vec![None; pages.len()];
    let mut claimed = vec![false; slots.len()];
    let mut last: Vec<((u16, bool), u32)> = Vec::new();
    let mut found: Vec<Option<(Vec<Slot>, bool)>> = vec![None; entries.len()];
    for (index, entry) in entries.iter().enumerate() {
        let title = normalize_title(&entry.title);
        // A title of no words ("-----", "2.1") names no line.
        let Some(target) = slot_of_page(entry.page - 1)
            .filter(|_| title.chars().count() >= 2 && title.chars().any(char::is_alphabetic))
        else {
            continue;
        };
        let window = [Some(target), target.checked_add(1), target.checked_sub(1)];
        for page in window
            .into_iter()
            .flatten()
            .filter(|&page| page < pages.len())
        {
            if norms[page].is_none() {
                norms[page] = Some(
                    order(page)
                        .iter()
                        .map(|&slot| normalize_title(&slots.line(pages, slot).text))
                        .collect(),
                );
            }
        }
        // A title read whole wins over one read as a fragment of a longer heading.
        let hit = [true, false].into_iter().find_map(|whole| {
            window
                .into_iter()
                .flatten()
                .enumerate()
                .find_map(|(at, page)| {
                    if page >= pages.len() || at > 0 && contents[page] {
                        return None;
                    }
                    let (lines, norms) = (order(page), norms[page].as_ref()?);
                    (0..lines.len())
                        .filter(|&start| !norms[start].is_empty())
                        .find_map(|start| {
                            let mut joined = String::new();
                            let mut existing = true;
                            for end in start..(start + MAX_TITLE_LINES).min(lines.len()) {
                                let slot = lines[end];
                                if claimed[slot.index()]
                                    || !eligible(slot)
                                    || context(slot) != context(lines[start])
                                {
                                    return None;
                                }
                                joined.push_str(&norms[end]);
                                // One heading, not several read together.
                                existing &= heading(slot)
                                    && (end == start || together(lines[end - 1], slot));
                                let matched = if whole {
                                    joined == title
                                } else {
                                    same_title(&joined, &title, existing)
                                };
                                if matched {
                                    return Some((page, start, end, existing));
                                }
                            }
                            None
                        })
                })
        });
        let Some((page, start, end, existing)) = hit else {
            continue;
        };
        let lines = &order(page)[start..=end];
        let first = lines[0];
        let (position, document) = (fences.place[first.index()], context(first));
        if last
            .iter()
            .any(|&(seen, at)| seen == document && position <= at)
        {
            continue;
        }
        if !existing {
            let joined = lines
                .iter()
                .map(|&slot| text(slot))
                .collect::<Vec<_>>()
                .join(" ");
            let carried_on = order(page).get(end + 1).is_some_and(|&next| {
                body_like(role[next.index()])
                    && together(lines[lines.len() - 1], next)
                    && text(next).starts_with(char::is_lowercase)
            });
            // A skeleton frame takes deeper bookmarks only within its chapters.
            let chaptered = tier == Tier::Full
                || entries
                    .iter()
                    .any(|chapter| chapter.level == 1 && chapter.page <= entry.page);
            if !promotes
                || !chaptered
                || generic_title().is_match(&entry.title)
                || joined.ends_with(',')
                || joined.starts_with(char::is_lowercase)
                || fences.continues[first.index()].is_some()
                || carried_on
            {
                continue;
            }
        }
        match last.iter_mut().find(|(seen, _)| *seen == document) {
            Some((_, at)) => *at = position,
            None => last.push((document, position)),
        }
        for &slot in lines {
            claimed[slot.index()] = true;
        }
        found[index] = Some((lines.to_vec(), existing));
    }
    // A title found within a heading takes the whole heading; within one it cuts only where a
    // numbered heading starts.
    for (lines, existing) in found.iter_mut().flatten() {
        if !*existing {
            continue;
        }
        let first = lines[0];
        let page = slots.page(first);
        let order = order(page);
        let base = slots.on(page).start;
        let same = |slot: Slot| heading(slot) && eligible(slot) && context(slot) == context(first);
        let numbered = |slot: Slot| enumerated_in(&evidence.readings[slot.index()]);
        let mut start = (fences.place[first.index()] - base) as usize;
        let mut end = (fences.place[lines[lines.len() - 1].index()] - base) as usize;
        if !numbered(order[start]) {
            while start > 0 && same(order[start - 1]) && together(order[start - 1], order[start]) {
                start -= 1;
            }
        }
        while end + 1 < order.len()
            && same(order[end + 1])
            && together(order[end], order[end + 1])
            && !numbered(order[end + 1])
        {
            end += 1;
        }
        *lines = order[start..=end.max(start)].to_vec();
    }
    let confirmed: Vec<bool> = found
        .iter()
        .zip(&entries)
        .map(|(hit, entry)| {
            hit.as_ref().is_some_and(|(lines, _)| {
                let found: String = lines
                    .iter()
                    .map(|&slot| normalize_title(&slots.line(pages, slot).text))
                    .collect();
                same_title(&found, &normalize_title(&entry.title), false)
            })
        })
        .collect();
    // A marker alone on the line over a found title is its heading's first line ("II." over
    // "Background Facts").
    for (lines, _) in found.iter_mut().flatten() {
        let first = lines[0];
        let page = slots.page(first);
        let at = (fences.place[first.index()] - slots.on(page).start) as usize;
        if let Some(&before) = at.checked_sub(1).map(|before| &order(page)[before]) {
            if role[before.index()] == Role::Open
                && !claimed[before.index()]
                && context(before) == context(first)
                && together(before, first)
                && marker_alone(evidence, before, text(before))
            {
                claimed[before.index()] = true;
                lines.insert(0, before);
            }
        }
    }
    let contexts: Vec<(u16, bool)> = found
        .iter()
        .map(|hit| {
            hit.as_ref()
                .map_or((0, false), |(lines, _)| context(lines[0]))
        })
        .collect();
    // The frame: every bookmark of a full set, a skeleton's chapters.
    let framed = |entry: &Entry| tier == Tier::Full || entry.level == 1;
    let mut levels: Vec<Option<usize>> = vec![None; entries.len()];
    let mut stacks: Vec<((u16, bool), Vec<usize>)> = Vec::new();
    let mut parents: Vec<Vec<usize>> = Vec::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        let stack = match stacks.iter().position(|(seen, _)| *seen == contexts[index]) {
            Some(at) => &mut stacks[at].1,
            None => {
                stacks.push((contexts[index], Vec::new()));
                &mut stacks.last_mut().expect("a stack").1
            }
        };
        while stack
            .last()
            .is_some_and(|&prior| entries[prior].level >= entry.level)
        {
            stack.pop();
        }
        parents.push(stack.clone());
        stack.push(index);
    }
    let mut frames: Vec<((u16, bool), Vec<usize>)> = Vec::new();
    for (index, entry) in entries
        .iter()
        .enumerate()
        .filter(|(index, entry)| found[*index].is_some() && framed(entry))
    {
        let frame = match frames.iter().position(|(seen, _)| *seen == contexts[index]) {
            Some(at) => &mut frames[at].1,
            None => {
                frames.push((contexts[index], Vec::new()));
                &mut frames.last_mut().expect("a frame").1
            }
        };
        while frame.last().is_some_and(|&level| level >= entry.level) {
            frame.pop();
        }
        frame.push(entry.level);
        levels[index] = Some(frame.len());
    }
    for index in 0..entries.len() {
        if found[index].is_some() && levels[index].is_none() {
            levels[index] = Some(
                parents[index]
                    .iter()
                    .rev()
                    .find_map(|&parent| levels[parent])
                    .map_or(1, |level| level + 1),
            );
        }
    }
    // Where each frame entry opens: its lines, else the top of its page.
    let mut opened: Vec<(u16, bool, u32)> = Vec::new();
    for (index, entry) in entries
        .iter()
        .enumerate()
        .filter(|(_, entry)| framed(entry))
    {
        let (document, place) = match &found[index] {
            Some((lines, _)) => (context(lines[0]), fences.place[lines[0].index()]),
            None => match slot_of_page(entry.page - 1) {
                Some(page) => ((0, false), slots.on(page).start),
                None => continue,
            },
        };
        match opened
            .iter_mut()
            .find(|(seen, translation, _)| (*seen, *translation) == document)
        {
            Some((_, _, at)) => *at = (*at).min(place),
            None => opened.push((document.0, document.1, place)),
        }
    }
    let mut anchors = Vec::new();
    for (index, hit) in found.into_iter().enumerate() {
        let Some((lines, _)) = hit else { continue };
        anchors.push(OutlineAnchor {
            lines,
            level: levels[index].unwrap_or(1).min(usize::from(u16::MAX)) as u16,
            confirmed: confirmed[index],
        });
    }
    Anchoring { anchors, opened }
}

/// Every heading's level, in each reading context, its headings in reading order (consecutive
/// heading lines going on with the one above them are one heading). A heading reads its level from
/// the level its anchor or a transcript's rule gave it, else from its look's rank among the
/// context's headings (larger type first, then capitals, bold, centred, underlined, then the one
/// met first). An unnumbered heading at the top of those readings starts the numbering ladder over
/// (a judgment's reasons after its headnote). An anchored heading found whole keeps its anchor's
/// level; any other a tagged PDF tags H1 to H6 takes its tag's level. A numbered heading takes its
/// depth on its run of the ladder (`ladder_levels`), set on the scale the run's anchored headings
/// show (the levels the document's own outline gave them, by the offset most of them agree on), or
/// else under the unnumbered heading that opens the run ("I. Introduction" under "Reasons for
/// Judgment"). An unnumbered heading takes its reading's step from the heading before it.
pub fn heading_levels(
    pages: &[Page],
    slots: &Slots,
    evidence: &Evidence,
    fences: &Fences,
    sequences: &Sequences,
    geometry: &Geometry,
    readings: &[Vec<MarkerReading>],
    decision: &mut Decision,
) {
    let mut families = Families::default();
    let text = |slot: Slot| slots.line(pages, slot).text.trim();
    for lines in &sequences.lines {
        let mut units: Vec<Vec<Slot>> = Vec::new();
        let mut previous: Option<Slot> = None;
        for &slot in lines {
            if decision.line(slot.index()).role != Role::Heading {
                previous = None;
                continue;
            }
            let joins = previous.is_some_and(|prior| slots.page(prior) == slots.page(slot))
                && decision.line(slot.index()).joined;
            match units.last_mut() {
                Some(current) if joins => current.push(slot),
                _ => units.push(vec![slot]),
            }
            previous = Some(slot);
        }
        if units.is_empty() {
            continue;
        }
        let texts: Vec<String> = units
            .iter()
            .map(|unit| {
                unit.iter()
                    .map(|&slot| text(slot))
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect();
        // A numbered title is set as its title's line is, whatever its number's type.
        let looks: Vec<Look> = units
            .iter()
            .zip(&texts)
            .map(|(unit, joined)| {
                let first = if unit.len() > 1 && marker_alone(evidence, unit[0], text(unit[0])) {
                    unit[1]
                } else {
                    unit[0]
                };
                look_of(
                    pages,
                    slots,
                    evidence,
                    fences,
                    geometry,
                    first,
                    joined,
                    families.of(readings, fences, unit[0]),
                )
                .0
            })
            .collect();
        // The level an anchor or a transcript's part gives its heading.
        let rules: std::collections::HashMap<Slot, Rule> = units
            .iter()
            .map(|unit| (unit[0], decision.line(unit[0].index()).rule))
            .collect();
        let given = |unit: &[Slot]| match rules[&unit[0]] {
            Rule::Anchor | Rule::TranscriptPart => usize::from(decision.given[unit[0].index()]),
            _ => 0,
        };
        let anchored = |unit: &[Slot]| {
            (rules[&unit[0]] == Rule::Anchor)
                .then(|| decision.anchor[unit[0].index()])
                .filter(|&anchor| anchor > 0)
        };
        let mut ranked: Vec<(Look, usize)> = Vec::new();
        for (k, look) in looks.iter().enumerate() {
            if given(&units[k]) == 0 && !ranked.iter().any(|(seen, _)| seen == look) {
                ranked.push((*look, k));
            }
        }
        ranked.sort_by_key(|(look, first)| (look.rank(), *first));
        let reads: Vec<usize> = units
            .iter()
            .zip(&looks)
            .map(|(unit, look)| match given(unit) {
                0 => ranked
                    .iter()
                    .position(|(seen, _)| seen == look)
                    .map_or(1, |rank| rank + 1),
                level => level,
            })
            .collect();
        let placed = |unit: &[Slot]| {
            anchored(unit).is_some_and(|anchor| decision.anchors[anchor as usize - 1].confirmed)
        };
        // The level a tagged PDF's H1 to H6 gives a heading no anchor names.
        let tagged = |unit: &[Slot]| match evidence.tag[unit[0].index()] {
            SourceTag::Heading(level) if level > 0 && anchored(unit).is_none() => {
                Some(usize::from(level))
            }
            _ => None,
        };
        let numbers: Vec<Vec<MarkerReading>> = texts
            .iter()
            .map(|text| crate::pdf_rules::evidence::heading_numbers(text))
            .collect();
        let breaks: Vec<bool> = numbers
            .iter()
            .zip(&reads)
            .map(|(number, &read)| number.is_empty() && read <= 1)
            .collect();
        let ladder = ladder_levels(&numbers, &breaks);
        // Each run's offset from its depths to the document's levels, by the anchored headings in it.
        let mut votes: Vec<(usize, isize, usize)> = Vec::new();
        let mut run = 0;
        for (k, unit) in units.iter().enumerate() {
            if let Some((depth, opens)) = ladder[k] {
                run += usize::from(opens);
                if placed(unit) {
                    let offset = reads[k] as isize - depth as isize;
                    match votes
                        .iter_mut()
                        .find(|(seen, value, _)| *seen == run && *value == offset)
                    {
                        Some((_, _, count)) => *count += 1,
                        None => votes.push((run, offset, 1)),
                    }
                }
            }
        }
        let offset_of = |run: usize| {
            votes
                .iter()
                .filter(|(seen, _, _)| *seen == run)
                .max_by_key(|&&(_, offset, count)| (count, std::cmp::Reverse(offset.abs())))
                .map(|&(_, offset, _)| offset.max(0) as usize)
        };
        // The level the current run of the ladder hangs from, where nothing in it was anchored.
        let (mut base, mut run) = (0, 0);
        let (mut prior_level, mut prior_read, mut prior_numbered): (
            Option<usize>,
            Option<usize>,
            bool,
        ) = (None, None, false);
        for (k, unit) in units.iter().enumerate() {
            let (rung, read) = (ladder[k], reads[k]);
            if let Some((_, true)) = rung {
                run += 1;
                // An attachment stands at the top whatever heads the part before it.
                let attachment = numbers[k].iter().any(|reading| {
                    ATTACHMENTS
                        .iter()
                        .any(|word| reading.family.starts_with(&format!("{word}:")))
                });
                base = if prior_numbered || attachment {
                    0
                } else {
                    prior_level.unwrap_or(0)
                };
            }
            let level = if placed(unit) {
                read
            } else if let Some(level) = tagged(unit) {
                level
            } else if let Some((depth, _)) = rung {
                offset_of(run).unwrap_or(base) + depth
            } else {
                match (prior_level, prior_read) {
                    (Some(level), Some(before)) => (level + read).saturating_sub(before).max(1),
                    _ => read,
                }
            };
            for &slot in unit {
                decision.moves[decision.place[slot.index()] as usize].level =
                    level.min(usize::from(u8::MAX)) as u8;
            }
            (prior_level, prior_read, prior_numbered) = (Some(level), Some(read), rung.is_some());
        }
    }
}
