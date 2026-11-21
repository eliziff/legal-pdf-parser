//! The outline the graph's units hang in: parents follow heading levels, the depth the walk opened each body unit at
//! and reading order within each constituent document. Source membership, quotation containers and
//! locators stay unchanged.

use legal_structure_model::{NodeKind, PrimaryRole, StructureNode};
use std::collections::HashMap;

/// A unit's document, quotation, and note, table cell or parallel text membership.
pub type OutlineContext = (String, Option<String>, Option<String>);

#[derive(Clone, Copy)]
enum Region {
    Heading,
    Text,
    UnderHeading,
    ListItem,
    Quote,
}

fn outline_region(node: &StructureNode, container: Option<NodeKind>) -> Option<Region> {
    let outside = matches!(container, Some(NodeKind::Footnote | NodeKind::Endnote))
        || matches!(
            node.role,
            Some(
                PrimaryRole::Furniture
                    | PrimaryRole::PageLabel
                    | PrimaryRole::LineLabel
                    | PrimaryRole::MarginalNote
            )
        )
        || node.grammar.as_deref() == Some("contents");
    let ranked = matches!(
        node.kind,
        NodeKind::Heading
            | NodeKind::ListItem
            | NodeKind::Paragraph
            | NodeKind::Prose
            | NodeKind::Quote
            | NodeKind::Table
    ) || (node.kind == NodeKind::Section && node.marker_range.is_some());
    if outside || !ranked {
        return None;
    }
    let front = matches!(
        node.role,
        Some(
            PrimaryRole::Caption
                | PrimaryRole::Title
                | PrimaryRole::Headnote
                | PrimaryRole::Judges
                | PrimaryRole::Counsel
                | PrimaryRole::Date
                | PrimaryRole::DocketNumber
                | PrimaryRole::Court
        )
    );
    Some(
        if container == Some(NodeKind::Cell) || node.kind == NodeKind::Table {
            Region::UnderHeading
        } else if node.role == Some(PrimaryRole::Heading) || node.kind == NodeKind::Heading {
            Region::Heading
        } else if node.kind == NodeKind::Quote {
            Region::Quote
        } else if front {
            Region::UnderHeading
        } else if node.kind == NodeKind::ListItem {
            Region::ListItem
        } else {
            Region::Text
        },
    )
}

#[derive(Default)]
struct Outline {
    headings: Vec<(usize, usize)>, // (level, node)
    /// The body units open, each at the depth the walk gave it, whether a heading has stood since, and whether it is
    /// numbered: (depth, node, parted, numbered).
    open: Vec<(usize, usize, bool, bool)>,
    previous: Option<usize>,
    /// The last unit of the text's own words since its heading: the one a quotation after it is set within.
    quoting: Option<usize>,
    /// The context's ranked units in reading order, each with whether it is a heading (TEMP lever "headnotes-held").
    seen: Vec<(usize, bool)>,
}

/// Sets ranked units' parents from the headings' levels (`level`, by a heading's line id) and the depth the walk
/// opened each body unit at (`depth`, by its first line's id), each in its context (`contexts`). Constituent
/// documents and a parallel translation scope the stacks (a column's list or heading never opens under the
/// other language's paragraph); quotation and object membership stay intact.
pub fn assemble_outline(
    nodes: &mut [StructureNode],
    contexts: &[OutlineContext],
    level: &dyn Fn(&str) -> Option<usize>,
    depth: &dyn Fn(&str) -> Option<usize>,
    parts: &dyn Fn(&str) -> bool,
    own: &dyn Fn(&str) -> Option<usize>,
) {
    let containers = nodes
        .iter()
        .map(|node| (node.id.as_str(), node.kind))
        .collect::<HashMap<_, _>>();
    let mut units = nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| {
            let container = contexts[index]
                .2
                .as_deref()
                .and_then(|id| id.strip_prefix("object:"))
                .and_then(|id| containers.get(id).copied());
            outline_region(node, container).map(|region| (index, region))
        })
        .collect::<Vec<_>>();
    // A quotation container opens before units with the same source start.
    units.sort_by_key(|&(index, _)| {
        (
            nodes[index].range.start,
            nodes[index].kind != NodeKind::Quote,
            nodes[index].range.end,
            nodes[index].id.clone(),
        )
    });
    // TEMP lever (remove before landing): a provision holds the unnumbered units after it only where LEGALPDF_LEVERS
    // names it, so one build reads both ways; unset, they hang as they did.
    let provision_holds =
        legal_structure_model::lever("provision-holds");
    // TEMP lever "headnotes-held" (remove before landing). A heading set between two of a provision's own units (a
    // subsection's headnote: "Idem" over "(2)") stands inside that provision. The outline is read once to find such
    // headings (`held`: each with the provision both units stand in), then again with each hanging under its provision
    // as its text does: it heads nothing, parts nothing, and the provision's extent takes it. A heading in capitals or
    // opening on a dash, a roman numeral alone, a line ending in a colon part what they stand between, and nothing is
    // held across one.
    let items_held = legal_structure_model::lever("items-held");
    let headnotes_held =
        legal_structure_model::lever("headnotes-held");
    let mut held: HashMap<usize, usize> = HashMap::new();
    loop {
    let mut parents = vec![None; nodes.len()];
    let mut documents = HashMap::<(&str, bool), Outline>::new();
    for &(index, region) in &units {
        let translated = contexts[index].2.as_deref() == Some("translation");
        let outline = documents
            .entry((&contexts[index].0, translated))
            .or_default();
        if let Some(&holder) = held.get(&index) {
            parents[index] = Some(holder);
            nodes[index].parent_id = Some(nodes[holder].id.clone());
            outline.previous = Some(index);
            continue;
        }
        let heading = outline.headings.last().map(|&(_, node)| node);
        // A passage the text quotes (its paragraphs, their numbers the quoted text's) stands within the unit quoting it.
        if contexts[index].1.is_some() && matches!(region, Region::Text | Region::ListItem) {
            let parent = outline.quoting.or(heading);
            parents[index] = parent;
            nodes[index].parent_id = parent.map(|parent| nodes[parent].id.clone());
            continue;
        }
        // A numbered body unit hangs where the walk opened it (`depth`): under the unit open at the depth above its own.
        // The walk's numbering runs on across a heading (a statute's running head, a section's own heading between its
        // subsections), so it finds its parent before one. An unnumbered unit stands beside the unit before it, and
        // under nothing a heading parts it from.
        // (`own`: a node's own depth, by its id, where its line opens more than one rung of a ladder: "39 (1) ...", the
        // section at its depth and its first subsection under it.)
        let mut stated = own(&nodes[index].id).or_else(|| nodes[index].line_ids.first().and_then(|id| depth(id)));
        // (TEMP lever "items-held") A numbered list item while a section is open, no heading between, is that section's own
        // paragraph ("34." / "1. Le quorum ..." / "2. Les décisions ..."): it stands one under the section, whatever depth the
        // walk gave it.
        if items_held && matches!(region, Region::ListItem) && nodes[index].marker_range.is_some() {
            if let Some(&(depth, ..)) = outline
                .open
                .iter()
                .rev()
                .find(|&&(_, open, parted, numbered)| numbered && !parted && nodes[open].kind == NodeKind::Section)
            {
                stated = Some(stated.map_or(depth + 1, |own| own.max(depth + 1)));
            }
        }
        // TEMP lever: a unit the page sets apart from the provision before it (`parts`, by its first line: a caption in
        // capitals no rule read as a heading, an enactment's history) parts that provision from what follows, as a
        // heading does.
        if provision_holds
            && nodes[index].marker_range.is_none()
            && matches!(region, Region::Text)
            && nodes[index].line_ids.first().is_some_and(|id| parts(id))
        {
            outline.open.iter_mut().for_each(|open| open.2 = true);
        }
        // An unnumbered unit while a provision is open (TEMP lever) is the provision's where no heading parts it from it
        // (its words run on in another column, its closing words): a provision holds what stands before the next one.
        // Parted, it stands under the heading. Either way it leaves the provisions open: the numbering runs on past it,
        // and a subsection after it finds its section.
        let provision = outline
            .open
            .iter()
            .rev()
            .find(|&&(_, open, ..)| nodes[open].kind == NodeKind::Section)
            .filter(|_| provision_holds && nodes[index].marker_range.is_none())
            .map(|&(_, open, parted, _)| (!parted).then_some(open).or(heading));
        let parent = match region {
            Region::Text | Region::ListItem if provision.is_some() => provision.flatten(),
            // (A numbered unit the walk gave no depth opens at the top of the numbering: it closes the units open before
            // it, so what the walk sets under it hangs under it and not under the unit before.)
            Region::Text | Region::ListItem
                if stated.is_some() || nodes[index].marker_range.is_some() =>
            {
                let numbered = nodes[index].marker_range.is_some();
                let beside = outline
                    .open
                    .last()
                    .filter(|&&(_, _, parted, _)| !parted)
                    .map_or(1, |&(depth, ..)| depth);
                let depth = if numbered {
                    stated.unwrap_or(1)
                } else {
                    beside
                };
                while outline.open.last().is_some_and(|&(open, ..)| open >= depth) {
                    outline.open.pop();
                }
                // A numbered unit hangs under a numbered one only: an unnumbered unit (a section's prose run on, an answer)
                // stands within the unit before it and never opens a level the numbering continues under.
                let parent = if numbered {
                    outline.open.iter().rev().find(|&&(.., numbered)| numbered)
                } else {
                    outline.open.last().filter(|&&(_, _, parted, _)| !parted)
                }
                .map(|&(_, node, ..)| node)
                .or(heading);
                outline.open.push((depth, index, false, numbered));
                parent
            }
            Region::Heading => {
                let level = nodes[index]
                    .line_ids
                    .iter()
                    .find_map(|id| level(id))
                    .unwrap_or(1);
                while outline
                    .headings
                    .last()
                    .is_some_and(|&(open, _)| open >= level)
                {
                    outline.headings.pop();
                }
                let parent = outline.headings.last().map(|&(_, node)| node);
                outline.headings.push((level, index));
                outline.open.iter_mut().for_each(|open| open.2 = true);
                parent
            }
            Region::Text | Region::ListItem | Region::UnderHeading => heading,
            Region::Quote => outline.previous.and_then(|previous| {
                if nodes[previous].kind == NodeKind::Quote {
                    parents[previous]
                } else {
                    Some(previous)
                }
            }),
        };
        parents[index] = parent;
        nodes[index].parent_id = parent.map(|parent| nodes[parent].id.clone());
        outline.seen.push((index, matches!(region, Region::Heading)));
        outline.previous = Some(index);
        outline.quoting = match region {
            Region::Heading => None,
            Region::Text | Region::ListItem => Some(index),
            Region::UnderHeading | Region::Quote => outline.quoting,
        };
    }
    if !headnotes_held || !held.is_empty() {
        break;
    }
    let numbered = |nodes: &[StructureNode], index: usize| {
        nodes[index].kind == NodeKind::Section && nodes[index].marker_range.is_some() && nodes[index].role != Some(PrimaryRole::Heading)
    };
    let headnote = |nodes: &[StructureNode], index: usize| {
        let words = nodes[index].label.as_deref().unwrap_or("").trim();
        nodes[index].kind == NodeKind::Heading
            && words.chars().filter(|c| c.is_alphabetic()).count() >= 3
            && words.chars().any(char::is_lowercase)
            && !words.starts_with(['\u{2014}', '\u{2013}', '-'])
            && !words.ends_with(':')
            && legal_structure_model::layout::roman_value(&words.to_uppercase()).is_none()
    };
    for outline in documents.values() {
        // A unit and the numbered provisions over it, innermost first.
        let chain = |mut index: usize| {
            let mut over = Vec::new();
            loop {
                if numbered(nodes, index) {
                    over.push(index);
                }
                match parents[index] {
                    Some(parent) => index = parent,
                    None => break over,
                }
            }
        };
        let (mut last, mut pending) = (None::<usize>, Vec::<usize>::new());
        for &(index, heading) in &outline.seen {
            if contexts[index].1.is_some() {
                continue;
            }
            if heading && !numbered(nodes, index) {
                if headnote(nodes, index) {
                    pending.push(index);
                } else {
                    (last, pending) = (None, Vec::new());
                }
            } else if numbered(nodes, index) {
                if let Some(before) = last.filter(|_| !pending.is_empty()) {
                    let over = chain(before);
                    let holder = parents[index].map(&chain).unwrap_or_default().into_iter().find(|provision| over.contains(provision));
                    if let Some(holder) = holder {
                        held.extend(pending.iter().map(|&heading| (heading, holder)));
                    }
                }
                pending.clear();
                last = Some(index);
            }
        }
    }
    if held.is_empty() {
        break;
    }
    }
}