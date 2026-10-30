//! The outline the graph's units hang in: parents follow heading levels, open numbering families and
//! reading order within each constituent document. Source membership, quotation containers and
//! locators stay unchanged.

use super::*;
use legal_structure_model::{NumberingMarker, PrimaryRole};

/// A unit's document, quotation, and note, table cell or parallel text membership.
pub(crate) type OutlineContext = (String, Option<String>, Option<String>);

#[derive(Clone, Copy)]
enum Region {
    Heading,
    Text,
    UnderHeading,
    ListItem,
    Quote,
}

fn outline_region(node: &StructureNode, container: Option<NodeKind>) -> Option<Region> {
    if matches!(container, Some(NodeKind::Footnote | NodeKind::Endnote))
        || matches!(
            node.role,
            Some(PrimaryRole::Furniture | PrimaryRole::PageLabel | PrimaryRole::LineLabel | PrimaryRole::MarginalNote)
        )
        || node.grammar.as_deref() == Some("contents")
    {
        return None;
    }
    if !matches!(
        node.kind,
        NodeKind::Heading
            | NodeKind::ListItem
            | NodeKind::Paragraph
            | NodeKind::Prose
            | NodeKind::Quote
            | NodeKind::Table
    ) && !(node.kind == NodeKind::Section && node.marker_range.is_some())
    {
        return None;
    }
    Some(
        if container == Some(NodeKind::Cell) || node.kind == NodeKind::Table {
            Region::UnderHeading
        } else if node.role == Some(PrimaryRole::Heading) || node.kind == NodeKind::Heading {
            Region::Heading
        } else if node.kind == NodeKind::Quote {
            Region::Quote
        } else if matches!(
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
        ) {
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
    headings: Vec<(usize, usize)>,       // (level, node)
    lists: Vec<(String, Option<usize>)>, // (family, parent)
    previous: Option<usize>,
    open_letter: Option<NumberingMarker>,
}

/// Sets ranked units' parents from the headings' levels (`level`, by a heading's line id) and the
/// shared numbering grammar, the units read in `text`, each in its context (`contexts`). Only
/// constituent documents scope the stacks; quotation and object membership stay intact.
pub(crate) fn assemble_outline(
    nodes: &mut [StructureNode],
    contexts: &[OutlineContext],
    text: &str,
    level: &dyn Fn(&str) -> Option<usize>,
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
    let mut parents = vec![None; nodes.len()];
    let coordinates = ScalarText::new(text);
    let mut documents = HashMap::<&str, Outline>::new();
    for (index, region) in units {
        let outline = documents.entry(&contexts[index].0).or_default();
        let heading = outline.headings.last().map(|&(_, node)| node);
        let parent = match region {
            Region::Heading => {
                let level = nodes[index].line_ids.iter().find_map(|id| level(id)).unwrap_or(1);
                while outline
                    .headings
                    .last()
                    .is_some_and(|&(open, _)| open >= level)
                {
                    outline.headings.pop();
                }
                let parent = outline.headings.last().map(|&(_, node)| node);
                outline.headings.push((level, index));
                outline.lists.clear();
                outline.open_letter = None;
                parent
            }
            Region::Text => {
                outline.lists.clear();
                outline.open_letter = None;
                heading
            }
            Region::UnderHeading => heading,
            Region::ListItem => {
                let node = &nodes[index];
                let marker = node
                    .marker_range
                    .and_then(|range| coordinates.slice(range.start..range.end))
                    .or(node.label.as_deref());
                let marker = marker.and_then(|value| {
                    legal_pdf_core::layout_analysis()
                        .numbering_marker(value, outline.open_letter.as_ref())
                });
                let family = marker.as_ref().map_or("", |marker| marker.family.as_str());
                let found = outline
                    .lists
                    .iter()
                    .rposition(|(open, _)| !family.is_empty() && open == family);
                let parent = if let Some(at) = found {
                    let parent = outline.lists[at].1;
                    outline.lists.truncate(at + 1);
                    parent
                } else if family.is_empty() && !outline.lists.is_empty() {
                    outline.lists.last().unwrap().1
                } else {
                    let parent = outline.previous.or(heading);
                    outline.lists.push((family.to_owned(), parent));
                    parent
                };
                if marker
                    .as_ref()
                    .is_some_and(|marker| marker.family.rsplit(':').next() == Some("a"))
                {
                    outline.open_letter = marker;
                }
                parent
            }
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
        outline.previous = Some(index);
    }
}
