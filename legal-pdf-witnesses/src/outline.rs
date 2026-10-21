//! Each unit's parent, chosen in reading order among the open units of its own context: a heading
//! nests under a more prominent heading, a numbered unit under the unit its numbering descends
//! from, other units under the nearest open heading or lead-in, and the printed contents or
//! bookmarks place a heading where they name its parent.

use crate::witness::{median, size};
use legal_pdf_core::model::{Line, Page};
use legal_structure_model::{NodeKind, NumberingMarker, ScalarRange, ScalarText, StructureNode};
use std::collections::HashMap;

/// Where a unit's outline lives: its document, quotation, and note, table cell or parallel text.
pub type Context = (String, Option<String>, Option<String>);

// Native font sizes and recognized glyph heights need separate baselines.
#[derive(Clone, Copy)]
struct BodySizes {
    font: f64,
    glyph: f64,
}

struct Unit {
    node: usize,
    context: Context,
    marker: Option<NumberingMarker>,
    style: (i32, bool, bool),
    rank: f32,
    level: Option<usize>,
    lead_in: bool,
}

fn heading(node: &StructureNode) -> bool {
    node.role == Some(legal_structure_model::PrimaryRole::Heading) || node.kind == NodeKind::Heading
}

fn outline_unit(node: &StructureNode) -> bool {
    if matches!(node.role, Some(legal_structure_model::PrimaryRole::Furniture | legal_structure_model::PrimaryRole::PageLabel)) {
        return false;
    }
    matches!(node.kind, NodeKind::Heading | NodeKind::ListItem | NodeKind::Paragraph | NodeKind::Prose)
        || node.kind == NodeKind::Section && node.marker_range.is_some()
}

/// The unit's own parts of its text, never the text between them.
fn membership_text(node: &StructureNode, text: &ScalarText<'_>) -> Option<String> {
    let ranges = if node.ranges.is_empty() { std::slice::from_ref(&node.range) } else { &node.ranges };
    let ranges = ranges.iter().map(|range| Some(ScalarRange {
        start: text.utf16_at_scalar(range.start)?,
        end: text.utf16_at_scalar(range.end)?,
    })).collect::<Option<Vec<_>>>()?;
    Some(text.membership_text_utf16(&ranges))
}

/// The type of the unit's own text on its first line (`source`, that line's place in the text):
/// its size in hundredths of the body size, and whether most of it is bold and italic.
fn unit_style(
    line: &Line,
    source: ScalarRange,
    node: &StructureNode,
    body: BodySizes,
) -> (i32, bool, bool) {
    let parts = if node.ranges.is_empty() {
        std::slice::from_ref(&node.range)
    } else {
        &node.ranges
    };
    let ranges = parts
        .iter()
        .filter_map(|part| {
            let (start, end) = (part.start.max(source.start), part.end.min(source.end));
            (start < end).then_some((start - source.start, end - source.start))
        })
        .collect::<Vec<_>>();
    let coordinates = ScalarText::new(&line.text);
    let (mut weight, mut relative_size, mut bold, mut italic) = (0usize, 0.0f32, 0usize, 0usize);
    for span in line.spans.iter().filter(|span| {
        span.start < span.end && coordinates.slice(span.start..span.end) == Some(span.text.as_str())
    }) {
        let count = ranges
            .iter()
            .map(|(lo, hi)| span.end.min(*hi).saturating_sub(span.start.max(*lo)))
            .sum::<usize>();
        let share = if body.font > 0.0 {
            (span.size / body.font) as f32
        } else {
            0.0
        };
        if count == 0 || share <= 0.0 {
            continue;
        }
        weight += count;
        relative_size += share * count as f32;
        bold += usize::from(span.flags & 16 != 0 || span.font.to_lowercase().contains("bold")) * count;
        italic += usize::from(span.flags & 2 != 0 || span.font.to_lowercase().contains("ital")) * count;
    }
    if weight == 0 && size(line) <= 0.0 && !ranges.is_empty() && body.glyph > 0.0 {
        let ratio = (line.bbox[3] - line.bbox[1]) / body.glyph;
        return ((ratio * 100.0).round() as i32, false, false);
    }
    let weight = weight.max(1) as f32;
    (
        (relative_size / weight * 100.0).round() as i32,
        bold as f32 / weight >= 0.5,
        italic as f32 / weight >= 0.5,
    )
}

/// Set every outline unit's parent. `contexts` holds each node's context; a parent link between
/// contexts is dropped, except a row's to its table and a cell's to its row. `line_range` gives a
/// page line's place in `text`.
pub fn assemble_outline(
    nodes: &mut [StructureNode],
    contexts: &[Context],
    text: &str,
    pages: &[Page],
    line_range: &dyn Fn(&str) -> Option<ScalarRange>,
    navigation: &HashMap<String, Option<String>>,
    heading_levels: &HashMap<String, usize>,
) {
    let lines = pages
        .iter()
        .flat_map(|page| &page.lines)
        .map(|line| (line.id.as_str(), line))
        .collect::<HashMap<_, _>>();
    let mut type_lines = HashMap::<Context, HashMap<&str, (f64, f64)>>::new();
    for (node, context) in nodes
        .iter()
        .zip(contexts)
        .filter(|(node, _)| outline_unit(node))
    {
        let own = type_lines.entry(context.clone()).or_default();
        for id in &node.line_ids {
            if let Some(line) = lines.get(id.as_str()) {
                own.entry(id)
                    .or_insert_with(|| (size(line), line.bbox[3] - line.bbox[1]));
            }
        }
    }
    let body_sizes = type_lines
        .into_iter()
        .map(|(context, lines)| {
            let body = BodySizes {
                font: median(
                    lines
                        .values()
                        .map(|(font, _)| *font)
                        .filter(|font| *font > 0.0)
                        .collect(),
                ),
                glyph: median(
                    lines
                        .values()
                        .filter(|(font, _)| *font <= 0.0)
                        .map(|(_, glyph)| *glyph)
                        .filter(|glyph| *glyph > 0.0)
                        .collect(),
                ),
            };
            (context, body)
        })
        .collect::<HashMap<_, _>>();
    let coordinates = ScalarText::new(text);
    let mut units = nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| outline_unit(node))
        .map(|(node_index, node)| {
            let source = |id: &String| Some((*lines.get(id.as_str())?, line_range(id)?));
            let first = node
                .line_ids
                .iter()
                .filter_map(source)
                .find(|(_, range)| node.content_start.is_none_or(|start| start < range.end))
                .or_else(|| node.line_ids.first().and_then(source));
            Unit {
                node: node_index,
                context: contexts[node_index].clone(),
                marker: None,
                style: first.map_or((0, false, false), |(line, source)| {
                    unit_style(line, source, node, body_sizes[&contexts[node_index]])
                }),
                rank: 0.0,
                level: node
                    .line_ids
                    .iter()
                    .find_map(|id| heading_levels.get(id).copied()),
                lead_in: membership_text(node, &coordinates)
                    .unwrap_or_default()
                    .trim()
                    .ends_with(':'),
            }
        })
        .collect::<Vec<_>>();
    units.sort_by_key(|unit| {
        (
            nodes[unit.node].range.start,
            nodes[unit.node].range.end,
            nodes[unit.node].id.clone(),
        )
    });
    let mut styles = HashMap::<Context, Vec<(i32, bool, bool)>>::new();
    let mut previous_marker = HashMap::<Context, NumberingMarker>::new();
    for unit in &mut units {
        styles.entry(unit.context.clone()).or_default().push(unit.style);
        let node = &nodes[unit.node];
        let marker = node.marker_range.and_then(|range| coordinates.slice(range.start..range.end)).or(node.label.as_deref());
        unit.marker = marker.and_then(|marker| legal_pdf_core::layout_analysis().numbering_marker(marker, previous_marker.get(&unit.context)));
        if let Some(marker) = &unit.marker {
            previous_marker.insert(unit.context.clone(), marker.clone());
        }
    }
    // A unit's rank is its type's place among its context's types, most prominent first.
    for ladder in styles.values_mut() {
        ladder.sort_unstable_by(|left, right| right.cmp(left));
        ladder.dedup();
    }
    for unit in &mut units {
        unit.rank = styles[&unit.context].iter().position(|style| style == &unit.style).unwrap_or(0) as f32;
    }
    let mut heading_targets = HashMap::<&str, Option<usize>>::new();
    for (position, unit) in units.iter().enumerate().filter(|(_, unit)| heading(&nodes[unit.node])) {
        heading_targets.insert(nodes[unit.node].id.as_str(), Some(position));
        for line in &nodes[unit.node].line_ids {
            heading_targets.entry(line.as_str()).and_modify(|prior| *prior = None).or_insert(Some(position));
        }
    }
    let navigation_claim = |node: &StructureNode| navigation.get(&node.id).or_else(|| node.line_ids.iter()
        .filter(|line| heading_targets.get(line.as_str()).is_some_and(Option::is_some))
        .find_map(|line| navigation.get(line)));
    let mut stacks = HashMap::<Context, Vec<usize>>::new();
    let mut chosen: Vec<Option<usize>> = vec![None; units.len()];
    for (position, unit) in units.iter().enumerate() {
        let stack = stacks.entry(unit.context.clone()).or_default();
        let node = &nodes[unit.node];
        let mut parent = stack.iter().rposition(|&candidate| {
            let parent = &units[candidate];
            if heading(node) {
                return heading(&nodes[parent.node])
                    && match (parent.level, unit.level) {
                        (Some(parent), Some(child)) => parent < child,
                        _ => parent.rank < unit.rank,
                    };
            }
            if let (Some(marker), Some(other)) = (&unit.marker, &parent.marker) {
                if marker.family == other.family && marker.prefix == other.prefix {
                    return false;
                }
                let mut components = other.prefix.clone();
                components.push(other.ordinal);
                if marker.prefix.starts_with(&components) {
                    return true;
                }
                // A newly opened count nests like the shared text parser's numbering stack.
                if marker.ordinal == 1 && marker.prefix.is_empty() && marker.family != other.family {
                    return true;
                }
            }
            heading(&nodes[parent.node]) || parent.lead_in
        }).map(|index| stack[index]);
        // A sibling closes itself and its descendants while retaining its enclosing ancestors.
        if let Some(sibling) = unit.marker.as_ref().and_then(|marker| stack.iter().rev().find(|&&candidate|
            units[candidate].marker.as_ref().is_some_and(|other| marker.family == other.family && marker.prefix == other.prefix))) {
            parent = chosen[*sibling];
        }
        let navigation_parent = navigation_claim(node).and_then(|claimed| match claimed {
            None => Some(None),
            Some(line) => heading_targets.get(line.as_str()).copied().flatten()
                .filter(|&parent| parent < position && units[parent].context == unit.context).map(Some),
        });
        if let Some(claimed) = navigation_parent.filter(|claimed| claimed.is_none_or(|parent| stack.contains(&parent))) {
            parent = claimed;
        }
        chosen[position] = parent;
        match parent.map(|parent| (parent, stack.iter().position(|&p| p == parent))) {
            Some((_, Some(at))) => stack.truncate(at + 1),
            // A parent off the stack reopens its own ancestry.
            Some((parent, None)) => {
                stack.clear();
                let mut ancestor = Some(parent);
                while let Some(at) = ancestor {
                    stack.push(at);
                    ancestor = chosen[at];
                }
                stack.reverse();
            }
            None => stack.clear(),
        }
        stack.push(position);
    }
    // A section and its first subsection can share one unit. Later subsections and their
    // marginal notes belong to that unit until a heading or the next numbered section.
    // Keep one open section per context, including each side of a parallel text.
    let mut sections = HashMap::<Context, (usize, Option<usize>)>::new();
    for (position, unit) in units.iter().enumerate() {
        if heading(&nodes[unit.node]) {
            sections.remove(&unit.context);
            continue;
        }
        let own = membership_text(&nodes[unit.node], &coordinates).unwrap_or_default();
        let mut words = own.split_whitespace();
        let number = words
            .next()
            .and_then(|word| legal_pdf_core::layout_analysis().numbering_marker(word, None))
            .is_some_and(|marker| {
                marker.family.split_once(':').is_some_and(|(wrap, count)| {
                    matches!(wrap, "" | ".") && count.split('.').all(|part| part == "1")
                })
            });
        if number {
            sections.remove(&unit.context);
            if words.next() == Some("(1)") {
                sections.insert(unit.context.clone(), (position, chosen[position]));
            }
        } else if let Some(&(section, parent)) = sections.get(&unit.context) {
            if chosen[position] == parent {
                chosen[position] = Some(section);
            }
        }
    }
    for (position, unit) in units.iter().enumerate() {
        nodes[unit.node].parent_id = chosen[position].map(|parent| nodes[units[parent].node].id.clone());
    }
    let positions = nodes.iter().enumerate().map(|(position, node)| (node.id.clone(), position)).collect::<HashMap<_, _>>();
    for position in 0..nodes.len() {
        let Some(parent) = nodes[position].parent_id.as_ref().map(|id| positions.get(id).copied()) else { continue };
        let grid = parent.is_some_and(|parent| matches!((nodes[position].kind, nodes[parent].kind),
            (NodeKind::Cell, NodeKind::Row) | (NodeKind::Row, NodeKind::Table)));
        if !grid && parent.is_none_or(|parent| contexts[parent] != contexts[position]) {
            nodes[position].parent_id = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use legal_structure_model::Derivation;

    #[test]
    fn new_numbering_families_nest_and_close_at_siblings_and_headings() {
        legal_structure_model::install_structure_analysis(legal_pdf_core::CHECKOUT_ENGINE);
        let entries = [
            ("Field kit", None),
            ("4. Carry a flask.", Some("4.")),
            ("(a) Use a blue cap.", Some("(a)")),
            ("(i) Check the seal.", Some("(i)")),
            ("(ii) Fill the flask.", Some("(ii)")),
            ("(b) Label the bottle.", Some("(b)")),
            ("5. Pack the pouch.", Some("5.")),
            ("(a) Keep it dry.", Some("(a)")),
            ("Shelter", None),
            ("(a) Choose a level site.", Some("(a)")),
            ("7. Check the pegs.", Some("7.")),
            ("(d) Count the spare pegs.", Some("(d)")),
            ("(a) A separate enclosure.", Some("(a)")),
        ];
        let text = entries
            .iter()
            .map(|(text, _)| *text)
            .collect::<Vec<_>>()
            .join("\n");
        let mut offset = 0;
        let mut nodes = entries
            .iter()
            .enumerate()
            .map(|(n, (text, marker))| {
                let range = ScalarRange {
                    start: offset,
                    end: offset + text.chars().count(),
                };
                offset = range.end + 1;
                let mut node = StructureNode::new(
                    format!("unit{n}"),
                    if marker.is_none() {
                        NodeKind::Heading
                    } else {
                        NodeKind::ListItem
                    },
                    range,
                    "invented-numbering",
                    Derivation::Native,
                    None,
                );
                node.marker_range = marker.map(|marker| ScalarRange {
                    start: range.start,
                    end: range.start + marker.chars().count(),
                });
                node
            })
            .collect::<Vec<_>>();
        let mut contexts = vec![("document".into(), None, None); nodes.len()];
        contexts[12].0 = "enclosure".into();
        assemble_outline(
            &mut nodes,
            &contexts,
            &text,
            &[],
            &|_| None,
            &HashMap::new(),
            &HashMap::new(),
        );
        let parents = nodes
            .iter()
            .map(|node| node.parent_id.as_deref())
            .collect::<Vec<_>>();
        assert_eq!(
            parents,
            [
                None,
                Some("unit0"),
                Some("unit1"),
                Some("unit2"),
                Some("unit2"),
                Some("unit1"),
                Some("unit0"),
                Some("unit6"),
                None,
                Some("unit8"),
                Some("unit8"),
                Some("unit8"),
                None,
            ]
        );
    }

    #[test]
    fn heading_ladder_survives_equal_type_and_respects_document_context() {
        let titles = [
            "Harbour handbook",
            "Navigation",
            "Read the tide chart.",
            "Equipment",
            "Inspect the compass.",
            "Island handbook",
            "Check the forecast.",
            "A separate enclosure.",
        ];
        let text = titles.join("\n");
        let mut offset = 0;
        let mut nodes = titles
            .iter()
            .enumerate()
            .map(|(n, title)| {
                let range = ScalarRange {
                    start: offset,
                    end: offset + title.chars().count(),
                };
                offset = range.end + 1;
                let mut node = StructureNode::new(
                    format!("unit{n}"),
                    if [0, 1, 3, 5].contains(&n) {
                        NodeKind::Heading
                    } else {
                        NodeKind::Prose
                    },
                    range,
                    "invented-outline",
                    Derivation::Native,
                    None,
                );
                node.line_ids = vec![format!("line{n}")];
                node
            })
            .collect::<Vec<_>>();
        let mut contexts = vec![("document".into(), None, None); nodes.len()];
        contexts[7].0 = "enclosure".into();
        let levels = [(0, 1), (1, 2), (3, 2), (5, 1)]
            .into_iter()
            .map(|(n, level)| (format!("line{n}"), level))
            .collect();
        assemble_outline(
            &mut nodes, &contexts, &text, &[], &|_| None, &HashMap::new(), &levels,
        );
        let parents = nodes
            .iter()
            .map(|node| node.parent_id.as_deref())
            .collect::<Vec<_>>();
        assert_eq!(parents, [
            None, Some("unit0"), Some("unit1"), Some("unit0"),
            Some("unit3"), None, Some("unit5"), None,
        ]);

        let navigation = HashMap::from([("line3".into(), None)]);
        assemble_outline(
            &mut nodes, &contexts, &text, &[], &|_| None, &navigation, &levels,
        );
        assert_eq!(nodes[3].parent_id, None);
        assert_eq!(nodes[4].parent_id.as_deref(), Some("unit3"));
    }
}
