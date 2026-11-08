//! The rendered text, assembled from what the walk wrote: each line as its manuscript edits leave it, joined to the line
//! before by its join, a space within one unit's lines and a blank line between units, a page's notes after its body;
//! each node's place in it; and the authority units read off it.

use legal_pdf_core::model::Page;
use legal_pdf_core::{Error, Result};
use legal_pdf_structure_model::TRANSITION_ORIGIN;
use legal_structure_model::{utf16_len, DocumentStructure, NodeKind, PrimaryRole, ScalarRange, ScalarText, StructureNode};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

/// A stretch of the rendered text that is a source line's text unchanged: where it stands in each (UTF-16), and the line.
pub struct Run {
    target: ScalarRange,
    source: ScalarRange,
    fragment: usize,
}

fn fault(message: &str) -> Error {
    Error::Message(message.to_owned())
}

/// A unit's text node, or a note: the nodes whose lines are theirs in the rendered text.
fn owns_text(node: &StructureNode) -> bool {
    (node.origin_id == TRANSITION_ORIGIN || note(node)) && !node.line_ids.is_empty()
        && matches!(node.kind, NodeKind::Paragraph | NodeKind::Heading | NodeKind::Section | NodeKind::ListItem
            | NodeKind::Prose | NodeKind::Footnote | NodeKind::Endnote)
}

fn note(node: &StructureNode) -> bool {
    matches!(node.kind, NodeKind::Footnote | NodeKind::Endnote)
}

/// A line's text with its manuscript edits applied (scalar offsets in the line), and the stretches of it left unchanged:
/// each its start and end in the output and its start in the line, in UTF-16.
fn edited(text: &str, edits: &[(usize, usize, String)]) -> Result<(String, Vec<(usize, usize, usize)>)> {
    let source = ScalarText::new(text);
    let mut output = String::with_capacity(text.len());
    let mut kept = Vec::new();
    let (mut cursor, mut prior) = (0, None);
    for (start, end, replacement) in edits {
        if start > end || *start < cursor || prior.is_some_and(|prior| *start <= prior) {
            return Err(fault("Invalid manuscript edit order"));
        }
        let unchanged = source.slice(cursor..*start).ok_or_else(|| fault("Manuscript edit starts outside its source line"))?;
        if source.utf16_at_scalar(*end).is_none() {
            return Err(fault("Manuscript edit ends outside its source line"));
        }
        let at = utf16_len(&output);
        kept.push((at, at + utf16_len(unchanged), source.utf16_at_scalar(cursor).unwrap()));
        output.push_str(unchanged);
        output.push_str(replacement);
        cursor = *end;
        prior = Some(*start);
    }
    let from = source.utf16_at_scalar(cursor).unwrap();
    let rest = source.slice_utf16(from..source.utf16_len()).unwrap();
    let at = utf16_len(&output);
    kept.push((at, at + utf16_len(rest), from));
    output.push_str(rest);
    Ok((output, kept))
}

/// Writes the structure's rendered text and each node's rendered membership, ranges and range; the runs it is made of.
pub fn render(pages: &[Page], structure: &mut DocumentStructure) -> Result<Vec<Run>> {
    let mut text = String::with_capacity(structure.text.len());
    let mut runs = Vec::new();
    {
        let source = ScalarText::new(&structure.text);
        let fragment_of: HashMap<&str, usize> = structure.source_fragments.iter().enumerate().map(|(at, f)| (f.id.as_str(), at)).collect();
        // Each line's unit, a note's over a body unit's.
        let mut owners = HashMap::<&str, &str>::new();
        let mut note_lines = HashSet::<&str>::new();
        for node in structure.nodes.iter().filter(|node| owns_text(node) && !note(node)) {
            owners.extend(node.line_ids.iter().map(|line| (line.as_str(), node.id.as_str())));
        }
        for node in structure.nodes.iter().filter(|node| owns_text(node) && note(node)) {
            owners.extend(node.line_ids.iter().map(|line| (line.as_str(), node.id.as_str())));
            note_lines.extend(node.line_ids.iter().map(String::as_str));
        }
        let mut separators = HashMap::new();
        for join in &structure.joins {
            separators.entry((join.from, join.to)).or_insert(join.separator.as_str());
        }
        let mut position = 0;
        let mut previous_start = None;
        for page in pages {
            let mut lines: Vec<_> = page.lines.iter().collect();
            lines.sort_by_key(|line| (line.reading_order, line.source_index));
            let mut previous_owner = None;
            for notes in [false, true] {
                if notes {
                    previous_owner = None;
                }
                for line in lines.iter().filter(|line| (note_lines.contains(line.id.as_str())
                    || matches!(line.region_type.as_str(), "footnote" | "endnote" | "note")) == notes) {
                    let edits = structure.manuscript_edits.get(&line.id).map(Vec::as_slice).unwrap_or_default();
                    if line.text.trim().is_empty() && edits.is_empty() {
                        continue;
                    }
                    let &at = fragment_of.get(line.id.as_str()).ok_or_else(|| Error::Message(format!("Missing PDF source fragment {}", line.id)))?;
                    let fragment = &structure.source_fragments[at];
                    if fragment.page_index != page.index || source.slice_utf16(fragment.range.start..fragment.range.end) != Some(line.text.as_str()) {
                        return Err(Error::Message(format!("PDF source fragment {} no longer matches its extraction", line.id)));
                    }
                    let (normalized, kept) = edited(&line.text, edits)?;
                    let trimmed = normalized.trim();
                    if line.text.is_empty() || trimmed.is_empty() {
                        continue;
                    }
                    let owner = owners.get(line.id.as_str()).copied().unwrap_or(line.id.as_str());
                    if !text.is_empty() {
                        let separator = previous_start.and_then(|from| separators.get(&(from, fragment.range.start)).copied())
                            .unwrap_or(if previous_owner == Some(owner) { " " } else { "\n\n" });
                        text.push_str(separator);
                        position += utf16_len(separator);
                    }
                    let leading = utf16_len(&normalized[..normalized.len() - normalized.trim_start().len()]);
                    let trailing = leading + utf16_len(trimmed);
                    for (kept_start, kept_end, from) in kept {
                        let (start, end) = (kept_start.max(leading), kept_end.min(trailing));
                        if start < end {
                            let source_start = fragment.range.start + from + start - kept_start;
                            runs.push(Run {
                                target: ScalarRange { start: position + start - leading, end: position + end - leading },
                                source: ScalarRange { start: source_start, end: source_start + end - start },
                                fragment: at,
                            });
                        }
                    }
                    text.push_str(trimmed);
                    position += utf16_len(trimmed);
                    previous_owner = Some(owner);
                    previous_start = Some(fragment.range.start);
                }
            }
        }
    }
    let by_source = by_source(&runs);
    let rendered = ScalarText::new(&text);
    for node in &mut structure.nodes {
        let own = if node.ranges.is_empty() { std::slice::from_ref(&node.range) } else { &node.ranges };
        let membership = project(&runs, &by_source, &rendered, own);
        let extent = (matches!(node.kind, NodeKind::Heading | NodeKind::Section) || node.role == Some(PrimaryRole::Heading)) && !node.extent.is_empty();
        let exact = if extent { project(&runs, &by_source, &rendered, &node.extent) } else { membership.clone() };
        // An empty mapping stays empty; a first-page or source-envelope fallback would invent membership.
        node.rendered_range = Some(exact.first().zip(exact.last()).map_or(ScalarRange { start: 0, end: 0 },
            |(first, last)| ScalarRange { start: first.start, end: last.end }));
        node.rendered_membership = membership;
        node.rendered_ranges = exact;
    }
    structure.rendered_text = (text != structure.text).then_some(text);
    Ok(runs)
}

/// The runs by where their source starts.
fn by_source(runs: &[Run]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..runs.len()).collect();
    order.sort_by_key(|&at| runs[at].source.start);
    order
}

/// Source ranges (UTF-16) in the rendered text: the runs they cover, one range over what only whitespace, a dropped
/// character or a line's own insertion parts.
fn project(runs: &[Run], by_source: &[usize], rendered: &ScalarText, source: &[ScalarRange]) -> Vec<ScalarRange> {
    let mut pieces = Vec::<(ScalarRange, usize)>::new();
    for range in source {
        let first = by_source.partition_point(|&at| runs[at].source.end <= range.start);
        for run in by_source[first..].iter().map(|&at| &runs[at]).take_while(|run| run.source.start < range.end) {
            let (start, end) = (range.start.max(run.source.start), range.end.min(run.source.end));
            if start < end {
                let shift = run.target.start;
                pieces.push((ScalarRange { start: shift + start - run.source.start, end: shift + end - run.source.start }, run.fragment));
            }
        }
    }
    pieces.sort_by_key(|(range, _)| (range.start, range.end));
    let mut exact = Vec::<(ScalarRange, usize)>::new();
    for (range, fragment) in pieces {
        match exact.last_mut() {
            Some((prior, line)) if range.start <= prior.end || *line == fragment
                || rendered.slice_utf16(prior.end..range.start).is_some_and(|gap| gap.chars().all(char::is_whitespace)) => {
                prior.end = prior.end.max(range.end);
                *line = fragment;
            }
            _ => exact.push((range, fragment)),
        }
    }
    exact.into_iter().map(|(range, _)| range).collect()
}

/// A unit's text from its stretches of the rendered text, each after the first after a space, with the stretches in
/// `cuts` (a note's reference mark) left out: the text, its mappings to the source, and where each cut stood in it.
fn unit_text(structure: &DocumentStructure, runs: &[Run], rendered: &ScalarText, pieces: &[ScalarRange], cuts: &[(ScalarRange, usize)])
    -> (String, Vec<Value>, Vec<Value>) {
    let (mut text, mut length, mut mappings, mut references) = (String::new(), 0, Vec::new(), Vec::new());
    let mut segment = |text: &mut String, length: &mut usize, a: usize, b: usize| {
        if a >= b {
            return;
        }
        text.push_str(rendered.slice_utf16(a..b).unwrap_or_default());
        let first = runs.partition_point(|run| run.target.end <= a);
        for run in runs[first..].iter().take_while(|run| run.target.start < b) {
            let (start, end) = (a.max(run.target.start), b.min(run.target.end));
            if start < end {
                let source = run.source.start + start - run.target.start;
                mappings.push(json!({"start": *length + start - a, "end": *length + end - a, "sourceStart": source,
                    "sourceEnd": source + end - start, "lineId": structure.source_fragments[run.fragment].id}));
            }
        }
        *length += b - a;
    };
    for (at, piece) in pieces.iter().enumerate() {
        if at > 0 {
            text.push(' ');
            length += 1;
        }
        let mut cursor = piece.start;
        for (cut, ordinal) in cuts.iter().filter(|(cut, _)| piece.start <= cut.start && cut.end <= piece.end) {
            segment(&mut text, &mut length, cursor, cut.start.max(cursor));
            references.push(json!([ordinal, length]));
            cursor = cursor.max(cut.end);
        }
        segment(&mut text, &mut length, cursor, piece.end);
    }
    (text, mappings, references)
}

fn source_of(structure: &DocumentStructure, node: &StructureNode, mappings: &[Value]) -> Value {
    let ranges: Vec<Value> = mappings.iter().map(|m| json!({"start": m["sourceStart"], "end": m["sourceEnd"]})).collect();
    json!({"documentId": structure.document_id, "textSha256": structure.text_sha256, "coordinateSpace": "source_utf16",
        "nodeIds": [node.id], "lineIds": node.line_ids, "ranges": ranges, "mappings": mappings})
}

/// The authority units: each body unit's text, its notes' marks left out and where each stood; then each note's body, a
/// note with its own mark (an author's "*") keeping the mark as the document prints it.
pub fn authority_units(structure: &DocumentStructure, runs: &[Run]) -> Vec<Value> {
    let rendered = ScalarText::new(structure.rendered_text.as_deref().unwrap_or(&structure.text));
    let by_source = by_source(runs);
    let notes: Vec<&StructureNode> = structure.nodes.iter().filter(|node| note(node)).collect();
    let mut cuts: Vec<(ScalarRange, usize)> = notes.iter().enumerate()
        .flat_map(|(at, node)| project(runs, &by_source, &rendered, &node.calls).into_iter().map(move |cut| (cut, at + 1)))
        .collect();
    cuts.sort_by_key(|(cut, _)| cut.start);
    let mut body: Vec<&StructureNode> = structure.nodes.iter()
        .filter(|node| owns_text(node) && !note(node) && !node.rendered_membership.is_empty()
            && !matches!(node.role, Some(PrimaryRole::Furniture | PrimaryRole::PageLabel | PrimaryRole::LineLabel)))
        .collect();
    body.sort_by_key(|node| node.rendered_membership[0].start);
    let mut units: Vec<Value> = body.iter().enumerate().map(|(at, node)| {
        let (text, mappings, references) = unit_text(structure, runs, &rendered, &node.rendered_membership, &cuts);
        json!({"key": format!("body:{at}"), "source": source_of(structure, node, &mappings), "kind": "body", "ordinal": at,
            "footnote_id": Value::Null, "page_numbers": node.page_indexes.first().map(|page| vec![page + 1]).unwrap_or_default(),
            "text": text, "footnote_refs": references})
    }).collect();
    units.extend(notes.iter().enumerate().map(|(at, node)| {
        let ordinal = at + 1;
        let label = node.printed_label();
        let number = label.parse::<u32>().ok();
        let (mut text, mut mappings, _) = unit_text(structure, runs, &rendered, &node.rendered_membership, &[]);
        if text.is_empty() {
            text = label.to_owned();
        }
        if number.is_none() && !label.is_empty() && text != label {
            let prefix = utf16_len(label) + 1;
            text = format!("{label} {text}");
            for mapping in &mut mappings {
                mapping["start"] = json!(mapping["start"].as_u64().unwrap_or(0) as usize + prefix);
                mapping["end"] = json!(mapping["end"].as_u64().unwrap_or(0) as usize + prefix);
            }
        }
        json!({"key": format!("footnote:{ordinal}"), "source": source_of(structure, node, &mappings), "kind": "footnote",
            "ordinal": ordinal, "footnote_id": ordinal, "note_number": number, "restart_sequence": node.series.parse::<usize>().unwrap_or(0),
            "page_numbers": node.page_indexes.iter().map(|page| page + 1).collect::<Vec<_>>(), "text": text, "footnote_refs": []})
    }));
    units
}
