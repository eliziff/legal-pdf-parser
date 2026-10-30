//! The rendered text: a manuscript's edits and the source lines a paragraph continues across.

use super::*;

pub(crate) fn append(text: &mut String, position: &mut usize, value: &str) {
    text.push_str(value);
    *position += utf16_len(value);
}

pub(crate) fn extend_range(range: Option<ScalarRange>, next: ScalarRange) -> Option<ScalarRange> {
    Some(range.map_or(next, |range| ScalarRange {
        start: range.start.min(next.start),
        end: range.end.max(next.end),
    }))
}

/// Preserve a native producer's witnessed dehyphenation as a manuscript-only edit.
pub(crate) fn attach_manuscript_edits(paragraphs: &[Paragraph], structure: &mut DocumentStructure)
    -> Vec<(legal_structure_model::SourceJoin, Option<legal_structure_model::SourceContinuation>)> {
    let source = ScalarText::new(&structure.text);
    let fragments: HashMap<_, _> = structure.source_fragments.iter()
        .map(|fragment| (fragment.id.as_str(), fragment)).collect();
    let mut edits = legal_structure_model::ManuscriptEdits::new();
    let mut joins = Vec::new();
    let mut diagnostics = Vec::new();
    let mut node_at = HashMap::<&str, usize>::new();
    for (index, node) in structure.nodes.iter().enumerate() {
        node_at.entry(node.id.as_str()).or_insert(index);
    }
    // What each candidate seam is checked against, indexed once: the fragments by where they start
    // and by page, and the joins by the fragment they join from.
    let mut fragment_starts: Vec<ScalarRange> = structure.source_fragments.iter().map(|fragment| fragment.range).collect();
    fragment_starts.sort_by_key(|range| range.start);
    let mut page_fragments = HashMap::<usize, Vec<ScalarRange>>::new();
    for fragment in &structure.source_fragments {
        page_fragments.entry(fragment.page_index).or_default().push(fragment.range);
    }
    let mut joins_from = HashMap::<usize, Vec<(usize, bool)>>::new();
    for join in &structure.joins {
        joins_from.entry(join.from).or_default().push((join.to, join.separator.is_empty()));
    }
    // The ranges of the nodes that own text (a role, or a leaf kind), each with its node (the first
    // node of its id) by where it starts; beside each, the furthest end of those up to it, and the
    // furthest of another node's: another node overlaps a part where one starting before the
    // part's end ends after its start.
    let mut owned: Vec<(usize, usize, usize)> = structure.nodes.iter()
        .filter(|other| other.role.is_some() || matches!(other.kind,
            NodeKind::Prose | NodeKind::Paragraph | NodeKind::Section | NodeKind::ListItem
            | NodeKind::Heading | NodeKind::Footnote | NodeKind::Endnote))
        .flat_map(|other| {
            let owner = node_at[other.id.as_str()];
            (if other.ranges.is_empty() { std::slice::from_ref(&other.range) } else { &other.ranges })
                .iter().map(move |range| (range.start, range.end, owner))
        })
        .collect();
    owned.sort_by_key(|&(start, _, _)| start);
    let mut furthest: Vec<(usize, usize, Option<usize>)> = Vec::with_capacity(owned.len());
    for &(_, end, owner) in &owned {
        furthest.push(match furthest.last().copied() {
            None => (end, owner, None),
            Some((first, first_owner, other)) if owner == first_owner => (first.max(end), owner, other),
            Some((first, _, _)) if end > first => (end, owner, Some(first)),
            Some((first, first_owner, other)) => (first, first_owner, Some(other.map_or(end, |other| other.max(end)))),
        });
    }
    let overlapped_by_another = |node: usize, part: ScalarRange| {
        let before = owned.partition_point(|&(start, _, _)| start < part.end);
        before > 0 && {
            let (first, first_owner, other) = furthest[before - 1];
            (if first_owner != node { Some(first) } else { other }).is_some_and(|end| part.start < end)
        }
    };
    let mut joined_here = HashSet::new();
    for paragraph in paragraphs {
        let Some(&index) = node_at.get(paragraph.id.as_str()) else { continue; };
        let node = &structure.nodes[index];
        let own = if matches!(node.kind, NodeKind::Heading)
            || node.role == Some(legal_structure_model::PrimaryRole::Heading) {
            node.heading_range.as_ref().map(std::slice::from_ref)
        } else { None }.unwrap_or_else(|| {
            if node.ranges.is_empty() { std::slice::from_ref(&node.range) } else { &node.ranges }
        });
        let output = ScalarText::new(&paragraph.text);
        for pair in paragraph.source_spans.windows(2) {
            let left = &pair[0];
            let right = &pair[1];
            let discover = || {
                if left.line_id == right.line_id || output.slice_utf16(left.end..right.start) != Some("") {
                    return None;
                }
                let a = *fragments.get(left.line_id.as_str())?;
                let b = *fragments.get(right.line_id.as_str())?;
                for (span, fragment) in [(left, a), (right, b)] {
                    let start = fragment.range.start.checked_add(span.source_start)?;
                    let end = fragment.range.start.checked_add(span.source_end)?;
                    if span.start >= span.end || span.source_start >= span.source_end
                        || end > fragment.range.end || span.end - span.start != end - start
                        || !own.iter().any(|range| range.start <= fragment.range.start && fragment.range.end <= range.end)
                        || output.slice_utf16(span.start..span.end).is_none()
                        || output.slice_utf16(span.start..span.end) != source.slice_utf16(start..end) {
                        return None;
                    }
                }
                let original = source.slice_utf16(a.range.start..a.range.end)?;
                let coordinates = ScalarText::new(original);
                let omitted_end = left.source_end.checked_add(1)?;
                let omitted = coordinates.slice_utf16(left.source_end..omitted_end)?;
                if !matches!(omitted, "-" | "\u{00ad}" | "\u{00ac}")
                    || omitted_end != a.range.end - a.range.start
                    || right.source_start != 0
                    || !output.slice_utf16(right.start..right.end)?.chars().next().is_some_and(char::is_lowercase) {
                    return None;
                }
                let start = coordinates.scalar_at_utf16(left.source_end)?;
                let end = coordinates.scalar_at_utf16(omitted_end)?;
                let edit = (start, end, String::new());
                if structure.manuscript_edits.get(&left.line_id).is_some_and(|existing| !existing.contains(&edit)) {
                    return None;
                }
                {
                    // No competing leaf ownership or intervening member may change these seed starts.
                    if a.range.start >= b.range.start || fragment_starts[
                            fragment_starts.partition_point(|fragment| fragment.start <= a.range.start)
                                ..fragment_starts.partition_point(|fragment| fragment.start < b.range.start)]
                            .iter().any(|fragment| own.iter().any(|range| range.start < fragment.end && fragment.start < range.end))
                        || [a.range, b.range].into_iter().any(|part| overlapped_by_another(index, part)) {
                        return None;
                    }
                    if joins_from.get(&a.range.start).is_some_and(|joins| joins.iter()
                        .any(|&(to, bare)| to != b.range.start || !bare)) { return None; }
                }
                let continuation = if a.page_index != b.page_index {
                    let first = |page| page_fragments.get(&page).into_iter().flatten()
                        .flat_map(|fragment| own.iter().filter_map(move |range| {
                            let start = range.start.max(fragment.start);
                            (start < range.end.min(fragment.end)).then_some(start)
                        })).min();
                    Some(legal_structure_model::SourceContinuation {
                        kind: if node.kind == NodeKind::Heading || node.role == Some(legal_structure_model::PrimaryRole::Heading) {
                            legal_structure_model::ContinuationKind::Heading
                        } else { legal_structure_model::ContinuationKind::Paragraph },
                        from: first(a.page_index)?, to: first(b.page_index)?, separator: Some(String::new()),
                    })
                } else { None };
                Some((a, b, edit, ScalarRange {
                    start: a.range.start + left.source_end, end: a.range.start + omitted_end,
                }, continuation))
            };
            let Some((a, b, edit, range, continuation)) = discover() else { continue; };
            if !structure.manuscript_edits.contains_key(&left.line_id) {
                let line = edits.entry(left.line_id.clone()).or_default();
                if !line.contains(&edit) { line.push(edit); }
                diagnostics.push(legal_structure_model::StructureDiagnostic {
                    code: "native_manuscript_dehyphenation".to_owned(),
                    severity: legal_structure_model::DiagnosticSeverity::Info,
                    ranges: vec![range], node_ids: vec![node.id.clone()],
                });
            }
            if !joins_from.contains_key(&a.range.start) && joined_here.insert(a.range.start) {
                joins.push((legal_structure_model::SourceJoin { from: a.range.start,
                    to: b.range.start, separator: String::new() }, continuation));
            }
        }
    }
    for line in edits.values_mut() { line.sort_by_key(|edit| (edit.0, edit.1)); }
    structure.manuscript_edits.extend(edits);
    structure.diagnostics.extend(diagnostics);
    joins
}

/// Retain only fully mapped page seams from a single native logical producer.
/// Source endpoints name fragment starts; producer run ends only delimit the separator.
pub(crate) fn attach_source_continuations(
    paragraphs: &[Paragraph],
    footnotes: &[Footnote],
    structure: &mut DocumentStructure,
) {
    if structure.continuations.is_some() {
        return;
    }
    let source = ScalarText::new(&structure.text);
    let fragments: HashMap<_, _> = structure.source_fragments.iter()
        .map(|fragment| (fragment.id.as_str(), fragment)).collect();
    let nodes: HashMap<_, _> = structure.nodes.iter()
        .map(|node| (node.id.as_str(), node)).collect();
    // The source lines by where they start, each with the furthest end so far, so a node finds
    // the lines it overlaps without reading every line.
    let mut by_start = (0..structure.source_fragments.len()).collect::<Vec<_>>();
    by_start.sort_by_key(|&slot| structure.source_fragments[slot].range.start);
    let reach = by_start.iter().scan(0, |end, &slot| {
        *end = (*end).max(structure.source_fragments[slot].range.end);
        Some(*end)
    }).collect::<Vec<_>>();
    let links = |node: &StructureNode, produced: &str,
                 spans: &[legal_pdf_core::model::SourceTextSpan], kind| {
        let heading = node.kind == NodeKind::Heading
            || node.role == Some(legal_structure_model::PrimaryRole::Heading);
        let own = heading.then_some(node.heading_range.as_ref()).flatten()
            .map(std::slice::from_ref).unwrap_or_else(|| {
                if node.ranges.is_empty() { std::slice::from_ref(&node.range) } else { &node.ranges }
            });
        let output = ScalarText::new(produced);
        let mut starts = HashMap::<usize, usize>::new();
        let mut ends = HashMap::<usize, usize>::new();
        let mut member_lines = HashSet::new();
        for range in own {
            let upto = by_start.partition_point(|&slot| structure.source_fragments[slot].range.start < range.end);
            let from = reach[..upto].partition_point(|end| *end <= range.start);
            for fragment in by_start[from..upto].iter().map(|&slot| &structure.source_fragments[slot]) {
                let start = range.start.max(fragment.range.start);
                let end = range.end.min(fragment.range.end);
                if start < end {
                    starts.entry(fragment.page_index).and_modify(|at| *at = (*at).min(start)).or_insert(start);
                    ends.entry(fragment.page_index).and_modify(|at| *at = (*at).max(end)).or_insert(end);
                    member_lines.insert(fragment.id.as_str());
                }
            }
        }
        let mut pages = Vec::<(usize, usize, usize)>::new();
        let mut mapped_lines = HashSet::new();
        let mut prior_end = 0;
        for (index, span) in spans.iter().enumerate() {
            let fragment = *fragments.get(span.line_id.as_str())?;
            let start = fragment.range.start.checked_add(span.source_start)?;
            let end = fragment.range.start.checked_add(span.source_end)?;
            if span.start < prior_end || span.start >= span.end || span.source_start >= span.source_end
                || end > fragment.range.end || span.end - span.start != end - start
                || !own.iter().any(|range| range.start <= start && end <= range.end)
                || output.slice_utf16(span.start..span.end).is_none()
                || output.slice_utf16(span.start..span.end) != source.slice_utf16(start..end) {
                return None;
            }
            prior_end = span.end;
            mapped_lines.insert(span.line_id.as_str());
            if let Some(last) = pages.last_mut().filter(|last| last.0 == fragment.page_index) {
                last.2 = index;
            } else {
                if pages.iter().any(|page| page.0 == fragment.page_index) { return None; }
                pages.push((fragment.page_index, index, index));
            }
        }
        // A partial producer map must not split the native object into partial facts.
        if mapped_lines != member_lines || pages.len() != starts.len() { return None; }
        let mut result = Vec::new();
        for pair in pages.windows(2) {
            let from = *starts.get(&pair[0].0)?;
            let to = *starts.get(&pair[1].0)?;
            let left = &spans[pair[0].2];
            let right = &spans[pair[1].1];
            let separator = output.slice_utf16(left.end..right.start)?;
            let left_end = fragments.get(left.line_id.as_str())?.range.start + left.source_end;
            let right_start = fragments.get(right.line_id.as_str())?.range.start + right.source_start;
            let suffix = source.slice_utf16(left_end..*ends.get(&pair[0].0)?)?;
            let prefix = source.slice_utf16(to..right_start)?;
            let left_fragment = fragments.get(left.line_id.as_str())?;
            let left_text = ScalarText::new(source.slice_utf16(left_fragment.range.start..left_fragment.range.end)?);
            let scalar = left_text.scalar_at_utf16(left.source_end)?;
            let removed_hyphen = suffix.chars().next().is_some_and(|glyph| matches!(glyph, '-' | '\u{00ad}' | '\u{00ac}'))
                && suffix.chars().count() == 1 && prefix.is_empty()
                && structure.manuscript_edits.get(&left.line_id).is_some_and(|edits|
                    edits.iter().any(|(start, end, replacement)| *start == scalar && *end == scalar + 1 && replacement.is_empty()));
            // Only an explicit matching manuscript edit may remove the printed seam glyph.
            if from >= to || !separator.chars().all(char::is_whitespace)
                || !(suffix.chars().all(char::is_whitespace) || removed_hyphen)
                || !prefix.chars().all(char::is_whitespace) {
                return None;
            }
            result.push(legal_structure_model::SourceContinuation {
                kind, from, to, separator: Some(separator.to_owned()),
            });
        }
        Some(result)
    };
    let mut continuations = Vec::new();
    for paragraph in paragraphs {
        if let Some(node) = nodes.get(paragraph.id.as_str()) {
            let kind = if node.kind == NodeKind::Heading
                || node.role == Some(legal_structure_model::PrimaryRole::Heading) {
                legal_structure_model::ContinuationKind::Heading
            } else { legal_structure_model::ContinuationKind::Paragraph };
            if let Some(found) = links(node, &paragraph.text, &paragraph.source_spans, kind) {
                continuations.extend(found);
            }
        }
    }
    let mut notes = HashMap::<&str, &legal_structure_model::Note>::new();
    for entry in &structure.notes {
        notes.entry(entry.id.as_str()).or_insert(entry);
    }
    for note in footnotes {
        let node = notes.get(note.pair_id.as_str())
            .and_then(|entry| entry.node_id.as_deref()).and_then(|id| nodes.get(id));
        if let Some(node) = node {
            if let Some(found) = links(node, &note.body, &note.source_spans,
                legal_structure_model::ContinuationKind::Note) {
                continuations.extend(found);
            }
        }
    }
    structure.continuations = Some(continuations);
}

pub(crate) struct RenderedSourceLine<'a> {
    pub(crate) source: ScalarRange,
    pub(crate) target: ScalarRange,
    pub(crate) text: &'a str,
    pub(crate) edits: &'a [(usize, usize, String)],
    pub(crate) trimmed_start: usize,
    pub(crate) trimmed_end: usize,
}

impl RenderedSourceLine<'_> {
    fn boundary(&self, source_offset: usize, end: bool) -> legal_pdf_core::Result<usize> {
        let original = ScalarText::new(self.text);
        let local = source_offset - self.source.start;
        let scalar = original.scalar_at_utf16(local).ok_or_else(||
            legal_pdf_core::Error::Message("Manuscript membership splits a source character".into()))?;
        let mut removed = 0;
        let mut inserted = 0;
        for (start, stop, replacement) in self.edits {
            if *start < scalar && scalar < *stop {
                return Err(legal_pdf_core::Error::Message(
                    "Manuscript membership splits a normalization edit".into()));
            }
            // Boundary insertions belong to the following member, except at the line end.
            if *stop < scalar || *stop == scalar && (*start < *stop || end && local == utf16_len(self.text)) {
                removed += original.utf16_at_scalar(*stop).unwrap() - original.utf16_at_scalar(*start).unwrap();
                inserted += utf16_len(replacement);
            }
        }
        let position = local - removed + inserted;
        Ok(self.target.start + position.clamp(self.trimmed_start, self.trimmed_end) - self.trimmed_start)
    }
}

pub(crate) fn manuscript_line(text: &str, edits: &[(usize, usize, String)]) -> legal_pdf_core::Result<String> {
    let source = ScalarText::new(text);
    let mut output = String::new();
    let mut cursor = 0;
    let mut prior_start = None;
    for (start, end, replacement) in edits {
        if start > end || *start < cursor || prior_start.is_some_and(|prior| *start <= prior) {
            return Err(legal_pdf_core::Error::Message("Invalid manuscript edit order".into()));
        }
        let unchanged = source.slice(cursor..*start).ok_or_else(||
            legal_pdf_core::Error::Message("Manuscript edit starts outside its source line".into()))?;
        if source.utf16_at_scalar(*end).is_none() {
            return Err(legal_pdf_core::Error::Message("Manuscript edit ends outside its source line".into()));
        }
        output.push_str(unchanged);
        output.push_str(replacement);
        cursor = *end;
        prior_start = Some(*start);
    }
    output.push_str(source.slice_utf16(source.utf16_at_scalar(cursor).unwrap()..source.utf16_len()).unwrap());
    Ok(output)
}

pub(crate) fn attach_rendered_text(
    paragraphs: &[Paragraph],
    pages: &[Page],
    footnotes: &[Footnote],
    structure: &mut DocumentStructure,
) -> legal_pdf_core::Result<()> {
    let source = ScalarText::new(&structure.text);
    let fragments: HashMap<_, _> = structure.source_fragments.iter()
        .map(|fragment| (fragment.id.as_str(), fragment)).collect();
    let mut text = String::with_capacity(structure.text.len());
    let mut mappings = Vec::new();
    let mut owners = HashMap::new();
    for paragraph in paragraphs {
        for line in &paragraph.line_ids {
            owners.insert(line.as_str(), paragraph.id.as_str());
        }
    }
    let mut note_lines = HashSet::new();
    for note in footnotes {
        for line in &note.body_line_ids {
            owners.insert(line.as_str(), note.pair_id.as_str());
            note_lines.insert(line.as_str());
        }
    }
    // Each join's separator by the fragments it joins: for a pair named twice, the first join.
    let mut separators = HashMap::new();
    for join in &structure.joins {
        separators.entry((join.from, join.to)).or_insert(join.separator.as_str());
    }
    let mut position = 0;
    let mut previous_source_start = None;
    for page in pages {
        let mut lines: Vec<_> = page.lines.iter().collect();
        lines.sort_by_key(|line| (line.reading_order, line.source_index));
        let mut previous_owner = None;
        for notes in [false, true] {
            if notes { previous_owner = None; }
            for line in lines.iter().filter(|line| {
                (note_lines.contains(line.id.as_str())
                    || matches!(line.region_type.as_str(), "footnote" | "endnote" | "note")) == notes
            }) {
                if line.text.trim().is_empty() && !structure.manuscript_edits.contains_key(&line.id) { continue; }
                let fragment = fragments.get(line.id.as_str()).ok_or_else(||
                    legal_pdf_core::Error::Message(format!("Missing PDF source fragment {}", line.id)))?;
                if fragment.page_index != page.index
                    || source.slice_utf16(fragment.range.start..fragment.range.end) != Some(line.text.as_str()) {
                    return Err(legal_pdf_core::Error::Message(format!("PDF source fragment {} no longer matches its extraction", line.id)));
                }
                let edits = structure.manuscript_edits.get(&line.id).map(Vec::as_slice).unwrap_or_default();
                let normalized = manuscript_line(&line.text, edits)?;
                let trimmed = normalized.trim();
                // An empty physical member owns no insertion, matching manuscript span projection.
                if line.text.is_empty() || trimmed.is_empty() { continue; }
                let owner = owners.get(line.id.as_str()).copied().unwrap_or(line.id.as_str());
                if !text.is_empty() {
                    let separator = previous_source_start
                        .and_then(|from| separators.get(&(from, fragment.range.start)).copied())
                        .unwrap_or(if previous_owner == Some(owner) { " " } else { "\n\n" });
                    append(&mut text, &mut position, separator);
                }
                let start = position;
                let leading = utf16_len(&normalized[..normalized.len() - normalized.trim_start().len()]);
                append(&mut text, &mut position, trimmed);
                mappings.push(RenderedSourceLine { source: fragment.range,
                    target: ScalarRange { start, end: position }, text: &line.text, edits,
                    trimmed_start: leading, trimmed_end: leading + utf16_len(trimmed) });
                previous_owner = Some(owner);
                previous_source_start = Some(fragment.range.start);
            }
        }
    }
    mappings.sort_by_key(|mapping| mapping.source.start);
    let rendered = ScalarText::new(&text);
    let project = |membership: &[ScalarRange]| -> legal_pdf_core::Result<Vec<ScalarRange>> {
        let mut ranges = Vec::<ScalarRange>::new();
        for range in membership {
            let first = mappings.partition_point(|mapping| mapping.source.end <= range.start);
            for mapping in mappings[first..].iter().take_while(|mapping| mapping.source.start < range.end) {
                let start = range.start.max(mapping.source.start);
                let end = range.end.min(mapping.source.end);
                if start < end {
                    let part = ScalarRange { start: mapping.boundary(start, false)?,
                        end: mapping.boundary(end, true)? };
                    if part.start < part.end { ranges.push(part); }
                }
            }
        }
        ranges.sort_by_key(|range| (range.start, range.end));
        let mut exact = Vec::<ScalarRange>::new();
        for range in ranges {
            if let Some(prior) = exact.last_mut().filter(|prior| range.start <= prior.end
                || rendered.slice_utf16(prior.end..range.start).is_some_and(|gap| gap.chars().all(char::is_whitespace))) {
                prior.end = prior.end.max(range.end);
            } else {
                exact.push(range);
            }
        }
        Ok(exact)
    };
    for node in &mut structure.nodes {
        let own = if node.ranges.is_empty() { std::slice::from_ref(&node.range) } else { &node.ranges };
        node.rendered_membership = project(own)?;
        let membership = if (matches!(node.kind, NodeKind::Heading | NodeKind::Section)
            || node.role == Some(legal_structure_model::PrimaryRole::Heading)) && !node.extent.is_empty() {
            node.extent.as_slice()
        } else { own };
        let exact = project(membership)?;
        // An empty mapping stays empty; a first-page or source-envelope fallback would invent membership.
        node.rendered_range = Some(exact.iter().copied().fold(None, extend_range)
            .unwrap_or(ScalarRange { start: 0, end: 0 }));
        node.rendered_ranges = exact;
    }
    structure.rendered_text = (text != structure.text).then_some(text);
    structure.refresh_revision()?;
    Ok(())
}
