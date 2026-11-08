//! The parse's text index: every line's place in the document's text, and the ranges of text
//! each set of lines covers.

use crate::line_numbers::line_labels;
use legal_pdf_core::model::{Line, Page};
use legal_structure_model::{ScalarRange, ScalarText};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedPdfLine {
    pub page_index: usize,
    pub line_id: String,
    pub range: ScalarRange,
}

#[derive(Debug, Clone)]
pub struct PdfTextIndex {
    pub text: String,
    pub lines: Vec<IndexedPdfLine>,
    pub line_slots: HashMap<String, usize>,
    pub page_ranges: HashMap<usize, ScalarRange>,
}

/// The range covering both ranges.
fn cover(range: ScalarRange, other: ScalarRange) -> ScalarRange {
    ScalarRange { start: range.start.min(other.start), end: range.end.max(other.end) }
}

impl PdfTextIndex {
    pub fn from_pages(pages: &[Page]) -> Self {
        let mut ordered_pages = pages.iter().collect::<Vec<_>>();
        ordered_pages.sort_by_key(|page| page.index);
        let mut index = Self { text: String::new(), lines: Vec::new(), line_slots: HashMap::new(), page_ranges: HashMap::new() };
        let mut cursor = 0;
        for page in ordered_pages {
            let mut ordered_lines = page.lines.iter().collect::<Vec<_>>();
            ordered_lines.sort_by(|left, right| left.reading_order.cmp(&right.reading_order).then_with(|| left.id.cmp(&right.id)));
            let (mut page_start, mut page_end) = (None, cursor);
            let mut previous: Option<&Line> = None;
            let line_numbers = line_labels(page).map(|line| line.id.as_str()).collect::<HashSet<_>>();
            for line in ordered_lines {
                if !index.lines.is_empty() {
                    // A number set on a line of its own with its paragraph beside it on the same row
                    // ("1." then "In this decision ...", a hanging "[35]" read apart from its text) reads
                    // as one printed line: the grammars see "1. In this decision ...", and the ranges
                    // still fall on their own source lines. A transcript's line number stays apart.
                    let beside = previous.is_some_and(|before| {
                        let label = before.text.trim();
                        !line_numbers.contains(before.id.as_str())
                            && crate::paragraphs::same_row(before, line)
                            && line.bbox[0] >= before.bbox[2] - 1.0
                            && !label.is_empty()
                            && (crate::paragraphs::marker_only_re().is_match(&before.text)
                                || (label.chars().count() <= 7 && !label.contains(' ') && legal_pdf_core::layout_analysis().numbering_marker(label, None).is_some()))
                    });
                    index.text.push(if beside { ' ' } else { '\n' });
                    cursor += 1;
                }
                previous = Some(line);
                let start = cursor;
                index.text.push_str(&line.text);
                cursor += line.text.chars().count();
                let range = ScalarRange { start, end: cursor };
                page_start.get_or_insert(start);
                page_end = range.end;
                index.line_slots.insert(line.id.clone(), index.lines.len());
                index.lines.push(IndexedPdfLine { page_index: page.index, line_id: line.id.clone(), range });
            }
            index.page_ranges.insert(page.index, ScalarRange { start: page_start.unwrap_or(cursor), end: page_end });
        }
        index
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn source_fragments(&self) -> Vec<legal_structure_model::SourceFragment> {
        let text = ScalarText::new(&self.text);
        let fragment = |line: &IndexedPdfLine| legal_structure_model::SourceFragment {
            id: line.line_id.clone(),
            page_index: line.page_index,
            range: ScalarRange { start: text.utf16(line.range.start), end: text.utf16(line.range.end) },
        };
        self.lines.iter().map(fragment).collect()
    }

    pub fn line(&self, line_id: &str) -> Option<&IndexedPdfLine> {
        self.line_slots.get(line_id).and_then(|slot| self.lines.get(*slot))
    }

    /// The places among the index's lines of the lines a range overlaps.
    pub fn overlapping(&self, range: ScalarRange) -> std::ops::Range<usize> {
        let start = self.lines.partition_point(|line| line.range.end <= range.start);
        let end = self.lines.partition_point(|line| line.range.start < range.end);
        start.min(end)..end
    }

    pub fn overlapping_lines(&self, range: ScalarRange) -> &[IndexedPdfLine] {
        &self.lines[self.overlapping(range)]
    }

    pub fn page_range(&self, page_index: usize) -> Option<ScalarRange> {
        self.page_ranges.get(&page_index).copied()
    }

    pub fn global_range(&self, line_id: &str, start: usize, end: usize) -> Option<ScalarRange> {
        let line = self.line(line_id)?;
        (start <= end && end <= line.range.end - line.range.start).then_some(ScalarRange { start: line.range.start + start, end: line.range.start + end })
    }

    pub fn line_ids(&self, range: ScalarRange) -> Vec<String> {
        self.overlapping_lines(range).iter().map(|line| line.line_id.clone()).collect()
    }

    pub fn range_for_line_ids<'a>(&self, line_ids: impl IntoIterator<Item = &'a String>) -> Option<ScalarRange> {
        line_ids.into_iter().filter_map(|line_id| self.line(line_id)).map(|line| line.range).reduce(cover)
    }

    pub fn page_indexes_for_line_ids<'a>(&self, line_ids: impl IntoIterator<Item = &'a String>) -> Vec<usize> {
        let mut pages = Vec::new();
        for page_index in line_ids.into_iter().filter_map(|line_id| self.line(line_id).map(|line| line.page_index)) {
            if !pages.contains(&page_index) {
                pages.push(page_index);
            }
        }
        pages
    }
}

pub fn source_membership(mut ranges: Vec<ScalarRange>) -> Vec<ScalarRange> {
    ranges.sort_by_key(|range| (range.start, range.end));
    let mut merged: Vec<ScalarRange> = Vec::new();
    for range in ranges {
        match merged.last_mut().filter(|last| range.start <= last.end) {
            Some(last) => last.end = last.end.max(range.end),
            None => merged.push(range),
        }
    }
    merged
}
