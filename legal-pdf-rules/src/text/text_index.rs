//! The parse's text index: every line's place in the document's text, and the ranges of text
//! each set of lines covers, in UTF-16 code units.

use legal_pdf_core::model::{Line, Page};
use legal_structure_model::{utf16_len, ScalarRange, SourceFragment};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PdfTextIndex {
    pub text: String,
    /// Each line's place in the text, in the text's order: the structure's source fragments.
    pub lines: Vec<SourceFragment>,
    pub line_slots: HashMap<String, usize>,
    pub page_ranges: HashMap<usize, ScalarRange>,
}

/// The range covering both ranges.
fn cover(range: ScalarRange, other: ScalarRange) -> ScalarRange {
    ScalarRange {
        start: range.start.min(other.start),
        end: range.end.max(other.end),
    }
}

impl PdfTextIndex {
    /// The index of pages as the structure stage leaves them: in order, each page's lines in reading order; a line
    /// joins the one before it on its printed line where it stands `beside` it.
    pub fn from_pages(pages: &[Page], beside: impl Fn(&Line, &Line) -> bool) -> Self {
        let mut index = Self {
            text: String::new(),
            lines: Vec::new(),
            line_slots: HashMap::new(),
            page_ranges: HashMap::new(),
        };
        let mut cursor = 0;
        for page in pages {
            let (mut page_start, mut page_end) = (None, cursor);
            let mut previous: Option<&Line> = None;
            // A page holds its lines as extracted; the decision writes each one's place in the reading order.
            let mut lines: Vec<&Line> = page.lines.iter().collect();
            lines.sort_by_key(|line| line.reading_order);
            for line in lines {
                if !index.lines.is_empty() {
                    // A number set on a line of its own with its paragraph beside it on the same row (`beside`: "1." then
                    // "In this decision ...", a hanging "[35]" read apart from its text) reads as one printed line: the grammars
                    // see "1. In this decision ...", and the ranges still fall on their own source lines.
                    index
                        .text
                        .push(if previous.is_some_and(|before| beside(before, line)) {
                            ' '
                        } else {
                            '\n'
                        });
                    cursor += 1;
                }
                previous = Some(line);
                let start = cursor;
                index.text.push_str(&line.text);
                cursor += utf16_len(&line.text);
                let range = ScalarRange { start, end: cursor };
                page_start.get_or_insert(start);
                page_end = range.end;
                index.line_slots.insert(line.id.clone(), index.lines.len());
                index.lines.push(SourceFragment {
                    id: line.id.clone(),
                    page_index: page.index,
                    range,
                });
            }
            index.page_ranges.insert(
                page.index,
                ScalarRange {
                    start: page_start.unwrap_or(cursor),
                    end: page_end,
                },
            );
        }
        index
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn line(&self, line_id: &str) -> Option<&SourceFragment> {
        self.line_slots
            .get(line_id)
            .and_then(|slot| self.lines.get(*slot))
    }

    /// The places among the index's lines of the lines a range overlaps.
    pub fn overlapping(&self, range: ScalarRange) -> std::ops::Range<usize> {
        let start = self
            .lines
            .partition_point(|line| line.range.end <= range.start);
        let end = self
            .lines
            .partition_point(|line| line.range.start < range.end);
        start.min(end)..end
    }

    pub fn overlapping_lines(&self, range: ScalarRange) -> &[SourceFragment] {
        &self.lines[self.overlapping(range)]
    }

    pub fn page_range(&self, page_index: usize) -> Option<ScalarRange> {
        self.page_ranges.get(&page_index).copied()
    }

    pub fn global_range(
        &self,
        line_id: &str,
        start: usize,
        end: usize,
    ) -> Option<ScalarRange> {
        let line = self.line(line_id)?;
        (start <= end && end <= line.range.end - line.range.start).then_some(ScalarRange {
            start: line.range.start + start,
            end: line.range.start + end,
        })
    }

    pub fn line_ids(&self, range: ScalarRange) -> Vec<String> {
        self.overlapping_lines(range)
            .iter()
            .map(|line| line.id.clone())
            .collect()
    }

    pub fn range_for_line_ids<'a>(
        &self,
        line_ids: impl IntoIterator<Item = &'a String>,
    ) -> Option<ScalarRange> {
        line_ids
            .into_iter()
            .filter_map(|line_id| self.line(line_id))
            .map(|line| line.range)
            .reduce(cover)
    }

    pub fn page_indexes_for_line_ids<'a>(
        &self,
        line_ids: impl IntoIterator<Item = &'a String>,
    ) -> Vec<usize> {
        let mut pages = Vec::new();
        for page_index in line_ids
            .into_iter()
            .filter_map(|line_id| self.line(line_id).map(|line| line.page_index))
        {
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
