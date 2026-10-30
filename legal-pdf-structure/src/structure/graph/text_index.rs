//! The parse's text index: every line's place in the document's text, and the ranges of text
//! each set of lines covers.

use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct IndexedPdfLine {
    pub(crate) page_index: usize,
    pub(crate) line_id: String,
    pub(crate) range: ScalarRange,
}

#[derive(Debug, Clone)]
pub struct PdfTextIndex {
    pub(crate) text: String,
    pub(crate) lines: Vec<IndexedPdfLine>,
    pub(crate) line_slots: HashMap<String, usize>,
    pub(crate) page_ranges: HashMap<usize, ScalarRange>,
}

impl PdfTextIndex {
    pub fn from_pages(pages: &[Page]) -> Self {
        let mut ordered_pages = pages.iter().collect::<Vec<_>>();
        ordered_pages.sort_by_key(|page| page.index);
        let mut text = String::new();
        let mut lines = Vec::new();
        let mut line_slots = HashMap::new();
        let mut page_ranges = HashMap::new();
        let mut scalar_cursor = 0;
        for page in ordered_pages {
            let mut ordered_lines = page.lines.iter().collect::<Vec<_>>();
            ordered_lines.sort_by(|left, right| {
                left.reading_order
                    .cmp(&right.reading_order)
                    .then_with(|| left.id.cmp(&right.id))
            });
            let mut page_start = None;
            let mut page_end = scalar_cursor;
            let mut previous: Option<&Line> = None;
            let line_numbers = line_labels(page).map(|line| line.id.as_str()).collect::<HashSet<_>>();
            for line in ordered_lines {
                if !lines.is_empty() {
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
                                || (label.chars().count() <= 7 && !label.contains(' ')
                                    && legal_pdf_core::layout_analysis().numbering_marker(label, None).is_some()))
                    });
                    text.push(if beside { ' ' } else { '\n' });
                    scalar_cursor += 1;
                }
                previous = Some(line);
                let start = scalar_cursor;
                text.push_str(&line.text);
                scalar_cursor += line.text.chars().count();
                let range = ScalarRange {
                    start,
                    end: scalar_cursor,
                };
                page_start.get_or_insert(start);
                page_end = range.end;
                let slot = lines.len();
                line_slots.insert(line.id.clone(), slot);
                lines.push(IndexedPdfLine {
                    page_index: page.index,
                    line_id: line.id.clone(),
                    range,
                });
            }
            page_ranges.insert(
                page.index,
                ScalarRange {
                    start: page_start.unwrap_or(scalar_cursor),
                    end: page_end,
                },
            );
        }
        Self {
            text,
            lines,
            line_slots,
            page_ranges,
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub(crate) fn source_fragments(&self) -> Vec<legal_structure_model::SourceFragment> {
        let text = ScalarText::new(&self.text);
        self.lines.iter().map(|line| legal_structure_model::SourceFragment {
            id: line.line_id.clone(),
            page_index: line.page_index,
            range: ScalarRange { start: text.utf16(line.range.start), end: text.utf16(line.range.end) },
        }).collect()
    }

    pub(crate) fn line(&self, line_id: &str) -> Option<&IndexedPdfLine> {
        self.line_slots
            .get(line_id)
            .and_then(|slot| self.lines.get(*slot))
    }

    /// The place among the index's lines of the line holding the scalar `at`.
    pub(crate) fn position_at(&self, at: usize) -> Option<usize> {
        let slot = self
            .lines
            .partition_point(|line| line.range.start <= at)
            .checked_sub(1)?;
        self.lines.get(slot).is_some_and(|line| {
            (line.range.start <= at && at < line.range.end)
                || (line.range.start == line.range.end && line.range.start == at)
        })
        .then_some(slot)
    }

    /// The places among the index's lines of the lines a range overlaps.
    pub(crate) fn overlapping(&self, range: ScalarRange) -> std::ops::Range<usize> {
        let start = self
            .lines
            .partition_point(|line| line.range.end <= range.start);
        let end = self
            .lines
            .partition_point(|line| line.range.start < range.end);
        start.min(end)..end
    }

    pub(crate) fn overlapping_lines(&self, range: ScalarRange) -> &[IndexedPdfLine] {
        &self.lines[self.overlapping(range)]
    }

    pub(crate) fn page_range(&self, page_index: usize) -> Option<ScalarRange> {
        self.page_ranges.get(&page_index).copied()
    }

    pub(crate) fn global_range(
        &self,
        line_id: &str,
        start: usize,
        end: usize,
    ) -> Option<ScalarRange> {
        let line = self.line(line_id)?;
        let length = line.range.end - line.range.start;
        (start <= end && end <= length).then_some(ScalarRange {
            start: line.range.start + start,
            end: line.range.start + end,
        })
    }

    pub(crate) fn line_ids(&self, range: ScalarRange) -> Vec<String> {
        self.overlapping_lines(range)
            .iter()
            .map(|line| line.line_id.clone())
            .collect()
    }

    pub(crate) fn range_for_line_ids<'a>(
        &self,
        line_ids: impl IntoIterator<Item = &'a String>,
    ) -> Option<ScalarRange> {
        line_ids
            .into_iter()
            .filter_map(|line_id| self.line(line_id))
            .fold(None, |range, line| {
                Some(range.map_or(line.range, |range: ScalarRange| ScalarRange {
                    start: range.start.min(line.range.start),
                    end: range.end.max(line.range.end),
                }))
            })
    }

    pub(crate) fn page_indexes_for_line_ids<'a>(
        &self,
        line_ids: impl IntoIterator<Item = &'a String>,
    ) -> Vec<usize> {
        line_ids.into_iter().fold(Vec::new(), |mut pages, line_id| {
            if let Some(page_index) = self.line(line_id).map(|line| line.page_index) {
                if !pages.contains(&page_index) {
                    pages.push(page_index);
                }
            }
            pages
        })
    }
}

pub(crate) fn source_membership(mut ranges: Vec<ScalarRange>) -> Vec<ScalarRange> {
    ranges.sort_by_key(|range| (range.start, range.end));
    let mut merged: Vec<ScalarRange> = Vec::new();
    for range in ranges {
        if let Some(last) = merged.last_mut().filter(|last| range.start <= last.end) {
            last.end = last.end.max(range.end);
        } else {
            merged.push(range);
        }
    }
    merged
}
