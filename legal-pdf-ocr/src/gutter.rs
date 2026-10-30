//! Lines the segmenter runs across a page's gutter. Columns set side by side at one leading (the
//! bilingual reports' English and French) can stand a mark in their gutter level with a row of
//! both, as the reports print a letter every few lines to locate them; the segmenter then reads
//! the gaps on either side of the mark, each no wider than a justified line's word space, as word
//! spaces, and takes the row of both columns for one line. Such a line is read as the lines it
//! holds: its left column's, the mark in the gutter, and its right column's.

use crate::indra::LineBox;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    Left,
    Right,
    /// Words on both sides of the gutter and none in it.
    Both,
    /// A word in the gutter.
    Across,
}

/// The page's lines, each line that runs across a gutter replaced in its place by its left
/// column's line, the mark in the gutter and its right column's line.
///
/// A line's words are its marks wider than three fifths of the page's line height; a narrower
/// mark is a letter or two, a stop or a speck, such as the reports' gutter letters, and fills no
/// gutter. A gutter is a band at least a line high wide, away from the page's edges, that hardly
/// any line's words cover (at most 15% of the lines covering the page's most covered column, as
/// pdf-inspector's column projection counts its empty valleys), between columns of eight lines or
/// more standing wholly on each side of its middle half, the lines that reach it on each side at
/// least eight line heights wide (a column of text, not of labels set before a hanging indent).
/// A line runs across the gutter when it has words on both sides of the gutter's middle half and
/// none in it, stands below lines of both columns (a running head or a title over the columns
/// stays whole), has lines of both columns within three lines of it, and at most half the lines
/// within three lines of it hold a word in the gutter (a table's rows across the page stay whole).
pub(crate) fn split_across_gutters(
    pixels: &[u8],
    width: usize,
    lines: Vec<LineBox>,
) -> Vec<LineBox> {
    if lines.len() < 16 {
        return lines;
    }
    let mut heights = lines.iter().map(|line| line.height()).collect::<Vec<_>>();
    heights.sort_unstable();
    let line_height = heights[heights.len() / 2].max(1);
    let marks = lines
        .iter()
        .map(|line| marks(pixels, width, *line, line_height))
        .collect::<Vec<_>>();
    let is_word = |mark: &&(usize, usize)| (mark.1 - mark.0) * 5 > line_height * 3;
    let mut cover = vec![0_usize; width];
    for mark in marks.iter().flatten().filter(is_word) {
        cover[mark.0..mark.1]
            .iter_mut()
            .for_each(|count| *count += 1);
    }
    let peak = cover.iter().copied().max().unwrap_or(0);
    let middle = |line: &LineBox| (line.top + line.bottom) / 2;
    let mut pieces = vec![None::<Vec<LineBox>>; lines.len()];
    let mut end = 0;
    while end < width {
        let start = end;
        while end < width && cover[end] * 100 <= peak * 15 {
            end += 1;
        }
        if end == start {
            end += 1;
            continue;
        }
        if start == 0 || end == width || end - start < line_height {
            continue;
        }
        let quarter = (end - start) / 4;
        let (inner_left, inner_right) = (start + quarter, end - quarter);
        let sides = marks
            .iter()
            .map(|marks| {
                let mut words = marks.iter().filter(is_word);
                if words
                    .clone()
                    .any(|mark| mark.0 < inner_right && mark.1 > inner_left)
                {
                    return Some(Side::Across);
                }
                match (
                    words.clone().any(|mark| mark.1 <= inner_left),
                    words.any(|mark| mark.0 >= inner_right),
                ) {
                    (true, true) => Some(Side::Both),
                    (true, false) => Some(Side::Left),
                    (false, true) => Some(Side::Right),
                    (false, false) => None,
                }
            })
            .collect::<Vec<_>>();
        let count = |side: Side| sides.iter().filter(|other| **other == Some(side)).count();
        if count(Side::Left) < 8 || count(Side::Right) < 8 {
            continue;
        }
        // Where the lines reaching the gutter begin (on its left) and end (on its right).
        let mut begins = Vec::new();
        let mut ends = Vec::new();
        for (marks, side) in marks.iter().zip(&sides) {
            let mut words = marks.iter().filter(is_word);
            match side {
                Some(Side::Left) if words.any(|mark| mark.1 + line_height >= start) => {
                    begins.push(marks[0].0);
                }
                Some(Side::Right) if words.any(|mark| mark.0 <= end + line_height) => {
                    ends.push(marks[marks.len() - 1].1);
                }
                _ => {}
            }
        }
        begins.sort_unstable();
        ends.sort_unstable();
        if begins.is_empty()
            || ends.is_empty()
            || start.saturating_sub(begins[begins.len() / 4]) < line_height * 8
            || ends[ends.len() * 3 / 4].saturating_sub(end) < line_height * 8
        {
            continue;
        }
        let reach = line_height * 3;
        for (index, line) in lines.iter().enumerate() {
            if sides[index] != Some(Side::Both) || pieces[index].is_some() {
                continue;
            }
            let y = middle(line);
            let near = lines
                .iter()
                .zip(&sides)
                .enumerate()
                .filter(|(other, (neighbour, _))| {
                    *other != index && middle(neighbour).abs_diff(y) <= reach
                })
                .map(|(_, (_, side))| *side)
                .collect::<Vec<_>>();
            let in_columns = [Side::Left, Side::Right].into_iter().all(|column| {
                lines.iter().zip(&sides).any(|(other, side)| {
                    *side == Some(column) && middle(other) + line_height / 2 < y
                }) && near.contains(&Some(column))
            });
            let across = near
                .iter()
                .filter(|side| **side == Some(Side::Across))
                .count();
            if !in_columns || across * 2 > near.len() {
                continue;
            }
            let held = &marks[index];
            let left = held
                .iter()
                .filter(|mark| mark.1 <= inner_left)
                .map(|mark| mark.1)
                .max();
            let right = held
                .iter()
                .filter(|mark| mark.0 >= inner_right)
                .map(|mark| mark.0)
                .min();
            let (Some(left), Some(right)) = (left, right) else {
                continue;
            };
            let mut split = vec![LineBox {
                right: left,
                ..*line
            }];
            let mut inner = held
                .iter()
                .filter(|mark| mark.0 < inner_right && mark.1 > inner_left);
            if let Some(first) = inner.next() {
                let last = inner.last().unwrap_or(first);
                split.push(LineBox {
                    left: first.0,
                    right: last.1,
                    ..*line
                });
            }
            split.push(LineBox {
                left: right,
                ..*line
            });
            pieces[index] = Some(split);
        }
    }
    lines
        .into_iter()
        .zip(pieces)
        .flat_map(|(line, pieces)| pieces.unwrap_or_else(|| vec![line]))
        .collect()
}

/// A line's marks, left to right: runs of columns with ink across the middle three fifths of its
/// height, a run going on over gaps narrower than a quarter of the page's line height (a letter's
/// spacing, not a word's).
fn marks(pixels: &[u8], width: usize, line: LineBox, line_height: usize) -> Vec<(usize, usize)> {
    let inset = line.height() / 5;
    let rows = line.top + inset..line.bottom - inset;
    let space = (line_height / 4).max(2);
    let mut marks: Vec<(usize, usize)> = Vec::new();
    for x in line.left..line.right {
        if rows.clone().any(|y| pixels[y * width + x] < 128) {
            match marks.last_mut() {
                Some(mark) if x - mark.1 < space => mark.1 = x + 1,
                _ => marks.push((x, x + 1)),
            }
        }
    }
    marks
}
