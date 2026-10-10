//! Lines Tesseract's page analysis leaves out: a paragraph's number alone in its margin, a page
//! number, a stamp's word, taken for noise beside the page's text. Ink no line holds, of a line's
//! height, is read as a line of its own; the recognizer keeps one only when it is sure of it.

use crate::kraken::LineBox;

/// The page's lines with the uncovered lines added, each placed before the line it stands beside
/// or after the line above it.
pub(crate) fn recover_uncovered_lines(
    pixels: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    mut boxes: Vec<LineBox>,
) -> Vec<LineBox> {
    if boxes.len() < 3 || width == 0 || height == 0 {
        return boxes;
    }
    let mut heights = boxes.iter().map(|line| line.height()).collect::<Vec<_>>();
    heights.sort_unstable();
    let line_height = heights[heights.len() / 2].max(1);
    let ink = |x: usize, y: usize| {
        let at = (y * width + x) * channels;
        let luminance = if channels >= 3 {
            (u32::from(pixels[at]) * 3 + u32::from(pixels[at + 1]) * 6 + u32::from(pixels[at + 2]))
                / 10
        } else {
            u32::from(pixels[at])
        };
        luminance < 128
    };
    // Ink within a quarter line of a line is that line's.
    let margin = line_height / 4;
    let mut seen = vec![false; width * height];
    for line in &boxes {
        for y in line.top.saturating_sub(margin)..(line.bottom + margin).min(height) {
            seen[y * width + line.left.saturating_sub(margin)
                ..y * width + (line.right + margin).min(width)]
                .fill(true);
        }
    }
    // Uncovered ink, piece by piece: (left, top, right, bottom, pixels).
    let mut pieces = Vec::<[usize; 5]>::new();
    let mut stack = Vec::new();
    for start in 0..width * height {
        if seen[start] || !ink(start % width, start / width) {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        let mut piece = [usize::MAX, usize::MAX, 0, 0, 0];
        while let Some(at) = stack.pop() {
            let (x, y) = (at % width, at / width);
            piece = [
                piece[0].min(x),
                piece[1].min(y),
                piece[2].max(x + 1),
                piece[3].max(y + 1),
                piece[4] + 1,
            ];
            for ny in y.saturating_sub(1)..(y + 2).min(height) {
                for nx in x.saturating_sub(1)..(x + 2).min(width) {
                    let next = ny * width + nx;
                    if !seen[next] && ink(nx, ny) {
                        seen[next] = true;
                        stack.push(next);
                    }
                }
            }
        }
        // A speck, a stop or a rule is no character; a character is at most a line high.
        let tall = piece[3] - piece[1];
        if tall * 10 >= line_height * 3 && tall * 10 <= line_height * 16 {
            pieces.push(piece);
        }
        if pieces.len() > 2000 {
            return boxes;
        }
    }
    // Characters on one row within a justified line's widest word space of each other are one
    // line.
    pieces.sort_by_key(|piece| piece[0]);
    let mut groups: Vec<[usize; 5]> = Vec::new();
    for piece in pieces {
        let joins = |group: &[usize; 5]| {
            let overlap = group[3]
                .min(piece[3])
                .saturating_sub(group[1].max(piece[1]));
            overlap * 2 >= (group[3] - group[1]).min(piece[3] - piece[1])
                && piece[0] <= group[2] + line_height * 3 / 2
        };
        match groups.iter_mut().rev().find(|group| joins(group)) {
            Some(group) => {
                *group = [
                    group[0].min(piece[0]),
                    group[1].min(piece[1]),
                    group[2].max(piece[2]),
                    group[3].max(piece[3]),
                    group[4] + piece[4],
                ];
            }
            None => groups.push(piece),
        }
    }
    for [left, top, right, bottom, count] in groups {
        let (wide, tall) = (right - left, bottom - top);
        let area = wide * tall;
        // Of a line's height, wide enough to read, neither a solid blot nor at the image's edge.
        if tall * 2 < line_height
            || tall * 10 > line_height * 16
            || wide * 10 < line_height * 3
            || count * 20 < area
            || count * 10 > area * 6
            || left <= 2
            || top <= 2
            || right + 2 >= width
            || bottom + 2 >= height
        {
            continue;
        }
        let found = LineBox {
            left,
            top,
            right,
            bottom,
            recovered: true,
        };
        let middle = (top + bottom) / 2;
        let beside = boxes
            .iter()
            .enumerate()
            .filter(|(_, line)| line.top <= middle && middle <= line.bottom && line.left >= right)
            .min_by_key(|(_, line)| line.left)
            .map(|(index, _)| index);
        let below = || boxes.iter().position(|line| line.top > middle);
        let at = beside.or_else(below).unwrap_or(boxes.len());
        boxes.insert(at, found);
    }
    boxes
}
