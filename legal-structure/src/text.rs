//! Coordinates over one unchanged Rust string.
//!
//! Bytes are UTF-8 boundaries, scalars are Rust `char`s, and UTF-16 offsets are
//! JavaScript code units. Exact conversions reject split characters; only the
//! named floor/ceil methods round. CR, LF, and CRLF are never normalized.

use std::ops::Range;

pub(crate) struct ScalarText<'a> {
    pub(crate) value: &'a str,
    checkpoints: Vec<[usize; 3]>,
    scalar_len: usize,
    utf16_len: usize,
}

impl<'a> ScalarText<'a> {
    pub(crate) fn new(value: &'a str) -> Self {
        if value.is_ascii() {
            return Self {
                value,
                checkpoints: Vec::new(),
                scalar_len: value.len(),
                utf16_len: value.len(),
            };
        }
        let mut checkpoints = Vec::new();
        let mut scalar_len = 0;
        let mut utf16_len = 0;
        for (scalar, (byte, character)) in value.char_indices().enumerate() {
            if scalar % 64 == 0 {
                checkpoints.push([scalar, byte, utf16_len]);
            }
            scalar_len = scalar + 1;
            utf16_len += character.len_utf16();
        }
        if checkpoints.last().is_none_or(|at| at[0] != scalar_len) {
            checkpoints.push([scalar_len, value.len(), utf16_len]);
        }
        Self {
            value,
            checkpoints,
            scalar_len,
            utf16_len,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.scalar_len
    }

    pub(crate) fn utf16_len(&self) -> usize {
        self.utf16_len
    }

    fn checkpoint(&self, target: usize, axis: usize) -> [usize; 3] {
        self.checkpoints[self.checkpoints.partition_point(|at| at[axis] <= target) - 1]
    }

    fn at_scalar(&self, scalar: usize) -> Option<[usize; 3]> {
        if scalar > self.scalar_len {
            return None;
        }
        if self.checkpoints.is_empty() {
            return Some([scalar, scalar, scalar]);
        }
        let mut at = self.checkpoint(scalar, 0);
        for character in self.value[at[1]..].chars().take(scalar - at[0]) {
            at[0] += 1;
            at[1] += character.len_utf8();
            at[2] += character.len_utf16();
        }
        Some(at)
    }

    fn at_byte(&self, byte: usize) -> Option<[usize; 3]> {
        if byte > self.value.len() || !self.value.is_char_boundary(byte) {
            return None;
        }
        if self.checkpoints.is_empty() {
            return Some([byte, byte, byte]);
        }
        let mut at = self.checkpoint(byte, 1);
        for character in self.value[at[1]..byte].chars() {
            at[0] += 1;
            at[1] += character.len_utf8();
            at[2] += character.len_utf16();
        }
        Some(at)
    }

    fn at_utf16(&self, utf16: usize) -> Option<([usize; 3], [usize; 3])> {
        if utf16 > self.utf16_len {
            return None;
        }
        if self.checkpoints.is_empty() {
            return Some(([utf16; 3], [utf16; 3]));
        }
        let mut floor = self.checkpoint(utf16, 2);
        if floor[2] == utf16 {
            return Some((floor, floor));
        }
        for character in self.value[floor[1]..].chars() {
            let ceil = [
                floor[0] + 1,
                floor[1] + character.len_utf8(),
                floor[2] + character.len_utf16(),
            ];
            if utf16 < ceil[2] {
                return Some((floor, ceil));
            }
            if utf16 == ceil[2] {
                return Some((ceil, ceil));
            }
            floor = ceil;
        }
        None
    }

    pub(crate) fn scalar_at_byte(&self, byte: usize) -> Option<usize> {
        self.at_byte(byte).map(|at| at[0])
    }

    pub(crate) fn scalar(&self, byte: usize) -> usize {
        self.scalar_at_byte(byte).unwrap()
    }

    pub(crate) fn byte_at_scalar(&self, scalar: usize) -> Option<usize> {
        self.at_scalar(scalar).map(|at| at[1])
    }

    pub(crate) fn byte(&self, scalar: usize) -> usize {
        self.byte_at_scalar(scalar).unwrap()
    }

    pub(crate) fn utf16_at_scalar(&self, scalar: usize) -> Option<usize> {
        self.at_scalar(scalar).map(|at| at[2])
    }

    pub(crate) fn utf16(&self, scalar: usize) -> usize {
        self.utf16_at_scalar(scalar).unwrap()
    }

    pub(crate) fn scalar_at_utf16(&self, utf16: usize) -> Option<usize> {
        self.at_utf16(utf16)
            .and_then(|(floor, ceil)| (floor[0] == ceil[0]).then_some(floor[0]))
    }

    pub(crate) fn utf16_at_byte(&self, byte: usize) -> Option<usize> {
        self.at_byte(byte).map(|at| at[2])
    }

    pub(crate) fn byte_at_utf16(&self, utf16: usize) -> Option<usize> {
        self.at_utf16(utf16)
            .and_then(|(floor, ceil)| (floor[1] == ceil[1]).then_some(floor[1]))
    }

    pub(crate) fn byte_at_utf16_floor(&self, utf16: usize) -> Option<usize> {
        self.at_utf16(utf16).map(|(floor, _)| floor[1])
    }

    pub(crate) fn byte_at_utf16_ceil(&self, utf16: usize) -> Option<usize> {
        self.at_utf16(utf16).map(|(_, ceil)| ceil[1])
    }

    pub(crate) fn slice(&self, range: Range<usize>) -> Option<&'a str> {
        self.value
            .get(self.byte_at_scalar(range.start)?..self.byte_at_scalar(range.end)?)
    }
}

pub(crate) fn utf16_len(value: &str) -> usize {
    value.encode_utf16().count()
}

/// The code points matched by ECMAScript `\s`: Unicode WhiteSpace plus line
/// terminators and BOM, deliberately excluding U+0085.
pub(crate) fn javascript_whitespace(character: char) -> bool {
    matches!(
        character,
        '\u{0009}'..='\u{000d}'
            | '\u{0020}'
            | '\u{00a0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200a}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202f}'
            | '\u{205f}'
            | '\u{3000}'
            | '\u{feff}'
    )
}

/// Collapse ECMAScript whitespace runs to one ASCII space and trim runs at
/// both ends. Non-whitespace code points, including U+0085, are unchanged.
pub(crate) fn normalize_javascript_whitespace(value: &str) -> String {
    let mut normalized = String::with_capacity(value.len());
    let mut separating = false;
    for character in value.chars() {
        if javascript_whitespace(character) {
            separating = !normalized.is_empty();
        } else {
            if separating {
                normalized.push(' ');
            }
            normalized.push(character);
            separating = false;
        }
    }
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coordinates_round_trip_empty_combining_and_non_bmp_text() {
        let empty = ScalarText::new("");
        assert_eq!(empty.byte_at_scalar(0), Some(0));
        assert_eq!(empty.scalar_at_byte(0), Some(0));
        assert_eq!(empty.byte_at_utf16(0), Some(0));
        assert_eq!(empty.scalar_at_utf16(0), Some(0));
        assert_eq!(empty.slice(0..0), Some(""));

        let value = "A\u{301}\u{1f9ab}\u{6587}";
        let text = ScalarText::new(value);
        let mut utf16 = 0;
        for (scalar, (byte, character)) in value.char_indices().enumerate() {
            assert_eq!(text.byte_at_scalar(scalar), Some(byte));
            assert_eq!(text.scalar_at_byte(byte), Some(scalar));
            assert_eq!(text.utf16_at_scalar(scalar), Some(utf16));
            assert_eq!(text.utf16_at_byte(byte), Some(utf16));
            assert_eq!(text.scalar_at_utf16(utf16), Some(scalar));
            assert_eq!(text.byte_at_utf16(utf16), Some(byte));
            utf16 += character.len_utf16();
        }
        assert_eq!(text.byte_at_scalar(text.len()), Some(value.len()));
        assert_eq!(text.scalar_at_byte(value.len()), Some(text.len()));
        assert_eq!(text.utf16_at_scalar(text.len()), Some(utf16));
        assert_eq!(text.byte_at_utf16(utf16), Some(value.len()));
        assert_eq!(text.scalar_at_utf16(utf16), Some(text.len()));
    }

    #[test]
    fn exact_coordinates_and_slices_reject_non_boundaries() {
        let text = ScalarText::new("a\u{1f9ab}b");
        assert_eq!(text.scalar_at_byte(2), None);
        assert_eq!(text.utf16_at_byte(2), None);
        assert_eq!(text.byte_at_scalar(4), None);
        assert_eq!(text.byte_at_utf16(2), None);
        assert_eq!(text.scalar_at_utf16(2), None);
        assert_eq!(text.byte_at_utf16_floor(2), Some(1));
        assert_eq!(text.byte_at_utf16_ceil(2), Some(5));
        assert_eq!(text.byte_at_utf16(1), Some(1));
        assert_eq!(text.byte_at_utf16(3), Some(5));
        assert_eq!(text.byte_at_utf16(4), Some(6));
        assert_eq!(text.slice(1..2), Some("\u{1f9ab}"));
        assert_eq!(text.slice(2..1), None);
        assert_eq!(text.slice(0..4), None);
        assert_eq!(text.byte_at_utf16_floor(5), None);
        assert_eq!(text.byte_at_utf16_ceil(5), None);
    }

    #[test]
    fn cr_lf_and_crlf_keep_original_coordinates() {
        let value = "a\rb\r\nc\nd";
        let text = ScalarText::new(value);
        assert_eq!(text.len(), 8);
        assert_eq!(text.utf16_len(), 8);
        assert_eq!(text.utf16_at_byte(value.find('\n').unwrap()), Some(4));
        assert_eq!(text.slice(1..5), Some("\rb\r\n"));
    }

    #[test]
    fn javascript_whitespace_matches_ecmascript_exactly() {
        let whitespace = [
            '\u{0009}', '\u{000a}', '\u{000b}', '\u{000c}', '\u{000d}', '\u{0020}', '\u{00a0}',
            '\u{1680}', '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}',
            '\u{2006}', '\u{2007}', '\u{2008}', '\u{2009}', '\u{200a}', '\u{2028}', '\u{2029}',
            '\u{202f}', '\u{205f}', '\u{3000}', '\u{feff}',
        ];
        assert!(whitespace.into_iter().all(javascript_whitespace));
        assert!(!javascript_whitespace('\u{0085}'));
        assert!(!javascript_whitespace('\u{180e}'));
    }

    #[test]
    fn javascript_whitespace_normalization_preserves_non_whitespace() {
        assert_eq!(
            normalize_javascript_whitespace("\r\n A\u{feff}\tB\u{0085}\n"),
            "A B\u{0085}"
        );
        assert_eq!(normalize_javascript_whitespace("\r"), "");
        assert_eq!(normalize_javascript_whitespace("\n"), "");
        assert_eq!(normalize_javascript_whitespace("\r\n"), "");
    }
}
