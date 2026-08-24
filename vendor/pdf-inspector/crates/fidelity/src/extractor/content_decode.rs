//! Bounded content-stream decoding.
//!
//! `lopdf::content::Content::decode` materializes every operator before any
//! caller can apply a limit. Count operators first and skip decode when the
//! cap is exceeded.

use crate::PdfError;
use lopdf::content::Content;

pub(crate) const MAX_PAGE_OPERATIONS: usize = 1_000_000;

pub(crate) fn decode_content_bounded(
    data: &[u8],
    max_operations: usize,
) -> Result<Option<Content>, PdfError> {
    if count_content_operators(data, max_operations.saturating_add(1)) > max_operations {
        return Ok(None);
    }
    Content::decode(data)
        .map(Some)
        .map_err(|error| PdfError::Parse(error.to_string()))
}

fn count_content_operators(data: &[u8], limit: usize) -> usize {
    let mut i = 0;
    let mut count = 0;
    while i < data.len() && count < limit {
        skip_content_space(data, &mut i);
        if i >= data.len() {
            break;
        }
        if data[i] == b'%' {
            skip_comment(data, &mut i);
            continue;
        }
        match data[i] {
            b'(' => i = skip_literal_string(data, i),
            b'<' => {
                if data.get(i + 1) == Some(&b'<') {
                    i += 2;
                } else {
                    i = skip_hex_string(data, i);
                }
            }
            b'>' => {
                i += 1;
                if data.get(i) == Some(&b'>') {
                    i += 1;
                }
            }
            b'[' | b']' => i += 1,
            b'/' => skip_name(data, &mut i),
            b'+' | b'-' | b'.' => skip_number(data, &mut i),
            byte if byte.is_ascii_digit() => skip_number(data, &mut i),
            byte if is_operator_byte(byte) => {
                let start = i;
                i += 1;
                while i < data.len() && is_operator_byte(data[i]) {
                    i += 1;
                }
                let token = &data[start..i];
                if token == b"true" || token == b"false" || token == b"null" {
                    continue;
                }
                count += 1;
                if token == b"BI" && (i >= data.len() || is_content_space(data[i])) {
                    i = skip_inline_image_after_bi(data, i);
                }
            }
            _ => i += 1,
        }
    }
    count
}

fn is_content_space(byte: u8) -> bool {
    matches!(byte, b'\0' | b'\t' | b'\n' | b'\x0c' | b'\r' | b' ')
}

fn is_operator_byte(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || matches!(byte, b'*' | b'\'' | b'"')
}

fn is_delimiter(byte: u8) -> bool {
    matches!(
        byte,
        b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%'
    )
}

fn skip_content_space(data: &[u8], i: &mut usize) {
    while *i < data.len() && is_content_space(data[*i]) {
        *i += 1;
    }
}

fn skip_comment(data: &[u8], i: &mut usize) {
    while *i < data.len() && data[*i] != b'\n' && data[*i] != b'\r' {
        *i += 1;
    }
}

fn skip_literal_string(data: &[u8], mut i: usize) -> usize {
    let mut depth = 1i32;
    i += 1;
    while i < data.len() && depth > 0 {
        match data[i] {
            b'\\' => {
                i += 1;
                if i < data.len() {
                    i += 1;
                }
            }
            b'(' => {
                depth += 1;
                i += 1;
            }
            b')' => {
                depth -= 1;
                i += 1;
            }
            _ => i += 1,
        }
    }
    i
}

fn skip_hex_string(data: &[u8], mut i: usize) -> usize {
    i += 1;
    while i < data.len() && data[i] != b'>' {
        i += 1;
    }
    if i < data.len() {
        i += 1;
    }
    i
}

fn skip_name(data: &[u8], i: &mut usize) {
    *i += 1;
    while *i < data.len() && !is_content_space(data[*i]) && !is_delimiter(data[*i]) {
        *i += 1;
    }
}

fn skip_number(data: &[u8], i: &mut usize) {
    if *i < data.len() && matches!(data[*i], b'+' | b'-') {
        *i += 1;
    }
    while *i < data.len() && data[*i].is_ascii_digit() {
        *i += 1;
    }
    if *i < data.len() && data[*i] == b'.' {
        *i += 1;
        while *i < data.len() && data[*i].is_ascii_digit() {
            *i += 1;
        }
    }
}

fn skip_inline_image_after_bi(data: &[u8], mut i: usize) -> usize {
    skip_content_space(data, &mut i);
    if let Some(pos) = data[i..].windows(4).position(|window| {
        is_content_space(window[0])
            && window[1] == b'E'
            && window[2] == b'I'
            && is_content_space(window[3])
    }) {
        return i + pos + 3;
    }
    i
}
