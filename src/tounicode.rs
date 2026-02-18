//! ToUnicode CMap parsing for PDF text extraction
//!
//! This module parses ToUnicode CMaps to convert CID-encoded text to Unicode.

use log::debug;
use lopdf::{Document, Object, ObjectId};
use std::collections::{HashMap, HashSet};

/// A parsed ToUnicode CMap mapping CIDs to Unicode strings
#[derive(Debug, Default, Clone)]
pub struct ToUnicodeCMap {
    /// Direct character mappings (CID -> Unicode codepoint(s))
    pub char_map: HashMap<u16, String>,
    /// Range mappings (start_cid, end_cid) -> base_unicode
    pub ranges: Vec<(u16, u16, u32)>,
    /// Byte width of source codes (1 or 2), determined from codespace and CMap entries
    pub code_byte_length: u8,
}

impl ToUnicodeCMap {
    /// Create a new empty CMap
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse a ToUnicode CMap from its decompressed content
    pub fn parse(content: &[u8]) -> Option<Self> {
        let text = String::from_utf8_lossy(content);
        let mut cmap = ToUnicodeCMap::new();
        let mut src_hex_lengths: Vec<usize> = Vec::new();

        // Parse begincodespacerange ... endcodespacerange to determine byte width
        let mut codespace_byte_len: Option<u8> = None;
        if let Some(cs_start) = text.find("begincodespacerange") {
            let section_start = cs_start + "begincodespacerange".len();
            if let Some(cs_end) = text[section_start..].find("endcodespacerange") {
                let section = &text[section_start..section_start + cs_end];
                // Parse hex values to determine byte length
                let mut in_hex = false;
                let mut hex_len = 0;
                for c in section.chars() {
                    if c == '<' {
                        in_hex = true;
                        hex_len = 0;
                    } else if c == '>' {
                        if in_hex && hex_len > 0 {
                            let byte_len = (hex_len + 1) / 2; // 2 hex digits = 1 byte
                            codespace_byte_len = Some(byte_len as u8);
                        }
                        in_hex = false;
                    } else if in_hex && c.is_ascii_hexdigit() {
                        hex_len += 1;
                    }
                }
            }
        }

        // Parse beginbfchar ... endbfchar sections
        let mut pos = 0;
        while let Some(start) = text[pos..].find("beginbfchar") {
            let section_start = pos + start + "beginbfchar".len();
            if let Some(end) = text[section_start..].find("endbfchar") {
                let section = &text[section_start..section_start + end];
                cmap.parse_bfchar_section(section, &mut src_hex_lengths);
                pos = section_start + end;
            } else {
                break;
            }
        }

        // Parse beginbfrange ... endbfrange sections
        pos = 0;
        while let Some(start) = text[pos..].find("beginbfrange") {
            let section_start = pos + start + "beginbfrange".len();
            if let Some(end) = text[section_start..].find("endbfrange") {
                let section = &text[section_start..section_start + end];
                cmap.parse_bfrange_section(section, &mut src_hex_lengths);
                pos = section_start + end;
            } else {
                break;
            }
        }

        if cmap.char_map.is_empty() && cmap.ranges.is_empty() {
            return None;
        }

        // Determine byte width: use codespace if available, otherwise infer from entries
        cmap.code_byte_length = if let Some(cs_len) = codespace_byte_len {
            // If codespace says 2-byte but ALL entries use 1-byte source codes
            // (hex length <= 2), treat as 1-byte. This handles the common case where
            // codespace is <0000><FFFF> but entries are <20>, <41>, etc.
            if cs_len == 2 && !src_hex_lengths.is_empty() && src_hex_lengths.iter().all(|&l| l <= 2)
            {
                1
            } else {
                cs_len
            }
        } else if !src_hex_lengths.is_empty() {
            // No codespace declaration: infer from entry hex lengths
            let max_hex_len = src_hex_lengths.iter().max().copied().unwrap_or(4);
            if max_hex_len <= 2 {
                1
            } else {
                2
            }
        } else {
            2 // Default to 2-byte
        };

        // Sort ranges by start CID for binary search in lookup()
        cmap.ranges.sort_unstable_by_key(|&(start, _, _)| start);

        Some(cmap)
    }

    /// Parse a bfchar section: <src> <dst> pairs
    fn parse_bfchar_section(&mut self, section: &str, src_hex_lengths: &mut Vec<usize>) {
        // Match pairs of hex values: <XXXX> <YYYY>
        let mut chars = section.chars().peekable();

        loop {
            // Skip whitespace
            while chars.peek().is_some_and(|c| c.is_whitespace()) {
                chars.next();
            }

            // Look for opening <
            if chars.peek() != Some(&'<') {
                break;
            }
            chars.next(); // consume <

            // Read source hex
            let mut src_hex = String::new();
            while chars.peek().is_some_and(|&c| c != '>') {
                if let Some(c) = chars.next() {
                    src_hex.push(c);
                }
            }
            chars.next(); // consume >

            // Track source hex length for byte width detection
            let trimmed_src = src_hex.trim();
            if !trimmed_src.is_empty() {
                src_hex_lengths.push(trimmed_src.len());
            }

            // Skip whitespace
            while chars.peek().is_some_and(|c| c.is_whitespace()) {
                chars.next();
            }

            // Look for opening <
            if chars.peek() != Some(&'<') {
                continue;
            }
            chars.next(); // consume <

            // Read destination hex
            let mut dst_hex = String::new();
            while chars.peek().is_some_and(|&c| c != '>') {
                if let Some(c) = chars.next() {
                    dst_hex.push(c);
                }
            }
            chars.next(); // consume >

            // Parse and store mapping
            if let (Some(src), Some(dst)) =
                (parse_hex_u16(&src_hex), hex_to_unicode_string(&dst_hex))
            {
                self.char_map.insert(src, dst);
            }
        }
    }

    /// Parse a bfrange section: <start> <end> <base> or <start> <end> [<u1> <u2> ...] triplets
    fn parse_bfrange_section(&mut self, section: &str, src_hex_lengths: &mut Vec<usize>) {
        let mut chars = section.chars().peekable();

        loop {
            // Skip whitespace
            while chars.peek().is_some_and(|c| c.is_whitespace()) {
                chars.next();
            }

            // Look for opening <
            if chars.peek() != Some(&'<') {
                break;
            }
            chars.next(); // consume <

            // Read start hex
            let mut start_hex = String::new();
            while chars.peek().is_some_and(|&c| c != '>') {
                if let Some(c) = chars.next() {
                    start_hex.push(c);
                }
            }
            chars.next(); // consume >

            // Track source hex length
            let trimmed_start = start_hex.trim();
            if !trimmed_start.is_empty() {
                src_hex_lengths.push(trimmed_start.len());
            }

            // Skip whitespace
            while chars.peek().is_some_and(|c| c.is_whitespace()) {
                chars.next();
            }

            // Read end hex
            if chars.peek() != Some(&'<') {
                continue;
            }
            chars.next();
            let mut end_hex = String::new();
            while chars.peek().is_some_and(|&c| c != '>') {
                if let Some(c) = chars.next() {
                    end_hex.push(c);
                }
            }
            chars.next();

            // Skip whitespace
            while chars.peek().is_some_and(|c| c.is_whitespace()) {
                chars.next();
            }

            // Read base - could be <hex> or [array]
            if chars.peek() == Some(&'<') {
                chars.next();
                let mut base_hex = String::new();
                while chars.peek().is_some_and(|&c| c != '>') {
                    if let Some(c) = chars.next() {
                        base_hex.push(c);
                    }
                }
                chars.next();

                // Store range mapping
                if let (Some(start), Some(end), Some(base)) = (
                    parse_hex_u16(&start_hex),
                    parse_hex_u16(&end_hex),
                    parse_hex_u32(&base_hex),
                ) {
                    self.ranges.push((start, end, base));
                }
            } else if chars.peek() == Some(&'[') {
                // Array format: [<unicode1> <unicode2> ...]
                // Each entry maps to start_cid + index
                chars.next(); // consume [
                if let (Some(start), Some(end)) =
                    (parse_hex_u16(&start_hex), parse_hex_u16(&end_hex))
                {
                    let mut cid = start;
                    loop {
                        // Skip whitespace
                        while chars.peek().is_some_and(|c| c.is_whitespace()) {
                            chars.next();
                        }
                        if chars.peek() == Some(&']') {
                            chars.next();
                            break;
                        }
                        if chars.peek() != Some(&'<') {
                            break;
                        }
                        chars.next(); // consume <
                        let mut hex = String::new();
                        while chars.peek().is_some_and(|&c| c != '>') {
                            if let Some(c) = chars.next() {
                                hex.push(c);
                            }
                        }
                        chars.next(); // consume >
                        if let Some(unicode_str) = hex_to_unicode_string(&hex) {
                            self.char_map.insert(cid, unicode_str);
                        }
                        if cid >= end {
                            // Skip remaining entries and closing bracket
                            while chars.peek().is_some_and(|&c| c != ']') {
                                chars.next();
                            }
                            if chars.peek() == Some(&']') {
                                chars.next();
                            }
                            break;
                        }
                        cid = cid.saturating_add(1);
                    }
                } else {
                    // Couldn't parse start/end, skip the array
                    while chars.peek().is_some_and(|&c| c != ']') {
                        chars.next();
                    }
                    if chars.peek() == Some(&']') {
                        chars.next();
                    }
                }
            }
        }
    }

    /// Look up a CID and return the Unicode string
    pub fn lookup(&self, cid: u16) -> Option<String> {
        // First check direct mappings
        if let Some(s) = self.char_map.get(&cid) {
            return Some(s.clone());
        }

        // Binary search through sorted ranges
        let idx = self
            .ranges
            .binary_search_by(|&(start, _, _)| start.cmp(&cid))
            .unwrap_or_else(|i| i);

        // Check the range at idx (where start == cid)
        if idx < self.ranges.len() {
            let (start, end, base) = self.ranges[idx];
            if cid >= start && cid <= end {
                let unicode = base + (cid - start) as u32;
                if let Some(c) = char::from_u32(unicode) {
                    return Some(c.to_string());
                }
            }
        }

        // Check the range before idx (cid may fall within a range that starts before it)
        if idx > 0 {
            let (start, end, base) = self.ranges[idx - 1];
            if cid >= start && cid <= end {
                let unicode = base + (cid - start) as u32;
                if let Some(c) = char::from_u32(unicode) {
                    return Some(c.to_string());
                }
            }
        }

        None
    }

    /// Decode a byte slice to a Unicode string, respecting the CMap's code byte width
    pub fn decode_cids(&self, bytes: &[u8]) -> String {
        let mut result = String::new();
        let mut unmapped_count = 0usize;

        if self.code_byte_length == 1 {
            // Single-byte codes: each byte is a code
            for &b in bytes {
                let code = b as u16;
                match self.lookup(code) {
                    Some(s) if !s.contains('\u{FFFD}') => result.push_str(&s),
                    _ => {
                        // For single-byte unmapped codes, try as Latin-1
                        // (the byte IS the character code in most legacy encodings)
                        if b >= 0x20 {
                            result.push(b as char);
                        }
                        unmapped_count += 1;
                    }
                }
            }
        } else {
            // Two-byte codes: CIDs are 2 bytes each (big-endian)
            for chunk in bytes.chunks(2) {
                if chunk.len() == 2 {
                    let cid = u16::from_be_bytes([chunk[0], chunk[1]]);
                    match self.lookup(cid) {
                        Some(s) if !s.contains('\u{FFFD}') => result.push_str(&s),
                        _ => {
                            // Do NOT blindly interpret CIDs as Unicode codepoints.
                            // CIDs are font-internal indices, not Unicode values.
                            // Unmapped 2-byte CIDs are skipped to avoid CJK garbage.
                            unmapped_count += 1;
                        }
                    }
                }
            }
        }

        // If too many codes were unmapped, signal failure by returning empty
        // so the caller can fall through to other decoding methods
        let total = if self.code_byte_length == 1 {
            bytes.len()
        } else {
            bytes.len() / 2
        };
        if total > 0 && unmapped_count > total / 2 {
            return String::new();
        }

        result
    }
}

/// Parse a hex string to u16
fn parse_hex_u16(hex: &str) -> Option<u16> {
    u16::from_str_radix(hex.trim(), 16).ok()
}

/// Parse a hex string to u32
fn parse_hex_u32(hex: &str) -> Option<u32> {
    u32::from_str_radix(hex.trim(), 16).ok()
}

/// Convert a hex string to a Unicode string
/// Handles both 2-byte (BMP) and 4-byte (supplementary) codepoints
fn hex_to_unicode_string(hex: &str) -> Option<String> {
    let hex = hex.trim();
    let mut result = String::new();

    // Process 4 hex digits at a time
    let mut i = 0;
    while i + 4 <= hex.len() {
        if let Ok(cp) = u32::from_str_radix(&hex[i..i + 4], 16) {
            if let Some(c) = char::from_u32(cp) {
                result.push(c);
            }
        }
        i += 4;
    }

    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

/// Collection of ToUnicode CMaps indexed by ToUnicode stream object number
#[derive(Debug, Default, Clone)]
pub struct FontCMaps {
    /// Map of ToUnicode object number to CMap
    by_obj_num: HashMap<u32, ToUnicodeCMap>,
}

impl FontCMaps {
    /// Build FontCMaps from a lopdf Document model.
    ///
    /// Iterates every page, collects fonts (including Form XObject fonts),
    /// and parses any `/ToUnicode` streams via lopdf's decompression.
    pub fn from_doc(doc: &Document) -> Self {
        let mut by_obj_num: HashMap<u32, ToUnicodeCMap> = HashMap::new();

        for (_page_num, &page_id) in doc.get_pages().iter() {
            // Page-level fonts (includes inherited parent resources)
            let fonts = doc.get_page_fonts(page_id).unwrap_or_default();
            Self::collect_cmaps_from_fonts(&fonts, doc, &mut by_obj_num);

            // Fonts inside Form XObjects referenced by this page
            Self::collect_cmaps_from_xobjects(doc, page_id, &mut by_obj_num);
        }

        FontCMaps { by_obj_num }
    }

    /// Parse ToUnicode CMaps from a set of font dictionaries.
    fn collect_cmaps_from_fonts(
        fonts: &std::collections::BTreeMap<Vec<u8>, &lopdf::Dictionary>,
        doc: &Document,
        by_obj_num: &mut HashMap<u32, ToUnicodeCMap>,
    ) {
        for font_dict in fonts.values() {
            let obj_ref = match font_dict
                .get(b"ToUnicode")
                .ok()
                .and_then(|o| o.as_reference().ok())
            {
                Some(r) => r,
                None => continue,
            };
            let obj_num = obj_ref.0;
            if by_obj_num.contains_key(&obj_num) {
                continue;
            }
            let stream = match doc.get_object(obj_ref).and_then(Object::as_stream) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let data = match stream.decompressed_content() {
                Ok(d) => d,
                Err(_) => continue,
            };
            if let Some(cmap) = ToUnicodeCMap::parse(&data) {
                debug!(
                    "CMap obj={:<6} code_byte_length={} char_map={} ranges={}",
                    obj_num,
                    cmap.code_byte_length,
                    cmap.char_map.len(),
                    cmap.ranges.len()
                );
                by_obj_num.insert(obj_num, cmap);
            }
        }
    }

    /// Walk Form XObjects in a page's resources and collect their font CMaps.
    fn collect_cmaps_from_xobjects(
        doc: &Document,
        page_id: ObjectId,
        by_obj_num: &mut HashMap<u32, ToUnicodeCMap>,
    ) {
        let (resource_dict, resource_ids) = match doc.get_page_resources(page_id) {
            Ok(r) => r,
            Err(_) => return,
        };

        let mut visited = HashSet::new();

        if let Some(resources) = resource_dict {
            Self::walk_xobject_fonts(resources, doc, by_obj_num, &mut visited);
        }
        for resource_id in resource_ids {
            if let Ok(resources) = doc.get_dictionary(resource_id) {
                Self::walk_xobject_fonts(resources, doc, by_obj_num, &mut visited);
            }
        }
    }

    /// Recursively collect font CMaps from XObjects in a resource dictionary.
    fn walk_xobject_fonts(
        resources: &lopdf::Dictionary,
        doc: &Document,
        by_obj_num: &mut HashMap<u32, ToUnicodeCMap>,
        visited: &mut HashSet<ObjectId>,
    ) {
        let xobject_dict = match resources.get(b"XObject") {
            Ok(Object::Reference(id)) => doc.get_object(*id).and_then(Object::as_dict).ok(),
            Ok(Object::Dictionary(dict)) => Some(dict),
            _ => None,
        };
        let xobject_dict = match xobject_dict {
            Some(d) => d,
            None => return,
        };

        for (_name, value) in xobject_dict.iter() {
            let id = match value {
                Object::Reference(id) => *id,
                _ => continue,
            };
            if !visited.insert(id) {
                continue;
            }
            let stream = match doc.get_object(id).and_then(Object::as_stream) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let is_form = stream
                .dict
                .get(b"Subtype")
                .and_then(|o| o.as_name())
                .is_ok_and(|n| n == b"Form");
            if !is_form {
                continue;
            }
            // Collect fonts from this Form XObject's Resources
            if let Ok(form_resources) = stream.dict.get(b"Resources").and_then(Object::as_dict) {
                // Extract font dict from the Form's resources
                let font_dict_obj = match form_resources.get(b"Font") {
                    Ok(Object::Reference(id)) => doc.get_object(*id).and_then(Object::as_dict).ok(),
                    Ok(Object::Dictionary(dict)) => Some(dict),
                    _ => None,
                };
                if let Some(font_dict) = font_dict_obj {
                    let mut fonts = std::collections::BTreeMap::new();
                    for (name, value) in font_dict.iter() {
                        let font = match value {
                            Object::Reference(id) => doc.get_dictionary(*id).ok(),
                            Object::Dictionary(dict) => Some(dict),
                            _ => None,
                        };
                        if let Some(font) = font {
                            fonts.insert(name.clone(), font);
                        }
                    }
                    Self::collect_cmaps_from_fonts(&fonts, doc, by_obj_num);
                }
                // Recurse into nested XObjects
                Self::walk_xobject_fonts(form_resources, doc, by_obj_num, visited);
            }
        }
    }

    /// Get a CMap by ToUnicode object number
    pub fn get_by_obj(&self, obj_num: u32) -> Option<&ToUnicodeCMap> {
        self.by_obj_num.get(&obj_num)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_bfchar_2byte() {
        let cmap_content = r#"
/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
1 begincodespacerange
<0000><FFFF>
endcodespacerange
3 beginbfchar
<0003> <0020>
<0024> <0041>
<0025> <0042>
endbfchar
endcmap
"#;
        let cmap = ToUnicodeCMap::parse(cmap_content.as_bytes()).unwrap();

        assert_eq!(cmap.code_byte_length, 2);
        assert_eq!(cmap.lookup(0x0003), Some(" ".to_string()));
        assert_eq!(cmap.lookup(0x0024), Some("A".to_string()));
        assert_eq!(cmap.lookup(0x0025), Some("B".to_string()));
    }

    #[test]
    fn test_parse_bfchar_1byte() {
        // This is the pattern that caused the CJK bug: codespace is <0000><FFFF>
        // but all source codes are 1-byte hex (e.g., <20>, <41>)
        let cmap_content = r#"
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
3 beginbfchar
<20> <0020>
<41> <0041>
<42> <0042>
endbfchar
"#;
        let cmap = ToUnicodeCMap::parse(cmap_content.as_bytes()).unwrap();

        // Should detect as 1-byte because all source codes are 1-byte hex
        assert_eq!(cmap.code_byte_length, 1);
        assert_eq!(cmap.lookup(0x0020), Some(" ".to_string()));
        assert_eq!(cmap.lookup(0x0041), Some("A".to_string()));
    }

    #[test]
    fn test_decode_cids_2byte() {
        let cmap_content = r#"
1 begincodespacerange
<0000><FFFF>
endcodespacerange
3 beginbfchar
<0003> <0020>
<0024> <0041>
<0025> <0042>
endbfchar
"#;
        let cmap = ToUnicodeCMap::parse(cmap_content.as_bytes()).unwrap();

        // "AB " in 2-byte CID encoding
        let cids = [0x00, 0x24, 0x00, 0x25, 0x00, 0x03];
        assert_eq!(cmap.decode_cids(&cids), "AB ");
    }

    #[test]
    fn test_decode_cids_1byte_no_cjk_garbage() {
        // Simulates the bug: CMap with 1-byte source codes
        let cmap_content = r#"
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
5 beginbfchar
<20> <0020>
<42> <0042>
<79> <0079>
<50> <0050>
<52> <0052>
endbfchar
"#;
        let cmap = ToUnicodeCMap::parse(cmap_content.as_bytes()).unwrap();
        assert_eq!(cmap.code_byte_length, 1);

        // "By" should decode to "By", NOT to CJK character 䉹
        let bytes = [0x42, 0x79];
        let result = cmap.decode_cids(&bytes);
        assert_eq!(result, "By");
        assert!(!result.contains('䉹'), "Should not produce CJK garbage");

        // "PR" should decode to "PR"
        let bytes2 = [0x50, 0x52];
        assert_eq!(cmap.decode_cids(&bytes2), "PR");
    }

    #[test]
    fn test_bfrange_array_format() {
        let cmap_content = r#"
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
1 beginbfrange
<0003> <0005> [<0041> <0042> <0043>]
endbfrange
"#;
        let cmap = ToUnicodeCMap::parse(cmap_content.as_bytes()).unwrap();

        assert_eq!(cmap.lookup(0x0003), Some("A".to_string()));
        assert_eq!(cmap.lookup(0x0004), Some("B".to_string()));
        assert_eq!(cmap.lookup(0x0005), Some("C".to_string()));
    }

    #[test]
    fn test_unmapped_2byte_cids_skipped() {
        let cmap_content = r#"
1 begincodespacerange
<0000><FFFF>
endcodespacerange
1 beginbfchar
<0041> <0041>
endbfchar
"#;
        let cmap = ToUnicodeCMap::parse(cmap_content.as_bytes()).unwrap();
        assert_eq!(cmap.code_byte_length, 2);

        // CID 0x4279 is unmapped - should NOT produce CJK character
        let bytes = [0x42, 0x79];
        let result = cmap.decode_cids(&bytes);
        assert!(
            !result.contains('䉹'),
            "Unmapped 2-byte CIDs should not produce CJK"
        );
    }
}
