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

    /// Per-byte CMap lookup without Latin-1 fallback.
    /// Returns `(raw_byte, Option<cmap_result>)` for each byte.
    /// Only meaningful for single-byte (code_byte_length==1) CMaps.
    pub fn lookup_bytes(&self, bytes: &[u8]) -> Vec<(u8, Option<String>)> {
        bytes
            .iter()
            .map(|&b| {
                let code = b as u16;
                let result = self.lookup(code).filter(|s| !s.contains('\u{FFFD}'));
                (b, result)
            })
            .collect()
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

    /// Get the minimum source CID across all mappings (char_map + ranges).
    fn min_source_cid(&self) -> Option<u16> {
        let char_min = self.char_map.keys().copied().min();
        let range_min = self.ranges.iter().map(|&(start, _, _)| start).min();
        match (char_min, range_min) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a @ Some(_), None) => a,
            (None, b @ Some(_)) => b,
            (None, None) => None,
        }
    }

    /// Remap a CMap that references pre-subsetting GIDs to sequential post-subsetting GIDs.
    /// Collects all source CIDs, sorts them, and reassigns to 1, 2, 3, ...
    pub fn remap_to_sequential(&self) -> ToUnicodeCMap {
        let mut cid_to_unicode: HashMap<u16, String> = HashMap::new();

        // Expand ranges first
        for &(start, end, base) in &self.ranges {
            for cid in start..=end {
                let unicode_cp = base + (cid - start) as u32;
                if let Some(ch) = char::from_u32(unicode_cp) {
                    cid_to_unicode.insert(cid, ch.to_string());
                }
            }
        }

        // char_map entries override range entries
        for (&cid, unicode) in &self.char_map {
            cid_to_unicode.insert(cid, unicode.clone());
        }

        // Sort old CIDs ascending
        let mut old_cids: Vec<u16> = cid_to_unicode.keys().copied().collect();
        old_cids.sort_unstable();

        // Build new CMap with sequential CIDs starting at 1
        let mut new_cmap = ToUnicodeCMap::new();
        for (i, &old_cid) in old_cids.iter().enumerate() {
            let new_cid = (i + 1) as u16; // GID 0 is .notdef, content CIDs start at 1
            if let Some(unicode) = cid_to_unicode.get(&old_cid) {
                new_cmap.char_map.insert(new_cid, unicode.clone());
            }
        }
        new_cmap.code_byte_length = self.code_byte_length;

        new_cmap
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

/// Navigate to the first DescendantFont dictionary of a Type0 font.
fn get_descendant_cid_font<'a>(
    font_dict: &'a lopdf::Dictionary,
    doc: &'a Document,
) -> Option<&'a lopdf::Dictionary> {
    let desc_fonts_obj = font_dict.get(b"DescendantFonts").ok()?;
    let arr = match desc_fonts_obj {
        Object::Array(arr) => arr,
        Object::Reference(r) => match doc.get_object(*r) {
            Ok(Object::Array(arr)) => arr,
            _ => return None,
        },
        _ => return None,
    };
    if arr.is_empty() {
        return None;
    }
    match &arr[0] {
        Object::Reference(r) => doc.get_dictionary(*r).ok(),
        Object::Dictionary(d) => Some(d),
        _ => None,
    }
}

/// Check if a CIDFont has an explicit (non-Identity) CIDToGIDMap.
fn has_explicit_cid_to_gid_map(cid_font_dict: &lopdf::Dictionary, doc: &Document) -> bool {
    match cid_font_dict.get(b"CIDToGIDMap").ok() {
        None => false,
        Some(Object::Name(n)) => n.as_slice() != b"Identity",
        Some(Object::Reference(r)) => match doc.get_object(*r) {
            Ok(Object::Name(n)) => n.as_slice() != b"Identity",
            Ok(Object::Stream(_)) => true,
            _ => false,
        },
        Some(_) => true,
    }
}

/// Get the starting CID from a CIDFont's W (widths) array.
fn get_w_array_start_cid(cid_font_dict: &lopdf::Dictionary, doc: &Document) -> Option<u16> {
    let w_obj = cid_font_dict.get(b"W").ok()?;
    let arr = match w_obj {
        Object::Array(arr) => arr,
        Object::Reference(r) => match doc.get_object(*r) {
            Ok(Object::Array(arr)) => arr,
            _ => return None,
        },
        _ => return None,
    };
    if arr.is_empty() {
        return None;
    }
    match &arr[0] {
        Object::Integer(n) => Some(*n as u16),
        Object::Reference(r) => match doc.get_object(*r) {
            Ok(Object::Integer(n)) => Some(*n as u16),
            _ => None,
        },
        _ => None,
    }
}

/// Detect and fix broken ToUnicode CMaps from subset fonts with GID mismatch.
///
/// Some PDF generators subset-embed fonts by renumbering GIDs sequentially (1, 2, 3...)
/// but fail to update the ToUnicode CMap, which still references original GID values.
/// This detects the mismatch and remaps the CMap to sequential positions.
fn try_remap_subset_cmap(
    cmap: ToUnicodeCMap,
    font_dict: &lopdf::Dictionary,
    doc: &Document,
    obj_num: u32,
) -> ToUnicodeCMap {
    // Only applies to Identity-H/V CID fonts
    let encoding = font_dict
        .get(b"Encoding")
        .ok()
        .and_then(|o| o.as_name().ok());
    if encoding != Some(b"Identity-H") && encoding != Some(b"Identity-V") {
        return cmap;
    }

    // CMap's minimum source CID must be > 2 (indicating old, non-sequential GIDs)
    let min_cid = match cmap.min_source_cid() {
        Some(c) if c > 2 => c,
        _ => return cmap,
    };

    // Navigate to DescendantFonts[0]
    let cid_font_dict = match get_descendant_cid_font(font_dict, doc) {
        Some(d) => d,
        None => return cmap,
    };

    // Must not have an explicit CIDToGIDMap (absent or /Identity is OK)
    if has_explicit_cid_to_gid_map(cid_font_dict, doc) {
        return cmap;
    }

    // W array must start at a low CID (≤ 2), indicating sequential post-subset GIDs
    let w_start = match get_w_array_start_cid(cid_font_dict, doc) {
        Some(c) if c <= 2 => c,
        _ => return cmap,
    };

    debug!(
        "Subset GID mismatch detected for obj={}: W starts at CID {}, CMap min CID {}. Remapping to sequential.",
        obj_num, w_start, min_cid
    );

    cmap.remap_to_sequential()
}

/// Build a ToUnicodeCMap from an embedded TrueType font's cmap table.
///
/// For Identity-H CID fonts, CID == GID. The TrueType cmap maps Unicode→GID,
/// so we reverse it to get GID→Unicode (i.e. CID→Unicode).
pub fn build_cmap_from_truetype(font_data: &[u8]) -> Option<ToUnicodeCMap> {
    let face = ttf_parser::Face::parse(font_data, 0).ok()?;

    let mut gid_to_unicode: HashMap<u16, char> = HashMap::new();

    // Iterate all Unicode codepoints that have a glyph mapping.
    // For each codepoint, the face gives us a GlyphId; reverse that to GID→Unicode.
    // We prefer the first (lowest) codepoint for each GID to handle duplicates.
    for subtable in face.tables().cmap.iter().flat_map(|cmap| cmap.subtables) {
        if !subtable.is_unicode() {
            continue;
        }
        subtable.codepoints(|cp| {
            if let Some(ch) = char::from_u32(cp) {
                if let Some(gid) = subtable.glyph_index(cp) {
                    let gid_val = gid.0;
                    // Keep the lowest codepoint per GID
                    gid_to_unicode.entry(gid_val).or_insert(ch);
                }
            }
        });
    }

    if gid_to_unicode.is_empty() {
        return None;
    }

    debug!(
        "TrueType cmap: {} GID→Unicode entries",
        gid_to_unicode.len()
    );

    let mut cmap = ToUnicodeCMap::new();
    for (gid, ch) in &gid_to_unicode {
        cmap.char_map.insert(*gid, ch.to_string());
    }
    cmap.code_byte_length = 2; // Identity-H uses 2-byte CIDs

    Some(cmap)
}

/// Build a ToUnicodeCMap from predefined CID→Unicode mapping based on CIDSystemInfo.
///
/// Supports Adobe-Korea1 (Korean) character collection. Can be extended for
/// Adobe-Japan1, Adobe-GB1, Adobe-CNS1 in the future.
fn build_cmap_from_cid_system_info(
    cid_font_dict: &lopdf::Dictionary,
    doc: &Document,
) -> Option<ToUnicodeCMap> {
    let csi_obj = cid_font_dict.get(b"CIDSystemInfo").ok()?;
    let csi_dict = match csi_obj {
        Object::Reference(r) => doc.get_dictionary(*r).ok()?,
        Object::Dictionary(d) => d,
        _ => return None,
    };
    let ordering = csi_dict.get(b"Ordering").ok().and_then(|o| {
        if let Object::String(bytes, _) = o {
            Some(String::from_utf8_lossy(bytes).to_string())
        } else {
            None
        }
    })?;

    match ordering.as_str() {
        "Korea1" => {
            use crate::adobe_korea1::ADOBE_KOREA1_CID_TO_UNICODE;
            let mut cmap = ToUnicodeCMap::new();
            for &(cid, unicode) in ADOBE_KOREA1_CID_TO_UNICODE.iter() {
                if let Some(ch) = char::from_u32(unicode as u32) {
                    cmap.char_map.insert(cid, ch.to_string());
                }
            }
            cmap.code_byte_length = 2;
            debug!(
                "Adobe-Korea1 predefined CMap: {} entries",
                cmap.char_map.len()
            );
            Some(cmap)
        }
        // Future: "Japan1", "GB1", "CNS1"
        _ => None,
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
    /// Also handles Identity-H/V CID fonts without ToUnicode by parsing
    /// the embedded TrueType cmap from FontFile2.
    fn collect_cmaps_from_fonts(
        fonts: &std::collections::BTreeMap<Vec<u8>, &lopdf::Dictionary>,
        doc: &Document,
        by_obj_num: &mut HashMap<u32, ToUnicodeCMap>,
    ) {
        // First pass: collect ToUnicode CMaps
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
                let cmap = try_remap_subset_cmap(cmap, font_dict, doc, obj_num);
                by_obj_num.insert(obj_num, cmap);
            }
        }

        // Second pass: Identity-H/V fonts without ToUnicode
        // Try: (1) embedded TrueType/OpenType cmap, (2) predefined CID→Unicode mapping
        for font_dict in fonts.values() {
            if font_dict.get(b"ToUnicode").is_ok() {
                continue;
            }
            let encoding = match font_dict
                .get(b"Encoding")
                .ok()
                .and_then(|o| o.as_name().ok())
            {
                Some(name) => name,
                None => continue,
            };
            if encoding != b"Identity-H" && encoding != b"Identity-V" {
                continue;
            }
            // Navigate: DescendantFonts[0]
            let desc_fonts_obj = match font_dict.get(b"DescendantFonts").ok() {
                Some(obj) => obj,
                None => continue,
            };
            let desc_fonts = match desc_fonts_obj {
                Object::Array(arr) => arr.clone(),
                Object::Reference(r) => match doc.get_object(*r) {
                    Ok(Object::Array(arr)) => arr.clone(),
                    _ => continue,
                },
                _ => continue,
            };
            if desc_fonts.is_empty() {
                continue;
            }
            let cid_font_dict = match &desc_fonts[0] {
                Object::Reference(r) => match doc.get_dictionary(*r) {
                    Ok(d) => d,
                    _ => continue,
                },
                Object::Dictionary(d) => d,
                _ => continue,
            };

            // Try to build CMap from embedded font (FontFile2 or FontFile3)
            let font_descriptor = cid_font_dict
                .get(b"FontDescriptor")
                .ok()
                .and_then(|o| match o {
                    Object::Reference(r) => doc.get_dictionary(*r).ok(),
                    Object::Dictionary(d) => Some(d),
                    _ => None,
                });

            let mut resolved = false;

            // Determine the font file reference (FontFile2 or FontFile3)
            let font_file_ref = font_descriptor.and_then(|fd| {
                fd.get(b"FontFile2")
                    .ok()
                    .and_then(|o| o.as_reference().ok())
                    .or_else(|| {
                        fd.get(b"FontFile3")
                            .ok()
                            .and_then(|o| o.as_reference().ok())
                    })
            });

            // The lookup key must match what get_font_file2_obj_num() returns:
            // font file obj_num if present, else CIDFont dict obj_num
            let lookup_key = font_file_ref
                .map(|r| r.0)
                .unwrap_or_else(|| match &desc_fonts[0] {
                    Object::Reference(r) => r.0,
                    _ => 0,
                });
            if lookup_key == 0 || by_obj_num.contains_key(&lookup_key) {
                continue;
            }

            // Try parsing embedded TrueType/OpenType cmap
            if let Some(ff_ref) = font_file_ref {
                if let Ok(stream) = doc.get_object(ff_ref).and_then(Object::as_stream) {
                    if let Ok(data) = stream.decompressed_content() {
                        if let Some(cmap) = build_cmap_from_truetype(&data) {
                            debug!(
                                "TrueType CMap obj={:<6} (embedded font) char_map={}",
                                lookup_key,
                                cmap.char_map.len()
                            );
                            by_obj_num.insert(lookup_key, cmap);
                            resolved = true;
                        }
                    }
                }
            }

            // Fallback: predefined CID→Unicode mapping from CIDSystemInfo
            if !resolved {
                if let Some(cmap) = build_cmap_from_cid_system_info(cid_font_dict, doc) {
                    debug!(
                        "Predefined CMap obj={:<6} (CIDSystemInfo) char_map={}",
                        lookup_key,
                        cmap.char_map.len()
                    );
                    by_obj_num.insert(lookup_key, cmap);
                }
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
    fn test_remap_to_sequential() {
        // Simulate a broken CMap where GIDs are from pre-subsetting:
        // Old GID 3 → space, old GID 36 → 'A', old GID 37 → 'B'
        // The subset font has sequential GIDs: 1=space, 2='A', 3='B'
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

        // Original CMap: CID 3 → space, CID 36 → 'A', CID 37 → 'B'
        assert_eq!(cmap.lookup(0x0003), Some(" ".to_string()));
        assert_eq!(cmap.lookup(0x0024), Some("A".to_string()));
        assert_eq!(cmap.lookup(0x0025), Some("B".to_string()));
        assert_eq!(cmap.lookup(0x0001), None);
        assert_eq!(cmap.lookup(0x0002), None);

        // After remapping: CID 1 → space, CID 2 → 'A', CID 3 → 'B'
        let remapped = cmap.remap_to_sequential();
        assert_eq!(remapped.lookup(0x0001), Some(" ".to_string()));
        assert_eq!(remapped.lookup(0x0002), Some("A".to_string()));
        assert_eq!(remapped.lookup(0x0003), Some("B".to_string()));
        assert_eq!(remapped.lookup(0x0024), None);
        assert_eq!(remapped.lookup(0x0025), None);
    }

    #[test]
    fn test_remap_to_sequential_with_ranges() {
        // CMap with a bfrange: old GIDs 100-102 → 'X', 'Y', 'Z'
        // Plus a bfchar: old GID 50 → space
        let cmap_content = r#"
1 begincodespacerange
<0000><FFFF>
endcodespacerange
1 beginbfchar
<0032> <0020>
endbfchar
1 beginbfrange
<0064> <0066> <0058>
endbfrange
"#;
        let cmap = ToUnicodeCMap::parse(cmap_content.as_bytes()).unwrap();

        assert_eq!(cmap.lookup(0x0032), Some(" ".to_string())); // CID 50
        assert_eq!(cmap.lookup(0x0064), Some("X".to_string())); // CID 100
        assert_eq!(cmap.lookup(0x0065), Some("Y".to_string())); // CID 101
        assert_eq!(cmap.lookup(0x0066), Some("Z".to_string())); // CID 102

        let remapped = cmap.remap_to_sequential();
        // Sorted old CIDs: 50, 100, 101, 102 → new CIDs: 1, 2, 3, 4
        assert_eq!(remapped.lookup(0x0001), Some(" ".to_string()));
        assert_eq!(remapped.lookup(0x0002), Some("X".to_string()));
        assert_eq!(remapped.lookup(0x0003), Some("Y".to_string()));
        assert_eq!(remapped.lookup(0x0004), Some("Z".to_string()));
        // Ranges should be cleared (all in char_map now)
        assert!(remapped.ranges.is_empty());
    }

    #[test]
    fn test_min_source_cid() {
        let cmap_content = r#"
1 begincodespacerange
<0000><FFFF>
endcodespacerange
2 beginbfchar
<0003> <0020>
<0024> <0041>
endbfchar
1 beginbfrange
<0030> <0032> <0058>
endbfrange
"#;
        let cmap = ToUnicodeCMap::parse(cmap_content.as_bytes()).unwrap();
        assert_eq!(cmap.min_source_cid(), Some(3));
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
