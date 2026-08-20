use lopdf::Document;
use std::borrow::Cow;
use std::path::Path;

pub const OCR_REASON_SUSPECTED_GARBLED_TEXT: &str = "suspected_garbled_text";
pub const OCR_REASON_SCANNED: &str = "scanned";
pub const OCR_REASON_NO_TEXT: &str = "no_text";
pub const OCR_REASON_VECTOR_TEXT: &str = "vector_text";

/// Repair malformed structure-element `/S` values before lopdf parses them.
/// Some generators emit `/S Code` instead of the valid `/S /Code`.
pub fn fix_bare_struct_names(buf: &[u8]) -> Cow<'_, [u8]> {
    if !contains_bytes(buf, b"/StructTreeRoot") {
        return Cow::Borrowed(buf);
    }
    const KNOWN_NAMES: &[&[u8]] = &[
        b"Document",
        b"Part",
        b"Art",
        b"Sect",
        b"Div",
        b"BlockQuote",
        b"Caption",
        b"TOC",
        b"TOCI",
        b"Index",
        b"NonStruct",
        b"Private",
        b"H",
        b"H1",
        b"H2",
        b"H3",
        b"H4",
        b"H5",
        b"H6",
        b"P",
        b"L",
        b"LI",
        b"Lbl",
        b"LBody",
        b"Table",
        b"TR",
        b"TH",
        b"TD",
        b"THead",
        b"TBody",
        b"TFoot",
        b"Span",
        b"Quote",
        b"Note",
        b"Reference",
        b"BibEntry",
        b"Code",
        b"Link",
        b"Annot",
        b"Figure",
        b"Formula",
        b"Form",
        b"Ruby",
        b"RB",
        b"RT",
        b"RP",
        b"Warichu",
        b"WT",
        b"WP",
    ];
    let pattern = b"/S ";
    let mut result: Option<Vec<u8>> = None;
    let mut pos = 0;
    while pos + pattern.len() < buf.len() {
        let Some(idx) = find_bytes(&buf[pos..], pattern).map(|index| index + pos) else {
            break;
        };
        let after = idx + pattern.len();
        if after < buf.len() && buf[after] == b'/' {
            pos = after;
            continue;
        }
        let mut matched = false;
        for name in KNOWN_NAMES {
            let end = after + name.len();
            if end <= buf.len()
                && &buf[after..end] == *name
                && (end >= buf.len() || matches!(buf[end], b'\n' | b'\r' | b' ' | b'/' | b'>'))
            {
                let out = result.get_or_insert_with(|| buf[..after].to_vec());
                if out.len() < after {
                    out.extend_from_slice(&buf[out.len()..after]);
                }
                out.push(b'/');
                out.extend_from_slice(name);
                pos = end;
                matched = true;
                log::debug!(
                    "fix_bare_struct_names: patched /S {} -> /S /{}",
                    String::from_utf8_lossy(name),
                    String::from_utf8_lossy(name)
                );
                break;
            }
        }
        if !matched {
            pos = after;
        }
    }
    match result {
        Some(mut out) => {
            if out.len() < buf.len() {
                out.extend_from_slice(&buf[out.len()..]);
            }
            Cow::Owned(out)
        }
        None => Cow::Borrowed(buf),
    }
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    find_bytes(haystack, needle).is_some()
}

/// Load a PDF from disk, returning the parsed document and page count.
///
/// `Document::load_metadata` for page count + `Document::load` for content
/// are combined here, but lopdf loads the full doc in `load()` so we extract
/// page count from it directly to avoid the metadata-only round-trip.
pub fn load_document_from_path<P: AsRef<Path>>(path: P) -> Result<(Document, u32), PdfError> {
    load_document_from_path_with_password(path, None)
}

/// Load a PDF file, decrypting with `password` if the file is encrypted.
pub fn load_document_from_path_with_password<P: AsRef<Path>>(
    path: P,
    password: Option<&str>,
) -> Result<(Document, u32), PdfError> {
    let buffer = std::fs::read(&path)?;
    load_document_from_mem_with_password(&buffer, password)
}

/// Load a PDF from a memory buffer.
pub fn load_document_from_mem(buffer: &[u8]) -> Result<(Document, u32), PdfError> {
    load_document_from_mem_with_password(buffer, None)
}

/// Load a PDF from a memory buffer, decrypting with `password` if encrypted.
pub fn load_document_from_mem_with_password(
    buffer: &[u8],
    password: Option<&str>,
) -> Result<(Document, u32), PdfError> {
    // Fix malformed struct element names before parsing. Some PDF generators
    // write bare names (/S Code) instead of proper PDF names (/S /Code), which
    // causes lopdf to silently drop the entire object.
    let fixed = fix_bare_struct_names(buffer);
    let buf = fixed.as_ref();

    let (empty_document, first_error) = match load_document_bytes(buf, password) {
        Ok(doc) => {
            let page_count = doc.get_pages().len() as u32;
            if page_count > 0 {
                return Ok((doc, page_count));
            }
            (Some(doc), None)
        }
        Err(error) => (None, Some(error)),
    };
    for repaired in repair_pdf_container_candidates(buf) {
        match load_document_bytes(&repaired, password) {
            Ok(doc) if !doc.get_pages().is_empty() => {
                log::debug!("loaded PDF after repairing malformed container bytes");
                let page_count = doc.get_pages().len() as u32;
                return Ok((doc, page_count));
            }
            Ok(_) => {}
            Err(error) if is_encrypted_lopdf_error(&error) => return Err(error.into()),
            Err(_) => {}
        }
    }
    match (empty_document, first_error) {
        (Some(doc), _) => Ok((doc, 0)),
        (_, Some(error)) => Err(error.into()),
        _ => unreachable!("initial PDF load produced neither a document nor an error"),
    }
}

fn load_document_bytes(buf: &[u8], password: Option<&str>) -> Result<Document, lopdf::Error> {
    match Document::load_mem(buf) {
        // Some encrypted PDFs load structurally but leave their streams
        // encrypted (`is_encrypted()` stays true); reading them yields garbage
        // until we re-load with a password. Others fail load_mem outright with
        // an encryption error. Handle both by re-loading with the password.
        Ok(doc) if doc.is_encrypted() => decrypt_document_bytes(buf, password),
        Ok(doc) => Ok(doc),
        Err(ref e) if is_encrypted_lopdf_error(e) => decrypt_document_bytes(buf, password),
        Err(e) => Err(e),
    }
}

/// Re-load an encrypted PDF, decrypting with `password`. Falls back to the
/// empty password (owner-only encryption, the common "protected" case) when a
/// non-empty password was supplied but rejected.
fn decrypt_document_bytes(buf: &[u8], password: Option<&str>) -> Result<Document, lopdf::Error> {
    let pw = password.unwrap_or("");
    match Document::load_mem_with_options(buf, lopdf::LoadOptions::with_password(pw)) {
        Ok(doc) => Ok(doc),
        Err(inner) if !pw.is_empty() => {
            Document::load_mem_with_options(buf, lopdf::LoadOptions::with_password(""))
                .map_err(|_| inner)
        }
        Err(inner) => Err(inner),
    }
}

fn repair_pdf_container_candidates(buf: &[u8]) -> Vec<Vec<u8>> {
    let mut candidates = Vec::new();

    add_repair_candidate(&mut candidates, append_missing_eof_marker(buf), buf);
    add_repair_candidate(&mut candidates, recover_startxref_pointer(buf), buf);
    add_repair_candidate(&mut candidates, strip_duplicated_pdf_prefix(buf), buf);

    let stripped = strip_leading_pdf_container_bytes(buf);
    if let Some(stripped_buf) = stripped.as_deref() {
        add_repair_candidate(&mut candidates, Some(stripped_buf.to_vec()), buf);
        add_repair_candidate(
            &mut candidates,
            append_missing_eof_marker(stripped_buf),
            buf,
        );
        add_repair_candidate(
            &mut candidates,
            recover_startxref_pointer(stripped_buf),
            buf,
        );
    }

    candidates
}

/// Some PDF writers emit a `startxref` pointer that doesn't actually point
/// at the cross-reference table — a single corrupted byte in the offset is
/// enough. lopdf trusts that pointer outright and fails to load rather than
/// searching for the real table, unlike pypdf/pdfium which both recover by
/// locating it directly. This finds the real (classic, non-stream) `xref`
/// table by scanning for the keyword — validating that a plausible
/// subsection header follows, not just any standalone "xref" token, since
/// this crate processes untrusted input and a coincidental match inside
/// unrelated stream/string content must not get "repaired" against a bogus
/// offset (lopdf would then load successfully against garbage instead of
/// returning a clean error) — and appends a corrected trailing
/// `startxref`/`%%EOF` block. lopdf's own `get_xref_start` always uses the
/// *last* `%%EOF` in the final 512 bytes of the buffer, so ours
/// transparently supersedes the broken one without needing to touch
/// anything already in the file.
///
/// Doesn't cover cross-reference *streams* (`N 0 obj << /Type /XRef ...`,
/// used by some PDF 1.5+ writers instead of a classic table) — recovering
/// those needs the containing object's number, not just a byte offset.
#[doc(hidden)]
pub fn recover_startxref_pointer(buf: &[u8]) -> Option<Vec<u8>> {
    let xref_pos = find_last_valid_xref_table_start(buf)?;

    let mut repaired = Vec::with_capacity(buf.len() + 32);
    repaired.extend_from_slice(buf);
    if !repaired.ends_with(b"\n") {
        repaired.push(b'\n');
    }
    repaired.extend_from_slice(format!("startxref\n{xref_pos}\n%%EOF\n").as_bytes());
    Some(repaired)
}

/// Finds the last standalone `xref` token in `buf` that is immediately
/// followed by a plausible classic cross-reference subsection header
/// (`<start-id> <count>`, e.g. "0 6") — the shape every real classic xref
/// table starts with. A single reverse byte scan: O(n) even on a
/// pathological buffer with many non-matching or non-standalone "xref"
/// occurrences, unlike repeatedly re-searching a shrinking prefix.
#[doc(hidden)]
pub fn find_last_valid_xref_table_start(buf: &[u8]) -> Option<usize> {
    const KEYWORD: &[u8] = b"xref";
    if buf.len() < KEYWORD.len() {
        return None;
    }
    let mut pos = buf.len() - KEYWORD.len();
    loop {
        if &buf[pos..pos + KEYWORD.len()] == KEYWORD {
            let before_ok = pos == 0 || buf[pos - 1].is_ascii_whitespace();
            let after_ok = buf
                .get(pos + KEYWORD.len())
                .is_none_or(|c| c.is_ascii_whitespace());
            if before_ok && after_ok && looks_like_xref_subsection_header(buf, pos + KEYWORD.len())
            {
                return Some(pos);
            }
        }
        if pos == 0 {
            return None;
        }
        pos -= 1;
    }
}

/// Checks that `buf[pos..]` starts (after whitespace) with two
/// whitespace-separated runs of ASCII digits — `<start-id> <count>`, the
/// first subsection header of a classic PDF cross-reference table.
fn looks_like_xref_subsection_header(buf: &[u8], pos: usize) -> bool {
    fn skip_ws(buf: &[u8], mut pos: usize) -> usize {
        while buf.get(pos).is_some_and(u8::is_ascii_whitespace) {
            pos += 1;
        }
        pos
    }
    fn skip_digits(buf: &[u8], mut pos: usize) -> usize {
        while buf.get(pos).is_some_and(u8::is_ascii_digit) {
            pos += 1;
        }
        pos
    }

    let pos = skip_ws(buf, pos);
    let after_first_digits = skip_digits(buf, pos);
    if after_first_digits == pos {
        return false; // no start-id
    }
    let sep = skip_ws(buf, after_first_digits);
    if sep == after_first_digits {
        return false; // start-id and count must be whitespace-separated
    }
    let after_count = skip_digits(buf, sep);
    if after_count == sep {
        return false; // no count
    }
    // The count run must end at whitespace/buffer-end, not run into trailing
    // garbage (e.g. a coincidental "xref\n0 6garbage" in stream content).
    buf.get(after_count).is_none_or(u8::is_ascii_whitespace)
}

fn add_repair_candidate(
    candidates: &mut Vec<Vec<u8>>,
    candidate: Option<Vec<u8>>,
    original: &[u8],
) {
    let Some(candidate) = candidate else {
        return;
    };
    if candidate.as_slice() == original {
        return;
    }
    if candidates.iter().any(|existing| existing == &candidate) {
        return;
    }
    candidates.push(candidate);
}

fn append_missing_eof_marker(buf: &[u8]) -> Option<Vec<u8>> {
    if contains_recent_eof_marker(buf) {
        return None;
    }

    let mut end = buf.len();
    while end > 0 && buf[end - 1].is_ascii_whitespace() {
        end -= 1;
    }

    if !buf[..end].ends_with(b"%%EO") {
        return None;
    }

    let mut repaired = Vec::with_capacity(end + 2);
    repaired.extend_from_slice(&buf[..end]);
    repaired.extend_from_slice(b"F\n");
    Some(repaired)
}

fn contains_recent_eof_marker(buf: &[u8]) -> bool {
    let start = buf.len().saturating_sub(1024);
    buf[start..].windows(b"%%EOF".len()).any(|w| w == b"%%EOF")
}

fn strip_leading_pdf_container_bytes(buf: &[u8]) -> Option<Vec<u8>> {
    let mut start = if buf.starts_with(&[0xEF, 0xBB, 0xBF]) {
        3
    } else {
        0
    };

    while start < buf.len() && buf[start].is_ascii_whitespace() {
        start += 1;
    }

    if start > 0 && buf[start..].starts_with(b"%PDF-") {
        Some(buf[start..].to_vec())
    } else {
        None
    }
}

/// Recover a PDF whose beginning was accidentally duplicated in front of the
/// complete file. The final `startxref` offset is relative to the later PDF
/// header, so only accept a suffix when that relative offset lands on a valid
/// classic cross-reference table. This avoids mistaking an embedded PDF stream
/// for the outer document.
#[doc(hidden)]
pub fn strip_duplicated_pdf_prefix(buf: &[u8]) -> Option<Vec<u8>> {
    let marker = b"startxref";
    let marker_pos = buf
        .windows(marker.len())
        .rposition(|window| window == marker)?;
    let mut number_start = marker_pos + marker.len();
    while buf.get(number_start).is_some_and(u8::is_ascii_whitespace) {
        number_start += 1;
    }
    let number_end = (number_start..buf.len()).find(|&index| !buf[index].is_ascii_digit())?;
    let xref_offset = std::str::from_utf8(&buf[number_start..number_end])
        .ok()?
        .parse::<usize>()
        .ok()?;

    buf.windows(b"%PDF-".len())
        .enumerate()
        .skip(1)
        .rev()
        .find_map(|(header, window)| {
            if window != b"%PDF-" {
                return None;
            }
            let xref = header.checked_add(xref_offset)?;
            (buf.get(xref..xref + 4) == Some(b"xref")
                && looks_like_xref_subsection_header(buf, xref + 4))
            .then(|| buf[header..].to_vec())
        })
}

/// Core processing pipeline operating on a pre-loaded document.

#[derive(Debug, thiserror::Error)]
pub enum PdfError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("PDF parsing error: {0}")]
    Parse(String),
    #[error("PDF is encrypted")]
    Encrypted,
    #[error("Invalid PDF structure")]
    InvalidStructure,
    #[error("Not a PDF: {0}")]
    NotAPdf(String),
}

impl From<lopdf::Error> for PdfError {
    fn from(e: lopdf::Error) -> Self {
        match e {
            lopdf::Error::IO(io_err) => PdfError::Io(io_err),
            lopdf::Error::Decryption(_)
            | lopdf::Error::InvalidPassword
            | lopdf::Error::AlreadyEncrypted
            | lopdf::Error::UnsupportedSecurityHandler(_) => PdfError::Encrypted,
            lopdf::Error::Unimplemented(msg) if msg.contains("encrypted") => PdfError::Encrypted,
            lopdf::Error::Parse(ref pe) if pe.to_string().contains("invalid file header") => {
                PdfError::NotAPdf("invalid PDF file header".to_string())
            }
            lopdf::Error::MissingXrefEntry
            | lopdf::Error::Xref(_)
            | lopdf::Error::IndirectObject { .. }
            | lopdf::Error::ObjectIdMismatch
            | lopdf::Error::InvalidObjectStream(_)
            | lopdf::Error::InvalidOffset(_) => PdfError::InvalidStructure,
            other => PdfError::Parse(other.to_string()),
        }
    }
}

/// Check whether a `lopdf::Error` represents an encryption-related failure.
pub fn is_encrypted_lopdf_error(e: &lopdf::Error) -> bool {
    matches!(
        e,
        lopdf::Error::Decryption(_)
            | lopdf::Error::InvalidPassword
            | lopdf::Error::AlreadyEncrypted
            | lopdf::Error::UnsupportedSecurityHandler(_)
    ) || matches!(e, lopdf::Error::Unimplemented(msg) if msg.contains("encrypted"))
}

// ---------------------------------------------------------------------------
// PDF validation helpers
// ---------------------------------------------------------------------------

/// Strip UTF-8 BOM and leading ASCII whitespace from a byte slice.
fn strip_bom_and_whitespace(bytes: &[u8]) -> &[u8] {
    let b = if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        &bytes[3..]
    } else {
        bytes
    };
    let start = b
        .iter()
        .position(|&c| !c.is_ascii_whitespace())
        .unwrap_or(b.len());
    &b[start..]
}

/// Case-insensitive prefix check on byte slices.
fn starts_with_ci(haystack: &[u8], needle: &[u8]) -> bool {
    if haystack.len() < needle.len() {
        return false;
    }
    haystack[..needle.len()]
        .iter()
        .zip(needle)
        .all(|(a, b)| a.eq_ignore_ascii_case(b))
}

/// Try to identify what kind of file the bytes represent.
fn detect_file_type_hint(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return "file is empty".to_string();
    }

    let trimmed = strip_bom_and_whitespace(bytes);

    // HTML
    if starts_with_ci(trimmed, b"<!doctype html")
        || starts_with_ci(trimmed, b"<html")
        || starts_with_ci(trimmed, b"<head")
        || starts_with_ci(trimmed, b"<body")
    {
        return "file appears to be HTML".to_string();
    }

    // XML (but not HTML)
    if trimmed.starts_with(b"<?xml") || trimmed.starts_with(b"<") {
        if starts_with_ci(trimmed, b"<?xml") {
            return "file appears to be XML".to_string();
        }
        if trimmed.starts_with(b"<") && !trimmed.starts_with(b"<%") {
            return "file appears to be XML".to_string();
        }
    }

    // JSON
    if trimmed.starts_with(b"{") || trimmed.starts_with(b"[") {
        return "file appears to be JSON".to_string();
    }

    // PNG
    if bytes.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
        return "file appears to be a PNG image".to_string();
    }

    // JPEG
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return "file appears to be a JPEG image".to_string();
    }

    // ZIP / Office documents
    if bytes.starts_with(&[0x50, 0x4B, 0x03, 0x04]) {
        return "file appears to be a ZIP archive (possibly an Office document)".to_string();
    }

    // If it looks like mostly printable ASCII/UTF-8, call it plain text
    let sample = &bytes[..bytes.len().min(512)];
    let printable = sample
        .iter()
        .filter(|&&b| b.is_ascii_graphic() || b.is_ascii_whitespace())
        .count();
    if printable > sample.len() * 3 / 4 {
        return "file appears to be plain text".to_string();
    }

    "file is not a PDF".to_string()
}

/// Validate that a byte buffer looks like a PDF (has `%PDF-` magic).
///
/// Scans the first 1024 bytes, allowing for a UTF-8 BOM and leading whitespace.
pub fn validate_pdf_bytes(buffer: &[u8]) -> Result<(), PdfError> {
    if buffer.is_empty() {
        return Err(PdfError::NotAPdf(detect_file_type_hint(buffer)));
    }

    let header = &buffer[..buffer.len().min(1024)];
    let trimmed = strip_bom_and_whitespace(header);

    if trimmed.starts_with(b"%PDF-") {
        Ok(())
    } else {
        Err(PdfError::NotAPdf(detect_file_type_hint(buffer)))
    }
}

/// Validate that a file on disk looks like a PDF.
///
/// Reads only the first 1024 bytes and delegates to [`validate_pdf_bytes`].
pub fn validate_pdf_file<P: AsRef<Path>>(path: P) -> Result<(), PdfError> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut buf = [0u8; 1024];
    let n = file.read(&mut buf)?;
    validate_pdf_bytes(&buf[..n])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_structure_names_are_repaired_conservatively() {
        let malformed = b"/StructTreeRoot << /S Code /K << /S H1 >> /S /P >>";
        assert_eq!(
            fix_bare_struct_names(malformed).as_ref(),
            b"/StructTreeRoot << /S /Code /K << /S /H1 >> /S /P >>"
        );

        let unknown = b"/StructTreeRoot << /S CustomRole >>";
        assert!(matches!(fix_bare_struct_names(unknown), Cow::Borrowed(_)));

        let no_structure_tree = b"<< /S Code >>";
        assert!(matches!(
            fix_bare_struct_names(no_structure_tree),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn xref_scan_rejects_tokens_without_a_subsection_header() {
        assert_eq!(
            find_last_valid_xref_table_start(b"Please refer to the xref appendix."),
            None
        );
        assert_eq!(
            find_last_valid_xref_table_start(b"startxref\n1234\n%%EOF"),
            None
        );
        assert_eq!(
            find_last_valid_xref_table_start(b"xref\n0 6garbage\n%%EOF"),
            None
        );
    }

    #[test]
    fn xref_scan_finds_the_last_valid_classic_table() {
        let bytes = b"xref\n0 3\n0000000000 65535 f \ntrailer\nsee the xref\n";
        assert_eq!(find_last_valid_xref_table_start(bytes), Some(0));
        assert!(recover_startxref_pointer(b"Please refer to the xref appendix.").is_none());
    }

    #[test]
    fn duplicated_prefix_repair_requires_a_matching_relative_xref() {
        let complete = b"%PDF-1.4\nxxxxx\nxref\n0 1\n0000000000 65535 f \nstartxref\n15\n%%EOF\n";
        let mut duplicated = b"%PDF-1.4\nbroken prefix bytes".to_vec();
        duplicated.extend_from_slice(complete);
        assert_eq!(
            strip_duplicated_pdf_prefix(&duplicated).as_deref(),
            Some(complete.as_slice())
        );
        assert!(strip_duplicated_pdf_prefix(
            b"%PDF-1.4\nouter %PDF-1.4\nembedded\nxref\n0 1\nstartxref\n7\n%%EOF\n"
        )
        .is_none());
    }
}
