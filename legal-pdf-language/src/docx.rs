use crate::{Error, Result};
use quick_xml::events::{BytesStart, Event};
use quick_xml::name::ResolveResult;
use quick_xml::reader::NsReader;
use regex::Regex;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::env;
use std::io::{Cursor, Read, Write};
use std::process::Command;
use std::sync::LazyLock;
use std::time::Duration;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

pub const MAX_DOCX_SUPRA_BYTES: usize = 25 * 1024 * 1024;
const W_NS: &str = "http://schemas.openxmlformats.org/wordprocessingml/2006/main";

fn file_bytes<'a>(files: &'a [(String, Vec<u8>)], name: &str) -> Option<&'a [u8]> {
    files
        .iter()
        .find(|(entry, _)| entry == name)
        .map(|(_, bytes)| bytes.as_slice())
}

fn replace_file(files: &mut Vec<(String, Vec<u8>)>, name: &str, bytes: Vec<u8>) {
    if let Some((_, value)) = files.iter_mut().find(|(entry, _)| entry == name) {
        *value = bytes;
    } else {
        files.push((name.to_owned(), bytes));
    }
}

#[derive(Clone)]
struct XmlAttribute {
    namespace: Option<String>,
    local: String,
    value: String,
}

#[derive(Clone)]
struct XmlElement {
    namespace: Option<String>,
    local: String,
    attributes: Vec<XmlAttribute>,
    children: Vec<XmlNode>,
    self_closing: bool,
}

#[derive(Clone)]
enum XmlNode {
    Element(XmlElement),
    Raw(Event<'static>),
}

struct XmlDocument {
    nodes: Vec<XmlNode>,
}

fn namespace_value(value: ResolveResult<'_>) -> Option<String> {
    match value {
        ResolveResult::Bound(namespace) => {
            Some(String::from_utf8_lossy(namespace.as_ref()).into_owned())
        }
        ResolveResult::Unbound | ResolveResult::Unknown(_) => None,
    }
}

fn decode_attribute(
    raw: &quick_xml::events::attributes::Attribute<'_>,
    decoder: quick_xml::encoding::Decoder,
) -> Result<String> {
    #[allow(deprecated)]
    raw.decode_and_unescape_value(decoder)
        .map(|value| value.into_owned())
        .map_err(|error| Error::Message(format!("XML attribute decoding failed: {error}")))
}

fn parse_element(
    reader: &NsReader<&[u8]>,
    start: &BytesStart<'_>,
    namespace: Option<String>,
    self_closing: bool,
) -> Result<XmlElement> {
    let local = std::str::from_utf8(start.local_name().as_ref())
        .map_err(|error| Error::Message(format!("XML name is not UTF-8: {error}")))?
        .to_owned();
    let mut attributes = Vec::new();
    for raw in start.attributes().with_checks(false) {
        let raw = raw.map_err(|error| Error::Message(format!("XML attribute failed: {error}")))?;
        let (resolved, local_name) = reader.resolver().resolve_attribute(raw.key);
        let local = std::str::from_utf8(local_name.as_ref())
            .map_err(|error| Error::Message(format!("XML attribute name is not UTF-8: {error}")))?
            .to_owned();
        attributes.push(XmlAttribute {
            namespace: namespace_value(resolved),
            local,
            value: decode_attribute(&raw, reader.decoder())?,
        });
    }
    Ok(XmlElement {
        namespace,
        local,
        attributes,
        children: Vec::new(),
        self_closing,
    })
}

fn parse_xml(raw: &[u8]) -> Result<XmlDocument> {
    let mut reader = NsReader::from_reader(raw);
    reader.config_mut().trim_text(false);
    let mut buffer = Vec::new();
    let mut roots = Vec::new();
    let mut stack = Vec::<XmlElement>::new();
    loop {
        let event = reader.read_event_into(&mut buffer)?;
        match event {
            Event::Start(start) => {
                let namespace = namespace_value(reader.resolver().resolve_element(start.name()).0);
                stack.push(parse_element(&reader, &start, namespace, false)?);
            }
            Event::Empty(start) => {
                let namespace = namespace_value(reader.resolver().resolve_element(start.name()).0);
                let node = XmlNode::Element(parse_element(&reader, &start, namespace, true)?);
                if let Some(parent) = stack.last_mut() {
                    parent.children.push(node);
                } else {
                    roots.push(node);
                }
            }
            Event::End(_) => {
                let element = stack
                    .pop()
                    .ok_or_else(|| Error::Message("XML has an unmatched end tag".to_owned()))?;
                let node = XmlNode::Element(element);
                if let Some(parent) = stack.last_mut() {
                    parent.children.push(node);
                } else {
                    roots.push(node);
                }
            }
            Event::Eof => break,
            Event::Decl(_) => {}
            other => {
                let node = XmlNode::Raw(other.into_owned());
                if let Some(parent) = stack.last_mut() {
                    parent.children.push(node);
                } else {
                    roots.push(node);
                }
            }
        }
        buffer.clear();
    }
    if !stack.is_empty() {
        return Err(Error::Message("XML has an unclosed element".to_owned()));
    }
    Ok(XmlDocument { nodes: roots })
}

impl XmlDocument {
    fn root(&self) -> Result<&XmlElement> {
        self.nodes
            .iter()
            .find_map(|node| match node {
                XmlNode::Element(element) => Some(element),
                XmlNode::Raw(_) => None,
            })
            .ok_or_else(|| Error::Message("XML has no root element".to_owned()))
    }
}

impl XmlElement {
    fn is(&self, namespace: &str, local: &str) -> bool {
        self.namespace.as_deref() == Some(namespace) && self.local == local
    }

    fn attribute(&self, namespace: Option<&str>, local: &str) -> Option<&str> {
        self.attributes
            .iter()
            .find(|attribute| {
                attribute.local == local
                    && namespace
                        .is_none_or(|namespace| attribute.namespace.as_deref() == Some(namespace))
            })
            .map(|attribute| attribute.value.as_str())
    }

    fn direct_elements(&self) -> impl Iterator<Item = &XmlElement> {
        self.children.iter().filter_map(|node| match node {
            XmlNode::Element(element) => Some(element),
            XmlNode::Raw(_) => None,
        })
    }
}

fn raw_text(event: &Event<'_>) -> Result<Option<String>> {
    match event {
        Event::Text(text) => {
            let decoded = text
                .decode()
                .map_err(|error| Error::Message(format!("XML text decoding failed: {error}")))?;
            quick_xml::escape::unescape(&decoded)
                .map(|value| Some(value.into_owned()))
                .map_err(|error| Error::Message(format!("XML text unescaping failed: {error}")))
        }
        Event::CData(text) => text
            .decode()
            .map(|value| Some(value.into_owned()))
            .map_err(|error| Error::Message(format!("XML text decoding failed: {error}"))),
        Event::GeneralRef(reference) => {
            if let Some(value) = reference
                .resolve_char_ref()
                .map_err(|error| Error::Message(format!("XML reference failed: {error}")))?
            {
                return Ok(Some(value.to_string()));
            }
            let name = reference
                .decode()
                .map_err(|error| Error::Message(format!("XML reference failed: {error}")))?;
            Ok(Some(
                match name.as_ref() {
                    "lt" => "<",
                    "gt" => ">",
                    "amp" => "&",
                    "apos" => "'",
                    "quot" => "\"",
                    _ => return Ok(Some(format!("&{name};"))),
                }
                .to_owned(),
            ))
        }
        _ => Ok(None),
    }
}

fn element_text(element: &XmlElement) -> Result<String> {
    let mut output = String::new();
    for child in &element.children {
        match child {
            XmlNode::Raw(event) => {
                if let Some(value) = raw_text(event)? {
                    output.push_str(&value);
                }
            }
            XmlNode::Element(child) => output.push_str(&element_text(child)?),
        }
    }
    Ok(output)
}

fn walk_elements<'a>(element: &'a XmlElement, output: &mut Vec<&'a XmlElement>) {
    output.push(element);
    for child in element.direct_elements() {
        walk_elements(child, output);
    }
}

fn docx_paragraph_text(element: &XmlElement, output: &mut String) -> Result<()> {
    if element.is(W_NS, "del") {
        return Ok(());
    }
    if element.is(W_NS, "t") {
        output.push_str(&element_text(element)?);
        return Ok(());
    }
    for child in element.direct_elements() {
        docx_paragraph_text(child, output)?;
    }
    Ok(())
}

fn normalize_docx_text(text: &str) -> String {
    let text = text
        .replace(['\u{201c}', '\u{201d}'], "\"")
        .replace(['\u{2018}', '\u{2019}'], "'")
        .replace(['\u{00a0}', '\u{2007}', '\u{202f}'], " ");
    legal_structure::normalize_javascript_whitespace(&text)
}

fn normalized_docx_paragraph(paragraph: &XmlElement) -> Result<String> {
    let mut text = String::new();
    docx_paragraph_text(paragraph, &mut text)?;
    Ok(normalize_docx_text(&text))
}

fn tolerant_docx_paragraphs(xml: &str) -> Result<Vec<String>> {
    static PARAGRAPHS: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?s)<w:p\b[^>]*>.*?</w:p>").expect("literal DOCX regex"));
    static DELETIONS: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?s)<w:del\b[^>]*>.*?</w:del>").expect("literal DOCX regex"));
    static TEXTS: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?s)<w:t\b[^>]*>(.*?)</w:t>").expect("literal DOCX regex"));
    PARAGRAPHS
        .find_iter(xml)
        .map(|paragraph| {
            let accepted = DELETIONS.replace_all(paragraph.as_str(), "");
            let mut value = String::new();
            for captures in TEXTS.captures_iter(&accepted) {
                value.push_str(
                    &quick_xml::escape::unescape(captures.get(1).unwrap().as_str())
                        .map_err(|error| Error::Message(error.to_string()))?,
                );
            }
            Ok(normalize_docx_text(&value))
        })
        .collect()
}

static SUPRA: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        &[
            r"(?i)supra,?",
            legal_structure::JS_WHITESPACE_CLASS,
            r"{1,4}(?:note|nn?\.?)",
            legal_structure::JS_WHITESPACE_CLASS,
            r"{1,4}([0-9]+)",
        ]
        .concat(),
    )
    .expect("literal DOCX supra regex")
});
static NUMBERED_SUPRA: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(supra)[^\n\r\u{2028}\u{2029}]{0,40}?[0-9]+")
        .expect("literal numbered supra regex")
});
static NUMBERING_RESTART: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)<w:numRestart\b").expect("literal DOCX numbering regex"));
static PARAGRAPH: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<w:p\b.*?</w:p>").expect("literal DOCX paragraph regex"));
static RUN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<w:r\b([^>]*)>(.*?)</w:r>").expect("literal DOCX run regex"));
static TEXT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<w:t\b[^>]*>(.*?)</w:t>").expect("literal DOCX text regex"));
static FIELD_MARKER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"<w:fldChar\b[^>]*\bw:fldCharType=(?:"(begin|end)"|'(begin|end)')[^>]*/?>"#)
        .expect("literal DOCX field regex")
});
static NOTEREF: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)NOTEREF").expect("literal NOTEREF regex"));
static RUN_PROPERTIES: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<w:rPr\b.*?</w:rPr>").expect("literal run properties regex"));
static FOOTNOTE_REFERENCE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"<w:footnoteReference\b[^>]*\bw:id=(?:"(-?[0-9]+)"|'(-?[0-9]+)')[^>]*/?>"#)
        .expect("literal DOCX footnote reference regex")
});
static CUSTOM_MARK: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?-u:\b)w:customMarkFollows=").expect("literal custom mark regex")
});
static BOOKMARK_ID: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"<w:bookmark(?:Start|End)\b[^>]*\bw:id=(?:"([0-9]+)"|'([0-9]+)')"#)
        .expect("literal DOCX bookmark regex")
});
static BOOKMARK_NAME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"<w:bookmarkStart\b[^>]*\bw:name=(?:"([^"]*)"|'([^']*)')"#)
        .expect("literal DOCX bookmark name regex")
});

#[derive(Debug)]
pub struct DocxSupraCleanup {
    pub bytes: Vec<u8>,
    pub detected: usize,
    pub converted: usize,
    pub already_linked: usize,
    pub review_required: usize,
    pub bookmarks_added: usize,
    pub restarted_numbering: bool,
    pub unsafe_or_split_fields: usize,
}

struct SupraAnalysis {
    detected: usize,
    already_linked: usize,
    ordinals: BTreeSet<usize>,
}

#[derive(Debug)]
struct ParagraphTextNode {
    text: String,
    visible_start: usize,
    visible_end: usize,
    xml_start: usize,
    run_start: usize,
    run_end: usize,
    run_attributes: String,
    run_properties: String,
    safe_to_replace: bool,
}

fn xml_text(value: &str) -> Result<String> {
    quick_xml::escape::unescape(value)
        .map(|value| value.into_owned())
        .map_err(|error| Error::Message(format!("XML text unescaping failed: {error}")))
}

fn escape_xml_text(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn element_is_open(xml: &str, offset: usize, tag: &str) -> bool {
    let prior = &xml[..offset];
    let open = prior
        .rfind(&format!("<{tag} "))
        .max(prior.rfind(&format!("<{tag}>")));
    open.is_some_and(|open| {
        prior
            .rfind(&format!("</{tag}>"))
            .is_none_or(|close| open > close)
    })
}

fn javascript_iu_word(character: Option<char>) -> bool {
    character.is_some_and(|character| {
        character.is_ascii_alphanumeric()
            || character == '_'
            || matches!(character, '\u{017f}' | '\u{212a}')
    })
}

fn javascript_iu_word_bounded(text: &str, start: usize, end: usize) -> bool {
    javascript_iu_word(text[..start].chars().next_back())
        != javascript_iu_word(text[start..].chars().next())
        && javascript_iu_word(text[..end].chars().next_back())
            != javascript_iu_word(text[end..].chars().next())
}

fn field_spans(paragraph: &str) -> Vec<(usize, usize)> {
    let mut stack = Vec::new();
    let mut spans = Vec::new();
    for marker in FIELD_MARKER.captures_iter(paragraph) {
        let whole = marker.get(0).unwrap();
        if marker.get(1).or_else(|| marker.get(2)).unwrap().as_str() == "begin" {
            stack.push(whole.start());
        } else if let Some(start) = stack.pop() {
            let field = &paragraph[start..whole.start()];
            if NOTEREF
                .find_iter(field)
                .any(|value| javascript_iu_word_bounded(field, value.start(), value.end()))
            {
                spans.push((start, whole.end()));
            }
        }
    }
    spans
}

fn paragraph_text_nodes(
    xml: &str,
    paragraph: &str,
    paragraph_offset: usize,
) -> Result<Vec<ParagraphTextNode>> {
    let mut nodes = Vec::new();
    let mut visible = 0;
    for run in RUN.captures_iter(paragraph) {
        let whole = run.get(0).unwrap();
        let body = run.get(2).unwrap();
        let texts = TEXT.captures_iter(body.as_str()).collect::<Vec<_>>();
        let properties = RUN_PROPERTIES
            .find(body.as_str())
            .map_or("", |value| value.as_str());
        let only_text = texts.len() == 1
            && legal_structure::normalize_javascript_whitespace(
                &body.as_str().replacen(properties, "", 1).replacen(
                    texts[0].get(0).unwrap().as_str(),
                    "",
                    1,
                ),
            )
            .is_empty();
        let safe = only_text
            && ["w:hyperlink", "w:fldSimple", "w:ins", "w:del"]
                .iter()
                .all(|tag| !element_is_open(xml, paragraph_offset + whole.start(), tag));
        for text in texts {
            let value = xml_text(text.get(1).unwrap().as_str())?;
            nodes.push(ParagraphTextNode {
                visible_start: visible,
                visible_end: visible + value.len(),
                xml_start: whole.start()
                    + (body.start() - whole.start())
                    + text.get(0).unwrap().start(),
                run_start: whole.start(),
                run_end: whole.end(),
                run_attributes: run.get(1).unwrap().as_str().to_owned(),
                run_properties: properties.to_owned(),
                text: value,
                safe_to_replace: safe,
            });
            visible = nodes.last().unwrap().visible_end;
        }
    }
    Ok(nodes)
}

fn analyze_supras(xml: &str) -> Result<SupraAnalysis> {
    let mut analysis = SupraAnalysis {
        detected: 0,
        already_linked: 0,
        ordinals: BTreeSet::new(),
    };
    for paragraph in PARAGRAPH.find_iter(xml) {
        let nodes = paragraph_text_nodes(xml, paragraph.as_str(), paragraph.start())?;
        let visible = nodes
            .iter()
            .map(|node| node.text.as_str())
            .collect::<String>();
        let fields = field_spans(paragraph.as_str());
        for matched in SUPRA.captures_iter(&visible) {
            let whole = matched.get(0).unwrap();
            if !javascript_iu_word_bounded(&visible, whole.start(), whole.end()) {
                continue;
            }
            analysis.detected += 1;
            let number = matched.get(1).unwrap();
            let node = nodes.iter().find(|node| {
                node.visible_start <= number.start() && node.visible_end >= number.end()
            });
            if node.is_some_and(|node| {
                fields
                    .iter()
                    .any(|&(start, end)| start <= node.xml_start && node.xml_start < end)
            }) {
                analysis.already_linked += 1;
            } else if let Ok(ordinal) = number.as_str().parse::<usize>() {
                if ordinal > 0 {
                    analysis.ordinals.insert(ordinal);
                }
            }
        }
    }
    Ok(analysis)
}

fn contains_numbered_supra(xml: &str) -> Result<bool> {
    for paragraph in PARAGRAPH.find_iter(xml) {
        let mut visible = String::new();
        for text in TEXT.captures_iter(paragraph.as_str()) {
            visible.push_str(&xml_text(text.get(1).unwrap().as_str())?);
        }
        if NUMBERED_SUPRA.captures_iter(&visible).any(|capture| {
            let whole = capture.get(0).unwrap();
            let word = capture.get(1).unwrap();
            javascript_iu_word_bounded(&visible, word.start(), word.end())
                && javascript_iu_word_bounded(&visible, word.start(), whole.end())
        }) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn read_docx_files(bytes: &[u8], wanted: Option<&[&str]>) -> Result<Vec<(String, Vec<u8>)>> {
    const MAX_EXPANDED_BYTES: u64 = 96 * 1024 * 1024;
    const MAX_XML_PART_BYTES: u64 = 16 * 1024 * 1024;
    const MAX_XML_TOTAL_BYTES: u64 = 32 * 1024 * 1024;
    if bytes.is_empty() || bytes.len() > MAX_DOCX_SUPRA_BYTES {
        return Err(Error::Message(
            "DOCX is empty or exceeds the read limit".to_owned(),
        ));
    }
    let mut archive = ZipArchive::new(Cursor::new(bytes))?;
    let mut files = Vec::with_capacity(wanted.map_or(archive.len(), |names| names.len()));
    let mut seen = HashSet::with_capacity(archive.len());
    let mut expanded = 0_u64;
    let mut xml_expanded = 0_u64;
    let mut declared_expanded = 0_u64;
    let mut declared_xml_expanded = 0_u64;
    let mut file_count = 0;
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        let name = entry.name().replace('\\', "/");
        if name.contains('\0')
            || name.starts_with('/')
            || name.as_bytes().get(1) == Some(&b':')
            || name.split('/').any(|component| component == "..")
        {
            return Err(Error::Message(
                "DOCX contains an unsafe package path".to_owned(),
            ));
        }
        let lower_name = name.to_ascii_lowercase();
        let xml = lower_name.ends_with(".xml") || lower_name.ends_with(".xml.rels");
        if !entry.is_dir() {
            file_count += 1;
            if file_count > 2_048 {
                return Err(Error::Message(
                    "DOCX has too many package entries".to_owned(),
                ));
            }
            declared_expanded = declared_expanded
                .checked_add(entry.size())
                .filter(|&size| size <= MAX_EXPANDED_BYTES)
                .ok_or_else(|| Error::Message("DOCX exceeds the expanded read limit".to_owned()))?;
            if xml {
                if entry.size() > MAX_XML_PART_BYTES {
                    return Err(Error::Message(
                        "DOCX XML part exceeds the read limit".to_owned(),
                    ));
                }
                declared_xml_expanded = declared_xml_expanded
                    .checked_add(entry.size())
                    .filter(|&size| size <= MAX_XML_TOTAL_BYTES)
                    .ok_or_else(|| {
                        Error::Message("DOCX XML parts exceed the read limit".to_owned())
                    })?;
            }
        }
        if !seen.insert(name.clone()) {
            return Err(Error::Message(format!(
                "DOCX contains duplicate package part {name}"
            )));
        }
        if wanted.is_some_and(|wanted| !wanted.contains(&name.as_str())) {
            continue;
        }
        let remaining = MAX_EXPANDED_BYTES.saturating_sub(expanded);
        let limit = if xml {
            remaining
                .min(MAX_XML_PART_BYTES)
                .min(MAX_XML_TOTAL_BYTES.saturating_sub(xml_expanded))
        } else {
            remaining
        };
        let mut value = Vec::with_capacity(entry.size().min(limit).min(1024 * 1024) as usize);
        entry
            .by_ref()
            .take(limit + 1)
            .read_to_end(&mut value)
            .map_err(|error| Error::io(&name, error))?;
        if value.len() as u64 > limit {
            let message = if xml {
                "DOCX XML part exceeds the read limit"
            } else {
                "DOCX exceeds the expanded read limit"
            };
            return Err(Error::Message(message.to_owned()));
        }
        expanded += value.len() as u64;
        if xml {
            xml_expanded += value.len() as u64;
        }
        files.push((name, value));
    }
    Ok(files)
}

fn write_docx_files(files: &[(String, Vec<u8>)]) -> Result<Vec<u8>> {
    let mut archive = ZipWriter::new(Cursor::new(Vec::new()));
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    for (name, value) in files {
        if name.ends_with('/') {
            archive.add_directory(name, options)?;
        } else {
            archive.start_file(name, options)?;
            archive
                .write_all(value)
                .map_err(|error| Error::io("DOCX output", error))?;
        }
    }
    Ok(archive.finish()?.into_inner())
}

fn docx_part(files: &[(String, Vec<u8>)], name: &str) -> Option<String> {
    file_bytes(files, name).map(|bytes| String::from_utf8_lossy(bytes).into_owned())
}

fn authority_text(
    element: &XmlElement,
    text: &mut String,
    references: &mut Vec<(i64, usize)>,
) -> Result<()> {
    if element.is(W_NS, "del") {
        return Ok(());
    }
    if element.is(W_NS, "t") {
        text.push_str(&element_text(element)?);
        return Ok(());
    }
    if element.is(W_NS, "tab") {
        text.push('\t');
        return Ok(());
    }
    if element.is(W_NS, "br") || element.is(W_NS, "cr") {
        text.push('\n');
        return Ok(());
    }
    if element.is(W_NS, "footnoteReference") {
        if let Some(id) = element.attribute(None, "id") {
            references.push((
                id.parse()
                    .map_err(|_| Error::Message(format!("Invalid DOCX footnote id {id}")))?,
                legal_structure::utf16_len(text),
            ));
        }
        return Ok(());
    }
    for child in element.direct_elements() {
        authority_text(child, text, references)?;
    }
    Ok(())
}

/// Return body paragraphs followed by Word footnotes in citation-review order.
pub fn docx_to_toa_text_units(bytes: &[u8]) -> Result<Vec<serde_json::Value>> {
    let files = read_docx_files(bytes, Some(&["word/document.xml", "word/footnotes.xml"]))?;
    let document = parse_xml(
        file_bytes(&files, "word/document.xml")
            .ok_or_else(|| Error::Message("DOCX has no word/document.xml".to_owned()))?,
    )?;
    let body = document
        .root()?
        .direct_elements()
        .find(|element| element.is(W_NS, "body"))
        .ok_or_else(|| Error::Message("DOCX has no document body".to_owned()))?;
    let mut elements = Vec::new();
    walk_elements(body, &mut elements);
    let mut body_units = Vec::new();
    for (ordinal, paragraph) in elements
        .into_iter()
        .filter(|element| element.is(W_NS, "p"))
        .enumerate()
    {
        let mut text = String::new();
        let mut references = Vec::new();
        authority_text(paragraph, &mut text, &mut references)?;
        if !text.trim().is_empty() {
            body_units.push((ordinal, text, references));
        }
    }

    let mut footnotes = Vec::new();
    let mut footnote_numbers = HashMap::new();
    if let Some(xml) = file_bytes(&files, "word/footnotes.xml") {
        let document = parse_xml(xml)?;
        let mut elements = Vec::new();
        walk_elements(document.root()?, &mut elements);
        for footnote in elements
            .into_iter()
            .filter(|element| element.is(W_NS, "footnote"))
        {
            let Some(raw_id) = footnote.attribute(None, "id") else {
                continue;
            };
            let raw_id: i64 = raw_id
                .parse()
                .map_err(|_| Error::Message(format!("Invalid DOCX footnote id {raw_id}")))?;
            if raw_id <= 0 {
                continue;
            }
            let mut text = String::new();
            authority_text(footnote, &mut text, &mut Vec::new())?;
            if text.trim().is_empty() {
                continue;
            }
            let ordinal = footnotes.len() + 1;
            footnote_numbers.insert(raw_id, ordinal);
            footnotes.push((raw_id, ordinal, text));
        }
    }

    let mut units = Vec::with_capacity(body_units.len() + footnotes.len());
    units.extend(body_units.into_iter().map(|(ordinal, text, references)| {
        let references = references
            .into_iter()
            .map(|(id, offset)| {
                serde_json::json!([
                    footnote_numbers
                        .get(&id)
                        .copied()
                        .map_or(id, |id| id as i64),
                    offset
                ])
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "key": format!("body:{ordinal}"),
            "kind": "body",
            "ordinal": ordinal,
            "footnote_id": serde_json::Value::Null,
            "text": text,
            "footnote_refs": references,
        })
    }));
    units.extend(footnotes.into_iter().map(|(raw_id, ordinal, text)| {
        serde_json::json!({
            "key": format!("footnote:{raw_id}"),
            "kind": "footnote",
            "ordinal": ordinal,
            "footnote_id": ordinal,
            "text": text,
            "footnote_refs": [],
        })
    }));
    Ok(units)
}

fn footnote_reference_ids(document: &str) -> Vec<usize> {
    FOOTNOTE_REFERENCE
        .captures_iter(document)
        .filter(|matched| !CUSTOM_MARK.is_match(matched.get(0).unwrap().as_str()))
        .filter_map(|matched| {
            matched
                .get(1)
                .or_else(|| matched.get(2))?
                .as_str()
                .parse()
                .ok()
        })
        .filter(|id| *id > 0)
        .collect()
}

fn add_target_bookmarks(
    mut xml: String,
    reference_ids: &[usize],
    ordinals: &BTreeSet<usize>,
) -> (String, HashMap<usize, String>, usize) {
    let mut bookmark_id = BOOKMARK_ID
        .captures_iter(&xml)
        .filter_map(|matched| {
            matched
                .get(1)
                .or_else(|| matched.get(2))?
                .as_str()
                .parse()
                .ok()
        })
        .max()
        .unwrap_or(0)
        + 1;
    let existing = BOOKMARK_NAME
        .captures_iter(&xml)
        .filter_map(|matched| matched.get(1).or_else(|| matched.get(2)))
        .map(|name| name.as_str().to_owned())
        .collect::<HashSet<_>>();
    let mut targets = HashMap::<usize, (usize, usize)>::new();
    for run in RUN.find_iter(&xml) {
        for matched in FOOTNOTE_REFERENCE.captures_iter(run.as_str()) {
            let Some(reference_id) = matched
                .get(1)
                .or_else(|| matched.get(2))
                .and_then(|id| id.as_str().parse().ok())
            else {
                continue;
            };
            targets
                .entry(reference_id)
                .or_insert((run.start(), run.end()));
        }
    }
    let mut names = HashMap::new();
    let mut edits = BTreeMap::<(usize, usize), Vec<(usize, String)>>::new();
    let mut added = 0;
    for &ordinal in ordinals {
        let Some(&reference_id) = ordinal
            .checked_sub(1)
            .and_then(|index| reference_ids.get(index))
        else {
            continue;
        };
        let name = format!("MikeSupraNote{ordinal}");
        names.insert(ordinal, name.clone());
        if existing.contains(&name) {
            continue;
        }
        let Some(&target) = targets.get(&reference_id) else {
            names.remove(&ordinal);
            continue;
        };
        edits.entry(target).or_default().push((bookmark_id, name));
        bookmark_id += 1;
        added += 1;
    }
    for ((start, end), bookmarks) in edits.into_iter().rev() {
        let mut replacement = String::new();
        for (id, name) in &bookmarks {
            replacement.push_str(&format!(
                r#"<w:bookmarkStart w:id="{id}" w:name="{name}"/>"#
            ));
        }
        replacement.push_str(&xml[start..end]);
        for (id, _) in bookmarks.iter().rev() {
            replacement.push_str(&format!(r#"<w:bookmarkEnd w:id="{id}"/>"#));
        }
        xml.replace_range(start..end, &replacement);
    }
    (xml, names, added)
}

fn plain_run(attributes: &str, properties: &str, text: &str) -> String {
    if text.is_empty() {
        String::new()
    } else {
        format!(
            r#"<w:r{attributes}>{properties}<w:t xml:space="preserve">{}</w:t></w:r>"#,
            escape_xml_text(text)
        )
    }
}

fn noteref_field(attributes: &str, properties: &str, name: &str, number: &str) -> String {
    format!(
        concat!(
            r#"<w:r{attributes}>{properties}<w:fldChar w:fldCharType="begin"/></w:r>"#,
            r#"<w:r{attributes}>{properties}<w:instrText xml:space="preserve"> NOTEREF {name} \h </w:instrText></w:r>"#,
            r#"<w:r{attributes}>{properties}<w:fldChar w:fldCharType="separate"/></w:r>"#,
            r#"{number_run}<w:r{attributes}>{properties}<w:fldChar w:fldCharType="end"/></w:r>"#
        ),
        attributes = attributes,
        properties = properties,
        name = name,
        number_run = plain_run(attributes, properties, number)
    )
}

fn convert_safe_paragraphs(xml: &str, names: &HashMap<usize, String>) -> Result<(String, usize)> {
    let mut output = String::with_capacity(xml.len());
    let mut cursor = 0;
    let mut converted = 0;
    for paragraph in PARAGRAPH.find_iter(xml) {
        output.push_str(&xml[cursor..paragraph.start()]);
        let nodes = paragraph_text_nodes(xml, paragraph.as_str(), paragraph.start())?;
        let visible = nodes
            .iter()
            .map(|node| node.text.as_str())
            .collect::<String>();
        let fields = field_spans(paragraph.as_str());
        let mut candidates = BTreeMap::<usize, Vec<(&ParagraphTextNode, usize, &str, &str)>>::new();
        for matched in SUPRA.captures_iter(&visible) {
            let whole = matched.get(0).unwrap();
            if !javascript_iu_word_bounded(&visible, whole.start(), whole.end()) {
                continue;
            }
            let number = matched.get(1).unwrap();
            let Ok(ordinal) = number.as_str().parse::<usize>() else {
                continue;
            };
            let Some(name) = names.get(&ordinal) else {
                continue;
            };
            let Some(node) = nodes.iter().find(|node| {
                node.visible_start <= number.start() && node.visible_end >= number.end()
            }) else {
                continue;
            };
            if !node.safe_to_replace
                || fields
                    .iter()
                    .any(|&(start, end)| start <= node.xml_start && node.xml_start < end)
            {
                continue;
            }
            candidates.entry(node.run_start).or_default().push((
                node,
                number.start(),
                number.as_str(),
                name,
            ));
        }
        let mut edits = Vec::new();
        for rows in candidates.values_mut() {
            rows.sort_by_key(|row| row.1);
            let node = rows[0].0;
            let mut replacement = String::new();
            let mut text_cursor = 0;
            for (_, start, number, name) in rows {
                let local = *start - node.visible_start;
                replacement.push_str(&plain_run(
                    &node.run_attributes,
                    &node.run_properties,
                    &node.text[text_cursor..local],
                ));
                replacement.push_str(&noteref_field(
                    &node.run_attributes,
                    &node.run_properties,
                    name,
                    number,
                ));
                text_cursor = local + number.len();
                converted += 1;
            }
            replacement.push_str(&plain_run(
                &node.run_attributes,
                &node.run_properties,
                &node.text[text_cursor..],
            ));
            edits.push((node.run_start, node.run_end, replacement));
        }
        let mut next = paragraph.as_str().to_owned();
        edits.sort_by_key(|edit| std::cmp::Reverse(edit.0));
        for (start, end, replacement) in edits {
            next.replace_range(start..end, &replacement);
        }
        output.push_str(&next);
        cursor = paragraph.end();
    }
    output.push_str(&xml[cursor..]);
    Ok((output, converted))
}

pub fn has_docx_supra_references(bytes: &[u8]) -> Result<bool> {
    let files = read_docx_files(bytes, Some(&["word/footnotes.xml"]))?;
    if let Some(xml) = docx_part(&files, "word/footnotes.xml") {
        if contains_numbered_supra(&xml)? {
            return Ok(true);
        }
    }
    let files = read_docx_files(bytes, Some(&["word/document.xml"]))?;
    docx_part(&files, "word/document.xml").map_or(Ok(false), |xml| contains_numbered_supra(&xml))
}

pub fn fix_docx_supra_cross_references(bytes: &[u8]) -> Result<DocxSupraCleanup> {
    let files = read_docx_files(
        bytes,
        Some(&[
            "word/document.xml",
            "word/footnotes.xml",
            "word/settings.xml",
        ]),
    )?;
    let document = docx_part(&files, "word/document.xml").ok_or_else(|| {
        Error::Message("DOCX does not contain ordinary Word footnotes".to_owned())
    })?;
    let footnotes = docx_part(&files, "word/footnotes.xml").ok_or_else(|| {
        Error::Message("DOCX does not contain ordinary Word footnotes".to_owned())
    })?;
    let body = analyze_supras(&document)?;
    let notes = analyze_supras(&footnotes)?;
    let detected = body.detected + notes.detected;
    let already_linked = body.already_linked + notes.already_linked;
    let restarted = NUMBERING_RESTART.is_match(&document)
        || docx_part(&files, "word/settings.xml")
            .is_some_and(|settings| NUMBERING_RESTART.is_match(&settings));
    let unchanged = |review_required, restarted_numbering| DocxSupraCleanup {
        bytes: bytes.to_vec(),
        detected,
        converted: 0,
        already_linked,
        review_required,
        bookmarks_added: 0,
        restarted_numbering,
        unsafe_or_split_fields: review_required,
    };
    if detected == 0 || restarted {
        return Ok(unchanged(
            detected.saturating_sub(already_linked),
            restarted,
        ));
    }
    let ordinals = body.ordinals.union(&notes.ordinals).copied().collect();
    let reference_ids = footnote_reference_ids(&document);
    let (bookmarked, names, bookmarks_added) =
        add_target_bookmarks(document, &reference_ids, &ordinals);
    let (next_document, body_converted) = convert_safe_paragraphs(&bookmarked, &names)?;
    let (next_footnotes, note_converted) = convert_safe_paragraphs(&footnotes, &names)?;
    let converted = body_converted + note_converted;
    let review_required = detected.saturating_sub(converted + already_linked);
    if converted == 0 {
        return Ok(unchanged(review_required, false));
    }
    let mut files = read_docx_files(bytes, None)?;
    replace_file(&mut files, "word/document.xml", next_document.into_bytes());
    replace_file(
        &mut files,
        "word/footnotes.xml",
        next_footnotes.into_bytes(),
    );
    Ok(DocxSupraCleanup {
        bytes: write_docx_files(&files)?,
        detected,
        converted,
        already_linked,
        review_required,
        bookmarks_added,
        restarted_numbering: false,
        unsafe_or_split_fields: review_required,
    })
}

fn nested_word_elements<'a>(element: &'a XmlElement, wanted: &str) -> Vec<&'a XmlElement> {
    let mut found = Vec::new();
    for child in element.direct_elements() {
        if child.is(W_NS, wanted) {
            found.push(child);
        } else if child.is(W_NS, "sdt") || child.is(W_NS, "sdtContent") {
            found.extend(nested_word_elements(child, wanted));
        }
    }
    found
}

fn paragraphs_under<'a>(
    element: &'a XmlElement,
    indexed: &HashMap<*const XmlElement, (usize, usize)>,
) -> Vec<(usize, usize)> {
    let mut found = Vec::new();
    let mut pending = vec![element];
    while let Some(current) = pending.pop() {
        if let Some(index) = indexed.get(&(current as *const XmlElement)) {
            found.push(*index);
        } else {
            let children = current.direct_elements().collect::<Vec<_>>();
            pending.extend(children.into_iter().rev());
        }
    }
    found
}

fn word_int(element: Option<&XmlElement>, name: &str) -> Option<usize> {
    element
        .and_then(|element| element.direct_elements().find(|child| child.is(W_NS, name)))
        .and_then(|element| element.attribute(None, "val"))
        .and_then(|value| value.parse().ok())
}

fn strip_heading_numbering(xml: &str) -> String {
    static PARAGRAPH_PROPERTIES: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?s)<w:pPr>(.*?)</w:pPr>").expect("literal DOCX heading regex")
    });
    static HEADING_STYLE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r#"<w:pStyle\b[^>]*w:val="Heading(\d+)""#)
            .expect("literal DOCX heading style regex")
    });
    static NUMBERING: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?s)<w:numPr\b.*?</w:numPr>").expect("literal DOCX numbering regex")
    });
    static OUTLINE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"<w:outlineLvl\b").expect("literal DOCX outline regex"));
    PARAGRAPH_PROPERTIES
        .replace_all(xml, |captures: &regex::Captures<'_>| {
            let inner = &captures[1];
            let Some(level) = HEADING_STYLE.captures(inner).and_then(|style| style.get(1)) else {
                return captures[0].to_owned();
            };
            if !NUMBERING.is_match(inner) {
                return captures[0].to_owned();
            }
            let mut inner = NUMBERING.replace_all(inner, "").into_owned();
            if !OUTLINE.is_match(&inner) {
                let Some(level) = level
                    .as_str()
                    .parse::<u8>()
                    .ok()
                    .filter(|level| (1..=6).contains(level))
                    .map(|level| level - 1)
                else {
                    return format!("<w:pPr>{inner}</w:pPr>");
                };
                inner.insert_str(0, &format!(r#"<w:outlineLvl w:val="{level}"/>"#));
            }
            format!("<w:pPr>{inner}</w:pPr>")
        })
        .into_owned()
}

fn normalize_heading_styles(xml: &str) -> String {
    static DEFAULT_STYLE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r#"<w:style\b[^>]*\bw:default="1""#).expect("literal DOCX default style regex")
    });
    static HEADING_STYLE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r#"(?s)<w:style\b[^>]*\bw:styleId="Heading\d+".*?</w:style>"#)
            .expect("literal DOCX heading style regex")
    });
    static HEADING_NAME: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r#"(?i)(<w:name\b[^>]*w:val=")Heading ([0-9])(")"#)
            .expect("literal DOCX heading name regex")
    });
    static STYLE_LEVEL: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r#"w:styleId="Heading([1-6])""#).expect("literal DOCX heading level regex")
    });
    static OUTLINE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"<w:outlineLvl\b").expect("literal DOCX outline regex"));
    static PARAGRAPH_PROPERTIES: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(<w:pPr[\s>][^<]*)").expect("literal DOCX style properties regex")
    });

    let xml = if DEFAULT_STYLE.is_match(xml) {
        xml.to_owned()
    } else {
        xml.replacen(
            "</w:styles>",
            r#"<w:style w:default="1" w:styleId="Normal" w:type="paragraph"><w:name w:val="Normal"/><w:qFormat/></w:style></w:styles>"#,
            1,
        )
    };
    HEADING_STYLE
        .replace_all(&xml, |captures: &regex::Captures<'_>| {
            let mut style = HEADING_NAME
                .replace(&captures[0], "${1}heading ${2}${3}")
                .into_owned();
            if !OUTLINE.is_match(&style) {
                if let Some(level) = STYLE_LEVEL
                    .captures(&style)
                    .and_then(|capture| capture.get(1))
                {
                    let level = level.as_str().parse::<u8>().unwrap_or(1) - 1;
                    style = PARAGRAPH_PROPERTIES
                        .replacen(
                            &style,
                            1,
                            format!("$1<w:outlineLvl w:val=\"{level}\"/>").as_str(),
                        )
                        .into_owned();
                }
            }
            style
        })
        .into_owned()
}

fn drafting_docx_input(bytes: &[u8]) -> Result<Vec<u8>> {
    static HEADING_STYLE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r#"<w:style\b[^>]*\bw:styleId="Heading\d+""#)
            .expect("literal DOCX heading style regex")
    });
    let inspected = read_docx_files(bytes, Some(&["word/document.xml", "word/styles.xml"]))?;
    let document = docx_part(&inspected, "word/document.xml")
        .ok_or_else(|| Error::Message("Drafting mode requires a valid DOCX".to_owned()))?;
    let stripped = strip_heading_numbering(&document);
    let mut changed = stripped != document;
    let styles = docx_part(&inspected, "word/styles.xml");
    let normalized_styles = if let Some(styles) = styles {
        if HEADING_STYLE.is_match(&styles) {
            let normalized = normalize_heading_styles(&styles);
            changed |= normalized != styles;
            Some(normalized)
        } else {
            None
        }
    } else {
        None
    };
    if !changed {
        return Ok(bytes.to_vec());
    }
    let mut files = read_docx_files(bytes, None)?;
    if stripped != document {
        replace_file(&mut files, "word/document.xml", stripped.into_bytes());
    }
    if let Some(styles) = normalized_styles {
        replace_file(&mut files, "word/styles.xml", styles.into_bytes());
    }
    write_docx_files(&files)
}

fn clean_process_error(bytes: &[u8]) -> String {
    legal_structure::normalize_javascript_whitespace(&String::from_utf8_lossy(bytes))
        .chars()
        .take(500)
        .collect()
}

fn pandoc_drafting_markdown(bytes: Vec<u8>) -> Result<String> {
    const MAX_OUTPUT: usize = MAX_DOCX_SUPRA_BYTES;
    const SYSTEM_ENV: [&str; 20] = [
        "APPDATA",
        "COMSPEC",
        "HOME",
        "LANG",
        "LC_ALL",
        "LOCALAPPDATA",
        "PATH",
        "PATHEXT",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "SHELL",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USERPROFILE",
        "WINDIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
    ];
    let mut command = Command::new("pandoc");
    command
        .args([
            "-f",
            "docx",
            "-t",
            "gfm",
            "--sandbox",
            "--wrap=none",
            "-o",
            "-",
        ])
        .env_clear();
    for (name, value) in env::vars_os() {
        let permitted = {
            let name_string = name.to_string_lossy();
            SYSTEM_ENV
                .iter()
                .any(|allowed| name_string.eq_ignore_ascii_case(allowed))
        };
        if permitted {
            command.env(name, value);
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000);
    }
    let output = crate::process::run(command, bytes, Duration::from_secs(120), MAX_OUTPUT, 8_192)
        .map_err(|error| match error {
        crate::process::RunError::Io(source) if source.kind() == std::io::ErrorKind::NotFound => {
            Error::Message(
                "Pandoc is required for drafting mode but was not found on PATH".to_owned(),
            )
        }
        crate::process::RunError::Io(source) => Error::Message(format!(
            "Pandoc conversion failed: {}",
            clean_process_error(source.to_string().as_bytes())
        )),
        crate::process::RunError::Timeout => {
            Error::Message("Pandoc conversion timed out".to_owned())
        }
    })?;
    if output.stdout_exceeded {
        return Err(Error::Message(
            "Pandoc conversion output exceeded 25 MiB".to_owned(),
        ));
    }
    if !output.status.success() {
        return Err(Error::Message(format!(
            "Pandoc conversion failed (exit {}): {}",
            output
                .status
                .code()
                .map_or_else(|| "unknown".to_owned(), |code| code.to_string()),
            clean_process_error(&output.stderr)
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn clean_drafting_markdown(markdown: String) -> String {
    static HTML_IMAGE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?i)<img\b[^>]*\/?>").expect("literal HTML image regex"));
    static MARKDOWN_IMAGE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"!\[[^\]]*\]\([^)]*\)(?:\{[^}]*\})?").expect("literal Markdown image regex")
    });
    static EMPTY_LINK: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(&format!(
            r"(?m)^\[\]\([^)]*\){}*$",
            legal_structure::JS_WHITESPACE_CLASS
        ))
        .expect("literal empty link regex")
    });
    static UNSAFE_LINK: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"(?i)\[[^\]]*\]\((?:data|javascript):[^)]*\)")
            .expect("literal unsafe link regex")
    });
    static ESCAPED_BRACKET: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\\([\[\]])").expect("literal escaped bracket regex"));
    let markdown = markdown.replace("\r\n", "\n").replace('\r', "\n");
    let markdown = HTML_IMAGE.replace_all(&markdown, "[Image omitted]");
    let markdown = MARKDOWN_IMAGE.replace_all(&markdown, "[Image omitted]");
    let markdown = EMPTY_LINK.replace_all(&markdown, "");
    let markdown = UNSAFE_LINK.replace_all(&markdown, "");
    let markdown = ESCAPED_BRACKET.replace_all(&markdown, "$1");
    legal_structure::trim_javascript_whitespace(&markdown).to_owned()
}

fn docx_document_xml(bytes: &[u8]) -> Result<Vec<u8>> {
    const MAX_DOCX_BYTES: usize = 50 * 1024 * 1024;
    const MAX_XML_BYTES: u64 = 32 * 1024 * 1024;
    if bytes.is_empty() || bytes.len() > MAX_DOCX_BYTES {
        return Err(Error::Message(
            "DOCX is empty or exceeds the read limit".to_owned(),
        ));
    }
    let mut archive = ZipArchive::new(Cursor::new(bytes))?;
    if archive.len() > 10_000 {
        return Err(Error::Message(
            "DOCX has too many package entries".to_owned(),
        ));
    }
    let mut document = archive
        .by_name("word/document.xml")
        .map_err(|_| Error::Message("DOCX has no word/document.xml".to_owned()))?;
    if document.size() > MAX_XML_BYTES {
        return Err(Error::Message(
            "DOCX document XML exceeds the read limit".to_owned(),
        ));
    }
    let mut xml = Vec::with_capacity(document.size() as usize);
    document
        .read_to_end(&mut xml)
        .map_err(|source| Error::io("word/document.xml", source))?;
    Ok(xml)
}

fn docx_structure_input(
    bytes: &[u8],
    include_tables: bool,
) -> Result<(Vec<String>, Vec<legal_structure::AuthoritativeTableCell>)> {
    let xml = docx_document_xml(bytes)?;
    let document = match parse_xml(&xml) {
        Ok(document) => document,
        Err(_) => {
            let xml = std::str::from_utf8(&xml)
                .map_err(|error| Error::Message(format!("DOCX XML is not UTF-8: {error}")))?;
            let paragraphs = tolerant_docx_paragraphs(xml)?;
            return Ok((paragraphs, Vec::new()));
        }
    };
    let root = document.root()?;
    let body = root
        .direct_elements()
        .find(|element| element.is(W_NS, "body"))
        .ok_or_else(|| Error::Message("DOCX has no document body".to_owned()))?;

    let mut all = Vec::new();
    walk_elements(body, &mut all);
    let canonical_elements = all
        .iter()
        .copied()
        .filter(|element| element.is(W_NS, "p") && !element.self_closing)
        .collect::<Vec<_>>();
    let paragraphs = canonical_elements
        .iter()
        .map(|paragraph| normalized_docx_paragraph(paragraph))
        .collect::<Result<Vec<_>>>()?;
    if !include_tables {
        return Ok((paragraphs, Vec::new()));
    }
    let mut starts = Vec::with_capacity(paragraphs.len());
    let mut text_length = 0;
    for (index, paragraph) in paragraphs.iter().enumerate() {
        text_length += usize::from(index > 0);
        starts.push(text_length);
        text_length += legal_structure::utf16_len(paragraph);
    }

    let mut by_element = canonical_elements
        .iter()
        .enumerate()
        .map(|(index, paragraph)| {
            let start = starts[index];
            (
                *paragraph as *const XmlElement,
                (
                    start,
                    start + legal_structure::utf16_len(&paragraphs[index]),
                ),
            )
        })
        .collect::<HashMap<_, _>>();
    let mut entry_paragraph = HashMap::new();
    let mut preceding = 0;
    for element in &all {
        entry_paragraph.insert(*element as *const XmlElement, preceding);
        if element.is(W_NS, "p") {
            if element.self_closing {
                let offset = starts.get(preceding).copied().unwrap_or(text_length);
                by_element.insert(*element as *const XmlElement, (offset, offset));
            } else {
                preceding += 1;
            }
        }
    }

    let mut table_cells = Vec::new();
    for (table_index, table) in all
        .iter()
        .copied()
        .filter(|element| element.is(W_NS, "tbl"))
        .enumerate()
    {
        let mut vertical_anchors = HashMap::<usize, usize>::new();
        for (row_index, row) in nested_word_elements(table, "tr").into_iter().enumerate() {
            let row_properties = row
                .direct_elements()
                .find(|element| element.is(W_NS, "trPr"));
            let mut column = 1 + word_int(row_properties, "gridBefore").unwrap_or(0);
            let mut next_vertical_anchors = HashMap::new();
            let mut horizontal_anchor = None;
            for cell in nested_word_elements(row, "tc") {
                let cell_properties = cell
                    .direct_elements()
                    .find(|element| element.is(W_NS, "tcPr"));
                let column_span = word_int(cell_properties, "gridSpan")
                    .filter(|value| *value > 0)
                    .unwrap_or(1);
                let column_end = column + column_span;
                let merge = |name| {
                    cell_properties
                        .and_then(|properties| {
                            properties
                                .direct_elements()
                                .find(|element| element.is(W_NS, name))
                        })
                        .map(|element| {
                            element
                                .attribute(None, "val")
                                .is_some_and(|value| value.eq_ignore_ascii_case("restart"))
                        })
                };
                let vertical_merge = merge("vMerge");
                let horizontal_merge = merge("hMerge");
                let vertical_continuation = vertical_merge == Some(false);
                let horizontal_continuation = horizontal_merge == Some(false);
                let vertical_anchor = vertical_continuation
                    .then(|| {
                        vertical_anchors.get(&column).copied().filter(|anchor| {
                            (column..column_end)
                                .all(|covered| vertical_anchors.get(&covered) == Some(anchor))
                        })
                    })
                    .flatten();
                let continuation_anchor = if (!vertical_continuation || vertical_anchor.is_some())
                    && (!horizontal_continuation || horizontal_anchor.is_some())
                    && (!vertical_continuation
                        || !horizontal_continuation
                        || vertical_anchor == horizontal_anchor)
                {
                    vertical_anchor.or(horizontal_anchor)
                } else {
                    None
                };
                let continuation = vertical_continuation || horizontal_continuation;
                let anchor = if continuation {
                    continuation_anchor
                } else {
                    let contents = paragraphs_under(cell, &by_element);
                    let empty_at = || {
                        let preceding = entry_paragraph
                            .get(&(cell as *const XmlElement))
                            .copied()
                            .unwrap_or(paragraphs.len());
                        preceding.checked_sub(1).map_or(0, |index| {
                            starts[index] + legal_structure::utf16_len(&paragraphs[index])
                        })
                    };
                    let start = contents.first().map_or_else(empty_at, |(start, _)| *start);
                    let end = contents.last().map_or_else(empty_at, |(_, end)| *end);
                    table_cells.push(legal_structure::AuthoritativeTableCell {
                        table: table_index + 1,
                        table_name: None,
                        row: row_index + 1,
                        column,
                        row_span: None,
                        column_span: (column_span > 1).then_some(column_span),
                        address: None,
                        display_value: None,
                        start,
                        end,
                    });
                    Some(table_cells.len() - 1)
                };
                if let Some(anchor) = anchor {
                    if vertical_continuation {
                        let span = row_index + 2 - table_cells[anchor].row;
                        let span = span.max(table_cells[anchor].row_span.unwrap_or(1));
                        table_cells[anchor].row_span = (span > 1).then_some(span);
                    }
                    if horizontal_continuation {
                        let span = column_end - table_cells[anchor].column;
                        let span = span.max(table_cells[anchor].column_span.unwrap_or(1));
                        table_cells[anchor].column_span = (span > 1).then_some(span);
                    }
                    if vertical_merge.is_some() {
                        for covered in column..column_end {
                            next_vertical_anchors.insert(covered, anchor);
                        }
                    }
                }
                horizontal_anchor = if horizontal_merge.is_some() {
                    anchor
                } else {
                    None
                };
                column = column_end;
            }
            vertical_anchors = next_vertical_anchors;
        }
    }
    Ok((paragraphs, table_cells))
}

/// Return the accepted model-visible DOCX text without running structure
/// detection. Drafting mode uses the same Pandoc adaptation as full analysis.
pub fn docx_text(bytes: &[u8], drafting: bool) -> Result<String> {
    if drafting {
        return drafting_docx_text(bytes);
    }
    docx_structure_input(bytes, false).map(|(paragraphs, _)| paragraphs.join("\n"))
}

/// Parse the accepted DOCX text and authoritative table coordinates once, then
/// feed the canonical Rust detector directly.
pub fn analyze_docx_bytes(
    bytes: &[u8],
    document_id: String,
) -> Result<legal_structure::DocumentStructure> {
    let (paragraphs, table_cells) = docx_structure_input(bytes, true)?;
    legal_structure::analyze_docx(document_id, paragraphs, &table_cells)
        .map_err(|error| Error::Message(error.to_string()))
}

fn drafting_docx_text(bytes: &[u8]) -> Result<String> {
    if bytes.is_empty() || bytes.len() > MAX_DOCX_SUPRA_BYTES {
        return Err(Error::Message(
            "Precedent DOCX exceeds the drafting read limit".to_owned(),
        ));
    }
    let input = drafting_docx_input(bytes).map_err(|error| match error {
        Error::Zip(_) => Error::Message("Precedent DOCX is corrupted or truncated".to_owned()),
        Error::Message(message) if message.contains("XML part exceeds") => {
            Error::Message("Precedent DOCX contains an oversized XML part".to_owned())
        }
        Error::Message(message) => Error::Message(
            message
                .strip_prefix("DOCX ")
                .map_or(message.clone(), |detail| format!("Precedent DOCX {detail}")),
        ),
        error => error,
    })?;
    let markdown = pandoc_drafting_markdown(input).map_err(|error| {
        if error.to_string().contains("was not found on PATH") {
            error
        } else {
            Error::Message(format!(
                "Precedent DOCX contains malformed XML in word/document.xml: {error}"
            ))
        }
    })?;
    let markdown = clean_drafting_markdown(markdown);
    if markdown.is_empty() {
        let text = docx_text(bytes, false)?;
        return if text.trim().is_empty() {
            Err(Error::Message(
                "Precedent DOCX has no readable drafting structure".to_owned(),
            ))
        } else {
            Ok(text)
        };
    }
    Ok(markdown)
}

/// Build the model-visible drafting document in one Rust operation. OOXML and
/// Pandoc adaptation stay here; the resulting text uses the canonical detector.
pub fn analyze_docx_drafting_bytes(
    bytes: &[u8],
    document_id: String,
) -> Result<legal_structure::DocumentStructure> {
    let markdown = drafting_docx_text(bytes)?;
    legal_structure::analyze_instrument(markdown, document_id, &[], true)
        .map_err(|error| Error::Message(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn authority_units_keep_utf16_references_and_word_footnote_order() {
        let document = format!(
            r#"<w:document xmlns:w="{W_NS}"><w:body>
                <w:p/>
                <w:p><w:r><w:t>A😀</w:t><w:footnoteReference w:id="7"/><w:tab/><w:t>B</w:t></w:r></w:p>
                <w:p><w:r><w:t>C</w:t><w:footnoteReference w:id="3"/></w:r></w:p>
            </w:body></w:document>"#
        );
        let footnotes = format!(
            r#"<w:footnotes xmlns:w="{W_NS}">
                <w:footnote w:id="-1"><w:p><w:r><w:t>separator</w:t></w:r></w:p></w:footnote>
                <w:footnote w:id="3"><w:p><w:r><w:t>First</w:t><w:tab/><w:t>note.</w:t></w:r></w:p></w:footnote>
                <w:footnote w:id="7"><w:p><w:r><w:t>Second</w:t><w:br/><w:t>note.</w:t></w:r></w:p></w:footnote>
            </w:footnotes>"#
        );
        let bytes = write_docx_files(&[
            ("word/document.xml".to_owned(), document.into_bytes()),
            ("word/footnotes.xml".to_owned(), footnotes.into_bytes()),
        ])
        .unwrap();

        assert_eq!(
            docx_to_toa_text_units(&bytes).unwrap(),
            vec![
                json!({"key":"body:1","kind":"body","ordinal":1,"footnote_id":null,"text":"A😀\tB","footnote_refs":[[2,3]]}),
                json!({"key":"body:2","kind":"body","ordinal":2,"footnote_id":null,"text":"C","footnote_refs":[[1,1]]}),
                json!({"key":"footnote:3","kind":"footnote","ordinal":1,"footnote_id":1,"text":"First\tnote.","footnote_refs":[]}),
                json!({"key":"footnote:7","kind":"footnote","ordinal":2,"footnote_id":2,"text":"Second\nnote.","footnote_refs":[]}),
            ]
        );
    }
}
